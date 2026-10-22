"""Cellosaurus text of the labelled lines as PIE's ``context_text``, keyed by ModelID.

Runs on the Mac: the H20 cannot reach OpenAI. The lines are the final split's
single-cell lines (train, val, test) and its labelled bulk-only lines; each is joined to
its Cellosaurus accession through DepMap Model.csv's RRID.

A line whose accession PIE's released ``context_text`` already holds takes PIE's vector
and text byte for byte; only the other lines are embedded. A released key names its
accession directly (Tahoe-100M's ``CVCL_0023``) or through its dataset's context map
(``RELEASED_NAMED_CONTEXTS``); released contexts with a stimulation (Jiang's
``a549_ifnb`` and the like) describe a stimulated line, not a basal one, and are never
reused. Each reused line is also re-rendered from its 56.0 record under its released key
and compared with PIE's released text; the result is recorded, not acted on.

A line PIE has no vector for is rendered with PIE's template (``render_context``) from
its record in PIE's release-pinned Cellosaurus client (56.0, PIE's pin) under its
accession as the "Context Name", which is how PIE names every accession-keyed context it
released (Tahoe's 50), then embedded by PIE's embedder (``text-embedding-3-large``,
3,072-d). A record Cellosaurus does not serve, or one PIE's parser refuses (an age range
such as ``52-54Y``), gives that line no text; such lines are listed with the reason,
next to those without an RRID.

This module imports only the standard library and PIE (with PIE's numpy and requests),
never ``src.*``, because it runs in PIE's environment; OPENAI_API_KEY is read from the
environment by PIE's embedder, only when some line needs embedding, and never written.
An OpenAI-compatible relay is used by setting OPENAI_BASE_URL (read by the ``openai``
client). Before any line is embedded, the released texts of the reused lines are
embedded through the same endpoint and compared with PIE's released vectors; a cosine
below ``ENDPOINT_MIN_COSINE`` (another model, or truncated vectors) stops the run::

    uv run --isolated --no-project --python 3.12 \\
        --with-editable 'third_party/pie[sources]' \\
        python -m src.data.build.cellosaurus_context --out data/ladder_context_text

Under ``--out``: ``cellosaurus/`` (the client's record cache, reused on a rerun),
``embed/context_text/`` (PIE's per-batch embedding checkpoint, reused on a rerun),
``context_text/`` (the source) and ``provenance.json``. An existing source is never
overwritten.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import pie
import requests
from pie.sources.common import provenance_path
from pie.sources.contract import (
    FORMAT_VERSION,
    Source,
    SourceMeta,
    read_descriptions,
    read_source,
    write_source,
)
from pie.sources.embed import openai as openai_embed
from pie.sources.text import contexts
from pie.utils import write_json

SPLIT = Path("configs/benchmarks/cell_line_geneeffect_split.json")
MODELS = Path("data/sl_dependency_v0/raw/depmap/Model.csv")
#: PIE's released context_text (arcinstitute/PIE_sources at this revision); the same
#: relative path on the Mac and the H20.
PIE_CONTEXT_TEXT = Path(
    "data/pie/hf/datasets/arcinstitute/pie_sources/"
    "cb1aaa4e7655605bdc70a9bd77bbd62016b8c7d7/context_text"
)
#: The split keys whose lines are read: single-cell lines and labelled bulk-only lines.
SPLIT_KEYS = ("train", "val", "test", "bulk_labelled")
CELLOSAURUS_RELEASE = "56.0"
SOURCE_NAME = "context_text"
PROVENANCE = "provenance.json"
#: Lowest cosine allowed between the endpoint's vector of a released text and PIE's
#: released vector of it; text-embedding-3-large reproduces its own vectors to ~1e-5.
ENDPOINT_MIN_COSINE = 0.999
#: The same pattern PIE's client accepts (pie.sources.text.context_file._ACCESSION).
ACCESSION = re.compile(r"CVCL_[A-Z0-9]{4}", re.ASCII)
#: PIE's released keys that are names, with the accession their dataset's context map
#: (preprocessed/contexts.yaml of the HF dataset repo) gives them.
RELEASED_NAMED_CONTEXTS = {
    # arcinstitute/PIE_replogle_nadig_essential
    "hepg2": "CVCL_0027",
    "jurkat": "CVCL_0367",
    "k562": "CVCL_0004",
    "rpe1": "CVCL_4388",
    # arcinstitute/PIE_x_atlas_orion
    "hct116": "CVCL_0291",
    "hek293t": "CVCL_0063",
    # arcinstitute/PIE_arc_vcc_25
    "ARC_H1": "CVCL_9771",
    "ARC_H1_VAL": "CVCL_9771",
}
STIMULATION_SEGMENT = "Stimulation: "


def split_lines(split: Mapping[str, object]) -> list[str]:
    """Sorted ModelIDs of the split's SPLIT_KEYS lists; a line listed twice raises."""
    lines = [str(model) for key in SPLIT_KEYS for model in split[key]]
    if len(set(lines)) != len(lines):
        raise ValueError("a ModelID appears in more than one split list")
    return sorted(lines)


def read_rrids(models: Path) -> dict[str, str]:
    """{ModelID: stripped RRID} of every Model.csv row."""
    with Path(models).open(newline="", encoding="utf-8") as handle:
        return {row["ModelID"]: row["RRID"].strip() for row in csv.DictReader(handle)}


def line_accessions(
    lines: Sequence[str], rrids: Mapping[str, str]
) -> tuple[dict[str, str], list[str]]:
    """({ModelID: accession} of lines with a valid CVCL RRID, sorted lines without).

    A line absent from Model.csv raises: the split and Model.csv are one DepMap release.
    """
    absent = [line for line in lines if line not in rrids]
    if absent:
        raise KeyError(f"{len(absent)} split lines are not in Model.csv: {absent[:10]}")
    accessions = {
        line: rrids[line]
        for line in lines
        if ACCESSION.fullmatch(rrids[line]) is not None
    }
    return accessions, sorted(line for line in lines if line not in accessions)


def released_keys(descriptions: Mapping[str, str]) -> dict[str, str]:
    """{accession: PIE key} of the released basal contexts.

    Stimulated contexts are left out. A key that is neither an accession nor in
    RELEASED_NAMED_CONTEXTS raises. Of two keys for one accession, the accession
    itself wins, then the codepoint-first name.
    """
    candidates: dict[str, list[str]] = {}
    for key, text in descriptions.items():
        if any(s.startswith(STIMULATION_SEGMENT) for s in text.split(" ;\n")):
            continue
        if ACCESSION.fullmatch(key) is not None:
            accession = key
        elif key in RELEASED_NAMED_CONTEXTS:
            accession = RELEASED_NAMED_CONTEXTS[key]
        else:
            raise ValueError(f"released context {key!r} has no known accession")
        candidates.setdefault(accession, []).append(key)
    return {
        accession: accession if accession in keys else sorted(keys)[0]
        for accession, keys in candidates.items()
    }


def _record(
    client: contexts.CellosaurusClient, accession: str
) -> tuple[dict | None, str | None]:
    """(record, None), or (None, reason) when Cellosaurus does not serve it."""
    try:
        return client.get(accession), None
    except requests.HTTPError as error:
        if error.response is not None and error.response.status_code == 404:
            return None, f"Cellosaurus {CELLOSAURUS_RELEASE} has no record (404)"
        raise


def _render(key: str, record: dict) -> tuple[str | None, str | None]:
    """(PIE's text of `record` named `key`, None), or (None, why PIE refused it)."""
    try:
        return contexts.render_context(key, record, None), None
    except ValueError as error:
        return None, f"PIE's renderer refused the record: {error}"


def plan_lines(
    accessions: Mapping[str, str],
    released: Mapping[str, str],
    released_texts: Mapping[str, str],
    client: contexts.CellosaurusClient,
) -> dict[str, dict[str, object]]:
    """Per line with an accession: reused (PIE key, render check) or embedded (text)."""
    plans: dict[str, dict[str, object]] = {}
    for line, accession in sorted(accessions.items()):
        record, reason = _record(client, accession)
        pie_key = released.get(accession)
        if pie_key is not None:
            text, render_reason = _render(pie_key, record) if record else (None, reason)
            plans[line] = {
                "accession": accession,
                "status": "reused",
                "pie_key": pie_key,
                "render_equal": text == released_texts[pie_key],
                "render_note": render_reason,
            }
            continue
        text, render_reason = _render(accession, record) if record else (None, reason)
        plans[line] = {
            "accession": accession,
            "status": "embedded" if text is not None else "no_text",
            "text": text,
            "reason": render_reason,
        }
    return plans


def _check_released(source: Source) -> None:
    meta = source.meta
    expected = (SOURCE_NAME, "dense", openai_embed.MODEL, openai_embed.DIM, "float32")
    found = (meta.name, meta.layout, meta.provenance["model"], meta.dim, meta.dtype)
    if found != expected:
        raise ValueError(f"PIE's released context_text is {found}, expected {expected}")


def check_endpoint(
    pie_source: Source,
    released_texts: Mapping[str, str],
    pie_keys: Sequence[str],
    resume_dir: Path,
) -> dict[str, object]:
    """Embed the released texts of ``pie_keys`` through the configured endpoint and
    compare each with PIE's released vector; raise if any cosine is below the bound."""
    if not pie_keys:
        raise ValueError("no reused line to check the embedding endpoint against")
    ours = openai_embed.embed_texts(
        [released_texts[key] for key in pie_keys], resume_dir=resume_dir
    ).astype(np.float64)
    theirs = np.stack(
        [np.asarray(pie_source.tokens(key)[0], dtype=np.float64) for key in pie_keys]
    )
    cosine = (ours * theirs).sum(axis=1) / (
        np.linalg.norm(ours, axis=1) * np.linalg.norm(theirs, axis=1)
    )
    endpoint = os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
    result = {
        "endpoint_host": urlsplit(endpoint).hostname,
        "keys": list(pie_keys),
        "min_cosine": float(cosine.min()),
        "median_cosine": float(np.median(cosine)),
        "bound": ENDPOINT_MIN_COSINE,
    }
    if cosine.min() < ENDPOINT_MIN_COSINE:
        worst = pie_keys[int(cosine.argmin())]
        raise ValueError(
            f"endpoint {result['endpoint_host']} does not reproduce PIE's "
            f"{openai_embed.MODEL} vectors: cosine {cosine.min():.6f} on {worst} "
            f"(median {result['median_cosine']:.6f}, bound {ENDPOINT_MIN_COSINE})"
        )
    return result


def build_context_text(
    out: Path,
    split_path: Path,
    models_path: Path,
    released_dir: Path,
    client: contexts.CellosaurusClient | None = None,
) -> Path:
    """Write ``<out>/context_text`` (reused and embedded rows) and its provenance.

    ``client`` defaults to the online client caching under ``<out>/cellosaurus``.
    """
    out = Path(out)
    source_dir = out / SOURCE_NAME
    for path in (source_dir, out / PROVENANCE):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
    lines = split_lines(json.loads(Path(split_path).read_text(encoding="utf-8")))
    accessions, without_rrid = line_accessions(lines, read_rrids(models_path))
    pie_source = read_source(released_dir)
    _check_released(pie_source)
    released_texts = read_descriptions(released_dir)
    released = released_keys(released_texts)
    if any(accession not in released for accession in accessions.values()):
        openai_embed.require_openai_key()
    cache_dir = out / "cellosaurus"
    if client is None:
        client = contexts.CellosaurusClient(
            cache_dir, CELLOSAURUS_RELEASE, offline=False
        )
    plans = plan_lines(accessions, released, released_texts, client)

    embedded = [line for line, plan in plans.items() if plan["status"] == "embedded"]
    reused = [line for line, plan in plans.items() if plan["status"] == "reused"]
    endpoint = (
        check_endpoint(
            pie_source,
            released_texts,
            sorted({str(plans[line]["pie_key"]) for line in reused}),
            out / "embed" / "endpoint_check",
        )
        if embedded
        else None
    )
    fresh = (
        openai_embed.embed_texts(
            [str(plans[line]["text"]) for line in embedded],
            resume_dir=out / "embed" / SOURCE_NAME,
        )
        if embedded
        else np.zeros((0, openai_embed.DIM), dtype=np.float32)
    )
    fresh_row = dict(zip(embedded, fresh, strict=True))
    keys = [line for line, plan in plans.items() if plan["status"] != "no_text"]
    rows, descriptions = [], {}
    for line in keys:
        plan = plans[line]
        if plan["status"] == "reused":
            pie_key = str(plan["pie_key"])
            rows.append(np.asarray(pie_source.tokens(pie_key)[0]))
            descriptions[line] = released_texts[pie_key]
        else:
            rows.append(fresh_row[line])
            descriptions[line] = str(plan["text"])
    embeddings = np.stack(rows).astype(np.float32, copy=False)

    record = client.provenance()
    meta = SourceMeta(
        format_version=FORMAT_VERSION,
        name=SOURCE_NAME,
        layout="dense",
        index="context",
        keys=keys,
        dim=openai_embed.DIM,
        dtype="float32",
        provenance={
            "tool_version": pie.__version__,
            "inputs": {
                "cellosaurus": {
                    "url": record["url"],
                    "release": record["release"],
                },
                "depmap_model": {"url": None, "release": "26Q1"},
                "pie_context_text": {"url": None, "release": None},
            },
            "model": openai_embed.MODEL,
            "revision": None,  # the OpenAI API serves the model by name only
            "batch": openai_embed.BATCH,
            "max_tokens": openai_embed.MAX_TOKENS,
            "prior": provenance_path(released_dir),
            "on_conflict": None,
            "params": {
                "cellosaurus_release": CELLOSAURUS_RELEASE,
                "context_name": "cellosaurus accession",
                "split_keys": list(SPLIT_KEYS),
                "reused": len(keys) - len(embedded),
                "embedded": len(embedded),
            },
        },
    )
    write_source(source_dir, meta, embeddings, descriptions=descriptions)

    no_text = {
        line: str(plan["reason"])
        for line, plan in plans.items()
        if plan["status"] == "no_text"
    }
    write_json(
        out / PROVENANCE,
        {
            "cellosaurus": record,
            "embedding_model": openai_embed.MODEL,
            "embedding_dim": openai_embed.DIM,
            "endpoint_check": endpoint,
            "split": {"path": str(split_path), "keys": list(SPLIT_KEYS)},
            "depmap_model": {"path": str(models_path)},
            "pie_context_text": {"path": str(released_dir)},
            "context_name": "cellosaurus accession",
            "counts": {
                "lines": len(lines),
                "with_text": len(keys),
                "reused": len(reused),
                "embedded": len(embedded),
                "without_rrid": len(without_rrid),
                "without_text": len(no_text),
                "render_equal": sum(
                    bool(plans[line]["render_equal"]) for line in reused
                ),
            },
            "lines_without_rrid": without_rrid,
            "lines_without_text": no_text,
            "lines": {
                line: {key: value for key, value in plan.items() if key != "text"}
                for line, plan in plans.items()
            },
        },
    )
    return source_dir


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True, help="output root")
    parser.add_argument("--split", type=Path, default=SPLIT, help="final split JSON")
    parser.add_argument("--models", type=Path, default=MODELS, help="DepMap Model.csv")
    parser.add_argument(
        "--pie-context-text",
        type=Path,
        default=PIE_CONTEXT_TEXT,
        help="PIE's released context_text source dir",
    )
    args = parser.parse_args(argv)
    source_dir = build_context_text(
        args.out, args.split, args.models, args.pie_context_text
    )
    print(f"wrote {source_dir}")


if __name__ == "__main__":
    main()
