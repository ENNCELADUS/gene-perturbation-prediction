"""Cellosaurus context text, offline against PIE's fixtures and a fake embedder."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pie.sources.text.tools")

from pie.sources.contract import (  # noqa: E402
    FORMAT_VERSION,
    SourceMeta,
    read_descriptions,
    read_source,
    write_source,
)
from pie.sources.embed import openai as openai_embed  # noqa: E402
from pie.sources.text import contexts  # noqa: E402

from src.data.builders import cellosaurus_context as cc  # noqa: E402
from src.experiments.run_dir import REPO_ROOT  # noqa: E402

FIXTURES = REPO_ROOT / "third_party/pie/tests/sources/text/fixtures/cellosaurus"
FAKE_KEY = "sk-fake-key-never-written-0123456789"
#: A synthetic record whose age PIE's renderer refuses.
REFUSED = "CVCL_Y009"
#: ModelID -> (split key, RRID). Y001 is released under its accession, Y002 under the
#: name "translineb", Y003 is not released; two lines lack a usable RRID, and the lines
#: outside the four read split keys must not enter.
LINES = {
    "ACH-000001": ("train", "CVCL_Y001"),
    "ACH-000002": ("val", "CVCL_Y002"),
    "ACH-000003": ("test", ""),
    "ACH-000004": ("bulk_labelled", "CVCL_Y003"),
    "ACH-000005": ("bulk_labelled", "not-an-rrid"),
    "ACH-000006": ("bulk_unlabelled", "CVCL_9771"),
    "ACH-000007": ("excluded", "CVCL_0186"),
    "ACH-000008": ("train", REFUSED),
}


def _record(accession: str) -> dict:
    path = FIXTURES / cc.CELLOSAURUS_RELEASE / f"{accession}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _write_inputs(root: Path, lines: dict[str, tuple[str, str]]) -> tuple[Path, Path]:
    split: dict[str, object] = {"train": [], "val": [], "test": [], "bulk_labelled": []}
    split["bulk_unlabelled"] = []
    split["excluded"] = {}
    for model, (key, _) in lines.items():
        if key == "excluded":
            split[key][model] = "synthetic"  # type: ignore[index]
        else:
            split[key].append(model)  # type: ignore[union-attr]
    split_path = root / "split.json"
    split_path.write_text(json.dumps(split), encoding="utf-8")
    models_path = root / "Model.csv"
    with models_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ModelID", "CellLineName", "RRID"])
        for model, (_, rrid) in lines.items():
            writer.writerow([model, f"line {model}", rrid])
    return split_path, models_path


def _write_released(root: Path) -> Path:
    descriptions = {
        "CVCL_Y001": contexts.render_context("CVCL_Y001", _record("CVCL_Y001"), None),
        "translineb": contexts.render_context("translineb", _record("CVCL_Y002"), None),
        "stimline_x": "Context Name: stimline_x ;\nStimulation: Fictional factor",
    }
    rows = np.random.default_rng(7).standard_normal((3, openai_embed.DIM))
    meta = SourceMeta(
        format_version=FORMAT_VERSION,
        name="context_text",
        layout="dense",
        index="context",
        keys=list(descriptions),
        dim=openai_embed.DIM,
        dtype="float32",
        provenance={"model": openai_embed.MODEL},
    )
    return write_source(
        root / "released" / "context_text",
        meta,
        rows.astype(np.float32),
        descriptions=descriptions,
    )


def _offline_client(cache: Path) -> contexts.CellosaurusClient:
    shutil.copytree(FIXTURES, cache)
    refused = _record("CVCL_Y003")
    record = refused["Cellosaurus"]["cell-line-list"][0]
    record["accession-list"] = [{"type": "primary", "value": REFUSED}]
    record["age"] = ">60Y"
    (cache / cc.CELLOSAURUS_RELEASE / f"{REFUSED}.json").write_text(
        json.dumps(refused), encoding="utf-8"
    )
    return contexts.CellosaurusClient(cache, cc.CELLOSAURUS_RELEASE, offline=True)


class FakeEmbedder:
    """Stands in for PIE's embed_texts; records every batch of texts it is given and
    returns PIE's released vector for a released text (``known``), 0.5 otherwise."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.known: dict[str, np.ndarray] = {}

    def __call__(self, texts, *, client=None, resume_dir=None, encoder=None):
        openai_embed.require_openai_key()
        self.calls.append(list(texts))
        rows = np.full((len(texts), openai_embed.DIM), 0.5, dtype=np.float32)
        for i, text in enumerate(texts):
            if text in self.known:
                rows[i] = self.known[text]
        return rows


@pytest.fixture
def setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(cc, "RELEASED_NAMED_CONTEXTS", {"translineb": "CVCL_Y002"})
    embedder = FakeEmbedder()
    monkeypatch.setattr(openai_embed, "embed_texts", embedder)
    monkeypatch.setenv("OPENAI_API_KEY", FAKE_KEY)
    released = _write_released(tmp_path)
    source, texts = read_source(released), read_descriptions(released)
    embedder.known = {text: source.tokens(key)[0] for key, text in texts.items()}
    return tmp_path, released, embedder


def _build(root: Path, released: Path, lines: dict) -> Path:
    split_path, models_path = _write_inputs(root, lines)
    out = root / "out"
    client = _offline_client(out / "cellosaurus")
    cc.build_context_text(out, split_path, models_path, released, client=client)
    return out


def test_lines_and_accessions_follow_the_split_and_model_csv(tmp_path: Path) -> None:
    split_path, models_path = _write_inputs(tmp_path, LINES)
    lines = cc.split_lines(json.loads(split_path.read_text(encoding="utf-8")))
    assert lines == [
        "ACH-000001",
        "ACH-000002",
        "ACH-000003",
        "ACH-000004",
        "ACH-000005",
        "ACH-000008",
    ]
    accessions, without = cc.line_accessions(lines, cc.read_rrids(models_path))
    assert accessions == {
        "ACH-000001": "CVCL_Y001",
        "ACH-000002": "CVCL_Y002",
        "ACH-000004": "CVCL_Y003",
        "ACH-000008": REFUSED,
    }
    assert without == ["ACH-000003", "ACH-000005"]
    with pytest.raises(KeyError, match="not in Model.csv"):
        cc.line_accessions([*lines, "ACH-999999"], cc.read_rrids(models_path))


def test_released_keys_skip_stimulations_and_prefer_the_accession(setup) -> None:
    _, released, _ = setup
    assert cc.released_keys(read_descriptions(released)) == {
        "CVCL_Y001": "CVCL_Y001",
        "CVCL_Y002": "translineb",
    }
    both = {"named": "Context Name: named", "CVCL_Y002": "Context Name: CVCL_Y002"}
    cc.RELEASED_NAMED_CONTEXTS["named"] = "CVCL_Y002"
    assert cc.released_keys(both) == {"CVCL_Y002": "CVCL_Y002"}
    with pytest.raises(ValueError, match="no known accession"):
        cc.released_keys({"unmapped": "Context Name: unmapped"})


def test_reused_lines_copy_pie_and_only_the_rest_are_embedded(setup) -> None:
    root, released, embedder = setup
    out = _build(root, released, LINES)
    source = read_source(out / "context_text")
    assert source.meta.name == "context_text"
    assert source.meta.layout == "dense"
    assert source.meta.index == "context"
    assert source.meta.keys == ["ACH-000001", "ACH-000002", "ACH-000004"]
    assert source.meta.dim == openai_embed.DIM
    assert source.embeddings.dtype == np.float32

    y003 = contexts.render_context("CVCL_Y003", _record("CVCL_Y003"), None)
    theirs = read_descriptions(released)
    assert embedder.calls == [[theirs["CVCL_Y001"], theirs["translineb"]], [y003]]
    pie = read_source(released)
    for line, key in [("ACH-000001", "CVCL_Y001"), ("ACH-000002", "translineb")]:
        assert source.tokens(line).tobytes() == pie.tokens(key).tobytes()
    assert np.all(source.tokens("ACH-000004") == 0.5)

    ours, theirs = read_descriptions(out / "context_text"), read_descriptions(released)
    assert ours["ACH-000001"] == theirs["CVCL_Y001"]
    assert ours["ACH-000002"] == theirs["translineb"]
    assert ours["ACH-000004"] == y003
    assert y003.startswith("Context Name: CVCL_Y003 ;\n")


def test_provenance_records_each_line(setup) -> None:
    root, released, _ = setup
    out = _build(root, released, LINES)
    provenance = json.loads((out / cc.PROVENANCE).read_text(encoding="utf-8"))
    assert provenance["counts"] == {
        "lines": 6,
        "with_text": 3,
        "reused": 2,
        "embedded": 1,
        "without_rrid": 2,
        "without_text": 1,
        "render_equal": 2,
    }
    lines = provenance["lines"]
    assert lines["ACH-000001"]["pie_key"] == "CVCL_Y001"
    assert lines["ACH-000002"]["pie_key"] == "translineb"
    assert lines["ACH-000002"]["render_equal"] is True
    assert lines["ACH-000004"]["status"] == "embedded"
    assert provenance["lines_without_rrid"] == ["ACH-000003", "ACH-000005"]
    assert list(provenance["lines_without_text"]) == ["ACH-000008"]
    assert "age '>60Y'" in provenance["lines_without_text"]["ACH-000008"]
    assert provenance["cellosaurus"]["release"] == "56.0"


def test_endpoint_check_is_recorded_and_refuses_another_model(
    setup, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, released, embedder = setup
    monkeypatch.setenv("OPENAI_BASE_URL", "https://relay.example/v1")
    out = _build(root, released, LINES)
    check = json.loads((out / "provenance.json").read_text())["endpoint_check"]
    assert check["endpoint_host"] == "relay.example"
    assert check["keys"] == ["CVCL_Y001", "translineb"]
    assert check["min_cosine"] > 0.999999

    embedder.known = {}  # the relay answers with another model's vectors
    split_path, models_path = _write_inputs(root, LINES)
    again = root / "again"
    client = _offline_client(again / "cellosaurus")
    with pytest.raises(ValueError, match="does not reproduce"):
        cc.build_context_text(again, split_path, models_path, released, client=client)
    assert not (again / "context_text").exists()


def test_all_lines_reused_needs_no_key(setup, monkeypatch: pytest.MonkeyPatch) -> None:
    root, released, embedder = setup
    monkeypatch.delenv("OPENAI_API_KEY")
    lines = {
        model: LINES[model] for model in ("ACH-000001", "ACH-000002", "ACH-000003")
    }
    out = _build(root, released, lines)
    assert embedder.calls == []
    assert read_source(out / "context_text").meta.keys == ["ACH-000001", "ACH-000002"]


def test_missing_key_stops_before_any_fetch(
    setup, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, released, embedder = setup
    monkeypatch.delenv("OPENAI_API_KEY")
    split_path, models_path = _write_inputs(root, LINES)
    with pytest.raises(Exception, match="OPENAI_API_KEY"):
        cc.build_context_text(root / "out", split_path, models_path, released)
    assert not (root / "out").exists()
    assert embedder.calls == []


def test_key_never_reaches_a_file(setup) -> None:
    root, released, _ = setup
    out = _build(root, released, LINES)
    for path in out.rglob("*"):
        if path.is_file():
            assert FAKE_KEY.encode() not in path.read_bytes(), path


def test_existing_source_is_never_overwritten(setup) -> None:
    root, released, _ = setup
    out = _build(root, released, LINES)
    with pytest.raises(FileExistsError):
        cc.build_context_text(out, root / "split.json", root / "Model.csv", released)
