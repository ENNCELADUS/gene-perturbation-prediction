#!/usr/bin/env python3
"""CLI: build verified Tx1-3B basal embedding caches (Wave 1 Phase B, Task 2).

Wires ``aivc_model.tx1_embed_cache.embed_lines`` to the real Tx1-3B encoder,
reusing the frozen Phase-A width-probe's model-loading logic
(``scripts/verify_tx1_obsm_width.py``: ``load_local_safetensors``,
``install_padding_metadata_fallback``, the ``loader_from_adata`` +
``Trainer.predict`` + ``output["cell_emb"]`` pattern) rather than
reimplementing it. Requires a CUDA GPU and the real Tahoe-100M / Tx1-3B
assets on the HPC; nothing here runs on the dev machine (see
``tests/test_tx1_embed_cache.py`` for the encoder-agnostic unit tests).

This CLI wires both Phase-A basal sources: the **Tahoe-100M DMSO** source
(``--shard-dir`` / ``--gene-metadata``) for the 38 Tahoe-sourced lines, and
the 4 Perturb-seq-sourced lines (the ``train_response_and_head`` role) via
``--perturbseq-source-config``, a JSON file mapping each Perturb-seq
``model_id`` to a per-source schema. Three of those four (Replogle K562,
Nadig Jurkat/HepG2) are h5ad files (an ``"h5ad_path"`` plus perturbation
column, control label, and Ensembl var column); the fourth (X-Atlas-Orion
HCT116) is parquet shards (a ``"shard_dir"``/``"shard_glob"`` plus a
gene-token-id-to-Ensembl metadata parquet, control label, and guide-QC
filter value). Each entry's optional ``"source_type"`` field
(``"h5ad"``, the default, or ``"xatlas_orion_parquet"``) selects which
schema it is parsed against. A JSON config was chosen over a repeatable
per-field flag because the 4 lines' schemas differ -- a flag per field per
line would not scale. Every in-scope manifest line with a non-Tahoe
``basal_source`` must have a configured entry before any GPU work begins;
an unconfigured line raises loudly, listing every missing ``model_id``,
rather than silently skipping it.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Mapping

# Codex P1-a: when this file is executed directly (``python
# scripts/build_tx1_basal_embeddings.py ...``), Python puts this file's own
# directory (``scripts/``) on ``sys.path``, not the repository root -- so
# ``from scripts.verify_tx1_obsm_width import ...`` below raises
# ModuleNotFoundError before any argument is even parsed, and
# ``from aivc_model...`` fails too unless this project's own packages are
# pip-installed into the interpreter running it (the HPC's dedicated
# ``.venv-tx1`` is not). Prepending the repo root and ``src/`` here makes
# both import families resolve without the caller having to set
# ``PYTHONPATH`` -- this must run before any of the local imports below.
_REPO_ROOT = Path(__file__).resolve().parents[1]
for _extra_path in (_REPO_ROOT, _REPO_ROOT / "src"):
    if str(_extra_path) not in sys.path:
        sys.path.insert(0, str(_extra_path))

import anndata as ad  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from aivc_model.gene_splits import sha256_file  # noqa: E402
from aivc_model.state_core import sha256_strings  # noqa: E402
from aivc_model.tx1_basal import load_line_manifest  # noqa: E402
from aivc_model.tx1_embed_cache import (  # noqa: E402
    MODEL_LABEL,
    EncoderFn,
    PerturbseqSource,
    PerturbseqSourceConfig,
    XatlasOrionSource,
    embed_lines,
    load_hvg_gene_order,
    verify_cache,
    write_run_manifest,
)
from scripts.verify_tx1_obsm_width import (  # noqa: E402
    install_padding_metadata_fallback,
    load_local_safetensors,
)

_LOGGER = logging.getLogger(__name__)

#: The basal source this CLI can embed without a ``--perturbseq-source-config``
#: entry (Global Constraint 1's documented enum value); every other
#: ``basal_source`` value requires a configured :class:`PerturbseqSource`.
_TAHOE_BASAL_SOURCE = "Tahoe-100M DMSO"

#: ``--perturbseq-source-config`` entry ``"source_type"`` values. ``"h5ad"``
#: is the default when an entry omits the field, so the 3 already-supported
#: h5ad sources' config entries need not add it.
_SOURCE_TYPE_H5AD = "h5ad"
_SOURCE_TYPE_XATLAS_PARQUET = "xatlas_orion_parquet"
_VALID_SOURCE_TYPES = frozenset({_SOURCE_TYPE_H5AD, _SOURCE_TYPE_XATLAS_PARQUET})

#: Required keys of an ``"h5ad"``-type entry, one-to-one with
#: :class:`PerturbseqSource`'s fields.
_H5AD_SOURCE_CONFIG_KEYS = (
    "h5ad_path",
    "perturbation_col",
    "control_label",
    "var_ensembl_col",
)

#: Required keys of an ``"xatlas_orion_parquet"``-type entry;
#: ``shard_glob``/``pass_guide_filter_value`` are optional, matching
#: :class:`XatlasOrionSource`'s defaults.
_XATLAS_SOURCE_CONFIG_REQUIRED_KEYS = (
    "shard_dir",
    "gene_metadata_path",
    "control_label",
)


def _build_tx1_encoder(
    model_dir: Path, batch_size: int, max_length: int
) -> tuple[EncoderFn, dict[str, object]]:
    """Build the real Tx1-3B ``EncoderFn``, reusing the width-probe's loader."""
    from composer import Trainer
    from tahoe_x1.utils.util import loader_from_adata

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required to embed Tx1-3B basal cells")
    install_padding_metadata_fallback()
    model, vocab, collator_config, load_report = load_local_safetensors(model_dir)
    trainer = Trainer(model=model, device="gpu")

    def encode(adata: ad.AnnData) -> np.ndarray:
        genes = adata.var["ensembl_id"].astype(str).tolist()
        gene_ids = np.asarray([vocab[gene] for gene in genes], dtype=int)
        loader = loader_from_adata(
            adata=adata,
            collator_cfg=collator_config,
            vocab=vocab,
            batch_size=batch_size,
            max_length=max_length,
            gene_ids=gene_ids,
            num_workers=0,
            prefetch_factor=None,
        )
        predictions = trainer.predict(loader, return_outputs=True)
        embeddings = torch.cat(
            [output["cell_emb"].detach().float().cpu() for output in predictions],
            dim=0,
        )
        return embeddings.numpy().astype(np.float32)

    return encode, load_report


def _verify_model_dir_matches_source_manifest(
    model_dir: Path, source_manifest: dict[str, object]
) -> None:
    """Defend against silently running on the wrong/corrupted Tx1 checkpoint.

    Resolves the ``tx-70m-merged``/``tahoe_x1_3b`` labeling bug (Global
    Constraint 2) at the source: before any GPU forward pass runs, every file
    ``--model-dir`` actually holds is hashed and compared against the frozen
    ``tx1_source_manifest.json`` this run was told to trust.
    """
    files = source_manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("--tx1-source-manifest has no non-empty 'files' entry")
    mismatched = [
        filename
        for filename, expected in files.items()
        if not isinstance(expected, dict)
        or sha256_file(model_dir / filename) != expected.get("sha256")
    ]
    if mismatched:
        raise ValueError(
            f"--model-dir does not match the registered Tx1 source manifest "
            f"for: {mismatched}"
        )


def _require_embedding_args(args: argparse.Namespace) -> None:
    names = (
        "model_dir",
        "shard_dir",
        "gene_metadata",
        "hvg_state_model_dir",
        "tx1_source_manifest",
    )
    missing = [
        f"--{name.replace('_', '-')}" for name in names if getattr(args, name) is None
    ]
    if missing:
        raise ValueError(
            f"missing required arguments (only optional with --verify-only): {missing}"
        )


def _load_perturbseq_source_config(path: Path) -> dict[str, PerturbseqSourceConfig]:
    """Load the per-``model_id`` Perturb-seq source config JSON.

    A JSON config was chosen over a repeatable per-field CLI flag because the
    4 Perturb-seq-sourced lines (Replogle K562, Nadig Jurkat/HepG2, X-Atlas
    HCT116) have differing schemas; see the module docstring. Each entry's
    ``"source_type"`` (default ``"h5ad"``) selects whether it is parsed as a
    :class:`PerturbseqSource` or a :class:`XatlasOrionSource`.

    Args:
        path: JSON file shaped ``{model_id: {source_type?, ...}}``, where an
            ``"h5ad"`` entry carries ``h5ad_path``/``perturbation_col``/
            ``control_label``/``var_ensembl_col`` and an
            ``"xatlas_orion_parquet"`` entry carries
            ``shard_dir``/``gene_metadata_path``/``control_label`` plus
            optional ``shard_glob``/``pass_guide_filter_value``.

    Returns:
        Per-``model_id`` :class:`PerturbseqSourceConfig` configs.

    Raises:
        ValueError: The file is not a JSON object, an entry is not an
            object, an entry names an unknown ``source_type``, or an entry
            is missing a key its ``source_type`` requires.
    """
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"--perturbseq-source-config at {path} must be a JSON object")
    sources: dict[str, PerturbseqSourceConfig] = {}
    for model_id, entry in raw.items():
        if not isinstance(entry, dict):
            raise ValueError(
                f"--perturbseq-source-config entry {model_id!r} must be a JSON object"
            )
        sources[str(model_id)] = _parse_source_entry(str(model_id), entry)
    return sources


def _parse_source_entry(
    model_id: str, entry: Mapping[str, object]
) -> PerturbseqSourceConfig:
    """Parse one ``--perturbseq-source-config`` entry, dispatched on ``source_type``."""
    source_type = str(entry.get("source_type", _SOURCE_TYPE_H5AD))
    if source_type not in _VALID_SOURCE_TYPES:
        raise ValueError(
            f"--perturbseq-source-config entry {model_id!r} has unknown "
            f"source_type {source_type!r}; expected one of "
            f"{sorted(_VALID_SOURCE_TYPES)}"
        )
    required_keys = (
        _H5AD_SOURCE_CONFIG_KEYS
        if source_type == _SOURCE_TYPE_H5AD
        else _XATLAS_SOURCE_CONFIG_REQUIRED_KEYS
    )
    missing = [key for key in required_keys if key not in entry]
    if missing:
        raise ValueError(
            f"--perturbseq-source-config entry {model_id!r} is missing keys: {missing}"
        )
    if source_type == _SOURCE_TYPE_H5AD:
        return PerturbseqSource(
            h5ad_path=Path(entry["h5ad_path"]),
            control_label=str(entry["control_label"]),
            perturbation_col=str(entry["perturbation_col"]),
            var_ensembl_col=str(entry["var_ensembl_col"]),
        )
    return XatlasOrionSource(
        shard_dir=Path(entry["shard_dir"]),
        gene_metadata_path=Path(entry["gene_metadata_path"]),
        control_label=str(entry["control_label"]),
        shard_glob=str(entry.get("shard_glob", "*.parquet")),
        pass_guide_filter_value=int(entry.get("pass_guide_filter_value", 1)),
    )


def _require_perturbseq_sources_configured(
    working: pd.DataFrame, perturbseq_sources: Mapping[str, PerturbseqSourceConfig]
) -> None:
    """Fail loudly, before any GPU work, if an in-scope line lacks a source.

    Args:
        working: The in-scope manifest slice (after ``--only-line``
            filtering).
        perturbseq_sources: Configured Perturb-seq sources, keyed by
            ``model_id``.

    Raises:
        ValueError: A ``basal_source`` other than Tahoe-100M DMSO appears in
            ``working`` without a matching ``perturbseq_sources`` entry.
    """
    non_tahoe = working[working["basal_source"] != _TAHOE_BASAL_SOURCE]
    unconfigured = sorted(
        set(non_tahoe["model_id"].astype(str)) - set(perturbseq_sources)
    )
    if unconfigured:
        raise ValueError(
            "the following in-scope lines have a non-Tahoe basal_source but "
            f"no --perturbseq-source-config entry: {unconfigured}"
        )


def _run_embedding(args: argparse.Namespace) -> dict[str, object]:
    """Build (or resume) every in-scope line's cache, then verify the result."""
    _require_embedding_args(args)
    manifest = load_line_manifest(args.line_manifest)
    only_lines = list(args.only_line) or None
    working = (
        manifest
        if only_lines is None
        else manifest[manifest["model_id"].isin(only_lines)]
    )
    perturbseq_sources = (
        _load_perturbseq_source_config(args.perturbseq_source_config)
        if args.perturbseq_source_config is not None
        else {}
    )
    _require_perturbseq_sources_configured(working, perturbseq_sources)
    source_manifest = json.loads(args.tx1_source_manifest.read_text())
    _verify_model_dir_matches_source_manifest(args.model_dir, source_manifest)
    _LOGGER.info("loading Tx1-3B checkpoint from %s", args.model_dir)
    encoder, load_report = _build_tx1_encoder(
        args.model_dir, args.batch_size, args.max_length
    )
    checkpoint_gene_order = load_hvg_gene_order(args.hvg_state_model_dir)
    _LOGGER.info("embedding %d in-scope line(s)", len(working["model_id"].unique()))
    line_entries = embed_lines(
        manifest,
        args.cache_dir,
        encoder=encoder,
        shard_dir=args.shard_dir,
        gene_metadata_path=args.gene_metadata,
        hvg_state_model_dir=args.hvg_state_model_dir,
        hvg_gene_symbol_col=args.hvg_gene_symbol_col,
        max_cells_per_line=args.max_cells_per_line,
        seed=args.seed,
        perturbseq_sources=perturbseq_sources,
        only_lines=only_lines,
    )
    config_snapshot = {
        "line_manifest_path": str(args.line_manifest),
        "line_manifest_sha256": sha256_file(args.line_manifest),
        "max_cells_per_line": args.max_cells_per_line,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "only_line": list(args.only_line),
        "hvg_gene_symbol_col": args.hvg_gene_symbol_col,
        "hvg_gene_order_sha256": sha256_strings(checkpoint_gene_order),
        "checkpoint_load_report": load_report,
        "perturbseq_source_config": (
            str(args.perturbseq_source_config)
            if args.perturbseq_source_config is not None
            else None
        ),
    }
    write_run_manifest(
        args.cache_dir,
        model_label=MODEL_LABEL,
        source_manifest=source_manifest,
        line_entries=line_entries,
        config_snapshot=config_snapshot,
    )
    _LOGGER.info("wrote run manifest; verifying cache at %s", args.cache_dir)
    # Codex P1-c: a --only-line shard must verify only its own in-scope
    # lines (only_lines=only_lines), not the full frozen manifest -- else
    # every shard's exit code reflects the *other* shards' lines being
    # absent, making exit codes useless for detecting real per-shard
    # failure. The full, unrestricted criterion still runs for an
    # unsharded invocation (only_lines is None) and for the explicit
    # --verify-only aggregation pass in main() below (which never passes
    # only_lines).
    return verify_cache(
        args.cache_dir,
        frozen_manifest_path=args.line_manifest,
        only_lines=only_lines,
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--line-manifest", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--shard-dir", type=Path)
    parser.add_argument("--gene-metadata", type=Path)
    parser.add_argument("--hvg-state-model-dir", type=Path)
    parser.add_argument("--hvg-gene-symbol-col", default="gene_symbol")
    parser.add_argument("--tx1-source-manifest", type=Path)
    parser.add_argument("--perturbseq-source-config", type=Path)
    parser.add_argument("--max-cells-per-line", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--only-line", action="append", default=[])
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Build (or verify) the configured Tx1 basal embedding cache."""
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    report = (
        verify_cache(args.cache_dir, frozen_manifest_path=args.line_manifest)
        if args.verify_only
        else _run_embedding(args)
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "verified":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
