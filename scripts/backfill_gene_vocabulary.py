#!/usr/bin/env python3
"""CLI: backfill ``gene_vocabulary.json`` for Phase C checkpoints that predate
``d576032`` (Wave 3 Codex gate P1-3 CARRY-FORWARD).

Two Phase C arms started training under code that predates the checkpoint
writer's ordered-vocabulary emission fix, so neither will ever produce
``gene_vocabulary.json``/``metadata.json``'s ``gene_vocabulary_sha256`` on its
own, and Phase D correctly refuses to authenticate a checkpoint that lacks
them (:func:`aivc_model.tx1_geneeffect_pipeline.verify_gene_vocabulary_authenticity`).
Re-training costs ~8.7 h per arm and is unnecessary: ``git diff --stat
4aca5ef..d576032 -- src/aivc_model/train.py`` is a pure 41-line addition that
never touches the vocabulary CONSTRUCTION path (``canonical_gene_order``/
``extra_genes``/the ``PerturbationVectorAdapter(...)`` call) -- only the
*write* of an artifact that construction path already determined. See
``.superpowers/sdd/phase-c/progress.md``'s "P1-3 CARRY-FORWARD RESOLVED"
section for the full reasoning this script implements.

**This reconstructs, it does not invent.** The ordered vocabulary comes from
running the SAME construction code ``aivc_model.train._build_e2e_model``
uses (via the same ``scripts.train_tx1_st_response`` assembly/split/adapter
sequence ``run_real_training`` uses, up to but never including the training
loop) -- never hand-assembled, sorted, or inferred here. Before trusting that
reconstruction, this cross-checks the recomputed train/val/test gene split
and source fingerprint against what the checkpoint's OWN existing
``metadata.json`` already recorded at original training time (``schema_version:
2``'s ``source_fingerprint``/``train_genes``/``val_genes``/``test_genes``
fields, written by ``aivc_model.prepare.ArtifactAuthority.metadata()``) --
if reconstruction cannot be proven to reproduce what this checkpoint was
actually trained with, this refuses to write anything rather than emit a
plausible-but-wrong file (a wrong-order vocabulary is worse than none: Phase
D would authenticate it and silently bind trainable weights to the wrong
gene symbols).

Usage (one arm; run on the SAME host/cache the arm trained against, e.g. the
HPC, under ``.venv-tx1`` with ``PYTHONPATH=src:.`` -- never on the local Mac,
which lacks the real Tx1 cache / STATE checkpoint)::

    PYTHONPATH=src:. .venv-tx1/bin/python scripts/backfill_gene_vocabulary.py \\
        --config configs/experiments/12_tx1_st_geneeffect/phase_c/tx1_arm.yaml \\
        --cache-dir data/tx1_basal_embeddings/v1 \\
        --line-manifest results/phase_a_tx1_20260724/cell_line_manifest.csv \\
        --response-cache-dir data/tx1_response_targets_cache \\
        --run-dir results/experiments/12_tx1_st_geneeffect/phase_c/runs/\\
tx1_st_geneeffect_phase_c_tx1_arm

Exits 0 (a clean no-op) if every targeted checkpoint already carries a
matching ``gene_vocabulary.json``/``gene_vocabulary_sha256``. Exits nonzero
with a named, actionable error if reconstruction cannot be trusted, or if an
existing artifact disagrees with the reconstruction (never overwritten).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from dataclasses import replace
from pathlib import Path
from typing import Mapping, Sequence

# Mirrors scripts/train_tx1_st_response.py's own bootstrap: executed
# directly, this file's own directory lands on sys.path, not the repo root,
# so `from aivc_model...`/`from scripts...` fail unless the packages are
# pip-installed (they are not, on the HPC's dedicated .venv-tx1).
_REPO_ROOT = Path(__file__).resolve().parents[1]
for _extra_path in (_REPO_ROOT, _REPO_ROOT / "src"):
    if str(_extra_path) not in sys.path:
        sys.path.insert(0, str(_extra_path))

from aivc_model import train as train_module  # noqa: E402
from aivc_model.gene_splits import FoldSpec, GeneAccessRecorder, sha256_file  # noqa: E402
from aivc_model.model import PerturbationVectorAdapter  # noqa: E402
from aivc_model.prepare import AivcConfig, GeneBags, load_config  # noqa: E402
from aivc_model.tx1_geneeffect_pipeline import (  # noqa: E402
    verify_gene_vocabulary_authenticity,
)
import scripts.train_tx1_st_response as train_tx1_cli  # noqa: E402

_LOGGER = logging.getLogger(__name__)

_CHECKPOINT_KINDS = ("best", "final")


class VocabularyBackfillError(RuntimeError):
    """Reconstruction cannot be trusted, or an existing artifact disagrees.

    Raised instead of ever writing a best-effort file: a wrong-order
    ``gene_vocabulary.json`` is worse than a missing one (P1-3 -- Phase D
    would authenticate it and silently bind trainable missing-gene weights
    to the wrong gene symbol).
    """


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments (mirrors train_tx1_st_response.py)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--line-manifest", type=Path, required=True)
    parser.add_argument(
        "--perturbseq-source-config",
        type=Path,
        default=train_tx1_cli._DEFAULT_PERTURBSEQ_SOURCE_CONFIG,
    )
    parser.add_argument(
        "--response-cache-dir",
        type=Path,
        required=True,
        help=(
            "The SAME --response-cache-dir the original run warmed and "
            "trained against. Required (not optional): omitting it forces "
            "assemble_and_project to re-materialize the raw per-line "
            "response matrices from scratch, the exact single-process "
            "memory blowup .superpowers/sdd/phase-c/progress.md's "
            "2026-07-26 incident already measured at ~620 GB RSS."
        ),
    )
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        choices=_CHECKPOINT_KINDS,
        default=list(_CHECKPOINT_KINDS),
        help="Which of models/{best,final} to backfill (default: both).",
    )
    return parser.parse_args(argv)


def _load_existing_metadata(checkpoint_dir: Path) -> dict[str, object]:
    metadata_path = checkpoint_dir / "metadata.json"
    if not metadata_path.is_file():
        raise VocabularyBackfillError(
            f"{metadata_path} does not exist -- cannot backfill a checkpoint "
            "with no metadata.json to record gene_vocabulary_sha256 into "
            "(is --run-dir/--checkpoints pointing at a real, completed "
            "Phase C checkpoint directory?)"
        )
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise VocabularyBackfillError(f"{metadata_path} is not valid JSON") from error
    if not isinstance(metadata, dict):
        raise VocabularyBackfillError(f"{metadata_path} must contain a JSON object")
    return metadata


def _assert_schema_and_fingerprint(
    metadata: Mapping[str, object],
    metadata_path: Path,
    expected_fingerprint: str,
) -> None:
    """Cheap checks, before any expensive assembly: right schema, right run.

    ``source_fingerprint`` hashes the line manifest, source config, cache
    manifest, and every referenced response source
    (``train_tx1_cli._source_fingerprint``) -- a mismatch here means the
    operator pointed this invocation's ``--cache-dir``/``--line-manifest``/
    ``--perturbseq-source-config``/``--config`` at different inputs than
    this checkpoint was actually trained against, before this function ever
    reads a single cell.
    """
    if metadata.get("schema_version") != 2:
        raise VocabularyBackfillError(
            f"{metadata_path} has no schema_version==2 (authority metadata) "
            "-- this is not an audited Phase C checkpoint this tool knows "
            "how to backfill"
        )
    recorded_fingerprint = metadata.get("source_fingerprint")
    if recorded_fingerprint != expected_fingerprint:
        raise VocabularyBackfillError(
            f"{metadata_path}'s recorded source_fingerprint "
            f"({recorded_fingerprint!r}) does not match the fingerprint "
            f"computed from THIS invocation's --cache-dir/--line-manifest/"
            f"--perturbseq-source-config ({expected_fingerprint!r}); wrong "
            "inputs for this checkpoint -- refusing to reconstruct against "
            "the wrong data"
        )


def _assert_split_authority_hash_if_present(
    run_dir: Path, metadata: Mapping[str, object], metadata_path: Path
) -> None:
    """If this run's own split-authority CSV is still on disk, it must agree
    with what the checkpoint recorded (`ArtifactAuthority.canonical_split_
    sha256`). A soft check (warns, does not fail) when the CSV/sidecar is
    absent -- the metadata-based checks below are already sufficient on
    their own.
    """
    manifest_path, sha_path = train_tx1_cli._split_authority_paths(run_dir)
    if not manifest_path.is_file() or not sha_path.is_file():
        _LOGGER.warning(
            "%s/%s not found; skipping the split-authority-CSV cross-check "
            "(the metadata-recorded train/val/test genes checked below are "
            "already sufficient)",
            manifest_path,
            sha_path,
        )
        return
    observed = sha256_file(manifest_path)
    recorded = metadata.get("canonical_split_sha256")
    if observed != recorded:
        raise VocabularyBackfillError(
            f"{manifest_path}'s current sha256 ({observed}) does not match "
            f"{metadata_path}'s recorded canonical_split_sha256 ({recorded}) "
            "-- this run's own split-authority artifact was modified after "
            "training, or this is the wrong run directory; refusing to "
            "reconstruct"
        )


def _assert_fold_matches_metadata(
    fold: FoldSpec, metadata: Mapping[str, object], metadata_path: Path
) -> None:
    """The core safety net: the recomputed fold must reproduce EXACTLY the
    train/val/test genes this checkpoint's own metadata.json already
    recorded (``ArtifactAuthority.metadata()``'s ``{train,val,test}_genes``).

    This is what makes reconstruction "verified, not assumed" (the ledger's
    own phrase): if the recomputation disagrees, the vocabulary this tool
    would build does NOT match what the checkpoint was actually trained
    with, and this refuses rather than guess.
    """
    roles = (
        ("train_genes", fold.train_genes),
        ("val_genes", fold.val_genes),
        ("test_genes", fold.test_genes),
    )
    for key, recomputed in roles:
        recorded = metadata.get(key)
        if not isinstance(recorded, list):
            raise VocabularyBackfillError(
                f"{metadata_path} has no usable {key!r} field -- cannot "
                "verify that reconstruction reproduces this checkpoint's "
                "own recorded split"
            )
        if sorted(str(gene) for gene in recorded) != sorted(
            str(gene) for gene in recomputed
        ):
            raise VocabularyBackfillError(
                f"recomputed {key} does not match {metadata_path}'s recorded "
                f"{key} -- reconstruction would NOT reproduce the "
                "vocabulary this checkpoint was actually trained with; "
                "refusing to proceed (investigate --config/--cache-dir/"
                "--line-manifest/--perturbseq-source-config/--response-"
                "cache-dir rather than overriding this check)"
            )


def _split_and_cross_check(
    config: AivcConfig,
    bags: GeneBags,
    run_dir: Path,
    checkpoint_metadata: Mapping[str, Mapping[str, object]],
) -> FoldSpec:
    """Recompute the fold (``_split_fold``, the real function ``run_real_
    training`` calls) and verify it against every checkpoint's own recorded
    split before anything downstream trusts it.

    Raises:
        VocabularyBackfillError: The recomputed split disagrees with any
            checkpoint's own recorded ``train_genes``/``val_genes``/
            ``test_genes``.
    """
    fold = train_tx1_cli._split_fold(bags.genes, config)
    for kind, metadata in checkpoint_metadata.items():
        metadata_path = run_dir / "models" / kind / "metadata.json"
        _assert_fold_matches_metadata(fold, metadata, metadata_path)
    return fold


def _authorized_views(
    config: AivcConfig, bags: GeneBags, fold: FoldSpec, run_dir: Path
) -> tuple[AivcConfig, GeneBags, GeneBags, tuple[str, ...]]:
    """Build the SAME authorized ``train_data``/``val_data`` views and
    ``canonical_gene_order`` ``run_real_training`` builds
    (``GeneBags.for_genes``/``for_prediction_genes``, ``canonical_gene_order
    = tuple(str(gene).upper() for gene in data.genes)`` from the FULL,
    unsplit authorized view -- NOT ``train_data.genes``/``val_data.genes``
    alone, which would silently drop the test-fold genes an ``esm2``
    tokenizer's canonical ordering needs), and points ``config.cv`` at this
    run's own split-authority files exactly as ``run_real_training`` does
    before calling ``_build_e2e_model``.
    """
    manifest_path, sha_path = train_tx1_cli._split_authority_paths(run_dir)
    config = replace(
        config,
        cv=replace(
            config.cv,
            outer_split_manifest=manifest_path,
            outer_split_sha256_file=sha_path,
        ),
    )
    data = replace(bags, access_recorder=GeneAccessRecorder(fold))
    train_data = data.for_genes(fold.train_genes, stage="adapter_fit")
    val_data = data.for_prediction_genes(
        fold.val_genes,
        stage="early_stopping_prediction_only",
        generation_targets=True,
    )
    canonical_gene_order = tuple(str(gene).upper() for gene in data.genes)
    return config, train_data, val_data, canonical_gene_order


def reconstruct_vocabulary_genes(
    config: AivcConfig,
    bags: GeneBags,
    run_dir: Path,
    checkpoint_metadata: Mapping[str, Mapping[str, object]],
) -> list[str]:
    """Reconstruct the exact ordered ``PerturbationVectorAdapter`` vocabulary
    through the real construction path, with NO training performed.

    Mirrors ``scripts.train_tx1_st_response.run_real_training`` up to (but
    never including) the training loop, then constructs the model through
    ``aivc_model.train._build_e2e_model`` -- the SAME function
    ``_run_audited_training`` calls -- rather than re-deriving the
    vocabulary's order independently.

    Args:
        checkpoint_metadata: ``{checkpoint_kind: existing metadata.json
            dict}``, cross-checked by :func:`_split_and_cross_check`.

    Raises:
        VocabularyBackfillError: The recomputed split disagrees with any
            checkpoint's own recorded split, or ``config.state.gene_
            tokenizer`` does not produce a ``PerturbationVectorAdapter``
            (this backfill only recovers THAT adapter's positional-
            vocabulary hazard).
    """
    fold = _split_and_cross_check(config, bags, run_dir, checkpoint_metadata)
    config, train_data, val_data, canonical_gene_order = _authorized_views(
        config, bags, fold, run_dir
    )
    model = train_module._build_e2e_model(
        config,
        train_data,
        extra_genes=(*fold.val_genes, *fold.test_genes),
        canonical_gene_order=canonical_gene_order,
        emit_checkpoint_output=False,
    )
    perturbations = model.perturbations
    if not isinstance(perturbations, PerturbationVectorAdapter):
        raise VocabularyBackfillError(
            f"config.state.gene_tokenizer={config.state.gene_tokenizer!r} "
            f"produced {type(perturbations).__name__}, not "
            "PerturbationVectorAdapter -- this backfill only recovers the "
            "positional-vocabulary hazard for PerturbationVectorAdapter "
            "checkpoints (Esm2PerturbationAdapter has no such hazard: see "
            "_save_model_checkpoint's own docstring)"
        )
    train_module._require_perturbation_vocabulary_coverage(
        model, train_data, val_data, fold, None
    )
    return list(perturbations.genes)


def _vocabulary_content(genes: Sequence[str]) -> str:
    """Byte-for-byte the same serialization ``_save_model_checkpoint`` uses."""
    return json.dumps(list(genes), indent=2) + "\n"


class _CheckpointPlan:
    """What (if anything) needs writing for one checkpoint dir.

    Split from the write itself so every checkpoint's plan can be validated
    -- and any conflict raised -- BEFORE any checkpoint's files are touched.
    A conflict on ``final`` must not leave ``best`` half-applied, and a
    conflict on the vocabulary file must not still apply the metadata write
    (or vice versa): see :func:`backfill_run`, which builds every plan first
    and only then applies any of them.
    """

    def __init__(
        self, directory: Path, write_vocab: bool, write_metadata: bool
    ) -> None:
        self.directory = directory
        self.write_vocab = write_vocab
        self.write_metadata = write_metadata

    @property
    def changed(self) -> bool:
        return self.write_vocab or self.write_metadata


def _plan_checkpoint_update(
    checkpoint_dir: Path, content: str, digest: str
) -> _CheckpointPlan:
    """Validate (never write) whether ``content``/``digest`` can be safely
    applied to ``checkpoint_dir``.

    Raises:
        VocabularyBackfillError: An existing ``gene_vocabulary.json``'s
            content, or an existing recorded ``gene_vocabulary_sha256``,
            differs from the reconstruction.
    """
    vocabulary_path = checkpoint_dir / "gene_vocabulary.json"
    write_vocab = True
    if vocabulary_path.is_file():
        existing = vocabulary_path.read_text(encoding="utf-8")
        if existing != content:
            raise VocabularyBackfillError(
                f"{vocabulary_path} already exists with DIFFERENT content "
                "than the reconstructed vocabulary; refusing to overwrite a "
                "possibly-authoritative file -- investigate before removing "
                "it manually"
            )
        write_vocab = False
    metadata = _load_existing_metadata(checkpoint_dir)
    existing_digest = metadata.get("gene_vocabulary_sha256")
    write_metadata = True
    if existing_digest is not None:
        if existing_digest != digest:
            raise VocabularyBackfillError(
                f"{checkpoint_dir / 'metadata.json'} already records "
                f"gene_vocabulary_sha256={existing_digest!r}, which does not "
                f"match the reconstructed vocabulary's sha256={digest!r}; "
                "refusing to overwrite a differing recorded hash"
            )
        write_metadata = False
    return _CheckpointPlan(checkpoint_dir, write_vocab, write_metadata)


def _apply_checkpoint_update(plan: _CheckpointPlan, content: str, digest: str) -> None:
    """Apply an already-validated plan. Never call this before every plan in
    the same run has been validated (see :func:`backfill_run`)."""
    if plan.write_vocab:
        vocabulary_path = plan.directory / "gene_vocabulary.json"
        vocabulary_path.parent.mkdir(parents=True, exist_ok=True)
        vocabulary_path.write_text(content, encoding="utf-8")
    if plan.write_metadata:
        metadata = _load_existing_metadata(plan.directory)
        metadata["gene_vocabulary_sha256"] = digest
        (plan.directory / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
        )


def _load_and_cross_check_metadata(
    checkpoint_dirs: Mapping[str, Path],
    run_dir: Path,
    *,
    cache_dir: Path,
    line_manifest: Path,
    perturbseq_source_config: Path,
) -> dict[str, dict[str, object]]:
    """Load every checkpoint's existing metadata.json and cross-check its
    schema/fingerprint/split-authority-hash, before anything expensive
    (reconstruction) runs.

    Raises:
        VocabularyBackfillError: A checkpoint directory is incomplete, or
            any cross-check fails.
    """
    for kind, checkpoint_dir in checkpoint_dirs.items():
        if not (checkpoint_dir / "pytorch_model.bin").is_file():
            raise VocabularyBackfillError(
                f"{checkpoint_dir / 'pytorch_model.bin'} does not exist -- "
                f"is this a completed run ({kind} checkpoint missing)?"
            )
    checkpoint_metadata = {
        kind: _load_existing_metadata(directory)
        for kind, directory in checkpoint_dirs.items()
    }
    expected_fingerprint = train_tx1_cli._source_fingerprint(
        line_manifest, perturbseq_source_config, cache_dir
    )
    for kind, directory in checkpoint_dirs.items():
        _assert_schema_and_fingerprint(
            checkpoint_metadata[kind],
            directory / "metadata.json",
            expected_fingerprint,
        )
        _assert_split_authority_hash_if_present(
            run_dir, checkpoint_metadata[kind], directory / "metadata.json"
        )
    return checkpoint_metadata


def backfill_run(
    config: AivcConfig,
    bags: GeneBags,
    *,
    run_dir: Path,
    cache_dir: Path,
    line_manifest: Path,
    perturbseq_source_config: Path,
    checkpoints: Sequence[str] = _CHECKPOINT_KINDS,
) -> dict[str, bool]:
    """Backfill ``gene_vocabulary.json``/``metadata.json`` for every requested
    checkpoint kind under ``run_dir/models/<kind>``.

    A self-contained, safe entry point: every cross-check
    (schema/fingerprint, split-authority hash, recomputed-split-vs-recorded)
    runs here regardless of what a caller already checked, so calling this
    directly (as the test suite does, bypassing ``main()``'s argv/CLI
    plumbing) is exactly as safe as the real CLI invocation.

    Args:
        bags: The SAME two-view ``GeneBags``
            ``scripts.train_tx1_st_response.assemble_and_project`` would
            produce for ``config`` (real data on a real run; a synthetic
            fixture in tests).
        cache_dir: The Tx1 embedding cache dir used to produce ``bags`` (only
            for the ``source_fingerprint`` cross-check -- not re-read here).

    Returns:
        ``{checkpoint_kind: changed}`` -- ``changed=False`` for every kind
        means this call was a complete no-op (already correctly backfilled).

    Raises:
        VocabularyBackfillError: Reconstruction cannot be trusted, or an
            existing artifact disagrees with the reconstruction.
    """
    checkpoint_dirs = {kind: run_dir / "models" / kind for kind in checkpoints}
    checkpoint_metadata = _load_and_cross_check_metadata(
        checkpoint_dirs,
        run_dir,
        cache_dir=cache_dir,
        line_manifest=line_manifest,
        perturbseq_source_config=perturbseq_source_config,
    )
    genes = reconstruct_vocabulary_genes(config, bags, run_dir, checkpoint_metadata)
    content = _vocabulary_content(genes)
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()

    # Validate EVERY checkpoint's plan before applying ANY of them: a
    # conflict on one checkpoint (or on just the vocabulary file, or just
    # the metadata hash) must never leave another checkpoint -- or this
    # one's other artifact -- half-applied.
    plans = {
        kind: _plan_checkpoint_update(directory, content, digest)
        for kind, directory in checkpoint_dirs.items()
    }
    for kind, plan in plans.items():
        _apply_checkpoint_update(plan, content, digest)
        verify_gene_vocabulary_authenticity(
            checkpoint_dirs[kind] / "gene_vocabulary.json",
            checkpoint_dirs[kind] / "pytorch_model.bin",
            config.state.backend,
        )
    return {kind: plan.changed for kind, plan in plans.items()}


def main(argv: Sequence[str] | None = None) -> None:
    """Assemble Phase C data for one arm's completed run, then backfill it."""
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = load_config(args.config)
    train_tx1_cli.validate_config_shape(config, check_paths=True)

    # Fail fast, before assemble_and_project's expensive (and, under a cold
    # --response-cache-dir, memory-heavy -- see --response-cache-dir's own
    # help text) raw-source read: if the operator pointed this invocation at
    # different inputs than this checkpoint actually trained against, these
    # cheap file-only checks already disagree. backfill_run repeats them
    # internally regardless (it must remain safe to call directly, as the
    # test suite does).
    checkpoint_dirs = {
        kind: args.run_dir / "models" / kind for kind in args.checkpoints
    }
    _load_and_cross_check_metadata(
        checkpoint_dirs,
        args.run_dir,
        cache_dir=args.cache_dir,
        line_manifest=args.line_manifest,
        perturbseq_source_config=args.perturbseq_source_config,
    )

    bags = train_tx1_cli.assemble_and_project(
        config,
        cache_dir=args.cache_dir,
        line_manifest=args.line_manifest,
        perturbseq_source_config=args.perturbseq_source_config,
        response_cache_dir=args.response_cache_dir,
    )
    changed = backfill_run(
        config,
        bags,
        run_dir=args.run_dir,
        cache_dir=args.cache_dir,
        line_manifest=args.line_manifest,
        perturbseq_source_config=args.perturbseq_source_config,
        checkpoints=args.checkpoints,
    )
    print(json.dumps({"run_dir": str(args.run_dir), "changed": changed}, indent=2))


if __name__ == "__main__":
    main()
