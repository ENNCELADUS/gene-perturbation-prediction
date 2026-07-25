#!/usr/bin/env python3
"""CLI: train one Phase C ST response arm (Wave 2 Task 7).

Wires Task 5's multi-line observed-response assembly
(``aivc_model.tx1_response_data.assemble_train_response_gene_bags``) to the
existing training entrypoint (``aivc_model.train.run_training``), for either
Phase C arm:

- ``hvg_arm.yaml``: ST's input is gene/HVG space (``state.input_view:
  checkpoint_hvg``), the same view as the response-encoder target -- the
  "encoder-unseen" attribution control.
- ``tx1_arm.yaml``: ST's input is the Tx1-3B basal embedding
  (``state.input_view: obsm``), distinct from the gene-space target.

Both arms train on the frozen manifest's 4 ``train_response_and_head`` lines
(C4/C5); which arm runs is entirely determined by ``--config``.

Runnable directly -- no ``PYTHONPATH`` needed. This file prepends the repo
root and ``src/`` to ``sys.path`` itself (mirroring
``scripts/build_tx1_basal_embeddings.py``'s fix for Phase B's Codex P1-a
finding: a plain ``python scripts/train_tx1_st_response.py`` otherwise fails
on ``from aivc_model...`` because the repo is not pip-installed into the
HPC's dedicated ``.venv-tx1``).

Invocation (dry run -- validates paths and widths, trains nothing)::

    .venv-tx1/bin/python scripts/train_tx1_st_response.py \\
        --config configs/experiments/12_tx1_st_geneeffect/phase_c/tx1_arm.yaml \\
        --cache-dir data/tx1_basal_embeddings/v1 \\
        --line-manifest results/phase_a_tx1_20260724/cell_line_manifest.csv \\
        --dry-run

Invocation (real training; run under ``accelerate launch --num_processes 4``
to satisfy ``required_world_size: 4`` -- see both Phase C configs)::

    accelerate launch --num_processes 4 scripts/train_tx1_st_response.py \\
        --config configs/experiments/12_tx1_st_geneeffect/phase_c/tx1_arm.yaml \\
        --cache-dir data/tx1_basal_embeddings/v1 \\
        --line-manifest results/phase_a_tx1_20260724/cell_line_manifest.csv

KNOWN LIMITATION (flagged, not fixed, in Task 7's report): the fold/gene
split below is name-keyed, like every other caller of ``GeneBags.for_genes``
/``for_prediction_genes``/``SealedGeneBags`` in this codebase. Task 5's
assembly keeps one bag per (gene, line) pair, so if the SAME gene symbol was
tested in more than one of the four training lines -- expected once real
multi-line data is assembled, since large Perturb-seq panels commonly
overlap -- the name-keyed lookup inside ``for_genes`` silently keeps only
one matching bag per requested name (a dict comprehension collapses same-
name keys to the last-seen index), dropping the rest with no error. This is
a pre-existing property of ``GeneBags.for_genes``/``SealedGeneBags``
(unchanged here); see ``tests/test_train_tx1_st_response.py``'s
``test_for_genes_silently_drops_a_duplicate_gene_name_across_lines`` for a
characterization test against the real implementation.
``run_real_training`` logs a WARNING naming the affected bag count
(``_warn_on_duplicate_gene_names``) so a real run at least surfaces this,
but does not (and, without a design decision about how a gene's role should
behave when it spans lines, safely cannot) change which bag survives.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

# Codex P1-a (Phase B): executed directly, this file's own directory lands on
# sys.path, not the repo root, so ``from aivc_model...`` fails unless the
# packages are pip-installed (they are not, on the HPC's .venv-tx1). Prepend
# the repo root and ``src/`` here, before any local import, exactly as
# scripts/build_tx1_basal_embeddings.py already does.
_REPO_ROOT = Path(__file__).resolve().parents[1]
for _extra_path in (_REPO_ROOT, _REPO_ROOT / "src"):
    if str(_extra_path) not in sys.path:
        sys.path.insert(0, str(_extra_path))

from sklearn.model_selection import train_test_split  # noqa: E402

from aivc_model import train as train_module  # noqa: E402
from aivc_model.distributed import run_rank_zero_or_raise  # noqa: E402
from aivc_model.gene_splits import (  # noqa: E402
    FoldSpec,
    GeneAccessRecorder,
    sha256_file,
)
from aivc_model.gwps_cache import sha256_strings  # noqa: E402
from aivc_model.prepare import (  # noqa: E402
    AivcConfig,
    GeneBags,
    SealedGeneBags,
    load_config,
)
from aivc_model.tx1_embed_cache import EMBEDDING_WIDTH  # noqa: E402
from aivc_model.tx1_response_data import assemble_train_response_gene_bags  # noqa: E402

_LOGGER = logging.getLogger(__name__)

#: Task 5's already-verified 4-line Perturb-seq source config (Phase B).
_DEFAULT_PERTURBSEQ_SOURCE_CONFIG = _REPO_ROOT / (
    "configs/experiments/12_tx1_st_geneeffect/phase_b/perturbseq_sources.json"
)

#: ``state.input_view`` values this CLI knows how to project onto (mirrors
#: ``prepare._VALID_STATE_INPUT_VIEWS``, not imported to avoid depending on
#: a private module constant).
_INPUT_VIEW_OBSM = "obsm"
_INPUT_VIEW_CHECKPOINT_HVG = "checkpoint_hvg"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments (mirrors build_tx1_basal_embeddings.py)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--line-manifest", type=Path, required=True)
    parser.add_argument(
        "--perturbseq-source-config",
        type=Path,
        default=_DEFAULT_PERTURBSEQ_SOURCE_CONFIG,
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="Training output dir; defaults to <data.output_dir>/runs/<train.run_id>.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _project_to_configured_view(bags: GeneBags, input_view: str) -> GeneBags:
    """Project Task 5's two-view assembly onto one arm's ST input space.

    Task 5's ``assemble_train_response_gene_bags`` always returns a
    two-view ``GeneBags``: ``input_bags``/``control_input`` are Tx1
    embedding space (2560-d), ``target_bags``/``control_target`` are
    gene/HVG space (2000-d) -- exactly what the Tx1 arm
    (``input_view == "obsm"``) needs, unchanged.

    The HVG arm (``input_view == "checkpoint_hvg"``) instead needs ST's
    input to equal its response-encoder target: repoint the input fields at
    the SAME gene-space arrays used as the target (C1's "same Python
    object" legacy convention -- ``target_bags=None`` afterwards makes
    ``effective_target_bags`` fall back to the identical ``input_bags``
    object), so ST never sees a Tx1 embedding in this arm. Raises
    ``ValueError`` for any other ``input_view``.
    """
    if input_view == _INPUT_VIEW_OBSM:
        return bags
    if input_view != _INPUT_VIEW_CHECKPOINT_HVG:
        raise ValueError(
            f"unsupported state.input_view {input_view!r}; expected "
            f"{_INPUT_VIEW_OBSM!r} or {_INPUT_VIEW_CHECKPOINT_HVG!r}"
        )
    return replace(
        bags,
        input_bags=bags.effective_target_bags,
        control_input=bags.effective_control_target,
        input_dim=bags.effective_target_dim,
        feature_names=bags.effective_target_feature_names,
        feature_fill_values=bags.effective_target_fill_values,
        latent_bags=bags.effective_target_bags,
        control_latent=bags.effective_control_target,
        latent_dim=bags.effective_target_dim,
        target_bags=None,
        control_target=None,
        target_dim=None,
        target_feature_names=None,
        target_fill_values=None,
    )


def assemble_and_project(
    config: AivcConfig,
    *,
    cache_dir: Path,
    line_manifest: Path,
    perturbseq_source_config: Path,
) -> GeneBags:
    """Assemble the 4-line response ``GeneBags`` and project it onto ``config``'s arm.

    Raises:
        ValueError: ``state.model_dir`` is unset, or any error Task 5's
            assembly itself raises (missing cache/source entries, gene
            order mismatch, unsupported ``input_view``, ...).
    """
    if config.state.model_dir is None:
        raise ValueError(
            "state.model_dir is required (the checkpoint dir supplying the "
            "HVG gene order)"
        )
    _LOGGER.info(
        "assembling response GeneBags: cache_dir=%s line_manifest=%s "
        "perturbseq_source_config=%s l2_normalize=%s",
        cache_dir,
        line_manifest,
        perturbseq_source_config,
        config.state.l2_normalize_input,
    )
    bags = assemble_train_response_gene_bags(
        cell_line_manifest_path=line_manifest,
        tx1_cache_dir=cache_dir,
        hvg_state_model_dir=config.state.model_dir,
        perturbseq_sources_path=perturbseq_source_config,
        l2_normalize=config.state.l2_normalize_input,
    )
    _LOGGER.info(
        "assembled %d bags (input_dim=%d, target_dim=%d); projecting onto "
        "input_view=%s",
        len(bags.genes),
        bags.input_dim,
        bags.effective_target_dim,
        config.state.input_view,
    )
    return _project_to_configured_view(bags, config.state.input_view)


def validate_config_shape(config: AivcConfig, *, check_paths: bool) -> None:
    """Validate ``config``'s own fields, independent of any assembled data.

    Catches the misconfigurations item 6 of the Task 7 brief calls out --
    config loading does no schema validation, so a misspelled or forgotten
    key silently takes its default. ``check_paths=False`` skips filesystem
    existence checks (for validating the checked-in configs' VALUES in an
    environment where the referenced, gitignored checkpoint assets are not
    present, e.g. CI); the real CLI always uses ``check_paths=True``.

    Raises:
        ValueError: A required section is missing, a width is wrong, or
            (when ``check_paths``) a configured path does not exist.
    """
    if config.response_encoder is None:
        raise ValueError("response_encoder config is required")
    if config.response_encoder.latent_dim != 128:
        raise ValueError(
            "response_encoder.latent_dim must be 128; got "
            f"{config.response_encoder.latent_dim}"
        )
    if config.gmm.trainable is not True:
        raise ValueError("gmm.trainable must be true for this training entrypoint")
    # ST's output is target/gene space in both arms (C9), so this must equal
    # the response encoder's own width -- not a hardcoded literal, since a
    # different released checkpoint could use a different HVG panel size.
    if config.state.output_dim != config.response_encoder.input_dim:
        raise ValueError(
            "state.output_dim must equal response_encoder.input_dim (ST's "
            "output is target/gene space in both arms -- C9); got "
            f"state.output_dim={config.state.output_dim}, "
            f"response_encoder.input_dim={config.response_encoder.input_dim}"
        )
    if config.state.input_view == _INPUT_VIEW_OBSM:
        if config.state.input_dim != EMBEDDING_WIDTH:
            raise ValueError(
                f"state.input_view=obsm requires state.input_dim={EMBEDDING_WIDTH} "
                f"(the Tx1 embedding width); got {config.state.input_dim}"
            )
        if not config.data.state_embed_key:
            raise ValueError(
                "state.input_view=obsm requires a non-empty data.state_embed_key"
            )
    elif config.state.input_view == _INPUT_VIEW_CHECKPOINT_HVG:
        # This arm's ST input equals its own response-encoder target (C1's
        # legacy single-view convention), so the two widths must agree.
        if config.state.input_dim != config.response_encoder.input_dim:
            raise ValueError(
                "state.input_view=checkpoint_hvg requires state.input_dim to "
                "equal response_encoder.input_dim; got "
                f"state.input_dim={config.state.input_dim}, "
                f"response_encoder.input_dim={config.response_encoder.input_dim}"
            )
    else:
        raise ValueError(f"unsupported state.input_view: {config.state.input_view!r}")
    for weight_name in (
        "pred_c_weight",
        "obs_c_weight",
        "occupancy_weight",
        "gmm_nll_weight",
        "pred_rank_weight",
    ):
        weight = getattr(config.loss, weight_name)
        if weight != 0.0:
            raise ValueError(
                f"loss.{weight_name} must be 0.0: these GeneBags carry no "
                "GeneEffect label (Phase C trains ST's response prediction "
                "only), and F.mse_loss(*, NaN) poisons total_loss "
                f"regardless of weight; got {weight}"
            )
    if config.train.gene_batch_size != 1:
        raise ValueError(
            "train.gene_batch_size must be 1 for this wave: gene_batch_size "
            "> 1 routes through model.py::_forward_gene_batch, a separate "
            "code path this wave's tests never exercise against an all-NaN "
            f"GeneEffect label; got {config.train.gene_batch_size}"
        )
    if check_paths:
        _validate_configured_paths_exist(config)


def _validate_configured_paths_exist(config: AivcConfig) -> None:
    if config.state.model_dir is None or not Path(config.state.model_dir).is_dir():
        raise ValueError(f"state.model_dir not found: {config.state.model_dir}")
    if config.state.backend == "state_checkpoint" and (
        config.state.checkpoint_path is None
        or not Path(config.state.checkpoint_path).is_file()
    ):
        raise ValueError(
            f"state.checkpoint_path not found: {config.state.checkpoint_path}"
        )
    # These two are optional, so only checked when actually set.
    for label, path in (
        ("state.warm_start_from", config.state.warm_start_from),
        ("state.known_perturbation_vectors", config.state.known_perturbation_vectors),
    ):
        if path is not None and not Path(path).is_file():
            raise ValueError(f"{label} not found: {path}")


def validate_config_against_bags(config: AivcConfig, bags: GeneBags) -> None:
    """Cross-check ``config``'s declared widths against the assembled data.

    Raises:
        ValueError: A configured width disagrees with what
            ``assemble_and_project`` actually produced.
    """
    if config.response_encoder is not None:
        if config.response_encoder.input_dim != bags.effective_target_dim:
            raise ValueError(
                "response_encoder.input_dim "
                f"{config.response_encoder.input_dim} != assembled target "
                f"width {bags.effective_target_dim}"
            )
    if config.state.input_dim is not None and config.state.input_dim != bags.input_dim:
        raise ValueError(
            f"state.input_dim {config.state.input_dim} != assembled input "
            f"width {bags.input_dim}"
        )
    if (
        config.state.output_dim is not None
        and config.state.output_dim != bags.effective_target_dim
    ):
        raise ValueError(
            f"state.output_dim {config.state.output_dim} != assembled "
            f"target width {bags.effective_target_dim}"
        )


def _split_fold(genes: Sequence[str], config: AivcConfig) -> FoldSpec:
    """Deterministic, non-stratified train/val/test split over distinct gene names.

    Phase C's assembled ``GeneBags`` carry no GeneEffect label (``y`` is an
    all-NaN placeholder), so -- unlike exp05's label-stratified canonical
    splits -- this is a plain partition of distinct gene names driven by
    ``config.split``'s fractions/seed.

    See the module docstring's KNOWN LIMITATION note: the resulting
    ``FoldSpec`` is a set of gene NAMES, and ``GeneBags.for_genes`` (used to
    build ``train_data``/``val_data`` from it) looks names up in a
    ``{name: index}`` dict, so a gene tested in more than one line only
    contributes one line's bag once split this way.
    """
    unique_genes = sorted({str(gene).upper() for gene in genes})
    if len(unique_genes) < 3:
        raise ValueError(
            f"need at least 3 distinct genes to split train/val/test; got "
            f"{len(unique_genes)}"
        )
    split = config.split
    train_genes, remainder = train_test_split(
        unique_genes,
        train_size=split.train_fraction,
        random_state=split.random_state,
        shuffle=True,
    )
    remainder_total = split.val_fraction + split.test_fraction
    val_share = split.val_fraction / remainder_total if remainder_total > 0 else 0.5
    if len(remainder) < 2:
        val_genes, test_genes = remainder, []
    else:
        val_genes, test_genes = train_test_split(
            remainder,
            train_size=val_share,
            random_state=split.random_state,
            shuffle=True,
        )
    return FoldSpec(
        outer_fold=0,
        train_genes=tuple(train_genes),
        val_genes=tuple(val_genes),
        test_genes=tuple(test_genes),
    )


def _source_fingerprint(line_manifest: Path, perturbseq_source_config: Path) -> str:
    """Hash the data-selection inputs (not the trained model) for provenance."""
    return sha256_strings(
        [sha256_file(line_manifest), sha256_file(perturbseq_source_config)]
    )


def _default_run_dir(config: AivcConfig) -> Path:
    run_id = config.train.run_id or "tx1_st_response"
    return Path(config.data.output_dir) / "runs" / run_id


def _warn_on_duplicate_gene_names(genes: np.ndarray) -> None:
    """Surface the KNOWN LIMITATION (module docstring) instead of relying on
    a human reading it before every run.

    Task 5's assembly keeps one bag per (gene, line) pair, so the SAME gene
    symbol legitimately repeats across lines with overlapping panels --
    ``GeneBags.for_genes``/``for_prediction_genes`` (used below) can only
    keep one bag per distinct requested name, silently dropping the rest.
    This does not raise (duplicate names are an expected property of real
    multi-line data, not necessarily a misconfiguration), but a silent drop
    should never be the ONLY signal a real run produces.
    """
    upper = [str(gene).upper() for gene in genes]
    duplicates = len(upper) - len(set(upper))
    if duplicates:
        _LOGGER.warning(
            "%d of %d assembled bags share a gene name already used by "
            "another line; GeneBags.for_genes keeps only one bag per "
            "distinct name once split, so %d line-specific bag(s) will be "
            "silently absent from train/val/test (see this module's KNOWN "
            "LIMITATION docstring)",
            duplicates,
            len(upper),
            duplicates,
        )


def _split_authority_paths(run_dir: Path) -> tuple[Path, Path]:
    """Deterministic paths for this run's split-authority artifact (no I/O).

    Pure so every DDP rank can compute the same paths independently; only
    the actual write (:func:`_write_split_authority`) needs rank-gating.
    """
    return (
        run_dir / "phase_c_gene_split.csv",
        run_dir / "phase_c_gene_split.csv.sha256",
    )


def _write_split_authority(manifest_path: Path, sha_path: Path, fold: FoldSpec) -> None:
    """Write this run's own gene split as its "canonical split" authority.

    ``train._canonical_split_sha256``/``_fold_artifact_authority`` (the
    audited path's provenance machinery) unconditionally require
    ``config.cv.outer_split_sha256_file`` to exist -- Phase C has no
    pre-registered canonical exp05 split to point at (its split is computed
    fresh, per run, from whatever ``--cache-dir``/``--line-manifest``
    assemble), so this writes ``_split_fold``'s own output as that
    authority artifact instead of a meaningless placeholder file. Only the
    file's presence and hex-digest shape are checked, so this is honest:
    the digest really does match the split this run used.

    Callers must gate this with ``aivc_model.distributed.
    run_rank_zero_or_raise`` under a real multi-rank launch -- unguarded,
    every DDP rank would race to write the same two files.
    """
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    rows = (
        [{"perturbation_gene": gene, "role": "train"} for gene in fold.train_genes]
        + [{"perturbation_gene": gene, "role": "val"} for gene in fold.val_genes]
        + [{"perturbation_gene": gene, "role": "test"} for gene in fold.test_genes]
    )
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    sha_path.write_text(sha256_file(manifest_path) + "\n", encoding="utf-8")


def run_real_training(
    config: AivcConfig,
    bags: GeneBags,
    *,
    line_manifest: Path,
    perturbseq_source_config: Path,
    run_dir: Path | None,
) -> dict[str, Path]:
    """Split, seal, and hand ``bags`` to ``aivc_model.train.run_training``."""
    _warn_on_duplicate_gene_names(bags.genes)
    fold = _split_fold(bags.genes, config)
    _LOGGER.info(
        "gene split: %d train, %d val, %d test",
        len(fold.train_genes),
        len(fold.val_genes),
        len(fold.test_genes),
    )
    recorder = GeneAccessRecorder(fold)
    data = replace(bags, access_recorder=recorder)
    train_data = data.for_genes(fold.train_genes, stage="adapter_fit")
    val_data = data.for_prediction_genes(
        fold.val_genes,
        stage="early_stopping_prediction_only",
        generation_targets=True,
    )
    sealed_test = SealedGeneBags(data, fold.test_genes)
    resolved_run_dir = run_dir or _default_run_dir(config)
    manifest_path, sha_path = _split_authority_paths(resolved_run_dir)
    # Build the accelerator here (not left to run_training) so the
    # split-authority write below can be gated to rank zero under a real
    # multi-rank `accelerate launch` -- unguarded, every rank would race to
    # write the same two files. The same instance is then handed to
    # run_training so it is not built a second time.
    accelerator = train_module._make_accelerator(config)
    run_rank_zero_or_raise(
        accelerator,
        "phase C split-authority write",
        lambda: _write_split_authority(manifest_path, sha_path, fold),
    )
    accelerator.wait_for_everyone()
    config = replace(
        config,
        cv=replace(
            config.cv,
            outer_split_manifest=manifest_path,
            outer_split_sha256_file=sha_path,
        ),
    )
    source_fingerprint = _source_fingerprint(line_manifest, perturbseq_source_config)
    canonical_gene_order = tuple(str(gene).upper() for gene in data.genes)
    return train_module.run_training(
        config,
        accelerator=accelerator,
        train_data=train_data,
        val_data=val_data,
        sealed_test=sealed_test,
        fold_spec=fold,
        run_dir_override=resolved_run_dir,
        source_fingerprint=source_fingerprint,
        canonical_gene_order=canonical_gene_order,
    )


def _dry_run_summary(
    config: AivcConfig, bags: GeneBags, args: argparse.Namespace
) -> dict[str, object]:
    return {
        "config": str(args.config),
        "cache_dir": str(args.cache_dir),
        "line_manifest": str(args.line_manifest),
        "perturbseq_source_config": str(args.perturbseq_source_config),
        "input_view": config.state.input_view,
        "state_input_dim": bags.input_dim,
        "response_encoder_input_dim": bags.effective_target_dim,
        "n_bags": int(len(bags.genes)),
        "n_control_cells": int(bags.control_input.shape[0]),
        "l2_normalize_input": config.state.l2_normalize_input,
        "run_dir": str(_default_run_dir(config)),
    }


def main() -> None:
    """Assemble Phase C data for one arm, then dry-run-validate or train it."""
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = load_config(args.config)
    # Path existence is checked in both modes: --dry-run's whole point is to
    # catch a bad config before training starts, not just before *this*
    # invocation happens to also want to train.
    validate_config_shape(config, check_paths=True)
    bags = assemble_and_project(
        config,
        cache_dir=args.cache_dir,
        line_manifest=args.line_manifest,
        perturbseq_source_config=args.perturbseq_source_config,
    )
    validate_config_against_bags(config, bags)
    if args.dry_run:
        summary = _dry_run_summary(config, bags, args)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    paths = run_real_training(
        config,
        bags,
        line_manifest=args.line_manifest,
        perturbseq_source_config=args.perturbseq_source_config,
        run_dir=args.run_dir,
    )
    print(f"run dir: {paths['run_dir']}")


if __name__ == "__main__":
    main()
