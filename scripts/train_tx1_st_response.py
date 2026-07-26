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
        --line-manifest results/phase_a_tx1_20260724/cell_line_manifest.csv \\
        --response-cache-dir data/tx1_response_targets_cache

**fix-round-4: warm the response cache BEFORE that multi-rank launch.**
``main()``'s real-training path calls ``assemble_and_project`` (this
module's own wrapper around Task 5's expensive, arm-independent raw-source
read) unconditionally, for every rank, *before* any accelerator/DDP object
exists -- there is no rank gating in that call. Under a cold
``--response-cache-dir``, 4 concurrent ``accelerate launch`` ranks would
each independently materialize the full per-line response matrices at the
same time, multiplying the ~195 GB single-process peak measured in the
2026-07-26 incident by ``--num_processes``. ``--warm-response-cache`` runs
that same assembly ONE TIME, single-process (plain ``python``, no
``accelerate``), writes ``--response-cache-dir``, and exits 0 without ever
calling ``run_real_training``/``run_training`` -- so it can never trip
``distributed.require_exact_world_size``'s 4-rank gate::

    .venv-tx1/bin/python scripts/train_tx1_st_response.py \\
        --config configs/experiments/12_tx1_st_geneeffect/phase_c/tx1_arm.yaml \\
        --cache-dir data/tx1_basal_embeddings/v1 \\
        --line-manifest results/phase_a_tx1_20260724/cell_line_manifest.csv \\
        --response-cache-dir data/tx1_response_targets_cache \\
        --warm-response-cache

**Both arms, the documented way (fix-round-3 Fix 4, fix-round-4):** use
``scripts/launch_phase_c_arms.sh``, which runs the warm step above once,
then runs the two arms SEQUENTIALLY by default, passing all three the same
``--response-cache-dir`` so every subsequent read (the second arm's own
ranks included) reuses the first's expensive response-target assembly (Fix
2) instead of repeating it. Do **not** launch either arm concurrently by
hand, and do **not** skip the warm step: the 2026-07-26 incident
(``.superpowers/sdd/phase-c/progress.md``) killed both single-process arms
after they independently climbed to ~621-625 GB RSS EACH on a shared node,
because each arm's own invocation of this script re-ran the identical,
arm-independent raw-source read at the same time -- 4 concurrent DDP ranks
against a cold cache reproduce the same failure mode, only worse. See
that script's own header for the (non-default, explicitly opt-in)
parallel-run override.

FIXED (Codex review, wave 2 round 1): ``GeneBags.for_genes``/
``for_prediction_genes``/``SealedGeneBags`` still resolve genes through a
``{name: index}`` dict, and Task 5's assembly still keeps one bag per
(gene, line) pair -- so a bare, repeated gene name would still silently
collapse to the last-seen line's bag. What changed is the KEY: Task 5's
assembly now keys every bag as ``f"{gene}@{model_id}"``
(``tx1_response_data.composite_gene_key``), so no two of the four training
lines' bags ever share a key, and every line's observed response survives
``for_genes``/``for_prediction_genes`` unchanged. The gene-level split
below (``_split_fold``) groups those composite keys by their BASE gene
(``tx1_response_data.base_gene_name``) before splitting, so every per-line
bag of one gene takes the SAME train/val/test role -- without that
grouping, ``KIF11@ACH-000551`` could land in train while
``KIF11@ACH-000995`` lands in test, which is cross-line leakage of the same
biological gene (a CV-protocol violation), not a real held-out split. See
``tests/test_train_tx1_st_response.py``'s
``test_split_fold_never_splits_a_base_gene_across_roles`` for the direct
leakage assertion. ``GeneBags.for_genes``'s own name-keyed dict lookup is
unchanged and is not the concern here: this module's keys are unique by
construction, so it never collapses anything for this data.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import replace
from pathlib import Path
from typing import Sequence

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
from aivc_model.tx1_response_cache_warmer import (  # noqa: E402
    assemble_response_bags,
    warm_response_cache_summary,
)
from aivc_model.tx1_response_data import (  # noqa: E402
    base_gene_name,
    referenced_source_paths,
    validate_response_sources_shape,
)

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
    parser.add_argument(
        "--response-cache-dir",
        type=Path,
        default=None,
        help=(
            "Fix-round-3, Fix 2: fingerprinted cache directory for the "
            "expensive, arm-independent observed-response target assembly. "
            "Pass the SAME directory to both arms' invocations so the "
            "second arm reuses the first's raw-source read instead of "
            "repeating it. Omit (the default) to never touch the cache, "
            "i.e. today's per-invocation-from-scratch behavior."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--warm-response-cache",
        action="store_true",
        help=(
            "fix-round-4: assemble and write --response-cache-dir's shared "
            "response-target cache, single-process, then exit 0 WITHOUT "
            "calling run_training. Run this once, under plain python (no "
            "accelerate), before `accelerate launch --num_processes 4 ...` "
            "against the SAME --response-cache-dir -- assemble_and_project "
            "has no rank gating, so a cold cache read concurrently by "
            "multiple DDP ranks would each pay the full raw-source-read "
            "cost at once. Requires --response-cache-dir; mutually "
            "exclusive with --dry-run."
        ),
    )
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
    response_cache_dir: Path | None = None,
) -> GeneBags:
    """Assemble the 4-line response ``GeneBags`` and project it onto ``config``'s arm.

    Args:
        response_cache_dir: Fix-round-3, Fix 2's shared response-targets
            cache directory. ``None`` (the default) never touches it.

    Raises:
        ValueError: ``state.model_dir`` is unset, or any error Task 5's
            assembly itself raises (missing cache/source entries, gene
            order mismatch, unsupported ``input_view``, ...).
    """
    bags = assemble_response_bags(
        config,
        cache_dir=cache_dir,
        line_manifest=line_manifest,
        perturbseq_source_config=perturbseq_source_config,
        response_cache_dir=response_cache_dir,
    )
    _LOGGER.info("projecting onto input_view=%s", config.state.input_view)
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
    """Deterministic, non-stratified train/val/test split grouped by BASE gene.

    Phase C's assembled ``GeneBags`` carry no GeneEffect label (``y`` is an
    all-NaN placeholder), so -- unlike exp05's label-stratified canonical
    splits -- this is a plain partition of distinct gene names driven by
    ``config.split``'s fractions/seed.

    ``genes`` here are Task 5's composite ``f"{gene}@{model_id}"`` bag keys
    (``tx1_response_data.composite_gene_key``), one per (gene, line) pair.
    The split groups them by BASE gene (``tx1_response_data.base_gene_name``)
    and assigns every per-line bag of one gene the SAME fold role (user
    ruling, 2026-07-25): splitting on the composite key directly would let
    e.g. ``KIF11@ACH-000551`` land in train while ``KIF11@ACH-000995`` lands
    in test -- cross-line leakage of the same biological gene, a violation
    of this repo's CV protocol, since ``GeneAccessRecorder`` authorizes by
    bag key, not by base gene identity.

    fix-round-6: the returned ``train_genes``/``val_genes``/``test_genes``
    keep each composite key's ORIGINAL case. The grouping key (which base
    genes get merged together) is still upper-cased internally -- so two
    lines whose raw sources happen to case a symbol differently still group
    into ONE fold role -- but the composite STRINGS handed back are exactly
    the ones ``genes`` was given. Force-uppercasing them (the old behavior)
    silently broke every ``Cxxorfyy``-style HGNC symbol (the only common
    gene symbols with lowercase letters, e.g. ``C10orf67``): a val/test-only
    ORF gene's uppercased composite key flowed into
    ``train._build_e2e_model``'s ``PerturbationVectorAdapter`` vocabulary,
    while evaluation (``train._final_prediction_tensor``) looked it up in
    its ORIGINAL case via ``GeneBags.for_genes``/``for_prediction_genes``
    (which always return the bag's own case, never the request string) --
    two different dict keys for the same gene, so the adapter's exact-string
    lookup raised a bare ``KeyError`` for that gene, only in POST-EPOCH
    evaluation, only for the lines/genes assigned to val/test (see
    .superpowers/sdd/phase-c/fix-round-6-report.md).
    """
    composite_genes = [str(gene) for gene in genes]
    composite_by_base: dict[str, list[str]] = {}
    for composite in composite_genes:
        composite_by_base.setdefault(base_gene_name(composite).upper(), []).append(
            composite
        )
    unique_base_genes = sorted(composite_by_base)
    if len(unique_base_genes) < 3:
        raise ValueError(
            f"need at least 3 distinct base genes to split train/val/test; got "
            f"{len(unique_base_genes)}"
        )
    split = config.split
    train_base, remainder_base = train_test_split(
        unique_base_genes,
        train_size=split.train_fraction,
        random_state=split.random_state,
        shuffle=True,
    )
    remainder_total = split.val_fraction + split.test_fraction
    val_share = split.val_fraction / remainder_total if remainder_total > 0 else 0.5
    if len(remainder_base) < 2:
        val_base, test_base = remainder_base, []
    else:
        val_base, test_base = train_test_split(
            remainder_base,
            train_size=val_share,
            random_state=split.random_state,
            shuffle=True,
        )

    def _expand(base_genes: Sequence[str]) -> tuple[str, ...]:
        """Every composite bag key for each of ``base_genes`` (sorted)."""
        return tuple(
            sorted(
                composite
                for base in base_genes
                for composite in composite_by_base[base]
            )
        )

    return FoldSpec(
        outer_fold=0,
        train_genes=_expand(train_base),
        val_genes=_expand(val_base),
        test_genes=_expand(test_base),
    )


def _source_fingerprint(
    line_manifest: Path,
    perturbseq_source_config: Path,
    tx1_cache_dir: Path,
) -> str:
    """Hash every data-selection input actually consumed, for provenance (C8).

    Task 6 (``7cb77c7``) applied this same honesty principle to the GWPS
    cache fingerprint. Before this fix (Codex P2-a), hashing only the line
    manifest and source-config JSON let two runs built from different
    sampled cells, Tx1 embeddings/HVG arrays, or a modified response source
    collide in ``ArtifactAuthority.source_fingerprint``, making incompatible
    checkpoints indistinguishable in provenance. Includes:

    - the line manifest and perturbseq source-config JSON (as before);
    - the verified Tx1 cache's own run ``manifest.json`` (assembly already
      requires this cache to pass an unrestricted ``verify_cache`` before
      this point -- its manifest covers every sampled cell, embedding, and
      HVG array actually consumed, by content hash);
    - every response source file the config JSON directly names
      (:func:`aivc_model.tx1_response_data.referenced_source_paths`), so a
      modified h5ad/parquet at the same configured path changes provenance
      too, not just a changed path/glob string.
    """
    hashes = [sha256_file(line_manifest), sha256_file(perturbseq_source_config)]
    cache_manifest_path = Path(tx1_cache_dir) / "manifest.json"
    if cache_manifest_path.is_file():
        hashes.append(sha256_file(cache_manifest_path))
    hashes.extend(
        sha256_file(path) for path in referenced_source_paths(perturbseq_source_config)
    )
    return sha256_strings(hashes)


def _default_run_dir(config: AivcConfig) -> Path:
    run_id = config.train.run_id or "tx1_st_response"
    return Path(config.data.output_dir) / "runs" / run_id


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
    tx1_cache_dir: Path,
    run_dir: Path | None,
) -> dict[str, Path]:
    """Split, seal, and hand ``bags`` to ``aivc_model.train.run_training``."""
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
    source_fingerprint = _source_fingerprint(
        line_manifest, perturbseq_source_config, tx1_cache_dir
    )
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


def _validate_widths_against_checkpoint(
    config: AivcConfig, checkpoint_hvg_width: int
) -> None:
    """Cross-check config widths against the checkpoint's own HVG width alone.

    Mirrors :func:`validate_config_against_bags`'s ``response_encoder``/
    ``output_dim`` checks, but sourced from ``checkpoint_hvg_width`` (a
    cheap ``var_dims.pkl`` read) instead of a real assembled ``GeneBags`` --
    ``tx1_response_data._build_line_gene_bags``'s ``_align_to_checkpoint_order``
    always makes the real assembled target width equal exactly this number
    (raising otherwise), so this catches the same misconfiguration
    `--dry-run` always caught, without assembling any data.

    Raises:
        ValueError: A configured width disagrees with ``checkpoint_hvg_width``.
    """
    if (
        config.response_encoder is not None
        and config.response_encoder.input_dim != checkpoint_hvg_width
    ):
        raise ValueError(
            "response_encoder.input_dim "
            f"{config.response_encoder.input_dim} != checkpoint HVG width "
            f"{checkpoint_hvg_width}"
        )
    if (
        config.state.output_dim is not None
        and config.state.output_dim != checkpoint_hvg_width
    ):
        raise ValueError(
            f"state.output_dim {config.state.output_dim} != checkpoint HVG "
            f"width {checkpoint_hvg_width}"
        )


def _dry_run_summary(config: AivcConfig, args: argparse.Namespace) -> dict[str, object]:
    """Validate every Phase C input cheaply, without reading a cell matrix.

    Codex fix-round-2 Defect 3: a real run's very first action
    (``assemble_and_project`` -> ``assemble_train_response_gene_bags`` ->
    ``_build_response_adata``) materializes every selected line's perturbed-
    cell expression matrix -- exactly the call that hung for 58+ minutes at
    100% CPU with SIGINT ignored against a dense, unchunked 66 GB h5ad,
    before a single byte was read. ``--dry-run``'s whole job is a cheap
    preflight that "validates every path and width and exit[s] without
    training", so it must never reach that call. This instead validates
    everything :func:`validate_response_sources_shape` can check from small
    metadata alone (manifest roles, cache verification, source presence, HVG
    gene-order agreement, gene-vocabulary coverage) plus the width
    cross-check :func:`_validate_widths_against_checkpoint` derives from the
    same cheap checkpoint read, and stops there -- typically sub-second to a
    few seconds, never proportional to a source's cell count.

    Raises:
        ValueError: ``state.model_dir`` is unset, or any check
            :func:`validate_response_sources_shape`/
            :func:`_validate_widths_against_checkpoint` performs fails.
    """
    if config.state.model_dir is None:
        raise ValueError(
            "state.model_dir is required (the checkpoint dir supplying the "
            "HVG gene order)"
        )
    sources_summary = validate_response_sources_shape(
        cell_line_manifest_path=args.line_manifest,
        tx1_cache_dir=args.cache_dir,
        hvg_state_model_dir=config.state.model_dir,
        perturbseq_sources_path=args.perturbseq_source_config,
    )
    _validate_widths_against_checkpoint(
        config, int(sources_summary["checkpoint_hvg_width"])
    )
    return {
        "config": str(args.config),
        "cache_dir": str(args.cache_dir),
        "line_manifest": str(args.line_manifest),
        "perturbseq_source_config": str(args.perturbseq_source_config),
        "input_view": config.state.input_view,
        "state_input_dim": config.state.input_dim,
        "response_encoder_input_dim": (
            config.response_encoder.input_dim
            if config.response_encoder is not None
            else None
        ),
        "response_max_cells_per_gene": config.data.response_max_cells_per_gene,
        "response_total_cells_per_line": config.data.response_total_cells_per_line,
        "response_cache_dir": (
            str(args.response_cache_dir) if args.response_cache_dir else None
        ),
        "l2_normalize_input": config.state.l2_normalize_input,
        "run_dir": str(_default_run_dir(config)),
        **sources_summary,
    }


def _warm_response_cache_summary(
    config: AivcConfig, args: argparse.Namespace
) -> dict[str, object]:
    """Thin CLI wrapper over ``tx1_response_cache_warmer.warm_response_cache_summary``.

    fix-round-4: that function must run once, single-process (plain
    ``python``, no ``accelerate``), strictly BEFORE any multi-rank
    ``accelerate launch`` reads the same ``--response-cache-dir`` -- see
    its docstring for the full 2026-07-26-incident memory-multiplication
    reasoning and the argument for why cache reuse across DDP ranks cannot
    be rank-dependent. This wrapper only adds ``args.config`` for CLI
    provenance; it never calls ``run_real_training``/``run_training``, so
    ``distributed.require_exact_world_size``'s 4-rank gate is never reached
    from this mode.

    Raises:
        ValueError: ``args.response_cache_dir`` is unset, or anything
            :func:`tx1_response_cache_warmer.warm_response_cache_summary`
            itself raises.
    """
    summary = warm_response_cache_summary(
        config,
        cache_dir=args.cache_dir,
        line_manifest=args.line_manifest,
        perturbseq_source_config=args.perturbseq_source_config,
        response_cache_dir=args.response_cache_dir,
    )
    return {"config": str(args.config), **summary}


def main() -> None:
    """Assemble Phase C data for one arm: warm the response cache, dry-run
    validate, or train it."""
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.dry_run and args.warm_response_cache:
        raise ValueError("--dry-run and --warm-response-cache are mutually exclusive")
    config = load_config(args.config)
    # Path existence is checked in every mode: --dry-run's/
    # --warm-response-cache's whole point is to catch a bad config before
    # training starts, not just before *this* invocation happens to train.
    validate_config_shape(config, check_paths=True)
    if args.dry_run:
        # Defect 3: never assemble (and thus never read a single cell's
        # expression values) for a dry run -- see _dry_run_summary.
        summary = _dry_run_summary(config, args)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    if args.warm_response_cache:
        # fix-round-4: assemble + write the shared response cache and stop
        # -- never reaches run_real_training/run_training (the DDP gate),
        # so this mode is safe to run single-process ahead of the real
        # multi-rank `accelerate launch`. See _warm_response_cache_summary.
        summary = _warm_response_cache_summary(config, args)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    bags = assemble_and_project(
        config,
        cache_dir=args.cache_dir,
        line_manifest=args.line_manifest,
        perturbseq_source_config=args.perturbseq_source_config,
        response_cache_dir=args.response_cache_dir,
    )
    validate_config_against_bags(config, bags)
    paths = run_real_training(
        config,
        bags,
        line_manifest=args.line_manifest,
        perturbseq_source_config=args.perturbseq_source_config,
        tx1_cache_dir=args.cache_dir,
        run_dir=args.run_dir,
    )
    print(f"run dir: {paths['run_dir']}")


if __name__ == "__main__":
    main()
