"""Config-driven response-target assembly, shared by the real-training path
and the fix-round-4 cache-warming preflight (Wave 2 Phase C).

``scripts/train_tx1_st_response.py`` has two callers of Task 5's expensive,
arm-independent observed-response assembly
(:func:`aivc_model.tx1_response_data.assemble_train_response_gene_bags`):
its real-training path (``assemble_and_project``, which additionally
projects the result onto one arm's ``state.input_view`` -- CLI-specific,
NOT duplicated here) and its ``--warm-response-cache`` preflight, which
must run the IDENTICAL assembly, single-process, before any multi-rank
``accelerate launch`` reads the same ``--response-cache-dir``.

Why a warm step is required, not optional (2026-07-26 incident,
``.superpowers/sdd/phase-c/progress.md``): a single-process run measured
peak RSS climbing to 194.6 GB after assembling all four
``train_response_and_head`` lines, before it even reached the DDP gate
(``distributed.require_exact_world_size``). That gate is only reached
AFTER assembly, so under a cold ``--response-cache-dir``, every one of N
``accelerate launch --num_processes N`` ranks would independently pay that
same ~195 GB peak at once -- N=4 projects to ~780 GB on a shared 2015 GB
node. Warming the cache once, single-process, before that launch, and
having every rank point at the SAME cache directory, means only the first
caller ever pays the raw-source-read cost; every subsequent caller
(the warm step's own repeat, and every DDP rank's ``assemble_and_project``
call) hits ``tx1_response_gene_bags_cache.load_response_targets_cache``
instead. Reuse is not rank-dependent: :func:`assemble_response_bags`'s
fingerprint (``tx1_response_data.response_targets_fingerprint``, computed
inside ``assemble_train_response_gene_bags``) is a pure function of file
contents and config values, never a rank id, hostname, or PID.

Factored into its own module (Wave 2 fix-round-4) rather than left inline
in the already-oversized CLI script, per this repo's file-size discipline
(new logic beyond ~80 lines belongs in a new ``aivc_model`` module).
"""

from __future__ import annotations

import logging
from pathlib import Path

from aivc_model.prepare import AivcConfig, GeneBags
from aivc_model.tx1_response_data import assemble_train_response_gene_bags

_LOGGER = logging.getLogger(__name__)


def assemble_response_bags(
    config: AivcConfig,
    *,
    cache_dir: Path,
    line_manifest: Path,
    perturbseq_source_config: Path,
    response_cache_dir: Path | None = None,
) -> GeneBags:
    """Assemble the 4-line response ``GeneBags`` for ``config``'s data settings.

    Always returns the UNPROJECTED two-view ``GeneBags`` Task 5's assembly
    produces (Tx1-space ``input_bags``, gene-space ``target_bags``);
    callers that need one arm's single-view projection (the real-training
    path) apply that themselves. The on-disk response-target cache this
    reads/writes is identical either way -- fix-round-3 Fix 2's fingerprint
    has no ``input_view``/``l2_normalize`` field (see
    ``tx1_response_gene_bags_cache``'s module docstring), so warming from
    either Phase C arm's config populates the SAME cache entry the other
    arm will hit.

    Args:
        response_cache_dir: Fix-round-3, Fix 2's shared response-targets
            cache directory. ``None`` never touches it (always rebuilds,
            never writes).

    Raises:
        ValueError: ``config.state.model_dir`` is unset, or any error
            ``assemble_train_response_gene_bags`` itself raises (missing
            cache/source entries, gene order mismatch, ...).
    """
    if config.state.model_dir is None:
        raise ValueError(
            "state.model_dir is required (the checkpoint dir supplying the "
            "HVG gene order)"
        )
    _LOGGER.info(
        "assembling response GeneBags: cache_dir=%s line_manifest=%s "
        "perturbseq_source_config=%s l2_normalize=%s "
        "response_max_cells_per_gene=%s response_total_cells_per_line=%s "
        "response_cache_dir=%s",
        cache_dir,
        line_manifest,
        perturbseq_source_config,
        config.state.l2_normalize_input,
        config.data.response_max_cells_per_gene,
        config.data.response_total_cells_per_line,
        response_cache_dir,
    )
    bags = assemble_train_response_gene_bags(
        cell_line_manifest_path=line_manifest,
        tx1_cache_dir=cache_dir,
        hvg_state_model_dir=config.state.model_dir,
        perturbseq_sources_path=perturbseq_source_config,
        l2_normalize=config.state.l2_normalize_input,
        max_cells_per_gene=config.data.response_max_cells_per_gene,
        total_cells_per_line=config.data.response_total_cells_per_line,
        response_cache_dir=response_cache_dir,
        seed=config.train.seed,
    )
    _LOGGER.info(
        "assembled %d bags (input_dim=%d, target_dim=%d)",
        len(bags.genes),
        bags.input_dim,
        bags.effective_target_dim,
    )
    return bags


def warm_response_cache_summary(
    config: AivcConfig,
    *,
    cache_dir: Path,
    line_manifest: Path,
    perturbseq_source_config: Path,
    response_cache_dir: Path | None,
) -> dict[str, object]:
    """Materialize and write ``response_cache_dir``, single-process, then stop.

    This is the ONLY caller allowed to build fix-round-3 Fix 2's response
    cache from raw sources under a cold cache while a multi-rank launch is
    imminent -- see the module docstring for the memory-multiplication
    reasoning. Never touches ``aivc_model.train.run_training`` or any
    ``Accelerator``, so ``distributed.require_exact_world_size``'s 4-rank
    gate can never be reached from here: a caller running this under plain
    ``python`` (world size 1) is exactly the intended usage, not a bug.

    Args:
        response_cache_dir: Required (unlike :func:`assemble_response_bags`,
            where it is optional) -- warming a cache that is never written
            anywhere is a caller error, not a silently-accepted no-op.

    Returns:
        A JSON-serializable summary (paths, assembled bag count, widths).
        The caller (``scripts/train_tx1_st_response.py``) prints this and
        returns normally -- returning at all means real success, so a
        "succeeded but exited non-zero" report (Phase B codex P1-c) cannot
        happen through this function.

    Raises:
        ValueError: ``response_cache_dir`` is ``None``, or anything
            :func:`assemble_response_bags` itself raises.
    """
    if response_cache_dir is None:
        raise ValueError(
            "--warm-response-cache requires --response-cache-dir: there is "
            "nothing to warm otherwise"
        )
    bags = assemble_response_bags(
        config,
        cache_dir=cache_dir,
        line_manifest=line_manifest,
        perturbseq_source_config=perturbseq_source_config,
        response_cache_dir=response_cache_dir,
    )
    return {
        "cache_dir": str(cache_dir),
        "line_manifest": str(line_manifest),
        "perturbseq_source_config": str(perturbseq_source_config),
        "response_cache_dir": str(response_cache_dir),
        "n_bags": int(len(bags.genes)),
        "input_dim": int(bags.input_dim),
        "target_dim": int(bags.effective_target_dim),
    }


__all__ = ["assemble_response_bags", "warm_response_cache_summary"]
