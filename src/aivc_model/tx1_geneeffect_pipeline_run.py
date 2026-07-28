"""Phase D Task 5: real-run orchestration -- the sibling of
:mod:`aivc_model.tx1_geneeffect_pipeline` (split out purely for CLAUDE.md's
600-line-per-file rule; that module owns the config schema and every
``--dry-run`` check, this one owns the chain ``--dry-run`` deliberately never
reaches: ST construction/checkpoint loading, streaming response pooling, head
training, and Phase F table emission).

Two things this module resolves that no earlier Phase D task did:

1. **Bounded-memory same-process execution.** Each ST response is moment-
   pooled immediately; the raw per-cell tensor is discarded before the next
   gene. No Phase D predicted-response tensor is persisted to disk.
2. **DepMap-to-ST-vocabulary gene-key translation.** ``load_depmap_gene_effect``
   (Task 1) indexes columns by ``depmap_column`` (e.g. ``"ACLY (47)"``);
   ``assemble_line_features`` (Task 3) expects ``gene_effect`` indexed by the
   ST-vocabulary gene SYMBOL used for response generation
   (post :data:`~aivc_model.tx1_predicted_response.SLICE_SYMBOL_ALIASES`
   resolution). :func:`build_training_gene_effect_by_line` is that
   translation, built once from the frozen slice's own
   ``gene_symbol``/``depmap_column`` pairing.

Wave 3 Codex gate (fix round 1) closed the gap the module docstring used to
flag here: Phase C's checkpoint writer now emits ``gene_vocabulary.json``
(the exact, ordered, full perturbation-adapter training vocabulary) beside
``pytorch_model.bin``, records its sha256 in that checkpoint's sibling
``metadata.json``, and this module authenticates ``state.gene_vocabulary_path``
against that recorded hash before ever constructing a model from it
(:func:`~aivc_model.tx1_geneeffect_pipeline.verify_gene_vocabulary_authenticity`,
P1-3) -- a same-genes-different-order file would otherwise pass gene-coverage
and ``load_state_dict`` alike while silently binding every positional
missing-gene weight to the wrong symbol.

That same gate also closed four other gaps in this module specifically:

* **P1-1**: :func:`run_training_pipeline` now runs an UNRESTRICTED
  ``tx1_embed_cache.verify_cache`` (Phase B's own "verified embedding
  caches" exit criterion) before any basal cache is read. The width check
  this module previously relied on alone
  (``tx1_geneeffect_pipeline.validate_widths``) only cross-checks the
  FIRST manifest entry's recorded shape -- a stale or corrupted entry for
  any other line passed silently.
* **P1-2**: now runs ``tx1_geneeffect_eval.verify_artifact_hashes`` before
  reading the frozen Phase-A registration/slice/panels -- that trust
  anchor was previously read unauthenticated here despite already existing
  precisely to catch this.
* **P1-4**: ``predicted_response_fingerprint`` now also hashes the FULL,
  ORDERED gene vocabulary (not just the sorted set of requested genes), so
  a reordered vocabulary can no longer reuse another construction's cached
  responses under an identical key.
* **P1-5**: resolves an accelerator device
  (:func:`~aivc_model.tx1_predicted_response.resolve_device`, config
  ``state.device``, default ``"auto"``) once, moves the forward-only model
  there, and threads it through every response-generation call -- ST
  inference no longer runs unconditionally on CPU.
"""

from __future__ import annotations

import importlib
import json
import logging
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd
import torch

from aivc_model.gene_splits import sha256_file
from aivc_model.tx1_embed_cache import verify_cache
from aivc_model.tx1_geneeffect_data import LineRoleSplit, load_depmap_gene_effect
from aivc_model.tx1_geneeffect_eval import (
    load_panels,
    load_slice,
    verify_artifact_hashes,
)
from aivc_model.tx1_geneeffect_head import Tx1GeneEffectHead
from aivc_model.tx1_geneeffect_pipeline import (
    Tx1GeneEffectHeadConfig,
    validate_config_shape,
    validate_gene_coverage,
    validate_roles,
    validate_widths,
    verify_gene_vocabulary_authenticity,
)
from aivc_model.tx1_geneeffect_predict import (
    assemble_test_line_features,
    build_test_line_predictions,
)
from aivc_model.tx1_geneeffect_train import TrainingConfig, train_tx1_geneeffect_head
from aivc_model.tx1_geneeffect_train_io import (
    assemble_split_features,
    build_provenance,
    save_checkpoint,
    write_provenance,
)
from aivc_model.tx1_predicted_response import (
    SLICE_SYMBOL_ALIASES,
    ForwardOnlyStateModel,
    construct_forward_only_model,
    load_forward_only_checkpoint,
    resolve_device,
)

_LOGGER = logging.getLogger(__name__)

__all__ = [
    "build_forward_only_model",
    "build_training_gene_effect_by_line",
    "emit_test_predictions",
    "run_training_pipeline",
]


def build_forward_only_model(
    config: Tx1GeneEffectHeadConfig, vocabulary_genes: Sequence[str]
) -> ForwardOnlyStateModel:
    """Construct a fresh forward-only ST + perturbation-adapter pair.

    ``vocabulary_genes`` must be Phase C's exact, ordered, full training
    vocabulary (see this module's docstring) -- both backends key their
    ``PerturbationVectorAdapter`` by this list's position.
    """
    state = config.state
    if state.backend == "linear_mock":
        # Test-only backend (mirrors aivc_model.model.load_state_model's own
        # backend switch): builds a tiny CPU model directly, with no
        # `state.tx.models.state_transition` import and no hparams
        # checkpoint file.
        from aivc_model.model import (
            LinearMockStateModel,
            PerturbationVectorAdapter,
            StateForwardAdapter,
        )
        from aivc_model.prepare import load_perturbation_vectors

        state_model = LinearMockStateModel(
            state.input_dim, state.output_dim, state.pert_dim
        )
        known_vectors = load_perturbation_vectors(state.known_perturbation_vectors)
        perturbations = PerturbationVectorAdapter(
            list(vocabulary_genes), known_vectors, state.pert_dim
        )
        return ForwardOnlyStateModel(StateForwardAdapter(state_model), perturbations)

    if state.hparams_checkpoint_path is None:
        raise ValueError(
            "state.hparams_checkpoint_path is required for backend=state_checkpoint"
        )
    module = importlib.import_module("state.tx.models.state_transition")
    model_cls = getattr(module, "StateTransitionPerturbationModel")
    return construct_forward_only_model(
        model_cls=model_cls,
        hparams_checkpoint_path=state.hparams_checkpoint_path,
        input_dim=state.input_dim,
        output_dim=state.output_dim,
        pert_dim=state.pert_dim,
        genes=vocabulary_genes,
        known_perturbation_vectors=state.known_perturbation_vectors,
        output_space=state.output_space,
    )


def build_training_gene_effect_by_line(
    depmap_frame: pd.DataFrame, slice_df: pd.DataFrame
) -> dict[str, pd.Series]:
    """Translate DepMap's ``depmap_column``-indexed matrix into the
    ST-vocabulary-symbol-indexed Series ``assemble_line_features`` expects.

    Built once from the frozen slice's own ``gene_symbol``/``depmap_column``
    pairing (with :data:`SLICE_SYMBOL_ALIASES` applied), not per line.
    """
    aliases = dict(SLICE_SYMBOL_ALIASES)
    column_by_symbol = {
        aliases.get(str(row.gene_symbol), str(row.gene_symbol)): str(row.depmap_column)
        for row in slice_df.itertuples(index=False)
    }
    return {
        str(model_id): pd.Series(
            {
                symbol: depmap_frame.loc[str(model_id), column]
                for symbol, column in column_by_symbol.items()
            }
        )
        for model_id in depmap_frame.index
    }


def _ensure_fresh_run_dir(resolved_run_dir: Path) -> None:
    """Refuse to write into a run directory that already exists (P2-3).

    A rerun with a fixed ``run_id`` would otherwise overwrite the checkpoint
    and provenance in place while leaving any OTHER prior file untouched --
    e.g. an old ``tx1_3b_st_predictions.csv`` from an earlier run that used
    ``--skip-test-predictions`` differently, sitting beside a checkpoint it
    was never actually produced from. A plausible, mismatched artifact is
    worse than a crash (D10): require a fresh directory instead.

    Checked first, before any other validation, so a doomed rerun fails
    immediately rather than after minutes of cache verification/training.

    Raises:
        FileExistsError: ``resolved_run_dir`` already exists.
    """
    if resolved_run_dir.exists():
        raise FileExistsError(
            f"run directory {resolved_run_dir} already exists; refusing to "
            "reuse it (Wave 3 Codex gate P2-3): a rerun could leave a stale "
            "output file -- e.g. an old tx1_3b_st_predictions.csv -- beside "
            "a new checkpoint. Choose a fresh run_id/run_dir, or remove the "
            "existing directory yourself if you are intentionally replacing "
            "its contents."
        )


def run_training_pipeline(
    config: Tx1GeneEffectHeadConfig,
    *,
    line_manifest_path: Path,
    phase_a_dir: Path,
    tx1_cache_dir: Path,
    depmap_gene_effect_path: Path,
    run_dir: Path | None = None,
    emit_test_predictions_flag: bool = True,
) -> dict[str, Path]:
    """Run the full Phase D chain: line selection -> predicted response ->
    head training -> (optionally) the Phase F table.

    D6/D12 are enforced by the functions this delegates to (Task 1's
    ``assert_training_role``/``build_line_role_split``, Task 3's
    ``_assert_examples_admissible``) -- never re-derived here. Wave 3 Codex
    gate P1-1/P1-2/P1-3/P1-5/P2-3 are enforced directly in this function
    (see the module docstring); ordered cheapest-check-first, and every
    trust check runs before any cache is read or model is constructed from
    it.

    Returns:
        Output paths: always ``run_dir``/``checkpoint``/``provenance``; also
        ``predictions`` when ``emit_test_predictions_flag``.
    """
    resolved_run_dir = (
        Path(run_dir) if run_dir else Path(config.output_dir) / "runs" / config.run_id
    )
    _ensure_fresh_run_dir(resolved_run_dir)  # P2-3

    validate_config_shape(config, check_paths=True)
    validate_widths(config, tx1_cache_dir)
    manifest, registration, split = validate_roles(
        line_manifest_path, config.validation_lines_path
    )
    # P1-2: authenticate the frozen Phase-A registration/slice/panels before
    # trusting anything read from phase_a_dir below.
    verify_artifact_hashes(Path(phase_a_dir))

    # P1-1: the Phase B exit criterion ("verified embedding caches") --
    # unrestricted (no only_lines), and run before `load_line_cache` is
    # ever reached (deep inside `assemble_split_features`/
    # `emit_test_predictions`). `validate_widths` above only cross-checks
    # ONE manifest entry's recorded shape; it is not a substitute.
    cache_report = verify_cache(
        Path(tx1_cache_dir), frozen_manifest_path=Path(line_manifest_path)
    )
    if cache_report["status"] != "verified":
        raise ValueError(
            f"Phase B basal-embedding cache at {tx1_cache_dir} failed "
            "verification (Wave 3 Codex gate P1-1); refusing to read it: "
            f"{cache_report['discrepancies']}"
        )

    phase_a_registration = json.loads(
        (Path(phase_a_dir) / "phase_a_registration.json").read_text()
    )
    slice_df = load_slice(Path(phase_a_dir) / "differentially_essential_slice.csv")
    validate_gene_coverage(slice_df, config.state.gene_vocabulary_path)
    vocabulary_genes = json.loads(Path(config.state.gene_vocabulary_path).read_text())

    # P1-3: authenticate state.gene_vocabulary_path against Phase C's own
    # recorded hash before it ever seeds a model's parameter positions.
    verify_gene_vocabulary_authenticity(
        config.state.gene_vocabulary_path,
        config.state.st_checkpoint_path,
        config.state.backend,
    )

    # P1-5: resolve an accelerator device once and move the model there --
    # every response-generation call below threads it through explicitly.
    device = resolve_device(config.state.device)
    model = build_forward_only_model(config, vocabulary_genes)
    load_forward_only_checkpoint(model, config.state.st_checkpoint_path)
    model.to(device)

    train_and_validation_ids = tuple(
        sorted(set(split.train_model_ids) | set(split.validation_model_ids))
    )
    depmap_frame = load_depmap_gene_effect(
        depmap_gene_effect_path,
        phase_a_registration,
        train_and_validation_ids,
        columns=[str(value) for value in slice_df["depmap_column"]],
    )
    gene_effect_by_line = build_training_gene_effect_by_line(depmap_frame, slice_df)

    phase_b_manifest_path = Path(tx1_cache_dir) / "manifest.json"
    train_lines, validation_lines = assemble_split_features(
        manifest,
        registration,
        arm=config.arm,
        tx1_cache_dir=tx1_cache_dir,
        gene_effect_by_line=gene_effect_by_line,
        model=model,
        moments=config.training.moments,
        cell_set_len=config.state.cell_set_len,
        window_macro_batch_size=config.state.response_window_macro_batch_size,
        seed=config.state.response_generation_seed,
        device=device,
    )

    training_config = TrainingConfig(
        hidden=config.training.hidden,
        moments=config.training.moments,
        lam=config.objective.lam,
        learning_rate=config.training.learning_rate,
        epochs=config.training.epochs,
        seed=config.training.seed,
    )
    head, result = train_tx1_geneeffect_head(
        train_lines, validation_lines, training_config
    )

    checkpoint_path = save_checkpoint(head, resolved_run_dir / "models" / "head.pt")
    provenance = build_provenance(
        st_checkpoint_path=config.state.st_checkpoint_path,
        phase_b_manifest_path=phase_b_manifest_path,
        depmap_gene_effect_path=depmap_gene_effect_path,
        arm=config.arm,
        config=training_config,
        result=result,
        response_generation={
            "response_generation_seed": config.state.response_generation_seed,
            "cell_set_len": config.state.cell_set_len,
            "response_window_macro_batch_size": (
                config.state.response_window_macro_batch_size
            ),
            "pooling_reducer_schema": "online_chan_mean_population_variance_v1",
            "accumulator_dtype": "float32",
            "pooled_output_dtype": "float32",
            "torch_version": str(torch.__version__),
            "torch_cuda_version": torch.version.cuda,
            "device": str(device),
            "allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "gene_vocabulary_sha256": sha256_file(
                Path(config.state.gene_vocabulary_path)
            ),
            "gene_panel_sha256": sha256_file(
                Path(phase_a_dir) / "differentially_essential_slice.csv"
            ),
            "input_dim": config.state.input_dim,
            "output_dim": config.state.output_dim,
            "pert_dim": config.state.pert_dim,
            "output_space": config.state.output_space,
        },
    )
    provenance_path = write_provenance(provenance, resolved_run_dir / "provenance.json")

    outputs = {
        "run_dir": resolved_run_dir,
        "checkpoint": checkpoint_path,
        "provenance": provenance_path,
    }
    if emit_test_predictions_flag:
        outputs["predictions"] = emit_test_predictions(
            config,
            model,
            head,
            manifest,
            split,
            slice_df,
            phase_a_registration,
            phase_a_dir=phase_a_dir,
            depmap_gene_effect_path=depmap_gene_effect_path,
            tx1_cache_dir=tx1_cache_dir,
            run_dir=resolved_run_dir,
            device=device,
        )
    return outputs


def emit_test_predictions(
    config: Tx1GeneEffectHeadConfig,
    model: ForwardOnlyStateModel,
    head: Tx1GeneEffectHead,
    manifest: pd.DataFrame,
    split: LineRoleSplit,
    slice_df: pd.DataFrame,
    phase_a_registration: Mapping[str, object],
    *,
    phase_a_dir: Path,
    depmap_gene_effect_path: Path,
    tx1_cache_dir: Path,
    run_dir: Path,
    device: torch.device | str = "cpu",
) -> Path:
    """Build and write the D1 ``tx1_3b_st`` predictions table for the 9
    frozen held-out lines (Task 4).

    Args:
        device: Where to run ST's forward pass for these 9 lines (P1-5);
            ``model`` must already be on this device.

    Returns:
        The path the table was written to.
    """
    role_by_id = {
        str(row.model_id): str(row.role) for row in manifest.itertuples(index=False)
    }
    depmap_frame = load_depmap_gene_effect(
        depmap_gene_effect_path,
        phase_a_registration,
        list(split.test_model_ids),
        columns=[str(value) for value in slice_df["depmap_column"]],
    )
    panels = load_panels(Path(phase_a_dir) / "k_label_panels.csv")
    frames: list[pd.DataFrame] = []
    for model_id in split.test_model_ids:
        line = assemble_test_line_features(
            model_id,
            role_by_id[model_id],
            model,
            config.arm,
            tx1_cache_dir,
            slice_df,
            depmap_frame.loc[model_id],
            moments=config.training.moments,
            cell_set_len=config.state.cell_set_len,
            window_macro_batch_size=config.state.response_window_macro_batch_size,
            seed=config.state.response_generation_seed,
            device=device,
        )
        frames.append(
            build_test_line_predictions(
                head,
                line,
                panels[panels["model_id"] == model_id],
                list(config.phase_f.k_schedule),
                method=config.phase_f.method,
                alpha=config.phase_f.alpha,
                residual=config.phase_f.residual,
            )
        )
        del line
    predictions = pd.concat(frames, ignore_index=True)
    out_path = Path(run_dir) / "tx1_3b_st_predictions.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(out_path, index=False)
    return out_path
