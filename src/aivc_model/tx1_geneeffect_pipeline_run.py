"""Phase D Task 5: real-run orchestration -- the sibling of
:mod:`aivc_model.tx1_geneeffect_pipeline` (split out purely for CLAUDE.md's
600-line-per-file rule; that module owns the config schema and every
``--dry-run`` check, this one owns the chain ``--dry-run`` deliberately never
reaches: ST construction/checkpoint loading, predicted-response generation +
caching, head training, and Phase F table emission).

Two things this module resolves that no earlier Phase D task did:

1. **Per-line predicted-response cache fingerprints.** Task 2's
   ``predicted_response_fingerprint`` bakes ``model_id`` into its hash (by
   design -- belt-and-suspenders against a caller pairing the wrong line's
   cache entry), so every line's real fingerprint differs. Wiring a real
   multi-line run surfaced that ``tx1_geneeffect_train_io.
   assemble_split_examples`` had never been exercised against that variance
   (its own test used one hand-typed literal fingerprint for every synthetic
   line) -- fixed in this task to take ``fingerprint_by_line: Mapping[str,
   str]`` instead of one shared string. :func:`warm_predicted_response_cache`
   is the function that computes and reuses those per-line fingerprints for
   real.
2. **DepMap-to-ST-vocabulary gene-key translation.** ``load_depmap_gene_effect``
   (Task 1) indexes columns by ``depmap_column`` (e.g. ``"ACLY (47)"``);
   ``assemble_line_examples`` (Task 3) expects ``gene_effect`` indexed by the
   ST-vocabulary gene SYMBOL the predicted-response cache keys its genes by
   (post :data:`~aivc_model.tx1_predicted_response.SLICE_SYMBOL_ALIASES`
   resolution). :func:`build_training_gene_effect_by_line` is that
   translation, built once from the frozen slice's own
   ``gene_symbol``/``depmap_column`` pairing.

An open gap this module does NOT resolve (Task 2's own docstring flagged it
first): ``state.gene_vocabulary_path`` must be Phase C's exact, ordered, full
perturbation-adapter training vocabulary --
:class:`~aivc_model.model.PerturbationVectorAdapter` keys its trainable
"missing" parameters by construction-time list POSITION, not gene identity,
so the wrong order/vocabulary silently loads the wrong gene's vector into
each slot once Phase C's checkpoint loads. No Phase C artifact under today's
``results/experiments/12_tx1_st_geneeffect/phase_c/`` output tree is confirmed
to hold that exact list in that exact order; producing it is a follow-up,
not something this task invents or guesses at.
"""

from __future__ import annotations

import importlib
import json
import logging
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from aivc_model.tx1_geneeffect_data import LineRoleSplit, load_depmap_gene_effect
from aivc_model.tx1_geneeffect_eval import load_panels, load_slice
from aivc_model.tx1_geneeffect_head import Tx1GeneEffectHead
from aivc_model.tx1_geneeffect_pipeline import (
    Tx1GeneEffectHeadConfig,
    validate_config_shape,
    validate_gene_coverage,
    validate_roles,
    validate_widths,
)
from aivc_model.tx1_geneeffect_predict import (
    assemble_test_line_examples,
    build_tx1_3b_st_predictions,
)
from aivc_model.tx1_geneeffect_train import TrainingConfig, train_tx1_geneeffect_head
from aivc_model.tx1_geneeffect_train_io import (
    assemble_split_examples,
    build_provenance,
    save_checkpoint,
    write_provenance,
)
from aivc_model.tx1_predicted_response import (
    SLICE_SYMBOL_ALIASES,
    ForwardOnlyStateModel,
    construct_forward_only_model,
    generate_predicted_response_for_line,
    load_forward_only_checkpoint,
)
from aivc_model.tx1_predicted_response_cache import (
    load_predicted_response_cache,
    predicted_response_fingerprint,
    write_predicted_response_cache,
)

_LOGGER = logging.getLogger(__name__)

__all__ = [
    "build_forward_only_model",
    "build_training_gene_effect_by_line",
    "emit_test_predictions",
    "run_training_pipeline",
    "warm_predicted_response_cache",
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
    ST-vocabulary-symbol-indexed Series ``assemble_line_examples`` expects.

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


def warm_predicted_response_cache(
    model: ForwardOnlyStateModel,
    tx1_cache_dir: Path,
    predicted_response_cache_dir: Path,
    model_id: str,
    role: str,
    genes: Sequence[str],
    arm: str,
    cell_set_len: int,
    seed: int,
    st_checkpoint_path: Path,
    phase_b_manifest_path: Path,
) -> str:
    """Ensure ``model_id``/``arm``'s predicted-response cache entry is fresh
    (D11); return its fingerprint (for
    ``tx1_geneeffect_train_io.assemble_split_examples``'s
    ``fingerprint_by_line``).

    Reuses an existing cache entry when its fingerprint already matches;
    otherwise forwards ST for every gene and writes a fresh entry. Never
    partially reuses a stale entry (D11).
    """
    fingerprint = predicted_response_fingerprint(
        st_checkpoint_path=st_checkpoint_path,
        phase_b_manifest_path=phase_b_manifest_path,
        model_id=model_id,
        genes=genes,
        seed=seed,
        arm=arm,
        cell_set_len=cell_set_len,
    )
    try:
        load_predicted_response_cache(
            predicted_response_cache_dir, model_id, arm, fingerprint
        )
        _LOGGER.info("predicted-response cache hit for %s/%s", model_id, arm)
        return fingerprint
    except (FileNotFoundError, ValueError) as exc:
        _LOGGER.info(
            "predicted-response cache miss for %s/%s (%s); regenerating",
            model_id,
            arm,
            exc,
        )
    responses = {
        gene: generate_predicted_response_for_line(
            tx1_cache_dir,
            model_id,
            role,
            model,
            gene,
            arm=arm,
            cell_set_len=cell_set_len,
            seed=seed,
            require_training_role=True,
        )
        for gene in genes
    }
    write_predicted_response_cache(
        predicted_response_cache_dir, model_id, arm, fingerprint, responses
    )
    return fingerprint


def run_training_pipeline(
    config: Tx1GeneEffectHeadConfig,
    *,
    line_manifest_path: Path,
    phase_a_dir: Path,
    tx1_cache_dir: Path,
    predicted_response_cache_dir: Path,
    depmap_gene_effect_path: Path,
    run_dir: Path | None = None,
    emit_test_predictions_flag: bool = True,
) -> dict[str, Path]:
    """Run the full Phase D chain: line selection -> predicted response ->
    head training -> (optionally) the Phase F table.

    D6/D12 are enforced by the functions this delegates to (Task 1's
    ``assert_training_role``/``build_line_role_split``, Task 3's
    ``_assert_examples_admissible``) -- never re-derived here.

    Returns:
        Output paths: always ``run_dir``/``checkpoint``/``provenance``; also
        ``predictions`` when ``emit_test_predictions_flag``.
    """
    validate_config_shape(config, check_paths=True)
    validate_widths(config, tx1_cache_dir)
    manifest, registration, split = validate_roles(
        line_manifest_path, config.validation_lines_path
    )
    role_by_id = {
        str(row.model_id): str(row.role) for row in manifest.itertuples(index=False)
    }

    phase_a_registration = json.loads(
        (Path(phase_a_dir) / "phase_a_registration.json").read_text()
    )
    slice_df = load_slice(Path(phase_a_dir) / "differentially_essential_slice.csv")
    resolved_genes = validate_gene_coverage(slice_df, config.state.gene_vocabulary_path)
    vocabulary_genes = json.loads(Path(config.state.gene_vocabulary_path).read_text())

    model = build_forward_only_model(config, vocabulary_genes)
    load_forward_only_checkpoint(model, config.state.st_checkpoint_path)

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
    fingerprint_by_line = {
        model_id: warm_predicted_response_cache(
            model,
            tx1_cache_dir,
            predicted_response_cache_dir,
            model_id,
            role_by_id[model_id],
            resolved_genes,
            config.arm,
            config.state.cell_set_len,
            config.state.response_generation_seed,
            config.state.st_checkpoint_path,
            phase_b_manifest_path,
        )
        for model_id in train_and_validation_ids
    }

    train_lines, validation_lines = assemble_split_examples(
        manifest,
        registration,
        arm=config.arm,
        tx1_cache_dir=tx1_cache_dir,
        predicted_response_cache_dir=predicted_response_cache_dir,
        fingerprint_by_line=fingerprint_by_line,
        gene_effect_by_line=gene_effect_by_line,
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

    resolved_run_dir = (
        Path(run_dir) if run_dir else Path(config.output_dir) / "runs" / config.run_id
    )
    checkpoint_path = save_checkpoint(head, resolved_run_dir / "models" / "head.pt")
    provenance = build_provenance(
        st_checkpoint_path=config.state.st_checkpoint_path,
        phase_b_manifest_path=phase_b_manifest_path,
        depmap_gene_effect_path=depmap_gene_effect_path,
        arm=config.arm,
        config=training_config,
        result=result,
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
) -> Path:
    """Build and write the D1 ``tx1_3b_st`` predictions table for the 9
    frozen held-out lines (Task 4).

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
    test_lines = [
        assemble_test_line_examples(
            model_id,
            role_by_id[model_id],
            model,
            config.arm,
            tx1_cache_dir,
            slice_df,
            depmap_frame.loc[model_id],
            cell_set_len=config.state.cell_set_len,
            seed=config.state.response_generation_seed,
        )
        for model_id in split.test_model_ids
    ]
    predictions = build_tx1_3b_st_predictions(
        head,
        test_lines,
        panels,
        k_schedule=list(config.phase_f.k_schedule),
        method=config.phase_f.method,
        alpha=config.phase_f.alpha,
        residual=config.phase_f.residual,
    )
    out_path = Path(run_dir) / "tx1_3b_st_predictions.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(out_path, index=False)
    return out_path
