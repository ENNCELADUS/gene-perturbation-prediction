"""Tests for src/aivc_model/tx1_geneeffect_predict.py -- Phase D Task 4:
emitting the Phase F artifact contract for the 9 held-out lines.

This is the seam Discovery A flagged as never exercised anywhere in this
repo: a real ``Tx1GeneEffectHead`` forward pass, through
``tx1_fewshot_calibration.make_predictions_long``, validated by the REAL
(not reimplemented) ``tx1_geneeffect_eval`` private validators. No GPU, no
real Phase C checkpoint, no real Tx1 embeddings, no real DepMap file exist
on this machine, so every test builds tiny synthetic fixtures directly, the
same way ``test_tx1_geneeffect_train.py`` and ``test_tx1_predicted_response.py``
do. Nothing here is allowed to skip silently.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from aivc_model.model import LinearMockStateModel, PerturbationVectorAdapter
from aivc_model.model import StateForwardAdapter
from aivc_model.tx1_embed_cache import EMBEDDING_WIDTH, write_line_cache
from aivc_model.tx1_fewshot_calibration import fit_ridge_calibration
from aivc_model.tx1_geneeffect_data import TEST_ROLE, TRAIN_HEAD_ROLE
from aivc_model.tx1_geneeffect_eval import (
    EvaluationContractError,
    _validate_evaluation_inputs,
    _validate_no_duplicate_prediction_keys,
    _validate_panel_aware_coverage,
)
from aivc_model.tx1_geneeffect_head import Tx1GeneEffectHead
from aivc_model.tx1_geneeffect_train import LineExamples, train_tx1_geneeffect_head
from aivc_model.tx1_geneeffect_predict import (
    assemble_test_line_features,
    assert_test_role,
    build_test_line_predictions,
    build_tx1_3b_st_predictions,
)
from aivc_model.tx1_predicted_response import ARM_HVG, ForwardOnlyStateModel

_OUTPUT_DIM = 4
_PERT_DIM = 3


# ---------------------------------------------------------------------------
# Shared synthetic fixture builders
# ---------------------------------------------------------------------------


def _forward_only_model(genes: list[str], input_dim: int) -> ForwardOnlyStateModel:
    """A small, freshly initialized ST + perturbation-adapter model."""
    state_model = LinearMockStateModel(input_dim, _OUTPUT_DIM, _PERT_DIM)
    known_vectors = {genes[0]: np.ones(_PERT_DIM, dtype=np.float32)}
    perturbations = PerturbationVectorAdapter(list(genes), known_vectors, _PERT_DIM)
    return ForwardOnlyStateModel(StateForwardAdapter(state_model), perturbations)


def _write_fake_line_cache(
    cache_dir: Path, model_id: str, n_cells: int, hvg_width: int = 3, seed: int = 0
) -> None:
    rng = np.random.default_rng(seed)
    embeddings = rng.normal(size=(n_cells, EMBEDDING_WIDTH)).astype(np.float32)
    hvg = rng.normal(size=(n_cells, hvg_width)).astype(np.float32)
    obs = pd.DataFrame(index=[f"{model_id}-{i}" for i in range(n_cells)])
    write_line_cache(
        cache_dir,
        model_id,
        embeddings,
        hvg,
        obs,
        hvg_gene_order=[f"HVG{i}" for i in range(hvg_width)],
    )


def _make_line_examples(
    model_id: str,
    role: str,
    *,
    n_genes: int,
    response_dim: int,
    basal_dim: int,
    n_response_cells: int = 6,
    n_basal_cells: int = 8,
    seed: int = 0,
) -> LineExamples:
    """A synthetic LineExamples with depmap_column-shaped gene identifiers."""
    rng = np.random.default_rng(seed)
    genes = tuple(f"GENE{i} ({i})" for i in range(n_genes))
    targets_np = rng.uniform(-2.0, 2.0, size=n_genes).astype(np.float32)
    response_bags = tuple(
        torch.as_tensor(
            rng.normal(size=(n_response_cells, response_dim)).astype(np.float32)
        )
        for _ in genes
    )
    basal_bag = torch.as_tensor(
        rng.normal(size=(n_basal_cells, basal_dim)).astype(np.float32)
    )
    return LineExamples(
        model_id=model_id,
        role=role,
        arm=ARM_HVG,
        basal_bag=basal_bag,
        genes=genes,
        response_bags=response_bags,
        targets=torch.as_tensor(targets_np),
    )


def _build_panels(
    model_ids: list[str],
    slice_genes: list[str],
    seed: int,
    n_panels: int,
    n_labels: int,
) -> pd.DataFrame:
    """A synthetic k_label_panels.csv-shaped DataFrame (mirrors the frozen
    contract's shape: exactly n_labels unique genes, label_order 1..n_labels,
    per (model_id, panel))."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for model_id in model_ids:
        for panel in range(n_panels):
            panel_seed = int(rng.integers(0, 2**31))
            order = rng.permutation(len(slice_genes))[:n_labels]
            for label_order, gene_idx in enumerate(order, start=1):
                rows.append(
                    {
                        "model_id": model_id,
                        "panel": panel,
                        "panel_seed": panel_seed,
                        "label_order": label_order,
                        "depmap_column": slice_genes[gene_idx],
                    }
                )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# assert_test_role
# ---------------------------------------------------------------------------


def test_assert_test_role_accepts_test_role() -> None:
    assert_test_role(TEST_ROLE, "ACH-1")  # must not raise


def test_assert_test_role_rejects_non_test_role() -> None:
    with pytest.raises(ValueError, match="held-out inference"):
        assert_test_role(TRAIN_HEAD_ROLE, "ACH-1")


# ---------------------------------------------------------------------------
# assemble_test_line_examples: full-slice coverage + the require_training_role
# =False opt-out, used deliberately and only after assert_test_role passes.
# ---------------------------------------------------------------------------


def test_assemble_test_line_examples_covers_full_slice_including_missing_target(
    tmp_path: Path,
) -> None:
    """Three slice genes: one plain, one under a HISTORICAL alias (CRIPTO ->
    TDGF1, the measured Task 0 alias), one with NO DepMap target on this
    line. All three must still get a response bag and a row; the missing
    target must be NaN, not a dropped gene -- unlike Task 3's training
    assembler, which would drop it."""
    model_id = "ACH-HELD-OUT"
    cache_dir = tmp_path / "tx1_cache"
    _write_fake_line_cache(cache_dir, model_id, n_cells=5, hvg_width=3)

    slice_df = pd.DataFrame(
        {
            "depmap_column": ["G1 (1)", "CRIPTO (2)", "G3 (3)"],
            "gene_symbol": ["G1", "CRIPTO", "G3"],
        }
    )
    # ST vocabulary carries the CURRENT HGNC symbol, not the historical one.
    model = _forward_only_model(["G1", "TDGF1", "G3"], input_dim=3)
    gene_effect = pd.Series({"G1 (1)": -1.5, "CRIPTO (2)": 0.75})  # G3 (3) absent

    line = assemble_test_line_features(
        model_id,
        TEST_ROLE,
        model,
        ARM_HVG,
        cache_dir,
        slice_df,
        gene_effect,
        moments=2,
        cell_set_len=4,
        seed=0,
    )

    assert line.role == TEST_ROLE
    assert line.genes == ("G1 (1)", "CRIPTO (2)", "G3 (3)")
    assert line.features.shape == (3, (_OUTPUT_DIM + 3) * 2)
    targets = line.targets.tolist()
    assert targets[0] == pytest.approx(-1.5)
    assert targets[1] == pytest.approx(0.75)
    assert np.isnan(targets[2])


def test_assemble_test_line_examples_rejects_non_test_role(tmp_path: Path) -> None:
    model_id = "ACH-TRAIN"
    cache_dir = tmp_path / "tx1_cache"
    _write_fake_line_cache(cache_dir, model_id, n_cells=5)
    slice_df = pd.DataFrame({"depmap_column": ["G1 (1)"], "gene_symbol": ["G1"]})
    model = _forward_only_model(["G1"], input_dim=3)

    with pytest.raises(ValueError, match="held-out inference"):
        assemble_test_line_features(
            model_id,
            TRAIN_HEAD_ROLE,
            model,
            ARM_HVG,
            cache_dir,
            slice_df,
            pd.Series(dtype=float),
            moments=2,
            cell_set_len=4,
            seed=0,
        )


def test_held_out_line_examples_can_never_enter_training() -> None:
    """D6, proven across the seam: a LineExamples this module assembles for
    held-out inference must still be rejected by Task 3's own training
    guard if a caller mistakenly fed it to train_tx1_geneeffect_head."""
    held_out = _make_line_examples(
        "ACH-HELD-OUT", TEST_ROLE, n_genes=3, response_dim=2, basal_dim=2, seed=1
    )
    with pytest.raises(ValueError, match="refusing to admit"):
        train_tx1_geneeffect_head(
            [held_out], [held_out], config=_tiny_training_config()
        )


def _tiny_training_config():
    from aivc_model.tx1_geneeffect_train import TrainingConfig

    return TrainingConfig(epochs=1)


# ---------------------------------------------------------------------------
# D2: the pooled features matrix -- not just the scalar -- reaches the
# few-shot ridge calibrator at its expected width.
# ---------------------------------------------------------------------------


def test_features_reach_ridge_calibrator_at_expected_width() -> None:
    response_dim, basal_dim, moments = 4, 3, 2
    line = _make_line_examples(
        "ACH-1",
        TEST_ROLE,
        n_genes=6,
        response_dim=response_dim,
        basal_dim=basal_dim,
        seed=2,
    )
    head = Tx1GeneEffectHead(
        response_dim=response_dim, basal_dim=basal_dim, hidden=16, moments=moments
    )

    predictions = build_test_line_predictions(
        head,
        line,
        panels_for_line=_build_panels(
            [line.model_id], list(line.genes), seed=3, n_panels=1, n_labels=2
        ),
        k_schedule=[0, 2],
    )
    assert set(predictions["k"].unique()) == {0, 2}

    from aivc_model.tx1_geneeffect_train import compute_line_features_and_predictions

    features, _ = compute_line_features_and_predictions(head, line)
    expected_width = (response_dim + basal_dim) * moments
    assert features.shape == (6, expected_width)

    # The reducer still consumes the complete pooled-feature vector, while
    # the supervised ridge only receives a k-compatible low-dimensional view.
    calibrator = fit_ridge_calibration(
        features_labeled=features[:2].numpy(),
        y_labeled=line.targets[:2].numpy(),
    )
    assert calibrator.feature_reducer.input_dim == expected_width
    assert calibrator.weights.shape == (1,)


# ---------------------------------------------------------------------------
# The central seam test: real head forward -> features -> make_predictions_
# long -> table, validated by the REAL _validate_panel_aware_coverage.
# ---------------------------------------------------------------------------


def _valid_two_line_fixture(k_schedule: list[int]):
    response_dim, basal_dim = 4, 3
    line_a = _make_line_examples(
        "ACH-A",
        TEST_ROLE,
        n_genes=12,
        response_dim=response_dim,
        basal_dim=basal_dim,
        seed=10,
    )
    line_b = _make_line_examples(
        "ACH-B",
        TEST_ROLE,
        n_genes=12,
        response_dim=response_dim,
        basal_dim=basal_dim,
        seed=11,
    )
    head = Tx1GeneEffectHead(response_dim=response_dim, basal_dim=basal_dim, hidden=16)
    panels = _build_panels(
        [line_a.model_id, line_b.model_id],
        list(line_a.genes),
        seed=12,
        n_panels=3,
        n_labels=4,
    )
    predictions = build_tx1_3b_st_predictions(
        head, [line_a, line_b], panels, k_schedule=k_schedule, method="tx1_3b_st"
    )
    return line_a, line_b, panels, predictions


def test_build_tx1_3b_st_predictions_passes_the_real_panel_aware_validator() -> None:
    k_schedule = [0, 2, 4]
    line_a, line_b, panels, predictions = _valid_two_line_fixture(k_schedule)
    slice_genes = set(line_a.genes)

    assert len(predictions) == 2 * 3 * len(k_schedule) * 12  # lines*panels*k*genes

    for line in (line_a, line_b):
        line_preds = predictions[
            (predictions["model_id"] == line.model_id)
            & (predictions["method"] == "tx1_3b_st")
        ]
        line_panels = panels[panels["model_id"] == line.model_id]
        # The REAL validator, not a reimplementation -- must not raise.
        _validate_panel_aware_coverage(
            line_preds,
            line_panels,
            line.model_id,
            "tx1_3b_st",
            slice_genes,
            k_schedule,
            max_listed=20,
        )


def test_build_tx1_3b_st_predictions_rejects_duplicate_line_ids() -> None:
    line = _make_line_examples(
        "ACH-A", TEST_ROLE, n_genes=4, response_dim=2, basal_dim=2, seed=13
    )
    head = Tx1GeneEffectHead(response_dim=2, basal_dim=2, hidden=8)
    panels = _build_panels(
        [line.model_id], list(line.genes), seed=14, n_panels=1, n_labels=1
    )

    with pytest.raises(ValueError, match="duplicate held-out line"):
        build_tx1_3b_st_predictions(head, [line, line], panels, k_schedule=[0])


def test_build_tx1_3b_st_predictions_rejects_non_test_role_line() -> None:
    bad = _make_line_examples(
        "ACH-TRAIN", TRAIN_HEAD_ROLE, n_genes=4, response_dim=2, basal_dim=2, seed=15
    )
    head = Tx1GeneEffectHead(response_dim=2, basal_dim=2, hidden=8)
    panels = _build_panels(
        [bad.model_id], list(bad.genes), seed=16, n_panels=1, n_labels=1
    )

    with pytest.raises(ValueError, match="held-out inference"):
        build_tx1_3b_st_predictions(head, [bad], panels, k_schedule=[0])


def test_build_tx1_3b_st_predictions_rejects_empty_lines() -> None:
    head = Tx1GeneEffectHead(response_dim=2, basal_dim=2, hidden=8)
    with pytest.raises(ValueError, match="non-empty"):
        build_tx1_3b_st_predictions(head, [], pd.DataFrame(), k_schedule=[0])


# ---------------------------------------------------------------------------
# Negative: a deliberately incomplete table must FAIL the real validator --
# proving the positive test above is not passing vacuously.
# ---------------------------------------------------------------------------


def test_incomplete_table_fails_the_real_panel_aware_validator() -> None:
    k_schedule = [0, 2, 4]
    line_a, _line_b, panels, predictions = _valid_two_line_fixture(k_schedule)
    slice_genes = set(line_a.genes)
    line_panels = panels[panels["model_id"] == line_a.model_id]

    line_preds = predictions[
        (predictions["model_id"] == line_a.model_id)
        & (predictions["method"] == "tx1_3b_st")
    ]
    # Drop every row for (panel=0, k=2): an entire required cell vanishes.
    incomplete = line_preds[~((line_preds["panel"] == 0) & (line_preds["k"] == 2))]

    with pytest.raises(EvaluationContractError, match="missing"):
        _validate_panel_aware_coverage(
            incomplete,
            line_panels,
            line_a.model_id,
            "tx1_3b_st",
            slice_genes,
            k_schedule,
            max_listed=20,
        )


def test_duplicate_prediction_key_fails_the_real_validator() -> None:
    k_schedule = [0, 2, 4]
    line_a, _line_b, panels, predictions = _valid_two_line_fixture(k_schedule)
    slice_genes = set(line_a.genes)
    test_set = {line_a.model_id, _line_b.model_id}

    tx1_preds = predictions[predictions["method"] == "tx1_3b_st"]
    duplicated_row = tx1_preds.iloc[[0]].copy()
    duplicated_row["base_pred"] = duplicated_row["base_pred"] + 1.0  # differs, too
    dirtied = pd.concat([tx1_preds, duplicated_row], ignore_index=True)

    with pytest.raises(EvaluationContractError, match="duplicate"):
        _validate_no_duplicate_prediction_keys(
            dirtied, slice_genes, test_set, {"tx1_3b_st"}
        )


def test_cross_method_y_true_mismatch_fails_full_strict_validation() -> None:
    """Exercises the cross-method y_true consistency path via the same
    _validate_evaluation_inputs the strict evaluate() gate calls."""
    k_schedule = [0, 2, 4]
    line_a, line_b, panels, tx1_predictions = _valid_two_line_fixture(k_schedule)

    manifest = pd.DataFrame(
        {"model_id": [line_a.model_id, line_b.model_id], "role": [TEST_ROLE, TEST_ROLE]}
    )
    slice_df = pd.DataFrame({"depmap_column": list(line_a.genes)})

    baseline_rows: list[dict[str, object]] = []
    for line in (line_a, line_b):
        y_true = line.targets.numpy()
        for gene, truth in zip(line.genes, y_true):
            baseline_rows.append(
                {
                    "model_id": line.model_id,
                    "depmap_column": gene,
                    "method": "copy_k562",
                    "base_pred": 0.0,
                    "y_true": float(truth),
                }
            )
    baseline = pd.DataFrame(baseline_rows)
    # Deliberately corrupt one (line, gene) pair's baseline y_true so it
    # disagrees with the tx1_3b_st table's y_true for the identical key.
    mismatch_mask = (baseline["model_id"] == line_a.model_id) & (
        baseline["depmap_column"] == line_a.genes[0]
    )
    baseline.loc[mismatch_mask, "y_true"] += 5.0

    combined = pd.concat([tx1_predictions, baseline], ignore_index=True)

    with pytest.raises(EvaluationContractError, match="y_true disagrees"):
        _validate_evaluation_inputs(
            combined,
            manifest,
            slice_df,
            panels,
            methods=["tx1_3b_st"],
            baseline_method="copy_k562",
            k_schedule=k_schedule,
            n_panels=3,
            n_labels=4,
        )
