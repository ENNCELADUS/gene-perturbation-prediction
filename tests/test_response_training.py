"""Tests for the Stage 1 response-model training objective and trainer.

The loss is the part worth testing hard: it is distributional, so the usual
"does it go down" check cannot distinguish a model that matches the mean
while collapsing spread -- exactly the failure the retired backbone showed.
These cover each term's discriminating behavior, the per-line held-out gene
split, the DDP even-step sharding invariants, held-out validation reduction,
best-checkpoint selection with early stopping, and that training actually
moves the module it is handed.
"""

from __future__ import annotations

import json

from accelerate import Accelerator
import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from aivc_model.response_training import (
    SELECTION_METRIC_NAME,
    ResponseLoss,
    ResponseLossWeights,
    TrainingConfig,
    _GeneIndexDataset,
    _heldout_model_losses,
    _is_better_metric,
    _pad_gene_indices,
    _per_anchor_mean_from_gathered,
    _record_objective_weights,
    _run_train_epoch,
    _select_best_epoch,
    _validate_anchor_weights,
    energy_distance,
    evaluate_response_model,
    mean_delta_mse,
    predict_bags,
    split_heldout_genes,
    train_response_model,
)


class _ShiftModel(nn.Module):
    """Adds a learnable per-gene shift to control cells; no head, like ST."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.shift = nn.Parameter(torch.zeros(dim))

    def forward(self, chunks, gene, batch_index_chunks):
        return tuple(chunk + self.shift for chunk in chunks)


def _bag(rows: int, dim: int, offset: float = 0.0, scale: float = 1.0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(rows * 100 + dim)
    return torch.randn(rows, dim, generator=generator) * scale + offset


# --- the two loss terms -------------------------------------------------


def test_mean_delta_mse_is_zero_when_the_mean_shift_matches() -> None:
    control_mean = torch.zeros(4)
    observed = torch.full((6, 4), 2.0)
    predicted = torch.full((3, 4), 2.0)
    assert float(mean_delta_mse(predicted, observed, control_mean)) == pytest.approx(
        0.0
    )


def test_mean_delta_mse_ignores_spread() -> None:
    """The mean term alone cannot see a collapsed prediction -- hence energy."""
    control_mean = torch.zeros(4)
    observed = _bag(64, 4, offset=1.0, scale=1.0)
    collapsed = observed.mean(dim=0, keepdim=True).repeat(64, 1)
    assert float(mean_delta_mse(collapsed, observed, control_mean)) == pytest.approx(
        0.0, abs=1e-6
    )


def test_energy_distance_detects_the_collapse_the_mean_term_misses() -> None:
    observed = _bag(64, 4, offset=1.0, scale=1.0)
    collapsed = observed.mean(dim=0, keepdim=True).repeat(64, 1)
    assert float(energy_distance(collapsed, observed)) > 0.1


def test_energy_distance_is_near_zero_for_samples_of_one_distribution() -> None:
    left = _bag(128, 3, offset=0.0, scale=1.0)
    right = torch.randn(128, 3, generator=torch.Generator().manual_seed(7))
    assert float(energy_distance(left, right)) < 0.15


def test_energy_distance_rejects_an_empty_bag() -> None:
    with pytest.raises(ValueError, match="at least one cell"):
        energy_distance(torch.zeros(0, 3), _bag(4, 3))


def test_response_loss_reports_both_terms() -> None:
    loss_fn = ResponseLoss()
    total, parts = loss_fn(_bag(8, 3), _bag(9, 3, offset=1.0), torch.zeros(3))
    assert {"mean_delta_mse", "energy_distance", "loss"} <= set(parts)
    assert float(total) == pytest.approx(parts["loss"])


def test_response_loss_weights_reject_an_all_zero_objective() -> None:
    with pytest.raises(ValueError, match="at least one loss term"):
        ResponseLossWeights(mean_delta=0.0, energy=0.0)


# --- held-out gene split ------------------------------------------------


def test_heldout_genes_are_split_per_line_and_are_deterministic() -> None:
    genes_by_line = {
        "ACH-000551": [f"G{i}" for i in range(20)],
        "ACH-000995": [f"G{i}" for i in range(20)],
    }
    first = split_heldout_genes(genes_by_line, fraction=0.2, seed=1)
    again = split_heldout_genes(genes_by_line, fraction=0.2, seed=1)
    assert first == again
    assert len(first["ACH-000551"]) == 4
    # Same gene names, different lines: the split is per line, so the two
    # held-out sets are chosen independently rather than mirrored.
    assert first["ACH-000551"] != first["ACH-000995"]


def test_heldout_split_is_stable_when_a_gene_is_added() -> None:
    """Adding a gene must not reshuffle the existing assignment."""
    base = {"L": [f"G{i}" for i in range(50)]}
    grown = {"L": [f"G{i}" for i in range(51)]}
    kept = split_heldout_genes(base, fraction=0.2, seed=3)["L"]
    after = split_heldout_genes(grown, fraction=0.2, seed=3)["L"]
    assert len(kept & after) >= len(kept) - 1


def test_heldout_split_rejects_a_line_too_small_to_hold_out() -> None:
    with pytest.raises(ValueError, match="cannot yield a held-out set"):
        split_heldout_genes({"L": ["A", "B"]}, fraction=0.2, seed=1)


# --- TrainingConfig validation -------------------------------------------


def test_training_config_rejects_a_sub_one_max_epochs() -> None:
    with pytest.raises(ValueError, match="max_epochs must be >= 1"):
        TrainingConfig(max_epochs=0)


def test_training_config_rejects_a_sub_one_patience() -> None:
    with pytest.raises(ValueError, match="patience must be >= 1"):
        TrainingConfig(patience=0)


def test_training_config_rejects_a_too_small_max_bag() -> None:
    with pytest.raises(ValueError, match="max_bag must be >= 2"):
        TrainingConfig(max_bag=1)


def test_training_config_rejects_a_sub_one_gene_batch_size() -> None:
    with pytest.raises(ValueError, match="gene_batch_size"):
        TrainingConfig(gene_batch_size=0)


def test_training_config_rejects_a_sub_one_validation_gene_batch_size() -> None:
    with pytest.raises(ValueError, match="validation_gene_batch_size"):
        TrainingConfig(validation_gene_batch_size=0)


# --- even-step DDP sharding -----------------------------------------------


def test_pad_gene_indices_pads_to_a_multiple_of_world_size() -> None:
    padded, is_padding = _pad_gene_indices(np.arange(5), world_size=3)
    assert len(padded) % 3 == 0
    assert len(padded) == len(is_padding)
    # Real indices are untouched and come first; padding repeats real data,
    # never invents an empty/zero record.
    assert list(padded[:5]) == [0, 1, 2, 3, 4]
    assert not any(is_padding[:5])
    assert all(is_padding[5:])
    assert set(padded[5:]) <= set(padded[:5])


def test_pad_gene_indices_is_a_noop_when_already_even() -> None:
    padded, is_padding = _pad_gene_indices(np.arange(6), world_size=3)
    assert list(padded) == [0, 1, 2, 3, 4, 5]
    assert not is_padding.any()


def test_pad_gene_indices_fills_complete_batches_on_every_rank() -> None:
    padded, is_padding = _pad_gene_indices(np.arange(10), world_size=2, batch_size=4)
    assert len(padded) == 16
    assert len(padded) % (2 * 4) == 0
    assert int(is_padding.sum()) == 6


def test_pad_gene_indices_rejects_an_empty_index_array() -> None:
    with pytest.raises(ValueError, match="no batches to shard"):
        _pad_gene_indices(np.array([], dtype=np.int64), world_size=2)


def _batches(dim: int = 3, n: int = 4, model_id: str = "ACH-000551") -> list[dict]:
    out = []
    for index in range(n):
        control = _bag(16, dim)
        out.append(
            {
                "gene": f"G{index}",
                "model_id": model_id,
                "control": control,
                "control_target": control,
                "observed": control + 1.5,
            }
        )
    return out


def test_a_padded_training_step_backpropagates_exactly_zero_gradient() -> None:
    """A padded step must run the full forward/backward, not be skipped.

    Skipping it would desync ranks under ``static_graph=True``; instead its
    loss is multiplied by zero before backward. Isolating a single padded
    step lets us check both halves directly: the gradient exists (so the
    backward genuinely ran) and it is exactly zero (so it did not move the
    model).
    """
    model = _ShiftModel(3)
    records = _batches(n=1)
    accelerator = Accelerator(cpu=True)
    loader = accelerator.prepare(
        DataLoader(_GeneIndexDataset(np.array([0]), np.array([True])), batch_size=1)
    )
    optimizer = accelerator.prepare(torch.optim.AdamW(model.parameters(), lr=0.0))
    config = TrainingConfig(max_bag=16, log_every=0)
    generator = torch.Generator().manual_seed(config.seed)

    with pytest.raises(ValueError, match="no non-padding training steps"):
        _run_train_epoch(
            model,
            loader,
            records,
            optimizer,
            ResponseLoss(),
            config,
            accelerator,
            generator,
            epoch=0,
        )
    assert model.shift.grad is not None
    assert torch.allclose(model.shift.grad, torch.zeros_like(model.shift.grad))


def test_every_rank_step_count_includes_padding_and_matches_loader_length() -> None:
    """Padding equalizes total step count -- the property DDP needs."""
    model = _ShiftModel(3)
    records = _batches(n=1)
    accelerator = Accelerator(cpu=True)
    loader = accelerator.prepare(
        DataLoader(
            _GeneIndexDataset(np.array([0, 0, 0]), np.array([False, True, True])),
            batch_size=1,
        )
    )
    optimizer = accelerator.prepare(torch.optim.AdamW(model.parameters(), lr=0.0))
    config = TrainingConfig(max_bag=16, log_every=0)
    generator = torch.Generator().manual_seed(config.seed)

    metrics, local_steps = _run_train_epoch(
        model,
        loader,
        records,
        optimizer,
        ResponseLoss(),
        config,
        accelerator,
        generator,
        epoch=0,
    )
    assert local_steps == 3
    # The two padded duplicates never enter the reported mean.
    single_metrics, single_steps = _run_train_epoch(
        model,
        accelerator.prepare(
            DataLoader(
                _GeneIndexDataset(np.array([0]), np.array([False])), batch_size=1
            )
        ),
        records,
        optimizer,
        ResponseLoss(),
        config,
        accelerator,
        torch.Generator().manual_seed(config.seed),
        epoch=0,
    )
    assert single_steps == 1
    assert metrics == pytest.approx(single_metrics)


def test_train_epoch_uses_one_optimizer_step_per_gene_batch() -> None:
    model = _ShiftModel(3)
    records = _batches(n=4)
    accelerator = Accelerator(cpu=True)
    loader = accelerator.prepare(
        DataLoader(
            _GeneIndexDataset(np.arange(4), np.zeros(4, dtype=bool)),
            batch_size=2,
        )
    )
    optimizer = accelerator.prepare(torch.optim.AdamW(model.parameters(), lr=0.0))
    config = TrainingConfig(max_bag=16, gene_batch_size=2, log_every=0)

    metrics, local_steps = _run_train_epoch(
        model,
        loader,
        records,
        optimizer,
        ResponseLoss(),
        config,
        accelerator,
        torch.Generator().manual_seed(config.seed),
        epoch=0,
    )

    assert local_steps == 2
    assert np.isfinite(metrics["loss"])


def test_mixed_padding_does_not_attenuate_the_real_record_gradient() -> None:
    records = _batches(n=1)

    def gradient(indices: np.ndarray, padding: np.ndarray) -> torch.Tensor:
        model = _ShiftModel(3)
        accelerator = Accelerator(cpu=True)
        loader = accelerator.prepare(
            DataLoader(_GeneIndexDataset(indices, padding), batch_size=len(indices))
        )
        optimizer = accelerator.prepare(
            torch.optim.AdamW(model.parameters(), lr=0.0, weight_decay=0.0)
        )
        config = TrainingConfig(max_bag=16, gene_batch_size=len(indices), log_every=0)
        _run_train_epoch(
            model,
            loader,
            records,
            optimizer,
            ResponseLoss(),
            config,
            accelerator,
            torch.Generator().manual_seed(config.seed),
            epoch=0,
        )
        assert model.shift.grad is not None
        return model.shift.grad.detach().clone()

    real_only = gradient(np.array([0]), np.array([False]))
    real_plus_padding = gradient(np.array([0, 0]), np.array([False, True]))
    assert torch.allclose(real_only, real_plus_padding)


# --- held-out per-anchor reduction ----------------------------------------


def test_gathered_sum_and_count_differs_from_mean_of_per_rank_means() -> None:
    """Sum-then-divide is required; averaging per-rank means is wrong.

    Rank 0 holds one held-out example at loss 10; rank 1 holds three at loss
    0. The correct pooled mean weights by count (10 / 4); naively averaging
    each rank's own mean (10 and 0) gives 5 -- a different, wrong answer
    whenever ranks carry unequal held-out counts, which even-step padding
    does nothing to prevent.
    """
    sum_matrix = torch.tensor([[10.0], [0.0]])
    count_matrix = torch.tensor([[1.0], [3.0]])
    correct = _per_anchor_mean_from_gathered(sum_matrix, count_matrix)
    naive_mean_of_means = (sum_matrix / count_matrix).mean(dim=0)
    assert float(correct[0]) == pytest.approx(2.5)
    assert float(naive_mean_of_means[0]) == pytest.approx(5.0)
    assert float(correct[0]) != pytest.approx(float(naive_mean_of_means[0]))


# --- selection helpers ------------------------------------------------


def test_is_better_metric_rejects_a_non_finite_candidate() -> None:
    assert _is_better_metric(float("nan"), 1.0, mode="min") is False


def test_is_better_metric_accepts_any_finite_value_over_a_non_finite_incumbent() -> (
    None
):
    """The NaN-handling bug that once froze checkpoint selection at epoch 1."""
    assert _is_better_metric(1e9, float("nan"), mode="min") is True


def test_is_better_metric_compares_normally_once_both_sides_are_finite() -> None:
    assert _is_better_metric(0.5, 1.0, mode="min") is True
    assert _is_better_metric(1.5, 1.0, mode="min") is False


def test_is_better_metric_rejects_an_unknown_mode() -> None:
    with pytest.raises(ValueError, match="unknown selection mode"):
        _is_better_metric(1.0, 2.0, mode="sideways")


def test_select_best_epoch_picks_the_minimum_and_keeps_the_earliest_tie() -> None:
    history = [
        {"epoch": 0, SELECTION_METRIC_NAME: 2.0},
        {"epoch": 1, SELECTION_METRIC_NAME: 1.0},
        {"epoch": 2, SELECTION_METRIC_NAME: 1.0},
    ]
    assert _select_best_epoch(history)["epoch"] == 1


def test_select_best_epoch_skips_non_finite_entries() -> None:
    history = [
        {"epoch": 0, SELECTION_METRIC_NAME: float("nan")},
        {"epoch": 1, SELECTION_METRIC_NAME: 3.0},
    ]
    assert _select_best_epoch(history)["epoch"] == 1


def test_select_best_epoch_rejects_an_empty_history() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        _select_best_epoch([])


def test_select_best_epoch_rejects_an_all_non_finite_history() -> None:
    with pytest.raises(ValueError, match="no epoch produced a finite"):
        _select_best_epoch([{"epoch": 0, SELECTION_METRIC_NAME: float("nan")}])


# --- anchor_weights validation --------------------------------------------


def test_validate_anchor_weights_rejects_weights_not_summing_to_one() -> None:
    with pytest.raises(ValueError, match="must sum to 1"):
        _validate_anchor_weights({"A": 0.4, "B": 0.4}, _batches(n=1, model_id="A"))


def test_validate_anchor_weights_rejects_an_unknown_heldout_model_id() -> None:
    with pytest.raises(ValueError, match="absent from anchor_weights"):
        _validate_anchor_weights({"A": 1.0}, _batches(n=1, model_id="B"))


def test_validate_anchor_weights_rejects_a_weighted_anchor_with_no_heldout_batch() -> (
    None
):
    with pytest.raises(ValueError, match="no held-out batch"):
        _validate_anchor_weights({"A": 0.5, "B": 0.5}, _batches(n=1, model_id="A"))


def test_validate_anchor_weights_allows_a_zero_weight_anchor_with_no_batch() -> None:
    _validate_anchor_weights({"A": 1.0, "B": 0.0}, _batches(n=1, model_id="A"))


def test_record_weights_make_anchor_means_contribute_declared_weights() -> None:
    records = _batches(n=1, model_id="A") + _batches(n=3, model_id="B")
    weights = _record_objective_weights(records, {"A": 0.5, "B": 0.5})

    assert weights == pytest.approx([2.0, 2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0])
    losses = [10.0, 0.0, 0.0, 0.0]
    objective = np.mean([loss * weight for loss, weight in zip(losses, weights)])
    assert objective == pytest.approx(5.0)


# --- training loop ------------------------------------------------------


def test_training_reduces_the_loss_and_updates_the_module(tmp_path) -> None:
    model = _ShiftModel(3)
    before = model.shift.detach().clone()
    report = train_response_model(
        model,
        _batches(n=6),
        _batches(n=3),
        anchor_weights={"ACH-000551": 1.0},
        out_dir=tmp_path,
        config=TrainingConfig(
            max_epochs=6, patience=6, learning_rate=0.2, max_bag=16, log_every=0
        ),
    )
    losses = [epoch["train_loss"] for epoch in report["epochs"]]
    assert losses[-1] < losses[0]
    assert not torch.allclose(before, model.shift.detach())


def test_validation_uses_its_own_gene_batch_size(tmp_path) -> None:
    class RecordingShiftModel(_ShiftModel):
        def __init__(self, dim: int) -> None:
            super().__init__(dim)
            self.condition_counts: list[int] = []

        def forward(self, chunks, gene, batch_index_chunks):
            self.condition_counts.append(len(gene))
            return super().forward(chunks, gene, batch_index_chunks)

    model = RecordingShiftModel(3)
    train_response_model(
        model,
        _batches(n=4),
        _batches(n=4),
        anchor_weights={"ACH-000551": 1.0},
        out_dir=tmp_path,
        config=TrainingConfig(
            max_epochs=1,
            patience=1,
            max_bag=16,
            gene_batch_size=4,
            validation_gene_batch_size=1,
            log_every=0,
        ),
    )

    assert model.condition_counts == [4, 1, 1, 1, 1]


def test_validation_losses_are_identical_for_batch_one_and_batch_many() -> None:
    model = _ShiftModel(3)
    records = _batches(n=4)
    loss_fn = ResponseLoss()
    batched = _heldout_model_losses(
        model, records, loss_fn, torch.device("cpu"), max_bag=16
    )[0]
    separate = [
        _heldout_model_losses(
            model, [record], loss_fn, torch.device("cpu"), max_bag=16
        )[0][0]
        for record in records
    ]

    assert [float(loss.detach()) for loss in batched] == pytest.approx(
        [float(loss.detach()) for loss in separate]
    )


def test_training_rejects_a_non_fixed_control_bag_before_fitting(tmp_path) -> None:
    heldout = _batches(n=1)
    heldout[0]["control"] = _bag(17, 3)
    heldout[0]["control_target"] = heldout[0]["control"]

    with pytest.raises(ValueError, match="fixed control bag must contain exactly 16"):
        train_response_model(
            _ShiftModel(3),
            _batches(n=1),
            heldout,
            anchor_weights={"ACH-000551": 1.0},
            out_dir=tmp_path,
            config=TrainingConfig(max_epochs=1, max_bag=16),
        )


def test_training_rejects_an_empty_train_batch_list(tmp_path) -> None:
    with pytest.raises(ValueError, match="no response batches"):
        train_response_model(
            _ShiftModel(3),
            [],
            _batches(n=1),
            anchor_weights={"ACH-000551": 1.0},
            out_dir=tmp_path,
        )


def test_training_rejects_an_empty_heldout_batch_list(tmp_path) -> None:
    with pytest.raises(ValueError, match="no held-out response batches"):
        train_response_model(
            _ShiftModel(3),
            _batches(n=1),
            [],
            anchor_weights={"ACH-000551": 1.0},
            out_dir=tmp_path,
        )


def test_training_report_has_the_documented_keys(tmp_path) -> None:
    report = train_response_model(
        _ShiftModel(3),
        _batches(n=4),
        _batches(n=2),
        anchor_weights={"ACH-000551": 1.0},
        out_dir=tmp_path,
        config=TrainingConfig(max_epochs=3, patience=3, max_bag=16, log_every=0),
    )
    assert {
        "epochs",
        "best_epoch",
        "best_metric_value",
        "selection_metric",
        "stopped_early",
        "stopped_at_epoch",
        "n_train_batches",
        "n_heldout_batches",
        "world_size",
        "config",
    } <= set(report)
    assert report["selection_metric"] == SELECTION_METRIC_NAME
    assert report["n_train_batches"] == 4
    assert report["n_heldout_batches"] == 2
    assert report["world_size"] == 1
    assert report["config"]["max_epochs"] == 3
    epoch_row = report["epochs"][0]
    assert SELECTION_METRIC_NAME in epoch_row
    assert "heldout_ACH-000551_loss" in epoch_row


def test_training_writes_the_train_log_and_both_checkpoints(tmp_path) -> None:
    train_response_model(
        _ShiftModel(3),
        _batches(n=4),
        _batches(n=2),
        anchor_weights={"ACH-000551": 1.0},
        out_dir=tmp_path,
        config=TrainingConfig(max_epochs=3, patience=3, max_bag=16, log_every=0),
    )
    assert (tmp_path / "train_log.csv").exists()
    for kind in ("best", "final"):
        checkpoint_dir = tmp_path / kind
        assert (checkpoint_dir / "pytorch_model.bin").exists()
        metadata = json.loads((checkpoint_dir / "metadata.json").read_text())
        assert metadata["checkpoint_kind"] == kind
        assert metadata["selection_metric"] == SELECTION_METRIC_NAME
        assert metadata["selection_mode"] == "min"
        assert np.isfinite(metadata["metric_value"])
    final_metadata = json.loads((tmp_path / "final" / "metadata.json").read_text())
    assert "best_metric_value" in final_metadata
    assert "stopped_early" in final_metadata
    assert "patience" in final_metadata


def test_early_stopping_triggers_once_patience_is_exhausted(tmp_path) -> None:
    """A frozen model's held-out loss never improves after epoch 0."""
    report = train_response_model(
        _ShiftModel(3),
        _batches(n=4),
        _batches(n=2),
        anchor_weights={"ACH-000551": 1.0},
        out_dir=tmp_path,
        config=TrainingConfig(
            max_epochs=10, patience=2, learning_rate=0.0, max_bag=16, log_every=0
        ),
    )
    assert report["stopped_early"] is True
    assert report["best_epoch"] == 0
    assert report["stopped_at_epoch"] == 2
    assert len(report["epochs"]) == 3


def test_anchor_weights_error_propagates_from_train_response_model(tmp_path) -> None:
    with pytest.raises(ValueError, match="no held-out batch"):
        train_response_model(
            _ShiftModel(3),
            _batches(n=2, model_id="A"),
            _batches(n=2, model_id="A"),
            anchor_weights={"A": 0.5, "B": 0.5},
            out_dir=tmp_path,
        )


def test_evaluate_reports_the_basal_copy_floor(tmp_path) -> None:
    """A trained shift must beat 'nothing happens', and the report must say so."""
    model = _ShiftModel(3)
    batches = _batches()
    train_response_model(
        model,
        batches,
        _batches(n=2),
        anchor_weights={"ACH-000551": 1.0},
        out_dir=tmp_path,
        config=TrainingConfig(
            max_epochs=8, patience=8, learning_rate=0.2, max_bag=16, log_every=0
        ),
    )
    report = evaluate_response_model(model, batches)
    assert report["basal_copy_loss"] > report["model_loss"]
    assert report["improvement_over_basal_copy"] > 0
    assert "ACH-000551" in report["per_line_model_loss"]
    assert set(report["per_anchor_metrics"]["ACH-000551"]) == {
        "mean_delta_mse",
        "energy_distance",
    }
    assert all(
        np.isfinite(value)
        for value in report["per_anchor_metrics"]["ACH-000551"].values()
    )
    assert np.isfinite(report["model_loss"])


class _WindowedModel(nn.Module):
    """Test double enforcing ST's fixed-window contract."""

    def __init__(self, dim: int, window: int) -> None:
        super().__init__()
        self.shift = nn.Parameter(torch.zeros(dim))
        self.cell_sentence_len = window

    def forward(self, chunks, gene, batch_index_chunks):
        for chunk in chunks:
            if chunk.shape[0] != self.cell_sentence_len:
                raise ValueError("STATE chunks must all equal cell_sentence_len")
        return tuple(chunk + self.shift for chunk in chunks)


def test_predict_bag_pads_and_trims_to_the_window() -> None:
    """A bag that is not a multiple of the window must still forward.

    Training only worked before because max_bag happened to equal the window;
    evaluation passed whole control bags and crashed on exactly this.
    """
    from aivc_model.response_training import predict_bag

    model = _WindowedModel(3, window=8)
    out = predict_bag(model, _bag(21, 3), "G0", seed=0)
    assert out.shape == (21, 3)


def test_predict_bags_combines_conditions_in_one_model_forward() -> None:
    class CountingWindowedModel(_WindowedModel):
        def __init__(self) -> None:
            super().__init__(dim=3, window=8)
            self.forward_calls = 0

        def forward(self, chunks, gene, batch_index_chunks):
            self.forward_calls += 1
            assert tuple(gene) == ("G1", "G1", "G2")
            return super().forward(chunks, gene, batch_index_chunks)

    model = CountingWindowedModel()
    outputs = predict_bags(
        model,
        [_bag(10, 3), _bag(6, 3)],
        ["G1", "G2"],
        seed=0,
    )

    assert model.forward_calls == 1
    assert [tuple(output.shape) for output in outputs] == [(10, 3), (6, 3)]


def test_evaluate_handles_bags_that_do_not_divide_the_window() -> None:
    model = _WindowedModel(3, window=8)
    batches = _batches()
    for batch in batches:
        batch["control"] = _bag(21, 3)
        batch["control_target"] = batch["control"]
        batch["observed"] = batch["control"] + 1.5
    report = evaluate_response_model(model, batches)
    assert np.isfinite(report["model_loss"])


def test_evaluate_caps_observed_bags_at_the_protocol_size() -> None:
    model = _ShiftModel(3)
    oversized = _batches(n=2)
    for batch in oversized:
        batch["observed"] = _bag(21, 3, offset=1.5)
    trimmed = [{**batch, "observed": batch["observed"][:16]} for batch in oversized]

    actual = evaluate_response_model(model, oversized, max_bag=16)
    expected = evaluate_response_model(model, trimmed, max_bag=16)

    assert actual["model_loss"] == pytest.approx(expected["model_loss"])
    assert actual["null_shuffle_loss"] == pytest.approx(expected["null_shuffle_loss"])


class _ForwardOnlyProxy(nn.Module):
    """Stand-in for ``DistributedDataParallel``: proxies ONLY ``forward``.

    DDP wraps the module and forwards ``__call__`` to the inner module's
    ``forward``; it does not expose the inner module's other methods. A
    trainer that reached for a custom method by name would raise
    ``AttributeError`` here, exactly as it would on a real multi-rank run.
    """

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        # DDP exposes the wrapped module as ``.module``; mirror that name so
        # attribute lookups (e.g. the ST window) resolve the same way.
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


def test_predict_bag_works_through_a_forward_only_wrapper() -> None:
    """Regression: DDP proxies only ``forward``.

    ``accelerator.prepare`` wraps the model in ``DistributedDataParallel``
    for any multi-rank launch. Calling ``predict_response_chunks`` on that
    wrapper raises ``AttributeError`` on the first training batch, so the
    whole point of the DDP support would fail at run time while every
    single-process test still passed.
    """
    inner = _WindowedModel(dim=3, window=4)
    wrapped = _ForwardOnlyProxy(inner)
    assert not hasattr(wrapped, "predict_response_chunks")
    control = _bag(6, 3)
    from aivc_model.response_training import predict_bag

    out = predict_bag(wrapped, control, "G", seed=0)
    assert out.shape == (6, 3)


def test_forward_only_state_model_exposes_forward() -> None:
    """The real module must route training through ``forward``, not a method."""
    from aivc_model.tx1_predicted_response import ForwardOnlyStateModel

    assert "forward" in vars(ForwardOnlyStateModel)


class _RealShapedAdapter(nn.Module):
    """Mirrors StateForwardAdapter: holds state_model, declares NO window itself."""

    def __init__(self, state_model: nn.Module) -> None:
        super().__init__()
        self.state_model = state_model


class _RealShapedModel(nn.Module):
    """Mirrors ForwardOnlyStateModel: the window lives at
    ``state_adapter.state_model.cell_sentence_len``, two levels down.

    Every earlier double put ``cell_sentence_len`` directly on the model,
    which is why they all passed while the real model failed.
    """

    def __init__(self, dim: int, window: int) -> None:
        super().__init__()
        self.shift = nn.Parameter(torch.zeros(dim))
        inner = nn.Module()
        inner.cell_sentence_len = window
        self.state_adapter = _RealShapedAdapter(inner)

    def forward(self, chunks, gene, batch_index_chunks):
        window = self.state_adapter.state_model.cell_sentence_len
        for chunk in chunks:
            if chunk.shape[0] != window:
                raise ValueError(
                    "STATE chunks must all equal the configured cell_sentence_len"
                )
        return tuple(chunk + self.shift for chunk in chunks)


def test_predict_bag_reads_the_window_from_state_model() -> None:
    """Regression: the window lives on state_model, not on the adapter.

    StateForwardAdapter has only ``state_model``; it declares no
    ``cell_sentence_len``. Looking for one on the adapter resolved to None on
    every real model, so predict_bag fell through to its single-chunk path
    and handed a whole bag to a model expecting fixed windows. That is the
    ValueError the first multi-rank run died on.
    """
    model = _RealShapedModel(dim=3, window=4)
    from aivc_model.response_training import predict_bag

    out = predict_bag(model, _bag(10, 3), "G", seed=0)
    assert out.shape == (10, 3)


def test_predict_bag_through_ddp_wrapper_with_real_shape() -> None:
    """The same, behind a forward-only wrapper -- the actual failing case."""
    wrapped = _ForwardOnlyProxy(_RealShapedModel(dim=3, window=4))
    from aivc_model.response_training import predict_bag

    out = predict_bag(wrapped, _bag(10, 3), "G", seed=0)
    assert out.shape == (10, 3)


def test_predict_bag_refuses_a_windowless_adapter() -> None:
    """A real adapter with no resolvable window must raise, not fall back.

    The silent fallback is what turned a missing attribute into a confusing
    chunk-size error thrown from deep inside forward_chunks.
    """
    model = _RealShapedModel(dim=3, window=4)
    del model.state_adapter.state_model.cell_sentence_len
    from aivc_model.response_training import predict_bag

    with pytest.raises(ValueError, match="cell_sentence_len"):
        predict_bag(model, _bag(10, 3), "G", seed=0)
