"""Tests for the rebuilt T2 GeneEffect readout head (tx1_geneeffect_head.py).

Covers: moment_pool correctness/invariance/masking, Tx1GeneEffectHead
permutation invariance and gradient flow, the correlation/variance/combined
losses, and the anti-collapse property that motivates replacing MSE.
"""

import torch
from torch import nn
import torch.nn.functional as F

from aivc_model.tx1_geneeffect_head import (
    FUSION_INTERACTION_RESIDUAL,
    Tx1GeneEffectHead,
    correlation_loss,
    moment_pool,
    rank_variance_loss,
    variance_matching_loss,
)

# ---------------------------------------------------------------------------
# moment_pool
# ---------------------------------------------------------------------------


def test_moment_pool_matches_manual_mean_var() -> None:
    torch.manual_seed(0)
    bag = torch.randn(20, 5)
    pooled = moment_pool(bag, moments=2)
    expected_mean = bag.mean(dim=0)
    expected_var = bag.var(dim=0, unbiased=False)
    assert pooled.shape == (10,)
    assert torch.allclose(pooled[:5], expected_mean, atol=1e-6)
    assert torch.allclose(pooled[5:], expected_var, atol=1e-6)


def test_moment_pool_moments_one_is_mean_only() -> None:
    bag = torch.randn(10, 3)
    pooled = moment_pool(bag, moments=1)
    assert pooled.shape == (3,)
    assert torch.allclose(pooled, bag.mean(dim=0), atol=1e-6)


def test_moment_pool_moments_three_appends_std() -> None:
    bag = torch.randn(15, 4)
    pooled = moment_pool(bag, moments=3)
    assert pooled.shape == (12,)
    expected_std = bag.var(dim=0, unbiased=False).sqrt()
    assert torch.allclose(pooled[8:], expected_std, atol=1e-3)


def test_moment_pool_moments_four_appends_skew_and_is_finite() -> None:
    bag = torch.randn(25, 2)
    pooled = moment_pool(bag, moments=4)
    assert pooled.shape == (8,)
    assert torch.isfinite(pooled).all()


def test_moment_pool_invalid_moments_raises() -> None:
    bag = torch.randn(5, 2)
    try:
        moment_pool(bag, moments=5)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for moments=5")


def test_moment_pool_permutation_invariant() -> None:
    torch.manual_seed(1)
    bag = torch.randn(12, 6)
    perm = torch.randperm(12)
    pooled = moment_pool(bag, moments=2)
    pooled_shuffled = moment_pool(bag[perm], moments=2)
    assert torch.allclose(pooled, pooled_shuffled, atol=1e-6)


def test_moment_pool_masking_ignores_padded_rows() -> None:
    torch.manual_seed(2)
    real = torch.randn(8, 4)
    padding = torch.full((3, 4), 999.0)
    bag = torch.cat([real, padding], dim=0)
    mask = torch.cat([torch.ones(8), torch.zeros(3)]).bool()
    pooled = moment_pool(bag, mask=mask, moments=2)
    expected = moment_pool(real, moments=2)
    assert torch.allclose(pooled, expected, atol=1e-6)


def test_moment_pool_empty_after_masking_raises() -> None:
    bag = torch.randn(4, 3)
    mask = torch.zeros(4).bool()
    try:
        moment_pool(bag, mask=mask, moments=2)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for an all-masked bag")


# ---------------------------------------------------------------------------
# Tx1GeneEffectHead
# ---------------------------------------------------------------------------


def _make_head(moments: int = 2) -> Tx1GeneEffectHead:
    torch.manual_seed(0)
    return Tx1GeneEffectHead(response_dim=16, basal_dim=12, hidden=32, moments=moments)


def _make_interaction_head() -> Tx1GeneEffectHead:
    torch.manual_seed(0)
    return Tx1GeneEffectHead(
        response_dim=16,
        basal_dim=12,
        hidden=32,
        moments=2,
        fusion=FUSION_INTERACTION_RESIDUAL,
        projection_dim=8,
    )


def test_concat_head_preserves_historical_initialization_and_output() -> None:
    torch.manual_seed(4)
    head = Tx1GeneEffectHead(response_dim=3, basal_dim=2, hidden=7, moments=2)
    torch.manual_seed(4)
    historical = nn.Sequential(
        nn.Linear(10, 7),
        nn.LayerNorm(7),
        nn.GELU(),
        nn.Linear(7, 7),
        nn.LayerNorm(7),
        nn.GELU(),
        nn.Linear(7, 1),
    )
    pooled = torch.randn(5, 10)
    assert torch.equal(head.forward_pooled(pooled), historical(pooled).squeeze(-1))


def test_interaction_forward_matches_forward_pooled_and_is_permutation_invariant() -> (
    None
):
    head = _make_interaction_head()
    response = torch.randn(10, 16)
    basal = torch.randn(7, 12)
    pooled = torch.cat([moment_pool(response), moment_pool(basal)])
    expected = head.forward_pooled(pooled)
    assert torch.allclose(head(response, basal), expected)
    assert torch.allclose(head(response.flip(0), basal.flip(0)), expected, atol=1e-6)


def test_interaction_context_and_parameter_gradients() -> None:
    head = _make_interaction_head()
    response = torch.randn(10, 16)
    basal_a = torch.randn(7, 12)
    basal_b = basal_a + 2.0
    score_a = head(response, basal_a)
    score_b = head(response, basal_b)
    assert not torch.allclose(score_a, score_b)
    (score_a + score_b).backward()
    for parameter in head.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_interaction_candidate_has_fewer_parameters_than_concat_at_real_widths() -> (
    None
):
    for basal_dim in (2560, 2000):
        concat = Tx1GeneEffectHead(2000, basal_dim, hidden=256, moments=2)
        interaction = Tx1GeneEffectHead(
            2000,
            basal_dim,
            hidden=256,
            moments=2,
            fusion=FUSION_INTERACTION_RESIDUAL,
            projection_dim=128,
        )
        assert sum(p.numel() for p in interaction.parameters()) < sum(
            p.numel() for p in concat.parameters()
        )


def test_interaction_head_rejects_nonpositive_projection_dim() -> None:
    for projection_dim in (0, -1):
        try:
            Tx1GeneEffectHead(
                3,
                2,
                fusion=FUSION_INTERACTION_RESIDUAL,
                projection_dim=projection_dim,
            )
        except ValueError as error:
            assert "projection_dim" in str(error)
        else:
            raise AssertionError("expected nonpositive projection_dim rejection")


def test_head_forward_returns_finite_scalar() -> None:
    head = _make_head()
    response_bag = torch.randn(10, 16)
    basal_bag = torch.randn(7, 12)
    output = head(response_bag, basal_bag)
    assert output.shape == ()
    assert torch.isfinite(output).item()


def test_head_permutation_invariant_in_both_bags() -> None:
    head = _make_head()
    torch.manual_seed(3)
    response_bag = torch.randn(9, 16)
    basal_bag = torch.randn(11, 12)
    response_perm = torch.randperm(9)
    basal_perm = torch.randperm(11)
    with torch.no_grad():
        baseline = head(response_bag, basal_bag)
        shuffled = head(response_bag[response_perm], basal_bag[basal_perm])
    assert torch.allclose(baseline, shuffled, atol=1e-5)


def test_head_backward_populates_grads_on_parameters() -> None:
    head = _make_head()
    response_bag = torch.randn(6, 16)
    basal_bag = torch.randn(5, 12)
    output = head(response_bag, basal_bag)
    output.backward()
    grads_present = [p.grad is not None for p in head.parameters()]
    assert all(grads_present)
    for p in head.parameters():
        assert torch.isfinite(p.grad).all()


def test_head_no_line_id_input_generalizes_to_different_bag_sizes() -> None:
    # A structural check that the head has no line-ID embedding table: it
    # must accept arbitrary N/M bag sizes without a fixed vocabulary.
    head = _make_head()
    with torch.no_grad():
        out_a = head(torch.randn(4, 16), torch.randn(3, 12))
        out_b = head(torch.randn(50, 16), torch.randn(200, 12))
    assert torch.isfinite(out_a).item()
    assert torch.isfinite(out_b).item()


def test_head_forward_batch_over_list_of_variable_n_bags() -> None:
    head = _make_head()
    torch.manual_seed(4)
    response_bags = [torch.randn(n, 16) for n in (5, 8, 3, 12, 6, 9, 4, 7)]
    basal_bags = [torch.randn(m, 12) for m in (4, 4, 10, 6, 6, 6, 3, 5)]
    predictions = head.forward_batch(response_bags, basal_bags)
    assert predictions.shape == (8,)
    assert torch.isfinite(predictions).all()


def test_head_forward_batch_padded_matches_list_form() -> None:
    head = _make_head()
    torch.manual_seed(5)
    response_bags = [torch.randn(n, 16) for n in (5, 3, 7)]
    basal_bags = [torch.randn(m, 12) for m in (4, 6, 2)]

    n_max = max(b.shape[0] for b in response_bags)
    m_max = max(b.shape[0] for b in basal_bags)
    padded_response = torch.zeros(3, n_max, 16)
    response_mask = torch.zeros(3, n_max).bool()
    for i, b in enumerate(response_bags):
        padded_response[i, : b.shape[0]] = b
        response_mask[i, : b.shape[0]] = True
    padded_basal = torch.zeros(3, m_max, 12)
    basal_mask = torch.zeros(3, m_max).bool()
    for i, b in enumerate(basal_bags):
        padded_basal[i, : b.shape[0]] = b
        basal_mask[i, : b.shape[0]] = True

    with torch.no_grad():
        from_list = head.forward_batch(response_bags, basal_bags)
        from_padded = head.forward_batch(
            padded_response,
            padded_basal,
            response_mask=response_mask,
            basal_mask=basal_mask,
        )
    assert torch.allclose(from_list, from_padded, atol=1e-5)


# ---------------------------------------------------------------------------
# correlation_loss
# ---------------------------------------------------------------------------


def test_correlation_loss_near_zero_for_positive_affine_relation() -> None:
    torch.manual_seed(6)
    target = torch.randn(64)
    pred = 3.0 * target + 0.7
    loss = correlation_loss(pred, target)
    assert loss.item() < 1e-4


def test_correlation_loss_high_for_independent_signals() -> None:
    torch.manual_seed(7)
    # Use two deterministic orthogonal-ish patterns rather than random
    # draws, so the test is not flaky.
    target = torch.arange(32, dtype=torch.float32)
    pred = torch.sin(torch.linspace(0, 40 * torch.pi, 32))
    loss = correlation_loss(pred, target)
    assert loss.item() > 0.8


def test_correlation_loss_gradient_flows() -> None:
    torch.manual_seed(8)
    target = torch.randn(20)
    pred = torch.randn(20, requires_grad=True)
    loss = correlation_loss(pred, target)
    loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
    assert pred.grad.abs().sum().item() > 0.0


def test_correlation_loss_constant_pred_is_finite_worst_case() -> None:
    target = torch.randn(10)
    pred = torch.full((10,), 5.0)
    loss = correlation_loss(pred, target)
    assert torch.isfinite(loss).item()
    assert loss.item() > 0.9


def test_collapsed_pred_gradients_are_finite_not_nan() -> None:
    # Regression for the anti-collapse contract: the std epsilon must sit
    # INSIDE the sqrt, or the backward pass produces NaN gradients on exactly
    # the constant (mean-collapse) prediction the loss exists to punish --
    # which would permanently poison an Adam-optimized training run.
    target = torch.randn(10)
    for loss_fn in (correlation_loss, rank_variance_loss):
        pred = torch.full((10,), 5.0, requires_grad=True)
        loss_fn(pred, target).backward()
        assert pred.grad is not None
        assert torch.isfinite(pred.grad).all(), (
            f"{loss_fn.__name__} produced a non-finite gradient on constant pred"
        )


# ---------------------------------------------------------------------------
# variance_matching_loss
# ---------------------------------------------------------------------------


def test_variance_matching_loss_near_zero_when_stds_match() -> None:
    torch.manual_seed(9)
    target = torch.randn(50) * 2.0 + 5.0
    pred = torch.randn(50) * 2.0 - 3.0
    loss = variance_matching_loss(pred, target)
    assert loss.item() < 0.5


def test_variance_matching_loss_large_for_collapsed_pred() -> None:
    torch.manual_seed(10)
    target = torch.randn(50) * 3.0
    pred = torch.full((50,), 1.23)
    loss = variance_matching_loss(pred, target)
    assert loss.item() > 5.0


# ---------------------------------------------------------------------------
# rank_variance_loss / anti-collapse
# ---------------------------------------------------------------------------


def test_rank_variance_loss_is_sum_of_parts() -> None:
    torch.manual_seed(11)
    pred = torch.randn(30)
    target = torch.randn(30)
    lam = 0.5
    combined = rank_variance_loss(pred, target, lam=lam)
    expected = correlation_loss(pred, target) + lam * variance_matching_loss(
        pred, target
    )
    assert torch.allclose(combined, expected, atol=1e-6)


def test_anti_collapse_mean_prediction_is_punished_far_more_than_mse() -> None:
    """The critical test: rank_variance_loss must punish the mean-collapse
    shortcut that MSE rewards.

    Construct a target with real spread. A constant prediction equal to the
    target mean is the exact MSE-optimal degenerate solution (it minimizes
    squared error over a location-only predictor) but has zero variance and
    ~zero rank correlation. An aligned (rank-correlated, variance-matched)
    prediction should score far better under rank_variance_loss even though
    it does *not* minimize MSE as tightly as the mean predictor might for a
    noisy signal.
    """
    torch.manual_seed(12)
    target = torch.randn(40) * 4.0 + 10.0

    mean_pred = torch.full_like(target, target.mean().item())
    aligned_pred = target.clone() + 0.01 * torch.randn_like(target)

    rank_var_mean = rank_variance_loss(mean_pred, target).item()
    rank_var_aligned = rank_variance_loss(aligned_pred, target).item()

    mse_mean = F.mse_loss(mean_pred, target).item()
    mse_aligned = F.mse_loss(aligned_pred, target).item()

    # rank_variance_loss strongly prefers the aligned predictor.
    assert rank_var_mean > 10.0 * rank_var_aligned
    assert rank_var_aligned < 0.1

    # Document the contrast: under MSE, the mean predictor is already very
    # competitive (its MSE equals target's own variance), while the aligned
    # predictor's MSE improvement is comparatively modest -- MSE alone does
    # not strongly separate the two solutions the way rank_variance_loss
    # does. The mean predictor's MSE is *not* far worse than the aligned
    # predictor's MSE, unlike the enormous separation in rank_variance_loss.
    assert mse_mean > 0.0
    assert mse_aligned < mse_mean
    mse_ratio = mse_mean / max(mse_aligned, 1e-8)
    rank_var_ratio = rank_var_mean / max(rank_var_aligned, 1e-8)
    assert rank_var_ratio > mse_ratio


def test_batched_forward_into_rank_variance_loss_trains_toward_correlation() -> None:
    """Optimizer steps on rank_variance_loss(head.forward_batch(...), y)
    should reduce the loss -- i.e. the batched forward composes with the
    loss into a trainable objective."""
    torch.manual_seed(13)
    head = Tx1GeneEffectHead(response_dim=8, basal_dim=6, hidden=24, moments=2)
    optimizer = torch.optim.Adam(head.parameters(), lr=5e-3)

    n_examples = 10
    response_bags = [torch.randn(6, 8) + i * 0.3 for i in range(n_examples)]
    basal_bags = [torch.randn(5, 6) for _ in range(n_examples)]
    target = torch.linspace(-2.0, 2.0, n_examples)

    losses = []
    for _ in range(60):
        optimizer.zero_grad()
        pred = head.forward_batch(response_bags, basal_bags)
        assert pred.shape == (n_examples,)
        loss = rank_variance_loss(pred, target, lam=1.0)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0]
