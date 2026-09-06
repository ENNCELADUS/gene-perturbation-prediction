"""Pure response losses, fixed holdout, padded forwarding and RNG boundaries."""

import pytest
import torch
from torch import nn
from src.experiments.prepare import split_heldout_genes
from src.model.response import energy_distance, mean_delta_mse, predict_bags
from src.model.response import response_terms
from src.data.batches import ResponseBatch


def test_batched_predictions_and_losses_match_independent_conditions():
    model = _WindowedModel(3, window=8).eval()
    controls = [_bag(10, 3), _bag(6, 3)]
    batched = predict_bags(model, controls, ["G1", "G2"], seed=0)
    independent = [
        predict_bags(model, [bag], [gene], seed=0)[0]
        for bag, gene in zip(controls, ["G1", "G2"], strict=True)
    ]
    for actual, expected, control in zip(batched, independent, controls, strict=True):
        torch.testing.assert_close(actual, expected)
        observed = control + 1.5
        torch.testing.assert_close(
            energy_distance(actual, observed), energy_distance(expected, observed)
        )
        torch.testing.assert_close(
            mean_delta_mse(actual, observed, control.mean(0)),
            mean_delta_mse(expected, observed, control.mean(0)),
        )


def _bag(rows: int, dim: int, offset: float = 0.0, scale: float = 1.0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(rows * 100 + dim)
    return torch.randn(rows, dim, generator=generator) * scale + offset


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(),
                reason="CUDA unavailable",
            ),
        ),
    ],
)
def test_response_terms_batch_equal_shapes_preserving_order_and_gradients(
    monkeypatch,
    device,
):
    shapes = [(32, 29, 36), (30, 31, 35), (32, 29, 36), (30, 31, 35)]
    predicted = tuple(
        _bag(n, 7, offset=i / 10).to(device).requires_grad_()
        for i, (n, _, _) in enumerate(shapes)
    )
    reference = tuple(x.detach().clone().requires_grad_() for x in predicted)
    observed = tuple(_bag(n, 7, offset=1).to(device) for _, n, _ in shapes)
    controls = tuple(_bag(n, 7).to(device) for _, _, n in shapes)
    batch = ResponseBatch(
        ("L",) * 4, ("A", "B", "C", "D"), controls, observed, controls
    )
    expected = {
        "mean_delta_mse": torch.stack(
            [
                mean_delta_mse(p, o, c.mean(0))
                for p, o, c in zip(reference, observed, controls, strict=True)
            ]
        ),
        "energy_distance": torch.stack(
            [energy_distance(p, o) for p, o in zip(reference, observed, strict=True)]
        ),
    }
    original = torch.cdist
    calls = []

    def counted(*args, **kwargs):
        calls.append(args[0].shape)
        return original(*args, **kwargs)

    monkeypatch.setattr(torch, "cdist", counted)
    actual = response_terms(predicted, batch)
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name])
    sum(value.sum() for value in actual.values()).backward()
    sum(value.sum() for value in expected.values()).backward()
    for p, ref in zip(predicted, reference, strict=True):
        torch.testing.assert_close(p.grad, ref.grad, atol=1e-6, rtol=1e-5)
    assert len(calls) == 6  # three distances per shape group, not per condition


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


class _RandpermWindowedModel(nn.Module):
    """Mirrors STATE collator randomness on each chunk's own device."""

    def __init__(self, window: int) -> None:
        super().__init__()
        self.cell_sentence_len = window

    def forward(self, chunks, gene, batch_index_chunks):
        return tuple(
            chunk[
                torch.randperm(
                    chunk.shape[0],
                    device=chunk.device,
                )
            ]
            for chunk in chunks
        )


def test_predict_bag_pads_and_trims_to_the_window() -> None:
    """A bag that is not a multiple of the window must still forward.

    Training only worked before because max_bag happened to equal the window;
    evaluation passed whole control bags and crashed on exactly this.
    """
    from src.model.response import predict_bag

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


def test_complete_state_windows_do_not_copy_control_cells():
    """The 128-cell production path must not launch an index copy per window."""
    controls = (_bag(128, 3), _bag(128, 3))

    class BorrowedWindows(_WindowedModel):
        def forward(self, chunks, gene, batch_index_chunks):
            for index, chunk in enumerate(chunks):
                source = controls[index // 2]
                assert chunk.untyped_storage().data_ptr() == (
                    source.untyped_storage().data_ptr()
                )
            return super().forward(chunks, gene, batch_index_chunks)

    predict_bags(BorrowedWindows(3, window=64), controls, ("G1", "G2"), seed=0)


@pytest.mark.parametrize("rows", [6, 8, 10, 16])
def test_state_window_values_and_gradients_match_indexed_reference(rows):
    from src.model.response import _chunk_control_cell_indices

    control = _bag(rows, 3).requires_grad_()
    reference = control.detach().clone().requires_grad_()
    model = _WindowedModel(3, window=8)
    expected_chunks = tuple(
        reference[torch.as_tensor(index)]
        for index in _chunk_control_cell_indices(rows, 8, 73)
    )
    expected = torch.cat(model(expected_chunks, "G", (None,) * len(expected_chunks)))[
        :rows
    ]
    actual = predict_bags(model, (control,), ("G",), seed=73)[0]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(control.grad, reference.grad, rtol=0, atol=0)


def test_predict_bags_seed_controls_randperm_despite_ambient_rng_changes() -> None:
    model = _RandpermWindowedModel(window=8).eval()
    control = torch.arange(30, dtype=torch.float32).reshape(10, 3)

    torch.manual_seed(11)
    torch.rand(5)
    first = predict_bags(model, [control], ["G"], seed=73)[0]
    torch.manual_seed(999)
    torch.rand(17)
    second = predict_bags(model, [control], ["G"], seed=73)[0]

    torch.testing.assert_close(first, second)


def test_predict_bags_different_seeds_can_change_padding_and_randperm() -> None:
    model = _RandpermWindowedModel(window=8).eval()
    control = torch.arange(30, dtype=torch.float32).reshape(10, 3)

    first = predict_bags(model, [control], ["G"], seed=73)[0]
    second = predict_bags(model, [control], ["G"], seed=74)[0]

    assert not torch.equal(first, second)


def test_predict_bags_preserves_caller_cpu_rng_state() -> None:
    model = _RandpermWindowedModel(window=8).eval()
    control = torch.arange(30, dtype=torch.float32).reshape(10, 3)
    torch.manual_seed(241)
    state_before = torch.random.get_rng_state().clone()

    predict_bags(model, [control], ["G"], seed=73)

    assert torch.equal(torch.random.get_rng_state(), state_before)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_predict_bags_seeds_and_restores_cuda_rng() -> None:
    device = torch.device("cuda", 0)
    model = _RandpermWindowedModel(window=8).eval().to(device)
    control = torch.arange(30, dtype=torch.float32, device=device).reshape(10, 3)
    torch.cuda.manual_seed(241)
    state_before = torch.cuda.get_rng_state(device).clone()

    first = predict_bags(model, [control], ["G"], seed=73)[0]
    state_after = torch.cuda.get_rng_state(device)
    torch.cuda.manual_seed(999)
    torch.rand(17, device=device)
    second = predict_bags(model, [control], ["G"], seed=73)[0]

    torch.testing.assert_close(first, second)
    assert torch.equal(state_after, state_before)


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
    from src.model.response import predict_bag

    out = predict_bag(wrapped, control, "G", seed=0)
    assert out.shape == (6, 3)


def test_forward_only_state_model_exposes_forward() -> None:
    """The real module must route training through ``forward``, not a method."""
    from src.model.state import ForwardOnlyStateModel

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
    from src.model.response import predict_bag

    out = predict_bag(model, _bag(10, 3), "G", seed=0)
    assert out.shape == (10, 3)


def test_predict_bag_through_ddp_wrapper_with_real_shape() -> None:
    """The same, behind a forward-only wrapper -- the actual failing case."""
    wrapped = _ForwardOnlyProxy(_RealShapedModel(dim=3, window=4))
    from src.model.response import predict_bag

    out = predict_bag(wrapped, _bag(10, 3), "G", seed=0)
    assert out.shape == (10, 3)


def test_predict_bag_refuses_a_windowless_adapter() -> None:
    """A real adapter with no resolvable window must raise, not fall back.

    The silent fallback is what turned a missing attribute into a confusing
    chunk-size error thrown from deep inside forward_chunks.
    """
    model = _RealShapedModel(dim=3, window=4)
    del model.state_adapter.state_model.cell_sentence_len
    from src.model.response import predict_bag

    with pytest.raises(ValueError, match="cell_sentence_len"):
        predict_bag(model, _bag(10, 3), "G", seed=0)
