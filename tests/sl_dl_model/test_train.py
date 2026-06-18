"""Tests for SlDlModel and StateDlProducer (Task 2.3).

All tests use state_backend="linear_mock" — no checkpoint required.
PYTORCH_ENABLE_MPS_FALLBACK is set at module import time so that the energy-
distance backward (aten::_cdist_backward) falls back to CPU on MPS-only Macs.
"""

from __future__ import annotations

import os

# Force MPS ops that lack a native kernel to fall back to CPU.
# Must be set before importing torch (or at least before any MPS dispatch).
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch

from sl_dl_model.config import SLDLConfig
from sl_dl_model.model import SlDlModel
from sl_dl_model.train import StateDlProducer  # noqa: F401 — used in _make_producer type hint


def _model(esm_dim: int = 8, input_dim: int = 6) -> SlDlModel:
    return SlDlModel(
        backend="linear_mock",
        checkpoint=None,
        esm_dim=esm_dim,
        adapter_hidden=16,
        pert_dim=5,
        input_dim=input_dim,
        output_dim=input_dim,
        pooling="mean_std",
        pair_hidden=(16,),
        include_coverage_flag=False,
    )


def test_embed_gene_shape() -> None:
    """embed_gene returns (2*output_dim,) via mean_std pooling."""
    model = _model().eval()
    e_g = model.embed_gene(torch.randn(8), torch.randn(10, 6))
    assert e_g.shape == (12,), f"expected (12,), got {e_g.shape}"


def test_score_pairs_shape_and_backprop() -> None:
    """score_pairs returns (B,) logits and gradients flow back through e_a."""
    model = _model()
    e_a = torch.randn(4, 12, requires_grad=True)
    e_b = torch.randn(4, 12)
    ge = torch.randn(4, 5)
    logits = model.score_pairs(e_a, e_b, ge)
    assert logits.shape == (4,), f"expected (4,), got {logits.shape}"
    logits.sum().backward()
    assert e_a.grad is not None, "gradient did not flow back through e_a"


def test_producer_emits_universe_table(tmp_path) -> None:
    """StateDlProducer.produce returns (n_gene, emb_dim) embeddings + mask."""
    from sl_dl_model.bags import GwpsBags
    from sl_dl_model.gene_embeddings import Esm2EmbeddingTable
    from sl_dl_model.train import StateDlProducer

    rng = np.random.default_rng(0)
    symbols = np.array(["A", "B", "C", "D"], dtype=object)
    esm = Esm2EmbeddingTable(
        dim=8,
        vectors_by_symbol={
            s: rng.standard_normal(8).astype("float32") for s in ["A", "B", "C", "D"]
        },
    )
    bags = GwpsBags(
        control_template=rng.standard_normal((8, 6)).astype("float32"),
        bags_by_symbol={
            "A": rng.standard_normal((8, 6)).astype("float32"),
            "B": rng.standard_normal((8, 6)).astype("float32"),
        },
        input_dim=6,
    )
    cfg = SLDLConfig(
        esm2_model="x",
        max_epochs=1,
        warmup_epochs=1,
        pert_dim=5,
        adapter_hidden=16,
        pair_hidden=(16,),
        include_coverage_flag=False,
        state_backend="linear_mock",
    )
    # 5-tuples: (gene_a, gene_b, label, ea, eb)
    pairs: list[tuple[str, str, int, float, float]] = [
        ("A", "B", 1, -1.0, -0.5),
        ("C", "D", 0, 0.1, 0.2),
        ("A", "C", 0, -1.0, 0.1),
    ]
    producer = StateDlProducer(
        cfg,
        esm=esm,
        bags=bags,
        train_pairs=pairs,
        input_dim=6,
        output_dim=6,
    )
    emb, mask = producer.produce(symbols, {"A", "B", "C", "D"})
    assert emb.shape[0] == 4, f"expected 4 rows, got {emb.shape[0]}"
    assert mask.shape == (4,), f"expected mask shape (4,), got {mask.shape}"
    # mask reflects bag coverage (A,B have bags); C,D have ESM2 but no bags → 0.
    assert mask.sum() == 2, f"expected 2 bag-covered genes, got mask={mask}"
    assert mask[0] == 1 and mask[1] == 1, "A and B should be bag-covered"
    assert mask[2] == 0 and mask[3] == 0, "C and D have no bags → not covered"


def test_distill_required_but_missing_vocab_raises() -> None:
    """FIX 2: real backend + lambda_distill>0 + missing vocab must fail loudly."""
    from pathlib import Path

    producer = _make_producer(
        esm_symbols=["A", "B"], bag_symbols=["A"], lambda_distill=0.5
    )
    # Switch to a real backend with a checkpoint whose sibling vocab does not exist.
    producer.config = SLDLConfig(
        esm2_model="x",
        state_backend="state_checkpoint",
        state_checkpoint=Path("/nonexistent/checkpoints/final.ckpt"),
        max_epochs=1,
        warmup_epochs=0,
        pert_dim=5,
        adapter_hidden=16,
        pair_hidden=(16,),
        include_coverage_flag=False,
        lambda_distill=0.5,
        lambda_distill_after_warmup=0.5,
    )
    producer._pert_vocab_loaded = False
    try:
        producer._ensure_pert_vocab()
        raise AssertionError("expected RuntimeError for missing required distill vocab")
    except RuntimeError as exc:
        assert "distill" in str(exc).lower()


def test_distill_not_required_when_weight_zero_does_not_raise() -> None:
    """FIX 2: lambda_distill=0 with missing vocab is fine (distill not requested)."""
    from pathlib import Path

    producer = _make_producer(esm_symbols=["A", "B"], lambda_distill=0.0)
    producer.config = SLDLConfig(
        esm2_model="x",
        state_backend="state_checkpoint",
        state_checkpoint=Path("/nonexistent/checkpoints/final.ckpt"),
        max_epochs=1,
        warmup_epochs=0,
        pert_dim=5,
        adapter_hidden=16,
        pair_hidden=(16,),
        include_coverage_flag=False,
        lambda_distill=0.0,
        lambda_distill_after_warmup=0.0,
    )
    producer._pert_vocab_loaded = False
    producer._ensure_pert_vocab()  # must not raise
    assert producer._pert_vocab is None


def test_global_mean_fallback_used_for_missing_esm() -> None:
    """FIX 4: with fallback_strategy=global_mean, a missing-ESM gene is embedded
    (not skipped) and counts via a nonzero embedding rather than silent zero."""
    producer = _make_producer(esm_symbols=["A", "B"], bag_symbols=["A", "B"])
    # C has a bag but no ESM2 vector; with global_mean fallback it still embeds.
    producer.bags.bags_by_symbol["C"] = np.zeros((8, 6), dtype="float32")
    producer.config = SLDLConfig(
        esm2_model="x",
        state_backend="linear_mock",
        max_epochs=1,
        warmup_epochs=0,
        pert_dim=5,
        adapter_hidden=16,
        pair_hidden=(16,),
        include_coverage_flag=False,
        fallback_strategy="global_mean",
    )
    symbols = np.array(["A", "B", "C"], dtype=object)
    emb, mask = producer.produce(symbols, {"A", "B"})
    # C is bag-covered → mask=1; embedding is nonzero because global_mean ESM2
    # fallback gives it a real ESM input rather than being skipped.
    assert mask[2] == 1
    assert np.linalg.norm(emb[2]) > 0, "global_mean fallback should embed gene C"


def test_zero_fallback_leaves_missing_esm_zero() -> None:
    """FIX 4: with fallback_strategy=zero, a missing-ESM gene embeds from a zero
    ESM input (deterministic) and is not silently dropped from the universe."""
    producer = _make_producer(esm_symbols=["A", "B"], bag_symbols=["A"])
    producer.config = SLDLConfig(
        esm2_model="x",
        state_backend="linear_mock",
        max_epochs=1,
        warmup_epochs=0,
        pert_dim=5,
        adapter_hidden=16,
        pair_hidden=(16,),
        include_coverage_flag=False,
        fallback_strategy="zero",
    )
    symbols = np.array(["A", "B", "Z"], dtype=object)  # Z missing from ESM + bags
    emb, mask = producer.produce(symbols, {"A", "B"})
    assert emb.shape[0] == 3
    assert mask[2] == 0  # Z not bag-covered


def test_score_matrix_applies_geneeffect_standardizer() -> None:
    """FIX 3: the DL score_matrix standardizes GeneEffect features (train-fold fit).

    The producer must expose a fitted standardizer after produce(); raw vs
    standardized features should differ when GeneEffect values are not already
    zero-mean/unit-variance.
    """
    producer = _make_producer(
        esm_symbols=["A", "B"],
        bag_symbols=["A"],
        pairs=[("A", "B", 1, 10.0, 12.0), ("A", "B", 0, 11.0, 9.0)],
    )
    symbols = np.array(["A", "B"], dtype=object)
    producer.produce(symbols, {"A", "B"})
    assert producer._ge_standardizer is not None, (
        "producer should fit a GeneEffect standardizer during produce()"
    )
    # The fitted mean should reflect the train pairs' GeneEffect scale (~9-12).
    assert float(np.abs(producer._ge_standardizer.mean_).max()) > 1.0


def _make_producer(
    include_coverage_flag: bool = False,
    esm_symbols: list[str] | None = None,
    bag_symbols: list[str] | None = None,
    pairs: list[tuple[str, str, int, float, float]] | None = None,
    max_epochs: int = 1,
    warmup_epochs: int = 0,
    lambda_distill: float = 0.0,
    lambda_bag: float = 0.0,
) -> "StateDlProducer":
    """Shared factory for StateDlProducer in tests."""
    from sl_dl_model.bags import GwpsBags
    from sl_dl_model.gene_embeddings import Esm2EmbeddingTable
    from sl_dl_model.train import StateDlProducer

    rng = np.random.default_rng(1)
    if esm_symbols is None:
        esm_symbols = ["A", "B"]
    if bag_symbols is None:
        bag_symbols = ["A"]
    if pairs is None:
        pairs = [("A", "B", 1, -1.0, -0.5)]

    esm = Esm2EmbeddingTable(
        dim=8,
        vectors_by_symbol={
            s: rng.standard_normal(8).astype("float32") for s in esm_symbols
        },
    )
    bags = GwpsBags(
        control_template=rng.standard_normal((8, 6)).astype("float32"),
        bags_by_symbol={
            s: rng.standard_normal((8, 6)).astype("float32") for s in bag_symbols
        },
        input_dim=6,
    )
    cfg = SLDLConfig(
        esm2_model="x",
        max_epochs=max_epochs,
        warmup_epochs=warmup_epochs,
        pert_dim=5,
        adapter_hidden=16,
        pair_hidden=(16,),
        include_coverage_flag=include_coverage_flag,
        state_backend="linear_mock",
        lambda_distill=lambda_distill,
        lambda_distill_after_warmup=lambda_distill,
        lambda_bag=lambda_bag,
    )
    return StateDlProducer(
        cfg, esm=esm, bags=bags, train_pairs=pairs, input_dim=6, output_dim=6
    )


def test_coverage_flag_produce_and_score_matrix() -> None:
    """With include_coverage_flag=True, produce + score_matrix run without crash."""
    from sl_dl_model.bags import GwpsBags
    from sl_dl_model.gene_embeddings import Esm2EmbeddingTable
    from sl_dl_model.train import StateDlProducer

    rng = np.random.default_rng(0)
    symbols = np.array(["A", "B"], dtype=object)
    esm = Esm2EmbeddingTable(
        dim=8,
        vectors_by_symbol={
            "A": rng.standard_normal(8).astype("float32"),
            "B": rng.standard_normal(8).astype("float32"),
        },
    )
    bags = GwpsBags(
        control_template=rng.standard_normal((8, 6)).astype("float32"),
        bags_by_symbol={
            "A": rng.standard_normal((8, 6)).astype("float32"),
        },
        input_dim=6,
    )
    cfg = SLDLConfig(
        esm2_model="x",
        max_epochs=1,
        warmup_epochs=0,
        pert_dim=5,
        adapter_hidden=16,
        pair_hidden=(16,),
        include_coverage_flag=True,  # Enable coverage flags
        state_backend="linear_mock",
    )
    pairs: list[tuple[str, str, int, float, float]] = [("A", "B", 1, -1.0, -0.5)]
    producer = StateDlProducer(
        cfg, esm=esm, bags=bags, train_pairs=pairs, input_dim=6, output_dim=6
    )
    emb, mask = producer.produce(symbols, {"A", "B"})
    assert emb.shape[0] == 2
    # Now score_matrix should not crash
    ge = np.array([-1.0, -0.5])
    scores = producer.score_matrix(symbols, ge)
    assert scores.shape == (2, 2), f"expected (2,2), got {scores.shape}"
    assert scores[0, 0] == 0.0 and scores[1, 1] == 0.0, "diagonal should be zero"


def test_coverage_mask_reflects_bag_membership() -> None:
    """FIX 3: coverage_mask=1 iff gene in bags_by_symbol, not just ESM2 coverage."""
    producer = _make_producer(esm_symbols=["A", "B"], bag_symbols=["A"])
    symbols = np.array(["A", "B"], dtype=object)
    emb, mask = producer.produce(symbols, {"A", "B"})
    # A has bag → mask=1; B has ESM2 but no bag → mask=0
    assert mask[0] == 1, "A has bag, should be covered"
    assert mask[1] == 0, "B has ESM2 but no bag, should NOT be covered"
    # Both should have nonzero embeddings (ESM2 coverage)
    assert np.linalg.norm(emb[0]) > 0, "A should have nonzero embedding"
    assert np.linalg.norm(emb[1]) > 0, "B should have nonzero embedding (ESM2 fallback)"


def test_distill_loss_wired_with_monkeypatch(monkeypatch) -> None:
    """FIX 2: distill flows grad via the encoded-token path (adapter -> pert_encoder).

    Discriminating setup: the checkpoint's pert_encoder maps pert_dim(5) ->
    hidden(3), so hidden != pert_dim. The encoded-token contract compares
    ``pert_encoder(adapter_raw)`` (width 3) against ``pert_encoder(onehot)``
    (width 3). A raw-vs-encoded comparison would instead pit the 5-wide adapter
    output against the 3-wide target and fail the MSE on shape.
    """
    from torch import nn

    # Monkeypatch _load_pert_vocab to return a small fake vocab (width = pert_dim).
    def _mock_load_pert_vocab(path):
        return {"A": np.eye(5, dtype=np.float32)[0]}  # one-hot for "A", width 5

    from sl_dl_model import train as train_mod

    monkeypatch.setattr(train_mod, "_load_pert_vocab", _mock_load_pert_vocab)

    # Build producer with distill enabled
    producer = _make_producer(
        esm_symbols=["A", "B"],
        bag_symbols=["A"],
        pairs=[("A", "B", 1, -1.0, -0.5)],
        max_epochs=1,
        warmup_epochs=0,
        lambda_distill=0.5,
    )

    # Materialize the model and attach a fake pert_encoder mapping pert_dim->hidden.
    model = producer._build_model()
    producer._model = model
    model.encoder.state.state_model.pert_encoder = nn.Linear(5, 3, bias=False)

    # Force pert_vocab load via our monkeypatched function
    # (use a dummy checkpoint path — it won't be read because we monkeypatched)
    from pathlib import Path

    producer.config = SLDLConfig(
        esm2_model="x",
        state_backend="state_checkpoint",  # not linear_mock so vocab loads
        state_checkpoint=Path("/fake/path/checkpoints/final.ckpt"),
        max_epochs=1,
        warmup_epochs=0,
        pert_dim=5,
        adapter_hidden=16,
        pair_hidden=(16,),
        include_coverage_flag=False,
        lambda_distill=0.5,
        lambda_distill_after_warmup=0.5,
    )
    producer._pert_vocab_loaded = False  # reset so _ensure_pert_vocab runs again

    result = producer._distill_part({"A"})
    assert result is not None, "_distill_part should return a tensor for in-vocab genes"
    assert torch.isfinite(result).all(), "distill loss should be finite"

    # Verify gradients flow back to adapter params through the frozen pert_encoder.
    result.backward()
    adapter_params = list(producer._model.encoder.adapter.parameters())
    assert any(p.grad is not None for p in adapter_params), (
        "adapter should receive gradient"
    )


def test_distill_part_returns_none_when_no_vocab() -> None:
    """FIX 2: _distill_part returns None when pert_vocab is None."""
    producer = _make_producer(lambda_distill=0.5)
    symbols = np.array(["A", "B"], dtype=object)
    producer.produce(symbols, {"A", "B"})
    # No pert_vocab loaded (linear_mock) → should return None
    result = producer._distill_part({"A"})
    assert result is None, "distill_part should return None when pert_vocab is None"


def test_skipped_pairs_warning(caplog) -> None:
    """FIX 4a: log warning when pairs are skipped due to missing ESM2."""
    import logging

    caplog.set_level(logging.WARNING)
    # "A" has ESM2, "B" does not → pair (A,B) skipped; pair (A,A) is trainable.
    producer = _make_producer(
        esm_symbols=["A"],
        bag_symbols=["A"],
        pairs=[("A", "B", 1, -1.0, -0.5), ("A", "A", 0, -1.0, -1.0)],
    )
    symbols = np.array(["A", "B"], dtype=object)
    producer.produce(symbols, {"A", "B"})
    # B has no ESM2 vector → pair (A,B) skipped
    assert any("skipped" in rec.message.lower() for rec in caplog.records), (
        "should warn about skipped pairs"
    )


def test_all_pairs_skipped_raises_error() -> None:
    """FIX 4b: raise RuntimeError when ALL pairs are skipped (empty ESM2 table)."""
    producer = _make_producer(esm_symbols=[], pairs=[("A", "B", 1, -1.0, -0.5)])
    symbols = np.array(["A", "B"], dtype=object)
    try:
        producer.produce(symbols, {"A", "B"})
        assert False, "should have raised RuntimeError for all skipped pairs"
    except RuntimeError as e:
        assert "no trainable pairs" in str(e).lower(), f"unexpected error: {e}"


def test_produce_uses_partialstate_not_accelerator(tmp_path, monkeypatch) -> None:
    """produce() must not instantiate Accelerator (DDP wrap removed)."""
    import sl_dl_model.train as train_mod

    # Fail loudly if any code path constructs an Accelerator.
    def _boom(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("Accelerator must not be constructed in produce()")

    monkeypatch.setattr(train_mod, "Accelerator", _boom, raising=False)

    rng = np.random.default_rng(1)
    symbols = np.array(["A", "B", "C", "D"], dtype=object)
    esm = train_mod.Esm2EmbeddingTable(
        dim=8,
        vectors_by_symbol={
            s: rng.standard_normal(8).astype("float32") for s in ["A", "B", "C", "D"]
        },
    )
    bags = train_mod.GwpsBags(
        control_template=rng.standard_normal((8, 6)).astype("float32"),
        bags_by_symbol={"A": rng.standard_normal((8, 6)).astype("float32")},
        input_dim=6,
    )
    cfg = SLDLConfig(
        esm2_model="x",
        max_epochs=1,
        warmup_epochs=1,
        pert_dim=5,
        adapter_hidden=16,
        pair_hidden=(16,),
        include_coverage_flag=False,
        state_backend="linear_mock",
    )
    producer = train_mod.StateDlProducer(
        cfg,
        esm=esm,
        bags=bags,
        train_pairs=[("A", "B", 1, -1.0, -0.5), ("C", "D", 0, 0.1, 0.2)],
        input_dim=6,
        output_dim=6,
    )
    emb, mask = producer.produce(symbols, {"A", "B", "C", "D"})
    assert emb.shape == (4, producer._model.emb_dim)
    assert mask.shape == (4,)


def test_score_matrix_caches_embeddings() -> None:
    """FIX 5: score_matrix reuses embeddings from produce, giving same results."""
    producer = _make_producer()
    symbols = np.array(["A", "B"], dtype=object)
    ge = np.array([-1.0, -0.5])
    # First call to score_matrix triggers produce internally
    scores1 = producer.score_matrix(symbols, ge)
    # Second call should reuse cached embeddings
    scores2 = producer.score_matrix(symbols, ge)
    assert np.allclose(scores1, scores2), "cached embeddings should give same results"
    assert scores1.shape == (2, 2)
    assert scores1[0, 0] == 0.0 and scores1[1, 1] == 0.0, "diagonal zeroed"
