from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from sl_dl_model.bags import GwpsBags
from sl_dl_model.exp08b_artifacts import (
    embedding_cache_path,
    generator_manifest_path,
    generator_weights_path,
    load_embedding_cache,
    load_generator_manifest,
)
from sl_dl_model.exp08b_config import Exp08bConfig
from sl_dl_model.exp08b_generator import (
    EmaBagScale,
    FixedWarmupBagScale,
    Step1GeneratorTrainer,
    build_bag_scale,
    select_generator_bag_sets,
)
from sl_dl_model.gene_embeddings import Esm2EmbeddingTable
from sl_dl_model.pert_vocab import load_pert_vocab


def test_generator_validation_split_comes_only_from_train_covered() -> None:
    train_symbols = {"A", "B", "C", "D", "E", "TEST_ONLY"}
    covered_symbols = {"A", "B", "C", "D", "E", "OUTSIDE_TRAIN"}

    train_bag, val_bag = select_generator_bag_sets(
        train_symbols=train_symbols,
        covered_symbols=covered_symbols,
        val_fraction=0.4,
        seed=17,
    )

    assert train_bag | val_bag == {"A", "B", "C", "D", "E"}
    assert train_bag.isdisjoint(val_bag)
    assert "TEST_ONLY" not in train_bag | val_bag
    assert "OUTSIDE_TRAIN" not in train_bag | val_bag
    assert len(val_bag) == 1


def test_generator_validation_split_is_deterministic() -> None:
    kwargs = {
        "train_symbols": {"A", "B", "C", "D", "E", "F"},
        "covered_symbols": {"A", "B", "C", "D", "E", "F"},
        "val_fraction": 0.2,
        "seed": 23,
    }

    first = select_generator_bag_sets(**kwargs)
    second = select_generator_bag_sets(**kwargs)

    assert first == second


def test_fixed_warmup_bag_scale_uses_median_and_clamp() -> None:
    scale = FixedWarmupBagScale(min_scale=1e-3)
    for value in (torch.tensor(10.0), torch.tensor(2.0), torch.tensor(6.0)):
        scale.observe(value)

    chosen = scale.finalize()

    assert chosen == 6.0
    assert scale.value == 6.0
    assert torch.isclose(scale.normalize(torch.tensor(12.0)), torch.tensor(2.0))


def test_fixed_warmup_bag_scale_clamps_small_values() -> None:
    scale = FixedWarmupBagScale(min_scale=1e-3)
    scale.observe(torch.tensor(0.0))
    scale.observe(torch.tensor(1e-8))

    chosen = scale.finalize()

    assert chosen == 1e-3
    assert torch.isclose(scale.normalize(torch.tensor(1e-3)), torch.tensor(1.0))


def test_fixed_warmup_bag_scale_requires_observations() -> None:
    scale = FixedWarmupBagScale(min_scale=1e-3)

    try:
        scale.finalize()
    except ValueError as exc:
        assert "no bag losses observed" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_ema_bag_scale_updates_and_normalizes() -> None:
    scale = EmaBagScale(min_scale=1e-3, decay=0.5)

    scale.observe(torch.tensor(10.0))
    assert scale.value == 10.0
    scale.observe(torch.tensor(2.0))

    assert scale.value == 6.0
    assert torch.isclose(scale.normalize(torch.tensor(12.0)), torch.tensor(2.0))


def test_build_bag_scale_selects_fixed_or_ema() -> None:
    fixed = build_bag_scale(Exp08bConfig(bag_scale_mode="fixed_warmup"))
    ema = build_bag_scale(Exp08bConfig(bag_scale_mode="ema", bag_scale_ema_decay=0.9))

    assert isinstance(fixed, FixedWarmupBagScale)
    assert isinstance(ema, EmaBagScale)


def test_build_bag_scale_rejects_unknown_mode() -> None:
    try:
        build_bag_scale(Exp08bConfig(bag_scale_mode="mystery"))
    except ValueError as exc:
        assert "unknown bag_scale_mode" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_load_pert_vocab_reads_checkpoint_sibling(tmp_path: Path) -> None:
    checkpoint = tmp_path / "state" / "checkpoints" / "final.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()

    torch.save(
        {"A": np.eye(3, dtype=np.float32)[0]},
        checkpoint.parent.parent / "pert_onehot_map.pt",
    )

    loaded = load_pert_vocab(checkpoint)

    assert loaded is not None
    assert set(loaded) == {"A"}
    np.testing.assert_allclose(loaded["A"], np.array([1.0, 0.0, 0.0], dtype=np.float32))


def test_load_pert_vocab_returns_none_when_sidecar_missing(tmp_path: Path) -> None:
    """Missing pert_onehot_map.pt must return None, NOT {}.

    The exp08 `_ensure_pert_vocab` raise-on-missing contract (train.py) keys off
    a `None` return to distinguish "absent" from "present but empty". Returning
    `{}` here would silently disable the distill anchor — the OOD-token fix the
    spec mandates at full weight (§3.2) — and would also break the existing
    `tests/sl_dl_model/test_train.py::test_distill_required_but_missing_vocab_raises`.
    """
    checkpoint = tmp_path / "state" / "checkpoints" / "final.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()

    assert load_pert_vocab(checkpoint) is None


def _tiny_esm_and_bags() -> tuple[Esm2EmbeddingTable, GwpsBags, np.ndarray]:
    rng = np.random.default_rng(7)
    symbols = np.array(["A", "B", "C"], dtype=object)
    esm = Esm2EmbeddingTable(
        dim=4,
        vectors_by_symbol={
            "A": rng.standard_normal(4).astype(np.float32),
            "B": rng.standard_normal(4).astype(np.float32),
            "C": rng.standard_normal(4).astype(np.float32),
        },
    )
    bags = GwpsBags(
        control_template=rng.standard_normal((5, 3)).astype(np.float32),
        bags_by_symbol={
            "A": rng.standard_normal((5, 3)).astype(np.float32),
            "B": rng.standard_normal((5, 3)).astype(np.float32),
        },
        input_dim=3,
    )
    return esm, bags, symbols


def test_step1_trainer_writes_fold_local_cache_and_manifest(tmp_path: Path) -> None:
    esm, bags, symbols = _tiny_esm_and_bags()
    cfg = Exp08bConfig(
        output_dir=tmp_path / "run",
        state_backend="linear_mock",
        pert_dim=3,
        adapter_hidden=8,
        max_epochs=1,
        warmup_epochs=1,
        lambda_bag=1.0,
        lambda_distill=0.0,
        lambda_distill_after_warmup=0.0,
    )
    trainer = Step1GeneratorTrainer(
        cfg,
        esm=esm,
        bags=bags,
        input_dim=3,
        output_dim=3,
    )

    result = trainer.train_fold(
        split_type="CV2",
        fold_id=0,
        symbols=symbols,
        train_symbols={"A", "B", "C"},
    )

    assert result.embedding_path == embedding_cache_path(cfg, "CV2", 0)
    assert result.manifest_path == generator_manifest_path(cfg, "CV2", 0)
    assert result.weights_path == generator_weights_path(cfg, "CV2", 0)
    assert result.embedding_path.exists()
    assert result.manifest_path.exists()
    assert result.weights_path.exists()

    cache = load_embedding_cache(result.embedding_path)
    assert cache["symbols"].tolist() == ["A", "B", "C"]
    assert cache["embeddings"].shape == (3, 6)
    assert cache["coverage_mask"].tolist() == [1, 1, 0]

    manifest = load_generator_manifest(result.manifest_path)
    assert manifest["split_type"] == "CV2"
    assert manifest["fold_id"] == 0
    assert manifest["generator_kind"] == "state_adapter"
    assert manifest["train_bag_gene_count"] == 1
    assert manifest["val_bag_gene_count"] == 1
    assert manifest["bag_scale"] >= 1e-3
    assert manifest["generator_weights_path"] == str(result.weights_path)


def test_step1_trainer_uses_partialstate_device_not_cuda_default() -> None:
    source = Path("src/sl_dl_model/exp08b_generator.py").read_text()

    assert 'torch.device("cuda" if torch.cuda.is_available() else "cpu")' not in source
    assert "PartialState().device" in source or "device=" in source


def test_distill_symbols_include_fold_train_vocab_independent_of_bag_split(
    tmp_path: Path,
) -> None:
    esm, bags, _symbols = _tiny_esm_and_bags()
    cfg = Exp08bConfig(output_dir=tmp_path / "run", state_backend="linear_mock")
    trainer = Step1GeneratorTrainer(
        cfg,
        esm=esm,
        bags=bags,
        input_dim=3,
        output_dim=3,
    )
    trainer._pert_vocab = {
        "A": np.eye(3, dtype=np.float32)[0],
        "B": np.eye(3, dtype=np.float32)[1],
        "UNCOVERED": np.eye(3, dtype=np.float32)[2],
    }

    distill_symbols = trainer.distill_symbols_for_fold(
        {"A", "B", "UNCOVERED", "NOT_IN_VOCAB"}
    )

    assert distill_symbols == {"A", "B", "UNCOVERED"}


def test_distill_required_but_missing_vocab_raises(tmp_path: Path) -> None:
    """Real backend + positive distill + missing pert_onehot_map.pt must fail loudly.

    The spec keeps the distill anchor at full weight as the OOD-token fix
    (§3.2); a real-backend run that requests distill but cannot load the STATE
    vocab must raise rather than silently degrade to bag-only.
    """
    esm, bags, _symbols = _tiny_esm_and_bags()
    cfg = Exp08bConfig(
        output_dir=tmp_path / "run",
        state_backend="state_checkpoint",
        state_checkpoint=tmp_path / "state" / "checkpoints" / "final.ckpt",
        lambda_distill=1.0,
        lambda_distill_after_warmup=1.0,
    )
    trainer = Step1GeneratorTrainer(
        cfg,
        esm=esm,
        bags=bags,
        input_dim=3,
        output_dim=3,
    )

    try:
        trainer.distill_symbols_for_fold({"A", "B"})
    except RuntimeError as exc:
        assert "distill" in str(exc).lower()
    else:
        raise AssertionError("expected RuntimeError for missing required distill vocab")


def test_distill_not_required_when_weight_zero_does_not_raise(tmp_path: Path) -> None:
    """lambda_distill == 0 with a missing vocab is fine (distill not requested)."""
    esm, bags, _symbols = _tiny_esm_and_bags()
    cfg = Exp08bConfig(
        output_dir=tmp_path / "run",
        state_backend="state_checkpoint",
        state_checkpoint=tmp_path / "state" / "checkpoints" / "final.ckpt",
        lambda_distill=0.0,
        lambda_distill_after_warmup=0.0,
    )
    trainer = Step1GeneratorTrainer(
        cfg,
        esm=esm,
        bags=bags,
        input_dim=3,
        output_dim=3,
    )

    assert trainer.distill_symbols_for_fold({"A", "B"}) == set()


def test_step1_generator_source_does_not_read_sl_labels() -> None:
    source = Path("src/sl_dl_model/exp08b_generator.py").read_text()

    assert "sl_label" not in source
    assert "SymmetricPairHead" not in source


def test_step1_distill_only_does_not_crash_at_warmup_boundary(
    tmp_path: Path, monkeypatch
) -> None:
    """lambda_bag == 0 must not raise when the warmup window observed no bags.

    Regression for the distill-only ablation: with lambda_bag == 0 the bag
    block never calls ``scale.observe``, so an unconditional ``finalize()`` at
    the warmup boundary would raise ``no bag losses observed``. The boundary
    finalize must be skipped, and the post-loop guard must default the unused
    scale to 1.0. ``_distill_term`` is monkeypatched to a param-free scalar so
    the linear_mock backend (empty pert-vocab, no STATE pert_encoder) still
    yields a trainable distill term.
    """
    esm, bags, symbols = _tiny_esm_and_bags()
    cfg = Exp08bConfig(
        output_dir=tmp_path / "run",
        state_backend="linear_mock",
        max_epochs=2,
        warmup_epochs=1,
        lambda_bag=0.0,
        lambda_distill=1.0,
        lambda_distill_after_warmup=1.0,
    )
    trainer = Step1GeneratorTrainer(
        cfg,
        esm=esm,
        bags=bags,
        input_dim=3,
        output_dim=3,
    )
    trainer._pert_vocab = {
        "A": np.eye(3, dtype=np.float32)[0],
        "B": np.eye(3, dtype=np.float32)[1],
        "C": np.eye(3, dtype=np.float32)[2],
    }
    monkeypatch.setattr(
        trainer,
        "_distill_term",
        lambda generator, symbol, device: torch.tensor(1.0, requires_grad=True),
    )

    result = trainer.train_fold(
        split_type="CV2",
        fold_id=0,
        symbols=symbols,
        train_symbols={"A", "B", "C"},
    )

    manifest = load_generator_manifest(result.manifest_path)
    assert manifest["bag_scale"] == 1.0
    assert manifest["distill_gene_count"] == 3
