import logging
import pickle
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from sl_dl_model.bags import build_gwps_bags, load_bags_npz, save_bags_npz
from sl_dl_model.config import SLDLConfig


def _toy_h5ad(path):
    n, d = 200, 6
    rng = np.random.default_rng(0)
    genes = ["non-targeting"] * 80 + ["AAAS"] * 60 + ["KRAS"] * 60
    obs = pd.DataFrame({"gene": genes, "gem_group": ["b0"] * n})
    adata = ad.AnnData(X=rng.normal(size=(n, d)).astype("float32"), obs=obs)
    adata.obsm["X_hvg"] = rng.normal(size=(n, d)).astype("float32")
    adata.write_h5ad(path)


def _toy_h5ad_for_state_alignment(path: Path) -> np.ndarray:
    genes = ["non-targeting"] * 3 + ["AAAS"] * 2
    obs = pd.DataFrame({"gene": genes})
    var = pd.DataFrame(
        {"gene_name": ["G_B", "G_A", "G_D", "G_C"]},
        index=["ens_b", "ens_a", "ens_d", "ens_c"],
    )
    x = np.arange(20, dtype=np.float32).reshape(5, 4)
    adata = ad.AnnData(X=x, obs=obs, var=var)
    adata.write_h5ad(path)
    return x


def _write_state_var_dims(checkpoint: Path, gene_names: list[str]) -> None:
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.touch()
    with (checkpoint.parent.parent / "var_dims.pkl").open("wb") as handle:
        pickle.dump({"input_dim": len(gene_names), "gene_names": gene_names}, handle)


def _toy_h5ad_no_control(path):
    """Synthetic h5ad with NO 'non-targeting' cells (only perturbation genes)."""
    n, d = 60, 6
    rng = np.random.default_rng(1)
    genes = ["AAAS"] * 30 + ["KRAS"] * 30
    obs = pd.DataFrame({"gene": genes})
    adata = ad.AnnData(X=rng.normal(size=(n, d)).astype("float32"), obs=obs)
    adata.write_h5ad(path)


def _toy_h5ad_single_cell_gene(path):
    """Synthetic h5ad where one gene ('BRCA1') has exactly 1 cell."""
    n, d = 82, 6
    rng = np.random.default_rng(2)
    # 80 control + 1 BRCA1 + 1 KRAS (KRAS also single-cell to keep it simple)
    genes = ["non-targeting"] * 80 + ["BRCA1"] * 1 + ["KRAS"] * 1
    obs = pd.DataFrame({"gene": genes})
    adata = ad.AnnData(X=rng.normal(size=(n, d)).astype("float32"), obs=obs)
    adata.write_h5ad(path)


def test_build_and_cache_bags(tmp_path):
    h5ad = tmp_path / "toy.h5ad"
    _toy_h5ad(h5ad)
    cfg = SLDLConfig(gwps_h5ad=h5ad, control_template_size=16, cells_per_bag=16)
    bags = build_gwps_bags(cfg, rng_seed=17)
    assert bags.input_dim == 6
    assert bags.control_template.shape == (16, 6)
    assert set(bags.bags_by_symbol) == {"AAAS", "KRAS"}
    assert bags.bags_by_symbol["KRAS"].shape[1] == 6

    npz = tmp_path / "bags.npz"
    save_bags_npz(bags, npz)
    loaded = load_bags_npz(npz)
    assert set(loaded.bags_by_symbol) == {"AAAS", "KRAS"}
    assert np.allclose(loaded.control_template, bags.control_template)


def test_build_gwps_bags_aligns_to_state_checkpoint_genes(tmp_path: Path) -> None:
    """Raw h5ad expression is projected into checkpoint gene order."""
    h5ad = tmp_path / "toy_state.h5ad"
    x = _toy_h5ad_for_state_alignment(h5ad)
    checkpoint = tmp_path / "state" / "checkpoints" / "final.ckpt"
    _write_state_var_dims(checkpoint, ["G_C", "G_A"])

    cfg = SLDLConfig(
        gwps_h5ad=h5ad,
        state_checkpoint=checkpoint,
        control_template_size=8,
        cells_per_bag=8,
    )
    bags = build_gwps_bags(cfg, rng_seed=17)

    assert bags.input_dim == 2
    assert np.allclose(bags.control_template, x[:3][:, [3, 1]])
    assert np.allclose(bags.bags_by_symbol["AAAS"], x[3:][:, [3, 1]])


# ---------------------------------------------------------------------------
# FIX 1 — HIGH: empty control template raises ValueError
# ---------------------------------------------------------------------------


def test_build_gwps_bags_raises_on_missing_control(tmp_path):
    """build_gwps_bags must raise ValueError when no 'non-targeting' cells exist.

    Without this guard, control_rows is empty → control_template shape (0, D)
    → downstream mean/std pooling silently produces NaN.
    """
    h5ad = tmp_path / "no_control.h5ad"
    _toy_h5ad_no_control(h5ad)
    cfg = SLDLConfig(gwps_h5ad=h5ad, control_template_size=16, cells_per_bag=16)
    with pytest.raises(ValueError, match="non-targeting"):
        build_gwps_bags(cfg, rng_seed=17)


def test_load_bags_npz_raises_on_empty_control_template(tmp_path):
    """load_bags_npz must raise ValueError when saved control_template has 0 rows."""
    # Build a valid bags object but manually zero out control rows before saving
    d = 6
    # Construct a GwpsBags-like NPZ manually with 0-row control
    symbols = np.array(["AAAS"], dtype=object)
    flat = np.zeros((5, d), dtype=np.float32)
    offsets = np.array([0, 5], dtype=np.int64)
    empty_control = np.zeros((0, d), dtype=np.float32)

    npz_path = tmp_path / "bad_bags.npz"
    np.savez(
        npz_path,
        control_template=empty_control,
        symbols=symbols,
        flat=flat,
        offsets=offsets,
        input_dim=np.int64(d),
    )
    with pytest.raises(ValueError, match="non-targeting"):
        load_bags_npz(npz_path)


# ---------------------------------------------------------------------------
# FIX 2 — MEDIUM: single-cell bags trigger a warning but are not dropped
# ---------------------------------------------------------------------------


def test_single_cell_bag_triggers_warning(tmp_path, caplog):
    """build_gwps_bags warns when any bag has <2 cells, but keeps the gene."""
    h5ad = tmp_path / "single_cell.h5ad"
    _toy_h5ad_single_cell_gene(h5ad)
    cfg = SLDLConfig(gwps_h5ad=h5ad, control_template_size=16, cells_per_bag=16)

    with caplog.at_level(logging.WARNING, logger="sl_dl_model.bags"):
        bags = build_gwps_bags(cfg, rng_seed=17)

    # Warning must have been emitted
    warning_messages = [rec.message.lower() for rec in caplog.records]
    assert any(
        "single-cell" in msg or "fewer than 2" in msg for msg in warning_messages
    ), f"Expected a warning about single-cell bags; got: {warning_messages}"

    # Gene must still be present (we warn, not drop)
    assert "BRCA1" in bags.bags_by_symbol
    assert bags.bags_by_symbol["BRCA1"].shape == (1, 6)
