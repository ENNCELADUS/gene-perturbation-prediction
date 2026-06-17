import anndata as ad
import numpy as np
import pandas as pd

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
