"""The ``.venv-states`` fits (``src.data.prepare.state_programs``) on synthetic
lines sharing two programs and a cycle program: recurrent programs found across
lines, cycle programs set aside, usages and scVI latents for every cell.

Needs cNMF, starCAT and scvi-tools (``.venv-states``); skipped without them.
"""

from __future__ import annotations

import json

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

LINES = tuple(f"ACH-L{i}" for i in range(6))
GENES = tuple(f"G{i}" for i in range(240))
#: Gene blocks of two recurrent programs, a cycle program and line identity.
BLOCKS = {"a": range(0, 30), "b": range(30, 60), "cycle": range(60, 90)}


def _world(root):
    rng = np.random.default_rng(0)
    base = rng.gamma(2.0, 1.0, size=len(GENES))
    paths = {}
    for index, model_id in enumerate(LINES):
        n = 120 if index else 15  # one line with few cells
        identity = base * rng.lognormal(0.0, 0.3, size=len(GENES))
        usage = rng.dirichlet([0.6, 0.6, 0.6], size=n)
        rate = np.tile(identity, (n, 1))
        for k, block in enumerate(BLOCKS.values()):
            rate[:, list(block)] *= 1 + 12 * usage[:, [k]]
        rate *= (
            rng.uniform(1.0, 3.0, size=(n, 1)) / rate.sum(axis=1, keepdims=True) * 2000
        )
        counts = rng.poisson(rate).astype(np.float32)
        path = root / f"{model_id}.h5ad"
        ad.AnnData(
            X=sparse.csr_matrix(counts),
            obs=pd.DataFrame(
                {"model_id": model_id}, index=[f"{model_id}-c{i}" for i in range(n)]
            ),
            var=pd.DataFrame(
                {
                    "ensembl_id": [f"ENSG{i:011d}" for i in range(len(GENES))],
                    "gene_symbol": list(GENES),
                },
                index=[f"v{i}" for i in range(len(GENES))],
            ),
        ).write_h5ad(path)
        paths[model_id] = str(path)
    return paths


def test_programs_and_scvi_on_synthetic_lines(tmp_path):
    pytest.importorskip("cnmf")
    pytest.importorskip("starcat")
    pytest.importorskip("scvi")
    from src.data.prepare import state_programs as sp

    paths = _world(tmp_path)
    train = list(LINES[:5])
    rng = np.random.default_rng(1)
    rows = []
    for model_id in train:
        n = ad.read_h5ad(paths[model_id]).n_obs
        chosen = np.sort(rng.choice(n, size=min(n, 40), replace=False))
        rows.append(
            pd.DataFrame(
                {
                    "model_id": model_id,
                    "row": chosen,
                    "cell_id": [f"{model_id}-c{i}" for i in chosen],
                    "chemistry": "10x 3' v3" if model_id < "ACH-L3" else "10x 3' v2",
                    "s_score": rng.normal(size=len(chosen)),
                    "g2m_score": rng.normal(size=len(chosen)),
                }
            )
        )
    second = ad.read_h5ad(paths["ACH-L1"])[::-1].copy()
    second.obs_names = [f"{n}-second" for n in second.obs_names]
    second.write_h5ad(tmp_path / "second.h5ad")
    sample = tmp_path / "sample_seed0.parquet"
    pd.concat(rows, ignore_index=True).to_parquet(sample)
    plan = {
        "root": str(tmp_path / "states"),
        "seeds": [0],
        "train": train,
        "programs": {
            "ks": [3, 4],
            "iterations": 4,
            "density_threshold": 2.0,
            "highly_variable": 120,
            "top_genes": 20,
            "robust_overlap": 10,
            "redundant_overlap": 6,
            "cycle_overlap": 6,
            "cluster_overlap": 6,
            "min_lines": 3,
        },
        "scvi": {
            "latent": 3,
            "hidden": 16,
            "layers": 1,
            "max_epochs": 3,
            "batch_size": 64,
        },
        "panel": list(GENES),
        "hvg": list(GENES[:100]),
        "cycle_genes": {
            "S": [GENES[i] for i in range(60, 75)],
            "G2M": [GENES[i] for i in range(75, 90)],
        },
        "var_ensembl_col": "ensembl_id",
        "encode": paths,
        "second": {"ACH-L1": str(tmp_path / "second.h5ad")},
        "samples": {"0": str(sample)},
    }
    sp.run(plan)
    folder = tmp_path / "states" / "programs" / "seed0"
    reference = dict(np.load(folder / "reference.npz"))
    kinds = list(reference["kinds"])
    assert kinds.count("recurrent") >= 1 and kinds[-1] == "background"
    spectra = reference["spectra"]
    assert spectra.shape == (len(kinds), len(reference["genes"]))
    assert (spectra.sum(axis=1) > 0).all()
    # Each recurrent program's spectrum itself (not only its top genes) sits on
    # one planted gene block, and the two recurrent programs on different ones.
    genes = list(reference["genes"])
    planted = []
    for row in np.flatnonzero(np.asarray(kinds) == "recurrent"):
        means = {
            name: spectra[
                row, [genes.index(GENES[i]) for i in block if GENES[i] in genes]
            ].mean()
            for name, block in BLOCKS.items()
        }
        rest = spectra[row, [i for i, g in enumerate(genes) if int(g[1:]) >= 90]].mean()
        best = max(means, key=means.get)
        assert means[best] > 2 * rest
        planted.append(best)
    assert set(planted) == {"a", "b"}
    # Per-line cNMF spectra are programs x genes and non-zero.
    line = dict(np.load(folder / "cnmf" / "ACH-L1.npz"))
    assert line["tpm_k3"].shape == (3, len(GENES)) and (line["tpm_k3"].sum(1) > 0).all()
    manifest = json.loads((folder / "manifest.json").read_text())
    assert "usage_to_weight" in manifest and manifest["starcat_check"]["cells"] == 15
    for model_id, path in paths.items():
        n = ad.read_h5ad(path).n_obs
        usages = np.load(folder / "usages" / f"{model_id}.npy")
        assert usages.shape == (n, len(kinds)) and (usages >= 0).all()
        latent = np.load(
            tmp_path / "states" / "scvi" / "seed0" / "latent" / f"{model_id}.npy"
        )
        assert latent.shape == (n, 3)
    assert (folder / "usages_second" / "ACH-L1.npy").is_file()
    sp.run(plan)  # resumes: every output exists
    # A resumed run reloads the saved scVI model (with its AnnData) and encodes
    # the same latents again.
    latent_dir = tmp_path / "states" / "scvi" / "seed0" / "latent"
    first = np.load(latent_dir / "ACH-L3.npy")
    (latent_dir / "ACH-L3.npy").unlink()
    sp.run(plan)
    assert np.allclose(np.load(latent_dir / "ACH-L3.npy"), first, atol=1e-5)


def test_program_spectra_orientation_and_alignment():
    pytest.importorskip("cnmf")
    from src.data.prepare import state_programs as sp

    panel = [f"G{i}" for i in range(6)]
    # cNMF's load_results: genes x programs, programs labelled 1..K.
    spectra = pd.DataFrame(
        [[9.0, 0.1], [8.0, 0.1], [0.1, 7.0], [0.1, 9.0], [1.0, 1.0]],
        index=panel[:5],
        columns=[1, 2],
    )
    top = pd.DataFrame({1: ["G0", "G1"], 2: ["G3", "G2"]})
    out = sp.program_spectra(spectra, top, panel)
    assert list(out.index) == [1, 2] and list(out.columns) == panel
    assert out.loc[1, "G0"] == 9.0 and out.loc[2, "G3"] == 9.0
    assert out.loc[2, "G5"] == 0.0  # a panel gene the line lacks
    # Rows follow top's program order, by label.
    flipped = sp.program_spectra(spectra, top[[2, 1]], panel)
    assert list(flipped.index) == [2, 1] and flipped.loc[2, "G3"] == 9.0
    with pytest.raises(ValueError, match="as columns"):
        sp.program_spectra(spectra.T, top, panel)
    empty = spectra.copy()
    empty[2] = 0.0
    with pytest.raises(ValueError, match="empty program spectrum"):
        sp.program_spectra(empty, top, panel)


def test_the_worker_pool_spawns_rather_than_forks(monkeypatch):
    """A pool forked after the scVI fit (torch threads, a CUDA context) deadlocked
    on the H20; workers are spawned, and the calls still run."""
    from functools import partial

    from src.data.prepare import state_programs

    contexts = []
    real = state_programs.ProcessPoolExecutor

    def recording(*args, **kwargs):
        contexts.append(kwargs.get("mp_context"))
        return real(*args, **kwargs)

    monkeypatch.setattr(state_programs, "ProcessPoolExecutor", recording)
    monkeypatch.setattr(state_programs, "_workers", lambda: 2)
    state_programs._in_processes([partial(int, "7"), partial(int, "8")], "test")
    assert [c.get_start_method() for c in contexts] == ["spawn"]
