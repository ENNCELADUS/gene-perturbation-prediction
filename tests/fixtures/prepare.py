"""The raw sources preparation reads: per-line basal h5ad with raw UMI, Perturb-seq
response sources and labels, for labelled, unlabelled and held-out lines."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import yaml
from scipy.sparse import csr_matrix

from src.data.tx1_cache import write_line_cache

K562, HEPG2, HCT116, JURKAT = "ACH-000551", "ACH-000739", "ACH-000971", "ACH-000995"
ANCHORS = (K562, HEPG2, HCT116, JURKAT)
TRAIN = (*ANCHORS, "ACH-T1", "ACH-T2", "ACH-U1")
VAL = ("ACH-V1", "ACH-V2", "ACH-V3")
TEST = ("ACH-S1", "ACH-S2", "ACH-S3")
UNLABELED = ("ACH-U1",)
HVG = ("H1", "H2", "H3", "H4", "H5", "H6")
PANEL_CANDIDATES = ("GA", "GB", "GC", "GD", "GE")
SOURCE_GENES = (*HVG, *PANEL_CANDIDATES, "X1", "X2")
CELLS_PER_LINE = 10
CACHED_CELLS = (9, 7, 5, 3, 1, 0)
CELLS_PER_CONTEXT = 4
RESPONSE_GENES = {"GA": 3, "GB": 3, "ZZ": 2}
N_CONTROLS = 6


def _counts(rng: np.random.Generator, n_cells: int, n_genes: int) -> np.ndarray:
    counts = rng.poisson(3.0, size=(n_cells, n_genes)).astype(np.float32)
    counts[:, -2:] += rng.integers(20, 60, size=(n_cells, 2))  # extra genes
    return counts


def write_basal_source(path: Path, model_id: str, seed: int, *, scale=1.0) -> None:
    rng = np.random.default_rng(seed)
    counts = _counts(rng, CELLS_PER_LINE, len(SOURCE_GENES)) * scale
    obs = pd.DataFrame(
        {"model_id": [model_id] * CELLS_PER_LINE},
        index=[f"{model_id}-c{i}" for i in range(CELLS_PER_LINE)],
    )
    var = pd.DataFrame(
        {
            "ensembl_id": [f"ENSG{i:011d}" for i in range(len(SOURCE_GENES))],
            "gene_symbol": list(SOURCE_GENES),
        },
        index=[f"v{i}" for i in range(len(SOURCE_GENES))],
    )
    ad.AnnData(X=csr_matrix(counts), obs=obs, var=var).write_h5ad(path)


def _write_response_source(path: Path, seed: int, genes: tuple[str, ...]) -> None:
    rng = np.random.default_rng(seed)
    labels = ["non-targeting"] * N_CONTROLS
    for gene, count in RESPONSE_GENES.items():
        labels += [gene] * count
    counts = _counts(rng, len(labels), len(genes))
    obs = pd.DataFrame(
        {"gene": labels}, index=[f"r{seed}-{i}" for i in range(len(labels))]
    )
    var = pd.DataFrame(
        {
            "gene_id": [f"ENSG{SOURCE_GENES.index(g):011d}" for g in genes],
            "gene_name": list(genes),
        },
        index=[f"v{i}" for i in range(len(genes))],
    )
    ad.AnnData(X=csr_matrix(counts), obs=obs, var=var).write_h5ad(path)


def write_line(root: Path, line: str, seed: int) -> None:
    """One line's raw-UMI source and its Tx1 cache."""
    write_basal_source(root / "basal_raw_umi" / f"{line}.h5ad", line, seed=seed)
    cell_ids = [f"{line}-c{i}" for i in CACHED_CELLS]
    write_line_cache(
        root / "tx1",
        line,
        np.random.default_rng(seed).normal(size=(len(cell_ids), 2560)),
        np.zeros((len(cell_ids), len(HVG)), dtype=np.float32),
        pd.DataFrame({"model_id": line}, index=cell_ids),
    )


def build_world(root: Path) -> dict:
    """A complete raw world: split, labels, sources, Tx1 cache, STATE, ESM2."""
    root.mkdir(parents=True, exist_ok=True)
    lines = (*TRAIN, *VAL, *TEST)
    (root / "split.json").write_text(
        json.dumps(
            {
                "train": list(TRAIN),
                "val": list(VAL),
                "test": list(TEST),
                "unlabeled_train": list(UNLABELED),
            }
        )
    )
    rng = np.random.default_rng(7)
    labeled = [line for line in lines if line not in UNLABELED]
    effects = rng.normal(size=(len(labeled), len(PANEL_CANDIDATES)))
    for row, line in enumerate(labeled):
        if line in VAL or line in TEST:
            effects[row] += 100.0  # would visibly shift any leaked fit
    frame = pd.DataFrame(
        effects,
        index=labeled,
        columns=[f"{g} ({i + 1})" for i, g in enumerate(PANEL_CANDIDATES)],
    )
    frame.loc[K562, "GE (5)"] = np.nan  # no copy-prior donor value -> off panel
    frame.to_csv(root / "gene_effect.csv")

    (root / "basal_raw_umi").mkdir()
    for index, line in enumerate(lines):
        write_line(root, line, seed=index)
    response = {}
    for index, anchor in enumerate(ANCHORS):
        genes = SOURCE_GENES if anchor != HCT116 else SOURCE_GENES[1:]  # no H1
        path = root / f"response_{anchor}.h5ad"
        _write_response_source(path, seed=100 + index, genes=genes)
        response[anchor] = {
            "source_type": "h5ad",
            "h5ad_path": str(path),
            "perturbation_col": "gene",
            "control_label": "non-targeting",
            "var_ensembl_col": "gene_id",
            "target_gene_symbol_col": "gene_name",
        }
    (root / "perturbseq_sources.json").write_text(json.dumps(response))

    (root / "state").mkdir()
    (root / "state" / "var_dims.pkl").write_bytes(
        pickle.dumps({"gene_names": list(HVG)})
    )
    esm2 = (*PANEL_CANDIDATES, "GF")
    np.savez(
        root / "esm2.npz",
        symbols=np.array(esm2),
        vectors=rng.normal(size=(len(esm2), 4)).astype(np.float32),
        resolved=np.ones(len(esm2), dtype=bool),
    )

    config = yaml.safe_load(Path("configs/geneeffect_joint.yaml").read_text())
    config["prepared_root"] = str(root / "prepared")
    config["features"]["cells_per_context"] = CELLS_PER_CONTEXT
    config["model"]["context_components"] = 4  # within the training lines' rank
    config["paths"].update(
        split=str(root / "split.json"),
        gene_effect=str(root / "gene_effect.csv"),
        basal_raw_umi=str(root / "basal_raw_umi"),
        tx1_cache=str(root / "tx1"),
        esm2_embeddings=str(root / "esm2.npz"),
        state_model_dir=str(root / "state"),
        perturbseq_sources=str(root / "perturbseq_sources.json"),
    )
    return config
