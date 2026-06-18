"""Shared fixtures for sl_benchmark_baseline tests."""

from __future__ import annotations

import os

# MPS fallback guard for exp08 (sl_dl_model). The bag-supervision loss uses
# torch.cdist, whose backward is unimplemented on the MPS backend; the energy
# distance therefore needs the CPU fallback. This env var must be set before
# torch initializes the MPS backend, so it lives here in conftest (imported
# before any test module) rather than inside an individual test file.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# macOS OpenMP load-order guard. xgboost and torch each bundle their own libomp;
# loading torch's runtime first and then fitting xgboost segfaults the process.
# conftest is imported before any test module, so importing xgboost here pins the
# safe (xgboost-then-torch) load order for the whole suite. Guarded so the suite
# still collects if xgboost is absent.
try:  # noqa: SIM105
    import xgboost  # noqa: F401
except ImportError:
    pass

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _write_synthetic_benchmark(path: Path) -> Path:
    """Write a tiny CV1-shaped benchmark CSV with 2 folds and both classes.

    Each fold has 8 train rows (4 pos, 4 neg) and 6 test rows (3 pos, 3 neg).
    Gene effects are chosen so positives skew more negative (more essential).
    """
    rows = []
    rng = np.random.default_rng(0)
    genes = [f"G{i}" for i in range(12)]
    pair_counter = 0
    for fold_id in (0, 1):
        for role, n_each in (("train", 4), ("test", 3)):
            for label in (1, 0):
                for _ in range(n_each):
                    a = genes[rng.integers(0, len(genes))]
                    b = genes[rng.integers(0, len(genes))]
                    while b == a:
                        b = genes[rng.integers(0, len(genes))]
                    # positives more essential (more negative gene effect)
                    base = -1.0 if label == 1 else 0.2
                    ea = base + rng.normal(0, 0.1)
                    eb = base + rng.normal(0, 0.1)
                    rows.append(
                        {
                            "pair_id": f"P{pair_counter}",
                            "fold_id": fold_id,
                            "split_role": role,
                            "sl_label": label,
                            "gene_a_symbol": a,
                            "gene_b_symbol": b,
                            "gene_a_k562_gene_effect": ea,
                            "gene_b_k562_gene_effect": eb,
                        }
                    )
                    pair_counter += 1
    frame = pd.DataFrame(rows)
    csv_path = path / "synthetic_sl.csv"
    frame.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def synthetic_benchmark_csv(tmp_path: Path) -> Path:
    """Provide a path to a freshly written synthetic benchmark CSV."""
    return _write_synthetic_benchmark(tmp_path)


def _write_synthetic_all_cv_benchmark(path: Path) -> Path:
    """Write a tiny benchmark CSV with CV1/CV2 sharing fold ids."""
    rows = []
    pair_counter = 0
    for split_type, offset in (("CV1", 0.0), ("CV2", 0.5)):
        for role, n_each in (("train", 3), ("test", 2)):
            for label in (1, 0):
                for index in range(n_each):
                    rows.append(
                        {
                            "pair_id": f"{split_type}_P{pair_counter}",
                            "split_type": split_type,
                            "fold_id": 0,
                            "split_role": role,
                            "sl_label": label,
                            "gene_a_symbol": f"{split_type}_GA{index}",
                            "gene_b_symbol": f"{split_type}_GB{index}",
                            "gene_a_k562_gene_effect": -1.0 + offset
                            if label == 1
                            else 0.2 + offset,
                            "gene_b_k562_gene_effect": -0.8 + offset
                            if label == 1
                            else 0.3 + offset,
                        }
                    )
                    pair_counter += 1
    frame = pd.DataFrame(rows)
    csv_path = path / "synthetic_all_cv_sl.csv"
    frame.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def synthetic_all_cv_benchmark_csv(tmp_path: Path) -> Path:
    """Provide a synthetic benchmark CSV with multiple split types."""
    return _write_synthetic_all_cv_benchmark(tmp_path)


def _write_synthetic_bags_npz(path: Path) -> Path:
    """Write a tiny cell-bags NPZ: 3 covered genes, 2-dim embeddings."""
    # gene G0: 2 cells, gene G1: 1 cell, gene G2: 3 cells
    cell_delta_pcs = np.array(
        [
            [1.0, 0.0],
            [3.0, 2.0],  # G0 -> mean [2, 1]
            [5.0, 5.0],  # G1 -> mean [5, 5]
            [0.0, 0.0],
            [2.0, 0.0],
            [4.0, 6.0],  # G2 -> mean [2, 2]
        ],
        dtype=np.float32,
    )
    bag_offsets = np.array([0, 2, 3, 6], dtype=np.int64)
    perturbation_gene = np.asarray(["G0", "G1", "G2"], dtype=object)
    npz_path = path / "synthetic_bags.npz"
    np.savez_compressed(
        npz_path,
        cell_delta_pcs=cell_delta_pcs,
        bag_offsets=bag_offsets,
        perturbation_gene=perturbation_gene,
    )
    return npz_path


@pytest.fixture
def synthetic_bags_npz(tmp_path: Path) -> Path:
    """Provide a path to a freshly written synthetic cell-bags NPZ."""
    return _write_synthetic_bags_npz(tmp_path)


def _write_augmented_benchmark_csv(path: Path) -> Path:
    """Deterministic benchmark guaranteeing both-covered and mixed test pairs.

    Covered genes C0-C3 (present in the augmented bags NPZ); uncovered U0-U1.
    Per fold (0, 1), each role/label includes at least one both-covered pair and
    one mixed pair, so the covered_pairs slice is always non-empty.
    """
    effects = {"C0": -1.2, "C1": -1.0, "C2": -0.9, "C3": -1.1, "U0": 0.2, "U1": 0.3}
    # (role, label, gene_a, gene_b)
    spec = [
        ("train", 1, "C0", "C1"),
        ("train", 1, "C2", "C3"),
        ("train", 0, "C0", "U0"),
        ("train", 0, "C1", "U1"),
        ("test", 1, "C0", "C2"),
        ("test", 1, "C1", "U0"),
        ("test", 0, "C1", "C3"),
        ("test", 0, "C2", "U1"),
    ]
    rows = []
    pair_counter = 0
    for fold_id in (0, 1):
        for role, label, a, b in spec:
            rows.append(
                {
                    "pair_id": f"P{pair_counter}",
                    "fold_id": fold_id,
                    "split_role": role,
                    "sl_label": label,
                    "gene_a_symbol": a,
                    "gene_b_symbol": b,
                    "gene_a_k562_gene_effect": effects[a],
                    "gene_b_k562_gene_effect": effects[b],
                }
            )
            pair_counter += 1
    csv_path = path / "synthetic_augmented_sl.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path


def _write_augmented_bags_npz(path: Path) -> Path:
    """Bags NPZ covering only C0-C3 (matches _write_augmented_benchmark_csv)."""
    cell_delta_pcs = np.array(
        [
            [1.0, 0.0],
            [3.0, 2.0],  # C0 -> [2, 1]
            [5.0, 5.0],  # C1 -> [5, 5]
            [0.0, 0.0],
            [4.0, 4.0],  # C2 -> [2, 2]
            [1.0, 3.0],  # C3 -> [1, 3]
        ],
        dtype=np.float32,
    )
    bag_offsets = np.array([0, 2, 3, 5, 6], dtype=np.int64)
    perturbation_gene = np.asarray(["C0", "C1", "C2", "C3"], dtype=object)
    npz_path = path / "synthetic_augmented_bags.npz"
    np.savez_compressed(
        npz_path,
        cell_delta_pcs=cell_delta_pcs,
        bag_offsets=bag_offsets,
        perturbation_gene=perturbation_gene,
    )
    return npz_path


@pytest.fixture
def synthetic_augmented_benchmark_csv(tmp_path: Path) -> Path:
    """Deterministic benchmark CSV with guaranteed covered/mixed test pairs."""
    return _write_augmented_benchmark_csv(tmp_path)


@pytest.fixture
def synthetic_augmented_bags_npz(tmp_path: Path) -> Path:
    """Bags NPZ covering exactly the covered genes of the augmented benchmark."""
    return _write_augmented_bags_npz(tmp_path)


@pytest.fixture
def synthetic_selectivity_fixture(tmp_path: Path) -> dict:
    """Tiny benchmark CSV with entrez ids + matching tiny DepMap CSVs.

    6 genes (entrez 10..60), 30 cell lines, CV1 with 2 folds. Returns a dict
    with ``benchmark_csv`` and ``depmap_dir`` paths.
    """
    rng = np.random.default_rng(3)
    entrez = [10, 20, 30, 40, 50, 60]
    symbols = [f"G{e}" for e in entrez]
    lines = [f"ACH-{i:04d}" for i in range(30)]

    # DepMap dir with the 5 standard files
    depmap_dir = tmp_path / "depmap"
    depmap_dir.mkdir()
    ge = pd.DataFrame(
        rng.normal(-0.3, 0.6, size=(len(lines), len(entrez))),
        index=lines,
        columns=[f"{s} ({e})" for s, e in zip(symbols, entrez, strict=True)],
    )
    ge.to_csv(depmap_dir / "CRISPRGeneEffect.csv")
    cn = pd.DataFrame(
        rng.uniform(0.6, 1.4, size=(len(lines), len(entrez))),
        index=lines,
        columns=[f"{s} ({e})" for s, e in zip(symbols, entrez, strict=True)],
    )
    cn.to_csv(depmap_dir / "PortalOmicsCNGeneLog2.csv")
    for fname, density in (
        ("OmicsSomaticMutationsMatrixDamaging.csv", 0.3),
        ("OmicsSomaticMutationsMatrixHotspot.csv", 0.1),
    ):
        mat = (rng.uniform(0, 1, size=(len(lines), len(entrez))) < density).astype(int)
        frame = pd.DataFrame(
            mat, columns=[f"{s} ({e})" for s, e in zip(symbols, entrez, strict=True)]
        )
        frame.insert(0, "ModelID", lines)
        frame.insert(0, "SequencingID", [f"s{i}" for i in range(len(lines))])
        frame.to_csv(depmap_dir / fname, index=False)
    expr = pd.DataFrame(
        rng.uniform(0, 7, size=(len(lines), len(entrez))),
        columns=[f"{s} ({e})" for s, e in zip(symbols, entrez, strict=True)],
    )
    expr.insert(0, "ModelID", lines)
    expr.to_csv(
        depmap_dir / "OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv",
        index=False,
    )

    # Benchmark CSV: 2 folds, both classes, entrez + negative_sampling_method
    rows = []
    counter = 0
    for fold_id in (0, 1):
        for role, n_each in (("train", 4), ("test", 3)):
            for label in (1, 0):
                for _ in range(n_each):
                    ia, ib = rng.integers(0, len(entrez), size=2)
                    while ib == ia:
                        ib = rng.integers(0, len(entrez))
                    base = -1.0 if label == 1 else 0.2
                    rows.append(
                        {
                            "pair_id": f"P{counter}",
                            "negative_sampling_method": "Rand",
                            "fold_id": fold_id,
                            "split_role": role,
                            "sl_label": label,
                            "gene_a_symbol": symbols[ia],
                            "gene_b_symbol": symbols[ib],
                            "gene_a_unified_id": int(entrez[ia]),
                            "gene_b_unified_id": int(entrez[ib]),
                            "gene_a_entrez_id": int(entrez[ia]),
                            "gene_b_entrez_id": int(entrez[ib]),
                            "gene_a_k562_gene_effect": base + rng.normal(0, 0.1),
                            "gene_b_k562_gene_effect": base + rng.normal(0, 0.1),
                        }
                    )
                    counter += 1
    csv_path = tmp_path / "bench_sel.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return {"benchmark_csv": csv_path, "depmap_dir": depmap_dir}
