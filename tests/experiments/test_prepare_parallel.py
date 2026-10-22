"""Parallel, memory-bounded preparation writes exactly what serial preparation wrote."""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from scipy.sparse import csr_matrix

from tests.fixtures import prepare as base
from src.experiments import prepare
from src.experiments.prepare import prepare_inputs
from tests.fixtures.xatlas import (
    CAPPED,
    RESPONSE_GENES,
    naive_xatlas_cells,
    response_labels,
    write_xatlas,
)

#: Response genes with an ESM2 vector; the rest are dropped from the cache.
RESOLVED_RESPONSE_GENES = (*RESPONSE_GENES[:34], "KIF11", "TP53")


def _write_rich_h5ad(path: Path, seed: int, symbols: tuple[str, ...], *, dense: bool):
    rng = np.random.default_rng(seed)
    labels = response_labels(rng, "non-targeting")
    counts = rng.poisson(0.9, size=(len(labels), len(symbols))).astype(np.float32)
    counts[:, -1] += rng.integers(0, 40, size=len(labels))
    obs = pd.DataFrame(
        {"gene": labels}, index=[f"r{seed}-{i}" for i in range(len(labels))]
    )
    var = pd.DataFrame(
        {
            "gene_id": [f"ENSG{9000 + i:011d}" for i in range(len(symbols))],
            "gene_name": list(symbols),
        },
        index=[f"v{i}" for i in range(len(symbols))],
    )
    matrix = counts if dense else csr_matrix(counts)
    ad.AnnData(X=matrix, obs=obs, var=var).write_h5ad(path)


def build_rich_world(root: Path, caps: dict) -> dict:
    """``test_prepare.build_world`` with the rich response anchors and ``caps``."""
    config = base.build_world(root)
    symbols = (*base.HVG, "H2", *base.PANEL_CANDIDATES, "X1", "X2")
    sources = {}
    for index, anchor in enumerate((base.K562, base.JURKAT, base.HEPG2)):
        anchor_symbols = symbols if anchor != base.HEPG2 else symbols[1:]  # no H1
        path = root / f"rich_{anchor}.h5ad"
        _write_rich_h5ad(
            path, seed=200 + index, symbols=anchor_symbols, dense=anchor == base.JURKAT
        )
        sources[anchor] = {
            "source_type": "h5ad",
            "h5ad_path": str(path),
            "perturbation_col": "gene",
            "control_label": "non-targeting",
            "var_ensembl_col": "gene_id",
            "target_gene_symbol_col": "gene_name",
        }
    sources[base.HCT116] = write_xatlas(root, seed=300)
    Path(config["paths"]["perturbseq_sources"]).write_text(json.dumps(sources))
    esm2 = (*base.PANEL_CANDIDATES, "GF", *RESOLVED_RESPONSE_GENES)
    np.savez(
        root / "esm2.npz",
        symbols=np.array(esm2),
        vectors=np.random.default_rng(5).normal(size=(len(esm2), 4)).astype(np.float32),
        resolved=np.ones(len(esm2), dtype=bool),
    )
    config["preparation"].update(caps)
    return config


def prepared_outputs(root: Path) -> dict[str, object]:
    """Every file preparation wrote, path-dependent fields removed."""
    root = Path(root)
    manifest = json.loads((root / "prepared_inputs.json").read_text())
    manifest.pop("settings")
    union = json.loads((root / "embedding_union.json").read_text())
    union.pop("esm2_table")
    conditions = pd.read_parquet(root / "response" / "conditions.parquet")
    outputs: dict[str, object] = {
        "manifest": manifest,
        "embedding_union": union,
        "embedding_union.csv": (root / "embedding_union.csv").read_text(),
        "common_gene_panel.csv": (root / "common_gene_panel.csv").read_text(),
        "conditions": {name: conditions[name].tolist() for name in conditions},
        "offsets": np.load(root / "response" / "offsets.npy"),
        "target_cells": np.load(root / "response" / "target_cells.npy"),
    }
    for path in sorted((root / "lines").glob("*.npz")):
        with np.load(path) as payload:
            for key in payload.files:
                outputs[f"lines/{path.stem}/{key}"] = payload[key]
    return outputs


def assert_identical(actual: dict[str, object], expected: dict[str, object]) -> None:
    """Same keys; arrays equal bit for bit (dtype, shape and bytes)."""
    assert sorted(actual) == sorted(expected)
    for key, value in expected.items():
        if isinstance(value, np.ndarray):
            other = actual[key]
            assert isinstance(other, np.ndarray), key
            assert (other.dtype, other.shape) == (value.dtype, value.shape), key
            assert other.tobytes() == value.tobytes(), key
        else:
            assert actual[key] == value, key


# The single-process reference: whole sources in memory, one cell at a time.
# Counts are integers, so every library size and HVG sum is exact whatever the
# summation order.


def naive_h5ad_cells(source: dict, hvg: tuple[str, ...], caps: dict, seed: int):
    """``(labels, library sizes, dense HVG counts)`` of a Perturb-seq h5ad."""
    from src.data.response import (
        cap_indices_by_group,
        group_candidate_indices_by_label,
    )

    adata = ad.read_h5ad(source["h5ad_path"])
    labels = adata.obs[source["perturbation_col"]].astype(str).to_numpy()
    candidate = np.flatnonzero(labels != source["control_label"])
    grouped = group_candidate_indices_by_label(candidate, labels[candidate])
    selected = cap_indices_by_group(grouped, caps["response_max_cells_per_gene"], seed)
    matrix = adata.X[selected]
    counts = np.asarray(
        matrix.toarray() if sparse.issparse(matrix) else matrix, dtype=np.float64
    )
    symbols = adata.var[source["target_gene_symbol_col"]].astype(str).tolist()
    aligned = np.zeros((len(selected), len(hvg)))
    for column, symbol in enumerate(symbols):
        if symbol in hvg:
            aligned[:, hvg.index(symbol)] += counts[:, column]
    return labels[selected], counts.sum(axis=1), aligned


def naive_response_targets(config: dict) -> tuple[float, list, list[np.ndarray]]:
    """``(T, keys, bags)`` of every anchor, before the ESM2 filter."""
    from src.data.expression import log_normalize
    from src.data.tx1_cache import load_hvg_gene_order

    sources = json.loads(Path(config["paths"]["perturbseq_sources"]).read_text())
    caps = config["preparation"]
    seed = caps["response_sampling_seed"]
    controls = []
    for anchor in (base.JURKAT, base.HEPG2):
        adata = ad.read_h5ad(sources[anchor]["h5ad_path"])
        rows = adata.obs["gene"].to_numpy() == "non-targeting"
        controls.append(np.asarray(adata.X[rows].sum(axis=1), dtype=np.float64).ravel())
    target_sum = float(np.median(np.concatenate(controls)))
    hvg = tuple(load_hvg_gene_order(Path(config["paths"]["state_model_dir"])))
    keys, bags = [], []
    for anchor in sorted(sources):
        source = sources[anchor]
        read = (
            naive_xatlas_cells
            if source["source_type"] == "xatlas_orion_parquet"
            else naive_h5ad_cells
        )
        labels, sizes, aligned = read(source, hvg, caps, seed)
        genes = pd.Series(labels).astype(str).str.strip().str.upper().to_numpy()
        for gene in sorted(set(genes)):
            rows = np.flatnonzero(genes == gene)
            keys.append((anchor, str(gene)))
            bags.append(log_normalize(aligned[rows], sizes[rows], target_sum))
    return target_sum, keys, bags


@pytest.fixture(scope="module")
def pooled(tmp_path_factory) -> dict:
    """The rich world prepared on process pools (the default)."""
    config = build_rich_world(tmp_path_factory.mktemp("capped"), CAPPED)
    prepare_inputs(config)
    return config


def test_response_cache_equals_single_process_reference(pooled):
    target_sum, keys, bags = naive_response_targets(pooled)
    root = Path(pooled["prepared_root"])
    manifest = json.loads((root / "prepared_inputs.json").read_text())
    assert manifest["expression_space"]["target_sum"] == target_sum
    outputs = prepared_outputs(root)
    resolved = set(outputs["embedding_union"]["esm2_order"])
    kept = [i for i, (_, gene) in enumerate(keys) if gene in resolved]
    assert len(kept) < len(keys)  # the ESM2 filter drops some conditions
    assert outputs["conditions"] == {
        "model_id": [keys[i][0] for i in kept],
        "gene": [keys[i][1] for i in kept],
        "n_cells": [len(bags[i]) for i in kept],
    }
    expected = np.concatenate([bags[i] for i in kept])
    np.testing.assert_array_equal(
        outputs["offsets"], np.cumsum([0, *(len(bags[i]) for i in kept)])
    )
    assert outputs["target_cells"].dtype == expected.dtype == np.float32
    assert outputs["target_cells"].tobytes() == expected.tobytes()
    assert not [path for path in root.iterdir() if path.name.startswith(".tmp")]


def test_process_pools_write_what_one_process_writes(pooled, tmp_path, monkeypatch):
    monkeypatch.setattr(prepare, "RESPONSE_PROCESSES", 1)
    monkeypatch.setattr(prepare, "LINE_PROCESSES", 1)
    caps = {key: pooled["preparation"][key] for key in CAPPED}
    config = build_rich_world(tmp_path / "serial", caps)
    prepare_inputs(config)
    assert_identical(
        prepared_outputs(Path(config["prepared_root"])),
        prepared_outputs(Path(pooled["prepared_root"])),
    )


def test_interrupted_pool_stops_running_workers():
    import functools
    import time

    from src.experiments.prepare import in_processes

    results = in_processes(
        [functools.partial(int, "1"), functools.partial(time.sleep, 60)], 2
    )
    assert next(results) == 1
    started = time.monotonic()
    results.close()  # what an interrupt does to the consuming loop
    assert time.monotonic() - started < 20
