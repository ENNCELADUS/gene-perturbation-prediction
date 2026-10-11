"""Extra single-cell sources: readers, cell calls, line naming through the map,
basal arms, the QC floor, never a 226 line, resumable per source."""

from __future__ import annotations

import gzip
import json
import pickle
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from scipy.io import mmwrite

from src.data.prepare.build_extra_single_cell_lines import MAP_COLUMNS
from src.data.prepare.build_extra_single_cell_sources import (
    assign_pool,
    call_hashtags,
    knee,
    read_library,
    run,
)

GENES = [f"ENSG{i:011d}" for i in range(1, 41)]
SYMBOLS = [f"G{i}" for i in range(1, 40)] + ["MT-CO1"]
TRAIN, VAL = "ACH-000001", "ACH-000002"
EXTRA_A, EXTRA_B, EXTRA_U = "ACH-000101", "ACH-000102", "ACH-000103"


def _counts(rng, n_cells, *, mito=0):
    counts = rng.poisson(3.0, size=(n_cells, len(GENES))).astype(np.int64)
    counts[:, -1] = mito
    return counts


def _write_mtx(directory: Path, counts: np.ndarray, cells: list[str]) -> None:
    directory.mkdir(parents=True)
    mmwrite(directory / "matrix.mtx", sparse.coo_matrix(counts.T))
    pd.DataFrame({"id": GENES, "name": SYMBOLS, "type": "Gene Expression"}).to_csv(
        directory / "features.tsv", sep="\t", header=False, index=False
    )
    (directory / "barcodes.tsv").write_text("\n".join(cells) + "\n")


def _library(path: str, identity: dict, *, arm="untreated", cells="deposited"):
    return {
        "reader": "tenx_mtx",
        "matrix": f"{path}/matrix.mtx",
        "features": f"{path}/features.tsv",
        "barcodes": f"{path}/barcodes.tsv",
        "cells": cells,
        "knee_min_umi": 10,
        "identity": identity,
        "arm": arm,
    }


def _map_rows(*rows):
    base = dict.fromkeys(MAP_COLUMNS, "")
    base.update(condition_class="basal_2d", raw_counts="true", wave="1", review="ok")
    return [{**base, **row} for row in rows]


def build_world(root: Path) -> dict:
    """Two sources of synthetic deposits, the map and the 26Q1 files."""
    rng = np.random.default_rng(0)
    pd.DataFrame(
        {"ensembl_id": GENES, "gene_symbol": SYMBOLS, "token_id": range(3, 43)}
    ).to_parquet(root / "gene_metadata.parquet", index=False)
    (root / "vocab.json").write_text(
        json.dumps({g: i + 3 for i, g in enumerate(GENES)})
    )
    (root / "state").mkdir()
    (root / "state" / "var_dims.pkl").write_bytes(pickle.dumps({"gene_names": SYMBOLS}))
    ids = [TRAIN, VAL, EXTRA_A, EXTRA_B, EXTRA_U]
    pd.DataFrame(
        {
            "ModelID": ids,
            "PatientID": ["P1", "P2", "P3", "P4", "P5"],
            "OncotreeLineage": "Lung",
        }
    ).to_csv(root / "Model.csv", index=False)
    pd.DataFrame({"G1 (1)": [0.1] * 4}, index=[TRAIN, VAL, EXTRA_A, EXTRA_B]).to_csv(
        root / "gene_effect.csv"
    )
    (root / "split.json").write_text(
        json.dumps({"train": [TRAIN], "val": [VAL], "test": [], "unlabeled_train": []})
    )

    raw = root / "raw"
    # geo: one line per library; the second library is a 226 line.
    a_counts = _counts(rng, 6)
    a_counts[0] = 0
    a_counts[0, :3] = 50  # three genes only: below the floor
    a_counts[1, -1] = 500  # mostly mitochondrial: above the floor
    _write_mtx(raw / "geo" / "a", a_counts, [f"a{i}" for i in range(6)])
    _write_mtx(raw / "geo" / "t", _counts(rng, 4), [f"t{i}" for i in range(4)])
    # labelled: an author table with arms; extra B in DMSO and drug arms.
    _write_mtx(raw / "labelled" / "x", _counts(rng, 6), [f"x{i}" for i in range(6)])
    pd.DataFrame(
        {
            "cell": [f"x{i}" for i in range(6)],
            "line": ["lineB"] * 4 + ["lineU", "lineB"],
            "arm": ["DMSO", "DMSO", "drug", "DMSO", "DMSO", "DMSO"],
            "quality": ["normal"] * 5 + ["low"],
        }
    ).to_csv(raw / "labelled" / "cells.csv", index=False)
    pd.DataFrame(
        _map_rows(
            {"source": "geo", "identifier": "Line-A", "model_id": EXTRA_A},
            {"source": "geo", "identifier": "Line-T", "model_id": TRAIN},
            {"source": "labelled", "identifier": "lineB", "model_id": EXTRA_B},
            {"source": "labelled", "identifier": "lineU", "model_id": EXTRA_U},
        ),
        columns=list(MAP_COLUMNS),
    ).to_csv(root / "map.csv", index=False)
    spec = {
        "candidates_path": str(root / "map.csv"),
        "model_path": str(root / "Model.csv"),
        "gene_effect_path": str(root / "gene_effect.csv"),
        "bulk_expression_path": str(root / "bulk.csv"),
        "split_path": str(root / "split.json"),
        "gene_metadata_path": str(root / "gene_metadata.parquet"),
        "gene_metadata_ensembl_column": "ensembl_id",
        "gene_metadata_symbol_column": "gene_symbol",
        "gene_metadata_token_column": "token_id",
        "minimum_token_id": 3,
        "tx1_vocab_path": str(root / "vocab.json"),
        "tx1_model_dir": str(root),
        "state_model_dir": str(root / "state"),
        "run_tx1_loader_probe": False,
        "tx1_loader_probe_max_length": 64,
        "raw_root": str(raw),
        "min_genes": 5,
        "max_mito_fraction": 0.25,
        "sources": {
            "geo": {
                "wave": 1,
                "files": {"a/matrix.mtx": "https://example/a"},
                "unpack": [],
                "gene_id_kind": "ensembl",
                "arms": ["untreated"],
                "libraries": [
                    _library("a", {"kind": "line", "identifier": "Line-A"}),
                    _library("t", {"kind": "line", "identifier": "Line-T"}),
                ],
            },
            "labelled": {
                "wave": 1,
                "files": {"cells.csv": "https://example/cells"},
                "unpack": [],
                "gene_id_kind": "ensembl",
                "arms": ["DMSO"],
                "libraries": [
                    _library(
                        "x",
                        {
                            "kind": "labels",
                            "path": "cells.csv",
                            "sep": ",",
                            "cell_column": "cell",
                            "label_column": "line",
                            "arm_column": "arm",
                            "keep": {"quality": ["normal"]},
                            "cell_prefix": "",
                            "cell_suffix": "",
                        },
                        arm=None,
                    )
                ],
            },
        },
    }
    (root / "spec.json").write_text(json.dumps(spec))
    return spec


def _report(out: Path, source: str) -> dict:
    return json.loads((out / "qc" / f"{source}.json").read_text())


def test_sources_write_basal_cells_of_their_chosen_lines_only(tmp_path):
    build_world(tmp_path)
    out = tmp_path / "out"
    run(tmp_path / "spec.json", 1, out)
    assert sorted(p.name for p in out.glob("*.h5ad")) == [
        f"{EXTRA_A}.h5ad",
        f"{EXTRA_B}.h5ad",
    ]
    line_a = ad.read_h5ad(out / f"{EXTRA_A}.h5ad")
    assert list(line_a.obs_names) == [f"{EXTRA_A}:0:a{i}" for i in range(2, 6)]
    assert set(line_a.obs["model_id"]) == {EXTRA_A}
    assert list(line_a.var.index) == GENES
    geo = _report(out, "geo")
    assert geo["dropped"] == {"QC floor": 2, "line of the 226": 4}
    assert geo["lines"][EXTRA_A]["cells"] == 4
    assert geo["libraries"][0]["mito_genes"] == 1

    line_b = ad.read_h5ad(out / f"{EXTRA_B}.h5ad")
    assert list(line_b.obs["arm"]) == ["DMSO"] * 3
    labelled = _report(out, "labelled")
    assert labelled["dropped"] == {
        "arm": 1,
        "no line": 1,
        "line not taken from this source": 1,
    }
    raw = read_library(tmp_path / "raw" / "labelled", _library("x", {}))
    np.testing.assert_array_equal(line_b.X.toarray(), raw.counts[[0, 1, 3]].toarray())


def test_sources_resume_and_refuse_interrupted_output(tmp_path):
    build_world(tmp_path)
    out = tmp_path / "out"
    run(tmp_path / "spec.json", 1, out)
    before = (out / "qc" / "geo.json").read_bytes()
    run(tmp_path / "spec.json", 1, out)
    assert (out / "qc" / "geo.json").read_bytes() == before
    (out / "qc" / "geo.json").unlink()
    with pytest.raises(FileExistsError, match="interrupted"):
        run(tmp_path / "spec.json", 1, out)


def test_a_label_the_map_lacks_raises(tmp_path):
    spec = build_world(tmp_path)
    spec["sources"]["geo"]["libraries"][0]["identity"]["identifier"] = "Unmapped"
    (tmp_path / "spec.json").write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="absent from the map"):
        run(tmp_path / "spec.json", 1, tmp_path / "out")


def test_readers_agree_on_tenx_h5_table_and_anndata(tmp_path):
    rng = np.random.default_rng(1)
    counts = _counts(rng, 3)
    cells = ["c0", "c1", "c2"]
    with h5py.File(tmp_path / "f.h5", "w") as handle:
        group = handle.create_group("matrix")
        matrix = sparse.csc_matrix(
            np.vstack([counts.T, np.array([[9, 0, 4]])])
        )  # genes plus one hashtag
        group["data"], group["indices"], group["indptr"] = (
            matrix.data,
            matrix.indices,
            matrix.indptr,
        )
        group["shape"] = np.array(matrix.shape)
        group["barcodes"] = np.array(cells, dtype="S")
        features = group.create_group("features")
        features["id"] = np.array([*GENES, "HTO1"], dtype="S")
        features["name"] = np.array([*SYMBOLS, "HTO1"], dtype="S")
        features["feature_type"] = np.array(
            ["Gene Expression"] * len(GENES) + ["Antibody Capture"], dtype="S"
        )
    h5 = read_library(tmp_path, {"reader": "tenx_h5", "h5": "f.h5"})
    np.testing.assert_array_equal(h5.counts.toarray(), counts)
    assert h5.tag_names == ["HTO1"] and h5.tags.toarray().ravel().tolist() == [9, 0, 4]

    with gzip.open(tmp_path / "dge.txt.gz", "wt") as handle:
        pd.DataFrame(counts.T, index=SYMBOLS, columns=cells).to_csv(handle, sep="\t")
    table = read_library(
        tmp_path,
        {"reader": "table", "path": "dge.txt.gz", "sep": "\t", "genes_by_cells": True},
    )
    np.testing.assert_array_equal(table.counts.toarray(), counts)
    assert table.genes == SYMBOLS and table.cells == cells

    adata = ad.AnnData(
        X=np.log1p(counts.astype(float)),
        obs=pd.DataFrame(index=cells),
        var=pd.DataFrame({"symbol": SYMBOLS}, index=GENES),
    )
    adata.raw = ad.AnnData(
        X=sparse.csr_matrix(counts), var=pd.DataFrame({"symbol": SYMBOLS}, index=GENES)
    )
    adata.write_h5ad(tmp_path / "p.h5ad")
    spec = {
        "reader": "anndata",
        "path": "p.h5ad",
        "gene_column": "index",
        "symbol_column": "symbol",
    }
    from_raw = read_library(tmp_path, {**spec, "counts": "raw"})
    np.testing.assert_array_equal(from_raw.counts.toarray(), counts)
    with pytest.raises(ValueError, match="integer"):
        read_library(tmp_path, {**spec, "counts": "X"})


def test_hashtag_calls_singlets_doublets_and_negatives():
    counts = np.array([[50, 2], [40, 35], [3, 1], [0, 80]])
    calls, tally = call_hashtags(
        counts, ["H1", "H2"], {"H1": "a", "H2": "b"}, min_count=10, min_ratio=3
    )
    assert calls == ["a", None, None, "b"]
    assert tally == {"singlet": 2, "doublet": 1, "negative": 1}


def test_knee_separates_cells_from_empty_droplets():
    rng = np.random.default_rng(0)
    totals = np.concatenate([rng.integers(2000, 5000, 300), rng.integers(10, 60, 5000)])
    cut = knee(totals, 10)
    assert cut == totals[:300].min()
    assert (totals >= cut).sum() == 300


def test_pool_assigns_clusters_by_bulk_and_withholds_close_calls():
    rng = np.random.default_rng(0)
    symbols = [f"S{i}" for i in range(60)]
    first, second = rng.gamma(2.0, 2.0, 60), rng.gamma(2.0, 2.0, 60)
    bulk = pd.DataFrame(
        np.log1p(np.vstack([first, second]) * 10),
        index=["M1", "M2"],
        columns=symbols,
    )
    counts = sparse.csr_matrix(
        np.vstack(
            [rng.poisson(first * 20, (40, 60)), rng.poisson(second * 20, (30, 60))]
        )
    )
    calls, table = assign_pool(
        counts, symbols, {"one": "M1", "two": "M2"}, bulk, clusters=2, min_margin=0.2
    )
    assert calls == ["one"] * 40 + ["two"] * 30
    assert all(row["margin"] >= 0.2 for row in table)
    calls, _ = assign_pool(
        counts, symbols, {"one": "M1", "two": "M2"}, bulk, clusters=2, min_margin=5.0
    )
    assert calls == [None] * 70


def test_tahoe_reader_keeps_the_listed_lines_cells(tmp_path):
    shards = tmp_path / "shards"
    shards.mkdir()
    pd.DataFrame(
        {
            "genes": [np.array([3, 4]), np.array([4, 5]), np.array([3])],
            "expressions": [np.array([2, 1]), np.array([7, 1]), np.array([9])],
            "BARCODE_SUB_LIB_ID": ["b0", "b1", "b2"],
            "cell_line_id": ["CVCL_1098", "CVCL_0001", "CVCL_1098"],
        }
    ).to_parquet(shards / "part-00001.parquet")
    pd.DataFrame(
        {"token_id": [3, 4, 5], "ensembl_id": GENES[:3], "gene_symbol": SYMBOLS[:3]}
    ).to_parquet(tmp_path / "meta.parquet")
    library = read_library(
        tmp_path,
        {
            "reader": "tahoe",
            "shards": str(shards),
            "gene_metadata": str(tmp_path / "meta.parquet"),
            "cell_lines": ["CVCL_1098"],
        },
    )
    assert library.cells == ["part-00001.parquet:0:b0", "part-00001.parquet:2:b2"]
    assert library.genes == GENES[:2]
    np.testing.assert_array_equal(library.counts.toarray(), [[2, 1], [9, 0]])
    assert list(library.obs["cell_line_id"]) == ["CVCL_1098"] * 2
