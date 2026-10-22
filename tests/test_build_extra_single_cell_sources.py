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

from src.data.build.build_extra_single_cell_lines import MAP_COLUMNS
from src.data.build.build_extra_single_cell_sources import (
    Library,
    _identify,
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
        "header": False,
        "gene_id_prefix": "",
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
                            "skiprows": 0,
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


def test_second_source_plan_builds_planned_lines_even_of_the_226(tmp_path):
    build_world(tmp_path)
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps({"lines": {TRAIN: {"primary": "elsewhere", "second": "geo"}}})
    )
    out = tmp_path / "second"
    run(tmp_path / "spec.json", 1, out, second_source=plan)
    assert [p.name for p in out.glob("*.h5ad")] == [f"{TRAIN}.h5ad"]
    line = ad.read_h5ad(out / f"{TRAIN}.h5ad")
    assert list(line.obs_names) == [f"{TRAIN}:1:t{i}" for i in range(4)]
    assert set(line.obs["source"]) == {"geo"}
    # Line A's six cells belong to no planned line (checked before the QC floor);
    # the unplanned source is not read.
    assert _report(out, "geo")["dropped"] == {"line not taken from this source": 6}
    assert not (out / "qc" / "labelled.json").exists()


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


def test_sources_option_builds_only_the_named_sources(tmp_path):
    build_world(tmp_path)
    out = tmp_path / "out"
    run(tmp_path / "spec.json", 1, out, ["labelled"])
    assert [p.name for p in (out / "qc").glob("*.json")] == ["labelled.json"]
    with pytest.raises(ValueError, match="not in the spec"):
        run(tmp_path / "spec.json", 1, out, ["nope"])


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
        {
            "reader": "table",
            "path": "dge.txt.gz",
            "sep": "\t",
            "skiprows": 0,
            "genes_by_cells": True,
        },
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
            "drug": ["DMSO_TF", "DMSO_TF", "DMSO_TF"],
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
            "drug": "DMSO_TF",
        },
    )
    assert library.cells == ["part-00001.parquet:0:b0", "part-00001.parquet:1:b2"]
    assert library.genes == GENES[:2]
    np.testing.assert_array_equal(library.counts.toarray(), [[2, 1], [9, 0]])
    assert list(library.obs["cell_line_id"]) == ["CVCL_1098"] * 2


def test_barcode_identity_reads_line_and_arm_from_cell_ids():
    cells = ["SUM149PT_AAAC", "SUM159PT_GGTA", "AAAC-1_C.R1", "AAAG-2_T.R1", "odd"]
    library = Library(sparse.csr_matrix((5, 1)), cells, ["g"], None, None, [], None)
    lines, arms, report = _identify(
        library,
        {
            "kind": "barcode",
            "pattern": r"^(?P<line>SUM\w+?)_",
            "identifier": None,
            "rename": {},
        },
        Path("."),
        {},
        Path("."),
    )
    assert lines == ["SUM149PT", "SUM159PT", None, None, None]
    assert report == {"matched_cells": 2}
    lines, arms, _ = _identify(
        library,
        {
            "kind": "barcode",
            "pattern": r"-(?P<line>\d+)_(?P<arm>[^_]+)$",
            "identifier": None,
            "rename": {"1": "huH-1"},
        },
        Path("."),
        {},
        Path("."),
    )
    assert lines == [None, None, "huH-1", "2", None]
    assert arms == [None, None, "C.R1", "T.R1", None]
    lines, arms, _ = _identify(
        library,
        {
            "kind": "barcode",
            "pattern": r"_(?P<arm>C\.R\d)$",
            "identifier": "KURAMOCHI",
            "rename": {},
        },
        Path("."),
        {},
        Path("."),
    )
    assert lines == [None, None, "KURAMOCHI", None, None]
    assert arms[2] == "C.R1"


def test_table_and_labels_skip_comment_header_lines(tmp_path):
    header = "####\n## pipeline 1.9\n####\n"
    (tmp_path / "mols.csv").write_text(header + "Cell_Index,G1,G2\n11,3,0\n12,1,4\n")
    (tmp_path / "tags.csv").write_text(
        header + "Cell_Index,Sample_Tag,Sample_Name\n11,ST1,WM793\n12,ST2,Multiplet\n"
    )
    library = read_library(
        tmp_path,
        {
            "reader": "table",
            "path": "mols.csv",
            "sep": ",",
            "skiprows": 3,
            "genes_by_cells": False,
        },
    )
    assert library.cells == ["11", "12"] and library.genes == ["G1", "G2"]
    lines, _, _ = _identify(
        library,
        {
            "kind": "labels",
            "path": "tags.csv",
            "sep": ",",
            "skiprows": 3,
            "cell_column": "Cell_Index",
            "label_column": "Sample_Name",
            "arm_column": None,
            "keep": {},
            "cell_prefix": "",
            "cell_suffix": "",
        },
        tmp_path,
        {},
        tmp_path,
    )
    assert lines == ["WM793", "Multiplet"]


def test_loom_reader_and_gzip_unpack(tmp_path):
    from src.data.build.build_extra_single_cell_sources import _unpack

    rng = np.random.default_rng(2)
    counts = _counts(rng, 5)
    loom = tmp_path / "x.loom"
    with h5py.File(loom, "w") as handle:
        handle["matrix"] = counts.T.astype(np.float32)
        handle.create_group("layers")["spliced"] = counts.T.astype(np.float32)
        rows = handle.create_group("row_attrs")
        rows["Gene"], rows["Accession"] = (
            np.array(SYMBOLS, dtype="S"),
            np.array(GENES, dtype="S"),
        )
        handle.create_group("col_attrs")["CellID"] = np.array(
            [f"s:c{i}x" for i in range(5)], dtype="S"
        )
    with open(loom, "rb") as source, gzip.open(tmp_path / "x.loom.gz", "wb") as out:
        out.write(source.read())
    loom.unlink()
    _unpack(tmp_path / "x.loom.gz", tmp_path)
    spec = {
        "reader": "anndata",
        "path": "x.loom",
        "counts": "layer:spliced",
        "gene_column": "Accession",
        "symbol_column": "index",
    }
    library = read_library(tmp_path, spec)
    np.testing.assert_array_equal(library.counts.toarray(), counts)
    assert library.genes == GENES and library.symbols == SYMBOLS
    assert library.cells == [f"s:c{i}x" for i in range(5)]


def test_label_table_cell_ids_may_be_its_unnamed_first_column(tmp_path):
    pd.DataFrame(
        {"cell_line_demuxlet": ["H1975", "A549"], "demuxlet_cls": ["SNG", "SNG"]},
        index=["AAAC", "GGTA"],
    ).to_csv(tmp_path / "meta.csv")
    library = Library(
        sparse.csr_matrix((2, 1)), ["GGTA", "AAAC"], ["g"], None, None, [], None
    )
    lines, _, _ = _identify(
        library,
        {
            "kind": "labels",
            "path": "meta.csv",
            "sep": ",",
            "skiprows": 0,
            "cell_column": None,
            "label_column": "cell_line_demuxlet",
            "arm_column": None,
            "keep": {"demuxlet_cls": ["SNG"]},
            "cell_prefix": "",
            "cell_suffix": "",
        },
        tmp_path,
        {},
        tmp_path,
    )
    assert lines == ["A549", "H1975"]


def test_mixed_species_mtx_keeps_prefixed_genes_and_hashtags_take_a_suffix(tmp_path):
    rng = np.random.default_rng(3)
    counts = np.hstack([_counts(rng, 3), rng.poisson(3.0, (3, 2))])
    directory = tmp_path / "m"
    directory.mkdir()
    mmwrite(directory / "matrix.mtx", sparse.coo_matrix(counts.T))
    ids = [f"hg19_{g}" for g in GENES] + ["mm10_ENSMUSG1", "mm10_ENSMUSG2"]
    names = [f"hg19_{s}" for s in SYMBOLS] + ["mm10_a", "mm10_b"]
    pd.DataFrame(
        {"gene_ids": ids, "gene": names, "feature_types": "Gene Expression"}
    ).to_csv(directory / "features.tsv", sep="\t", index=False)
    pd.DataFrame({"barcode": ["c0-1", "c1-1", "c2-1"], "call": "hg19"}).to_csv(
        directory / "barcodes.tsv", sep="\t", index=False
    )
    spec = {**_library("m", {}), "header": True, "gene_id_prefix": "hg19_"}
    library = read_library(tmp_path, spec)
    assert library.genes == GENES and library.symbols == SYMBOLS
    np.testing.assert_array_equal(library.counts.toarray(), counts[:, : len(GENES)])
    assert library.cells == ["c0-1", "c1-1", "c2-1"]

    hto = tmp_path / "hto"
    hto.mkdir()
    mmwrite(hto / "matrix.mtx", sparse.coo_matrix(np.array([[40, 1, 0], [0, 30, 2]])))
    (hto / "features.tsv").write_text(
        "H1\tH1\tAntibody Capture\nH2\tH2\tAntibody Capture\n"
    )
    (hto / "barcodes.tsv").write_text("c0\nc1\nc2\n")
    hto_spec = {**_library("hto", {}), "reader": "tenx_mtx"}
    lines, _, report = _identify(
        library,
        {
            "kind": "hashtag",
            "counts": hto_spec,
            "cell_suffix": "-1",
            "hashtags": {"H1": "a", "H2": "b"},
            "min_count": 10,
            "min_ratio": 3,
        },
        tmp_path,
        {},
        tmp_path,
    )
    assert lines == ["a", "b", None]


def test_r_written_label_table_keeps_its_row_names_as_cell_ids(tmp_path):
    (tmp_path / "meta.csv").write_text(
        '"cell_line_demuxlet","demuxlet_cls"\n'
        '"AAAC","H1975","SNG"\n"GGTA","A549","DBL"\n"TTTG","H838","SNG"\n'
    )
    library = Library(
        sparse.csr_matrix((3, 1)), ["TTTG", "AAAC", "GGTA"], ["g"], None, None, [], None
    )
    lines, _, _ = _identify(
        library,
        {
            "kind": "labels",
            "path": "meta.csv",
            "sep": ",",
            "skiprows": 0,
            "cell_column": None,
            "label_column": "cell_line_demuxlet",
            "arm_column": None,
            "keep": {"demuxlet_cls": ["SNG"]},
            "cell_prefix": "",
            "cell_suffix": "",
        },
        tmp_path,
        {},
        tmp_path,
    )
    assert lines == ["H838", "H1975", None]


def test_lognorm_counts_recover_integers_and_refuse_other_values():
    from src.data.build.build_extra_single_cell_sources import lognorm_counts

    rng = np.random.default_rng(4)
    counts = rng.poisson(2.0, size=(6, 30)).astype(np.int64)
    counts[:, 0] += 1  # every cell has a value
    totals = counts.sum(axis=1)
    logged = sparse.csr_matrix(np.log1p(counts * (1e4 / totals[:, None])))
    logged.data = logged.data.astype(np.float32)
    np.testing.assert_array_equal(lognorm_counts(logged, totals).toarray(), counts)
    with pytest.raises(ValueError, match="totals|integers"):
        lognorm_counts(logged, totals + 7)  # normalised over other genes
    corrected = sparse.csr_matrix(np.log1p(counts * 0.37))
    with pytest.raises(ValueError, match="integers"):
        lognorm_counts(corrected, np.round(totals * 0.37))


def test_anndata_reader_inverts_lognormalised_raw_with_obs_totals(tmp_path):
    rng = np.random.default_rng(5)
    counts = _counts(rng, 4)
    totals = counts.sum(axis=1)
    adata = ad.AnnData(
        X=np.zeros((4, 2), dtype=np.float32),
        obs=pd.DataFrame({"nCount_RNA": totals}, index=[f"c{i}" for i in range(4)]),
        var=pd.DataFrame(index=GENES[:2]),
    )
    adata.raw = ad.AnnData(
        X=sparse.csr_matrix(np.log1p(counts * (1e4 / totals[:, None]))),
        obs=adata.obs,
        var=pd.DataFrame({"gene_symbol": SYMBOLS}, index=GENES),
    )
    adata.write_h5ad(tmp_path / "e.h5ad")
    library = read_library(
        tmp_path,
        {
            "reader": "anndata",
            "path": "e.h5ad",
            "counts": "lognorm:raw:nCount_RNA",
            "gene_column": "index",
            "symbol_column": "gene_symbol",
        },
    )
    np.testing.assert_array_equal(library.counts.toarray(), counts)
    assert library.symbols == SYMBOLS


def test_triplets_reader_and_rt_hash_identity_follow_the_authors_maps(tmp_path):
    cells = [
        "01B_A01_RT_BC_1_Lig_BC_7",  # RT well A1: line X, DMSO singlet
        "01B_A01_RT_BC_2_Lig_BC_3",  # RT well A2: line Y, drug singlet
        "01B_A02_RT_BC_1_Lig_BC_9",  # line X, two hashes too close: no arm
        "01B_A02_RT_BC_100_Lig_BC_1",  # RT barcode outside the map: no line
    ]
    with gzip.open(tmp_path / "umi.gz", "wt") as handle:
        handle.write("1\t1\t3\n2\t1\t1\n3\t2\t5\n1\t3\t2\n3\t4\t1\n")
    with gzip.open(tmp_path / "cells.gz", "wt") as handle:
        handle.write("".join(f"{cell}\tsciPlex\n" for cell in cells))
    with gzip.open(tmp_path / "genes.gz", "wt") as handle:
        handle.write("".join(f"{g}.1\t{s}\n" for g, s in zip(GENES[:3], SYMBOLS)))
    (tmp_path / "rt.txt").write_text("RT_BC_1 AAAA\nRT_BC_2 CCCC\nRT_BC_100 GGGG\n")
    (tmp_path / "map.csv").write_text(
        "Well Position,Barcode,Cell Line\nA1,AAAA,X\nA2,CCCC,Y\n"
    )
    with gzip.open(tmp_path / "hash.gz", "wt") as handle:
        handle.write(
            f"sciPlex\t{cells[0]}\tDMSO_0nM_10H01\t1\t40\n"
            f"sciPlex\t{cells[0]}\tDrug_5nM_10A01\t1\t2\n"
            f"sciPlex\t{cells[1]}\tDrug_5nM_10A01\t1\t30\n"
            f"sciPlex\t{cells[2]}\tDMSO_0nM_10H01\t1\t25\n"
            f"sciPlex\t{cells[2]}\tDrug_5nM_10A01\t1\t20\n"
        )
    library = read_library(
        tmp_path,
        {
            "reader": "triplets",
            "matrix": "umi.gz",
            "cell_table": "cells.gz",
            "gene_table": "genes.gz",
        },
    )
    assert library.cells == cells and library.genes == [f"{g}.1" for g in GENES[:3]]
    np.testing.assert_array_equal(
        library.counts.toarray(), [[3, 1, 0], [0, 0, 5], [2, 0, 0], [0, 0, 1]]
    )
    lines, arms, report = _identify(
        library,
        {
            "kind": "rt_hash",
            "rt_barcodes": "rt.txt",
            "rt_map": "map.csv",
            "hash_table": "hash.gz",
            "min_count": 20,
            "min_ratio": 3,
        },
        tmp_path,
        {},
        tmp_path,
    )
    assert lines == ["X", "Y", "X", None]
    assert arms == ["DMSO", "Drug", None, None]
    assert report["hashtags"] == {"singlet": 2, "doublet": 1, "negative": 1}


def test_mtx_names_reader_takes_names_from_r_vectors(tmp_path):
    rng = np.random.default_rng(6)
    counts = _counts(rng, 3)
    mmwrite(tmp_path / "m.mtx", sparse.coo_matrix(counts.T))
    pd.Series(SYMBOLS, index=range(1, len(SYMBOLS) + 1)).to_csv(tmp_path / "g.csv")
    (tmp_path / "c.csv").write_text('"","x"\n"1","A431S"\n"2","A431R"\n"3","A431S"\n')
    library = read_library(
        tmp_path,
        {
            "reader": "mtx_names",
            "matrix": "m.mtx",
            "gene_table": "g.csv",
            "cell_table": "c.csv",
        },
    )
    assert library.cells == ["1", "2", "3"] and library.genes == SYMBOLS
    np.testing.assert_array_equal(library.counts.toarray(), counts)
    lines, _, _ = _identify(
        library,
        {
            "kind": "labels",
            "path": "c.csv",
            "sep": ",",
            "skiprows": 0,
            "cell_column": None,
            "label_column": "x",
            "arm_column": None,
            "keep": {},
            "cell_prefix": "",
            "cell_suffix": "",
        },
        tmp_path,
        {},
        tmp_path,
    )
    assert lines == ["A431S", "A431R", "A431S"]


def test_tenx_bam_reader_counts_one_umi_per_representative_read(tmp_path):
    pysam = pytest.importorskip("pysam")
    header = {"HD": {"VN": "1.6"}, "SQ": [{"SN": "chr1", "LN": 10000}]}
    reads = [  # (CB, UB, GX, GN, xf)
        ("AAAC-1", "U1", "ENSG1", "G1", 25),
        ("AAAC-1", "U1", "ENSG1", "G1", 17),  # duplicate read of that molecule
        ("AAAC-1", "U2", "ENSG1", "G1", 25),
        ("AAAC-1", "U3", "ENSG2", "G2", 25),
        ("GGTA-1", "U1", "ENSG2", "G2", 25),
        ("GGTA-1", "U4", "ENSG1;ENSG2", "G1;G2", 25),  # multi-gene: not counted
        ("TTTG-1", "U5", "ENSG1", "G1", 1),  # low-support molecule
    ]
    with pysam.AlignmentFile(tmp_path / "x.bam", "wb", header=header) as handle:
        for index, (cb, ub, gx, gn, xf) in enumerate(reads):
            read = pysam.AlignedSegment(handle.header)
            read.query_name = f"r{index}"
            read.reference_id, read.reference_start = 0, 100 + index
            read.cigarstring, read.query_sequence = "4M", "ACGT"
            read.set_tags([("CB", cb), ("UB", ub), ("GX", gx), ("GN", gn), ("xf", xf)])
            handle.write(read)
    library = read_library(
        tmp_path, {"reader": "tenx_bam", "bam": "x.bam", "cell_prefix": "E1_"}
    )
    assert library.cells == ["E1_AAAC-1", "E1_GGTA-1"]
    assert library.genes == ["ENSG1", "ENSG2"] and library.symbols == ["G1", "G2"]
    np.testing.assert_array_equal(library.counts.toarray(), [[2, 1], [0, 1]])


def test_downloads_list_the_recount_reference_once(tmp_path):
    from src.data.build.build_extra_single_cell_sources import downloads

    spec = {
        "raw_root": str(tmp_path / "raw"),
        "recount": {
            "index_dir": str(tmp_path / "ref"),
            "files": {"genome.fa.gz": "https://example/genome"},
        },
        "sources": {
            name: {
                "wave": 2,
                "files": {f"{name}.sra": f"https://example/{name}"},
                "recount": {},
            }
            for name in ("one", "two")
        },
    }
    (tmp_path / "spec.json").write_text(json.dumps(spec))
    listed = downloads(tmp_path / "spec.json", 2)
    assert [url for url, _ in listed] == [
        "https://example/one",
        "https://example/genome",
        "https://example/two",
    ]
