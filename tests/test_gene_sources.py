import json
import pickle

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.gene_sources import (
    SourcePaths,
    SourceTable,
    align,
    align_sources,
    coverage,
    main,
    read_gene2vec,
    read_pie_dense,
)


def write_pie_dense(path, name, keys, vectors, dtype):
    path.mkdir(parents=True)
    vectors = np.asarray(vectors, dtype=dtype)
    meta = {
        "format_version": 1,
        "name": name,
        "layout": "dense",
        "index": "pert",
        "keys": list(keys),
        "dim": int(vectors.shape[1]),
        "dtype": dtype,
        "provenance": {},
    }
    (path / "meta.json").write_text(json.dumps(meta))
    np.save(path / "embeddings.npy", vectors)


@pytest.fixture
def sources(tmp_path):
    pie = tmp_path / "pie_sources"
    write_pie_dense(
        pie / "gene_text",
        "gene_text",
        ["TP53", "C1orf112", "KRAS"],
        [[1, 2], [3, 4], [5, 6]],
        "float32",
    )
    write_pie_dense(
        pie / "string_space",
        "string_space",
        ["kras", "MYC"],
        [[7, 8, 9], [1, 1, 1]],
        "float16",
    )
    protein = tmp_path / "protein"
    protein.mkdir()
    for name, symbols in (("esmc_600m", ["TP53", "KRAS"]), ("esm3_open", ["TP53"])):
        np.savez(
            protein / f"{name}.npz",
            symbols=np.asarray(symbols),
            accessions=np.asarray([f"P{i}" for i in range(len(symbols))]),
            vectors=np.arange(4 * len(symbols), dtype=np.float32).reshape(
                len(symbols), 4
            ),
        )
    gene2vec = tmp_path / "gene2vec_embs.pt"
    torch.save(
        {
            "ENSG1": torch.tensor([1.0, 2.0], dtype=torch.float64),
            "ENSG2": torch.tensor([3.0, 4.0], dtype=torch.float64),
            "MYC": torch.tensor(
                [9.0, 9.0], dtype=torch.float64
            ),  # a symbol key: never reached
        },
        gene2vec,
    )
    names = tmp_path / "gene_name_id_dict_gc30M.pkl"
    with open(names, "wb") as handle:
        pickle.dump({"TP53": "ENSG1", "c1orf112": "ENSG2", "MYC": "ENSG9"}, handle)
    return SourcePaths(pie, protein, gene2vec, names)


AXIS = ["TP53", "C1ORF112", "KRAS", "MYC"]


def test_alignment_is_case_insensitive_and_masks_genes_without_a_key(sources):
    aligned = align_sources(AXIS, sources)
    text = aligned["gene_text"]
    assert text.present.tolist() == [True, True, True, False]
    np.testing.assert_array_equal(text.vectors[:3], [[1, 2], [3, 4], [5, 6]])
    assert np.isnan(text.vectors[3]).all()
    string = aligned["string_space"]
    assert string.vectors.dtype == np.float32
    assert string.present.tolist() == [False, False, True, True]
    np.testing.assert_array_equal(string.vectors[2], [7, 8, 9])
    assert aligned["esm3_open"].present.tolist() == [True, False, False, False]
    # Gene2vec goes symbol -> Ensembl -> vector; MYC's Ensembl ID has none.
    g2v = aligned["gene2vec"]
    assert g2v.present.tolist() == [True, True, False, False]
    np.testing.assert_array_equal(g2v.vectors[1], [3.0, 4.0])


def test_keys_equal_up_to_case_are_refused():
    table = SourceTable("gene_text", ("TP53", "tp53"), np.zeros((2, 2), np.float32))
    with pytest.raises(ValueError, match="equal up to case"):
        align(table, ["TP53"])


def test_the_axis_must_be_unique_upper_case():
    table = SourceTable("gene_text", ("TP53",), np.zeros((1, 2), np.float32))
    with pytest.raises(ValueError, match="upper-case"):
        align(table, ["tp53"])
    with pytest.raises(ValueError, match="upper-case"):
        align(table, ["TP53", "TP53"])


def test_pie_dir_with_a_wrong_dtype_is_refused(tmp_path):
    write_pie_dense(tmp_path / "s", "string_space", ["A"], [[1.0]], "float16")
    meta = json.loads((tmp_path / "s" / "meta.json").read_text())
    meta["dtype"] = "float32"
    (tmp_path / "s" / "meta.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="against meta"):
        read_pie_dense(tmp_path / "s")


def test_gene2vec_reads_only_symbols_with_a_vector(sources):
    table = read_gene2vec(sources.gene2vec, sources.gene_names)
    assert table.keys == ("TP53", "c1orf112")
    assert table.vectors.dtype == np.float32


def test_coverage_counts_axis_and_selective_shares(sources):
    record = coverage(
        align_sources(AXIS, sources, ["gene_text"])["gene_text"], ["KRAS", "MYC"]
    )
    assert record == {
        "dim": 2,
        "covered": 3,
        "share_of_axis": 0.75,
        "selective_covered": 1,
        "share_of_selective": 0.5,
    }
    with pytest.raises(ValueError, match="off the axis"):
        coverage(align_sources(AXIS, sources, ["gene_text"])["gene_text"], ["NOPE"])


def test_cli_writes_each_source_and_coverage(sources, tmp_path):
    genes = pd.DataFrame(
        {"selective": [True, False, True, False]}, index=pd.Index(AXIS, name="gene")
    )
    genes.to_parquet(tmp_path / "genes.parquet")
    out = tmp_path / "out"
    argv = ["--genes", str(tmp_path / "genes.parquet"), "--out", str(out)]
    argv += [
        "--pie-sources",
        str(sources.pie_sources),
        "--protein",
        str(sources.protein),
    ]
    argv += [
        "--gene2vec",
        str(sources.gene2vec),
        "--gene-names",
        str(sources.gene_names),
    ]
    assert main(argv) == 0
    record = json.loads((out / "coverage.json").read_text())
    assert record["axis"] == 4 and record["selective"] == 2
    assert record["sources"]["esmc_600m"]["covered"] == 2
    assert record["sources"]["esmc_600m"]["share_of_selective"] == 1.0
    assert len(record["sources"]["gene2vec"]["inputs"]) == 2
    with np.load(out / "string_space.npz") as payload:
        assert payload["axis"].tolist() == AXIS
        assert payload["present"].tolist() == [False, False, True, True]
        assert payload["vectors"].shape == (4, 3)
