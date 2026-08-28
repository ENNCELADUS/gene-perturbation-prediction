"""Strict Exp13 GeneEffect data-contract tests."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from aivc_model.gene_splits import sha256_file
from aivc_model.geneeffect_data import (
    Exp13Split,
    QScFeatures,
    build_g_var,
    build_q_sc_shards,
    build_residual_data,
    build_scored_universe,
    compute_q_sc,
    load_exp13_split,
    load_geneeffect_long,
    load_source_registry,
    parse_gene_symbol,
    verify_q_sc_shards,
)


def _split() -> Exp13Split:
    return Exp13Split(
        train=("T1", "T2", "T3", "T4", "T5", "ACH-000779", "ACH-001086"),
        val=("V1", "V2", "V3"),
        test=("E1", "E2", "E3"),
        unlabeled_train=("ACH-000779", "ACH-001086"),
    )


def _labels() -> pd.DataFrame:
    rows = []
    values = {
        "A": [1, 2, 3, 4, 5],
        "B": [0, 0, 0, 0, 0],
        "TIE": [-2, -1, 0, 1, 2],
        "LOW": [1, 1, 1, 1, 1],
    }
    for gene, train_values in values.items():
        rows.extend(
            {"model_id": f"T{i}", "gene_symbol": gene, "gene_effect": value}
            for i, value in enumerate(train_values, 1)
        )
        rows.extend(
            {"model_id": model_id, "gene_symbol": gene, "gene_effect": float(i)}
            for i, model_id in enumerate(("V1", "V2", "V3"), 1)
        )
        rows.extend(
            {"model_id": model_id, "gene_symbol": gene, "gene_effect": float(i)}
            for i, model_id in enumerate(("E1", "E2", "E3"), 1)
        )
    rows.append({"model_id": "T1", "gene_symbol": "SPARSE", "gene_effect": 1.0})
    return pd.DataFrame(rows)


def test_load_tracked_split_and_reject_membership_change(tmp_path: Path) -> None:
    tracked = Path("configs/benchmarks/cell_line_geneeffect_226_split.json")
    split = load_exp13_split(tracked)
    assert len(split.all_model_ids) == 226
    assert len(split.supervised_train) == 170

    payload = json.loads(tracked.read_text())
    payload["train"][0], payload["test"][0] = (
        payload["test"][0],
        payload["train"][0],
    )
    path = tmp_path / "split.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="SHA-256"):
        load_exp13_split(path)


def test_geneeffect_symbol_parsing_and_duplicate_rejection(tmp_path: Path) -> None:
    assert parse_gene_symbol("TP53 (7157)") == "TP53"
    assert parse_gene_symbol("C10orf105 (118812)") == "C10ORF105"
    for malformed in ("TP53", "TP53 (legacy) (7157)", "TP53  (7157)"):
        with pytest.raises(ValueError, match="invalid"):
            parse_gene_symbol(malformed)

    split = _split()
    frame = pd.DataFrame(
        [[1.0, 2.0]] * 11,
        index=[*split.supervised_train, *split.val, *split.test],
        columns=["TP53 (7157)", "tp53 (9999)"],
    )
    path = tmp_path / "geneeffect.csv"
    frame.to_csv(path)
    with pytest.raises(ValueError, match="duplicate symbols"):
        load_geneeffect_long(path, split)


def test_geneeffect_long_uses_model_id_and_enforces_only_declared_missing(
    tmp_path: Path,
) -> None:
    split = _split()
    frame = pd.DataFrame(
        {"TP53 (7157)": np.arange(11, dtype=float)},
        index=[*split.supervised_train, *split.val, *split.test],
    )
    path = tmp_path / "geneeffect.csv"
    frame.to_csv(path)
    long = load_geneeffect_long(path, split)
    assert list(long.columns) == ["model_id", "gene_symbol", "gene_effect"]
    assert set(long["model_id"]) == set(split.all_model_ids) - set(
        split.unlabeled_train
    )

    frame = frame.drop(index="V1")
    frame.to_csv(path)
    with pytest.raises(ValueError, match="unlabeled_train"):
        load_geneeffect_long(path, split)


def test_geneeffect_long_rejects_nonnumeric_and_infinite_values(tmp_path: Path) -> None:
    split = _split()
    ids = [*split.supervised_train, *split.val, *split.test]
    frame = pd.DataFrame({"TP53 (7157)": np.arange(11, dtype=object)}, index=ids)
    path = tmp_path / "geneeffect.csv"
    frame.loc["T1", "TP53 (7157)"] = "corrupt"
    frame.to_csv(path)
    with pytest.raises(ValueError, match="nonnumeric"):
        load_geneeffect_long(path, split)
    frame.loc["T1", "TP53 (7157)"] = np.inf
    frame.to_csv(path)
    with pytest.raises(ValueError, match="infinite"):
        load_geneeffect_long(path, split)


def test_scored_universe_intersection_preserves_esm2_order_and_drop_reasons() -> None:
    universe = build_scored_universe(_labels(), _split(), ["B", "SPARSE", "A"])
    assert universe.symbols == ("B", "A")
    assert universe.manifest["scored_symbols"] == ["B", "A"]
    reasons = universe.coverage.set_index("gene_symbol")["drop_reason"]
    assert "train_finite_lt5" in reasons["SPARSE"]
    assert reasons["TIE"] == "esm2_unresolved"


def test_scored_universe_rejects_duplicate_gene_line_rows() -> None:
    labels = pd.concat([_labels(), _labels().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        build_scored_universe(labels, _split(), ["A"])


def test_residual_means_are_train_only_and_exclude_unlabeled_train() -> None:
    labels = _labels()
    universe = build_scored_universe(labels, _split(), ["A", "B"])
    residual = build_residual_data(labels, _split(), universe)
    assert residual.targets.gene_mean["A"] == 3.0
    held_out = residual.targets.long.query("model_id == 'V1' and gene_symbol == 'A'")
    assert held_out.iloc[0]["residual"] == -2.0
    assert residual.manifest["fit_line_count"] == 5
    assert residual.manifest["excluded_unlabeled_train"] == [
        "ACH-000779",
        "ACH-001086",
    ]


def test_g_var_uses_population_variance_percentile_and_includes_ties() -> None:
    labels = _labels()
    universe = build_scored_universe(labels, _split(), ["LOW", "A", "TIE", "B"])
    residual = build_residual_data(labels, _split(), universe)
    result = build_g_var(residual, _split(), universe)
    # A and TIE both have population variance 2; B and LOW have zero.
    assert result.threshold == 2.0
    assert result.symbols == ("A", "TIE")
    assert result.manifest["ties_included"] is True


def test_source_registry_requires_exact_membership_and_raw_umi(tmp_path: Path) -> None:
    split = _split()
    rows = [
        {
            "model_id": model_id,
            "source_path": f"/{model_id}.h5ad",
            "source_kind": "h5ad",
            "matrix_semantics": "raw_umi_counts",
        }
        for model_id in split.all_model_ids
    ]
    path = tmp_path / "registry.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    registry = load_source_registry(path, split)
    assert list(registry.index) == list(split.all_model_ids)

    rows[0]["matrix_semantics"] = "processed_cpm"
    pd.DataFrame(rows).to_csv(path, index=False)
    with pytest.raises(ValueError, match="non-raw-UMI"):
        load_source_registry(path, split)


def test_q_sc_raw_counts_and_explicit_unavailable_mask() -> None:
    adata = SimpleNamespace(
        X=sparse.csr_matrix([[0, 1], [2, 1], [4, 1]]),
        var=pd.DataFrame({"gene_symbol": ["A", "B"]}),
    )
    result = compute_q_sc(adata, ["B", "MISSING", "A"])
    assert isinstance(result, QScFeatures)
    np.testing.assert_allclose(result.values[0], [1.0, 1.0, 0.0])
    np.testing.assert_allclose(result.values[2], [2.0, 2 / 3, 8 / 3])
    assert result.available.tolist() == [True, False, True]
    assert np.isnan(result.values[1]).all()

    adata.X = np.array([[0.5, 1.0]])
    with pytest.raises(ValueError, match="raw UMI"):
        compute_q_sc(adata, ["A"])


def test_q_sc_shards_resume_hash_and_unrestricted_verification(tmp_path: Path) -> None:
    source_a = tmp_path / "A.h5ad"
    source_b = tmp_path / "B.h5ad"
    source_a.write_bytes(b"source-a")
    source_b.write_bytes(b"source-b")
    registry = pd.DataFrame(
        {
            "source_path": [str(source_a), str(source_b)],
            "source_kind": ["h5ad", "h5ad"],
            "matrix_semantics": ["raw_umi_counts", "raw_umi_counts"],
        },
        index=pd.Index(["A", "B"], name="model_id"),
    )
    adata = SimpleNamespace(
        X=np.array([[0, 1], [2, 3]], dtype=int),
        var=pd.DataFrame({"gene_symbol": ["G1", "G2"]}),
    )
    calls = []

    def reader(path: Path) -> SimpleNamespace:
        calls.append(path)
        adata.obs = pd.DataFrame({"model_id": [path.stem, path.stem]})
        return adata

    output = tmp_path / "q_sc"
    manifest = build_q_sc_shards(registry, output, ["G2", "MISSING"], reader=reader)
    assert manifest["line_count"] == 2
    assert len(calls) == 2
    verification = verify_q_sc_shards(registry, output, ["G2", "MISSING"])
    assert verification["status"] == "passed"
    assert verification["manifest_sha256"] == sha256_file(output / "manifest.json")
    assert set(verification["shard_sha256"]) == {"A", "B"}
    with np.load(output / "A.npz", allow_pickle=False) as shard:
        assert shard["model_id"].item() == "A"
        assert shard["gene_symbols"].tolist() == ["G2", "MISSING"]
        assert shard["values"].shape == (2, 3)
        assert shard["available"].tolist() == [True, False]
        assert len(shard["source_sha256"].item()) == 64

    calls.clear()
    build_q_sc_shards(registry, output, ["G2", "MISSING"], reader=reader, resume=True)
    assert calls == []

    np.savez(output / "EXTRA.npz", x=np.array([1]))
    report = verify_q_sc_shards(registry, output, ["G2", "MISSING"])
    assert report["status"] == "failed"
    assert "extra shard: EXTRA.npz" in report["discrepancies"]

    (output / "EXTRA.npz").unlink()
    with np.load(output / "A.npz", allow_pickle=False) as shard:
        payload = {key: shard[key] for key in shard.files}
    payload["values"] = payload["values"].copy()
    payload["values"][0, 0] += 1
    np.savez(output / "A.npz", **payload)
    report = verify_q_sc_shards(registry, output, ["G2", "MISSING"])
    assert "A: shard SHA-256 mismatch" in report["discrepancies"]

    calls.clear()
    build_q_sc_shards(registry, output, ["G2", "MISSING"], reader=reader, resume=True)
    assert calls == [source_a]


def test_q_sc_builder_refuses_nonresume_overwrite_and_verifier_never_crashes(
    tmp_path: Path,
) -> None:
    source = tmp_path / "A.h5ad"
    source.write_bytes(b"source")
    registry = pd.DataFrame(
        {
            "source_path": [str(source)],
            "source_kind": ["h5ad"],
            "matrix_semantics": ["raw_umi_counts"],
        },
        index=pd.Index(["A"], name="model_id"),
    )
    output = tmp_path / "q_sc"
    output.mkdir()
    (output / "stale").write_text("stale")
    with pytest.raises(FileExistsError, match="nonempty"):
        build_q_sc_shards(registry, output, ["G1"], reader=lambda _: None)

    np.savez(
        output / "A.npz",
        model_id=np.asarray("A"),
        gene_symbols=np.asarray(["G1"]),
        values=np.asarray([1.0]),
        available=np.asarray([True]),
        source_sha256=np.asarray("bad"),
    )
    (output / "manifest.json").write_text("[]")
    report = verify_q_sc_shards(registry, output, ["G1"])
    assert report["status"] == "failed"
    assert "manifest root is not an object" in report["discrepancies"]
    assert "A: values shape mismatch" in report["discrepancies"]
