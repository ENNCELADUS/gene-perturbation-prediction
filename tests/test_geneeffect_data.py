"""Strict Exp13 GeneEffect data-contract tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.geneeffect import (
    Exp13Split,
    build_g_var,
    build_residual_data,
    build_scored_universe,
    fit_residual_scale,
    fit_selective_genes,
    load_exp13_split,
    load_geneeffect_long,
    load_source_registry,
    parse_gene_symbol,
)
from src.data.prepared import load_inputs
from tests.test_joint_data import make_prepared_fixture


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


def _toy(column: str, values: dict[str, list[float]], lines: list[str]) -> pd.DataFrame:
    """Long labels with ``column`` set per gene and line, and 0 in the other."""
    rows = [
        {
            "model_id": line,
            "gene_symbol": gene,
            "gene_effect": 0.0,
            "residual": 0.0,
            column: value,
        }
        for gene, gene_values in values.items()
        for line, value in zip(lines, gene_values)
    ]
    return pd.DataFrame(rows)


def test_selective_genes_threshold_fraction_and_train_only() -> None:
    train = ["T1", "T2", "T3", "T4", "T5"]
    values = {
        "TWO": [-1, -0.6, 0, 0, 0],  # 2 of 5 depend: selective
        "AT_THRESHOLD": [-0.5, -0.5, -0.5, 0, 0],  # -0.5 is not < -0.5
        "BOUND": [-1, -1, -1, 0, 0],  # fraction 0.6 is not < 0.6
        "ALL": [-1, -1, -1, -1, -1],  # fraction 1.0
        "FEW": [-1, 0, 0, 0, 0],  # one dependent line < min_lines
        "NAN": [-1, -1, np.nan, 0, 0],  # 2 of 4 finite rows: fraction 0.5
    }
    labels = _toy("gene_effect", values, train)
    # Val rows must not count: they would make FEW selective.
    leak = _toy("gene_effect", {"FEW": [-5.0, -5.0]}, ["V1", "V2"])
    labels = pd.concat([labels, leak], ignore_index=True)
    genes = [*values, "ABSENT"]
    result = fit_selective_genes(labels, train, genes, min_lines=2, max_fraction=0.6)
    assert result == frozenset({"TWO", "NAN"})
    assert fit_selective_genes(
        labels, train, genes, min_lines=2, max_fraction=0.7
    ) == frozenset({"TWO", "BOUND", "NAN"})
    assert fit_selective_genes(
        labels, train, genes, min_lines=3, max_fraction=1.0
    ) == frozenset({"BOUND"})


def test_selective_genes_fraction_boundary_is_exact() -> None:
    train = [f"T{i}" for i in range(10)]
    labels = _toy("gene_effect", {"G": [-1.0] * 7 + [0.0] * 3}, train)
    # 7 / 10 == 0.7 exactly in the comparison, so the gene is not below 0.7.
    assert (
        fit_selective_genes(labels, train, ["G"], min_lines=1, max_fraction=0.7)
        == frozenset()
    )
    assert fit_selective_genes(
        labels, train, ["G"], min_lines=1, max_fraction=0.71
    ) == frozenset({"G"})


def test_selective_genes_validate_arguments() -> None:
    labels = _toy("gene_effect", {"G": [-1.0, 0.0]}, ["T1", "T2"])
    for kwargs in (
        {"min_lines": 0, "max_fraction": 0.5},
        {"min_lines": 1, "max_fraction": 0.0},
        {"min_lines": 1, "max_fraction": 1.5},
    ):
        with pytest.raises(ValueError):
            fit_selective_genes(labels, ["T1", "T2"], ["G"], **kwargs)
    with pytest.raises(ValueError, match="train_lines"):
        fit_selective_genes(labels, [], ["G"], min_lines=1, max_fraction=0.5)
    with pytest.raises(ValueError, match="missing columns"):
        fit_selective_genes(
            labels.drop(columns="gene_effect"),
            ["T1"],
            ["G"],
            min_lines=1,
            max_fraction=0.5,
        )


def test_residual_scale_population_sd_floor_and_short_genes() -> None:
    train = ["T1", "T2", "T3", "T4", "T5"]
    labels = _toy(
        "residual",
        {
            "A": [1, 2, 3, 4, 5],  # SD sqrt(2)
            "B": [0, 0, 0, 0, 0],  # SD 0, below the floor
            "C": [0, 10, 0, 10, 0],  # SD sqrt(24)
        },
        train,
    )
    # D has one training row (so it takes the floor); the val row must not count.
    extra = pd.concat(
        [
            _toy("residual", {"D": [7.0]}, ["T1"]),
            _toy("residual", {"A": [1000.0]}, ["V1"]),
        ]
    )
    labels = pd.concat([labels, extra], ignore_index=True)
    scale = fit_residual_scale(
        labels, train, ["A", "B", "C", "D", "ABSENT"], floor_percentile=50
    )
    # Finite SDs are {sqrt(2), 0, sqrt(24)}: the median is sqrt(2).
    expected = [np.sqrt(2), np.sqrt(2), np.sqrt(24), np.sqrt(2), np.sqrt(2)]
    assert list(scale.index) == ["A", "B", "C", "D", "ABSENT"]
    np.testing.assert_allclose(scale.to_numpy(), expected)


def test_residual_scale_rejects_nonpositive_floor_and_missing_data() -> None:
    train = ["T1", "T2", "T3"]
    labels = _toy("residual", {"A": [0, 0, 0], "B": [0, 1, 2]}, train)
    # The 0th percentile of {0, sqrt(2/3)} is 0: no positive scale exists.
    with pytest.raises(ValueError, match="positive"):
        fit_residual_scale(labels, train, ["A", "B"], floor_percentile=0)
    with pytest.raises(ValueError, match="two training rows"):
        fit_residual_scale(labels, ["T1"], ["A", "B"], floor_percentile=50)
    with pytest.raises(ValueError, match="between 0 and 100"):
        fit_residual_scale(labels, train, ["A", "B"], floor_percentile=101)


def test_selective_genes_and_residual_scale_round_trip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.data.prepared as prepared

    config = make_prepared_fixture(tmp_path)
    config["features"]["selective_min_lines"] = 2
    wide = pd.read_csv(config["paths"]["gene_effect"], index_col=0).astype(float)
    # G0: two of the five supervised training lines depend on it.
    wide.iloc[:5, wide.columns.get_loc("G0 (1)")] = [-1.0, -1.0, 0.0, 0.0, 0.0]
    wide.to_csv(config["paths"]["gene_effect"])
    fitted = load_inputs(config)
    assert fitted.selective_genes == frozenset({"G0"})
    assert list(fitted.residual_scale.index) == list(fitted.genes)
    assert np.isfinite(fitted.residual_scale.to_numpy()).all()
    assert (fitted.residual_scale.to_numpy() > 0).all()

    state = fitted.preprocessing_state()
    assert state["selective_genes"] == ["G0"]
    assert state["residual_scale"]["symbols"] == list(fitted.genes)

    def forbidden(*args, **kwargs):
        raise AssertionError("restore must not refit")

    monkeypatch.setattr(prepared, "fit_selective_genes", forbidden)
    monkeypatch.setattr(prepared, "fit_residual_scale", forbidden)
    restored = load_inputs(config, preprocessing=state)
    assert restored.selective_genes == fitted.selective_genes
    pd.testing.assert_series_equal(restored.residual_scale, fitted.residual_scale)

    for missing in ("selective_genes", "residual_scale"):
        broken = {k: v for k, v in state.items() if k != missing}
        with pytest.raises(ValueError, match=missing):
            load_inputs(config, preprocessing=broken)
    both = {
        k: v for k, v in state.items() if k not in ("selective_genes", "residual_scale")
    }
    with pytest.raises(ValueError, match="selective_genes, residual_scale"):
        load_inputs(config, preprocessing=both)
    state["residual_scale"]["values"][0] = 0.0
    with pytest.raises(ValueError, match="finite and positive"):
        load_inputs(config, preprocessing=state)


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
