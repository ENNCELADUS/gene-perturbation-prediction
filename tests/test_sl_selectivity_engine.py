"""Unit tests for the cross-cell-line Selectivity engine."""

from __future__ import annotations

import numpy as np
import pandas as pd

from sl_benchmark_baseline.selectivity import (
    SelectivityTable,
    UniverseSelectivity,
    align_selectivity_to_universe,
    build_defective_mask,
    build_selectivity_table,
    load_ach_indexed_matrix,
    load_gene_effect_matrix,
    load_modelid_column_matrix,
    parse_entrez_columns,
)


def test_parse_entrez_columns():
    cols = ["A1BG (1)", "TP53 (7157)", "weird_no_entrez"]
    mapping = parse_entrez_columns(cols)
    assert mapping == {1: "A1BG (1)", 7157: "TP53 (7157)"}


def test_load_gene_effect_matrix(tmp_path):
    path = tmp_path / "ge.csv"
    pd.DataFrame(
        {"GENEA (10)": [-0.1, -0.9], "GENEB (20)": [0.0, -1.5]},
        index=["ACH-1", "ACH-2"],
    ).to_csv(path)
    lines, vecs = load_gene_effect_matrix(path)
    assert list(lines) == ["ACH-1", "ACH-2"]
    np.testing.assert_allclose(vecs[10], [-0.1, -0.9])
    np.testing.assert_allclose(vecs[20], [0.0, -1.5])


def test_load_ach_indexed_matrix_reindexes(tmp_path):
    path = tmp_path / "cn.csv"
    pd.DataFrame({"GENEA (10)": [1.0, 0.5]}, index=["ACH-2", "ACH-9"]).to_csv(path)
    lines = pd.Index(["ACH-1", "ACH-2"])
    vecs = load_ach_indexed_matrix(path, lines)
    # ACH-1 absent -> NaN; ACH-2 -> 1.0; ACH-9 dropped (not in lines)
    assert np.isnan(vecs[10][0])
    np.testing.assert_allclose(vecs[10][1], 1.0)


def test_load_modelid_column_matrix_reindexes(tmp_path):
    path = tmp_path / "mut.csv"
    pd.DataFrame(
        {
            "Unnamed: 0": [0, 1],
            "SequencingID": ["s1", "s2"],
            "ModelID": ["ACH-2", "ACH-9"],
            "ModelConditionID": ["mc1", "mc2"],
            "IsDefaultEntryForModel": [True, True],
            "IsDefaultEntryForMC": [True, True],
            "GENEA (10)": [1, 0],
        }
    ).to_csv(path, index=False)
    lines = pd.Index(["ACH-1", "ACH-2"])
    vecs = load_modelid_column_matrix(path, lines)
    assert np.isnan(vecs[10][0])  # ACH-1 absent
    np.testing.assert_allclose(vecs[10][1], 1.0)  # ACH-2


def test_build_defective_mask_or_across_channels():
    n = 5
    entrez = 10
    damaging = {10: np.array([1.0, 0.0, 0.0, 0.0, 0.0])}
    hotspot = {10: np.array([0.0, 1.0, 0.0, 0.0, 0.0])}
    cn = {10: np.array([1.0, 1.0, 0.5, 1.0, np.nan])}  # line 2 is a loss
    expr = {10: np.array([9.0, 9.0, 9.0, 0.0, 9.0])}  # line 3 lowest decile
    mask = build_defective_mask(
        entrez,
        n,
        damaging,
        hotspot,
        cn,
        expr,
        cn_loss_thr=0.8,
        expr_low_quantile=0.10,
    )
    # lines 0(dmg),1(hotspot),2(cn loss),3(low expr) defective; line 4 not
    assert mask.tolist() == [True, True, True, True, False]


def test_build_defective_mask_missing_channels():
    n = 3
    entrez = 99  # absent from all dicts
    mask = build_defective_mask(entrez, n, {}, {}, {}, {}, 0.8, 0.10)
    assert mask.tolist() == [False, False, False]


def _toy_inputs():
    # 3 cell lines, genes 10 and 20
    ge = {
        10: np.array([-1.0, -0.5, 0.0]),
        20: np.array([0.2, -2.0, -1.0]),
    }
    damaging = {
        10: np.array([1.0, 0.0, 0.0]),  # line 0 defective for gene 10
        20: np.array([0.0, 1.0, 0.0]),  # line 1 defective for gene 20
    }
    return ge, damaging


def test_build_selectivity_table_values():
    ge, damaging = _toy_inputs()
    table = build_selectivity_table(
        entrez_order=(10, 20),
        gene_effect_vectors=ge,
        damaging=damaging,
        hotspot={},
        cn_log2={},
        expr={},
        cn_loss_thr=0.8,
        expr_low_quantile=0.10,
        n_min=1,
    )
    assert isinstance(table, SelectivityTable)
    i, j = table.index_by_entrez[10], table.index_by_entrez[20]
    # sel(10->20) = mean(intact d20) - mean(def d20)
    #             = mean([-2.0,-1.0]) - mean([0.2]) = -1.5 - 0.2 = -1.7
    np.testing.assert_allclose(table.sel_matrix[i, j], -1.7, atol=1e-6)
    # sel(20->10) = mean([-1.0,0.0]) - mean([-0.5]) = -0.5 - (-0.5) = 0.0
    np.testing.assert_allclose(table.sel_matrix[j, i], 0.0, atol=1e-6)
    # pan_essential
    np.testing.assert_allclose(table.pan_essential[i], -0.5, atol=1e-6)
    np.testing.assert_allclose(table.pan_essential[j], -0.9333333, atol=1e-6)
    assert table.coverage_flag.tolist() == [1, 1]


def test_build_selectivity_table_n_min_fallback():
    ge, damaging = _toy_inputs()
    table = build_selectivity_table(
        entrez_order=(10, 20),
        gene_effect_vectors=ge,
        damaging=damaging,
        hotspot={},
        cn_log2={},
        expr={},
        cn_loss_thr=0.8,
        expr_low_quantile=0.10,
        n_min=2,  # gene10 has only 1 defective line -> fallback
    )
    i = table.index_by_entrez[10]
    assert table.coverage_flag[i] == 0
    np.testing.assert_allclose(table.sel_matrix[i, :], 0.0)


def test_align_selectivity_to_universe_gather_and_missing():
    table = build_selectivity_table(
        entrez_order=(10, 20),
        gene_effect_vectors={
            10: np.array([-1.0, -0.5, 0.0]),
            20: np.array([0.2, -2.0, -1.0]),
        },
        damaging={
            10: np.array([1.0, 0.0, 0.0]),
            20: np.array([0.0, 1.0, 0.0]),
        },
        hotspot={},
        cn_log2={},
        expr={},
        cn_loss_thr=0.8,
        expr_low_quantile=0.10,
        n_min=1,
    )
    # universe order: [20, 10, 999(missing)]
    uni = align_selectivity_to_universe(table, [20, 10, 999], sel_lambda=0.0)
    assert isinstance(uni, UniverseSelectivity)
    assert uni.sel_matrix.shape == (3, 3)
    # sel(20->10) lives at [0,1] in universe order; equals table sel(20->10)=0.0
    np.testing.assert_allclose(uni.sel_matrix[0, 1], 0.0, atol=1e-6)
    # sel(10->20) at [1,0] = -1.7
    np.testing.assert_allclose(uni.sel_matrix[1, 0], -1.7, atol=1e-6)
    # missing gene row/col all zero, coverage 0
    np.testing.assert_allclose(uni.sel_matrix[2, :], 0.0)
    assert uni.coverage_flag[2] == 0
    assert uni.coverage_flag[0] == 1 and uni.coverage_flag[1] == 1


def test_align_selectivity_lambda_penalty():
    table = build_selectivity_table(
        entrez_order=(10, 20),
        gene_effect_vectors={
            10: np.array([-1.0, -0.5, 0.0]),
            20: np.array([0.2, -2.0, -1.0]),
        },
        damaging={10: np.array([1.0, 0.0, 0.0]), 20: np.array([0.0, 1.0, 0.0])},
        hotspot={},
        cn_log2={},
        expr={},
        cn_loss_thr=0.8,
        expr_low_quantile=0.10,
        n_min=1,
    )
    uni0 = align_selectivity_to_universe(table, [10, 20], sel_lambda=0.0)
    uni1 = align_selectivity_to_universe(table, [10, 20], sel_lambda=1.0)
    # pan_essential(gene20) ~ -0.9333; penalty on col j=20: -1*max(0,0.9333)
    j = 1
    delta = uni1.sel_matrix[:, j] - uni0.sel_matrix[:, j]
    np.testing.assert_allclose(delta, -0.9333333, atol=1e-5)


def test_load_modelid_column_matrix_dedups_nondefault_rows(tmp_path):
    """Duplicate ModelIDs (multi-profile) collapse to the Yes default entry."""
    path = tmp_path / "mut_dups.csv"
    pd.DataFrame(
        {
            "SequencingID": ["s1", "s2", "s3"],
            "ModelID": ["ACH-2", "ACH-2", "ACH-9"],
            "IsDefaultEntryForModel": ["Yes", "No", "Yes"],
            "GENEA (10)": [1, 0, 1],
        }
    ).to_csv(path, index=False)
    lines = pd.Index(["ACH-1", "ACH-2"])
    vecs = load_modelid_column_matrix(path, lines)
    # ACH-1 absent -> NaN; ACH-2 uses the "Yes" row (value 1), not the "No" row
    assert np.isnan(vecs[10][0])
    np.testing.assert_allclose(vecs[10][1], 1.0)
