"""Gate tests for the paper table-extraction script.

Anchors are read directly from on-disk artifacts and cross-checked against the
spec Numbers Ledger (docs/superpowers/specs/2026-06-22-paper-story-design.md).
Do NOT edit an anchor to make a test pass; if disk disagrees with the ledger,
reconcile against the experiment docs first.
"""
from __future__ import annotations

import importlib.util
import pathlib
import subprocess

_SPEC = importlib.util.spec_from_file_location(
    "make_tables", pathlib.Path("docs/report/scripts/make_tables.py")
)
mt = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mt)

EXP06 = "results/experiments/06_k562_sl_pair_dependency_only_mvp/official_metrics_summary.csv"
EXP07 = "results/experiments/07_k562_sl_pair_perturbseq_augmented/augmented_no_flag/summary.csv"
EXP09 = "results/experiments/09_k562_sl_pair_cross_cell_line_selectivity/run/summary.csv"
EXP08_DIRS = "results/experiments/08_k562_sl_pair_state_dl/phase2_bce*"


def test_floor_cv1_modelB_auroc():
    # verified on disk: CV1,B,auroc,0.7947182192429787
    assert mt.fmt(mt.read_metric(EXP06, "CV1", "B", "auroc")) == "0.795"


def test_floor_cv2_modelB_auroc_matches_ledger():
    # spec ledger: exp06 CV2 Model B AUROC = 0.704
    assert mt.fmt(mt.read_metric(EXP06, "CV2", "B", "auroc")) == "0.704"


def test_transcriptome_cv2_lift_full_universe():
    # spec ledger C3: exp07 CV2 B_transcript NDCG@10 = 0.094 (full_universe slice)
    val = mt.read_metric(EXP07, "CV2", "B_transcript", "ndcg@10", slice="full_universe")
    assert mt.fmt(val) == "0.094"


def test_decomposition_cv3_nonpanessential_auroc():
    # spec ledger C5: exp09 CV3 B_xcl non_pan_essential AUROC = 0.583
    val = mt.read_metric(EXP09, "CV3", "B_xcl", "auroc", slice="non_pan_essential")
    assert mt.fmt(val) == "0.583"


def test_method_best_fold_cv2_auroc():
    # spec ledger CNEW: exp08 Phase2 BCE CV2 best-fold AUROC = 0.6667 -> 0.667
    val = mt.read_best_fold_metric(EXP08_DIRS, "CV2", "auroc", slice="full_universe")
    assert mt.fmt(val) == "0.667"


def test_fmt_is_three_dp():
    assert mt.fmt(0.7) == "0.700"


def test_all_tables_generated_and_clean():
    subprocess.run(
        ["python", "docs/report/scripts/make_tables.py", "--all"], check=True
    )
    for name in ["floor", "transcriptome", "method", "decomposition", "foundation", "benchmark"]:
        txt = pathlib.Path(f"docs/report/tables/tab_{name}.tex").read_text()
        assert "\\toprule" in txt and "|" not in txt  # booktabs, no vertical rules
    method_txt = pathlib.Path("docs/report/tables/tab_method.tex").read_text().lower()
    assert "5-fold" in method_txt  # F2 preliminary flag present


def test_benchmark_published_grsmf_mean_ndcg():
    import pandas as pd
    df = pd.read_csv("docs/report/tables/benchmark_published.csv")
    val = df.loc[df["model"] == "GRSMF", "mean_ndcg10"].iloc[0]
    assert abs(float(val) - 0.317) < 1e-9


def test_benchmark_published_slgnn_cv3_ndcg_collapse():
    import pandas as pd
    df = pd.read_csv("docs/report/tables/benchmark_published.csv")
    val = df.loc[df["model"] == "SLGNN", "cv3_ndcg10"].iloc[0]
    assert abs(float(val) - 0.000) < 1e-9


def test_mean_over_cvs_matches_simple_average():
    assert abs(mt.mean_over_cvs([0.7947, 0.7035, 0.5956]) - 0.6979333333) < 1e-6


def test_benchmark_functional_floor_mean_auroc():
    # exp06 Model B mean-over-CV AUROC, recomputed from artifacts (~0.698).
    vals = [mt.read_metric(mt.EXP06, s, "B", "auroc") for s in ("CV1", "CV2", "CV3")]
    assert abs(mt.mean_over_cvs(vals) - 0.698) < 0.005


def test_benchmark_table_has_both_rowgroups():
    subprocess.run(
        ["python", "docs/report/scripts/make_tables.py", "--table", "benchmark"],
        check=True,
    )
    text = pathlib.Path("docs/report/tables/tab_benchmark.tex").read_text()
    assert "\\toprule" in text and "|" not in text
    assert "Label-graph" in text and "Functional" in text
    assert "GRSMF" in text and "DDGCN" in text


def test_difficulty_ladder_figure_builds():
    r = subprocess.run(
        ["python", "docs/report/scripts/make_figures.py"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    assert pathlib.Path("docs/report/figures/fig_difficulty_ladder.pdf").exists()
