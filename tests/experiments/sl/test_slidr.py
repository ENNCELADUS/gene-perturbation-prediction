"""The SLIdR port: R's Irwin-Hall CDF, and a planted SL pair on a synthetic world."""

import json

import numpy as np
import pandas as pd
import pytest

from src.experiments.sl.slidr import UNTESTED, irwin_hall_cdf, load_config, run


def test_irwin_hall_cdf_matches_the_closed_form_for_one_and_two_uniforms():
    x = np.array([0.0, 0.3, 0.9, 1.0])
    assert irwin_hall_cdf(x, 1) == pytest.approx(x)
    x = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    two = np.where(x <= 1, x**2 / 2, 1 - (2 - x) ** 2 / 2)
    assert irwin_hall_cdf(x, 2) == pytest.approx(two)
    # above 20 lines R switches to N(n/2, n/12): the median stays at 0.5
    assert irwin_hall_cdf(np.array([15.0]), 30) == pytest.approx([0.5])


def depmap(frame: pd.DataFrame, omics: bool) -> pd.DataFrame:
    """A DepMap file layout: ``SYMBOL (Entrez)`` columns; omics tables with a
    ModelID column and default entries."""
    out = frame.rename(columns={g: f"{g} ({i})" for i, g in enumerate(frame.columns)})
    if not omics:
        return out
    return out.rename_axis("ModelID").reset_index().assign(IsDefaultEntryForModel="Yes")


def test_a_planted_pair_gets_the_best_score_and_untestable_pairs_tie(tmp_path):
    rng = np.random.default_rng(0)
    fit = [f"F{i}" for i in range(40)]
    lines = fit + ["V0", "E0", "B0", "B1"]
    genes = ["DRV", "SLP"] + [f"G{i}" for i in range(30)]
    effect = pd.DataFrame(
        rng.normal(size=(len(lines), len(genes))), index=lines, columns=genes
    )
    mutated = pd.DataFrame(0, index=effect.index, columns=["DRV", "G0", "G1"])
    mutated.loc[fit[:10], "DRV"] = 1
    mutated.loc[fit[10:], "G0"] = 1  # every line has an alteration
    effect.loc[fit[:10], "SLP"] -= 4  # SLP is lethal only where DRV is mutated
    # G1 is never altered, G2 lost (0 copies) in one line only: neither is a driver
    copy_number = pd.DataFrame(1.0, index=effect.index, columns=genes)
    copy_number.loc["F0", "G2"] = 0.0
    depmap(effect, False).to_csv(tmp_path / "ge.csv", index=True)
    depmap(mutated, True).to_csv(tmp_path / "mut.csv", index=False)
    # DepMap's copy number: lines in the index, and columns with an Ensembl ID only
    depmap(copy_number, False).assign(ENSG00000100101=1.0).to_csv(tmp_path / "cn.csv")
    pairs = [("DRV", "SLP"), ("SLP", "DRV"), ("DRV", "G5"), ("G0", "G5"), ("G1", "G2")]
    pd.DataFrame(
        [{"model_id": m, "gene_a": a, "gene_b": b, "label": 1}
         for m in ("V0", "E0", "B0") for a, b in pairs]
    ).to_parquet(tmp_path / "labels.parquet")  # fmt: skip
    # lineage A: the 40 fitted lines, V0 and E0; lineage B: B0 and one fitted line
    split = {"train": fit[:30] + ["B1"], "bulk_labelled": fit[30:], "val": ["V0"],
             "test": ["E0", "B0"]}  # fmt: skip
    (tmp_path / "split.json").write_text(json.dumps(split))
    pd.DataFrame({"model_id": ["V0", "E0", "B0"], "lineage": ["A", "A", "B"]}).to_csv(
        tmp_path / "split.csv", index=False
    )
    pd.DataFrame(
        {"ModelID": fit + ["B1"], "OncotreeLineage": ["A"] * 40 + ["B"]}
    ).to_csv(tmp_path / "Model.csv", index=False)
    (tmp_path / "config.yaml").write_text(
        f"kind: sl_slidr\nlabels: {tmp_path}/labels.parquet\n"
        f"split: {tmp_path}/split.json\nmodel: {tmp_path}/Model.csv\n"
        f"gene_effect: {tmp_path}/ge.csv\n"
        f"mutations: [{tmp_path}/mut.csv]\ncopy_number: {tmp_path}/cn.csv\n"
        f"output_root: {tmp_path}\ntask: slidr\nmin_lines: 5\nmin_mutated: 2\n"
        "wt_pvalue: 0.1\n"
    )
    out = run(load_config(tmp_path / "config.yaml"), "r0")
    table = pd.read_parquet(out / "scores.parquet")
    scores = table[table["model_id"] == "E0"].set_index(["gene_a", "gene_b"])
    # both orientations of the planted pair read DRV as the driver
    assert scores.loc[("DRV", "SLP"), "score"] == scores.loc[("SLP", "DRV"), "score"]
    assert scores.loc[("DRV", "SLP"), "score"] > 3 * scores.loc[("DRV", "G5"), "score"]
    assert scores["covered"].tolist() == [True, True, True, True, False]
    assert scores.loc[("G1", "G2"), "score"] == pytest.approx(UNTESTED)
    # lineage B has one fitted line, fewer than min_lines: nothing is tested there
    assert not table.loc[table["model_id"] == "B0", "covered"].any()
    assert len(table) == 15
    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["lineages"]["A"]["fit_lines"] == 40
    assert metrics["lineages"]["B"]["tests"] == 0
