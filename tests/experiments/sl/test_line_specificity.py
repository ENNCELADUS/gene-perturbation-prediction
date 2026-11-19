"""Line-specific signal in SL labels on synthetic screens: a line effect that two
studies share repeats, one that they do not share does not."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.experiments.sl.line_specificity import (
    advantage,
    correlations,
    load_config,
    pair_share,
    run,
    study_cells,
)

LINES = ["L0", "L1", "L2", "L3"]
PAIRS = [(f"A{i:03d}", f"B{i:03d}") for i in range(300)]


def labels(shared: bool) -> pd.DataFrame:
    """Two studies on four lines: score = pair effect + line-specific effect + noise.
    ``shared``: both studies see one line-specific effect; else each has its own.
    Study S2 reports a score whose low side is SL."""
    rng = np.random.default_rng(0)
    pair = rng.normal(size=len(PAIRS))
    line_effect = rng.normal(size=(2, len(LINES), len(PAIRS)))
    frames = []
    for s, study in enumerate(("S1", "S2")):
        for c, line in enumerate(LINES):
            latent = (
                pair + line_effect[0 if shared else s, c]
                + 0.3 * rng.normal(size=len(PAIRS))
            )  # fmt: skip
            frames.append(
                pd.DataFrame(
                    {
                        "study_id": study,
                        "list_id": f"{study}|{line}",
                        "model_id": line,
                        "design": "combinatorial",
                        "condition": "genetic",
                        "gene_a": [a for a, _ in PAIRS],
                        "gene_b": [b for _, b in PAIRS],
                        "score": latent if s == 0 else -latent,
                        "label": (latent >= np.quantile(latent, 0.9)).astype(float),
                    }
                )  # fmt: skip
            )
    return pd.concat(frames, ignore_index=True)


def metrics(tmp_path, shared: bool) -> dict:
    (tmp_path / "split.json").write_text(
        json.dumps({"train": LINES[:2], "val": LINES[2:3], "test": LINES[3:]})
    )
    labels(shared).to_parquet(tmp_path / "labels.parquet")
    (tmp_path / "config.yaml").write_text(
        f"kind: sl_line_specificity\nseed: 0\noutput_root: {tmp_path}\ntask: t\n"
        f"labels: {tmp_path}/labels.parquet\nsplit: {tmp_path}/split.json\n"
        "bootstrap: 50\nmin_pairs: 20\nmin_lines: 3\n"
    )
    out = run(load_config(tmp_path / "config.yaml"), f"shared_{shared}", workers=2)
    assert "Table 2" in (out / "results.md").read_text()
    return json.loads((out / "metrics.json").read_text())


def test_a_shared_line_effect_repeats(tmp_path):
    found = metrics(tmp_path, shared=True)
    row = found["study_pairs"][0]
    rank, call = row["rank"], row["call"]
    assert (row["shared_lines"], row["pairs"]) == (4, 300)
    assert (rank["same_line_comparisons"], rank["other_line_comparisons"]) == (4, 12)
    assert rank["line_part_repeat"] > 0.8 and rank["line_part_repeat_interval"][0] > 0.7
    assert rank["advantage"] > 0.3 and rank["advantage_interval"][0] > 0.2
    # every line is shared, so no screen's level is left to remove
    assert rank["advantage"] == pytest.approx(rank["same_line"] - rank["other_line"])
    assert call["advantage"] > 0.2 and call["line_part_repeat"] > 0.4
    assert call["positive_agreement_same_line"] > call["positive_agreement_other_line"]
    pooled = found["pooled"]["rank"]
    assert pooled["ceiling"] == pytest.approx(rank["line_part_repeat"] ** 0.5)
    # half of the variance is the pair effect, in both studies
    assert [r["pair_share_rank"] for r in found["within_study"]] == pytest.approx(
        [0.48, 0.48], abs=0.08
    )


def test_an_unshared_line_effect_does_not_repeat(tmp_path):
    row = metrics(tmp_path, shared=False)["study_pairs"][0]
    for kind in ("rank", "call"):
        low, high = row[kind]["line_part_repeat_interval"]
        assert low < 0 < high and abs(row[kind]["line_part_repeat"]) < 0.1
        assert abs(row[kind]["advantage"]) < 0.1


def test_a_noisy_screen_does_not_decide_the_advantage():
    """One shared line whose screen is noisy in the second study: its plain
    correlation is below another line's, the advantage still shows the line."""
    rng = np.random.default_rng(2)
    pair, line = rng.normal(size=4000), rng.normal(size=(3, 4000))
    a = pair + line[:2] + 0.3 * rng.normal(size=(2, 4000))  # lines L0, L1
    b = pair + line[[0, 2]] + 0.3 * rng.normal(size=(2, 4000))  # lines L0, L2
    b[0] += 3 * rng.normal(size=4000)  # the shared line's screen is noisy
    match = np.array([[True, False], [False, False]])
    r = correlations(a, b, np.ones(4000), min_pairs=20)
    assert r[0, 0] < r[0, 1] and advantage(r, match) > 0.05


def test_a_score_whose_low_side_is_sl_is_turned():
    cells, studies = study_cells(labels(shared=True), set(LINES))
    assert studies.set_index("study_id")["sl_on_low_side"].to_dict() == {
        "S1": False, "S2": True
    }  # fmt: skip
    # after the turn, SL pairs are on the high side in both studies
    high = cells[cells["label"] == 1].groupby("study_id")["value"].mean()
    assert (high > 0.9).all()


def test_pair_share_and_masked_correlations():
    rng = np.random.default_rng(1)
    pair = rng.normal(size=2000)
    matrix = pair + rng.normal(size=(6, 2000))
    assert pair_share(matrix) == pytest.approx(0.5, abs=0.05)
    matrix[0, :1000] = np.nan  # a pair needs two lines, not all
    assert pair_share(matrix) == pytest.approx(0.5, abs=0.05)
    a, b = matrix[:2].copy(), matrix[2:4].copy()
    r = correlations(a, b, np.ones(2000), min_pairs=20)
    seen = np.isfinite(a[0])
    assert r[0, 1] == pytest.approx(np.corrcoef(a[0, seen], b[1, seen])[0, 1])
    assert np.isnan(correlations(a[:, :10], b[:, :10], np.ones(10), min_pairs=20)).all()
