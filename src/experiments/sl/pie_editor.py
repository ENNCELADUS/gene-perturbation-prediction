"""The PIE editor of the SL backbone (SL protocol §13): the context editor is PIE's
predicted profile after A's loss in place of A's expression set low; the reader (the
default linear expression model of the backbone run), the line encoder and the
conditional score do not change.

The row after A's loss is the line's raw pseudo-bulk (log1p CPM of summed UMI) plus
PIE's predicted change (a change of the mean over cells of log1p CP10K) times each
gene's slope from PIE's input to the raw pseudo-bulk over the SL training lines (the
affine bridge's least squares), then the reader's quantile normalisation, the line's
bridge (its fold's for a labelled single-cell training line) and the reader. A term
is B's prediction after minus before; the conditional score and the logistic head
are the backbone's. Rows: PIE's profile of the line, PIE's profile of another line
(one seeded permutation of the screened lines, no line keeps its own), each on the
rows the simple edit covers (0 elsewhere, as the simple edit) and on every row PIE
can edit; and the reference rows, the measured Perturb-seq change of the line
against that of another line, on the rows whose lost genes both lines perturbed.
Scores go to ``scores/<name>.parquet`` for the benchmark harness."""

from __future__ import annotations

import argparse
import os
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.extra_lines import load_bulk_lines
from src.data.io import jsonable, write_json, write_text
from src.data.pseudobulk import read_pseudobulk
from src.data.splits import load_geneeffect_split
from src.experiments.joint.config import load_config as load_joint_config
from src.experiments.linear_model.base import base_on
from src.experiments.linear_model.config import load_linear_model_config
from src.experiments.run_dir import absolute, bind
from src.experiments.sl.benchmark_baselines import benchmark_rows
from src.experiments.sl.conditional import (
    KEYS,
    decision,
    features,
    fit_head,
    predictions,
    readers_and_rows,
    terms,
)
from src.experiments.sl.pie_pass import CHANGE, MEASURED, cell_inputs, load_config
from src.experiments.stage_one.pie import read_tables
from src.linear_model.bridge import fit_bridge

Change = Callable[[np.ndarray, np.ndarray], np.ndarray]


class ProfileEdit:
    """A lost: the line's raw pseudo-bulk plus ``slope`` times a change on the
    panel (``change(model ids, lost genes)``, in PIE's unit), then ``normalize``
    and the reader key's bridge. Every query moves the row."""

    def __init__(
        self,
        genes: Sequence[str],
        change: Change,
        raw: pd.DataFrame,
        panel: Sequence[str],
        slope: np.ndarray,
        normalize: Callable[[pd.DataFrame], pd.DataFrame],
        bridges: Mapping[int, Any],
    ) -> None:
        self.genes = pd.Index(genes)
        self.change = change
        self.lines = raw.index
        self.columns = raw.columns
        self.raw = raw.to_numpy(dtype=np.float64)
        self.at = raw.columns.get_indexer(list(panel))
        if (self.at < 0).any():
            raise ValueError("panel genes outside the raw rows")
        self.slope = np.asarray(slope, dtype=np.float64)
        self.normalize = normalize
        self.bridges = bridges

    def moves(self, line: np.ndarray, lost: np.ndarray) -> np.ndarray:
        return np.ones(len(line), dtype=bool)

    def after(self, line: np.ndarray, lost: np.ndarray) -> pd.DataFrame:
        """The raw rows after the loss (before normalisation)."""
        values = self.raw[line].copy()
        moved = self.change(self.lines[line].to_numpy(), self.genes[lost].to_numpy())
        values[:, self.at] += self.slope * np.nan_to_num(moved, nan=0.0)
        return pd.DataFrame(values, columns=self.columns)

    def __call__(self, line: np.ndarray, lost: np.ndarray, key: int) -> np.ndarray:
        edited = self.bridges[key].apply(self.normalize(self.after(line, lost)))
        return edited.to_numpy()


class ChangeStore:
    """The changes of the (source line, lost gene) pairs ``pairs`` asks for, read
    once per gene from ``directory/<gene>.npy`` (lines x panel)."""

    def __init__(self, directory: Path, lines: Sequence[str], pairs: pd.DataFrame,
                 width: int) -> None:  # fmt: skip
        pairs = pairs.drop_duplicates().sort_values(["gene", "source"])
        pairs = pairs.reset_index(drop=True)
        self.index = pd.MultiIndex.from_frame(pairs[["source", "gene"]])
        self.values = np.empty((len(pairs), width), dtype=np.float16)
        position = {m: i for i, m in enumerate(lines)}
        for gene, group in pairs.groupby("gene", sort=False):
            array = np.load(Path(directory) / f"{gene}.npy")
            if array.shape != (len(lines), width):
                raise ValueError(f"{gene}: change of shape {array.shape}")
            self.values[group.index] = array[[position[m] for m in group["source"]]]

    def __call__(self, sources: np.ndarray, genes: np.ndarray) -> np.ndarray:
        at = self.index.get_indexer(pd.MultiIndex.from_arrays([sources, genes]))
        if (at < 0).any():
            raise KeyError("a change the store was not asked for")
        return self.values[at].astype(np.float64)


def derangement(lines: Sequence[str], seed: int) -> dict[str, str]:
    """One seeded permutation of ``lines`` without a fixed point: a random order,
    each line mapped to the next."""
    order = [str(m) for m in np.random.default_rng(seed).permutation(sorted(lines))]
    return dict(zip(order, order[1:] + order[:1]))


def unit_slope(pie_input: pd.DataFrame, raw: pd.DataFrame) -> np.ndarray:
    """Per gene: the slope of the least-squares line from PIE's input to the raw
    pseudo-bulk across lines (``fit_bridge``)."""
    return fit_bridge(pie_input, raw.loc[pie_input.index, pie_input.columns]).slope


def queries(rows: pd.DataFrame) -> pd.DataFrame:
    """Every (line, lost gene) a term edits: A of each row, B of a pair."""
    pair = rows["design"] != "anchor"
    return pd.concat(
        [
            rows[["model_id", "gene_a"]].set_axis(["line", "gene"], axis=1),
            rows.loc[pair, ["model_id", "gene_b"]].set_axis(["line", "gene"], axis=1),
        ]
    ).drop_duplicates()


def measured_store(root: Path, name: str) -> tuple[pd.Index, np.ndarray]:
    dataset, context = name.split("/")
    with np.load(root / MEASURED / f"{dataset}__{context}.npz") as payload:
        return pd.Index(payload["perturbations"]), payload["change"]


def conditional(values: np.ndarray) -> np.ndarray:
    return -values.sum(axis=1)


def movement(edit: ProfileEdit, line: np.ndarray, lost: np.ndarray,
             before: np.ndarray) -> dict[str, float]:  # fmt: skip
    """How far the change moves the edited rows after normalisation: per edit, the
    median absolute rank change of the genes whose raw value moved, the share of
    the raw change's norm left after normalisation, and their Pearson; medians over
    edits."""
    from scipy.stats import rankdata

    raw_before = edit.raw[line]
    raw_after = edit.after(line, lost).to_numpy()
    after = edit.normalize(pd.DataFrame(raw_after, columns=edit.columns)).to_numpy()
    shift, moved = raw_after - raw_before, after - before
    ranks = np.abs(rankdata(after, axis=1) - rankdata(before, axis=1))
    rank, kept, pearson = [], [], []
    for i in range(len(line)):
        changed = np.abs(shift[i]) > 0
        if not changed.any():
            continue
        rank.append(np.median(ranks[i, changed]))
        kept.append(np.linalg.norm(moved[i]) / np.linalg.norm(shift[i]))
        if np.ptp(moved[i]) > 0:
            pearson.append(np.corrcoef(shift[i], moved[i])[0, 1])
    return {
        "edits": len(rank),
        "genes": int(raw_before.shape[1]),
        "median_abs_rank_change": float(np.median(rank)),
        "norm_kept": float(np.median(kept)),
        "pearson_raw_normalised": float(np.median(pearson)),
    }


def run(config: Mapping[str, Any], run_id: str) -> Path:
    run_dir = absolute(config["output_root"]) / config["task"] / run_id
    if (run_dir / "metrics.json").exists():
        raise FileExistsError(f"{run_dir} is finished; use a new --run-id")
    run_dir.mkdir(parents=True, exist_ok=True)
    bind(run_dir, config)
    seed = int(config["seed"])
    split_path = absolute(config["split"])
    split = load_geneeffect_split(split_path)
    linear_model = load_linear_model_config(absolute(config["linear_model"]))
    base = base_on(
        linear_model,
        split,
        load_bulk_lines(split_path, split),
        [*split.val, *split.test],
        other_split=True,
    )
    leaked = set(base.labelled) & {*split.val, *split.test}
    if leaked:
        raise ValueError(f"SL held-out lines among the fit lines: {sorted(leaked)}")
    readers, assignment, line_rows, _, bridges = readers_and_rows(base)
    residual = predictions(readers, assignment, line_rows)
    rows, _ = benchmark_rows(
        pd.read_parquet(absolute(config["labels"])), split, set(base.definitions.genes)
    )

    root = absolute(config["pie_root"])
    tables = read_tables(root)
    panel, space = tables.panel, list(line_rows.columns)
    keyed = set(tables.genes.index[tables.genes["in_panel"]])
    if panel != [g for g in space if g in keyed]:
        raise ValueError("the PIE root's panel is not the reader's space with a key")
    joint = load_joint_config(absolute(linear_model["joint_config"]))
    raw = read_pseudobulk(absolute(joint["prepared_root"])).loc[line_rows.index, space]
    unit_lines = [
        m for m in split.supervised_train
        if (absolute(config["ladder_root"]) / "cells" / f"{m}.npz").is_file()
    ]  # fmt: skip
    slope = unit_slope(
        cell_inputs(absolute(config["ladder_root"]), unit_lines, panel),
        raw.loc[unit_lines, panel].fillna(base.pseudobulk_fill.loc[panel]),
    )

    def edit(genes: Sequence[str], change: Change) -> ProfileEdit:
        return ProfileEdit(
            genes, change, raw, panel, slope, base.normalize_pseudobulk, bridges
        )

    # the zero change reproduces each line's row and gives zero terms
    zero = edit(tables.genes.index, lambda m, g: np.zeros((len(m), len(panel))))
    every = np.arange(len(raw))
    gap = max(
        float(np.abs(zero(every[[i]], np.array([0]), assignment[m])
                     - line_rows.loc[[m]].to_numpy()).max())
        for i, m in enumerate(raw.index)
    )  # fmt: skip
    sample = rows[rows["split"] != "train"].sample(20_000, random_state=seed)
    zero_terms = terms(sample, line_rows, None, assignment, readers, zero)
    zero_terms = zero_terms[np.isfinite(zero_terms)]

    # PIE's changes of every (line, lost gene) and (other line, lost gene) asked for
    lines = list(tables.lines.index)
    unknown = sorted(set(rows["model_id"]) - set(lines))
    if unknown:
        raise ValueError(f"rows of lines PIE did not pass: {unknown[:5]}")
    other = derangement(lines, seed)
    lost = list(tables.genes.index[tables.genes["lost"]])
    asked = queries(rows)
    asked = asked[asked["gene"].isin(set(lost))]
    pairs = pd.concat(
        [
            asked.rename(columns={"line": "source"}),
            asked.assign(source=asked["line"].map(other))[["source", "gene"]],
        ]
    )
    store = ChangeStore(root / "pie" / CHANGE, lines, pairs, len(panel))
    own = edit(lost, store)
    permuted = edit(lost, lambda m, g: store(pd.Series(m).map(other).to_numpy(), g))
    reading = {
        "pie": terms(rows, line_rows, None, assignment, readers, own),
        "pie_permuted": terms(rows, line_rows, None, assignment, readers, permuted),
    }
    pie_covered = np.isfinite(reading["pie"]).all(axis=1)

    # the simple edit's covered rows, from the backbone run
    simple = pd.read_parquet(
        absolute(config["simple_scores"]) / "conditional.parquet",
        columns=[*KEYS, "covered"],
    )
    simple_covered = rows[KEYS].merge(simple, on=KEYS, how="left")["covered"].to_numpy()
    if pd.isna(simple_covered).any():
        raise ValueError("rows the backbone run did not score")
    simple_covered = simple_covered.astype(bool)

    train = (rows["split"] == "train").to_numpy()
    val = (rows["split"] == "val").to_numpy()
    scores: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    selection = {}
    for name, covered in (("pie", simple_covered), ("pie_all", pie_covered)):
        own_terms, other_terms = (
            np.where(covered[:, None], np.nan_to_num(reading[k], nan=0.0), 0.0)
            for k in ("pie", "pie_permuted")
        )
        x, x_other = (features(rows, t, residual, line_rows)
                      for t in (own_terms, other_terms))  # fmt: skip
        head, selection[name] = fit_head(
            rows[train], x[train], rows[val], x[val],
            config["head_penalties"], int(config["top_k"]),
        )  # fmt: skip
        scores |= {
            f"{name}_conditional": (conditional(own_terms), covered),
            f"{name}_conditional_permuted": (conditional(other_terms), covered),
            f"{name}_head": (decision(head, x), covered),
            f"{name}_head_permuted": (decision(head, x_other), covered),
        }

    # reference rows: the measured change of the line against another line's
    own_measured = np.full(len(rows), np.nan)
    other_measured = np.full(len(rows), np.nan)
    references = {}
    for line, (mine, theirs) in config["references"].items():
        stores = [measured_store(root, name) for name in (mine, theirs)]
        both = stores[0][0].intersection(stores[1][0])
        at = np.flatnonzero((rows["model_id"] == line).to_numpy())
        chosen = rows.iloc[at]
        values = []
        for perturbations, change in stores:
            index = perturbations

            def lookup(models, genes, index=index, change=change):
                return change[index.get_indexer(genes)].astype(np.float64)

            values.append(
                conditional(
                    terms(
                        chosen, line_rows, None, assignment, readers, edit(both, lookup)
                    )
                )  # fmt: skip
            )
        own_measured[at], other_measured[at] = values
        references[line] = {
            "own": mine,
            "other": theirs,
            "perturbed_in_both": len(both),
            "rows": int(np.isfinite(values[0]).sum()),
        }
    measured_covered = np.isfinite(own_measured)
    scores["measured_own"] = (own_measured, measured_covered)
    scores["measured_other"] = (other_measured, measured_covered)

    out = run_dir / "scores"
    out.mkdir(exist_ok=True)
    for name, (values, covered) in scores.items():
        frame = rows[KEYS].assign(score=values, covered=covered).drop_duplicates(KEYS)
        staging = out / f".{name}.parquet.tmp"
        frame.to_parquet(staging, index=False)
        os.replace(staging, out / f"{name}.parquet")

    held = ~train
    rng = np.random.default_rng(seed)
    probe = rng.choice(np.flatnonzero(held & pie_covered), 2000, replace=False)
    probe_lines = raw.index.get_indexer(rows["model_id"].to_numpy()[probe])
    probe_lost = own.genes.get_indexer(rows["gene_a"].to_numpy()[probe])
    keep = probe_lost >= 0
    probe_lines, probe_lost = probe_lines[keep], probe_lost[keep]
    before = base.normalize_pseudobulk(raw.iloc[probe_lines]).to_numpy()
    metrics = {
        "run_id": run_id,
        "rows": {side: int((rows["split"] == side).sum())
                 for side in ("train", "val", "test")},
        "lines_passed": len(lines),
        "panel": len(panel),
        "lost_genes_with_change": len(lost),
        "unit_lines": len(unit_lines),
        "unit_slope_quartiles": np.quantile(slope, [0, 0.25, 0.5, 0.75, 1]).tolist(),
        "zero_change": {"largest_row_difference": gap,
                        "largest_abs_term": float(np.abs(zero_terms).max()),
                        "terms": int(zero_terms.size)},
        "coverage": {
            side: {"simple_edit": float(simple_covered[mask].mean()),
                   "pie": float(pie_covered[mask].mean())}
            for side, mask in (("val", val), ("test", held & ~val), ("held", held))
        },
        "movement": movement(own, probe_lines, probe_lost, before),
        "terms_held_covered": {
            name: float(np.median(np.abs(values[held & pie_covered, 0])))
            for name, values in reading.items()
        },
        "permutation": other,
        "head": selection,
        "references": references,
    }  # fmt: skip
    write_json(run_dir / "metrics.json", jsonable(metrics))
    write_text(run_dir / "results.md", results_markdown(metrics))
    return run_dir


def results_markdown(metrics: Mapping[str, Any]) -> str:
    """The checks of the run; the SL rows are in the benchmark harness's report."""
    zero, move = metrics["zero_change"], metrics["movement"]
    coverage = metrics["coverage"]
    lines = [
        f"# SL backbone, PIE editor: {metrics['run_id']}",
        "",
        f"PIE's change on {metrics['panel']:,} panel genes of "
        f"{metrics['lines_passed']} screened lines for "
        f"{metrics['lost_genes_with_change']:,} lost genes; unit slope fitted on "
        f"{metrics['unit_lines']} SL training lines (quartiles "
        + ", ".join(f"{q:.2f}" for q in metrics["unit_slope_quartiles"]) + ").",
        "",
        f"- Zero change: largest difference from the line's row "
        f"{zero['largest_row_difference']:.2e}; largest |term| "
        f"{zero['largest_abs_term']:.2e} over {zero['terms']:,} terms.",
        "- Held-out rows the editor can edit (validation / test): simple edit "
        f"{coverage['val']['simple_edit']:.1%} / {coverage['test']['simple_edit']:.1%}"
        f", PIE {coverage['val']['pie']:.1%} / {coverage['test']['pie']:.1%}; all "
        f"held-out {coverage['held']['simple_edit']:.1%} against "
        f"{coverage['held']['pie']:.1%}.",
        f"- Movement after normalisation ({move['edits']:,} held-out edits, "
        f"{move['genes']:,} genes): median absolute rank change "
        f"{move['median_abs_rank_change']:.1f}, norm kept {move['norm_kept']:.3f}, "
        f"Pearson raw against normalised change {move['pearson_raw_normalised']:.3f}.",
        "- Median |term of A's loss on B| on covered held-out rows: "
        + ", ".join(f"{k} {v:.2e}" for k, v in metrics["terms_held_covered"].items())
        + ".",
        "",
        "| Reference line | Own | Other | Perturbed in both | Rows |",
        "| --- | --- | --- | ---: | ---: |",
        *(f"| {line} | {r['own']} | {r['other']} | {r['perturbed_in_both']:,} | "
          f"{r['rows']:,} |" for line, r in metrics["references"].items()),
        "",
    ]  # fmt: skip
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("config", type=Path)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    out = run(load_config(absolute(args.config)), args.run_id)
    print(f"SL backbone, PIE editor: {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
