"""The ``report`` step: state identity, the mass deposit and QC explain, recurrence,
stability, determinism and the frozen PIE's state gap, and ``results.md``."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data import cell_states as cs
from src.data.io import write_json
from src.experiments.run_dir import absolute
from src.experiments.states.telemetry import (
    DELTAS,
    QC_NAMES,
    REFERENCE_GROUPS,
    Telemetry,
    load_npz,
    npz_keys,
)
from src.experiments.states.vocabularies import (
    project_line,
    read_memberships,
    vocabulary_labels,
)


def membership_masses(t: Telemetry, second: bool = False) -> dict[str, dict[str, Any]]:
    """Per line (or second deposit), every membership key's masses and the cell
    count, each memberships file read once."""

    def make() -> dict[str, dict[str, Any]]:
        lines = list(t.second_sources) if second else t.projected
        out = {}
        for model_id in lines:
            memberships = read_memberships(t, model_id, second)
            out[model_id] = {key: cs.masses(q) for key, q in memberships.items()}
            out[model_id]["n"] = next(iter(memberships.values())).shape[0]
        return out

    return t.cached(("masses", second), make)


def line_masses(
    t: Telemetry, vocabulary: str, seed: int, lines: Sequence[str], second: bool = False
) -> pd.DataFrame:
    """Lines x states masses (unmerged) of one vocabulary at one seed."""
    key = cs.membership_key(vocabulary, seed)
    table = membership_masses(t, second)
    rows = {m: table[m][key] for m in lines}
    return pd.DataFrame(rows, index=vocabulary_labels(t, seed, vocabulary)).T


def identity_section(t: Telemetry, vocabulary: str) -> dict[str, Any]:
    """Each state's mass-weighted mean, over training lines, of its cells'
    line-centred reference-program scores; the three highest programs and every
    group's best."""
    lines = t.train
    labels = vocabulary_labels(t, t.seed, vocabulary)
    table = membership_masses(t)
    key = cs.membership_key(vocabulary, t.seed)
    stacked, weights, shares = [], [], []
    names: list[str] = []
    for model_id in lines:
        line = npz_keys(
            t.layout.line(model_id),
            "reference_programs",
            "reference_listed",
            f"identity__{vocabulary}",
        )
        names = [str(n) for n in line["reference_programs"]]
        shares.append(line["reference_listed"])
        stacked.append(np.nan_to_num(line[f"identity__{vocabulary}"]))
        weights.append(table[model_id][key] * table[model_id]["n"])
    weights = np.asarray(weights)
    total = weights.sum(axis=0)
    mean = (
        np.einsum("lk,lkp->kp", weights, np.asarray(stacked))
        / np.maximum(total, 1e-12)[:, None]
    )
    frame = pd.DataFrame(mean, index=labels, columns=names)
    listed = dict(zip(names, np.mean(shares, axis=0).tolist(), strict=True))
    groups = {
        group: [n for n in names if n.startswith(f"{group}:")]
        for group in REFERENCE_GROUPS
    }
    states = {}
    for label in labels:
        row = frame.loc[label].sort_values(ascending=False)
        states[label] = {
            "top": [[name, float(value)] for name, value in row.head(3).items()],
            "best_per_group": {
                group: [best, float(row[best])]
                for group, members in groups.items()
                if members
                for best in [row[members].idxmax()]
            },
        }
    return {
        "states": states,
        "programs_scored": {group: len(members) for group, members in groups.items()},
        # The share of each group's members the training lines' sources list.
        "members_listed": {
            group: float(np.mean([listed[n] for n in members]))
            for group, members in groups.items()
            if members
        },
    }


def line_qc(t: Telemetry) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Each training line's median and SD of log UMI, genes detected and
    mitochondrial fraction over its cells."""

    def make() -> dict[str, tuple[np.ndarray, np.ndarray]]:
        out = {}
        for model_id in t.train:
            qc = t.cell_arrays(model_id, "qc")[0][:, list(cs.QC_FEATURES)]
            qc = qc.astype(np.float64)
            out[model_id] = (np.median(qc, axis=0), qc.std(axis=0))
        return out

    return t.cached("line_qc", make)


def per_state_group_share(
    frame: pd.DataFrame,
    groups: pd.Series,
    strata: pd.Series,
    *,
    permutations: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    """Per state (column of lines x states masses), the share of its mass's
    variance across lines explained by ``groups`` (eta squared), against
    ``groups`` permuted among the lines of each stratum (the noise diagnosis's
    null, state by state)."""
    from src.context_prior.noise import between_share

    lines = list(frame.index)
    labels = groups.loc[lines].astype(str).to_numpy()
    levels = np.asarray(sorted(set(labels)))
    values = frame.to_numpy(dtype=np.float64)
    finite = np.isfinite(values).astype(np.float64)
    values = np.where(finite > 0, values, 0.0)
    stratum = strata.loc[lines].to_numpy()
    blocks = [np.flatnonzero(stratum == s) for s in sorted(set(stratum))]

    def share(assigned: np.ndarray) -> np.ndarray:
        onehot = (assigned[:, None] == levels[None, :]).astype(np.float64)
        return between_share(values, finite, onehot)

    observed = share(labels)
    rng = np.random.default_rng(seed)
    null = np.empty((permutations, values.shape[1]))
    for k in range(permutations):
        shuffled = labels.copy()
        for block in blocks:
            shuffled[block] = labels[rng.permutation(block)]
        null[k] = share(shuffled)
    # A state whose mass does not vary has no defined share: NaN, not a p-value.
    return {
        str(state): {
            "observed": float(observed[j]),
            "null_q95": float(np.nanquantile(null[:, j], 0.95))
            if np.isfinite(observed[j])
            else float("nan"),
            "p_value": float(
                (1 + (null[:, j] >= observed[j]).sum()) / (1 + permutations)
            )
            if np.isfinite(observed[j])
            else float("nan"),
        }
        for j, state in enumerate(frame.columns)
    }


def confound_section(
    t: Telemetry, vocabulary: str, frame: pd.DataFrame
) -> dict[str, Any]:
    """Mass explained by deposit (eta squared against permutations within
    lineage) and by the lines' median QC (R squared), and each state's
    within-line QC shift in pooled within-line SD units."""
    from src.context_prior.noise import group_share

    train = list(frame.index)
    deposit = t.deposits.loc[train]
    counts = deposit.value_counts()
    small = set(counts.index[counts < int(t.setup.noise["deposit"]["min_lines"])])
    grouped = deposit.where(~deposit.isin(small), "other")
    lineage = t.lines.loc[train, "lineage"].astype(str)
    share = group_share(
        frame,
        grouped,
        lineage,
        permutations=int(t.config["report"]["permutations"]),
        seed=t.seed,
    )
    per_state = per_state_group_share(
        frame,
        grouped,
        lineage,
        permutations=int(t.config["report"]["permutations"]),
        seed=t.seed,
    )
    summaries = line_qc(t)
    medians, spreads, shifts, weights = [], [], [], []
    for model_id in train:
        medians.append(summaries[model_id][0])
        spreads.append(summaries[model_id][1])
        key = f"qcshift__{vocabulary}"
        shifts.append(np.nan_to_num(npz_keys(t.layout.line(model_id), key)[key]))
        weights.append(frame.loc[model_id].to_numpy())
    design = np.column_stack([np.ones(len(train)), np.asarray(medians)])
    r2 = {}
    for state in frame.columns:
        y = frame[state].to_numpy()
        fitted = design @ np.linalg.lstsq(design, y, rcond=None)[0]
        total = ((y - y.mean()) ** 2).sum()
        r2[state] = (
            float(1 - ((y - fitted) ** 2).sum() / total) if total > 0 else float("nan")
        )
    pooled_sd = np.sqrt(np.mean(np.asarray(spreads) ** 2, axis=0))
    w = np.asarray(weights)
    shift = (
        np.einsum("lk,lkq->kq", w, np.asarray(shifts))
        / np.maximum(w.sum(axis=0), 1e-12)[:, None]
    )
    return {
        "deposit_per_state": per_state,
        # The mean over states of the same share (noise.group_share).
        "deposit": share,
        "pooled_as_other": sorted(small),
        "qc_r2": r2,
        "qc_shift_sd": {
            state: dict(zip(QC_NAMES, (shift[k] / pooled_sd).tolist(), strict=True))
            for k, state in enumerate(frame.columns)
        },
    }


def composition_section(
    frame: pd.DataFrame, mass: float, floor: float
) -> dict[str, Any]:
    """Recurrence (share of lines holding each state above ``mass``), each
    state's spread across lines, and the singular values of the centred
    log-ratio masses (its K - 1 contrasts)."""
    values = np.maximum(frame.to_numpy(dtype=np.float64), floor)
    values /= values.sum(axis=1, keepdims=True)
    clr = np.log(values) - np.log(values).mean(axis=1, keepdims=True)
    clr -= clr.mean(axis=0)
    singular = np.linalg.svd(clr, compute_uv=False)[: max(frame.shape[1] - 1, 1)]
    energy = singular**2
    held = (frame.to_numpy() >= mass).sum(axis=1)
    return {
        "recurrence": {s: float((frame[s] >= mass).mean()) for s in frame.columns},
        "mass_mean": {s: float(frame[s].mean()) for s in frame.columns},
        "mass_sd": {s: float(frame[s].std()) for s in frame.columns},
        "states_per_line_median": float(np.median(held)),
        "clr_variance_share": (energy / energy.sum()).tolist()
        if energy.sum() > 0
        else [],
    }


def second_section(t: Telemetry, vocabulary: str) -> dict[str, Any] | None:
    """Agreement of state masses between a line's two deposits: per state the
    Pearson correlation across lines, per line the total-variation distance,
    beside the distance between different lines' primary deposits."""
    key = cs.membership_key(vocabulary, t.seed)
    seconds = membership_masses(t, second=True)
    lines = [m for m in t.second_sources if key in seconds[m]]
    if len(lines) < 3:
        return None
    primary = line_masses(t, vocabulary, t.seed, lines)
    second = line_masses(t, vocabulary, t.seed, lines, second=True)
    distance = 0.5 * np.abs(primary.to_numpy() - second.to_numpy()).sum(axis=1)
    values = primary.to_numpy()
    other = [
        0.5 * np.abs(values[i] - values[j]).sum()
        for i in range(len(lines))
        for j in range(i + 1, len(lines))
    ]
    pearson = {}
    for state in primary.columns:
        a, b = primary[state].to_numpy(), second[state].to_numpy()
        pearson[state] = (
            float(np.corrcoef(a, b)[0, 1])
            if a.std() > 0 and b.std() > 0
            else float("nan")
        )
    return {
        "lines": len(lines),
        "pearson": pearson,
        "tvd_median": float(np.median(distance)),
        "tvd_between_lines_median": float(np.median(other)),
    }


def stability_section(t: Telemetry, vocabulary: str) -> dict[str, Any]:
    """States of each pair of fitting seeds matched by the Hungarian method on
    the correlation of their memberships over a fixed draw of cells per training
    line (``matched_states``)."""
    from src.data.ladder_cells import line_rng

    def draw() -> dict[str, np.ndarray]:
        """Every membership key's rows of a fixed draw of cells per line, each
        memberships file read once."""
        per_line = int(t.config["report"]["stability_cells"])
        parts: dict[str, list[np.ndarray]] = {}
        for model_id in t.train:
            memberships = read_memberships(t, model_id)
            n = next(iter(memberships.values())).shape[0]
            rows = line_rng(model_id, 0, "state_stability").permutation(n)[:per_line]
            for key, q in memberships.items():
                parts.setdefault(key, []).append(q[rows])
        return {key: np.vstack(values) for key, values in parts.items()}

    drawn = t.cached("stability_draw", draw)
    table = membership_masses(t)
    stacked = {s: drawn[cs.membership_key(vocabulary, s)] for s in t.seeds}
    line_masses_by_seed = {
        s: [table[m][cs.membership_key(vocabulary, s)] for m in t.train]
        for s in t.seeds
    }
    return {
        f"{a}-{b}": matched_states(
            stacked[a],
            stacked[b],
            np.asarray(line_masses_by_seed[a]),
            np.asarray(line_masses_by_seed[b]),
        )
        for i, a in enumerate(t.seeds)
        for b in t.seeds[i + 1 :]
    }


def matched_states(
    qa: np.ndarray, qb: np.ndarray, ma: np.ndarray, mb: np.ndarray
) -> dict[str, Any]:
    """Two fits' states matched by the Hungarian method on the correlation of
    their memberships (cells x states ``qa``, ``qb``); per line (lines x states
    masses ``ma``, ``mb``), the total-variation distance with every unmatched
    state set against a zero-mass counterpart, and the unmatched mass."""
    from scipy.optimize import linear_sum_assignment

    ka = qa.shape[1]
    corr = np.nan_to_num(np.corrcoef(qa.T, qb.T)[:ka, ka:])
    rows, cols = linear_sum_assignment(-corr)
    unmatched_a = np.setdiff1d(np.arange(ka), rows)
    unmatched_b = np.setdiff1d(np.arange(qb.shape[1]), cols)
    unmatched = ma[:, unmatched_a].sum(axis=1) + mb[:, unmatched_b].sum(axis=1)
    distance = 0.5 * (np.abs(ma[:, rows] - mb[:, cols]).sum(axis=1) + unmatched)
    return {
        "matched_correlation": corr[rows, cols].tolist(),
        "matched_mass_tvd_mean": float(distance.mean()),
        "unmatched_states": [int(unmatched_a.size), int(unmatched_b.size)],
        "unmatched_mass_mean": float(unmatched.mean()),
        "states": [int(ka), int(qb.shape[1])],
    }


def determinism_section(t: Telemetry) -> dict[str, Any]:
    """Validation lines from a deposit no training line comes from, projected
    again: whether their memberships are identical to the stored ones."""
    train_deposits = set(t.deposits.loc[t.train])
    unseen = [
        m
        for m in t.projected
        if t.lines.loc[m, "role"] != "train_cells"
        and t.deposits[m] not in train_deposits
    ]
    out = {}
    for model_id in unseen:
        again = project_line(t, model_id)
        stored = load_npz(t.layout.memberships(model_id))
        out[model_id] = bool(
            set(again) == set(stored)
            and all(np.array_equal(again[k], stored[k]) for k in stored)
        )
    return {"lines": out, "deposits": {m: str(t.deposits[m]) for m in unseen}}


def _main_pass(
    pie_root: Path, gene: str, line_index: np.ndarray, basis: Mapping[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """The main pass's uncentred deltas of a gene on the given lines."""
    shard = load_npz(pie_root / "genes" / f"{gene}.npz")
    if "dE" in shard:
        dE = shard["dE"]
    else:
        full = np.load(pie_root / "full" / f"{gene}.npy")
        dE = (full - basis["mean"]) @ basis["components"].T
    return {
        "dH": shard["dH"][line_index],
        "dE": np.asarray(dE)[line_index],
        "own_h": shard["own_h"][line_index],
        "own_dp": np.asarray(shard["own_dp"]).reshape(-1, 1)[line_index],
    }


def gap_section(t: Telemetry, set_name: str) -> dict[str, Any] | None:
    """The frozen PIE's gap ``sum_k pi_k D(x_k) - D(x)`` per (line, gene), for
    each delta, against the set's own ``|whole`` context; its size and its
    spread across lines within a gene, each relative to the spread across lines
    of the whole-line delta within the gene; and the noise floor, the
    ``|whole`` context against the main pass, on the same scale."""
    folder = t.layout.pie_states / set_name
    if not (folder / "index.json").is_file():
        return None
    index = json.loads((folder / "index.json").read_text())
    current = cs.set_fingerprints(
        pd.read_parquet(t.layout.state_set(set_name) / "inputs.parquet"),
        pd.read_parquet(t.layout.state_set(set_name) / "masses.parquet"),
    )
    if index.get("fingerprints") != current:
        raise ValueError(
            f"{folder} was computed from other state inputs than "
            f"{t.layout.state_set(set_name)}; move it aside and rerun the pass"
        )
    contexts = pd.read_parquet(folder / "contexts.parquet")
    genes = list(index["genes"])
    pie_root = t.ladder_root / "pie"
    basis = load_npz(pie_root / "basis.npz")
    means = load_npz(pie_root / "frame_means.npz")
    panel_position = {g: i for i, g in enumerate(t.panel)}
    lines = list(dict.fromkeys(contexts["model_id"]))
    line_index = np.asarray([t.lines.index.get_loc(m) for m in lines])
    roles = t.lines.loc[lines, "role"].to_numpy()
    owner = contexts["model_id"].to_numpy()
    is_whole = contexts["state"].to_numpy() == cs.WHOLE
    mass = np.nan_to_num(contexts["mass"].to_numpy(dtype=np.float64))
    # Rows of each line's whole context, and the lines x contexts mixing matrix.
    whole_rows = np.asarray([np.flatnonzero((owner == m) & is_whole)[0] for m in lines])
    mixing = np.zeros((len(lines), len(contexts)))
    for i, model_id in enumerate(lines):
        states = (owner == model_id) & ~is_whole
        mixing[i, states] = mass[states]
    out: dict[str, Any] = {"lines": len(lines), "genes": len(genes)}
    for name in DELTAS:
        values = np.load(folder / f"{name}.npy", mmap_mode="r")
        # Per role: gap energy, gap spread, whole-line spread, noise energy.
        sums = {role: np.zeros(4) for role in ("train_cells", "val")}
        for g, gene in enumerate(genes):
            array = np.asarray(values[g], dtype=np.float64).reshape(len(contexts), -1)
            array = array + means[name][panel_position[gene]]
            whole = array[whole_rows]
            gap = mixing @ array - whole
            main = _main_pass(pie_root, gene, line_index, basis)[name]
            noise = whole - np.asarray(main, dtype=np.float64).reshape(len(lines), -1)
            for role, total in sums.items():
                keep = roles == role
                if keep.sum() < 2:
                    continue
                total += [
                    (gap[keep] ** 2).sum(),
                    ((gap[keep] - gap[keep].mean(axis=0)) ** 2).sum(),
                    ((whole[keep] - whole[keep].mean(axis=0)) ** 2).sum(),
                    (noise[keep] ** 2).sum(),
                ]
        out[name] = {
            role: {
                "gap_ratio": float(total[0] / total[2]),
                "gap_spread_ratio": float(total[1] / total[2]),
                "noise_ratio": float(total[3] / total[2]),
            }
            for role, total in sums.items()
            if total[2] > 0
        }
    run = t.layout.pie_states / "run.json"
    if run.is_file():
        timing = json.loads(run.read_text()).get("sets", {}).get(set_name)
        out["seconds_per_context"] = timing
    return out


def step_report(t: Telemetry) -> dict[str, Any]:
    out_dir = absolute(t.config["output_root"]) / t.name
    report: dict[str, Any] = {"vocabularies": {}, "pie_gap": {}}
    for vocabulary in cs.vocabularies(t.config):
        frame = line_masses(t, vocabulary, t.seed, t.train)
        report["vocabularies"][vocabulary] = {
            "states": list(frame.columns),
            "identity": identity_section(t, vocabulary),
            "confound": confound_section(t, vocabulary, frame),
            "composition": composition_section(
                frame,
                float(t.config["report"]["recurrence_mass"]),
                float(t.config["report"]["clr_floor"]),
            ),
            "second_deposit": second_section(t, vocabulary),
            "stability": stability_section(t, vocabulary),
        }
    report["determinism"] = determinism_section(t)
    for set_name in cs.state_sets(t.config):
        report["pie_gap"][set_name] = gap_section(t, set_name)
    write_json(out_dir / "report.json", report, keep_nan=True)
    (out_dir / "results.md").write_text(results_markdown(t, report))
    return {"output": str(out_dir)}


def _f(value: Any, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "–"
    return f"{value:.{digits}f}"


def results_markdown(t: Telemetry, report: Mapping[str, Any]) -> str:
    lines = [
        f"# State telemetry ({t.name})",
        "",
        "Frozen state vocabularies fitted on the single-cell training lines (at "
        "most 500 distinct cells each) and projected on the lines of "
        f"{', '.join(t.config['project']['roles'])}; the frozen PIE on their state "
        "inputs. Telemetry only: nothing here says whether states help predict "
        "GeneEffect, and nothing is chosen in code.",
        "",
        "## Composition, confound and stability (training lines, first fitting seed)",
        "",
        "Deposit η²: the mean over states of the share of each state's mass across "
        "lines explained by deposit (null q95), then the largest state's; per state "
        "in the identity tables. Seed match: the matched states' mean correlation "
        "(the mean unmatched mass per line, when the fits' state counts differ).",
        "",
        "| Vocabulary | States | States per line ≥ mass | Deposit η² mean (null q95); "
        "max state | QC R² max | Second deposit TVD (between lines) | "
        "Seed match (unmatched mass) |",
        "| --- | ---: | ---: | --- | ---: | --- | --- |",
    ]
    for vocabulary, entry in report["vocabularies"].items():
        deposit = entry["confound"]["deposit"]
        per_state = entry["confound"]["deposit_per_state"]
        largest = max((v["observed"] for v in per_state.values()), default=None)
        r2 = [v for v in entry["confound"]["qc_r2"].values() if np.isfinite(v)]
        second = entry["second_deposit"]
        stability = list(entry["stability"].values())
        matched = [np.mean(p["matched_correlation"]) for p in stability]
        unmatched = [p["unmatched_mass_mean"] for p in stability]
        agreement = (
            f"{_f(second['tvd_median'])} ({_f(second['tvd_between_lines_median'])}), "
            f"{second['lines']} lines"
            if second
            else "–"
        )
        cells = [
            vocabulary,
            str(len(entry["states"])),
            _f(entry["composition"]["states_per_line_median"], 1),
            f"{_f(deposit['observed'])} ({_f(deposit['null_q95'])}); {_f(largest)}",
            _f(max(r2) if r2 else None),
            agreement,
            f"{_f(float(np.mean(matched)) if matched else None)} "
            f"({_f(float(np.mean(unmatched)) if unmatched else None)})",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "## Identity (best reference program per state)", ""]
    for vocabulary, entry in report["vocabularies"].items():
        lines += [
            f"### {vocabulary}",
            "",
            "| State | Mass (mean) | Deposit η² (null q95) | Top programs |",
            "| --- | ---: | --- | --- |",
        ]
        composition = entry["composition"]
        per_state = entry["confound"]["deposit_per_state"]
        for state, identity in entry["identity"]["states"].items():
            top = "; ".join(f"{name} {value:+.3f}" for name, value in identity["top"])
            share = per_state[state]
            cells = [
                state,
                _f(composition["mass_mean"][state]),
                f"{_f(share['observed'])} ({_f(share['null_q95'])})",
                top,
            ]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    lines += [
        "## The frozen PIE's gap",
        "",
        "Pooled over the genes and the role's lines: the gap's energy, and its "
        "spread across lines within a gene, over the spread across lines of the "
        "whole-line delta within the gene; the noise floor is the set's `|whole` "
        "context against the main pass on the same scale.",
        "",
        "| Set | Delta | Gap / spread (train) | Gap spread (train) | Noise (train) | "
        "Gap / spread (val) | Noise (val) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for set_name, entry in report["pie_gap"].items():
        if entry is None:
            lines.append(f"| {set_name} | not computed | | | | | |")
            continue
        for name in DELTAS:
            train = entry[name].get("train_cells", {})
            val = entry[name].get("val", {})
            cells = [
                set_name,
                name,
                _f(train.get("gap_ratio"), 4),
                _f(train.get("gap_spread_ratio"), 4),
                _f(train.get("noise_ratio"), 4),
                _f(val.get("gap_ratio"), 4),
                _f(val.get("noise_ratio"), 4),
            ]
            lines.append("| " + " | ".join(cells) + " |")
    determinism = report["determinism"]["lines"]
    lines += [
        "",
        "## Projection of lines from deposits no training line comes from",
        "",
        f"{sum(determinism.values())} of {len(determinism)} such lines project "
        "identically when projected again.",
        "",
    ]
    return "\n".join(lines)
