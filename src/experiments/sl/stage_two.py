"""Stage two, first run (SL protocol §10), two heads on the SL screen benchmark: a
viability-embedding head, an MLP over stage one's 32-d V(c, A) and V(c, B), and a PARIS
head, the same MLP over PARIS's pair evidence (fitted on the reference lines' measured
GeneEffect) beside the arm's predicted GeneEffect of A and B in line c. Both are fitted
on the SL training lines and scored per held-out line and screen against
pair-identity, pan-essentiality, A-blind and context-ablated controls."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.io import read_json, write_json, write_parquet, write_text
from src.experiments.config import read_yaml, require_keys
from src.experiments.run_dir import bind, number
from src.experiments.sl.paris import reference_lines

SCREENS = ("harle_2025", "ito_2021", "klingbeil_2024")
VIABILITY_WIDTH = 32
#: An arm is a stage-one report row (``prior`` or ``run``) or the measurement.
ARM_KEYS = {"prior": {"kind", "run"}, "run": {"kind", "config"}, "measured": {"kind"}}
CONTROLS = ("pair_identity", "pan_essentiality", "a_blind")
PARIS_COLUMNS = ("importance_max", "importance_sum", "r_sum", "r_gap", "selected")


def load_config(path: Path) -> dict[str, Any]:
    config = read_yaml(path)
    require_keys(
        config,
        {"kind", "seed", "output_root", "task", "ladder_config", "benchmark", "paris",
         "arms", "head", "top_k", "bootstrap"},
        "config",
    )  # fmt: skip
    if config["kind"] != "sl_stage_two":
        raise ValueError(f"{path} is a {config['kind']} config")
    for name, arm in config["arms"].items():
        if arm.get("kind") not in ARM_KEYS:
            raise ValueError(f"arms.{name}.kind must be one of {sorted(ARM_KEYS)}")
        require_keys(arm, ARM_KEYS[arm["kind"]], f"arms.{name}")
    require_keys(
        config["head"],
        {"hidden", "learning_rate", "weight_decay", "batch_size", "max_epochs"},
        "head",
    )
    return config


# --- inputs ---------------------------------------------------------------------


def head_input(v_a: np.ndarray, v_b: np.ndarray, screens: pd.Series) -> np.ndarray:
    """``[V_A + V_B ; |V_A − V_B| ; screen one-hot]``: the pair's order cannot
    matter."""
    return np.concatenate(
        [v_a + v_b, np.abs(v_a - v_b), screen_onehot(screens)], axis=1
    )


def screen_onehot(screens: pd.Series) -> np.ndarray:
    return np.stack([(screens == s).to_numpy(float) for s in SCREENS], axis=1)


def paris_features(selected: pd.DataFrame, rows: pd.DataFrame) -> np.ndarray:
    """PARIS's evidence for each row's pair in both directions (A's expression for
    B's dependency, B's for A's), order-free: the larger and summed importance, the
    summed r and its gap, and how many directions PARIS selected; an unselected
    direction reads importance 0 and r 0."""
    table = selected.set_index(["dependency_gene", "feature_gene"])[["importance", "r"]]
    ab = table.reindex(pd.MultiIndex.from_arrays([rows["gene_b"], rows["gene_a"]]))
    ba = table.reindex(pd.MultiIndex.from_arrays([rows["gene_a"], rows["gene_b"]]))
    count = ab["importance"].notna().to_numpy(int) + ba["importance"].notna().to_numpy(
        int
    )
    ab, ba = ab.fillna(0.0).to_numpy(), ba.fillna(0.0).to_numpy()
    return np.stack(
        [np.maximum(ab[:, 0], ba[:, 0]), ab[:, 0] + ba[:, 0], ab[:, 1] + ba[:, 1],
         np.abs(ab[:, 1] - ba[:, 1]), count],
        axis=1,
    )  # fmt: skip


def effect_features(effects: pd.DataFrame, rows: pd.DataFrame) -> np.ndarray:
    """The two genes' GeneEffect in the row's line, order-free: sum, gap, minimum."""
    line = effects.index.get_indexer(rows["model_id"])
    if (line < 0).any():
        raise ValueError("a row's line has no GeneEffect row")
    values = effects.to_numpy(dtype=np.float64)
    a = values[line, effects.columns.get_indexer(rows["gene_a"])]
    b = values[line, effects.columns.get_indexer(rows["gene_b"])]
    return np.stack([a + b, np.abs(a - b), np.minimum(a, b)], axis=1)


def context_weights(contexts: pd.Series, labels: pd.Series) -> np.ndarray:
    """Every (screen, line) context weighs 1, split equally between its positives
    and negatives, so neither a large screen nor the majority class dominates."""
    frame = pd.DataFrame({"context": contexts.to_numpy(), "label": labels.to_numpy()})
    count = frame.groupby(["context", "label"])["label"].transform("size")
    classes = frame.groupby("context")["label"].transform("nunique")
    return (1.0 / (count * classes)).to_numpy(dtype=np.float64)


def pair_identity(training: pd.DataFrame, rows: pd.DataFrame) -> np.ndarray:
    """The pair's hit rate over the same screen's training lines; NaN when the
    pair was never labelled there."""
    keys = ["screen", "gene_a", "gene_b"]
    rate = training.groupby(keys)["label"].mean()
    return rate.reindex(pd.MultiIndex.from_frame(rows[keys])).to_numpy()


def varying_pairs(training: pd.DataFrame, rows: pd.DataFrame) -> np.ndarray:
    """Pairs labelled both SL and not SL among the same screen's training lines."""
    keys = ["screen", "gene_a", "gene_b"]
    both = training.groupby(keys)["label"].nunique() > 1
    return both.reindex(
        pd.MultiIndex.from_frame(rows[keys]), fill_value=False
    ).to_numpy(bool)


# --- the head -------------------------------------------------------------------


def fit_head(
    train: tuple[np.ndarray, np.ndarray, np.ndarray],
    validation: Sequence[tuple[np.ndarray, np.ndarray]],
    params: Mapping[str, Any],
    seed: int,
) -> tuple[Any, dict[str, Any]]:
    """MLP ``Linear(hidden) → ReLU → Linear(1)`` on training-standardised inputs,
    minimising context-weighted binary cross-entropy; the epoch kept is the one with
    the highest mean AUPRC lift over the validation contexts (``(x, label)`` each).
    Returns ``score(x)`` and the selection record."""
    import torch
    from torch import nn

    torch.manual_seed(seed)
    x, y, w = train
    center, scale = x.mean(axis=0), x.std(axis=0)
    scale = np.where(scale > 0, scale, 1.0)

    def tensor(values: np.ndarray) -> torch.Tensor:
        return torch.as_tensor((values - center) / scale, dtype=torch.float32)

    model = nn.Sequential(
        nn.Linear(x.shape[1], int(params["hidden"])),
        nn.ReLU(),
        nn.Linear(int(params["hidden"]), 1),
    )
    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=float(params["learning_rate"]),
        weight_decay=float(params["weight_decay"]),
    )
    inputs, targets = tensor(x), torch.as_tensor(y, dtype=torch.float32)
    weights = torch.as_tensor(w / w.mean(), dtype=torch.float32)
    generator = torch.Generator().manual_seed(seed)

    def score(values: np.ndarray) -> np.ndarray:
        model.eval()
        with torch.no_grad():
            return model(tensor(values)).squeeze(-1).numpy().astype(np.float64)

    def lift() -> float:
        lifts = [
            ranking_metrics(lab, score(v), 1).get("auprc_lift") for v, lab in validation
        ]
        return float(np.mean([v for v in lifts if v is not None]))

    best, best_lift, best_epoch, history = None, -np.inf, 0, []
    for epoch in range(1, int(params["max_epochs"]) + 1):
        model.train()
        order = torch.randperm(len(inputs), generator=generator)
        for batch in order.split(int(params["batch_size"])):
            loss = nn.functional.binary_cross_entropy_with_logits(
                model(inputs[batch]).squeeze(-1), targets[batch], weight=weights[batch]
            )
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        value = lift()
        history.append(value)
        if value > best_lift:
            best_lift, best_epoch = value, epoch
            best = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best)
    return score, {
        "epoch": best_epoch,
        "validation_lift": best_lift,
        "history": history,
    }


# --- metrics --------------------------------------------------------------------


def ranking_metrics(label: np.ndarray, score: np.ndarray, k: int) -> dict[str, Any]:
    from sklearn.metrics import roc_auc_score

    from src.eval.metrics import average_precision

    keep = np.isfinite(score)
    label, score = label[keep].astype(int), score[keep]
    positives = int(label.sum())
    out: dict[str, Any] = {"pairs": int(len(label)), "positives": positives}
    if not 0 < positives < len(label):
        return out
    rate = positives / len(label)
    auprc = average_precision(label, score)
    top = label[np.argsort(-score, kind="stable")[:k]]
    return out | {
        "positive_rate": rate,
        "auprc": auprc,
        "auprc_lift": auprc / rate,
        "auroc": float(roc_auc_score(label, score)),
        "precision_at_k": float(top.mean()),
        "recall_at_k": float(top.sum() / positives),
    }


def dyadic_difference(
    label: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    gene_a: np.ndarray,
    gene_b: np.ndarray,
    reps: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """AUPRC(first) − AUPRC(second) with a two-way dyadic bootstrap interval: genes
    are resampled and a pair weighs the product of its two genes' counts, since a
    pair has two endpoints (SL protocol §7)."""
    from sklearn.metrics import average_precision_score

    keep = np.isfinite(first) & np.isfinite(second)
    label, first, second = label[keep].astype(int), first[keep], second[keep]
    genes, index = np.unique(
        np.concatenate([gene_a[keep], gene_b[keep]]), return_inverse=True
    )
    left, right = index[: keep.sum()], index[keep.sum() :]

    def difference(weight: np.ndarray | None) -> float:
        return float(
            average_precision_score(label, first, sample_weight=weight)
            - average_precision_score(label, second, sample_weight=weight)
        )

    draws = []
    for _ in range(reps):
        counts = rng.multinomial(len(genes), np.full(len(genes), 1.0 / len(genes)))
        weight = (counts[left] * counts[right]).astype(np.float64)
        chosen = label[weight > 0]
        if 0 < chosen.sum() < len(chosen):
            draws.append(difference(weight))
    low, high = np.quantile(draws, [0.025, 0.975]) if draws else (np.nan, np.nan)
    return {"difference": difference(None), "interval": [float(low), float(high)]}


# --- the run --------------------------------------------------------------------


def arm_effects(setup: Any, arm: Mapping[str, Any], ladder_config: str) -> pd.DataFrame:
    """The arm's GeneEffect on the validation, test and (out of fold) single-cell
    training lines, read as the stage-one report reads a row; the measured arm reads
    DepMap. GeneEffect is in the same units across fits, unlike V."""
    from src.experiments.stage_one.report import row_predictions

    ladder = setup.ladder
    if arm["kind"] == "measured":
        units = setup.truth(ladder.role_lines("val", "test", "train_cells"))
    else:
        parts = row_predictions(setup, arm, {}, ladder_config)
        if any(part is None for part in parts.values()):
            raise FileNotFoundError(f"{arm} is not a finished row")
        units = pd.concat(parts.values())
    genes = ladder.genes
    units = units[[str(g) for g in genes.index]]
    return units * genes["sigma_g"].to_numpy() + genes["mu_g"].to_numpy()


class Viability:
    """A stage-one run's V from its main run, for every line: each fold run is
    another network whose V is in other coordinates (median per-dimension
    correlation 0.3 with the main run's), so one head cannot read both. A training
    line's V is in sample for stage one (fitted on its GeneEffect, never on SL
    labels)."""

    def __init__(self, config: str, ladder_config: str):
        from src.experiments.stage_one.config import load_stage_one_config
        from src.experiments.stage_one.report import finished_run

        path = Path(config)
        run_dir = finished_run(
            load_stage_one_config(path), path.stem, None, ladder_config
        )
        if run_dir is None:
            raise FileNotFoundError(f"{config}: its main run is not finished")
        meta = read_json(run_dir / "V.json")
        self.rows = {m: i for i, m in enumerate(meta["lines"])}
        self.genes = {g: i for i, g in enumerate(meta["genes"])}
        self.values = np.load(run_dir / "V.npy", mmap_mode="r")

    def pairs(self, rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """V(c, gene_a) and V(c, gene_b) of every row, in float32."""
        out_a = np.empty((len(rows), VIABILITY_WIDTH), np.float32)
        out_b = np.empty_like(out_a)
        a = rows["gene_a"].map(self.genes).to_numpy()
        b = rows["gene_b"].map(self.genes).to_numpy()
        for model_id, index in rows.groupby("model_id").indices.items():
            values = self.values[self.rows[model_id]]
            out_a[index], out_b[index] = values[a[index]], values[b[index]]
        return out_a, out_b

    def mean(self, lines: Sequence[str]) -> np.ndarray:
        """Each gene's V averaged over ``lines`` (genes x 32)."""
        total = np.zeros((len(self.genes), VIABILITY_WIDTH), np.float64)
        for model_id in lines:
            total += self.values[self.rows[model_id]]
        return (total / len(lines)).astype(np.float32)


def benchmark_rows(
    path: Path, frame: Sequence[str], held: Sequence[str], genes: Sequence[str]
) -> dict[str, Any]:
    """Labelled rows whose two genes stage one decodes: training rows of the
    single-cell stratum (frame lines), and held-out rows whose pair was labelled in
    a training line of the same screen (the pair-identity control's universe)."""
    pairs = pd.read_parquet(path)
    pairs = pairs[pairs["label"].notna()]
    pairs = pairs.assign(
        label=pairs["label"].astype(int),
        decoded=pairs["gene_a"].isin(genes) & pairs["gene_b"].isin(genes),
    )
    training = pairs[(pairs["split"] == "train") & (pairs["stratum"] == "single_cell")]
    evaluated = pairs[pairs["split"].isin(["val", "test"])]
    if not set(training["model_id"]) <= set(frame):
        raise ValueError("a single-cell SL training line is not a stage-one frame line")
    if not set(evaluated["model_id"]) <= set(held):
        raise ValueError("an SL held-out line is not a stage-one held-out line")
    if set(evaluated["model_id"]) & set(training["model_id"]):
        raise ValueError("a line is both an SL training and an SL held-out line")
    seen = pd.Series(pair_identity(training, evaluated), index=evaluated.index).notna()
    return {
        "training": training[training["decoded"]].reset_index(drop=True),
        "held": evaluated[seen & evaluated["decoded"]].reset_index(drop=True),
        "dropped": {
            "training_rows_gene_not_decoded": int((~training["decoded"]).sum()),
            "held_rows_gene_not_decoded": int((~evaluated["decoded"]).sum()),
            "held_rows_pair_not_in_training": int((~seen).sum()),
        },
    }


def run(config: Mapping[str, Any], run_id: str) -> Path:
    from src.experiments.stage_one.setup import load_setup

    run_dir = Path(config["output_root"]) / config["task"] / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    bind(run_dir, config)
    setup = load_setup(config["ladder_config"])
    ladder = setup.ladder
    frame, held = ladder.frame, ladder.role_lines("val", "test")
    seed = int(config["seed"])
    # One universe for every arm: the decode panel, the genes V exists for.
    rows = benchmark_rows(Path(config["benchmark"]), frame, held, ladder.panel)
    training, evaluated = rows["training"], rows["held"]
    reference = reference_lines(ladder, pd.read_parquet(config["benchmark"]))
    validation = evaluated[evaluated["split"] == "val"]
    groups = list(validation.groupby(["screen", "model_id"]).indices.values())
    weights = context_weights(
        training["screen"] + "|" + training["model_id"], training["label"]
    )
    selected = pd.read_parquet(config["paris"])
    paris = {"training": paris_features(selected, training),
             "validation": paris_features(selected, validation),
             "held": paris_features(selected, evaluated)}  # fmt: skip
    screens = {"training": training["screen"], "validation": validation["screen"],
               "held": evaluated["screen"]}  # fmt: skip
    gene_mean = ladder.genes["m_g"]
    shared = {
        "paris": paris["held"][:, 0],
        "pair_identity": pair_identity(training, evaluated),
        "pan_essentiality": -(
            gene_mean.reindex(evaluated["gene_a"]).to_numpy()
            + gene_mean.reindex(evaluated["gene_b"]).to_numpy()
        ),
    }

    def fit(name: str, x: Mapping[str, np.ndarray]) -> np.ndarray:
        """Fit a head on the training rows it can read (complete inputs) and score
        the held-out rows; a row with a missing input scores NaN."""
        complete = np.isfinite(x["training"]).all(axis=1)
        score, record = fit_head(
            (x["training"][complete], training["label"].to_numpy()[complete],
             weights[complete]),
            [(x["validation"][i], validation["label"].to_numpy()[i]) for i in groups],
            config["head"],
            seed,
        )  # fmt: skip
        selection[name] = record | {"training_rows": int(complete.sum())}
        return score(x["held"])

    scores, selection = [], {}
    columns = [
        "screen",
        "model_id",
        "screen_line",
        "split",
        "gene_a",
        "gene_b",
        "label",
    ]
    varying = varying_pairs(training, evaluated)
    tables = {"training": training, "validation": validation, "held": evaluated}
    # Every line reads each gene's frame mean GeneEffect: the PARIS head with no line
    # in it, the same for every arm.
    lines = pd.unique(pd.concat([t["model_id"] for t in tables.values()]))
    blind = pd.DataFrame(
        np.tile(gene_mean.to_numpy(), (len(lines), 1)),
        index=lines,
        columns=gene_mean.index,
    )
    shared["paris_head_context_ablated"] = fit(
        "paris_head_context_ablated",
        {
            k: np.c_[paris[k], effect_features(blind, t), screen_onehot(screens[k])]
            for k, t in tables.items()
        },
    )
    for name, arm in config["arms"].items():
        effects = arm_effects(setup, arm, config["ladder_config"])
        ge = {k: effect_features(effects, t) for k, t in tables.items()}
        held_scores = dict(shared) | {"a_blind": -ge["held"][:, 0]}
        held_scores["paris_head"] = fit(
            f"{name}/paris_head",
            {k: np.c_[paris[k], ge[k], screen_onehot(screens[k])] for k in tables},
        )
        if arm["kind"] == "run":
            viability = Viability(arm["config"], config["ladder_config"])
            mean_v = viability.mean(reference)

            def gene_mean_v(table: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
                index = viability.genes
                return (
                    mean_v[table["gene_a"].map(index).to_numpy()],
                    mean_v[table["gene_b"].map(index).to_numpy()],
                )

            held_scores["v_head"] = fit(
                f"{name}/v_head",
                {
                    k: head_input(*viability.pairs(t), t["screen"])
                    for k, t in tables.items()
                },
            )
            # Each gene's V averaged over the reference lines: no line in it.
            held_scores["v_head_context_ablated"] = fit(
                f"{name}/v_head_context_ablated",
                {
                    k: head_input(*gene_mean_v(t), t["screen"])
                    for k, t in tables.items()
                },
            )
        for scorer, scored in held_scores.items():
            scores.append(
                evaluated[columns].assign(
                    arm=name, scorer=scorer, score=scored, varying=varying
                )
            )
        print(f"{name}: scored", flush=True)
    counts = {
        "training_rows": len(training),
        "training_lines": int(training["model_id"].nunique()),
        "held_rows": len(evaluated),
        "reference_lines": len(reference),
        "paris_pairs": len(selected),
        **rows["dropped"],
    }
    scores = pd.concat(scores, ignore_index=True)
    write_parquet(run_dir / "scores.parquet", scores)
    metrics = evaluate(scores, int(config["top_k"]), int(config["bootstrap"]), seed)
    metrics |= {"counts": counts, "selection": selection}
    write_json(run_dir / "metrics.json", metrics)
    write_text(run_dir / "results.md", results_markdown(metrics, config))
    print(f"{run_dir}: done", flush=True)
    return run_dir


def evaluate(scores: pd.DataFrame, k: int, reps: int, seed: int) -> dict[str, Any]:
    """Per (split, screen, line, arm): every scorer's metrics on all pairs and on
    the pairs whose training-line call varies, and the head's AUPRC contrasts."""
    rng = np.random.default_rng(seed)
    out: dict[str, Any] = {"rows": [], "contrasts": []}
    keys = ["split", "screen", "screen_line", "arm"]
    for (split, screen, line, arm), group in scores.groupby(keys, sort=True):
        by = {s: g.reset_index(drop=True) for s, g in group.groupby("scorer")}
        for scorer, frame in by.items():
            label, score = frame["label"].to_numpy(), frame["score"].to_numpy()
            varying = frame["varying"].to_numpy(bool)
            out["rows"].append(
                {"split": split, "screen": screen, "line": line, "arm": arm,
                 "scorer": scorer, "all": ranking_metrics(label, score, k),
                 "varying": ranking_metrics(label[varying], score[varying], k)}
            )  # fmt: skip
        for name in [s for s in by if f"{s}_context_ablated" in by]:
            head = by[name]
            others = [*CONTROLS, f"{name}_context_ablated"]
            if name == "paris_head":
                others.append("paris")
            for other in others:
                out["contrasts"].append(
                    {"split": split, "screen": screen, "line": line, "arm": arm,
                     "contrast": f"{name} - {other}",
                     **dyadic_difference(
                         head["label"].to_numpy(), head["score"].to_numpy(),
                         by[other]["score"].to_numpy(), head["gene_a"].to_numpy(),
                         head["gene_b"].to_numpy(), reps, rng)}
                )  # fmt: skip
    return out


def results_markdown(metrics: Mapping[str, Any], config: Mapping[str, Any]) -> str:
    counts = metrics["counts"]
    lines = [
        "# SL stage two: first run",
        "",
        f"Arms: {', '.join(config['arms'])}; PARIS pairs from {config['paris']} "
        f"({counts['paris_pairs']}). Training rows {counts['training_rows']} over "
        f"{counts['training_lines']} single-cell SL training lines; "
        f"held-out rows {counts['held_rows']}; left out: "
        + ", ".join(
            f"{k.replace('_', ' ')} {v}" for k, v in counts.items() if "_not_" in k
        )
        + f". Top K = {config['top_k']}. Epochs chosen on validation: "
        + ", ".join(f"{k} {v['epoch']}" for k, v in metrics["selection"].items())
        + ". Nothing is chosen here; this is not a calibrated SL probability.",
        "",
        "| Split | Screen | Line | Arm | Scorer | Pairs | SL | AUPRC | Lift | AUROC "
        "| P@K | Varying pairs | Varying SL | Varying AUPRC |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: "
        "| ---: | ---: | ---: |",
    ]
    for row in metrics["rows"]:
        a, v = row["all"], row["varying"]
        lines.append(
            f"| {row['split']} | {row['screen']} | {row['line']} | {row['arm']} | "
            f"{row['scorer']} | {a['pairs']} | {a['positives']} | "
            f"{number(a.get('auprc'))} | {number(a.get('auprc_lift'), 2)} | "
            f"{number(a.get('auroc'))} | {number(a.get('precision_at_k'), 2)} | "
            f"{v['pairs']} | {v['positives']} | {number(v.get('auprc'))} |"
        )
    lines += [
        "",
        "## AUPRC contrasts (95% two-way dyadic bootstrap)",
        "",
        "| Split | Screen | Line | Arm | Contrast | Difference |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in metrics["contrasts"]:
        low, high = row["interval"]
        lines.append(
            f"| {row['split']} | {row['screen']} | {row['line']} | {row['arm']} | "
            f"{row['contrast']} | {number(row['difference'])} "
            f"[{number(low)}, {number(high)}] |"
        )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("config", type=Path)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    run(load_config(args.config), args.run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
