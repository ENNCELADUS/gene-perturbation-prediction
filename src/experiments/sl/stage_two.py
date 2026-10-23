"""Stage two, first run (SL protocol §10): a viability-embedding head, an MLP over stage
one's 32-d viability embeddings V(c, A) and V(c, B) of a screened pair in line c, fitted
on the SL training lines (V out of fold) and scored per held-out line and screen of the
SL screen benchmark against pair-identity, pan-essentiality, A-blind and
context-ablated controls. One arm per stage-one run."""

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

SCREENS = ("harle_2025", "ito_2021", "klingbeil_2024")
VIABILITY_WIDTH = 32
CONTRASTS = ("pair_identity", "head_context_ablated", "pan_essentiality", "a_blind")


def load_config(path: Path) -> dict[str, Any]:
    config = read_yaml(path)
    require_keys(
        config,
        {"kind", "seed", "output_root", "task", "ladder_config", "benchmark", "arms",
         "head", "top_k", "bootstrap"},
        "config",
    )  # fmt: skip
    if config["kind"] != "sl_stage_two":
        raise ValueError(f"{path} is a {config['kind']} config")
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
    onehot = np.stack([(screens == s).to_numpy(float) for s in SCREENS], axis=1)
    return np.concatenate([v_a + v_b, np.abs(v_a - v_b), onehot], axis=1)


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


class Viability:
    """A stage-one run's V: out of fold for frame lines (the fold run that left the
    line out), the main run for held-out lines."""

    def __init__(self, run: str, ladder_config: str, folds: Mapping[str, int]):
        from src.experiments.stage_one.report import finished_run

        run_path = Path(run)
        config = read_json(run_path / "run_config.json")["config"]["run"]
        self.folds = folds
        self.sources = {}
        for k in [None, *sorted(set(folds.values()))]:
            run_dir = finished_run(config, run_path.name, k, ladder_config)
            if run_dir is None:
                raise FileNotFoundError(f"{run} (fold {k}) is not a finished run")
            meta = read_json(run_dir / "V.json")
            values = np.load(run_dir / "V.npy", mmap_mode="r")
            self.sources[k] = (
                meta,
                {m: i for i, m in enumerate(meta["lines"])},
                values,
            )
        genes = self.sources[None][0]["genes"]
        if any(meta["genes"] != genes for meta, _, _ in self.sources.values()):
            raise ValueError(f"{run}: a fold run's genes are not the main run's")
        self.genes = {g: i for i, g in enumerate(genes)}

    def line(self, model_id: str) -> np.ndarray:
        """The line's V (genes x 32), from the run that did not train on it."""
        k = self.folds.get(model_id)
        meta, rows, values = self.sources[k]
        row = rows[model_id]
        if k is not None and meta["in_sample"][row]:
            raise ValueError(f"fold {k}'s run trained on {model_id}, its own line")
        return values[row]

    def pairs(self, rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """V(c, gene_a) and V(c, gene_b) of every row, in float32."""
        out_a = np.empty((len(rows), VIABILITY_WIDTH), np.float32)
        out_b = np.empty_like(out_a)
        a = rows["gene_a"].map(self.genes).to_numpy()
        b = rows["gene_b"].map(self.genes).to_numpy()
        for model_id, index in rows.groupby("model_id").indices.items():
            values = self.line(model_id)
            out_a[index], out_b[index] = values[a[index]], values[b[index]]
        return out_a, out_b

    def frame_mean(self, lines: Sequence[str]) -> np.ndarray:
        """Each gene's V averaged over ``lines`` (genes x 32), out of fold."""
        total = np.zeros((len(self.genes), VIABILITY_WIDTH), np.float64)
        for model_id in lines:
            total += self.line(model_id)
        return (total / len(lines)).astype(np.float32)


def benchmark_rows(
    path: Path, frame: Sequence[str], held: Sequence[str], genes: Mapping[str, int]
) -> dict[str, Any]:
    """Labelled rows whose two genes stage one decodes: training rows of the
    single-cell stratum (frame lines), and held-out rows whose pair was labelled in
    a training line of the same screen (the pair-identity control's universe)."""
    pairs = pd.read_parquet(path)
    pairs = pairs[pairs["label"].notna()]
    decoded = pairs["gene_a"].isin(genes) & pairs["gene_b"].isin(genes)
    pairs = pairs.assign(label=pairs["label"].astype(int))
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
        "screened_lines": set(pairs["model_id"].dropna()),
        "training": training[decoded].reset_index(drop=True),
        "held": evaluated[seen & decoded].reset_index(drop=True),
        "dropped": {
            "training_rows_gene_not_decoded": int((~decoded[training.index]).sum()),
            "held_rows_gene_not_decoded": int((~decoded[evaluated.index]).sum()),
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
    gene_mean = ladder.genes["m_g"]

    scores, selection, counts = [], {}, {}
    for name, run_path in config["arms"].items():
        viability = Viability(run_path, config["ladder_config"], ladder.folds)
        rows = benchmark_rows(Path(config["benchmark"]), frame, held, viability.genes)
        training, evaluated = rows["training"], rows["held"]
        reference = [m for m in frame if m not in rows["screened_lines"]]
        mean_v = viability.frame_mean(reference)

        def inputs(table: pd.DataFrame, ablated: bool) -> np.ndarray:
            if ablated:  # each gene's V averaged over the reference lines
                a = mean_v[table["gene_a"].map(viability.genes).to_numpy()]
                b = mean_v[table["gene_b"].map(viability.genes).to_numpy()]
            else:
                a, b = viability.pairs(table)
            return head_input(a, b, table["screen"])

        weights = context_weights(
            training["screen"] + "|" + training["model_id"], training["label"]
        )
        validation_rows = evaluated[evaluated["split"] == "val"]
        groups = list(validation_rows.groupby(["screen", "model_id"]).indices.values())
        held_scores = {}
        for head, ablated in (("head", False), ("head_context_ablated", True)):
            val_x = inputs(validation_rows, ablated)
            score, record = fit_head(
                (inputs(training, ablated), training["label"].to_numpy(), weights),
                [(val_x[i], validation_rows["label"].to_numpy()[i]) for i in groups],
                config["head"],
                seed,
            )
            held_scores[head] = score(inputs(evaluated, ablated))
            selection[f"{name}/{head}"] = record
        predicted = (
            pd.read_parquet(Path(run_path) / "predictions.parquet")
            .loc[list(held)]
            .mul(ladder.genes["sigma_g"])
            .add(ladder.genes["mu_g"])
        )  # GeneEffect units on held-out lines
        line = predicted.index.get_indexer(evaluated["model_id"])
        a_col = predicted.columns.get_indexer(evaluated["gene_a"])
        b_col = predicted.columns.get_indexer(evaluated["gene_b"])
        values = predicted.to_numpy()
        held_scores |= {
            "pair_identity": pair_identity(training, evaluated),
            "pan_essentiality": -(
                gene_mean.reindex(evaluated["gene_a"]).to_numpy()
                + gene_mean.reindex(evaluated["gene_b"]).to_numpy()
            ),
            "a_blind": -(values[line, a_col] + values[line, b_col]),
        }
        columns = ["screen", "model_id", "screen_line", "split", "gene_a", "gene_b",
                   "label"]  # fmt: skip
        varying = varying_pairs(training, evaluated)
        for scorer, scored in held_scores.items():
            scores.append(
                evaluated[columns].assign(
                    arm=name, scorer=scorer, score=scored, varying=varying
                )
            )
        counts[name] = {
            "training_rows": len(training),
            "training_lines": int(training["model_id"].nunique()),
            "held_rows": len(evaluated),
            "reference_lines": len(reference),
            **rows["dropped"],
        }
        print(f"{name}: {counts[name]}", flush=True)
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
        head = by["head"]
        for other in CONTRASTS:
            out["contrasts"].append(
                {"split": split, "screen": screen, "line": line, "arm": arm,
                 "contrast": f"head - {other}",
                 **dyadic_difference(
                     head["label"].to_numpy(), head["score"].to_numpy(),
                     by[other]["score"].to_numpy(), head["gene_a"].to_numpy(),
                     head["gene_b"].to_numpy(), reps, rng)}
            )  # fmt: skip
    return out


def results_markdown(metrics: Mapping[str, Any], config: Mapping[str, Any]) -> str:
    counts = next(iter(metrics["counts"].values()))  # every arm decodes the panel
    lines = [
        "# SL stage two: first run",
        "",
        f"Viability-embedding head per arm ({', '.join(config['arms'])}). "
        f"Training rows {counts['training_rows']} over "
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
