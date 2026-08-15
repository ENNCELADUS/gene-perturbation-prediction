"""One-shot evaluation of sealed HCT116 predictions against DepMap GeneEffect."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

from aivc_model.gene_splits import sha256_file

LOGGER = logging.getLogger(__name__)


def _metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    thresholds: tuple[float, ...],
    top_fraction: float,
) -> dict[str, float]:
    values = {
        "spearman": float(spearmanr(y_true, y_pred).statistic),
        "pearson": float(pearsonr(y_true, y_pred).statistic),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }
    for threshold in thresholds:
        labels = y_true < threshold
        suffix = str(abs(threshold)).replace(".", "p")
        values[f"auroc_lt_neg_{suffix}"] = (
            float(roc_auc_score(labels, -y_pred))
            if np.unique(labels).size == 2
            else math.nan
        )
        values[f"auprc_lt_neg_{suffix}"] = (
            float(average_precision_score(labels, -y_pred))
            if labels.any()
            else math.nan
        )
    labels = y_true < thresholds[0]
    top_count = max(1, int(math.ceil(len(y_pred) * top_fraction)))
    selected = np.argsort(y_pred)[:top_count]
    prevalence = float(labels.mean())
    values["top5_enrichment_lt_neg_0p5"] = (
        float(labels[selected].mean() / prevalence) if prevalence > 0 else math.nan
    )
    return values


def _scope_mask(frame: pd.DataFrame, scope: str) -> np.ndarray:
    if scope == "e_all":
        return np.ones(len(frame), dtype=bool)
    column = f"is_{scope}"
    if column not in frame:
        raise ValueError(f"prediction table is missing scope column {column}")
    return frame[column].astype(bool).to_numpy()


def evaluate(
    predictions_dir: Path,
    hct116_labels_path: Path,
    k562_transfer_path: Path,
    k562_split_path: Path,
    prediction_contract_path: Path,
    prediction_seal_path: Path,
    evaluation_contract_path: Path,
    output_dir: Path,
) -> Path:
    """Evaluate predictions and write all registered metrics."""
    contract = json.loads(evaluation_contract_path.read_text())
    prediction_contract_hash = sha256_file(prediction_contract_path)
    if contract["prediction_contract_sha256"] != prediction_contract_hash:
        raise ValueError("prediction contract hash mismatch")
    if contract["prediction_seal_sha256"] != sha256_file(prediction_seal_path):
        raise ValueError("prediction seal does not match frozen evaluation contract")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"output directory is not empty: {output_dir}")

    prediction_manifest_path = predictions_dir / "prediction_manifest.json"
    prediction_path = predictions_dir / "predictions_label_blind.csv"
    prediction_manifest = json.loads(prediction_manifest_path.read_text())
    prediction_seal = json.loads(prediction_seal_path.read_text())
    if prediction_manifest.get("ground_truth_accessed") is not False:
        raise ValueError("predictions were not produced under the label-blind contract")
    if prediction_manifest.get("prediction_contract_sha256") != sha256_file(
        prediction_contract_path
    ):
        raise ValueError("predictions do not match the prediction contract")
    if prediction_manifest["predictions_sha256"] != sha256_file(prediction_path):
        raise ValueError("sealed prediction hash mismatch")
    expected_seal = {
        "schema_version": 1,
        "prediction_manifest_sha256": sha256_file(prediction_manifest_path),
        "predictions_sha256": sha256_file(prediction_path),
        "prediction_contract_sha256": sha256_file(prediction_contract_path),
    }
    if prediction_seal != expected_seal:
        raise ValueError("independent prediction seal does not match prediction files")
    if prediction_manifest["k562_split_manifest_sha256"] != sha256_file(
        k562_split_path
    ):
        raise ValueError("K562 split authority mismatch")

    expected_hashes = {
        "hct116_label_table_sha256": sha256_file(hct116_labels_path),
        "k562_transfer_table_sha256": sha256_file(k562_transfer_path),
    }
    if any(contract[key] != value for key, value in expected_hashes.items()):
        raise ValueError("evaluation label hash does not match frozen contract")

    predictions = pd.read_csv(prediction_path)
    if predictions["perturbation_gene"].duplicated().any():
        raise ValueError("predictions must contain one row per gene")
    labels = pd.read_csv(hct116_labels_path)
    labels = labels.loc[
        labels["has_depmap_label"].astype(bool),
        ["perturbation_gene", "depmap_gene_effect", "match_status"],
    ]
    labels = labels.loc[labels["match_status"].eq("exact_symbol_match")]
    if labels["perturbation_gene"].duplicated().any():
        raise ValueError("HCT116 label table has duplicate exact-symbol genes")
    merged = predictions.merge(
        labels, on="perturbation_gene", how="inner", validate="one_to_one"
    )
    merged = merged.rename(columns={"depmap_gene_effect": "y_true"})
    if merged.empty or not np.isfinite(merged[["y_pred", "y_true"]]).all().all():
        raise ValueError("label merge is empty or nonfinite")

    k562 = pd.read_csv(k562_transfer_path).loc[
        lambda frame: frame["has_depmap_label"].astype(bool),
        ["perturbation_gene", "depmap_gene_effect"],
    ]
    if k562["perturbation_gene"].duplicated().any():
        raise ValueError("K562 transfer table has duplicate labelled genes")
    if not np.isfinite(k562["depmap_gene_effect"]).all():
        raise ValueError("K562 transfer table has nonfinite labels")
    k562 = k562.rename(columns={"depmap_gene_effect": "y_pred_k562_transfer"})
    split = pd.read_csv(k562_split_path)
    train_mean = float(
        k562.merge(split, on="perturbation_gene", validate="one_to_one")
        .loc[lambda frame: frame["split_role"].eq("train"), "y_pred_k562_transfer"]
        .mean()
    )
    merged = merged.merge(
        k562, on="perturbation_gene", how="left", validate="one_to_one"
    )
    merged["y_pred_k562_train_mean"] = train_mean

    thresholds = tuple(float(value) for value in contract["essentiality_thresholds"])
    top_fraction = float(contract["top_fraction"])
    systems = {
        "final_observed_response": "y_pred",
        "k562_geneeffect_transfer": "y_pred_k562_transfer",
        "k562_train_mean": "y_pred_k562_train_mean",
    }
    metric_rows: list[dict[str, object]] = []
    bootstrap_rows: list[dict[str, object]] = []
    comparison_rows: list[dict[str, object]] = []
    for scope in contract["scopes"]:
        scoped = merged.loc[_scope_mask(merged, str(scope))].reset_index(drop=True)
        scope_seed = int.from_bytes(
            hashlib.sha256(str(scope).encode()).digest()[:4], "little"
        )
        cohorts = {
            "scope_complete": (
                scoped,
                {
                    "final_observed_response": systems["final_observed_response"],
                    "k562_train_mean": systems["k562_train_mean"],
                },
            ),
            "k562_transfer_paired": (
                scoped.dropna(subset=["y_pred_k562_transfer"]).reset_index(drop=True),
                systems,
            ),
        }
        for cohort, (frame, cohort_systems) in cohorts.items():
            if len(frame) < 2:
                continue
            rng = np.random.default_rng(
                int(contract["bootstrap_seed"])
                + scope_seed
                + int.from_bytes(hashlib.sha256(cohort.encode()).digest()[:4], "little")
            )
            bootstrap_indices = rng.integers(
                0,
                len(frame),
                size=(int(contract["bootstrap_replicates"]), len(frame)),
            )
            y_true = frame["y_true"].to_numpy(dtype=np.float64)
            for system, column in cohort_systems.items():
                y_pred = frame[column].to_numpy(dtype=np.float64)
                values = _metrics(
                    y_true, y_pred, thresholds=thresholds, top_fraction=top_fraction
                )
                metric_rows.extend(
                    {
                        "scope": scope,
                        "cohort": cohort,
                        "system": system,
                        "n_genes": len(frame),
                        "metric": key,
                        "estimate": value,
                    }
                    for key, value in values.items()
                )
                samples = [
                    _metrics(
                        y_true[indices],
                        y_pred[indices],
                        thresholds=thresholds,
                        top_fraction=top_fraction,
                    )
                    for indices in bootstrap_indices
                ]
                for key in values:
                    distribution = np.asarray(
                        [sample[key] for sample in samples], dtype=np.float64
                    )
                    distribution = distribution[np.isfinite(distribution)]
                    bootstrap_rows.append(
                        {
                            "scope": scope,
                            "cohort": cohort,
                            "system": system,
                            "metric": key,
                            "ci_low": float(np.quantile(distribution, 0.025))
                            if len(distribution)
                            else math.nan,
                            "ci_high": float(np.quantile(distribution, 0.975))
                            if len(distribution)
                            else math.nan,
                            "n_bootstrap_valid": int(len(distribution)),
                        }
                    )
        paired = scoped.dropna(subset=["y_pred", "y_pred_k562_transfer"])
        if len(paired) >= 2:
            paired = paired.reset_index(drop=True)

            def paired_values(indices: np.ndarray) -> dict[str, float]:
                sample = paired.iloc[indices]
                final_residual = sample["y_true"] - sample["y_pred"]
                transfer_residual = sample["y_true"] - sample["y_pred_k562_transfer"]
                return {
                    "final_minus_k562_transfer_mae": float(
                        np.abs(final_residual).mean() - np.abs(transfer_residual).mean()
                    ),
                    "final_minus_k562_transfer_rmse": float(
                        np.sqrt(np.square(final_residual).mean())
                        - np.sqrt(np.square(transfer_residual).mean())
                    ),
                    "delta_spearman": float(
                        spearmanr(
                            sample["y_pred"] - sample["y_pred_k562_transfer"],
                            sample["y_true"] - sample["y_pred_k562_transfer"],
                        ).statistic
                    ),
                }

            estimates = paired_values(np.arange(len(paired)))
            paired_rng = np.random.default_rng(
                int(contract["bootstrap_seed"]) + scope_seed
            )
            distributions = {key: [] for key in estimates}
            for _ in range(int(contract["bootstrap_replicates"])):
                sample_indices = paired_rng.integers(0, len(paired), len(paired))
                values = paired_values(sample_indices)
                for key, value in values.items():
                    if np.isfinite(value):
                        distributions[key].append(value)
            comparison_rows.extend(
                {
                    "scope": scope,
                    "cohort": "k562_transfer_paired",
                    "comparison": key,
                    "n_genes": len(paired),
                    "estimate": estimate,
                    "ci_low": float(np.quantile(distributions[key], 0.025)),
                    "ci_high": float(np.quantile(distributions[key], 0.975)),
                    "n_bootstrap_valid": len(distributions[key]),
                }
                for key, estimate in estimates.items()
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    merged_path = output_dir / "predictions_with_hct116_geneeffect.csv"
    metrics_path = output_dir / "metrics.csv"
    bootstrap_path = output_dir / "bootstrap_ci.csv"
    comparisons_path = output_dir / "paired_comparisons.csv"
    merged.to_csv(merged_path, index=False)
    pd.DataFrame(metric_rows).to_csv(metrics_path, index=False)
    pd.DataFrame(bootstrap_rows).to_csv(bootstrap_path, index=False)
    pd.DataFrame(comparison_rows).to_csv(comparisons_path, index=False)
    result_manifest = {
        "schema_version": 1,
        "prediction_manifest_sha256": sha256_file(prediction_manifest_path),
        "prediction_seal_sha256": sha256_file(prediction_seal_path),
        "evaluation_contract_sha256": sha256_file(evaluation_contract_path),
        "hct116_label_table_sha256": expected_hashes["hct116_label_table_sha256"],
        "merged_genes": int(len(merged)),
        "outputs": {
            path.name: sha256_file(path)
            for path in (merged_path, metrics_path, bootstrap_path, comparisons_path)
        },
    }
    manifest_path = output_dir / "evaluation_manifest.json"
    manifest_path.write_text(
        json.dumps(result_manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-dir", type=Path, required=True)
    parser.add_argument("--hct116-labels", type=Path, required=True)
    parser.add_argument("--k562-transfer", type=Path, required=True)
    parser.add_argument("--k562-split", type=Path, required=True)
    parser.add_argument("--prediction-contract", type=Path, required=True)
    parser.add_argument("--prediction-seal", type=Path, required=True)
    parser.add_argument("--evaluation-contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    manifest = evaluate(
        args.predictions_dir,
        args.hct116_labels,
        args.k562_transfer,
        args.k562_split,
        args.prediction_contract,
        args.prediction_seal,
        args.evaluation_contract,
        args.output_dir,
    )
    LOGGER.info("HCT116 evaluation ready: %s", manifest)


if __name__ == "__main__":
    main()
