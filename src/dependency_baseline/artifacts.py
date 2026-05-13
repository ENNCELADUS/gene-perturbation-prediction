"""Artifact, checkpoint, and manifest helpers for experiment runs."""

from __future__ import annotations

import json
import os
import platform
import random
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dependency_baseline.config import BaselineConfig
from dependency_baseline.metrics import (
    SUMMARY_GROUP_COLUMNS,
    rank_predictions,
    summarize_metrics,
    summarize_rankings,
    topk_candidates,
)


@dataclass(frozen=True)
class CvPaths:
    run_dir: Path
    fold_metrics_csv: Path
    summary_csv: Path
    predictions_csv: Path
    config_json: Path
    manifest_json: Path
    splits_csv: Path
    model_manifest_csv: Path
    topk_candidates_csv: Path


@dataclass(frozen=True)
class FinalFitPaths:
    run_dir: Path
    final_model_manifest_csv: Path
    final_rankings_csv: Path
    manifest_json: Path


@dataclass(frozen=True)
class RunContext:
    run_id: str
    run_dir: Path
    feature_path: Path
    command: tuple[str, ...]
    config_path: Path | None


class ArtifactStore:
    """Incremental result writer for one experiment run."""

    def __init__(
        self,
        run_dir: Path,
        result_formats: tuple[str, ...],
        topk_values: tuple[int, ...],
        save_predictions: bool,
        save_rankings: bool,
    ) -> None:
        self.run_dir = run_dir
        self.result_formats = result_formats
        self.topk_values = topk_values
        self.save_predictions = save_predictions
        self.save_rankings = save_rankings
        self.completed_jobs_path = run_dir / "completed_jobs.jsonl"
        self.rankings_dir = run_dir / "rankings"
        self.tables = {
            name: self._load_table(name)
            for name in (
                "fold_metrics",
                "predictions",
                "splits",
                "model_manifest",
                "summary_metrics",
                "topk_candidates",
                "final_model_manifest",
                "final_rankings",
                "ranking_summary",
            )
        }
        self.completed_jobs = self._load_completed_jobs()

    def append_fold_result(
        self,
        metric_row: dict[str, object],
        predictions: pd.DataFrame,
        model_manifest_row: dict[str, object],
    ) -> None:
        """Persist one completed internal CV fit."""
        job = str(metric_row["job_key"])
        self.tables["fold_metrics"] = concat_dedupe(
            self.tables["fold_metrics"],
            pd.DataFrame([metric_row]),
            ["job_key"],
        )
        if self.save_predictions:
            self.tables["predictions"] = concat_dedupe(
                self.tables["predictions"],
                predictions,
                ["job_key", "perturbation_gene"],
            )
        self.tables["model_manifest"] = concat_dedupe(
            self.tables["model_manifest"],
            pd.DataFrame([model_manifest_row]),
            ["job_key"],
        )
        self._refresh_summaries()
        self._write_core_tables()
        self._mark_completed(job, model_manifest_row)

    def append_external_result(
        self,
        metric_row: dict[str, object],
        predictions: pd.DataFrame,
    ) -> None:
        """Persist one external evaluation result."""
        self.tables["fold_metrics"] = concat_dedupe(
            self.tables["fold_metrics"],
            pd.DataFrame([metric_row]),
            ["job_key"],
        )
        if self.save_predictions:
            self.tables["predictions"] = concat_dedupe(
                self.tables["predictions"],
                predictions,
                ["job_key", "perturbation_gene"],
            )
        self._refresh_summaries()
        self._write_core_tables()

    def write_splits(self, splits: pd.DataFrame) -> None:
        self.tables["splits"] = concat_dedupe(
            self.tables["splits"],
            splits,
            ["evaluation_scope", "fold", "perturbation_gene", "split"],
        )
        self._write_table("splits", self.tables["splits"])

    def append_final_result(
        self,
        manifest_row: dict[str, object],
        ranking: pd.DataFrame,
    ) -> None:
        self.tables["final_model_manifest"] = concat_dedupe(
            self.tables["final_model_manifest"],
            pd.DataFrame([manifest_row]),
            ["job_key"],
        )
        self.tables["final_rankings"] = concat_dedupe(
            self.tables["final_rankings"],
            ranking,
            ["job_key", "perturbation_gene"],
        )
        self._write_table("final_model_manifest", self.tables["final_model_manifest"])
        self._write_table("final_rankings", self.tables["final_rankings"])

    def _refresh_summaries(self) -> None:
        fold_metrics = self.tables["fold_metrics"]
        if not fold_metrics.empty:
            self.tables["summary_metrics"] = summarize_metrics(fold_metrics)
        if self.save_rankings and not self.tables["predictions"].empty:
            self._write_group_rankings(self.tables["predictions"])
            topk = topk_candidates(self.tables["predictions"], self.topk_values)
            self.tables["topk_candidates"] = topk
            self.tables["ranking_summary"] = summarize_rankings(topk)

    def _write_core_tables(self) -> None:
        for table_name in (
            "fold_metrics",
            "predictions",
            "model_manifest",
            "summary_metrics",
            "topk_candidates",
            "ranking_summary",
        ):
            if table_name == "predictions" and not self.save_predictions:
                continue
            self._write_table(table_name, self.tables[table_name])

    def _write_group_rankings(self, predictions: pd.DataFrame) -> None:
        self.rankings_dir.mkdir(parents=True, exist_ok=True)
        for key, group in predictions.groupby(SUMMARY_GROUP_COLUMNS, dropna=False):
            evaluation_scope, feature_set, model, weighting = key
            filename = "__".join(
                safe_name(str(value))
                for value in (evaluation_scope, feature_set, model, weighting)
            )
            write_formats(
                self.rankings_dir / filename,
                rank_predictions(group),
                self.result_formats,
            )

    def _write_table(self, table_name: str, table: pd.DataFrame) -> None:
        if table.empty:
            return
        write_formats(self.run_dir / table_name, table, self.result_formats)

    def _load_table(self, table_name: str) -> pd.DataFrame:
        parquet_path = self.run_dir / f"{table_name}.parquet"
        csv_path = self.run_dir / f"{table_name}.csv"
        if parquet_path.exists():
            return pd.read_parquet(parquet_path)
        if csv_path.exists():
            return pd.read_csv(csv_path)
        return pd.DataFrame()

    def _load_completed_jobs(self) -> set[str]:
        if not self.completed_jobs_path.exists():
            return set()
        completed: set[str] = set()
        for line in self.completed_jobs_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                completed.add(str(json.loads(line)["job_key"]))
        return completed

    def _mark_completed(
        self,
        job: str,
        model_manifest_row: dict[str, object],
    ) -> None:
        if job in self.completed_jobs:
            return
        payload = {
            "job_key": job,
            "completed_at": utc_now(),
            "checkpoint_path": model_manifest_row.get("checkpoint_path"),
        }
        with self.completed_jobs_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
        self.completed_jobs.add(job)


def create_run_context(
    *,
    config: BaselineConfig,
    features_npz: Path | None,
    run_id: str | None,
    resume: bool,
    command: tuple[str, ...],
    config_path: Path | None,
) -> RunContext:
    """Create or resume a run directory and update latest-run pointer."""
    feature_path = (
        features_npz or config.data.output_dir / "replogle_k562_delta_features.npz"
    )
    resolved_run_id = run_id or config.experiment.run_id or default_run_id()
    run_dir = config.data.output_dir / "runs" / resolved_run_id
    if run_dir.exists() and not resume and any(run_dir.iterdir()):
        msg = f"Run directory already exists; use --resume or a new --run-id: {run_dir}"
        raise FileExistsError(msg)
    run_dir.mkdir(parents=True, exist_ok=True)
    (config.data.output_dir / "latest_run.txt").write_text(
        str(run_dir),
        encoding="utf-8",
    )
    return RunContext(
        run_id=resolved_run_id,
        run_dir=run_dir,
        feature_path=feature_path,
        command=command,
        config_path=config_path,
    )


def manifest_base(
    config: BaselineConfig,
    context: RunContext,
    command_name: str,
    resume: bool,
) -> dict[str, object]:
    """Build common run manifest fields."""
    return {
        "experiment_name": config.experiment.name,
        "run_id": context.run_id,
        "run_dir": str(context.run_dir),
        "command_name": command_name,
        "command": list(context.command),
        "config_path": str(context.config_path) if context.config_path else None,
        "features_npz": str(context.feature_path),
        "started_at": utc_now(),
        "resume": resume,
        "git_sha": git_sha(short=False),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "pid": os.getpid(),
        "packages": package_versions(),
        "resolved_config": jsonable(asdict(config)),
    }


def write_cv_config(
    config: BaselineConfig,
    feature_path: Path,
    output_path: Path,
) -> None:
    """Write the resolved CV configuration."""
    write_json(
        output_path,
        {
            "n_splits": config.cv.n_splits,
            "n_repeats": config.cv.n_repeats,
            "random_state": config.cv.random_state,
            "stratify_bins": config.cv.stratify_bins,
            "model_set": config.cv.model_set,
            "features_npz": str(feature_path),
            "experiment": jsonable(asdict(config.experiment)),
            "selection": jsonable(asdict(config.selection)),
            "models": jsonable(config.models or {}),
        },
    )


def summarize_results(results_dir: Path) -> tuple[Path, Path | None]:
    """Rebuild summary metrics and ranking summary for a result directory."""
    fold_metrics = read_table(results_dir / "fold_metrics")
    summary = summarize_metrics(fold_metrics)
    write_formats(results_dir / "summary_metrics", summary, ("parquet", "csv"))
    topk_base = results_dir / "topk_candidates"
    ranking_summary_path: Path | None = None
    if (
        topk_base.with_suffix(".parquet").exists()
        or topk_base.with_suffix(".csv").exists()
    ):
        ranking_summary = summarize_rankings(read_table(topk_base))
        write_formats(
            results_dir / "ranking_summary",
            ranking_summary,
            ("parquet", "csv"),
        )
        ranking_summary_path = results_dir / "ranking_summary.csv"
    return results_dir / "summary_metrics.csv", ranking_summary_path


def set_seed(seed: int) -> None:
    """Set NumPy, Python, and optional PyTorch seeds."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def checkpoint_path(
    run_dir: Path,
    evaluation_scope: str,
    fold: int,
    feature_set: str,
    model: str,
    weighting: str,
) -> Path:
    """Return checkpoint path for one CV or final model."""
    if evaluation_scope == "final":
        return (
            run_dir
            / "models"
            / "final"
            / f"{feature_set}__{model}__{weighting}.joblib"
        )
    return (
        run_dir
        / "models"
        / "cv"
        / safe_name(evaluation_scope)
        / f"fold_{fold}"
        / f"{feature_set}__{model}__{weighting}.joblib"
    )


def job_key(
    evaluation_scope: str,
    fold: int,
    feature_set: str,
    model: str,
    weighting: str,
) -> str:
    """Stable unique key for one model fit/evaluation job."""
    return "|".join(
        str(value)
        for value in (evaluation_scope, fold, feature_set, model, weighting)
    )


def safe_name(value: str) -> str:
    """Convert a display name to a filesystem-safe token."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def concat_dedupe(
    existing: pd.DataFrame,
    new_rows: pd.DataFrame,
    key_cols: list[str],
) -> pd.DataFrame:
    """Append rows and keep the last row for duplicate keys."""
    if existing.empty:
        combined = new_rows.copy()
    else:
        combined = pd.concat([existing, new_rows], ignore_index=True)
    return combined.drop_duplicates(subset=key_cols, keep="last").reset_index(drop=True)


def write_formats(
    base_path: Path,
    table: pd.DataFrame,
    formats: tuple[str, ...],
) -> None:
    """Write a table in all configured formats."""
    base_path.parent.mkdir(parents=True, exist_ok=True)
    for result_format in formats:
        if result_format == "csv":
            _atomic_write_csv(table, base_path.with_suffix(".csv"))
        elif result_format == "parquet":
            _atomic_write_parquet(table, base_path.with_suffix(".parquet"))
        else:
            msg = f"Unsupported result format: {result_format}"
            raise ValueError(msg)


def read_table(base_path: Path) -> pd.DataFrame:
    """Read a table from parquet if available, otherwise CSV."""
    parquet_path = base_path.with_suffix(".parquet")
    csv_path = base_path.with_suffix(".csv")
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)
    msg = f"Missing table: {base_path}.parquet or {base_path}.csv"
    raise FileNotFoundError(msg)


def write_json(path: Path, payload: dict[str, object]) -> None:
    """Atomically write a JSON payload."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


def utc_now() -> str:
    """Return a compact UTC timestamp."""
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def default_run_id() -> str:
    """Generate a default run id from UTC timestamp and git SHA."""
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    short_sha = git_sha(short=True)
    return f"{timestamp}_{short_sha or 'nogit'}"


def git_sha(short: bool = False) -> str | None:
    """Return the current git SHA when available."""
    args = ["git", "rev-parse", "--short" if short else "HEAD"]
    try:
        result = subprocess.run(
            args,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def package_versions() -> dict[str, str | None]:
    """Collect key package versions for run provenance."""
    packages = [
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
        "joblib",
        "torch",
        "xgboost",
    ]
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def jsonable(value: Any) -> Any:
    """Recursively convert paths and tuples into JSON-compatible values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def _atomic_write_csv(table: pd.DataFrame, path: Path) -> None:
    tmp_path = path.with_name(f".{path.name}.tmp")
    table.to_csv(tmp_path, index=False)
    tmp_path.replace(path)


def _atomic_write_parquet(table: pd.DataFrame, path: Path) -> None:
    tmp_path = path.with_name(f".{path.name}.tmp")
    table.to_parquet(tmp_path, index=False)
    tmp_path.replace(path)
