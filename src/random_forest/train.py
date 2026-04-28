"""Train random forest baseline."""

from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestRegressor

from src.utils.data import build_pseudobulk_matrices
from src.utils.metrics import build_label_matrix


def run(config: dict) -> dict:
    """Fit a multi-output random forest gene scorer."""
    data = build_pseudobulk_matrices(config)
    model_config = config["model_config"]
    X_train = data["matrices"]["train"]
    train_conditions = data["conditions"]["train"]
    if X_train.shape[0] == 0:
        raise ValueError("Random forest training requires at least one train condition")

    labels = build_label_matrix(
        train_conditions,
        data["gene_name_to_idx"],
        len(data["gene_names"]),
    )
    model = RandomForestRegressor(
        n_estimators=int(model_config.get("n_estimators", 300)),
        max_depth=model_config.get("max_depth"),
        random_state=config["run_config"].get("seed"),
        n_jobs=int(model_config.get("n_jobs", 1)),
    )
    model.fit(X_train, labels)
    checkpoint_path = _checkpoint_path(config)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "gene_names": data["gene_names"]}, checkpoint_path)
    return {"checkpoint_path": str(checkpoint_path), "n_train": X_train.shape[0]}


def _checkpoint_path(config: dict) -> Path:
    path = config["run_config"].get("save_checkpoint_path")
    if path:
        return Path(path)
    study_name = config["run_config"]["study_name"]
    return Path("model") / "random_forest" / study_name / "model.joblib"
