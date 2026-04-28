"""Train random forest baseline."""

from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestRegressor
from tqdm.auto import tqdm

from src.utils.data import build_pseudobulk_matrices
from src.utils.distributed import disable_tqdm
from src.utils.metrics import build_label_matrix


def run(config: dict) -> dict:
    """Fit a multi-output random forest gene scorer."""
    with tqdm(
        total=4,
        desc="random_forest train",
        unit="step",
        dynamic_ncols=True,
        disable=disable_tqdm(config),
    ) as progress:
        data = build_pseudobulk_matrices(config)
        progress.update()
        model_config = config["model_config"]
        X_train = data["matrices"]["train"]
        train_conditions = data["conditions"]["train"]
        if X_train.shape[0] == 0:
            raise ValueError(
                "Random forest training requires at least one train condition"
            )

        labels = build_label_matrix(
            train_conditions,
            data["gene_name_to_idx"],
            len(data["gene_names"]),
        )
        progress.update()
        model = _fit_random_forest_with_progress(
            X_train=X_train,
            labels=labels,
            model_config=model_config,
            seed=config["run_config"].get("seed"),
            disable_tqdm=disable_tqdm(config),
        )
        progress.update()
        checkpoint_path = _checkpoint_path(config)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": model, "gene_names": data["gene_names"]}, checkpoint_path)
        progress.update()
    return {"checkpoint_path": str(checkpoint_path), "n_train": X_train.shape[0]}


def _fit_random_forest_with_progress(
    X_train,
    labels,
    model_config: dict,
    seed: int | None,
    disable_tqdm: bool,
) -> RandomForestRegressor:
    n_estimators = int(model_config.get("n_estimators", 300))
    if n_estimators < 1:
        raise ValueError("Random forest n_estimators must be at least 1")
    estimator_chunk_size = max(1, int(model_config.get("estimator_chunk_size", 1)))
    model = RandomForestRegressor(
        n_estimators=0,
        max_depth=model_config.get("max_depth"),
        min_samples_leaf=int(model_config.get("min_samples_leaf", 1)),
        random_state=seed,
        n_jobs=int(model_config.get("n_jobs", 1)),
        warm_start=True,
    )
    with tqdm(
        total=n_estimators,
        desc="random_forest trees",
        unit="tree",
        dynamic_ncols=True,
        disable=disable_tqdm,
    ) as progress:
        fitted_estimators = 0
        while fitted_estimators < n_estimators:
            next_estimators = min(
                fitted_estimators + estimator_chunk_size,
                n_estimators,
            )
            model.set_params(n_estimators=next_estimators)
            model.fit(X_train, labels)
            progress.update(next_estimators - fitted_estimators)
            fitted_estimators = next_estimators
    return model


def _checkpoint_path(config: dict) -> Path:
    path = config["run_config"].get("save_checkpoint_path")
    if path:
        return Path(path)
    study_name = config["run_config"]["study_name"]
    return Path("model") / "random_forest" / study_name / "model.joblib"
