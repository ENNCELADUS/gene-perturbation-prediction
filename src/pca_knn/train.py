"""Train PCA+kNN baseline artifacts."""

from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from src.utils.data import build_pseudobulk_matrices
from src.utils.distributed import disable_tqdm
from src.utils.metrics import build_label_matrix


def run(config: dict) -> dict:
    """Fit PCA on train condition profiles and persist nearest-neighbor labels."""
    with tqdm(
        total=5,
        desc="pca_knn train",
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
            raise ValueError("PCA+kNN training requires at least one train condition")

        standardize = bool(model_config.get("standardize", False))
        scaler = StandardScaler() if standardize else None
        X_proc = scaler.fit_transform(X_train) if scaler is not None else X_train
        progress.update()
        n_components = min(
            int(model_config.get("n_components", 16)),
            X_proc.shape[0],
            X_proc.shape[1],
        )
        pca = PCA(
            n_components=n_components,
            random_state=config["run_config"].get("seed"),
        )
        train_embeddings = pca.fit_transform(X_proc)
        progress.update()
        labels = build_label_matrix(
            train_conditions,
            data["gene_name_to_idx"],
            len(data["gene_names"]),
        )
        progress.update()
        checkpoint_path = _checkpoint_path(config)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "pca": pca,
                "scaler": scaler,
                "train_embeddings": train_embeddings,
                "train_labels": labels,
                "train_conditions": train_conditions,
                "gene_names": data["gene_names"],
            },
            checkpoint_path,
        )
        progress.update()
    return {"checkpoint_path": str(checkpoint_path), "n_train": X_train.shape[0]}


def _checkpoint_path(config: dict) -> Path:
    path = config["run_config"].get("save_checkpoint_path")
    if path:
        return Path(path)
    study_name = config["run_config"]["study_name"]
    return Path("model") / "pca_knn" / study_name / "model.joblib"
