import sys
import torch
import logging
import numpy as np
import pandas as pd
import scipy.sparse as sp
from pathlib import Path

# Optional dependency for PCA (used by this RF model)
try:
    import scanpy as sc
except Exception:  # pragma: no cover
    sc = None

try:
    from sklearn.ensemble import RandomForestRegressor
except Exception:  # pragma: no cover
    RandomForestRegressor = None

# Add scGPT to path (reusing logic from wrapper.py / baseline.py)
current_dir = Path(__file__).parent.parent.parent
scgpt_path = current_dir / "scGPT"

if not scgpt_path.exists():
    scgpt_path = Path.cwd() / "scGPT"

if str(scgpt_path) not in sys.path:
    sys.path.insert(0, str(scgpt_path))

try:
    from scgpt.tokenizer.gene_tokenizer import GeneVocab
except ImportError:
    logging.warning("Could not import scgpt directly. Checking path...")
    if not scgpt_path.exists():
        raise ImportError(f"scGPT directory not found at {scgpt_path}")
    else:
        raise


class RandomForestWrapper:
    """
    Random Forest model following the *ridge-style* pipeline:

    1) Pseudo-bulk mean profiles per perturbation (group_means)
    2) mean_perturbed as baseline
    3) PCA on genes -> loadings G (n_genes x k)
    4) Project each perturbation delta into PCA space:
           delta = (group_mean - mean_perturbed) in gene space
           z = argmin ||G z - delta||^2  (least squares)
    5) Train RF to map:
           x = P_val = G[target_gene_index, :]  (R^k)
           y = z (R^k)
    6) Inference:
           z_hat = RF(x)
           y_hat = G z_hat + mean_perturbed

    Output is a single mean profile per target_gene, broadcast to batch size.
    """

    def __init__(self, config, logger=None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.vocab = None

        rf_cfg = self.config.get("random_forest", {}) or {}

        self.top_k = int(rf_cfg.get("top_k", 50))

        self.n_estimators = int(rf_cfg.get("n_estimators", 300))
        self.max_depth = rf_cfg.get("max_depth", None)
        self.min_samples_split = int(rf_cfg.get("min_samples_split", 2))
        self.min_samples_leaf = int(rf_cfg.get("min_samples_leaf", 1))
        self.max_features = rf_cfg.get("max_features", "auto")
        self.random_state = int(rf_cfg.get("random_state", 0))
        self.n_jobs = int(rf_cfg.get("n_jobs", -1))

        self.G = None
        self.mean_unperturbed = None
        self.mean_perturbed = None
        self.gene2idx = None
        self.var_names = None

        self.rf = None
        self._is_fitted = False

        self._load_vocab()

    def _load_vocab(self):
        model_dir = Path(self.config["paths"]["model_dir"])
        vocab_file = model_dir / "vocab.json"
        if not vocab_file.exists():
            raise FileNotFoundError(f"Vocab file not found: {vocab_file}")

        self.vocab = GeneVocab.from_file(vocab_file)

        special_tokens = [self.config["model"]["pad_token"], "<cls>", "<eoc>"]
        for s in special_tokens:
            if s not in self.vocab:
                self.vocab.append_token(s)

        self.logger.info(f"Loaded vocab from {vocab_file}")

    def fit(self, train_loader):
        raise NotImplementedError(
            "RandomForestWrapper expects AnnData input. "
            "Call fit_from_adata(train_adata, condition_key=..., control_key=...)."
        )

    def fit_from_adata(self, adata, condition_key="condition", control_key="ctrl"):
        if sc is None:
            raise ImportError(
                "scanpy is required for RandomForestWrapper (PCA). Please install scanpy."
            )
        if RandomForestRegressor is None:
            raise ImportError(
                "scikit-learn is required for RandomForestWrapper. Please install scikit-learn."
            )

        self.logger.info("Fitting random forest model from AnnData...")

        adata = adata.copy()
        adata.obs[condition_key] = adata.obs[condition_key].astype(str)

        self.var_names = adata.var_names
        self.gene2idx = {g: i for i, g in enumerate(self.var_names)}

        n_genes = adata.n_vars
        self.top_k = min(int(self.top_k), n_genes)
        if self.top_k < 1:
            raise ValueError("top_k must be >= 1")

        # ---- group means ----
        X = adata.X
        if sp.issparse(X):
            df = pd.DataFrame.sparse.from_spmatrix(
                X, index=adata.obs[condition_key].values, columns=self.var_names
            )
        else:
            df = pd.DataFrame(
                np.asarray(X),
                index=adata.obs[condition_key].values,
                columns=self.var_names,
            )

        group_means = df.groupby(level=0, observed=True).mean()

        # ---- mean profiles ----
        ctrl_mask = adata.obs[condition_key] == str(control_key)
        if ctrl_mask.sum() == 0:
            raise ValueError(
                f"No control cells found where {condition_key} == '{control_key}'."
            )

        self.mean_unperturbed = (
            np.asarray(adata[ctrl_mask].X.mean(axis=0)).ravel().astype(np.float32)
        )
        self.mean_perturbed = (
            np.asarray(adata[~ctrl_mask].X.mean(axis=0)).ravel().astype(np.float32)
        )

        # Remove control from training targets
        if str(control_key) in group_means.index:
            group_means = group_means.drop(index=str(control_key))

        # ---- PCA -> G ----
        self.logger.info(f"Running PCA (n_comps={self.top_k})...")
        sc.pp.pca(adata, n_comps=self.top_k, svd_solver="arpack")
        if "PCs" not in adata.varm:
            raise RuntimeError("scanpy PCA did not produce adata.varm['PCs'].")

        G = np.asarray(adata.varm["PCs"], dtype=np.float64)  # (n_genes, k)
        self.G = G.astype(np.float32, copy=False)

        # ---- Training set in PCA space ----
        target_genes = [str(tg) for tg in group_means.index.tolist()]
        X_list, Y_list = [], []

        for tg in target_genes:
            idx = self.gene2idx.get(tg, None)
            if idx is None:
                continue

            x = G[idx, :]  # (k,)
            delta = group_means.loc[tg].values.astype(
                np.float64
            ) - self.mean_perturbed.astype(np.float64)  # (n_genes,)

            z, *_ = np.linalg.lstsq(G, delta, rcond=None)  # (k,)

            X_list.append(x)
            Y_list.append(z)

        if len(X_list) < 5:
            raise ValueError(
                f"Too few training perturbations after filtering (n={len(X_list)}). Cannot fit random forest."
            )

        X_train = np.vstack(X_list)  # (n_samples, k)
        Y_train = np.vstack(Y_list)  # (n_samples, k)

        self.logger.info(
            f"Training RandomForestRegressor: samples={X_train.shape[0]}, features={X_train.shape[1]}, "
            f"targets_dim={Y_train.shape[1]}, n_estimators={self.n_estimators}"
        )

        self.rf = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self.rf.fit(X_train, Y_train)

        self._is_fitted = True

    def _predict_profile_for_target(self, target_gene: str) -> np.ndarray:
        if not self._is_fitted or self.rf is None:
            raise RuntimeError(
                "RandomForestWrapper not fitted. Call fit_from_adata() first."
            )

        idx = self.gene2idx.get(str(target_gene), None)
        if idx is None:
            return self.mean_perturbed

        x = self.G[idx, :].astype(np.float64, copy=False).reshape(1, -1)  # (1, k)
        z_hat = self.rf.predict(x).reshape(-1)  # (k,)
        y_hat = (self.G.astype(np.float64) @ z_hat.reshape(-1, 1)).reshape(
            -1
        ) + self.mean_perturbed.astype(np.float64)

        return y_hat.astype(np.float32, copy=False)

    def predict(
        self,
        batch_data,
        gene_ids,
        include_zero_gene="batch-wise",
        amp=True,
        target_gene=None,
        **kwargs,
    ):
        if not self._is_fitted:
            raise RuntimeError(
                "RandomForestWrapper has not been fitted. Call fit_from_adata() first."
            )
        if target_gene is None:
            raise ValueError(
                "RandomForestWrapper.predict() requires target_gene. Update main.py to pass target_gene=target."
            )

        batch_size = batch_data.x.shape[0]
        n_genes = batch_data.x.shape[2]

        profile = self._predict_profile_for_target(str(target_gene))

        if profile.shape[0] != n_genes:
            self.logger.warning(
                f"Gene count mismatch: predicted profile has {profile.shape[0]} genes, input expects {n_genes}. "
                "Truncating/padding."
            )
            if profile.shape[0] > n_genes:
                profile = profile[:n_genes]
            else:
                profile = np.pad(
                    profile, (0, n_genes - profile.shape[0]), mode="constant"
                )

        pred = torch.from_numpy(profile).float().unsqueeze(0).expand(batch_size, -1)
        if batch_data.x.is_cuda:
            pred = pred.to(batch_data.x.device)
        return pred
