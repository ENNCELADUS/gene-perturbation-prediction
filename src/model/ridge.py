import sys
import torch
import logging
import numpy as np
import pandas as pd
import scipy.sparse as sp
from pathlib import Path

# Optional dependency for PCA (used by this ridge baseline)
try:
    import scanpy as sc
except Exception:  # pragma: no cover
    sc = None

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


class RidgeWrapper:
    """
    Ridge (PCA-space) linear model.

    This implements the ridge/PCA method in the user's reference snippet:
      - Pseudo-bulk means per perturbation (group_means)
      - mean_unperturbed / mean_perturbed
      - PCA gene loadings G (n_genes x k)
      - Fit W (k x k) with ridge penalties:
            W = inv(G^T G + λI) * G^T * Δ * P * inv(P^T P + λI)

    At inference, for a given target_gene:
        yhat = G * W * P_val^T + mean_perturbed
    and the predicted mean profile is broadcast to batch size.
    """

    def __init__(self, config, logger=None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.vocab = None

        # Hyperparams (optionally configurable)
        ridge_cfg = self.config.get("ridge", {}) or self.config.get("baseline", {})
        self.top_k = int(ridge_cfg.get("top_k", 50))
        self.lmbda = float(ridge_cfg.get("ridge_lambda", 0.1))

        # Learned state
        self.G = None  # (n_genes, top_k)
        self.W = None  # (top_k, top_k)
        self.mean_unperturbed = None
        self.mean_perturbed = None
        self.gene2idx = None
        self.var_names = None

        self._is_fitted = False
        self._load_vocab()

    def _load_vocab(self):
        # Vocab needed for data loader compatibility
        model_dir = Path(self.config["paths"]["model_dir"])
        vocab_file = model_dir / "vocab.json"
        if not vocab_file.exists():
            raise FileNotFoundError(f"Vocab file not found: {vocab_file}")

        self.vocab = GeneVocab.from_file(vocab_file)

        # Ensure special tokens (matching ScGPTWrapper logic)
        special_tokens = [self.config["model"]["pad_token"], "<cls>", "<eoc>"]
        for s in special_tokens:
            if s not in self.vocab:
                self.vocab.append_token(s)

        self.logger.info(f"Loaded vocab from {vocab_file}")

    def fit(self, train_loader):
        # Keep API compatibility, but this ridge model expects AnnData.
        raise NotImplementedError(
            "RidgeWrapper expects AnnData input. "
            "Call fit_from_adata(train_adata, condition_key=..., control_key=...)."
        )

    def fit_from_adata(self, adata, condition_key="condition", control_key="ctrl"):
        if sc is None:
            raise ImportError(
                "scanpy is required for RidgeWrapper (PCA). Please install scanpy."
            )

        self.logger.info("Fitting ridge model from AnnData...")

        adata = adata.copy()
        adata.obs[condition_key] = adata.obs[condition_key].astype(str)

        self.var_names = adata.var_names
        self.gene2idx = {g: i for i, g in enumerate(self.var_names)}

        n_genes = adata.n_vars
        self.top_k = min(int(self.top_k), n_genes)
        if self.top_k < 1:
            raise ValueError("top_k must be >= 1")

        # ---- 1) group_means (pseudo-bulk means per target gene) ----
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
        if str(control_key) in group_means.index:
            group_means = group_means.drop(index=str(control_key))

        # ---- 2) mean profiles ----
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

        # ---- 3) PCA gene loadings ----
        perturbed_genes = set(adata.obs[condition_key].unique().tolist())
        perturbed_genes.discard(str(control_key))
        perturbed_mask = self.var_names.isin(list(perturbed_genes))

        self.logger.info(f"Running PCA (n_comps={self.top_k})...")
        sc.pp.pca(adata, n_comps=self.top_k, svd_solver="arpack")
        if "PCs" not in adata.varm:
            raise RuntimeError("scanpy PCA did not produce adata.varm['PCs'].")

        G = np.asarray(adata.varm["PCs"], dtype=np.float64)  # (n_genes, top_k)
        P = G[perturbed_mask, :]  # (n_perturbed_genes, top_k)

        # Align columns
        group_means = group_means.loc[:, self.var_names]
        delta = (group_means.values - self.mean_perturbed).T  # (n_genes, n_targets)

        if P.shape[0] == 0:
            self.logger.warning(
                "No perturbed genes overlap with var_names. Falling back to mean_perturbed."
            )
            self.G = G.astype(np.float32, copy=False)
            self.W = None
            self._is_fitted = True
            return

        # ---- 4) ridge fit W ----
        lmbda = float(self.lmbda)
        I = np.eye(self.top_k, dtype=np.float64)

        self.W = (
            np.linalg.inv(G.T @ G + lmbda * I)
            @ G.T
            @ delta
            @ P
            @ np.linalg.inv(P.T @ P + lmbda * I)
        ).astype(np.float32, copy=False)

        self.G = G.astype(np.float32, copy=False)
        self._is_fitted = True

        self.logger.info(
            f"Ridge fitted: n_genes={n_genes}, top_k={self.top_k}, "
            f"n_targets={group_means.shape[0]}, lambda={self.lmbda}"
        )

    def _predict_profile_for_target(self, target_gene: str) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError(
                "RidgeWrapper has not been fitted. Call fit_from_adata() first."
            )

        if self.G is None or self.W is None or self.gene2idx is None:
            return self.mean_perturbed

        idx = self.gene2idx.get(str(target_gene), None)
        if idx is None:
            return self.mean_perturbed

        P_val = self.G[idx : idx + 1, :]  # (1, top_k)
        yhat = (self.G @ self.W @ P_val.T).reshape(-1) + self.mean_perturbed
        return yhat.astype(np.float32, copy=False)

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
                "RidgeWrapper has not been fitted. Call fit_from_adata() first."
            )

        batch_size = batch_data.x.shape[0]
        n_genes = batch_data.x.shape[2]

        if target_gene is None:
            raise ValueError(
                "RidgeWrapper.predict() requires target_gene. "
                "Update main.py to pass target_gene=target."
            )

        profile = self._predict_profile_for_target(str(target_gene))

        # Match expected gene count
        if profile.shape[0] != n_genes:
            self.logger.warning(
                f"Gene count mismatch: predicted profile has {profile.shape[0]} genes, "
                f"input expects {n_genes} genes. Truncating/padding."
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
