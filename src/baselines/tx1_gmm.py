"""Basal Tx1 population distributions with train-only per-gene ridge readouts."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from src.data.splits import assert_fit_eligible, validate_fixed_split

if TYPE_CHECKING:
    from src.data.prepared import PreparedInputs


@dataclass
class Tx1GMMRidge:
    cell_scaler: StandardScaler
    gmm: GaussianMixture
    context_scaler: StandardScaler
    genes: tuple[str, ...]
    gene_means: np.ndarray
    coefficients: np.ndarray
    intercepts: np.ndarray
    diagnostics: dict

    @staticmethod
    def _bag(value, width=None):
        bag = np.asarray(value, dtype=np.float64)
        if (
            bag.ndim != 2
            or min(bag.shape) == 0
            or (width is not None and bag.shape[1] != width)
            or not np.isfinite(bag).all()
        ):
            raise ValueError("Tx1 bags must be nonempty finite matrices of equal width")
        return bag

    @staticmethod
    def _features(bag, cell_scaler, gmm):
        scaled = cell_scaler.transform(bag)
        responsibilities = gmm.predict_proba(scaled)
        occupancy = responsibilities.mean(axis=0)
        positive = occupancy > 0
        entropy = -np.sum(occupancy[positive] * np.log(occupancy[positive]))
        return np.concatenate(
            (
                occupancy,
                [
                    entropy,
                    np.exp(entropy),
                    responsibilities.max(axis=1).mean(),
                    -gmm.score_samples(scaled).mean(),
                ],
            )
        )

    @property
    def feature_names(self):
        return [f"occupancy_{i}" for i in range(self.gmm.n_components)] + [
            "occupancy_entropy",
            "effective_components",
            "assignment_confidence",
            "nll",
        ]

    @classmethod
    def fit(cls, inputs: "PreparedInputs", *, n_components: int = 64):
        """Fit only labeled train contexts; a smaller K is useful for unit fixtures."""
        validate_fixed_split(inputs.split)
        train_ids = inputs.split.supervised_train
        if len(train_ids) < 3:
            raise ValueError("GMM-ridge requires at least three labeled train contexts")
        if type(n_components) is not int or n_components < 1:
            raise ValueError("n_components must be a positive integer")
        bags = {}
        width = None
        for model_id in train_ids:
            assert_fit_eligible(model_id, inputs.split)
            bag = cls._bag(inputs.lines[model_id].controls_tx1, width)
            width = bag.shape[1]
            bags[model_id] = bag
        cells_per_line = min(len(bag) for bag in bags.values())
        rng = np.random.default_rng(0)
        positions = {
            model_id: np.sort(rng.choice(len(bag), cells_per_line, replace=False))
            for model_id, bag in bags.items()
        }
        fit_cells = np.concatenate(
            [bags[model_id][positions[model_id]] for model_id in train_ids]
        )
        if len(fit_cells) < max(2, n_components):
            raise ValueError("balanced training cells cannot support the requested GMM")
        cell_scaler = StandardScaler().fit(fit_cells)
        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type="diag",
            random_state=0,
            reg_covar=1e-4,
            max_iter=200,
            n_init=1,
            tol=1e-3,
        ).fit(cell_scaler.transform(fit_cells))
        contexts = np.stack(
            [cls._features(bags[model_id], cell_scaler, gmm) for model_id in train_ids]
        )
        if not np.isfinite(contexts).all():
            raise ValueError("non-finite GMM context features")
        context_scaler = StandardScaler().fit(contexts)
        x = context_scaler.transform(contexts)
        labels = inputs.labels.loc[inputs.labels.model_id.isin(train_ids)]
        if labels.duplicated(["model_id", "gene_symbol"]).any():
            raise ValueError("duplicate training ModelID/gene labels")
        if not inputs.genes or len(set(inputs.genes)) != len(inputs.genes):
            raise ValueError("common gene panel must be nonempty and unique")
        means = inputs.train_gene_means.loc[list(inputs.genes)].to_numpy(dtype=float)
        if not np.isfinite(means).all():
            raise ValueError("training gene means must be finite")
        targets = labels.pivot(
            index="model_id", columns="gene_symbol", values="gene_effect"
        )
        targets = targets.reindex(index=train_ids, columns=inputs.genes).to_numpy(
            dtype=float
        )
        coefficients, intercepts, counts = [], [], {}
        for i, gene in enumerate(inputs.genes):
            valid = np.isfinite(targets[:, i])
            counts[gene] = int(valid.sum())
            if counts[gene] < 3:
                raise ValueError(f"{gene} has fewer than three finite train contexts")
            ridge = Ridge(alpha=1.0).fit(x[valid], targets[valid, i] - means[i])
            coefficients.append(ridge.coef_)
            intercepts.append(ridge.intercept_)
        diagnostics = {
            "method": "tx1_gmm_ridge",
            "seed": 0,
            "ridge_alpha": 1.0,
            "gmm_components": n_components,
            "gmm_parameters": gmm.get_params(),
            "tx1_width": width,
            "gmm_converged": bool(gmm.converged_),
            "gmm_iterations": int(gmm.n_iter_),
            "gmm_lower_bound": float(gmm.lower_bound_),
            "gmm_weights": gmm.weights_.tolist(),
            "train_mean_occupancy": contexts[:, :n_components].mean(0).tolist(),
            "train_mean_assignment_confidence": float(contexts[:, -2].mean()),
            "fit_model_ids": list(train_ids),
            "fit_cells_per_line": cells_per_line,
            "fit_cell_positions": {
                key: value.tolist() for key, value in positions.items()
            },
            "prepared_bag_lengths": {key: len(value) for key, value in bags.items()},
            "fit_cell_rows": len(fit_cells),
            "gene_train_observations": counts,
        }
        return cls(
            cell_scaler,
            gmm,
            context_scaler,
            tuple(inputs.genes),
            means,
            np.stack(coefficients),
            np.asarray(intercepts),
            diagnostics,
        )

    def context_features(self, bags) -> pd.DataFrame:
        if not bags:
            raise ValueError("at least one context bag is required")
        features = np.stack(
            [
                self._features(
                    self._bag(bag, self.cell_scaler.n_features_in_),
                    self.cell_scaler,
                    self.gmm,
                )
                for bag in bags.values()
            ]
        )
        if not np.isfinite(features).all():
            raise ValueError("non-finite GMM context features")
        return pd.DataFrame(
            features, index=pd.Index(bags, name="model_id"), columns=self.feature_names
        )

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        """Return residual predictions, with explicit context and gene axes."""
        if (
            list(features.columns) != self.feature_names
            or features.index.has_duplicates
        ):
            raise ValueError("context feature order or ModelID uniqueness mismatch")
        x = self.context_scaler.transform(features.to_numpy(dtype=float))
        prediction = x @ self.coefficients.T + self.intercepts
        if not np.isfinite(prediction).all():
            raise ValueError("non-finite GMM-ridge predictions")
        return pd.DataFrame(prediction, index=features.index, columns=self.genes)
