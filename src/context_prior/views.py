"""Context views a query line can supply, computed identically from bulk and
bridged pseudo-bulk: expression components, pathway scores, predicted genotype."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge

from src.context_prior.reference import Reference
from src.data.context_pca import ContextPCA, fit_context_pca

#: A lineage with fewer training-side lines is pooled into "other".
MIN_LINEAGE_LINES = 20


def fit_expression_components(
    expression: pd.DataFrame, n_components: int
) -> ContextPCA:
    """Label-free, eigen-scaled principal components of expression rows."""
    return fit_context_pca(expression.to_numpy(dtype=np.float64), n_components)


def expression_components(pca: ContextPCA, expression: pd.DataFrame) -> pd.DataFrame:
    """Component scores of expression rows whose columns are in the fit's order."""
    scores = pca.transform(expression.to_numpy(dtype=np.float64))
    columns = [f"pc{i + 1}" for i in range(scores.shape[1])]
    return pd.DataFrame(scores, index=expression.index, columns=columns)


def pathway_scores(
    expression: pd.DataFrame, hallmark: pd.DataFrame, progeny: pd.DataFrame
) -> pd.DataFrame:
    """Hallmark: centred mean within-line rank of the set's genes; PROGENy:
    weighted sum of its footprint genes. Gene sets with no present gene are left out."""
    genes = expression.columns
    ranks = expression.rank(axis=1, method="average").to_numpy() / len(genes) - 0.5
    values = expression.to_numpy(dtype=np.float64)
    columns: dict[str, np.ndarray] = {}
    for name, members in hallmark.groupby("gene_set")["gene"]:
        index = genes.get_indexer(members.unique())
        index = index[index >= 0]
        if index.size:
            columns[f"hallmark:{name}"] = ranks[:, index].mean(axis=1)
    for name, rows in progeny.groupby("pathway"):
        index = genes.get_indexer(rows["gene"])
        present = index >= 0
        if present.any():
            columns[f"progeny:{name}"] = (
                values[:, index[present]] @ rows["weight"].to_numpy()[present]
            )
    return pd.DataFrame(columns, index=expression.index)


@dataclass(frozen=True)
class GenotypeEncoders:
    """Driver status, MSI score and lineage predicted from expression components."""

    drivers: Mapping[str, Any]  # gene -> classifier, or the prevalence if one class
    msi: Ridge
    lineage: Any  # LogisticRegression, or the one class seen
    lineage_classes: tuple[str, ...]

    def predict(self, components: pd.DataFrame) -> pd.DataFrame:
        x = components.to_numpy(dtype=np.float64)
        columns: dict[str, np.ndarray] = {}
        for gene, model in self.drivers.items():
            columns[f"driver:{gene}"] = (
                model.predict_proba(x)[:, 1]
                if isinstance(model, LogisticRegression)
                else np.full(len(x), float(model))
            )
        columns["msi"] = self.msi.predict(x)
        if isinstance(self.lineage, LogisticRegression):
            probability = self.lineage.predict_proba(x)
            classes = list(self.lineage.classes_)
        else:
            probability, classes = np.ones((len(x), 1)), [self.lineage]
        for name in self.lineage_classes:
            columns[f"lineage:{name}"] = (
                probability[:, classes.index(name)]
                if name in classes
                else np.zeros(len(x))
            )
        return pd.DataFrame(columns, index=components.index)


def _fit_encoders(
    components: pd.DataFrame,
    reference: Reference,
    lineage: pd.Series,
    drivers: Sequence[str],
    lineage_classes: Sequence[str],
) -> GenotypeEncoders:
    rows = components.index
    driver_rows = rows.intersection(reference.drivers.index)
    models: dict[str, Any] = {}
    for gene in drivers:
        y = reference.drivers.loc[driver_rows, gene].to_numpy()
        if 0 < y.sum() < len(y):
            models[gene] = LogisticRegression(C=1.0, max_iter=2000).fit(
                components.loc[driver_rows].to_numpy(), y
            )
        else:
            models[gene] = float(y.mean()) if len(y) else 0.0
    msi_rows = rows.intersection(reference.msi.index)
    msi = Ridge(alpha=1.0).fit(
        components.loc[msi_rows].to_numpy(), reference.msi.loc[msi_rows].to_numpy()
    )
    labelled = rows.intersection(lineage.dropna().index)
    labels = lineage.loc[labelled].where(
        lineage.loc[labelled].isin(lineage_classes), "other"
    )
    if labels.nunique() > 1:
        classifier = LogisticRegression(C=1.0, max_iter=2000).fit(
            components.loc[labelled].to_numpy(), labels.to_numpy()
        )
    else:
        classifier = str(labels.iloc[0]) if len(labels) else "other"
    return GenotypeEncoders(models, msi, classifier, tuple(lineage_classes))


def fit_genotype(
    components: pd.DataFrame,
    reference: Reference,
    lineage: pd.Series,
    folds: Mapping[str, int],
) -> tuple[GenotypeEncoders, pd.DataFrame]:
    """Encoders fitted on every row, and each row's own out-of-sample features.

    A row's features come from the encoders of the inner fold that excludes it;
    query rows use the returned encoders, fitted on every row. The driver list and
    the lineage classes are fixed from every row so each inner fold yields the same
    columns.
    """
    rows = list(components.index)
    drivers = tuple(reference.drivers.columns)
    counts = lineage.reindex(rows).value_counts()
    classes = (*sorted(counts.index[counts >= MIN_LINEAGE_LINES]), "other")
    encoders = _fit_encoders(components, reference, lineage, drivers, classes)
    own = encoders.predict(components)
    for fold in sorted({folds[m] for m in rows}):
        held = [m for m in rows if folds[m] == fold]
        rest = [m for m in rows if folds[m] != fold]
        inner = _fit_encoders(
            components.loc[rest], reference, lineage, drivers, classes
        )
        own.loc[held] = inner.predict(components.loc[held])[own.columns].to_numpy()
    return encoders, own
