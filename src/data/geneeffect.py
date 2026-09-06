"""data / geneeffect."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final
import numpy as np
import pandas as pd
from src.data.residual_target import ResidualTargets, build_residual_targets

SPLIT_COUNTS: Final = {"train": 172, "val": 27, "test": 27}


UNLABELED_TRAIN: Final = ("ACH-000779", "ACH-001086")


RAW_UMI_SEMANTICS: Final = "raw_umi_counts"


SPLIT_SCHEMA_VERSION: Final = "cell-line-geneeffect-226-split-v1"


PINNED_SPLIT_SHA256: Final = (
    "1c91d3a821f50bb5aadfb9fac86b889311e194380c64b484463c1f437c41ed72"
)


PINNED_COPY_PRIOR_SHA256: Final = (
    "b9a20999cdfdf94fd88f004d0d661e3e6d3c3214b9f8ed1b1ad9a0c276d3f785"
)


_REGISTRY_COLUMNS: Final = (
    "model_id",
    "source_path",
    "source_kind",
    "matrix_semantics",
)


_GENE_EFFECT_COLUMN_RE: Final = re.compile(r"^(?P<symbol>\S+) \((?P<entrez>\d+)\)$")


@dataclass(frozen=True)
class Exp13Split:
    """The fixed split authority, keyed only by DepMap ModelID."""

    train: tuple[str, ...]
    val: tuple[str, ...]
    test: tuple[str, ...]
    unlabeled_train: tuple[str, ...]

    @property
    def all_model_ids(self) -> tuple[str, ...]:
        return self.train + self.val + self.test

    @property
    def supervised_train(self) -> tuple[str, ...]:
        excluded = set(self.unlabeled_train)
        return tuple(model_id for model_id in self.train if model_id not in excluded)

    def to_manifest(self) -> dict[str, object]:
        return {
            "counts": {name: len(getattr(self, name)) for name in SPLIT_COUNTS},
            "train": list(self.train),
            "val": list(self.val),
            "test": list(self.test),
            "unlabeled_train": list(self.unlabeled_train),
        }


@dataclass(frozen=True)
class ScoredUniverse:
    """Coverage-qualified genes in the explicit ESM2 input order."""

    symbols: tuple[str, ...]
    coverage: pd.DataFrame
    manifest: dict[str, object]


@dataclass(frozen=True)
class ResidualData:
    """Train-only gene means and residual targets for the scored universe."""

    targets: ResidualTargets
    manifest: dict[str, object]


@dataclass(frozen=True)
class VariableGenes:
    """Upper-quartile train-residual-variance genes, with ties included."""

    symbols: tuple[str, ...]
    variance: pd.Series
    threshold: float
    manifest: dict[str, object]


def _unique_strings(values: Sequence[object], label: str) -> tuple[str, ...]:
    result = tuple(str(value) for value in values)
    if any(not value for value in result):
        raise ValueError(f"{label} contains an empty value")
    duplicates = sorted(value for value, count in Counter(result).items() if count > 1)
    if duplicates:
        raise ValueError(f"{label} contains duplicates: {duplicates[:10]}")
    return result


def load_exp13_split(path: Path) -> Exp13Split:
    """Load and strictly validate the tracked 172/27/27 split authority."""
    path = Path(path)
    observed_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed_sha256 != PINNED_SPLIT_SHA256:
        raise ValueError(
            f"Exp13 split SHA-256 mismatch: {observed_sha256} != {PINNED_SPLIT_SHA256}"
        )
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != SPLIT_SCHEMA_VERSION:
        raise ValueError(f"split schema_version must be {SPLIT_SCHEMA_VERSION!r}")
    required = {"train", "val", "test", "unlabeled_train"}
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"split is missing keys: {missing}")
    groups = {
        name: _unique_strings(payload[name], f"split {name}") for name in required
    }
    for name, expected in SPLIT_COUNTS.items():
        if len(groups[name]) != expected:
            raise ValueError(
                f"split {name} has {len(groups[name])} ModelIDs; expected {expected}"
            )
    membership = groups["train"] + groups["val"] + groups["test"]
    if len(set(membership)) != sum(SPLIT_COUNTS.values()):
        raise ValueError("ModelID membership overlaps across train/val/test")
    if groups["unlabeled_train"] != UNLABELED_TRAIN:
        raise ValueError(
            "unlabeled_train must be exactly PC9 ACH-000779 and HeLa ACH-001086"
        )
    if not set(groups["unlabeled_train"]).issubset(groups["train"]):
        raise ValueError("unlabeled_train contains a non-train ModelID")
    return Exp13Split(
        train=groups["train"],
        val=groups["val"],
        test=groups["test"],
        unlabeled_train=groups["unlabeled_train"],
    )


def parse_gene_symbol(column: object) -> str:
    """Parse and normalize DepMap's ``SYMBOL (EntrezID)`` header."""
    text = str(column).strip()
    match = _GENE_EFFECT_COLUMN_RE.fullmatch(text)
    if match is None:
        raise ValueError(f"invalid DepMap GeneEffect column: {text!r}")
    return match.group("symbol").upper()


def load_geneeffect_long(path: Path, split: Exp13Split) -> pd.DataFrame:
    """Load CRISPRGeneEffect.csv into canonical ModelID/symbol long form."""
    wide = pd.read_csv(path, index_col=0)
    wide.index = wide.index.astype(str)
    if not wide.index.is_unique:
        duplicates = sorted(wide.index[wide.index.duplicated()].unique())
        raise ValueError(f"GeneEffect contains duplicate ModelIDs: {duplicates[:10]}")
    symbols = [parse_gene_symbol(column) for column in wide.columns]
    if len(symbols) != len(set(symbols)):
        duplicates = sorted(
            symbol for symbol, count in Counter(symbols).items() if count > 1
        )
        raise ValueError(
            f"GeneEffect columns map to duplicate symbols: {duplicates[:10]}"
        )
    expected_missing = set(split.unlabeled_train)
    observed_missing = set(split.all_model_ids) - set(wide.index)
    if observed_missing != expected_missing:
        raise ValueError(
            "split label coverage does not match unlabeled_train: "
            f"missing={sorted(observed_missing)}"
        )
    selected_ids = [
        model_id for model_id in split.all_model_ids if model_id in wide.index
    ]
    selected = wide.loc[selected_ids]
    numeric = selected.apply(pd.to_numeric, errors="coerce")
    invalid = selected.notna() & numeric.isna()
    if invalid.any().any():
        invalid_positions = np.argwhere(invalid.to_numpy())[:10]
        examples = [
            (
                str(selected.index[row]),
                str(selected.columns[column]),
                str(selected.iloc[row, column]),
            )
            for row, column in invalid_positions
        ]
        raise ValueError(
            f"GeneEffect contains nonnumeric nonmissing values: {examples}"
        )
    infinite = np.isinf(numeric.to_numpy())
    if infinite.any():
        raise ValueError(f"GeneEffect contains {int(infinite.sum())} infinite values")
    numeric.columns = symbols
    long = (
        numeric.rename_axis("model_id")
        .reset_index()
        .melt(id_vars="model_id", var_name="gene_symbol", value_name="gene_effect")
    )
    return long


def build_scored_universe(
    labels: pd.DataFrame,
    split: Exp13Split,
    esm2_symbols: Sequence[str],
) -> ScoredUniverse:
    """Intersect label coverage with ESM2 resolution, preserving ESM2 order."""
    required = {"model_id", "gene_symbol", "gene_effect"}
    missing = sorted(required - set(labels.columns))
    if missing:
        raise ValueError(f"labels is missing columns: {missing}")
    if labels[["model_id", "gene_symbol"]].duplicated().any():
        raise ValueError("labels contains duplicate (model_id, gene_symbol) rows")
    unknown = sorted(set(labels["model_id"]) - set(split.all_model_ids))
    if unknown:
        raise ValueError(f"labels contains ModelIDs outside the split: {unknown[:10]}")
    ordered = _unique_strings(esm2_symbols, "ESM2 symbols")
    counts: dict[str, pd.Series] = {}
    for name in SPLIT_COUNTS:
        ids = set(getattr(split, name))
        counts[name] = (
            labels.loc[labels["model_id"].isin(ids)]
            .groupby("gene_symbol")["gene_effect"]
            .count()
        )
    all_label_symbols = set(labels["gene_symbol"].astype(str))
    esm2_set = set(ordered)
    rows = []
    for symbol in [*ordered, *sorted(all_label_symbols - esm2_set)]:
        train = int(counts["train"].get(symbol, 0))
        val = int(counts["val"].get(symbol, 0))
        test = int(counts["test"].get(symbol, 0))
        reasons = []
        if train < 5:
            reasons.append("train_finite_lt5")
        if val < 3:
            reasons.append("val_finite_lt3")
        if test < 3:
            reasons.append("test_finite_lt3")
        if symbol not in esm2_set:
            reasons.append("esm2_unresolved")
        rows.append(
            {
                "gene_symbol": symbol,
                "train_finite": train,
                "val_finite": val,
                "test_finite": test,
                "esm2_resolved": symbol in esm2_set,
                "included": not reasons,
                "drop_reason": "|".join(reasons),
            }
        )
    coverage = pd.DataFrame(rows)
    included = set(coverage.loc[coverage["included"], "gene_symbol"])
    symbols = tuple(symbol for symbol in ordered if symbol in included)
    reason_counts = (
        coverage.loc[~coverage["included"], "drop_reason"].value_counts().to_dict()
    )
    manifest = {
        "esm2_input_count": len(ordered),
        "label_symbol_count": len(all_label_symbols),
        "scored_gene_count": len(symbols),
        "scored_symbols": list(symbols),
        "drop_reason_counts": {
            str(key): int(value) for key, value in reason_counts.items()
        },
    }
    return ScoredUniverse(symbols=symbols, coverage=coverage, manifest=manifest)


def restrict_scored_universe_to_copy_prior(
    universe: ScoredUniverse, copy_prior_symbols: Sequence[str]
) -> ScoredUniverse:
    """Apply the one-universe K562 finite-label gate in existing gene order."""
    eligible = set(_unique_strings(copy_prior_symbols, "copy-prior symbols"))
    coverage = universe.coverage.copy()
    prior_missing = coverage["included"] & ~coverage["gene_symbol"].isin(eligible)
    coverage.loc[prior_missing, "included"] = False
    coverage.loc[prior_missing, "drop_reason"] = "copy_prior_missing"
    symbols = tuple(symbol for symbol in universe.symbols if symbol in eligible)
    reason_counts = (
        coverage.loc[~coverage["included"], "drop_reason"].value_counts().to_dict()
    )
    manifest = {
        **universe.manifest,
        "pre_copy_prior_gene_count": len(universe.symbols),
        "scored_gene_count": len(symbols),
        "scored_symbols": list(symbols),
        "copy_prior_missing_count": int(prior_missing.sum()),
        "drop_reason_counts": {
            str(key): int(value) for key, value in reason_counts.items()
        },
    }
    return ScoredUniverse(symbols=symbols, coverage=coverage, manifest=manifest)


def build_residual_data(
    labels: pd.DataFrame, split: Exp13Split, universe: ScoredUniverse
) -> ResidualData:
    """Fit train-only means and residuals for the final scored universe."""
    selected = labels.loc[labels["gene_symbol"].isin(universe.symbols)].copy()
    targets = build_residual_targets(selected, split.supervised_train)
    if tuple(targets.gene_mean.index) != tuple(sorted(universe.symbols)):
        raise ValueError("residual target genes disagree with the scored universe")
    return ResidualData(
        targets=targets,
        manifest={
            "fit_line_count": len(split.supervised_train),
            "excluded_unlabeled_train": list(split.unlabeled_train),
            "gene_count": len(targets.gene_mean),
            "row_count": len(targets.long),
        },
    )


def build_g_var(
    residual_data: ResidualData,
    split: Exp13Split,
    universe: ScoredUniverse,
) -> VariableGenes:
    """Select genes at or above the train residual-variance 75th percentile."""
    long = residual_data.targets.long
    train = long.loc[long["model_id"].isin(split.supervised_train)]
    grouped = train.groupby("gene_symbol")["residual"]
    counts = grouped.count().reindex(universe.symbols, fill_value=0)
    eligible = counts[counts >= 5].index
    variance = grouped.var(ddof=0).reindex(eligible)
    if variance.empty or not np.isfinite(variance.to_numpy()).all():
        raise ValueError("G_var has no finite eligible train residual variances")
    threshold = float(np.percentile(variance.to_numpy(), 75, method="linear"))
    selected_set = set(variance.index[variance >= threshold])
    symbols = tuple(symbol for symbol in universe.symbols if symbol in selected_set)
    return VariableGenes(
        symbols=symbols,
        variance=variance,
        threshold=threshold,
        manifest={
            "variance_ddof": 0,
            "eligible_gene_count": len(variance),
            "percentile": 75,
            "percentile_method": "linear",
            "threshold": threshold,
            "selected_gene_count": len(symbols),
            "ties_included": True,
            "symbols": list(symbols),
        },
    )


def load_source_registry(path: Path, split: Exp13Split) -> pd.DataFrame:
    """Validate the complete raw-UMI source registry for all 226 ModelIDs."""
    registry = pd.read_csv(path, dtype=str)
    missing_columns = sorted(set(_REGISTRY_COLUMNS) - set(registry.columns))
    if missing_columns:
        raise ValueError(f"source registry is missing columns: {missing_columns}")
    registry = registry.loc[:, _REGISTRY_COLUMNS].copy()
    if registry.isna().any().any() or (registry == "").any().any():
        raise ValueError("source registry contains missing values")
    if registry["model_id"].duplicated().any():
        duplicates = sorted(
            registry.loc[registry["model_id"].duplicated(False), "model_id"].unique()
        )
        raise ValueError(
            f"source registry contains duplicate ModelIDs: {duplicates[:10]}"
        )
    if registry["source_path"].duplicated().any():
        raise ValueError("source registry contains duplicate source_path values")
    expected = set(split.all_model_ids)
    observed = set(registry["model_id"])
    if observed != expected:
        raise ValueError(
            f"source registry ModelID mismatch: missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )
    bad = registry.loc[registry["matrix_semantics"] != RAW_UMI_SEMANTICS]
    if not bad.empty:
        raise ValueError(
            "source registry contains non-raw-UMI semantics for ModelIDs: "
            f"{sorted(bad['model_id'])[:10]}"
        )
    bad_kind = registry.loc[registry["source_kind"] != "h5ad"]
    if not bad_kind.empty:
        raise ValueError(
            "source registry contains unsupported source_kind for ModelIDs: "
            f"{sorted(bad_kind['model_id'])[:10]}"
        )
    return registry.set_index("model_id", verify_integrity=True).loc[
        list(split.all_model_ids)
    ]
