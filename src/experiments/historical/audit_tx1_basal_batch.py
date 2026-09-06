#!/usr/bin/env python3
"""Audit Tahoe-DMSO versus CCLE-bulk basal source confounding."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

GENE_RE = re.compile(r"^(.*?) \((\d+)\)$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reservoir_cells(
    shard_dir: Path,
    target_cellosaurus_ids: set[str],
    cells_per_line: int,
    seed: int,
) -> tuple[dict[str, list[tuple[np.ndarray, np.ndarray]]], dict[str, int]]:
    """Deterministically reservoir-sample DMSO cells from all materialized shards."""
    rng = np.random.default_rng(seed)
    reservoirs: dict[str, list[tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
    seen: dict[str, int] = defaultdict(int)
    paths = sorted(shard_dir.glob("*.parquet"))
    if not paths:
        raise ValueError(f"No parquet shards found under {shard_dir}")
    for path in paths:
        frame = pd.read_parquet(path, columns=["genes", "expressions", "cell_line_id"])
        frame = frame[frame["cell_line_id"].isin(target_cellosaurus_ids)]
        for row in frame.itertuples(index=False):
            line = str(row.cell_line_id)
            seen[line] += 1
            cell = (
                np.asarray(row.genes, dtype=np.int32),
                np.asarray(row.expressions, dtype=np.float32),
            )
            if len(reservoirs[line]) < cells_per_line:
                reservoirs[line].append(cell)
                continue
            replacement = int(rng.integers(0, seen[line]))
            if replacement < cells_per_line:
                reservoirs[line][replacement] = cell
    missing = sorted(target_cellosaurus_ids - set(reservoirs))
    if missing:
        raise ValueError(f"No DMSO cells found for: {missing}")
    return dict(reservoirs), dict(seen)


def tahoe_pseudobulk(
    reservoirs: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    gene_metadata: pd.DataFrame,
) -> pd.DataFrame:
    token_to_symbol = gene_metadata.set_index("token_id")["gene_symbol"].astype(str)
    max_token = int(gene_metadata["token_id"].max())
    rows: dict[str, np.ndarray] = {}
    for line, cells in reservoirs.items():
        counts = np.zeros(max_token + 1, dtype=np.float64)
        for genes, values in cells:
            valid = (genes >= 3) & (genes <= max_token) & (values > 0)
            np.add.at(counts, genes[valid], values[valid])
        token_counts = pd.Series(counts[token_to_symbol.index], index=token_to_symbol)
        symbol_counts = token_counts.groupby(level=0).sum()
        total = float(symbol_counts.sum())
        if total <= 0:
            raise ValueError(f"Non-positive pseudobulk library for {line}")
        rows[line] = np.log2(1.0 + 1_000_000.0 * symbol_counts / total).to_numpy()
    symbols = token_to_symbol.groupby(token_to_symbol).first().index
    return pd.DataFrame.from_dict(rows, orient="index", columns=symbols)


def load_ccle_expression(path: Path, model_ids: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "IsDefaultEntryForModel" in frame:
        frame = frame[frame["IsDefaultEntryForModel"].astype(bool)]
    frame = frame.drop_duplicates("ModelID", keep="first").set_index("ModelID")
    missing = sorted(set(model_ids) - set(frame.index.astype(str)))
    if missing:
        raise ValueError(f"Missing paired OmicsExpression rows: {missing}")
    metadata = {
        "Unnamed: 0",
        "SequencingID",
        "ModelConditionID",
        "IsDefaultEntryForMC",
        "IsDefaultEntryForModel",
    }
    values = frame.loc[model_ids].drop(columns=list(metadata & set(frame.columns)))
    values.columns = [
        match.group(1) if (match := GENE_RE.match(str(column))) else str(column)
        for column in values.columns
    ]
    return values.T.groupby(level=0).mean().T


def _partial_r2(response: np.ndarray, reduced: np.ndarray, full: np.ndarray) -> float:
    reduced_fit = reduced @ np.linalg.lstsq(reduced, response, rcond=None)[0]
    full_fit = full @ np.linalg.lstsq(full, response, rcond=None)[0]
    reduced_rss = float(np.square(response - reduced_fit).sum())
    full_rss = float(np.square(response - full_fit).sum())
    return (reduced_rss - full_rss) / reduced_rss


def audit_metrics(
    tahoe: pd.DataFrame,
    ccle: pd.DataFrame,
    lineages: pd.Series,
    train_ids: list[str],
    evaluation_ids: list[str],
    n_genes: int = 2000,
) -> dict[str, object]:
    common = sorted(set(tahoe.columns) & set(ccle.columns))
    tahoe = tahoe.loc[evaluation_ids, common]
    ccle = ccle.loc[evaluation_ids, common]
    training_variance = (
        tahoe.loc[train_ids].var(axis=0) + ccle.loc[train_ids].var(axis=0)
    ) / 2
    genes = training_variance.nlargest(min(n_genes, len(common))).index
    tahoe = tahoe[genes]
    ccle = ccle[genes]

    def training_z(frame: pd.DataFrame) -> np.ndarray:
        mean = frame.loc[train_ids].mean(axis=0)
        std = frame.loc[train_ids].std(axis=0, ddof=1).replace(0, np.nan)
        return ((frame - mean) / std).fillna(0).to_numpy()

    tahoe_z = training_z(tahoe)
    ccle_z = training_z(ccle)
    correlation = np.corrcoef(tahoe_z, ccle_z)
    similarity = correlation[: len(evaluation_ids), len(evaluation_ids) :]
    ranks = []
    for index in range(len(evaluation_ids)):
        order = np.argsort(-similarity[index])
        ranks.append(int(np.flatnonzero(order == index)[0]) + 1)
    matched = np.diag(similarity)
    unmatched = similarity[~np.eye(len(similarity), dtype=bool)]

    combined = pd.concat([tahoe.assign(source="Tahoe"), ccle.assign(source="CCLE")])
    sources = combined.pop("source").to_numpy()
    groups = np.asarray(evaluation_ids + evaluation_ids)
    scaled = StandardScaler().fit_transform(combined)
    pcs = PCA(n_components=min(20, len(combined) - 2), random_state=0).fit_transform(
        scaled
    )
    predictions = np.empty(len(sources), dtype=object)
    for train, test in LeaveOneGroupOut().split(pcs, sources, groups):
        model = LogisticRegression(max_iter=2000, random_state=0)
        model.fit(pcs[train], sources[train])
        predictions[test] = model.predict(pcs[test])

    source = pd.get_dummies(pd.Series(sources), drop_first=True).to_numpy()
    line = pd.get_dummies(pd.Series(groups), drop_first=True).to_numpy()
    intercept = np.ones((len(groups), 1))
    source_design = np.column_stack([intercept, source])
    line_design = np.column_stack([intercept, line])
    full_design = np.column_stack([intercept, source, line])
    source_partial = _partial_r2(scaled, line_design, full_design)
    line_partial = _partial_r2(scaled, source_design, full_design)
    encoded_lineages = lineages.loc[evaluation_ids].to_dict()
    test_mask = np.asarray([model_id not in train_ids for model_id in evaluation_ids])
    risk = "high" if accuracy_score(sources, predictions) >= 0.80 else "moderate"
    return {
        "status": "complete",
        "audit_scope": "paired Tahoe-DMSO pseudobulk versus DepMap/CCLE bulk",
        "interpretation_limit": (
            "This crossed-source proxy identifies source separation independently "
            "of cell-line identity; it is not a direct Perturb-seq-control versus "
            "Tahoe comparison because those sources have no overlapping lines."
        ),
        "selected_genes": int(len(genes)),
        "gene_selection": "top training-line mean within-source variance",
        "paired_lines": int(len(evaluation_ids)),
        "heldout_lines_in_descriptive_audit": int(test_mask.sum()),
        "source_classifier_leave_one_line_out_accuracy": float(
            accuracy_score(sources, predictions)
        ),
        "source_silhouette": float(silhouette_score(pcs, sources)),
        "line_silhouette": float(silhouette_score(pcs, groups)),
        "partial_source_r2_after_line": source_partial,
        "partial_line_r2_after_source": line_partial,
        "matched_line_top1_accuracy_all": float(np.mean(np.asarray(ranks) == 1)),
        "matched_line_median_rank_all": float(np.median(ranks)),
        "matched_line_top1_accuracy_heldout": float(
            np.mean(np.asarray(ranks)[test_mask] == 1)
        ),
        "matched_line_median_rank_heldout": float(
            np.median(np.asarray(ranks)[test_mask])
        ),
        "matched_cross_source_correlation_mean": float(np.mean(matched)),
        "unmatched_cross_source_correlation_mean": float(np.mean(unmatched)),
        "risk_level": risk,
        "registered_response": (
            "Keep the unadjusted estimator as the sole gate; additionally report "
            "a Tahoe-only head-training sensitivity and a lineage-balanced macro "
            "estimate. Do not fit a line-level source covariate because every "
            "held-out line has the Tahoe source."
        ),
        "lineages": encoded_lineages,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dmso-shards", type=Path, required=True)
    parser.add_argument("--gene-metadata", type=Path, required=True)
    parser.add_argument("--expression", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cells-per-line", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260724)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.manifest)
    paired = manifest[manifest["basal_source"] == "Tahoe-100M DMSO"].copy()
    paired = paired[paired["omics_expression_available"].astype(bool)]
    paired = paired.sort_values("model_id")
    mapping = paired.set_index("cellosaurus_id")["model_id"].to_dict()
    reservoirs, observed = reservoir_cells(
        args.dmso_shards, set(mapping), args.cells_per_line, args.seed
    )
    gene_metadata = pd.read_parquet(args.gene_metadata)
    pseudobulk = tahoe_pseudobulk(reservoirs, gene_metadata).rename(index=mapping)
    model_ids = paired["model_id"].tolist()
    ccle = load_ccle_expression(args.expression, model_ids)
    train_ids = paired.loc[paired["role"] == "train_head", "model_id"].tolist()
    metrics = audit_metrics(
        pseudobulk,
        ccle,
        paired.set_index("model_id")["lineage"],
        train_ids,
        model_ids,
    )
    metrics.update(
        {
            "seed": args.seed,
            "cells_per_line": args.cells_per_line,
            "observed_dmso_cells": {
                mapping[key]: observed[key] for key in sorted(mapping)
            },
            "source_sha256": {
                "manifest": sha256_file(args.manifest),
                "gene_metadata": sha256_file(args.gene_metadata),
                "depmap_omics_expression": sha256_file(args.expression),
            },
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
