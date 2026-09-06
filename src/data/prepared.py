"""Read-only opening of fixed inputs for joint GeneEffect training."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
import torch

import src.data.geneeffect as geneeffect_data
import src.data.residual_target as residual_target
from src.data.embeddings import load_esm2_embeddings
from src.data.q_sc import QScFeatures, load_q_sc_line
from src.data.response_cache import ResponseTargetsCache
from src.data.response_cache import open_response_targets_cache
from src.data.splits import FixedSplit, load_geneeffect_226_split
from src.data.tx1_cache import load_hvg_gene_order, open_line_cache

PREPARED_METADATA_FILENAME: Final[str] = "prepared_inputs.json"
PREPARED_METADATA_SCHEMA: Final[str] = "geneeffect-joint-prepared-v1"


@dataclass(frozen=True)
class PreparedLine:
    """One line's fixed paired basal arrays and aligned q_sc summaries."""

    controls_tx1: np.ndarray
    basal_hvg: np.ndarray
    q_sc: QScFeatures


@dataclass(frozen=True)
class PreparedInputs:
    """Fixed labels, preprocessing, feature orders, and opened cache views."""

    split: FixedSplit
    labels: pd.DataFrame
    genes: tuple[str, ...]
    train_gene_means: pd.Series
    variable_genes: frozenset[str]
    tx1_cache: Path
    q_sc_cache: Path
    response_cache: Path
    hvg_order: tuple[str, ...]
    response_holdout: frozenset[tuple[str, str]]
    esm2_symbols: tuple[str, ...] = ()
    esm2_vectors: np.ndarray = field(
        default_factory=lambda: np.empty((0, 0), dtype=np.float32),
        repr=False,
        compare=False,
    )
    lines: Mapping[str, PreparedLine] = field(
        default_factory=dict, repr=False, compare=False
    )
    response_targets: ResponseTargetsCache | None = field(
        default=None, repr=False, compare=False
    )
    response_anchors: tuple[str, ...] = ()

    def preprocessing_state(self) -> dict[str, object]:
        """Return checkpoint-ready fitted state, including actual ESM2 vectors."""
        return {
            "gene_means": {
                "symbols": list(self.genes),
                "values": [float(self.train_gene_means[gene]) for gene in self.genes],
            },
            "variable_genes": [
                gene for gene in self.genes if gene in self.variable_genes
            ],
            "esm2_symbols": list(self.esm2_symbols),
            "esm2_vectors": torch.from_numpy(
                np.asarray(self.esm2_vectors, dtype=np.float32)
            ).clone(),
        }


def _path(config: Mapping[str, Any], key: str) -> Path:
    paths = config.get("paths")
    if not isinstance(paths, Mapping) or key not in paths:
        raise ValueError(f"config.paths.{key} is required")
    return Path(paths[key])


def _features(config: Mapping[str, Any]) -> Mapping[str, Any]:
    value = config.get("features", {})
    if not isinstance(value, Mapping):
        raise ValueError("config.features must be a mapping")
    return value


def _unique_upper(values: Sequence[object], label: str) -> tuple[str, ...]:
    result = tuple(str(value).strip().upper() for value in values)
    if not result or any(not value for value in result):
        raise ValueError(f"{label} must contain non-empty genes")
    duplicates = sorted(value for value, count in Counter(result).items() if count > 1)
    if duplicates:
        raise ValueError(f"{label} contains duplicates: {duplicates[:10]}")
    return result


def _read_panel(path: Path) -> tuple[str, ...]:
    if not path.is_file():
        raise FileNotFoundError(
            f"missing prepared common gene panel {path}; run "
            "`hpc/run.sh prepare <config>`"
        )
    frame = pd.read_csv(path)
    if tuple(frame.columns) != ("gene_symbol",):
        raise ValueError(
            f"prepared common gene panel {path} must have one gene_symbol column"
            "; run `hpc/run.sh prepare <config>`"
        )
    return _unique_upper(frame["gene_symbol"].tolist(), "common gene panel")


def _read_metadata(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"missing prepared input metadata {path}; run `hpc/run.sh prepare <config>`"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"unable to read prepared input metadata {path}: {exc}"
            "; run `hpc/run.sh prepare <config>`"
        ) from exc
    if not isinstance(payload, Mapping):
        raise ValueError(
            f"prepared input metadata {path} must be a JSON object"
            "; run `hpc/run.sh prepare <config>`"
        )
    if payload.get("schema_version") != PREPARED_METADATA_SCHEMA:
        raise ValueError(
            f"prepared input metadata {path} must use {PREPARED_METADATA_SCHEMA!r}"
            "; run `hpc/run.sh prepare <config>`"
        )
    required = {
        "split",
        "common_gene_panel",
        "hvg_order",
        "esm2_order",
        "response_anchors",
        "response_conditions",
        "response_holdout",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(
            f"prepared input metadata {path} is missing fields {missing}"
            "; run `hpc/run.sh prepare <config>`"
        )
    return payload


def _metadata_pairs(values: object, label: str) -> tuple[tuple[str, str], ...]:
    if not isinstance(values, list):
        raise ValueError(f"prepared metadata {label} must be a list")
    pairs: list[tuple[str, str]] = []
    for index, value in enumerate(values):
        if not isinstance(value, Mapping) or set(value) != {"model_id", "gene"}:
            raise ValueError(
                f"prepared metadata {label}[{index}] must contain model_id and gene"
            )
        model_id = str(value["model_id"]).strip()
        gene = str(value["gene"]).strip().upper()
        if not model_id or not gene:
            raise ValueError(f"prepared metadata {label}[{index}] has an empty key")
        pairs.append((model_id, gene))
    if len(set(pairs)) != len(pairs):
        raise ValueError(f"prepared metadata {label} contains duplicate keys")
    return tuple(pairs)


def _validate_metadata_split(metadata: Mapping[str, Any], split: FixedSplit) -> None:
    expected = {
        "train": list(split.train),
        "val": list(split.val),
        "test": list(split.test),
        "unlabeled_train": list(split.unlabeled_train),
    }
    if metadata["split"] != expected:
        raise ValueError(
            "prepared metadata split membership/order does not match split file"
        )


def _restore_gene_means(
    preprocessing: Mapping[str, Any], genes: tuple[str, ...]
) -> pd.Series:
    state = preprocessing.get("gene_means")
    if not isinstance(state, Mapping):
        raise ValueError("preprocessing.gene_means must contain symbols and values")
    symbols = _unique_upper(state.get("symbols", []), "checkpoint gene means")
    values = np.asarray(state.get("values", []), dtype=np.float64)
    if (
        symbols != genes
        or values.shape != (len(genes),)
        or not np.isfinite(values).all()
    ):
        raise ValueError("checkpoint gene means do not match the prepared gene panel")
    return pd.Series(
        values, index=pd.Index(genes, name="gene_symbol"), name="gene_mean"
    )


def _restore_variable_genes(
    preprocessing: Mapping[str, Any], genes: tuple[str, ...]
) -> frozenset[str]:
    values = preprocessing.get("variable_genes")
    if not isinstance(values, (list, tuple)):
        raise ValueError("preprocessing.variable_genes must be an ordered list")
    restored = _unique_upper(values, "checkpoint variable genes")
    unknown = sorted(set(restored) - set(genes))
    if unknown:
        raise ValueError(
            f"checkpoint variable genes are outside the panel: {unknown[:10]}"
        )
    return frozenset(restored)


def _esm2_state(
    config: Mapping[str, Any],
    preprocessing: Mapping[str, Any] | None,
    expected_order: tuple[str, ...],
) -> tuple[tuple[str, ...], np.ndarray]:
    if preprocessing is not None:
        if "esm2_symbols" not in preprocessing or "esm2_vectors" not in preprocessing:
            raise ValueError(
                "checkpoint preprocessing requires esm2_symbols and esm2_vectors"
            )
        symbols = _unique_upper(
            preprocessing.get("esm2_symbols", []), "checkpoint ESM2 symbols"
        )
        raw_vectors = preprocessing.get("esm2_vectors")
        if isinstance(raw_vectors, torch.Tensor):
            vectors = raw_vectors.detach().cpu().numpy()
        else:
            vectors = np.asarray(raw_vectors)
        vectors = np.asarray(vectors, dtype=np.float32)
    else:
        path = _path(config, "esm2_embeddings")
        if not path.is_file():
            raise FileNotFoundError(
                f"missing prepared ESM2 table {path}; run `hpc/run.sh prepare <config>`"
            )
        table = load_esm2_embeddings(path)
        symbols = tuple(table.vectors_by_symbol)
        vectors = np.stack([table.vectors_by_symbol[symbol] for symbol in symbols])
    if symbols != expected_order:
        raise ValueError("ESM2 symbol order does not match prepared metadata")
    if vectors.ndim != 2 or vectors.shape[0] != len(symbols):
        raise ValueError("ESM2 vectors do not align with their ordered symbols")
    if vectors.dtype != np.dtype(np.float32) or not bool(np.isfinite(vectors).all()):
        raise ValueError("ESM2 vectors must be finite float32")
    configured_dim = _features(config).get("esm2_dim")
    if configured_dim is not None and int(configured_dim) != vectors.shape[1]:
        raise ValueError("ESM2 vector width does not match config.features.esm2_dim")
    return symbols, vectors


def _paired_indices(
    model_id: str, cell_ids: Sequence[object], count: int
) -> np.ndarray:
    identifiers = tuple(str(value) for value in cell_ids)
    ranked = sorted(
        range(len(identifiers)),
        key=lambda index: hashlib.sha256(
            f"{model_id}|{identifiers[index]}".encode()
        ).digest(),
    )
    selected = ranked[: min(count, len(ranked))]
    while len(selected) < count:
        selected.append(selected[len(selected) % len(ranked)])
    return np.asarray(selected, dtype=np.int64)


def _open_lines(
    model_ids: Sequence[str],
    *,
    tx1_cache: Path,
    q_sc_cache: Path,
    genes: tuple[str, ...],
    hvg_order: tuple[str, ...],
    cells_per_context: int,
) -> dict[str, PreparedLine]:
    result: dict[str, PreparedLine] = {}
    for model_id in model_ids:
        embeddings, hvg, obs = open_line_cache(
            tx1_cache, model_id, expected_hvg_order=hvg_order
        )
        indices = _paired_indices(model_id, obs.index, cells_per_context)
        controls = np.array(embeddings[indices], dtype=np.float32, copy=True)
        basal_hvg = np.array(hvg[indices], dtype=np.float32, copy=True)
        if not np.isfinite(controls).all() or not np.isfinite(basal_hvg).all():
            raise ValueError(f"{model_id}: selected paired basal arrays are non-finite")
        result[model_id] = PreparedLine(
            controls_tx1=controls,
            basal_hvg=basal_hvg,
            q_sc=load_q_sc_line(q_sc_cache, model_id, genes),
        )
    return result


def load_inputs(
    config: Mapping[str, Any],
    *,
    preprocessing: Mapping[str, Any] | None = None,
    include_test: bool = False,
) -> PreparedInputs:
    """Open prepared joint-training inputs without raw reads or cache writes."""
    split = load_geneeffect_226_split(_path(config, "split"))
    prepared_root_value = config.get("prepared_root")
    if prepared_root_value is None:
        raise ValueError("config.prepared_root is required")
    prepared_root = Path(prepared_root_value)
    metadata = _read_metadata(prepared_root / PREPARED_METADATA_FILENAME)
    _validate_metadata_split(metadata, split)

    genes = _read_panel(_path(config, "common_gene_panel"))
    if tuple(metadata["common_gene_panel"]) != genes:
        raise ValueError("prepared metadata common gene panel/order does not match CSV")
    hvg_order = _unique_upper(metadata["hvg_order"], "prepared HVG order")
    state_model_dir = _path(config, "state_model_dir")
    if not (state_model_dir / "var_dims.pkl").is_file():
        raise FileNotFoundError(
            f"missing STATE gene order {state_model_dir / 'var_dims.pkl'}; "
            "run `hpc/run.sh prepare <config>`"
        )
    state_hvg_order = _unique_upper(
        load_hvg_gene_order(state_model_dir), "STATE HVG order"
    )
    if hvg_order != state_hvg_order:
        raise ValueError("prepared HVG order does not match STATE model order")
    expected_hvg_dim = _features(config).get("hvg_dim", 2_000)
    if len(hvg_order) != int(expected_hvg_dim):
        raise ValueError(
            "prepared HVG order width does not match config.features.hvg_dim"
        )

    esm2_order = _unique_upper(metadata["esm2_order"], "prepared ESM2 order")
    esm2_symbols, esm2_vectors = _esm2_state(config, preprocessing, esm2_order)
    esm2_symbol_set = set(esm2_symbols)
    missing_esm2 = [gene for gene in genes if gene not in esm2_symbol_set]
    if missing_esm2:
        raise ValueError(
            f"common panel genes missing from ESM2 state: {missing_esm2[:10]}"
        )

    raw_labels = geneeffect_data.load_geneeffect_long(
        _path(config, "gene_effect"), split
    )
    raw_labels = raw_labels.loc[raw_labels["gene_symbol"].isin(genes)].copy()
    train_ids = split.supervised_train
    if preprocessing is None:
        gene_means = residual_target.fit_gene_means(raw_labels, train_ids)
        if tuple(gene_means.index) != tuple(sorted(genes)):
            raise ValueError("train-fit gene means do not cover the prepared panel")
        gene_means = gene_means.reindex(genes)
    else:
        gene_means = _restore_gene_means(preprocessing, genes)
    raw_labels["residual"] = raw_labels["gene_effect"] - raw_labels["gene_symbol"].map(
        gene_means
    )
    if preprocessing is None:
        features = _features(config)
        variable_genes = geneeffect_data.fit_variable_gene_membership(
            raw_labels,
            train_ids,
            genes,
            min_observations=int(features.get("variable_gene_min_observations", 5)),
            percentile=float(features.get("variable_gene_percentile", 75.0)),
        )
    else:
        variable_genes = _restore_variable_genes(preprocessing, genes)

    exposed_ids = set((*split.train, *split.val))
    if include_test:
        exposed_ids.update(split.test)
    labels = raw_labels.loc[
        raw_labels["model_id"].isin(exposed_ids)
        & np.isfinite(raw_labels["gene_effect"])
        & np.isfinite(raw_labels["residual"]),
        ["model_id", "gene_symbol", "gene_effect", "residual"],
    ].reset_index(drop=True)
    if not include_test and set(labels["model_id"]) & set(split.test):
        raise AssertionError("test labels entered default prepared inputs")

    anchors_raw = metadata["response_anchors"]
    if not isinstance(anchors_raw, list):
        raise ValueError("prepared response_anchors must be a list")
    response_anchors = tuple(str(value).strip() for value in anchors_raw)
    if len(response_anchors) != 4 or len(set(response_anchors)) != 4:
        raise ValueError("prepared response_anchors must contain four unique ModelIDs")
    if not set(response_anchors).issubset(split.supervised_train):
        raise ValueError("prepared response anchors must be labeled training lines")
    response_conditions = _metadata_pairs(
        metadata["response_conditions"], "response_conditions"
    )
    response_holdout_ordered = _metadata_pairs(
        metadata["response_holdout"], "response_holdout"
    )
    missing_response_esm2 = sorted(
        {gene for _, gene in response_conditions} - set(esm2_symbols)
    )
    if missing_response_esm2:
        raise ValueError(
            f"response genes missing from ESM2 state: {missing_response_esm2[:10]}"
        )
    response_holdout = frozenset(response_holdout_ordered)
    if not response_holdout or not response_holdout.issubset(response_conditions):
        raise ValueError(
            "prepared response holdout must be a non-empty condition subset"
        )

    tx1_cache = _path(config, "tx1_cache")
    q_sc_cache = _path(config, "q_sc_cache")
    response_cache = _path(config, "response_cache")
    needed_lines = tuple(
        model_id
        for model_id in split.all_model_ids
        if model_id in exposed_ids and model_id not in split.unlabeled_train
    )
    cells_per_context = int(_features(config).get("cells_per_context", 128))
    if cells_per_context <= 0:
        raise ValueError("config.features.cells_per_context must be positive")
    lines = _open_lines(
        needed_lines,
        tx1_cache=tx1_cache,
        q_sc_cache=q_sc_cache,
        genes=genes,
        hvg_order=hvg_order,
        cells_per_context=cells_per_context,
    )
    response_targets = open_response_targets_cache(
        response_cache, expected_hvg_order=hvg_order
    )
    if response_targets.keys != response_conditions:
        raise ValueError(
            "response cache condition order does not match prepared metadata"
        )
    response_key_set = set(response_targets.keys)
    if any(model_id not in response_anchors for model_id, _ in response_key_set):
        raise ValueError("response cache contains a condition outside the four anchors")
    for anchor in response_anchors:
        anchor_keys = {key for key in response_key_set if key[0] == anchor}
        if not anchor_keys - response_holdout or not anchor_keys & response_holdout:
            raise ValueError(
                f"response anchor {anchor} lacks train or holdout conditions"
            )

    return PreparedInputs(
        split=split,
        labels=labels,
        genes=genes,
        train_gene_means=gene_means,
        variable_genes=variable_genes,
        tx1_cache=tx1_cache,
        q_sc_cache=q_sc_cache,
        response_cache=response_cache,
        hvg_order=hvg_order,
        response_holdout=response_holdout,
        esm2_symbols=esm2_symbols,
        esm2_vectors=esm2_vectors,
        lines=lines,
        response_targets=response_targets,
        response_anchors=response_anchors,
    )


__all__ = [
    "PREPARED_METADATA_FILENAME",
    "PREPARED_METADATA_SCHEMA",
    "PreparedInputs",
    "PreparedLine",
    "load_inputs",
]
