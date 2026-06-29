"""Queue helpers for exp08b two-pass runs."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

from sl_dl_model import fold_queue as fq
from sl_dl_model.exp08b_artifacts import (
    embedding_cache_path,
    generator_manifest_path,
    load_generator_manifest,
    step2_output_dir,
)
from sl_dl_model.exp08b_config import Exp08bConfig, metric_model_name_for

_STEP2_FINGERPRINT_ALLOWLIST = frozenset(
    {
        "split_types",
        "folds",
        "ranking_k",
        "seed",
        "fallback_strategy",
        "include_coverage_flag",
        "pair_hidden",
        "lr",
        "max_epochs",
        "batch_pairs",
        "max_grad_norm",
    }
)


def step_results_dir(config: Exp08bConfig, step: str) -> Path:
    """Return the queue directory for one exp08b pass."""
    return Path(config.output_dir) / config.fold_results_subdir / step


def step_result_path(
    config: Exp08bConfig, step: str, split_type: str, fold_id: int
) -> Path:
    """Return one pass-scoped success marker path."""
    return fq.result_path(step_results_dir(config, step), split_type, fold_id)


def step_failed_path(
    config: Exp08bConfig, step: str, split_type: str, fold_id: int
) -> Path:
    """Return one pass-scoped failure marker path."""
    return fq.failed_path(step_results_dir(config, step), split_type, fold_id)


def step2_metric_config(config: Exp08bConfig) -> Exp08bConfig:
    """Project config fields that affect cached-embedding SL-head metrics."""
    updates: dict[str, object] = {}
    for name in fq._FINGERPRINT_FIELDS:
        if name not in _STEP2_FINGERPRINT_ALLOWLIST and hasattr(config, name):
            updates[name] = None
    for name in fq._FINGERPRINT_PATH_FIELDS:
        if name not in _STEP2_FINGERPRINT_ALLOWLIST and hasattr(config, name):
            updates[name] = None
    updates["state_backend"] = "linear_mock"
    updates["output_dir"] = step2_output_dir(config)
    return replace(config, **updates)


def step2_fold_fingerprint(
    config: Exp08bConfig, split_type: str, fold_id: int
) -> str:
    """Hash Step 2 metric config plus the Step 1 files for one fold."""
    h = hashlib.sha256()
    h.update(fq.fingerprint(step2_metric_config(config)).encode())
    h.update(
        fq._path_signature(embedding_cache_path(config, split_type, fold_id)).encode()
    )
    manifest_sig = fq._path_signature(
        generator_manifest_path(config, split_type, fold_id)
    )
    h.update(manifest_sig.encode())
    return h.hexdigest()[:16]


def step2_metric_model_name(
    config: Exp08bConfig, split_type: str, fold_id: int
) -> str:
    """Read the Step 1 manifest and return the official metric model label."""
    manifest = load_generator_manifest(
        generator_manifest_path(config, split_type, fold_id)
    )
    return metric_model_name_for(str(manifest["generator_kind"]))


def read_step2_result_cache_fp(
    results_dir: Path, split_type: str, fold_id: int
) -> str | None:
    """Return a Step 2 result's cached-input fingerprint, if present."""
    return _read_cache_fp(fq.result_path(results_dir, split_type, fold_id))


def read_step2_failed_cache_fp(
    results_dir: Path, split_type: str, fold_id: int
) -> str | None:
    """Return a Step 2 failure marker's cached-input fingerprint, if present."""
    return _read_cache_fp(fq.failed_path(results_dir, split_type, fold_id))


def step2_result_matches_cache(
    results_dir: Path,
    split_type: str,
    fold_id: int,
    *,
    fingerprint: str,
    cache_fp: str,
) -> bool:
    """Return whether a Step 2 result marker matches this run and cache."""
    return (
        _read_marker_cache_fp(
            fq.result_path(results_dir, split_type, fold_id), fingerprint
        )
        == cache_fp
    )


def step2_failed_matches_cache(
    results_dir: Path,
    split_type: str,
    fold_id: int,
    *,
    fingerprint: str,
    cache_fp: str,
) -> bool:
    """Return whether a Step 2 failure marker matches this run and cache."""
    return (
        _read_marker_cache_fp(
            fq.failed_path(results_dir, split_type, fold_id), fingerprint
        )
        == cache_fp
    )


def _read_cache_fp(path: Path) -> str | None:
    """Read ``cache_fp`` from a marker, treating malformed markers as stale."""
    payload = _read_marker(path)
    if payload is None:
        return None
    value = payload.get("cache_fp")
    return str(value) if value is not None else None


def _read_marker_cache_fp(path: Path, fingerprint: str) -> str | None:
    """Read a marker cache fingerprint only when the run fingerprint matches."""
    payload = _read_marker(path)
    if payload is None or payload.get("fingerprint") != fingerprint:
        return None
    value = payload.get("cache_fp")
    return str(value) if value is not None else None


def _read_marker(path: Path) -> dict | None:
    """Read a marker payload, treating malformed markers as stale."""
    if not path.exists():
        return None
    try:
        payload = fq.read_json(path)
    except (OSError, UnicodeDecodeError, ValueError):
        return None
    if isinstance(payload, dict):
        return payload
    return None
