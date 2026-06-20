"""Filesystem coordination primitives for the exp08 fold work-queue.

Replaces the single end-of-run NCCL all-gather collective. Every rank
walks the same job list and uses these primitives to claim, run, and record
``(split_type, fold_id)`` jobs through the filesystem only — no
``torch.distributed`` collective is involved (Guard G1).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from sl_dl_model.config import SLDLConfig

# Scalar config fields that change a fold's computed metrics. output_dir, the
# queue knobs, and path fields (handled separately, below) are excluded so they
# don't bust the fingerprint for the wrong reason.
_FINGERPRINT_FIELDS = (
    "split_types",
    "folds",
    "ranking_k",
    "seed",
    "fallback_strategy",
    "include_coverage_flag",
    "esm2_model",
    "state_backend",
    "pooling",
    "pair_hidden",
    "adapter_hidden",
    "pert_dim",
    "control_template_size",
    "cells_per_bag",
    "lambda_sl",
    "lambda_distill",
    "lambda_distill_after_warmup",
    "lambda_bag",
    "lambda_rank",
    "warmup_epochs",
    "max_epochs",
    "batch_pairs",
    "lr",
    "early_stop_patience",
    "max_grad_norm",
    "embedding_method",
)

# Path fields whose *contents* change a fold's metrics. Folded into the
# fingerprint as a (path, size, mtime_ns) stat signature so a cache rebuilt at
# the same path (new size/mtime) busts stale reuse without reading multi-GB
# files. ``None``-valued fields contribute a sentinel so toggling them matters.
# ``input_csv`` is handled separately (content-hashed) because it is small and
# an in-place edit keeping size+mtime should still bust reuse.
_FINGERPRINT_PATH_FIELDS = (
    "esm2_npz",
    "bags_npz",
    "gwps_h5ad",
    "gwps_overlap_csv",
    "state_checkpoint",
)


# STATE sidecar files derived from ``state_checkpoint.parent.parent``. They are
# not config fields but their contents change a fold's metrics: ``var_dims.pkl``
# is read by bags.py and ``pert_onehot_map.pt`` by train.py. They can change
# while ``state_checkpoint`` itself does not (e.g. a cache rebuild touches the
# sidecar but not the .ckpt), so they fold in as their own stat signatures.
_STATE_SIDECAR_NAMES = ("var_dims.pkl", "pert_onehot_map.pt")


def _path_signature(value: object) -> str:
    """Return a ``(path, size, mtime_ns)`` signature for a path-valued field.

    Uses ``os.stat`` (no read) so multi-GB caches cost O(1). A missing or
    ``None`` path returns a sentinel that still distinguishes set-vs-unset.
    """
    if value is None:
        return "<none>"
    path = Path(value)
    try:
        st = path.stat()
    except OSError:
        return f"{path}:<absent>"
    return f"{path}:{st.st_size}:{st.st_mtime_ns}"


def fingerprint(config: SLDLConfig) -> str:
    """Return a short hash of result-affecting config fields + cache signatures.

    Reused result/failed files are trusted only when their stored fingerprint
    matches the current run's. This prevents mixing incompatible fold results
    when the same ``output_dir`` is reused after the input CSV, config, model
    parameters, or a cache file (ESM2/bags/gwps) changes — including a cache
    regenerated at the same path (detected via size + mtime).

    Args:
        config: The run configuration.

    Returns:
        A 16-hex-char fingerprint string.
    """
    h = hashlib.sha256()
    # input_csv: content-hashed (small) so even an in-place same-size edit busts.
    input_path = Path(config.input_csv)
    if input_path.exists():
        h.update(b"input_csv=")
        h.update(input_path.read_bytes())
    else:
        h.update(f"input_csv=<absent:{input_path}>".encode())
    for name in _FINGERPRINT_FIELDS:
        h.update(f"{name}={getattr(config, name, None)!r}".encode())
    for name in _FINGERPRINT_PATH_FIELDS:
        h.update(f"{name}={_path_signature(getattr(config, name, None))}".encode())
    # STATE sidecars (var_dims.pkl, pert_onehot_map.pt) live next to the
    # checkpoint and affect results; skip for the linear_mock backend, which
    # ignores the checkpoint entirely.
    if getattr(config, "state_backend", None) != "linear_mock":
        ckpt = getattr(config, "state_checkpoint", None)
        sidecar_root = Path(ckpt).parent.parent if ckpt is not None else None
        for sidecar in _STATE_SIDECAR_NAMES:
            value = sidecar_root / sidecar if sidecar_root is not None else None
            h.update(f"{sidecar}={_path_signature(value)}".encode())
    return h.hexdigest()[:16]


def run_token() -> str:
    """Return a per-run token used to scope intra-run claim markers.

    Claims are *per-run* (design decision C): a hard-crashed worker leaves an
    orphan claim dir, and the next run must ignore it. Keying the ``.claims``
    subdir by this token guarantees a fresh run never collides with a prior
    run's claims, so the only cross-run state is the ``.result.json`` files.

    Resolution order:

    1. ``SLURM_JOB_ID`` — distinct per cluster job (the production case).
    2. ``SL_DL_RUN_ID`` — explicit override (set by a launcher to share one
       token across ranks of the same local/CLI run).
    3. ``local-<parent-pid>`` — fallback for a bare local run; the parent pid
       is stable across the ranks an ``accelerate launch`` spawns.

    Returns:
        A non-empty token string.
    """
    slurm = os.environ.get("SLURM_JOB_ID")
    if slurm:
        return slurm
    explicit = os.environ.get("SL_DL_RUN_ID")
    if explicit:
        return explicit
    return f"local-{os.getppid()}"


def fold_results_dir(config: SLDLConfig) -> Path:
    """Return the per-run fold-results directory under ``output_dir``."""
    return Path(config.output_dir) / config.fold_results_subdir


def result_path(results_dir: Path, split: str, fold: int) -> Path:
    """Return the success-result JSON path for one job."""
    return results_dir / f"{split}_fold{fold}.result.json"


def failed_path(results_dir: Path, split: str, fold: int) -> Path:
    """Return the quarantine-marker path for one job."""
    return results_dir / f"{split}_fold{fold}.failed"


def claim_path(
    results_dir: Path,
    split: str,
    fold: int,
    run_token: str | None = None,
    *,
    _default_token=run_token,
) -> Path:
    """Return the atomic-claim directory path for one job, scoped by run token.

    Args:
        results_dir: The fold-results directory.
        split: CV split type.
        fold: Fold id.
        run_token: Per-run token; defaults to :func:`run_token`. Claims live
            under ``.claims/<run_token>/`` so a prior run's orphan claims never
            block a resume.
    """
    token = run_token if run_token is not None else _default_token()
    return results_dir / ".claims" / token / f"{split}_fold{fold}"


def atomic_write_json(path: Path, obj: object) -> None:
    """Write ``obj`` as JSON atomically (temp file in the same dir + replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, path)


def read_json(path: Path) -> object:
    """Read and parse a JSON file written by :func:`atomic_write_json`."""
    return json.loads(Path(path).read_text())


def try_claim(
    results_dir: Path, split: str, fold: int, run_token: str | None = None
) -> bool:
    """Atomically claim one job. Return ``True`` if this caller won the claim.

    Uses ``os.mkdir`` (POSIX/Lustre-atomic). A returned ``False`` means another
    rank in the *same run* already owns the job. Claims are scoped by
    ``run_token`` (see :func:`run_token`), so a prior run's orphan claim never
    blocks a resume.
    """
    claim = claim_path(results_dir, split, fold, run_token)
    claim.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.mkdir(claim)
        return True
    except FileExistsError:
        return False


def is_done(
    results_dir: Path, split: str, fold: int, fingerprint: str | None = None
) -> bool:
    """Return ``True`` if this job has a success-result file for this run.

    When ``fingerprint`` is given, a result file whose stored fingerprint does
    not match is treated as *not* done (stale results from a different config /
    input are recomputed rather than reused).

    Args:
        results_dir: The fold-results directory.
        split: CV split type.
        fold: Fold id.
        fingerprint: Current run fingerprint (see :func:`fingerprint`); when
            ``None``, only file existence is checked (back-compat).
    """
    path = result_path(results_dir, split, fold)
    if not path.exists():
        return False
    if fingerprint is None:
        return True
    return _stored_fingerprint(path) == fingerprint


def is_failed(
    results_dir: Path, split: str, fold: int, fingerprint: str | None = None
) -> bool:
    """Return ``True`` if this job has a quarantine marker for this run.

    A ``.failed`` marker from a different fingerprint is treated as *not*
    failed, so a config/input change re-runs the fold instead of inheriting an
    old failure.
    """
    path = failed_path(results_dir, split, fold)
    if not path.exists():
        return False
    if fingerprint is None:
        return True
    return _stored_fingerprint(path) == fingerprint


def write_result(
    results_dir: Path,
    split: str,
    fold: int,
    rows: object,
    fingerprint: str,
) -> None:
    """Atomically write a fold's success result with its run fingerprint."""
    atomic_write_json(
        result_path(results_dir, split, fold),
        {"fingerprint": fingerprint, "rows": rows},
    )


def read_result_rows(
    results_dir: Path, split: str, fold: int, fingerprint: str
) -> list | None:
    """Return the stored rows iff the result exists and the fingerprint matches.

    Args:
        results_dir: The fold-results directory.
        split: CV split type.
        fold: Fold id.
        fingerprint: Current run fingerprint; a mismatch (or missing file)
            returns ``None`` so the caller recomputes.

    Returns:
        The stored row list, or ``None`` on missing file / fingerprint mismatch.
    """
    path = result_path(results_dir, split, fold)
    if not path.exists():
        return None
    payload = read_json(path)
    if not isinstance(payload, dict) or payload.get("fingerprint") != fingerprint:
        return None
    return payload.get("rows")


def write_failed(
    results_dir: Path,
    split: str,
    fold: int,
    marker: dict,
    fingerprint: str,
) -> None:
    """Atomically write a fold's quarantine marker with its run fingerprint."""
    atomic_write_json(
        failed_path(results_dir, split, fold),
        {"fingerprint": fingerprint, **marker},
    )


def _stored_fingerprint(path: Path) -> str | None:
    """Return the ``fingerprint`` field of a result/failed file, or ``None``."""
    payload = read_json(path)
    if isinstance(payload, dict):
        return payload.get("fingerprint")
    return None
