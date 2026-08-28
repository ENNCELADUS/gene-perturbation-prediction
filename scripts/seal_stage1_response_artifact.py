#!/usr/bin/env python3
"""Seal Stage-1 compatibility/inputs with incomplete historical lineage."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Mapping
import uuid

import numpy as np
import torch

from aivc_model.gene_embeddings import load_esm2_embeddings
from aivc_model.stage1_artifact import seal_stage1_artifact


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_MANIFEST_NAME = "stage1_model_manifest.json"
_BUNDLE_NAME = "stage2_bundle.json"
_BUNDLE_SCHEMA = "exp13-stage2-bundle-v1"
_LEGACY_ESM_KEY = "perturbations.esm_matrix"

_CODE_RELATIVE_PATHS = {
    "train_geneeffect_response_model.py": Path(
        "scripts/train_geneeffect_response_model.py"
    ),
    "gene_embeddings.py": Path("src/aivc_model/gene_embeddings.py"),
    "response_training.py": Path("src/aivc_model/response_training.py"),
    "stage1_config.py": Path("src/aivc_model/stage1_config.py"),
    "state_core.py": Path("src/aivc_model/state_core.py"),
    "tx1_predicted_response.py": Path("src/aivc_model/tx1_predicted_response.py"),
}


def _require_file(path: Path, label: str) -> Path:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing or is not a file: {path}")
    return path


def _load_flat_tensor_state(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in payload.items()
    ):
        raise ValueError("Stage-1 checkpoint must be a flat tensor state dict")
    return payload


def _vector_digest(vector: torch.Tensor) -> str:
    array = vector.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def reconstruct_stage1_gene_vocabulary(
    checkpoint_path: Path, esm2_embeddings_path: Path
) -> tuple[str, ...]:
    """Recover legacy checkpoint row identities by exact vector matching."""
    state = _load_flat_tensor_state(checkpoint_path)
    legacy = state.get(_LEGACY_ESM_KEY)
    if legacy is None:
        raise ValueError(
            "Historical checkpoint has no perturbations.esm_matrix to authenticate"
        )
    if legacy.ndim != 2 or legacy.shape[0] == 0:
        raise ValueError("Legacy perturbations.esm_matrix must be a non-empty matrix")

    table = load_esm2_embeddings(esm2_embeddings_path)
    actual = legacy.detach().to(device="cpu", dtype=torch.float32)
    if actual.shape[1] != table.dim:
        raise ValueError(
            "Legacy perturbations.esm_matrix width does not match Stage-1 ESM2: "
            f"{actual.shape[1]} != {table.dim}"
        )

    candidates_by_digest: dict[str, list[tuple[str, torch.Tensor]]] = {}
    for symbol, vector in table.vectors_by_symbol.items():
        tensor = torch.as_tensor(vector, dtype=torch.float32)
        candidates_by_digest.setdefault(_vector_digest(tensor), []).append(
            (symbol, tensor)
        )
    genes: list[str] = []
    for row_index, row in enumerate(actual):
        matches = [
            symbol
            for symbol, vector in candidates_by_digest.get(_vector_digest(row), ())
            if vector.shape == row.shape and torch.equal(vector, row)
        ]
        if len(matches) != 1:
            raise ValueError(
                "Legacy ESM row must match exactly one resolved Stage-1 ESM vector: "
                f"row={row_index}, matches={matches[:10]}, count={len(matches)}"
            )
        genes.append(matches[0])
    if len(set(genes)) != len(genes):
        raise ValueError("Legacy ESM matrix resolves to duplicate gene identities")
    return tuple(genes)


def _provenance_maps(
    *,
    repository_root: Path,
    stage1_config: Path,
    split_json: Path,
    perturbseq_sources: Path,
) -> tuple[dict[str, Path], dict[str, Path], dict[str, Path]]:
    compatibility_code_paths = {
        name: _require_file(
            repository_root / relative, f"seal/load compatibility code:{name}"
        )
        for name, relative in _CODE_RELATIVE_PATHS.items()
    }
    config_paths = {"stage1_response": _require_file(stage1_config, "Stage-1 config")}
    source_paths = {
        "split_json": _require_file(split_json, "Exp13 split"),
        "perturbseq_sources": _require_file(
            perturbseq_sources, "Perturb-seq source registry"
        ),
    }
    return compatibility_code_paths, config_paths, source_paths


def _bundle_payload(
    compatibility_code_paths: Mapping[str, Path],
    config_paths: Mapping[str, Path],
    source_paths: Mapping[str, Path],
) -> dict[str, object]:
    return {
        "schema_version": _BUNDLE_SCHEMA,
        "compatibility_code_paths": {
            name: str(path)
            for name, path in sorted(compatibility_code_paths.items())
        },
        "config_paths": {
            name: str(path) for name, path in sorted(config_paths.items())
        },
        "source_paths": {
            name: str(path) for name, path in sorted(source_paths.items())
        },
    }


def _temporary_path(directory: Path, label: str) -> Path:
    return directory / f".{label}.{uuid.uuid4().hex}.tmp"


def _publish_fresh(source: Path, destination: Path) -> None:
    """Atomically publish a completed file without replacing an existing one."""
    os.link(source, destination)
    source.unlink()


def _write_bundle_temp(path: Path, payload: Mapping[str, object]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _seal(
    *,
    run_dir: Path,
    stage1_config: Path,
    esm2_embeddings: Path,
    state_hparams: Path,
    split_json: Path,
    perturbseq_sources: Path,
    repository_root: Path,
    dry_run: bool,
) -> dict[str, object]:
    run_dir = run_dir.resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Stage-1 run directory is missing: {run_dir}")
    checkpoint = _require_file(
        run_dir / "best" / "pytorch_model.bin", "selected Stage-1 checkpoint"
    )
    metadata = _require_file(
        run_dir / "best" / "metadata.json", "selected checkpoint metadata"
    )
    run_manifest = _require_file(run_dir / "run_manifest.json", "run manifest")
    objective = _require_file(run_dir / "stage1_objective.json", "Stage-1 objective")
    esm2_embeddings = _require_file(esm2_embeddings, "Stage-1 ESM2 table")
    state_hparams = _require_file(state_hparams, "STATE hparams checkpoint")
    compatibility_code_paths, config_paths, source_paths = _provenance_maps(
        repository_root=repository_root.resolve(),
        stage1_config=stage1_config,
        split_json=split_json,
        perturbseq_sources=perturbseq_sources,
    )
    manifest_path = run_dir / _MANIFEST_NAME
    bundle_path = run_dir / _BUNDLE_NAME
    for path in (manifest_path, bundle_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing seal output: {path}")

    genes = reconstruct_stage1_gene_vocabulary(checkpoint, esm2_embeddings)
    bundle = _bundle_payload(
        compatibility_code_paths, config_paths, source_paths
    )
    if dry_run:
        with tempfile.TemporaryDirectory(prefix="exp13-stage1-seal-") as directory:
            temporary_manifest = Path(directory) / _MANIFEST_NAME
            manifest = seal_stage1_artifact(
                checkpoint_path=checkpoint,
                manifest_path=temporary_manifest,
                stage1_genes=genes,
                esm2_embeddings_path=esm2_embeddings,
                state_hparams_path=state_hparams,
                run_manifest_path=run_manifest,
                checkpoint_metadata_path=metadata,
                stage1_objective_path=objective,
                compatibility_code_paths=compatibility_code_paths,
                config_paths=config_paths,
                source_paths=source_paths,
            )
    else:
        temporary_manifest = _temporary_path(run_dir, _MANIFEST_NAME)
        temporary_bundle = _temporary_path(run_dir, _BUNDLE_NAME)
        bundle_published = False
        try:
            manifest = seal_stage1_artifact(
                checkpoint_path=checkpoint,
                manifest_path=temporary_manifest,
                stage1_genes=genes,
                esm2_embeddings_path=esm2_embeddings,
                state_hparams_path=state_hparams,
                run_manifest_path=run_manifest,
                checkpoint_metadata_path=metadata,
                stage1_objective_path=objective,
                compatibility_code_paths=compatibility_code_paths,
                config_paths=config_paths,
                source_paths=source_paths,
            )
            _write_bundle_temp(temporary_bundle, bundle)
            _publish_fresh(temporary_bundle, bundle_path)
            bundle_published = True
            _publish_fresh(temporary_manifest, manifest_path)
        except BaseException:
            temporary_manifest.unlink(missing_ok=True)
            temporary_bundle.unlink(missing_ok=True)
            if bundle_published and not manifest_path.exists():
                bundle_path.unlink(missing_ok=True)
            raise

    return {
        "status": (
            "compatibility_inputs_validated"
            if dry_run
            else "compatibility_inputs_sealed"
        ),
        "dry_run": dry_run,
        "training_data_provenance_status": (
            manifest.training_data_provenance_status
        ),
        "training_data_provenance_missing_identities": list(
            manifest.training_data_provenance_missing_identities
        ),
        "stage1_gene_count": len(genes),
        "stage1_genes": list(genes),
        "manifest_path": str(manifest_path),
        "bundle_path": str(bundle_path),
        "writes_planned": [_MANIFEST_NAME, _BUNDLE_NAME] if dry_run else [],
        "manifest": asdict(manifest),
        "bundle": bundle,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--stage1-config",
        type=Path,
        default=_REPOSITORY_ROOT
        / "configs/experiments/13_geneeffect_226/stage1_response.yaml",
    )
    parser.add_argument("--esm2-embeddings", type=Path, required=True)
    parser.add_argument("--state-hparams", type=Path, required=True)
    parser.add_argument(
        "--split-json",
        type=Path,
        default=_REPOSITORY_ROOT
        / "configs/benchmarks/cell_line_geneeffect_226_split.json",
    )
    parser.add_argument(
        "--perturbseq-sources",
        type=Path,
        default=_REPOSITORY_ROOT
        / "configs/experiments/13_geneeffect_226/perturbseq_sources.json",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = _seal(
        run_dir=args.run_dir,
        stage1_config=args.stage1_config,
        esm2_embeddings=args.esm2_embeddings,
        state_hparams=args.state_hparams,
        split_json=args.split_json,
        perturbseq_sources=args.perturbseq_sources,
        repository_root=_REPOSITORY_ROOT,
        dry_run=args.dry_run,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
