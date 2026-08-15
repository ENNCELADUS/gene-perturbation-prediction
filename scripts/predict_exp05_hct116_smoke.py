"""Run label-blind observed-response GeneEffect predictions from an HCT116 cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from aivc_model.gene_splits import sha256_file
from aivc_model.model import MLPHead
from aivc_model.response import ResponseEncoder, TrainableDiagonalGMM

LOGGER = logging.getLogger(__name__)


def _padded_group_chunks(
    start: int,
    stop: int,
    *,
    chunk_size: int,
    rng: np.random.Generator,
) -> tuple[tuple[np.ndarray, int, int], ...]:
    """Shuffle a group, retaining all real cells and padding its final window."""
    indices = np.arange(start, stop, dtype=np.int64)
    if len(indices) == 0:
        raise ValueError("response group cannot be empty")
    rng.shuffle(indices)
    chunks: list[tuple[np.ndarray, int, int]] = []
    for offset in range(0, len(indices), chunk_size):
        real_indices = indices[offset : offset + chunk_size]
        padding_count = chunk_size - len(real_indices)
        if padding_count:
            padding = rng.choice(indices, size=padding_count, replace=True)
            chunk_indices = np.concatenate([real_indices, padding])
        else:
            chunk_indices = real_indices
        chunks.append((chunk_indices, len(real_indices), padding_count))
    return tuple(chunks)


def _aggregate_sample_gene_rows(
    sample_gene_rows: list[dict[str, object]],
    split_roles: dict[str, str],
) -> pd.DataFrame:
    """Average sample-gene predictions equally into one row per gene."""
    frame = pd.DataFrame(sample_gene_rows)
    rows: list[dict[str, object]] = []
    for gene, group in frame.groupby("perturbation_gene", sort=True):
        role = split_roles.get(str(gene), "not_in_k562_pool")
        rows.append(
            {
                "perturbation_gene": gene,
                "k562_split_role": role,
                "is_e_shared_train": role == "train",
                "is_e_double_shift": role == "not_in_k562_pool",
                "is_e_selection_exposed": role in {"validation", "internal_test"},
                "y_pred": float(group["y_pred"].mean()),
                "n_real_cells": int(group["n_real_cells"].sum()),
                "n_padding_draws": int(group["n_padding_draws"].sum()),
                "n_chunks": int(group["n_chunks"].sum()),
                "n_samples": int(len(group)),
            }
        )
    return pd.DataFrame(rows)


def _substate(state: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    return {
        key.removeprefix(prefix): value
        for key, value in state.items()
        if key.startswith(prefix)
    }


def _build_modules(
    checkpoint_path: Path,
    *,
    covariance_floor: float,
    device: torch.device,
) -> tuple[ResponseEncoder, TrainableDiagonalGMM, MLPHead]:
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    encoder_state = _substate(state, "response_encoder.")
    pooler_state = _substate(state, "response_pooler.")
    head_state = _substate(state, "c_head.")
    input_dim = int(encoder_state["linear.weight"].shape[1])
    latent_dim = int(encoder_state["linear.weight"].shape[0])
    n_components = int(pooler_state["means"].shape[0])
    linear_weights = [
        value for key, value in sorted(head_state.items()) if key.endswith("weight")
    ]
    hidden_units = tuple(int(weight.shape[0]) for weight in linear_weights[:-1])
    encoder = ResponseEncoder(input_dim=input_dim, latent_dim=latent_dim)
    pooler = TrainableDiagonalGMM(
        latent_dim=latent_dim,
        n_components=n_components,
        covariance_floor=covariance_floor,
        init_scale=0.0,
    )
    head = MLPHead(
        input_dim=pooler.output_dim,
        hidden_units=hidden_units,
        dropout=0.0,
    )
    encoder.load_state_dict(encoder_state, strict=True)
    pooler.load_state_dict(pooler_state, strict=True)
    head.load_state_dict(head_state, strict=True)
    return encoder.to(device).eval(), pooler.to(device).eval(), head.to(device).eval()


def predict_cache(
    cache_dir: Path,
    checkpoint_path: Path,
    output_dir: Path,
    *,
    covariance_floor: float = 1e-4,
    device_name: str = "auto",
    chunk_size: int = 64,
    control_panel_size: int = 4096,
    random_seed: int = 42,
    k562_split_manifest: Path | None = None,
    prediction_contract_path: Path | None = None,
) -> Path:
    """Predict one label-free GeneEffect value per sample/perturbation group."""
    manifest = json.loads((cache_dir / "manifest.json").read_text())
    qa = json.loads((cache_dir / "qa.json").read_text())
    if manifest.get("label_blind") is not True or qa.get("label_blind") is not True:
        raise ValueError("HCT116 cache must be label-blind")
    if chunk_size < 1 or control_panel_size < 1:
        raise ValueError("chunk_size and control_panel_size must be positive")
    prediction_contract: dict[str, object] | None = None
    if prediction_contract_path is not None:
        prediction_contract = json.loads(prediction_contract_path.read_text())
        expected_contract = {
            "schema_version": 2,
            "chunk_size": chunk_size,
            "pad_short": True,
            "sample_gene_aggregation": "mean_chunks",
            "gene_aggregation": "equal_mean_sample_gene",
            "control_panel_size": control_panel_size,
            "random_seed": random_seed,
            "covariance_floor": covariance_floor,
            "checkpoint_sha256": manifest["sources"]["frozen_checkpoint"],
            "preprocessing_contract_sha256": manifest["sources"]["transform_contract"],
            "k562_split_manifest_sha256": (
                sha256_file(k562_split_manifest)
                if k562_split_manifest is not None
                else None
            ),
        }
        if prediction_contract != expected_contract:
            raise ValueError("runtime inputs do not match the prediction contract")
        if manifest["parameters"]["max_cells_per_group"] is not None:
            raise ValueError(
                "registered prediction requires an uncapped response cache"
            )
    expected_checkpoint = manifest["sources"]["frozen_checkpoint"]
    if sha256_file(checkpoint_path) != expected_checkpoint:
        raise ValueError("checkpoint hash does not match cache authority")
    required_arrays = (
        "response_cells.npy",
        "control_cells.npy",
        "response_group_samples.npy",
        "response_group_genes.npy",
        "response_group_offsets.npy",
        "response_group_counts.npy",
        "control_group_samples.npy",
        "control_group_offsets.npy",
    )
    for filename in required_arrays:
        metadata = manifest["files"][filename]
        path = cache_dir / filename
        if sha256_file(path) != metadata["sha256"]:
            raise ValueError(f"cache array hash mismatch: {filename}")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if (
            list(array.shape) != metadata["shape"]
            or array.dtype.str != metadata["dtype"]
        ):
            raise ValueError(f"cache array metadata mismatch: {filename}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved_device = (
        ("cuda" if torch.cuda.is_available() else "cpu")
        if device_name == "auto"
        else device_name
    )
    device = torch.device(resolved_device)
    encoder, pooler, head = _build_modules(
        checkpoint_path,
        covariance_floor=covariance_floor,
        device=device,
    )
    responses = np.load(cache_dir / "response_cells.npy", mmap_mode="r")
    controls = np.load(cache_dir / "control_cells.npy", mmap_mode="r")
    group_samples = np.load(cache_dir / "response_group_samples.npy").astype(str)
    group_genes = np.load(cache_dir / "response_group_genes.npy").astype(str)
    group_offsets = np.load(cache_dir / "response_group_offsets.npy")
    group_counts = np.load(cache_dir / "response_group_counts.npy")
    control_samples = np.load(cache_dir / "control_group_samples.npy").astype(str)
    control_offsets = np.load(cache_dir / "control_group_offsets.npy")
    control_ranges = {
        sample: (int(control_offsets[index]), int(control_offsets[index + 1]))
        for index, sample in enumerate(control_samples)
    }
    split_roles: dict[str, str] = {}
    split_sha256: str | None = None
    if k562_split_manifest is not None:
        split = pd.read_csv(k562_split_manifest)
        if list(split.columns) != ["perturbation_gene", "split_role"]:
            raise ValueError("K562 split manifest schema is invalid")
        if split["perturbation_gene"].duplicated().any():
            raise ValueError("K562 split manifest genes must be unique")
        allowed_roles = {"train", "validation", "internal_test"}
        if not set(split["split_role"].astype(str)).issubset(allowed_roles):
            raise ValueError("K562 split manifest contains an invalid role")
        split_roles = dict(
            zip(
                split["perturbation_gene"].astype(str),
                split["split_role"].astype(str),
                strict=True,
            )
        )
        split_sha256 = sha256_file(k562_split_manifest)
    elif prediction_contract is not None:
        raise ValueError("registered prediction requires the K562 split manifest")
    control_latents: dict[str, torch.Tensor] = {}
    sample_gene_rows: list[dict[str, object]] = []
    with torch.inference_mode():
        for index, (sample, gene) in enumerate(
            zip(group_samples, group_genes, strict=True)
        ):
            if sample not in control_latents:
                start, stop = control_ranges[sample]
                panel_indices = np.arange(start, stop, dtype=np.int64)
                if len(panel_indices) > control_panel_size:
                    digest = hashlib.sha256(sample.encode()).digest()
                    sample_seed = random_seed + int.from_bytes(digest[:4], "little")
                    rng = np.random.default_rng(sample_seed)
                    panel_indices = np.sort(
                        rng.choice(
                            panel_indices,
                            size=control_panel_size,
                            replace=False,
                        )
                    )
                values = torch.as_tensor(
                    np.asarray(controls[panel_indices]).copy(),
                    dtype=torch.float32,
                    device=device,
                )
                control_latents[sample] = encoder(values)
            start, stop = int(group_offsets[index]), int(group_offsets[index + 1])
            count = int(group_counts[index])
            if stop - start != count:
                raise ValueError("response group offset/count mismatch")
            digest = hashlib.sha256(f"{sample}\0{gene}".encode()).digest()
            group_seed = random_seed + int.from_bytes(digest[:4], "little")
            chunks = _padded_group_chunks(
                start,
                stop,
                chunk_size=chunk_size,
                rng=np.random.default_rng(group_seed),
            )
            group_predictions: list[float] = []
            padding_draws = 0
            for chunk_indices, _, padding_count in chunks:
                values = torch.as_tensor(
                    np.asarray(responses[chunk_indices]).copy(),
                    dtype=torch.float32,
                    device=device,
                )
                prediction = head(pooler(encoder(values), control_latents[sample]))
                group_predictions.append(float(prediction.cpu()))
                padding_draws += padding_count
            sample_gene_rows.append(
                {
                    "sample": sample,
                    "perturbation_gene": gene,
                    "y_pred": float(np.mean(group_predictions)),
                    "n_real_cells": count,
                    "n_padding_draws": padding_draws,
                    "n_chunks": len(chunks),
                }
            )
    predictions = _aggregate_sample_gene_rows(sample_gene_rows, split_roles)
    if predictions.empty or not np.isfinite(predictions["y_pred"]).all():
        raise RuntimeError("prediction smoke produced empty or nonfinite output")
    sample_gene_predictions_path = (
        output_dir / "sample_gene_predictions_label_blind.csv"
    )
    pd.DataFrame(sample_gene_rows).to_csv(sample_gene_predictions_path, index=False)
    predictions_path = output_dir / "predictions_label_blind.csv"
    predictions.to_csv(predictions_path, index=False)
    output_manifest = {
        "schema_version": 2,
        "label_blind": True,
        "ground_truth_accessed": False,
        "route": "observed_response_encoder_gmm_c_head",
        "state_batch_covariate": "not_used",
        "cache_manifest_sha256": sha256_file(cache_dir / "manifest.json"),
        "checkpoint_sha256": expected_checkpoint,
        "covariance_floor": covariance_floor,
        "device": str(device),
        "prediction_rows": int(len(predictions)),
        "sample_gene_prediction_rows": int(len(sample_gene_rows)),
        "chunk_size": chunk_size,
        "pad_short": True,
        "sample_gene_aggregation": "mean_chunks",
        "gene_aggregation": "equal_mean_sample_gene",
        "control_panel_size": control_panel_size,
        "random_seed": random_seed,
        "k562_split_manifest_sha256": split_sha256,
        "prediction_contract_sha256": (
            sha256_file(prediction_contract_path)
            if prediction_contract_path is not None
            else None
        ),
        "sample_gene_predictions_sha256": sha256_file(sample_gene_predictions_path),
        "predictions_sha256": sha256_file(predictions_path),
    }
    manifest_path = output_dir / "prediction_manifest.json"
    manifest_path.write_text(
        json.dumps(output_manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--covariance-floor", type=float, default=1e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--control-panel-size", type=int, default=4096)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--k562-split-manifest", type=Path)
    parser.add_argument("--prediction-contract", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    manifest = predict_cache(
        args.cache_dir,
        args.checkpoint,
        args.output_dir,
        covariance_floor=args.covariance_floor,
        device_name=args.device,
        chunk_size=args.chunk_size,
        control_panel_size=args.control_panel_size,
        random_seed=args.random_seed,
        k562_split_manifest=args.k562_split_manifest,
        prediction_contract_path=args.prediction_contract,
    )
    LOGGER.info("label-blind predictions ready: %s", manifest)


if __name__ == "__main__":
    main()
