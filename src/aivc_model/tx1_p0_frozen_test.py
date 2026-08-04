"""Prediction-first, post-hoc P0 evaluation on the opened test cohort."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from aivc_model.tx1_embed_cache import verify_cache
from aivc_model.tx1_geneeffect_eval import line_bootstrap_ci
from aivc_model.tx1_p0_inputs import _load_authorities, _load_gene_effect
from aivc_model.tx1_p0_representation import fit_outer_fold, load_representation

EXPECTED_TEST_LINES = 9
PREDICTION_PROTOCOL = "tx1_geneeffect_p0_frozen_test_v1"
FROZEN_PCA_COMPONENTS = 8
FROZEN_RIDGE_ALPHA = 1.0
FROZEN_SHUFFLE_SEED = 20260804
EXPECTED_CACHE_MANIFEST_SHA256 = (
    "ac06f60bbc0b9045571d00f10a80f50a7bdd86c54dcc6fd09e6571a343981927"
)
EXPECTED_TX1_MODEL_SHA256 = (
    "424911f1d7425001db3dc6792193ce6470b6b15ab7ec10a35267cc27bd46634c"
)
EXPECTED_METHODS = {
    "copy_k562",
    "train_gene_mean",
    "hvg_nearest",
    "hvg_ridge",
    "hvg_shuffled_ridge",
    "tx1_nearest",
    "tx1_ridge",
    "tx1_shuffled_ridge",
    "previous_hvg",
    "previous_tx1",
}
EXPECTED_COMPARATORS = {
    "previous_hvg": {
        "method": "hvg_st",
        "predictions_sha256": (
            "aee95e55226fc9beba74c6bbb9419dc84a5f8de3e2165ec44789ba33d68b1b56"
        ),
        "manifest_sha256": (
            "e167f53ecfc42afdb7a28aad1dce6f9aaa12445ad13fdfe6405439436c28ba89"
        ),
        "head_sha256": (
            "626a2a6abfbfd8a77c35806b8508d9cd765c7488dfcdb5292a0ed92eb3b6fa9e"
        ),
        "reason": "post_hoc_head_redesign_after_test_opening",
    },
    "previous_tx1": {
        "method": "tx1_3b_st",
        "predictions_sha256": (
            "bbcb1ad2c83d40f053987de63daffef44a204a0f113198750fc53e4ee71695c0"
        ),
        "manifest_sha256": (
            "856896d4f268ea20f299db50897bd59310953ff907d6c325ee834e2d4f2cb080"
        ),
        "head_sha256": (
            "d16daa7db1583bdb1820cb56133f15098cf0f9a95491b1022b2cfe73f0f3f9bc"
        ),
        "reason": "post_hoc_adapter_after_test_opening",
    },
}


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _test_ids(manifest_path: Path) -> list[str]:
    manifest = pd.read_csv(manifest_path)
    if not {"model_id", "role"}.issubset(manifest.columns):
        raise ValueError("manifest must contain model_id and role")
    ids = sorted(manifest.loc[manifest["role"] == "test", "model_id"].astype(str))
    if len(ids) != EXPECTED_TEST_LINES or len(set(ids)) != EXPECTED_TEST_LINES:
        raise ValueError("frozen test cohort must contain exactly 9 unique lines")
    return ids


def _pooled_test_context(
    cache_root: Path,
    test_ids: list[str],
    *,
    array_filename: str,
    prefix: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    expected_width: int | None = None
    for model_id in test_ids:
        path = cache_root / model_id / array_filename
        values = np.load(path, mmap_mode="r")
        if values.ndim != 2 or values.shape[0] < 1:
            raise ValueError(f"{model_id}: test cache must be non-empty 2D")
        width = int(values.shape[1])
        if expected_width is None:
            expected_width = width
        elif width != expected_width:
            raise ValueError("test cache widths differ across lines")
        mean = np.asarray(values, dtype=np.float64).mean(axis=0)
        if not np.isfinite(mean).all():
            raise ValueError(f"{model_id}: pooled test context is non-finite")
        row: dict[str, object] = {"model_id": model_id}
        row.update({f"{prefix}_{i:04d}": float(value) for i, value in enumerate(mean)})
        rows.append(row)
    return pd.DataFrame(rows).set_index("model_id")


def _validate_exposure_ledger(path: Path, test_ids: list[str]) -> dict[str, str]:
    ledger = pd.read_csv(path)
    required = {
        "model_id",
        "role",
        "pretraining_exact_context_status",
        "geneeffect_label_role",
        "formal_eligibility",
    }
    if not required.issubset(ledger.columns):
        raise ValueError("exposure ledger is missing required columns")
    test = ledger.loc[ledger["role"] == "test"].copy()
    if len(test) != len(test_ids) or set(test["model_id"].astype(str)) != set(test_ids):
        raise ValueError("exposure ledger does not exactly cover frozen test lines")
    expected = {
        "geneeffect_label_role": "opened_binding_historical",
        "pretraining_exact_context_status": "known_present",
        "formal_eligibility": "ineligible",
    }
    for column, value in expected.items():
        if set(test[column].astype(str)) != {value}:
            raise ValueError(f"unexpected test exposure status in {column}")
    return {"sha256": sha256_file(path), **expected}


def _load_comparator(
    path: Path,
    manifest_path: Path,
    *,
    output_method: str,
    source_method: str,
    expected_sha256: str,
    test_ids: list[str],
    slice_frame: pd.DataFrame,
) -> pd.DataFrame:
    if sha256_file(path) != expected_sha256:
        raise ValueError(f"{output_method}: comparator SHA256 mismatch")
    contract = EXPECTED_COMPARATORS[output_method]
    if sha256_file(manifest_path) != contract["manifest_sha256"]:
        raise ValueError(f"{output_method}: comparator manifest SHA256 mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required_manifest = {
        "formal": False,
        "mode": "diagnostic_prediction_only",
        "predictions_sha256": expected_sha256,
        "head_checkpoint_sha256": contract["head_sha256"],
        "expected_head_checkpoint_sha256": contract["head_sha256"],
        "reason": contract["reason"],
    }
    if any(manifest.get(key) != value for key, value in required_manifest.items()):
        raise ValueError(f"{output_method}: comparator manifest semantics differ")
    frame = pd.read_csv(
        path,
        usecols=["model_id", "depmap_column", "method", "k", "panel", "base_pred"],
    )
    selected = frame.loc[
        (frame["method"] == source_method) & (frame["k"] == 0) & (frame["panel"] == 0),
        ["model_id", "depmap_column", "base_pred"],
    ].copy()
    expected_keys = {
        (model_id, depmap_column)
        for model_id in test_ids
        for depmap_column in slice_frame["depmap_column"].astype(str)
    }
    actual_keys = set(
        zip(
            selected["model_id"].astype(str),
            selected["depmap_column"].astype(str),
            strict=True,
        )
    )
    if len(selected) != len(expected_keys) or actual_keys != expected_keys:
        raise ValueError(
            f"{output_method}: comparator coverage differs from frozen test"
        )
    selected["base_pred"] = pd.to_numeric(selected["base_pred"], errors="coerce")
    if not np.isfinite(selected["base_pred"].to_numpy(dtype=float)).all():
        raise ValueError(f"{output_method}: comparator predictions are non-finite")
    gene_by_column = dict(
        zip(
            slice_frame["depmap_column"].astype(str),
            slice_frame["gene_symbol"].astype(str),
            strict=True,
        )
    )
    selected["gene_symbol"] = selected["depmap_column"].map(gene_by_column)
    selected["method"] = output_method
    return selected[["model_id", "depmap_column", "gene_symbol", "method", "base_pred"]]


def build_frozen_predictions(
    *,
    phase_a_dir: Path,
    manifest_path: Path,
    raw_gene_effect_path: Path,
    cache_root: Path,
    exposure_ledger_path: Path,
    representation_paths: Mapping[str, Path],
    cache_arrays: Mapping[str, tuple[str, str]],
    comparator_paths: Mapping[str, Path],
    comparator_manifest_paths: Mapping[str, Path],
    pca_components: int = FROZEN_PCA_COMPONENTS,
    ridge_alpha: float = FROZEN_RIDGE_ALPHA,
    shuffle_seed: int = FROZEN_SHUFFLE_SEED,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Fit on 29 train_head lines and predict test lines without test labels."""
    if (
        pca_components != FROZEN_PCA_COMPONENTS
        or ridge_alpha != FROZEN_RIDGE_ALPHA
        or shuffle_seed != FROZEN_SHUFFLE_SEED
    ):
        raise ValueError("frozen-test hyperparameters cannot be changed")
    train, slice_frame, registration = _load_authorities(phase_a_dir, manifest_path)
    train_ids = train["model_id"].astype(str).tolist()
    test_ids = _test_ids(manifest_path)
    exposure = _validate_exposure_ledger(exposure_ledger_path, test_ids)
    train_long, prior_frame = _load_gene_effect(
        raw_gene_effect_path, registration, train_ids, slice_frame
    )
    genes = slice_frame["gene_symbol"].astype(str).tolist()
    depmap_by_gene = dict(
        zip(
            slice_frame["gene_symbol"].astype(str),
            slice_frame["depmap_column"].astype(str),
            strict=True,
        )
    )
    labels = train_long.pivot(
        index="model_id", columns="gene_symbol", values="gene_effect"
    ).reindex(index=train_ids, columns=genes)
    prior = prior_frame.set_index("gene_symbol")["gene_effect"].reindex(genes)
    if labels.isna().any().any() or prior.isna().any():
        raise ValueError("training labels or K562 prior lost frozen-gene coverage")
    cache_report = verify_cache(cache_root, frozen_manifest_path=manifest_path)
    if cache_report.get("status") != "verified":
        raise ValueError("full Tx1 cache verification failed")
    if set(representation_paths) != set(cache_arrays):
        raise ValueError("representation and cache-array names must match")

    rows: list[dict[str, object]] = []
    shared_predictions = {
        "copy_k562": prior.to_numpy(dtype=float),
        "train_gene_mean": labels.to_numpy(dtype=float).mean(axis=0),
    }
    for model_id in test_ids:
        for method, prediction in shared_predictions.items():
            for gene, value in zip(genes, prediction, strict=True):
                rows.append(
                    {
                        "model_id": model_id,
                        "depmap_column": depmap_by_gene[gene],
                        "gene_symbol": gene,
                        "method": method,
                        "base_pred": float(value),
                    }
                )

    representation_metadata: dict[str, object] = {}
    for name in sorted(representation_paths):
        train_features = load_representation(representation_paths[name], train_ids)
        array_filename, prefix = cache_arrays[name]
        test_features = _pooled_test_context(
            cache_root,
            test_ids,
            array_filename=array_filename,
            prefix=prefix,
        )
        if list(train_features.columns) != list(test_features.columns):
            raise ValueError(f"{name}: train/test feature columns differ")
        dropped: list[int] = []
        for line_index, model_id in enumerate(test_ids):
            predictions = fit_outer_fold(
                train_features.to_numpy(dtype=float),
                labels.to_numpy(dtype=float),
                test_features.loc[model_id].to_numpy(dtype=float),
                prior.to_numpy(dtype=float),
                pca_components=pca_components,
                ridge_alpha=ridge_alpha,
                shuffle_seed=shuffle_seed + line_index,
            )
            dropped.append(predictions.dropped_constant_feature_count)
            for suffix, prediction in (
                ("nearest", predictions.nearest_neighbor),
                ("ridge", predictions.ridge),
                ("shuffled_ridge", predictions.shuffled_ridge),
            ):
                for gene, value in zip(genes, prediction, strict=True):
                    rows.append(
                        {
                            "model_id": model_id,
                            "depmap_column": depmap_by_gene[gene],
                            "gene_symbol": gene,
                            "method": f"{name}_{suffix}",
                            "base_pred": float(value),
                        }
                    )
        representation_metadata[name] = {
            "cache_array": array_filename,
            "train_representation_sha256": sha256_file(representation_paths[name]),
            "test_cache_array_sha256": {
                model_id: sha256_file(cache_root / model_id / array_filename)
                for model_id in test_ids
            },
            "dropped_constant_feature_count_by_test_line": dropped,
        }

    if set(comparator_paths) != set(EXPECTED_COMPARATORS) or set(
        comparator_manifest_paths
    ) != set(EXPECTED_COMPARATORS):
        raise ValueError("both frozen previous-model comparators are required")
    comparator_metadata: dict[str, object] = {}
    for name in sorted(comparator_paths):
        contract = EXPECTED_COMPARATORS[name]
        comparator = _load_comparator(
            comparator_paths[name],
            comparator_manifest_paths[name],
            output_method=name,
            source_method=str(contract["method"]),
            expected_sha256=str(contract["predictions_sha256"]),
            test_ids=test_ids,
            slice_frame=slice_frame,
        )
        rows.extend(comparator.to_dict(orient="records"))
        comparator_metadata[name] = {
            "predictions_sha256": sha256_file(comparator_paths[name]),
            "manifest_sha256": sha256_file(comparator_manifest_paths[name]),
            "head_checkpoint_sha256": contract["head_sha256"],
            "reason": contract["reason"],
        }

    predictions = pd.DataFrame(rows).sort_values(
        ["model_id", "method", "gene_symbol"], kind="stable"
    )
    expected_methods = 2 + 3 * len(representation_paths) + len(comparator_paths)
    if len(predictions) != EXPECTED_TEST_LINES * len(genes) * expected_methods:
        raise ValueError("prediction table has incomplete frozen-test coverage")
    metadata: dict[str, object] = {
        "protocol_id": PREDICTION_PROTOCOL,
        "formal": False,
        "post_hoc": True,
        "test_labels_accessed": False,
        "prediction_first": True,
        "train_role": "train_head",
        "test_role": "test",
        "n_train_lines": len(train_ids),
        "n_test_lines": len(test_ids),
        "n_genes": len(genes),
        "config": {
            "pca_components": pca_components,
            "ridge_alpha": ridge_alpha,
            "shuffle_seed": shuffle_seed,
        },
        "representations": representation_metadata,
        "comparators": comparator_metadata,
        "exposure": exposure,
        "input_sha256": {
            "manifest": sha256_file(manifest_path),
            "phase_a_registration": sha256_file(
                phase_a_dir / "phase_a_registration.json"
            ),
            "phase_a_slice": sha256_file(
                phase_a_dir / "differentially_essential_slice.csv"
            ),
            "raw_gene_effect": sha256_file(raw_gene_effect_path),
        },
        "cache_manifest": {
            "sha256": sha256_file(cache_root / "manifest.json"),
            "tx1_source_manifest": json.loads(
                (cache_root / "manifest.json").read_text(encoding="utf-8")
            )["tx1_source_manifest"],
        },
    }
    return predictions.reset_index(drop=True), metadata


def write_predictions(
    predictions: pd.DataFrame, metadata: Mapping[str, object], output_dir: Path
) -> None:
    """Atomically write the prediction-only artifact."""
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
    )
    try:
        prediction_path = temporary / "predictions.csv"
        predictions.to_csv(prediction_path, index=False)
        payload = dict(metadata)
        payload["predictions_sha256"] = sha256_file(prediction_path)
        (temporary / "prediction_manifest.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output_dir)
    except Exception:
        import shutil

        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _load_test_truth(
    raw_gene_effect_path: Path,
    test_ids: list[str],
    slice_frame: pd.DataFrame,
) -> pd.DataFrame:
    columns = slice_frame["depmap_column"].astype(str).tolist()
    header = pd.read_csv(raw_gene_effect_path, nrows=0)
    id_column = str(header.columns[0])
    ids = pd.read_csv(raw_gene_effect_path, usecols=[id_column], dtype=str)[id_column]
    positions = {model_id: index for index, model_id in enumerate(ids)}
    missing = sorted(set(test_ids) - set(positions))
    if missing:
        raise ValueError(f"GeneEffect CSV is missing test rows: {missing}")
    selected_positions = {positions[model_id] for model_id in test_ids}
    frame = pd.read_csv(
        raw_gene_effect_path,
        usecols=[id_column, *columns],
        index_col=id_column,
        skiprows=lambda row: row != 0 and row - 1 not in selected_positions,
    ).apply(pd.to_numeric, errors="coerce")
    frame.index = frame.index.astype(str)
    frame = frame.reindex(index=test_ids, columns=columns)
    if not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError("test GeneEffect truth is incomplete or non-finite")
    return frame


def evaluate_predictions(
    *,
    prediction_dir: Path,
    phase_a_dir: Path,
    manifest_path: Path,
    raw_gene_effect_path: Path,
    exposure_ledger_path: Path,
    expected_prediction_manifest_sha256: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Open test labels only after validating the frozen prediction artifact."""
    prediction_manifest_path = prediction_dir / "prediction_manifest.json"
    if sha256_file(prediction_manifest_path) != expected_prediction_manifest_sha256:
        raise ValueError("prediction manifest differs from pre-registered SHA256")
    metadata = json.loads(prediction_manifest_path.read_text(encoding="utf-8"))
    prediction_path = prediction_dir / "predictions.csv"
    if metadata.get("protocol_id") != PREDICTION_PROTOCOL:
        raise ValueError("unexpected frozen-test prediction protocol")
    if metadata.get("test_labels_accessed") is not False:
        raise ValueError("prediction manifest does not attest label-free prediction")
    expected_semantics = {
        "formal": False,
        "post_hoc": True,
        "prediction_first": True,
        "n_train_lines": 29,
        "n_test_lines": 9,
        "n_genes": 587,
        "config": {
            "pca_components": FROZEN_PCA_COMPONENTS,
            "ridge_alpha": FROZEN_RIDGE_ALPHA,
            "shuffle_seed": FROZEN_SHUFFLE_SEED,
        },
    }
    if any(metadata.get(key) != value for key, value in expected_semantics.items()):
        raise ValueError("prediction manifest semantics differ from frozen contract")
    recorded_comparators = metadata.get("comparators")
    if not isinstance(recorded_comparators, Mapping):
        raise ValueError("prediction manifest comparator provenance is missing")
    for name, contract in EXPECTED_COMPARATORS.items():
        recorded = recorded_comparators.get(name)
        if not isinstance(recorded, Mapping) or any(
            recorded.get(key) != value
            for key, value in {
                "predictions_sha256": contract["predictions_sha256"],
                "manifest_sha256": contract["manifest_sha256"],
                "head_checkpoint_sha256": contract["head_sha256"],
                "reason": contract["reason"],
            }.items()
        ):
            raise ValueError(f"prediction manifest {name} provenance differs")
    cache_manifest = metadata.get("cache_manifest")
    if (
        not isinstance(cache_manifest, Mapping)
        or cache_manifest.get("sha256") != EXPECTED_CACHE_MANIFEST_SHA256
    ):
        raise ValueError("prediction cache-manifest provenance differs")
    tx1_source = cache_manifest.get("tx1_source_manifest")
    if (
        not isinstance(tx1_source, Mapping)
        or tx1_source.get("model_label") != "tahoe_x1_3b"
        or tx1_source.get("status") != "verified"
        or not isinstance(tx1_source.get("files"), Mapping)
        or tx1_source["files"].get("model.safetensors", {}).get("sha256")
        != EXPECTED_TX1_MODEL_SHA256
    ):
        raise ValueError("prediction Tx1 source/checkpoint provenance differs")
    if sha256_file(prediction_path) != metadata.get("predictions_sha256"):
        raise ValueError("prediction artifact SHA256 mismatch")
    train, slice_frame, registration = _load_authorities(phase_a_dir, manifest_path)
    del train
    test_ids = _test_ids(manifest_path)
    expected_inputs = {
        "manifest": sha256_file(manifest_path),
        "phase_a_registration": sha256_file(phase_a_dir / "phase_a_registration.json"),
        "phase_a_slice": sha256_file(
            phase_a_dir / "differentially_essential_slice.csv"
        ),
        "raw_gene_effect": sha256_file(raw_gene_effect_path),
    }
    recorded_inputs = metadata.get("input_sha256")
    if not isinstance(recorded_inputs, Mapping) or any(
        recorded_inputs.get(name) != digest for name, digest in expected_inputs.items()
    ):
        raise ValueError("prediction manifest authority hashes do not match")
    registered_gene_effect_sha = str(
        registration["sources"]["depmap_gene_effect"]["sha256"]
    )
    if expected_inputs["raw_gene_effect"] != registered_gene_effect_sha:
        raise ValueError("GeneEffect CSV SHA256 differs from Phase-A registration")
    exposure = _validate_exposure_ledger(exposure_ledger_path, test_ids)
    recorded_exposure = metadata.get("exposure")
    if (
        not isinstance(recorded_exposure, Mapping)
        or recorded_exposure.get("sha256") != exposure["sha256"]
    ):
        raise ValueError("prediction exposure-ledger SHA256 mismatch")
    predictions = pd.read_csv(prediction_path)
    required = {"model_id", "depmap_column", "gene_symbol", "method", "base_pred"}
    if not required.issubset(predictions.columns):
        raise ValueError("prediction artifact is missing required columns")
    if predictions.duplicated(["model_id", "depmap_column", "method"]).any():
        raise ValueError("prediction artifact has duplicate keys")
    if set(predictions["model_id"].astype(str)) != set(test_ids):
        raise ValueError("prediction artifact does not exactly cover test lines")
    methods = sorted(predictions["method"].astype(str).unique())
    if set(methods) != EXPECTED_METHODS:
        raise ValueError("prediction artifact does not contain the frozen method set")
    expected_rows = len(test_ids) * len(slice_frame) * len(EXPECTED_METHODS)
    if len(predictions) != expected_rows:
        raise ValueError("prediction artifact has incomplete frozen coverage")
    if not np.isfinite(pd.to_numeric(predictions["base_pred"], errors="coerce")).all():
        raise ValueError("prediction artifact contains non-finite values")
    gene_by_column = dict(
        zip(
            slice_frame["depmap_column"].astype(str),
            slice_frame["gene_symbol"].astype(str),
            strict=True,
        )
    )
    expected_symbols = predictions["depmap_column"].astype(str).map(gene_by_column)
    if expected_symbols.isna().any() or not expected_symbols.equals(
        predictions["gene_symbol"].astype(str)
    ):
        raise ValueError("prediction artifact gene mapping differs from Phase A")
    truth = _load_test_truth(raw_gene_effect_path, test_ids, slice_frame)
    baseline = "copy_k562"
    rows: list[dict[str, object]] = []
    for model_id in test_ids:
        line = predictions.loc[predictions["model_id"] == model_id]
        baseline_values = (
            line.loc[line["method"] == baseline]
            .set_index("depmap_column")["base_pred"]
            .reindex(truth.columns)
        )
        baseline_rho = float(spearmanr(baseline_values, truth.loc[model_id]).statistic)
        if not np.isfinite(baseline_rho):
            raise ValueError(f"{model_id}: copy-K562 Spearman is non-finite")
        for method in methods:
            values = (
                line.loc[line["method"] == method]
                .set_index("depmap_column")["base_pred"]
                .reindex(truth.columns)
            )
            if values.isna().any():
                raise ValueError(
                    f"{model_id}/{method}: prediction coverage is incomplete"
                )
            rho = float(spearmanr(values, truth.loc[model_id]).statistic)
            if not np.isfinite(rho):
                raise ValueError(f"{model_id}/{method}: Spearman is non-finite")
            rows.append(
                {
                    "model_id": model_id,
                    "method": method,
                    "k": 0,
                    "rho": rho,
                    "copy_k562_rho": baseline_rho,
                    "delta_rho": rho - baseline_rho,
                }
            )
    per_line = pd.DataFrame(rows)
    summary_rows: list[dict[str, object]] = []
    for method in methods:
        subset = per_line.loc[per_line["method"] == method]
        point, lo, hi = line_bootstrap_ci(subset["delta_rho"].to_numpy())
        summary_rows.append(
            {
                "method": method,
                "k": 0,
                "macro_rho": float(subset["rho"].mean()),
                "delta_rho": point,
                "ci_lo": lo,
                "ci_hi": hi,
                "negative_transfer_rate": float((subset["delta_rho"] < 0).mean()),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("method", kind="stable")
    direct_comparisons: list[dict[str, object]] = []
    for candidate, previous in (
        ("tx1_ridge", "previous_tx1"),
        ("hvg_ridge", "previous_hvg"),
    ):
        candidate_rho = per_line.loc[per_line["method"] == candidate].set_index(
            "model_id"
        )["rho"]
        previous_rho = per_line.loc[per_line["method"] == previous].set_index(
            "model_id"
        )["rho"]
        paired = candidate_rho.reindex(test_ids) - previous_rho.reindex(test_ids)
        point, lo, hi = line_bootstrap_ci(paired.to_numpy())
        direct_comparisons.append(
            {
                "candidate": candidate,
                "previous": previous,
                "delta_rho": point,
                "ci_lo": lo,
                "ci_hi": hi,
            }
        )
    verdict: dict[str, object] = {
        "protocol_id": PREDICTION_PROTOCOL,
        "status": "diagnostic_complete",
        "formal": False,
        "post_hoc_after_test_opening": True,
        "eligible_for_model_selection": False,
        "test_labels_accessed_during_prediction": False,
        "test_labels_accessed_during_evaluation": True,
        "n_test_lines": EXPECTED_TEST_LINES,
        "n_genes": len(slice_frame),
        "prediction_manifest_sha256": expected_prediction_manifest_sha256,
        "exposure_ledger_sha256": exposure["sha256"],
        "direct_comparisons": direct_comparisons,
        "tx1_scope": (
            "held-out-label transfer only; exact test contexts were present "
            "in Tx1 pretraining"
        ),
    }
    return per_line, summary.reset_index(drop=True), verdict
