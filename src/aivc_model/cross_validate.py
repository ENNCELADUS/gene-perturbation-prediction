"""Strict frozen-manifest cross-validation orchestration for exp05."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from accelerate import Accelerator
import numpy as np
import pandas as pd

from aivc_model.gene_splits import (
    CANONICAL_GENE_COUNT,
    FINAL_RESPONSE_STAGES,
    FIT_STAGES,
    SELECTION_STAGES,
    FoldSpec,
    assert_gene_access,
    attach_gene_provenance,
    load_canonical_outer_manifest,
    make_inner_fold_spec,
)
from aivc_model.prepare import (
    AivcConfig,
    ExternalGeneBags,
    GeneBags,
    SealedGeneBags,
    load_config,
    load_external_gene_bags,
    load_gene_bags,
)
from aivc_model.train import run_training


def run_training_fold(
    config: AivcConfig,
    data: GeneBags,
    external: ExternalGeneBags | None,
    fold_spec: FoldSpec,
    run_dir: Path,
    source_fingerprint: str,
    accelerator: Accelerator | None = None,
) -> dict[str, Path]:
    """Run one outer fold through role-limited GeneBags views."""
    assert_gene_access(
        "fine_tuning",
        fold_spec.train_genes,
        fold_spec,
        checkpoint_frozen=False,
    )
    assert_gene_access(
        "early_stopping",
        fold_spec.val_genes,
        fold_spec,
        checkpoint_frozen=False,
    )
    train_data = data.for_genes(fold_spec.train_genes, stage="fine_tuning")
    val_data = data.for_genes(fold_spec.val_genes, stage="early_stopping")
    manifest = _manifest_from_bags(data)
    train_data = replace(
        train_data,
        metadata=attach_gene_provenance(
            train_data.metadata,
            manifest,
            "perturbation_gene",
            "fine_tuning",
        ),
    )
    val_data = replace(
        val_data,
        metadata=attach_gene_provenance(
            val_data.metadata,
            manifest,
            "perturbation_gene",
            "gene_effect",
        ),
    )
    sealed_test = SealedGeneBags(data, fold_spec.test_genes)
    return run_training(
        config,
        accelerator=accelerator,
        train_data=train_data,
        val_data=val_data,
        sealed_test=sealed_test,
        external_override=external,
        fold_spec=fold_spec,
        run_dir_override=run_dir,
        source_fingerprint=source_fingerprint,
    )


def run_cross_validation(
    config_path: Path,
    accelerator: Accelerator | None = None,
) -> Path:
    """Execute all frozen outer folds and aggregate audited final artifacts."""
    config = load_config(config_path)
    data = load_gene_bags(config)
    labels = pd.DataFrame(
        {
            "perturbation_gene": [str(gene).upper() for gene in data.genes],
            "depmap_gene_effect": np.asarray(data.y, dtype=float),
        }
    )
    manifest_path, expected_sha256 = _manifest_authority(config)
    manifest = load_canonical_outer_manifest(
        manifest_path,
        labels,
        expected_sha256,
    )
    _assert_canonical_universe(manifest)
    if isinstance(data, GeneBags):
        data = replace(
            data,
            metadata=attach_gene_provenance(
                data.metadata,
                manifest,
                "perturbation_gene",
                "gwps_response",
            ),
        )
    run_id = config.train.run_id or "state_aivc_cv"
    run_dir = config.data.output_dir / "runs" / run_id
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    external = load_external_gene_bags(
        config,
        data,
        artifacts_dir,
        project_scvi=False,
    )
    fingerprint = _source_fingerprint(config_path, expected_sha256, labels)

    folds: list[FoldSpec] = []
    outputs: list[dict[str, Path]] = []
    for outer_fold in sorted(manifest["outer_fold"].unique()):
        fold = make_inner_fold_spec(
            manifest,
            labels,
            int(outer_fold),
            config.cv.inner_val_fraction,
            config.cv.random_state,
        )
        fold_dir = run_dir / f"fold_{fold.outer_fold}"
        folds.append(fold)
        outputs.append(
            run_training_fold(
                config=config,
                data=data,
                external=external,
                fold_spec=fold,
                run_dir=fold_dir,
                source_fingerprint=fingerprint,
                accelerator=accelerator,
            )
        )
    _aggregate_outputs(
        run_dir,
        manifest,
        folds,
        outputs,
        manifest_path,
        expected_sha256,
        fingerprint,
        config,
        exact_feature_matches=getattr(data, "input_dim", None),
    )
    return run_dir


def _manifest_authority(config: AivcConfig) -> tuple[Path, str]:
    manifest_path = config.cv.outer_split_manifest
    sha_path = config.cv.outer_split_sha256_file
    if manifest_path is None or sha_path is None:
        raise ValueError("canonical outer manifest and SHA-256 file are required")
    text = sha_path.read_text(encoding="utf-8")
    if len(text) != 65 or not text.endswith("\n"):
        raise ValueError("outer split SHA-256 file must contain one digest and newline")
    return manifest_path, text[:-1]


def _manifest_from_bags(data: GeneBags) -> pd.DataFrame:
    if data.gene_outer_folds is not None:
        folds = np.asarray(data.gene_outer_folds, dtype=np.int64)
    elif "outer_fold" in data.metadata:
        folds = data.metadata["outer_fold"].to_numpy(dtype=np.int64)
    else:
        raise ValueError("GeneBags must carry canonical outer_fold provenance")
    return pd.DataFrame(
        {
            "perturbation_gene": [str(gene).upper() for gene in data.genes],
            "outer_fold": folds,
        }
    )


def _assert_canonical_universe(manifest: pd.DataFrame) -> None:
    if (
        len(manifest) != CANONICAL_GENE_COUNT
        or manifest["perturbation_gene"].nunique() != CANONICAL_GENE_COUNT
    ):
        raise ValueError("canonical universe must contain exactly 9338 unique genes")


def _source_fingerprint(
    config_path: Path,
    split_sha256: str,
    labels: pd.DataFrame,
) -> str:
    digest = hashlib.sha256()
    if config_path.exists():
        digest.update(config_path.read_bytes())
    digest.update(split_sha256.encode("ascii"))
    digest.update(
        labels.sort_values("perturbation_gene").to_csv(index=False).encode("utf-8")
    )
    return digest.hexdigest()


def _aggregate_outputs(
    run_dir: Path,
    manifest: pd.DataFrame,
    folds: list[FoldSpec],
    outputs: list[dict[str, Path]],
    manifest_path: Path,
    split_sha256: str,
    source_fingerprint: str,
    config: AivcConfig,
    exact_feature_matches: int | None,
) -> None:
    artifacts_dir = run_dir / "artifacts"
    predictions = _concat_output(outputs, "predictions")
    metrics = _concat_output(outputs, "fold_metrics")
    access_audit = _concat_output(outputs, "fit_access_audit")
    external_qa = _concat_output(outputs, "external_alignment_qa")
    _assert_predictions(predictions, manifest)
    _assert_access_audit(access_audit, folds)
    _write_frame(metrics, artifacts_dir / "fold_metrics.csv")
    _write_frame(predictions, artifacts_dir / "predictions.csv")
    _write_frame(access_audit, artifacts_dir / "fit_access_audit.csv")
    _write_frame(external_qa, artifacts_dir / "external_alignment_qa.csv")
    _write_frame(_gene_split_rows(manifest, folds), artifacts_dir / "gene_splits.csv")
    _write_frame(_summarize_metrics(metrics), run_dir / "summary.csv")
    fold_seeds = {
        str(fold.outer_fold): int(config.cv.random_state + fold.outer_fold + 1)
        for fold in folds
    }
    payload: dict[str, Any] = {
        "canonical_split_path": str(manifest_path),
        "canonical_split_sha256": split_sha256,
        "canonical_gene_count": CANONICAL_GENE_COUNT,
        "esm_coverage": "9338/9338",
        "source_fingerprint": source_fingerprint,
        "fold_seeds": fold_seeds,
        "checkpoint_dimensions": _checkpoint_dimensions(config),
        "exact_feature_matches": exact_feature_matches,
        "fold_fit_audit_summaries": [
            str(paths["fit_audit_summary"]) for paths in outputs
        ],
        "artifacts": {
            "fold_metrics": str(artifacts_dir / "fold_metrics.csv"),
            "predictions": str(artifacts_dir / "predictions.csv"),
            "gene_splits": str(artifacts_dir / "gene_splits.csv"),
            "fit_access_audit": str(artifacts_dir / "fit_access_audit.csv"),
            "external_alignment_qa": str(artifacts_dir / "external_alignment_qa.csv"),
        },
    }
    (run_dir / "run_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _checkpoint_dimensions(config: AivcConfig) -> dict[str, int | None]:
    state = getattr(config, "state", None)
    return {
        "input_dim": getattr(state, "input_dim", None),
        "output_dim": getattr(state, "output_dim", None),
        "pert_dim": getattr(state, "pert_dim", None),
    }


def _concat_output(outputs: list[dict[str, Path]], key: str) -> pd.DataFrame:
    frames = [pd.read_csv(paths[key]) for paths in outputs]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _assert_predictions(predictions: pd.DataFrame, manifest: pd.DataFrame) -> None:
    required = {"perturbation_gene", "outer_fold", "inner_role", "evaluation_scope"}
    if not required <= set(predictions.columns):
        raise ValueError(
            f"predictions are missing columns {sorted(required - set(predictions))}"
        )
    internal_scopes = {
        "internal_outer_test",
        "generation_quality_outer_test",
        "observed_b_oracle_outer_test",
    }
    internal = predictions.loc[predictions["evaluation_scope"].isin(internal_scopes)]
    canonical = manifest.set_index("perturbation_gene")["outer_fold"]
    observed = internal["perturbation_gene"].map(canonical)
    if observed.isna().any() or not np.array_equal(
        observed.to_numpy(dtype=np.int64),
        internal["outer_fold"].to_numpy(dtype=np.int64),
    ):
        raise ValueError(
            "derived prediction outer_fold conflicts with canonical manifest"
        )
    for scope in internal_scopes:
        rows = predictions.loc[predictions["evaluation_scope"] == scope]
        if (
            rows["perturbation_gene"].nunique() != CANONICAL_GENE_COUNT
            or rows.duplicated(["perturbation_gene", "evaluation_scope"]).any()
        ):
            raise ValueError(f"each canonical gene must appear once in {scope}")


def _assert_access_audit(audit: pd.DataFrame, folds: list[FoldSpec]) -> None:
    if audit.empty:
        raise ValueError("fit access audit is empty")
    fold_by_id = {fold.outer_fold: fold for fold in folds}
    for row in audit.to_dict("records"):
        stage = str(row["stage"])
        fold = fold_by_id[int(row["outer_fold"])]
        checkpoint_frozen = bool(row.get("checkpoint_frozen", False))
        genes_text = row.get("genes")
        if isinstance(genes_text, str):
            genes = tuple(gene for gene in genes_text.split(";") if gene)
            assert_gene_access(stage, genes, fold, checkpoint_frozen)
            continue
        expected_genes = _stage_genes(stage, fold)
        expected_hash = _gene_set_sha256(expected_genes)
        if str(row.get("gene_set_sha256")) != expected_hash:
            raise ValueError(
                f"{stage} audit gene set does not match its authorized role"
            )
        assert_gene_access(stage, expected_genes, fold, checkpoint_frozen)


def _stage_genes(stage: str, fold: FoldSpec) -> tuple[str, ...]:
    if stage in FIT_STAGES:
        return fold.train_genes
    if stage in SELECTION_STAGES:
        return fold.val_genes
    if stage in FINAL_RESPONSE_STAGES:
        return fold.test_genes
    raise ValueError(f"unknown gene-access stage {stage!r}")


def _gene_set_sha256(genes: tuple[str, ...]) -> str:
    value = "\n".join(sorted(str(gene).upper() for gene in genes))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _gene_split_rows(manifest: pd.DataFrame, folds: list[FoldSpec]) -> pd.DataFrame:
    rows = []
    for fold in folds:
        roles = {
            **dict.fromkeys(fold.train_genes, "inner_train"),
            **dict.fromkeys(fold.val_genes, "inner_validation"),
            **dict.fromkeys(fold.test_genes, "outer_test"),
        }
        for row in manifest.itertuples(index=False):
            rows.append(
                {
                    "perturbation_gene": row.perturbation_gene,
                    "outer_fold": int(row.outer_fold),
                    "evaluation_outer_fold": fold.outer_fold,
                    "inner_role": roles[str(row.perturbation_gene)],
                }
            )
    return pd.DataFrame(rows)


def _summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty or "evaluation_scope" not in metrics:
        return pd.DataFrame()
    numeric = [
        column
        for column in metrics.select_dtypes(include=[np.number]).columns
        if column != "outer_fold"
    ]
    rows = []
    for scope, frame in metrics.groupby("evaluation_scope", sort=True):
        for metric in numeric:
            values = frame[metric].dropna().to_numpy(dtype=float)
            if values.size:
                rows.append(
                    {
                        "evaluation_scope": scope,
                        "metric": metric,
                        "mean": float(values.mean()),
                        "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
                        "n_folds": int(values.size),
                    }
                )
    return pd.DataFrame(rows)


def _write_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
