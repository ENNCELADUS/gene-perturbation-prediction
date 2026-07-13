"""Leakage-regression tests for the exp05 outer cross-validation protocol."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import types

import numpy as np
import pandas as pd
import pytest

from aivc_model import cross_validate as cv
from aivc_model.gene_splits import FoldSpec
from aivc_model.prepare import AivcConfig, GeneBags, SealedGeneBags, load_config


def _toy_bags(genes: tuple[str, ...] = ("A", "B", "C", "D")) -> GeneBags:
    bags = tuple(
        np.full((2, 2), index + 1, dtype=np.float32)
        for index, _gene in enumerate(genes)
    )
    metadata = pd.DataFrame(
        {
            "perturbation_gene": genes,
            "depmap_gene_effect": np.linspace(-1.0, 0.0, len(genes)),
            "outer_fold": [1] * (len(genes) - 1) + [0],
        }
    )
    return GeneBags(
        genes=np.asarray(genes, dtype=object),
        y=metadata["depmap_gene_effect"].to_numpy(dtype=np.float32),
        input_bags=bags,
        latent_bags=tuple(bag.copy() for bag in bags),
        control_input=np.zeros((2, 2), dtype=np.float32),
        control_latent=np.zeros((2, 2), dtype=np.float32),
        cell_type_bags=None,
        control_cell_type=None,
        batch_bags=None,
        control_batch=None,
        feature_names=np.asarray(["X", "Y"], dtype=object),
        metadata=metadata,
        input_dim=2,
        latent_dim=2,
        gene_outer_folds=metadata["outer_fold"].to_numpy(dtype=np.int64),
    )


def _audited_config(tmp_path: Path) -> AivcConfig:
    config_path = tmp_path / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        f"""
data:
  h5ad_path: {tmp_path / "unused.h5ad"}
  overlap_csv: {tmp_path / "unused.csv"}
  output_dir: {tmp_path / "outputs"}
state:
  backend: linear_mock
  input_dim: 2
  output_dim: 2
  pert_dim: 2
projector:
  teacher: obsm
  latent_dim: 2
  ridge_alpha: 0.1
gmm:
  n_components: 2
  max_fit_cells: 8
train:
  run_id: audited
  seed: 13
  max_epochs: 1
  cell_set_len: 2
  freeze_state: true
  device: cpu
""",
        encoding="utf-8",
    )
    return load_config(config_path)


def test_gene_bag_views_preserve_only_requested_genes() -> None:
    bags = _toy_bags()
    selected = bags.for_genes(("C", "A"), stage="fine_tuning")
    assert selected.genes.tolist() == ["C", "A"]
    assert selected.metadata["perturbation_gene"].tolist() == ["C", "A"]
    np.testing.assert_array_equal(selected.input_bags[0], bags.input_bags[2])


def test_sealed_outer_test_supports_exactly_two_post_freeze_routes() -> None:
    bags = _toy_bags()
    sealed = SealedGeneBags(bags, ("B", "D"))
    with pytest.raises(ValueError, match="selected checkpoint is frozen"):
        sealed.open("generation_quality_outer_test", checkpoint_frozen=False)
    for stage in ("generation_quality_outer_test", "observed_b_oracle_outer_test"):
        assert sealed.open(stage, checkpoint_frozen=True).genes.tolist() == ["B", "D"]
    label_view = sealed.label_view(checkpoint_frozen=True)
    assert label_view.genes.tolist() == ["B", "D"]
    assert all(bag.shape[0] == 0 for bag in label_view.input_bags)
    with pytest.raises(ValueError, match="only be opened"):
        sealed.open("fine_tuning", checkpoint_frozen=True)


def test_run_training_fold_never_passes_outer_test_responses_to_fit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = _toy_bags()
    fold = FoldSpec(0, ("A", "C"), ("B",), ("D",))
    captured: list[dict[str, object]] = []

    def fake_run_training(*_args: object, **kwargs: object) -> dict[str, Path]:
        train_data = kwargs["train_data"]
        val_data = kwargs["val_data"]
        sealed_test = kwargs["sealed_test"]
        assert isinstance(train_data, GeneBags)
        assert isinstance(val_data, GeneBags)
        assert isinstance(sealed_test, SealedGeneBags)
        captured.append(
            {
                "train_genes": train_data.genes.tolist(),
                "val_genes": val_data.genes.tolist(),
                "train_hash": hashlib.sha256(
                    train_data.input_bags[0].tobytes()
                ).hexdigest(),
                "sealed": sealed_test,
            }
        )
        run_dir = Path(str(kwargs["run_dir_override"]))
        run_dir.mkdir(parents=True, exist_ok=True)
        audit = run_dir / "fit_audit_summary.json"
        audit.write_text(json.dumps({"checkpoint_sha256": captured[-1]["train_hash"]}))
        return {"fit_audit_summary": audit}

    monkeypatch.setattr(cv, "run_training", fake_run_training)
    common = {
        "config": types.SimpleNamespace(),
        "external": None,
        "fold_spec": fold,
        "source_fingerprint": "source",
    }
    cv.run_training_fold(data=data, run_dir=tmp_path / "first", **common)
    changed_bags = list(data.input_bags)
    changed_bags[3] = np.full_like(changed_bags[3], 999.0)
    changed = replace(data, input_bags=tuple(changed_bags))
    cv.run_training_fold(data=changed, run_dir=tmp_path / "changed", **common)

    assert captured[0]["train_genes"] == captured[1]["train_genes"] == ["A", "C"]
    assert captured[0]["val_genes"] == captured[1]["val_genes"] == ["B"]
    assert captured[0]["train_hash"] == captured[1]["train_hash"]


def test_audited_training_writes_three_final_scopes_and_fit_audit(
    tmp_path: Path,
) -> None:
    data = _toy_bags()
    fold = FoldSpec(0, ("A", "B"), ("C",), ("D",))

    paths = cv.run_training_fold(
        config=_audited_config(tmp_path),
        data=data,
        external=None,
        fold_spec=fold,
        run_dir=tmp_path / "fold_0",
        source_fingerprint="source",
    )

    predictions = pd.read_csv(paths["predictions"])
    assert set(predictions["evaluation_scope"]) == {
        "internal_outer_test",
        "generation_quality_outer_test",
        "observed_b_oracle_outer_test",
    }
    audit = pd.read_csv(paths["fit_access_audit"])
    assert set(audit["stage"]) >= {
        "projector_fit",
        "gmm_fit",
        "fine_tuning",
        "early_stopping",
        "generation_quality_outer_test",
        "observed_b_oracle_outer_test",
    }
    assert not audit.loc[audit["stage"].str.endswith("fit"), "checkpoint_frozen"].any()
    fit_summary = json.loads(paths["fit_audit_summary"].read_text())
    assert fit_summary["checkpoint_sha256"]
    assert fit_summary["state_sha256"]


def test_changing_outer_test_responses_cannot_change_fitted_artifacts(
    tmp_path: Path,
) -> None:
    data = _toy_bags()
    fold = FoldSpec(0, ("A", "B"), ("C",), ("D",))
    config = _audited_config(tmp_path)
    first = cv.run_training_fold(
        config=config,
        data=data,
        external=None,
        fold_spec=fold,
        run_dir=tmp_path / "first",
        source_fingerprint="source",
    )
    changed_input = list(data.input_bags)
    changed_latent = list(data.latent_bags)
    changed_input[3] = np.full_like(changed_input[3], 999.0)
    changed_latent[3] = np.full_like(changed_latent[3], 999.0)
    changed = replace(
        data,
        input_bags=tuple(changed_input),
        latent_bags=tuple(changed_latent),
    )
    second = cv.run_training_fold(
        config=config,
        data=changed,
        external=None,
        fold_spec=fold,
        run_dir=tmp_path / "changed",
        source_fingerprint="source",
    )
    first_audit = json.loads(first["fit_audit_summary"].read_text())
    second_audit = json.loads(second["fit_audit_summary"].read_text())
    for key in (
        "adapter_sha256",
        "state_sha256",
        "scvi_sha256",
        "gmm_sha256",
        "normalizer_sha256",
        "projector_sha256",
        "selected_layer",
        "best_epoch",
        "checkpoint_sha256",
    ):
        assert first_audit[key] == second_audit[key]


def test_cross_validation_writes_each_outer_test_gene_once_per_final_scope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    genes = tuple(f"GENE{index:05d}" for index in range(9338))
    labels = pd.DataFrame(
        {
            "perturbation_gene": genes,
            "depmap_gene_effect": np.linspace(-2.0, 1.0, len(genes)),
        }
    )
    manifest = pd.DataFrame(
        {"perturbation_gene": genes, "outer_fold": np.arange(len(genes)) % 5}
    )
    manifest_path = tmp_path / "outer.csv"
    manifest.to_csv(manifest_path, index=False)
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    sha_path = tmp_path / "outer.csv.sha256"
    sha_path.write_text(f"{digest}\n", encoding="utf-8")
    config = types.SimpleNamespace(
        data=types.SimpleNamespace(output_dir=tmp_path),
        train=types.SimpleNamespace(run_id="toy", seed=42),
        cv=types.SimpleNamespace(
            outer_split_manifest=manifest_path,
            outer_split_sha256_file=sha_path,
            inner_val_fraction=0.1,
            random_state=42,
        ),
    )
    monkeypatch.setattr(cv, "load_config", lambda _path: config)
    monkeypatch.setattr(
        cv,
        "load_gene_bags",
        lambda _config: types.SimpleNamespace(
            genes=np.asarray(genes, dtype=object),
            y=labels["depmap_gene_effect"].to_numpy(),
        ),
    )
    monkeypatch.setattr(cv, "load_external_gene_bags", lambda *_args, **_kwargs: None)

    def fake_fold_runner(**kwargs: object) -> dict[str, Path]:
        fold = kwargs["fold_spec"]
        assert isinstance(fold, FoldSpec)
        fold_dir = Path(str(kwargs["run_dir"]))
        artifacts = fold_dir / "artifacts"
        artifacts.mkdir(parents=True)
        rows = []
        for scope in (
            "internal_outer_test",
            "generation_quality_outer_test",
            "observed_b_oracle_outer_test",
        ):
            rows.extend(
                {
                    "perturbation_gene": gene,
                    "outer_fold": fold.outer_fold,
                    "inner_role": "outer_test",
                    "evaluation_scope": scope,
                    "y_true": 0.0,
                    "y_pred": 0.0,
                }
                for gene in fold.test_genes
            )
        rows.append(
            {
                "perturbation_gene": f"ADAMSON_ONLY_{fold.outer_fold}",
                "outer_fold": fold.outer_fold,
                "inner_role": "external_test",
                "evaluation_scope": "external:adamson_k562",
                "y_true": 0.0,
                "y_pred": 0.0,
            }
        )
        predictions = artifacts / "predictions.csv"
        pd.DataFrame(rows).to_csv(predictions, index=False)
        metrics = artifacts / "fold_metrics.csv"
        pd.DataFrame(
            [
                {"outer_fold": fold.outer_fold, "evaluation_scope": scope, "mse": 0.0}
                for scope in (
                    "internal_outer_test",
                    "generation_quality_outer_test",
                    "observed_b_oracle_outer_test",
                )
            ]
        ).to_csv(metrics, index=False)
        audit = artifacts / "fit_access_audit.csv"
        pd.DataFrame(
            [
                {
                    "stage": "fine_tuning",
                    "outer_fold": fold.outer_fold,
                    "genes": ";".join(fold.train_genes),
                    "checkpoint_frozen": False,
                }
            ]
        ).to_csv(audit, index=False)
        qa = artifacts / "external_alignment_qa.csv"
        pd.DataFrame(columns=["outer_fold", "source_name"]).to_csv(qa, index=False)
        summary = fold_dir / "fit_audit_summary.json"
        summary.write_text(json.dumps({"checkpoint_sha256": str(fold.outer_fold)}))
        return {
            "predictions": predictions,
            "fold_metrics": metrics,
            "fit_access_audit": audit,
            "external_alignment_qa": qa,
            "fit_audit_summary": summary,
        }

    monkeypatch.setattr(cv, "run_training_fold", fake_fold_runner)
    run_dir = cv.run_cross_validation(tmp_path / "config.yaml")
    predictions = pd.read_csv(run_dir / "artifacts" / "predictions.csv")
    for scope in (
        "internal_outer_test",
        "generation_quality_outer_test",
        "observed_b_oracle_outer_test",
    ):
        rows = predictions.query("evaluation_scope == @scope")
        assert rows["perturbation_gene"].nunique() == 9338
        assert not rows.duplicated(["perturbation_gene", "evaluation_scope"]).any()
    assert (run_dir / "artifacts" / "gene_splits.csv").exists()
    assert (run_dir / "summary.csv").exists()
    run_manifest = json.loads((run_dir / "run_manifest.json").read_text())
    assert run_manifest["canonical_split_sha256"] == digest
    assert run_manifest["esm_coverage"] == "9338/9338"
