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
from aivc_model import gene_splits as gene_splits_module
from aivc_model import train as train_module
from aivc_model.gene_splits import FoldSpec, GeneAccessRecorder
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
    split_path = tmp_path / "toy_outer.csv"
    split_path.write_text(
        "perturbation_gene,outer_fold\nA,1\nB,1\nC,1\nD,0\n",
        encoding="utf-8",
    )
    split_sha_path = tmp_path / "toy_outer.csv.sha256"
    split_sha_path.write_text(
        hashlib.sha256(split_path.read_bytes()).hexdigest() + "\n",
        encoding="utf-8",
    )
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
cv:
  outer_split_manifest: {split_path}
  outer_split_sha256_file: {split_sha_path}
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


def _fingerprinted_config(tmp_path: Path) -> AivcConfig:
    config = _audited_config(tmp_path)
    cache_dir = tmp_path / "gwps_cache"
    cache_dir.mkdir()
    (cache_dir / "manifest.json").write_text(
        '{"source_fingerprint":"gwps"}\n', encoding="utf-8"
    )
    labels = tmp_path / "labels.csv"
    labels.write_text("perturbation_gene,label\nA,-1\n", encoding="utf-8")
    split = tmp_path / "outer.csv"
    split.write_text("perturbation_gene,outer_fold\nA,0\n", encoding="utf-8")
    split_sha = tmp_path / "outer.csv.sha256"
    split_sha.write_text(hashlib.sha256(split.read_bytes()).hexdigest() + "\n")
    esm = tmp_path / "esm2.npz"
    esm.write_bytes(b"esm")
    checkpoint = tmp_path / "state.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    model_dir = tmp_path / "state"
    model_dir.mkdir()
    (model_dir / "var_dims.pkl").write_bytes(b"sidecar")
    return replace(
        config,
        data=replace(
            config.data,
            prepared_cache_dir=cache_dir,
            overlap_csv=labels,
        ),
        cv=replace(
            config.cv,
            outer_split_manifest=split,
            outer_split_sha256_file=split_sha,
        ),
        state=replace(
            config.state,
            esm2_npz=esm,
            checkpoint_path=checkpoint,
            model_dir=model_dir,
        ),
    )


def test_experiment_source_fingerprint_changes_with_esm_cache(tmp_path: Path) -> None:
    config = _fingerprinted_config(tmp_path)

    first = cv.experiment_source_fingerprint(config)
    assert config.state.esm2_npz is not None
    config.state.esm2_npz.write_bytes(b"new-cache")
    second = cv.experiment_source_fingerprint(config)

    assert first != second


def test_experiment_source_fingerprint_hashes_small_state_sidecar_bytes(
    tmp_path: Path,
) -> None:
    config = _fingerprinted_config(tmp_path)

    first = cv.experiment_source_fingerprint(config)
    assert config.state.model_dir is not None
    (config.state.model_dir / "var_dims.pkl").write_bytes(b"changed-sidecar")
    second = cv.experiment_source_fingerprint(config)

    assert first != second


def test_gene_bag_views_preserve_only_requested_genes() -> None:
    bags = _toy_bags()
    with pytest.raises(ValueError, match="GeneAccessRecorder"):
        bags.for_genes(("C", "A"), stage="fine_tuning")
    bags = replace(
        bags,
        access_recorder=GeneAccessRecorder(FoldSpec(0, ("A", "C"), ("B",), ("D",))),
    )
    selected = bags.for_genes(("C", "A"), stage="fine_tuning")
    assert selected.genes.tolist() == ["C", "A"]
    assert selected.metadata["perturbation_gene"].tolist() == ["C", "A"]
    np.testing.assert_array_equal(selected.input_bags[0], bags.input_bags[2])


def test_actual_gene_view_rejects_unauthorized_access_without_recording() -> None:
    fold = FoldSpec(0, ("A", "B"), ("C",), ("D",))
    recorder = GeneAccessRecorder(fold)
    bags = replace(_toy_bags(), access_recorder=recorder)

    with pytest.raises(ValueError, match="outer-test"):
        bags.for_genes(("D",), stage="projector_fit")

    assert recorder.to_frame().empty
    selected = bags.for_genes(fold.train_genes, stage="fine_tuning")
    assert selected.genes.tolist() == ["A", "B"]
    assert recorder.to_frame()["stage"].tolist() == ["fine_tuning"]


def test_aggregation_rejects_unauthorized_emitted_event() -> None:
    fold = FoldSpec(0, ("A", "B"), ("C",), ("D",))
    audit = pd.DataFrame(
        [
            {
                "stage": "fine_tuning",
                "outer_fold": 0,
                "gene_count": 2,
                "gene_set_sha256": cv._gene_set_sha256(fold.train_genes),
                "checkpoint_frozen": False,
            },
            {
                "stage": "projector_fit",
                "outer_fold": 0,
                "gene_count": 1,
                "gene_set_sha256": cv._gene_set_sha256(fold.test_genes),
                "checkpoint_frozen": False,
            },
        ]
    )

    with pytest.raises(ValueError, match="authorized role"):
        cv._assert_access_audit(audit, [fold])


def test_sealed_outer_test_supports_exactly_two_post_freeze_routes() -> None:
    bags = replace(
        _toy_bags(),
        access_recorder=GeneAccessRecorder(FoldSpec(0, ("A",), ("C",), ("B", "D"))),
    )
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
        "normalizer_fit",
        "fine_tuning",
        "early_stopping",
        "observed_b_oracle_fit",
        "observed_b_oracle_selection",
        "generation_quality_outer_test",
        "observed_b_oracle_outer_test",
    }
    assert set(audit["stage"]).isdisjoint({"state_fit", "scvi_fit", "layer_selection"})
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
        "oracle_best_epoch",
        "oracle_checkpoint_sha256",
    ):
        assert first_audit[key] == second_audit[key]


def test_audited_fold_artifacts_share_exact_fit_authority(tmp_path: Path) -> None:
    run_dir = tmp_path / "authority"
    fold = FoldSpec(0, ("A", "B"), ("C",), ("D",))
    config = _fingerprinted_config(tmp_path)

    cv.run_training_fold(
        config=config,
        data=_toy_bags(),
        external=None,
        fold_spec=fold,
        run_dir=run_dir,
        source_fingerprint="source",
    )

    expected_hash = train_module._sha256_strings(fold.train_genes)
    metadata_paths = [
        run_dir / "artifacts" / "ridge_projector_fit" / "metadata.json",
        run_dir / "artifacts" / "fixed_gmm_fit" / "metadata.json",
        run_dir / "artifacts" / "normalizer_fit" / "metadata.json",
        run_dir / "artifacts" / "esm_adapter_fit" / "metadata.json",
        run_dir / "artifacts" / "c_head_fit" / "metadata.json",
        run_dir / "artifacts" / "observed_b_oracle_fit" / "metadata.json",
        run_dir / "models" / "best" / "metadata.json",
        run_dir / "models" / "final" / "metadata.json",
    ]
    for path in metadata_paths:
        metadata = json.loads(path.read_text(encoding="utf-8"))
        assert metadata["source_fingerprint"] == "source"
        assert metadata["canonical_split_sha256"]
        assert metadata["outer_fold"] == 0
        assert metadata["fit_stage"] == "inner_train"
        assert metadata["fit_genes_sha256"] == expected_hash
        assert metadata["train_genes"] == ["A", "B"]
        assert metadata["val_genes"] == ["C"]
        assert metadata["test_genes"] == ["D"]


def test_projector_cache_rejects_source_change_and_test_gene_contamination(
    tmp_path: Path,
) -> None:
    config = _audited_config(tmp_path)
    data = _toy_bags(("A", "B"))
    split = train_module.GeneSplit(
        train=np.asarray([0, 1], dtype=np.int64),
        val=np.asarray([], dtype=np.int64),
        test=np.asarray([], dtype=np.int64),
    )
    fold = FoldSpec(0, ("A", "B"), ("C",), ("D",))
    first_authority = train_module._fold_artifact_authority(
        config,
        fold,
        source_fingerprint="aaa",
        canonical_split_sha256="split",
    )
    metadata = train_module._projector_cache_metadata(
        config, data, split, authority=first_authority
    )
    train_module._write_projector_cache(tmp_path, metadata, np.eye(2), np.zeros(2))
    changed_authority = replace(first_authority, source_fingerprint="bbb")
    changed = train_module._projector_cache_metadata(
        config, data, split, authority=changed_authority
    )
    contaminated = {
        **metadata,
        "train_genes": ["A", "B", "D"],
        "fit_genes_sha256": train_module._sha256_strings(("A", "B", "D")),
    }

    assert train_module._load_projector_cache(tmp_path, changed) is None
    assert not train_module._cache_metadata_matches(tmp_path, contaminated)


def test_observed_b_oracle_selection_depends_on_validation_response_only(
    tmp_path: Path,
) -> None:
    data = _toy_bags()
    fold = FoldSpec(0, ("A", "B"), ("C",), ("D",))
    config = replace(
        _audited_config(tmp_path),
        train=replace(
            _audited_config(tmp_path).train,
            max_epochs=20,
            learning_rate=0.1,
        ),
    )
    first = cv.run_training_fold(
        config=config,
        data=data,
        external=None,
        fold_spec=fold,
        run_dir=tmp_path / "oracle_first",
        source_fingerprint="source",
    )
    changed_input = list(data.input_bags)
    changed_latent = list(data.latent_bags)
    changed_input[2] = np.full_like(changed_input[2], -100.0)
    changed_latent[2] = np.full_like(changed_latent[2], -100.0)
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
        run_dir=tmp_path / "oracle_changed_val",
        source_fingerprint="source",
    )
    first_audit = json.loads(first["fit_audit_summary"].read_text())
    second_audit = json.loads(second["fit_audit_summary"].read_text())
    assert first_audit["checkpoint_sha256"] == second_audit["checkpoint_sha256"]
    assert (
        first_audit["oracle_checkpoint_sha256"]
        != second_audit["oracle_checkpoint_sha256"]
    )


def test_run_training_fold_rejects_nonempty_run_directory(tmp_path: Path) -> None:
    run_dir = tmp_path / "fold_0"
    run_dir.mkdir()
    (run_dir / "stale.txt").write_text("stale", encoding="utf-8")

    with pytest.raises(FileExistsError, match="fresh run directory"):
        cv.run_training_fold(
            config=_audited_config(tmp_path),
            data=_toy_bags(),
            external=None,
            fold_spec=FoldSpec(0, ("A", "B"), ("C",), ("D",)),
            run_dir=run_dir,
            source_fingerprint="source",
        )


def test_audited_esm_uses_exact_canonical_manifest_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical_order = ("D", "B", "A", "C")
    manifest = pd.DataFrame(
        {
            "perturbation_gene": canonical_order,
            "outer_fold": [0, 1, 1, 1],
        }
    )
    manifest_path = tmp_path / "outer.csv"
    manifest.to_csv(manifest_path, index=False)
    sha_path = tmp_path / "outer.csv.sha256"
    sha_path.write_text(
        f"{hashlib.sha256(manifest_path.read_bytes()).hexdigest()}\n",
        encoding="utf-8",
    )
    esm_path = tmp_path / "esm2.npz"
    np.savez(
        esm_path,
        symbols=np.asarray(canonical_order, dtype=object),
        vectors=np.arange(12, dtype=np.float32).reshape(4, 3),
        resolved=np.ones(4, dtype=bool),
    )
    base = _audited_config(tmp_path)
    config = replace(
        base,
        cv=replace(
            base.cv,
            outer_split_manifest=manifest_path,
            outer_split_sha256_file=sha_path,
        ),
        state=replace(
            base.state,
            gene_tokenizer="esm2",
            esm2_npz=esm_path,
            esm2_adapter_hidden=4,
            require_resolved_esm2=True,
        ),
    )
    monkeypatch.setattr(train_module, "CANONICAL_GENE_COUNT", 4)
    monkeypatch.setattr(gene_splits_module, "CANONICAL_GENE_COUNT", 4)
    monkeypatch.setattr(
        gene_splits_module,
        "CANONICAL_OUTER_FOLDS",
        frozenset({0, 1}),
    )

    paths = cv.run_training_fold(
        config=config,
        data=_toy_bags(),
        external=None,
        fold_spec=FoldSpec(0, ("A", "C"), ("B",), ("D",)),
        run_dir=tmp_path / "esm_fold",
        source_fingerprint="source",
        canonical_gene_order=canonical_order,
    )

    evidence = json.loads(paths["runtime_evidence"].read_text())
    assert evidence["esm_resolved_count"] == 4
    assert (
        evidence["esm_gene_order_sha256"]
        == hashlib.sha256("\n".join(canonical_order).encode("utf-8")).hexdigest()
    )


def test_scvi_fit_event_is_emitted_only_when_fit_boundary_executes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _audited_config(tmp_path)
    config = replace(base, projector=replace(base.projector, teacher="scvi"))
    monkeypatch.setattr(
        train_module,
        "_fit_audited_scvi_latents",
        lambda _config, train, val, _split, _artifacts, _accelerator, _authority: (
            train,
            val,
        ),
    )
    monkeypatch.setattr(
        train_module,
        "_project_audited_scvi_data",
        lambda _config, data, _artifacts: data,
    )

    paths = cv.run_training_fold(
        config=config,
        data=_toy_bags(),
        external=None,
        fold_spec=FoldSpec(0, ("A", "B"), ("C",), ("D",)),
        run_dir=tmp_path / "scvi_fold",
        source_fingerprint="source",
    )

    audit = pd.read_csv(paths["fit_access_audit"])
    assert audit["stage"].tolist().count("scvi_fit") == 1


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
    monkeypatch.setattr(cv, "experiment_source_fingerprint", lambda _config: "source")

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
                    "gene_count": len(fold.train_genes),
                    "gene_set_sha256": cv._gene_set_sha256(fold.train_genes),
                    "checkpoint_frozen": False,
                }
            ]
        ).to_csv(audit, index=False)
        qa = artifacts / "external_alignment_qa.csv"
        pd.DataFrame(columns=["outer_fold", "source_name"]).to_csv(qa, index=False)
        summary = fold_dir / "fit_audit_summary.json"
        summary.write_text(json.dumps({"checkpoint_sha256": str(fold.outer_fold)}))
        runtime_evidence = fold_dir / "runtime_evidence.json"
        runtime_evidence.write_text(
            json.dumps(
                {
                    "esm_resolved_count": 9338,
                    "esm_total_count": 9338,
                    "esm_gene_order_sha256": hashlib.sha256(
                        "\n".join(genes).encode("utf-8")
                    ).hexdigest(),
                    "state_input_dim": 2000,
                    "state_output_dim": 2000,
                    "state_pert_dim": 512,
                    "state_feature_match_count": 2000,
                }
            ),
            encoding="utf-8",
        )
        return {
            "predictions": predictions,
            "fold_metrics": metrics,
            "fit_access_audit": audit,
            "external_alignment_qa": qa,
            "fit_audit_summary": summary,
            "runtime_evidence": runtime_evidence,
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
    canonical_output = run_dir / "artifacts" / "gene_splits.csv"
    assert len(pd.read_csv(canonical_output)) == 9338
    assert hashlib.sha256(canonical_output.read_bytes()).hexdigest() == digest
    assert len(pd.read_csv(run_dir / "artifacts" / "fold_roles.csv")) == 5 * 9338
    assert (run_dir / "summary.csv").exists()
    run_manifest = json.loads((run_dir / "run_manifest.json").read_text())
    assert run_manifest["canonical_split_sha256"] == digest
    assert run_manifest["esm_resolved_count"] == 9338
    assert run_manifest["esm_total_count"] == 9338
    assert run_manifest["state_feature_match_count"] == 2000
    assert run_manifest["checkpoint_dimensions"]["pert_dim"] == 512
