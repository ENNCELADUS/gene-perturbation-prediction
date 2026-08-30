from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import weakref

import numpy as np
import pandas as pd
import pytest
import torch

from aivc_model.geneeffect_stage2_runner import (
    GeneEffectSupervisionCache,
    ResponseAssembly,
    Stage2Preflight,
    assemble_response_supervision,
    authenticate_stage1_seal,
    build_frozen_feature_store,
    build_joint_batch_factories,
    build_warmup_batch_factories,
    load_stage2_bundle_spec,
    preflight_stage2,
    run_registered_baselines,
    run_full_stage2,
    _authenticated_target_esm2_sha256,
    _formal_distributed_runtime,
)
from aivc_model.stage1_artifact import Stage1ArtifactManifest, sha256_file
from aivc_model.stage1_config import Stage1ObjectiveConfig
from aivc_model.stage2_artifacts import Stage2RunLayout
from aivc_model.state_core import sha256_strings


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_bundle_accepts_absolute_seal_for_relative_stage1_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.chdir(repo)
    stage1_config = _write(
        repo / "configs/experiments/13_geneeffect_226/stage1_response.yaml", "x: 1\n"
    )
    stage1_esm = _write(repo / "data/esm2/stage1.npz", "esm")
    cell_manifest = _write(repo / "results/cell_line_manifest.csv", "model_id\n")
    perturbseq = _write(repo / "configs/perturbseq_sources.json", "{}")
    state_model_dir = repo / "model/state"
    response_cache = repo / "data/response_cache"
    state_model_dir.mkdir(parents=True)
    response_cache.mkdir(parents=True)
    config = SimpleNamespace(
        paths=SimpleNamespace(
            stage1_config=stage1_config.relative_to(repo),
            stage1_esm_embeddings=stage1_esm.relative_to(repo),
            cell_line_manifest=cell_manifest.relative_to(repo),
            state_model_dir=state_model_dir.relative_to(repo),
            perturbseq_sources=perturbseq.relative_to(repo),
            response_cache=response_cache.relative_to(repo),
        )
    )
    bundle = load_stage2_bundle_spec(config)

    assert bundle.stage1_config.resolve() == stage1_config.resolve()


def _sealed_stage1(tmp_path: Path) -> tuple[SimpleNamespace, Stage1ArtifactManifest]:
    root = tmp_path / "stage1"
    checkpoint = _write(root / "best" / "pytorch_model.bin", "weights")
    state_hparams = _write(tmp_path / "state.ckpt", "state")
    run_manifest = _write(
        root / "run_manifest.json",
        json.dumps(
            {
                "best_epoch": 4,
                "selection_metric": "heldout_loss",
                "best_metric_value": 0.25,
                "input_sha256": {
                    "state_checkpoint": sha256_file(state_hparams),
                    "esm2_embeddings": "1" * 64,
                },
            }
        ),
    )
    metadata = _write(
        root / "best" / "metadata.json",
        json.dumps(
            {
                "checkpoint_kind": "best",
                "epoch": 4,
                "selection_metric": "heldout_loss",
                "metric_value": 0.25,
            }
        ),
    )
    objective = _write(root / "stage1_objective.json", "{}")
    genes = ("TP53", "KRAS")
    manifest = Stage1ArtifactManifest(
        schema_version=1,
        checkpoint_sha256=sha256_file(checkpoint),
        stage1_genes=genes,
        stage1_gene_vocabulary_sha256=sha256_strings(genes),
        esm2_artifact_sha256="1" * 64,
        state_hparams_sha256=sha256_file(state_hparams),
        compatibility_code_sha256={"code": "2" * 64},
        training_code_provenance_status="unavailable",
        training_code_provenance_reason=(
            "historical_run_has_no_immutable_training_code_identity"
        ),
        training_data_provenance_status="incomplete",
        training_data_provenance_missing_identities=(
            "cell_line_manifest",
            "tx1_basal_cache",
            "response_cache",
            "perturbseq_source_content",
        ),
        training_data_provenance_reason=(
            "historical_run_manifest_does_not_hash_all_training_data_inputs"
        ),
        config_sha256={"config": "3" * 64},
        source_sha256={"source": "4" * 64},
        legacy_esm_matrix_sha256=None,
        run_manifest_sha256=sha256_file(run_manifest),
        checkpoint_metadata_sha256=sha256_file(metadata),
        stage1_objective_sha256=sha256_file(objective),
    )
    manifest_path = root / "stage1_model_manifest.json"
    manifest.write(manifest_path)
    config = SimpleNamespace(
        paths=SimpleNamespace(
            stage1_manifest=manifest_path,
            stage1_checkpoint=checkpoint,
            state_hparams=state_hparams,
        )
    )
    return config, manifest


def test_authenticate_stage1_seal_checks_selected_run(tmp_path: Path) -> None:
    config, expected = _sealed_stage1(tmp_path)

    observed = authenticate_stage1_seal(
        config, target_esm_symbols=("TP53", "KRAS", "EGFR")
    )

    assert observed == expected


def test_authenticated_target_esm2_rejects_post_preflight_replacement(
    tmp_path: Path,
) -> None:
    esm2 = _write(tmp_path / "esm2.npz", "original")
    state = SimpleNamespace(
        report={"esm2": {"embedding_sha256": sha256_file(esm2)}},
        config=SimpleNamespace(paths=SimpleNamespace(esm2_embeddings=esm2)),
    )
    assert _authenticated_target_esm2_sha256(state) == sha256_file(esm2)
    esm2.write_text("replacement", encoding="utf-8")
    with pytest.raises(ValueError, match="changed after preflight"):
        _authenticated_target_esm2_sha256(state)


def _feature_store_fixture(tmp_path: Path, gene_count: int = 19):
    source = _write(tmp_path / "ACH-1.h5ad", "authenticated raw source")
    esm2 = _write(tmp_path / "esm2.npz", "authenticated esm")
    source_digest = sha256_file(source)
    genes = tuple(f"G{index}" for index in range(gene_count))
    model_ids = ("ACH-000001",)
    state = SimpleNamespace(
        config=SimpleNamespace(
            seeds=SimpleNamespace(projection=3, collator=5),
            joint=SimpleNamespace(response_batch_size=4),
            paths=SimpleNamespace(esm2_embeddings=esm2),
        ),
        stage1_manifest=SimpleNamespace(checkpoint_sha256="b" * 64),
        source_registry=pd.DataFrame(
            {"source_path": [str(source)]}, index=pd.Index(model_ids)
        ),
        report={
            "esm2": {"embedding_sha256": sha256_file(esm2)},
            "tx1_cache": {"source_sha256": {model_ids[0]: source_digest}},
        },
    )
    data = SimpleNamespace(
        genes=genes,
        model_ids=model_ids,
        e_g=np.zeros((gene_count, 1280), dtype=np.float32),
        z_c=np.zeros((1, 5120), dtype=np.float32),
        controls={model_ids[0]: np.zeros((2, 3), dtype=np.float32)},
        basal_hvg={model_ids[0]: np.zeros((2, 3), dtype=np.float32)},
        q_sc={
            model_ids[0]: (
                np.zeros((gene_count, 3), dtype=np.float32),
                np.ones(gene_count, dtype=bool),
            )
        },
        hvg_indices={},
    )
    return state, data, source


def test_frozen_feature_generation_bounds_live_prediction_bags_by_chunk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import aivc_model.geneeffect_stage2_runner as runner

    state, data, _source = _feature_store_fixture(tmp_path, gene_count=41)
    live = 0
    peak = 0

    def fake_predict(_model, controls, genes, *, seed):
        nonlocal live, peak
        assert len(controls) == len(genes)
        result = []
        for _gene in genes:
            prediction = torch.zeros((2, 3), dtype=torch.float32)
            live += 1
            peak = max(peak, live)

            def released() -> None:
                nonlocal live
                live -= 1

            weakref.finalize(prediction, released)
            result.append(prediction)
        return result

    def fake_features(*args, **kwargs):
        return SimpleNamespace(
            delta_proj=torch.zeros(256),
            s=torch.zeros(6),
            hvg_panel_mask=torch.tensor(False),
            own_gene_shift_mask=torch.tensor(False),
        )

    monkeypatch.setattr(runner, "predict_bags", fake_predict)
    monkeypatch.setattr(runner, "compute_condition_features", fake_features)
    backbone = torch.nn.Linear(1, 1)

    _projection, manifest = build_frozen_feature_store(
        state, data, backbone, tmp_path / "features"
    )

    assert peak <= state.config.joint.response_batch_size
    assert live == 0
    assert manifest["shards"]["ACH-000001"]["source_sha256"] == sha256_file(_source)


def test_frozen_feature_generation_never_claims_post_preflight_source_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import aivc_model.geneeffect_stage2_runner as runner

    state, data, source = _feature_store_fixture(tmp_path, gene_count=2)

    def drifting_predict(_model, controls, genes, *, seed):
        source.write_text("drifted during generation", encoding="utf-8")
        return [torch.zeros((2, 3), dtype=torch.float32) for _gene in genes]

    monkeypatch.setattr(runner, "predict_bags", drifting_predict)
    monkeypatch.setattr(
        runner,
        "compute_condition_features",
        lambda *args, **kwargs: SimpleNamespace(
            delta_proj=torch.zeros(256),
            s=torch.zeros(6),
            hvg_panel_mask=torch.tensor(False),
            own_gene_shift_mask=torch.tensor(False),
        ),
    )

    with pytest.raises(ValueError, match="changed during generation"):
        build_frozen_feature_store(
            state, data, torch.nn.Linear(1, 1), tmp_path / "features"
        )
    assert not (tmp_path / "features" / "manifest.json").exists()


def test_preflight_loads_contract_and_verifies_both_caches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import aivc_model.geneeffect_stage2_runner as runner

    files = {
        name: _write(tmp_path / name, "asset")
        for name in (
            "split.json",
            "gene_effect.csv",
            "registry.csv",
            "esm2.npz",
            "esm2_universe_manifest.json",
            "esm2_provenance_manifest.json",
            "esm2_uniprot_mapping.json",
            "esm2_uniprot_mapping.csv",
            "copy_prior.csv",
            "copy_prior_manifest.json",
            "registration.json",
        )
    }
    files["copy_prior.csv"].write_text(
        "gene_symbol,gene_effect\nTP53,-0.5\n", encoding="utf-8"
    )
    tx1_cache = tmp_path / "tx1"
    q_sc_cache = tmp_path / "q_sc"
    tx1_cache.mkdir()
    q_sc_cache.mkdir()
    source = _write(tmp_path / "ACH-1.h5ad", "raw")
    model_ids = tuple(f"ACH-{index:06d}" for index in range(226))
    split = SimpleNamespace(
        train=model_ids[:172],
        val=model_ids[172:199],
        test=model_ids[199:],
        all_model_ids=model_ids,
    )
    config = SimpleNamespace(
        source_sha256="a" * 64,
        features=SimpleNamespace(esm2_dim=1280),
        paths=SimpleNamespace(
            split=files["split.json"],
            gene_effect=files["gene_effect.csv"],
            source_registry=files["registry.csv"],
            tx1_registration=files["registration.json"],
            esm2_embeddings=files["esm2.npz"],
            esm2_universe_manifest=files["esm2_universe_manifest.json"],
            esm2_provenance_manifest=files["esm2_provenance_manifest.json"],
            esm2_uniprot_mapping_json=files["esm2_uniprot_mapping.json"],
            esm2_uniprot_mapping_csv=files["esm2_uniprot_mapping.csv"],
            copy_prior=files["copy_prior.csv"],
            copy_prior_manifest=files["copy_prior_manifest.json"],
            stage1_manifest=files["registry.csv"],
            tx1_cache=tx1_cache,
            q_sc_cache=q_sc_cache,
        ),
    )
    universe = SimpleNamespace(
        symbols=("TP53",),
        manifest={},
        coverage=pd.DataFrame(
            columns=[
                "gene_symbol",
                "train_finite",
                "val_finite",
                "test_finite",
                "included",
                "drop_reason",
            ]
        ),
    )
    residual = SimpleNamespace()
    g_var = SimpleNamespace(symbols=("TP53",), manifest={})
    registry = pd.DataFrame(
        {
            "source_path": [str(source)] * len(model_ids),
            "source_kind": ["h5ad"] * len(model_ids),
            "matrix_semantics": ["raw_umi_counts"] * len(model_ids),
        },
        index=pd.Index(model_ids, name="model_id"),
    )
    stage1 = SimpleNamespace(
        checkpoint_sha256="b" * 64,
        stage1_genes=("TP53",),
        training_code_provenance_status="unavailable",
        training_code_provenance_reason=(
            "historical_run_has_no_immutable_training_code_identity"
        ),
        training_data_provenance_status="incomplete",
        training_data_provenance_missing_identities=("tx1_basal_cache",),
        training_data_provenance_reason="historical training data incomplete",
    )
    monkeypatch.setattr(runner, "load_stage2_config", lambda path: config)
    monkeypatch.setattr(runner, "load_exp13_split", lambda path: split)
    monkeypatch.setattr(
        runner,
        "load_geneeffect_long",
        lambda path, value: pd.DataFrame({"gene_symbol": ["TP53"]}),
    )
    monkeypatch.setattr(
        runner,
        "load_esm2_embeddings",
        lambda path: SimpleNamespace(dim=1280, vectors_by_symbol={"TP53": object()}),
    )
    monkeypatch.setattr(
        runner, "build_scored_universe", lambda labels, value, symbols: universe
    )
    monkeypatch.setattr(
        runner, "build_residual_data", lambda labels, value, genes: residual
    )
    monkeypatch.setattr(runner, "build_g_var", lambda residuals, value, genes: g_var)
    monkeypatch.setattr(
        runner,
        "_authenticate_copy_prior",
        lambda value, split_value, labels: pd.Series({"TP53": -0.5}),
    )
    monkeypatch.setattr(
        runner,
        "restrict_scored_universe_to_copy_prior",
        lambda value, symbols: value,
    )
    monkeypatch.setattr(
        runner,
        "authenticate_target_esm2",
        lambda *args, **kwargs: (
            {},
            {
                "requested_model_id": "tiny-esm2",
                "loaded_model": {
                    "state_sha256": "d" * 64,
                    "config_sha256": "f" * 64,
                },
                "tokenizer": {"vocabulary_config_sha256": "e" * 64},
            },
        ),
    )
    monkeypatch.setattr(runner, "load_source_registry", lambda path, value: registry)
    cache_calls = []

    def fake_verify_cache(
        path: Path,
        *,
        expected_model_ids: tuple[str, ...],
        expected_source_sha256: dict[str, str],
        expected_matrix_semantics: str,
        expected_tx1_source_manifest: dict[str, object],
    ):
        cache_calls.append(
            (
                path,
                expected_model_ids,
                expected_source_sha256,
                expected_matrix_semantics,
                expected_tx1_source_manifest,
            )
        )
        return {"status": "verified", "discrepancies": []}

    monkeypatch.setattr(runner, "verify_cache", fake_verify_cache)
    registered_source = {"model_revision": "1" * 40}
    monkeypatch.setattr(
        runner,
        "authenticate_tx1_registration",
        lambda path: (registered_source, "f" * 64),
    )
    monkeypatch.setattr(
        runner,
        "verify_q_sc_shards",
        lambda rows, path, symbols: {"status": "passed", "discrepancies": []},
    )
    monkeypatch.setattr(runner, "authenticate_stage1_seal", lambda *a, **k: stage1)
    monkeypatch.setattr(
        runner, "authenticate_universe_stage1_vocabulary", lambda *a, **k: None
    )
    monkeypatch.setattr(runner, "sha256_file", lambda path: "c" * 64)

    bundle = SimpleNamespace()
    monkeypatch.setattr(runner, "load_stage2_bundle_spec", lambda value: bundle)

    result = preflight_stage2(tmp_path / "config.yaml")

    assert result.report["status"] == "passed"
    assert result.report["cell_lines"]["total"] == 226
    assert result.report["stage1"]["training_data_provenance_status"] == "incomplete"
    assert result.report["stage1"]["training_data_provenance_missing_identities"] == [
        "tx1_basal_cache"
    ]
    assert cache_calls == [
        (
            tx1_cache,
            model_ids,
            {model_id: "c" * 64 for model_id in model_ids},
            "raw_umi_counts",
            registered_source,
        )
    ]


def test_full_run_records_failure_without_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import aivc_model.geneeffect_stage2_runner as runner

    class FakeAccelerator:
        num_processes = 1
        process_index = 0
        local_process_index = 0
        is_main_process = True
        device = torch.device("cpu")
        mixed_precision = "bf16"

        @staticmethod
        def wait_for_everyone() -> None:
            return None

    monkeypatch.setattr(
        runner, "_create_accelerator", lambda precision: FakeAccelerator()
    )

    config_file = _write(tmp_path / "config.yaml", "config")
    split_file = _write(tmp_path / "split.json", "{}")
    stage1_manifest_file = _write(tmp_path / "stage1_model_manifest.json", "{}")
    stage1_checkpoint = _write(tmp_path / "stage1" / "best" / "model.bin", "x")
    _write(tmp_path / "stage1" / "stage1_objective.json", "{}")
    esm2_file = _write(tmp_path / "esm2.npz", "x")
    esm2_universe_manifest = _write(tmp_path / "esm2_universe_manifest.json", "{}")
    esm2_provenance_manifest = _write(tmp_path / "esm2_provenance_manifest.json", "{}")
    esm2_uniprot_mapping_json = _write(tmp_path / "esm2_uniprot_mapping.json", "{}")
    esm2_uniprot_mapping_csv = _write(
        tmp_path / "esm2_uniprot_mapping.csv", "gene_symbol\n"
    )
    copy_prior_file = _write(
        tmp_path / "copy_prior.csv", "gene_symbol,gene_effect\nTP53,0\n"
    )
    output_root = tmp_path / "runs"
    config = SimpleNamespace(
        source_sha256="a" * 64,
        paths=SimpleNamespace(
            output_root=output_root,
            split=split_file,
            stage1_manifest=stage1_manifest_file,
            stage1_checkpoint=stage1_checkpoint,
            esm2_embeddings=esm2_file,
            esm2_universe_manifest=esm2_universe_manifest,
            esm2_provenance_manifest=esm2_provenance_manifest,
            esm2_uniprot_mapping_json=esm2_uniprot_mapping_json,
            esm2_uniprot_mapping_csv=esm2_uniprot_mapping_csv,
            copy_prior=copy_prior_file,
        ),
        seeds=SimpleNamespace(train=7, collator=8, projection=9),
        features=SimpleNamespace(cells_per_context=128, cell_set_len=64),
        joint=SimpleNamespace(conditions_per_rank=256),
        distributed=SimpleNamespace(mixed_precision="bf16"),
        snapshot=lambda: {"config": "snapshot"},
    )
    state = Stage2Preflight(
        config=config,
        split=object(),
        universe=SimpleNamespace(manifest={"genes": ["TP53"]}),
        residual_data=object(),
        variable_genes=SimpleNamespace(manifest={"genes": ["TP53"]}),
        source_registry=object(),
        copy_prior=pd.Series({"TP53": 0.0}),
        stage1_manifest=SimpleNamespace(
            checkpoint_sha256="b" * 64,
            training_code_provenance_status="unavailable",
            training_code_provenance_reason=(
                "historical_run_has_no_immutable_training_code_identity"
            ),
            training_data_provenance_status="incomplete",
            training_data_provenance_missing_identities=("tx1_basal_cache",),
            training_data_provenance_reason="historical training data incomplete",
        ),
        bundle=SimpleNamespace(),
        report={
            "status": "passed",
            "esm2": {
                "embedding_sha256": sha256_file(esm2_file),
                "universe_manifest_sha256": sha256_file(esm2_universe_manifest),
                "provenance_manifest_sha256": sha256_file(esm2_provenance_manifest),
                "uniprot_mapping_json_sha256": sha256_file(esm2_uniprot_mapping_json),
                "uniprot_mapping_csv_sha256": sha256_file(esm2_uniprot_mapping_csv),
            },
            "tx1_cache": {"manifest_sha256": "1" * 64},
            "q_sc_cache": {"manifest_sha256": "2" * 64},
            "tx1_registration": {
                "registration_sha256": "3" * 64,
                "source_manifest_sha256": "4" * 64,
            },
        },
    )
    monkeypatch.setattr(runner, "preflight_stage2", lambda path: state)
    monkeypatch.setattr(runner, "load_stage2_config", lambda path: config)
    monkeypatch.setattr(
        runner,
        "_formal_distributed_runtime",
        lambda accelerator, value: {
            "world_size": 4,
            "mixed_precision": "bf16",
            "conditions_per_rank": 256,
            "global_conditions_per_step": 1024,
            "rank_topology": [],
        },
    )
    monkeypatch.setattr(runner, "PINNED_SPLIT_SHA256", sha256_file(split_file))
    monkeypatch.setattr(
        runner,
        "assemble_response_supervision",
        lambda state: ResponseAssembly(
            bags=object(),
            batch_factory=lambda epoch: (),
            heldout_batch_factory=lambda epoch: (),
            before_metrics={},
            batch_count=2,
            train_records=(),
            heldout_records=(),
        ),
    )

    def pin_response(layout, state, response):
        path = layout.root / "response_targets" / "lineage.json"
        path.parent.mkdir()
        runner._atomic_write_strict_json(path, {"lineage_sha256": "5" * 64})
        return "5" * 64, sha256_file(path)

    monkeypatch.setattr(runner, "_write_response_lineage_artifact", pin_response)
    monkeypatch.setattr(
        runner,
        "construct_stage2_backbone",
        lambda state, response: (
            torch.nn.Linear(1, 1),
            SimpleNamespace(
                loaded_keys=("state_adapter.weight",),
                dropped_keys=(),
                legacy_esm_matrix_authenticated=True,
                trainable=False,
            ),
        ),
    )
    monkeypatch.setattr(
        runner,
        "build_dependency_data",
        lambda state: (_ for _ in ()).throw(FileNotFoundError("missing basal asset")),
    )

    with pytest.raises(FileNotFoundError, match="missing basal asset"):
        run_full_stage2(config_file, run_id="trial-1")

    run_root = output_root / "trial-1"
    assert (run_root / "failure.json").is_file()
    assert not (run_root / "complete.json").exists()
    failure = json.loads((run_root / "failure.json").read_text())
    assert failure["phase"] == "training_assembly"
    manifest = json.loads((run_root / "run_manifest.json").read_text())
    assert manifest["status"] == "initialized"


def test_residual_target_artifact_persists_recomputable_contract(
    tmp_path: Path,
) -> None:
    import hashlib
    import aivc_model.geneeffect_stage2_runner as runner

    genes = ("G1", "G2")
    model_ids = ("T1", "V1", "E1")
    targets = np.asarray([[1.0, 2.0, 0.0], [3.0, 0.0, 5.0]], dtype=np.float32)
    label_mask = np.asarray([[True, True, False], [True, False, True]], dtype=bool)
    mu_train = np.asarray([1.0, 3.0], dtype=np.float64)
    residual_digest = hashlib.sha256()
    residual_digest.update("\n".join(genes).encode())
    residual_digest.update("\n".join(model_ids).encode())
    residual_digest.update(targets.tobytes())
    residual_digest.update(label_mask.tobytes())
    mu_digest = hashlib.sha256()
    mu_digest.update("\n".join(genes).encode())
    mu_digest.update(mu_train.tobytes())
    centering_digest = hashlib.sha256(b"T1").hexdigest()
    data = SimpleNamespace(
        genes=genes,
        model_ids=model_ids,
        targets=targets,
        label_mask=label_mask,
        residual_target_sha256=residual_digest.hexdigest(),
        mu_train_sha256=mu_digest.hexdigest(),
        centering_fit_model_ids_sha256=centering_digest,
    )
    state = SimpleNamespace(
        residual_data=SimpleNamespace(
            targets=SimpleNamespace(gene_mean=pd.Series(mu_train, index=genes))
        ),
        split=SimpleNamespace(supervised_train=("T1",)),
    )
    layout = Stage2RunLayout(tmp_path)
    _write(tmp_path / "run_manifest.json", '{"run_id":"trial"}')

    artifact_sha256 = runner._write_residual_targets_artifact(layout, state, data)

    artifact = tmp_path / "residual_targets.npz"
    assert artifact_sha256 == sha256_file(artifact)
    with np.load(artifact, allow_pickle=False) as loaded:
        assert set(loaded.files) == {
            "gene_symbols",
            "model_ids",
            "residual_targets",
            "label_mask",
            "mu_train",
            "centering_model_ids",
        }
        assert tuple(loaded["gene_symbols"].tolist()) == genes
        assert tuple(loaded["model_ids"].tolist()) == model_ids
        assert tuple(loaded["centering_model_ids"].tolist()) == ("T1",)
        assert loaded["residual_targets"].dtype == np.float32
        assert loaded["label_mask"].dtype == np.bool_
        assert loaded["mu_train"].dtype == np.float64
    manifest = json.loads((tmp_path / "run_manifest.json").read_text())
    assert manifest["residual_targets_artifact_sha256"] == artifact_sha256
    assert manifest["residual_target_sha256"] == residual_digest.hexdigest()
    assert manifest["mu_train_sha256"] == mu_digest.hexdigest()
    assert manifest["centering_fit_model_ids_sha256"] == centering_digest


def test_full_run_rejects_unsafe_run_id_before_preflight(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="run_id"):
        run_full_stage2(
            tmp_path / "config.yaml",
            run_id="../escape",
        )


def test_full_run_rejects_nonformal_topology_before_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import aivc_model.geneeffect_stage2_runner as runner

    class FakeAccelerator:
        num_processes = 1
        process_index = 0
        local_process_index = 0
        is_main_process = True
        device = torch.device("cpu")
        mixed_precision = "bf16"

        @staticmethod
        def wait_for_everyone() -> None:
            return None

    config = SimpleNamespace(
        source_sha256="a" * 64,
        distributed=SimpleNamespace(mixed_precision="bf16"),
        joint=SimpleNamespace(conditions_per_rank=256),
    )
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr(runner, "load_stage2_config", lambda path: config)
    monkeypatch.setattr(
        runner, "_create_accelerator", lambda precision: FakeAccelerator()
    )
    monkeypatch.setattr(
        runner,
        "preflight_stage2",
        lambda *args: (_ for _ in ()).throw(AssertionError("preflight must not run")),
    )

    with pytest.raises(RuntimeError, match="2- or 4-rank"):
        run_full_stage2(
            tmp_path / "config.yaml",
            run_id="trial-1",
        )


@pytest.mark.parametrize("world_size", [2, 4])
def test_formal_runtime_records_auto_detected_topology(
    monkeypatch: pytest.MonkeyPatch, world_size: int
) -> None:
    import aivc_model.geneeffect_stage2_runner as runner

    accelerator = SimpleNamespace(
        num_processes=world_size,
        process_index=world_size - 1,
        local_process_index=world_size - 1,
        mixed_precision="bf16",
        device=torch.device(f"cuda:{world_size - 1}"),
    )
    config = SimpleNamespace(
        distributed=SimpleNamespace(mixed_precision="bf16"),
        joint=SimpleNamespace(conditions_per_rank=256),
    )
    topology = [
        {
            "rank": rank,
            "local_rank": rank,
            "device": f"cuda:{rank}",
            "device_name": "NVIDIA H20",
            "hostname": "hpc",
        }
        for rank in range(world_size)
    ]
    monkeypatch.setenv("WORLD_SIZE", str(world_size))
    monkeypatch.setenv("RANK", str(world_size - 1))
    monkeypatch.setenv("LOCAL_RANK", str(world_size - 1))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "NVIDIA H20")
    monkeypatch.setattr(runner.socket, "gethostname", lambda: "hpc")

    def gather(output: list[object | None], local: object) -> None:
        assert local == topology[world_size - 1]
        output[:] = topology

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)

    runtime = runner._formal_distributed_runtime(accelerator, config)

    assert runtime == {
        "world_size": world_size,
        "mixed_precision": "bf16",
        "conditions_per_rank": 256,
        "global_conditions_per_step": 256 * world_size,
        "rank_topology": topology,
    }


def test_formal_runtime_rejects_unregistered_auto_detected_world_size() -> None:
    accelerator = SimpleNamespace(num_processes=3)
    config = SimpleNamespace()

    with pytest.raises(RuntimeError, match="2- or 4-rank"):
        _formal_distributed_runtime(accelerator, config)


def test_state_window_must_match_configured_cell_set_len() -> None:
    import aivc_model.geneeffect_stage2_runner as runner

    backbone = SimpleNamespace(
        state_adapter=SimpleNamespace(state_model=SimpleNamespace(cell_sentence_len=64))
    )
    runner._assert_configured_state_window(backbone, 64)
    with pytest.raises(ValueError, match="cell_sentence_len"):
        runner._assert_configured_state_window(backbone, 32)


def test_calibration_closures_require_joint_train_mode() -> None:
    import aivc_model.geneeffect_stage2_runner as runner

    class CalibrationModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = torch.nn.Linear(2, 2)
            self.head = torch.nn.Linear(2, 1)
            self.backbone_frozen = False

    model = CalibrationModel()
    observed: list[tuple[bool, bool, bool]] = []
    closure = runner._checked_calibration_closure(
        model,
        lambda: (
            observed.append(
                (model.training, model.backbone.training, model.head.training)
            )
            or torch.ones((), requires_grad=True)
        ),
    )

    model.eval()
    with pytest.raises(RuntimeError, match="train mode"):
        closure()
    assert observed == []

    model.train()
    assert closure().item() == 1.0
    assert observed == [(True, True, True)]

    model.backbone.weight.requires_grad_(False)
    with pytest.raises(RuntimeError, match="non-trainable joint parameters"):
        closure()


def test_long_rank_zero_action_removes_success_status(tmp_path: Path) -> None:
    import aivc_model.geneeffect_stage2_runner as runner

    class FakeAccelerator:
        num_processes = 1
        is_main_process = True
        waits = 0

        def wait_for_everyone(self) -> None:
            self.waits += 1

    accelerator = FakeAccelerator()
    status = tmp_path / ".status.json"
    called: list[str] = []
    runner._run_rank_zero_long_action(
        accelerator,
        "test action",
        status,
        lambda: called.append("done"),
    )

    assert called == ["done"]
    assert accelerator.waits == 2
    assert not status.exists()


def _response_assembly_case(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    before_metrics: dict[str, object],
) -> tuple[SimpleNamespace, SimpleNamespace, dict[str, object]]:
    import aivc_model.geneeffect_stage2_runner as runner

    cache = tmp_path / "response"
    _write(cache / "response_targets" / "manifest.json", "{}")
    bags = SimpleNamespace(
        genes=("TP53@ACH-000551", "KRAS@ACH-000551"),
        control_batch=np.asarray(["ACH-000551", "ACH-000551"]),
        control_input=np.ones((2, 4), dtype=np.float32),
        effective_control_target=np.ones((2, 3), dtype=np.float32),
        effective_target_bags=(
            np.ones((2, 3), dtype=np.float32),
            np.full((2, 3), 2.0, dtype=np.float32),
        ),
    )
    stage1 = SimpleNamespace(
        train=SimpleNamespace(
            max_cells_per_gene=8,
            total_cells_per_line=16,
            max_bag=4,
            data_seed=7,
        ),
        objective=Stage1ObjectiveConfig(
            anchor_weights=(("ACH-000001", 0.75), ("ACH-000551", 0.25)),
            required_anchor_metrics=("energy_distance", "mean_delta_mse"),
        ),
    )
    state = SimpleNamespace(
        split=SimpleNamespace(all_model_ids=("ACH-000001", "ACH-000551")),
        bundle=SimpleNamespace(
            stage1_config=tmp_path / "stage1.yaml",
            response_cache_dir=cache,
            cell_line_manifest=tmp_path / "lines.csv",
            state_model_dir=tmp_path / "state",
            perturbseq_sources=tmp_path / "sources.json",
        ),
        config=SimpleNamespace(
            paths=SimpleNamespace(
                tx1_cache=tmp_path / "tx1",
                stage1_checkpoint=tmp_path / "stage1" / "best" / "model.bin",
            ),
            joint=SimpleNamespace(response_batch_size=1),
            seeds=SimpleNamespace(train=11),
        ),
    )
    monkeypatch.setattr(runner, "load_stage1_config", lambda path: stage1)
    assembly_kwargs = {}

    def fake_assemble(**kwargs: object) -> object:
        assembly_kwargs.update(kwargs)
        return bags

    monkeypatch.setattr(runner, "assemble_train_response_gene_bags", fake_assemble)

    _write(
        tmp_path / "stage1" / "run_manifest.json",
        json.dumps(
            {
                "heldout_genes": {"ACH-000551": ["KRAS"]},
                "best_metric_value": 25.856595039367676,
            }
        ),
    )
    _write(
        tmp_path / "stage1" / "heldout_metrics.json",
        json.dumps(before_metrics),
    )
    return state, bags, assembly_kwargs


def test_response_assembly_builds_valid_deterministic_batches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    before_metrics = {
        "model_loss": 15.625461436575137,
        "per_line_model_loss": {"ACH-000551": 10.0408},
    }
    state, _, assembly_kwargs = _response_assembly_case(
        tmp_path, monkeypatch, before_metrics
    )
    assembly = assemble_response_supervision(state)
    first = list(assembly.batch_factory(3))
    second = list(assembly.batch_factory(3))

    assert assembly.batch_count == 1
    assert [batch.genes for batch in first] == [batch.genes for batch in second]
    assert [batch.genes for batch in first] == [("TP53",)]
    assert first[0].objective_weights.item() == 0.25
    assert [batch.genes for batch in assembly.heldout_batch_factory(0)] == [("KRAS",)]
    assert assembly.before_metrics == before_metrics
    assert assembly_kwargs["expected_cache_model_ids"] == (
        "ACH-000001",
        "ACH-000551",
    )


@pytest.mark.parametrize("model_loss", ["invalid", float("inf")])
def test_response_assembly_rejects_invalid_heldout_model_loss(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model_loss: object,
) -> None:
    state, _, _ = _response_assembly_case(
        tmp_path,
        monkeypatch,
        {"model_loss": model_loss},
    )

    with pytest.raises(
        ValueError,
        match="Stage 1 heldout_metrics model_loss must be a finite number",
    ):
        assemble_response_supervision(state)


def test_response_weights_preserve_anchor_mass_under_unequal_gene_counts() -> None:
    import aivc_model.geneeffect_stage2_runner as runner

    counts = {"ACH-000001": 1, "ACH-000002": 2, "ACH-000003": 3, "ACH-000004": 4}
    weights = {model_id: float(index) for index, model_id in enumerate(counts, 1)}
    labels = tuple(
        f"G{position}@{model_id}"
        for model_id, count in counts.items()
        for position in range(count)
    )
    bags = SimpleNamespace(
        genes=labels,
        control_batch=np.asarray(list(counts)),
        control_input=np.ones((4, 2), dtype=np.float32),
        effective_control_target=np.ones((4, 3), dtype=np.float32),
        effective_target_bags=tuple(np.ones((1, 3), dtype=np.float32) for _ in labels),
    )
    records = runner._response_records(bags, weights)
    runner._normalize_response_weights(records)

    mass = {
        model_id: sum(
            float(row["weight"]) for row in records if row["model_id"] == model_id
        )
        for model_id in counts
    }
    assert mass == weights


def test_response_lineage_pins_cache_targets_membership_and_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import aivc_model.geneeffect_stage2_runner as runner
    from aivc_model.tx1_response_gene_bags_cache import (
        write_response_targets_cache,
    )

    fingerprint = "f" * 64
    cache = tmp_path / "cache"
    genes = ("TP53@ACH-1", "KRAS@ACH-1")
    targets = (
        np.ones((2, 3), dtype=np.float32),
        np.full((1, 3), 2.0, dtype=np.float32),
    )
    write_response_targets_cache(
        cache,
        fingerprint,
        genes=genes,
        target_bags=targets,
        metadata=pd.DataFrame(
            {"model_id": ["ACH-1", "ACH-1"], "gene": ["TP53", "KRAS"]}
        ),
    )
    source = _write(tmp_path / "source.h5ad", "source")
    cell_manifest = _write(tmp_path / "lines.csv", "lines")
    source_config = _write(tmp_path / "sources.json", "{}")
    tx1_manifest = _write(tmp_path / "tx1" / "manifest.json", "{}")
    var_dims = _write(tmp_path / "state" / "var_dims.pkl", "dims")
    stage1_checkpoint = _write(tmp_path / "stage1" / "best" / "model.bin", "checkpoint")
    _write(
        tmp_path / "stage1" / "run_manifest.json",
        '{"heldout_genes":{"ACH-1":["KRAS"]}}',
    )
    _write(tmp_path / "stage1" / "heldout_metrics.json", '{"model_loss":1.0}')
    stage1_config = SimpleNamespace(
        train=SimpleNamespace(
            max_cells_per_gene=8, total_cells_per_line=16, data_seed=7
        )
    )
    monkeypatch.setattr(runner, "load_stage1_config", lambda path: stage1_config)
    monkeypatch.setattr(runner, "referenced_source_paths", lambda path: (source,))
    monkeypatch.setattr(
        runner, "response_targets_fingerprint", lambda **kwargs: fingerprint
    )
    control_tx1 = np.ones((2, 4), dtype=np.float32)
    control_hvg = np.ones((2, 3), dtype=np.float32)
    train_record = {
        "record_id": genes[0],
        "gene": "TP53",
        "model_id": "ACH-1",
        "control_tx1": control_tx1,
        "observed_hvg": targets[0],
        "control_hvg": control_hvg,
        "anchor_weight": 1.0,
        "weight": 1.0,
    }
    heldout_record = {
        **train_record,
        "record_id": genes[1],
        "gene": "KRAS",
        "observed_hvg": targets[1],
    }
    response = ResponseAssembly(
        bags=SimpleNamespace(),
        batch_factory=lambda epoch: (),
        heldout_batch_factory=lambda epoch: (),
        before_metrics={"model_loss": 1.0},
        batch_count=1,
        train_records=(train_record,),
        heldout_records=(heldout_record,),
    )
    state = SimpleNamespace(
        bundle=SimpleNamespace(
            stage1_config=tmp_path / "stage1.yaml",
            response_cache_dir=cache,
            cell_line_manifest=cell_manifest,
            perturbseq_sources=source_config,
            state_model_dir=var_dims.parent,
        ),
        config=SimpleNamespace(
            paths=SimpleNamespace(
                tx1_cache=tx1_manifest.parent,
                stage1_checkpoint=stage1_checkpoint,
            )
        ),
    )
    layout = Stage2RunLayout(tmp_path / "run")
    layout.root.mkdir()
    _write(layout.root / "run_manifest.json", '{"run_id":"trial"}')

    lineage_sha, artifact_sha = runner._write_response_lineage_artifact(
        layout, state, response
    )

    lineage = json.loads((layout.root / "response_targets/lineage.json").read_text())
    assert lineage["lineage_sha256"] == lineage_sha
    assert sha256_file(layout.root / "response_targets/lineage.json") == artifact_sha
    assert lineage["train_records"][0]["record_id"] == genes[0]
    assert lineage["heldout_records"][0]["record_id"] == genes[1]
    assert lineage["source_identities"]["referenced_source_sha256"] == {
        str(source): sha256_file(source)
    }
    run = json.loads((layout.root / "run_manifest.json").read_text())
    assert run["response_lineage_sha256"] == lineage_sha
    assert run["response_lineage_artifact_sha256"] == artifact_sha


def test_response_lineage_rejects_cache_target_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import aivc_model.geneeffect_stage2_runner as runner
    from aivc_model.tx1_response_gene_bags_cache import (
        write_response_targets_cache,
    )

    fingerprint = "f" * 64
    cache = tmp_path / "cache"
    target = np.ones((1, 2), dtype=np.float32)
    write_response_targets_cache(
        cache,
        fingerprint,
        genes=("TP53@ACH-1",),
        target_bags=(target,),
        metadata=pd.DataFrame({"model_id": ["ACH-1"]}),
    )
    np.save(
        cache / "response_targets/target_cells.npy", np.zeros((1, 2), dtype=np.float32)
    )
    monkeypatch.setattr(
        runner,
        "load_stage1_config",
        lambda path: SimpleNamespace(
            train=SimpleNamespace(
                max_cells_per_gene=1, total_cells_per_line=1, data_seed=1
            )
        ),
    )
    monkeypatch.setattr(runner, "referenced_source_paths", lambda path: ())
    monkeypatch.setattr(
        runner, "response_targets_fingerprint", lambda **kwargs: fingerprint
    )
    response = ResponseAssembly(
        bags=SimpleNamespace(),
        batch_factory=lambda epoch: (),
        heldout_batch_factory=lambda epoch: (),
        before_metrics={},
        batch_count=1,
        train_records=(
            {
                "record_id": "TP53@ACH-1",
                "gene": "TP53",
                "model_id": "ACH-1",
                "observed_hvg": target,
            },
        ),
        heldout_records=(),
    )
    state = SimpleNamespace(
        bundle=SimpleNamespace(
            stage1_config=tmp_path / "stage1.yaml",
            response_cache_dir=cache,
            cell_line_manifest=tmp_path / "lines",
            perturbseq_sources=tmp_path / "sources",
            state_model_dir=tmp_path / "state",
        ),
        config=SimpleNamespace(
            paths=SimpleNamespace(
                tx1_cache=tmp_path / "tx1",
                stage1_checkpoint=tmp_path / "stage1/best/model.bin",
            )
        ),
    )

    with pytest.raises(ValueError, match="differs from assembled record"):
        runner._write_response_lineage_artifact(
            Stage2RunLayout(tmp_path / "run"), state, response
        )


def test_paired_sampling_uses_barcode_hash_order_and_deterministic_padding() -> None:
    import aivc_model.geneeffect_stage2_runner as runner

    obs = pd.DataFrame(index=["cell-c", "cell-a", "cell-b"])
    first, metadata = runner._paired_sample_indices("ACH-000001", obs, count=5)
    second, _ = runner._paired_sample_indices("ACH-000001", obs, count=5)

    assert np.array_equal(first, second)
    assert len(set(first[:3].tolist())) == 3
    assert first[3:].tolist() == first[:2].tolist()
    assert metadata["distinct_count"] == 3
    assert metadata["padding_fraction"] == pytest.approx(0.4)


def _dependency_toctou_state(tmp_path: Path) -> tuple[SimpleNamespace, Path, Path]:
    model_id = "ACH-000001"
    tx1_line = tmp_path / "tx1" / model_id
    tx1_line.mkdir(parents=True)
    for filename in ("embeddings.npy", "hvg.npy", "obs.parquet"):
        (tx1_line / filename).write_bytes(filename.encode())
    tx1_manifest = _write(tmp_path / "tx1" / "manifest.json", "{}")
    q_sc_dir = tmp_path / "q_sc"
    q_sc_dir.mkdir()
    q_sc_path = q_sc_dir / f"{model_id}.npz"
    np.savez(
        q_sc_path,
        gene_symbols=np.asarray(["TP53"]),
        values=np.ones((1, 3), dtype=np.float32),
        available=np.ones(1, dtype=bool),
    )
    q_sc_manifest = _write(q_sc_dir / "manifest.json", "{}")
    esm2 = _write(tmp_path / "esm2.npz", "esm")
    state = SimpleNamespace(
        universe=SimpleNamespace(symbols=("TP53",)),
        split=SimpleNamespace(all_model_ids=(model_id,)),
        bundle=SimpleNamespace(state_model_dir=tmp_path / "state"),
        config=SimpleNamespace(
            features=SimpleNamespace(cells_per_context=2),
            paths=SimpleNamespace(
                esm2_embeddings=esm2,
                tx1_cache=tmp_path / "tx1",
                q_sc_cache=q_sc_dir,
            ),
        ),
        report={
            "tx1_cache": {
                "manifest_sha256": sha256_file(tx1_manifest),
                "line_artifact_sha256": {
                    model_id: {
                        filename: sha256_file(tx1_line / filename)
                        for filename in (
                            "embeddings.npy",
                            "hvg.npy",
                            "obs.parquet",
                        )
                    }
                },
            },
            "q_sc_cache": {
                "manifest_sha256": sha256_file(q_sc_manifest),
                "shard_sha256": {model_id: sha256_file(q_sc_path)},
            },
        },
    )
    return state, tx1_line, q_sc_path


def test_dependency_load_rejects_tx1_mutation_after_precheck(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import aivc_model.geneeffect_stage2_runner as runner

    state, tx1_line, _ = _dependency_toctou_state(tmp_path)
    monkeypatch.setattr(runner, "_authenticated_target_esm2_sha256", lambda state: "x")
    monkeypatch.setattr(
        runner,
        "load_esm2_embeddings",
        lambda path: SimpleNamespace(vectors_by_symbol={"TP53": np.ones(1280)}),
    )
    monkeypatch.setattr(
        runner,
        "load_hvg_gene_order",
        lambda path: [f"G{index}" for index in range(2_000)],
    )

    def mutate_after_load(cache_dir: Path, model_id: str):
        (tx1_line / "embeddings.npy").write_bytes(b"mutated")
        return (
            np.ones((2, 2560), dtype=np.float32),
            np.ones((2, 2000), dtype=np.float32),
            pd.DataFrame(index=["a", "b"]),
        )

    monkeypatch.setattr(runner, "load_line_cache", mutate_after_load)
    with pytest.raises(ValueError, match="Tx1 post-load embeddings.npy SHA-256"):
        runner.build_dependency_data(state)


def test_dependency_load_rejects_qsc_mutation_after_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import aivc_model.geneeffect_stage2_runner as runner

    state, _, q_sc_path = _dependency_toctou_state(tmp_path)
    monkeypatch.setattr(runner, "_authenticated_target_esm2_sha256", lambda state: "x")
    monkeypatch.setattr(
        runner,
        "load_esm2_embeddings",
        lambda path: SimpleNamespace(vectors_by_symbol={"TP53": np.ones(1280)}),
    )
    monkeypatch.setattr(
        runner,
        "load_hvg_gene_order",
        lambda path: [f"G{index}" for index in range(2_000)],
    )
    monkeypatch.setattr(
        runner,
        "load_line_cache",
        lambda cache_dir, model_id: (
            np.ones((2, 2560), dtype=np.float32),
            np.ones((2, 2000), dtype=np.float32),
            pd.DataFrame(index=["a", "b"]),
        ),
    )
    real_load = np.load

    class MutatingLoad:
        def __init__(self, path: Path) -> None:
            self.path = Path(path)
            self.inner = real_load(path, allow_pickle=False)

        def __enter__(self):
            return self.inner

        def __exit__(self, *args):
            self.inner.close()
            self.path.write_bytes(self.path.read_bytes() + b"tamper")

    monkeypatch.setattr(
        runner.np,
        "load",
        lambda path, allow_pickle=False: MutatingLoad(Path(path)),
    )
    with pytest.raises(ValueError, match="q_sc post-load SHA-256"):
        runner.build_dependency_data(state)


def test_response_batch_device_helper_moves_every_tensor() -> None:
    import aivc_model.geneeffect_stage2_runner as runner
    from aivc_model.geneeffect_training import ResponseSupervisionBatch

    batch = ResponseSupervisionBatch(
        controls_tx1=(torch.ones(2, 4),),
        observed_hvg=(torch.ones(2, 3),),
        control_hvg=(torch.ones(2, 3),),
        genes=("TP53",),
        objective_weights=torch.ones(1),
    )

    moved = runner.response_batch_to_device(batch, torch.device("cpu"))

    tensors = (
        *moved.controls_tx1,
        *moved.observed_hvg,
        *moved.control_hvg,
        moved.objective_weights,
    )
    assert all(tensor.device.type == "cpu" for tensor in tensors)


def test_registered_baselines_cover_every_required_method() -> None:
    genes = ("G1", "G2", "G3")
    train = ("T1", "T2", "T3", "T4")
    val = ("V1",)
    test = ("E1",)
    model_ids = (*train, *val, *test)
    rows = [
        {
            "model_id": model_id,
            "gene_symbol": gene,
            "gene_effect": float(line_index + gene_index / 10),
        }
        for line_index, model_id in enumerate(model_ids)
        for gene_index, gene in enumerate(genes)
    ]
    state = SimpleNamespace(
        residual_data=SimpleNamespace(targets=SimpleNamespace(long=pd.DataFrame(rows))),
        copy_prior=pd.Series({gene: float(index) for index, gene in enumerate(genes)}),
        split=SimpleNamespace(
            supervised_train=train,
            val=val,
            test=test,
            unlabeled_train=(),
        ),
        config=SimpleNamespace(
            seeds=SimpleNamespace(train=7),
            loss=SimpleNamespace(minimum_observations=3),
        ),
    )
    data = SimpleNamespace(
        genes=genes,
        model_ids=model_ids,
        z_c=np.asarray(
            [[index, index**2] for index in range(len(model_ids))], dtype=np.float32
        ),
    )

    result = run_registered_baselines(state, data)

    assert set(result.predictions["method"]) == {
        "gene_mean",
        "copy_prior",
        "nearest_line[z_c]",
        "context_pca_ridge[z_c]",
    }
    assert set(result.predictions["slice"]) == {"val", "test"}
    assert (
        result.summary["slices"]["val"]["methods"]["gene_mean"]["evaluation_status"]
        == "not_evaluable_constant_prediction"
    )


def test_split_factories_do_not_materialize_batches_eagerly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import aivc_model.geneeffect_stage2_runner as runner

    split_path = _write(tmp_path / "split.json", "{}")
    labels_path = _write(tmp_path / "labels.csv", "x")
    validation_ids = tuple(f"V{index}" for index in range(27))
    state = SimpleNamespace(
        split=SimpleNamespace(supervised_train=("T1",), val=validation_ids),
        config=SimpleNamespace(
            joint=SimpleNamespace(
                genes_per_batch=8,
                contexts_per_gene=32,
                conditions_per_rank=256,
            ),
            paths=SimpleNamespace(split=split_path, gene_effect=labels_path),
        ),
    )
    data = SimpleNamespace(mu_train_sha256="a" * 64)
    index = SimpleNamespace(rows=(), objective_weight=1.0)
    monkeypatch.setattr(runner, "_validation_batch_indices", lambda *args: (index,))

    def fail_if_eager(*args, **kwargs):
        raise AssertionError("validation batch was built eagerly")

    monkeypatch.setattr(runner, "_online_conditions", fail_if_eager)
    monkeypatch.setattr(runner, "_supervision_from_index", fail_if_eager)

    cache = SimpleNamespace(gather=fail_if_eager)
    supervision_cache = SimpleNamespace(gather=fail_if_eager)
    _, warmup_metric, _ = build_warmup_batch_factories(
        state,
        data,
        cache,
        supervision_cache,
        process_index=0,
        num_processes=1,
    )
    _, _, joint_metric, _ = build_joint_batch_factories(
        state,
        data,
        process_index=0,
        num_processes=1,
    )

    assert warmup_metric.batch_kind == "precomputed"
    assert joint_metric.batch_kind == "online"


def test_warmup_factories_gather_device_supervision_without_legacy_transfer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import aivc_model.geneeffect_stage2_runner as runner

    split_path = _write(tmp_path / "split.json", "{}")
    labels_path = _write(tmp_path / "labels.csv", "x")
    train_index = SimpleNamespace(
        rows=(
            SimpleNamespace(
                gene_index=2,
                context_indices=(2, 0, 2),
                label_mask=(True, True, False),
            ),
            SimpleNamespace(
                gene_index=0,
                context_indices=(1, 2, 0),
                label_mask=(True, False, True),
            ),
        ),
        objective_weight=0.0,
    )
    validation_index = SimpleNamespace(
        rows=(
            SimpleNamespace(
                gene_index=1,
                context_indices=(1, 2),
                label_mask=(True, True),
            ),
        ),
        objective_weight=1.0,
    )
    state = SimpleNamespace(
        split=SimpleNamespace(supervised_train=("T0",), val=("V0", "V1")),
        config=SimpleNamespace(
            joint=SimpleNamespace(genes_per_batch=2),
            paths=SimpleNamespace(split=split_path, gene_effect=labels_path),
        ),
    )
    data = SimpleNamespace(
        genes=("G0", "G1", "G2"),
        model_ids=("T0", "V0", "V1"),
        targets=np.asarray(
            [[0.0, 1.0, 2.0], [10.0, 11.0, 12.0], [20.0, 21.0, 22.0]],
            dtype=np.float32,
        ),
        label_mask=np.asarray(
            [[True, False, False], [True, True, True], [True, True, True]],
            dtype=bool,
        ),
        g_var_mask=np.asarray([False, True, True], dtype=bool),
        residual_target_sha256="a" * 64,
        centering_fit_model_ids_sha256="b" * 64,
        mu_train_sha256="c" * 64,
    )
    supervision_cache = GeneEffectSupervisionCache(data, device="cpu")
    feature_calls: list[tuple[tuple[int, int], ...]] = []

    def gather_features(pairs):
        feature_calls.append(tuple(pairs))
        return SimpleNamespace()

    monkeypatch.setattr(
        runner, "_epoch_batch_indices", lambda *args, **kwargs: (train_index,)
    )
    monkeypatch.setattr(
        runner,
        "_validation_batch_indices",
        lambda *args, **kwargs: (validation_index,),
    )

    def fail_legacy(*args, **kwargs):
        raise AssertionError("legacy CPU supervision path was used")

    monkeypatch.setattr(runner, "_supervision_from_index", fail_legacy)

    def fail_transfer(*args, **kwargs):
        raise AssertionError("per-batch Tensor.to device transfer was used")

    monkeypatch.setattr(torch.Tensor, "to", fail_transfer)
    train_factory, metric, _ = build_warmup_batch_factories(
        state,
        data,
        SimpleNamespace(gather=gather_features),
        supervision_cache,
        process_index=0,
        num_processes=1,
    )

    train_batch = next(iter(train_factory(0)))
    validation_batch = next(iter(metric.batch_factory()))

    assert feature_calls == [
        ((2, 2), (2, 0), (2, 2), (0, 1), (0, 2), (0, 0)),
        ((1, 1), (1, 2)),
    ]
    assert train_batch.objective_weight == 0.0
    assert train_batch.supervision.gene_symbols == ("G2", "G0")
    assert train_batch.supervision.context_model_ids_by_gene == (
        ("V1", "T0", "V1"),
        ("V0", "V1", "T0"),
    )
    assert torch.equal(
        train_batch.supervision.target,
        torch.tensor([[22.0, 20.0, 22.0], [1.0, 2.0, 0.0]]),
    )
    assert torch.equal(
        train_batch.supervision.label_mask,
        torch.tensor([[True, True, False], [False, False, True]]),
    )
    assert torch.equal(train_batch.supervision.g_var_mask, torch.tensor([True, False]))
    assert validation_batch.supervision.gene_symbols == ("G1",)
    assert torch.equal(
        validation_batch.supervision.target, torch.tensor([[11.0, 12.0]])
    )
    supervision_cache.close()
    with pytest.raises(RuntimeError, match="supervision cache is closed"):
        supervision_cache.gather(train_index)


def test_validation_batches_preserve_gene_order_without_g_var_anchors() -> None:
    import aivc_model.geneeffect_stage2_runner as runner

    data = SimpleNamespace(
        genes=tuple(f"G{index}" for index in range(7)),
        model_ids=("T0", "V0", "V1"),
        label_mask=np.ones((7, 3), dtype=bool),
        g_var_mask=np.zeros(7, dtype=bool),
    )

    batches = runner._validation_batch_indices(data, ("V0", "V1"), 3)

    assert [row.gene_index for batch in batches for row in batch.rows] == list(range(7))
    assert all(row.context_indices == (1, 2) for batch in batches for row in batch.rows)


def test_epoch_sharding_uses_explicit_accelerator_topology(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import aivc_model.geneeffect_stage2_runner as runner

    row = SimpleNamespace(
        gene_index=0, context_indices=(0, 1, 2), label_mask=(True, True, True)
    )
    batches = tuple(
        SimpleNamespace(rows=(row,), objective_weight=float(index + 1))
        for index in range(3)
    )
    monkeypatch.setattr(runner, "build_epoch_batches", lambda *args, **kwargs: batches)
    calls: list[tuple[int, int]] = []

    def shard(value, *, rank: int, world_size: int):
        calls.append((rank, world_size))
        return (value[rank],)

    monkeypatch.setattr(runner, "shard_batches", shard)
    monkeypatch.setenv("RANK", "99")
    monkeypatch.setenv("WORLD_SIZE", "100")
    data = SimpleNamespace(
        model_ids=("T0", "T1", "T2"),
        label_mask=np.ones((1, 3), dtype=bool),
        g_var_mask=np.ones(1, dtype=bool),
    )
    config = SimpleNamespace(
        joint=SimpleNamespace(genes_per_batch=1, contexts_per_gene=3),
        seeds=SimpleNamespace(train=7),
    )

    single = runner._epoch_batch_indices(
        data, data.model_ids, config, 0, process_index=0, num_processes=1
    )
    multi = runner._epoch_batch_indices(
        data, data.model_ids, config, 0, process_index=1, num_processes=2
    )

    assert [batch.objective_weight for batch in single] == [1.0, 2.0, 3.0]
    assert [batch.objective_weight for batch in multi] == [2.0]
    assert calls == [(1, 2)]


def test_epoch_sharding_retains_zero_objective_padding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import aivc_model.geneeffect_stage2_runner as runner
    from aivc_model.geneeffect_sampler import GeneContextBatchIndex, GeneContextRow

    row = GeneContextRow(
        gene_index=0, context_indices=(0, 1, 2), label_mask=(True, True, True)
    )
    batches = tuple(GeneContextBatchIndex(rows=(row,)) for _ in range(3))
    monkeypatch.setattr(runner, "build_epoch_batches", lambda *args, **kwargs: batches)
    data = SimpleNamespace(
        model_ids=("T0", "T1", "T2"),
        label_mask=np.ones((1, 3), dtype=bool),
        g_var_mask=np.ones(1, dtype=bool),
    )
    config = SimpleNamespace(
        joint=SimpleNamespace(genes_per_batch=1, contexts_per_gene=3),
        seeds=SimpleNamespace(train=7),
    )

    rank_one = runner._epoch_batch_indices(
        data, data.model_ids, config, 0, process_index=1, num_processes=2
    )

    assert [batch.objective_weight for batch in rank_one] == [1.0, 0.0]


def test_metric_json_normalizes_undefined_but_rejects_infinity(
    tmp_path: Path,
) -> None:
    import aivc_model.geneeffect_stage2_runner as runner

    normalized = runner._json_metrics({"undefined": float("nan")})
    assert normalized == {"undefined": None}
    path = tmp_path / "metrics.json"
    runner._atomic_write_strict_json(path, normalized)
    assert "NaN" not in path.read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="infinite"):
        runner._json_metrics({"invalid": float("inf")})


def test_response_metrics_forbid_historical_improvement_claim() -> None:
    import aivc_model.geneeffect_stage2_runner as runner

    payload = runner._response_metric_record(
        {"model_loss": 0.5},
        0.4,
        response_lineage_sha256="a" * 64,
        response_lineage_artifact_sha256="b" * 64,
    )

    assert set(payload) == {
        "before_stage2",
        "after_stage2",
        "comparison_status",
        "delta_reported",
        "hard_guard_applied",
    }
    assert payload["comparison_status"] == (
        "not_comparable_historical_input_lineage_incomplete"
    )
    assert payload["delta_reported"] is False
    assert payload["before_stage2"] == {
        "input_lineage_status": "historical_unverified_inputs",
        "metrics": {"model_loss": 0.5},
    }
    assert payload["after_stage2"] == {
        "input_lineage_status": "current_authenticated_inputs",
        "metrics": {"model_loss": 0.4},
        "response_lineage_sha256": "a" * 64,
        "response_lineage_artifact_sha256": "b" * 64,
    }
    assert not any("improvement" in key or "delta" == key for key in payload)


def test_complete_sentinel_must_match_active_run(tmp_path: Path) -> None:
    import aivc_model.geneeffect_stage2_runner as runner

    layout = Stage2RunLayout(tmp_path)
    _write(tmp_path / "complete.json", '{"status":"complete","run_id":"trial"}')
    runner._assert_complete_sentinel(layout, "trial")

    with pytest.raises(RuntimeError, match="active run"):
        runner._assert_complete_sentinel(layout, "different")


def test_model_and_baseline_coverage_must_match_exactly() -> None:
    import aivc_model.geneeffect_stage2_runner as runner

    model = pd.DataFrame(
        [("val", "V1", "G1")], columns=["split", "model_id", "gene_symbol"]
    )
    baseline = pd.DataFrame(
        [("val", "V1", "G2")], columns=["split", "model_id", "gene_symbol"]
    )

    with pytest.raises(RuntimeError, match="incomparable truth-mask coverage"):
        runner._assert_model_baseline_coverage(model, baseline)
