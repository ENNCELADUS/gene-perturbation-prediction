"""Tests for scripts/train_tx1_geneeffect_head.py and its two library
modules -- Phase D Task 5 configuration, dry-run CLI validation, and
fail-fast orchestration gates.

The former full local training-chain E2E cases were intentionally removed:
they imposed excessive CPU load on developer machines. Lightweight module
tests cover response generation, pooled-feature assembly, head training, and
Phase F validation independently; this file retains path/config/hash guards
that do not materialize the full training chain.

No GPU, no real Phase C checkpoint, no real DepMap file, and no real Tx1
embeddings exist on this machine, so every fixture here is synthetic, built
the same way every other Phase D test file builds them
(``tx1_embed_cache.write_line_cache``, the shared ``conftest`` manifest
helpers). Nothing here is allowed to skip silently.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import scripts.train_tx1_geneeffect_head as cli
import aivc_model.tx1_geneeffect_pipeline_run as tx1_geneeffect_pipeline_run_module
from aivc_model.gene_splits import sha256_file
from aivc_model.model import LinearMockStateModel, PerturbationVectorAdapter
from aivc_model.model import StateForwardAdapter
from aivc_model.tx1_basal import load_line_manifest
from aivc_model.tx1_embed_cache import EMBEDDING_WIDTH, write_line_cache
from aivc_model.tx1_geneeffect_data import (
    generate_validation_line_registration,
)
from aivc_model.tx1_geneeffect_eval import (
    K_SCHEDULE,
    EvaluationContractError,
    verify_artifact_hashes as _real_verify_artifact_hashes,
)
from aivc_model.tx1_geneeffect_head import Tx1GeneEffectHead
from aivc_model.tx1_geneeffect_pipeline import (
    ObjectiveConfig,
    PhaseFConfig,
    StateArmConfig,
    TrainingArmConfig,
    Tx1GeneEffectHeadConfig,
    load_pipeline_config,
    validate_config_shape,
    validate_widths,
    verify_gene_vocabulary_authenticity,
)
from aivc_model.tx1_geneeffect_pipeline_run import run_training_pipeline
from aivc_model.tx1_geneeffect_prediction_run import (
    load_and_verify_head_provenance,
    verify_expected_head_sha256,
)
from aivc_model.tx1_geneeffect_train import SELECTION_METRIC_NAME
from aivc_model.tx1_predicted_response import ForwardOnlyStateModel, resolve_device
from conftest import tx1_manifest_row as _manifest_row
from conftest import write_tx1_cache_run_manifest as _write_cache_run_manifest
from conftest import write_tx1_line_manifest as _write_manifest

_REAL_CONFIG_DIR = Path("configs/experiments/12_tx1_st_geneeffect/phase_d")
_REAL_CONFIGS = ("tx1_arm.yaml", "hvg_arm.yaml")

_RESPONSE_DIM = 4
_PERT_DIM = 3
_CELL_SET_LEN = 4
_N_PLAIN_GENES = 50
_N_LABELS = 50
_N_PANELS = 2
_TEST_LINES = ("ACH-T1", "ACH-T2")


def test_prediction_only_cli_never_dispatches_training(
    monkeypatch, tmp_path: Path
) -> None:
    config = object()
    captured: dict[str, object] = {}
    monkeypatch.setattr(cli, "load_pipeline_config", lambda _path: config)

    def fake_prediction_pipeline(received_config, **kwargs):
        captured["config"] = received_config
        captured.update(kwargs)
        return {"predictions": tmp_path / "predictions.csv"}

    monkeypatch.setattr(cli, "run_prediction_pipeline", fake_prediction_pipeline)

    def fail_training(*_args, **_kwargs):
        raise AssertionError("prediction-only CLI dispatched training")

    monkeypatch.setattr(cli, "run_training_pipeline", fail_training)
    checkpoint = tmp_path / "head.pt"
    run_dir = tmp_path / "diagnostic"
    exit_code = cli.main(
        [
            "--config",
            str(tmp_path / "config.yaml"),
            "--line-manifest",
            str(tmp_path / "lines.csv"),
            "--tx1-cache-dir",
            str(tmp_path / "cache"),
            "--depmap-gene-effect",
            str(tmp_path / "gene_effect.csv"),
            "--run-dir",
            str(run_dir),
            "--prediction-only-head-checkpoint",
            str(checkpoint),
            "--expected-head-sha256",
            "expected-digest",
        ]
    )

    assert exit_code == 0
    assert captured["config"] is config
    assert captured["head_checkpoint_path"] == checkpoint
    assert captured["expected_head_sha256"] == "expected-digest"
    assert captured["run_dir"] == run_dir


def test_prediction_only_cli_requires_expected_head_digest(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(cli, "load_pipeline_config", lambda _path: object())
    with pytest.raises(ValueError, match="requires --expected-head-sha256"):
        cli.main(
            [
                "--config",
                str(tmp_path / "config.yaml"),
                "--line-manifest",
                str(tmp_path / "lines.csv"),
                "--tx1-cache-dir",
                str(tmp_path / "cache"),
                "--depmap-gene-effect",
                str(tmp_path / "gene_effect.csv"),
                "--run-dir",
                str(tmp_path / "diagnostic"),
                "--prediction-only-head-checkpoint",
                str(tmp_path / "head.pt"),
            ]
        )


# ---------------------------------------------------------------------------
# Both checked-in configs parse and pass shape preflight (D7/config discipline)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", _REAL_CONFIGS)
def test_real_configs_parse_and_pass_shape_preflight(name: str) -> None:
    """``load_pipeline_config`` + ``validate_config_shape`` must not raise.

    ``check_paths=False``: the referenced checkpoint/gene-vocabulary assets
    are gitignored/not-yet-produced (Phase C is a separate, still-running
    task), so this checks only the config's own field VALUES.
    """
    config = load_pipeline_config(_REAL_CONFIG_DIR / name)
    validate_config_shape(config, check_paths=False)  # must not raise


def test_real_validation_lines_json_satisfies_the_5_line_contract() -> None:
    """P2-2's dedicated pin on the actual committed contract file (D6/D12:
    the effective split is 28 train / 5 validation / 9 test). The runtime
    checks in ``load_validation_line_registration`` are deliberately generic
    (also exercised by tests using smaller synthetic registrations), so this
    is the place the real 5-line requirement is pinned: a hand-edit to the
    checked-in file that keeps ``validation_model_ids`` and
    ``validation_count`` self-consistent at some OTHER number would pass
    every runtime check but must still fail this dedicated regression test.
    """
    config = load_pipeline_config(_REAL_CONFIG_DIR / "tx1_arm.yaml")
    registration = json.loads(config.validation_lines_path.read_text())
    ids = registration["validation_model_ids"]
    assert registration["validation_count"] == 5
    assert len(ids) == 5
    assert len(set(ids)) == 5


def test_both_real_configs_are_byte_identical_except_the_encoder_view() -> None:
    """D7: scientific and response-generation settings match between arms."""
    tx1 = load_pipeline_config(_REAL_CONFIG_DIR / "tx1_arm.yaml")
    hvg = load_pipeline_config(_REAL_CONFIG_DIR / "hvg_arm.yaml")
    assert tx1.objective == hvg.objective
    assert tx1.training == hvg.training
    assert tx1.phase_f == hvg.phase_f
    assert tx1.validation_lines_path == hvg.validation_lines_path
    assert tx1.state.output_dim == hvg.state.output_dim == 2000
    assert tx1.state.cell_set_len == hvg.state.cell_set_len == 64
    assert (
        tx1.state.response_window_macro_batch_size
        == hvg.state.response_window_macro_batch_size
        == 64
    )
    assert tx1.state.response_generation_seed == hvg.state.response_generation_seed == 0
    assert tx1.state.input_dim == EMBEDDING_WIDTH
    assert hvg.state.input_dim != EMBEDDING_WIDTH


# ---------------------------------------------------------------------------
# validate_config_shape: named rejections
# ---------------------------------------------------------------------------


def _base_config(**overrides: object) -> Tx1GeneEffectHeadConfig:
    state = StateArmConfig(
        backend="linear_mock",
        hparams_checkpoint_path=None,
        st_checkpoint_path=None,
        known_perturbation_vectors=None,
        gene_vocabulary_path=None,
        input_dim=EMBEDDING_WIDTH,
        output_dim=4,
        pert_dim=3,
        output_space=None,
        cell_set_len=4,
        response_window_macro_batch_size=2,
        response_generation_seed=0,
    )
    objective = ObjectiveConfig(lam=1.0, selection_metric=SELECTION_METRIC_NAME)
    training = TrainingArmConfig(
        hidden=8, moments=2, learning_rate=0.01, epochs=2, seed=0
    )
    phase_f = PhaseFConfig(
        k_schedule=tuple(K_SCHEDULE), method="tx1_3b_st", alpha=1.0, residual=True
    )
    config = Tx1GeneEffectHeadConfig(
        arm="tx1_arm",
        run_id="unit",
        output_dir=Path("unused"),
        validation_lines_path=Path("unused"),
        state=state,
        objective=objective,
        training=training,
        phase_f=phase_f,
    )
    for key, value in overrides.items():
        if key.startswith("state."):
            config = _replace(config, state=_replace(config.state, **{key[6:]: value}))
        elif key.startswith("objective."):
            config = _replace(
                config, objective=_replace(config.objective, **{key[10:]: value})
            )
        elif key.startswith("training."):
            config = _replace(
                config, training=_replace(config.training, **{key[9:]: value})
            )
        elif key.startswith("phase_f."):
            config = _replace(
                config, phase_f=_replace(config.phase_f, **{key[8:]: value})
            )
        else:
            config = _replace(config, **{key: value})
    return config


def _write_prediction_only_provenance_fixture(
    tmp_path: Path, **overrides: object
) -> tuple[Tx1GeneEffectHeadConfig, Path, Path, Tx1GeneEffectHead]:
    st_checkpoint = tmp_path / "st.bin"
    depmap = tmp_path / "depmap.csv"
    st_checkpoint.write_bytes(b"st-checkpoint")
    depmap.write_bytes(b"depmap")
    head_checkpoint = tmp_path / "run" / "models" / "head.pt"
    head_checkpoint.parent.mkdir(parents=True)
    head_checkpoint.write_bytes(b"head")
    config = _base_config(**{"state.st_checkpoint_path": st_checkpoint})
    head = Tx1GeneEffectHead(
        response_dim=config.state.output_dim,
        basal_dim=config.state.input_dim,
        hidden=config.training.hidden,
        moments=config.training.moments,
    )
    provenance: dict[str, object] = {
        "arm": config.arm,
        "config": {
            "moments": config.training.moments,
            "hidden": config.training.hidden,
            "lam": config.objective.lam,
            "learning_rate": config.training.learning_rate,
            "epochs": config.training.epochs,
            "seed": config.training.seed,
        },
        "selection_metric": config.objective.selection_metric,
        "st_checkpoint_sha256": sha256_file(st_checkpoint),
        "depmap_gene_effect_sha256": sha256_file(depmap),
    }
    for key, value in overrides.items():
        if key.startswith("config."):
            provenance_config = provenance["config"]
            assert isinstance(provenance_config, dict)
            provenance_config[key[7:]] = value
        else:
            provenance[key] = value
    (head_checkpoint.parent.parent / "provenance.json").write_text(
        json.dumps(provenance)
    )
    return config, head_checkpoint, depmap, head


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("arm", "wrong_arm"),
        ("config.moments", 99),
        ("config.hidden", 99),
        ("config.lam", 99.0),
        ("config.learning_rate", 99.0),
        ("config.epochs", 99),
        ("config.seed", 99),
        ("selection_metric", "wrong"),
        ("st_checkpoint_sha256", "wrong"),
        ("depmap_gene_effect_sha256", "wrong"),
    ],
)
def test_prediction_only_rejects_phase_d_provenance_mismatch(
    tmp_path: Path, key: str, value: object
) -> None:
    config, checkpoint, depmap, head = _write_prediction_only_provenance_fixture(
        tmp_path, **{key: value}
    )
    with pytest.raises(ValueError, match="provenance mismatch"):
        load_and_verify_head_provenance(config, checkpoint, depmap, head)


def test_prediction_only_rejects_head_architecture_mismatch(tmp_path: Path) -> None:
    config, checkpoint, depmap, _head = _write_prediction_only_provenance_fixture(
        tmp_path
    )
    wrong_head = Tx1GeneEffectHead(
        response_dim=config.state.output_dim,
        basal_dim=config.state.input_dim,
        hidden=config.training.hidden + 1,
        moments=config.training.moments,
    )
    with pytest.raises(ValueError, match="head architecture"):
        load_and_verify_head_provenance(config, checkpoint, depmap, wrong_head)


def test_prediction_only_rejects_unpinned_head_bytes(tmp_path: Path) -> None:
    checkpoint = tmp_path / "head.pt"
    checkpoint.write_bytes(b"actual-head")
    with pytest.raises(ValueError, match="head SHA-256 mismatch"):
        verify_expected_head_sha256(checkpoint, "wrong-digest")


def _replace(obj: object, **kwargs: object) -> object:
    from dataclasses import replace

    return replace(obj, **kwargs)


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"arm": "bogus_arm"}, "arm must be one of"),
        ({"state.backend": "gpu_magic"}, "state.backend must be one of"),
        ({"state.input_dim": 0}, "state.input_dim must be positive"),
        (
            {"state.response_window_macro_batch_size": 0},
            "state.response_window_macro_batch_size must be positive",
        ),
        ({"state.input_dim": 999}, f"requires state.input_dim={EMBEDDING_WIDTH}"),
        ({"objective.lam": -1.0}, "objective.lam must be >= 0"),
        (
            {"objective.selection_metric": "val_c_loss"},
            "objective.selection_metric must equal",
        ),
        ({"training.moments": 1}, "training.moments must equal 2"),
        ({"training.epochs": 0}, "training.epochs must be positive"),
        ({"phase_f.k_schedule": (0, 5, 10)}, "phase_f.k_schedule must equal"),
        ({"phase_f.method": "copy_k562"}, "phase_f.method must equal"),
        ({"phase_f.alpha": 0.0}, "phase_f.alpha must be positive"),
    ],
)
def test_validate_config_shape_rejects_bad_configs(
    overrides: dict[str, object], match: str
) -> None:
    config = _base_config(**overrides)
    with pytest.raises(ValueError, match=match):
        validate_config_shape(config, check_paths=False)


def test_load_pipeline_config_requires_explicit_keys(tmp_path: Path) -> None:
    """A misspelled/omitted key raises at LOAD time, not silently defaulted
    (D10) -- this repo's own config loading is otherwise all ``.get(key,
    default)``, exactly how a typo goes unnoticed."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text("arm: tx1_arm\nrun:\n  run_id: x\n")
    with pytest.raises(ValueError, match="config section 'state' is missing key"):
        load_pipeline_config(config_path)


def test_load_pipeline_config_requires_run_output_dir(tmp_path: Path) -> None:
    """Omitting a key from a LATER section (``run.output_dir``, checked
    after every ``state``/``objective``/``training``/``phase_f`` key is
    already present) is named just as loudly as an early one."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
arm: tx1_arm
run:
  run_id: x
data:
  validation_lines_path: unused.json
state:
  backend: linear_mock
  input_dim: {EMBEDDING_WIDTH}
  output_dim: 4
  pert_dim: 3
  cell_set_len: 4
  response_window_macro_batch_size: 2
  response_generation_seed: 0
objective:
  lam: 1.0
  selection_metric: {SELECTION_METRIC_NAME}
training:
  hidden: 8
  moments: 2
  learning_rate: 0.05
  epochs: 3
  seed: 0
phase_f:
  k_schedule: {list(K_SCHEDULE)}
  method: tx1_3b_st
  alpha: 1.0
  residual: true
"""
    )
    with pytest.raises(ValueError, match="'run' is missing key 'output_dir'"):
        load_pipeline_config(config_path)


# ---------------------------------------------------------------------------
# Shared fixture: a tiny 6-line manifest, 51-gene slice, and cache
# ---------------------------------------------------------------------------


def _gene_symbols() -> list[str]:
    return [f"GENE{i:03d}" for i in range(_N_PLAIN_GENES)] + ["CRIPTO"]


def _resolved_vocabulary_genes() -> list[str]:
    """The ST vocabulary: CRIPTO is perturbed under its current HGNC symbol
    TDGF1 in every real anchor (the measured Task 0 alias); this fixture
    exercises that same alias-resolution path end to end."""
    return [f"GENE{i:03d}" for i in range(_N_PLAIN_GENES)] + ["TDGF1"]


def _depmap_columns() -> list[str]:
    return [f"{gene} ({1000 + i})" for i, gene in enumerate(_gene_symbols())]


def _write_manifest_fixture(tmp_path: Path) -> Path:
    rows = [
        _manifest_row(model_id=f"ACH-L{i}", lineage="Lung", role="train_head")
        for i in (1, 2, 3)
    ]
    rows.append(
        _manifest_row(
            model_id="ACH-A1",
            lineage="Blood",
            role="train_response_and_head",
            basal_source="Perturb-seq non-targeting control",
        )
    )
    rows.extend(
        _manifest_row(model_id=model_id, lineage="Skin", role="test")
        for model_id in _TEST_LINES
    )
    manifest_path = tmp_path / "manifest.csv"
    return _write_manifest(manifest_path, rows)


def _write_phase_a_fixture(
    tmp_path: Path, depmap_path: Path, manifest_path: Path
) -> Path:
    """Write a tiny ``phase_a_dir`` -- manifest/slice/panels + a registration
    recording every hash ``verify_artifact_hashes`` (P1-2) actually reads,
    not just the ``sources.depmap_gene_effect`` entry earlier fixtures here
    used alone. ``manifest_path`` is copied in (never symlinked) as
    ``cell_line_manifest.csv``, matching the real frozen layout where the
    line manifest lives directly under ``phase_a_dir``.
    """
    phase_a_dir = tmp_path / "phase_a"
    phase_a_dir.mkdir()
    manifest_copy_path = phase_a_dir / "cell_line_manifest.csv"
    manifest_copy_path.write_bytes(Path(manifest_path).read_bytes())

    slice_path = phase_a_dir / "differentially_essential_slice.csv"
    slice_df = pd.DataFrame(
        {"depmap_column": _depmap_columns(), "gene_symbol": _gene_symbols()}
    )
    slice_df.to_csv(slice_path, index=False)

    rng = np.random.default_rng(7)
    panel_rows = []
    columns = _depmap_columns()
    for model_id in _TEST_LINES:
        for panel in range(_N_PANELS):
            order = rng.permutation(len(columns))[:_N_LABELS]
            panel_seed = int(rng.integers(0, 2**31))
            for label_order, gene_idx in enumerate(order, start=1):
                panel_rows.append(
                    {
                        "model_id": model_id,
                        "panel": panel,
                        "panel_seed": panel_seed,
                        "label_order": label_order,
                        "depmap_column": columns[gene_idx],
                    }
                )
    panels_path = phase_a_dir / "k_label_panels.csv"
    pd.DataFrame(panel_rows).to_csv(panels_path, index=False)

    registration_path = phase_a_dir / "phase_a_registration.json"
    registration_path.write_text(
        json.dumps(
            {
                "sources": {
                    "depmap_gene_effect": {
                        "path": "unused",
                        "sha256": sha256_file(depmap_path),
                    }
                },
                "artifacts": {
                    "cell_line_manifest_sha256": sha256_file(manifest_copy_path),
                    "differential_slice_sha256": sha256_file(slice_path),
                    "k_label_panels_sha256": sha256_file(panels_path),
                },
            }
        )
    )
    return phase_a_dir


def _write_depmap_fixture(tmp_path: Path, model_ids: list[str]) -> Path:
    rng = np.random.default_rng(11)
    columns = _depmap_columns()
    frame = pd.DataFrame(
        rng.uniform(-2.0, 2.0, size=(len(model_ids), len(columns))),
        index=pd.Index(model_ids, name="model_id"),
        columns=columns,
    )
    depmap_path = tmp_path / "CRISPRGeneEffect.csv"
    frame.to_csv(depmap_path)
    return depmap_path


def _write_tx1_cache_fixture(
    tmp_path: Path, model_ids: list[str], *, input_dim: int, hvg_width: int = 3
) -> Path:
    cache_dir = tmp_path / "tx1_cache"
    line_arrays = {}
    for index, model_id in enumerate(model_ids):
        n_cells = 5 + index
        rng = np.random.default_rng(100 + index)
        embeddings = rng.normal(size=(n_cells, input_dim)).astype(np.float32)
        hvg = rng.normal(size=(n_cells, hvg_width)).astype(np.float32)
        obs = pd.DataFrame(index=[f"{model_id}-c{i}" for i in range(n_cells)])
        line_arrays[model_id] = write_line_cache(
            cache_dir,
            model_id,
            embeddings,
            hvg,
            obs,
            hvg_gene_order=[f"H{i}" for i in range(hvg_width)],
        )
    _write_cache_run_manifest(
        cache_dir, line_arrays, [f"H{i}" for i in range(hvg_width)]
    )
    return cache_dir


def _write_fake_st_checkpoint(
    path: Path, *, input_dim: int, vocabulary: list[str]
) -> None:
    """A flat state dict shaped like Phase C's real ``AivcModel.state_dict()``
    (the model's own real keys plus fake dropped-head keys) -- mirrors
    ``test_tx1_predicted_response.py``'s ``_fake_full_checkpoint`` helper."""
    state_model = LinearMockStateModel(input_dim, _RESPONSE_DIM, _PERT_DIM)
    perturbations = PerturbationVectorAdapter(vocabulary, {}, _PERT_DIM)
    source = ForwardOnlyStateModel(StateForwardAdapter(state_model), perturbations)
    checkpoint = dict(source.state_dict())
    checkpoint["response_encoder.linear.weight"] = torch.zeros(2, 2)
    checkpoint["response_pooler.means"] = torch.zeros(2, 2)
    checkpoint["c_head.net.0.weight"] = torch.zeros(2, 2)
    checkpoint["control_expression_mean"] = torch.zeros(_RESPONSE_DIM)
    torch.save(checkpoint, path)


def _write_config_yaml(
    tmp_path: Path,
    *,
    input_dim: int,
    st_checkpoint_path: Path,
    gene_vocabulary_path: Path,
    validation_lines_path: Path,
    output_dir: Path,
    arm: str = "tx1_arm",
    run_id: str = "smoke",
) -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
arm: {arm}
run:
  run_id: {run_id}
  output_dir: {output_dir}
data:
  validation_lines_path: {validation_lines_path}
state:
  backend: linear_mock
  hparams_checkpoint_path: null
  st_checkpoint_path: {st_checkpoint_path}
  known_perturbation_vectors: null
  gene_vocabulary_path: {gene_vocabulary_path}
  input_dim: {input_dim}
  output_dim: {_RESPONSE_DIM}
  pert_dim: {_PERT_DIM}
  output_space: null
  cell_set_len: {_CELL_SET_LEN}
  response_window_macro_batch_size: 2
  response_generation_seed: 0
objective:
  lam: 1.0
  selection_metric: {SELECTION_METRIC_NAME}
training:
  hidden: 8
  moments: 2
  learning_rate: 0.05
  epochs: 3
  seed: 0
phase_f:
  k_schedule: {list(K_SCHEDULE)}
  method: tx1_3b_st
  alpha: 1.0
  residual: true
"""
    )
    return config_path


def _full_fixture(tmp_path: Path) -> dict[str, Path]:
    manifest_path = _write_manifest_fixture(tmp_path)
    manifest = load_line_manifest(manifest_path)
    all_ids = manifest["model_id"].astype(str).tolist()

    registration = generate_validation_line_registration(
        manifest, manifest_path, validation_count=1, seed=0
    )
    validation_lines_path = tmp_path / "validation_lines.json"
    validation_lines_path.write_text(json.dumps(registration, sort_keys=True))

    depmap_path = _write_depmap_fixture(tmp_path, all_ids)
    phase_a_dir = _write_phase_a_fixture(tmp_path, depmap_path, manifest_path)
    tx1_cache_dir = _write_tx1_cache_fixture(
        tmp_path, all_ids, input_dim=EMBEDDING_WIDTH
    )

    vocabulary = _resolved_vocabulary_genes()
    gene_vocabulary_path = tmp_path / "gene_vocabulary.json"
    gene_vocabulary_path.write_text(json.dumps(vocabulary))

    st_checkpoint_path = tmp_path / "pytorch_model.bin"
    _write_fake_st_checkpoint(
        st_checkpoint_path, input_dim=EMBEDDING_WIDTH, vocabulary=vocabulary
    )

    output_dir = tmp_path / "phase_d_out"
    config_path = _write_config_yaml(
        tmp_path,
        input_dim=EMBEDDING_WIDTH,
        st_checkpoint_path=st_checkpoint_path,
        gene_vocabulary_path=gene_vocabulary_path,
        validation_lines_path=validation_lines_path,
        output_dir=output_dir,
    )
    return {
        "manifest_path": manifest_path,
        "phase_a_dir": phase_a_dir,
        "tx1_cache_dir": tx1_cache_dir,
        "depmap_path": depmap_path,
        "config_path": config_path,
        "output_dir": output_dir,
    }


def _argv(fixture: dict[str, Path], predicted_cache_dir: Path) -> list[str]:
    del predicted_cache_dir
    return [
        "--config",
        str(fixture["config_path"]),
        "--line-manifest",
        str(fixture["manifest_path"]),
        "--phase-a-dir",
        str(fixture["phase_a_dir"]),
        "--tx1-cache-dir",
        str(fixture["tx1_cache_dir"]),
        "--depmap-gene-effect",
        str(fixture["depmap_path"]),
    ]


def _install_lenient_hash_check(monkeypatch: pytest.MonkeyPatch) -> None:
    """Decouple these real-run tests from the real frozen-registration anchor.

    Mirrors ``test_evaluate_tx1_backbone.py``'s helper of the same name and
    for the same reason: ``verify_artifact_hashes`` (now wired into
    ``run_training_pipeline`` for Wave 3 Codex gate P1-2) pins its
    registration-IDENTITY check to ``FROZEN_REGISTRATION_SHA256`` -- the
    sha256 of the real, committed ``results/phase_a_tx1_20260724/
    phase_a_registration.json``. These tests build tiny synthetic
    ``phase_a_registration.json`` fixtures instead, so this patches
    ``tx1_geneeffect_pipeline_run.verify_artifact_hashes`` to trust whatever
    registration file is actually on disk in the fixture at call time -- the
    per-ARTIFACT hash comparisons (what P1-2's tests below actually
    exercise) still run for real, unweakened. The real frozen anchor is
    tested directly, against the real committed file, in
    ``test_tx1_geneeffect_eval.py``.
    """

    def _lenient_verify(phase_a_dir: Path) -> None:
        registration_path = Path(phase_a_dir) / "phase_a_registration.json"
        expected = sha256_file(registration_path)
        _real_verify_artifact_hashes(phase_a_dir, expected_registration_sha256=expected)

    monkeypatch.setattr(
        tx1_geneeffect_pipeline_run_module, "verify_artifact_hashes", _lenient_verify
    )


def _write_state_checkpoint_fixture(tmp_path: Path) -> dict[str, Path]:
    """A ``backend=state_checkpoint`` variant of :func:`_full_fixture`, for
    Wave 3 Codex gate P1-3 (``verify_gene_vocabulary_authenticity`` is a
    no-op for ``_full_fixture``'s ``linear_mock`` backend).

    ``st_checkpoint_path`` is a dummy file -- never loaded by these tests,
    since they exercise only the authentication step, which
    ``run_training_pipeline``/``dry_run_summary`` both run strictly BEFORE
    ``build_forward_only_model`` would ever read it (the real ``state``
    package IS importable on this machine, but constructing a real
    Tx1-3B-shaped ST model just to prove a preceding guard fired first is
    unnecessary weight for a unit test).
    """
    manifest_path = _write_manifest_fixture(tmp_path)
    manifest = load_line_manifest(manifest_path)
    all_ids = manifest["model_id"].astype(str).tolist()

    registration = generate_validation_line_registration(
        manifest, manifest_path, validation_count=1, seed=0
    )
    validation_lines_path = tmp_path / "validation_lines.json"
    validation_lines_path.write_text(json.dumps(registration, sort_keys=True))

    depmap_path = _write_depmap_fixture(tmp_path, all_ids)
    phase_a_dir = _write_phase_a_fixture(tmp_path, depmap_path, manifest_path)
    tx1_cache_dir = _write_tx1_cache_fixture(
        tmp_path, all_ids, input_dim=EMBEDDING_WIDTH
    )

    vocabulary = _resolved_vocabulary_genes()
    gene_vocabulary_path = tmp_path / "gene_vocabulary.json"
    gene_vocabulary_path.write_text(json.dumps(vocabulary))

    checkpoint_dir = tmp_path / "phase_c_checkpoint"
    checkpoint_dir.mkdir()
    st_checkpoint_path = checkpoint_dir / "pytorch_model.bin"
    st_checkpoint_path.write_bytes(b"dummy-checkpoint-bytes-never-loaded")
    (checkpoint_dir / "metadata.json").write_text(
        json.dumps({"gene_vocabulary_sha256": sha256_file(gene_vocabulary_path)})
    )

    hparams_path = tmp_path / "hparams.ckpt"
    hparams_path.write_bytes(b"unused-in-these-tests")

    output_dir = tmp_path / "phase_d_out_state_checkpoint"
    config_path = tmp_path / "config_state_checkpoint.yaml"
    config_path.write_text(
        f"""
arm: tx1_arm
run:
  run_id: state_checkpoint_smoke
  output_dir: {output_dir}
data:
  validation_lines_path: {validation_lines_path}
state:
  backend: state_checkpoint
  hparams_checkpoint_path: {hparams_path}
  st_checkpoint_path: {st_checkpoint_path}
  known_perturbation_vectors: null
  gene_vocabulary_path: {gene_vocabulary_path}
  input_dim: {EMBEDDING_WIDTH}
  output_dim: {_RESPONSE_DIM}
  pert_dim: {_PERT_DIM}
  output_space: null
  cell_set_len: {_CELL_SET_LEN}
  response_window_macro_batch_size: 2
  response_generation_seed: 0
objective:
  lam: 1.0
  selection_metric: {SELECTION_METRIC_NAME}
training:
  hidden: 8
  moments: 2
  learning_rate: 0.05
  epochs: 3
  seed: 0
phase_f:
  k_schedule: {list(K_SCHEDULE)}
  method: tx1_3b_st
  alpha: 1.0
  residual: true
"""
    )
    return {
        "manifest_path": manifest_path,
        "phase_a_dir": phase_a_dir,
        "tx1_cache_dir": tx1_cache_dir,
        "depmap_path": depmap_path,
        "config_path": config_path,
        "output_dir": output_dir,
        "gene_vocabulary_path": gene_vocabulary_path,
        "st_checkpoint_path": st_checkpoint_path,
    }


# ---------------------------------------------------------------------------
# --dry-run: validates without materializing cell matrices
# ---------------------------------------------------------------------------


def test_dry_run_end_to_end_against_tiny_fixtures(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fixture = _full_fixture(tmp_path)
    argv = _argv(fixture, tmp_path / "unused_predicted_cache") + ["--dry-run"]
    exit_code = cli.main(argv)
    assert exit_code == 0

    summary = json.loads(capsys.readouterr().out)
    assert summary["arm"] == "tx1_arm"
    assert summary["n_train_lines"] == 3  # 3 train_head + 1 anchor - 1 validation
    assert summary["n_validation_lines"] == 1
    assert summary["n_test_lines"] == 2
    assert summary["n_anchor_lines"] == 1
    assert summary["n_slice_genes"] == _N_PLAIN_GENES + 1
    assert summary["n_resolved_genes"] == _N_PLAIN_GENES + 1


def test_dry_run_never_materializes_a_cell_matrix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The dry-run's core assertion: it must not read a single cell's
    expression values. Both the basal-cache reader and ST-forward generation
    are monkeypatched to raise; a real (non-dry-run) run reaches both (see
    ``test_e2e_smoke_...`` below, unaffected by this monkeypatch)."""
    import aivc_model.tx1_embed_cache as tx1_embed_cache_module
    import aivc_model.tx1_predicted_response as tx1_predicted_response_module

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("must not run during --dry-run")

    monkeypatch.setattr(tx1_embed_cache_module, "load_line_cache", _boom)
    monkeypatch.setattr(
        tx1_predicted_response_module, "generate_predicted_response_for_line", _boom
    )
    monkeypatch.setattr(
        tx1_predicted_response_module, "generate_predicted_response", _boom
    )

    fixture = _full_fixture(tmp_path)
    argv = _argv(fixture, tmp_path / "unused_predicted_cache") + ["--dry-run"]
    exit_code = cli.main(argv)  # must not raise
    assert exit_code == 0


def test_dry_run_rejects_stale_validation_registration_without_reading_cells(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A validation registration derived from a DIFFERENT manifest must fail
    --dry-run loudly (D6), before any cell/response read."""
    import aivc_model.tx1_embed_cache as tx1_embed_cache_module

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("must not run before the role check fails")

    monkeypatch.setattr(tx1_embed_cache_module, "load_line_cache", _boom)

    fixture = _full_fixture(tmp_path)
    registration = json.loads((tmp_path / "validation_lines.json").read_text())
    registration["derived_from_sha256"] = "0" * 64
    (tmp_path / "validation_lines.json").write_text(json.dumps(registration))

    argv = _argv(fixture, tmp_path / "unused_predicted_cache") + ["--dry-run"]
    with pytest.raises(ValueError, match="is stale"):
        cli.main(argv)


def test_dry_run_rejects_a_bad_path(tmp_path: Path) -> None:
    fixture = _full_fixture(tmp_path)
    argv = _argv(fixture, tmp_path / "unused_predicted_cache") + ["--dry-run"]
    bad_depmap_index = argv.index("--depmap-gene-effect") + 1
    argv[bad_depmap_index] = str(tmp_path / "does_not_exist.csv")
    with pytest.raises(FileNotFoundError):
        cli.main(argv)


def test_validate_widths_rejects_hvg_width_mismatch(tmp_path: Path) -> None:
    """validate_widths (not validate_config_shape's EMBEDDING_WIDTH-only
    check) must independently catch an hvg_arm width disagreeing with what
    Phase B's cache manifest actually recorded."""
    tx1_cache_dir = _write_tx1_cache_fixture(
        tmp_path, ["ACH-ONLY"], input_dim=EMBEDDING_WIDTH, hvg_width=3
    )
    config = _base_config(arm="hvg_arm", **{"state.input_dim": 999})
    with pytest.raises(ValueError, match="disagrees with the Phase B cache"):
        validate_widths(config, tx1_cache_dir)


# ---------------------------------------------------------------------------
# Wave 3 Codex gate fix round 1
# ---------------------------------------------------------------------------


# --- P1-1: the Phase B cache must be verified before any read ---------------


def test_e2e_smoke_rejects_a_corrupted_phase_b_cache_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``validate_widths`` (the guard ``run_training_pipeline`` used to rely
    on alone) only cross-checks the FIRST manifest entry's recorded shape --
    corrupting a recorded array HASH (which ``validate_widths`` never reads
    at all, for any entry) is invisible to it regardless of position.
    ``verify_cache``, now run unconditionally before any cache read, must
    catch it."""
    _install_lenient_hash_check(monkeypatch)
    fixture = _full_fixture(tmp_path)
    cache_manifest_path = fixture["tx1_cache_dir"] / "manifest.json"
    cache_manifest = json.loads(cache_manifest_path.read_text())
    # ACH-T2 is not the first line `write_line_cache` recorded (see
    # `_write_tx1_cache_fixture`'s iteration order over `all_ids`).
    cache_manifest["lines"]["ACH-T2"]["arrays"]["embeddings.npy"]["sha256"] = "0" * 64
    cache_manifest_path.write_text(json.dumps(cache_manifest))

    argv = _argv(fixture, tmp_path / "predicted_response_cache")
    with pytest.raises(ValueError, match="failed.*verification"):
        cli.main(argv)


# --- P1-2: the frozen Phase-A registration/slice/panels must be verified ----


def test_e2e_smoke_rejects_a_tampered_phase_a_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The frozen Phase-A registration/slice/panels were previously read
    completely unauthenticated in ``run_training_pipeline`` -- a slice file
    modified after the registration was written (so its recorded hash no
    longer matches) must now be rejected before it is ever used to select
    training genes."""
    _install_lenient_hash_check(monkeypatch)
    fixture = _full_fixture(tmp_path)
    slice_path = fixture["phase_a_dir"] / "differentially_essential_slice.csv"
    tampered = pd.read_csv(slice_path)
    tampered.loc[0, "gene_symbol"] = "TAMPERED_GENE"
    tampered.to_csv(slice_path, index=False)  # registration's recorded hash now stale

    argv = _argv(fixture, tmp_path / "predicted_response_cache")
    with pytest.raises(EvaluationContractError, match="SHA-256 mismatch"):
        cli.main(argv)


# --- P1-3: state.gene_vocabulary_path must be authenticated -----------------


def test_verify_gene_vocabulary_authenticity_is_a_noop_for_linear_mock(
    tmp_path: Path,
) -> None:
    verify_gene_vocabulary_authenticity(
        tmp_path / "does-not-exist.json", None, "linear_mock"
    )  # must not raise


def test_verify_gene_vocabulary_authenticity_requires_checkpoint_path(
    tmp_path: Path,
) -> None:
    vocab_path = tmp_path / "gene_vocabulary.json"
    vocab_path.write_text(json.dumps(["G1", "G2"]))

    with pytest.raises(ValueError, match="st_checkpoint_path"):
        verify_gene_vocabulary_authenticity(vocab_path, None, "state_checkpoint")


def _write_checkpoint_dir(
    tmp_path: Path, *, vocabulary: list[str], metadata: dict[str, object] | None
) -> tuple[Path, Path]:
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    checkpoint_path = checkpoint_dir / "pytorch_model.bin"
    checkpoint_path.write_bytes(b"unused")
    vocab_path = tmp_path / "gene_vocabulary.json"
    vocab_path.write_text(json.dumps(vocabulary))
    if metadata is not None:
        (checkpoint_dir / "metadata.json").write_text(json.dumps(metadata))
    return checkpoint_path, vocab_path


def test_verify_gene_vocabulary_authenticity_requires_sibling_metadata(
    tmp_path: Path,
) -> None:
    checkpoint_path, vocab_path = _write_checkpoint_dir(
        tmp_path, vocabulary=["G1", "G2"], metadata=None
    )

    with pytest.raises(ValueError, match="metadata.json"):
        verify_gene_vocabulary_authenticity(
            vocab_path, checkpoint_path, "state_checkpoint"
        )


def test_verify_gene_vocabulary_authenticity_requires_recorded_hash(
    tmp_path: Path,
) -> None:
    checkpoint_path, vocab_path = _write_checkpoint_dir(
        tmp_path, vocabulary=["G1", "G2"], metadata={"other_key": "value"}
    )

    with pytest.raises(ValueError, match="gene_vocabulary_sha256"):
        verify_gene_vocabulary_authenticity(
            vocab_path, checkpoint_path, "state_checkpoint"
        )


def test_verify_gene_vocabulary_authenticity_rejects_reordered_vocabulary(
    tmp_path: Path,
) -> None:
    """The exact P1-3 hazard: the SAME genes in a DIFFERENT construction
    order -- gene coverage passes, but PerturbationVectorAdapter binds
    every positional missing-gene weight to the wrong symbol."""
    checkpoint_path, vocab_path = _write_checkpoint_dir(
        tmp_path, vocabulary=["G1", "G2", "G3"], metadata=None
    )
    (checkpoint_path.parent / "metadata.json").write_text(
        json.dumps({"gene_vocabulary_sha256": sha256_file(vocab_path)})
    )
    vocab_path.write_text(json.dumps(["G3", "G1", "G2"]))  # reordered post-hoc

    with pytest.raises(ValueError, match="gene_vocabulary_sha256"):
        verify_gene_vocabulary_authenticity(
            vocab_path, checkpoint_path, "state_checkpoint"
        )


def test_verify_gene_vocabulary_authenticity_accepts_matching_hash(
    tmp_path: Path,
) -> None:
    checkpoint_path, vocab_path = _write_checkpoint_dir(
        tmp_path, vocabulary=["G1", "G2", "G3"], metadata=None
    )
    (checkpoint_path.parent / "metadata.json").write_text(
        json.dumps({"gene_vocabulary_sha256": sha256_file(vocab_path)})
    )

    verify_gene_vocabulary_authenticity(
        vocab_path, checkpoint_path, "state_checkpoint"
    )  # must not raise


def test_dry_run_rejects_reordered_gene_vocabulary(tmp_path: Path) -> None:
    """P1-3, at the wiring level: ``--dry-run`` must call
    ``verify_gene_vocabulary_authenticity`` and reject a vocabulary
    reordered after Phase C's checkpoint recorded its hash --
    ``validate_gene_coverage`` alone (order-blind) would pass this fixture."""
    fixture = _write_state_checkpoint_fixture(tmp_path)
    vocabulary = json.loads(fixture["gene_vocabulary_path"].read_text())
    fixture["gene_vocabulary_path"].write_text(json.dumps(list(reversed(vocabulary))))

    argv = _argv(fixture, tmp_path / "unused_predicted_cache") + ["--dry-run"]
    with pytest.raises(ValueError, match="gene_vocabulary_sha256"):
        cli.main(argv)


def test_dry_run_rejects_checkpoint_missing_vocabulary_metadata(
    tmp_path: Path,
) -> None:
    """A checkpoint that predates Phase C's gene_vocabulary.json/
    metadata.json emission fix must fail loudly and actionably (a live
    Phase C run on HPC has not produced this file under the fixed pipeline
    yet), not be silently trusted."""
    fixture = _write_state_checkpoint_fixture(tmp_path)
    (fixture["st_checkpoint_path"].parent / "metadata.json").unlink()

    argv = _argv(fixture, tmp_path / "unused_predicted_cache") + ["--dry-run"]
    with pytest.raises(ValueError, match="metadata.json"):
        cli.main(argv)


def test_dry_run_accepts_a_matching_state_checkpoint_vocabulary(
    tmp_path: Path,
) -> None:
    """Sanity check: an untouched, hash-matching vocabulary/checkpoint pair
    must not be falsely rejected by P1-3's new authentication step."""
    fixture = _write_state_checkpoint_fixture(tmp_path)

    argv = _argv(fixture, tmp_path / "unused_predicted_cache") + ["--dry-run"]
    exit_code = cli.main(argv)  # must not raise

    assert exit_code == 0


def test_config_device_defaults_to_auto_and_resolves(tmp_path: Path) -> None:
    config = _base_config()
    assert config.state.device == "auto"
    resolved = resolve_device(config.state.device)
    assert resolved == torch.device("cpu") or resolved == torch.device("cuda")


def test_config_device_can_be_pinned_explicitly(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
arm: tx1_arm
run:
  run_id: x
  output_dir: {tmp_path / "out"}
data:
  validation_lines_path: {tmp_path / "unused.json"}
state:
  backend: linear_mock
  input_dim: {EMBEDDING_WIDTH}
  output_dim: 4
  pert_dim: 3
  cell_set_len: 4
  response_window_macro_batch_size: 2
  response_generation_seed: 0
  device: cpu
objective:
  lam: 1.0
  selection_metric: {SELECTION_METRIC_NAME}
training:
  hidden: 8
  moments: 2
  learning_rate: 0.05
  epochs: 3
  seed: 0
phase_f:
  k_schedule: {list(K_SCHEDULE)}
  method: tx1_3b_st
  alpha: 1.0
  residual: true
"""
    )
    config = load_pipeline_config(config_path)
    assert config.state.device == "cpu"
    assert resolve_device(config.state.device) == torch.device("cpu")


# --- P2-3: a rerun requires a fresh run directory ---------------------------


def test_run_training_pipeline_rejects_existing_run_dir_before_any_validation(
    tmp_path: Path,
) -> None:
    """The fresh-run-dir check runs FIRST, before config/path validation --
    proven here by pointing every other argument at nonexistent paths and
    still getting ``FileExistsError``, not some later validation error. A
    rerun with a fixed run_id must never overwrite a checkpoint in place
    while leaving another prior run's output (e.g. a stale
    tx1_3b_st_predictions.csv) untouched beside it."""
    run_dir = tmp_path / "runs" / "existing"
    run_dir.mkdir(parents=True)
    config = _base_config()

    with pytest.raises(FileExistsError, match="already exists"):
        run_training_pipeline(
            config,
            line_manifest_path=tmp_path / "does-not-exist.csv",
            phase_a_dir=tmp_path / "does-not-exist-dir",
            tx1_cache_dir=tmp_path / "does-not-exist-cache",
            depmap_gene_effect_path=tmp_path / "does-not-exist.csv",
            run_dir=run_dir,
        )
