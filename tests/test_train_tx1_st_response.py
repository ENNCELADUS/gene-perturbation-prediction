"""Tests for scripts/train_tx1_st_response.py (Wave 2 Phase C, Task 7).

Covers: both checked-in Phase C configs parse and pass config-shape
preflight (item 6 of the brief), the per-arm view projection, a full
``--dry-run`` exercised against tiny local fixtures (no real Tx1/HPC
assets), bad-config rejection, a characterization test documenting
``GeneBags.for_genes``'s general name-keyed-dict behavior, and (fix round 1)
the base-gene-grouped split's leakage guarantee and the widened source
fingerprint.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

import scripts.train_tx1_st_response as cli
from aivc_model import train as train_module
from aivc_model.gene_splits import FoldSpec, GeneAccessRecorder
from aivc_model.prepare import GeneBags, SplitConfig, load_config
from aivc_model.tx1_embed_cache import EMBEDDING_WIDTH, write_line_cache
from aivc_model.tx1_response_data import base_gene_name, composite_gene_key
from conftest import tx1_manifest_row as _manifest_row
from conftest import write_tx1_cache_run_manifest as _write_cache_run_manifest
from conftest import write_tx1_line_manifest as _write_manifest
from conftest import write_tx1_perturbseq_h5ad_ensembl_index as _write_perturbseq_h5ad

_REAL_CONFIG_DIR = Path("configs/experiments/12_tx1_st_geneeffect/phase_c")
_REAL_CONFIGS = ("hvg_arm.yaml", "tx1_arm.yaml")


# --- item 6: both checked-in configs parse and pass shape preflight --------


@pytest.mark.parametrize("name", _REAL_CONFIGS)
def test_real_configs_parse_and_pass_shape_preflight(name: str) -> None:
    """``load_config`` + ``validate_config_shape`` must not raise for either arm.

    ``check_paths=False``: the referenced checkpoint assets
    (``model/checkpoints/state/...``) are gitignored and not present in a
    fresh checkout/CI, so this checks only the config's own field values,
    not filesystem state.
    """
    config = load_config(_REAL_CONFIG_DIR / name)
    cli.validate_config_shape(config, check_paths=False)  # must not raise


def test_hvg_arm_config_values() -> None:
    config = load_config(_REAL_CONFIG_DIR / "hvg_arm.yaml")
    assert config.state.input_view == "checkpoint_hvg"
    assert config.state.input_dim == 2000
    assert config.state.warm_start_from is None, (
        "HVG arm's input already matches the checkpoint's own width; no "
        "fresh basal_encoder.0 (and thus no warm_start_from) is needed"
    )


def test_tx1_arm_config_values() -> None:
    config = load_config(_REAL_CONFIG_DIR / "tx1_arm.yaml")
    assert config.state.input_view == "obsm"
    assert config.state.input_dim == 2560
    assert config.data.state_embed_key
    assert config.state.warm_start_from == config.state.checkpoint_path, (
        "Tx1 arm needs Task 4's shape-filtered warm start (fresh "
        "basal_encoder.0), triggered by warm_start_from being set"
    )


@pytest.mark.parametrize("name", _REAL_CONFIGS)
def test_both_arms_share_output_and_response_encoder_width(name: str) -> None:
    config = load_config(_REAL_CONFIG_DIR / name)
    assert config.state.output_dim == 2000, "ST output is gene space in both arms (C9)"
    assert config.response_encoder is not None
    assert config.response_encoder.input_dim == 2000, (
        "response encoder is target-space, stays 2000 even for the Tx1 "
        "arm's 2560-d ST input"
    )
    assert config.response_encoder.latent_dim == 128
    assert config.state.output_space == "gene"


@pytest.mark.parametrize("name", _REAL_CONFIGS)
def test_both_arms_record_l2_normalize_explicitly(name: str) -> None:
    """C7: the flag must appear explicitly, at its measured-correct default."""
    config = load_config(_REAL_CONFIG_DIR / name)
    assert config.state.l2_normalize_input is False


@pytest.mark.parametrize("name", _REAL_CONFIGS)
def test_both_arms_zero_the_nan_y_dependent_loss_weights(name: str) -> None:
    """Phase C's GeneBags carry no GeneEffect label; these MUST be 0.0.

    ``F.mse_loss(*, NaN)`` is NaN regardless of its weight
    (``0.0 * NaN == NaN``), so a nonzero default here would poison
    ``total_loss`` even though ``model.py``'s ``_weighted_loss_sum`` skips
    exactly-zero-weighted terms.
    """
    config = load_config(_REAL_CONFIG_DIR / name)
    assert config.loss.pred_c_weight == 0.0
    assert config.loss.obs_c_weight == 0.0
    assert config.loss.occupancy_weight == 0.0
    assert config.loss.gmm_nll_weight == 0.0
    assert config.loss.pred_rank_weight == 0.0


@pytest.mark.parametrize("name", _REAL_CONFIGS)
def test_both_arms_use_gene_batch_size_one(name: str) -> None:
    """gene_batch_size > 1 reintroduces an unguarded NaN-y hazard in pred_rank."""
    config = load_config(_REAL_CONFIG_DIR / name)
    assert config.train.gene_batch_size == 1


@pytest.mark.parametrize("name", _REAL_CONFIGS)
def test_both_arms_draw_lines_from_the_frozen_manifest_role(name: str) -> None:
    """C4/C5: no hardcoded line list anywhere in the config files' text."""
    text = (_REAL_CONFIG_DIR / name).read_text(encoding="utf-8")
    for hardcoded in ("ACH-000551", "ACH-000971", "ACH-000995", "ACH-000739"):
        assert hardcoded not in text


# --- validate_config_shape: bad-config rejection ---------------------------


def _minimal_config_yaml(tmp_path: Path, **overrides: str) -> Path:
    values = {
        "response_encoder_input_dim": "2000",
        "response_encoder_latent_dim": "128",
        "gmm_trainable": "true",
        "state_output_dim": "2000",
        "state_input_view": "checkpoint_hvg",
        "state_input_dim": "2000",
        "state_embed_key": "null",
        "pred_c_weight": "0.0",
        "obs_c_weight": "0.0",
        "occupancy_weight": "0.0",
        "gmm_nll_weight": "0.0",
        "pred_rank_weight": "0.0",
        "gene_batch_size": "1",
    }
    values.update(overrides)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
data:
  h5ad_path: UNUSED
  overlap_csv: UNUSED
  output_dir: {tmp_path / "outputs"}
  state_embed_key: {values["state_embed_key"]}
state:
  backend: state_checkpoint
  input_dim: {values["state_input_dim"]}
  output_dim: {values["state_output_dim"]}
  input_view: {values["state_input_view"]}
response_encoder:
  input_dim: {values["response_encoder_input_dim"]}
  latent_dim: {values["response_encoder_latent_dim"]}
gmm:
  trainable: {values["gmm_trainable"]}
loss:
  pred_c_weight: {values["pred_c_weight"]}
  obs_c_weight: {values["obs_c_weight"]}
  occupancy_weight: {values["occupancy_weight"]}
  gmm_nll_weight: {values["gmm_nll_weight"]}
  pred_rank_weight: {values["pred_rank_weight"]}
train:
  gene_batch_size: {values["gene_batch_size"]}
""",
        encoding="utf-8",
    )
    return config_path


def test_validate_config_shape_accepts_a_well_formed_config(tmp_path: Path) -> None:
    config = load_config(_minimal_config_yaml(tmp_path))
    cli.validate_config_shape(config, check_paths=False)  # must not raise


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"response_encoder_input_dim": "1999"}, "response_encoder.input_dim"),
        ({"response_encoder_latent_dim": "64"}, "latent_dim"),
        ({"gmm_trainable": "false"}, "gmm.trainable"),
        ({"state_output_dim": "2560"}, "state.output_dim"),
        ({"pred_c_weight": "1.0"}, "loss.pred_c_weight"),
        ({"obs_c_weight": "0.25"}, "loss.obs_c_weight"),
        ({"gene_batch_size": "4"}, "gene_batch_size"),
    ],
)
def test_validate_config_shape_rejects_bad_configs(
    tmp_path: Path, override: dict[str, str], match: str
) -> None:
    config = load_config(_minimal_config_yaml(tmp_path, **override))
    with pytest.raises(ValueError, match=match):
        cli.validate_config_shape(config, check_paths=False)


def test_validate_config_shape_rejects_obsm_without_input_dim_2560(
    tmp_path: Path,
) -> None:
    config = load_config(
        _minimal_config_yaml(
            tmp_path,
            state_input_view="obsm",
            state_input_dim="2000",
            state_embed_key="X_tx1_3b",
        )
    )
    with pytest.raises(ValueError, match="2560"):
        cli.validate_config_shape(config, check_paths=False)


def test_validate_config_shape_rejects_obsm_without_state_embed_key(
    tmp_path: Path,
) -> None:
    config = load_config(
        _minimal_config_yaml(
            tmp_path,
            state_input_view="obsm",
            state_input_dim="2560",
            state_embed_key="null",
        )
    )
    with pytest.raises(ValueError, match="state_embed_key"):
        cli.validate_config_shape(config, check_paths=False)


def test_validate_config_shape_check_paths_rejects_missing_checkpoint(
    tmp_path: Path,
) -> None:
    config_path = _minimal_config_yaml(tmp_path)
    text = config_path.read_text(encoding="utf-8").replace(
        "backend: state_checkpoint",
        f"backend: state_checkpoint\n  model_dir: {tmp_path / 'no_such_dir'}\n"
        f"  checkpoint_path: {tmp_path / 'no_such_checkpoint.ckpt'}",
    )
    config_path.write_text(text, encoding="utf-8")
    config = load_config(config_path)
    with pytest.raises(ValueError, match="not found"):
        cli.validate_config_shape(config, check_paths=True)


# --- _project_to_configured_view --------------------------------------------


def _tiny_two_view_bags() -> GeneBags:
    n_control = 4
    genes = ("G0", "G1")
    input_bags = tuple(np.full((3, 2560), np.nan, dtype=np.float32) for _ in genes)
    target_bags = tuple(
        np.ones((3, 5), dtype=np.float32) * index for index, _ in enumerate(genes)
    )
    metadata = pd.DataFrame({"perturbation_gene": genes})
    return GeneBags(
        genes=np.asarray(genes, dtype=object),
        y=np.full(len(genes), np.nan, dtype=np.float32),
        input_bags=input_bags,
        latent_bags=target_bags,
        control_input=np.zeros((n_control, 2560), dtype=np.float32),
        control_latent=np.ones((n_control, 5), dtype=np.float32),
        cell_type_bags=None,
        control_cell_type=None,
        batch_bags=None,
        control_batch=None,
        feature_names=np.asarray([f"tx1_{i}" for i in range(2560)], dtype=object),
        metadata=metadata,
        input_dim=2560,
        latent_dim=5,
        target_bags=target_bags,
        control_target=np.ones((n_control, 5), dtype=np.float32),
        target_dim=5,
        target_feature_names=np.asarray([f"GENE{i}" for i in range(5)], dtype=object),
    )


def test_project_to_configured_view_obsm_is_a_no_op() -> None:
    bags = _tiny_two_view_bags()
    projected = cli._project_to_configured_view(bags, "obsm")
    assert projected is bags


def test_project_to_configured_view_checkpoint_hvg_repoints_input_at_target() -> None:
    bags = _tiny_two_view_bags()
    projected = cli._project_to_configured_view(bags, "checkpoint_hvg")
    assert projected.input_dim == bags.target_dim
    assert projected.input_bags is bags.target_bags
    assert projected.control_input is bags.control_target  # C1: same object
    assert projected.feature_names is bags.target_feature_names
    assert projected.target_bags is None
    assert projected.control_target is None
    assert projected.target_dim is None
    # effective_* falls back to input_* now -- confirms a real single view.
    assert projected.effective_target_bags is projected.input_bags
    assert projected.effective_control_target is projected.control_input


def test_project_to_configured_view_rejects_unknown_view() -> None:
    bags = _tiny_two_view_bags()
    with pytest.raises(ValueError, match="input_view"):
        cli._project_to_configured_view(bags, "bogus")


# --- full --dry-run against tiny local fixtures -----------------------------


def _write_var_dims(model_dir: Path, gene_names: list[str]) -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)
    with (model_dir / "var_dims.pkl").open("wb") as handle:
        pickle.dump({"gene_names": gene_names}, handle)
    return model_dir


def _dry_run_fixture(tmp_path: Path) -> dict[str, Path]:
    """One ``train_response_and_head`` line plus a ``test`` line (never touched)."""
    hvg_genes = ["GENE0", "GENE1", "GENE2"]
    hvg_dir = _write_var_dims(tmp_path / "hvg_state", hvg_genes)
    manifest_rows = [
        _manifest_row(
            model_id="ACH-A",
            cellosaurus_id="CVCL_A",
            cell_line_name="LineA",
            basal_source="Perturb-seq non-targeting control",
            role="train_response_and_head",
        ),
        _manifest_row(
            model_id="ACH-T",
            cellosaurus_id="CVCL_T",
            cell_line_name="LineT",
            basal_source="Tahoe-100M DMSO",
            role="test",
        ),
    ]
    manifest_path = _write_manifest(tmp_path / "manifest.csv", manifest_rows)
    h5ad_path = _write_perturbseq_h5ad(
        tmp_path / "line_a.h5ad",
        n_control=5,
        n_other=4,
        perturbation_col="gene",
        control_label="non-targeting",
        n_genes=len(hvg_genes),
    )
    sources = {
        "ACH-A": {
            "h5ad_path": str(h5ad_path),
            "perturbation_col": "gene",
            "control_label": "non-targeting",
            "var_ensembl_col": "gene_id",
            "target_gene_symbol_col": "gene_symbol",
        }
    }
    sources_path = tmp_path / "sources.json"
    sources_path.write_text(json.dumps(sources), encoding="utf-8")

    cache_dir = tmp_path / "cache"
    n_control_cache = 6
    embeddings = np.tile(
        (np.arange(n_control_cache, dtype=np.float32) + 1.0)[:, None],
        (1, EMBEDDING_WIDTH),
    )
    hvg_matrix = np.arange(n_control_cache * len(hvg_genes), dtype=np.float32).reshape(
        n_control_cache, len(hvg_genes)
    )
    arrays_a = write_line_cache(
        cache_dir,
        "ACH-A",
        embeddings,
        hvg_matrix,
        pd.DataFrame(index=[f"ctrl{i}" for i in range(n_control_cache)]),
        hvg_gene_order=hvg_genes,
    )
    # ACH-T (role=test) is never read by the assembly, but an unrestricted
    # verify_cache pass (Codex P1-d) requires every frozen-manifest line to
    # have SOME cache entry -- mirrors the real Phase B contract (all 42
    # lines cached, not just the 4 response ones).
    arrays_t = write_line_cache(
        cache_dir,
        "ACH-T",
        embeddings[:2],
        hvg_matrix[:2],
        pd.DataFrame(index=["ctrl-t0", "ctrl-t1"]),
        hvg_gene_order=hvg_genes,
    )
    _write_cache_run_manifest(
        cache_dir, {"ACH-A": arrays_a, "ACH-T": arrays_t}, hvg_genes
    )
    return {
        "manifest": manifest_path,
        "cache_dir": cache_dir,
        "sources": sources_path,
        "hvg_dir": hvg_dir,
    }


def _write_multi_gene_perturbseq_h5ad(
    path: Path,
    *,
    hvg_genes: list[str],
    control_n: int,
    gene_cells: dict[str, int],
) -> Path:
    """A tiny multi-perturbation-gene h5ad (``write_tx1_perturbseq_h5ad_
    ensembl_index`` only supports a single hardcoded perturbed gene)."""
    labels = ["non-targeting"] * control_n
    for gene, n in gene_cells.items():
        labels += [gene] * n
    n_cells = len(labels)
    rng = np.random.default_rng(0)
    counts = rng.integers(1, 10, size=(n_cells, len(hvg_genes))).astype(np.float32)
    obs = pd.DataFrame(
        {"gene": labels}, index=[f"cell{index}" for index in range(n_cells)]
    )
    var_index = pd.Index(
        [f"ENSG{index:011d}" for index in range(len(hvg_genes))], name="gene_id"
    )
    var = pd.DataFrame({"gene_symbol": hvg_genes})
    var.index = var_index
    ad.AnnData(X=csr_matrix(counts), obs=obs, var=var).write_h5ad(path)
    return path


def _multi_gene_dry_run_fixture(tmp_path: Path) -> dict[str, Path]:
    """Like ``_dry_run_fixture``, but with 3 distinct perturbed genes -- the
    minimum ``_split_fold`` needs to produce a non-empty train/val/test."""
    hvg_genes = ["GENE0", "GENE1", "GENE2"]
    hvg_dir = _write_var_dims(tmp_path / "hvg_state", hvg_genes)
    manifest_rows = [
        _manifest_row(
            model_id="ACH-A",
            cellosaurus_id="CVCL_A",
            cell_line_name="LineA",
            basal_source="Perturb-seq non-targeting control",
            role="train_response_and_head",
        ),
    ]
    manifest_path = _write_manifest(tmp_path / "manifest.csv", manifest_rows)
    h5ad_path = _write_multi_gene_perturbseq_h5ad(
        tmp_path / "line_a.h5ad",
        hvg_genes=hvg_genes,
        control_n=5,
        gene_cells={"PERT1": 4, "PERT2": 4, "PERT3": 4},
    )
    sources = {
        "ACH-A": {
            "h5ad_path": str(h5ad_path),
            "perturbation_col": "gene",
            "control_label": "non-targeting",
            "var_ensembl_col": "gene_id",
            "target_gene_symbol_col": "gene_symbol",
        }
    }
    sources_path = tmp_path / "sources.json"
    sources_path.write_text(json.dumps(sources), encoding="utf-8")

    cache_dir = tmp_path / "cache"
    n_control_cache = 6
    embeddings = np.tile(
        (np.arange(n_control_cache, dtype=np.float32) + 1.0)[:, None],
        (1, EMBEDDING_WIDTH),
    )
    hvg_matrix = np.arange(n_control_cache * len(hvg_genes), dtype=np.float32).reshape(
        n_control_cache, len(hvg_genes)
    )
    arrays_a = write_line_cache(
        cache_dir,
        "ACH-A",
        embeddings,
        hvg_matrix,
        pd.DataFrame(index=[f"ctrl{i}" for i in range(n_control_cache)]),
        hvg_gene_order=hvg_genes,
    )
    _write_cache_run_manifest(cache_dir, {"ACH-A": arrays_a}, hvg_genes)
    return {
        "manifest": manifest_path,
        "cache_dir": cache_dir,
        "sources": sources_path,
        "hvg_dir": hvg_dir,
    }


def _write_dry_run_config(tmp_path: Path, hvg_dir: Path, *, input_view: str) -> Path:
    input_dim = 2560 if input_view == "obsm" else 3
    embed_key = "X_tx1_3b" if input_view == "obsm" else "null"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
data:
  h5ad_path: UNUSED
  overlap_csv: UNUSED
  output_dir: {tmp_path / "outputs"}
  state_embed_key: {embed_key}
state:
  backend: linear_mock
  model_dir: {hvg_dir}
  input_dim: {input_dim}
  output_dim: 3
  pert_dim: 4
  input_view: {input_view}
  output_space: gene
  l2_normalize_input: false
response_encoder:
  input_dim: 3
  latent_dim: 128
gmm:
  trainable: true
loss:
  pred_c_weight: 0.0
  obs_c_weight: 0.0
  occupancy_weight: 0.0
  gmm_nll_weight: 0.0
  pred_rank_weight: 0.0
train:
  gene_batch_size: 1
""",
        encoding="utf-8",
    )
    return config_path


@pytest.mark.parametrize("input_view", ["obsm", "checkpoint_hvg"])
def test_dry_run_end_to_end_against_tiny_fixtures(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], input_view: str
) -> None:
    """Both arms' config-shape + cheap source validation runs against small
    local fixtures -- no real Tx1/HPC assets, and (Defect 3) no cell matrix
    is ever read; see ``test_dry_run_never_materializes_a_cell_matrix``."""
    fixture = _dry_run_fixture(tmp_path)
    config_path = _write_dry_run_config(
        tmp_path, fixture["hvg_dir"], input_view=input_view
    )
    argv = [
        "train_tx1_st_response.py",
        "--config",
        str(config_path),
        "--cache-dir",
        str(fixture["cache_dir"]),
        "--line-manifest",
        str(fixture["manifest"]),
        "--perturbseq-source-config",
        str(fixture["sources"]),
        "--dry-run",
    ]
    import sys

    old_argv = sys.argv
    sys.argv = argv
    try:
        cli.main()
    finally:
        sys.argv = old_argv

    summary = json.loads(capsys.readouterr().out)
    assert summary["input_view"] == input_view
    assert summary["response_encoder_input_dim"] == 3
    assert summary["state_input_dim"] == (2560 if input_view == "obsm" else 3)
    assert summary["n_lines"] == 1
    assert summary["checkpoint_hvg_width"] == 3
    assert summary["lines"]["ACH-A"]["role"] == "train_response_and_head"
    assert summary["lines"]["ACH-A"]["source_type"] == "h5ad"
    assert summary["lines"]["ACH-A"]["hvg_vocabulary_coverage"] == 1.0
    # the excluded test-role line never appears in the dry-run summary either.
    assert "ACH-T" not in summary["lines"]


def test_dry_run_never_materializes_a_cell_matrix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Defect 3's core assertion: --dry-run must not read a single cell's
    expression values. ``_materialize_rows`` is the shared function every
    h5ad-backed builder calls right before reading the bulk matrix
    (tx1_basal.py); making it raise proves --dry-run never reaches it,
    while a real (non-dry-run) assembly still would (see
    ``test_run_real_training_end_to_end_against_tiny_fixtures``, unaffected
    by this monkeypatch)."""
    import aivc_model.tx1_basal as tx1_basal_module

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("_materialize_rows must not run during --dry-run")

    monkeypatch.setattr(tx1_basal_module, "_materialize_rows", _boom)

    fixture = _dry_run_fixture(tmp_path)
    config_path = _write_dry_run_config(tmp_path, fixture["hvg_dir"], input_view="obsm")
    argv = [
        "train_tx1_st_response.py",
        "--config",
        str(config_path),
        "--cache-dir",
        str(fixture["cache_dir"]),
        "--line-manifest",
        str(fixture["manifest"]),
        "--perturbseq-source-config",
        str(fixture["sources"]),
        "--dry-run",
    ]
    import sys

    old_argv = sys.argv
    sys.argv = argv
    try:
        cli.main()  # must not raise
    finally:
        sys.argv = old_argv


def test_dry_run_rejects_inadmissible_role_without_reading_cells(
    tmp_path: Path,
) -> None:
    """A bad line role/basal_source must fail --dry-run loudly (C5), not hang
    or silently admit the line."""
    fixture = _dry_run_fixture(tmp_path)
    # Corrupt ACH-A's basal_source so it fails _assert_admissible_role's C5
    # check, without touching anything else about the fixture.
    manifest_text = (
        fixture["manifest"]
        .read_text(encoding="utf-8")
        .replace("Perturb-seq non-targeting control", "Tahoe-100M DMSO")
    )
    fixture["manifest"].write_text(manifest_text, encoding="utf-8")
    config_path = _write_dry_run_config(tmp_path, fixture["hvg_dir"], input_view="obsm")
    argv = [
        "train_tx1_st_response.py",
        "--config",
        str(config_path),
        "--cache-dir",
        str(fixture["cache_dir"]),
        "--line-manifest",
        str(fixture["manifest"]),
        "--perturbseq-source-config",
        str(fixture["sources"]),
        "--dry-run",
    ]
    import sys

    old_argv = sys.argv
    sys.argv = argv
    try:
        with pytest.raises(ValueError, match="basal_source"):
            cli.main()
    finally:
        sys.argv = old_argv


def test_dry_run_fails_loudly_on_a_bad_config(tmp_path: Path) -> None:
    """A config that would silently drop half its meaning must fail dry-run."""
    fixture = _dry_run_fixture(tmp_path)
    config_path = _write_dry_run_config(
        tmp_path, fixture["hvg_dir"], input_view="checkpoint_hvg"
    )
    # Corrupt response_encoder.input_dim so it disagrees with the real
    # assembled target width (3).
    text = config_path.read_text(encoding="utf-8").replace(
        "input_dim: 3\n  latent_dim: 128", "input_dim: 999\n  latent_dim: 128"
    )
    config_path.write_text(text, encoding="utf-8")

    import sys

    argv = [
        "train_tx1_st_response.py",
        "--config",
        str(config_path),
        "--cache-dir",
        str(fixture["cache_dir"]),
        "--line-manifest",
        str(fixture["manifest"]),
        "--perturbseq-source-config",
        str(fixture["sources"]),
        "--dry-run",
    ]
    old_argv = sys.argv
    sys.argv = argv
    try:
        with pytest.raises(ValueError, match="response_encoder.input_dim"):
            cli.main()
    finally:
        sys.argv = old_argv


# --- GeneBags.for_genes's general name-keyed-dict behavior -----------------


def test_for_genes_silently_drops_a_duplicate_gene_name_across_lines() -> None:
    """Characterizes ``GeneBags.for_genes``'s general mechanism, unchanged by
    the fix round below.

    ``GeneBags.for_genes`` (shared, generic CV-protocol machinery) looks
    gene names up in a ``{name: index}`` dict, which silently collapses a
    truly duplicate requested name to the last-seen index. This is NOT
    itself the bug Codex P1-b flagged -- Task 5's assembly now avoids ever
    requesting a duplicate by keying every bag as ``f"{gene}@{model_id}"``
    (see ``test_split_fold_never_splits_a_base_gene_across_roles`` below),
    so this dict's behavior is simply never exercised on a real duplicate
    for Phase C data. Pinned so the next bad surprise here is a broken
    test, not a broken production run.
    """
    genes = np.asarray(["TP53", "TP53"], dtype=object)  # same gene, 2 lines
    bags = GeneBags(
        genes=genes,
        y=np.full(2, np.nan, dtype=np.float32),
        input_bags=(
            np.full((1, 4), np.nan, dtype=np.float32),
            np.full((1, 4), np.nan, dtype=np.float32),
        ),
        latent_bags=(
            np.full((1, 2), 1.0, dtype=np.float32),
            np.full((1, 2), 2.0, dtype=np.float32),
        ),
        control_input=np.zeros((1, 4), dtype=np.float32),
        control_latent=np.zeros((1, 2), dtype=np.float32),
        cell_type_bags=None,
        control_cell_type=None,
        batch_bags=np.asarray(["ACH-LINE-1", "ACH-LINE-2"], dtype=object),
        control_batch=None,
        feature_names=None,
        metadata=pd.DataFrame({"perturbation_gene": genes, "model_id": genes}),
        input_dim=4,
        latent_dim=2,
        access_recorder=GeneAccessRecorder(
            FoldSpec(outer_fold=0, train_genes=("TP53",), val_genes=(), test_genes=())
        ),
    )

    view = bags.for_genes(("TP53",), stage="adapter_fit")

    # Only ONE of the two "TP53" bags survives; the other is silently gone,
    # with no error and no count mismatch signal.
    assert len(view.genes) == 1
    assert len(bags.genes) == 2


# --- _split_fold: composite-key grouping and leakage prevention (fix 1) ----


def _split_only_config(seed: int) -> SimpleNamespace:
    """A minimal duck-typed config exposing only what ``_split_fold`` reads."""
    return SimpleNamespace(split=SplitConfig(random_state=seed))


def test_split_fold_never_splits_a_base_gene_across_roles() -> None:
    """The required leakage test: no base gene may appear in two fold roles.

    Six base genes, each with 2-4 per-line composite bags (mirroring
    real overlapping-panel data), repeated over several seeds. Without the
    base-gene grouping this directly asserts, ``KIF11@ACH-000551`` could
    land in train while ``KIF11@ACH-000995`` lands in test -- cross-line
    leakage of the same biological gene, a CV-protocol violation.
    """
    lines = ("ACH-000551", "ACH-000971", "ACH-000995", "ACH-000739")
    # A deterministic, varied 2-4 lines per gene (mirrors overlapping panels
    # of differing size across genes) -- fixed, not hash()-derived, so this
    # test's composition never depends on PYTHONHASHSEED.
    lines_per_gene = {
        "KIF11": 4,
        "TP53": 2,
        "MYC": 3,
        "EGFR": 2,
        "BRCA1": 4,
        "PTEN": 3,
    }
    composite_genes = [
        composite_gene_key(gene, line)
        for gene, n_lines in lines_per_gene.items()
        for line in lines[:n_lines]
    ]
    for seed in range(10):
        fold = cli._split_fold(composite_genes, _split_only_config(seed))  # type: ignore[arg-type]

        role_by_composite = {
            gene: role
            for role, genes in (
                ("train", fold.train_genes),
                ("val", fold.val_genes),
                ("test", fold.test_genes),
            )
            for gene in genes
        }
        # Every composite bag was assigned exactly one role.
        assert set(role_by_composite) == set(composite_genes)

        # THE LEAKAGE ASSERTION: every base gene's per-line bags share ONE role.
        roles_by_base: dict[str, set[str]] = {}
        for composite, role in role_by_composite.items():
            roles_by_base.setdefault(base_gene_name(composite), set()).add(role)
        multi_role_genes = {
            base: roles for base, roles in roles_by_base.items() if len(roles) > 1
        }
        assert not multi_role_genes, (
            f"base gene(s) split across multiple fold roles (seed={seed}): "
            f"{multi_role_genes}"
        )


def test_split_fold_preserves_every_composite_bag() -> None:
    """Codex P1-b's core ask: no line's bag is dropped by the split itself."""
    composite_genes = [
        composite_gene_key(gene, line)
        for gene in ("KIF11", "TP53", "MYC")
        for line in ("ACH-A", "ACH-B")
    ]
    fold = cli._split_fold(composite_genes, _split_only_config(0))  # type: ignore[arg-type]
    all_assigned = sorted(fold.train_genes + fold.val_genes + fold.test_genes)
    assert all_assigned == sorted(composite_genes)


# --- _source_fingerprint: covers the actual training inputs (fix 4) --------


def test_source_fingerprint_changes_when_cache_manifest_changes(
    tmp_path: Path,
) -> None:
    """Codex P2-a: two runs against different Tx1 caches must not collide."""
    line_manifest = tmp_path / "manifest.csv"
    line_manifest.write_text("model_id\nACH-A\n", encoding="utf-8")
    sources = tmp_path / "sources.json"
    sources.write_text("{}", encoding="utf-8")
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "manifest.json").write_text('{"lines": {}}', encoding="utf-8")

    before = cli._source_fingerprint(line_manifest, sources, cache_dir)
    (cache_dir / "manifest.json").write_text(
        '{"lines": {"ACH-A": {}}}', encoding="utf-8"
    )
    after = cli._source_fingerprint(line_manifest, sources, cache_dir)
    assert before != after


def test_source_fingerprint_changes_when_a_referenced_h5ad_is_modified(
    tmp_path: Path,
) -> None:
    """Codex P2-a: a modified response source at the SAME configured path
    must not collide in provenance with the original."""
    line_manifest = tmp_path / "manifest.csv"
    line_manifest.write_text("model_id\nACH-A\n", encoding="utf-8")
    h5ad_path = tmp_path / "line_a.h5ad"
    h5ad_path.write_bytes(b"original bytes")
    sources = tmp_path / "sources.json"
    sources.write_text(
        json.dumps(
            {
                "ACH-A": {
                    "source_type": "h5ad",
                    "h5ad_path": str(h5ad_path),
                    "perturbation_col": "gene",
                    "control_label": "non-targeting",
                    "var_ensembl_col": "gene_id",
                }
            }
        ),
        encoding="utf-8",
    )
    cache_dir = tmp_path / "cache"  # no manifest.json -- exercises the
    # "cache not yet built" branch, still a valid fingerprint input.

    before = cli._source_fingerprint(line_manifest, sources, cache_dir)
    h5ad_path.write_bytes(b"modified bytes -- same path, different content")
    after = cli._source_fingerprint(line_manifest, sources, cache_dir)
    assert before != after


def test_source_fingerprint_deterministic_for_unchanged_inputs(
    tmp_path: Path,
) -> None:
    line_manifest = tmp_path / "manifest.csv"
    line_manifest.write_text("model_id\nACH-A\n", encoding="utf-8")
    sources = tmp_path / "sources.json"
    sources.write_text("{}", encoding="utf-8")
    cache_dir = tmp_path / "cache"
    first = cli._source_fingerprint(line_manifest, sources, cache_dir)
    second = cli._source_fingerprint(line_manifest, sources, cache_dir)
    assert first == second


# --- real (non-dry-run) training: closes the "missing cv:" gap ------------


def _write_real_training_config(tmp_path: Path, hvg_dir: Path) -> Path:
    """Like ``_write_dry_run_config``, but sized for an actual (if tiny)
    training run and, crucially, WITHOUT a ``cv:`` section -- proving
    ``run_real_training`` synthesizes its own split-authority file rather
    than requiring one pre-declared in the checked-in YAML."""
    config_path = tmp_path / "real_config.yaml"
    config_path.write_text(
        f"""
data:
  h5ad_path: UNUSED
  overlap_csv: UNUSED
  output_dir: {tmp_path / "outputs"}
  state_embed_key: null
state:
  backend: linear_mock
  model_dir: {hvg_dir}
  input_dim: 3
  output_dim: 3
  pert_dim: 4
  input_view: checkpoint_hvg
  output_space: gene
  l2_normalize_input: false
response_encoder:
  input_dim: 3
  latent_dim: 128
gmm:
  trainable: true
  n_components: 2
  init_scale: 0.02
loss:
  pred_c_weight: 0.0
  obs_c_weight: 0.0
  occupancy_weight: 0.0
  gmm_nll_weight: 0.0
  pred_rank_weight: 0.0
train:
  run_id: real_training_test
  max_epochs: 2
  cell_set_len: 2
  gene_batch_size: 1
  eval_control_panel_size: 2
  eval_window_macro_batch_size: 2
  required_world_size: 4
  device: cpu
""",
        encoding="utf-8",
    )
    return config_path


def test_run_real_training_end_to_end_against_tiny_fixtures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The non-dry-run path actually trains, without a pre-declared ``cv:``.

    Regression test for a real bug an independent review caught: both
    checked-in Phase C configs omit ``cv:`` (there is no pre-registered
    canonical split to reference), but
    ``train._run_audited_training``/``_canonical_split_sha256`` require
    ``config.cv.outer_split_sha256_file`` unconditionally -- without
    ``run_real_training`` synthesizing one itself, the documented
    ``accelerate launch ... scripts/train_tx1_st_response.py`` invocation
    would crash before a single batch trained.
    """
    fixture = _multi_gene_dry_run_fixture(tmp_path)
    config_path = _write_real_training_config(tmp_path, fixture["hvg_dir"])
    monkeypatch.setattr(
        train_module,
        "require_exact_world_size",
        lambda *_args, **_kwargs: None,
    )

    argv = [
        "train_tx1_st_response.py",
        "--config",
        str(config_path),
        "--cache-dir",
        str(fixture["cache_dir"]),
        "--line-manifest",
        str(fixture["manifest"]),
        "--perturbseq-source-config",
        str(fixture["sources"]),
    ]
    import sys

    old_argv = sys.argv
    sys.argv = argv
    try:
        cli.main()
    finally:
        sys.argv = old_argv

    run_dir = tmp_path / "outputs" / "runs" / "real_training_test"
    assert (run_dir / "train_log.csv").is_file()
    assert (run_dir / "phase_c_gene_split.csv").is_file()
    assert (run_dir / "phase_c_gene_split.csv.sha256").is_file()
    split = pd.read_csv(run_dir / "phase_c_gene_split.csv")
    assert set(split["role"]) <= {"train", "val", "test"}
    assert len(split) == 3  # PERT1/PERT2/PERT3, one role each
