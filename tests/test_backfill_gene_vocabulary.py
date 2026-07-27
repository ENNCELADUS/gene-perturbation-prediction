"""Tests for scripts/backfill_gene_vocabulary.py (Phase D unblock, P1-3
CARRY-FORWARD).

Strategy: produce a REAL, fully audited Phase C run via
``scripts.train_tx1_st_response.main()`` (HEAD already includes d576032, so
this run's checkpoints DO get a real ``gene_vocabulary.json``/
``gene_vocabulary_sha256`` written by ``aivc_model.train._save_model_
checkpoint``). Strip exactly what a pre-d576032 checkpoint would lack
(``gene_vocabulary.json`` and the ``gene_vocabulary_sha256`` metadata key),
then prove ``backfill_gene_vocabulary`` reconstructs BYTE-IDENTICAL output
from the stripped state -- the one test that actually matters, since any
serialization difference changes the hash Phase D authenticates against.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

import scripts.backfill_gene_vocabulary as backfill_cli
import scripts.train_tx1_st_response as train_cli
from aivc_model import train as train_module
from aivc_model.tx1_embed_cache import EMBEDDING_WIDTH, write_line_cache
from aivc_model.tx1_geneeffect_pipeline import verify_gene_vocabulary_authenticity
from conftest import tx1_manifest_row as _manifest_row
from conftest import write_tx1_cache_run_manifest as _write_cache_run_manifest
from conftest import write_tx1_line_manifest as _write_manifest

_HVG_GENES = ["GENE0", "GENE1", "GENE2"]
_GENE_CELLS = {"PERT1": 4, "PERT2": 4, "PERT3": 4}


def _write_var_dims(model_dir: Path, gene_names: list[str]) -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)
    with (model_dir / "var_dims.pkl").open("wb") as handle:
        pickle.dump({"gene_names": gene_names}, handle)
    return model_dir


def _write_multi_gene_perturbseq_h5ad(
    path: Path,
    *,
    hvg_genes: list[str],
    control_n: int,
    gene_cells: dict[str, int],
) -> Path:
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


def _fixture(tmp_path: Path) -> dict[str, Path]:
    """One ``train_response_and_head`` line, 3 perturbed genes -- the
    minimum ``_split_fold`` needs for a non-degenerate train/val/test split
    (proven by ``test_train_tx1_st_response.py``'s own real-training test to
    yield exactly one gene per role under the default split fractions).
    """
    hvg_dir = _write_var_dims(tmp_path / "hvg_state", _HVG_GENES)
    manifest_path = _write_manifest(
        tmp_path / "manifest.csv",
        [
            _manifest_row(
                model_id="ACH-A",
                cellosaurus_id="CVCL_A",
                cell_line_name="LineA",
                basal_source="Perturb-seq non-targeting control",
                role="train_response_and_head",
            )
        ],
    )
    h5ad_path = _write_multi_gene_perturbseq_h5ad(
        tmp_path / "line_a.h5ad",
        hvg_genes=_HVG_GENES,
        control_n=5,
        gene_cells=_GENE_CELLS,
    )
    sources_path = tmp_path / "sources.json"
    sources_path.write_text(
        json.dumps(
            {
                "ACH-A": {
                    "h5ad_path": str(h5ad_path),
                    "perturbation_col": "gene",
                    "control_label": "non-targeting",
                    "var_ensembl_col": "gene_id",
                    "target_gene_symbol_col": "gene_symbol",
                }
            }
        ),
        encoding="utf-8",
    )

    cache_dir = tmp_path / "cache"
    n_control_cache = 6
    embeddings = np.tile(
        (np.arange(n_control_cache, dtype=np.float32) + 1.0)[:, None],
        (1, EMBEDDING_WIDTH),
    )
    hvg_matrix = np.arange(n_control_cache * len(_HVG_GENES), dtype=np.float32).reshape(
        n_control_cache, len(_HVG_GENES)
    )
    arrays_a = write_line_cache(
        cache_dir,
        "ACH-A",
        embeddings,
        hvg_matrix,
        pd.DataFrame(index=[f"ctrl{i}" for i in range(n_control_cache)]),
        hvg_gene_order=_HVG_GENES,
    )
    _write_cache_run_manifest(cache_dir, {"ACH-A": arrays_a}, _HVG_GENES)
    return {
        "manifest": manifest_path,
        "cache_dir": cache_dir,
        "sources": sources_path,
        "hvg_dir": hvg_dir,
    }


def _write_config(tmp_path: Path, hvg_dir: Path, run_id: str) -> Path:
    """A tiny, real (``linear_mock``) audited Phase C config -- CPU seconds.

    Mirrors ``test_train_tx1_st_response.py``'s own
    ``_write_real_training_config``: no ``cv:`` section (proving neither
    that entrypoint nor this backfill tool needs a pre-declared canonical
    split), ``state_onehot`` tokenizer (``StateConfig``'s own default,
    matching both real Phase C arms), 3 HVG genes / pert_dim 4.
    """
    config_path = tmp_path / f"{run_id}.yaml"
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
  run_id: {run_id}
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


def _run_real_training(
    tmp_path: Path,
    fixture: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    run_id: str,
) -> Path:
    """Produce a REAL, complete Phase C run directory via the real CLI."""
    config_path = _write_config(tmp_path, fixture["hvg_dir"], run_id)
    monkeypatch.setattr(
        train_module, "require_exact_world_size", lambda *_a, **_k: None
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
    old_argv = sys.argv
    sys.argv = argv
    try:
        train_cli.main()
    finally:
        sys.argv = old_argv
    return tmp_path / "outputs" / "runs" / run_id


def _strip_to_legacy_state(run_dir: Path) -> dict[str, tuple[bytes, str]]:
    """Delete gene_vocabulary.json + its metadata hash, simulating a
    pre-d576032 checkpoint. Returns the ORIGINAL (ground-truth) bytes/hash
    per checkpoint kind, for the byte-identity assertion later.
    """
    original: dict[str, tuple[bytes, str]] = {}
    for kind in ("best", "final"):
        checkpoint_dir = run_dir / "models" / kind
        vocabulary_path = checkpoint_dir / "gene_vocabulary.json"
        metadata_path = checkpoint_dir / "metadata.json"
        assert vocabulary_path.is_file(), "HEAD must emit gene_vocabulary.json"
        original_bytes = vocabulary_path.read_bytes()
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        original_hash = metadata.pop("gene_vocabulary_sha256")
        original[kind] = (original_bytes, original_hash)
        vocabulary_path.unlink()
        metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
        )
    return original


def test_backfill_reproduces_byte_identical_vocabulary_and_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The test that matters: from a stripped (legacy-simulating) run
    directory, the backfilled gene_vocabulary.json/metadata.json are
    BYTE-IDENTICAL to what _save_model_checkpoint originally wrote.
    """
    fixture = _fixture(tmp_path)
    run_dir = _run_real_training(tmp_path, fixture, monkeypatch, "backfill_test")
    original = _strip_to_legacy_state(run_dir)

    config = train_cli.load_config(
        _write_config(tmp_path, fixture["hvg_dir"], "backfill_test")
    )
    bags = train_cli.assemble_and_project(
        config,
        cache_dir=fixture["cache_dir"],
        line_manifest=fixture["manifest"],
        perturbseq_source_config=fixture["sources"],
        response_cache_dir=None,
    )
    changed = backfill_cli.backfill_run(
        config,
        bags,
        run_dir=run_dir,
        cache_dir=fixture["cache_dir"],
        line_manifest=fixture["manifest"],
        perturbseq_source_config=fixture["sources"],
    )
    assert changed == {"best": True, "final": True}

    for kind in ("best", "final"):
        checkpoint_dir = run_dir / "models" / kind
        vocabulary_path = checkpoint_dir / "gene_vocabulary.json"
        metadata = json.loads((checkpoint_dir / "metadata.json").read_text())
        original_bytes, original_hash = original[kind]
        assert vocabulary_path.read_bytes() == original_bytes, (
            f"{kind}: backfilled gene_vocabulary.json is not byte-identical "
            "to what _save_model_checkpoint originally wrote"
        )
        assert metadata["gene_vocabulary_sha256"] == original_hash
        # The real Phase D authenticator must accept it (force
        # backend="state_checkpoint": this fixture trains with linear_mock,
        # which the authenticator legitimately no-ops for -- see its own
        # docstring -- so this proves the REAL check logic, decoupled from
        # what backend produced the checkpoint under test).
        verify_gene_vocabulary_authenticity(
            vocabulary_path, checkpoint_dir / "pytorch_model.bin", "state_checkpoint"
        )


def test_rerun_against_already_backfilled_run_is_a_clean_noop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    run_dir = _run_real_training(tmp_path, fixture, monkeypatch, "noop_test")
    _strip_to_legacy_state(run_dir)
    config = train_cli.load_config(
        _write_config(tmp_path, fixture["hvg_dir"], "noop_test")
    )
    bags = train_cli.assemble_and_project(
        config,
        cache_dir=fixture["cache_dir"],
        line_manifest=fixture["manifest"],
        perturbseq_source_config=fixture["sources"],
        response_cache_dir=None,
    )
    kwargs = dict(
        run_dir=run_dir,
        cache_dir=fixture["cache_dir"],
        line_manifest=fixture["manifest"],
        perturbseq_source_config=fixture["sources"],
    )
    first = backfill_cli.backfill_run(config, bags, **kwargs)
    assert first == {"best": True, "final": True}

    before = {
        kind: (
            (run_dir / "models" / kind / "gene_vocabulary.json").read_bytes(),
            (run_dir / "models" / kind / "metadata.json").read_bytes(),
        )
        for kind in ("best", "final")
    }
    second = backfill_cli.backfill_run(config, bags, **kwargs)
    assert second == {"best": False, "final": False}, "re-run must be a clean no-op"
    after = {
        kind: (
            (run_dir / "models" / kind / "gene_vocabulary.json").read_bytes(),
            (run_dir / "models" / kind / "metadata.json").read_bytes(),
        )
        for kind in ("best", "final")
    }
    assert before == after, "re-running must not modify already-correct files"


def test_conflicting_vocabulary_file_causes_loud_refusal_not_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    run_dir = _run_real_training(tmp_path, fixture, monkeypatch, "conflict_test")
    _strip_to_legacy_state(run_dir)
    config = train_cli.load_config(
        _write_config(tmp_path, fixture["hvg_dir"], "conflict_test")
    )
    bags = train_cli.assemble_and_project(
        config,
        cache_dir=fixture["cache_dir"],
        line_manifest=fixture["manifest"],
        perturbseq_source_config=fixture["sources"],
        response_cache_dir=None,
    )
    # Plant a WRONG (reordered) vocabulary before the tool ever runs.
    best_dir = run_dir / "models" / "best"
    corrupted = json.dumps(["PERT3", "PERT2", "PERT1"], indent=2) + "\n"
    (best_dir / "gene_vocabulary.json").write_text(corrupted, encoding="utf-8")

    with pytest.raises(backfill_cli.VocabularyBackfillError, match="DIFFERENT content"):
        backfill_cli.backfill_run(
            config,
            bags,
            run_dir=run_dir,
            cache_dir=fixture["cache_dir"],
            line_manifest=fixture["manifest"],
            perturbseq_source_config=fixture["sources"],
        )

    # The corrupted file must survive untouched -- never silently overwritten.
    assert (best_dir / "gene_vocabulary.json").read_text(encoding="utf-8") == corrupted
    best_metadata = json.loads((best_dir / "metadata.json").read_text())
    assert "gene_vocabulary_sha256" not in best_metadata


def test_conflicting_recorded_hash_causes_loud_refusal_not_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    run_dir = _run_real_training(tmp_path, fixture, monkeypatch, "hash_conflict_test")
    _strip_to_legacy_state(run_dir)
    config = train_cli.load_config(
        _write_config(tmp_path, fixture["hvg_dir"], "hash_conflict_test")
    )
    bags = train_cli.assemble_and_project(
        config,
        cache_dir=fixture["cache_dir"],
        line_manifest=fixture["manifest"],
        perturbseq_source_config=fixture["sources"],
        response_cache_dir=None,
    )
    best_dir = run_dir / "models" / "best"
    metadata = json.loads((best_dir / "metadata.json").read_text())
    metadata["gene_vocabulary_sha256"] = "0" * 64
    (best_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )

    with pytest.raises(
        backfill_cli.VocabularyBackfillError, match="differing recorded hash"
    ):
        backfill_cli.backfill_run(
            config,
            bags,
            run_dir=run_dir,
            cache_dir=fixture["cache_dir"],
            line_manifest=fixture["manifest"],
            perturbseq_source_config=fixture["sources"],
        )
    # Not overwritten with the reconstructed hash.
    after = json.loads((best_dir / "metadata.json").read_text())
    assert after["gene_vocabulary_sha256"] == "0" * 64
    # And no gene_vocabulary.json was written for best either (refusal must
    # not leave a half-applied state for THIS checkpoint dir).
    assert not (best_dir / "gene_vocabulary.json").is_file()


def test_missing_metadata_fails_with_named_actionable_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    run_dir = _run_real_training(tmp_path, fixture, monkeypatch, "missing_meta_test")
    _strip_to_legacy_state(run_dir)
    (run_dir / "models" / "best" / "metadata.json").unlink()
    config = train_cli.load_config(
        _write_config(tmp_path, fixture["hvg_dir"], "missing_meta_test")
    )
    bags = train_cli.assemble_and_project(
        config,
        cache_dir=fixture["cache_dir"],
        line_manifest=fixture["manifest"],
        perturbseq_source_config=fixture["sources"],
        response_cache_dir=None,
    )
    with pytest.raises(backfill_cli.VocabularyBackfillError, match="metadata.json"):
        backfill_cli.backfill_run(
            config,
            bags,
            run_dir=run_dir,
            cache_dir=fixture["cache_dir"],
            line_manifest=fixture["manifest"],
            perturbseq_source_config=fixture["sources"],
        )


def test_wrong_source_inputs_fail_via_fingerprint_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pointing the tool at a DIFFERENT (but validly structured) cache/
    manifest/source-config than the checkpoint actually trained against must
    be refused, not silently reconstructed against the wrong data.
    """
    fixture = _fixture(tmp_path)
    run_dir = _run_real_training(tmp_path, fixture, monkeypatch, "wrong_inputs_test")
    _strip_to_legacy_state(run_dir)

    other_root = tmp_path / "other"
    other_root.mkdir()
    other_fixture = _fixture(other_root)
    config = train_cli.load_config(
        _write_config(tmp_path, fixture["hvg_dir"], "wrong_inputs_test")
    )
    bags = train_cli.assemble_and_project(
        config,
        cache_dir=other_fixture["cache_dir"],
        line_manifest=other_fixture["manifest"],
        perturbseq_source_config=other_fixture["sources"],
        response_cache_dir=None,
    )
    with pytest.raises(
        backfill_cli.VocabularyBackfillError, match="source_fingerprint"
    ):
        backfill_cli.backfill_run(
            config,
            bags,
            run_dir=run_dir,
            cache_dir=other_fixture["cache_dir"],
            line_manifest=other_fixture["manifest"],
            perturbseq_source_config=other_fixture["sources"],
        )


def test_cli_end_to_end_via_argv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The full CLI (argv parsing, config validation, assembly, backfill)."""
    fixture = _fixture(tmp_path)
    run_dir = _run_real_training(tmp_path, fixture, monkeypatch, "cli_test")
    original = _strip_to_legacy_state(run_dir)
    config_path = _write_config(tmp_path, fixture["hvg_dir"], "cli_test")

    argv = [
        "backfill_gene_vocabulary.py",
        "--config",
        str(config_path),
        "--run-dir",
        str(run_dir),
        "--cache-dir",
        str(fixture["cache_dir"]),
        "--line-manifest",
        str(fixture["manifest"]),
        "--perturbseq-source-config",
        str(fixture["sources"]),
        "--response-cache-dir",
        str(tmp_path / "response_cache"),
    ]
    old_argv = sys.argv
    sys.argv = argv
    try:
        backfill_cli.main()
    finally:
        sys.argv = old_argv

    for kind in ("best", "final"):
        vocabulary_path = run_dir / "models" / kind / "gene_vocabulary.json"
        assert vocabulary_path.read_bytes() == original[kind][0]
