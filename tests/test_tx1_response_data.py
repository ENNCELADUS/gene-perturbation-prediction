"""Tests for src/aivc_model/tx1_response_data.py -- multi-line observed-response
``GeneBags`` assembly (Wave 2 Phase C, Task 5).

No GPU and no real Tx1/Perturb-seq/X-Atlas-Orion data exist on this machine,
so every test builds tiny synthetic fixtures in ``tmp_path``: a frozen-manifest-
shaped CSV, a fake Tx1 embedding cache (written via
``tx1_embed_cache.write_line_cache``, never via real Tx1 inference), and
either a Perturb-seq h5ad or an X-Atlas-Orion parquet shard as the raw
response source. A skip here would hide a real gap, so nothing in this file
is allowed to skip silently.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import NamedTuple

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

import aivc_model.tx1_response_data as tx1_response_data_module
from aivc_model.tx1_embed_cache import EMBEDDING_WIDTH, write_line_cache
from aivc_model.tx1_response_data import (
    _assert_admissible_role,
    _assert_target_order_matches,
    _select_train_response_lines,
    assemble_train_response_gene_bags,
    base_gene_name,
    composite_gene_key,
)
from conftest import tx1_manifest_row as _manifest_row
from conftest import write_tx1_cache_run_manifest as _write_cache_run_manifest
from conftest import write_tx1_line_manifest as _write_manifest
from conftest import write_tx1_xatlas_gene_metadata as _write_xatlas_gene_metadata
from conftest import write_tx1_xatlas_shard as _write_xatlas_shard

_HVG_GENES = ["GENE0", "GENE1", "GENE2"]
_TOKENS = [0, 1, 2]


# --- local synthetic-fixture builders ---------------------------------------


def _write_var_dims(model_dir: Path, gene_names: list[str]) -> Path:
    """Write a minimal released-checkpoint ``var_dims.pkl``."""
    model_dir.mkdir(parents=True, exist_ok=True)
    with (model_dir / "var_dims.pkl").open("wb") as handle:
        pickle.dump({"gene_names": gene_names}, handle)
    return model_dir


def _embeddings(n_cells: int) -> np.ndarray:
    """A valid-width (``EMBEDDING_WIDTH``), non-zero-per-row synthetic array."""
    values = (np.arange(n_cells, dtype=np.float32) + 1.0)[:, None]
    return np.tile(values, (1, EMBEDDING_WIDTH))


def _obs(n_cells: int, prefix: str) -> pd.DataFrame:
    return pd.DataFrame(index=[f"{prefix}{index}" for index in range(n_cells)])


def _write_line_cache(
    cache_dir: Path, model_id: str, n_control: int
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Write a fake Phase B cache entry; returns the raw arrays for comparison
    plus the array metadata ``write_line_cache`` recorded (needed to build a
    valid run manifest via ``write_tx1_cache_run_manifest``)."""
    embeddings = _embeddings(n_control)
    hvg = np.arange(n_control * len(_HVG_GENES), dtype=np.float32).reshape(
        n_control, len(_HVG_GENES)
    )
    arrays = write_line_cache(
        cache_dir,
        model_id,
        embeddings,
        hvg,
        _obs(n_control, f"{model_id}-ctrl"),
        hvg_gene_order=_HVG_GENES,
    )
    return embeddings, hvg, arrays


def _write_multi_gene_perturbseq_h5ad(
    path: Path,
    *,
    control_label: str,
    perturbation_col: str,
    control_n: int,
    gene_cells: dict[str, int],
) -> Path:
    """Perturb-seq h5ad with Ensembl-only ``var.index`` (name ``gene_id``) and
    a ``gene_symbol`` column, supporting multiple distinct perturbed genes --
    unlike conftest's single-gene ``write_tx1_perturbseq_h5ad_ensembl_index``.
    """
    labels = [control_label] * control_n
    for gene, n in gene_cells.items():
        labels += [gene] * n
    n_cells = len(labels)
    rng = np.random.default_rng(0)
    counts = rng.integers(1, 10, size=(n_cells, len(_HVG_GENES))).astype(np.float32)
    obs = pd.DataFrame(
        {perturbation_col: labels}, index=[f"cell{index}" for index in range(n_cells)]
    )
    var_index = pd.Index(
        [f"ENSG{index:011d}" for index in range(len(_HVG_GENES))], name="gene_id"
    )
    var = pd.DataFrame({"gene_symbol": _HVG_GENES})
    var.index = var_index
    adata = ad.AnnData(X=csr_matrix(counts), obs=obs, var=var)
    adata.write_h5ad(path)
    return path


def _write_multi_gene_xatlas_shard(
    path: Path,
    *,
    control_label: str,
    control_n: int,
    gene_cells: dict[str, int],
) -> Path:
    """X-Atlas-Orion shard supporting multiple distinct perturbed genes."""

    def _row(barcode: str, gene_target: str) -> dict[str, object]:
        return {
            "gene_token_id": np.array(_TOKENS),
            "gene_expression": np.array([1.0, 2.0, 3.0], dtype=np.float32),
            "cell_barcode": barcode,
            "sample": "HCT116_Batch1",
            "gene_target": gene_target,
            "pass_guide_filter": 1,
        }

    rows = [_row(f"ctrl{index}", control_label) for index in range(control_n)]
    for gene, n in gene_cells.items():
        rows += [_row(f"{gene}_{index}", gene) for index in range(n)]
    return _write_xatlas_shard(path, rows)


class _TwoLineFixture(NamedTuple):
    manifest_path: Path
    cache_dir: Path
    hvg_dir: Path
    sources_path: Path
    control_embeddings: dict[str, np.ndarray]
    control_hvg: dict[str, np.ndarray]
    expected_counts: dict[tuple[str, str], int]


def _build_two_line_fixture(tmp_path: Path) -> _TwoLineFixture:
    """Two ``train_response_and_head`` lines (h5ad + X-Atlas-Orion) plus a
    ``test``-role line that must never be touched (C5), sharing one HVG
    checkpoint gene order.

    Line A (h5ad, ACH-A): control=5, PERT_A_ONLY=2, PERT_SHARED=3.
    Line B (X-Atlas, ACH-B): control=4, PERT_B_ONLY=1, PERT_SHARED=6.
    """
    hvg_dir = _write_var_dims(tmp_path / "hvg_state", list(_HVG_GENES))

    manifest_rows = [
        _manifest_row(
            model_id="ACH-A",
            cellosaurus_id="CVCL_A",
            cell_line_name="LineA",
            basal_source="Perturb-seq non-targeting control",
            role="train_response_and_head",
        ),
        _manifest_row(
            model_id="ACH-B",
            cellosaurus_id="CVCL_B",
            cell_line_name="LineB",
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

    h5ad_path = _write_multi_gene_perturbseq_h5ad(
        tmp_path / "line_a.h5ad",
        control_label="non-targeting",
        perturbation_col="gene",
        control_n=5,
        gene_cells={"PERT_A_ONLY": 2, "PERT_SHARED": 3},
    )
    xatlas_dir = tmp_path / "line_b_shards"
    xatlas_dir.mkdir()
    _write_multi_gene_xatlas_shard(
        xatlas_dir / "shard1.parquet",
        control_label="Non-Targeting",
        control_n=4,
        gene_cells={"PERT_B_ONLY": 1, "PERT_SHARED": 6},
    )
    gene_metadata_path = _write_xatlas_gene_metadata(
        tmp_path / "xatlas_genes.parquet", _TOKENS
    )

    sources = {
        "ACH-A": {
            "source_type": "h5ad",
            "h5ad_path": str(h5ad_path),
            "perturbation_col": "gene",
            "control_label": "non-targeting",
            "var_ensembl_col": "gene_id",
            "target_gene_symbol_col": "gene_symbol",
        },
        "ACH-B": {
            "source_type": "xatlas_orion_parquet",
            "shard_dir": str(xatlas_dir),
            "gene_metadata_path": str(gene_metadata_path),
            "control_label": "Non-Targeting",
        },
    }
    sources_path = tmp_path / "perturbseq_sources.json"
    sources_path.write_text(json.dumps(sources))

    cache_dir = tmp_path / "cache"
    embeddings_a, hvg_a, arrays_a = _write_line_cache(cache_dir, "ACH-A", n_control=5)
    embeddings_b, hvg_b, arrays_b = _write_line_cache(cache_dir, "ACH-B", n_control=4)
    # ACH-T (role=test) is never read by the assembly, but verify_cache's
    # unrestricted completeness check (Codex P1-d) requires every frozen-
    # manifest line to have SOME cache entry, mirroring the real Phase B
    # contract (the cache covers all 42 lines, not just the 4 response ones).
    _, _, arrays_t = _write_line_cache(cache_dir, "ACH-T", n_control=2)
    _write_cache_run_manifest(
        cache_dir,
        {"ACH-A": arrays_a, "ACH-B": arrays_b, "ACH-T": arrays_t},
        _HVG_GENES,
    )

    return _TwoLineFixture(
        manifest_path=manifest_path,
        cache_dir=cache_dir,
        hvg_dir=hvg_dir,
        sources_path=sources_path,
        control_embeddings={"ACH-A": embeddings_a, "ACH-B": embeddings_b},
        control_hvg={"ACH-A": hvg_a, "ACH-B": hvg_b},
        expected_counts={
            ("PERT_A_ONLY", "ACH-A"): 2,
            ("PERT_SHARED", "ACH-A"): 3,
            ("PERT_B_ONLY", "ACH-B"): 1,
            ("PERT_SHARED", "ACH-B"): 6,
        },
    )


# --- manifest role filtering (C4/C5) -----------------------------------------


def test_select_train_response_lines_filters_by_role() -> None:
    manifest = pd.DataFrame(
        [
            _manifest_row(model_id="ACH-1", role="test"),
            _manifest_row(model_id="ACH-2", role="train_head"),
            _manifest_row(model_id="ACH-3", role="train_response_and_head"),
        ]
    )
    selected = _select_train_response_lines(manifest)
    assert selected["model_id"].tolist() == ["ACH-3"]


def test_select_train_response_lines_raises_when_none_selected() -> None:
    manifest = pd.DataFrame(
        [
            _manifest_row(model_id="ACH-1", role="test"),
            _manifest_row(model_id="ACH-2", role="train_head"),
        ]
    )
    with pytest.raises(ValueError, match="train_response_and_head"):
        _select_train_response_lines(manifest)


def test_assert_admissible_role_rejects_test_role() -> None:
    row = _manifest_row(
        model_id="ACH-1",
        role="test",
        basal_source="Perturb-seq non-targeting control",
    )
    with pytest.raises(ValueError, match="C5"):
        _assert_admissible_role(pd.Series(row))


def test_assert_admissible_role_rejects_non_perturbseq_basal_source() -> None:
    row = _manifest_row(
        model_id="ACH-1",
        role="train_response_and_head",
        basal_source="Tahoe-100M DMSO",
    )
    with pytest.raises(ValueError, match="basal_source"):
        _assert_admissible_role(pd.Series(row))


def test_assert_admissible_role_accepts_valid_row() -> None:
    row = _manifest_row(
        model_id="ACH-1",
        role="train_response_and_head",
        basal_source="Perturb-seq non-targeting control",
    )
    _assert_admissible_role(pd.Series(row))  # must not raise


# --- HVG gene-order mismatch guard (requirement #3) -------------------------


def test_assert_target_order_matches_passes_when_equal() -> None:
    order = np.asarray(["G0", "G1"], dtype=object)
    matrix = np.zeros((3, 2), dtype=np.float32)
    _assert_target_order_matches("ACH-A", matrix, order, order)  # must not raise


def test_assert_target_order_matches_raises_on_width_mismatch() -> None:
    order = np.asarray(["G0", "G1"], dtype=object)
    matrix = np.zeros((3, 3), dtype=np.float32)  # 3 columns, order has 2
    with pytest.raises(ValueError, match="width"):
        _assert_target_order_matches("ACH-A", matrix, order, order)


def test_assert_target_order_matches_raises_on_reordered_names() -> None:
    checkpoint_order = np.asarray(["G0", "G1"], dtype=object)
    resolved_order = np.asarray(["G1", "G0"], dtype=object)  # same set, swapped
    matrix = np.zeros((3, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="mispairing"):
        _assert_target_order_matches("ACH-A", matrix, resolved_order, checkpoint_order)


# --- end-to-end two-line assembly -------------------------------------------


def test_assemble_two_line_shapes_and_per_gene_bag_lengths(tmp_path: Path) -> None:
    fixture = _build_two_line_fixture(tmp_path)
    bags = assemble_train_response_gene_bags(
        cell_line_manifest_path=fixture.manifest_path,
        tx1_cache_dir=fixture.cache_dir,
        hvg_state_model_dir=fixture.hvg_dir,
        perturbseq_sources_path=fixture.sources_path,
    )

    assert bags.input_dim == EMBEDDING_WIDTH
    assert bags.target_dim == len(_HVG_GENES)
    assert bags.control_input.shape == (9, EMBEDDING_WIDTH)  # 5 + 4 control cells
    assert bags.control_target.shape == (9, len(_HVG_GENES))
    assert len(bags.genes) == 4  # 2 genes x 2 lines, PERT_SHARED once per line
    # Codex P1-b: every bag is keyed as f"{gene}@{model_id}", so PERT_SHARED's
    # two per-line bags are distinct keys, not a name collision.
    assert set(bags.genes) == {
        composite_gene_key("PERT_A_ONLY", "ACH-A"),
        composite_gene_key("PERT_SHARED", "ACH-A"),
        composite_gene_key("PERT_B_ONLY", "ACH-B"),
        composite_gene_key("PERT_SHARED", "ACH-B"),
    }
    assert len(set(bags.genes)) == 4, "composite keys must be unique, not collapsed"

    actual_counts = {
        (base_gene_name(str(gene)), str(bags.metadata.iloc[index]["model_id"])): int(
            bags.metadata.iloc[index]["n_cells"]
        )
        for index, gene in enumerate(bags.genes)
    }
    assert actual_counts == fixture.expected_counts

    for index in range(len(bags.genes)):
        n_cells = int(bags.metadata.iloc[index]["n_cells"])
        assert bags.input_bags[index].shape == (n_cells, EMBEDDING_WIDTH)
        assert bags.target_bags[index].shape == (n_cells, len(_HVG_GENES))
        assert np.isnan(bags.input_bags[index]).all()  # NaN input placeholder
        model_id = str(bags.metadata.iloc[index]["model_id"])
        assert bags.batch_bags[index].tolist() == [model_id] * n_cells

    # latent falls back to target (expression-as-latent), same objects.
    assert bags.latent_bags == bags.target_bags
    assert bags.latent_dim == bags.target_dim
    assert np.array_equal(bags.control_latent, bags.control_target)
    assert np.isnan(bags.y).all()  # response-only container carries no labels


def test_assemble_provenance_columns_present_and_correct(tmp_path: Path) -> None:
    fixture = _build_two_line_fixture(tmp_path)
    bags = assemble_train_response_gene_bags(
        cell_line_manifest_path=fixture.manifest_path,
        tx1_cache_dir=fixture.cache_dir,
        hvg_state_model_dir=fixture.hvg_dir,
        perturbseq_sources_path=fixture.sources_path,
    )
    assert {"model_id", "cell_line_name", "basal_source", "n_cells"}.issubset(
        bags.metadata.columns
    )
    for index, gene in enumerate(bags.genes):
        row = bags.metadata.iloc[index]
        if row["model_id"] == "ACH-A":
            assert row["cell_line_name"] == "LineA"
        else:
            assert row["model_id"] == "ACH-B"
            assert row["cell_line_name"] == "LineB"
        assert row["basal_source"] == "Perturb-seq non-targeting control"
    # the excluded test-role line never appears anywhere.
    assert "ACH-T" not in set(bags.metadata["model_id"])
    assert "ACH-T" not in set(bags.control_batch.tolist())


def test_assemble_l2_normalize_false_leaves_norms_untouched(tmp_path: Path) -> None:
    fixture = _build_two_line_fixture(tmp_path)
    bags = assemble_train_response_gene_bags(
        cell_line_manifest_path=fixture.manifest_path,
        tx1_cache_dir=fixture.cache_dir,
        hvg_state_model_dir=fixture.hvg_dir,
        perturbseq_sources_path=fixture.sources_path,
        l2_normalize=False,
    )
    expected = np.concatenate(
        [fixture.control_embeddings["ACH-A"], fixture.control_embeddings["ACH-B"]],
        axis=0,
    )
    np.testing.assert_array_equal(bags.control_input, expected)
    assert set(bags.metadata["l2_normalize"].unique().tolist()) == {False}


def test_assemble_l2_normalize_true_produces_unit_norm_rows(tmp_path: Path) -> None:
    fixture = _build_two_line_fixture(tmp_path)
    bags = assemble_train_response_gene_bags(
        cell_line_manifest_path=fixture.manifest_path,
        tx1_cache_dir=fixture.cache_dir,
        hvg_state_model_dir=fixture.hvg_dir,
        perturbseq_sources_path=fixture.sources_path,
        l2_normalize=True,
    )
    norms = np.linalg.norm(bags.control_input, axis=1)
    np.testing.assert_allclose(norms, np.ones_like(norms), rtol=1e-5)
    assert set(bags.metadata["l2_normalize"].unique().tolist()) == {True}


def test_assemble_hvg_gene_order_mismatch_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Simulates a checkpoint-order/resolved-order disagreement end to end by
    monkeypatching ``resolve_state_gene_order`` to scramble the result --
    the real function can never itself disagree with
    ``load_hvg_gene_order`` since both read the same ``var_dims.pkl``, so
    this exercises the wiring around a hypothetical resolution bug."""
    fixture = _build_two_line_fixture(tmp_path)
    real = tx1_response_data_module.resolve_state_gene_order

    def _scrambled(adata, model_dir, symbol_col):
        indices, names = real(adata, model_dir, symbol_col)
        return indices[::-1], names[::-1]

    monkeypatch.setattr(
        tx1_response_data_module, "resolve_state_gene_order", _scrambled
    )
    with pytest.raises(ValueError, match="mispairing"):
        assemble_train_response_gene_bags(
            cell_line_manifest_path=fixture.manifest_path,
            tx1_cache_dir=fixture.cache_dir,
            hvg_state_model_dir=fixture.hvg_dir,
            perturbseq_sources_path=fixture.sources_path,
        )


def test_assemble_embedding_width_mismatch_raises(tmp_path: Path) -> None:
    fixture = _build_two_line_fixture(tmp_path)
    # Overwrite ACH-A's cached embeddings with the wrong width, bypassing
    # write_line_cache's own validation (which would refuse it).
    bad = np.zeros((5, 16), dtype=np.float32)
    np.save(fixture.cache_dir / "ACH-A" / "embeddings.npy", bad)
    with pytest.raises(ValueError, match=str(EMBEDDING_WIDTH)):
        assemble_train_response_gene_bags(
            cell_line_manifest_path=fixture.manifest_path,
            tx1_cache_dir=fixture.cache_dir,
            hvg_state_model_dir=fixture.hvg_dir,
            perturbseq_sources_path=fixture.sources_path,
        )


def test_assemble_missing_cache_entry_names_the_model_id(tmp_path: Path) -> None:
    fixture = _build_two_line_fixture(tmp_path)
    manifest_rows = [
        _manifest_row(
            model_id="ACH-A",
            basal_source="Perturb-seq non-targeting control",
            role="train_response_and_head",
        ),
        _manifest_row(
            model_id="ACH-MISSING",
            basal_source="Perturb-seq non-targeting control",
            role="train_response_and_head",
        ),
    ]
    manifest_path = _write_manifest(tmp_path / "manifest2.csv", manifest_rows)
    with pytest.raises(ValueError, match="ACH-MISSING"):
        assemble_train_response_gene_bags(
            cell_line_manifest_path=manifest_path,
            tx1_cache_dir=fixture.cache_dir,
            hvg_state_model_dir=fixture.hvg_dir,
            perturbseq_sources_path=fixture.sources_path,
        )
