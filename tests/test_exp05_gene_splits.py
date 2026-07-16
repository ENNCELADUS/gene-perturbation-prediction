"""Tests for the immutable exp05 canonical outer-fold manifest."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aivc_model.gene_splits import (
    FoldSpec,
    assert_gene_access,
    attach_gene_provenance,
    build_canonical_outer_manifest,
    load_canonical_outer_manifest,
    make_inner_fold_spec,
    sha256_file,
)


def _labels(count: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "perturbation_gene": [f"GENE{index:05d}" for index in range(count)],
            "depmap_gene_effect": np.linspace(-2.0, 1.0, count),
        }
    )


def _write_manifest(tmp_path: Path, labels: pd.DataFrame) -> tuple[Path, str]:
    path = tmp_path / "manifest.csv"
    pd.DataFrame(
        {
            "perturbation_gene": labels["perturbation_gene"],
            "outer_fold": np.arange(len(labels)) % 5,
        }
    ).to_csv(path, index=False)
    return path, sha256_file(path)


def test_canonical_outer_manifest_freezes_all_9338_genes_once() -> None:
    labels = _labels(9338)
    manifest = build_canonical_outer_manifest(labels, n_splits=5, seed=42)
    assert manifest.columns.tolist() == ["perturbation_gene", "outer_fold"]
    assert len(manifest) == 9338
    assert manifest["perturbation_gene"].nunique() == 9338
    assert set(manifest["outer_fold"]) == {0, 1, 2, 3, 4}
    assert manifest.equals(build_canonical_outer_manifest(labels, n_splits=5, seed=42))


def test_manifest_loader_rejects_any_universe_or_hash_change(tmp_path: Path) -> None:
    labels = _labels(20)
    path, digest = _write_manifest(tmp_path, labels)
    changed = labels.iloc[:-1].copy()
    with pytest.raises(ValueError, match="canonical gene universe"):
        load_canonical_outer_manifest(path, changed, digest)
    with pytest.raises(ValueError, match="SHA-256"):
        load_canonical_outer_manifest(path, labels, "0" * 64)


def test_manifest_loader_accepts_exact_canonical_manifest(tmp_path: Path) -> None:
    labels = _labels(9338)
    manifest = build_canonical_outer_manifest(labels, n_splits=5, seed=42)
    path = tmp_path / "manifest.csv"
    manifest.to_csv(path, index=False)

    loaded = load_canonical_outer_manifest(path, labels, sha256_file(path))

    assert loaded.equals(manifest)


def _toy_labels_and_manifest(count: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = _labels(count)
    manifest = pd.DataFrame(
        {
            "perturbation_gene": labels["perturbation_gene"],
            "outer_fold": np.arange(count) % 5,
        }
    )
    return labels, manifest


def _toy_fold_spec() -> FoldSpec:
    return FoldSpec(
        outer_fold=0,
        train_genes=("A", "B"),
        val_genes=("C",),
        test_genes=("D",),
    )


def test_inner_split_consumes_frozen_outer_assignment() -> None:
    labels, manifest = _toy_labels_and_manifest(50)
    fold = make_inner_fold_spec(manifest, labels, 0, 0.1, 42)
    expected_test = set(manifest.query("outer_fold == 0")["perturbation_gene"])
    assert set(fold.test_genes) == expected_test
    assert set(fold.train_genes).isdisjoint(fold.val_genes)
    assert set(fold.train_genes).isdisjoint(fold.test_genes)
    assert set(fold.val_genes).isdisjoint(fold.test_genes)
    assert set(fold.train_genes) | set(fold.val_genes) | set(fold.test_genes) == set(
        manifest["perturbation_gene"]
    )


def test_outer_test_labels_do_not_change_inner_split() -> None:
    labels, manifest = _toy_labels_and_manifest(50)
    first = make_inner_fold_spec(manifest, labels, 0, 0.1, 42)
    changed = labels.copy()
    changed.loc[
        changed["perturbation_gene"].isin(first.test_genes),
        "depmap_gene_effect",
    ] = 999.0
    second = make_inner_fold_spec(manifest, changed, 0, 0.1, 42)
    assert first.train_genes == second.train_genes
    assert first.val_genes == second.val_genes


@pytest.mark.parametrize(
    "source_kind",
    ["gene_effect", "gwps_response", "transition", "prompt", "fine_tuning"],
)
def test_every_gene_derived_row_inherits_canonical_outer_fold(
    source_kind: str,
) -> None:
    manifest = pd.DataFrame({"perturbation_gene": ["A", "B"], "outer_fold": [0, 1]})
    rows = pd.DataFrame({"gene": ["a", "A", "b"]})
    result = attach_gene_provenance(rows, manifest, "gene", source_kind)
    assert result[["perturbation_gene", "outer_fold"]].to_dict("records") == [
        {"perturbation_gene": "A", "outer_fold": 0},
        {"perturbation_gene": "A", "outer_fold": 0},
        {"perturbation_gene": "B", "outer_fold": 1},
    ]
    assert set(result["source_kind"]) == {source_kind}


def test_gene_provenance_rejects_missing_and_conflicting_fold() -> None:
    manifest = pd.DataFrame({"perturbation_gene": ["A", "B"], "outer_fold": [0, 1]})
    with pytest.raises(ValueError, match="canonical manifest"):
        attach_gene_provenance(
            pd.DataFrame({"gene": ["C"]}), manifest, "gene", "prompt"
        )
    with pytest.raises(ValueError, match="conflicting outer_fold"):
        attach_gene_provenance(
            pd.DataFrame({"gene": ["A"], "outer_fold": [1]}),
            manifest,
            "gene",
            "prompt",
        )


@pytest.mark.parametrize(
    "stage",
    [
        "adapter_fit",
        "state_fit",
        "response_encoder_fit",
        "gmm_fit",
        "c_head_fit",
        "transition_supervision",
        "gene_prompt_fit",
        "fine_tuning",
    ],
)
def test_outer_test_gene_is_rejected_from_every_fit_stage(stage: str) -> None:
    fold = _toy_fold_spec()
    with pytest.raises(ValueError, match="outer-test"):
        assert_gene_access(stage, [fold.test_genes[0]], fold, checkpoint_frozen=False)


def test_selection_accepts_only_inner_validation_genes() -> None:
    fold = _toy_fold_spec()
    assert_gene_access(
        "early_stopping_prediction_only",
        fold.val_genes,
        fold,
        checkpoint_frozen=False,
    )
    with pytest.raises(ValueError, match="inner-validation"):
        assert_gene_access("layer_selection", fold.train_genes, fold, False)


def test_outer_test_response_routes_are_disabled() -> None:
    fold = _toy_fold_spec()
    gene = fold.test_genes[0]
    for stage in (
        "generation_quality_outer_test",
        "observed_b_shared_oracle_outer_test",
    ):
        with pytest.raises(ValueError, match="unknown gene-access stage"):
            assert_gene_access(stage, [gene], fold, checkpoint_frozen=True)
