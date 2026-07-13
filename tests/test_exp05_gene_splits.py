"""Tests for the immutable exp05 canonical outer-fold manifest."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aivc_model.gene_splits import (
    build_canonical_outer_manifest,
    load_canonical_outer_manifest,
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
