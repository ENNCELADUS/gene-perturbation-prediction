"""Tests for the Exp13 source-bound Tx1 cache CLI wiring."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd

import scripts.build_exp13_tx1_cache as cli
from aivc_model.gene_splits import sha256_file


def test_verify_binds_registry_source_bytes(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "ACH-A.h5ad"
    source.write_bytes(b"raw-umi-source")
    registry = pd.DataFrame(
        {
            "source_path": [str(source)],
            "source_kind": ["h5ad"],
            "matrix_semantics": ["raw_umi_counts"],
        },
        index=pd.Index(["ACH-A"], name="model_id"),
    )
    observed = {}

    def fake_verify(path: Path, **kwargs):
        observed.update(kwargs)
        return {"status": "verified"}

    monkeypatch.setattr(cli, "verify_cache", fake_verify)
    source_manifest = {"model_revision": "1" * 40}
    monkeypatch.setattr(
        cli,
        "authenticate_tx1_registration",
        lambda path: (source_manifest, "f" * 64),
    )
    report = cli._verify(
        SimpleNamespace(
            cache_dir=tmp_path / "cache",
            tx1_registration=tmp_path / "registration.json",
        ),
        registry,
    )
    assert report["status"] == "verified"
    assert observed == {
        "expected_model_ids": ("ACH-A",),
        "expected_source_sha256": {"ACH-A": sha256_file(source)},
        "expected_matrix_semantics": "raw_umi_counts",
        "expected_tx1_source_manifest": source_manifest,
    }
