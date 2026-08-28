"""Tests for the strict Exp13 q_sc cache CLI."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from aivc_model.geneeffect_data import Exp13Split
from scripts import build_exp13_q_sc_cache as cli


def _split() -> Exp13Split:
    return Exp13Split(
        train=(
            "ACH-000551",
            "T2",
            "T3",
            "T4",
            "T5",
            "ACH-000779",
            "ACH-001086",
        ),
        val=("V1", "V2", "V3"),
        test=("E1", "E2", "E3"),
        unlabeled_train=("ACH-000779", "ACH-001086"),
    )


def _inputs(tmp_path: Path) -> dict[str, Path]:
    split = tmp_path / "split.json"
    split.write_text("{}")
    ids = [*_split().supervised_train, *_split().val, *_split().test]
    gene_effect = tmp_path / "gene_effect.csv"
    frame = pd.DataFrame(
        {
            "A (1)": np.arange(len(ids), dtype=float),
            "B (2)": [*range(5), 1.0, 2.0, 3.0, 1.0, np.nan, np.nan],
        },
        index=ids,
    )
    frame.to_csv(gene_effect)
    esm2 = tmp_path / "esm2.npz"
    np.savez(
        esm2,
        symbols=np.asarray(["B", "A"], dtype=object),
        vectors=np.ones((2, 4), dtype=np.float32),
        resolved=np.asarray([True, True]),
    )
    copy_prior = tmp_path / "copy_prior.csv"
    copy_prior.write_text("gene_symbol,gene_effect\nA,0.0\nB,0.0\n")
    copy_prior_manifest = tmp_path / "copy_prior_manifest.json"
    copy_prior_manifest.write_text(
        json.dumps(
            {
                "schema_version": "exp13-copy-prior-v1",
                "donor": {
                    "model_id": "ACH-000551",
                    "split": "train",
                    "unlabeled": False,
                },
                "source": {"sha256": cli._sha256(gene_effect)},
                "split": {"sha256": cli._sha256(split)},
                "output": {"sha256": cli._sha256(copy_prior)},
            }
        )
    )
    esm2_provenance = tmp_path / "esm2.provenance.json"
    provenance_payload = {"authenticated": True}
    esm2_provenance.write_text(json.dumps(provenance_payload))
    esm2_universe = tmp_path / "esm2_universe.json"
    esm2_universe.write_text(
        json.dumps(
            {
                "schema_version": "exp13-esm2-universes-v2",
                "status": "authenticated_complete",
                "scored_symbols": ["A"],
                "scored_gene_count": 1,
                "coverage_thresholds": {"train": 5, "val": 3, "test": 3},
                "input_sha256": {
                    "split": cli._sha256(split),
                    "gene_effect": cli._sha256(gene_effect),
                    "copy_prior": cli._sha256(copy_prior),
                    "copy_prior_manifest": cli._sha256(copy_prior_manifest),
                },
                "copy_prior_eligible_candidates": {"symbols": ["A"]},
                "final_evaluated_universe": {
                    "symbols": ["A"],
                    "count": 1,
                    "symbols_sha256": cli._symbols_sha256(("A",)),
                    "unresolved_candidate_symbols": [],
                    "unresolved_candidate_count": 0,
                },
                "embedding_union": {
                    "verified_npz": {"artifact_sha256": cli._sha256(esm2)},
                    "provenance_manifest": {
                        "sha256": cli._sha256(esm2_provenance),
                        "payload": provenance_payload,
                    },
                },
            }
        )
    )
    source_dir = tmp_path / "sources"
    source_dir.mkdir()
    rows = []
    for model_id in _split().all_model_ids:
        source = source_dir / f"{model_id}.h5ad"
        source.write_bytes(model_id.encode())
        rows.append(
            {
                "model_id": model_id,
                "source_path": str(source),
                "source_kind": "h5ad",
                "matrix_semantics": "raw_umi_counts",
            }
        )
    registry = tmp_path / "registry.csv"
    pd.DataFrame(rows).to_csv(registry, index=False)
    return {
        "split": split,
        "gene_effect": gene_effect,
        "esm2": esm2,
        "esm2_universe": esm2_universe,
        "esm2_provenance": esm2_provenance,
        "copy_prior": copy_prior,
        "copy_prior_manifest": copy_prior_manifest,
        "registry": registry,
        "output": tmp_path / "q_sc",
    }


def test_builds_then_unrestricted_verifies_coverage_qualified_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _inputs(tmp_path)
    monkeypatch.setattr(cli, "load_exp13_split", lambda _: _split())
    monkeypatch.setattr(
        cli, "PINNED_GENE_EFFECT_SHA256", cli._sha256(paths["gene_effect"])
    )
    monkeypatch.setattr(
        cli,
        "load_and_authenticate_esm2_provenance",
        lambda *args, **kwargs: {"authenticated": True},
    )
    calls: list[str] = []

    def reader(path: Path) -> SimpleNamespace:
        calls.append(path.stem)
        return SimpleNamespace(
            X=np.asarray([[0, 1], [2, 3]], dtype=int),
            var=pd.DataFrame({"gene_symbol": ["A", "B"]}),
            obs=pd.DataFrame({"model_id": [path.stem, path.stem]}),
        )

    report = cli.build_or_verify_q_sc_cache(
        split_path=paths["split"],
        gene_effect_path=paths["gene_effect"],
        esm2_path=paths["esm2"],
        esm2_universe_manifest_path=paths["esm2_universe"],
        esm2_provenance_manifest_path=paths["esm2_provenance"],
        copy_prior_path=paths["copy_prior"],
        copy_prior_manifest_path=paths["copy_prior_manifest"],
        registry_path=paths["registry"],
        output_dir=paths["output"],
        reader=reader,
    )

    assert report["status"] == "passed"
    assert len(calls) == len(_split().all_model_ids)
    assert report["gene_universe"]["scored_symbols"] == ["A"]
    assert len(report["gene_universe"]["symbols_sha256"]) == 64
    qualification = report["coverage_qualification"]
    assert qualification["thresholds"] == {"train": 5, "val": 3, "test": 3}
    assert qualification["test_labels_role"] == "coverage_qualification_only"
    assert qualification["test_labels_used_for_fit"] is False
    assert qualification["fit_operations"] == []
    assert report["verification"]["lines_present"] == len(_split().all_model_ids)


def test_verify_only_never_builds_and_detects_extra_shard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _inputs(tmp_path)
    monkeypatch.setattr(cli, "load_exp13_split", lambda _: _split())
    monkeypatch.setattr(
        cli, "PINNED_GENE_EFFECT_SHA256", cli._sha256(paths["gene_effect"])
    )
    monkeypatch.setattr(
        cli,
        "load_and_authenticate_esm2_provenance",
        lambda *args, **kwargs: {"authenticated": True},
    )

    def reader(path: Path) -> SimpleNamespace:
        return SimpleNamespace(
            X=np.asarray([[1, 0]], dtype=int),
            var=pd.DataFrame({"gene_symbol": ["A", "B"]}),
            obs=pd.DataFrame({"model_id": [path.stem]}),
        )

    cli.build_or_verify_q_sc_cache(
        split_path=paths["split"],
        gene_effect_path=paths["gene_effect"],
        esm2_path=paths["esm2"],
        esm2_universe_manifest_path=paths["esm2_universe"],
        esm2_provenance_manifest_path=paths["esm2_provenance"],
        copy_prior_path=paths["copy_prior"],
        copy_prior_manifest_path=paths["copy_prior_manifest"],
        registry_path=paths["registry"],
        output_dir=paths["output"],
        reader=reader,
    )
    np.savez(paths["output"] / "EXTRA.npz", x=np.asarray([1]))
    monkeypatch.setattr(
        cli,
        "build_q_sc_shards",
        lambda *args, **kwargs: pytest.fail("verify-only attempted to build"),
    )

    report = cli.build_or_verify_q_sc_cache(
        split_path=paths["split"],
        gene_effect_path=paths["gene_effect"],
        esm2_path=paths["esm2"],
        esm2_universe_manifest_path=paths["esm2_universe"],
        esm2_provenance_manifest_path=paths["esm2_provenance"],
        copy_prior_path=paths["copy_prior"],
        copy_prior_manifest_path=paths["copy_prior_manifest"],
        registry_path=paths["registry"],
        output_dir=paths["output"],
        verify_only=True,
    )

    assert report["status"] == "failed"
    assert report["mode"] == "verify_only"
    assert "extra shard: EXTRA.npz" in report["verification"]["discrepancies"]


def test_main_writes_json_report_and_returns_nonzero_on_failed_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report_path = tmp_path / "reports" / "q_sc.json"
    failed = {
        "schema_version": cli.SCHEMA_VERSION,
        "status": "failed",
        "verification": {"discrepancies": ["missing shard: A.npz"]},
    }
    monkeypatch.setattr(cli, "build_or_verify_q_sc_cache", lambda **_: failed)

    code = cli.main(
        [
            "--split",
            "split.json",
            "--gene-effect",
            "labels.csv",
            "--esm2",
            "esm2.npz",
            "--esm2-universe-manifest",
            "esm2_universe.json",
            "--esm2-provenance-manifest",
            "esm2.provenance.json",
            "--copy-prior",
            "copy_prior.csv",
            "--copy-prior-manifest",
            "copy_prior_manifest.json",
            "--registry",
            "registry.csv",
            "--output-dir",
            "q_sc",
            "--report",
            str(report_path),
            "--verify-only",
        ]
    )

    assert code == 1
    assert json.loads(report_path.read_text()) == failed


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("final_evaluated_universe", "count", 2),
        ("final_evaluated_universe", "symbols_sha256", "0" * 64),
        ("final_evaluated_universe", "unresolved_candidate_symbols", ["B"]),
        (None, "scored_gene_count", 2),
        ("coverage_thresholds", "train", 6),
    ],
)
def test_rejects_tampered_final_universe_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    section: str | None,
    key: str,
    value: object,
) -> None:
    paths = _inputs(tmp_path)
    monkeypatch.setattr(cli, "load_exp13_split", lambda _: _split())
    monkeypatch.setattr(
        cli, "PINNED_GENE_EFFECT_SHA256", cli._sha256(paths["gene_effect"])
    )
    monkeypatch.setattr(
        cli,
        "load_and_authenticate_esm2_provenance",
        lambda *args, **kwargs: {"authenticated": True},
    )
    payload = json.loads(paths["esm2_universe"].read_text())
    target = payload if section is None else payload[section]
    target[key] = value
    paths["esm2_universe"].write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="final symbols"):
        cli.build_or_verify_q_sc_cache(
            split_path=paths["split"],
            gene_effect_path=paths["gene_effect"],
            esm2_path=paths["esm2"],
            esm2_universe_manifest_path=paths["esm2_universe"],
            esm2_provenance_manifest_path=paths["esm2_provenance"],
            copy_prior_path=paths["copy_prior"],
            copy_prior_manifest_path=paths["copy_prior_manifest"],
            registry_path=paths["registry"],
            output_dir=paths["output"],
            verify_only=True,
        )


def test_rejects_duplicate_resolved_esm2_symbols(tmp_path: Path) -> None:
    path = tmp_path / "esm2.npz"
    np.savez(
        path,
        symbols=np.asarray(["A", "A"], dtype=object),
        vectors=np.ones((2, 3), dtype=np.float32),
        resolved=np.asarray([True, True]),
    )
    with pytest.raises(ValueError, match="duplicated"):
        cli._load_resolved_esm2_symbols(path)
