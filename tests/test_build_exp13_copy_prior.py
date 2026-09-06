"""Tests for the authenticated Exp13 K562 copy-prior materializer."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from src.data.geneeffect import ScoredUniverse, restrict_scored_universe_to_copy_prior
from src.data.prepare import build_exp13_copy_prior as builder


SPLIT_PATH = Path("configs/benchmarks/cell_line_geneeffect_226_split.json")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source(path: Path) -> Path:
    pd.DataFrame(
        {
            "TP53 (7157)": [-0.5],
            "KRAS (3845)": [None],
            "MYC (4609)": [-1.25],
        },
        index=pd.Index([builder.DONOR_MODEL_ID], name="ModelID"),
    ).to_csv(path)
    return path


def _materialized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, Path]:
    source = _source(tmp_path / "CRISPRGeneEffect.csv")
    monkeypatch.setattr(builder, "PINNED_GENE_EFFECT_SHA256", _sha256(source))
    expected_output = "gene_symbol,gene_effect\nTP53,-0.5\nMYC,-1.25\n".encode("utf-8")
    expected_output_sha256 = hashlib.sha256(expected_output).hexdigest()
    monkeypatch.setattr(builder, "PINNED_COPY_PRIOR_SHA256", expected_output_sha256)
    output = tmp_path / "copy_prior.csv"
    manifest = tmp_path / "copy_prior_manifest.json"
    builder.materialize_copy_prior(source, SPLIT_PATH, output, manifest)
    return source, output, manifest


def _config(source: Path, output: Path, manifest: Path) -> SimpleNamespace:
    return SimpleNamespace(
        paths=SimpleNamespace(
            gene_effect=source,
            split=SPLIT_PATH,
            copy_prior=output,
            copy_prior_manifest=manifest,
        )
    )


def _labels() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model_id": ["ACH-000551", "ACH-000551", "ACH-000551"],
            "gene_symbol": ["TP53", "KRAS", "MYC"],
            "gene_effect": [-0.5, None, -1.25],
        }
    )


def test_materializes_train_side_k562_row_with_exact_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, output, manifest_path = _materialized(tmp_path, monkeypatch)

    frame = pd.read_csv(output)
    assert frame.to_dict("records") == [
        {"gene_symbol": "TP53", "gene_effect": -0.5},
        {"gene_symbol": "MYC", "gene_effect": -1.25},
    ]
    manifest = json.loads(manifest_path.read_text())
    assert manifest["schema_version"] == "exp13-copy-prior-v1"
    assert manifest["donor"] == {
        "model_id": "ACH-000551",
        "split": "train",
        "unlabeled": False,
    }
    assert manifest["source"]["sha256"] == _sha256(source)
    assert manifest["split"]["sha256"] == builder.PINNED_SPLIT_SHA256
    assert manifest["output"]["sha256"] == _sha256(output)
    assert manifest["counts"] == {
        "source_gene_count": 3,
        "output_gene_count": 2,
        "dropped_gene_count": 1,
    }
    assert manifest["drop_reason_counts"] == {"missing_gene_effect": 1}
    assert manifest["donor_missing"]["count"] == 1
    assert manifest["donor_missing"]["symbols"] == ["KRAS"]

    with pytest.raises(FileExistsError, match="overwrite"):
        builder.materialize_copy_prior(source, SPLIT_PATH, output, manifest_path)


def test_materializer_rejects_wrong_depmap_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source(tmp_path / "CRISPRGeneEffect.csv")
    monkeypatch.setattr(builder, "PINNED_GENE_EFFECT_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="GeneEffect SHA-256 mismatch"):
        builder.materialize_copy_prior(
            source, SPLIT_PATH, tmp_path / "prior.csv", tmp_path / "manifest.json"
        )


def test_materializer_rejects_unpinned_output_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source(tmp_path / "CRISPRGeneEffect.csv")
    monkeypatch.setattr(builder, "PINNED_GENE_EFFECT_SHA256", _sha256(source))
    with pytest.raises(ValueError, match="pinned artifact SHA-256"):
        builder.materialize_copy_prior(
            source, SPLIT_PATH, tmp_path / "prior.csv", tmp_path / "manifest.json"
        )


def test_materializer_rejects_aliased_output_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source(tmp_path / "CRISPRGeneEffect.csv")
    monkeypatch.setattr(builder, "PINNED_GENE_EFFECT_SHA256", _sha256(source))
    output = tmp_path / "copy_prior"
    with pytest.raises(ValueError, match="must be distinct"):
        builder.materialize_copy_prior(source, SPLIT_PATH, output, output)


def test_one_universe_gate_drops_copy_prior_missing_for_every_method() -> None:
    coverage = pd.DataFrame(
        {
            "gene_symbol": ["A", "B", "C"],
            "included": [True, True, False],
            "drop_reason": ["", "", "train_finite_lt5"],
        }
    )
    universe = ScoredUniverse(
        symbols=("A", "B"),
        coverage=coverage,
        manifest={"scored_gene_count": 2, "scored_symbols": ["A", "B"]},
    )

    restricted = restrict_scored_universe_to_copy_prior(universe, ("B",))

    assert restricted.symbols == ("B",)
    assert restricted.manifest["pre_copy_prior_gene_count"] == 2
    assert restricted.manifest["copy_prior_missing_count"] == 1
    drops = restricted.coverage.set_index("gene_symbol")["drop_reason"]
    assert drops["A"] == "copy_prior_missing"
    assert drops["C"] == "train_finite_lt5"


def test_output_only_crash_state_is_safely_recoverable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, output, manifest = _materialized(tmp_path, monkeypatch)
    manifest.unlink()

    recovered = builder.materialize_copy_prior(source, SPLIT_PATH, output, manifest)

    assert manifest.is_file()
    assert recovered["output"]["sha256"] == _sha256(output)
