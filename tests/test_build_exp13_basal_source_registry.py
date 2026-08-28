"""Tests for the fail-closed Exp13 raw-UMI registry builder."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from aivc_model.geneeffect_data import load_exp13_split, load_source_registry
from scripts.build_exp13_basal_source_registry import build_registry


SPLIT_PATH = Path("configs/benchmarks/cell_line_geneeffect_226_split.json")


def _sources(tmp_path: Path) -> tuple[Path, Path, Path, tuple[str, ...]]:
    split = load_exp13_split(SPLIT_PATH)
    ids = split.all_model_ids
    kinker = tmp_path / "kinker_umi_152" / "h5ad"
    raw_27 = tmp_path / "raw_umi_27" / "h5ad"
    original_dir = tmp_path / "original"
    kinker.mkdir(parents=True)
    raw_27.mkdir(parents=True)
    original_dir.mkdir()
    for model_id in ids[:152]:
        (kinker / f"{model_id}.h5ad").write_bytes(b"kinker")
    for model_id in ids[152:179]:
        (raw_27 / f"{model_id}.h5ad").write_bytes(b"raw-27")
    original_rows = []
    for model_id in ids[179:]:
        source = original_dir / f"{model_id}.h5ad"
        source.write_bytes(b"original")
        original_rows.append(
            {
                "model_id": model_id,
                "source_path": source.name,
                "source_kind": "h5ad",
                "matrix_semantics": "raw_umi_counts",
            }
        )
    original_registry = original_dir / "registry.csv"
    pd.DataFrame(original_rows).to_csv(original_registry, index=False)
    return kinker, raw_27, original_registry, ids


def test_builds_exact_fresh_registry_in_split_order(tmp_path: Path) -> None:
    kinker, raw_27, original, ids = _sources(tmp_path)
    output = tmp_path / "configs" / "registry.csv"
    report = build_registry(SPLIT_PATH, kinker, raw_27, original, output)
    assert report["status"] == "passed"
    assert report["source_counts"] == {
        "kinker_umi_152": 152,
        "raw_umi_27": 27,
        "original_47": 47,
    }
    frame = pd.read_csv(output)
    assert tuple(frame.columns) == (
        "model_id",
        "source_path",
        "source_kind",
        "matrix_semantics",
    )
    assert tuple(frame["model_id"]) == ids
    assert set(frame["source_kind"]) == {"h5ad"}
    assert set(frame["matrix_semantics"]) == {"raw_umi_counts"}
    assert len(load_source_registry(output, load_exp13_split(SPLIT_PATH))) == 226

    with pytest.raises(FileExistsError, match="overwrite"):
        build_registry(SPLIT_PATH, kinker, raw_27, original, output)


def test_dry_run_reports_missing_and_extra_without_writing(tmp_path: Path) -> None:
    kinker, raw_27, original, ids = _sources(tmp_path)
    (kinker / f"{ids[0]}.h5ad").rename(kinker / "ACH-999999.h5ad")
    output = tmp_path / "registry.csv"
    report = build_registry(SPLIT_PATH, kinker, raw_27, original, output, dry_run=True)
    assert report["status"] == "failed"
    assert report["missing_model_ids"] == [ids[0]]
    assert report["extra_model_ids"] == ["ACH-999999"]
    assert not output.exists()

    with pytest.raises(ValueError, match="audit failed"):
        build_registry(SPLIT_PATH, kinker, raw_27, original, output)


def test_rejects_missing_source_and_non_raw_original_registry(tmp_path: Path) -> None:
    kinker, raw_27, original, _ = _sources(tmp_path)
    frame = pd.read_csv(original)
    missing_path = original.parent / frame.loc[0, "source_path"]
    missing_path.unlink()
    report = build_registry(
        SPLIT_PATH,
        kinker,
        raw_27,
        original,
        tmp_path / "registry.csv",
        dry_run=True,
    )
    assert report["missing_source_paths"] == [str(missing_path.resolve())]

    frame.loc[0, "matrix_semantics"] = "processed_cpm"
    frame.to_csv(original, index=False)
    with pytest.raises(ValueError, match="non-raw-UMI"):
        build_registry(
            SPLIT_PATH,
            kinker,
            raw_27,
            original,
            tmp_path / "registry.csv",
            dry_run=True,
        )


def test_optional_obs_model_id_verification_is_fail_closed(tmp_path: Path) -> None:
    kinker, raw_27, original, ids = _sources(tmp_path)
    calls: list[Path] = []

    def reader(path: Path) -> SimpleNamespace:
        calls.append(path)
        model_id = path.stem
        if model_id == ids[-1]:
            model_id = "ACH-WRONG"
        return SimpleNamespace(obs=pd.DataFrame({"model_id": [model_id]}))

    report = build_registry(
        SPLIT_PATH,
        kinker,
        raw_27,
        original,
        tmp_path / "registry.csv",
        dry_run=True,
        verify_obs_model_id=True,
        reader=reader,
    )
    assert len(calls) == 226
    assert report["status"] == "failed"
    assert report["obs_model_id_discrepancies"] == [
        f"{ids[-1]}: AnnData obs.model_id values are ['ACH-WRONG']"
    ]
