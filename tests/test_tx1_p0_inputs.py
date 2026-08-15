"""Tests for authenticated Tx1 P0 input materialization."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import aivc_model.tx1_p0_inputs as inputs_module
from aivc_model.tx1_p0_inputs import build_p0_inputs, write_p0_inputs
from conftest import tx1_manifest_row, write_tx1_line_manifest


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> dict[str, Path]:
    phase_a_dir = tmp_path / "phase_a"
    phase_a_dir.mkdir()
    manifest_path = phase_a_dir / "cell_line_manifest.csv"
    train_ids = [f"ACH-H{index:04d}" for index in range(29)]
    rows = [
        tx1_manifest_row(model_id=model_id, role="train_head") for model_id in train_ids
    ]
    rows.extend(
        tx1_manifest_row(
            model_id=f"ACH-A{index:04d}",
            basal_source="Perturb-seq non-targeting control",
            role="train_response_and_head",
        )
        for index in range(4)
    )
    rows[29]["model_id"] = "ACH-000551"
    rows.extend(
        tx1_manifest_row(model_id=f"ACH-T{index:04d}", role="test")
        for index in range(9)
    )
    write_tx1_line_manifest(manifest_path, rows)

    genes = [f"G{index:03d}" for index in range(587)]
    depmap_columns = [f"{gene} ({index})" for index, gene in enumerate(genes)]
    slice_path = phase_a_dir / "differentially_essential_slice.csv"
    pd.DataFrame({"depmap_column": depmap_columns, "gene_symbol": genes}).to_csv(
        slice_path, index=False
    )
    (phase_a_dir / "k_label_panels.csv").write_text("model_id\n", encoding="utf-8")

    gene_effect_path = tmp_path / "CRISPRGeneEffect.csv"
    all_ids = [*train_ids, "ACH-000551", *[f"ACH-T{i:04d}" for i in range(9)]]
    gene_values = np.vstack(
        [
            np.arange(587, dtype=float) + line_index / 100.0
            for line_index in range(len(all_ids))
        ]
    )
    gene_values[-9:, :] = 1e9  # test labels must never enter a training output
    pd.DataFrame(gene_values, index=all_ids, columns=depmap_columns).to_csv(
        gene_effect_path
    )
    expression_path = tmp_path / "expression.csv"
    pd.DataFrame(
        {
            "E1": np.arange(29, dtype=float),
            "E2": np.arange(29, dtype=float) + 0.5,
        },
        index=train_ids,
    ).to_csv(expression_path)
    registration = {
        "sources": {
            "depmap_gene_effect": {"sha256": _sha(gene_effect_path)},
            "depmap_omics_expression": {"sha256": _sha(expression_path)},
        }
    }
    (phase_a_dir / "phase_a_registration.json").write_text(
        json.dumps(registration), encoding="utf-8"
    )

    cache_root = tmp_path / "cache"
    for index, model_id in enumerate(train_ids):
        line_dir = cache_root / model_id
        line_dir.mkdir(parents=True)
        np.save(
            line_dir / "embeddings.npy",
            np.array([[index, 1.0, 2.0], [index + 2.0, 3.0, 4.0]]),
        )
        np.save(
            line_dir / "hvg.npy",
            np.array([[index, 5.0], [index + 4.0, 7.0]]),
        )
    return {
        "phase_a_dir": phase_a_dir,
        "manifest_path": manifest_path,
        "gene_effect_path": gene_effect_path,
        "expression_path": expression_path,
        "cache_root": cache_root,
    }


@pytest.fixture(autouse=True)
def _authenticated_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(inputs_module, "verify_artifact_hashes", lambda _: None)
    monkeypatch.setattr(
        inputs_module,
        "verify_cache",
        lambda *_args, **_kwargs: {"status": "verified", "discrepancies": []},
    )


def test_materializes_only_train_head_labels_prior_and_mean_context(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    result = build_p0_inputs(**paths)

    assert result.gene_effect_long.shape == (29 * 587, 3)
    assert set(result.gene_effect_long["model_id"]) == {
        f"ACH-H{index:04d}" for index in range(29)
    }
    assert not result.gene_effect_long["model_id"].str.startswith("ACH-T").any()
    assert result.gene_effect_long["gene_effect"].max() < 1e9
    assert result.copy_k562_prior.shape == (587, 2)
    first = result.line_context.set_index("model_id").loc["ACH-H0000"]
    assert first["tx1_mean_0000"] == pytest.approx(1.0)
    assert first["tx1_std_0000"] == pytest.approx(1.0)
    assert first["hvg_mean_0000"] == pytest.approx(2.0)
    assert first["hvg_std_0000"] == pytest.approx(2.0)
    assert first["expression__E1"] == pytest.approx(0.0)


def test_gene_effect_missing_train_line_is_rejected(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    frame = pd.read_csv(paths["gene_effect_path"], index_col=0).iloc[1:]
    frame.to_csv(paths["gene_effect_path"])
    registration_path = paths["phase_a_dir"] / "phase_a_registration.json"
    registration = json.loads(registration_path.read_text())
    registration["sources"]["depmap_gene_effect"]["sha256"] = _sha(
        paths["gene_effect_path"]
    )
    registration_path.write_text(json.dumps(registration))

    with pytest.raises(ValueError, match="missing rows"):
        build_p0_inputs(**paths)


def test_nonfinite_cache_and_unverified_cache_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path)
    bad = np.load(paths["cache_root"] / "ACH-H0000" / "hvg.npy")
    bad[0, 0] = np.nan
    np.save(paths["cache_root"] / "ACH-H0000" / "hvg.npy", bad)
    with pytest.raises(ValueError, match="non-finite"):
        build_p0_inputs(**paths)

    monkeypatch.setattr(
        inputs_module,
        "verify_cache",
        lambda *_args, **_kwargs: {"status": "failed", "discrepancies": ["bad"]},
    )
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    with pytest.raises(ValueError, match="verification failed"):
        build_p0_inputs(**_fixture(fresh))


def test_expression_requires_complete_train_head_coverage(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    expression = pd.read_csv(paths["expression_path"], index_col=0).iloc[1:]
    expression.to_csv(paths["expression_path"])
    registration_path = paths["phase_a_dir"] / "phase_a_registration.json"
    registration = json.loads(registration_path.read_text())
    registration["sources"]["depmap_omics_expression"]["sha256"] = _sha(
        paths["expression_path"]
    )
    registration_path.write_text(json.dumps(registration))
    with pytest.raises(ValueError, match="every train_head"):
        build_p0_inputs(**paths)


def test_expression_supports_depmap_default_model_schema(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    train_ids = [f"ACH-H{index:04d}" for index in range(29)]
    defaults = pd.DataFrame(
        {
            "Unnamed: 0": np.arange(29),
            "SequencingID": [f"SEQ-{index}" for index in range(29)],
            "ModelID": train_ids,
            "IsDefaultEntryForModel": True,
            "E1 (1)": np.arange(29, dtype=float),
            "E2 (2)": np.arange(29, dtype=float) + 0.5,
        }
    )
    non_default_before = defaults.iloc[[0]].copy()
    non_default_before["SequencingID"] = "SEQ-non-default-before"
    non_default_before["IsDefaultEntryForModel"] = False
    non_default_before[["E1 (1)", "E2 (2)"]] = 999.0
    non_default_after = defaults.iloc[[0]].copy()
    non_default_after["SequencingID"] = "SEQ-non-default-after"
    non_default_after["IsDefaultEntryForModel"] = False
    non_default_after[["E1 (1)", "E2 (2)"]] = -999.0
    expression = pd.concat(
        [non_default_before, defaults, non_default_after], ignore_index=True
    )
    expression.to_csv(paths["expression_path"], index=False)
    registration_path = paths["phase_a_dir"] / "phase_a_registration.json"
    registration = json.loads(registration_path.read_text())
    registration["sources"]["depmap_omics_expression"]["sha256"] = _sha(
        paths["expression_path"]
    )
    registration_path.write_text(json.dumps(registration))

    result = build_p0_inputs(**paths)

    assert result.line_context["expression__E1 (1)"].tolist() == pytest.approx(
        np.arange(29, dtype=float)
    )


def test_expression_rejects_duplicate_default_rows(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    train_ids = [f"ACH-H{index:04d}" for index in range(29)]
    expression = pd.DataFrame(
        {
            "ModelID": [*train_ids, train_ids[0]],
            "IsDefaultEntryForModel": True,
            "E1 (1)": np.arange(30, dtype=float),
        }
    )
    expression.to_csv(paths["expression_path"], index=False)
    registration_path = paths["phase_a_dir"] / "phase_a_registration.json"
    registration = json.loads(registration_path.read_text())
    registration["sources"]["depmap_omics_expression"]["sha256"] = _sha(
        paths["expression_path"]
    )
    registration_path.write_text(json.dumps(registration))

    with pytest.raises(ValueError, match="duplicate default ModelID"):
        build_p0_inputs(**paths)


def test_writes_deterministic_atomic_artifacts(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    first = build_p0_inputs(**paths)
    second = build_p0_inputs(**paths)
    pd.testing.assert_frame_equal(first.gene_effect_long, second.gene_effect_long)
    assert first.provenance == second.provenance

    output = tmp_path / "output"
    write_p0_inputs(first, output)
    provenance = json.loads((output / "provenance.json").read_text())
    assert set(provenance["output_sha256"]) == {
        "gene_effect_long.csv",
        "copy_k562_prior.csv",
        "line_context.csv",
        "tx1_context.csv",
        "hvg_context.csv",
        "tx1_moments_context.csv",
        "hvg_moments_context.csv",
        "ccle_expression_context.csv",
        "multiview_context.csv",
    }
    tx1 = pd.read_csv(output / "tx1_context.csv")
    hvg = pd.read_csv(output / "hvg_context.csv")
    assert tx1.columns.tolist() == [
        "model_id",
        "tx1_mean_0000",
        "tx1_mean_0001",
        "tx1_mean_0002",
    ]
    assert hvg.columns.tolist() == ["model_id", "hvg_mean_0000", "hvg_mean_0001"]
    tx1_moments = pd.read_csv(output / "tx1_moments_context.csv")
    hvg_moments = pd.read_csv(output / "hvg_moments_context.csv")
    assert tx1_moments.shape[1] == 1 + 2 * 3
    assert hvg_moments.shape[1] == 1 + 2 * 2
    assert tx1["model_id"].tolist() == hvg["model_id"].tolist()
    with pytest.raises(FileExistsError):
        write_p0_inputs(first, output)
