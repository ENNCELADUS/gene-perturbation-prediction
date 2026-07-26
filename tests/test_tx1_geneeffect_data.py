"""Tests for src/aivc_model/tx1_geneeffect_data.py -- Phase D Task 1: line-role
selection, the derived validation-line registration, and the DepMap
GeneEffect label loader.

No GPU and no real DepMap/Tx1 data exist on this machine, so every test
builds tiny synthetic fixtures in ``tmp_path``: a frozen-manifest-shaped CSV
(via the shared ``conftest`` helpers) and a fake ``CRISPRGeneEffect.csv``.
Nothing here is allowed to skip silently.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from aivc_model.gene_splits import sha256_file
from aivc_model.tx1_basal import load_line_manifest
from aivc_model.tx1_geneeffect_data import (
    TEST_ROLE,
    TRAIN_HEAD_ROLE,
    TRAIN_RESPONSE_ROLE,
    assert_training_role,
    build_line_role_split,
    generate_validation_line_registration,
    load_depmap_gene_effect,
    load_validation_line_registration,
    select_line_roles,
)
from conftest import tx1_manifest_row as _manifest_row
from conftest import write_tx1_line_manifest as _write_manifest


# --- synthetic manifest builder ---------------------------------------------
#
# Mirrors the real frozen manifest's shape (29 train_head + 4
# train_response_and_head + 9 test = 42) closely enough to exercise the
# 28/5/9 split, but with a simple 5-lineage train_head structure (sizes
# 6/6/6/6/5) so the stratified allocation is easy to reason about by hand:
# each of the 5 lineages gets exactly one validation slot.


def _synthetic_manifest(path: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    lineage_sizes = {"LinA": 6, "LinB": 6, "LinC": 6, "LinD": 6, "LinE": 5}
    counter = 0
    for lineage, size in lineage_sizes.items():
        for _ in range(size):
            rows.append(
                _manifest_row(
                    model_id=f"ACH-H{counter:04d}",
                    lineage=lineage,
                    role=TRAIN_HEAD_ROLE,
                )
            )
            counter += 1
    anchor_ids = ["ACH-A0001", "ACH-A0002", "ACH-A0003", "ACH-A0004"]
    for model_id in anchor_ids:
        rows.append(
            _manifest_row(
                model_id=model_id,
                lineage="Anchor",
                basal_source="Perturb-seq non-targeting control",
                role=TRAIN_RESPONSE_ROLE,
            )
        )
    for index in range(9):
        rows.append(
            _manifest_row(
                model_id=f"ACH-T{index:04d}", lineage="TestLineage", role=TEST_ROLE
            )
        )
    _write_manifest(path, rows)
    return load_line_manifest(path)


# --- select_line_roles / assert_training_role -------------------------------


def test_select_line_roles_splits_combined_training_pool_and_test(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "cell_line_manifest.csv"
    manifest = _synthetic_manifest(manifest_path)

    training, test = select_line_roles(manifest)

    assert len(training) == 29 + 4  # train_head + train_response_and_head
    assert len(test) == 9
    assert set(training["model_id"]) & set(test["model_id"]) == set()
    assert len(training) + len(test) == len(manifest)


def test_select_line_roles_rejects_unrecognized_role(tmp_path: Path) -> None:
    manifest_path = tmp_path / "cell_line_manifest.csv"
    manifest = _synthetic_manifest(manifest_path)
    corrupted = manifest.copy()
    corrupted.loc[0, "role"] = "mystery_role"

    with pytest.raises(ValueError, match="unrecognized role"):
        select_line_roles(corrupted)


@pytest.mark.parametrize("role", [TRAIN_HEAD_ROLE, TRAIN_RESPONSE_ROLE])
def test_assert_training_role_accepts_training_roles(role: str) -> None:
    assert_training_role(role, "ACH-000001")  # must not raise


def test_assert_training_role_rejects_test_role() -> None:
    """A test-role line requesting entry to the training path must raise."""
    with pytest.raises(ValueError, match="refusing to admit role='test'"):
        assert_training_role(TEST_ROLE, "ACH-000178")


def test_assert_training_role_rejects_arbitrary_role() -> None:
    with pytest.raises(ValueError):
        assert_training_role("not_a_real_role", "ACH-000001")


# --- generate_validation_line_registration / build_line_role_split ---------


def test_generate_validation_line_registration_is_deterministic_and_stratified(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "cell_line_manifest.csv"
    manifest = _synthetic_manifest(manifest_path)

    first = generate_validation_line_registration(manifest, manifest_path)
    second = generate_validation_line_registration(manifest, manifest_path)

    assert first == second
    assert len(first["validation_model_ids"]) == 5
    assert first["derived_from_sha256"] == sha256_file(manifest_path)
    # one validation line drawn from each of the 5 equally-sized lineages
    lineage_by_id = dict(zip(manifest["model_id"], manifest["lineage"]))
    drawn_lineages = {lineage_by_id[mid] for mid in first["validation_model_ids"]}
    assert drawn_lineages == {"LinA", "LinB", "LinC", "LinD", "LinE"}


def test_build_line_role_split_gives_28_5_9(tmp_path: Path) -> None:
    manifest_path = tmp_path / "cell_line_manifest.csv"
    manifest = _synthetic_manifest(manifest_path)
    registration = generate_validation_line_registration(manifest, manifest_path)

    split = build_line_role_split(manifest, registration)

    assert len(split.train_model_ids) == 28
    assert len(split.validation_model_ids) == 5
    assert len(split.test_model_ids) == 9
    assert len(split.anchor_model_ids) == 4


def test_validation_lines_disjoint_from_anchors_and_test(tmp_path: Path) -> None:
    manifest_path = tmp_path / "cell_line_manifest.csv"
    manifest = _synthetic_manifest(manifest_path)
    registration = generate_validation_line_registration(manifest, manifest_path)
    split = build_line_role_split(manifest, registration)

    validation = set(split.validation_model_ids)
    anchors = set(split.anchor_model_ids)
    test = set(split.test_model_ids)
    train_head_ids = set(manifest.loc[manifest["role"] == TRAIN_HEAD_ROLE, "model_id"])

    assert validation & anchors == set()
    assert validation & test == set()
    assert validation.issubset(train_head_ids)
    # anchors always stay in training, never spent on model selection
    assert anchors.issubset(set(split.train_model_ids))


def test_build_line_role_split_rejects_validation_line_overlapping_anchor(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "cell_line_manifest.csv"
    manifest = _synthetic_manifest(manifest_path)
    registration = generate_validation_line_registration(manifest, manifest_path)
    corrupted = dict(registration)
    kept = registration["validation_model_ids"][:-1]
    corrupted["validation_model_ids"] = list(kept) + ["ACH-A0001"]

    with pytest.raises(ValueError, match="overlap the .*anchors"):
        build_line_role_split(manifest, corrupted)


def test_build_line_role_split_rejects_validation_line_overlapping_test(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "cell_line_manifest.csv"
    manifest = _synthetic_manifest(manifest_path)
    registration = generate_validation_line_registration(manifest, manifest_path)
    corrupted = dict(registration)
    kept = registration["validation_model_ids"][:-1]
    corrupted["validation_model_ids"] = list(kept) + ["ACH-T0000"]

    with pytest.raises(ValueError, match="overlap the frozen test lines"):
        build_line_role_split(manifest, corrupted)


# --- load_validation_line_registration: freshness gate ----------------------


def test_load_validation_line_registration_round_trips(tmp_path: Path) -> None:
    manifest_path = tmp_path / "cell_line_manifest.csv"
    manifest = _synthetic_manifest(manifest_path)
    registration = generate_validation_line_registration(manifest, manifest_path)
    registration_path = tmp_path / "validation_lines.json"
    registration_path.write_text(json.dumps(registration))

    loaded = load_validation_line_registration(
        registration_path, manifest, manifest_path
    )

    assert loaded == registration


def test_load_validation_line_registration_raises_on_stale_manifest(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "cell_line_manifest.csv"
    manifest = _synthetic_manifest(manifest_path)
    registration = generate_validation_line_registration(manifest, manifest_path)
    registration_path = tmp_path / "validation_lines.json"
    registration_path.write_text(json.dumps(registration))

    # Mutate the manifest on disk after the registration was derived -- its
    # sha256 must now disagree with the recorded derived_from_sha256.
    rows = pd.read_csv(manifest_path)
    rows.loc[0, "dmso_cells"] = 999999
    rows.to_csv(manifest_path, index=False)
    mutated_manifest = load_line_manifest(manifest_path)

    with pytest.raises(ValueError, match="is stale"):
        load_validation_line_registration(
            registration_path, mutated_manifest, manifest_path
        )


def test_load_validation_line_registration_raises_on_disjointness_violation(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "cell_line_manifest.csv"
    manifest = _synthetic_manifest(manifest_path)
    registration = generate_validation_line_registration(manifest, manifest_path)
    # Hash still matches the manifest (untouched); only the validation ids
    # are corrupted to include a frozen test line.
    registration["validation_model_ids"] = list(
        registration["validation_model_ids"][:-1]
    ) + ["ACH-T0000"]
    registration_path = tmp_path / "validation_lines.json"
    registration_path.write_text(json.dumps(registration))

    with pytest.raises(ValueError, match="overlap the frozen test lines"):
        load_validation_line_registration(registration_path, manifest, manifest_path)


# --- P2-2: a truncated or duplicated registration must not silently pass ----


def test_load_validation_line_registration_rejects_duplicate_id(
    tmp_path: Path,
) -> None:
    """A duplicated id keeps the list length equal to validation_count (5),
    so nothing about the recorded count looks wrong -- only an explicit
    uniqueness check catches it. Without this fix, `_assert_validation_
    disjoint`'s set() conversion would collapse the duplicate invisibly and
    this would silently train with only 4 UNIQUE validation lines."""
    manifest_path = tmp_path / "cell_line_manifest.csv"
    manifest = _synthetic_manifest(manifest_path)
    registration = generate_validation_line_registration(manifest, manifest_path)
    corrupted = dict(registration)
    ids = list(registration["validation_model_ids"])
    corrupted["validation_model_ids"] = [ids[0]] + ids[:-1]  # ids[0] duplicated
    assert len(corrupted["validation_model_ids"]) == registration["validation_count"]
    registration_path = tmp_path / "validation_lines.json"
    registration_path.write_text(json.dumps(corrupted))

    with pytest.raises(ValueError, match="duplicate id"):
        load_validation_line_registration(registration_path, manifest, manifest_path)


def test_load_validation_line_registration_rejects_truncated_list(
    tmp_path: Path,
) -> None:
    """Dropping one id without updating validation_count must raise -- the
    manifest hash is untouched (this is not the staleness case) and every
    remaining id is a genuine, disjoint train_head line, so nothing but an
    explicit count check catches the registration silently training with
    fewer than its own claimed validation-line count."""
    manifest_path = tmp_path / "cell_line_manifest.csv"
    manifest = _synthetic_manifest(manifest_path)
    registration = generate_validation_line_registration(manifest, manifest_path)
    corrupted = dict(registration)
    corrupted["validation_model_ids"] = list(registration["validation_model_ids"])[:-1]
    assert len(corrupted["validation_model_ids"]) < registration["validation_count"]
    registration_path = tmp_path / "validation_lines.json"
    registration_path.write_text(json.dumps(corrupted))

    with pytest.raises(ValueError, match="validation_count"):
        load_validation_line_registration(registration_path, manifest, manifest_path)


def test_load_validation_line_registration_accepts_self_consistent_registration(
    tmp_path: Path,
) -> None:
    """Sanity check: an untouched, genuinely-generated registration (unique
    ids, count matches) must still round-trip -- P2-2's new checks must not
    reject a valid registration."""
    manifest_path = tmp_path / "cell_line_manifest.csv"
    manifest = _synthetic_manifest(manifest_path)
    registration = generate_validation_line_registration(manifest, manifest_path)
    registration_path = tmp_path / "validation_lines.json"
    registration_path.write_text(json.dumps(registration))

    loaded = load_validation_line_registration(
        registration_path, manifest, manifest_path
    )  # must not raise

    assert loaded == registration


# --- load_depmap_gene_effect -------------------------------------------------


def _write_fake_gene_effect_csv(path: Path) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "GENE1 (1)": [-0.1, -1.5],
            "GENE2 (2)": [0.3, -0.2],
        },
        index=["ACH-000001", "ACH-000002"],
    )
    frame.to_csv(path)
    return frame


def test_load_depmap_gene_effect_hash_mismatch_raises(tmp_path: Path) -> None:
    csv_path = tmp_path / "CRISPRGeneEffect.csv"
    _write_fake_gene_effect_csv(csv_path)
    registration = {"sources": {"depmap_gene_effect": {"sha256": "0" * 64}}}

    with pytest.raises(ValueError, match="content-hash resolution"):
        load_depmap_gene_effect(csv_path, registration, ["ACH-000001"])


def test_load_depmap_gene_effect_joins_by_model_id_and_preserves_request_order(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "CRISPRGeneEffect.csv"
    _write_fake_gene_effect_csv(csv_path)
    registration = {
        "sources": {"depmap_gene_effect": {"sha256": sha256_file(csv_path)}}
    }

    result = load_depmap_gene_effect(
        csv_path, registration, ["ACH-000002", "ACH-000001"]
    )

    assert list(result.index) == ["ACH-000002", "ACH-000001"]
    assert result.loc["ACH-000001", "GENE1 (1)"] == pytest.approx(-0.1)
    assert result.loc["ACH-000002", "GENE2 (2)"] == pytest.approx(-0.2)


def test_load_depmap_gene_effect_missing_model_id_raises(tmp_path: Path) -> None:
    csv_path = tmp_path / "CRISPRGeneEffect.csv"
    _write_fake_gene_effect_csv(csv_path)
    registration = {
        "sources": {"depmap_gene_effect": {"sha256": sha256_file(csv_path)}}
    }

    with pytest.raises(ValueError, match="missing rows"):
        load_depmap_gene_effect(csv_path, registration, ["ACH-999999"])


def test_load_depmap_gene_effect_missing_column_raises(tmp_path: Path) -> None:
    csv_path = tmp_path / "CRISPRGeneEffect.csv"
    _write_fake_gene_effect_csv(csv_path)
    registration = {
        "sources": {"depmap_gene_effect": {"sha256": sha256_file(csv_path)}}
    }

    with pytest.raises(ValueError, match="missing columns"):
        load_depmap_gene_effect(
            csv_path, registration, ["ACH-000001"], columns=["NOPE (999)"]
        )


def test_load_depmap_gene_effect_restricts_to_requested_columns(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "CRISPRGeneEffect.csv"
    _write_fake_gene_effect_csv(csv_path)
    registration = {
        "sources": {"depmap_gene_effect": {"sha256": sha256_file(csv_path)}}
    }

    result = load_depmap_gene_effect(
        csv_path, registration, ["ACH-000001"], columns=["GENE2 (2)"]
    )

    assert list(result.columns) == ["GENE2 (2)"]


def test_no_name_keyed_join_path_exists_by_construction() -> None:
    """Structural guard: the module must never reference DepMap's
    ``CellLineName`` column (``"K-562"`` there vs. ``"K562"`` in this repo's
    code) -- a name-keyed join is impossible by construction, not merely by
    convention.
    """
    import aivc_model.tx1_geneeffect_data as module

    source = Path(module.__file__).read_text()
    assert "CellLineName" not in source
    assert "cell_line_name" not in source
