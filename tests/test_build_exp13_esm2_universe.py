from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
import shlex
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from aivc_model.geneeffect_data import Exp13Split
from aivc_model.geneeffect_stage2_runner import authenticate_target_esm2
from aivc_model.esm2_provenance import (
    SCHEMA_VERSION as ESM2_PROVENANCE_SCHEMA,
    build_embedding_artifact_record,
)
from aivc_model.stage1_artifact import Stage1ArtifactManifest
from aivc_model.state_core import sha256_strings
from scripts.build_exp13_esm2_universe import (
    AuthenticatedStage1Vocabulary,
    CoverageUniverse,
    build_coverage_universe,
    build_embedding_union,
    build_precompute_command,
    authenticate_stage1_manifest,
    authenticate_vocabulary,
    inspect_npz_coverage,
    load_authenticated_vocabulary,
    require_npz_coverage,
    require_pinned_gene_effect,
    write_universe_artifacts,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _split() -> Exp13Split:
    return Exp13Split(
        train=("T1", "T2", "T3", "T4", "T5"),
        val=("V1", "V2", "V3"),
        test=("E1", "E2", "E3"),
        unlabeled_train=(),
    )


def _labels() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model_id in _split().all_model_ids:
        rows.append({"model_id": model_id, "gene_symbol": "PASS", "gene_effect": 1.0})
        rows.append(
            {
                "model_id": model_id,
                "gene_symbol": "LOW_TRAIN",
                "gene_effect": np.nan if model_id == "T5" else 2.0,
            }
        )
        rows.append(
            {
                "model_id": model_id,
                "gene_symbol": "LOW_VAL",
                "gene_effect": np.nan if model_id == "V3" else 3.0,
            }
        )
    return pd.DataFrame(rows)


def _checkpoint(path: Path, vocabulary_sha256: str) -> Path:
    torch.save(
        {
            "perturbations.gene_vocabulary_sha256": torch.tensor(
                list(bytes.fromhex(vocabulary_sha256)), dtype=torch.uint8
            )
        },
        path,
    )
    return path


def _stage1(symbols: tuple[str, ...], tmp_path: Path) -> AuthenticatedStage1Vocabulary:
    vocabulary_sha256 = sha256_strings(np.asarray(symbols, dtype=object))
    return authenticate_vocabulary(
        symbols,
        vocabulary_sha256,
        source="fixture.json",
        source_sha256=_digest("fixture"),
        checkpoint_path=_checkpoint(
            tmp_path / f"checkpoint-{vocabulary_sha256[:8]}.pt", vocabulary_sha256
        ),
    )


def test_coverage_universe_has_explicit_drop_reasons() -> None:
    universe = build_coverage_universe(_labels(), _split())
    assert universe.symbols == ("PASS",)
    drops = {row["gene_symbol"]: row for row in universe.dropped}
    assert drops["LOW_TRAIN"]["reasons"] == ["train_finite_lt5"]
    assert drops["LOW_VAL"]["reasons"] == ["val_finite_lt3"]


def test_coverage_universe_rejects_duplicate_pairs() -> None:
    labels = pd.concat([_labels(), _labels().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        build_coverage_universe(labels, _split())


def test_stage1_only_gene_enters_embedding_union_but_not_scored_metrics(
    tmp_path: Path,
) -> None:
    split_path = tmp_path / "split.json"
    labels_path = tmp_path / "labels.csv"
    split_path.write_text("split")
    labels_path.write_text("labels")
    scored_csv = tmp_path / "scored.csv"
    union_csv = tmp_path / "union.csv"
    manifest_path = tmp_path / "manifest.json"
    scored = build_coverage_universe(_labels(), _split())
    stage1 = _stage1(("PASS", "STAGE1_ONLY"), tmp_path)

    manifest = write_universe_artifacts(
        scored,
        scored,
        stage1,
        scored_csv,
        union_csv,
        manifest_path,
        split_path=split_path,
        gene_effect_path=labels_path,
        copy_prior_path=labels_path,
        copy_prior_manifest_path=split_path,
        expected_upper_bound_count=1,
        expected_candidate_count=1,
    )

    assert scored_csv.read_text() == "gene_symbol\nPASS\n"
    assert union_csv.read_text() == "gene_symbol\nPASS\nSTAGE1_ONLY\n"
    assert manifest["metrics_membership"] == (
        "copy_prior_candidates_intersect_esm2_resolved"
    )
    assert manifest["coverage_qualified_upper_bound"]["symbols"] == ["PASS"]
    assert manifest["copy_prior_eligible_candidates"]["symbols"] == ["PASS"]
    assert manifest["embedding_union"]["symbols"] == ["PASS", "STAGE1_ONLY"]
    assert manifest["embedding_union"]["stage1_only_symbols"] == ["STAGE1_ONLY"]
    assert json.loads(manifest_path.read_text()) == manifest
    assert (
        manifest["copy_prior_eligible_candidates"]["csv_sha256"]
        == hashlib.sha256(scored_csv.read_bytes()).hexdigest()
    )


def test_embedding_union_is_sorted_without_changing_separate_scored_order() -> None:
    scored = ("B", "A")
    assert build_embedding_union(scored, ("A", "C", "D")) == ("A", "B", "C", "D")
    assert scored == ("B", "A")


def test_authenticated_vocabulary_fails_closed_on_hash_or_shape(tmp_path: Path) -> None:
    path = tmp_path / "vocabulary.json"
    path.write_text(json.dumps(["A", "STAGE1_ONLY"]))
    expected = sha256_strings(np.asarray(["A", "STAGE1_ONLY"], dtype=object))
    checkpoint = _checkpoint(tmp_path / "checkpoint.pt", expected)
    loaded = load_authenticated_vocabulary(path, expected, checkpoint)
    assert loaded.symbols == ("A", "STAGE1_ONLY")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_authenticated_vocabulary(path, _digest("stale"), checkpoint)
    path.write_text(json.dumps({"stage1_genes": ["A"]}))
    with pytest.raises(ValueError, match="list or tuple"):
        load_authenticated_vocabulary(path, expected, checkpoint)
    wrong_checkpoint = _checkpoint(tmp_path / "wrong.pt", _digest("wrong-vocabulary"))
    path.write_text(json.dumps(["A", "STAGE1_ONLY"]))
    with pytest.raises(ValueError, match="checkpoint vocabulary SHA-256 mismatch"):
        load_authenticated_vocabulary(path, expected, wrong_checkpoint)


def test_sealed_manifest_authenticates_vocabulary(tmp_path: Path) -> None:
    symbols = ("A", "STAGE1_ONLY")
    manifest = Stage1ArtifactManifest(
        schema_version=1,
        checkpoint_sha256=_digest("checkpoint"),
        stage1_genes=symbols,
        stage1_gene_vocabulary_sha256=sha256_strings(np.asarray(symbols, dtype=object)),
        esm2_artifact_sha256=_digest("esm2"),
        state_hparams_sha256=_digest("hparams"),
        compatibility_code_sha256={"code": _digest("code")},
        training_code_provenance_status="unavailable",
        training_code_provenance_reason=(
            "historical_run_has_no_immutable_training_code_identity"
        ),
        training_data_provenance_status="incomplete",
        training_data_provenance_missing_identities=(
            "cell_line_manifest",
            "tx1_basal_cache",
            "response_cache",
            "perturbseq_source_content",
        ),
        training_data_provenance_reason=(
            "historical_run_manifest_does_not_hash_all_training_data_inputs"
        ),
        config_sha256={"config": _digest("config")},
        source_sha256={"source": _digest("source")},
        legacy_esm_matrix_sha256=None,
        run_manifest_sha256=_digest("run"),
        checkpoint_metadata_sha256=_digest("metadata"),
        stage1_objective_sha256=_digest("objective"),
    )
    path = tmp_path / "stage1_model_manifest.json"
    manifest.write(path)
    manifest_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    authenticated = authenticate_stage1_manifest(path, manifest_sha256)
    assert authenticated.symbols == symbols
    assert authenticated.checkpoint_sha256 == _digest("checkpoint")
    with pytest.raises(ValueError, match="manifest SHA-256 mismatch"):
        authenticate_stage1_manifest(path, _digest("stale-manifest"))
    payload = json.loads(path.read_text())
    payload["stage1_genes"] = ["A", "TAMPERED"]
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="vocabulary SHA256 mismatch"):
        authenticate_stage1_manifest(
            path, hashlib.sha256(path.read_bytes()).hexdigest()
        )


def _write_npz(
    path: Path, symbols: list[str], resolved: list[bool], *, width: int = 3
) -> None:
    np.savez(
        path,
        symbols=np.asarray(symbols, dtype=object),
        vectors=np.ones((len(symbols), width), dtype=np.float32),
        resolved=np.asarray(resolved, dtype=bool),
    )


def _write_provenance(path: Path, npz: Path, symbols: list[str]) -> Path:
    union_csv = "gene_symbol\n" + "".join(f"{symbol}\n" for symbol in symbols)
    with np.load(npz, allow_pickle=True) as payload_npz:
        resolved = payload_npz["resolved"].tolist()
    mapping_json = path.with_suffix(".mapping.json")
    mapping_csv = path.with_suffix(".mapping.csv")
    mapping_records = [
        {
            "gene_symbol": symbol,
            "resolved": bool(is_resolved),
            "primary_accession": f"P{index:05d}" if is_resolved else None,
            "entry_id": f"GENE{index}_HUMAN" if is_resolved else None,
            "isoform_identifier": f"P{index:05d}" if is_resolved else None,
            "isoform_policy": (
                "canonical_reviewed_top_hit" if is_resolved else None
            ),
            "sequence_sha256": (
                _digest(f"sequence:{symbol}") if is_resolved else None
            ),
        }
        for index, (symbol, is_resolved) in enumerate(
            zip(symbols, resolved, strict=True), start=1
        )
    ]
    mapping_json.write_text(
        json.dumps(
            {"schema_version": "esm2-uniprot-mapping-v1", "records": mapping_records}
        ),
        encoding="utf-8",
    )
    pd.DataFrame(mapping_records).to_csv(mapping_csv, index=False, lineterminator="\n")
    payload = {
        "schema_version": ESM2_PROVENANCE_SCHEMA,
        "requested_model_id": "facebook/esm2_t33_650M_UR50D",
        "loaded_model": {
            "class": "TinyModel",
            "state_sha256": _digest("model"),
            "config_sha256": _digest("model-config"),
        },
        "tokenizer": {
            "class": "TinyTokenizer",
            "vocabulary_config_sha256": _digest("tokenizer"),
        },
        "sequence_source": {
            "benchmark_csv_sha256": hashlib.sha256(union_csv.encode()).hexdigest(),
            "sequence_cache_sha256": _digest("sequence-cache"),
            "symbol_columns": ["gene_symbol"],
            "uniprot_mapping_json_path": str(mapping_json),
            "uniprot_mapping_json_sha256": hashlib.sha256(
                mapping_json.read_bytes()
            ).hexdigest(),
            "uniprot_mapping_csv_path": str(mapping_csv),
            "uniprot_mapping_csv_sha256": hashlib.sha256(
                mapping_csv.read_bytes()
            ).hexdigest(),
            "sequence_sha256_by_symbol": {
                symbol: (_digest(f"sequence:{symbol}") if is_resolved else None)
                for symbol, is_resolved in zip(symbols, resolved, strict=True)
            }
        },
        "embedding_artifact": build_embedding_artifact_record(npz),
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_npz_must_cover_stage1_only_union_member(tmp_path: Path) -> None:
    path = tmp_path / "esm2.npz"
    _write_npz(path, ["PASS"], [True])
    report = inspect_npz_coverage(path, ["PASS", "STAGE1_ONLY"])
    assert report.missing == ("STAGE1_ONLY",)
    with pytest.raises(ValueError, match=r"missing 1/2"):
        require_npz_coverage(path, ["PASS", "STAGE1_ONLY"], expected_width=3)


def test_npz_allows_unresolved_candidate_but_not_unresolved_stage1(
    tmp_path: Path,
) -> None:
    path = tmp_path / "esm2.npz"
    _write_npz(path, ["CANDIDATE", "STAGE1"], [False, True])
    report = require_npz_coverage(
        path,
        ["CANDIDATE", "STAGE1"],
        must_resolve_symbols=["STAGE1"],
        expected_width=3,
    )
    assert report.missing == ("CANDIDATE",)
    with pytest.raises(ValueError, match="must-resolve"):
        require_npz_coverage(
            path,
            ["CANDIDATE", "STAGE1"],
            must_resolve_symbols=["CANDIDATE", "STAGE1"],
            expected_width=3,
        )


def test_npz_must_exactly_match_union_order_without_extras(tmp_path: Path) -> None:
    path = tmp_path / "esm2.npz"
    _write_npz(path, ["B", "A", "EXTRA"], [True, True, True])
    with pytest.raises(ValueError, match="order/universe"):
        require_npz_coverage(path, ["A", "B"], expected_width=3)


def test_npz_coverage_rejects_malformed_resolved_dtype(tmp_path: Path) -> None:
    path = tmp_path / "esm2.npz"
    np.savez(
        path,
        symbols=np.asarray(["A"], dtype=object),
        vectors=np.ones((1, 3), dtype=np.float32),
        resolved=np.asarray([1], dtype=np.int64),
    )
    with pytest.raises(ValueError, match="bool"):
        inspect_npz_coverage(path, ["A"])


def test_artifacts_are_fresh_and_precompute_targets_union(tmp_path: Path) -> None:
    split_path = tmp_path / "split"
    labels_path = tmp_path / "labels"
    split_path.write_text("split")
    labels_path.write_text("labels")
    scored_csv = tmp_path / "scored.csv"
    union_csv = tmp_path / "embedding union.csv"
    manifest_path = tmp_path / "manifest.json"
    scored = build_coverage_universe(_labels(), _split())
    write_universe_artifacts(
        scored,
        scored,
        _stage1(("PASS", "STAGE1_ONLY"), tmp_path),
        scored_csv,
        union_csv,
        manifest_path,
        split_path=split_path,
        gene_effect_path=labels_path,
        copy_prior_path=labels_path,
        copy_prior_manifest_path=split_path,
        expected_upper_bound_count=1,
        expected_candidate_count=1,
    )
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_universe_artifacts(
            scored,
            scored,
            _stage1(("PASS", "STAGE1_ONLY"), tmp_path),
            scored_csv,
            union_csv,
            manifest_path,
            split_path=split_path,
            gene_effect_path=labels_path,
            copy_prior_path=labels_path,
            copy_prior_manifest_path=split_path,
            expected_upper_bound_count=1,
            expected_candidate_count=1,
        )
    command = build_precompute_command(
        union_csv,
        tmp_path / "esm2.npz",
        tmp_path / "sequences.json",
        local_files_only=True,
    )
    parsed = shlex.split(command)
    assert parsed[parsed.index("--benchmark-csv") + 1] == str(union_csv)
    assert "--require-complete-coverage" not in parsed
    assert "--local-files-only" in parsed


def test_writer_reauthenticates_stage1_vocabulary(tmp_path: Path) -> None:
    split_path = tmp_path / "split"
    labels_path = tmp_path / "labels"
    split_path.write_text("split")
    labels_path.write_text("labels")
    stale = replace(
        _stage1(("PASS", "STAGE1_ONLY"), tmp_path),
        vocabulary_sha256=_digest("stale"),
    )
    with pytest.raises(ValueError, match="vocabulary SHA-256 is stale"):
        write_universe_artifacts(
            build_coverage_universe(_labels(), _split()),
            build_coverage_universe(_labels(), _split()),
            stale,
            tmp_path / "scored.csv",
            tmp_path / "union.csv",
            tmp_path / "manifest.json",
            split_path=split_path,
            gene_effect_path=labels_path,
            copy_prior_path=labels_path,
            copy_prior_manifest_path=split_path,
            expected_upper_bound_count=1,
            expected_candidate_count=1,
        )


def test_outputs_must_be_distinct_and_verified_npz_is_hash_bound(
    tmp_path: Path,
) -> None:
    split_path = tmp_path / "split"
    labels_path = tmp_path / "labels"
    split_path.write_text("split")
    labels_path.write_text("labels")
    scored = build_coverage_universe(_labels(), _split())
    stage1 = _stage1(("PASS", "STAGE1_ONLY"), tmp_path)
    shared = tmp_path / "shared.csv"
    with pytest.raises(ValueError, match="must be distinct"):
        write_universe_artifacts(
            scored,
            scored,
            stage1,
            shared,
            shared,
            tmp_path / "manifest.json",
            split_path=split_path,
            gene_effect_path=labels_path,
            copy_prior_path=labels_path,
            copy_prior_manifest_path=split_path,
            expected_upper_bound_count=1,
            expected_candidate_count=1,
        )

    npz = tmp_path / "esm2.npz"
    _write_npz(npz, ["PASS", "STAGE1_ONLY"], [True, True], width=1_280)
    report = require_npz_coverage(npz, ["PASS", "STAGE1_ONLY"])
    provenance = _write_provenance(
        tmp_path / "esm2.provenance.json", npz, ["PASS", "STAGE1_ONLY"]
    )
    manifest = write_universe_artifacts(
        scored,
        scored,
        stage1,
        tmp_path / "scored.csv",
        tmp_path / "union.csv",
        tmp_path / "verified-manifest.json",
        split_path=split_path,
        gene_effect_path=labels_path,
        copy_prior_path=labels_path,
        copy_prior_manifest_path=split_path,
        expected_upper_bound_count=1,
        expected_candidate_count=1,
        verified_npz_path=npz,
        expected_npz_sha256=report.artifact_sha256,
        esm2_provenance_path=provenance,
    )
    assert manifest["embedding_union"]["verified_npz"] == {
        "path": str(npz),
        "artifact_sha256": hashlib.sha256(npz.read_bytes()).hexdigest(),
        "resolved_count": 2,
        "vector_width": 1_280,
    }
    assert manifest["final_evaluated_universe"] == {
        "symbols": ["PASS"],
        "count": 1,
        "symbols_sha256": manifest["final_evaluated_universe"]["symbols_sha256"],
        "unresolved_candidate_symbols": [],
        "unresolved_candidate_count": 0,
    }

    config = SimpleNamespace(
        features=SimpleNamespace(esm2_dim=1_280),
        paths=SimpleNamespace(
            esm2_embeddings=npz,
            esm2_provenance_manifest=provenance,
            esm2_uniprot_mapping_json=provenance.with_suffix(".mapping.json"),
            esm2_uniprot_mapping_csv=provenance.with_suffix(".mapping.csv"),
            esm2_universe_manifest=tmp_path / "verified-manifest.json",
            split=split_path,
            gene_effect=labels_path,
            copy_prior=labels_path,
            copy_prior_manifest=split_path,
        ),
    )
    authenticate_target_esm2(
        config,
        coverage_qualified_symbols=("PASS",),
        candidate_symbols=("PASS",),
        coverage_drop_report=manifest["coverage_qualified_upper_bound"][
            "drop_report"
        ],
        candidate_drop_report=manifest["copy_prior_eligible_candidates"][
            "drop_report"
        ],
        scored_symbols=("PASS",),
        embedding_symbols=("PASS", "STAGE1_ONLY"),
    )
    manifest_path = tmp_path / "verified-manifest.json"
    pristine_manifest = manifest_path.read_text(encoding="utf-8")
    tampered_manifest = json.loads(pristine_manifest)
    tampered_manifest["final_evaluated_universe"]["unresolved_candidate_count"] = 1
    manifest_path.write_text(json.dumps(tampered_manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="final evaluated record mismatch"):
        authenticate_target_esm2(
            config,
            coverage_qualified_symbols=("PASS",),
            candidate_symbols=("PASS",),
            coverage_drop_report=manifest["coverage_qualified_upper_bound"][
                "drop_report"
            ],
            candidate_drop_report=manifest["copy_prior_eligible_candidates"][
                "drop_report"
            ],
            scored_symbols=("PASS",),
            embedding_symbols=("PASS", "STAGE1_ONLY"),
        )
    manifest_path.write_text(pristine_manifest, encoding="utf-8")
    tampered_manifest = json.loads(pristine_manifest)
    tampered_manifest["copy_prior_eligible_candidates"]["drop_report"] = [
        {"gene_symbol": "FAKE", "reasons": ["copy_prior_missing"]}
    ]
    manifest_path.write_text(json.dumps(tampered_manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="candidate drop report mismatch"):
        authenticate_target_esm2(
            config,
            coverage_qualified_symbols=("PASS",),
            candidate_symbols=("PASS",),
            coverage_drop_report=manifest["coverage_qualified_upper_bound"][
                "drop_report"
            ],
            candidate_drop_report=manifest["copy_prior_eligible_candidates"][
                "drop_report"
            ],
            scored_symbols=("PASS",),
            embedding_symbols=("PASS", "STAGE1_ONLY"),
        )
    manifest_path.write_text(pristine_manifest, encoding="utf-8")
    pristine = provenance.read_text(encoding="utf-8")
    payload = json.loads(pristine)
    payload["loaded_model"]["state_sha256"] = _digest("tampered-model")
    provenance.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="provenance authentication mismatch"):
        authenticate_target_esm2(
            config,
            coverage_qualified_symbols=("PASS",),
            candidate_symbols=("PASS",),
            coverage_drop_report=manifest["coverage_qualified_upper_bound"][
                "drop_report"
            ],
            candidate_drop_report=manifest["copy_prior_eligible_candidates"][
                "drop_report"
            ],
            scored_symbols=("PASS",),
            embedding_symbols=("PASS", "STAGE1_ONLY"),
        )
    provenance.write_text(pristine, encoding="utf-8")
    payload = json.loads(pristine)
    payload["sequence_source"]["sequence_sha256_by_symbol"]["PASS"] = _digest(
        "tampered-sequence"
    )
    provenance.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="identity/sequence mismatch"):
        authenticate_target_esm2(
            config,
            coverage_qualified_symbols=("PASS",),
            candidate_symbols=("PASS",),
            coverage_drop_report=manifest["coverage_qualified_upper_bound"][
                "drop_report"
            ],
            candidate_drop_report=manifest["copy_prior_eligible_candidates"][
                "drop_report"
            ],
            scored_symbols=("PASS",),
            embedding_symbols=("PASS", "STAGE1_ONLY"),
        )


def test_manifest_drops_unresolved_candidate_but_requires_stage1(
    tmp_path: Path,
) -> None:
    split_path = tmp_path / "split"
    labels_path = tmp_path / "labels"
    split_path.write_text("split")
    labels_path.write_text("labels")
    npz = tmp_path / "esm2.npz"
    _write_npz(
        npz,
        ["MISS", "PASS", "STAGE1_ONLY"],
        [False, True, True],
        width=1_280,
    )
    provenance = _write_provenance(
        tmp_path / "esm2.provenance.json",
        npz,
        ["MISS", "PASS", "STAGE1_ONLY"],
    )
    report = inspect_npz_coverage(npz, ["MISS", "PASS", "STAGE1_ONLY"])
    candidate = CoverageUniverse(("MISS", "PASS"), ())
    manifest = write_universe_artifacts(
        candidate,
        candidate,
        _stage1(("STAGE1_ONLY",), tmp_path),
        tmp_path / "scored.csv",
        tmp_path / "union.csv",
        tmp_path / "manifest.json",
        split_path=split_path,
        gene_effect_path=labels_path,
        copy_prior_path=labels_path,
        copy_prior_manifest_path=split_path,
        expected_upper_bound_count=2,
        expected_candidate_count=2,
        verified_npz_path=npz,
        expected_npz_sha256=report.artifact_sha256,
        esm2_provenance_path=provenance,
    )
    assert manifest["final_evaluated_universe"]["symbols"] == ["PASS"]
    assert manifest["final_evaluated_universe"][
        "unresolved_candidate_symbols"
    ] == ["MISS"]


def test_gene_effect_input_is_sha_pinned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "gene_effect.csv"
    path.write_text("wrong release")
    with pytest.raises(ValueError, match="GeneEffect SHA-256 mismatch"):
        require_pinned_gene_effect(path)
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(
        "scripts.build_exp13_esm2_universe.PINNED_GENE_EFFECT_SHA256", observed
    )
    assert require_pinned_gene_effect(path) == observed
