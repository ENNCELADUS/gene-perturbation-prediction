"""Offline unit tests for scripts/precompute_esm2_embeddings.py.

Tests cover:
  - FIX 1: corrupt JSON cache recovery in load_or_fetch_sequences
  - FIX 2: check_resolution helper raises/warns correctly
  - FIX 3: truncate_sequence helper logs and truncates correctly
  - Existing smoke: universe_symbols reads both columns
"""

from __future__ import annotations

import importlib
import hashlib
import json
import logging
import sys
import types
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch

from aivc_model.gene_embeddings import (
    Esm2EmbeddingTable,
    require_complete_esm_coverage,
)
from aivc_model.esm2_provenance import load_and_authenticate_esm2_provenance

# ---------------------------------------------------------------------------
# Lazy import: stub out heavy optional imports before loading the script
# ---------------------------------------------------------------------------


def _import_module() -> types.ModuleType:
    """Import precompute_esm2_embeddings with torch/transformers stubbed."""
    stubbed_modules = (
        "torch",
        "transformers",
        "transformers.EsmModel",
        "transformers.EsmTokenizer",
    )
    original_modules = {name: sys.modules.get(name) for name in stubbed_modules}

    # Build minimal stubs so the module-level imports don't explode.
    if "torch" not in sys.modules:
        torch_stub = types.ModuleType("torch")
        torch_stub.cuda = types.SimpleNamespace(is_available=lambda: False)  # type: ignore[attr-defined]
        torch_stub.no_grad = lambda: _NullContext()  # type: ignore[attr-defined]
        sys.modules["torch"] = torch_stub

    for mod in ("transformers", "transformers.EsmModel", "transformers.EsmTokenizer"):
        if mod not in sys.modules:
            sys.modules[mod] = types.ModuleType(mod)

    # Force reload so the module picks up stubs if first import in this process.
    spec_path = (
        Path(__file__).parent.parent / "scripts" / "precompute_esm2_embeddings.py"
    )
    spec = importlib.util.spec_from_file_location(
        "precompute_esm2_embeddings", spec_path
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
    finally:
        for name, original in original_modules.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original
    return mod


class _NullContext:
    def __enter__(self) -> "_NullContext":
        return self

    def __exit__(self, *_: object) -> None:
        pass


MOD = _import_module()


def _record(sequence: str = "MSEQ"):
    return MOD.UniProtSequenceRecord(
        primary_accession="P04637",
        entry_id="P53_HUMAN",
        isoform_identifier="P04637",
        isoform_policy=MOD.ISOFORM_POLICY,
        sequence=sequence,
    )


class _UniProtResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def __enter__(self) -> "_UniProtResponse":
        return self

    def __exit__(self, *_: object) -> None:
        pass

    def read(self) -> bytes:
        return json.dumps(self.payload).encode()


def _uniprot_hit(*, symbol: str = "TP53", reviewed: bool = True) -> dict[str, object]:
    return {
        "primaryAccession": "P04637",
        "uniProtkbId": "P53_HUMAN",
        "entryType": (
            MOD.REVIEWED_ENTRY_TYPE if reviewed else "UniProtKB unreviewed (TrEMBL)"
        ),
        "genes": [{"geneName": {"value": symbol}}],
        "uniProtKBCrossReferences": [{"database": "GeneID", "id": "7157"}],
        "sequence": {"value": "MSEQ"},
    }


def test_fetch_sequence_uses_exact_symbol_and_validates_returned_identity() -> None:
    with patch.object(
        MOD.urllib.request,
        "urlopen",
        return_value=_UniProtResponse({"results": [_uniprot_hit()]}),
    ) as request:
        assert MOD.fetch_sequence("TP53") == _record()
    assert "gene_exact%3ATP53" in request.call_args.args[0]

    for hit in (
        _uniprot_hit(symbol="TP53BP1"),
        _uniprot_hit(reviewed=False),
    ):
        with patch.object(
            MOD.urllib.request,
            "urlopen",
            return_value=_UniProtResponse({"results": [hit]}),
        ):
            assert MOD.fetch_sequence("TP53") is None


def test_fetch_sequence_validates_requested_gene_id() -> None:
    hit = _uniprot_hit()
    hit["uniProtKBCrossReferences"] = [
        {"database": "GeneID", "id": "9999"}
    ]
    with patch.object(
        MOD.urllib.request,
        "urlopen",
        return_value=_UniProtResponse({"results": [hit]}),
    ):
        assert MOD.fetch_sequence("TP53", "7157") is None


# ---------------------------------------------------------------------------
# FIX 1 — corrupt JSON cache recovery
# ---------------------------------------------------------------------------


def test_corrupt_cache_logs_warning_and_recovers(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """load_or_fetch_sequences must recover from a truncated JSON cache.

    RED: before fix, json.JSONDecodeError propagates and the call raises.
    GREEN: after fix, it logs a warning and falls back to an empty dict.
    """
    cache = tmp_path / "cache.json"
    cache.write_text('{"BRCA1": "MPIGSKERP', encoding="utf-8")  # truncated JSON

    with patch.object(MOD, "fetch_sequence", return_value=_record()) as mock_fetch:
        with caplog.at_level(logging.WARNING, logger="precompute_esm2"):
            result = MOD.load_or_fetch_sequences(["PTEN"], cache)

    # Should have fallen back and fetched the symbol.
    assert "PTEN" in result
    assert result["PTEN"] == _record()
    mock_fetch.assert_called_once_with("PTEN")

    # A warning must have been emitted about the corrupt cache.
    warning_messages = [
        r.message for r in caplog.records if r.levelno == logging.WARNING
    ]
    assert any(
        "corrupt" in str(m).lower() or "json" in str(m).lower()
        for m in warning_messages
    ), f"Expected a warning about corrupt JSON; got: {warning_messages}"


def test_load_or_fetch_sequences_uses_identifier_for_missing_symbol(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "cache.json"
    with patch.object(MOD, "fetch_sequence", return_value=_record()) as mock_fetch:
        result = MOD.load_or_fetch_sequences(
            ["PTEN"], cache, identifiers={"PTEN": "5728"}
        )

    assert result == {"PTEN": _record()}
    mock_fetch.assert_called_once_with("PTEN", "5728")


def test_legacy_sequence_cache_requires_explicit_refetch(tmp_path: Path) -> None:
    cache = tmp_path / "cache.json"
    cache.write_text('{"PTEN":"MSEQ"}', encoding="utf-8")
    with pytest.raises(ValueError, match="--refetch-legacy-cache"):
        MOD.load_or_fetch_sequences(["PTEN"], cache)
    with patch.object(MOD, "fetch_sequence", return_value=_record()) as fetch:
        records = MOD.load_or_fetch_sequences(
            ["PTEN"], cache, refetch_legacy_cache=True
        )
    assert records == {"PTEN": _record()}
    fetch.assert_called_once_with("PTEN")


def test_identifiers_from_csv_normalizes_integer_ids(tmp_path: Path) -> None:
    csv = tmp_path / "genes.csv"
    pd.DataFrame(
        {"perturbation_gene": ["tp53", "KRAS"], "entrez": [7157, 3845]}
    ).to_csv(csv, index=False)

    assert MOD.identifiers_from_csv(csv, "perturbation_gene", "entrez") == {
        "TP53": "7157",
        "KRAS": "3845",
    }


# ---------------------------------------------------------------------------
# FIX 2 — check_resolution helper
# ---------------------------------------------------------------------------


def test_check_resolution_raises_on_all_unresolved() -> None:
    """check_resolution must raise RuntimeError when no symbols resolved."""
    resolved = np.zeros(5, dtype=bool)
    with pytest.raises(RuntimeError, match="no sequences resolved"):
        MOD.check_resolution(resolved, n_symbols=5)


def test_check_resolution_warns_on_low_resolution(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """check_resolution must warn when resolved fraction is < 50%."""
    resolved = np.array([True, False, False, False, False], dtype=bool)  # 20%
    with caplog.at_level(logging.WARNING, logger="precompute_esm2"):
        MOD.check_resolution(resolved, n_symbols=5)
    warning_messages = [
        r.message for r in caplog.records if r.levelno == logging.WARNING
    ]
    assert any(
        "low" in str(m).lower() or "resol" in str(m).lower() for m in warning_messages
    ), f"Expected low-resolution warning; got: {warning_messages}"


def test_check_resolution_silent_when_majority_resolved(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """check_resolution must not warn when resolved fraction >= 50%."""
    resolved = np.array([True, True, True, False, False], dtype=bool)  # 60%
    with caplog.at_level(logging.WARNING, logger="precompute_esm2"):
        MOD.check_resolution(resolved, n_symbols=5)
    warning_messages = [
        r.message for r in caplog.records if r.levelno == logging.WARNING
    ]
    assert not warning_messages, f"Unexpected warnings: {warning_messages}"


# ---------------------------------------------------------------------------
# FIX 3 — truncate_sequence helper
# ---------------------------------------------------------------------------


def test_truncate_sequence_returns_full_when_short(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """truncate_sequence returns seq unchanged and emits no warning for short seqs."""
    seq = "M" * 100
    with caplog.at_level(logging.WARNING, logger="precompute_esm2"):
        result = MOD.truncate_sequence(seq, symbol="GENE1")
    assert result == seq
    assert not caplog.records


def test_truncate_sequence_truncates_and_warns_for_long_seq(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """truncate_sequence must truncate to 1022 and emit a warning for long seqs."""
    seq = "A" * 2000
    with caplog.at_level(logging.WARNING, logger="precompute_esm2"):
        result = MOD.truncate_sequence(seq, symbol="HUGEGENE")
    assert len(result) == 1022
    warning_messages = [
        r.message for r in caplog.records if r.levelno == logging.WARNING
    ]
    assert any("HUGEGENE" in str(m) for m in warning_messages), (
        f"Expected symbol name in warning; got: {warning_messages}"
    )
    assert any(
        "2000" in str(m) or "truncat" in str(m).lower() for m in warning_messages
    ), f"Expected length or 'truncat' in warning; got: {warning_messages}"


def test_truncate_sequence_custom_max_len(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """truncate_sequence respects a custom max_len parameter."""
    seq = "C" * 50
    with caplog.at_level(logging.WARNING, logger="precompute_esm2"):
        result = MOD.truncate_sequence(seq, symbol="GENE2", max_len=10)
    assert len(result) == 10
    assert caplog.records  # must have warned


# ---------------------------------------------------------------------------
# FIX 5 — residue mean pooling excludes special tokens (BOS/EOS/pad)
# ---------------------------------------------------------------------------


def test_mean_pool_residues_excludes_special_tokens() -> None:
    """mean_pool_residues averages only residue rows (special_tokens_mask == 0)."""
    # 4 token positions, dim 2. Rows 0 and 3 are special (BOS/EOS); 1,2 residues.
    hidden = np.array(
        [[100.0, 100.0], [1.0, 2.0], [3.0, 4.0], [-100.0, -100.0]],
        dtype=np.float32,
    )
    special = np.array([1, 0, 0, 1])  # 1 = special token, excluded from the mean
    pooled = MOD.mean_pool_residues(hidden, special)
    # Mean of rows 1 and 2 only: ([1,2] + [3,4]) / 2 = [2, 3].
    assert pooled.shape == (2,)
    np.testing.assert_allclose(pooled, np.array([2.0, 3.0]), rtol=1e-6)


def test_mean_pool_residues_all_special_falls_back_to_all_tokens() -> None:
    """If every token is special (degenerate), fall back to a full mean (no NaN)."""
    hidden = np.array([[2.0, 4.0], [6.0, 8.0]], dtype=np.float32)
    special = np.array([1, 1])
    pooled = MOD.mean_pool_residues(hidden, special)
    assert np.isfinite(pooled).all()
    np.testing.assert_allclose(pooled, np.array([4.0, 6.0]), rtol=1e-6)


# ---------------------------------------------------------------------------
# Existing smoke: universe_symbols
# ---------------------------------------------------------------------------


def test_universe_symbols_returns_sorted_unique_upper(tmp_path: Path) -> None:
    """universe_symbols deduplicates and upper-cases both symbol columns."""
    csv = tmp_path / "benchmark.csv"
    df = pd.DataFrame(
        {
            "gene_a_symbol": ["brca1", "tp53", "pten"],
            "gene_b_symbol": ["TP53", "KRAS", "brca1"],
        }
    )
    df.to_csv(csv, index=False)

    result = MOD.universe_symbols(csv)

    assert result == sorted({"BRCA1", "TP53", "PTEN", "KRAS"})
    assert result == sorted(set(result))  # sorted


def test_symbols_from_csv_supports_single_gene_column(tmp_path: Path) -> None:
    csv = tmp_path / "genes.csv"
    pd.DataFrame({"perturbation_gene": ["tp53", "KRAS", "TP53"]}).to_csv(
        csv, index=False
    )
    assert MOD.symbols_from_csv(csv, ("perturbation_gene",)) == ["KRAS", "TP53"]


def test_exp05_esm_asset_must_resolve_all_canonical_genes() -> None:
    canonical = ["A", "B", "C"]
    table = Esm2EmbeddingTable(
        dim=4,
        vectors_by_symbol={
            "A": np.ones(4, dtype=np.float32),
            "B": np.ones(4, dtype=np.float32),
        },
    )
    with pytest.raises(ValueError, match="2/3"):
        require_complete_esm_coverage(canonical, table)


def test_complete_esm_coverage_requires_exact_uppercase_order() -> None:
    canonical = ["B", "A"]
    table = Esm2EmbeddingTable(
        dim=2,
        vectors_by_symbol={
            "A": np.ones(2, dtype=np.float32),
            "B": np.ones(2, dtype=np.float32),
        },
    )
    with pytest.raises(ValueError, match="order"):
        require_complete_esm_coverage(canonical, table)


def test_complete_esm_coverage_allows_resolved_extra_tokens() -> None:
    canonical = ["B", "A"]
    table = Esm2EmbeddingTable(
        dim=2,
        vectors_by_symbol={
            "B": np.ones(2, dtype=np.float32),
            "EXTERNAL": np.ones(2, dtype=np.float32),
            "A": np.ones(2, dtype=np.float32),
        },
    )

    require_complete_esm_coverage(canonical, table)


def _run_asset_cli(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    resolved: np.ndarray,
    *,
    strict: bool,
) -> Path:
    csv = tmp_path / "genes.csv"
    pd.DataFrame({"perturbation_gene": ["A", "B", "C"]}).to_csv(csv, index=False)
    output = tmp_path / "esm2.npz"
    sequence_cache = tmp_path / "sequences.json"
    sequence_cache.write_text(
        json.dumps({symbol: "M" for symbol in ("A", "B", "C")}),
        encoding="utf-8",
    )
    argv = [
        "precompute_esm2_embeddings.py",
        "--benchmark-csv",
        str(csv),
        "--symbol-column",
        "perturbation_gene",
        "--out",
        str(output),
        "--seq-cache",
        str(sequence_cache),
    ]
    if strict:
        argv.append("--require-complete-coverage")
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(
        MOD,
        "load_or_fetch_sequences",
        lambda symbols, cache, **kwargs: {symbol: _record("M") for symbol in symbols},
    )
    monkeypatch.setattr(
        MOD,
        "embed_sequences",
        lambda *args, **kwargs: (
            np.ones((3, 4), dtype=np.float32),
            resolved,
            {
                "model_class": "tests.TinyModel",
                "model_state_sha256": "1" * 64,
                "model_config_sha256": "3" * 64,
                "tokenizer_class": "tests.TinyTokenizer",
                "tokenizer_vocabulary_config_sha256": "2" * 64,
            },
        ),
    )
    MOD.main()
    return output


@pytest.mark.parametrize("existing_output", [False, True])
def test_strict_asset_rejects_incomplete_coverage_before_output_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    existing_output: bool,
) -> None:
    output = tmp_path / "esm2.npz"
    if existing_output:
        output.write_bytes(b"existing-asset")

    with pytest.raises(ValueError, match="2/3"):
        _run_asset_cli(
            monkeypatch,
            tmp_path,
            np.asarray([True, True, False]),
            strict=True,
        )

    if existing_output:
        assert output.read_bytes() == b"existing-asset"
    else:
        assert not output.exists()


def test_strict_asset_writes_when_coverage_is_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = _run_asset_cli(
        monkeypatch,
        tmp_path,
        np.asarray([True, True, True]),
        strict=True,
    )

    with np.load(output, allow_pickle=True) as payload:
        assert payload["symbols"].tolist() == ["A", "B", "C"]
        assert int(payload["resolved"].sum()) == 3
    sidecar = json.loads(
        output.with_suffix(".npz.provenance.json").read_text(encoding="utf-8")
    )
    assert sidecar["embedding_artifact"]["sha256"] == hashlib.sha256(
        output.read_bytes()
    ).hexdigest()
    assert list(sidecar["sequence_source"]["sequence_sha256_by_symbol"]) == [
        "A",
        "B",
        "C",
    ]


def test_model_state_hash_binds_sorted_names_shapes_dtypes_and_bytes() -> None:
    first = torch.nn.Module()
    first.register_parameter(
        "weight", torch.nn.Parameter(torch.tensor([[1.0, 2.0]], dtype=torch.float32))
    )
    second = torch.nn.Module()
    second.register_parameter(
        "weight", torch.nn.Parameter(torch.tensor([[1.0, 3.0]], dtype=torch.float32))
    )
    assert MOD.hash_model_state(first) == MOD.hash_model_state(first)
    assert MOD.hash_model_state(first) != MOD.hash_model_state(second)


def test_tokenizer_hash_binds_vocabulary_and_configuration() -> None:
    class TinyTokenizer:
        model_max_length = 1024
        special_tokens_map = {"cls_token": "<cls>"}

        def __init__(self, vocab: dict[str, int]) -> None:
            self._vocab = vocab
            self.init_kwargs = {"do_lower_case": False}

        def get_vocab(self) -> dict[str, int]:
            return self._vocab

    first = TinyTokenizer({"A": 0, "B": 1})
    second = TinyTokenizer({"A": 0, "B": 2})
    assert MOD.hash_tokenizer_vocabulary_config(first) != (
        MOD.hash_tokenizer_vocabulary_config(second)
    )


def test_provenance_rejects_tampered_vectors_model_hash_and_sequence_mapping(
    tmp_path: Path,
) -> None:
    benchmark = tmp_path / "genes.csv"
    benchmark.write_text("gene_symbol\nA\n", encoding="utf-8")
    cache = tmp_path / "sequences.json"
    cache.write_text('{"A":"MSEQ"}', encoding="utf-8")
    npz = tmp_path / "esm2.npz"
    sidecar = tmp_path / "esm2.provenance.json"
    mapping_json = tmp_path / "esm2.mapping.json"
    mapping_csv = tmp_path / "esm2.mapping.csv"
    MOD.write_embedding_with_provenance(
        npz,
        sidecar,
        symbols=["A"],
        vectors=np.ones((1, 4), dtype=np.float32),
        resolved=np.ones(1, dtype=bool),
        sequences={"A": "MSEQ"},
        uniprot_records={"A": _record()},
        sequence_cache=cache,
        mapping_json_output=mapping_json,
        mapping_csv_output=mapping_csv,
        benchmark_csv=benchmark,
        symbol_columns=("gene_symbol",),
        requested_model_id="tiny-esm2",
        runtime_identity={
            "model_class": "tests.TinyModel",
            "model_state_sha256": "1" * 64,
            "model_config_sha256": "3" * 64,
            "tokenizer_class": "tests.TinyTokenizer",
            "tokenizer_vocabulary_config_sha256": "2" * 64,
        },
    )
    pristine_npz = npz.read_bytes()
    pristine_sidecar = sidecar.read_text(encoding="utf-8")
    pristine_mapping = mapping_json.read_text(encoding="utf-8")

    mapping_payload = json.loads(pristine_mapping)
    mapping_payload["records"][0]["primary_accession"] = "TAMPERED"
    mapping_json.write_text(json.dumps(mapping_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="mapping artifact SHA256 mismatch"):
        load_and_authenticate_esm2_provenance(
            sidecar,
            npz,
            mapping_json_path=mapping_json,
            mapping_csv_path=mapping_csv,
        )
    mapping_json.write_text(pristine_mapping, encoding="utf-8")
    mapping_payload = json.loads(pristine_mapping)
    mapping_payload["records"][0]["sequence_sha256"] = "4" * 64
    mapping_json.write_text(json.dumps(mapping_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="mapping artifact SHA256 mismatch"):
        load_and_authenticate_esm2_provenance(
            sidecar,
            npz,
            mapping_json_path=mapping_json,
            mapping_csv_path=mapping_csv,
        )
    mapping_json.write_text(pristine_mapping, encoding="utf-8")

    np.savez(
        npz,
        symbols=np.asarray(["A"], dtype=object),
        vectors=np.zeros((1, 4), dtype=np.float32),
        resolved=np.ones(1, dtype=bool),
    )
    with pytest.raises(ValueError, match="sha256 mismatch"):
        load_and_authenticate_esm2_provenance(
            sidecar,
            npz,
            mapping_json_path=mapping_json,
            mapping_csv_path=mapping_csv,
        )

    npz.write_bytes(pristine_npz)
    payload = json.loads(pristine_sidecar)
    payload["loaded_model"]["state_sha256"] = "tampered"
    sidecar.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="loaded_model hash"):
        load_and_authenticate_esm2_provenance(
            sidecar,
            npz,
            mapping_json_path=mapping_json,
            mapping_csv_path=mapping_csv,
        )

    payload = json.loads(pristine_sidecar)
    payload["sequence_source"]["sequence_sha256_by_symbol"] = {"B": "3" * 64}
    sidecar.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="sequence mapping membership"):
        load_and_authenticate_esm2_provenance(
            sidecar,
            npz,
            mapping_json_path=mapping_json,
            mapping_csv_path=mapping_csv,
        )


def test_writer_rejects_identity_for_unresolved_symbol_before_publish(
    tmp_path: Path,
) -> None:
    benchmark = tmp_path / "genes.csv"
    benchmark.write_text("gene_symbol\nA\n", encoding="utf-8")
    cache = tmp_path / "sequences.json"
    cache.write_text("{}", encoding="utf-8")
    output = tmp_path / "esm2.npz"
    with pytest.raises(ValueError, match="exact membership"):
        MOD.write_embedding_with_provenance(
            output,
            tmp_path / "esm2.provenance.json",
            symbols=["A"],
            vectors=np.ones((1, 4), dtype=np.float32),
            resolved=np.zeros(1, dtype=bool),
            sequences={"A": "MSEQ"},
            uniprot_records={"A": _record()},
            sequence_cache=cache,
            mapping_json_output=tmp_path / "esm2.mapping.json",
            mapping_csv_output=tmp_path / "esm2.mapping.csv",
            benchmark_csv=benchmark,
            symbol_columns=("gene_symbol",),
            requested_model_id="tiny-esm2",
            runtime_identity={
                "model_class": "tests.TinyModel",
                "model_state_sha256": "1" * 64,
                "model_config_sha256": "3" * 64,
                "tokenizer_class": "tests.TinyTokenizer",
                "tokenizer_vocabulary_config_sha256": "2" * 64,
            },
        )
    assert not output.exists()
