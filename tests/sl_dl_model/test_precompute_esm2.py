"""Offline unit tests for scripts/precompute_esm2_embeddings.py.

Tests cover:
  - FIX 1: corrupt JSON cache recovery in load_or_fetch_sequences
  - FIX 2: check_resolution helper raises/warns correctly
  - FIX 3: truncate_sequence helper logs and truncates correctly
  - Existing smoke: universe_symbols reads both columns
"""

from __future__ import annotations

import importlib
import logging
import sys
import types
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

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
        Path(__file__).parent.parent.parent
        / "scripts"
        / "precompute_esm2_embeddings.py"
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

    with patch.object(MOD, "fetch_sequence", return_value="MSEQ") as mock_fetch:
        with caplog.at_level(logging.WARNING, logger="precompute_esm2"):
            result = MOD.load_or_fetch_sequences(["PTEN"], cache)

    # Should have fallen back and fetched the symbol.
    assert "PTEN" in result
    assert result["PTEN"] == "MSEQ"
    mock_fetch.assert_called_once_with("PTEN")

    # A warning must have been emitted about the corrupt cache.
    warning_messages = [
        r.message for r in caplog.records if r.levelno == logging.WARNING
    ]
    assert any(
        "corrupt" in str(m).lower() or "json" in str(m).lower()
        for m in warning_messages
    ), f"Expected a warning about corrupt JSON; got: {warning_messages}"


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
