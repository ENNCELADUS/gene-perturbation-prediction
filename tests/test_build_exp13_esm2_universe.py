from pathlib import Path
import shlex
import numpy as np
import pandas as pd
import pytest
from src.data.geneeffect import Exp13Split
from src.data.prepare.build_exp13_esm2_universe import (
    build_coverage_universe,
    build_embedding_union,
    build_precompute_command,
    inspect_npz_coverage,
    require_npz_coverage,
    write_embedding_union,
    restrict_coverage_universe_to_copy_prior,
)


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


def test_embedding_union_is_sorted_without_changing_separate_scored_order() -> None:
    scored = ("B", "A")
    assert build_embedding_union(scored, ("A", "C", "D")) == ("A", "B", "C", "D")
    assert scored == ("B", "A")


def _write_npz(
    path: Path, symbols: list[str], resolved: list[bool], *, width: int = 3
) -> None:
    np.savez(
        path,
        symbols=np.asarray(symbols, dtype=object),
        vectors=np.ones((len(symbols), width), dtype=np.float32),
        resolved=np.asarray(resolved, dtype=bool),
    )


def test_npz_must_cover_response_only_union_member(tmp_path: Path) -> None:
    path = tmp_path / "esm2.npz"
    _write_npz(path, ["PASS"], [True])
    report = inspect_npz_coverage(path, ["PASS", "RESPONSE_ONLY"])
    assert report.missing == ("RESPONSE_ONLY",)
    with pytest.raises(ValueError, match=r"missing 1/2"):
        require_npz_coverage(path, ["PASS", "RESPONSE_ONLY"], expected_width=3)


def test_npz_allows_unresolved_candidate_but_not_unresolved_response(
    tmp_path: Path,
) -> None:
    path = tmp_path / "esm2.npz"
    _write_npz(path, ["CANDIDATE", "RESPONSE"], [False, True])
    report = require_npz_coverage(
        path,
        ["CANDIDATE", "RESPONSE"],
        must_resolve_symbols=["RESPONSE"],
        expected_width=3,
    )
    assert report.missing == ("CANDIDATE",)
    with pytest.raises(ValueError, match="must-resolve"):
        require_npz_coverage(
            path,
            ["CANDIDATE", "RESPONSE"],
            must_resolve_symbols=["CANDIDATE", "RESPONSE"],
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


def test_writer_separates_scored_and_response_union_and_reports_exclusions(tmp_path):
    path = tmp_path / "esm2.npz"
    _write_npz(path, ["B", "A", "RESPONSE", "UNRESOLVED"], [True, True, True, False])
    result = write_embedding_union(
        scored_symbols=("B", "A", "UNRESOLVED"),
        response_symbols=("RESPONSE", "UNRESOLVED"),
        esm2_path=path,
        output_dir=tmp_path / "prepared",
    )
    assert result["common_gene_panel"] == ["B", "A"]
    assert result["embedding_union"] == ["A", "B", "RESPONSE", "UNRESOLVED"]
    assert result["esm2_order"] == ["B", "A", "RESPONSE"]
    assert result["unresolved_symbols"] == ["UNRESOLVED"]
    panel = pd.read_csv(tmp_path / "prepared/common_gene_panel.csv")
    assert panel.gene_symbol.tolist() == ["B", "A"]
    command = shlex.split(
        build_precompute_command(
            tmp_path / "union with spaces.csv",
            path,
            tmp_path / "sequences.json",
            local_files_only=True,
        )
    )
    assert command[4:6] == [
        "src.data.prepare.precompute_esm2_embeddings",
        "--benchmark-csv",
    ]
    assert "--local-files-only" in command


def test_common_panel_keeps_finite_k562_availability_restriction():
    candidates = build_coverage_universe(_labels(), _split())
    assert restrict_coverage_universe_to_copy_prior(candidates, ("PASS",)).symbols == (
        "PASS",
    )
    with pytest.raises(ValueError, match="empty"):
        restrict_coverage_universe_to_copy_prior(candidates, ("OTHER",))
