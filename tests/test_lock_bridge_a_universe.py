import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.lock_bridge_a_universe import (
    assert_complete_clique_topology,
    assert_frozen_counts,
    check_pool_membership,
    check_vocabulary_coverage,
    drop_nonqueryable_pairs,
    filter_exp05_ready_universe,
    lock_bridge_a_universe,
    summarize_gi_scores,
    universe_genes,
)


def _synthetic_pairs() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "gene_a": "A",
                "gene_b": "B",
                "gi_score": -4.0,
                "is_strong_sl": True,
                "gene_a_canonical": "A",
                "gene_b_canonical": "B",
                "exp05_observed_pair_covered": True,
            },
            {
                "gene_a": "A",
                "gene_b": "C",
                "gi_score": -1.0,
                "is_strong_sl": False,
                "gene_a_canonical": "A",
                "gene_b_canonical": "C",
                "exp05_observed_pair_covered": True,
            },
            {
                "gene_a": "B",
                "gene_b": "D",
                "gi_score": 2.0,
                "is_strong_sl": False,
                "gene_a_canonical": "B",
                "gene_b_canonical": "D",
                "exp05_observed_pair_covered": False,
            },
        ]
    )


def test_filter_exp05_ready_universe_keeps_only_covered_rows() -> None:
    covered = filter_exp05_ready_universe(_synthetic_pairs())
    assert len(covered) == 2
    assert set(covered["gene_b"]) == {"B", "C"}


def test_filter_exp05_ready_universe_requires_coverage_column() -> None:
    with pytest.raises(ValueError, match="missing required column"):
        filter_exp05_ready_universe(pd.DataFrame({"gene_a": ["A"]}))


def test_universe_genes_is_sorted_union_of_canonical_symbols() -> None:
    covered = filter_exp05_ready_universe(_synthetic_pairs())
    assert universe_genes(covered) == ["A", "B", "C"]


def test_assert_frozen_counts_passes_on_matching_counts() -> None:
    assert_frozen_counts(n_pairs=2, n_genes=3, expected_pairs=2, expected_genes=3)


def test_assert_frozen_counts_raises_on_mismatch() -> None:
    with pytest.raises(ValueError, match="drifted from the frozen bound"):
        assert_frozen_counts(
            n_pairs=2, n_genes=3, expected_pairs=83_028, expected_genes=408
        )


def _clique_pairs(genes: list[str]) -> pd.DataFrame:
    """Build a synthetic complete-clique pair table over ``genes``."""
    rows = [
        {"gene_a_canonical": a, "gene_b_canonical": b}
        for a, b in itertools.combinations(sorted(genes), 2)
    ]
    return pd.DataFrame(rows)


def test_assert_complete_clique_topology_passes_on_genuine_clique() -> None:
    genes = ["A", "B", "C", "D"]
    covered = _clique_pairs(genes)
    assert_complete_clique_topology(covered, genes)  # must not raise


def test_assert_complete_clique_topology_raises_on_self_pair() -> None:
    genes = ["A", "B", "C"]
    covered = pd.concat(
        [
            _clique_pairs(genes),
            pd.DataFrame([{"gene_a_canonical": "A", "gene_b_canonical": "A"}]),
        ],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="self-pair"):
        assert_complete_clique_topology(covered, genes)


def test_assert_complete_clique_topology_raises_on_missing_and_duplicate() -> None:
    """A missing pair masked by a compensating duplicate must still raise.

    The row count matches C(n, 2) exactly, so a naive count-only check would
    pass this malformed table; the topology check must reject it anyway.
    """
    genes = ["A", "B", "C", "D"]
    covered = _clique_pairs(genes)
    # Drop the (A, B) pair and duplicate (C, D) instead, so the row count is
    # unchanged (still C(4, 2) == 6) but the pair set is wrong.
    covered = covered[
        ~((covered["gene_a_canonical"] == "A") & (covered["gene_b_canonical"] == "B"))
    ].reset_index(drop=True)
    duplicate_row = covered[
        (covered["gene_a_canonical"] == "C") & (covered["gene_b_canonical"] == "D")
    ]
    covered = pd.concat([covered, duplicate_row], ignore_index=True)
    assert len(covered) == 6  # still C(4, 2); a count-only check would pass

    with pytest.raises(ValueError, match="duplicate canonical pair"):
        assert_complete_clique_topology(covered, genes)


def test_summarize_gi_scores_reports_strong_sl_and_gi_range() -> None:
    covered = filter_exp05_ready_universe(_synthetic_pairs())
    summary = summarize_gi_scores(covered)
    assert summary["strong_sl_pair_count"] == 1
    assert summary["gi_score_min"] == -4.0
    assert summary["gi_score_max"] == -1.0
    assert summary["gi_score_median"] == -2.5


def _write_esm2_npz(path: Path, symbols: list[str]) -> None:
    vectors = np.zeros((len(symbols), 4), dtype=np.float32)
    resolved = np.ones(len(symbols), dtype=bool)
    np.savez(
        path,
        symbols=np.array(symbols, dtype=object),
        vectors=vectors,
        resolved=resolved,
    )


def test_check_vocabulary_coverage_splits_found_and_missing(tmp_path: Path) -> None:
    npz_path = tmp_path / "esm2.npz"
    _write_esm2_npz(npz_path, ["A", "B"])
    found, missing = check_vocabulary_coverage(["A", "B", "C"], npz_path)
    assert found == ["A", "B"]
    assert missing == ["C"]


def test_drop_nonqueryable_pairs_drops_every_pair_touching_missing_gene() -> None:
    covered = filter_exp05_ready_universe(_synthetic_pairs())
    dropped = drop_nonqueryable_pairs(covered, ["C"])
    assert dropped["gene_b"].tolist() == ["B"]


def test_check_pool_membership_counts_split_roles() -> None:
    pool = pd.DataFrame(
        {
            "perturbation_gene": ["A", "B", "C"],
            "split_role": ["train", "train", "validation"],
        }
    )
    missing, role_counts = check_pool_membership(["A", "B", "C", "D"], pool)
    assert missing == ["D"]
    assert role_counts == {"train": 2, "validation": 1}


def _clique_synthetic_pairs() -> pd.DataFrame:
    """Build a genuine complete-triangle pair table (A-B, A-C, B-C; all covered).

    Unlike ``_synthetic_pairs()``, every pair among the three genes is
    present exactly once, so this table satisfies the complete-clique
    topology invariant and is safe to feed through the full
    ``lock_bridge_a_universe`` pipeline.
    """
    return pd.DataFrame(
        [
            {
                "gene_a": "A",
                "gene_b": "B",
                "gi_score": -4.0,
                "is_strong_sl": True,
                "gene_a_canonical": "A",
                "gene_b_canonical": "B",
                "exp05_observed_pair_covered": True,
            },
            {
                "gene_a": "A",
                "gene_b": "C",
                "gi_score": -1.0,
                "is_strong_sl": False,
                "gene_a_canonical": "A",
                "gene_b_canonical": "C",
                "exp05_observed_pair_covered": True,
            },
            {
                "gene_a": "B",
                "gene_b": "C",
                "gi_score": -2.0,
                "is_strong_sl": False,
                "gene_a_canonical": "B",
                "gene_b_canonical": "C",
                "exp05_observed_pair_covered": True,
            },
        ]
    )


def test_lock_bridge_a_universe_writes_artifacts_and_manifest(tmp_path: Path) -> None:
    pairs_csv = tmp_path / "pairs.csv"
    _clique_synthetic_pairs().to_csv(pairs_csv, index=False)

    pool_csv = tmp_path / "pool.csv"
    pd.DataFrame(
        {
            "perturbation_gene": ["A", "B", "C"],
            "split_role": ["train", "train", "validation"],
        }
    ).to_csv(pool_csv, index=False)

    npz_path = tmp_path / "esm2.npz"
    _write_esm2_npz(npz_path, ["A", "B", "C"])

    output_dir = tmp_path / "universe"
    manifest = lock_bridge_a_universe(
        pairs_csv,
        pool_csv,
        npz_path,
        output_dir,
        expected_pair_count=3,
        expected_gene_count=3,
    )

    assert manifest["frozen_gene_count"] == 3
    assert manifest["frozen_pair_count"] == 3
    assert manifest["complete_clique_verified"] is True
    assert manifest["expected_complete_clique_pair_count"] == 3
    assert manifest["esm2_vocabulary_check"]["missing_count"] == 0
    assert (output_dir / "candidate_universe_pairs.csv").exists()
    assert (output_dir / "candidate_universe_genes.txt").read_text() == "A\nB\nC\n"
    on_disk = json.loads((output_dir / "candidate_universe_manifest.json").read_text())
    assert on_disk == manifest


def test_lock_bridge_a_universe_raises_on_frozen_count_mismatch(tmp_path: Path) -> None:
    pairs_csv = tmp_path / "pairs.csv"
    _synthetic_pairs().to_csv(pairs_csv, index=False)
    pool_csv = tmp_path / "pool.csv"
    pd.DataFrame(
        {"perturbation_gene": ["A", "B", "C"], "split_role": ["train"] * 3}
    ).to_csv(pool_csv, index=False)
    npz_path = tmp_path / "esm2.npz"
    _write_esm2_npz(npz_path, ["A", "B", "C"])

    with pytest.raises(ValueError, match="drifted from the frozen bound"):
        lock_bridge_a_universe(
            pairs_csv,
            pool_csv,
            npz_path,
            tmp_path / "universe",
            expected_pair_count=83_028,
            expected_gene_count=408,
        )


def test_lock_bridge_a_universe_raises_when_vocab_missing_by_default(
    tmp_path: Path,
) -> None:
    """A frozen universe must be queryable by construction: fail closed."""
    pairs_csv = tmp_path / "pairs.csv"
    _synthetic_pairs().to_csv(pairs_csv, index=False)
    pool_csv = tmp_path / "pool.csv"
    pd.DataFrame(
        {
            "perturbation_gene": ["A", "B", "C"],
            "split_role": ["train", "train", "validation"],
        }
    ).to_csv(pool_csv, index=False)
    npz_path = tmp_path / "esm2.npz"
    _write_esm2_npz(npz_path, ["A", "B"])  # "C" is absent from the vocabulary.

    output_dir = tmp_path / "universe"
    with pytest.raises(ValueError, match="missing from the ESM2 perturbation"):
        lock_bridge_a_universe(
            pairs_csv,
            pool_csv,
            npz_path,
            output_dir,
            expected_pair_count=2,
            expected_gene_count=3,
        )
    assert not output_dir.exists(), "no artifact may be written when unqueryable"


def test_lock_bridge_a_universe_drop_nonqueryable_recomputes_universe(
    tmp_path: Path,
) -> None:
    """--drop-nonqueryable drops exactly the offending genes/pairs and logs it."""
    pairs_csv = tmp_path / "pairs.csv"
    _synthetic_pairs().to_csv(pairs_csv, index=False)
    pool_csv = tmp_path / "pool.csv"
    pd.DataFrame(
        {
            "perturbation_gene": ["A", "B", "C"],
            "split_role": ["train", "train", "validation"],
        }
    ).to_csv(pool_csv, index=False)
    npz_path = tmp_path / "esm2.npz"
    _write_esm2_npz(npz_path, ["A", "B"])  # "C" is absent from the vocabulary.

    output_dir = tmp_path / "universe"
    manifest = lock_bridge_a_universe(
        pairs_csv,
        pool_csv,
        npz_path,
        output_dir,
        expected_pair_count=2,
        expected_gene_count=3,
        drop_nonqueryable=True,
    )

    assert manifest["dropped_nonqueryable_genes"] == ["C"]
    assert manifest["dropped_pair_count"] == 1
    assert manifest["frozen_gene_count"] == 2
    assert manifest["frozen_pair_count"] == 1
    assert manifest["esm2_vocabulary_check"] == {
        "found_count": 2,
        "missing_count": 0,
        "missing_genes": [],
    }
    assert (output_dir / "candidate_universe_genes.txt").read_text() == "A\nB\n"
    written_pairs = pd.read_csv(output_dir / "candidate_universe_pairs.csv")
    assert written_pairs["gene_b"].tolist() == ["B"]
    on_disk = json.loads((output_dir / "candidate_universe_manifest.json").read_text())
    assert on_disk == manifest
