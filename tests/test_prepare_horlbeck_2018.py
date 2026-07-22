import csv
import json
from pathlib import Path

import pytest

from scripts.prepare_horlbeck_2018 import (
    DEFAULT_ARCHIVE,
    DEFAULT_CDT,
    DEFAULT_CONDITIONS,
    DEFAULT_FIXED_SPLIT,
    DEFAULT_HGNC,
    prepare_horlbeck,
    read_k562_gi_matrix,
    resolve_gene_coverage,
    verify_frozen_inputs,
)


def _write_cdt(path: Path, reverse_score: float = -4.0) -> None:
    rows = [
        ["GID", "", "NAME", "GWEIGHT", "OLD1", "B"],
        ["AID", "", "", "", "ARRY0X", "ARRY1X"],
        ["EWEIGHT", "", "", "", "1.000000", "1.000000"],
        ["GENE0X", "OLD1", "OLD1", "1.000000", "0.0", "-4.0"],
        ["GENE1X", "B", "B", "1.000000", str(reverse_score), "0.0"],
    ]
    with path.open("w", newline="") as handle:
        csv.writer(handle, delimiter="\t").writerows(rows)


def test_read_k562_gi_matrix_validates_symmetry(tmp_path: Path) -> None:
    path = tmp_path / "matrix.cdt"
    _write_cdt(path)
    genes, scores = read_k562_gi_matrix(path)
    assert genes == ["OLD1", "B"]
    assert scores[0][1] == -4.0

    _write_cdt(path, reverse_score=-3.5)
    with pytest.raises(ValueError, match="not exactly symmetric"):
        read_k562_gi_matrix(path)


def test_resolve_gene_coverage_requires_observation_and_fixed_pool() -> None:
    resolved = resolve_gene_coverage(
        "OLD1",
        replogle_cells={"NEW1": 10},
        fixed_pool={"NEW1"},
        hgnc_aliases={"OLD1": {"NEW1"}},
        min_cells=8,
    )
    assert resolved.canonical_symbol == "NEW1"
    assert resolved.match_method == "hgnc_alias"
    assert resolved.exp05_observed_covered

    below_threshold = resolve_gene_coverage(
        "OLD1",
        replogle_cells={"NEW1": 7},
        fixed_pool={"NEW1"},
        hgnc_aliases={"OLD1": {"NEW1"}},
        min_cells=8,
    )
    assert not below_threshold.exp05_observed_covered

    ambiguous = resolve_gene_coverage(
        "OLD1",
        replogle_cells={"NEW1": 10, "NEW2": 10},
        fixed_pool={"NEW1", "NEW2"},
        hgnc_aliases={"OLD1": {"NEW1", "NEW2"}},
        min_cells=8,
    )
    assert ambiguous.canonical_symbol is None
    assert not ambiguous.exp05_observed_covered


def test_prepare_horlbeck_freezes_pair_and_coverage_contract(tmp_path: Path) -> None:
    cdt = tmp_path / "matrix.cdt"
    rows = [
        ["GID", "", "NAME", "GWEIGHT", "A", "OLD1", "C"],
        ["AID", "", "", "", "ARRY0X", "ARRY1X", "ARRY2X"],
        ["EWEIGHT", "", "", "", "1", "1", "1"],
        ["GENE0X", "A", "A", "1", "0", "-3", "-3.1"],
        ["GENE1X", "OLD1", "OLD1", "1", "-3", "0", "2"],
        ["GENE2X", "C", "C", "1", "-3.1", "2", "0"],
    ]
    with cdt.open("w", newline="") as handle:
        csv.writer(handle, delimiter="\t").writerows(rows)

    hgnc = tmp_path / "hgnc.tsv"
    with hgnc.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["status", "symbol", "prev_symbol", "alias_symbol"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerow(
            {
                "status": "Approved",
                "symbol": "NEW1",
                "prev_symbol": "OLD1",
                "alias_symbol": "",
            }
        )

    conditions = tmp_path / "conditions.csv"
    with conditions.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["source_name", "n_parsed_genes", "parsed_genes", "n_cells"],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "source_name": "replogle_k562_gwps",
                    "n_parsed_genes": "1",
                    "parsed_genes": "A",
                    "n_cells": "10",
                },
                {
                    "source_name": "replogle_k562_gwps",
                    "n_parsed_genes": "1",
                    "parsed_genes": "NEW1",
                    "n_cells": "9",
                },
                {
                    "source_name": "wrong_source",
                    "n_parsed_genes": "1",
                    "parsed_genes": "C",
                    "n_cells": "100",
                },
                {
                    "source_name": "replogle_k562_gwps",
                    "n_parsed_genes": "1",
                    "parsed_genes": "C",
                    "n_cells": "7",
                },
            ]
        )

    fixed = tmp_path / "fixed.csv"
    with fixed.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["perturbation_gene"])
        writer.writeheader()
        writer.writerows([{"perturbation_gene": gene} for gene in ["A", "NEW1", "C"]])

    output_dir = tmp_path / "output"
    summary = prepare_horlbeck(cdt, hgnc, conditions, fixed, output_dir, min_cells=8)
    assert summary["unique_off_diagonal_pair_count"] == 3
    assert summary["strong_sl_pair_count"] == 1
    assert summary["observed_replogle_gene_count"] == 2
    assert summary["exp05_observed_gene_count"] == 2
    assert summary["exp05_observed_pair_count"] == 1

    with (output_dir / "k562_gene_pairs.csv").open(newline="") as handle:
        pairs = list(csv.DictReader(handle))
    assert len(pairs) == 3
    assert (
        next(row for row in pairs if row["gi_score"] == "-3.0")["is_strong_sl"]
        == "False"
    )
    assert json.loads((output_dir / "coverage_summary.json").read_text()) == summary


def test_frozen_hash_check_rejects_changed_input(tmp_path: Path) -> None:
    changed = tmp_path / "changed"
    changed.write_text("changed")
    with pytest.raises(ValueError, match="Frozen archive SHA-256 mismatch"):
        verify_frozen_inputs(changed, changed, changed, changed, changed, min_cells=8)
    with pytest.raises(ValueError, match="Frozen coverage requires min_cells=8"):
        verify_frozen_inputs(changed, changed, changed, changed, changed, min_cells=7)


@pytest.mark.skipif(
    not all(
        path.exists()
        for path in [
            DEFAULT_ARCHIVE,
            DEFAULT_CDT,
            DEFAULT_HGNC,
            DEFAULT_CONDITIONS,
            DEFAULT_FIXED_SPLIT,
        ]
    ),
    reason="gitignored Horlbeck/Replogle artifacts are not installed",
)
def test_real_horlbeck_artifacts_match_frozen_invariants(tmp_path: Path) -> None:
    verify_frozen_inputs(
        DEFAULT_ARCHIVE,
        DEFAULT_CDT,
        DEFAULT_HGNC,
        DEFAULT_CONDITIONS,
        DEFAULT_FIXED_SPLIT,
        min_cells=8,
    )
    summary = prepare_horlbeck(
        DEFAULT_CDT,
        DEFAULT_HGNC,
        DEFAULT_CONDITIONS,
        DEFAULT_FIXED_SPLIT,
        tmp_path,
        min_cells=8,
    )
    assert summary["matrix_gene_count"] == 448
    assert summary["unique_off_diagonal_pair_count"] == 100_128
    assert summary["strong_sl_pair_count"] == 1_523
    assert summary["observed_replogle_gene_count"] == 436
    assert summary["exp05_observed_gene_count"] == 408
