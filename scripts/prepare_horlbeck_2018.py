"""Normalize the Horlbeck 2018 K562 GI map and audit exp05 coverage."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path


DEFAULT_CDT = Path("data/sl_dependency_v0/raw/horlbeck_2018/K562_gene.cdt")
DEFAULT_ARCHIVE = Path("data/sl_dependency_v0/raw/horlbeck_2018/GI_map_treeview.zip")
DEFAULT_HGNC = Path(
    "data/sl_dependency_v0/raw/horlbeck_2018/hgnc_complete_set_20260722.txt"
)
DEFAULT_CONDITIONS = Path(
    "data/sl_dependency_v0/interim/k562_perturbseq_sl_condition_coverage_conditions.csv"
)
DEFAULT_FIXED_SPLIT = Path(
    "data/sl_dependency_v0/splits/k562_pool_depmap_fixed_seed42.csv"
)
DEFAULT_OUTPUT_DIR = Path("data/sl_dependency_v0/processed/horlbeck_2018")
REPLOGLE_SOURCE = "replogle_k562_gwps"
STRONG_SL_THRESHOLD = -3.0
FROZEN_INPUT_SHA256 = {
    "archive": "d389b3ebdd2c88f86cc446f272f3ff6526df635d666cd17d67440bf10c0f9115",
    "cdt": "e3f7c4caa860cda8daec94be56e6900d9737781499e5357eaf7b5f18e7534c19",
    "hgnc": "25042e23a296bfa046bf7bc65ee110139cc03a9a55fb7f15ab34b19226cf6417",
    "conditions": "76f8f055549c17a466d13f8c81d7788a8d8ae08eac0817703d4848c653d63397",
    "fixed_split": "c33dbe6cbdd78d0e590a9810eab8a47f74b1a7d1ff3d9872ceca26d65cd3d6ef",
}


@dataclass(frozen=True)
class GeneCoverage:
    """Alignment of one Horlbeck symbol to the Replogle/exp05 vocabulary."""

    horlbeck_symbol: str
    canonical_symbol: str | None
    match_method: str
    replogle_n_cells: int
    observed_replogle_min_cells: bool
    in_exp05_fixed_pool: bool
    exp05_observed_covered: bool


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_frozen_inputs(
    archive_path: Path,
    cdt_path: Path,
    hgnc_path: Path,
    conditions_path: Path,
    fixed_split_path: Path,
    min_cells: int,
) -> None:
    """Refuse to reproduce the frozen result from changed input bytes."""
    if min_cells != 8:
        raise ValueError(
            f"Frozen coverage requires min_cells=8, got {min_cells}; "
            "use --allow-unfrozen-inputs for a non-binding analysis"
        )
    inputs = {
        "archive": archive_path,
        "cdt": cdt_path,
        "hgnc": hgnc_path,
        "conditions": conditions_path,
        "fixed_split": fixed_split_path,
    }
    for name, path in inputs.items():
        actual = sha256_file(path)
        expected = FROZEN_INPUT_SHA256[name]
        if actual != expected:
            raise ValueError(
                f"Frozen {name} SHA-256 mismatch for {path}: "
                f"expected {expected}, got {actual}"
            )


def read_k562_gi_matrix(path: Path) -> tuple[list[str], list[list[float]]]:
    """Read and validate the K562 gene-level Java TreeView matrix."""
    with path.open(newline="") as handle:
        rows = list(csv.reader(handle, delimiter="\t"))
    if len(rows) < 4 or rows[0][:4] != ["GID", "", "NAME", "GWEIGHT"]:
        raise ValueError(f"Unexpected Horlbeck CDT header in {path}")

    column_genes = rows[0][4:]
    row_genes = [row[2] for row in rows[3:]]
    if column_genes != row_genes:
        raise ValueError("Horlbeck K562 matrix row and column genes differ")
    if len(row_genes) != len(set(row_genes)):
        raise ValueError("Horlbeck K562 matrix contains duplicate gene symbols")

    scores = [[float(value) for value in row[4:]] for row in rows[3:]]
    size = len(row_genes)
    if any(len(row) != size for row in scores):
        raise ValueError("Horlbeck K562 matrix is not square")
    for row_index, row in enumerate(scores):
        for column_index, value in enumerate(row):
            if not math.isfinite(value):
                raise ValueError("Horlbeck K562 matrix contains a non-finite score")
            if value != scores[column_index][row_index]:
                raise ValueError(
                    "Horlbeck K562 matrix is not exactly symmetric at "
                    f"({row_index}, {column_index})"
                )
    return row_genes, scores


def load_replogle_cells(path: Path) -> dict[str, int]:
    """Load single-gene K562 GWPS condition counts."""
    with path.open(newline="") as handle:
        rows = csv.DictReader(handle)
        result = {
            row["parsed_genes"]: int(row["n_cells"])
            for row in rows
            if row["source_name"] == REPLOGLE_SOURCE and row["n_parsed_genes"] == "1"
        }
    if not result:
        raise ValueError(f"No {REPLOGLE_SOURCE} single-gene conditions found in {path}")
    return result


def load_fixed_pool(path: Path) -> set[str]:
    """Load the frozen exp05 gene-role manifest."""
    with path.open(newline="") as handle:
        genes = {row["perturbation_gene"] for row in csv.DictReader(handle)}
    if not genes:
        raise ValueError(f"No perturbation genes found in {path}")
    return genes


def load_hgnc_aliases(path: Path) -> dict[str, set[str]]:
    """Map approved, previous, and alias symbols to approved HGNC symbols."""
    aliases: dict[str, set[str]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["status"] != "Approved":
                continue
            current = row["symbol"]
            symbols = [current]
            symbols.extend(row["prev_symbol"].split("|"))
            symbols.extend(row["alias_symbol"].split("|"))
            for symbol in symbols:
                normalized = symbol.strip().strip('"').upper()
                if normalized:
                    aliases.setdefault(normalized, set()).add(current)
    return aliases


def resolve_gene_coverage(
    horlbeck_symbol: str,
    replogle_cells: dict[str, int],
    fixed_pool: set[str],
    hgnc_aliases: dict[str, set[str]],
    min_cells: int,
) -> GeneCoverage:
    """Resolve a Horlbeck symbol conservatively and determine coverage."""
    canonical: str | None
    method: str
    if horlbeck_symbol in replogle_cells:
        canonical = horlbeck_symbol
        method = "exact"
    else:
        casefold_matches = [
            symbol
            for symbol in replogle_cells
            if symbol.upper() == horlbeck_symbol.upper()
        ]
        if len(casefold_matches) == 1:
            canonical = casefold_matches[0]
            method = "casefold"
        else:
            candidates = hgnc_aliases.get(horlbeck_symbol.upper(), set())
            if len(candidates) == 1:
                canonical = next(iter(candidates))
                method = "hgnc_alias"
            elif candidates:
                canonical = None
                method = "ambiguous_hgnc_alias"
            else:
                canonical = None
                method = "unresolved"

    n_cells = replogle_cells.get(canonical, 0) if canonical else 0
    observed = n_cells >= min_cells
    in_fixed_pool = canonical in fixed_pool if canonical else False
    return GeneCoverage(
        horlbeck_symbol=horlbeck_symbol,
        canonical_symbol=canonical,
        match_method=method,
        replogle_n_cells=n_cells,
        observed_replogle_min_cells=observed,
        in_exp05_fixed_pool=in_fixed_pool,
        exp05_observed_covered=observed and in_fixed_pool,
    )


def _write_gene_coverage(path: Path, coverage: list[GeneCoverage]) -> None:
    fieldnames = list(GeneCoverage.__dataclass_fields__)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in coverage:
            writer.writerow({field: getattr(item, field) for field in fieldnames})


def _write_pairs(
    path: Path,
    genes: list[str],
    scores: list[list[float]],
    coverage_by_gene: dict[str, GeneCoverage],
) -> None:
    fieldnames = [
        "gene_a",
        "gene_b",
        "gi_score",
        "is_strong_sl",
        "gene_a_canonical",
        "gene_b_canonical",
        "gene_a_replogle_n_cells",
        "gene_b_replogle_n_cells",
        "both_observed_replogle_min_cells",
        "both_in_exp05_fixed_pool",
        "exp05_observed_pair_covered",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row_index, gene_a in enumerate(genes):
            coverage_a = coverage_by_gene[gene_a]
            for column_index in range(row_index + 1, len(genes)):
                gene_b = genes[column_index]
                coverage_b = coverage_by_gene[gene_b]
                gi_score = scores[row_index][column_index]
                writer.writerow(
                    {
                        "gene_a": gene_a,
                        "gene_b": gene_b,
                        "gi_score": gi_score,
                        "is_strong_sl": gi_score < STRONG_SL_THRESHOLD,
                        "gene_a_canonical": coverage_a.canonical_symbol or "",
                        "gene_b_canonical": coverage_b.canonical_symbol or "",
                        "gene_a_replogle_n_cells": coverage_a.replogle_n_cells,
                        "gene_b_replogle_n_cells": coverage_b.replogle_n_cells,
                        "both_observed_replogle_min_cells": (
                            coverage_a.observed_replogle_min_cells
                            and coverage_b.observed_replogle_min_cells
                        ),
                        "both_in_exp05_fixed_pool": (
                            coverage_a.in_exp05_fixed_pool
                            and coverage_b.in_exp05_fixed_pool
                        ),
                        "exp05_observed_pair_covered": (
                            coverage_a.exp05_observed_covered
                            and coverage_b.exp05_observed_covered
                        ),
                    }
                )


def prepare_horlbeck(
    cdt_path: Path,
    hgnc_path: Path,
    conditions_path: Path,
    fixed_split_path: Path,
    output_dir: Path,
    min_cells: int,
) -> dict[str, object]:
    """Create normalized pair and coverage artifacts and return the summary."""
    genes, scores = read_k562_gi_matrix(cdt_path)
    replogle_cells = load_replogle_cells(conditions_path)
    fixed_pool = load_fixed_pool(fixed_split_path)
    hgnc_aliases = load_hgnc_aliases(hgnc_path)
    coverage = [
        resolve_gene_coverage(gene, replogle_cells, fixed_pool, hgnc_aliases, min_cells)
        for gene in genes
    ]
    coverage_by_gene = {item.horlbeck_symbol: item for item in coverage}

    output_dir.mkdir(parents=True, exist_ok=True)
    gene_path = output_dir / "k562_gene_coverage.csv"
    pair_path = output_dir / "k562_gene_pairs.csv"
    summary_path = output_dir / "coverage_summary.json"
    _write_gene_coverage(gene_path, coverage)
    _write_pairs(pair_path, genes, scores, coverage_by_gene)

    pair_count = len(genes) * (len(genes) - 1) // 2
    observed_gene_count = sum(item.observed_replogle_min_cells for item in coverage)
    exp05_gene_count = sum(item.exp05_observed_covered for item in coverage)
    strong_sl_count = sum(
        scores[row][column] < STRONG_SL_THRESHOLD
        for row in range(len(genes))
        for column in range(row + 1, len(genes))
    )
    summary: dict[str, object] = {
        "source_score_surface": "K562_gene.cdt matrix cells",
        "normalized_gi_column": "gi_score",
        "sl_sign_convention": "negative GI is synergistic/synthetic sick-lethal",
        "strong_sl_rule": "gi_score < -3.0",
        "matrix_gene_count": len(genes),
        "unique_off_diagonal_pair_count": pair_count,
        "strong_sl_pair_count": strong_sl_count,
        "replogle_source": REPLOGLE_SOURCE,
        "min_cells_per_gene": min_cells,
        "observed_replogle_gene_count": observed_gene_count,
        "observed_replogle_pair_count": observed_gene_count
        * (observed_gene_count - 1)
        // 2,
        "exp05_observed_gene_count": exp05_gene_count,
        "exp05_observed_pair_count": exp05_gene_count * (exp05_gene_count - 1) // 2,
        "exp05_observed_pair_fraction": (
            exp05_gene_count * (exp05_gene_count - 1) / (len(genes) * (len(genes) - 1))
        ),
        "missing_exp05_observed_genes": sorted(
            item.horlbeck_symbol for item in coverage if not item.exp05_observed_covered
        ),
    }
    with summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return summary


def main() -> None:
    """Run the Horlbeck normalization and coverage audit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--cdt", type=Path, default=DEFAULT_CDT)
    parser.add_argument("--hgnc", type=Path, default=DEFAULT_HGNC)
    parser.add_argument("--conditions", type=Path, default=DEFAULT_CONDITIONS)
    parser.add_argument("--fixed-split", type=Path, default=DEFAULT_FIXED_SPLIT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-cells", type=int, default=8)
    parser.add_argument(
        "--allow-unfrozen-inputs",
        action="store_true",
        help="Skip frozen SHA-256 checks for an explicitly non-binding analysis.",
    )
    args = parser.parse_args()
    if not args.allow_unfrozen_inputs:
        verify_frozen_inputs(
            args.archive,
            args.cdt,
            args.hgnc,
            args.conditions,
            args.fixed_split,
            args.min_cells,
        )
    summary = prepare_horlbeck(
        args.cdt,
        args.hgnc,
        args.conditions,
        args.fixed_split,
        args.output_dir,
        args.min_cells,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
