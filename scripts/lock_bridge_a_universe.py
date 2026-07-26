"""Lock and freeze the Bridge A candidate universe of K562 gene pairs.

Filters the Horlbeck 2018 K562 GI pair table to the exp05-ready coverage
bound -- both genes trained in the exp05 fixed pool, DepMap-label-qualified,
and observed with >=8 Replogle cells -- asserts the frozen 408-gene /
83,028-pair contract, verifies the covered pair table is exactly the
complete graph over those genes (no self-pairs, no duplicate or reversed
pairs, no missing pairs offset by a compensating duplicate), cross-checks
every universe gene against the trained model's ESM2 perturbation
vocabulary and the exp05 fixed pool split roles, and writes frozen
candidate-universe artifacts plus a SHA-256 provenance manifest.

Column note: the exp05-ready bound is encoded by the pair column
``exp05_observed_pair_covered`` (fixed-pool membership AND >=8 observed
Replogle cells for both genes), not by ``both_in_exp05_fixed_pool`` alone
(fixed-pool membership only). On the frozen Horlbeck output,
``both_in_exp05_fixed_pool`` yields 409 genes / 83,436 pairs -- one gene
(ZNF830) is in the fixed training pool but has <8 observed Replogle cells,
so it fails the exp05-ready bound. ``exp05_observed_pair_covered`` yields
exactly 408 genes / 83,028 pairs, matching this script's frozen contract and
the source ``coverage_summary.json`` written by
``scripts/prepare_horlbeck_2018.py``.

Queryability note: the frozen universe must be queryable by the model's
ESM2 perturbation vocabulary by construction. By default, any universe gene
absent from the vocabulary aborts the run with a ``ValueError`` *before* any
artifact is written -- a frozen-but-unqueryable universe is never produced.
Pass ``--drop-nonqueryable`` to instead drop those genes (and every pair
touching them) from the universe, recompute counts, and proceed; the drop is
recorded in the manifest under ``dropped_nonqueryable_genes`` /
``dropped_pair_count``.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
from collections import Counter
from pathlib import Path

import pandas as pd

from aivc_model.gene_splits import sha256_file
from aivc_model.gene_embeddings import load_esm2_embeddings

logger = logging.getLogger("lock_bridge_a_universe")

DEFAULT_PAIRS_CSV = Path(
    "data/sl_dependency_v0/processed/horlbeck_2018/k562_gene_pairs.csv"
)
DEFAULT_POOL_CSV = Path(
    "data/sl_dependency_v0/splits/k562_pool_depmap_fixed_seed42.csv"
)
DEFAULT_ESM2_NPZ = Path("data/esm2/k562_gwps_depmap_esm2_650M.npz")
DEFAULT_OUTPUT_DIR = Path("results/experiments/05_aivc_a_to_b_to_c/bridge_a/universe")

COVERAGE_COLUMN = "exp05_observed_pair_covered"
GI_SCORE_COLUMN = "gi_score"
STRONG_SL_COLUMN = "is_strong_sl"
BOUND_NAME = "exp05_ready / exp05_observed_pair_covered"
EXPECTED_PAIR_COUNT = 83_028
EXPECTED_GENE_COUNT = 408


def filter_exp05_ready_universe(pairs: pd.DataFrame) -> pd.DataFrame:
    """Filter Horlbeck K562 pairs to the exp05-ready Bridge A bound.

    The exp05-ready bound requires both genes to be members of the exp05
    fixed training pool AND observed with >=8 Replogle cells; this is
    encoded by ``exp05_observed_pair_covered``, not by pool membership
    (``both_in_exp05_fixed_pool``) alone.

    Args:
        pairs: Horlbeck K562 gene-pair table with a boolean
            ``exp05_observed_pair_covered`` column.

    Returns:
        The subset of rows where ``exp05_observed_pair_covered`` is true.

    Raises:
        ValueError: If the required coverage column is absent.
    """
    if COVERAGE_COLUMN not in pairs.columns:
        raise ValueError(f"missing required column: {COVERAGE_COLUMN}")
    covered = pairs[pairs[COVERAGE_COLUMN].astype(bool)]
    return covered.reset_index(drop=True)


def universe_genes(covered: pd.DataFrame) -> list[str]:
    """Return the sorted union of canonical genes touched by covered pairs."""
    genes = set(covered["gene_a_canonical"]) | set(covered["gene_b_canonical"])
    return sorted(genes)


def assert_frozen_counts(
    n_pairs: int, n_genes: int, expected_pairs: int, expected_genes: int
) -> None:
    """Fail loudly if the frozen exp05-ready bound has drifted.

    Args:
        n_pairs: Observed covered-pair count.
        n_genes: Observed distinct-gene count.
        expected_pairs: Frozen pair count the caller expects.
        expected_genes: Frozen gene count the caller expects.

    Raises:
        ValueError: If either observed count differs from expected.
    """
    if n_pairs != expected_pairs or n_genes != expected_genes:
        raise ValueError(
            "exp05-ready Bridge A universe drifted from the frozen bound: "
            f"expected {expected_genes} genes / {expected_pairs} pairs, "
            f"observed {n_genes} genes / {n_pairs} pairs"
        )


def assert_complete_clique_topology(covered: pd.DataFrame, genes: list[str]) -> None:
    """Verify the covered pair table is exactly the complete graph over ``genes``.

    A row-count and gene-count match (e.g. 408 genes / 83,028 pairs) does
    not by itself prove the pair table is the complete graph: a table with
    self-pairs, duplicate unordered pairs, reversed duplicates, or a missing
    pair offset by a compensating duplicate can satisfy the same counts.
    This check canonicalizes every pair as a sorted
    ``(gene_a_canonical, gene_b_canonical)`` tuple and asserts the resulting
    tuple set equals exactly ``itertools.combinations(sorted(genes), 2)``.

    Args:
        covered: exp05-ready covered pair table with ``gene_a_canonical`` /
            ``gene_b_canonical`` columns.
        genes: Canonical gene symbols spanning the universe.

    Raises:
        ValueError: If any self-pair exists, any canonical pair (including
            reversed duplicates) appears more than once, or the canonical
            pair set does not equal the complete C(n, 2) clique over
            ``genes``.
    """
    gene_a = covered["gene_a_canonical"]
    gene_b = covered["gene_b_canonical"]

    self_pairs = sorted(set(gene_a[gene_a == gene_b]))
    if self_pairs:
        raise ValueError(
            f"{len(self_pairs)} self-pair gene(s) found in the Bridge A "
            "candidate universe (gene_a_canonical == gene_b_canonical): "
            f"{self_pairs}"
        )

    canonical_pairs = [
        (a, b) if a < b else (b, a) for a, b in zip(gene_a, gene_b, strict=True)
    ]
    pair_counts = Counter(canonical_pairs)
    duplicates = sorted(pair for pair, count in pair_counts.items() if count > 1)
    if duplicates:
        raise ValueError(
            f"{len(duplicates)} duplicate canonical pair(s) found in the "
            "Bridge A candidate universe (unordered pairs, including "
            f"reversed duplicates, must appear exactly once): {duplicates}"
        )

    observed = set(pair_counts)
    expected = set(itertools.combinations(sorted(genes), 2))
    missing = expected - observed
    unexpected = observed - expected
    if missing or unexpected:
        raise ValueError(
            "Bridge A candidate universe is not the complete graph over "
            f"{len(genes)} genes: {len(missing)} missing pair(s), "
            f"{len(unexpected)} unexpected pair(s); expected exactly "
            f"C({len(genes)}, 2) = {len(expected)} pairs."
        )


def check_vocabulary_coverage(
    genes: list[str], esm2_npz: Path
) -> tuple[list[str], list[str]]:
    """Split universe genes into those found/missing in the ESM2 vocabulary.

    Args:
        genes: Upper-case canonical universe gene symbols.
        esm2_npz: Path to the precomputed ESM2 ``.npz`` embedding asset. Its
            resolved symbols are the model's queryable perturbation
            vocabulary.

    Returns:
        A tuple ``(found, missing)`` of gene-symbol lists.
    """
    table = load_esm2_embeddings(esm2_npz)
    vocab = set(table.vectors_by_symbol)
    found = [gene for gene in genes if gene.upper() in vocab]
    missing = [gene for gene in genes if gene.upper() not in vocab]
    return found, missing


def drop_nonqueryable_pairs(
    covered: pd.DataFrame, missing_genes: list[str]
) -> pd.DataFrame:
    """Drop every pair touching an ESM2-vocabulary-missing gene.

    Args:
        covered: exp05-ready covered pair table.
        missing_genes: Canonical gene symbols absent from the ESM2
            perturbation vocabulary.

    Returns:
        The subset of ``covered`` where neither ``gene_a_canonical`` nor
        ``gene_b_canonical`` is in ``missing_genes``.
    """
    missing_set = set(missing_genes)
    keep = ~(
        covered["gene_a_canonical"].isin(missing_set)
        | covered["gene_b_canonical"].isin(missing_set)
    )
    return covered[keep].reset_index(drop=True)


def load_pool_manifest(path: Path) -> pd.DataFrame:
    """Load and validate the exp05 fixed pool manifest.

    Raises:
        ValueError: If the manifest columns do not match the expected schema.
    """
    pool = pd.read_csv(path)
    expected_columns = ["perturbation_gene", "split_role"]
    if pool.columns.tolist() != expected_columns:
        raise ValueError(f"pool manifest columns must be {expected_columns}")
    return pool


def check_pool_membership(
    genes: list[str], pool: pd.DataFrame
) -> tuple[list[str], dict[str, int]]:
    """Confirm universe genes are exp05 fixed-pool members; count split roles.

    Args:
        genes: Canonical universe gene symbols.
        pool: Fixed pool manifest with ``perturbation_gene``/``split_role``.

    Returns:
        A tuple ``(missing_from_pool, split_role_counts)``.
    """
    roles = dict(zip(pool["perturbation_gene"], pool["split_role"], strict=True))
    missing_from_pool = [gene for gene in genes if gene not in roles]
    role_counts: dict[str, int] = {}
    for gene in genes:
        role = roles.get(gene)
        if role is not None:
            role_counts[role] = role_counts.get(role, 0) + 1
    return missing_from_pool, role_counts


def summarize_gi_scores(covered: pd.DataFrame) -> dict[str, float | int]:
    """Summarize strong-SL count and gi_score min/median/max within the universe."""
    gi_scores = covered[GI_SCORE_COLUMN]
    return {
        "strong_sl_pair_count": int(covered[STRONG_SL_COLUMN].astype(bool).sum()),
        "gi_score_min": float(gi_scores.min()),
        "gi_score_median": float(gi_scores.median()),
        "gi_score_max": float(gi_scores.max()),
    }


def write_pair_and_gene_artifacts(
    covered: pd.DataFrame, genes: list[str], output_dir: Path
) -> tuple[Path, Path]:
    """Write the frozen candidate-universe pair CSV and gene list.

    Returns:
        A tuple ``(pairs_path, genes_path)``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    columns = [
        "gene_a",
        "gene_b",
        "gene_a_canonical",
        "gene_b_canonical",
        "gi_score",
        "is_strong_sl",
    ]
    pairs_path = output_dir / "candidate_universe_pairs.csv"
    genes_path = output_dir / "candidate_universe_genes.txt"
    covered[columns].to_csv(pairs_path, index=False)
    genes_path.write_text("\n".join(genes) + "\n")
    return pairs_path, genes_path


def build_manifest(
    *,
    pairs_csv: Path,
    pool_csv: Path,
    esm2_npz: Path,
    pairs_path: Path,
    genes_path: Path,
    n_genes: int,
    n_pairs: int,
    found: list[str],
    missing: list[str],
    role_counts: dict[str, int],
    gi_summary: dict[str, float | int],
    dropped_nonqueryable_genes: list[str],
    dropped_pair_count: int,
    complete_clique_verified: bool,
    expected_complete_clique_pair_count: int,
) -> dict[str, object]:
    """Assemble the provenance manifest for the frozen candidate universe."""
    return {
        "bound_name": BOUND_NAME,
        "frozen_gene_count": n_genes,
        "frozen_pair_count": n_pairs,
        "complete_clique_verified": complete_clique_verified,
        "expected_complete_clique_pair_count": expected_complete_clique_pair_count,
        "input_sha256": {
            "pairs_csv": sha256_file(pairs_csv),
            "pool_csv": sha256_file(pool_csv),
            "esm2_npz": sha256_file(esm2_npz),
        },
        "output_sha256": {
            "candidate_universe_pairs_csv": sha256_file(pairs_path),
            "candidate_universe_genes_txt": sha256_file(genes_path),
        },
        "esm2_vocabulary_check": {
            "found_count": len(found),
            "missing_count": len(missing),
            "missing_genes": missing,
        },
        "dropped_nonqueryable_genes": dropped_nonqueryable_genes,
        "dropped_pair_count": dropped_pair_count,
        "exp05_fixed_pool_membership": {
            "all_genes_in_pool": True,
            "split_role_counts": role_counts,
        },
        "gi_score_summary": gi_summary,
        "filter_column": COVERAGE_COLUMN,
        "filter_column_note": (
            "The exp05-ready bound is 'both genes trained in the exp05 "
            "fixed pool AND DepMap-label-qualified AND observed with >=8 "
            "Replogle cells', encoded by exp05_observed_pair_covered. Pool "
            "membership alone (both_in_exp05_fixed_pool) is a looser "
            "superset that yields 409 genes / 83,436 pairs on this data -- "
            "one gene (ZNF830) is fixed-pool-trained but has <8 observed "
            "Replogle cells, so it fails the exp05-ready bound. "
            "exp05_observed_pair_covered reproduces exactly 408 genes / "
            "83,028 pairs, matching this script's frozen contract and the "
            "source coverage_summary.json written by "
            "scripts/prepare_horlbeck_2018.py."
        ),
        "note": (
            "Bridge A candidate universe: K562 gene pairs where both genes "
            "are exp05-ready (trained in the fixed pool, DepMap-label-"
            "qualified, and >=8 Replogle cells). Frozen from the Horlbeck "
            "2018 K562 GI map (exp05_observed_pair_covered == True) and "
            "cross-checked against the model's ESM2 perturbation vocabulary "
            "and the exp05 fixed pool split-role manifest."
        ),
    }


def lock_bridge_a_universe(
    pairs_csv: Path,
    pool_csv: Path,
    esm2_npz: Path,
    output_dir: Path,
    expected_pair_count: int,
    expected_gene_count: int,
    drop_nonqueryable: bool = False,
) -> dict[str, object]:
    """Filter, verify, freeze, and report the Bridge A candidate universe.

    The frozen universe must be queryable by construction: every gene in it
    must resolve in the model's ESM2 perturbation vocabulary. By default a
    missing gene aborts the run before any artifact is written. Pass
    ``drop_nonqueryable=True`` to instead drop the offending genes (and
    every pair touching them) and proceed with the recomputed universe.

    Raises:
        ValueError: If the observed counts drift from the frozen bound, if
            a universe gene is absent from the exp05 fixed pool, or if
            ``drop_nonqueryable`` is False and a universe gene is missing
            from the ESM2 perturbation vocabulary.
    """
    logger.info("loading Horlbeck K562 pairs from %s", pairs_csv)
    pairs = pd.read_csv(pairs_csv)
    covered = filter_exp05_ready_universe(pairs)
    genes = universe_genes(covered)
    assert_frozen_counts(
        len(covered), len(genes), expected_pair_count, expected_gene_count
    )
    logger.info(
        "frozen exp05-ready bound confirmed: %d genes / %d pairs",
        len(genes),
        len(covered),
    )

    found, missing = check_vocabulary_coverage(genes, esm2_npz)
    dropped_nonqueryable_genes: list[str] = []
    dropped_pair_count = 0
    if missing:
        if not drop_nonqueryable:
            raise ValueError(
                f"{len(missing)}/{len(genes)} Bridge A universe genes are "
                "missing from the ESM2 perturbation vocabulary and cannot "
                f"be queried by the frozen model: {missing}. Universe "
                "locking aborted before writing any artifact -- a frozen "
                "universe must be queryable by construction. Pass "
                "--drop-nonqueryable to drop these genes (and every pair "
                "touching them) instead of failing."
            )
        pre_drop_pair_count = len(covered)
        covered = drop_nonqueryable_pairs(covered, missing)
        genes = universe_genes(covered)
        dropped_nonqueryable_genes = missing
        dropped_pair_count = pre_drop_pair_count - len(covered)
        logger.warning(
            "--drop-nonqueryable: dropped %d nonqueryable gene(s) (%s) and "
            "%d pair(s) touching them; universe recomputed to %d genes / "
            "%d pairs",
            len(dropped_nonqueryable_genes),
            dropped_nonqueryable_genes,
            dropped_pair_count,
            len(genes),
            len(covered),
        )
        found, missing = check_vocabulary_coverage(genes, esm2_npz)

    if missing:
        # Unreachable in practice: drop_nonqueryable removes every missing
        # gene above. Kept as a hard guard so a frozen-but-unqueryable
        # universe can never be written.
        raise AssertionError(
            "internal error: frozen Bridge A universe still has "
            f"{len(missing)} gene(s) missing from the ESM2 perturbation "
            f"vocabulary after drop-nonqueryable resolution: {missing}"
        )
    logger.info(
        "all %d frozen universe genes are queryable in the ESM2 "
        "perturbation vocabulary",
        len(found),
    )

    assert_complete_clique_topology(covered, genes)
    expected_complete_clique_pair_count = len(genes) * (len(genes) - 1) // 2
    logger.info(
        "complete_clique_verified: %d genes / %d pairs form the complete "
        "graph (no self-pairs, no duplicate/reversed pairs, no missing "
        "pairs)",
        len(genes),
        expected_complete_clique_pair_count,
    )

    pool = load_pool_manifest(pool_csv)
    missing_from_pool, role_counts = check_pool_membership(genes, pool)
    if missing_from_pool:
        raise ValueError(
            f"{len(missing_from_pool)} universe genes are absent from the "
            f"exp05 fixed pool: {missing_from_pool}"
        )
    logger.info("exp05 fixed pool split-role breakdown: %s", role_counts)

    gi_summary = summarize_gi_scores(covered)
    logger.info(
        "strong-SL pairs: %d; gi_score min=%.4f median=%.4f max=%.4f",
        gi_summary["strong_sl_pair_count"],
        gi_summary["gi_score_min"],
        gi_summary["gi_score_median"],
        gi_summary["gi_score_max"],
    )

    pairs_path, genes_path = write_pair_and_gene_artifacts(covered, genes, output_dir)
    manifest = build_manifest(
        pairs_csv=pairs_csv,
        pool_csv=pool_csv,
        esm2_npz=esm2_npz,
        pairs_path=pairs_path,
        genes_path=genes_path,
        n_genes=len(genes),
        n_pairs=len(covered),
        found=found,
        missing=missing,
        role_counts=role_counts,
        gi_summary=gi_summary,
        dropped_nonqueryable_genes=dropped_nonqueryable_genes,
        dropped_pair_count=dropped_pair_count,
        complete_clique_verified=True,
        expected_complete_clique_pair_count=expected_complete_clique_pair_count,
    )
    manifest_path = output_dir / "candidate_universe_manifest.json"
    with manifest_path.open("w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")

    logger.info(
        "FROZEN Bridge A candidate universe: %d genes / %d pairs "
        "(%d strong-SL); %d dropped-nonqueryable genes / %d dropped pairs; "
        "wrote %s",
        len(genes),
        len(covered),
        gi_summary["strong_sl_pair_count"],
        len(dropped_nonqueryable_genes),
        dropped_pair_count,
        output_dir,
    )
    return manifest


def main() -> None:
    """CLI entry point: lock and freeze the Bridge A candidate universe."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs-csv", type=Path, default=DEFAULT_PAIRS_CSV)
    parser.add_argument("--pool-csv", type=Path, default=DEFAULT_POOL_CSV)
    parser.add_argument("--esm2-npz", type=Path, default=DEFAULT_ESM2_NPZ)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--expected-pair-count", type=int, default=EXPECTED_PAIR_COUNT)
    parser.add_argument("--expected-gene-count", type=int, default=EXPECTED_GENE_COUNT)
    parser.add_argument(
        "--drop-nonqueryable",
        action="store_true",
        default=False,
        help=(
            "Drop universe genes (and every pair touching them) missing "
            "from the ESM2 perturbation vocabulary instead of raising. "
            "Default: raise before writing any artifact."
        ),
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    lock_bridge_a_universe(
        args.pairs_csv,
        args.pool_csv,
        args.esm2_npz,
        args.output_dir,
        args.expected_pair_count,
        args.expected_gene_count,
        drop_nonqueryable=args.drop_nonqueryable,
    )


if __name__ == "__main__":
    main()
