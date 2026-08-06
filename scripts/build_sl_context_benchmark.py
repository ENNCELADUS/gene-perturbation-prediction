"""Build an unsplit cell-line-conditioned SL benchmark from the formal raw table.

The production CLI intentionally has one fixed input.  It does not read the
previously derived high-quality table or any Horlbeck/Feng benchmark artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd


LOGGER = logging.getLogger(__name__)
INPUT_PATH = Path("data/SL_Benchmark_Formal/sl_integrated_pairs.csv")
DEFAULT_OUTPUT_DIR = Path("data/SL_Benchmark_Formal/derived/context_screen_v1")
BENCHMARK_FILENAME = "sl_context_pairs.csv"
CONTEXT_FILENAME = "context_inventory.csv"
MANIFEST_FILENAME = "manifest.json"

ALLOWED_GENE_STATUSES = frozenset({"approved", "updated"})
ATOMIC_CONTEXT_RE = re.compile(r"^[A-Z0-9]+$")
EXCLUDED_CONTEXTS = frozenset(
    {
        "FULLBASAL",
        "MESC",
        "MULTIPLE",
        "PAN-CANCER",
        "PAN_CANCER",
        "UNKNOWN",
        "UNSPECIFIED",
    }
)
REQUIRED_COLUMNS = frozenset(
    {
        "conflict",
        "cell_lines",
        "en",
        "ep",
        "evidence_types",
        "gene_a_status",
        "gene_b_status",
        "has_human_evidence",
        "is_sl",
        "label",
        "label_tier",
        "min_fdr",
        "n_cell_lines",
        "n_evidence",
        "n_neg",
        "n_pos",
        "n_unl",
        "organisms",
        "pair_human_ortholog",
        "qc_flag",
        "sources",
    }
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build an unsplit pair-context benchmark using only "
            "data/SL_Benchmark_Formal/sl_integrated_pairs.csv."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the benchmark, context inventory, and manifest.",
    )
    parser.add_argument(
        "--min-class-count",
        type=int,
        default=10,
        help="Minimum positive and negative pair count required per context.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=100_000,
        help="Number of raw rows read at a time.",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalise_context(value: object) -> str | None:
    context = str(value).strip().upper()
    if (
        not context
        or context.startswith("CTX:")
        or context in EXCLUDED_CONTEXTS
        or ATOMIC_CONTEXT_RE.fullmatch(context) is None
    ):
        return None
    return context


def _selection_masks(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    common = (
        frame["qc_flag"].isna()
        & frame["has_human_evidence"].eq(True)
        & frame["organisms"].eq("human")
        & frame["gene_a_status"].isin(ALLOWED_GENE_STATUSES)
        & frame["gene_b_status"].isin(ALLOWED_GENE_STATUSES)
        & frame["conflict"].eq(0)
        & frame["sources"].eq("screen")
        & frame["evidence_types"].eq("experimental_screen")
    )
    positive = (
        frame["label"].eq("positive")
        & frame["label_tier"].eq("experimental")
        & frame["is_sl"].eq(True)
        & frame["ep"].eq(True)
        & frame["n_pos"].eq(frame["n_evidence"])
        & frame["n_neg"].eq(0)
        & frame["n_unl"].eq(0)
    )
    negative = (
        frame["label"].eq("negative")
        & frame["label_tier"].eq("experimental_negative")
        & frame["en"].eq(True)
        & frame["n_neg"].eq(frame["n_evidence"])
        & frame["n_pos"].eq(0)
        & frame["n_unl"].eq(0)
    )
    return common & positive, common & negative


def select_atomic_rows(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Select and explode defensible pair-context rows from one raw chunk."""
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        msg = f"Missing required columns: {sorted(missing)}"
        raise ValueError(msg)

    positive, negative = _selection_masks(frame)
    selected = frame.loc[positive | negative].copy()
    stats = {
        "raw_rows": int(len(frame)),
        "quality_label_rows": int(len(selected)),
        "count_mismatch_rows": 0,
        "invalid_pair_rows": 0,
        "invalid_context_tokens": 0,
        "atomic_rows": 0,
    }
    if selected.empty:
        return pd.DataFrame(), stats

    selected["context_tokens"] = selected["cell_lines"].map(
        lambda value: [
            token.strip()
            for token in str(value).split(";")
            if token.strip() and str(value) != "nan"
        ]
    )
    expected_count = pd.to_numeric(selected["n_cell_lines"], errors="coerce")
    evidence_count = pd.to_numeric(selected["n_evidence"], errors="coerce")
    count_match = (
        expected_count.eq(evidence_count)
        & expected_count.eq(selected["context_tokens"].map(len))
        & expected_count.gt(0)
    )
    stats["count_mismatch_rows"] = int((~count_match).sum())
    selected = selected.loc[count_match].copy()
    if selected.empty:
        return pd.DataFrame(), stats

    pair_valid = selected["pair_human_ortholog"].astype(str).str.count(r"\|").eq(1)
    pair_parts = (
        selected["pair_human_ortholog"]
        .astype(str)
        .str.split("|", n=1, expand=True, regex=False)
    )
    gene_left = pair_parts[0].str.strip().str.upper()
    gene_right = pair_parts[1].str.strip().str.upper()
    pair_valid &= gene_left.ne("") & gene_right.ne("") & gene_left.ne(gene_right)
    stats["invalid_pair_rows"] = int((~pair_valid).sum())
    selected = selected.loc[pair_valid].copy()
    gene_left = gene_left.loc[pair_valid]
    gene_right = gene_right.loc[pair_valid]
    selected["gene_a"] = gene_left.where(gene_left < gene_right, gene_right)
    selected["gene_b"] = gene_right.where(gene_left < gene_right, gene_left)
    selected["sl_label"] = selected["label"].eq("positive").astype("int8")

    exploded = selected.explode("context_tokens", ignore_index=False)
    exploded["context"] = exploded["context_tokens"].map(_normalise_context)
    stats["invalid_context_tokens"] = int(exploded["context"].isna().sum())
    exploded = exploded.loc[exploded["context"].notna()].copy()
    exploded["pair_id"] = exploded["gene_a"] + "|" + exploded["gene_b"]
    exploded["label_name"] = exploded["sl_label"].map({1: "positive", 0: "negative"})
    exploded["label_semantics"] = exploded["sl_label"].map(
        {1: "experimental_screen_hit", 0: "experimental_screen_non_hit"}
    )
    exploded["pair_is_ordered"] = False
    exploded["label_confidence"] = "silver_inferred"
    exploded["context_assignment"] = "unanimous_row_evidence_count_match"
    exploded["source_n_evidence"] = pd.to_numeric(
        exploded["n_evidence"], errors="raise"
    ).astype(int)
    exploded = exploded.rename(columns={"min_fdr": "source_row_min_fdr"})
    stats["atomic_rows"] = int(len(exploded))
    columns = [
        "pair_id",
        "gene_a",
        "gene_b",
        "context",
        "sl_label",
        "label_name",
        "label_semantics",
        "pair_is_ordered",
        "label_confidence",
        "context_assignment",
        "source_n_evidence",
        "source_row_min_fdr",
    ]
    return exploded[columns].reset_index(drop=True), stats


def _sum_stats(stats: Iterable[dict[str, int]]) -> dict[str, int]:
    total: dict[str, int] = {}
    for chunk_stats in stats:
        for key, value in chunk_stats.items():
            total[key] = total.get(key, 0) + value
    return total


def finalise_benchmark(
    atomic_rows: pd.DataFrame, min_class_count: int
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Resolve duplicates and retain contexts that support binary evaluation."""
    if min_class_count < 1:
        raise ValueError("min_class_count must be at least 1")
    if atomic_rows.empty:
        raise ValueError("No atomic pair-context rows passed preprocessing")

    keys = ["pair_id", "context"]
    label_counts = atomic_rows.groupby(keys)["sl_label"].nunique()
    conflicting_keys = label_counts[label_counts > 1].index
    conflict_index = pd.MultiIndex.from_frame(atomic_rows[keys]).isin(conflicting_keys)
    resolved = atomic_rows.loc[~conflict_index].copy()
    resolved["source_row_count"] = resolved.groupby(keys)["pair_id"].transform("size")
    resolved = resolved.sort_values(
        keys + ["sl_label", "source_row_min_fdr"], na_position="last"
    )
    resolved = resolved.drop_duplicates(keys, keep="first")

    inventory = (
        resolved.groupby(["context", "sl_label"])["pair_id"]
        .nunique()
        .unstack(fill_value=0)
        .rename(columns={0: "n_negative", 1: "n_positive"})
        .reset_index()
    )
    for column in ["n_negative", "n_positive"]:
        if column not in inventory:
            inventory[column] = 0
    inventory["n_pairs"] = inventory["n_positive"] + inventory["n_negative"]
    inventory["minority_class_count"] = inventory[["n_positive", "n_negative"]].min(
        axis=1
    )
    anchor_rows = pd.concat(
        [
            resolved[["context", "gene_a", "sl_label"]].rename(
                columns={"gene_a": "gene"}
            ),
            resolved[["context", "gene_b", "sl_label"]].rename(
                columns={"gene_b": "gene"}
            ),
        ],
        ignore_index=True,
    )
    anchor_labels = anchor_rows.groupby(["context", "gene"])["sl_label"].nunique()
    anchor_summary = (
        anchor_labels.groupby(level="context")
        .agg(
            n_unique_genes="size",
            n_genes_with_both_labels=lambda values: int((values == 2).sum()),
        )
        .reset_index()
    )
    inventory = inventory.merge(anchor_summary, on="context", validate="one_to_one")
    inventory["included_in_pair_classification_table"] = inventory[
        "minority_class_count"
    ].ge(min_class_count)
    inventory = inventory.sort_values(
        ["included_in_pair_classification_table", "n_pairs", "context"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    included_contexts = set(
        inventory.loc[inventory["included_in_pair_classification_table"], "context"]
    )
    benchmark = resolved.loc[resolved["context"].isin(included_contexts)].copy()
    benchmark["source_file"] = INPUT_PATH.name
    benchmark = benchmark.sort_values(
        ["context", "pair_id", "sl_label"], ascending=[True, True, False]
    ).reset_index(drop=True)
    benchmark = benchmark[
        [
            "pair_id",
            "gene_a",
            "gene_b",
            "context",
            "sl_label",
            "label_name",
            "label_semantics",
            "pair_is_ordered",
            "label_confidence",
            "context_assignment",
            "source_n_evidence",
            "source_row_count",
            "source_row_min_fdr",
            "source_file",
        ]
    ]
    pair_context_structure = benchmark.groupby("pair_id").agg(
        n_contexts=("context", "nunique"),
        n_labels=("sl_label", "nunique"),
    )
    stats = {
        "pair_context_label_conflicts_excluded": int(len(conflicting_keys)),
        "deduplicated_atomic_rows": int(len(resolved)),
        "all_atomic_contexts": int(len(inventory)),
        "included_contexts": int(len(included_contexts)),
        "benchmark_rows": int(len(benchmark)),
        "benchmark_unique_pairs": int(benchmark["pair_id"].nunique()),
        "benchmark_unique_genes": int(
            len(set(benchmark["gene_a"]) | set(benchmark["gene_b"]))
        ),
        "benchmark_positive_rows": int(benchmark["sl_label"].sum()),
        "benchmark_negative_rows": int((benchmark["sl_label"] == 0).sum()),
        "benchmark_multi_context_pairs": int(
            (pair_context_structure["n_contexts"] > 1).sum()
        ),
        "benchmark_cross_context_label_change_pairs": int(
            (pair_context_structure["n_labels"] > 1).sum()
        ),
        "benchmark_max_contexts_per_pair": int(
            pair_context_structure["n_contexts"].max()
        ),
    }
    return benchmark, inventory, stats


def _manifest(
    output_dir: Path,
    min_class_count: int,
    selection_stats: dict[str, int],
    final_stats: dict[str, int],
) -> dict[str, Any]:
    benchmark_path = output_dir / BENCHMARK_FILENAME
    context_path = output_dir / CONTEXT_FILENAME
    return {
        "schema_version": "sl-context-screen-v1",
        "input": {
            "path": str(INPUT_PATH),
            "sha256": _sha256(INPUT_PATH),
            "only_allowed_source": True,
        },
        "outputs": {
            BENCHMARK_FILENAME: _sha256(benchmark_path),
            CONTEXT_FILENAME: _sha256(context_path),
        },
        "selection": {
            "human_gene_mapping": (
                "human-only; both endpoints approved_or_updated; pair_human_ortholog"
            ),
            "evidence": "screen and experimental_screen only",
            "labels": {
                "positive": "experimental tier; ep; all evidence positive",
                "negative": "experimental_negative tier; en; all evidence negative",
            },
            "context_assignment": (
                "explode only when n_evidence == n_cell_lines == token_count; "
                "the row label must be unanimous"
            ),
            "atomic_context": "uppercase alphanumeric token; pseudo-contexts excluded",
            "minimum_positive_and_negative_rows_per_context": min_class_count,
            "negative_sampling": "none; retain experimentally screened non-hits",
            "class_balancing": "none",
            "cv_or_train_validation_test_split": "none",
        },
        "feng2024_alignment": {
            "doi": "10.1038/s41467-024-52900-7",
            "adopted": [
                "standardize human gene names",
                "canonicalize undirected pairs",
                "retain explicit binary labels and natural imbalance",
                "treat experimental screen non-hits as stronger negatives",
            ],
            "not_adopted": [
                "direct reads from SynLethDB or Horlbeck/Feng artifacts",
                "random/expression/dependency negative sampling",
                "CV1/CV2/CV3 or train/validation/test splits",
            ],
        },
        "provenance_limit": (
            "The integrated CSV is the sole direct input, but it lacks sufficient "
            "study identifiers to independently audit every upstream dataset lineage."
        ),
        "selection_counts": selection_stats,
        "final_counts": final_stats,
    }


def build(output_dir: Path, min_class_count: int, chunksize: int) -> dict[str, Any]:
    """Build and write the benchmark artifacts."""
    if not INPUT_PATH.is_file():
        raise FileNotFoundError(INPUT_PATH)
    if chunksize < 1:
        raise ValueError("chunksize must be at least 1")

    atomic_frames: list[pd.DataFrame] = []
    chunk_stats: list[dict[str, int]] = []
    for chunk in pd.read_csv(
        INPUT_PATH,
        usecols=sorted(REQUIRED_COLUMNS),
        chunksize=chunksize,
        low_memory=False,
    ):
        atomic, stats = select_atomic_rows(chunk)
        chunk_stats.append(stats)
        if not atomic.empty:
            atomic_frames.append(atomic)
    if not atomic_frames:
        raise ValueError("No rows passed raw-table selection")

    benchmark, inventory, final_stats = finalise_benchmark(
        pd.concat(atomic_frames, ignore_index=True), min_class_count
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    benchmark.to_csv(output_dir / BENCHMARK_FILENAME, index=False)
    inventory.to_csv(output_dir / CONTEXT_FILENAME, index=False)
    manifest = _manifest(
        output_dir,
        min_class_count,
        _sum_stats(chunk_stats),
        final_stats,
    )
    (output_dir / MANIFEST_FILENAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    args = _parse_args()
    manifest = build(args.output_dir, args.min_class_count, args.chunksize)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    LOGGER.info(
        "Final counts: %s", json.dumps(manifest["final_counts"], sort_keys=True)
    )


if __name__ == "__main__":
    main()
