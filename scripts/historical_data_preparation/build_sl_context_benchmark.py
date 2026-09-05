"""Build the cell-line-conditioned SL benchmark from the integrated raw table.

The integrated CSV is the only pair-label input. The tracked split contract
supplies context identities and registered-data evidence, never extra labels.
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
DEFAULT_OUTPUT_DIR = Path("data/SL_Benchmark_Formal/derived/context_screen_v2")
BENCHMARK_FILENAME = "sl_context_pairs.csv"
CONTEXT_FILENAME = "context_inventory.csv"
AUDIT_FILENAME = "filter_audit.csv"
STATISTICS_FILENAME = "context_statistics.csv"
MANIFEST_FILENAME = "manifest.json"
SPLIT_MANIFEST_PATH = Path("configs/benchmarks/context_screen_v2_split.json")

ALLOWED_GENE_STATUSES = frozenset({"approved", "updated"})
ATOMIC_CONTEXT_RE = re.compile(r"^[A-Z0-9]+$")
EXCLUDED_CONTEXTS = frozenset(
    {
        "FULLBASAL",
        "MESC",
        # str(NaN).upper() matches the atomic-token pattern, so guard it here
        # rather than only at the tokenizer that happens to check for it.
        "NAN",
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
            "Build the split pair-context benchmark using only "
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


def validate_split_evidence(contract: dict[str, Any]) -> None:
    """Fail closed if the files registering split contexts have drifted."""
    evidence = contract["registration_evidence"]
    paths: dict[str, Path] = {}
    for name, specification in evidence.items():
        path = Path(specification["path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        if _sha256(path) != specification["sha256"]:
            raise ValueError(f"Registration-evidence hash mismatch: {path}")
        paths[name] = path

    contexts = contract["contexts"]
    for context, specification in contexts.items():
        if _normalise_context(specification["canonical_name"]) != context:
            raise ValueError(f"Label context identity mismatch for {context}")
        scope = specification.get("evaluation_scope")
        if scope not in {"sl_and_gene_effect", "sl_only"}:
            raise ValueError(f"Unsupported evaluation scope for {context}")
        if scope == "sl_only" and contract["assignments"].get(context) != "test":
            raise ValueError(f"SL-only context must be assigned to test: {context}")
        if not str(specification.get("screen_cluster", "")).strip():
            raise ValueError(f"Missing screen cluster for {context}")
    expected_model_ids = {
        specification["model_id"] for specification in contexts.values()
    }
    basal_payload = json.loads(paths["basal_registry"].read_text(encoding="utf-8"))
    if basal_payload.get("schema_version") != "sl-context-basal-registry-v1":
        raise ValueError("Unsupported basal registry schema")
    basal = pd.DataFrame(basal_payload["contexts"], dtype=str)
    required_basal_columns = {
        "context",
        "model_id",
        "canonical_name",
        "cellosaurus_id",
        "basal_source",
        "artifact_path",
        "artifact_sha256",
        "artifact_status",
    }
    missing_columns = required_basal_columns - set(basal.columns)
    if missing_columns:
        raise ValueError(f"Basal registry missing columns: {sorted(missing_columns)}")
    if basal["context"].duplicated().any() or basal["model_id"].duplicated().any():
        raise ValueError("Basal registry contains duplicate contexts or ModelIDs")
    registered_contexts = set(basal["context"])
    if registered_contexts != set(contexts):
        raise ValueError("Basal registry contexts must exactly match split contexts")
    basal = basal.set_index("context")
    artifact_hashes: dict[Path, str] = {}
    for context, specification in contexts.items():
        row = basal.loc[context]
        for field in ("model_id", "canonical_name", "cellosaurus_id", "basal_source"):
            if str(row[field]) != specification[field]:
                raise ValueError(f"Basal registry {field} mismatch for {context}")
        artifact_path = Path(str(row["artifact_path"]))
        if not artifact_path.is_file():
            raise FileNotFoundError(artifact_path)
        if artifact_path not in artifact_hashes:
            artifact_hashes[artifact_path] = _sha256(artifact_path)
        actual_hash = artifact_hashes[artifact_path]
        if actual_hash != str(row["artifact_sha256"]):
            raise ValueError(f"Basal artifact hash mismatch for {context}")
        artifact_status = str(row["artifact_status"]).strip()
        if artifact_status not in {"source_registered", "tx1_contract_verified"}:
            raise ValueError(f"Unsupported basal artifact status for {context}")
        if artifact_status == "tx1_contract_verified":
            provenance_path = Path(str(row.get("provenance_path", "")))
            provenance_hash = str(row.get("provenance_sha256", ""))
            if not provenance_path.is_file():
                raise FileNotFoundError(provenance_path)
            if _sha256(provenance_path) != provenance_hash:
                raise ValueError(f"Basal provenance hash mismatch for {context}")

    model_metadata = pd.read_csv(paths["model_metadata"]).set_index("ModelID")
    missing_metadata = expected_model_ids - set(model_metadata.index)
    if missing_metadata:
        raise ValueError(
            f"Split contexts missing model metadata: {sorted(missing_metadata)}"
        )
    for context, specification in contexts.items():
        model_id = specification["model_id"]
        if (
            str(model_metadata.loc[model_id, "StrippedCellLineName"])
            != specification["canonical_name"]
        ):
            raise ValueError(f"DepMap identity mismatch for {context}")

    gene_effect_ids = set(
        pd.read_csv(paths["gene_effect"], usecols=[0]).iloc[:, 0].astype(str)
    )
    required_gene_effect = {
        specification["model_id"]
        for specification in contexts.values()
        if specification["evaluation_scope"] == "sl_and_gene_effect"
    }
    missing_gene_effect = required_gene_effect - gene_effect_ids
    if missing_gene_effect:
        raise ValueError(
            f"Split contexts missing GeneEffect rows: {sorted(missing_gene_effect)}"
        )
    unexpected_gene_effect = {
        specification["model_id"]
        for specification in contexts.values()
        if specification["evaluation_scope"] == "sl_only"
    } & gene_effect_ids
    if unexpected_gene_effect:
        raise ValueError(
            "SL-only contexts unexpectedly have GeneEffect rows: "
            f"{sorted(unexpected_gene_effect)}"
        )


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


def _common_conditions(frame: pd.DataFrame) -> dict[str, pd.Series]:
    """Named row-eligibility conditions, kept separable so they can be audited."""
    return {
        "qc_flag_empty": frame["qc_flag"].isna(),
        "has_human_evidence": frame["has_human_evidence"].eq(True),
        "organisms_human": frame["organisms"].eq("human"),
        "gene_status_approved_or_updated": frame["gene_a_status"].isin(
            ALLOWED_GENE_STATUSES
        )
        & frame["gene_b_status"].isin(ALLOWED_GENE_STATUSES),
        "conflict_zero": frame["conflict"].eq(0),
        "sources_screen_only": frame["sources"].eq("screen"),
        "evidence_experimental_screen_only": frame["evidence_types"].eq(
            "experimental_screen"
        ),
    }


def _positive_conditions(frame: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "label_positive": frame["label"].eq("positive"),
        "tier_experimental": frame["label_tier"].eq("experimental"),
        "is_sl_true": frame["is_sl"].eq(True),
        "ep_true": frame["ep"].eq(True),
        "all_evidence_positive": frame["n_pos"].eq(frame["n_evidence"])
        & frame["n_neg"].eq(0)
        & frame["n_unl"].eq(0),
    }


def _all_true(masks: Iterable[pd.Series], index: pd.Index) -> pd.Series:
    combined = pd.Series(True, index=index)
    for mask in masks:
        combined &= mask
    return combined


def _selection_masks(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    common = _all_true(_common_conditions(frame).values(), frame.index)
    positive = _all_true(_positive_conditions(frame).values(), frame.index)
    negative = (
        frame["label"].eq("negative")
        & frame["label_tier"].eq("experimental_negative")
        & frame["en"].eq(True)
        & frame["n_neg"].eq(frame["n_evidence"])
        & frame["n_pos"].eq(0)
        & frame["n_unl"].eq(0)
    )
    return common & positive, common & negative


def audit_positive_losses(frame: pd.DataFrame) -> pd.DataFrame:
    """Count, per context and per condition, positive-labelled rows the filter drops.

    Each condition is evaluated independently rather than cumulatively, so a
    context that loses all of its positives can be traced to the single rule
    responsible instead of to whichever rule happens to run first.
    """
    labelled = frame.loc[frame["label"].eq("positive")]
    if labelled.empty:
        return pd.DataFrame(columns=["context", "condition", "positives_dropped"])

    tokens = labelled["cell_lines"].map(
        lambda value: [
            normalised
            for token in str(value).split(";")
            if (normalised := _normalise_context(token)) is not None
        ]
    )
    conditions = {**_common_conditions(labelled), **_positive_conditions(labelled)}
    counts: dict[tuple[str, str], int] = {}
    for name, mask in conditions.items():
        failing = tokens.loc[~mask]
        for context_list in failing:
            for context in context_list:
                counts[(context, name)] = counts.get((context, name), 0) + 1
    if not counts:
        return pd.DataFrame(columns=["context", "condition", "positives_dropped"])
    return pd.DataFrame(
        [
            {"context": context, "condition": name, "positives_dropped": value}
            for (context, name), value in counts.items()
        ]
    )


def select_atomic_rows(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Select and explode defensible pair-context rows from one raw chunk."""
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        msg = f"Missing required columns: {sorted(missing)}"
        raise ValueError(msg)

    positive, negative = _selection_masks(frame)
    selected = frame.loc[positive | negative].copy()
    # Chunked reads keep a running index, so this is the global raw-CSV row number.
    # It links contexts exploded from one aggregate row; it cannot link separate
    # rows produced by the same underlying screen, because the source carries no
    # study or evidence identifier.
    selected["source_row_id"] = selected.index.astype("int64")
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
        "source_row_id",
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
            "source_row_id",
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


def context_statistics(benchmark: pd.DataFrame) -> pd.DataFrame:
    """Per-context properties a benchmark user must see before trusting a context.

    Positive-anchor concentration is the load-bearing column: a context whose
    positives all share one gene has a single-gene indicator as its label
    function, which no row count reveals.
    """
    multi_context = _multi_context_rows(benchmark)
    rows: list[dict[str, Any]] = []
    for context, frame in benchmark.groupby("context", sort=True):
        positives = frame.loc[frame["sl_label"].eq(1)]
        genes = pd.concat([positives["gene_a"], positives["gene_b"]], ignore_index=True)
        gene_counts = genes.value_counts()
        n_positive = int(len(positives))
        shared = frame.loc[frame["source_row_id"].isin(multi_context)]
        rows.append(
            {
                "context": context,
                "n_positive": n_positive,
                "n_negative": int((frame["sl_label"] == 0).sum()),
                "n_pairs": int(len(frame)),
                "positive_prior": round(n_positive / len(frame), 6),
                "n_distinct_positive_genes": int(gene_counts.size),
                "top_positive_gene": (
                    str(gene_counts.index[0]) if gene_counts.size else ""
                ),
                "top_positive_gene_share": (
                    round(float(gene_counts.iloc[0]) / n_positive, 6)
                    if n_positive
                    else 0.0
                ),
                "n_rows_sharing_source_row_with_other_context": int(len(shared)),
            }
        )
    return pd.DataFrame(rows)


def _multi_context_rows(benchmark: pd.DataFrame) -> set[int]:
    """Source rows exploded into more than one retained context."""
    spread = benchmark.groupby("source_row_id")["context"].nunique()
    return set(spread[spread > 1].index)


def apply_context_split(
    benchmark: pd.DataFrame, contract: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Assign contexts, then remove complete source rows crossing split sides."""
    if contract.get("schema_version") != "sl-context-screen-v2-split-v4":
        raise ValueError("Unsupported context split schema")

    post_filter_min_class_count = int(contract["post_filter_min_class_count"])
    if post_filter_min_class_count < 1:
        raise ValueError("Post-filter minimum class count must be at least 1")
    context_specs = contract["contexts"]
    pinned_train = set(contract["pinned_train_contexts"])

    configured = set(context_specs)
    missing = configured - set(benchmark["context"])
    if missing:
        raise ValueError(f"Split contexts absent from benchmark: {sorted(missing)}")
    if not pinned_train <= configured:
        raise ValueError("Pinned train contexts must be configured and registered")
    response_anchors = {
        context
        for context, specification in context_specs.items()
        if specification["response_anchor"]
    }
    if response_anchors != pinned_train:
        raise ValueError("Pinned train contexts must equal configured response anchors")

    model_ids = {
        context: str(spec["model_id"]) for context, spec in context_specs.items()
    }
    evaluation_scopes = {
        context: str(spec["evaluation_scope"])
        for context, spec in context_specs.items()
    }
    screen_clusters = {
        context: str(spec["screen_cluster"]) for context, spec in context_specs.items()
    }
    if len(set(model_ids.values())) != len(model_ids):
        raise ValueError("Split context ModelIDs must be unique")
    assignments = contract["assignments"]
    if set(assignments) != configured:
        raise ValueError("Published assignments must exactly match configured contexts")
    if set(assignments.values()) != {"train", "validation", "test"}:
        raise ValueError("Split must contain train, validation, and test contexts")
    if any(assignments[context] != "train" for context in pinned_train):
        raise ValueError("Pinned train contexts must be assigned to train")

    selected = benchmark.loc[benchmark["context"].isin(configured)].copy()
    selected["model_id"] = selected["context"].map(model_ids)
    selected["split"] = selected["context"].map(assignments)
    selected["evaluation_scope"] = selected["context"].map(evaluation_scopes)
    selected["screen_cluster"] = selected["context"].map(screen_clusters)
    split_counts = selected.groupby("source_row_id")["split"].nunique()
    crossing_source_rows = set(split_counts.index[split_counts > 1])
    dropped = selected.loc[selected["source_row_id"].isin(crossing_source_rows)]
    selected = selected.loc[
        ~selected["source_row_id"].isin(crossing_source_rows)
    ].copy()
    if (selected.groupby("source_row_id")["split"].nunique() > 1).any():
        raise AssertionError("A source row remains on more than one split side")

    post_counts = (
        selected.groupby("context")["sl_label"]
        .agg(n_positive="sum", n_pairs="size")
        .reindex(sorted(configured), fill_value=0)
    )
    post_counts["n_negative"] = post_counts["n_pairs"] - post_counts["n_positive"]
    post_ineligible = post_counts.index[
        post_counts["n_positive"].lt(post_filter_min_class_count)
        | post_counts["n_negative"].lt(post_filter_min_class_count)
    ]
    if len(post_ineligible):
        raise ValueError(
            "Row-level leakage removal erased a context or class: "
            f"{sorted(post_ineligible)}"
        )

    columns = list(selected.columns)
    columns.remove("model_id")
    columns.remove("split")
    columns.remove("evaluation_scope")
    columns.remove("screen_cluster")
    context_index = columns.index("context") + 1
    columns[context_index:context_index] = [
        "model_id",
        "split",
        "evaluation_scope",
        "screen_cluster",
    ]
    selected = (
        selected[columns]
        .sort_values(
            ["split", "context", "pair_id", "sl_label"],
            ascending=[True, True, True, False],
        )
        .reset_index(drop=True)
    )
    pair_split_counts = selected.groupby("pair_id")["split"].nunique()
    crossing_pairs = int((pair_split_counts > 1).sum())
    if crossing_pairs:
        raise ValueError(f"Canonical pairs cross split sides: {crossing_pairs}")
    dropped_by_context = {
        context: {
            "positive": int((frame["sl_label"] == 1).sum()),
            "negative": int((frame["sl_label"] == 0).sum()),
            "total": int(len(frame)),
        }
        for context, frame in dropped.groupby("context", sort=True)
    }
    retained_by_context = {
        context: {
            "model_id": model_ids[context],
            "split": assignments[context],
            "evaluation_scope": evaluation_scopes[context],
            "screen_cluster": screen_clusters[context],
            "positive": int(frame["sl_label"].sum()),
            "negative": int((frame["sl_label"] == 0).sum()),
            "total": int(len(frame)),
        }
        for context, frame in selected.groupby("context", sort=True)
    }
    stats = {
        "assignments": assignments,
        "cross_split_source_rows_dropped": int(len(crossing_source_rows)),
        "rows_dropped": int(len(dropped)),
        "rows_dropped_by_context": dropped_by_context,
        "retained_by_context": retained_by_context,
        "retained_rows": int(len(selected)),
        "retained_unique_pairs": int(selected["pair_id"].nunique()),
        "retained_unique_genes": int(
            len(set(selected["gene_a"]) | set(selected["gene_b"]))
        ),
        "source_rows_crossing_splits_after_filter": 0,
        "pairs_crossing_splits_after_filter": 0,
    }
    return selected, stats


def _manifest(
    output_dir: Path,
    min_class_count: int,
    selection_stats: dict[str, int],
    pre_split_stats: dict[str, int],
    split_stats: dict[str, Any],
) -> dict[str, Any]:
    benchmark_path = output_dir / BENCHMARK_FILENAME
    context_path = output_dir / CONTEXT_FILENAME
    return {
        "schema_version": "sl-context-screen-v2",
        "input": {
            "path": str(INPUT_PATH),
            "sha256": _sha256(INPUT_PATH),
            "only_allowed_source": True,
        },
        "outputs": {
            BENCHMARK_FILENAME: _sha256(benchmark_path),
            CONTEXT_FILENAME: _sha256(context_path),
            AUDIT_FILENAME: _sha256(output_dir / AUDIT_FILENAME),
            STATISTICS_FILENAME: _sha256(output_dir / STATISTICS_FILENAME),
        },
        "provenance": {
            "builder": {
                "path": (
                    "scripts/historical_data_preparation/build_sl_context_benchmark.py"
                ),
                "sha256": _sha256(
                    Path("scripts/historical_data_preparation")
                    / "build_sl_context_benchmark.py"
                ),
            },
            "source_row_id": (
                "global raw-CSV row number; links contexts exploded from one "
                "aggregate row. It cannot link separate rows from the same "
                "experimental screen — the source has no study or evidence ID, so "
                "no independence claim may rest on it."
            ),
            "split": (
                "context assignment comes from the tracked split manifest; complete "
                "source rows crossing assignment sides are removed from every side"
            ),
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
            "cv_or_train_validation_test_split": str(SPLIT_MANIFEST_PATH),
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
                "Feng2024 CV1/CV2/CV3 split definitions",
            ],
        },
        "provenance_limit": (
            "The integrated CSV is the sole direct input, but it lacks sufficient "
            "study identifiers to independently audit every upstream dataset lineage."
        ),
        "selection_counts": selection_stats,
        "pre_split_counts": pre_split_stats,
        "split": {
            "manifest_path": str(SPLIT_MANIFEST_PATH),
            "manifest_sha256": _sha256(SPLIT_MANIFEST_PATH),
            **split_stats,
        },
        "final_counts": {
            "benchmark_rows": split_stats["retained_rows"],
            "benchmark_unique_pairs": split_stats["retained_unique_pairs"],
            "benchmark_unique_genes": split_stats["retained_unique_genes"],
            "included_contexts": len(split_stats["retained_by_context"]),
            "benchmark_positive_rows": sum(
                values["positive"]
                for values in split_stats["retained_by_context"].values()
            ),
            "benchmark_negative_rows": sum(
                values["negative"]
                for values in split_stats["retained_by_context"].values()
            ),
        },
    }


def build(output_dir: Path, min_class_count: int, chunksize: int) -> dict[str, Any]:
    """Build and write the benchmark artifacts."""
    if not INPUT_PATH.is_file():
        raise FileNotFoundError(INPUT_PATH)
    if chunksize < 1:
        raise ValueError("chunksize must be at least 1")
    if not SPLIT_MANIFEST_PATH.is_file():
        raise FileNotFoundError(SPLIT_MANIFEST_PATH)
    split_contract = json.loads(SPLIT_MANIFEST_PATH.read_text(encoding="utf-8"))
    label_input = split_contract["label_input"]
    if (
        Path(label_input["path"]) != INPUT_PATH
        or _sha256(INPUT_PATH) != label_input["sha256"]
    ):
        raise ValueError(
            "Canonical split label input does not match the tracked contract"
        )
    if min_class_count != int(split_contract["pre_split_min_class_count"]):
        raise ValueError(
            "Pre-split minimum class count does not match the tracked contract"
        )
    validate_split_evidence(split_contract)

    atomic_frames: list[pd.DataFrame] = []
    audit_frames: list[pd.DataFrame] = []
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
        audit = audit_positive_losses(chunk)
        if not audit.empty:
            audit_frames.append(audit)
    if not atomic_frames:
        raise ValueError("No rows passed raw-table selection")

    benchmark, inventory, pre_split_stats = finalise_benchmark(
        pd.concat(atomic_frames, ignore_index=True), min_class_count
    )
    benchmark, split_stats = apply_context_split(benchmark, split_contract)
    inventory = inventory.rename(
        columns={
            "included_in_pair_classification_table": "passed_pre_split_min_class_gate"
        }
    )
    inventory["included_in_pair_classification_table"] = inventory["context"].isin(
        split_stats["retained_by_context"]
    )
    inventory["model_id"] = inventory["context"].map(
        {
            context: specification["model_id"]
            for context, specification in split_contract["contexts"].items()
        }
    )
    inventory["split"] = inventory["context"].map(split_contract["assignments"])
    audit = (
        pd.concat(audit_frames, ignore_index=True)
        .groupby(["context", "condition"], as_index=False)["positives_dropped"]
        .sum()
        .sort_values(
            ["positives_dropped", "context", "condition"], ascending=[False, True, True]
        )
        if audit_frames
        else pd.DataFrame(columns=["context", "condition", "positives_dropped"])
    )
    statistics = context_statistics(benchmark)
    statistics.insert(
        1,
        "model_id",
        statistics["context"].map(
            {
                context: spec["model_id"]
                for context, spec in split_contract["contexts"].items()
            }
        ),
    )
    statistics.insert(
        2, "split", statistics["context"].map(split_contract["assignments"])
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    benchmark.to_csv(output_dir / BENCHMARK_FILENAME, index=False)
    inventory.to_csv(output_dir / CONTEXT_FILENAME, index=False)
    audit.to_csv(output_dir / AUDIT_FILENAME, index=False)
    statistics.to_csv(output_dir / STATISTICS_FILENAME, index=False)
    manifest = _manifest(
        output_dir,
        min_class_count,
        _sum_stats(chunk_stats),
        pre_split_stats,
        split_stats,
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
