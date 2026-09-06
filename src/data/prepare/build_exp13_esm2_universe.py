"""data / prepare / build exp13 esm2 universe."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shlex
import uuid
import numpy as np
import pandas as pd
from src.data.geneeffect import Exp13Split

COVERAGE_THRESHOLDS = {"train": 5, "val": 3, "test": 3}


@dataclass(frozen=True)
class NpzCoverage:
    required_count: int
    resolved_count: int
    missing: tuple[str, ...]
    vector_width: int
    artifact_sha256: str


@dataclass(frozen=True)
class CoverageUniverse:
    symbols: tuple[str, ...]
    dropped: tuple[dict[str, object], ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _unique_symbols(symbols: Sequence[object], label: str) -> tuple[str, ...]:
    if isinstance(symbols, (str, bytes)) or not isinstance(symbols, (list, tuple)):
        raise ValueError(f"{label} must be a list or tuple")
    if any(not isinstance(symbol, str) or not symbol for symbol in symbols):
        raise ValueError(f"{label} must contain nonempty strings")
    if any(symbol != symbol.upper() for symbol in symbols):
        raise ValueError(f"{label} must contain uppercase symbols")
    duplicates = sorted(
        symbol for symbol, count in Counter(symbols).items() if count > 1
    )
    if duplicates:
        raise ValueError(f"{label} contains duplicate symbols: {duplicates[:10]}")
    return tuple(symbols)


def build_coverage_universe(
    labels: pd.DataFrame, split: Exp13Split
) -> CoverageUniverse:
    """Return coverage-qualified scored candidates and an explicit drop report."""
    required = {"model_id", "gene_symbol", "gene_effect"}
    missing_columns = sorted(required - set(labels.columns))
    if missing_columns:
        raise ValueError(f"labels is missing columns: {missing_columns}")
    if labels[["model_id", "gene_symbol"]].duplicated().any():
        raise ValueError("labels contains duplicate (model_id, gene_symbol) rows")
    unknown = sorted(set(labels["model_id"]) - set(split.all_model_ids))
    if unknown:
        raise ValueError(f"labels contains ModelIDs outside the split: {unknown[:10]}")
    if labels["gene_symbol"].isna().any():
        raise ValueError("labels contains missing gene symbols")
    symbols = tuple(sorted(set(labels["gene_symbol"].astype(str))))
    _unique_symbols(symbols, "GeneEffect symbols")
    counts = {
        name: (
            labels.loc[labels["model_id"].isin(set(getattr(split, name)))]
            .groupby("gene_symbol", sort=False)["gene_effect"]
            .count()
        )
        for name in COVERAGE_THRESHOLDS
    }
    qualified: list[str] = []
    dropped: list[dict[str, object]] = []
    for symbol in symbols:
        observed = {name: int(counts[name].get(symbol, 0)) for name in counts}
        reasons = [
            f"{name}_finite_lt{minimum}"
            for name, minimum in COVERAGE_THRESHOLDS.items()
            if observed[name] < minimum
        ]
        if reasons:
            dropped.append(
                {"gene_symbol": symbol, "finite_counts": observed, "reasons": reasons}
            )
        else:
            qualified.append(symbol)
    if not qualified:
        raise ValueError("no genes satisfy the Exp13 coverage thresholds")
    return CoverageUniverse(tuple(qualified), tuple(dropped))


def restrict_coverage_universe_to_copy_prior(
    universe: CoverageUniverse, copy_prior_symbols: Sequence[str]
) -> CoverageUniverse:
    """Apply the global finite-donor gate before freezing the ESM2 universe."""
    eligible = set(_unique_symbols(copy_prior_symbols, "copy-prior symbols"))
    kept = tuple(symbol for symbol in universe.symbols if symbol in eligible)
    dropped = list(universe.dropped)
    dropped.extend(
        {"gene_symbol": symbol, "reasons": ["copy_prior_missing"]}
        for symbol in universe.symbols
        if symbol not in eligible
    )
    if not kept:
        raise ValueError("copy-prior coverage produced an empty scored universe")
    return CoverageUniverse(kept, tuple(dropped))


def coverage_qualified_symbols(
    labels: pd.DataFrame, split: Exp13Split
) -> tuple[str, ...]:
    return build_coverage_universe(labels, split).symbols


def build_embedding_union(
    scored_symbols: Sequence[str], response_symbols: Sequence[str]
) -> tuple[str, ...]:
    """Return the deterministic union consumed by the sorting precompute script."""
    scored = _unique_symbols(scored_symbols, "scored GeneEffect universe")
    response = _unique_symbols(response_symbols, "response vocabulary")
    return tuple(sorted(set(scored) | set(response)))


def inspect_npz_coverage(
    npz_path: Path, required_symbols: Sequence[str]
) -> NpzCoverage:
    required = _unique_symbols(required_symbols, "required universe")
    try:
        with np.load(npz_path, allow_pickle=True) as payload:
            expected_keys = {"symbols", "vectors", "resolved"}
            if set(payload.files) != expected_keys:
                raise ValueError(
                    f"ESM2 NPZ keys {sorted(payload.files)} != {sorted(expected_keys)}"
                )
            symbols = tuple(str(symbol) for symbol in payload["symbols"].tolist())
            vectors = np.asarray(payload["vectors"])
            resolved = np.asarray(payload["resolved"])
    except (OSError, EOFError, KeyError) as exc:
        raise ValueError(f"cannot read ESM2 NPZ {npz_path}: {exc}") from exc
    _unique_symbols(symbols, "ESM2 NPZ")
    if vectors.ndim != 2 or vectors.shape[0] != len(symbols):
        raise ValueError("ESM2 NPZ vectors shape does not match symbols")
    if resolved.dtype != np.dtype(bool) or resolved.shape != (len(symbols),):
        raise ValueError("ESM2 NPZ resolved must be a one-dimensional bool array")
    if not np.issubdtype(vectors.dtype, np.number):
        raise ValueError("ESM2 NPZ vectors must be numeric")
    if resolved.any() and not np.isfinite(vectors[resolved]).all():
        raise ValueError("ESM2 NPZ contains non-finite resolved vectors")
    resolved_symbols = {
        symbol
        for symbol, is_resolved in zip(symbols, resolved, strict=True)
        if is_resolved
    }
    missing = tuple(symbol for symbol in required if symbol not in resolved_symbols)
    return NpzCoverage(
        len(required),
        len(required) - len(missing),
        missing,
        int(vectors.shape[1]),
        _sha256(npz_path),
    )


def require_npz_coverage(
    npz_path: Path,
    required_symbols: Sequence[str],
    *,
    must_resolve_symbols: Sequence[str] | None = None,
    expected_width: int = 1_280,
) -> NpzCoverage:
    report = inspect_npz_coverage(npz_path, required_symbols)
    required = _unique_symbols(required_symbols, "required universe")
    must_resolve = _unique_symbols(
        required if must_resolve_symbols is None else must_resolve_symbols,
        "must-resolve ESM2 universe",
    )
    if not set(must_resolve).issubset(required):
        raise ValueError("must-resolve ESM2 symbols must belong to the NPZ universe")
    missing_required = tuple(
        symbol for symbol in must_resolve if symbol in set(report.missing)
    )
    if missing_required:
        raise ValueError(
            "ESM2 NPZ must-resolve coverage incomplete: "
            f"missing {len(missing_required)}/{len(must_resolve)} symbols; "
            f"examples={list(missing_required[:20])}"
        )
    with np.load(npz_path, allow_pickle=True) as payload:
        observed = tuple(str(symbol) for symbol in payload["symbols"].tolist())
    if observed != required:
        raise ValueError("ESM2 NPZ symbol order/universe does not exactly match target")
    if report.vector_width != expected_width:
        raise ValueError(
            f"ESM2 NPZ vector width {report.vector_width} != {expected_width}"
        )
    with np.load(npz_path, allow_pickle=True) as payload:
        if payload["vectors"].dtype != np.dtype(np.float32):
            raise ValueError("ESM2 NPZ vectors must have dtype float32")
    return report


def build_precompute_command(
    embedding_union_csv: Path,
    esm_out: Path,
    sequence_cache: Path,
    *,
    provenance_out: Path | None = None,
    model: str = "facebook/esm2_t33_650M_UR50D",
    cache_dir: Path | None = None,
    local_files_only: bool = False,
) -> str:
    command = [
        "uv",
        "run",
        "python",
        "-m",
        "src.data.prepare.precompute_esm2_embeddings",
        "--benchmark-csv",
        str(embedding_union_csv),
        "--symbol-column",
        "gene_symbol",
        "--out",
        str(esm_out),
        "--seq-cache",
        str(sequence_cache),
        "--provenance-out",
        str(provenance_out or esm_out.with_suffix(esm_out.suffix + ".provenance.json")),
        "--model",
        model,
    ]
    if cache_dir is not None:
        command.extend(("--cache-dir", str(cache_dir)))
    if local_files_only:
        command.append("--local-files-only")
    return shlex.join(command)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_embedding_union(*, scored_symbols, response_symbols, esm2_path, output_dir):
    """Record dependency/response union and supplied table coverage, without seals."""
    from src.data.embeddings import load_esm2_embeddings

    union = build_embedding_union(tuple(scored_symbols), tuple(response_symbols))
    report = inspect_npz_coverage(Path(esm2_path), union)
    table = load_esm2_embeddings(Path(esm2_path))
    resolved = set(table.vectors_by_symbol)
    panel = tuple(gene for gene in scored_symbols if gene in resolved)
    if not panel:
        raise ValueError("no scored genes resolve in supplied ESM2 table")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(
        output_dir / "embedding_union.csv",
        pd.DataFrame({"gene_symbol": union}).to_csv(index=False),
    )
    _atomic_write_text(
        output_dir / "common_gene_panel.csv",
        pd.DataFrame({"gene_symbol": panel}).to_csv(index=False),
    )
    payload = {
        "embedding_union": list(union),
        "common_gene_panel": list(panel),
        "unresolved_symbols": list(report.missing),
        "esm2_table": str(esm2_path),
        "esm2_order": list(table.vectors_by_symbol),
        "vector_width": table.dim,
    }
    _atomic_write_text(
        output_dir / "embedding_union.json", json.dumps(payload, indent=2) + "\n"
    )
    return payload


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scored-panel", type=Path, required=True)
    parser.add_argument(
        "--response-conditions",
        type=Path,
        required=True,
        help="CSV with perturbation_gene column",
    )
    parser.add_argument("--esm2", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    write_embedding_union(
        scored_symbols=tuple(pd.read_csv(args.scored_panel).gene_symbol),
        response_symbols=tuple(
            sorted(set(pd.read_csv(args.response_conditions).perturbation_gene))
        ),
        esm2_path=args.esm2,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
