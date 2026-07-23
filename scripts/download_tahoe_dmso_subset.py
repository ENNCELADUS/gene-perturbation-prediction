"""Materialize the Tahoe-100M DMSO-control subset from remote Parquet shards."""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import fsspec
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

LOGGER = logging.getLogger(__name__)


def _url(base_url: str, shard: int, shard_count: int) -> str:
    name = f"train-{shard:05d}-of-{shard_count:05d}.parquet"
    return f"{base_url.rstrip('/')}/{name}"


def _read_table(url: str, columns: list[str] | None = None):
    for attempt in range(5):
        try:
            with fsspec.open(url, block_size=8 * 1024 * 1024).open() as source:
                return pq.read_table(
                    source,
                    columns=columns,
                    filters=[("drug", "=", "DMSO_TF")],
                )
        except Exception:
            if attempt == 4:
                raise
            wait_seconds = 2**attempt
            LOGGER.warning(
                "Remote read failed for %s; retrying in %ds",
                url,
                wait_seconds,
                exc_info=True,
            )
            time.sleep(wait_seconds)
    raise AssertionError("unreachable")


def _scan_shard(base_url: str, shard: int, shard_count: int) -> tuple[int, int]:
    table = _read_table(_url(base_url, shard, shard_count), ["drug"])
    return shard, table.num_rows


def _extract_shard(
    base_url: str,
    output_dir: Path,
    shard: int,
    shard_count: int,
    excluded_cell_line_ids: frozenset[str],
) -> tuple[int, int]:
    destination = output_dir / "shards" / f"part-{shard:05d}.parquet"
    if destination.exists():
        return shard, pq.read_metadata(destination).num_rows

    table = _read_table(_url(base_url, shard, shard_count))
    if excluded_cell_line_ids:
        keep = pc.invert(
            pc.is_in(
                table["cell_line_id"],
                value_set=pa.array(sorted(excluded_cell_line_ids)),
            )
        )
        table = table.filter(keep)
    if table.num_rows:
        partial = destination.with_suffix(".parquet.part")
        pq.write_table(table, partial, compression="zstd")
        partial.replace(destination)
    return shard, table.num_rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--base-url",
        default=(
            "https://hf-mirror.com/datasets/tahoebio/Tahoe-100M/"
            "resolve/2dc57900b7981cfcf5e211527169a0b006546a95/data"
        ),
    )
    parser.add_argument("--shard-count", type=int, default=3388)
    parser.add_argument(
        "--source-revision",
        default="2dc57900b7981cfcf5e211527169a0b006546a95",
    )
    parser.add_argument("--scan-workers", type=int, default=16)
    parser.add_argument("--extract-workers", type=int, default=4)
    parser.add_argument("--min-cell-lines", type=int, default=45)
    parser.add_argument(
        "--exclude-cell-line-id",
        action="append",
        default=["CVCL_0027", "CVCL_0367", "CVCL_1098"],
        help="Cellosaurus IDs excluded from the training subset.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "shards").mkdir(exist_ok=True)
    contract = {
        "source": "tahoebio/Tahoe-100M",
        "source_base_url": args.base_url,
        "source_revision": args.source_revision,
        "source_shard_count": args.shard_count,
        "drug_filter": "DMSO_TF",
        "excluded_cell_line_ids": sorted(set(args.exclude_cell_line_id)),
        "minimum_expected_cell_lines": args.min_cell_lines,
    }
    contract_path = args.output_dir / "extraction_contract.json"
    existing_parts = list((args.output_dir / "shards").glob("part-*.parquet"))
    if contract_path.exists():
        existing_contract = json.loads(contract_path.read_text())
        if existing_contract != contract:
            raise RuntimeError(
                "Existing Tahoe extraction contract does not match this run"
            )
    elif existing_parts:
        raise RuntimeError("Existing Tahoe shards have no extraction contract")
    else:
        contract_path.write_text(json.dumps(contract, indent=2) + "\n")
    scan_path = args.output_dir / "dmso_shard_scan.json"

    if scan_path.exists():
        scan = json.loads(scan_path.read_text())
        if set(map(int, scan)) != set(range(args.shard_count)):
            raise RuntimeError("Tahoe shard scan is incomplete")
        hits = {int(key): int(value) for key, value in scan.items() if value}
        LOGGER.info("Loaded scan with %d DMSO-positive shards", len(hits))
    else:
        scan: dict[int, int] = {}
        with ThreadPoolExecutor(max_workers=args.scan_workers) as executor:
            futures = {
                executor.submit(
                    _scan_shard,
                    args.base_url,
                    shard,
                    args.shard_count,
                ): shard
                for shard in range(args.shard_count)
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                shard, rows = future.result()
                scan[shard] = rows
                if completed % 100 == 0 or rows:
                    LOGGER.info(
                        "Scanned %d/%d shards; shard=%d DMSO_rows=%d",
                        completed,
                        args.shard_count,
                        shard,
                        rows,
                    )
        scan_partial = scan_path.with_suffix(".json.part")
        scan_partial.write_text(
            json.dumps(dict(sorted(scan.items())), indent=2) + "\n"
        )
        scan_partial.replace(scan_path)
        hits = {shard: rows for shard, rows in scan.items() if rows}

    excluded = frozenset(args.exclude_cell_line_id)
    counts: Counter[str] = Counter()
    with ThreadPoolExecutor(max_workers=args.extract_workers) as executor:
        futures = {
            executor.submit(
                _extract_shard,
                args.base_url,
                args.output_dir,
                shard,
                args.shard_count,
                excluded,
            ): shard
            for shard in sorted(hits)
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            shard, rows = future.result()
            counts["rows"] += rows
            counts["completed_shards"] += 1
            LOGGER.info(
                "Extracted %d/%d positive shards; shard=%d kept_rows=%d",
                completed,
                len(hits),
                shard,
                rows,
            )

    cell_counts: Counter[str] = Counter()
    for path in sorted((args.output_dir / "shards").glob("*.parquet")):
        column = pq.read_table(path, columns=["cell_line_id"])["cell_line_id"]
        cell_counts.update(column.to_pylist())
    summary = {
        "source": "tahoebio/Tahoe-100M",
        "source_base_url": args.base_url,
        "source_revision": args.source_revision,
        "drug_filter": "DMSO_TF",
        "excluded_cell_line_ids": sorted(excluded),
        "source_shard_count": args.shard_count,
        "dmso_positive_source_shards": len(hits),
        "materialized_rows": sum(cell_counts.values()),
        "cell_line_counts": dict(sorted(cell_counts.items())),
    }
    leaked_ids = excluded.intersection(cell_counts)
    if leaked_ids:
        raise RuntimeError(f"Held-out cell lines survived filtering: {leaked_ids}")
    if summary["materialized_rows"] == 0:
        raise RuntimeError("Tahoe DMSO extraction produced zero rows")
    if len(cell_counts) < args.min_cell_lines:
        raise RuntimeError(
            f"Tahoe DMSO extraction retained only {len(cell_counts)} cell lines; "
            f"expected at least {args.min_cell_lines}"
        )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    LOGGER.info("Complete: %s", summary)


if __name__ == "__main__":
    main()
