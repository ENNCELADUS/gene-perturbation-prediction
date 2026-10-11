"""Re-count the extra single-cell libraries whose deposits hold no raw counts.

A source of the spec (``configs/data/extra_single_cell_sources_26Q1.json``)
with a ``recount`` entry names, per library, its SRA runs (``<run>.sra`` among
the source's files), its kb technology and which ``fasterq-dump`` reads are the
barcode read and the cDNA read. ``index`` builds the kallisto index and
transcript-to-gene map from the spec's reference genome and GTF (kb's standard
workflow) once. ``count`` dumps each library's runs (technical reads included),
counts them with ``kb count`` against that index and keeps kb's unfiltered gene
matrix in Cell Ranger layout at
``<raw_root>/<source>/kb/<library>/counts_unfiltered/cellranger/``, which the
source builder then reads like a deposit. The dumped FASTQ are deleted; a
library whose ``kb/<library>/done.json`` exists is skipped, and an interrupted
library's partial output is removed before it is counted again.

uv run python -m src.data.prepare.recount_extra_single_cell_sources \
    --spec configs/data/extra_single_cell_sources_26Q1.json --wave 2 index
uv run python -m src.data.prepare.recount_extra_single_cell_sources \
    --spec configs/data/extra_single_cell_sources_26Q1.json --wave 2 count \
    [--sources gse142750_recount] [--kb kb] [--fasterq-dump fasterq-dump]
"""  # noqa: E501

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

#: Keys a library's re-count takes (all required).
RECOUNT = ("technology", "runs", "reads")
#: Keys of the spec's ``recount`` block.
#: ``files`` maps the genome and GTF file names to their URLs.
REFERENCE = ("index_dir", "files", "genome", "gtf", "threads", "memory")


def _require(spec: Mapping[str, Any], keys: Sequence[str], where: str) -> None:
    missing = sorted(set(keys) - set(spec))
    if missing:
        raise ValueError(f"{where}: missing keys {missing}")


def _run(command: Sequence[str]) -> str:
    print("+", " ".join(command), flush=True)
    done = subprocess.run(command, check=True, capture_output=True, text=True)
    return done.stdout + done.stderr


def _versions(kb: str) -> list[str]:
    """kb's, kallisto's and bustools' versions, from ``kb info`` (which exits 1)."""
    info = subprocess.run([kb, "info"], capture_output=True, text=True)
    return [
        line
        for line in (info.stdout + info.stderr).splitlines()
        if line.startswith(("kb_python ", "kallisto: ", "bustools: "))
    ]


def index_paths(recount: Mapping[str, Any]) -> dict[str, Path]:
    directory = Path(recount["index_dir"])
    return {
        "index": directory / "kallisto_standard.idx",
        "t2g": directory / "kallisto_standard.t2g.txt",
        "cdna": directory / "kallisto_standard.cdna.fa",
        "done": directory / "kallisto_standard.json",
    }


def build_index(spec: Mapping[str, Any], kb: str) -> Path:
    """kb's standard index of the reference genome and GTF; returns its record."""
    recount = spec["recount"]
    _require(recount, REFERENCE, "recount")
    paths = index_paths(recount)
    if paths["done"].exists():
        return paths["done"]
    directory = Path(recount["index_dir"])
    genome, gtf = directory / recount["genome"], directory / recount["gtf"]
    _run(
        [
            kb, "ref", "--overwrite", "-t", str(recount["threads"]),
            "--tmp", str(directory / "kb_tmp"),
            "-i", str(paths["index"]), "-g", str(paths["t2g"]),
            "-f1", str(paths["cdna"]), str(genome), str(gtf),
        ]
    )  # fmt: skip
    record = {
        "workflow": "standard",
        "genome": recount["genome"],
        "gtf": recount["gtf"],
        "versions": _versions(kb),
    }
    paths["done"].write_text(json.dumps(record, indent=2) + "\n")
    return paths["done"]


def count_library(
    source_root: Path,
    library: str,
    recount: Mapping[str, Any],
    reference: Mapping[str, Any],
    *,
    kb: str,
    fasterq_dump: str,
) -> Path:
    """Count one library's runs; returns ``kb/<library>/done.json``."""
    _require(recount, RECOUNT, f"{source_root.name} recount {library}")
    out = source_root / "kb" / library
    done = out / "done.json"
    if done.exists():
        return done
    paths = index_paths(reference)
    if not paths["done"].exists():
        raise FileNotFoundError(f"{paths['done']}: build the index first")
    work = source_root / "kb" / f".{library}.tmp"
    for stale in (out, work):
        if stale.exists():
            shutil.rmtree(stale)
    work.mkdir(parents=True)
    threads = str(reference["threads"])
    barcode, cdna = recount["reads"]
    fastqs = []
    for run in recount["runs"]:
        sra = source_root / f"{run}.sra"
        if not sra.exists():
            raise FileNotFoundError(f"{sra}: not downloaded")
        _run(
            [
                fasterq_dump, "--split-files", "--include-technical",
                "-e", threads, "-t", str(work / "fqd"), "-O", str(work), str(sra),
            ]
        )  # fmt: skip
        fastqs += [str(work / f"{run}_{read}.fastq") for read in (barcode, cdna)]
    log = _run(
        [
            kb, "count", "-i", str(paths["index"]), "-g", str(paths["t2g"]),
            "-x", recount["technology"], "-t", threads, "-m", reference["memory"],
            "--tmp", str(work / "kb_tmp"), "--cellranger", "-o", str(work / "out"),
            *fastqs,
        ]
    )  # fmt: skip
    (work / "out" / "kb_count.log").write_text(log)
    for fastq in fastqs:
        Path(fastq).unlink()
    for bus in (work / "out").glob("*.bus"):
        bus.unlink()
    os.replace(work / "out", out)
    shutil.rmtree(work)
    run_info = json.loads((out / "run_info.json").read_text())
    record = {
        "library": library,
        "runs": list(recount["runs"]),
        "technology": recount["technology"],
        "reads": list(recount["reads"]),
        "index": json.loads(paths["done"].read_text()),
        "kallisto": {
            key: run_info[key]
            for key in ("n_processed", "n_pseudoaligned", "p_pseudoaligned")
        },
    }
    done.write_text(json.dumps(record, indent=2) + "\n")
    return done


def count(
    spec: Mapping[str, Any],
    wave: int,
    *,
    kb: str,
    fasterq_dump: str,
    only: Sequence[str] | None = None,
) -> None:
    unknown = sorted(set(only or ()) - set(spec["sources"]))
    if unknown:
        raise ValueError(f"--sources not in the spec: {unknown}")
    for name, source in spec["sources"].items():
        if source["wave"] != wave or "recount" not in source:
            continue
        if only and name not in only:
            continue
        root = Path(spec["raw_root"]) / name
        for library, recount in source["recount"].items():
            print(f"{name}: counting {library}", flush=True)
            count_library(
                root,
                library,
                recount,
                spec["recount"],
                kb=kb,
                fasterq_dump=fasterq_dump,
            )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--wave", type=int, required=True)
    parser.add_argument("--kb", default="kb")
    parser.add_argument("--fasterq-dump", default="fasterq-dump")
    parser.add_argument("--sources", help="comma-separated source keys (default: all)")
    parser.add_argument("step", choices=("index", "count"))
    args = parser.parse_args(argv)
    spec = json.loads(args.spec.read_text())
    if args.step == "index":
        print(build_index(spec, args.kb))
        return 0
    only = None if args.sources is None else args.sources.split(",")
    count(spec, args.wave, kb=args.kb, fasterq_dump=args.fasterq_dump, only=only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
