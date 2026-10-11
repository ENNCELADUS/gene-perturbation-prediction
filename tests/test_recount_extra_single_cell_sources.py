"""Re-count of extra single-cell libraries: the runs' barcode and cDNA reads go to
kb in order, the Cell Ranger-layout matrix is kept, FASTQ are deleted, a counted
library is skipped and an interrupted one is counted again."""

from __future__ import annotations

import json
import stat
import sys
from pathlib import Path

import pytest

from src.data.prepare.recount_extra_single_cell_sources import (
    build_index,
    count,
    index_paths,
)

FAKE_FASTERQ_DUMP = """
import sys
from pathlib import Path
args = sys.argv[1:]
out = Path(args[args.index("-O") + 1])
run = Path(args[-1]).stem
for read in (1, 2, 3):
    (out / f"{run}_{read}.fastq").write_text(f"{run} read {read}\\n")
"""

FAKE_KB = """
import json, sys
from pathlib import Path
args = sys.argv[1:]
log = Path(__file__).with_suffix(".calls")
with log.open("a") as handle:
    handle.write(json.dumps(args) + "\\n")
if args[:1] == ["info"]:  # kb prints its versions, then exits 1
    print("usage: kb\\nkb_python 0.0.test")
    print("kallisto: 0.0 (fake)\\nbustools: 0.0 (fake)")
    sys.exit(1)
elif args[:1] == ["ref"]:
    for flag in ("-i", "-g", "-f1"):
        Path(args[args.index(flag) + 1]).write_text("ref")
elif args[:1] == ["count"]:
    out = Path(args[args.index("-o") + 1])
    matrices = out / "counts_unfiltered" / "cellranger"
    matrices.mkdir(parents=True)
    (matrices / "matrix.mtx.gz").write_text("m")
    (out / "output.bus").write_text("bus")
    fastqs = [a for a in args if a.endswith(".fastq")]
    for fastq in fastqs:
        assert Path(fastq).exists(), fastq
    (out / "run_info.json").write_text(json.dumps(
        {"n_processed": 10, "n_pseudoaligned": 8, "p_pseudoaligned": 80.0}
    ))
"""


def _tool(directory: Path, name: str, body: str) -> str:
    path = directory / name
    path.write_text(f"#!{sys.executable}\n{body}")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return str(path)


def _world(tmp_path: Path) -> dict:
    raw = tmp_path / "raw" / "src"
    raw.mkdir(parents=True)
    for run in ("SRR1", "SRR2", "SRR3"):
        (raw / f"{run}.sra").write_text(run)
    reference = tmp_path / "ref"
    reference.mkdir()
    for name in ("genome.fa.gz", "genes.gtf.gz"):
        (reference / name).write_text(name)
    return {
        "raw_root": str(tmp_path / "raw"),
        "recount": {
            "index_dir": str(reference),
            "files": {},
            "genome": "genome.fa.gz",
            "gtf": "genes.gtf.gz",
            "threads": 2,
            "memory": "1G",
        },
        "sources": {
            "src": {
                "wave": 2,
                "files": {},
                "recount": {
                    "lib_a": {
                        "technology": "10XV3",
                        "runs": ["SRR1", "SRR2"],
                        "reads": [2, 3],
                    },
                    "lib_b": {"technology": "10XV2", "runs": ["SRR3"], "reads": [1, 2]},
                },
            },
            "other": {"wave": 1, "files": {}},
        },
    }


def test_count_passes_each_runs_barcode_and_cdna_reads_and_keeps_matrices(tmp_path):
    spec = _world(tmp_path)
    tools = tmp_path / "tools"
    tools.mkdir()
    kb = _tool(tools, "kb", FAKE_KB)
    dump = _tool(tools, "fasterq-dump", FAKE_FASTERQ_DUMP)
    with pytest.raises(FileNotFoundError, match="index"):
        count(spec, 2, kb=kb, fasterq_dump=dump)
    build_index(spec, kb)
    record = json.loads(index_paths(spec["recount"])["done"].read_text())
    assert record["workflow"] == "standard"
    assert record["versions"] == [
        "kb_python 0.0.test",
        "kallisto: 0.0 (fake)",
        "bustools: 0.0 (fake)",
    ]
    count(spec, 2, kb=kb, fasterq_dump=dump)

    calls = [json.loads(line) for line in (tools / "kb.calls").read_text().splitlines()]
    counts = [call for call in calls if call[0] == "count"]
    assert [c[c.index("-x") + 1] for c in counts] == ["10XV3", "10XV2"]
    assert [Path(a).name for a in counts[0] if a.endswith(".fastq")] == [
        "SRR1_2.fastq",
        "SRR1_3.fastq",
        "SRR2_2.fastq",
        "SRR2_3.fastq",
    ]
    assert [Path(a).name for a in counts[1] if a.endswith(".fastq")] == [
        "SRR3_1.fastq",
        "SRR3_2.fastq",
    ]
    out = tmp_path / "raw" / "src" / "kb"
    assert (
        out / "lib_a" / "counts_unfiltered" / "cellranger" / "matrix.mtx.gz"
    ).exists()
    assert not list(out.rglob("*.fastq")) and not list(out.rglob("*.bus"))
    assert sorted(p.name for p in out.iterdir()) == ["lib_a", "lib_b"]
    done = json.loads((out / "lib_a" / "done.json").read_text())
    assert done["runs"] == ["SRR1", "SRR2"] and done["kallisto"]["n_processed"] == 10

    before = len(calls)
    count(spec, 2, kb=kb, fasterq_dump=dump)
    after = (tools / "kb.calls").read_text().splitlines()
    assert len(after) == before  # both libraries skipped


def test_an_interrupted_library_is_counted_again(tmp_path):
    spec = _world(tmp_path)
    tools = tmp_path / "tools"
    tools.mkdir()
    kb = _tool(tools, "kb", FAKE_KB)
    dump = _tool(tools, "fasterq-dump", FAKE_FASTERQ_DUMP)
    build_index(spec, kb)
    kb_dir = tmp_path / "raw" / "src" / "kb"
    (kb_dir / ".lib_b.tmp").mkdir(parents=True)
    (kb_dir / ".lib_b.tmp" / "SRR3_1.fastq").write_text("stale")
    (kb_dir / "lib_b").mkdir()
    count(spec, 2, kb=kb, fasterq_dump=dump, only=["src"])
    assert (kb_dir / "lib_b" / "done.json").exists()
    assert not (kb_dir / ".lib_b.tmp").exists()
    with pytest.raises(ValueError, match="not in the spec"):
        count(spec, 2, kb=kb, fasterq_dump=dump, only=["nope"])
