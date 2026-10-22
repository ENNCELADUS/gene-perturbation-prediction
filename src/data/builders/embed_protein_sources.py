"""Protein gene sources for the stage-one gene encoder (ladder design §4.4): ESM C 600M
and ESM3-open over the reviewed human UniProt sequences behind PIE's ``esm2`` source.

Steps run in order, each skipped when its output exists: ``proteins`` in ``.venv-pie``,
``afdb`` anywhere, ``esmc`` and ``esm3`` in an ``esm`` environment, hence the
function-level imports.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import logging
import os
import platform
import tarfile
import time
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.io import replace_with, write_json, write_npz

log = logging.getLogger(__name__)

STEPS = ("proteins", "afdb", "esmc", "esm3")
# PIE's esm2 windowing (pie.sources.embed.esm2).
WINDOW = 1022
STRIDE = 511
ESM_VERSION = "3.2.3"
CHUNK = 500

AFDB_VERSION = "v6"
AFDB_ARCHIVE = f"UP000005640_9606_HUMAN_{AFDB_VERSION}.tar"
AFDB_URL = (
    f"https://ftp.ebi.ac.uk/pub/databases/alphafold/{AFDB_VERSION}/{AFDB_ARCHIVE}"
)
DOWNLOAD_ATTEMPTS = 20
DOWNLOAD_BLOCK = 1 << 24

PROTEINS = "proteins.tsv"
STRUCTURES = "esm3_open_structures.tsv"
MATCHED = "matched"


@dataclass(frozen=True)
class ModelPin:
    """One ESM model: ``esm``'s loader name and the snapshot it must read."""

    name: str  # output stem
    esm_name: str
    repo: str
    revision: str
    data_root_key: str  # esm.utils.constants.esm3.data_root's model key


ESMC = ModelPin(
    name="esmc_600m",
    esm_name="esmc_600m",
    repo="EvolutionaryScale/esmc-600m-2024-12",
    revision="e4d83bc7e10fd55c92e598e545f4a76bf04a6e5c",
    data_root_key="esmc-600",
)
ESM3 = ModelPin(
    name="esm3_open",
    esm_name="esm3_sm_open_v1",
    repo="EvolutionaryScale/esm3-sm-open-v1",
    revision="47f0545b2b6daf26a93439a3cd610f4f7f3d5478",
    data_root_key="esm3",
)


def windows(length: int) -> list[tuple[int, int]]:
    """Residue spans a protein is encoded in: one span up to WINDOW residues, else
    windows starting every STRIDE residues (PIE's loop, tail windows included)."""
    if length < 1:
        raise ValueError("empty protein sequence")
    if length <= WINDOW:
        return [(0, length)]
    return [(start, min(start + WINDOW, length)) for start in range(0, length, STRIDE)]


def pool_residues(length: int, states: Callable[[int, int], np.ndarray]) -> np.ndarray:
    """Mean over residues of per-residue states, ``states(start, end)`` giving the
    (end - start, D) states of one span with BOS and EOS already dropped; a residue
    in several windows is first averaged over them."""
    spans = windows(length)
    if len(spans) == 1:
        return np.asarray(states(0, length), dtype=np.float32).mean(axis=0)
    total: np.ndarray | None = None
    count = np.zeros(length, dtype=np.float32)
    for start, end in spans:
        block = np.asarray(states(start, end), dtype=np.float32)
        if block.shape[0] != end - start:
            raise ValueError(
                f"{block.shape[0]} states for a {end - start}-residue span"
            )
        if total is None:
            total = np.zeros((length, block.shape[1]), dtype=np.float32)
        total[start:end] += block
        count[start:end] += 1.0
    return (total / count[:, None]).mean(axis=0)


def residue_states(hidden: np.ndarray, n_residues: int) -> np.ndarray:
    """Positions 1..n of a (1, n + 2, D) final-layer output: BOS (0) and EOS (n + 1)
    dropped, as PIE's esm2 drops them."""
    if hidden.ndim != 3 or hidden.shape[0] != 1 or hidden.shape[1] != n_residues + 2:
        raise ValueError(f"hidden states {hidden.shape} for {n_residues} residues")
    return hidden[0, 1 : n_residues + 1]


def afdb_member(accession: str, fragment: int = 1) -> str:
    """The archive's PDB member of one AFDB model."""
    return f"AF-{accession}-F{fragment}-model_{AFDB_VERSION}.pdb.gz"


def structure_status(
    accession: str,
    sequence: str,
    members: frozenset[str],
    structure_sequence: str | None,
) -> str:
    """Why a protein does or does not get a structure: ``matched`` only when its single
    AFDB model has exactly the UniProt sequence (an archive split into fragments, F2
    onwards, never covers the whole protein)."""
    if afdb_member(accession) not in members:
        return "no_model"
    if afdb_member(accession, 2) in members:
        return "fragmented"
    if structure_sequence is None:
        raise ValueError(f"{accession}: model present but its sequence was not read")
    if structure_sequence.upper() != sequence.upper():
        return "sequence_mismatch"
    return MATCHED


def pending_chunks(n_rows: int, chunk: int, chunk_dir: Path) -> list[int]:
    """Start rows of the chunks without a finished file."""
    return [
        start
        for start in range(0, n_rows, chunk)
        if not (chunk_dir / f"{start:06d}.npz").exists()
    ]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_proteins(out: Path) -> pd.DataFrame:
    table = pd.read_csv(out / PROTEINS, sep="\t", dtype=str, keep_default_na=False)
    if list(table.columns) != ["symbol", "uniprot_accession", "sequence"]:
        raise ValueError(f"{out / PROTEINS}: unexpected columns {list(table.columns)}")
    return table


def proteins(pie_sources: Path, out: Path) -> None:
    """``proteins.tsv``: PIE's UniProt match over the released esm2 source's keys in
    their row order, PIE's claim order, so the rows PIE claimed are claimed again
    (NCBI's ``gene_info``, PIE's other input, is unreachable from the H20 and only
    decides which symbols are tried)."""
    from pie.sources import uniprot
    from pie.sources.common import input_record

    meta = json.loads((pie_sources / "esm2" / "meta.json").read_text())
    keys = [str(key) for key in meta["keys"]]
    stream_path = uniprot.fetch_reviewed_human(out / "uniprot")
    stream = pd.read_csv(stream_path, sep="\t", dtype=str).rename(
        columns=uniprot.TSV_COLUMNS
    )
    matched, unmatched = uniprot.match_symbols(stream, keys)
    # build_proteins_table's filter: invalid sequences, then repeated symbols.
    keep = [
        isinstance(s, str) and uniprot.valid_sequence(s) for s in matched["sequence"]
    ]
    matched = matched.loc[keep].drop_duplicates(subset=["symbol"], keep="first")
    table = pd.DataFrame(
        {
            "symbol": matched["symbol"].tolist(),
            "uniprot_accession": matched["accession"].tolist(),
            "sequence": matched["sequence"].tolist(),
        }
    )
    ours = uniprot.proteins_sha256(table)
    pie_sha = meta["provenance"]["proteins_sha256"]
    missing = sorted(set(keys) - set(table["symbol"]))
    replace_with(out / PROTEINS, lambda t: table.to_csv(t, sep="\t", index=False))
    write_json(
        out / "proteins.json",
        {
            "created": _now(),
            "uniprot_reviewed_human": {
                key: value
                for key, value in input_record(stream_path, uniprot.STREAM_URL).items()
                if key != "sha256"
            },
            "uniprot_query": uniprot.QUERY,
            "pie_esm2_inputs": meta["provenance"]["inputs"],
            "esm2_keys": len(keys),
            "proteins": len(table),
            "esm2_keys_without_protein": missing,
            "unmatched": len(unmatched),
            "row_order_equals_esm2": table["symbol"].tolist() == keys,
            "proteins_sha256": ours,
            "pie_proteins_sha256": pie_sha,
            "proteins_sha256_equals_pie": ours == pie_sha,
        },
        sort_keys=True,
    )
    log.info(
        "proteins: %d of %d esm2 keys; table sha256 %s PIE's",
        len(table),
        len(keys),
        "equals" if ours == pie_sha else "differs from",
    )


def afdb(afdb_dir: Path, out: Path) -> None:
    """The AFDB human-proteome archive, resumed by byte range, and ``afdb.json``."""
    afdb_dir.mkdir(parents=True, exist_ok=True)
    dest = afdb_dir / AFDB_ARCHIVE
    head = urllib.request.Request(AFDB_URL, method="HEAD")
    with urllib.request.urlopen(head, timeout=120) as response:
        total = int(response.headers["Content-Length"])
        headers = {
            k: response.headers[k]
            for k in ("Last-Modified", "ETag")
            if k in response.headers
        }
    if not dest.exists():
        part = dest.with_name(dest.name + ".part")
        attempt = 0
        while not part.exists() or part.stat().st_size < total:
            have = part.stat().st_size if part.exists() else 0
            request = urllib.request.Request(
                AFDB_URL, headers={"Range": f"bytes={have}-"}
            )
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    if response.status != 206:
                        raise OSError(f"range request answered {response.status}")
                    with open(part, "ab") as handle:
                        for block in iter(lambda: response.read(DOWNLOAD_BLOCK), b""):
                            handle.write(block)
                if part.stat().st_size == have:
                    raise OSError("no bytes received")
            except OSError as error:
                attempt += 1
                log.warning("afdb: attempt %d failed: %s", attempt, error)
                if attempt == DOWNLOAD_ATTEMPTS:
                    raise
                time.sleep(min(60, 5 * attempt))
        if part.stat().st_size != total:
            raise OSError(f"{part}: {part.stat().st_size} bytes, expected {total}")
        os.replace(part, dest)
    if dest.stat().st_size != total:
        raise OSError(f"{dest}: {dest.stat().st_size} bytes, the server has {total}")
    write_json(
        out / "afdb.json",
        {
            "created": _now(),
            "url": AFDB_URL,
            "version": AFDB_VERSION,
            "path": str(dest),
            "bytes": total,
            "headers": headers,
        },
        sort_keys=True,
    )


def _pinned_snapshot(pin: ModelPin) -> Path:
    """The snapshot ``esm``'s loaders read; refuses any revision but the pinned one."""
    from esm.utils.constants.esm3 import data_root

    path = Path(data_root(pin.data_root_key))
    if path.name != pin.revision:
        raise ValueError(f"{pin.repo}: snapshot {path.name}, pinned {pin.revision}")
    return path


def _load(pin: ModelPin, device: str) -> Any:
    import torch
    from esm.models.esm3 import ESM3 as Esm3Model
    from esm.models.esmc import ESMC as EsmcModel

    _pinned_snapshot(pin)
    loader = EsmcModel if pin is ESMC else Esm3Model
    return loader.from_pretrained(pin.esm_name, device=torch.device(device)).eval()


class FinalStates:
    """Captures the transformer stack's last LayerNorm output on every forward."""

    def __init__(self, model: Any) -> None:
        self.value: Any = None
        model.transformer.norm.register_forward_hook(self._keep)

    def _keep(self, module: Any, inputs: Any, output: Any) -> None:
        self.value = output

    def take(self) -> np.ndarray:
        value, self.value = self.value, None
        if value is None:
            raise RuntimeError("the final LayerNorm did not run")
        return value.detach().float().cpu().numpy()


def embed_protein(
    model: Any,
    hook: FinalStates,
    sequence: str,
    coordinates: Any | None,
) -> np.ndarray:
    from esm.sdk.api import ESMProtein, LogitsConfig

    def states(start: int, end: int) -> np.ndarray:
        protein = ESMProtein(
            sequence=sequence[start:end],
            coordinates=None if coordinates is None else coordinates[start:end],
        )
        model.logits(model.encode(protein), LogitsConfig(sequence=True))
        return residue_states(hook.take(), end - start)

    return pool_residues(len(sequence), states)


def _archive_index(tar: tarfile.TarFile) -> dict[str, tarfile.TarInfo]:
    """Member name -> header, read once (tarfile's own name lookup is a linear scan)."""
    return {member.name: member for member in tar.getmembers()}


def _read_structure(tar: tarfile.TarFile, member: tarfile.TarInfo) -> Any:
    from esm.utils.structure.protein_chain import ProteinChain

    handle = tar.extractfile(member)
    if handle is None:
        raise ValueError(f"{member.name} is not a file in the archive")
    text = gzip.decompress(handle.read()).decode()
    return ProteinChain.from_pdb(io.StringIO(text), is_predicted=True)


def structures(out: Path, afdb_path: Path) -> pd.DataFrame:
    """``esm3_open_structures.tsv``: every protein's AFDB status (written once)."""
    path = out / STRUCTURES
    if path.exists():
        return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    table = read_proteins(out)
    with tarfile.open(afdb_path, "r:") as tar:
        index = _archive_index(tar)
        members = frozenset(index)
        status = []
        for accession, sequence in zip(table["uniprot_accession"], table["sequence"]):
            chain_sequence = None
            if (
                afdb_member(accession) in members
                and afdb_member(accession, 2) not in members
            ):
                chain_sequence = _read_structure(
                    tar, index[afdb_member(accession)]
                ).sequence
            status.append(
                structure_status(accession, sequence, members, chain_sequence)
            )
    frame = table[["symbol", "uniprot_accession"]].assign(status=status)
    replace_with(path, lambda t: frame.to_csv(t, sep="\t", index=False))
    log.info("structures: %s", frame["status"].value_counts().to_dict())
    return frame


def embed(pin: ModelPin, out: Path, device: str, chunk: int) -> None:
    """``<model>.npz`` and ``<model>.json``; chunks resume, final files are atomic."""
    import torch

    table = read_proteins(out)
    afdb_record: dict[str, Any] | None = None
    if pin is ESM3:
        afdb_record = json.loads((out / "afdb.json").read_text())
        status = structures(out, Path(afdb_record["path"]))
        if status["symbol"].tolist() != table["symbol"].tolist():
            raise ValueError(f"{STRUCTURES} does not follow {PROTEINS}")
        table = table.loc[(status["status"] == MATCHED).to_numpy()].reset_index(
            drop=True
        )
    chunk_dir = out / "chunks" / pin.name
    chunk_dir.mkdir(parents=True, exist_ok=True)
    pending = pending_chunks(len(table), chunk, chunk_dir)
    if pending:
        model = _load(pin, device)
        hook = FinalStates(model)
        tar = tarfile.open(afdb_record["path"], "r:") if afdb_record else None
        index = _archive_index(tar) if tar is not None else {}
        try:
            for start in pending:
                rows = table.iloc[start : start + chunk]
                vectors = []
                for symbol, accession, sequence in rows.itertuples(index=False):
                    coordinates = None
                    if tar is not None:
                        chain = _read_structure(tar, index[afdb_member(accession)])
                        if chain.sequence.upper() != sequence.upper():
                            raise ValueError(f"{symbol}: structure sequence changed")
                        coordinates = torch.tensor(chain.atom37_positions)
                    vectors.append(embed_protein(model, hook, sequence, coordinates))
                block = np.stack(vectors).astype(np.float32)
                if not np.isfinite(block).all():
                    raise ValueError(
                        f"{pin.name}: non-finite embeddings in chunk {start}"
                    )
                write_npz(
                    chunk_dir / f"{start:06d}.npz",
                    symbols=rows["symbol"].to_numpy(dtype=str),
                    vectors=block,
                )
                log.info(
                    "%s: %d of %d proteins", pin.name, start + len(rows), len(table)
                )
        finally:
            if tar is not None:
                tar.close()
    blocks, symbols = [], []
    for start in range(0, len(table), chunk):
        with np.load(chunk_dir / f"{start:06d}.npz") as payload:
            symbols.extend(str(s) for s in payload["symbols"])
            blocks.append(payload["vectors"])
    if symbols != table["symbol"].tolist():
        raise ValueError(f"{pin.name}: chunks do not follow {PROTEINS}")
    vectors = np.concatenate(blocks).astype(np.float32)
    write_npz(
        out / f"{pin.name}.npz",
        symbols=np.asarray(symbols, dtype=str),
        accessions=table["uniprot_accession"].to_numpy(dtype=str),
        vectors=vectors,
    )
    protein_record = json.loads((out / "proteins.json").read_text())
    record: dict[str, Any] = {
        "created": _now(),
        "model": pin.repo,
        "revision": pin.revision,
        "esm": ESM_VERSION,
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else device,
        "dtype": "bfloat16 weights and autocast on cuda (esm's from_pretrained)",
        "layer": "output of transformer.norm, the stack's last LayerNorm",
        "pooling": "mean over residues, BOS and EOS dropped (PIE's esm2 rule)",
        "window": WINDOW,
        "stride": STRIDE,
        "dim": int(vectors.shape[1]),
        "proteins": int(len(read_proteins(out))),
        "embedded": int(len(symbols)),
        "uniprot_reviewed_human": protein_record["uniprot_reviewed_human"],
        "proteins_sha256_equals_pie": protein_record["proteins_sha256_equals_pie"],
        "python": platform.python_version(),
    }
    if afdb_record is not None:
        counts = structures(out, Path(afdb_record["path"]))["status"].value_counts()
        record["structure"] = {
            "afdb": {k: afdb_record[k] for k in ("url", "version", "bytes")},
            "member": f"AF-<accession>-F1-model_{AFDB_VERSION}.pdb.gz",
            "rule": "used only when its sequence equals the UniProt sequence; "
            "otherwise the protein has no entry",
            "status": {str(k): int(v) for k, v in counts.items()},
        }
    write_json(out / f"{pin.name}.json", record, sort_keys=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--steps", required=True, help=f"comma-separated, of {STEPS}")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pie-sources", type=Path, help="PIE_sources dir (proteins)")
    parser.add_argument(
        "--afdb-dir", type=Path, help="where the AFDB archive goes (afdb)"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--chunk", type=int, default=CHUNK)
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    steps = [s for s in args.steps.split(",") if s]
    unknown = sorted(set(steps) - set(STEPS))
    if unknown:
        parser.error(f"unknown steps {unknown}; expected some of {STEPS}")
    args.out.mkdir(parents=True, exist_ok=True)
    for step in STEPS:
        if step not in steps:
            continue
        if step == "proteins" and not (args.out / PROTEINS).exists():
            if args.pie_sources is None:
                parser.error("proteins needs --pie-sources")
            proteins(args.pie_sources, args.out)
        elif step == "afdb" and not (args.out / "afdb.json").exists():
            if args.afdb_dir is None:
                parser.error("afdb needs --afdb-dir")
            afdb(args.afdb_dir, args.out)
        elif step == "esmc" and not (args.out / f"{ESMC.name}.json").exists():
            embed(ESMC, args.out, args.device, args.chunk)
        elif step == "esm3" and not (args.out / f"{ESM3.name}.json").exists():
            embed(ESM3, args.out, args.device, args.chunk)
        else:
            log.info("%s: output exists, skipped", step)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
