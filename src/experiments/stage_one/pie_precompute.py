"""The frozen PIE's deltas for every line x panel gene (PIE stage-one design §3.2,
Task 2), run in PIE's environment (``.venv-pie``).

``python -m src.experiments.stage_one.pie_precompute CONFIG --steps STEPS [--shard I/N]
[--device cuda]`` (``hpc/run.sh pie-precompute``) reads the ladder root that
``src.experiments.stage_one.ladder_inputs`` wrote (``lines.parquet``, ``genes.parquet``,
``mean/pie_input.parquet``) and the Cellosaurus ``context_text`` source. Steps,
each skipping what exists:

- ``rescore``: PIE_replogle_xdataset re-scored on Replogle ``test_seen`` and
  ``test_unseen`` with its DepMap source masked, beside Task 1's unmasked
  numbers (what masking costs PIE on its own task).
- ``pilot``: the gene pass (below) on ``pie.pilot_genes`` random panel genes,
  every line, keeping each pair's full predicted change.
- ``basis``: the ``pie.components`` principal components of the pilot's
  frame-line predicted changes (the profile delta's basis); the pilot report
  ``pilot.md`` (timing, variance across lines within a gene against variance
  between genes per delta, the basis's variance kept, and agreement with PIE's
  own data path). People read it before ``full`` runs.
- ``full``: the gene pass on every panel gene (``--shard I/N`` takes every
  N-th), projecting each predicted change on the basis.
- ``assemble``: every delta centred within its gene on the frame lines' mean
  (the means stored) into gene x line arrays under ``pie/``.

On a states config (``configs/stage_one/states/``, which names the ladder
config), ``--steps states|assemble_states --set NAME[,NAME]|ALL`` (design §3.5,
Task 8; ``src.experiments.states.state_telemetry`` writes the inputs):

- ``states``: the frozen PIE on every (line, state) pseudo-context of the sets
  (``<ModelID>|<set>|<state>``, the line's Cellosaurus text, the state's input
  as control mean, the set's ``|whole`` context among them) for the pilot's
  genes; ``--shard I/N`` takes every N-th line. Per line
  ``states/pie/<set>/lines/<ModelID>.npz`` (with the line's state-input
  fingerprint), skipped when present with the same fingerprint, refused with
  another; an assembled set is skipped.
- ``assemble_states``: per set, genes x contexts arrays centred on the main
  pass's frame means, ``states/pie/<set>/``, with each line's fingerprint.

The gene pass, for gene g and every line c: PIE's inputs are the line's control
mean on the decode panel (``pie_input``), its Cellosaurus text where a record
exists, g's perturbation sources and g's same-perturbation evidence from the
checkpoint's training data (our dataset is not in that evidence, so nothing is
excluded and the same-context blocks are empty). PIE's ``depmap_gene_effect``
token is never served. The reference pass of each line has the same context and
no perturbation source or response evidence. Per pair:

- ``dH``: last decoder-layer gene features pooled over the panel, knockdown
  minus reference;
- ``dE``: the predicted change in mean expression (softmax expectation over the
  delta-p bin centres at PIE's temperature), as basis coordinates; the full
  panel is kept for the selective genes and the pilot;
- ``dL``: the mean-pooled latent array, knockdown minus reference;
- ``own``: g's own decoder feature, its own predicted change and its control
  mean.

It reports and decides nothing.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from src.data.io import write_json, write_npz
from src.experiments.run_dir import REPO_ROOT, absolute, record_step, shard
from src.experiments.stage_one.pie import (
    EXPERIMENT,
    ROW_SETS,
    SHARD_ARRAYS,
    Layout,
    Tables,
    depmap_masked_prediction,
    exclusive,
    layout,
    pie_environment,
    pilot_genes,
    read_tables,
    shard_genes,
    write_dataset,
)
from src.experiments.stage_one.pie_decode import run_genes
from src.experiments.stage_one.pie_reproduction import CKPT
from src.experiments.stage_one.pie_states import run_state_steps

log = logging.getLogger(__name__)

STEPS: Final[tuple[str, ...]] = ("rescore", "pilot", "basis", "full", "assemble")
#: Steps on a states config (``configs/stage_one/states/``), with ``--set``.
STATE_STEPS: Final[tuple[str, ...]] = ("states", "assemble_states")
#: Task 1's run, whose unmasked xdataset scores the masked ones sit beside.
TASK_ONE_RUN: Final[Path] = (
    REPO_ROOT / "outputs/pie_reproduction/pie_reproduction_20261016"
)


# --- paths ------------------------------------------------------------------------


# --- the ladder's tables ------------------------------------------------------------


# --- PIE --------------------------------------------------------------------------


# --- the decode, as PIE runs it, with what does not change hoisted --------------
#
# PieModel._decode_group step for step (third_party/pie/src/pie/model/pie.py,
# trunk.py, evidence.py), with three exact hoists: the decoder's memory keys and
# values once per batch, not per gene chunk; the control-mean encoding once per
# line batch and chunk for the whole run (it depends on the line, not on the
# perturbation); the evidence-block encoders on one row, then expanded (the
# blocks are the perturbation's, the same on every line). Under bf16 the chunk
# runs through torch.compile, which fuses the norms, casts and the delta-p
# expectation; the agreement check holds it to PIE's own forward.


# --- the masked re-score ----------------------------------------------------------


def rescore(out: Layout, device: str) -> dict[str, Any]:
    """PIE's evaluation of the checkpoint on Replogle with DepMap masked, beside
    Task 1's unmasked scores."""
    from pie.config import EvalConfig
    from pie.evaluate import run_eval

    from src.experiments.stage_one.pie_reproduction import (
        SPLITS_URI,
        compare_metrics,
        read_table,
    )

    tables = {}
    with pie_environment(out) as experiment:
        for row_set in ROW_SETS:
            metrics = experiment / "eval" / row_set / f"metrics_{CKPT}.csv"
            granular = experiment / "eval" / row_set / f"granular_{CKPT}.csv"
            if not granular.is_file():
                metrics.unlink(missing_ok=True)
                config = EvalConfig(
                    experiment_name=EXPERIMENT,
                    run_dir=str(experiment),
                    ckpt=CKPT,
                    split_path=f"{SPLITS_URI}/replogle_xdataset/{row_set}.json",
                    row_set=row_set,
                    preprocessed_dirs=None,
                    save_predictions=False,
                    overwrite=False,
                    device=device,
                    batch_size=16,
                )
                with depmap_masked_prediction():
                    run_eval(config)
            unmasked = read_table(
                TASK_ONE_RUN
                / "runs"
                / EXPERIMENT
                / "eval"
                / row_set
                / f"metrics_{CKPT}.csv"
            )
            table = compare_metrics(unmasked, read_table(metrics)).rename(
                columns={"released": "unmasked", "reproduced": "masked"}
            )
            tables[row_set] = table.assign(row_set=row_set)
    joined = pd.concat(tables.values(), ignore_index=True)
    joined.to_csv(out.pilot / "rescore_comparison.csv", index=False)
    overall = joined.loc[joined["context"] == "all"]
    return {
        row_set: {
            row["metric"]: [float(row["unmasked"]), float(row["masked"])]
            for row in overall.loc[overall["row_set"] == row_set].to_dict("records")
        }
        for row_set in ROW_SETS
    }


# --- the basis and the pilot report -----------------------------------------------


def fit_basis(
    changes: np.ndarray, components: int, device: str
) -> dict[str, np.ndarray]:
    """Leading principal components of predicted changes (pairs x panel): mean,
    components (k x panel), and the share of the centred variance each keeps."""
    import torch

    x = torch.from_numpy(np.asarray(changes, dtype=np.float32)).to(device)
    mean = x.mean(dim=0)
    x.sub_(mean)
    total = float(x.pow(2).sum(dtype=torch.float64))
    k = min(components, min(x.shape) - 1)
    _, s, v = torch.svd_lowrank(x, q=k + 10, niter=6)
    s, v = s[:k], v[:, :k]
    return {
        "mean": mean.cpu().numpy(),
        "components": v.T.contiguous().cpu().numpy(),
        "explained": ((s.double() ** 2) / total).cpu().numpy(),
    }


def variance_split(values: np.ndarray) -> dict[str, float]:
    """``values`` genes x lines x width: the mean over genes of the variance
    across lines within a gene, against the variance between the genes' means
    (both summed over the width)."""
    means = values.mean(axis=1)
    within = float(((values - means[:, None]) ** 2).sum(axis=2).mean())
    between = float(((means - means.mean(axis=0)) ** 2).sum(axis=1).mean())
    return {"within_gene": within, "between_genes": between, "ratio": within / between}


def build_basis(out: Layout, tables: Tables, device: str, seed: int) -> dict[str, Any]:
    """The profile delta's basis (``pie/basis.npz``) and ``pilot/pilot.md``."""
    genes = pilot_genes(tables, int(out.config["pie"]["pilot_genes"]), seed)
    missing = [g for g in genes if not (out.genes_dir / f"{g}.npz").is_file()]
    if missing:
        raise FileNotFoundError(f"pilot genes without a shard: {missing[:5]}")
    frame = [tables.lines.index.get_loc(m) for m in tables.frame]
    changes = np.vstack([np.load(out.full_dir / f"{g}.npy")[frame] for g in genes])
    basis = fit_basis(changes, int(out.config["pie"]["components"]), device)
    write_npz(out.pie / "basis.npz", **basis)
    shards = {}
    for gene in genes:
        with np.load(out.genes_dir / f"{gene}.npz") as payload:
            shards[gene] = {k: payload[k] for k in payload.files}
        full = np.load(out.full_dir / f"{gene}.npy")
        shards[gene]["dE"] = (full - basis["mean"]) @ basis["components"].T
    n_lines = len(tables.lines)
    split = {
        name: variance_split(
            np.stack([np.asarray(shards[g][name]).reshape(n_lines, -1) for g in genes])
        )
        for name in ("dH", "dE", "dL", "own_h", "own_dp")
    }
    run = json.loads((out.pie / "run.json").read_text())
    report = {
        "genes": len(genes),
        "selective": len(set(genes) & tables.selective),
        "lines": len(tables.lines),
        "frame_pairs_in_basis": int(changes.shape[0]),
        "components": int(basis["components"].shape[0]),
        "variance_kept": float(basis["explained"].sum()),
        "variance_split": split,
        "agreement": {
            name: max(
                v["agreement"][name]
                for k, v in run["steps"].items()
                if k.startswith("pilot") and "agreement" in v
            )
            for name in ("float32", "pie_bf16", "ours_bf16")
        },
        "pilot_seconds_per_gene": [
            v["seconds_per_gene"]
            for k, v in run["steps"].items()
            if k.startswith("pilot") and "seconds_per_gene" in v
        ],
        "rescore": run["steps"].get("rescore"),
    }
    write_pilot_report(out, report)
    return report


def write_pilot_report(out: Layout, report: Mapping[str, Any]) -> None:
    lines = [
        "# Frozen-PIE pilot",
        "",
        f"{report['genes']} random panel genes ({report['selective']} selective) x "
        f"{report['lines']} lines; DepMap source masked in every pass.",
        "",
        "## Masked re-score (Replogle, `all` row: unmasked -> masked)",
        "",
        "| Row set | Metric | Unmasked (Task 1) | Masked |",
        "| --- | --- | ---: | ---: |",
    ]
    for row_set, metrics in (report["rescore"] or {}).items():
        if row_set == "finished":
            continue
        for metric, (unmasked, masked) in metrics.items():
            lines.append(f"| {row_set} | {metric} | {unmasked:.4f} | {masked:.4f} |")
    lines += [
        "",
        "## Variance across lines within a gene against variance between genes",
        "",
        "| Delta | Within gene | Between genes | Ratio |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, split in report["variance_split"].items():
        lines.append(
            f"| {name} | {split['within_gene']:.4g} | {split['between_genes']:.4g} | "
            f"{split['ratio']:.4g} |"
        )
    seconds = report["pilot_seconds_per_gene"]
    lines += [
        "",
        f"Profile-delta basis: {report['components']} components on "
        f"{report['frame_pairs_in_basis']} frame-line pairs keep "
        f"{report['variance_kept']:.4f} of the centred variance.",
        "",
        f"Seconds per gene (all lines, per process): {seconds}.",
        "",
        "Largest delta-p logit difference from PIE's own dataset, collate and "
        f"float32 forward: our float32 decode {report['agreement']['float32']:.3g}; "
        f"PIE's own bf16 forward {report['agreement']['pie_bf16']:.3g}, our compiled "
        f"bf16 decode {report['agreement']['ours_bf16']:.3g}.",
        "",
    ]
    (out.pilot / "pilot.md").write_text("\n".join(lines))


# --- assembly ---------------------------------------------------------------------


def assemble(out: Layout, tables: Tables) -> dict[str, Any]:
    """Gene x line arrays under ``pie/``, each delta centred within its gene on
    the frame lines' mean (``<name>_frame_mean``); pilot genes, computed before
    the basis, are projected now."""
    basis = dict(np.load(out.pie / "basis.npz"))
    genes = tables.panel
    missing = [g for g in genes if not (out.genes_dir / f"{g}.npz").is_file()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} panel genes without a shard: {missing[:5]}"
        )
    frame = np.asarray([tables.lines.index.get_loc(m) for m in tables.frame])
    n_lines = len(tables.lines)
    arrays: dict[str, np.ndarray] = {}
    means: dict[str, np.ndarray] = {}
    for g, gene in enumerate(genes):
        with np.load(out.genes_dir / f"{gene}.npz") as payload:
            shard = {k: payload[k] for k in payload.files}
        if "dE" not in shard:
            full = np.load(out.full_dir / f"{gene}.npy")
            shard["dE"] = (full - basis["mean"]) @ basis["components"].T
        for name in (*SHARD_ARRAYS, "dE"):
            value = np.asarray(shard[name], dtype=np.float32).reshape(n_lines, -1)
            if name not in arrays:
                width = value.shape[1]
                arrays[name] = np.lib.format.open_memmap(
                    out.pie / f".{name}.npy.tmp",
                    mode="w+",
                    dtype=np.float32,
                    shape=(len(genes), n_lines, width),
                )
                means[name] = np.zeros((len(genes), width), dtype=np.float32)
            centre = value[frame].mean(axis=0)
            arrays[name][g] = value - centre
            means[name][g] = centre
    for name, array in arrays.items():
        array.flush()
        del array
        os.replace(out.pie / f".{name}.npy.tmp", out.pie / f"{name}.npy")
    write_npz(out.pie / "frame_means.npz", **means)
    write_json(
        out.pie / "index.json",
        {
            "genes": genes,
            "lines": list(tables.lines.index),
            "arrays": {
                name: [len(genes), n_lines, int(m.shape[1])]
                for name, m in means.items()
            },
            "centring": "each gene's frame-line mean subtracted; frame_means.npz",
            "full_change": "pie/full/<gene>.npy, lines x panel, for the selective "
            "and pilot genes",
        },
    )
    return {"genes": len(genes), "lines": n_lines}


# --- state pseudo-contexts (design §3.5, Task 8) ------------------------------------
#
# Each (line, state) is a context of its own, named ``<ModelID>|<set>|<state>``
# (``src.data.cell_states``), whose control mean is the state's input and whose
# Cellosaurus text is its owner line's. The main dataset and checkpoint are
# reused. The pass is context-major: a batch of contexts runs its own reference
# pass, then every gene, and its cached context tokens and control encodings
# are evicted after it. The encoder reads no expression, so it runs once per
# distinct owner in a batch (exact). The last batch is padded with copies of its
# last context so the compiled decode sees one batch size.


# --- entry point ------------------------------------------------------------------


def run(
    config: Mapping[str, Any], steps: Sequence[str], shard: tuple[int, int], device: str
) -> None:
    out = layout(config)
    out.pilot.mkdir(parents=True, exist_ok=True)
    if set(steps) - {"rescore"}:
        # The re-score reads PIE's own data alone; every other step our lines.
        tables = read_tables(out.root)
        with exclusive(out):
            write_dataset(out, tables)
    seed = int(config["seed"])
    for step in steps:
        if step == "rescore":
            record_step(out.pie / "run.json", "rescore", rescore(out, device))
        elif step == "pilot":
            genes = shard_genes(
                pilot_genes(tables, int(config["pie"]["pilot_genes"]), seed), shard
            )
            result = run_genes(
                out, tables, genes, device, keep_full=set(genes), basis=None
            )
            record_step(
                out.pie / "run.json", f"pilot_shard_{shard[0]}_of_{shard[1]}", result
            )
        elif step == "basis":
            record_step(
                out.pie / "run.json", "basis", build_basis(out, tables, device, seed)
            )
        elif step == "full":
            basis_path = out.pie / "basis.npz"
            if not basis_path.is_file():
                raise FileNotFoundError(f"{basis_path}: run the pilot and basis first")
            basis = dict(np.load(basis_path))
            genes = shard_genes(tables.panel, shard)
            result = run_genes(
                out, tables, genes, device, keep_full=tables.selective, basis=basis
            )
            record_step(
                out.pie / "run.json", f"full_shard_{shard[0]}_of_{shard[1]}", result
            )
        elif step == "assemble":
            record_step(out.pie / "run.json", "assemble", assemble(out, tables))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--steps", required=True)
    parser.add_argument("--shard", type=shard, default=(0, 1))
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--set",
        dest="sets",
        default=None,
        help="states steps: the sets of state inputs (comma-separated, or ALL)",
    )
    args = parser.parse_args(argv)
    steps = [s for s in args.steps.split(",") if s]
    unknown = [s for s in steps if s not in (*STEPS, *STATE_STEPS)]
    if unknown:
        parser.error(f"unknown steps {unknown}; choose from {STEPS + STATE_STEPS}")
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    if set(steps) & set(STATE_STEPS):
        if set(steps) - set(STATE_STEPS) or args.sets is None:
            parser.error(f"{STATE_STEPS} run alone, with --set, on a states config")
        from src.data.cell_states import state_sets
        from src.experiments.states.config import load_states_config

        states_config = load_states_config(absolute(args.config))
        every = state_sets(states_config)
        sets = every if args.sets == "ALL" else args.sets.split(",")
        if set(sets) - set(every):
            parser.error(f"unknown sets {sorted(set(sets) - set(every))}")
        run_state_steps(states_config, steps, sets, args.shard, args.device)
        return 0
    if args.sets is not None:
        parser.error("--set belongs to the states steps")
    from src.experiments.stage_one.config import load_ladder_config

    run(load_ladder_config(absolute(args.config)), steps, args.shard, args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
