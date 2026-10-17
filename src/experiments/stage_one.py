"""Stage-one runs on the ladder inputs (PIE stage-one design §3.3, §4, Task 3;
protocol §16), in ``.venv-pie``.

``python -m src.experiments.stage_one run CONFIG [--run-id ID] [--fold K]
[--device cuda]`` (``hpc/run.sh stage-one``) trains one config (``kind: train``):
the target is ``GeneEffect - m_g`` over the frame's 981 lines in units of
``sigma_g`` (``--fold K`` drops fold K from training and refits ``m_g`` without
it); ``best.pt`` is chosen on validation selective Spearman; then every line is
predicted once by ``best.pt``: ``predictions.parquet`` (every line x every gene, in
scoring units ``(m_g - mu_g) / sigma_g + z``; the 180 genes outside the decode
panel predict the base), ``V.npy`` (lines x panel genes x 32, fp16: the viability
embedding, ``V.json`` marks the lines trained on as in-sample) and ``row.json``
(validation; test once, in the main run only; the fold's single-cell training lines
out of fold, in a fold run; the training lines in-sample). A run directory
``<output_root>/<task>/<run id>[__fold<k>]`` belongs to one config; a run whose
``row.json`` exists is skipped, and ``last.pt`` resumes an interrupted one.

``queue --gpus 0,1,2,3 [--folds main,0,1,2,3,4] CONFIG...``
(``hpc/run.sh stage-one-queue``) runs every training config and fold, one process
per GPU, skipping configs of another kind and runs whose ``row.json`` exists.

``report CONFIG [--device cuda]`` (``hpc/run.sh stage-one-report``) scores the rows
a report config lists, trained and untrained (the base; the default prior refitted
on the frame, ``src.experiments.stage_one_prior``; a per-gene ridge on the
line-centred decoder-feature delta, solved in its dual on the device), on
validation and test and out of fold over the single-cell training lines, with
the listed contrasts' 95% paired line-bootstrap intervals: ``results.md`` and
``results.json`` in ``<output_root>/<task>/<report config stem>/``.

Nothing is chosen in code: people read the report.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.context_prior.targets import Definitions
from src.data import ladder_reader as reader
from src.data.splits import FixedSplit
from src.experiments.run_dir import (
    assert_bound,
    bind,
    locked,
    read_json,
    write_json,
    write_text,
)

log = logging.getLogger(__name__)

SPLITS = ("val", "test")
ROW = "row.json"
#: The metrics of the report's tables, in order.
REPORTED = (
    ("selective_spearman", "Selective Spearman"),
    ("selective_spearman_scored", "Genes scored"),
    ("within_gene_r2", "Within-gene R²"),
    ("selective_spearman_panel", "Selective Spearman, panel"),
    ("aupr_lift", "AUPR lift"),
    ("residual_pearson", "Residual Pearson"),
    ("absolute_pearson", "Per-line Pearson"),
    ("rmse", "RMSE"),
    ("calibration_slope", "Calibration slope"),
)


# --- setup -------------------------------------------------------------------------


@dataclass(frozen=True)
class Setup:
    """The ladder and the labels and split it was built from."""

    ladder_config: Mapping[str, Any]
    ladder: reader.Ladder
    split: FixedSplit
    bulk_labelled: tuple[str, ...]
    #: Lines outside the split sharing a patient with a held-out line.
    excluded: tuple[str, ...]
    gene_effect: pd.DataFrame
    #: The GeneEffect matrix the labels were read from.
    label_source: Path

    @property
    def definitions(self) -> Definitions:
        """The protocol's definitions (377-line mean, residual SD, gene sets), as
        ``genes.parquet`` records them."""
        genes = self.ladder.genes
        axis = [str(g) for g in genes.index]
        return Definitions(
            genes=tuple(axis),
            gene_means=genes["mu_g"].astype(float),
            selective=tuple(g for g, s in zip(axis, genes["selective"]) if s),
            variable=tuple(g for g, v in zip(axis, genes["variable"]) if v),
            residual_scale=genes["sigma_g"].astype(float),
        )

    def truth(self, lines: Sequence[str]) -> pd.DataFrame:
        return reader.scoring_truth(self.ladder, self.gene_effect, lines)

    def targets(self, fold: int | None) -> reader.Targets:
        return reader.training_targets(
            self.ladder,
            self.gene_effect,
            self.split,
            self.bulk_labelled,
            exclude_fold=fold,
        )


def load_setup(ladder_config_path: str | Path) -> Setup:
    from src.data.depmap import read_depmap_matrix
    from src.data.extra_lines import load_bulk_lines
    from src.data.splits import load_geneeffect_split
    from src.experiments.config import (
        load_config,
        load_ladder_config,
        load_prior_config,
    )

    ladder_config = load_ladder_config(Path(ladder_config_path))
    prior = load_prior_config(Path(ladder_config["prior_config"]))
    joint = load_config(Path(prior["joint_config"]))
    split_path = Path(joint["paths"]["split"])
    split = load_geneeffect_split(split_path)
    bulk = load_bulk_lines(split_path, split)
    return Setup(
        ladder_config=ladder_config,
        ladder=reader.read_ladder(Path(ladder_config["ladder_root"])),
        split=split,
        bulk_labelled=tuple(bulk.labelled),
        excluded=tuple(bulk.excluded),
        gene_effect=read_depmap_matrix(Path(joint["paths"]["gene_effect"])),
        label_source=Path(joint["paths"]["gene_effect"]),
    )


def run_directory(config: Mapping[str, Any], run_id: str, fold: int | None) -> Path:
    name = run_id if fold is None else f"{run_id}__fold{fold}"
    return Path(config["output_root"]) / config["task"] / name


def predictions_frame(
    setup: Setup,
    m_g: pd.Series,
    z: np.ndarray,
    lines: Sequence[str],
    genes: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Scoring units ``(m_g - mu_g) / sigma_g + z`` on every gene; ``z`` covers
    ``genes`` (default the decode panel; genes x lines), every other gene predicts
    the base."""
    frame = reader.base_prediction(setup.ladder, m_g, lines)
    frame.loc[:, list(setup.ladder.panel if genes is None else genes)] += z.T.astype(
        np.float64
    )
    return frame


def run_genes(setup: Setup, config: Mapping[str, Any]) -> tuple[list[str], dict]:
    """The genes a run trains on and predicts (``data.genes``) and their coverage:
    the decode panel, or the selective genes in it that have a full-panel profile
    delta (``pie/full/``), the same set for both matched profile rows."""
    ladder = setup.ladder
    if config["data"]["genes"] == "panel":
        return ladder.panel, {"genes": "panel", "count": len(ladder.panel)}
    selective = [g for g in ladder.panel if bool(ladder.genes.at[g, "selective"])]
    found = reader.full_panel_genes(ladder, selective)
    return found, {
        "genes": "selective_full",
        "count": len(found),
        "selective_in_panel": len(selective),
        "without_full_panel": [g for g in selective if g not in set(found)],
    }


def bind_genes(run_dir: Path, genes: Sequence[str]) -> None:
    """A run keeps the genes it started with: the full-panel set is found from the
    files present, so a resume after files were added or removed would train on
    another set; it is refused instead."""
    path = run_dir / "genes.json"
    with locked(run_dir):
        if path.is_file():
            if read_json(path) != list(genes):
                raise ValueError(
                    f"{run_dir} was started on another gene set ({path}); "
                    "use a new --run-id"
                )
            return
        write_json(path, list(genes))


# --- inputs on the device ------------------------------------------------------------


class Resident:
    """The inputs a training config reads, on the device: gene sources on the
    panel; the delta (bf16 on CUDA, by gene chunk from its memory map) with its
    frame mean and the own-gene slots; the profile delta's states; a line-level
    input; the line tokens and Cellosaurus text; PIE's input as each gene's own
    evidence; the full-panel profile delta streamed by gene chunk (``FullStream``).
    ``batch(genes, lines)`` assembles a ``PairBatch`` from positions in ``genes``
    (the run's genes) and line positions; a permuted control reads, for each gene,
    another line's delta and slots (a fixed permutation of the lines, seeded)."""

    def __init__(
        self,
        setup: Setup,
        config: Mapping[str, Any],
        fold: int | None,
        targets: reader.Targets,
        device: torch.device,
        genes: Sequence[str],
    ) -> None:
        ladder, data = setup.ladder, config["data"]
        arm, kind = config["model"]["arm"], data["input"]
        self.device = device
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        panel = ladder.panel
        n_lines, n_genes = len(ladder.line_ids), len(panel)
        at = {g: i for i, g in enumerate(panel)}
        self.panel_index = torch.tensor([at[g] for g in genes], device=device)
        self.full = None
        self.full_record: dict[str, Any] | None = None
        position = {g: i for i, g in enumerate(str(g) for g in ladder.genes.index)}
        on_panel = [position[g] for g in panel]
        sources = reader.gene_sources(ladder, data["gene_sources"])
        self.gene_vectors = {
            name: torch.from_numpy(vectors[on_panel]).to(device)
            for name, (vectors, _) in sources.items()
        }
        self.gene_present = {
            name: torch.from_numpy(present[on_panel]).to(device)
            for name, (_, present) in sources.items()
        }
        self.source_dims = {name: v.shape[1] for name, (v, _) in sources.items()}
        self.delta = self.delta_mean = self.before = self.shift = None
        self.own = self.own_mean = self.line = self.permutation = None
        self.tokens = self.text = self.evidence = None
        self.width, self.own_width, self.variance_kept = 0, 0, None
        self.line_classes: list[str] | None = None
        slots = list(data["own_slots"]) + (["own_x"] if kind == "own_x" else [])
        if kind in reader.PIE_ARRAYS or kind == "dE_full" or slots:
            index = reader.check_pie_index(ladder)
            means = reader.frame_means(ladder, index)
        if kind in ("dH", "dE", "dL"):
            self.delta = reader.resident(
                reader.open_delta(ladder, kind, index), device, dtype
            )
            self.width = int(self.delta.shape[2])
            self.delta_mean = torch.from_numpy(means[kind]).to(device)
        if kind == "dE_full":
            profile = reader.full_profile(ladder, reader.pie_input(ladder), genes)
            if profile.genes != tuple(genes):
                raise ValueError(
                    f"no full-panel profile delta for {profile.missing[:10]}"
                )
            frame = [ladder.line_ids.index(m) for m in ladder.frame]
            self.full = reader.FullStream(profile, frame, device, dtype)
            self.before = torch.from_numpy(profile.before).to(device)
            self.width = int(profile.before.shape[1])
            self.full_record = {
                "genes": len(profile.genes),
                "frame_mean_pass_seconds": self.full.seconds,
                "frame_mean_pass_bytes": self.full.bytes_read,
            }
            log.info(
                "full-panel profile delta: %d genes, frame-mean pass %.0f s",
                len(profile.genes),
                self.full.seconds,
            )
        if kind == "dE":
            states = reader.profile_states(
                ladder, reader.pie_input(ladder), means, reader.basis(ladder)
            )
            self.before = torch.from_numpy(states.before).to(device)
            self.shift = torch.from_numpy(states.shift).to(device)
            self.variance_kept = states.variance_kept
        if slots:
            widths = [int(index["arrays"][name][2]) for name in slots]
            self.own = torch.empty(
                (n_genes, n_lines, sum(widths)), dtype=dtype, device=device
            )
            start = 0
            for name, width in zip(slots, widths):
                self.own[:, :, start : start + width] = reader.resident(
                    reader.open_delta(ladder, name, index), device, dtype
                )
                start += width
            self.own_mean = torch.from_numpy(
                np.concatenate([means[name] for name in slots], axis=1)
            ).to(device)
            self.own_width = sum(widths)
            if kind == "own_x":
                self.width, self.own_width = self.own_width, 0
        if data["permuted"]:
            generator = torch.Generator().manual_seed(int(config["seed"]))
            self.permutation = torch.stack(
                [torch.randperm(n_lines, generator=generator) for _ in range(n_genes)]
            ).to(device)
        if kind in ("components", "lineage", "text"):
            line = self._line_input(setup, data, kind, fold, targets)
            self.line = torch.tensor(line, device=device)
            self.width = line.shape[1]
        text, text_present = reader.context_text(ladder)
        self.text_dim = text.shape[1]
        if arm != "film":
            tokens = reader.line_tokens(ladder)
            self.tokens = (
                torch.from_numpy(tokens.values).to(device),
                torch.from_numpy(tokens.present).to(device),
            )
            self.token_genes, self.token_present = self._token_sources(
                setup, tokens.axis
            )
            self.text = (
                torch.from_numpy(text).to(device),
                torch.from_numpy(text_present).to(device),
            )
        if arm in ("direct_readout", "hybrid"):
            # Copies: pandas 3 hands out read-only arrays.
            self.evidence = (
                torch.tensor(
                    reader.pie_input(ladder).to_numpy(np.float32).T, device=device
                ),
                torch.tensor(reader.pie_listed(ladder).to_numpy(bool).T, device=device),
            )

    def _line_input(self, setup, data, kind, fold, targets) -> np.ndarray:
        ladder = setup.ladder
        if kind == "components":
            name = "main" if fold is None else f"fold{fold}"
            frame = pd.read_parquet(
                Path(data["prior_run"]) / "components" / f"{name}.parquet"
            )
            if [str(m) for m in frame.index] != ladder.line_ids:
                raise ValueError(
                    f"components/{name}.parquet is not on the ladder's lines"
                )
            return frame.to_numpy(np.float32)
        if kind == "lineage":
            lineage = ladder.lines["lineage"].astype(str)
            self.line_classes = sorted(set(lineage.loc[list(targets.lines)]))
            return np.stack(
                [(lineage == name).to_numpy(np.float32) for name in self.line_classes],
                axis=1,
            )
        text, present = reader.context_text(ladder)
        return np.concatenate([text, present[:, None].astype(np.float32)], axis=1)

    def _token_sources(self, setup, axis):
        """The token axis's gene sources: ``gene_text`` from PIE's source dir (the
        tokens span genes outside the GeneEffect axis); every other source masked."""
        path = Path(setup.ladder_config["paths"]["gene_text"])
        vectors, present = {}, {}
        for name, dim in self.source_dims.items():
            if name == "gene_text":
                values, listed = reader.text_on_axis(path, axis)
            else:
                values = np.zeros((len(axis), dim), np.float32)
                listed = np.zeros(len(axis), bool)
            vectors[name] = torch.from_numpy(values).to(self.device)
            present[name] = torch.from_numpy(listed).to(self.device)
        return vectors, present

    def dims(self):
        from src.model.stage_one import InputDims

        return InputDims(
            gene_sources=self.source_dims,
            delta=self.width,
            own=self.own_width,
            text=self.text_dim,
        )

    def batch(self, genes: torch.Tensor, lines: torch.Tensor):
        from src.model.stage_one import PairBatch

        run_genes, genes = genes, self.panel_index[genes]
        rows = lines.unsqueeze(0).expand(len(genes), -1)
        if self.permutation is not None:
            rows = self.permutation[genes][:, lines]
        pick = (genes[:, None], rows)
        batch = PairBatch(
            gene_vectors={k: v[genes] for k, v in self.gene_vectors.items()},
            gene_present={k: v[genes] for k, v in self.gene_present.items()},
        )
        if self.delta is not None:
            batch.delta = self.delta[pick]
            batch.delta_mean = self.delta_mean[genes]
        if self.full is not None:
            batch.delta = self.full.rows(run_genes.cpu().numpy(), rows.cpu().numpy())
            batch.shift = self.full.means[run_genes]
        if self.before is not None:
            # A permuted control reads the other line's before-state too.
            batch.before = self.before[lines if self.permutation is None else rows]
            if self.full is None:
                batch.shift = self.shift[genes]
        if self.own is not None:
            batch.own = self.own[pick]
            batch.own_mean = self.own_mean[genes]
        if self.line is not None:
            batch.line = self.line[lines]
        if self.tokens is not None:
            batch.token_genes, batch.token_gene_present = (
                self.token_genes,
                self.token_present,
            )
            batch.token_values = self.tokens[0][lines]
            batch.token_present = self.tokens[1][lines]
            batch.text, batch.text_present = self.text[0][lines], self.text[1][lines]
        if self.evidence is not None:
            batch.evidence = self.evidence[0][genes][:, lines]
            batch.evidence_have = self.evidence[1][genes][:, lines]
        return batch


# --- one run -------------------------------------------------------------------------


def make_model(config: Mapping[str, Any], resident: Resident) -> torch.nn.Module:
    from src.model.stage_one import build_model, direct_readouts

    model = build_model(config, resident.dims())
    for direct in direct_readouts(model):
        direct.token_dropout = float(config["train"]["token_dropout"])
        direct.evidence_dropout = float(config["train"]["evidence_dropout"])
    return model.to(resident.device)


def warm_start_trunk(model: torch.nn.Module, path: str | Path) -> list[str]:
    """The direct readout's shared modules from a released PIE checkpoint (at PIE's
    width), by ``trunk_name_map`` and the gene encoder's ``gene_text`` projection
    from PIE's gene query MLP; returns the tensors left fresh."""
    from pie.model.geneeffect import trunk_name_map, warm_start
    from pie.predict import load_checkpoint

    from src.model.stage_one import direct_readouts

    loaded = load_checkpoint(Path(path))
    layers = loaded.config.model.model_dump()
    names = {}
    for prefix, module in model.named_modules():
        if module in direct_readouts(model):
            head = f"{prefix}." if prefix else ""
            names |= trunk_name_map(module.trunk, layers, prefix=f"{head}trunk.")
            names[f"{head}gene_encoder.projections.gene_text"] = "gene_query_mlp"
    return warm_start(model, loaded.state_dict, names)


def run(config_path: Path, run_id: str | None, fold: int | None, device: str) -> Path:
    from src.experiments.config import load_stage_one_config
    from src.training.stage_one_trainer import TrainingData, fit, load_strict, predict

    config = load_stage_one_config(config_path)
    if config["kind"] != "train":
        raise ValueError(f"{config_path} is a {config['kind']} config, not a run")
    run_dir = run_directory(config, run_id or Path(config_path).stem, fold)
    run_dir.mkdir(parents=True, exist_ok=True)
    bind(run_dir, {"run": config, "fold": fold})
    if (run_dir / ROW).is_file():
        log.info("%s: done", run_dir)
        return run_dir
    torch.manual_seed(int(config["seed"]))
    np.random.seed(int(config["seed"]))
    setup = load_setup(config["data"]["ladder_config"])
    ladder = setup.ladder
    targets = setup.targets(fold)
    torch_device = torch.device(device)
    genes, coverage = run_genes(setup, config)
    bind_genes(run_dir, genes)
    resident = Resident(setup, config, fold, targets, torch_device, genes)
    model = make_model(config, resident)
    fresh = None
    trunk = config["model"]["trunk"]
    if trunk is not None and trunk["warm_start"] is not None:
        fresh = warm_start_trunk(model, trunk["warm_start"])
    panel, lines = ladder.panel, ladder.line_ids
    position = {m: i for i, m in enumerate(lines)}
    table = targets.values.loc[:, genes].to_numpy(np.float32)
    labels = torch.full((len(lines), len(genes)), torch.nan, device=torch_device)
    trained = torch.tensor([position[m] for m in targets.lines], device=torch_device)
    labels[trained] = torch.from_numpy(table).to(torch_device)
    data = TrainingData(resident.batch, labels, trained, len(genes))
    definitions = setup.definitions
    val = ladder.role_lines("val")
    val_positions = torch.tensor([position[m] for m in val], device=torch_device)
    val_truth = setup.truth(val)

    def validate(module: torch.nn.Module) -> dict[str, Any]:
        z = predict(module, resident.batch, val_positions, len(genes), config)
        scores = wide_scores(
            predictions_frame(setup, targets.m_g, z, val, genes),
            val_truth,
            definitions,
            panel,
        )
        return {f"val_{key}": value for key, value in scores.items()}

    result = fit(model, data, config, run_dir, setup.split, validate)
    best = make_model(config, resident)
    load_strict(result.best, best, config, setup.split)
    viability = np.lib.format.open_memmap(
        run_dir / ".V.npy.tmp",
        mode="w+",
        dtype=np.float16,
        shape=(len(lines), len(genes), 32),
    )
    z = predict(
        best,
        resident.batch,
        torch.arange(len(lines), device=torch_device),
        len(genes),
        config,
        viability=viability,
    )
    viability.flush()
    del viability
    os.replace(run_dir / ".V.npy.tmp", run_dir / "V.npy")
    in_sample = set(targets.lines)
    write_json(
        run_dir / "V.json",
        {
            "layout": "lines x the run's genes x 32, float16",
            "lines": lines,
            "genes": genes,
            "in_sample": [m in in_sample for m in lines],
        },
    )
    predictions = predictions_frame(setup, targets.m_g, z, lines, genes)
    temporary = run_dir / ".predictions.parquet.tmp"
    predictions.astype(np.float32).to_parquet(temporary)
    os.replace(temporary, run_dir / "predictions.parquet")
    scored = {"val": val}
    if fold is None:
        scored["test"] = ladder.role_lines("test")
    else:
        scored["oof"] = [
            m for m in ladder.role_lines("train_cells") if ladder.folds[m] == fold
        ]
    scored["train_in_sample"] = [
        m for m in ladder.role_lines("train_cells") if m in in_sample
    ]
    scores = {
        name: wide_scores(
            predictions.loc[group], setup.truth(group), definitions, panel
        )
        for name, group in scored.items()
    }
    write_json(
        run_dir / ROW,
        {
            "fold": fold,
            "best_epoch": result.state.best_epoch,
            "best_val_selective_spearman": result.state.best_score,
            "epochs": result.state.next_epoch - 1,
            "updates": result.state.global_step,
            "scores": scores,
            "history": result.history,
            "lines_trained": len(targets.lines),
            "warm_start_fresh": fresh,
            "lineage_classes": resident.line_classes,
            "profile_variance_kept": resident.variance_kept,
            "genes": coverage,
            "full_panel_stream": resident.full_record,
        },
    )
    log.info(
        "%s: validation selective Spearman %s",
        run_dir,
        scores["val"]["selective_spearman"],
    )
    return run_dir


def wide_scores(
    prediction: pd.DataFrame,
    truth: pd.DataFrame,
    definitions: Definitions,
    panel: Sequence[str],
) -> dict[str, Any]:
    from src.eval.geneeffect import wide_accuracy

    return wide_accuracy(prediction, truth, definitions, panel=panel)


# --- the queue ---------------------------------------------------------------------


def queue(configs: Sequence[Path], gpus: Sequence[str], folds: Sequence[str]) -> int:
    """Every training config x fold without a ``row.json``, one process per GPU;
    returns the number of runs that failed."""
    from src.experiments.config import load_stage_one_config

    jobs = []
    for path in configs:
        config = load_stage_one_config(path)
        if config["kind"] != "train":
            print(f"skip {path}: a {config['kind']} config", flush=True)
            continue
        for value in folds:
            fold = None if value == "main" else int(value)
            run_dir = run_directory(config, Path(path).stem, fold)
            if (run_dir / "run_config.json").is_file():
                # Every existing run directory is checked before anything starts.
                assert_bound(run_dir, {"run": config, "fold": fold})
            if (run_dir / ROW).is_file():
                print(f"skip {run_dir}: done", flush=True)
                continue
            jobs.append((Path(path), fold, run_dir))
    free, running, failed = list(gpus), {}, []
    while jobs or running:
        while free and jobs:
            path, fold, run_dir = jobs.pop(0)
            gpu = free.pop(0)
            logs = run_dir.parent / "logs"
            logs.mkdir(parents=True, exist_ok=True)
            argv = [sys.executable, "-m", "src.experiments.stage_one", "run", str(path)]
            if fold is not None:
                argv += ["--fold", str(fold)]
            handle = open(logs / f"{run_dir.name}.log", "a")
            process = subprocess.Popen(
                argv,
                stdout=handle,
                stderr=subprocess.STDOUT,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu},
            )
            running[process] = (gpu, run_dir, handle)
            print(f"start {run_dir.name} on GPU {gpu}", flush=True)
        for process in list(running):
            if process.poll() is None:
                continue
            gpu, run_dir, handle = running.pop(process)
            handle.close()
            free.append(gpu)
            status = (
                "done" if process.returncode == 0 else f"failed ({process.returncode})"
            )
            if process.returncode != 0:
                failed.append(run_dir.name)
            print(f"{status} {run_dir.name}", flush=True)
        time.sleep(5)
    print(f"queue finished; failed: {failed or 'none'}", flush=True)
    return len(failed)


# --- the per-gene ridge on the decoder-feature delta --------------------------------

#: Panel genes solved at a time (a 64-gene block of dH is ~0.2 GB in fp32).
RIDGE_GENES = 64


def dh_ridge(
    setup: Setup, penalties: Sequence[float], device: torch.device
) -> dict[float, pd.DataFrame]:
    """Per penalty, a ridge per panel gene from the line-centred ``dH`` (768-d) to
    the training target, in its dual (``(K + lambda I) alpha = y``, ``K`` the lines'
    Gram matrix), on the device: the main fit on the frame predicts validation and
    test; fold k's fit, without fold k and its base, predicts fold k's frame lines.
    A pair without a label leaves its fit exactly (its row of dH and its target are
    zeroed). Scoring units on every predicted line; other genes predict the base."""
    ladder = setup.ladder
    index = reader.check_pie_index(ladder)
    features = reader.open_delta(ladder, "dH", index)
    panel, lines = ladder.panel, ladder.line_ids
    position = {m: i for i, m in enumerate(lines)}
    folds = ladder.folds
    queries = [(None, ladder.role_lines("val", "test"))] + [
        (k, [m for m in ladder.frame if folds[m] == k])
        for k in sorted(set(folds.values()))
    ]
    fits = []
    for fold, query in queries:
        targets = setup.targets(fold)
        fits.append(
            (
                targets,
                torch.tensor([position[m] for m in targets.lines], device=device),
                torch.tensor([position[m] for m in query], device=device),
                query,
                targets.values.loc[:, panel].to_numpy(np.float32),
            )
        )
    z = {
        p: [np.empty((len(panel), len(f[3])), np.float32) for f in fits]
        for p in penalties
    }
    for start in range(0, len(panel), RIDGE_GENES):
        end = min(start + RIDGE_GENES, len(panel))
        x = torch.from_numpy(np.array(features[start:end], dtype=np.float32)).to(device)
        for number, (_, train, query, _, values) in enumerate(fits):
            target = torch.from_numpy(np.ascontiguousarray(values[:, start:end].T)).to(
                device
            )
            have = torch.isfinite(target)
            xt = x[:, train] * have.unsqueeze(-1)
            gram = xt @ xt.transpose(1, 2)
            cross = x[:, query] @ xt.transpose(1, 2)
            target = torch.nan_to_num(target).unsqueeze(-1)
            eye = torch.eye(len(train), device=device)
            for penalty in penalties:
                factor = torch.linalg.cholesky(gram + penalty * eye)
                alpha = torch.cholesky_solve(target, factor)
                z[penalty][number][start:end] = (
                    (cross @ alpha).squeeze(-1).cpu().numpy()
                )
        if start // RIDGE_GENES % 25 == 0:
            log.info("dH ridge: %d of %d panel genes", end, len(panel))
    return {
        penalty: pd.concat(
            [
                predictions_frame(setup, fit[0].m_g, values, fit[3])
                for fit, values in zip(fits, z[penalty])
            ]
        )
        for penalty in penalties
    }


def ridge_fingerprint(setup: Setup, ladder_config: str | Path) -> dict[str, Any]:
    """What the ridge's predictions depend on: the ladder config, the label file,
    the lines with their roles and folds, the panel and the dH file."""

    def stat(path: Path) -> list:
        info = Path(path).stat()
        return [str(Path(path).resolve()), info.st_size, info.st_mtime_ns]

    ladder = setup.ladder
    return {
        "ladder_config": str(Path(ladder_config).resolve()),
        "ladder": dict(setup.ladder_config),
        "labels": stat(setup.label_source),
        "lines": {
            m: [str(ladder.lines.at[m, "role"]), int(ladder.lines.at[m, "fold"])]
            for m in ladder.line_ids
        },
        "panel": ladder.panel,
        "dH": stat(ladder.root / "pie" / "dH.npy"),
    }


def cached_ridge(
    setup: Setup,
    ladder_config: str | Path,
    penalties: Sequence[float],
    out: Path,
    device: str,
) -> dict[float, pd.DataFrame]:
    """The ridge's predictions per penalty, from ``out/dh_ridge/`` when its
    ``fingerprint.json`` matches the current inputs (refused otherwise), else
    computed and written with the fingerprint."""
    from src.experiments.run_dir import jsonable

    folder = out / "dh_ridge"
    fingerprint = jsonable(ridge_fingerprint(setup, ladder_config))
    record = folder / "fingerprint.json"
    if record.is_file() and read_json(record) != fingerprint:
        raise ValueError(f"{folder} was computed from other inputs; move it aside")
    cached = {p: folder / f"penalty_{p:g}.parquet" for p in penalties}
    if any(not path.is_file() for path in cached.values()):
        folder.mkdir(exist_ok=True)
        write_json(record, fingerprint)
        for penalty, frame in dh_ridge(setup, penalties, torch.device(device)).items():
            frame.astype(np.float32).to_parquet(cached[penalty])
    return {penalty: pd.read_parquet(path) for penalty, path in cached.items()}


def split_scores(
    setup: Setup, split: str, frame: pd.DataFrame, truth: pd.DataFrame
) -> dict[str, Any]:
    """A row's scores on one split; out of fold, the rank metrics are scored within
    each fold and averaged (``fold_accuracy``), since each fold's base differs."""
    from src.eval.geneeffect import fold_accuracy

    ladder = setup.ladder
    if split == "oof":
        return fold_accuracy(
            frame, truth, setup.definitions, ladder.folds, panel=ladder.panel
        )
    return wide_scores(frame, truth, setup.definitions, ladder.panel)


# --- the report -----------------------------------------------------------------------


def _same_ladder(path: str | Path, report_ladder: str | Path, what: str) -> None:
    if Path(path).resolve() != Path(report_ladder).resolve():
        raise ValueError(f"{what} read {path}, the report {report_ladder}")


def finished_run(
    config: Mapping[str, Any], stem: str, fold: int | None, report_ladder: str | Path
) -> Path | None:
    """A finished run's directory (its ``row.json`` exists), checked to be bound to
    this config and fold and to read the report's ladder; None when not finished."""
    run_dir = run_directory(config, stem, fold)
    if not (run_dir / ROW).is_file():
        return None
    assert_bound(run_dir, {"run": config, "fold": fold})
    _same_ladder(config["data"]["ladder_config"], report_ladder, str(run_dir))
    return run_dir


def finished_prior(run: str | Path, report_ladder: str | Path) -> Path | None:
    """The prior reference's run directory once its ``row.json`` exists, checked to
    read the report's ladder."""
    run_dir = Path(run)
    if not (run_dir / "row.json").is_file():
        return None
    bound = read_json(run_dir / "run_config.json")["config"]
    if bound["kind"] != "prior_reference":
        raise ValueError(f"{run_dir} is not a prior reference run")
    _same_ladder(bound["ladder_config"], report_ladder, str(run_dir))
    return run_dir


def _row_predictions(
    setup: Setup,
    row: Mapping[str, Any],
    ridge: Mapping[float, pd.DataFrame],
    report_ladder: str | Path,
) -> dict[str, pd.DataFrame | None]:
    """A report row's predictions (scoring units) on ``val``, ``test`` and, out of
    fold, ``oof`` (the single-cell training lines); None where a run is missing."""
    from src.experiments.config import load_stage_one_config

    ladder = setup.ladder
    val, test = ladder.role_lines("val"), ladder.role_lines("test")
    cells = ladder.role_lines("train_cells")
    folds = sorted(set(ladder.folds.values()))
    by_fold = {k: [m for m in cells if ladder.folds[m] == k] for k in folds}
    kind = row["kind"]
    if kind == "base":
        m_g = ladder.genes["m_g"]
        oof = pd.concat(
            [
                reader.base_prediction(ladder, setup.targets(k).m_g, by_fold[k])
                for k in folds
            ]
        )
        held = reader.base_prediction(ladder, m_g, [*val, *test])
        return {"val": held.loc[val], "test": held.loc[test], "oof": oof.loc[cells]}
    if kind == "dh_ridge":
        frame = ridge[float(row["penalty"])]
        return {"val": frame.loc[val], "test": frame.loc[test], "oof": frame.loc[cells]}
    out: dict[str, pd.DataFrame | None] = {"val": None, "test": None, "oof": None}
    if kind == "prior":
        run_dir = finished_prior(row["run"], report_ladder)
        if run_dir is None:
            return out
        for split, wanted in (("val", val), ("test", test), ("oof", cells)):
            frame = pd.read_parquet(run_dir / f"predictions_{split}.parquet")
            out[split] = frame.loc[wanted]
        return out
    config = load_stage_one_config(Path(row["config"]))
    stem = Path(row["config"]).stem
    main = finished_run(config, stem, None, report_ladder)
    if main is not None:
        frame = pd.read_parquet(main / "predictions.parquet")
        out["val"], out["test"] = frame.loc[val], frame.loc[test]
    parts = []
    for k in folds:
        run_dir = finished_run(config, stem, k, report_ladder)
        if run_dir is None:
            return out
        parts.append(pd.read_parquet(run_dir / "predictions.parquet").loc[by_fold[k]])
    out["oof"] = pd.concat(parts).loc[cells]
    return out


def _contrast(
    setup: Setup,
    left: pd.DataFrame,
    right: pd.DataFrame,
    truth: pd.DataFrame,
    repeats: int,
) -> dict[str, Any]:
    """Selective-Spearman and within-gene R² differences, left minus right, with
    95% paired line-bootstrap intervals (the same resamples for both). The Spearman
    difference is over the selective genes defined on both sides (a row predicting
    the base for a gene, as every row does outside the decode panel, is undefined
    there), so the two macro means cover the same genes; within-gene R² is defined
    for every gene and covers them all."""
    from src.context_prior.noise import line_sums, r2_gain
    from src.eval.geneeffect import BOOTSTRAP_SEED, defined_genes, gain

    definitions = setup.definitions
    selective = list(definitions.selective)
    common = set(defined_genes(left, truth, selective)) & set(
        defined_genes(right, truth, selective)
    )
    genes = [g for g in selective if g in common]
    sums = [
        line_sums(
            frame[selective].to_numpy(np.float64), truth[selective].to_numpy(np.float64)
        )
        for frame in (left, right)
    ]
    spearman = {"difference": None, "interval": [None, None]}
    if genes:
        spearman = gain(
            left.astype(np.float64),
            right.astype(np.float64),
            truth,
            definitions,
            repeats,
            genes=genes,
        )
    return {
        "selective_spearman": spearman | {"genes": len(genes)},
        "within_gene_r2": r2_gain(
            sums[0][0], sums[1][0], sums[0][1], repeats=repeats, seed=BOOTSTRAP_SEED
        ),
    }


def report(config_path: Path, device: str) -> Path:
    from src.experiments.config import load_stage_one_config

    config = load_stage_one_config(config_path)
    if config["kind"] != "report":
        raise ValueError(f"{config_path} is a {config['kind']} config, not a report")
    out = Path(config["output_root"]) / config["task"] / Path(config_path).stem
    out.mkdir(parents=True, exist_ok=True)
    setup = load_setup(config["ladder_config"])
    ladder = setup.ladder
    groups = {
        "val": ladder.role_lines("val"),
        "test": ladder.role_lines("test"),
        "oof": ladder.role_lines("train_cells"),
    }
    truth = {name: setup.truth(lines) for name, lines in groups.items()}
    penalties = sorted(
        {float(r["penalty"]) for r in config["rows"] if r["kind"] == "dh_ridge"}
    )
    ridge = cached_ridge(setup, config["ladder_config"], penalties, out, device)
    predictions = {
        row["name"]: _row_predictions(setup, row, ridge, config["ladder_config"])
        for row in config["rows"]
    }
    scores = {
        name: {
            split: None
            if frame is None
            else split_scores(setup, split, frame, truth[split])
            for split, frame in splits.items()
        }
        for name, splits in predictions.items()
    }
    contrasts = []
    for left, right in config["contrasts"]:
        entry = {"left": left, "right": right}
        for split in SPLITS:
            a, b = predictions[left][split], predictions[right][split]
            entry[split] = (
                None
                if a is None or b is None
                else _contrast(
                    setup, a, b, truth[split], int(config["bootstrap_repeats"])
                )
            )
        contrasts.append(entry)
    training = {}
    for row in config["rows"]:
        if row["kind"] == "run":
            run_config = load_stage_one_config(Path(row["config"]))
            run_dir = finished_run(
                run_config, Path(row["config"]).stem, None, config["ladder_config"]
            )
            training[row["name"]] = (
                None if run_dir is None else read_json(run_dir / ROW)
            )
    payload = {
        "config": config,
        "lines": {name: len(lines) for name, lines in groups.items()},
        "scores": scores,
        "contrasts": contrasts,
        "training": {
            name: None
            if record is None
            else {
                key: record[key]
                for key in ("best_epoch", "epochs", "updates", "lines_trained")
            }
            | {"train_in_sample": record["scores"]["train_in_sample"]}
            for name, record in training.items()
        },
    }
    write_json(out / "results.json", payload)
    write_text(out / "results.md", results_markdown(payload))
    log.info("report: %s", out / "results.md")
    return out


def _number(value: Any, digits: int = 4) -> str:
    """Undefined (None or NaN) is written as such, never as 0."""
    if value is None or not np.isfinite(float(value)):
        return "undefined"
    return f"{float(value):.{digits}f}"


def _interval(entry: Mapping[str, Any]) -> str:
    low, high = entry["interval"]
    return f"{_number(entry['difference'])} [{_number(low)}, {_number(high)}]"


def results_markdown(payload: Mapping[str, Any]) -> str:
    config = payload["config"]
    counts = payload["lines"]
    out = [
        f"# Stage-one report: {Path(config['task'])} ({len(config['rows'])} rows)",
        "",
        "Scoring units: every prediction is GeneEffect re-centred on the protocol's "
        "377-line mean (`(m_g - mu_g) / sigma_g + z`); metrics as protocol §6 and "
        "§14 (within-gene R² pooled over the selective genes; calibration slope: "
        "measured residual on predicted, each gene's offset removed). "
        f"Validation {counts['val']} lines, test {counts['test']}, out of fold "
        f"{counts['oof']} single-cell training lines (each scored by the run without "
        "its fold; each fold's base differs, so the rank metrics are scored within "
        "each fold and averaged, and within-gene R², RMSE and per-line Pearson are "
        "pooled). Selective Spearman covers the 4,323 protocol genes (genes scored: "
        "those defined), the panel column the selective genes in the decode panel; a "
        "contrast's Spearman difference covers the genes defined on both sides. "
        "`undefined`: a prediction constant within every gene, or a run not "
        "finished. Nothing is chosen here; this is not SL evidence.",
    ]
    titles = {"val": "Validation", "test": "Test", "oof": "Out of fold (telemetry)"}
    for split, title in titles.items():
        header = " | ".join(label for _, label in REPORTED)
        out += [
            "",
            f"## {title}",
            "",
            f"| Row | {header} |",
            "|" + " --- |" * (len(REPORTED) + 1),
        ]
        for name, splits in payload["scores"].items():
            entry = splits[split]
            cells = [
                "undefined" if entry is None else _number(entry[key])
                for key, _ in REPORTED
            ]
            out.append(f"| {name} | " + " | ".join(cells) + " |")
    out += [
        "",
        "## Contrasts (left minus right; 95% paired line-bootstrap intervals)",
        "",
        "| Left | Right | Val selective Spearman | Test selective Spearman "
        "| Val within-gene R² | Test within-gene R² |",
        "|" + " --- |" * 6,
    ]
    for entry in payload["contrasts"]:
        cells = [entry["left"], entry["right"]]
        for metric in ("selective_spearman", "within_gene_r2"):
            for split in SPLITS:
                result = entry[split]
                text = "undefined" if result is None else _interval(result[metric])
                if result is not None and metric == "selective_spearman":
                    text += f" ({result[metric]['genes']} genes)"
                cells.append(text)
        out.append("| " + " | ".join(cells) + " |")
    out += [
        "",
        "## Training (selected epoch, and the gap from training to validation)",
        "",
        "| Row | Best epoch | Epochs | Updates | Lines trained "
        "| Train selective Spearman | Val selective Spearman "
        "| Train within-gene R² | Val within-gene R² |",
        "|" + " --- |" * 9,
    ]
    for name, record in payload["training"].items():
        if record is None:
            out.append(f"| {name} |" + " not run |" * 8)
            continue
        train, val = record["train_in_sample"], payload["scores"][name]["val"]
        out.append(
            f"| {name} | {record['best_epoch']} | {record['epochs']} "
            f"| {record['updates']} "
            f"| {record['lines_trained']} | {_number(train['selective_spearman'])} "
            f"| {_number(None if val is None else val['selective_spearman'])} "
            f"| {_number(train['within_gene_r2'])} "
            f"| {_number(None if val is None else val['within_gene_r2'])} |"
        )
    return "\n".join(out) + "\n"


# --- entry point --------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    one = commands.add_parser("run")
    one.add_argument("config", type=Path)
    one.add_argument("--run-id")
    one.add_argument("--fold", type=int)
    one.add_argument("--device", default="cuda")
    many = commands.add_parser("queue")
    many.add_argument("configs", type=Path, nargs="+")
    many.add_argument("--gpus", required=True)
    many.add_argument("--folds", default="main")
    summary = commands.add_parser("report")
    summary.add_argument("config", type=Path)
    summary.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    if args.command == "run":
        run(args.config, args.run_id, args.fold, args.device)
        return 0
    if args.command == "queue":
        folds = [f for f in args.folds.split(",") if f]
        bad = [f for f in folds if f != "main" and not f.isdigit()]
        if bad:
            parser.error(f"--folds takes main and fold numbers, got {bad}")
        gpus = [g for g in args.gpus.split(",") if g]
        return int(queue(args.configs, gpus, folds) > 0)
    report(args.config, args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
