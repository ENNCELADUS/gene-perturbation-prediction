"""What a stage-one run keeps on the device: the ladder's arrays for the run's genes and
lines, read once, and the batches the model reads from them."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.data import ladder_reader as reader
from src.data.io import read_json
from src.experiments.stage_one.setup import Setup

log = logging.getLogger(__name__)


def centred(vectors: np.ndarray, present: np.ndarray, mean: np.ndarray) -> np.ndarray:
    """Source vectors minus the source's panel mean; an absent gene stays zero."""
    return np.where(present[:, None], vectors - mean, 0.0).astype(np.float32)


class Resident:
    """The inputs a training config reads, on the device. ``batch(genes, lines)``
    assembles a ``PairBatch`` from positions in the run's genes and line positions;
    a permuted control reads, for each gene, another line's delta and slots (a
    fixed, seeded permutation of the lines)."""

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
        # Each source's mean over the panel genes it covers (label-free, over the
        # gene axis; no per-gene parameter): gene_text vectors share half their
        # energy along it, which leaves the gene encoder's conditions nearly equal
        # across genes and the FiLM arms learning almost nothing gene-specific.
        self.source_means = {
            name: vectors[on_panel][present[on_panel]].mean(axis=0)
            if data["centre_gene_sources"]
            else np.zeros(vectors.shape[1], np.float32)
            for name, (vectors, present) in sources.items()
        }
        self.gene_vectors = {
            name: torch.from_numpy(
                centred(vectors[on_panel], present[on_panel], self.source_means[name])
            ).to(device)
            for name, (vectors, present) in sources.items()
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
        self.prior = None
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
        if data["prior_run"] is not None and kind != "components":
            self.prior = torch.from_numpy(prior_feature(setup, data, targets.m_g)).to(
                device
            )
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
        path = Path(setup.config["paths"]["gene_text"])
        vectors, present = {}, {}
        for name, dim in self.source_dims.items():
            if name == "gene_text":
                values, listed = reader.text_on_axis(path, axis)
            else:
                values = np.zeros((len(axis), dim), np.float32)
                listed = np.zeros(len(axis), bool)
            values = centred(values, listed, self.source_means[name])
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
        if self.prior is not None:
            # Never permuted: the permuted control keeps the line's own prior.
            batch.prior = self.prior[genes][:, lines]
        return batch


def prior_feature(setup: Setup, data: Mapping[str, Any], m_g: pd.Series) -> np.ndarray:
    """The prior reference's prediction minus the run's base, in z units (panel
    genes x ladder lines): frame lines out of fold, validation and test lines from
    the fit on the whole frame. A fold run's training lines read predictions whose
    fits saw the held-out fold's labels, so its out-of-fold telemetry is slightly
    optimistic; validation and test are unaffected."""
    ladder = setup.ladder
    run_dir = finished_prior(data["prior_run"], data["ladder_config"])
    if run_dir is None:
        raise FileNotFoundError(f"{data['prior_run']} has no row.json")
    predictions = pd.concat(
        [
            pd.read_parquet(run_dir / f"predictions_{name}.parquet")
            for name in ("oof", "val", "test")
        ],
        verify_integrity=True,
    )
    predictions.index = [str(m) for m in predictions.index]
    missing = sorted(set(ladder.line_ids) - set(predictions.index))
    if missing:
        raise ValueError(f"the prior predicts no {missing[:5]}")
    panel = list(ladder.panel)
    base = reader.base_prediction(ladder, m_g, ladder.line_ids).loc[:, panel]
    values = predictions.loc[ladder.line_ids, panel] - base
    return values.to_numpy(np.float32).T.copy()


def same_ladder(path: str | Path, report_ladder: str | Path, what: str) -> None:
    if Path(path).resolve() != Path(report_ladder).resolve():
        raise ValueError(f"{what} read {path}, the report {report_ladder}")


def finished_prior(run: str | Path, report_ladder: str | Path) -> Path | None:
    """The prior reference's run directory once its ``row.json`` exists, checked to
    read the report's ladder."""
    run_dir = Path(run)
    if not (run_dir / "row.json").is_file():
        return None
    bound = read_json(run_dir / "run_config.json")["config"]
    if bound["kind"] != "prior_reference":
        raise ValueError(f"{run_dir} is not a prior reference run")
    same_ladder(bound["ladder_config"], report_ladder, str(run_dir))
    return run_dir
