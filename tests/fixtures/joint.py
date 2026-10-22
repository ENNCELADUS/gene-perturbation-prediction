"""The joint model's synthetic world: ten lines over a 2000-HVG panel, their
``PreparedInputs`` built in memory, a matching config, and a STATE-shaped released
checkpoint."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

pytest.importorskip("accelerate")
pytest.importorskip("state.tx.models.state_transition")

import dataclasses

import pytest

from src.data.context_pca import fit_context_pca, pooled_context  # noqa: E402
from src.data.prepared import PreparedInputs, PreparedLine  # noqa: E402
from src.data.prior_offsets import (
    PriorOffsets,
)
from src.data.q_sc import QScFeatures  # noqa: E402
from src.data.splits import FixedSplit  # noqa: E402
from src.experiments.joint.config import load_config  # noqa: E402
from src.model.state import build_state  # noqa: E402

CONFIG = Path("configs/geneeffect_joint.yaml")


HVG = 2000
TX1 = 8
ESM2 = 4
CELLS = 8
SENTENCE = 4
ANCHORS = ("ACH-A0", "ACH-A1", "ACH-A2", "ACH-A3")
TRAIN = (*ANCHORS, "ACH-T0", "ACH-T1")
VAL = ("ACH-V0", "ACH-V1", "ACH-V2")  # three: Spearman needs three lines
TEST = ("ACH-X0",)
GENES = ("G0", "G1", "G2", "G3", "G4", "G5")
SELECTIVE = ("G0", "G2", "G3", "G5")
HVG_ORDER = ("G0", "G1", *(f"H{i}" for i in range(HVG - 2)))
RESPONSE_GENES = ("G0", "G2", "G4")
CONTEXT_COMPONENTS = 4


class FakeResponseTargets:
    """The prepared response cache's interface: ordered keys and one bag per key."""

    def __init__(self, bags: dict[tuple[str, str], np.ndarray]) -> None:
        self.keys = tuple(bags)
        self._bags = list(bags.values())

    def target_bag(self, index: int) -> np.ndarray:
        return self._bags[index]


def make_inputs(seed: int = 0) -> PreparedInputs:
    rng = np.random.default_rng(seed)
    lines = (*TRAIN, *VAL, *TEST)
    labels = pd.DataFrame(
        [(m, g, float(rng.normal(-0.3, 0.5))) for m in lines for g in GENES],
        columns=["model_id", "gene_symbol", "gene_effect"],
    )
    means = (
        labels[labels.model_id.isin(TRAIN)].groupby("gene_symbol").gene_effect.mean()
    )
    labels["residual"] = labels.gene_effect - labels.gene_symbol.map(means)
    scale = (
        labels[labels.model_id.isin(TRAIN)].groupby("gene_symbol").residual.std(ddof=0)
    )
    prepared_lines = {
        m: PreparedLine(
            controls_tx1=rng.normal(size=(CELLS, TX1)).astype(np.float32),
            basal_hvg=np.log1p(rng.poisson(1.0, size=(CELLS, HVG))).astype(np.float32),
            q_sc=QScFeatures(
                symbols=GENES,
                values=rng.random((len(GENES), 3)),
                available=np.array([True, True, True, False, True, True]),
            ),
        )
        for m in lines
    }
    bags = {
        (anchor, gene): np.log1p(rng.poisson(1.5, size=(5, HVG))).astype(np.float32)
        for anchor in ANCHORS
        for gene in RESPONSE_GENES
    }
    return PreparedInputs(
        split=FixedSplit(train=TRAIN, val=VAL, test=TEST),
        labels=labels,
        genes=GENES,
        train_gene_means=means.reindex(list(GENES)),
        variable_genes=frozenset(GENES),
        selective_genes=frozenset(SELECTIVE),
        residual_scale=scale.reindex(list(GENES)),
        context_pca=fit_context_pca(
            np.stack([pooled_context(prepared_lines[m].controls_tx1) for m in TRAIN]),
            CONTEXT_COMPONENTS,
        ),
        hvg_order=HVG_ORDER,
        esm2_symbols=GENES,
        esm2_vectors=rng.normal(size=(len(GENES), ESM2)).astype(np.float32),
        lines=prepared_lines,
        response_targets=FakeResponseTargets(bags),
        response_anchors=ANCHORS,
        target_sum=1000.0,
    )


def write_released_state(path: Path, *, dropout: float = 0.0) -> Path:
    """A STATE-shaped released checkpoint: 2000-wide basal encoder, batch encoder."""
    hparams = dict(
        input_dim=HVG,
        hidden_dim=8,
        output_dim=HVG,
        pert_dim=6,
        batch_dim=3,
        batch_encoder=True,
        cell_set_len=SENTENCE,
        predict_residual=True,
        output_space="gene",
        embed_key="X_hvg",
        gene_names=list(HVG_ORDER),
        transformer_backbone_kwargs={
            "n_embd": 8,
            "n_layer": 1,
            "n_head": 2,
            "resid_pdrop": dropout,
            "embd_pdrop": dropout,
            "attn_pdrop": dropout,
        },
        n_encoder_layers=1,
        n_decoder_layers=1,
        dropout=0.0,
    )
    torch.manual_seed(1)
    state = build_state(hparams)
    torch.save({"hyper_parameters": hparams, "state_dict": state.state_dict()}, path)
    return path


def make_config(
    root: Path, *, dropout: float = 0.0, state: bool = True, **train
) -> dict:
    """The production config at toy sizes; ``state=False`` drops both STATE blocks."""
    config = load_config(CONFIG)
    config["output_root"] = str(root / "runs")
    config["paths"]["state_checkpoint"] = str(
        write_released_state(root / "released.ckpt", dropout=dropout)
    )
    config["model"].update(
        cell_sentence_len=SENTENCE,
        esm2_adapter_hidden=4,
        factor_rank=4,
        context_components=CONTEXT_COMPONENTS,
    )
    config["model"]["head_blocks"].update(use_delta_proj=state, use_s=state)
    config["train"].update(
        max_epochs=2,
        patience=5,
        dependency_batch_size=4,
        genes_per_block=2,
        **train,
    )
    return config


def with_prior(inputs, per_line: dict[str, float]):
    frame = pd.DataFrame({gene: pd.Series(per_line) for gene in inputs.genes}).loc[
        :, list(inputs.genes)
    ]
    prior = PriorOffsets({"run_id": "r", "reference": {}}, frame, inputs.residual_scale)
    return dataclasses.replace(inputs, prior=prior)
