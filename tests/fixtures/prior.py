"""The context prior's synthetic world: a ``RunBase`` of bulk and pseudo-bulk lines over
twelve genes, and a small prior config with its run base on disk."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.context_prior.bridge import BridgeBase
from src.context_prior.reference import Reference
from src.context_prior.space import quantile_reference
from src.context_prior.targets import Definitions
from src.data.splits import FixedSplit
from src.experiments.prior import base as prior_base
from src.experiments.prior.config import validate_prior_config

GENES = [f"G{i}" for i in range(12)]


def base(seed=0):
    rng = np.random.default_rng(seed)
    sc = [f"S{i}" for i in range(20)]  # S18, S19 have no bulk row
    paired = sc[:18]
    extras = [f"E{i}" for i in range(10)]
    signal = rng.normal(size=(len(sc) + 12, 12))
    pseudo = pd.DataFrame(
        signal + 0.3 * rng.normal(size=signal.shape),
        index=[*sc, *[f"V{i}" for i in range(6)], *[f"T{i}" for i in range(6)]],
        columns=GENES,
    )
    bulk = pd.DataFrame(
        np.vstack([2 * signal[:18] + 1, rng.normal(size=(10, 12))]),
        index=[*paired, *extras],
        columns=GENES,
    )
    oracle = pd.DataFrame(
        rng.normal(size=(6, 12)), index=[f"V{i}" for i in range(6)], columns=GENES
    )
    return BridgeBase(
        bulk=bulk,
        oracle=oracle,
        pseudobulk=pseudo,
        paired=tuple(paired),
        single_cell_train=tuple(sc),
        val=tuple(f"V{i}" for i in range(6)),
        test=tuple(f"T{i}" for i in range(6)),
        folds={m: i % 4 for i, m in enumerate(sc)},
    )


CONFIG = yaml.safe_load(Path("configs/context_prior/bridge_remedies.yaml").read_text())
TARGETS = GENES[:8]


def tiny_config(tmp_path: Path) -> dict:
    config = copy.deepcopy(CONFIG)
    config["output_root"] = str(tmp_path / "out")
    config["prior"] = {
        "components": 4,
        "folds": 4,
        "bootstrap_repeats": 20,
        "selected_genes": 2,
    }
    config["experiments"] = {
        "affine": copy.deepcopy(CONFIG["experiments"]["affine"]),
        "affine_again": {
            "kind": "affine",
            "settings": [{}],
            "block_sets": ["components", "selected"],
        },
    }
    return validate_prior_config(config)


def synthetic_base(config) -> prior_base.RunBase:
    """The bridging test's lines; labels follow expression with noise."""
    bridge = base()
    extras = [m for m in bridge.bulk.index if m not in set(bridge.paired)]
    expression = pd.concat([bridge.pseudobulk, bridge.bulk.loc[extras]])
    lines = list(expression.index)
    rng = np.random.default_rng(1)
    effect = (
        -0.5
        + 0.5 * expression.loc[:, GENES[1:9]].to_numpy()
        + 0.2 * rng.normal(size=(len(lines), len(TARGETS)))
    )
    gene_effect = pd.DataFrame(effect, index=lines, columns=TARGETS)
    gene_effect.iloc[0, 0] = np.nan  # a missing training label
    definitions = Definitions(
        genes=tuple(TARGETS),
        gene_means=pd.Series(-0.5, index=TARGETS),
        selective=tuple(TARGETS[:6]),
        variable=tuple(TARGETS),
        residual_scale=pd.Series(0.5, index=TARGETS),
    )
    side = pd.Index(list(bridge.bulk.index), name="model_id")
    reference = Reference(
        paralogs=pd.DataFrame(
            {"gene": ["G0", "G2"], "paralog": ["G1", "G3"], "identity": [50.0, 40.0]}
        ),
        complexes=pd.DataFrame({"complex_id": [1, 1], "gene": ["G4", "G5"]}),
        hallmark=pd.DataFrame({"gene_set": ["S"], "gene": ["G7"]}),
        progeny=pd.DataFrame({"pathway": ["P"], "gene": ["G8"], "weight": [1.0]}),
        drivers=pd.DataFrame({"KRAS": 0}, index=side),
        msi=pd.Series(0.0, index=side),
    )
    return prior_base.RunBase(
        config=config,
        split=FixedSplit(
            train=bridge.single_cell_train, val=bridge.val, test=bridge.test
        ),
        gene_effect=gene_effect,
        definitions=definitions,
        labelled=(*bridge.paired, *extras),
        models=pd.DataFrame({"patient_id": lines, "lineage": "Lung"}, index=lines),
        reference=reference,
        bridge=bridge,
        filled_lines=0,
        profile=quantile_reference(bridge.bulk),
        pseudobulk_fill=bridge.pseudobulk.loc[list(bridge.single_cell_train)].mean(),
    )
