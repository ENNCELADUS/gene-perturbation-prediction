"""The stage-one ladder's synthetic world: split, bulk and labels files, per-line h5ad
and a ladder config over training, held-out and bulk-only lines."""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import yaml
from scipy.sparse import csr_matrix

from src.data.splits import SPLIT_SCHEMA_VERSION
from src.experiments.stage_one.config import validate_ladder_config

TRAIN = tuple(f"ACH-T{i}" for i in range(10))  # T8 has no bulk; T9 is lymphoid
UNLABELLED = ("ACH-U0",)
VAL = ("ACH-V0", "ACH-V1", "ACH-V2")
TEST = ("ACH-S0", "ACH-S1", "ACH-S2")
BULK_LABELLED = tuple(f"ACH-B{i}" for i in range(6))  # B5 is myeloid
BULK_UNLABELLED = ("ACH-C0", "ACH-C1")
EFFECT_GENES = (*(f"G{i}" for i in range(8)), "C1ORF1")
BULK_GENES = (*EFFECT_GENES, "B1", "B2", "B3")
HVG = ("H1", "H2", "H3", "H4")
SOURCE = (*(f"G{i}" for i in range(8)), "C1orf1", "B1", "B2", "B3", *HVG, "MT-CO1")
PROGRAMS = ("S", "G2M", "P1")
CELLS = 30
TARGET_SUM = 5000.0


def _lineage(model_id: str) -> str:
    return {"ACH-T9": "Lymphoid", "ACH-B5": "Myeloid"}.get(model_id, "Lung")


def build_world(root: Path) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    split_lines = (*TRAIN, *UNLABELLED, *VAL, *TEST)
    (root / "split.json").write_text(
        json.dumps(
            {
                "schema_version": SPLIT_SCHEMA_VERSION,
                "train": [*TRAIN, *UNLABELLED],
                "val": list(VAL),
                "test": list(TEST),
                "unlabeled_train": list(UNLABELLED),
                "bulk_labelled": list(BULK_LABELLED),
                "bulk_unlabelled": list(BULK_UNLABELLED),
                "excluded": {},
            }
        )
    )
    every = (*split_lines, *BULK_LABELLED, *BULK_UNLABELLED)
    pd.DataFrame(
        {
            "ModelID": every,
            "PatientID": [f"P-{m}" for m in every],
            "OncotreeLineage": [_lineage(m) for m in every],
            "RRID": "",
        }
    ).to_csv(root / "Model.csv", index=False)

    labelled = (*TRAIN, *VAL, *TEST, *BULK_LABELLED)
    effects = rng.normal(-0.4, 0.6, size=(len(labelled), len(EFFECT_GENES)))
    effects[len(TRAIN) : len(TRAIN) + len(VAL) + len(TEST)] += 100.0  # leak marker
    pd.DataFrame(
        effects,
        index=pd.Index(labelled),
        columns=[f"{g} ({i + 1})" for i, g in enumerate(EFFECT_GENES)],
    ).to_csv(root / "gene_effect.csv")

    with_bulk = [m for m in every if m not in ("ACH-T8", *TEST)]
    bulk = rng.normal(4.0, 1.5, size=(len(with_bulk), len(BULK_GENES))).clip(0)
    for row, model_id in enumerate(with_bulk):
        if model_id in VAL:
            bulk[row] += 100.0  # validation bulk RNA must never be read
    frame = pd.DataFrame(
        bulk, columns=[f"{g} ({i + 1})" for i, g in enumerate(BULK_GENES)]
    )
    frame.insert(0, "ModelID", with_bulk)
    frame.insert(1, "IsDefaultEntryForModel", "Yes")
    frame.insert(2, "SequencingID", [f"CDS-{i}" for i in range(len(with_bulk))])
    frame.to_csv(root / "bulk.csv", index=False)

    (root / "basal_raw_umi").mkdir()
    programs = root / "prepared" / "noise" / "programs"
    programs.mkdir(parents=True)
    (root / "prepared" / "lines").mkdir()
    for index, model_id in enumerate(split_lines):
        line_rng = np.random.default_rng(100 + index)
        genes = (
            SOURCE if model_id != "ACH-T1" else tuple(g for g in SOURCE if g != "B3")
        )
        counts = line_rng.poisson(4.0, size=(CELLS, len(genes))).astype(np.float32)
        counts[:, : len(EFFECT_GENES)] *= 1 + index % 3
        ad.AnnData(
            X=csr_matrix(counts),
            obs=pd.DataFrame(
                {"model_id": model_id},
                index=[f"{model_id}-c{i}" for i in range(CELLS)],
            ),
            var=pd.DataFrame(
                {
                    "ensembl_id": [f"ENSG{SOURCE.index(g):011d}" for g in genes],
                    "gene_symbol": list(genes),
                },
                index=[f"v{i}" for i in range(len(genes))],
            ),
        ).write_h5ad(root / "basal_raw_umi" / f"{model_id}.h5ad")
        np.savez(
            programs / f"{model_id}.npz",
            scores=line_rng.normal(size=(CELLS, len(PROGRAMS))).astype(np.float32),
        )
        np.savez(
            root / "prepared" / "lines" / f"{model_id}.npz",
            controls_tx1=np.zeros((4, 8), dtype=np.float32),
            basal_hvg=line_rng.normal(size=(4, len(HVG))).astype(np.float32),
            q_sc_values=np.zeros((len(EFFECT_GENES), 3), dtype=np.float32),
            q_sc_available=np.ones(len(EFFECT_GENES), dtype=bool),
        )
    (root / "prepared" / "prepared_inputs.json").write_text(
        json.dumps(
            {
                "expression_space": {"target_sum": TARGET_SUM},
                "common_gene_panel": list(EFFECT_GENES),
                "hvg_order": list(HVG),
                "lines": list(split_lines),
            }
        )
    )
    gene_text = root / "gene_text"
    gene_text.mkdir()
    keys = [*(f"G{i}" for i in range(7)), "C1orf1", "OTHER"]  # G7 has no text
    (gene_text / "meta.json").write_text(json.dumps({"keys": keys}))

    joint = yaml.safe_load(Path("configs/geneeffect_joint.yaml").read_text())
    joint["prepared_root"] = str(root / "prepared")
    joint["features"]["selective_min_lines"] = 2
    joint["paths"].update(
        split=str(root / "split.json"),
        gene_effect=str(root / "gene_effect.csv"),
        basal_raw_umi=str(root / "basal_raw_umi"),
        tx1_model_dir=str(root / "tx1_model"),
    )
    (root / "joint.yaml").write_text(yaml.safe_dump(joint))
    prior = yaml.safe_load(Path("configs/context_prior/default_prior.yaml").read_text())
    prior["joint_config"] = str(root / "joint.yaml")
    prior["paths"].update(
        model=str(root / "Model.csv"), bulk_expression=str(root / "bulk.csv")
    )
    (root / "prior.yaml").write_text(yaml.safe_dump(prior))
    noise = yaml.safe_load(
        Path("configs/context_prior/noise_diagnosis.yaml").read_text()
    )
    noise["cell_states"].update(
        programs=list(PROGRAMS), cluster_components=2, cluster_counts=[2, 3]
    )
    (root / "noise.yaml").write_text(yaml.safe_dump(noise))
    config = yaml.safe_load(Path("configs/stage_one/ladder_inputs.yaml").read_text())
    config.update(
        prior_config=str(root / "prior.yaml"),
        noise_config=str(root / "noise.yaml"),
        ladder_root=str(root / "ladder"),
        frame={"folds": 2},
    )
    config["paths"].update(
        gene_text=str(gene_text),
        context_text=str(root / "ladder" / "context_text"),
        tx1_cache=str(root / "tx1"),
    )
    return validate_ladder_config(config)
