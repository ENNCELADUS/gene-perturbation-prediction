"""Real tiny STATE train/checkpoint/evaluate and independent export recovery."""

import copy
import json
from pathlib import Path

import pandas as pd
import pytest
import torch
import yaml

from src.experiments.config import load_config
from src.experiments import geneeffect
from src.training.checkpoint import load_checkpoint
from test_joint_training import tiny_training_config, assert_tree_equal


def full_tiny_config(root):
    config = load_config(Path("configs/geneeffect_joint.yaml"))
    tiny = tiny_training_config(root)
    for key, value in tiny.items():
        if isinstance(value, dict):
            config[key].update(value)
        else:
            config[key] = value
    config["output_root"] = str(root / "runs")
    return config


def test_real_training_independent_test_and_export_retry(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    config = full_tiny_config(tmp_path / "inputs")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    run_dir = geneeffect.run_training(path, run_id="tiny")
    record = json.loads((run_dir / "run.json").read_text())
    assert record["training"]["status"] == "completed"
    assert record["evaluation"]["status"] == "not_started"
    checkpoint = run_dir / "best.pt"
    saved = load_checkpoint(checkpoint)
    before = checkpoint.read_bytes()
    training = copy.deepcopy(record["training"])
    Path(config["paths"]["esm2_embeddings"]).unlink()
    Path(config["paths"]["state_checkpoint"]).unlink()
    from src.data import geneeffect as data, residual_target

    def forbidden(*a, **kw):
        pytest.fail("checkpoint evaluation refitted preprocessing or optimized")

    monkeypatch.setattr(data, "fit_variable_gene_membership", forbidden)
    monkeypatch.setattr(residual_target, "fit_gene_means", forbidden)
    monkeypatch.setattr(torch.optim.AdamW, "step", forbidden)
    original_export = geneeffect.export_evaluation

    def fail_export(*a, **kw):
        raise OSError("deliberate export failure")

    monkeypatch.setattr(geneeffect, "export_evaluation", fail_export)
    with pytest.raises(OSError, match="deliberate"):
        geneeffect.evaluate_checkpoint(checkpoint, split="test")
    failed = json.loads((run_dir / "run.json").read_text())
    assert failed["training"] == training
    assert failed["evaluation"]["status"] == "failed"
    assert failed["evaluation"]["phase"] == "export"
    monkeypatch.setattr(geneeffect, "export_evaluation", original_export)
    result = geneeffect.evaluate_checkpoint(checkpoint, split="test")
    after = json.loads((run_dir / "run.json").read_text())
    assert after["training"] == training
    assert after["evaluation"]["status"] == "completed"
    assert all(key.startswith("test_") for key in result.metrics)
    assert set(result.predictions.model_id) == {"ACH-TEST"}
    exported = pd.read_parquet(run_dir / "evaluation/best/test/predictions.parquet")
    pd.testing.assert_frame_equal(exported, result.predictions)
    assert checkpoint.read_bytes() == before
    assert_tree_equal(saved["model_state"], load_checkpoint(checkpoint)["model_state"])
    # Config disagreement is rejected before changing completed training state.
    conflict = copy.deepcopy(config)
    conflict["train"]["head_learning_rate"] *= 2
    path.write_text(yaml.safe_dump(conflict))
    with pytest.raises(ValueError, match="conflicts"):
        geneeffect.run_training(path, resume=run_dir / "last.pt")
    assert json.loads((run_dir / "run.json").read_text())["training"] == training
    path.write_text(yaml.safe_dump(config))
    geneeffect.run_training(path, resume=run_dir / "last.pt")
    assert json.loads((run_dir / "run.json").read_text())["training"] == training


def test_real_raw_preparation_writes_order_and_original_holdout(tmp_path, monkeypatch):
    """Tiny raw h5ads exercise the production assembly, q_sc writer and readers."""
    import pickle
    import anndata as ad
    import numpy as np
    from src.data import geneeffect as data
    from src.data.geneeffect import Exp13Split
    from src.data.prepared import load_inputs
    from src.data.tx1_cache import write_line_cache
    from src.experiments.prepare import prepare_inputs, split_heldout_genes
    from conftest import tx1_manifest_row, write_tx1_line_manifest

    anchors = ("ACH-000551", "A1", "A2", "A3")
    split = Exp13Split(
        (*anchors, "TRAIN", "TRAIN2"), ("V1", "V2", "V3"), ("T1", "T2", "T3"), ()
    )
    monkeypatch.setattr(data, "load_exp13_split", lambda _: split)
    config = load_config(Path("configs/geneeffect_joint.yaml"))
    config["prepared_root"] = str(tmp_path / "prepared")
    config["features"].update(cells_per_context=4, esm2_dim=3)
    config["precision"] = "no"
    for key in (
        "split",
        "gene_effect",
        "source_registry",
        "cell_line_manifest",
        "perturbseq_sources",
    ):
        config["paths"][key] = str(tmp_path / key)
    for key in ("tx1_cache", "q_sc_cache", "state_model_dir", "response_cache"):
        config["paths"][key] = str(tmp_path / key)
    config["paths"]["common_gene_panel"] = str(
        tmp_path / "prepared/common_gene_panel.csv"
    )
    config["paths"]["esm2_embeddings"] = str(tmp_path / "esm2.npz")
    Path(config["paths"]["split"]).write_text(
        json.dumps(
            {
                name: list(getattr(split, name))
                for name in ("train", "val", "test", "unlabeled_train")
            }
        )
    )
    rows = len(split.all_model_ids)
    labels = pd.DataFrame(
        {
            "G0 (1)": np.arange(rows),
            "G1 (2)": np.arange(rows) ** 2,
            "G2 (3)": [np.nan, *np.arange(rows - 1)],
        },
        index=split.all_model_ids,
    )
    # G2 meets 5/3/3 availability but its missing K562 donor excludes it.
    labels.to_csv(config["paths"]["gene_effect"])
    hvg = ["G1", "G0", *[f"HVG{i}" for i in range(2, 2000)]]
    state = Path(config["paths"]["state_model_dir"])
    state.mkdir()
    (state / "var_dims.pkl").write_bytes(pickle.dumps({"gene_names": hvg}))
    registry, manifest, sources = [], [], {}
    response_genes = [f"R{i}" for i in range(20)]
    original_holdout = split_heldout_genes(
        {anchor: response_genes for anchor in anchors}, fraction=0.1, seed=13
    )
    excluded = next(
        gene
        for gene in response_genes
        if all(gene not in held for held in original_holdout.values())
    )
    esm = ["G0", "G1", "G2", *response_genes]
    np.savez(
        config["paths"]["esm2_embeddings"],
        symbols=np.asarray(esm),
        vectors=np.ones((len(esm), 3), dtype=np.float32),
        resolved=np.asarray([gene != excluded for gene in esm]),
    )
    for index, model_id in enumerate(split.all_model_ids):
        basal = tmp_path / f"basal-{model_id}.h5ad"
        ad.AnnData(
            X=np.array([[1, 2], [3, 4]], dtype=np.float32),
            obs=pd.DataFrame({"model_id": [model_id] * 2}, index=["b1", "b2"]),
            var=pd.DataFrame({"gene_symbol": ["G0", "G1"]}, index=["e0", "e1"]),
        ).write_h5ad(basal)
        registry.append(
            dict(
                model_id=model_id,
                source_path=str(basal),
                source_kind="h5ad",
                matrix_semantics="raw_umi_counts",
            )
        )
        write_line_cache(
            Path(config["paths"]["tx1_cache"]),
            model_id,
            np.ones((2, 2560), dtype=np.float32) * (index + 1),
            np.ones((2, 2000), dtype=np.float32),
            pd.DataFrame(index=["b1", "b2"]),
            hvg_gene_order=hvg,
        )
        if model_id in anchors:
            manifest.append(
                tx1_manifest_row(
                    model_id=model_id,
                    cellosaurus_id=f"CVCL_{index}",
                    cell_line_name=f"Line{index}",
                    basal_source="Perturb-seq non-targeting control",
                    role="train_response_and_head",
                )
            )
            source = tmp_path / f"response-{model_id}.h5ad"
            source_names = ["G0", "G1", *hvg[2:]]
            targets = np.tile(np.arange(1, 2001, dtype=np.float32), (42, 1))
            ad.AnnData(
                X=targets,
                obs=pd.DataFrame(
                    {
                        "gene": ["non-targeting"] * 2
                        + [gene for gene in response_genes for _ in range(2)]
                    },
                    index=[f"c{i}" for i in range(42)],
                ),
                var=pd.DataFrame(
                    {"gene_symbol": source_names},
                    index=pd.Index(
                        [f"ENSG{i:011d}" for i in range(2000)], name="gene_id"
                    ),
                ),
            ).write_h5ad(source)
            sources[model_id] = dict(
                source_type="h5ad",
                h5ad_path=str(source),
                perturbation_col="gene",
                control_label="non-targeting",
                var_ensembl_col="gene_id",
                target_gene_symbol_col="gene_symbol",
            )
    pd.DataFrame(registry).to_csv(config["paths"]["source_registry"], index=False)
    write_tx1_line_manifest(Path(config["paths"]["cell_line_manifest"]), manifest)
    Path(config["paths"]["perturbseq_sources"]).write_text(json.dumps(sources))
    destination = prepare_inputs(config)
    payload = json.loads(destination.read_text())
    assert payload["hvg_order"] == hvg
    assert payload["common_gene_panel"] == ["G0", "G1"]
    assert {row["gene"] for row in payload["excluded_response_conditions"]} == {
        excluded
    }
    for anchor in anchors:
        assert {
            row["gene"]
            for row in payload["response_holdout"]
            if row["model_id"] == anchor
        } == original_holdout[anchor]
    opened = load_inputs(config, include_test=True)
    assert (
        opened.response_targets.target_bag(0)[0, 0]
        > opened.response_targets.target_bag(0)[0, 1]
    )
    assert payload["newly_encoded_tx1_lines"] == []
    # Malformed prepared metadata names the exact source and repair command.
    payload["common_gene_panel"].reverse()
    destination.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="hpc/run.sh prepare") as caught:
        load_inputs(config)
    assert str(destination) in str(caught.value)


def test_baseline_exports_use_current_axes_and_requested_split(tmp_path, monkeypatch):
    from dataclasses import replace
    import math
    import numpy as np
    from src.data import prepared
    from src.data.splits import FixedSplit
    from src.experiments.baselines import run_baselines
    from test_joint_data import make_prepared_fixture

    config = make_prepared_fixture(tmp_path / "inputs")
    inputs = prepared.load_inputs(config)
    labels = inputs.labels.copy()
    labels.loc[labels.model_id == "ACH-A0", "model_id"] = "ACH-000551"
    train = tuple(
        "ACH-000551" if line == "ACH-A0" else line for line in inputs.split.train
    )
    lines = {
        "ACH-000551" if line == "ACH-A0" else line: value
        for line, value in inputs.lines.items()
    }
    inputs = replace(
        inputs,
        labels=labels,
        lines=lines,
        split=FixedSplit(
            train=train,
            val=inputs.split.val,
            test=inputs.split.test,
            unlabeled_train=inputs.split.unlabeled_train,
        ),
    )
    opened = []

    def open_inputs(config, *, include_test=False):
        opened.append(include_test)
        return inputs

    monkeypatch.setattr(prepared, "load_inputs", open_inputs)
    result = run_baselines(config, split="val", out_dir=tmp_path / "baseline")
    assert opened == [False]
    assert set(result.predictions.model_id) == {"ACH-VAL"}
    for frame in result.predictions.groupby("method"):
        prediction = frame[1]
        np.testing.assert_allclose(
            prediction.geneeffect_prediction,
            prediction.residual_prediction
            + prediction.gene_symbol.map(inputs.train_gene_means),
        )
    assert all(
        key.startswith("val_") for metrics in result.summary.values() for key in metrics
    )
    assert math.isfinite(
        result.summary["gene_mean"]["val_geneeffect_pearson_macro_per_line"]
    )
