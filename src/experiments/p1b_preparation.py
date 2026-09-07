"""One-time P1-B bundle construction and source provenance audit."""

import hashlib
import json
import random
from pathlib import Path
import numpy as np
import torch
from src.data.p1b import (
    SOURCE_ANCHORS,
    EXTERNAL_ANCHOR,
    build_snapshot,
    common_coordinates,
    source_vocabularies,
    ResponseView,
)
from src.data.response_cache import open_response_targets_cache
from src.model.initialization import build_joint_model
from src.model.p1b import configure_parameters
from src.experiments.geneeffect import _write_json, _revision
from src.training.p1b import save


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def restore_backbone(template):
    from copy import deepcopy
    from state.tx.models.state_transition import StateTransitionPerturbationModel
    from src.data.embeddings import Esm2EmbeddingTable
    from src.model.initialization import _suppress_checkpoint_output
    from src.model.state import ForwardOnlyStateModel, StateForwardAdapter
    from src.model.perturbation import Esm2PerturbationAdapter

    preprocessing = template["preprocessing"]
    architecture = template["architecture"]
    vectors = preprocessing["esm2_vectors"].numpy()
    symbols = preprocessing["esm2_symbols"]
    with _suppress_checkpoint_output():
        state = StateTransitionPerturbationModel(
            **deepcopy(architecture["state_hparams"])
        )
    table = Esm2EmbeddingTable(vectors.shape[1], dict(zip(symbols, vectors)))
    perturbations = Esm2PerturbationAdapter(
        symbols,
        table,
        architecture["esm2_adapter_hidden"],
        architecture["state_hparams"]["pert_dim"],
    )
    backbone = ForwardOnlyStateModel(StateForwardAdapter(state), perturbations)
    backbone.load_state_dict(template["model_state"], strict=True)
    return backbone


def prepare_bundle(checkpoint, directory):
    from src.data.prepared import load_inputs
    from src.training.checkpoint import load_checkpoint

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    _write_json(directory / "status.json", {"status": "running"})
    try:
        joint = load_checkpoint(checkpoint)
        config = joint["config"]
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        inputs = load_inputs(config, preprocessing=joint["preprocessing"])
        sources_path = Path(config["paths"]["perturbseq_sources"])
        sources = json.loads(sources_path.read_text())
        if set(sources) != set((*SOURCE_ANCHORS, EXTERNAL_ANCHOR)):
            raise ValueError("P1-B requires exactly the approved four sources")
        vocabularies = source_vocabularies(sources)
        if len(inputs.hvg_order) != 2000:
            raise ValueError("P1-B requires unchanged 2000-output architecture")
        coordinates = common_coordinates(inputs.hvg_order, vocabularies.values())
        native_map_path = (
            Path(config["paths"]["state_model_dir"]) / "pert_onehot_map.pt"
        )
        native_map = torch.load(native_map_path, map_location="cpu", weights_only=False)
        native_genes = {str(g) for g in native_map}
        bundle = build_snapshot(inputs, coordinates, native_genes)
        counts = {k: len(v) for k, v in bundle["splits"].items()}
        if counts != {"train": 27361, "val": 3047, "external": 2377}:
            raise ValueError(f"P1-B membership counts changed: {counts}")
        if (
            max(
                sum(bundle["keys"][i][0] == a for i in bundle["splits"]["train"])
                for a in SOURCE_ANCHORS
            )
            != 16399
        ):
            raise ValueError("largest anchor must have 16399 training conditions")
        external_panels = {
            name: len(bundle["panels"][f"external/{EXTERNAL_ANCHOR}/{name}"]["indices"])
            for name in ("seen", "unseen", "native_common", "native_all")
        }
        if external_panels != {
            "seen": 2373,
            "unseen": 4,
            "native_common": 2006,
            "native_all": 2009,
        }:
            raise ValueError(f"Jurkat strata changed: {external_panels}")
        # Reset immediately before exactly the same public constructor as P0.
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        model = build_joint_model(config, inputs)
        reference = torch.load(
            config["paths"]["state_checkpoint"], map_location="cpu", weights_only=False
        )
        destination = model.backbone.state_adapter.state_model.state_dict()
        loaded = [
            n
            for n, t in reference["state_dict"].items()
            if n in destination and tuple(t.shape) == tuple(destination[n].shape)
        ]
        skipped = [n for n in destination if n not in loaded]
        _, report = configure_parameters(model.backbone, loaded)
        bundle["loaded_keys"] = loaded
        bundle["response_cache"] = str(inputs.response_cache.resolve())
        initial = {
            "config": config,
            "preprocessing": joint["preprocessing"],
            "architecture": model.architecture,
            "model_state": model.backbone.state_dict(),
        }
        save(directory / "B-init.pt", initial)
        # Only the model construction fields are retained, not the P0 optimizer.
        joint_template = {k: joint[k] for k in initial}
        joint_template["model_state"] = {
            k.removeprefix("backbone."): v
            for k, v in joint["model_state"].items()
            if k.startswith("backbone.")
        }
        save(directory / "B-joint.pt", joint_template)
        root = inputs.response_cache / "response_targets"
        cache_identity = {
            name: digest(root / name)
            for name in (
                "manifest.json",
                "metadata.parquet",
                "offsets.npy",
                "target_cells.npy",
            )
        }
        bundle["cache_identity"] = cache_identity
        target_stat = (root / "target_cells.npy").stat()
        bundle["target_stat"] = {
            "size": target_stat.st_size,
            "mtime_ns": target_stat.st_mtime_ns,
        }
        save(directory / "bundle.pt", bundle)
        record = {
            "schema": "p1b-v1",
            "revision": _revision(),
            "checkpoint": str(Path(checkpoint).resolve()),
            "checkpoint_sha256": digest(checkpoint),
            "native_checkpoint_sha256": digest(config["paths"]["state_checkpoint"]),
            "source_registry_sha256": digest(sources_path),
            "native_vocabulary_sha256": digest(native_map_path),
            "bundle_sha256": digest(directory / "bundle.pt"),
            "counts": counts,
            "external_panels": external_panels,
            "model_files": {
                name: digest(directory / f"{name}.pt") for name in ("B-init", "B-joint")
            },
            "coordinates": coordinates,
            "measured_genes": [inputs.hvg_order[i] for i in coordinates],
            "source_missing_genes": {
                a: [g for g in inputs.hvg_order if g not in v]
                for a, v in vocabularies.items()
            },
            "loaded_keys": loaded,
            "shape_skipped_keys": skipped,
            "parameters": report,
            "native": {
                "status": "unavailable",
                "reason": (
                    "Original normalization/output-scale and batch mapping are "
                    "not verified. int_counts:false is insufficient; external "
                    "training config unavailable."
                ),
            },
            "exposure": (
                "Jurkat held out only from interface adaptation; "
                "ST/Tx1 pretraining non-exposure unverified"
            ),
        }
        _write_json(directory / "manifest.json", record)
        _write_json(directory / "status.json", {"status": "completed"})
    except Exception as exc:
        _write_json(directory / "status.json", {"status": "failed", "error": str(exc)})
        raise


def open_bundle(directory):
    directory = Path(directory)
    if json.loads((directory / "status.json").read_text())["status"] != "completed":
        raise ValueError("P1-B preparation incomplete")
    manifest = json.loads((directory / "manifest.json").read_text())
    if digest(directory / "bundle.pt") != manifest["bundle_sha256"]:
        raise ValueError("prepared bundle identity changed")
    bundle = torch.load(directory / "bundle.pt", map_location="cpu", weights_only=False)
    root = Path(bundle["response_cache"]) / "response_targets"
    target_stat = (root / "target_cells.npy").stat()
    if {"size": target_stat.st_size, "mtime_ns": target_stat.st_mtime_ns} != bundle[
        "target_stat"
    ]:
        raise ValueError("response target file changed after preparation")
    # Runtime never recursively fingerprints/rebuilds raw caches. Small identity
    # files bind keys, layout and the preparation-time full target hash.
    for name in ("manifest.json", "metadata.parquet", "offsets.npy"):
        if digest(root / name) != bundle["cache_identity"][name]:
            raise ValueError(f"response cache identity changed: {name}")
    cache = open_response_targets_cache(
        Path(bundle["response_cache"]), expected_hvg_order=bundle["hvg_order"]
    )
    return bundle, ResponseView(bundle, cache), manifest


def load_state(directory, state, manifest):
    path = Path(directory) / f"{state}.pt"
    if digest(path) != manifest["model_files"][state]:
        raise ValueError("prepared model identity changed")
    return restore_backbone(torch.load(path, map_location="cpu", weights_only=False))
