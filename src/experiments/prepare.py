"""Prepare fixed joint-training inputs once, before starting training workers."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def split_heldout_genes(
    genes_by_line: Mapping[str, Sequence[str]], *, fraction: float, seed: int
) -> dict[str, frozenset[str]]:
    """Preserve historical per-line SHA256 ranking, before ESM2 exclusions."""
    if not 0 < fraction < 1:
        raise ValueError(f"fraction must be in (0, 1), got {fraction}")
    result = {}
    for model_id, genes in genes_by_line.items():
        unique = sorted(set(map(str, genes)))
        n_hold = int(len(unique) * fraction)
        if n_hold < 1:
            raise ValueError(
                f"{model_id}: {len(unique)} genes cannot yield a held-out set "
                f"at fraction {fraction}"
            )
        ranked = sorted(
            unique,
            key=lambda gene: hashlib.sha256(
                f"{seed}|{model_id}|{gene}".encode()
            ).hexdigest(),
        )
        result[model_id] = frozenset(ranked[:n_hold])
    return result


def _prepare_tx1(config, registry):
    from src.data.tx1_cache import (
        embed_registry_lines,
        load_hvg_gene_order,
        open_line_cache,
    )

    paths, settings = config["paths"], config["preparation"]
    cache = Path(paths["tx1_cache"])
    hvg_order = load_hvg_gene_order(Path(paths["state_model_dir"]))
    missing = []
    for model_id in registry.index.astype(str):
        try:
            open_line_cache(cache, model_id, expected_hvg_order=hvg_order)
        except (FileNotFoundError, ValueError, OSError):
            missing.append(model_id)
    if missing:
        from src.model.tx1 import _build_tx1_encoder

        encoder, _ = _build_tx1_encoder(
            Path(paths["tx1_model_dir"]),
            settings["tx1_batch_size"],
            settings["tx1_max_length"],
        )
        embed_registry_lines(
            registry,
            cache,
            encoder=encoder,
            hvg_state_model_dir=Path(paths["state_model_dir"]),
            var_ensembl_col=settings["var_ensembl_col"],
            hvg_gene_symbol_col=settings["hvg_gene_symbol_col"],
            max_cells_per_line=config["features"]["cells_per_context"],
            seed=0,
            only_lines=missing,
        )
        for model_id in missing:
            # This sidecar describes ONLY newly encoded cells, never old caches.
            (cache / model_id / "encoder_settings.json").write_text(
                json.dumps(
                    {
                        "collator_seed": 0,
                        "batch_size": settings["tx1_batch_size"],
                        "max_length": settings["tx1_max_length"],
                        "model_dir": paths["tx1_model_dir"],
                    },
                    indent=2,
                )
                + "\n"
            )
    return missing


def prepare_inputs(config: Mapping[str, Any]) -> Path:
    """Build aligned panel, basal caches and response targets; write metadata last."""
    import numpy as np
    import pandas as pd
    from src.data.geneeffect import (
        load_exp13_split,
        load_geneeffect_long,
        load_source_registry,
    )
    from src.data.prepared import load_inputs
    from src.data.q_sc import build_q_sc_shards
    from src.data.response import assemble_train_response_gene_bags
    from src.data.response_cache import write_response_targets_cache
    from src.data.tx1_cache import load_hvg_gene_order
    from src.data.prepare.build_exp13_esm2_universe import (
        build_coverage_universe,
        restrict_coverage_universe_to_copy_prior,
        write_embedding_union,
    )
    from src.experiments.config import validate_config

    validate_config(config)
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("prepare must run in one process before training launch")
    paths, settings = config["paths"], config["preparation"]
    root = Path(config["prepared_root"])
    split = load_exp13_split(Path(paths["split"]))
    labels = load_geneeffect_long(Path(paths["gene_effect"]), split)
    if "ACH-000551" not in split.supervised_train:
        raise ValueError("K562 copy-prior donor must be a labeled training line")
    donor = labels.loc[
        (labels.model_id == "ACH-000551") & np.isfinite(labels.gene_effect),
        "gene_symbol",
    ]
    candidates = restrict_coverage_universe_to_copy_prior(
        build_coverage_universe(labels, split), tuple(donor)
    )
    registry = load_source_registry(Path(paths["source_registry"]), split)
    encoded = _prepare_tx1(config, registry)
    hvg_order = tuple(
        str(gene) for gene in load_hvg_gene_order(Path(paths["state_model_dir"]))
    )
    bags = assemble_train_response_gene_bags(
        cell_line_manifest_path=Path(paths["cell_line_manifest"]),
        tx1_cache_dir=Path(paths["tx1_cache"]),
        hvg_state_model_dir=Path(paths["state_model_dir"]),
        perturbseq_sources_path=Path(paths["perturbseq_sources"]),
        max_cells_per_gene=settings["response_max_cells_per_gene"],
        total_cells_per_line=settings["response_total_cells_per_line"],
        control_cells_per_line=config["features"]["cells_per_context"],
        seed=settings["response_sampling_seed"],
    )
    # target_feature_names is produced by raw alignment plus per-line order checks.
    # Never upgrade a dimension-only historical header with an assumed order.
    if (
        bags.target_feature_names is None
        or tuple(bags.target_feature_names) != hvg_order
    ):
        raise ValueError("assembled response target gene order differs from STATE")
    metadata = bags.metadata.reset_index(drop=True)
    genes_by_line = {
        str(key): tuple(frame.perturbation_gene)
        for key, frame in metadata.groupby("model_id", sort=False)
    }
    anchors = tuple(genes_by_line)
    if len(anchors) != 4 or not set(anchors).issubset(split.supervised_train):
        raise ValueError("response sources must contain four labeled training anchors")
    holdout = split_heldout_genes(
        genes_by_line,
        fraction=settings["response_holdout_fraction"],
        seed=settings["response_holdout_seed"],
    )
    union = write_embedding_union(
        scored_symbols=candidates.symbols,
        response_symbols=tuple(sorted(set(metadata.perturbation_gene))),
        esm2_path=Path(paths["esm2_embeddings"]),
        output_dir=root,
    )
    panel = tuple(union["common_gene_panel"])
    panel_path = Path(paths["common_gene_panel"])
    panel_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"gene_symbol": panel}).to_csv(panel_path, index=False)
    resolved = set(union["esm2_order"])
    keep = [i for i, gene in enumerate(metadata.perturbation_gene) if gene in resolved]
    conditions = [
        {"model_id": str(row.model_id), "gene": str(row.perturbation_gene)}
        for row in metadata.itertuples()
    ]
    selected = [conditions[i] for i in keep]
    surviving_holdout = [
        row for row in selected if row["gene"] in holdout[row["model_id"]]
    ]
    for model_id in anchors:
        held = {row["gene"] for row in surviving_holdout if row["model_id"] == model_id}
        train = {row["gene"] for row in selected if row["model_id"] == model_id} - held
        if not held or not train:
            raise ValueError(
                f"{model_id}: ESM2 filtering emptied response train or fixed holdout"
            )
    write_response_targets_cache(
        Path(paths["response_cache"]),
        genes=[bags.genes[i] for i in keep],
        target_bags=[bags.effective_target_bags[i] for i in keep],
        metadata=metadata.iloc[keep].reset_index(drop=True),
        hvg_order=tuple(bags.target_feature_names),
    )
    build_q_sc_shards(registry, Path(paths["q_sc_cache"]), panel, resume=True)
    payload = {
        "schema_version": "geneeffect-joint-prepared-v1",
        "split": {
            name: list(getattr(split, name))
            for name in ("train", "val", "test", "unlabeled_train")
        },
        "common_gene_panel": list(panel),
        "hvg_order": list(hvg_order),
        "esm2_order": union["esm2_order"],
        "response_anchors": list(anchors),
        "response_conditions": selected,
        "response_holdout": surviving_holdout,
        "excluded_response_conditions": [
            dict(row, reason="unresolved_esm2")
            for row in conditions
            if row["gene"] not in resolved
        ],
        "excluded_dependency_genes": [
            *candidates.dropped,
            *[
                {"gene_symbol": gene, "reasons": ["unresolved_esm2"]}
                for gene in candidates.symbols
                if gene not in resolved
            ],
        ],
        "preparation": dict(settings),
        "inputs": dict(paths),
        "newly_encoded_tx1_lines": encoded,
        "layouts": {
            "tx1": "per ModelID embeddings.npy/hvg.npy/obs.parquet",
            "q_sc": "per ModelID .npz",
            "response": "response_targets arrays + metadata.parquet + ordered manifest",
        },
    }
    destination = root / "prepared_inputs.json"
    temporary = root / "prepared_inputs.json.tmp"
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, destination)
    # Exercise the actual readers, including test-cache shape/order boundaries.
    load_inputs(config, include_test=True)
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args(argv)
    from src.experiments.config import load_config

    print(prepare_inputs(load_config(args.config)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
