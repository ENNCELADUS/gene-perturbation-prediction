"""Prepare fixed joint-training inputs once, in STATE's log expression space.

Tx1 reads raw UMI. Every other expression quantity is ``log1p(x * T / L_cell)``
then STATE's HVG slice, with ``L_cell`` the cell's UMI total over every gene
of its source matrix and ``T`` the median ``L_cell`` of the non-targeting
cells in the Nadig 2025 Jurkat and HepG2 sources (data STATE's Replogle
checkpoint was trained on). Skipped when ``prepared_inputs.json`` exists.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

_LOGGER = logging.getLogger(__name__)

#: Jurkat and HepG2: the response sources whose controls define ``T``.
TARGET_SUM_SOURCES: Final[tuple[str, str]] = ("ACH-000995", "ACH-000739")
#: K562 supplies the copy-prior baseline, so scored genes need its label.
COPY_PRIOR_DONOR: Final[str] = "ACH-000551"


def _encode_missing_tx1(config: Mapping[str, Any], registry) -> list[str]:
    from src.data.tx1_cache import encode_lines, load_hvg_gene_order, missing_lines

    paths, settings = config["paths"], config["preparation"]
    cache = Path(paths["tx1_cache"])
    missing = missing_lines(cache, registry.index)
    if not missing:
        return []
    from src.model.tx1 import _build_tx1_encoder

    _LOGGER.info("Encoding %d lines missing from the Tx1 cache", len(missing))
    encoder, _ = _build_tx1_encoder(
        Path(paths["tx1_model_dir"]),
        settings["tx1_batch_size"],
        settings["tx1_max_length"],
    )
    return encode_lines(
        registry.loc[missing],
        cache,
        encoder=encoder,
        hvg_order=tuple(load_hvg_gene_order(Path(paths["state_model_dir"]))),
        var_ensembl_col=settings["var_ensembl_col"],
        hvg_gene_symbol_col=settings["hvg_gene_symbol_col"],
        max_cells_per_line=config["features"]["cells_per_context"],
        seed=0,
    )


def preparation_settings(config: Mapping[str, Any]) -> dict[str, Any]:
    """Every config value that changes what preparation writes."""
    return {
        "preparation": dict(config["preparation"]),
        "cells_per_context": config["features"]["cells_per_context"],
        "paths": dict(config["paths"]),
    }


def prepare_inputs(config: Mapping[str, Any]) -> Path:
    """Write the prepared root; return its manifest path (written last)."""
    from src.experiments.config import validate_config

    validate_config(config)
    root = Path(config["prepared_root"])
    from src.data.prepared import PREPARED_METADATA_FILENAME

    manifest_path = root / PREPARED_METADATA_FILENAME
    produced_by = preparation_settings(config)
    if manifest_path.is_file():
        recorded = json.loads(manifest_path.read_text()).get("settings")
        if recorded != produced_by:
            raise ValueError(
                f"{root} was prepared with different settings ({recorded}); "
                "choose a new prepared_root for these settings"
            )
        _LOGGER.info("Prepared inputs already exist at %s", root)
        return manifest_path

    import numpy as np

    from src.data.basal import symbol_column
    from src.data.expression import median_library_size
    from src.data.geneeffect import load_geneeffect_long, load_source_registry
    from src.data.prepare.build_exp13_esm2_universe import (
        build_coverage_universe,
        restrict_coverage_universe_to_copy_prior,
        write_embedding_union,
    )
    from src.data.prepared import EXPRESSION_TRANSFORM, prepare_line
    from src.data.prepared import write_prepared_line
    from src.data.response import (
        build_response_targets,
        control_library_sizes,
        load_response_sources,
    )
    from src.data.response_cache import write_response_targets
    from src.data.splits import assert_fit_eligible, load_geneeffect_226_split
    from src.data.tx1_cache import (
        load_hvg_gene_order,
        load_line_cache,
        read_registry_source,
    )

    paths, settings = config["paths"], config["preparation"]
    cells_per_context = config["features"]["cells_per_context"]
    split = load_geneeffect_226_split(Path(paths["split"]))
    labels = load_geneeffect_long(Path(paths["gene_effect"]), split)
    assert_fit_eligible(COPY_PRIOR_DONOR, split)
    donor = labels.loc[
        (labels.model_id == COPY_PRIOR_DONOR) & np.isfinite(labels.gene_effect),
        "gene_symbol",
    ]
    candidates = restrict_coverage_universe_to_copy_prior(
        build_coverage_universe(labels, split), tuple(donor)
    )
    registry = load_source_registry(Path(paths["source_registry"]), split)
    hvg_order = tuple(load_hvg_gene_order(Path(paths["state_model_dir"])))
    encoded = _encode_missing_tx1(config, registry)
    _LOGGER.info("Tx1 cache ready; %d lines newly encoded", len(encoded))

    sources = load_response_sources(Path(paths["perturbseq_sources"]))
    for anchor in sources:
        assert_fit_eligible(anchor, split)
    target_sum = median_library_size(
        *(
            control_library_sizes(sources[model_id], model_id)
            for model_id in TARGET_SUM_SOURCES
        )
    )
    _LOGGER.info("Expression target sum T = %.1f", target_sum)

    keys, bags = build_response_targets(
        sources,
        hvg_order,
        target_sum,
        max_cells_per_gene=settings["response_max_cells_per_gene"],
        total_cells_per_line=settings["response_total_cells_per_line"],
        seed=settings["response_sampling_seed"],
    )
    union = write_embedding_union(
        scored_symbols=candidates.symbols,
        response_symbols=tuple(sorted({gene for _, gene in keys})),
        esm2_path=Path(paths["esm2_embeddings"]),
        output_dir=root,
    )
    resolved = set(union["esm2_order"])
    keep = [index for index, (_, gene) in enumerate(keys) if gene in resolved]
    if not keep:
        raise ValueError("no response condition resolves in the ESM2 table")
    _LOGGER.info("Writing %d of %d response conditions", len(keep), len(keys))
    write_response_targets(
        root / "response", [keys[i] for i in keep], [bags[i] for i in keep]
    )
    del bags

    genes = tuple(union["common_gene_panel"])
    for model_id, row in registry.iterrows():
        source = read_registry_source(
            Path(row["source_path"]),
            model_id=str(model_id),
            var_ensembl_col=settings["var_ensembl_col"],
        )
        embeddings, _, obs = load_line_cache(Path(paths["tx1_cache"]), str(model_id))
        column = symbol_column(source.var, settings["hvg_gene_symbol_col"])
        line = prepare_line(
            source.X,
            source.obs_names.astype(str),
            source.var[column].astype(str).tolist(),
            embeddings,
            obs.index.astype(str).tolist(),
            model_id=str(model_id),
            hvg_order=hvg_order,
            genes=genes,
            target_sum=target_sum,
            cells_per_context=cells_per_context,
        )
        write_prepared_line(root / "lines" / f"{model_id}.npz", line)
    _LOGGER.info("Prepared %d basal lines", len(registry))

    manifest = {
        "expression_space": {
            "transform": EXPRESSION_TRANSFORM,
            "target_sum": target_sum,
            "library_size": "all_genes",
            "target_sum_sources": list(TARGET_SUM_SOURCES),
        },
        "common_gene_panel": list(genes),
        "hvg_order": list(hvg_order),
        "response_anchors": sorted(sources),
        "settings": produced_by,
    }
    temporary = manifest_path.with_name(manifest_path.name + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, manifest_path)
    return manifest_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args(argv)
    from src.experiments.config import load_config

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    print(prepare_inputs(load_config(args.config)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
