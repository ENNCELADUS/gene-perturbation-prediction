"""Prepare fixed joint-training inputs once, in STATE's log expression space.

Tx1 reads raw UMI. Every other expression quantity is ``log1p(x * T / L_cell)``
then STATE's HVG slice, with ``L_cell`` the cell's UMI total over every gene
of its source matrix and ``T`` the median ``L_cell`` of the non-targeting
cells in the Nadig 2025 Jurkat and HepG2 sources (data STATE's Replogle
checkpoint was trained on).

The lines are the split's (``paths.split``: the 226 and the built single-cell
extras); each line's raw-UMI cells are ``<paths.basal_raw_umi>/<ModelID>.h5ad``.
Once ``prepared_inputs.json`` exists, a run prepares only the lines without
``lines/<ModelID>.npz``, in the recorded space (``T``, panel, HVG order); every
other file stays as written.

The two ``T`` sources and every response anchor are read at once, one process
each; each anchor process writes its reduced cells to a temporary part file
that this process turns into log-space bags once ``T`` is known. The basal
lines then run on a process pool. Every result is what one process computing
them in turn would write.
"""

from __future__ import annotations

import argparse
import json
import logging
import logging.handlers
import multiprocessing
import os
import shutil
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any, Final
from src.data.io import write_json

_LOGGER = logging.getLogger(__name__)

#: Jurkat and HepG2: the response sources whose controls define ``T``.
TARGET_SUM_SOURCES: Final[tuple[str, str]] = ("ACH-000995", "ACH-000739")
#: K562 supplies the copy-prior baseline, so scored genes need its label.
COPY_PRIOR_DONOR: Final[str] = "ACH-000551"
#: Processes reading the ``T`` sources and the response anchors, all at once
#: (two ``T`` sources plus four anchors in the joint config).
RESPONSE_PROCESSES: Final[int] = 8
#: Processes preparing basal lines; each holds one line's source in memory.
LINE_PROCESSES: Final[int] = 16


def _forward_logs_to(queue, level: int) -> None:
    """Worker initializer: send every log record to the parent's handlers."""
    root = logging.getLogger()
    root.handlers[:] = [logging.handlers.QueueHandler(queue)]
    root.setLevel(level)


def in_processes(calls: Sequence[Callable[[], Any]], processes: int) -> Iterator[Any]:
    """Each call's result, in order, from up to ``processes`` spawned processes.

    One process (or one call) runs them here, in turn. Worker log records go
    through this process's logging handlers.
    """
    if processes <= 1 or len(calls) <= 1:
        for call in calls:
            yield call()
        return
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    root = logging.getLogger()
    handlers = root.handlers or [logging.lastResort]
    listener = logging.handlers.QueueListener(
        queue, *handlers, respect_handler_level=True
    )
    listener.start()
    pool = ProcessPoolExecutor(
        max_workers=min(processes, len(calls)),
        mp_context=context,
        initializer=_forward_logs_to,
        initargs=(queue, root.getEffectiveLevel()),
    )
    finished = False
    try:
        futures = [pool.submit(call) for call in calls]
        for future in futures:
            yield future.result()
        finished = True
    finally:
        if not finished:
            # Interrupted or failed: stop running workers now instead of
            # waiting for their whole source read to finish.
            for process in list(pool._processes.values()):
                process.terminate()
        pool.shutdown(wait=True, cancel_futures=True)
        listener.stop()


def _prepare_basal_line(
    model_id: str,
    source_path: Path,
    output_path: Path,
    *,
    tx1_cache: Path,
    var_ensembl_col: str,
    hvg_order: tuple[str, ...],
    genes: tuple[str, ...],
    target_sum: float,
    cells_per_context: int,
) -> None:
    """Write ``lines/<ModelID>.npz`` of one basal line."""
    from src.data.basal import symbol_column
    from src.data.prepared import prepare_line, write_prepared_line
    from src.data.tx1_cache import load_line_cache, read_basal_source

    source = read_basal_source(
        source_path, model_id=model_id, var_ensembl_col=var_ensembl_col
    )
    embeddings, _, obs = load_line_cache(tx1_cache, model_id)
    column = symbol_column(source.var)
    line = prepare_line(
        source.X,
        source.obs_names.astype(str),
        source.var[column].astype(str).tolist(),
        embeddings,
        obs.index.astype(str).tolist(),
        model_id=model_id,
        hvg_order=hvg_order,
        genes=genes,
        target_sum=target_sum,
        cells_per_context=cells_per_context,
    )
    write_prepared_line(output_path, line)


def basal_sources(config: Mapping[str, Any]) -> dict[str, Path]:
    """ModelID to raw-UMI h5ad of every line of the split. Raises, listing them,
    if any file is missing."""
    from src.data.splits import load_geneeffect_split

    paths = config["paths"]
    split = load_geneeffect_split(Path(paths["split"]))
    directory = Path(paths["basal_raw_umi"])
    sources = {
        model_id: directory / f"{model_id}.h5ad" for model_id in split.all_model_ids
    }
    missing = sorted(
        model_id for model_id, path in sources.items() if not path.is_file()
    )
    if missing:
        raise FileNotFoundError(
            f"{directory}: no raw-UMI h5ad for {len(missing)} lines: {missing[:10]}"
        )
    return sources


def _encode_missing_tx1(
    config: Mapping[str, Any], sources: Mapping[str, Path]
) -> list[str]:
    from src.data.tx1_cache import encode_lines, load_hvg_gene_order, missing_lines

    paths, settings = config["paths"], config["preparation"]
    cache = Path(paths["tx1_cache"])
    missing = missing_lines(cache, list(sources))
    if not missing:
        return []
    from src.model.tx1 import build_tx1_encoder

    _LOGGER.info("Encoding %d lines missing from the Tx1 cache", len(missing))
    encoder, _ = build_tx1_encoder(
        Path(paths["tx1_model_dir"]),
        settings["tx1_batch_size"],
        settings["tx1_max_length"],
    )
    return encode_lines(
        {model_id: sources[model_id] for model_id in missing},
        cache,
        encoder=encoder,
        hvg_order=tuple(load_hvg_gene_order(Path(paths["state_model_dir"]))),
        var_ensembl_col=settings["var_ensembl_col"],
        max_cells_per_line=config["features"]["cells_per_context"],
        seed=0,
    )


#: Paths that say which lines exist, not how a line is prepared: the prior
#: export is read by training and evaluation only, and the split grows the root
#: (its lines are recorded in the manifest; the gene panel of an existing root is
#: the recorded one). A recorded value of either is ignored too.
_LINE_SET_PATHS: Final[frozenset[str]] = frozenset({"prior", "split"})


def preparation_settings(config: Mapping[str, Any]) -> dict[str, Any]:
    """Every config value that changes how preparation writes a line."""
    return {
        "preparation": dict(config["preparation"]),
        "cells_per_context": config["features"]["cells_per_context"],
        "paths": {
            key: value
            for key, value in config["paths"].items()
            if key not in _LINE_SET_PATHS
        },
    }


def _prepare_lines(
    config: Mapping[str, Any],
    root: Path,
    sources: Mapping[str, Path],
    *,
    genes: tuple[str, ...],
    hvg_order: tuple[str, ...],
    target_sum: float,
) -> None:
    """Tx1-encode the uncached lines of ``sources``, then write their npz."""
    paths, settings = config["paths"], config["preparation"]
    encoded = _encode_missing_tx1(config, sources)
    _LOGGER.info("Tx1 cache ready; %d lines newly encoded", len(encoded))
    line_calls = [
        partial(
            _prepare_basal_line,
            model_id,
            path,
            root / "lines" / f"{model_id}.npz",
            tx1_cache=Path(paths["tx1_cache"]),
            var_ensembl_col=settings["var_ensembl_col"],
            hvg_order=hvg_order,
            genes=genes,
            target_sum=target_sum,
            cells_per_context=config["features"]["cells_per_context"],
        )
        for model_id, path in sources.items()
    ]
    for done, _ in enumerate(in_processes(line_calls, LINE_PROCESSES), start=1):
        if done % 25 == 0:
            _LOGGER.info("Prepared basal line %d of %d", done, len(line_calls))
    _LOGGER.info("Prepared %d basal lines", len(line_calls))


def prepare_inputs(config: Mapping[str, Any]) -> Path:
    """Write the prepared root, or grow it by its missing lines; return its
    manifest path (written last)."""
    from src.experiments.joint.config import validate_config

    validate_config(config)
    root = Path(config["prepared_root"])
    from src.data.prepared import PREPARED_METADATA_FILENAME

    manifest_path = root / PREPARED_METADATA_FILENAME
    produced_by = preparation_settings(config)
    sources = basal_sources(config)
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        recorded = manifest.get("settings")
        if isinstance(recorded, dict) and isinstance(recorded.get("paths"), dict):
            paths = recorded["paths"]
            recorded = {
                **recorded,
                "paths": {k: v for k, v in paths.items() if k not in _LINE_SET_PATHS},
            }
        if recorded != produced_by:
            raise ValueError(
                f"{root} was prepared with different settings "
                f"({manifest.get('settings')}); choose a new prepared_root for "
                "these settings"
            )
        missing = {
            model_id: path
            for model_id, path in sources.items()
            if not (root / "lines" / f"{model_id}.npz").is_file()
        }
        if missing:
            _LOGGER.info("Growing %s by %d lines", root, len(missing))
            _prepare_lines(
                config,
                root,
                missing,
                genes=tuple(manifest["common_gene_panel"]),
                hvg_order=tuple(manifest["hvg_order"]),
                target_sum=float(manifest["expression_space"]["target_sum"]),
            )
        if manifest.get("lines") != list(sources):
            write_json(manifest_path, {**manifest, "lines": list(sources)})
        return manifest_path

    import numpy as np

    from src.data.expression import median_library_size
    from src.data.labels import load_geneeffect_long
    from src.data.build.build_exp13_esm2_universe import (
        build_coverage_universe,
        restrict_coverage_universe_to_copy_prior,
        write_embedding_union,
    )
    from src.data.prepared import EXPRESSION_TRANSFORM
    from src.data.response import (
        control_library_sizes,
        load_response_sources,
        part_conditions,
        response_bags,
        write_response_part,
    )
    from src.data.response_cache import write_response_targets
    from src.data.splits import assert_fit_eligible, load_geneeffect_split
    from src.data.tx1_cache import load_hvg_gene_order

    paths, settings = config["paths"], config["preparation"]
    split = load_geneeffect_split(Path(paths["split"]))
    labels = load_geneeffect_long(Path(paths["gene_effect"]), split)
    assert_fit_eligible(COPY_PRIOR_DONOR, split)
    donor = labels.loc[
        (labels.model_id == COPY_PRIOR_DONOR) & np.isfinite(labels.gene_effect),
        "gene_symbol",
    ]
    candidates = restrict_coverage_universe_to_copy_prior(
        build_coverage_universe(labels, split), tuple(donor)
    )
    hvg_order = tuple(load_hvg_gene_order(Path(paths["state_model_dir"])))

    responses = load_response_sources(Path(paths["perturbseq_sources"]))
    for anchor in responses:
        assert_fit_eligible(anchor, split)
    anchors = sorted(responses)
    part_dir = root / f".tmp-response-parts-{uuid.uuid4().hex}"
    part_dir.mkdir(parents=True)
    try:
        calls = [
            partial(control_library_sizes, responses[model_id], model_id)
            for model_id in TARGET_SUM_SOURCES
        ] + [
            partial(
                write_response_part,
                part_dir / f"{model_id}.npz",
                responses[model_id],
                model_id,
                hvg_order,
                max_cells_per_gene=settings["response_max_cells_per_gene"],
                seed=settings["response_sampling_seed"],
            )
            for model_id in anchors
        ]
        results = list(in_processes(calls, RESPONSE_PROCESSES))
        target_sum = median_library_size(*results[: len(TARGET_SUM_SOURCES)])
        _LOGGER.info("Expression target sum T = %.1f", target_sum)
        parts = dict(zip(anchors, results[len(TARGET_SUM_SOURCES) :], strict=True))

        conditions = [
            (model_id, gene, n_cells)
            for model_id in anchors
            for gene, n_cells in part_conditions(parts[model_id])
        ]
        union = write_embedding_union(
            scored_symbols=candidates.symbols,
            response_symbols=tuple(sorted({gene for _, gene, _ in conditions})),
            esm2_path=Path(paths["esm2_embeddings"]),
            output_dir=root,
        )
        resolved = set(union["esm2_order"])
        kept = [condition for condition in conditions if condition[1] in resolved]
        if not kept:
            raise ValueError("no response condition resolves in the ESM2 table")
        _LOGGER.info("Writing %d of %d response conditions", len(kept), len(conditions))
        keys = [(model_id, gene) for model_id, gene, _ in kept]
        write_response_targets(
            root / "response",
            keys,
            [n_cells for _, _, n_cells in kept],
            len(hvg_order),
            response_bags(parts, keys, target_sum),
        )
    finally:
        shutil.rmtree(part_dir, ignore_errors=True)

    genes = tuple(union["common_gene_panel"])
    _prepare_lines(
        config, root, sources, genes=genes, hvg_order=hvg_order, target_sum=target_sum
    )

    manifest = {
        "expression_space": {
            "transform": EXPRESSION_TRANSFORM,
            "target_sum": target_sum,
            "library_size": "all_genes",
            "target_sum_sources": list(TARGET_SUM_SOURCES),
        },
        "common_gene_panel": list(genes),
        "hvg_order": list(hvg_order),
        "response_anchors": anchors,
        "settings": produced_by,
        "lines": list(sources),
    }
    write_json(manifest_path, manifest)
    return manifest_path


def _pseudobulk_line(
    model_id: str,
    source_path: Path,
    *,
    var_ensembl_col: str,
    genes: tuple[str, ...],
):
    from src.data.basal import symbol_column
    from src.data.pseudobulk import pseudobulk
    from src.data.tx1_cache import read_basal_source

    source = read_basal_source(
        source_path, model_id=model_id, var_ensembl_col=var_ensembl_col
    )
    column = symbol_column(source.var)
    return pseudobulk(source.X, source.var[column].astype(str).tolist(), genes)


def prepare_pseudobulk(config: Mapping[str, Any], genes: Sequence[str]) -> Path:
    """Write ``<prepared_root>/pseudobulk/`` for every line preparation covers,
    or append the rows of lines it lacks; existing rows stay as written.

    Reads each line's raw source (preparation stays the only reader of raw data);
    raises when the existing pseudo-bulk is for other genes.
    """
    import numpy as np
    import pandas as pd

    from src.data.pseudobulk import PSEUDOBULK_DIR, TRANSFORM, read_pseudobulk

    prepared = Path(config["prepared_root"])
    root = prepared / PSEUDOBULK_DIR
    manifest_path = root / "manifest.json"
    sources = basal_sources(config)
    if manifest_path.is_file():
        if json.loads(manifest_path.read_text())["genes"] != list(genes):
            raise ValueError(f"{root} holds pseudo-bulk for other genes")
        frame = read_pseudobulk(prepared)
    else:
        frame = pd.DataFrame(columns=list(genes), dtype=np.float32)
    missing = [model_id for model_id in sources if model_id not in frame.index]
    if not missing:
        return manifest_path
    settings = config["preparation"]
    calls = [
        partial(
            _pseudobulk_line,
            model_id,
            sources[model_id],
            var_ensembl_col=settings["var_ensembl_col"],
            genes=tuple(genes),
        )
        for model_id in missing
    ]
    added = pd.DataFrame(
        np.stack(list(in_processes(calls, LINE_PROCESSES))),
        index=missing,
        columns=list(genes),
    )
    frame = pd.concat([frame, added]) if len(frame) else added
    frame.index.name = "model_id"
    root.mkdir(parents=True, exist_ok=True)
    temporary = root / "pseudobulk.parquet.tmp"
    frame.to_parquet(temporary)
    os.replace(temporary, root / "pseudobulk.parquet")
    write_json(
        manifest_path,
        {"transform": TRANSFORM, "genes": list(genes), "lines": list(frame.index)},
    )
    return manifest_path


NOISE_DIR: Final[str] = "noise"


def _noise_line(
    model_id: str,
    source_path: Path,
    programs_path: Path,
    *,
    var_ensembl_col: str,
    genes: tuple[str, ...],
    cells: tuple[int, ...],
    programs: Mapping[str, tuple[str, ...]],
    target_sum: float,
):
    """One line's draw pseudo-bulks (draw name to row) and the shares of the
    scoring set and of each program its source lists; writes its per-cell program
    scores to ``programs_path``."""
    import numpy as np

    from src.data.basal import symbol_column
    from src.data.noise_inputs import cell_draws, program_scores
    from src.data.pseudobulk import pseudobulk
    from src.data.tx1_cache import read_basal_source

    source = read_basal_source(
        source_path, model_id=model_id, var_ensembl_col=var_ensembl_col
    )
    symbols = source.var[symbol_column(source.var)]
    symbols = symbols.astype(str).tolist()
    draws = {
        name: pseudobulk(source.X[rows], symbols, genes)
        for name, rows in cell_draws(
            model_id, source.obs_names.astype(str), cells
        ).items()
    }
    scores, listed = program_scores(source.X, symbols, genes, programs, target_sum)
    temporary = programs_path.with_name(f".{programs_path.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, scores=scores)
    os.replace(temporary, programs_path)
    return draws, listed


def prepare_noise_inputs(
    config: Mapping[str, Any],
    genes: Sequence[str],
    *,
    cells: Sequence[int],
    programs: Mapping[str, Sequence[str]],
) -> Path:
    """Write ``<prepared_root>/noise/`` for every line of the split: cell-draw
    pseudo-bulk (``subsamples.parquet``, rows ``(model_id, draw)``, the prepared
    pseudo-bulk transform over ``genes``) and per-cell program scores over the same
    genes (``programs/<ModelID>.npz``, every basal cell, in ``programs`` order),
    then ``manifest.json`` with the share of the scoring set and of each program
    every line's source lists (``listed``); return the manifest path.

    A root written with the same settings is reused; one written with other
    settings is refused. Reads raw data (preparation is its only reader).
    """
    import numpy as np
    import pandas as pd

    from src.data.prepared import read_manifest

    prepared = Path(config["prepared_root"])
    root = prepared / NOISE_DIR
    manifest_path = root / "manifest.json"
    prepared_manifest = read_manifest(prepared)
    settings = {
        "genes": list(genes),
        "cells": sorted(int(n) for n in cells),
        "programs": {name: list(members) for name, members in programs.items()},
        "target_sum": float(prepared_manifest["expression_space"]["target_sum"]),
    }
    sources = basal_sources(config)
    if manifest_path.is_file():
        recorded = json.loads(manifest_path.read_text())
        if recorded["settings"] != settings:
            raise ValueError(f"{root} was prepared with other settings")
        if recorded["lines"] == list(sources):
            return manifest_path
        raise ValueError(
            f"{root} was prepared for other lines; move it aside and rerun"
        )
    (root / "programs").mkdir(parents=True, exist_ok=True)
    options = config["preparation"]
    calls = [
        partial(
            _noise_line,
            model_id,
            path,
            root / "programs" / f"{model_id}.npz",
            var_ensembl_col=options["var_ensembl_col"],
            genes=tuple(genes),
            cells=tuple(settings["cells"]),
            programs={name: tuple(m) for name, m in settings["programs"].items()},
            target_sum=settings["target_sum"],
        )
        for model_id, path in sources.items()
    ]
    rows, index, listed = [], [], {}
    for model_id, (draws, shares) in zip(
        sources, in_processes(calls, LINE_PROCESSES), strict=True
    ):
        listed[model_id] = shares
        for name, values in draws.items():
            rows.append(values)
            index.append((model_id, name))
    frame = pd.DataFrame(
        np.stack(rows),
        index=pd.MultiIndex.from_tuples(index, names=["model_id", "draw"]),
        columns=list(genes),
    )
    temporary = root / "subsamples.parquet.tmp"
    frame.to_parquet(temporary)
    os.replace(temporary, root / "subsamples.parquet")
    scoring = {str(gene).upper() for gene in genes}
    write_json(
        manifest_path,
        {
            "settings": settings,
            "lines": list(sources),
            "members_in_genes": {
                name: len({str(g).upper() for g in members} & scoring)
                for name, members in settings["programs"].items()
            },
            "listed": listed,
        },
    )
    return manifest_path


def prepare_second_source(
    config: Mapping[str, Any], genes: Sequence[str], second_root: Path
) -> Path:
    """Write ``<prepared_root>/noise/second_source/``: pseudo-bulk, in the
    prepared transform over ``genes``, of every split line with a second-deposit
    h5ad ``<second_root>/<ModelID>.h5ad``; rows of lines already written stay as
    written and new files add rows. Refuses a root for other genes or another
    second-deposit directory."""
    import numpy as np
    import pandas as pd

    from src.data.splits import load_geneeffect_split

    root = Path(config["prepared_root"]) / NOISE_DIR / "second_source"
    manifest_path = root / "manifest.json"
    settings = {"genes": list(genes), "second_root": str(second_root)}
    split = load_geneeffect_split(Path(config["paths"]["split"]))
    available = {
        model_id: Path(second_root) / f"{model_id}.h5ad"
        for model_id in split.all_model_ids
        if (Path(second_root) / f"{model_id}.h5ad").is_file()
    }
    if manifest_path.is_file():
        recorded = json.loads(manifest_path.read_text())
        if recorded["settings"] != settings:
            raise ValueError(f"{root} holds another second-deposit pseudo-bulk")
        frame = pd.read_parquet(root / "pseudobulk.parquet")
    else:
        frame = pd.DataFrame(columns=list(genes), dtype=np.float32)
    missing = [m for m in available if m not in frame.index]
    if not missing and manifest_path.is_file():
        return manifest_path
    options = config["preparation"]
    calls = [
        partial(
            _pseudobulk_line,
            model_id,
            available[model_id],
            var_ensembl_col=options["var_ensembl_col"],
            genes=tuple(genes),
        )
        for model_id in missing
    ]
    if calls:
        added = pd.DataFrame(
            np.stack(list(in_processes(calls, LINE_PROCESSES))),
            index=missing,
            columns=list(genes),
        )
        frame = pd.concat([frame, added]) if len(frame) else added
    frame.index.name = "model_id"
    root.mkdir(parents=True, exist_ok=True)
    temporary = root / "pseudobulk.parquet.tmp"
    frame.to_parquet(temporary)
    os.replace(temporary, root / "pseudobulk.parquet")
    write_json(manifest_path, {"settings": settings, "lines": list(frame.index)})
    return manifest_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument(
        "--pseudobulk-genes",
        type=Path,
        help="a DepMap bulk matrix; also grow the pseudo-bulk over its genes",
    )
    args = parser.parse_args(argv)
    from src.experiments.joint.config import load_config

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    config = load_config(args.config)
    print(prepare_inputs(config))
    if args.pseudobulk_genes is not None:
        from src.data.depmap import read_depmap_matrix

        genes = list(read_depmap_matrix(args.pseudobulk_genes).columns)
        print(prepare_pseudobulk(config, genes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
