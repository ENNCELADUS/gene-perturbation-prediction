"""Naming each called cell's line in a deposited library: the knee cell call, hashtag
and RT-barcode demultiplexing, pools matched to DepMap bulk RNA, and the identity
kinds a source spec may name."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from src.data.basal import align_columns
from src.data.builders.single_cell_readers import Library, read_library, require
from src.data.depmap import read_depmap_matrix

IDENTITIES = {
    "line": ("identifier",),
    "labels": (
        "path",
        "sep",
        "cell_column",
        "label_column",
        "arm_column",
        "keep",
        "cell_prefix",
        "cell_suffix",
        "skiprows",
    ),
    "barcode": ("pattern", "identifier", "rename"),
    "hashtag": ("counts", "cell_suffix", "hashtags", "min_count", "min_ratio"),
    "pool": ("members", "clusters", "min_margin"),
    "rt_hash": ("rt_barcodes", "rt_map", "hash_table", "min_count", "min_ratio"),
}


def knee(totals: np.ndarray, min_umi: int) -> float:
    """UMI total at the inflection of the log-log barcode-rank curve: the left end
    of its steepest segment over distinct totals of at least ``min_umi`` (ignoring
    the first ten ranks). Barcodes at or above it are cells."""
    values = totals[totals >= min_umi]
    unique, counts = np.unique(values, return_counts=True)
    unique, counts = unique[::-1].astype(np.float64), counts[::-1]
    last_rank = np.cumsum(counts)
    allowed = last_rank[:-1] >= 10
    if not allowed.any():
        return float(min_umi)
    mid_rank = last_rank - (counts - 1) / 2
    slope = np.diff(np.log10(unique)) / np.diff(np.log10(mid_rank))
    step = np.flatnonzero(allowed)[np.argmin(slope[allowed])]
    return float(unique[step])


def call_hashtags(
    counts: np.ndarray,
    names: Sequence[str],
    hashtags: Mapping[str, str],
    *,
    min_count: int,
    min_ratio: float,
) -> tuple[list[str | None], dict[str, int]]:
    """Each cell's identifier from its hashtag counts: a singlet's top tag
    (at least ``min_count`` and ``min_ratio`` times the second); negatives and
    doublets get None."""
    columns = [list(names).index(tag) for tag in hashtags]
    values = np.asarray(counts)[:, columns].astype(np.float64)
    order = np.sort(values, axis=1)
    top = order[:, -1]
    second = order[:, -2] if values.shape[1] > 1 else np.zeros(len(values))
    best = np.argmax(values, axis=1)
    identifiers = list(hashtags.values())
    calls, tally = [], Counter()
    for index in range(len(values)):
        if top[index] < min_count:
            calls.append(None)
            tally["negative"] += 1
        elif top[index] >= min_ratio * second[index] and top[index] > second[index]:
            calls.append(identifiers[best[index]])
            tally["singlet"] += 1
        else:
            calls.append(None)
            tally["doublet"] += 1
    return calls, dict(tally)


def assign_pool(
    counts: sparse.csr_matrix,
    symbols: Sequence[str],
    members: Mapping[str, str],
    bulk: pd.DataFrame,
    *,
    clusters: int,
    min_margin: float,
) -> tuple[list[str | None], list[dict[str, Any]]]:
    """Each cell's identifier in a pool without labels: cluster the cells, then
    give each cluster the member (identifier to ModelID) whose DepMap bulk RNA
    deviation best correlates with the cluster's, when it beats the second best
    by ``min_margin``; other clusters' cells get None."""
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    missing = sorted(set(members.values()) - set(bulk.index))
    if missing:
        raise ValueError(f"pool members without DepMap bulk RNA: {missing}")
    genes = [g for g in dict.fromkeys(s.upper() for s in symbols) if g in bulk.columns]
    aligned, _ = align_columns(counts, [s.upper() for s in symbols], genes)
    library = np.asarray(counts.sum(axis=1), dtype=np.float64).ravel()
    logged = sparse.diags(1e4 / np.maximum(library, 1)) @ aligned
    logged.data = np.log1p(logged.data)
    variance = np.asarray(logged.power(2).mean(axis=0)).ravel() - np.square(
        np.asarray(logged.mean(axis=0)).ravel()
    )
    top = np.argsort(variance)[::-1][:2000]
    reduced = PCA(
        n_components=min(30, len(top), counts.shape[0] - 1), random_state=0
    ).fit_transform(logged[:, top].toarray())
    labels = KMeans(n_clusters=clusters, n_init=10, random_state=0).fit_predict(reduced)
    profile = bulk.loc[list(members.values()), genes]
    markers = profile.var(axis=0).sort_values(ascending=False).index[:1000]
    position = [genes.index(g) for g in markers]
    member_dev = profile[markers] - profile[markers].mean(axis=0)
    means = np.vstack(
        [
            np.asarray(logged[labels == k][:, position].mean(axis=0)).ravel()
            for k in range(clusters)
        ]
    )
    cluster_dev = means - means.mean(axis=0)
    identifiers = list(members)
    table, assigned = [], {}
    for k in range(clusters):
        r = [float(np.corrcoef(cluster_dev[k], row)[0, 1]) for row in member_dev.values]
        ranked = np.argsort(r)[::-1]
        margin = r[ranked[0]] - (r[ranked[1]] if len(r) > 1 else -1.0)
        chosen = identifiers[ranked[0]] if margin >= min_margin else None
        assigned[k] = chosen
        table.append(
            {
                "cluster": k,
                "cells": int((labels == k).sum()),
                "r": dict(zip(identifiers, r, strict=True)),
                "margin": margin,
                "assigned": chosen,
            }
        )
    return [assigned[k] for k in labels], table


def identify(
    library: Library,
    identity: Mapping[str, Any],
    root: Path,
    members: Mapping[str, str],
    bulk_path: Path,
) -> tuple[list[str | None], list[str | None], dict[str, Any]]:
    """Per-cell identifier and arm (None where the identity rule names no line or
    no arm), and the rule's report."""
    kind = identity["kind"]
    require(identity, IDENTITIES[kind], f"{kind} identity")
    n = len(library.cells)
    if kind == "line":
        return [identity["identifier"]] * n, [None] * n, {}
    if kind == "labels":
        if identity["path"] is None:
            table = library.obs.assign(_cell=library.obs.index.astype(str))
            cell_column = "_cell"
        else:
            table = pd.read_csv(
                root / identity["path"],
                sep=identity["sep"],
                skiprows=identity["skiprows"],
                dtype=str,
            )
            cell_column = identity["cell_column"]
            if cell_column is None:  # cell ids are the table's unnamed first column
                if isinstance(table.index, pd.RangeIndex):
                    table = table.rename(columns={table.columns[0]: "_cell"})
                else:  # an R table: header one field short, ids became the index
                    table = table.rename_axis("_cell").reset_index()
                cell_column = "_cell"
        for column, values in identity["keep"].items():
            table = table.loc[table[column].astype(str).isin(values)]
        cells = identity["cell_prefix"] + table[cell_column] + identity["cell_suffix"]
        table = table.set_index(cells.to_numpy())
        if not table.index.is_unique:
            raise ValueError(f"{identity['path']}: duplicate labelled cells")
        found = table.reindex(library.cells)
        arms = (
            [None] * n
            if identity["arm_column"] is None
            else found[identity["arm_column"]].tolist()
        )
        labels = found[identity["label_column"]].tolist()
        clean = [None if pd.isna(v) else str(v) for v in labels]
        return (
            clean,
            [None if pd.isna(a) else str(a) for a in arms],
            {"labelled_cells": int(sum(v is not None for v in clean))},
        )
    if kind == "barcode":
        # The line (group ``line``, else ``identifier``) and arm (group ``arm``)
        # carried by each cell id; ``rename`` turns a captured line into a name.
        pattern = re.compile(identity["pattern"])
        lines, arms = [], []
        for cell in library.cells:
            match = pattern.search(cell)
            groups = match.groupdict() if match else {}
            line = groups.get("line", identity["identifier"]) if match else None
            lines.append(None if line is None else identity["rename"].get(line, line))
            arms.append(groups.get("arm"))
        return lines, arms, {"matched_cells": int(sum(v is not None for v in lines))}
    if kind == "hashtag":
        if identity["counts"] is None:
            tags, names = library.tags, library.tag_names
            rows = np.arange(n)
        else:
            other = read_library(root, identity["counts"])
            if other.tags is not None:  # 10x features typed as antibody capture
                tags, names = other.tags, other.tag_names
            else:
                tags, names = other.counts, other.genes
            suffix = identity["cell_suffix"]
            position = {cell + suffix: i for i, cell in enumerate(other.cells)}
            rows = np.asarray([position.get(cell, -1) for cell in library.cells])
        if tags is None:
            raise ValueError("hashtag identity: the library has no tag features")
        dense = np.zeros((n, tags.shape[1]))
        present = rows >= 0
        dense[present] = tags[rows[present]].toarray()
        calls, tally = call_hashtags(
            dense,
            names,
            identity["hashtags"],
            min_count=identity["min_count"],
            min_ratio=identity["min_ratio"],
        )
        calls = [c if ok else None for c, ok in zip(calls, present, strict=True)]
        return calls, [None] * n, {"hashtags": tally}
    if kind == "pool":
        if library.symbols is None:
            raise ValueError("pool identity needs gene symbols")
        if len(identity["members"]) < 2:
            raise ValueError("a pool has at least two members")
        bulk = read_depmap_matrix(bulk_path)
        pool = {name: members[name] for name in identity["members"]}
        calls, table = assign_pool(
            library.counts,
            library.symbols,
            pool,
            bulk,
            clusters=identity["clusters"],
            min_margin=identity["min_margin"],
        )
        return calls, [None] * n, {"clusters": table}
    return _rt_hash(library, identity, root)


def _rt_hash(
    library: Library, identity: Mapping[str, Any], root: Path
) -> tuple[list[str | None], list[str | None], dict[str, Any]]:
    """sci-Plex: the line from the cell's RT barcode (the fifth ``_`` field of a
    ``P5_P7_RT_BC_<n>_Lig_BC_<m>`` id, named in ``rt_barcodes`` as ``RT_BC_<n>``)
    through the authors' RT-well map, and the arm from its hash: a singlet's top
    oligo (``call_hashtags``) up to its first ``_`` (``DMSO_0nM_10H01`` is DMSO)."""
    barcodes = pd.read_csv(
        root / identity["rt_barcodes"], sep=r"\s+", header=None, dtype=str
    )
    sequence = dict(zip(barcodes[0], barcodes[1], strict=True))
    wells = pd.read_csv(root / identity["rt_map"], dtype=str)
    line_of = dict(zip(wells["Barcode"], wells["Cell Line"], strict=True))
    lines = [
        line_of.get(sequence.get(f"RT_BC_{cell.split('_')[4]}"))
        for cell in library.cells
    ]
    hashes = pd.read_csv(
        root / identity["hash_table"],
        sep="\t",
        header=None,
        names=["sample", "cell", "oligo", "axis", "umis"],
        dtype={"cell": str, "oligo": str, "umis": np.int64},
    )
    oligos = sorted(set(hashes["oligo"]))
    position = {cell: i for i, cell in enumerate(library.cells)}
    found = hashes["cell"].map(position)
    hashes = hashes.loc[found.notna()]
    tags = sparse.csr_matrix(
        (
            hashes["umis"].to_numpy(),
            (
                found.dropna().astype(np.int64).to_numpy(),
                hashes["oligo"].map({o: i for i, o in enumerate(oligos)}).to_numpy(),
            ),
        ),
        shape=(len(library.cells), len(oligos)),
    )
    calls, tally = call_hashtags(
        tags.toarray(),
        oligos,
        {oligo: oligo for oligo in oligos},
        min_count=identity["min_count"],
        min_ratio=identity["min_ratio"],
    )
    arms = [None if call is None else call.split("_")[0] for call in calls]
    report = {
        "rt_lines": dict(Counter(line for line in lines if line is not None)),
        "hashtags": tally,
    }
    return lines, arms, report
