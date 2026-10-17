"""Cell-state vocabularies, memberships and state inputs (PIE stage-one design
§3.5, Task 8).

A **vocabulary** is fitted on the single-cell training lines only, each line
weighing equally (``fit_sample``), then frozen and **projected** on every cell of
a line: soft memberships ``q_ik`` summing to one over the vocabulary's states.

- ``cycle``: a Gaussian mixture on each cell's S and G2/M program scores
  (absolute), its components named non-cycling, S and G2/M.
- ``programs``: recurrent programs from consensus NMF per training line
  (``src.data.prepare.state_programs``, ``.venv-states``); a cell's memberships
  are its non-negative usages of the recurrent programs and the background
  program, cycle usages dropped, normalised to one (``program_memberships``).
- Mixtures (``fit_mixture``): a Gaussian mixture with full covariances on a
  per-cell representation, line-centred (each line's mean over all of its cells
  subtracted) or absolute: scVI's latent, the HVG cells' and the Tx1-3B
  embeddings' leading principal components, or the QC columns (log UMI, genes
  detected, mitochondrial fraction) standardised.

For line c with per-cell PIE input ``u_i = log1p(CP10K_i)`` on the decode panel
(``per_cell_log_cp10k``: Task 2's gene order, whole-library denominator and
unlisted-gene fill), a state's mass is ``pi_k = sum_i q_ik / n`` and its input
``x_k = sum_i q_ik u_i / sum_i q_ik``, so ``sum_k pi_k x_k`` is the line's PIE
input exactly. A state whose effective cell count ``(sum q)^2 / sum q^2`` is
below a floor is merged into its nearest state (by input), its membership column
added, so mass and weighted expression are kept (``merge_small``). Controls:
memberships permuted across a line's cells (``permute_memberships``), which keeps
every state's mass, effective count and overlap and the line input; a
cycling-score-weighted whole-line input (``cycling_weights``).

Context names (``context_name``) are ``<ModelID>|<set>|<state>``; the frozen PIE
reads a context's Cellosaurus text through its owner, the ModelID.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import pandas as pd
from scipy import sparse

from src.data.basal import align_columns
from src.data.ladder_cells import (
    PIE_TARGET_SUM,
    _scaled,
    library_sizes,
    line_rng,
    upper_symbols,
)

SEPARATOR: Final[str] = "|"
WHOLE: Final[str] = "whole"
CYCLING_SET: Final[str] = "cycling_weighted"
CYCLE: Final[str] = "cycle"
PROGRAMS: Final[str] = "programs"
#: Per-cell representations a mixture vocabulary sits on, and whether each is
#: line-centred.
MIXTURE_FAMILIES: Final[dict[str, tuple[str, bool]]] = {
    "scvi_centred": ("scvi", True),
    "scvi_absolute": ("scvi", False),
    "hvg": ("hvg", True),
    "tx1": ("tx1", True),
    "qc": ("qc", True),
}
#: The vocabularies whose memberships are permuted as the primary control (QC
#: states are themselves a control).
PERMUTED_FAMILIES: Final[tuple[str, ...]] = (
    CYCLE,
    PROGRAMS,
    "scvi_centred",
    "scvi_absolute",
    "hvg",
    "tx1",
)
#: ``cells/<ModelID>.npz`` qc columns (``ladder_inputs.QC_COLUMNS``).
QC_FEATURES: Final[tuple[int, ...]] = (0, 1, 2)  # log UMI, genes, mito fraction
CYCLE_FEATURES: Final[tuple[int, ...]] = (4, 5)  # S and G2/M scores
#: Program usages of a cell with no usage at all go to the background program.
BACKGROUND: Final[str] = "background"


# --- names --------------------------------------------------------------------


def context_name(model_id: str, set_name: str, state: str) -> str:
    for part in (model_id, set_name, state):
        if SEPARATOR in part or not part:
            raise ValueError(
                f"context name part {part!r} is empty or holds {SEPARATOR}"
            )
    return SEPARATOR.join((model_id, set_name, state))


def context_owner(name: str) -> str:
    """The ModelID owning a context (whose Cellosaurus text PIE reads)."""
    owner, _, _ = name.split(SEPARATOR)
    return owner


def membership_key(vocabulary: str, seed: int) -> str:
    return f"{vocabulary}@seed{seed}"


def permuted_set(vocabulary: str, seed: int) -> str:
    return f"{vocabulary}__perm{seed}"


def vocabularies(config: Mapping[str, Any]) -> list[str]:
    """Every vocabulary a states config fits, in report order."""
    counts = config["fit"]["mixture_counts"]
    return [
        CYCLE,
        PROGRAMS,
        *(f"{family}_k{k}" for family in MIXTURE_FAMILIES for k in counts),
    ]


def family(vocabulary: str) -> str:
    if vocabulary in (CYCLE, PROGRAMS):
        return vocabulary
    head, _, k = vocabulary.rpartition("_k")
    if head not in MIXTURE_FAMILIES or not k.isdigit():
        raise ValueError(f"unknown vocabulary {vocabulary!r}")
    return head


def state_sets(config: Mapping[str, Any]) -> list[str]:
    """The sets of state inputs the frozen PIE reads, in queue order: every
    vocabulary at the first fitting seed and the cycling-weighted input, then
    each permutation seed's permuted vocabularies."""
    real = vocabularies(config)
    permuted = [v for v in real if family(v) in PERMUTED_FAMILIES]
    return [
        *real,
        CYCLING_SET,
        *(
            permuted_set(v, seed)
            for seed in config["inputs"]["permutation_seeds"]
            for v in permuted
        ),
    ]


def base_vocabulary(set_name: str) -> str:
    return set_name.split("__perm")[0]


# --- the fit sample -------------------------------------------------------------


def fit_sample(
    cell_counts: Mapping[str, int],
    roles: Mapping[str, str],
    split: Any,
    *,
    cells_per_line: int,
    seed: int,
) -> pd.DataFrame:
    """Row positions of the distinct cells each training line contributes to a
    fit: at most ``cells_per_line``, drawn without replacement, so a line with
    fewer cells contributes all of them and weighs less (duplicated cells would
    let a full-covariance mixture spend components on a small line's copies).
    Refuses any line that is not a labelled single-cell training line."""
    from src.data.splits import assert_fit_eligible

    rows = []
    for model_id, n in cell_counts.items():
        if roles[model_id] != "train_cells":
            raise ValueError(f"{model_id} ({roles[model_id]}) may not enter a fit")
        assert_fit_eligible(model_id, split)
        rng = line_rng(model_id, seed, "state_fit_sample")
        chosen = np.sort(rng.choice(n, size=min(n, cells_per_line), replace=False))
        rows.append(pd.DataFrame({"model_id": model_id, "row": chosen}))
    return pd.concat(rows, ignore_index=True)


def resolve_chemistry(
    table: pd.DataFrame, source: str, model_id: str, arms: Sequence[str]
) -> np.ndarray:
    """Each cell's chemistry from the reviewed table (``source``, ``model_id``,
    ``arm``, ``chemistry``; an empty ``model_id`` or ``arm`` matches any): the
    most specific matching row (source, ModelID and arm, then ModelID, then arm,
    then the source) that names a chemistry, a ``mixed`` row naming none. Fails
    closed when a cell resolves to no row."""
    rows = table.loc[table["source"].astype(str) == source].fillna("")
    out = np.empty(len(arms), dtype=object)
    by_arm: dict[str, str] = {}
    for arm in dict.fromkeys(str(a) for a in arms):
        candidates = rows.loc[
            rows["model_id"].isin(["", model_id])
            & rows["arm"].isin(["", arm])
            & (rows["chemistry"] != "mixed")
        ]
        if candidates.empty:
            raise ValueError(
                f"{model_id}: no chemistry row of source {source!r} covers arm {arm!r}"
            )
        rank = (candidates["model_id"] != "") * 2 + (candidates["arm"] != "")
        by_arm[arm] = str(candidates.loc[rank.idxmax(), "chemistry"])
    for i, arm in enumerate(arms):
        out[i] = by_arm[str(arm)]
    return out.astype(str)


def state_fingerprint(
    contexts: Sequence[str], inputs: np.ndarray, mass: np.ndarray
) -> str:
    """SHA-256 of a line's contexts, their inputs (float32) and masses (float64),
    to refuse frozen-PIE outputs computed from other state inputs."""
    import hashlib

    digest = hashlib.sha256()
    digest.update("\n".join(contexts).encode())
    digest.update(np.ascontiguousarray(inputs, dtype=np.float32).tobytes())
    digest.update(np.ascontiguousarray(mass, dtype=np.float64).tobytes())
    return digest.hexdigest()


def set_fingerprints(inputs: pd.DataFrame, masses: pd.DataFrame) -> dict[str, str]:
    """Each line's ``state_fingerprint`` in a set's ``inputs.parquet`` and
    ``masses.parquet`` (contexts in their table order)."""
    owners = masses["model_id"].astype(str).to_numpy()
    values = inputs.to_numpy(dtype=np.float32)
    mass = masses["mass"].to_numpy(dtype=np.float64)
    names = [str(c) for c in inputs.index]
    out = {}
    for owner in dict.fromkeys(owners):
        rows = np.flatnonzero(owners == owner)
        out[owner] = state_fingerprint(
            [names[i] for i in rows], values[rows], mass[rows]
        )
    return out


# --- mixtures ---------------------------------------------------------------------


def centre_lines(features: np.ndarray, centre: np.ndarray | None) -> np.ndarray:
    return features if centre is None else features - centre


def fit_pca(features: np.ndarray, components: int, seed: int) -> dict[str, np.ndarray]:
    """Leading principal components (mean, components)."""
    from sklearn.decomposition import PCA

    k = min(components, *features.shape)
    pca = PCA(n_components=k, random_state=seed).fit(features)
    return {
        "pca_mean": pca.mean_.astype(np.float64),
        "pca_components": pca.components_.astype(np.float64),
    }


def fit_mixture(
    features: np.ndarray, k: int, *, seed: int, inits: int
) -> dict[str, np.ndarray]:
    """A Gaussian mixture with full covariances (weights, means, covariances)."""
    from sklearn.mixture import GaussianMixture

    model = GaussianMixture(
        n_components=k, covariance_type="full", n_init=inits, random_state=seed
    ).fit(np.asarray(features, dtype=np.float64))
    return {
        "weights": model.weights_,
        "means": model.means_,
        "covariances": model.covariances_,
    }


def mixture_responsibilities(
    params: Mapping[str, np.ndarray], features: np.ndarray
) -> np.ndarray:
    """Each row's posterior over the mixture's components, computed row by row
    from the stored parameters (deterministic, independent of the other rows)."""
    from scipy.linalg import solve_triangular
    from scipy.special import logsumexp

    x = np.asarray(features, dtype=np.float64)
    weights, means, covariances = (
        params["weights"],
        params["means"],
        params["covariances"],
    )
    d = x.shape[1]
    log = np.empty((x.shape[0], len(weights)))
    for j, (w, mu, cov) in enumerate(zip(weights, means, covariances, strict=True)):
        lower = np.linalg.cholesky(cov)
        z = solve_triangular(lower, (x - mu).T, lower=True)
        log[:, j] = (
            np.log(w)
            - 0.5 * d * np.log(2 * np.pi)
            - np.log(np.diag(lower)).sum()
            - 0.5 * (z**2).sum(axis=0)
        )
    return np.exp(log - logsumexp(log, axis=1, keepdims=True))


def mixture_features(
    params: Mapping[str, np.ndarray], features: np.ndarray, centre: np.ndarray | None
) -> np.ndarray:
    """A representation as the mixture reads it: line-centred when the fit was,
    standardised and projected on its principal components when it was."""
    x = np.asarray(features, dtype=np.float64)
    if bool(params["centred"]):
        if centre is None:
            raise ValueError("a line-centred vocabulary needs the line's centre")
        x = x - centre
    if "scale" in params:
        x = x / params["scale"]
    if "pca_components" in params:
        x = (x - params["pca_mean"]) @ params["pca_components"].T
    return x


def project(
    params: Mapping[str, np.ndarray], features: np.ndarray, centre: np.ndarray | None
) -> np.ndarray:
    """A frozen mixture's memberships of a line's cells."""
    return mixture_responsibilities(params, mixture_features(params, features, centre))


def cycle_labels(means: np.ndarray) -> list[str]:
    """Names of a cycle mixture's components from their (S, G2/M) means: the one
    lowest on both is non-cycling; with three, the one higher in S than G2/M is
    S and the other G2/M."""
    order = np.argsort(means.max(axis=1), kind="stable")
    labels = [""] * len(means)
    labels[order[0]] = "noncycling"
    rest = [int(i) for i in order[1:]]
    if len(rest) == 2:
        s = max(rest, key=lambda i: means[i, 0] - means[i, 1])
        labels[s] = "s"
        labels[rest[0] if rest[1] == s else rest[1]] = "g2m"
    else:
        for n, i in enumerate(rest):
            labels[i] = f"cycling{n}"
    return labels


def cycling_probability(q: np.ndarray, labels: Sequence[str]) -> np.ndarray:
    """A cell's membership of the cycling phases (everything but non-cycling)."""
    return 1.0 - q[:, list(labels).index("noncycling")]


def program_memberships(usages: np.ndarray, kinds: Sequence[str]) -> np.ndarray:
    """Usage-to-weight rule of the program vocabulary: the usages of the
    recurrent programs and the background program (cycle usages dropped),
    normalised to sum to one; a cell with none of them is all background."""
    kinds = np.asarray(kinds)
    kept = np.asarray(usages, dtype=np.float64)[:, kinds != "cycle"]
    background = list(kinds[kinds != "cycle"]).index(BACKGROUND)
    total = kept.sum(axis=1)
    out = np.zeros_like(kept)
    has = total > 0
    out[has] = kept[has] / total[has, None]
    out[~has, background] = 1.0
    return out


# --- state inputs -----------------------------------------------------------------


@dataclass(frozen=True)
class LineCells:
    """A line's per-cell PIE input on the decode panel: ``u`` (cells x panel,
    ``log1p`` CP10K, zero where the source does not list the gene), the listed
    mask and the value PIE's input takes at an unlisted gene."""

    u: sparse.csr_matrix
    listed: np.ndarray
    fill: np.ndarray

    @property
    def n_cells(self) -> int:
        return int(self.u.shape[0])

    def weighted_inputs(self, weights: np.ndarray) -> np.ndarray:
        """Columns of ``weights`` (cells x states) as state inputs: the
        weighted mean of ``u`` per state, PIE's fill at unlisted genes."""
        w = np.asarray(weights, dtype=np.float64)
        total = w.sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            inputs = np.asarray((self.u.T @ w).T) / total[:, None]
        inputs[:, ~self.listed] = self.fill[~self.listed]
        return inputs

    def mean(self) -> np.ndarray:
        return self.weighted_inputs(np.ones((self.n_cells, 1)))[0]


def per_cell_log_cp10k(
    counts: object,
    symbols: Sequence[str],
    panel: Sequence[str],
    pie_input: np.ndarray,
    listed: np.ndarray,
) -> LineCells:
    """``u_i = log1p(CP10K_i)`` over the panel as Task 2 computes PIE's input
    (``ladder_cells.gene_summaries``): whole-library denominator, upper-case
    symbols with duplicates summed; an unlisted gene takes the line's
    ``pie_input`` value (the frame mean). Refuses a line whose listed genes
    differ from ``listed`` (``mean/pie_listed.parquet``)."""
    matrix = sparse.csr_matrix(counts, dtype=np.float64)
    library = library_sizes(matrix)
    aligned, present = align_columns(matrix, upper_symbols(symbols), list(panel))
    if not np.array_equal(present, np.asarray(listed, dtype=bool)):
        raise ValueError("the source lists other panel genes than PIE's input records")
    aligned.eliminate_zeros()
    u = _scaled(aligned, library, PIE_TARGET_SUM)
    u.data = np.log1p(u.data)
    fill = np.where(present, np.nan, np.asarray(pie_input, dtype=np.float64))
    return LineCells(u=u.tocsr(), listed=present, fill=fill)


def effective_count(q: np.ndarray) -> np.ndarray:
    """``(sum_i q_ik)^2 / sum_i q_ik^2`` per state (0 for an empty state)."""
    total = q.sum(axis=0)
    square = (q**2).sum(axis=0)
    out = np.zeros_like(total)
    np.divide(total**2, square, out=out, where=square > 0)
    return out


@dataclass(frozen=True)
class StateInputs:
    """One set's states of one line, after merging."""

    labels: list[str]
    mass: np.ndarray
    effective: np.ndarray
    inputs: np.ndarray  # states x panel
    merged: list[list[str]]  # the states each one absorbed


def merge_small(
    q: np.ndarray, cells: LineCells, labels: Sequence[str], minimum: float
) -> tuple[np.ndarray, list[str], list[list[str]]]:
    """Merge, one at a time and smallest first, every state whose effective cell
    count is below ``minimum`` into the state whose input is nearest (Euclidean
    over the panel), adding its membership column, until none is below or one
    state is left. An empty state is dropped (it holds no mass)."""
    q = np.array(q, dtype=np.float64)
    labels = list(labels)
    merged: list[list[str]] = [[] for _ in labels]
    while q.shape[1] > 1:
        effective = effective_count(q)
        small = np.flatnonzero(effective < minimum)
        if not small.size:
            break
        k = int(small[np.argmin(effective[small])])
        others = [j for j in range(q.shape[1]) if j != k]
        if q[:, k].sum() > 0:
            inputs = cells.weighted_inputs(q)
            alive = [j for j in others if q[:, j].sum() > 0]
            distance = ((inputs[alive] - inputs[k]) ** 2).sum(axis=1)
            target = alive[int(np.argmin(distance))]
            q[:, target] += q[:, k]
        else:
            target = others[int(np.argmax(q[:, others].sum(axis=0)))]
        merged[target] += [labels[k], *merged[k]]
        q = np.delete(q, k, axis=1)
        del labels[k], merged[k]
    return q, labels, merged


def state_masses_and_inputs(
    q: np.ndarray, cells: LineCells, labels: Sequence[str], minimum: float
) -> StateInputs:
    """Masses, effective counts and inputs of one line's states after
    ``merge_small``; ``sum_k mass_k inputs_k`` is the line's PIE input."""
    q, labels, merged = merge_small(q, cells, labels, minimum)
    return StateInputs(
        labels=labels,
        mass=q.sum(axis=0) / cells.n_cells,
        effective=effective_count(q),
        inputs=cells.weighted_inputs(q),
        merged=merged,
    )


def permute_memberships(
    q: np.ndarray, model_id: str, vocabulary: str, seed: int
) -> np.ndarray:
    """The line's membership rows permuted across its cells: every state's mass,
    effective count and overlap are kept, its link to expression broken."""
    rng = line_rng(model_id, seed, f"membership_permutation|{vocabulary}")
    return np.asarray(q)[rng.permutation(len(q))]


def cycling_weights(probability: np.ndarray, floor: float) -> np.ndarray:
    """Each cell weighted by its cycling membership plus ``floor``, normalised
    to sum to one (a sensitivity row for proliferation weighting)."""
    w = np.asarray(probability, dtype=np.float64) + floor
    return w / w.sum()


def masses(q: np.ndarray) -> np.ndarray:
    return np.asarray(q, dtype=np.float64).mean(axis=0)


__all__ = [
    "BACKGROUND",
    "CYCLE",
    "CYCLING_SET",
    "LineCells",
    "MIXTURE_FAMILIES",
    "PROGRAMS",
    "StateInputs",
    "WHOLE",
    "base_vocabulary",
    "context_name",
    "context_owner",
    "cycle_labels",
    "cycling_probability",
    "cycling_weights",
    "effective_count",
    "family",
    "fit_mixture",
    "fit_pca",
    "fit_sample",
    "masses",
    "membership_key",
    "merge_small",
    "mixture_features",
    "mixture_responsibilities",
    "per_cell_log_cp10k",
    "permute_memberships",
    "permuted_set",
    "program_memberships",
    "project",
    "resolve_chemistry",
    "set_fingerprints",
    "state_fingerprint",
    "state_masses_and_inputs",
    "state_sets",
    "vocabularies",
]
