"""GO semantic similarity of genes for SLMGAE's two GO views, in Python.

The released GO matrices of SLMGAE came from GOSemSim (R, Wang's measure, best-match
average). This module computes the same measure for any gene list with sparse and dense
matrix products (no loop over gene pairs) and builds the official k-nearest-neighbour
graph. numpy, pandas and scipy only.
"""

import gzip
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

#: Wang's weights of the two relations the measure follows.
WEIGHTS = {"is_a": 0.8, "part_of": 0.6}
ASPECTS = {"BP": "P", "CC": "C"}


def read_obo(path: Path) -> dict[str, list[tuple[str, float]]]:
    """Term -> weighted parents (``is_a`` 0.8, ``part_of`` 0.6) of the non-obsolete
    terms of a go-basic.obo file."""
    parents: dict[str, list[tuple[str, float]]] = {}
    term = None
    for line in Path(path).read_text().splitlines():
        if line.startswith("["):
            term = None if line != "[Term]" else ""
        elif term is None or not line.strip():
            continue
        elif line.startswith("id: "):
            term = line[4:].strip()
            parents[term] = []
        elif line.startswith("is_obsolete: true"):
            parents.pop(term, None)
            term = None
        elif line.startswith("is_a: "):
            parents[term].append((line[6:].split()[0], WEIGHTS["is_a"]))
        elif line.startswith("relationship: part_of "):
            parents[term].append((line[22:].split()[0], WEIGHTS["part_of"]))
    return parents


def read_gaf(path: Path, aspect: str) -> pd.DataFrame:
    """Unique (symbol, term) rows of a GAF file for one aspect (``BP`` or ``CC``),
    without the rows with a ``NOT`` qualifier."""
    with gzip.open(path, "rt") as handle:
        table = pd.read_csv(
            handle,
            sep="\t",
            comment="!",
            header=None,
            usecols=[2, 3, 4, 8],
            names=["symbol", "qualifier", "term", "aspect"],
            dtype=str,
        )
    keep = (table["aspect"] == ASPECTS[aspect]) & ~table["qualifier"].str.contains(
        "NOT", na=False
    )
    return table.loc[keep, ["symbol", "term"]].drop_duplicates(ignore_index=True)


def term_similarity(
    parents: dict[str, list[tuple[str, float]]], terms: list[str]
) -> np.ndarray:
    """Wang's similarity of every pair of ``terms``. S_A(t) is the largest product of
    edge weights on a path from A up to t (1 at A); SV(A) = sum of S_A;
    sim(A, B) = sum over shared ancestors t of (S_A(t) + S_B(t)) / (SV(A) + SV(B))."""
    closure: dict[str, int] = {}
    order: list[str] = []  # parents before children

    def visit(root: str) -> None:
        stack = [(root, False)]
        while stack:
            term, done = stack.pop()
            if done:
                closure[term] = len(order)
                order.append(term)
            elif term not in closure and term in parents:
                closure[term] = -1  # on the stack: GO is a DAG, so no cycle
                stack.append((term, True))
                stack.extend((p, False) for p, _ in parents[term] if p not in closure)

    for term in terms:
        visit(term)
    # a term pushed twice before it finished is placed once; its parents come first
    size = len(order)
    index = {t: i for i, t in enumerate(order)}
    s = np.zeros((size, size))
    for term in order:
        i = index[term]
        for parent, weight in parents[term]:
            if parent in index:
                np.maximum(s[i], weight * s[index[parent]], out=s[i])
        s[i, i] = 1.0
    rows = np.array([index[t] for t in terms])
    s = s[rows]
    sv = s.sum(axis=1)
    shared = s @ (s > 0).T.astype(float)  # sum over t of S_A(t) 1[t in T_B]
    return (shared + shared.T) / (sv[:, None] + sv[None, :])


def gene_similarity(term_sim: np.ndarray, gene_terms: sp.csr_matrix) -> np.ndarray:
    """Best-match average of two genes' term sets:
    (sum_i max_j sim_ij + sum_j max_i sim_ij) / (m + n). A gene without a term has
    similarity 0 to every gene."""
    gene_terms = sp.csr_matrix(gene_terms)
    best = np.zeros(
        (term_sim.shape[0], gene_terms.shape[0])
    )  # term -> gene: best match
    for gene in range(gene_terms.shape[0]):
        own = gene_terms.indices[gene_terms.indptr[gene] : gene_terms.indptr[gene + 1]]
        if len(own):
            best[:, gene] = term_sim[:, own].max(axis=1)
    forward = gene_terms @ best  # [g, h]: sum over g's terms of the best match in h
    count = np.diff(gene_terms.indptr).astype(float)
    size = count[:, None] + count[None, :]
    return np.divide(
        forward + forward.T, size, out=np.zeros_like(forward), where=size > 0
    )


def knn_graph(sim: np.ndarray, k: int) -> sp.coo_matrix:
    """The official graph: each gene keeps its k most similar genes (similarity above
    0, never itself); an edge stays when either gene keeps it. Binary, symmetric."""
    sim = np.asarray(sim, dtype=float)
    n = len(sim)
    k = min(k, n - 1)
    rows, cols = [], []
    for start in range(0, n, 2048):
        block = sim[start : start + 2048].copy()
        block[np.arange(len(block)), start + np.arange(len(block))] = 0
        top = np.argpartition(block, n - k, axis=1)[:, n - k :]
        row = np.repeat(start + np.arange(len(block)), k)
        col = top.ravel()
        keep = block[row - start, col] > 0
        rows.append(row[keep])
        cols.append(col[keep])
    row, col = np.concatenate(rows), np.concatenate(cols)
    graph = sp.coo_matrix((np.ones(len(row)), (row, col)), shape=(n, n)).tocsr()
    return (graph + graph.T).astype(bool).astype(float).tocoo()


def go_similarity(
    annotations: pd.DataFrame,
    parents: dict[str, list[tuple[str, float]]],
    genes: list[str],
) -> np.ndarray:
    """Gene similarity of ``genes`` (rows of ``annotations``: symbol, term). A term
    absent from the ontology (obsolete) is dropped."""
    annotations = annotations[
        annotations["symbol"].isin(set(genes)) & annotations["term"].isin(parents)
    ]
    terms = sorted(annotations["term"].unique())
    gene_row = {g: i for i, g in enumerate(genes)}
    term_col = {t: i for i, t in enumerate(terms)}
    gene_terms = sp.csr_matrix(
        (
            np.ones(len(annotations)),
            (
                annotations["symbol"].map(gene_row).to_numpy(),
                annotations["term"].map(term_col).to_numpy(),
            ),
        ),
        shape=(len(genes), len(terms)),
    )
    return gene_similarity(term_similarity(parents, terms), gene_terms)


def read_released(path: Path, n: int) -> np.ndarray:
    """A released GO matrix (Human_GOsim*.txt) as a dense symmetric matrix. The file is
    a ragged upper triangle: value k of line i is the similarity of genes i and
    i + 1 + k (line i holds n - 1 - i values and a trailing tab). The diagonal is 0."""
    out = np.zeros((n, n))
    lines = Path(path).read_text().split("\n")
    for i, line in enumerate(lines[: n - 1]):
        values = np.array([v for v in line.split("\t") if v], dtype=float)
        if len(values) != n - 1 - i:
            raise ValueError(
                f"line {i} holds {len(values)} values, expected {n - 1 - i}"
            )
        out[i, i + 1 :] = values
    return out + out.T
