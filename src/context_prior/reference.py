"""The pinned reference tables of the linear context prior."""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class Reference:
    """Gene-pair lists, gene sets and training-side line labels.

    Attributes:
        paralogs: ``gene, paralog, identity``, closest first within each gene.
        complexes: ``complex_id, gene``.
        hallmark: ``gene_set, gene``.
        progeny: ``pathway, gene, weight``.
        drivers: training-side lines x driver genes, 0/1.
        msi: training-side line -> MSI score.
    """

    paralogs: pd.DataFrame
    complexes: pd.DataFrame
    hallmark: pd.DataFrame
    progeny: pd.DataFrame
    drivers: pd.DataFrame
    msi: pd.Series


def load_reference(directory: Path, *, blocked: Collection[str]) -> Reference:
    """Load the tables; raise if a line table holds any ``blocked`` line."""
    directory = Path(directory)

    def read(name: str) -> pd.DataFrame:
        return pd.read_csv(directory / name, sep="\t")

    drivers = read("drivers.tsv").astype({"model_id": str}).set_index("model_id")
    msi = read("msi.tsv").astype({"model_id": str}).set_index("model_id")["msi_score"]
    leaked = sorted((set(drivers.index) | set(msi.index)) & set(blocked))
    if leaked:
        raise ValueError(
            f"reference tables hold held-out or patient-sharing lines: {leaked[:10]}"
        )
    return Reference(
        paralogs=read("paralogs.tsv.gz"),
        complexes=read("complexes.tsv"),
        hallmark=read("hallmark.tsv"),
        progeny=read("progeny.tsv"),
        drivers=drivers,
        msi=msi,
    )
