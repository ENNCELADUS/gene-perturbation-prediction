"""Map an X-Atlas/Orion HCT116 h5ad to HCT116 DepMap GeneEffect."""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import anndata as ad
import numpy as np
import pandas as pd

from scripts.assemble_exp05_fixed_datasets import (
    DEFAULT_GENE_EFFECT,
    read_gene_effect_row,
    write_immutable_csv,
)

HCT116_MODEL = {
    "cell_line_id": "ACH-000971",
    "cell_line_name": "HCT116",
    "ccle_name": "HCT116_LARGE_INTESTINE",
}


def build_xatlas_overlap(
    h5ad_path: Path,
    gene_effect: dict[str, float],
    *,
    target_col: str,
    pass_filter_col: str | None,
    min_cells: int,
) -> pd.DataFrame:
    """Build exp05-compatible HCT116 condition-to-label rows."""
    adata = ad.read_h5ad(h5ad_path, backed="r")
    try:
        obs = adata.obs
        if target_col not in obs:
            raise ValueError(f"X-Atlas obs is missing {target_col!r}")
        mask = np.ones(len(obs), dtype=bool)
        if pass_filter_col is not None:
            if pass_filter_col not in obs:
                raise ValueError(f"X-Atlas obs is missing {pass_filter_col!r}")
            mask = obs[pass_filter_col].astype(bool).to_numpy()
        counts = obs.loc[mask, target_col].astype(str).value_counts()
    finally:
        adata.file.close()

    rows = []
    for condition, count in counts.items():
        gene = str(condition).upper()
        if (
            int(count) < min_cells
            or re.fullmatch(r"[A-Z0-9-]+", gene) is None
            or gene in {"NON-TARGETING", "NON_TARGETING", "CONTROL"}
            or gene not in gene_effect
        ):
            continue
        rows.append(
            {
                **HCT116_MODEL,
                "source_perturbation_label": str(condition),
                "perturbation_gene": gene,
                "perturbation_label_type": "single_gene",
                "perturbation_modality": "CRISPRi",
                "depmap_gene_column": gene,
                "has_depmap_label": True,
                "depmap_gene_effect": gene_effect[gene],
                "n_cells_or_pseudobulk": int(count),
                "is_control_candidate": False,
                "source_dataset": "xatlas_orion_hct116",
            }
        )
    if not rows:
        raise ValueError("X-Atlas produced no HCT116 GeneEffect matches")
    return pd.DataFrame(rows).sort_values("perturbation_gene").reset_index(drop=True)


def main() -> None:
    """Build the HCT116 external overlap after the processed h5ad is local."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5ad", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gene-effect", type=Path, default=DEFAULT_GENE_EFFECT)
    parser.add_argument("--target-col", default="gene_target")
    parser.add_argument("--pass-filter-col", default="pass_guide_filter")
    parser.add_argument("--min-cells", type=int, default=8)
    args = parser.parse_args()
    gene_effect = read_gene_effect_row(
        args.gene_effect,
        HCT116_MODEL["cell_line_id"],
    )
    overlap = build_xatlas_overlap(
        args.h5ad,
        gene_effect,
        target_col=args.target_col,
        pass_filter_col=args.pass_filter_col,
        min_cells=args.min_cells,
    )
    write_immutable_csv(overlap, args.out)


if __name__ == "__main__":
    main()
