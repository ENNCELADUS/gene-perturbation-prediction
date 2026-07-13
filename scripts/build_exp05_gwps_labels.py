"""Build the canonical exp05 GWPS/DepMap label intersection."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import anndata as ad
import pandas as pd

GENE_COLUMN_RE = re.compile(r"^(?P<symbol>.+) \((?P<entrez>\d+)\)$")
DEFAULT_GWPS_H5AD = Path(
    "data/sl_dependency_v0/raw/replogle/K562_gwps_normalized_singlecell_01.h5ad"
)
DEFAULT_GENE_EFFECT_CSV = Path("data/sl_dependency_v0/raw/depmap/CRISPRGeneEffect.csv")
DEFAULT_OUTPUT_CSV = Path("data/sl_dependency_v0/interim/k562_gwps_depmap_overlap.csv")


def build_gwps_label_table(
    gwps_h5ad: Path,
    gene_effect_csv: Path,
    model_id: str,
) -> pd.DataFrame:
    """Return the deterministic numeric GWPS/DepMap gene intersection."""
    if not gene_effect_csv.is_file():
        raise ValueError(f"{model_id} not found in {gene_effect_csv}")
    effects = pd.read_csv(gene_effect_csv, index_col=0)
    if model_id not in effects.index:
        raise ValueError(f"{model_id} not found in {gene_effect_csv}")

    adata = ad.read_h5ad(gwps_h5ad, backed="r")
    try:
        gwps_genes = {
            str(value).upper()
            for value in adata.obs["gene"].astype(str).unique()
            if str(value) != "non-targeting"
        }
    finally:
        adata.file.close()

    rows: list[dict[str, object]] = []
    numeric = pd.to_numeric(effects.loc[model_id], errors="coerce").dropna()
    for column, value in numeric.items():
        match = GENE_COLUMN_RE.match(str(column))
        if match is None:
            continue
        symbol = match.group("symbol").upper()
        if symbol not in gwps_genes:
            continue
        rows.append(
            {
                "perturbation_gene": symbol,
                "depmap_model_id": model_id,
                "depmap_entrez_id": match.group("entrez"),
                "depmap_gene_effect": float(value),
                "has_depmap_label": True,
            }
        )
    return pd.DataFrame(rows).sort_values("perturbation_gene").reset_index(drop=True)


def main() -> None:
    """Build and write the canonical exp05 label table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gwps-h5ad", type=Path, default=DEFAULT_GWPS_H5AD)
    parser.add_argument("--gene-effect-csv", type=Path, default=DEFAULT_GENE_EFFECT_CSV)
    parser.add_argument("--model-id", default="ACH-000551")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_CSV)
    args = parser.parse_args()

    labels = build_gwps_label_table(args.gwps_h5ad, args.gene_effect_csv, args.model_id)
    if labels["perturbation_gene"].duplicated().any():
        raise ValueError("duplicate perturbation_gene rows in GWPS/DepMap overlap")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
