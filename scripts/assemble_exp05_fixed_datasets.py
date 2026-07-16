"""Assemble the unified Replogle plus non-Replogle K562 gene pool."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
import re

import anndata as ad
import numpy as np
import pandas as pd

from aivc_model.gene_splits import build_fixed_split_manifest

DEFAULT_ADAMSON_OVERLAP = Path(
    "data/sl_dependency_v0/interim/k562_adamson_depmap_overlap.csv"
)
DEFAULT_DIXIT_H5AD = Path("data/sl_dependency_v0/raw/dixit/dixit_2016.h5ad")
DEFAULT_GENE_EFFECT = Path("data/sl_dependency_v0/raw/depmap/CRISPRGeneEffect.csv")
DEFAULT_EXTERNAL_OVERLAP = Path(
    "data/sl_dependency_v0/interim/k562_non_replogle_depmap_overlap.csv"
)
DEFAULT_FIXED_MANIFEST = Path(
    "data/sl_dependency_v0/splits/k562_pool_depmap_fixed_seed42.csv"
)
K562_MODE = {
    "cell_line_id": "ACH-000551",
    "cell_line_name": "K-562",
    "ccle_name": "K562_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE",
}


def labels_from_predictions(path: Path) -> pd.DataFrame:
    """Recover one label per canonical gene from audited internal predictions."""
    frame = pd.read_csv(path)
    required = {"perturbation_gene", "y_true", "evaluation_scope"}
    if not required <= set(frame):
        raise ValueError(
            f"predictions are missing columns {sorted(required - set(frame))}"
        )
    frame = frame.loc[frame["evaluation_scope"] == "internal_outer_test"].copy()
    frame["perturbation_gene"] = frame["perturbation_gene"].astype(str).str.upper()
    frame["y_true"] = pd.to_numeric(frame["y_true"], errors="coerce")
    if not np.isfinite(frame["y_true"]).all():
        raise ValueError("internal predictions contain non-finite y_true values")
    grouped = frame.groupby("perturbation_gene", sort=True)["y_true"]
    spread = grouped.max() - grouped.min()
    if (spread > 1e-8).any():
        raise ValueError("internal predictions contain conflicting y_true values")
    return grouped.mean().rename("depmap_gene_effect").reset_index()


def read_gene_effect_row(path: Path, model_id: str) -> dict[str, float]:
    """Read one DepMap model row without loading the full matrix into memory."""
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        for row in reader:
            if row and row[0] == model_id:
                result: dict[str, float] = {}
                for column, raw_value in zip(header[1:], row[1:], strict=True):
                    if not raw_value:
                        continue
                    gene = column.rsplit(" (", 1)[0].upper()
                    result[gene] = float(raw_value)
                return result
    raise ValueError(f"DepMap model {model_id!r} is absent from {path}")


def build_dixit_overlap(
    h5ad_path: Path,
    gene_effect: dict[str, float],
    *,
    min_cells: int,
) -> pd.DataFrame:
    """Map exact single-gene Dixit conditions to K562 GeneEffect labels."""
    adata = ad.read_h5ad(h5ad_path, backed="r")
    try:
        counts = adata.obs["perturbation_name"].astype(str).value_counts()
    finally:
        adata.file.close()
    rows = []
    for condition, count in counts.items():
        gene = str(condition).upper()
        if (
            int(count) < min_cells
            or re.fullmatch(r"[A-Z0-9-]+", gene) is None
            or gene.startswith("INTERGENIC")
            or gene not in gene_effect
        ):
            continue
        rows.append(
            {
                **K562_MODE,
                "source_perturbation_label": str(condition),
                "perturbation_gene": gene,
                "perturbation_label_type": "single_gene",
                "perturbation_modality": "CRISPR-KO",
                "depmap_gene_column": gene,
                "has_depmap_label": True,
                "depmap_gene_effect": gene_effect[gene],
                "n_cells_or_pseudobulk": int(count),
                "is_control_candidate": False,
                "source_dataset": "dixit_2016",
            }
        )
    if not rows:
        raise ValueError("Dixit produced no exact single-gene DepMap matches")
    return pd.DataFrame(rows).sort_values("perturbation_gene").reset_index(drop=True)


def combine_external_overlap(
    adamson_overlap: pd.DataFrame,
    dixit_overlap: pd.DataFrame,
) -> pd.DataFrame:
    """Combine only finite single-gene external conditions."""
    adamson = adamson_overlap.copy()
    if "perturbation_label_type" in adamson:
        adamson = adamson.loc[adamson["perturbation_label_type"] == "single_gene"]
    adamson = adamson.loc[adamson["has_depmap_label"].astype(bool)].copy()
    adamson["depmap_gene_effect"] = pd.to_numeric(
        adamson["depmap_gene_effect"], errors="coerce"
    )
    adamson = adamson.loc[np.isfinite(adamson["depmap_gene_effect"])]
    combined = pd.concat([adamson, dixit_overlap], ignore_index=True, sort=False)
    combined["perturbation_gene"] = (
        combined["perturbation_gene"].astype(str).str.upper()
    )
    keys = ["source_dataset", "source_perturbation_label", "perturbation_gene"]
    return combined.drop_duplicates(keys).sort_values(keys).reset_index(drop=True)


def build_pool_labels(
    replogle_labels: pd.DataFrame,
    non_replogle_overlap: pd.DataFrame,
) -> pd.DataFrame:
    """Add only truly Replogle-unseen genes to the split universe."""
    external = non_replogle_overlap[
        ["perturbation_gene", "depmap_gene_effect"]
    ].copy()
    external["perturbation_gene"] = (
        external["perturbation_gene"].astype(str).str.upper()
    )
    grouped = external.groupby("perturbation_gene", sort=True)["depmap_gene_effect"]
    spread = grouped.max() - grouped.min()
    if (spread > 1e-8).any():
        raise ValueError("non-Replogle sources contain conflicting GeneEffect labels")
    external = grouped.mean().reset_index()
    replogle_genes = set(replogle_labels["perturbation_gene"].astype(str).str.upper())
    unseen = external.loc[~external["perturbation_gene"].isin(replogle_genes)]
    pooled = pd.concat([replogle_labels, unseen], ignore_index=True)
    return pooled.sort_values("perturbation_gene").reset_index(drop=True)


def write_immutable_csv(frame: pd.DataFrame, path: Path) -> str:
    """Write deterministic CSV bytes and refuse a conflicting overwrite."""
    content = frame.to_csv(index=False).encode("utf-8")
    if path.exists() and path.read_bytes() != content:
        raise FileExistsError(f"refusing to overwrite non-identical {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    Path(f"{path}.sha256").write_text(f"{digest}\n", encoding="ascii")
    return digest


def main() -> None:
    """Build the fixed role manifest and K562 external overlap table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-csv", type=Path, required=True)
    parser.add_argument("--adamson-overlap", type=Path, default=DEFAULT_ADAMSON_OVERLAP)
    parser.add_argument("--dixit-h5ad", type=Path, default=DEFAULT_DIXIT_H5AD)
    parser.add_argument("--gene-effect", type=Path, default=DEFAULT_GENE_EFFECT)
    parser.add_argument("--external-out", type=Path, default=DEFAULT_EXTERNAL_OVERLAP)
    parser.add_argument("--split-out", type=Path, default=DEFAULT_FIXED_MANIFEST)
    parser.add_argument("--train-fraction", type=float, default=0.85)
    parser.add_argument("--validation-fraction", type=float, default=0.075)
    parser.add_argument("--min-cells", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    labels = labels_from_predictions(args.predictions_csv)
    gene_effect = read_gene_effect_row(args.gene_effect, K562_MODE["cell_line_id"])
    dixit = build_dixit_overlap(
        args.dixit_h5ad,
        gene_effect,
        min_cells=args.min_cells,
    )
    external = combine_external_overlap(pd.read_csv(args.adamson_overlap), dixit)
    pool_labels = build_pool_labels(labels, external)
    manifest = build_fixed_split_manifest(
        pool_labels,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    write_immutable_csv(external, args.external_out)
    write_immutable_csv(manifest, args.split_out)


if __name__ == "__main__":
    main()
