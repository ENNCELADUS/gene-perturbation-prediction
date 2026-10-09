"""Write the run's DepMap inputs in the layout the released notebooks read.

The paper used DepMap 23Q4. The H20 cannot reach figshare, so phase 1 uses the pinned
26Q1 files already on it: expression is reduced to one default profile per model (26Q1
has several per model) and written with ModelID as an unnamed index, the layout of the
23Q4 OmicsExpressionProteinCodingGenesTPMLogp1.csv; Model.csv stands in for the Drive
metadata.csv; CRISPRGeneEffect.csv already has the layout the perturber reads.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DEPMAP_26Q1, OFFICIAL_DATA, paths, sha256  # noqa: E402

NON_GENE_COLUMNS = [
    "SequencingID",
    "ModelConditionID",
    "ModelID",
    "IsDefaultEntryForMC",
    "IsDefaultEntryForModel",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    p = paths(args.run_dir)
    p["inputs"].mkdir(parents=True, exist_ok=True)

    sources = {
        "expression": DEPMAP_26Q1
        / "OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv",
        "metadata": DEPMAP_26Q1 / "Model.csv",
        "gene_effect": DEPMAP_26Q1 / "CRISPRGeneEffect.csv",
    }
    expression = pd.read_csv(sources["expression"], index_col=0)
    expression = expression[expression["IsDefaultEntryForModel"] == "Yes"]
    if expression["ModelID"].duplicated().any():
        raise ValueError("more than one default expression profile for a ModelID")
    expression = expression.set_index("ModelID").drop(
        columns=[c for c in NON_GENE_COLUMNS if c != "ModelID"]
    )
    expression.index.name = None
    expression.to_csv(p["expression"])
    shutil.copyfile(sources["metadata"], p["metadata"])
    shutil.copyfile(sources["gene_effect"], p["gene_effect"])

    gene_effect_lines = set(pd.read_csv(p["gene_effect"], usecols=[0]).iloc[:, 0])
    provenance = {
        "depmap_release": "DepMap Public 26Q1",
        "sources": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in sources.items()
        },
        "official_data": {
            path.name: sha256(path) for path in sorted(OFFICIAL_DATA.iterdir())
        },
        "expression_lines": int(len(expression)),
        "expression_genes": int(expression.shape[1]),
        "gene_effect_lines": len(gene_effect_lines),
        "lines_with_expression_and_gene_effect": len(
            gene_effect_lines & set(expression.index)
        ),
    }
    (p["inputs"] / "provenance.json").write_text(
        json.dumps(provenance, indent=1) + "\n"
    )
    print(
        json.dumps(
            {
                k: v
                for k, v in provenance.items()
                if k not in ("sources", "official_data")
            }
        )
    )


if __name__ == "__main__":
    main()
