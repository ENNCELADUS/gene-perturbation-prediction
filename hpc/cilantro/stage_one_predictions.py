"""The released stage one's GeneEffect predictions on the validation and test lines.

Reads the run's stage-one rows (generated_df/gene2vec_emb_mat.h5: one row per line and
gene in the line's top 2,048 expressed, its 640-d input and Cilantro's label) and its
trained regressor (models/gene2vec_5x.pth), keeps the rows of the split's validation and
test lines, maps Ensembl IDs to DepMap symbols through the notebooks' symbol map (an
Ensembl ID several symbols share is dropped), and writes ``model_id, gene_symbol,
prediction, cilantro_label`` as CSV. The regressor trained on every row of these lines,
so its predictions on them are in-sample. No retraining; the noise diagnosis's
learning curve scores the file (configs/context_prior/noise_diagnosis.yaml).

``.venv-cilantro/bin/python hpc/cilantro/stage_one_predictions.py --run-dir
outputs/cilantro_reproduction/as_released_gc95M --out
outputs/cilantro_analysis/stage_one_split_lines.csv``
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OFFICIAL_CODE, ROOT, SYMBOL_MAP  # noqa: E402

SPLIT = ROOT / "configs" / "benchmarks" / "cell_line_geneeffect_split.json"
KEY = re.compile(r"\('([^']+)', 'gene_([^']+)'\)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    import numpy as np
    import pandas as pd
    import torch

    sys.path.insert(0, str(OFFICIAL_CODE / "nn_helpers"))
    from net_layers import FiLMNet

    split = json.loads(SPLIT.read_text())
    wanted = set(split["val"]) | set(split["test"])
    with open(SYMBOL_MAP, "rb") as handle:
        symbol_to_ensembl = pickle.load(handle)
    shared = Counter(symbol_to_ensembl.values())
    symbol_of = {e: s for s, e in symbol_to_ensembl.items() if shared[e] == 1}

    # A fixed-format HDF table: read whole (about 11 GB), then keep the split's lines.
    df = pd.read_hdf(args.run_dir / "generated_df" / "gene2vec_emb_mat.h5", "table")
    parsed = [KEY.fullmatch(k) for k in df.index]
    if not all(parsed):
        raise ValueError("a stage-one row key is not ('ModelID', 'gene_ENSG')")
    keep = np.array([m.group(1) in wanted for m in parsed])
    keys = [m.groups() for m, k in zip(parsed, keep) if k]
    x = df.iloc[keep, :640].to_numpy(dtype=np.float32)
    labels = df["viability score"].to_numpy(dtype=float)[keep]
    del df

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = FiLMNet(protein_size=128)
    net.load_state_dict(
        torch.load(
            args.run_dir / "models" / "gene2vec_5x.pth",
            weights_only=True,
            map_location="cpu",
        )
    )
    net = net.to(device).eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(x), 65536):
            batch = torch.from_numpy(x[start : start + 65536]).to(device)
            out = net(batch[:, :512], batch[:, 512:])[0].squeeze(1)
            predictions.append(out.double().cpu().numpy())

    rows = pd.DataFrame(
        {
            "model_id": [line for line, _ in keys],
            "ensembl": [gene for _, gene in keys],
            "prediction": np.concatenate(predictions),
            "cilantro_label": labels,
        }
    )
    rows["gene_symbol"] = rows["ensembl"].map(symbol_of)
    mapped = rows.dropna(subset=["gene_symbol"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    mapped[["model_id", "gene_symbol", "prediction", "cilantro_label"]].to_csv(
        args.out, index=False
    )
    record = {
        "rows": len(rows),
        "rows_mapped": len(mapped),
        "lines": int(rows["model_id"].nunique()),
        "lines_wanted": len(wanted),
        "genes_mapped": int(mapped["gene_symbol"].nunique()),
        "in_sample": True,
    }
    args.out.with_suffix(".json").write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps(record, indent=1))


if __name__ == "__main__":
    main()
