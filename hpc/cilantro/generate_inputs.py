"""Released notebook 2_perturber, second half: collect per-gene delta embeddings into
sub_embs_only.csv."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import GENEFORMER_MODEL, configure, mark_done, paths  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    configure(args.run_dir)
    p = paths(args.run_dir)

    from data_generator import data_generator

    dg = data_generator(
        is_hvg=False,
        ko_file_dir=str(p["perturb"] / "perturbed_embs") + "/",
        sub_file_dir=str(p["perturb"] / "subtracted_embs") + "/",
        output_dir=str(p["generated"]),
        model_path=str(GENEFORMER_MODEL),
        model_type="Pretrained",
        emb_mode="cell",
        emb_dim=512,
        n_classes=0,
        dataset=str(p["tokenized"]),
    )
    dg.proc_input_subs()
    mark_done(
        p["generated"] / "sub_embs",
        rows=len(dg.adata.uns["viability_dict"]),
        genes=len(dg.adata.uns["embedding_differences"]),
    )


if __name__ == "__main__":
    main()
