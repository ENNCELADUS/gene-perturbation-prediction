"""Unperturbed Geneformer embedding of every line, which the released data_generator
reads.

data_generator.remove_nan() loads orig_embedding_pretrained.csv; no released notebook
writes it. It is produced with the extractor settings the released perturber uses (cell
mode, penultimate layer, patient_id label). data_generator puts it in the AnnData's
obsm, which requires the AnnData's line order; the extractor sorts lines by length, so
the rows are put back in that order.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    DICTIONARIES,
    GENEFORMER_MODEL,
    configure,
    mark_done,
    paths,
    run_config,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    configure(args.run_dir)
    p = paths(args.run_dir)

    import anndata as ad
    from datasets import load_from_disk
    from geneformer import EmbExtractor

    dataset = load_from_disk(str(p["tokenized"]))
    embex = EmbExtractor(
        model_type="Pretrained",
        num_classes=0,
        emb_mode="cell",
        emb_layer=-1,
        emb_label=["patient_id"],
        forward_batch_size=50,
        max_ncells=len(dataset),
        nproc=16,
        token_dictionary_file=str(
            DICTIONARIES[run_config(args.run_dir)["dictionary"]]["token"]
        ),
    )
    p["orig_embs"].parent.mkdir(parents=True, exist_ok=True)
    embs = embex.extract_embs(
        str(GENEFORMER_MODEL),
        str(p["tokenized"]),
        str(p["orig_embs"].parent),
        p["orig_embs"].stem,
    )
    order = ad.read_h5ad(p["sl_hvg_h5ad"], backed="r").obs_names
    embs = (
        embs.set_index("patient_id").loc[order].rename_axis("patient_id").reset_index()
    )
    embs.to_csv(p["orig_embs"])
    mark_done(p["orig_embs"].parent, lines=int(len(embs)), dims=int(embs.shape[1] - 1))


if __name__ == "__main__":
    main()
