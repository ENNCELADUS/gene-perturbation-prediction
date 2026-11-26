"""Train ConvE on PrimeKG and save the entity embeddings.

The entity embeddings are used as the structural gene-prior features in
``src/preprocess_datasets.py`` (see ``PROMPTCASL_PRIMEKG_EMB``).

Run from the repository root:

    python -m scripts.train_conve_primekg --dim 768 --epochs 10 --device 0
"""

import argparse
import os
import random

import numpy as np
import torch
from pykeen.datasets import PrimeKG
from pykeen.evaluation import RankBasedEvaluator
from pykeen.models import ConvE
from pykeen.training import LCWATrainingLoop
from torch.optim import Adam

from src.config import EXTERNAL_DIR

SEED = 42


def _array(representation):
    if hasattr(representation, "weight"):
        return representation.weight.detach().cpu().numpy()
    return representation().detach().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=768, help="entity embedding dimension")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--out_dir", default=EXTERNAL_DIR)
    args = ap.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    dataset = PrimeKG(create_inverse_triples=True)
    model = ConvE(
        triples_factory=dataset.training,
        embedding_dim=args.dim,
        input_channels=1,
        output_channels=32,
        embedding_height=16,
        embedding_width=16,
        kernel_height=3,
        kernel_width=3,
        input_dropout=0.2,
        feature_map_dropout=0.2,
        output_dropout=0.3,
    ).to(device)

    optimizer = Adam(params=model.get_grad_params())
    loop = LCWATrainingLoop(model=model, optimizer=optimizer, triples_factory=dataset.training)
    loop.train(num_epochs=args.epochs, batch_size=args.batch_size,
               triples_factory=dataset.training, use_tqdm=True)

    evaluator = RankBasedEvaluator()
    result = evaluator.evaluate(
        model=model,
        mapped_triples=dataset.validation.mapped_triples,
        additional_filter_triples=dataset.training.mapped_triples,
        batch_size=8192,
        use_tqdm=True,
    )
    print(result.to_dict())

    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"dim{args.dim}"
    torch.save(model.state_dict(), os.path.join(args.out_dir, f"conv_primekg_model_{tag}.pth"))
    torch.save(optimizer.state_dict(), os.path.join(args.out_dir, f"conv_primekg_optimizer_{tag}.pth"))
    np.save(os.path.join(args.out_dir, f"conv_primekg_entity_emb{args.dim}.npy"),
            _array(model.entity_representations[0]))
    np.save(os.path.join(args.out_dir, f"conv_primekg_relation_emb{args.dim}.npy"),
            _array(model.relation_representations[0]))
    print(f"saved to {args.out_dir}")


if __name__ == "__main__":
    main()
