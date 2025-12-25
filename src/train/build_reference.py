"""
Build reference database for retrieval-based perturbation prediction.

For each condition, generate predicted profiles using the finetuned forward model.
"""

import argparse
from pathlib import Path
from typing import Dict, List
import yaml

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle

from ..data import load_perturb_data
from ..data.forward_collator import ForwardModelDataset, collate_forward_batch
from ..model.scgpt_forward import ScGPTForward


def parse_args():
    parser = argparse.ArgumentParser(description="Build retrieval reference database")
    parser.add_argument("--config", type=str, required=True, help="Config file")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Finetuned model checkpoint"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/forward/reference_db.pkl",
        help="Output path",
    )
    parser.add_argument(
        "--n_samples_per_condition",
        type=int,
        default=200,
        help="Control samples per condition",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def generate_predictions_for_condition(
    model: ScGPTForward,
    control_adata,
    condition: str,
    vocab: Dict[str, int],
    n_samples: int = 200,
    batch_size: int = 32,
    device: str = "cuda",
) -> np.ndarray:
    """
    Generate predicted perturbed profiles for a condition.

    Args:
        model: Finetuned forward model
        control_adata: AnnData of control cells
        condition: Perturbation condition
        vocab: Gene vocabulary
        n_samples: Number of control cells to sample
        batch_size: Batch size for inference
        device: Device

    Returns:
        Array of predicted expression profiles, shape (n_samples, n_genes)
    """
    model.model.eval()

    # Sample control cells
    n_control = len(control_adata)
    if n_control > n_samples:
        indices = np.random.choice(n_control, n_samples, replace=False)
    else:
        indices = np.arange(n_control)
        if n_control < n_samples:
            # Upsample if not enough control cells
            extra = np.random.choice(n_control, n_samples - n_control, replace=True)
            indices = np.concatenate([indices, extra])

    predictions = []

    # Process in batches
    for i in range(0, len(indices), batch_size):
        batch_indices = indices[i : i + batch_size]

        # Get control expressions
        control_exprs = []
        for idx in batch_indices:
            expr = (
                control_adata.X[idx].toarray().flatten()
                if hasattr(control_adata.X, "toarray")
                else control_adata.X[idx]
            )
            control_exprs.append(expr)
        control_exprs = np.array(control_exprs)

        # Bin expressions
        control_binned = []
        for expr in control_exprs:
            expr_clip = np.clip(expr, 0, None)
            max_val = expr_clip.max() if expr_clip.max() > 0 else 1.0
            binned = np.floor(expr_clip / max_val * 50).astype(int)
            binned = np.clip(binned, 0, 50)
            control_binned.append(binned)
        control_binned = np.array(control_binned)

        # Get gene IDs - use 'gene_name' column if available (gene symbols),
        # otherwise fall back to var_names (may be Ensembl IDs)
        if "gene_name" in control_adata.var.columns:
            gene_names = control_adata.var["gene_name"].tolist()
        else:
            gene_names = control_adata.var_names.tolist()
        gene_ids = np.array([vocab.get(g, vocab.get("<pad>", 0)) for g in gene_names])

        # Tokenize and pad
        from scgpt.tokenizer.gene_tokenizer import tokenize_and_pad_batch

        # Safeguard: ensure no all-zero rows exist to avoid empty tensors.
        # PyTorch transformer's to_padded_tensor fails with empty sequences.
        for i in range(len(control_binned)):
            if np.count_nonzero(control_binned[i]) == 0:
                control_binned[i, 0] = 1  # Set minimal placeholder value

        batch_data = tokenize_and_pad_batch(
            control_binned,
            gene_ids,
            max_len=1200,
            vocab=vocab,
            pad_token="<pad>",
            pad_value=-2,
            append_cls=False,
            include_zero_gene=False,
            return_pt=True,
        )

        # Move to device
        genes = batch_data["genes"].to(device)
        values = batch_data["values"].to(device)
        padding_mask = genes == vocab["<pad>"]

        # Forward pass
        with torch.no_grad():
            outputs = model(
                gene_ids=genes,
                values=values,
                padding_mask=padding_mask,
            )

        # Get predictions (unbinned expression values)
        pred = outputs["mlm_output"].cpu().numpy()  # (batch, seq_len)
        predictions.append(pred)

    predictions = np.concatenate(predictions, axis=0)  # (n_samples, seq_len)
    return predictions


def build_reference_database(
    model: ScGPTForward,
    dataset,
    conditions: List[str],
    n_samples_per_condition: int,
    device: str,
) -> Dict:
    """
    Build reference database for all conditions.

    Returns:
        Dictionary with:
        - predictions: Dict[condition, np.ndarray] of predicted profiles
        - conditions: List of condition names
        - gene_names: List of gene names
    """
    print(f"Building reference database for {len(conditions)} conditions...")

    control_adata = dataset.control_adata

    reference_db = {
        "predictions": {},
        "conditions": conditions,
        "gene_names": dataset.adata.var_names.tolist(),
    }

    for condition in tqdm(conditions, desc="Generating predictions"):
        preds = generate_predictions_for_condition(
            model,
            control_adata,
            condition,
            model.vocab,
            n_samples=n_samples_per_condition,
            device=device,
        )
        reference_db["predictions"][condition] = preds

    return reference_db


def main():
    args = parse_args()
    config = load_config(args.config)

    print("=" * 60)
    print("Building Retrieval Reference Database")
    print("=" * 60)

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load data
    print("\n[1/3] Loading dataset...")
    cond_split_config = config.get("condition_split", {})
    dataset = load_perturb_data(
        h5ad_path=config["data"]["h5ad_path"],
        condition_split_path=cond_split_config.get("output_path"),
        **{k: v for k, v in cond_split_config.items() if k != "output_path"},
    )

    # All conditions (train + val + test)
    all_conditions = (
        dataset.train_conditions + dataset.val_conditions + dataset.test_conditions
    )

    print(f"  - Total conditions: {len(all_conditions)}")
    print(f"  - Control cells: {len(dataset.control_adata)}")

    # Load model
    print("\n[2/3] Loading finetuned model...")
    model = ScGPTForward(
        checkpoint_path="model/scGPT/best_model.pt",  # Base checkpoint
        device=device,
    )

    # Load finetuned weights
    print(f"  Loading finetuned weights from {args.checkpoint}")
    finetuned_state = torch.load(args.checkpoint, map_location=device)
    model.model.load_state_dict(finetuned_state, strict=False)

    # Build reference database
    print("\n[3/3] Generating predictions...")
    reference_db = build_reference_database(
        model,
        dataset,
        all_conditions,
        n_samples_per_condition=args.n_samples_per_condition,
        device=device,
    )

    # Save database
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        pickle.dump(reference_db, f)

    print("\n" + "=" * 60)
    print(f"Reference database saved to: {output_path}")
    print(f"  - Conditions: {len(reference_db['conditions'])}")
    print(f"  - Samples per condition: {args.n_samples_per_condition}")
    print(
        f"  - Total predictions: {sum(v.shape[0] for v in reference_db['predictions'].values())}"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
