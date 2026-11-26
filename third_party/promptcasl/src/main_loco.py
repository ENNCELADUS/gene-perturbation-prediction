"""LOCO (leave-one-cell-line-out) transfer training and evaluation.

--loco_mode np1 trains on the balanced NP1 folds; --loco_mode full trains on the
imbalanced full fold.
"""

import torch
import yaml
import pandas as pd
import logging
import argparse
import os
import warnings
from datetime import datetime
from transformers import AutoModel
from torch.utils.data import DataLoader, ConcatDataset

from .train_model import train_model, validate_model, load_omics_emb_dict, load_omics_dict
from .data_loader import load_data_LOCO, GenePairDataset
from .model import (
    LLMEncoder, GeneAttentionAggregator, GenePairContextAttention,
    Predictor, OmicsEncoder, PairFiLM,
)
from .eval import load_models_from_log, _load_pathway_encoder
from src.config import BIOMEDBERT

warnings.filterwarnings("ignore")
os.environ["WANDB_MODE"] = "offline"
import wandb

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ALL_CELL_LINES = ["A549", "PK1", "IPC298", "MEWO", "GI1", "HS936T"]

DEFAULT_NP1_DATASET   = os.path.join(PROJECT_ROOT, "data/datasets/cell_line_biomedbert_kg50shared")
DEFAULT_FULL_DATASET  = os.path.join(PROJECT_ROOT, "data/datasets/cell_line_biomedbert_v1_kg50shared_full_eval")


# ── Helpers ──────────────────────────────────────────────────────────────────

def setup_logger(name, log_file_path, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    fh = logging.FileHandler(log_file_path)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)
    return logger


def apply_lora(model, args, logger=None):
    from peft import get_peft_model, LoraConfig
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["query", "value"],
        lora_dropout=args.lora_dropout,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    msg = f"LoRA applied: {trainable:,} / {total:,} params ({100*trainable/total:.2f}%)"
    print(msg)
    if logger:
        logger.info(msg)
    return model


def enable_gradient_checkpointing(model, logger=None):
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    msg = "Gradient checkpointing enabled."
    print(msg)
    if logger:
        logger.info(msg)


# ── Argument parsing ─────────────────────────────────────────────────────────

def parse_arguments():
    parser = argparse.ArgumentParser(description="LOCO transfer training for PromptCASL.")

    # LOCO-specific
    parser.add_argument("--loco_mode", type=str, required=True, choices=["np1", "full"],
                        help="np1: balanced NP1 train data (2.1); full: imbalanced full train data (2.2).")
    parser.add_argument("--target_cell_line", type=str, default=None,
                        help="Target (held-out) cell line. If not set, runs all cell lines in --cell_line_set sequentially.")
    parser.add_argument("--cell_line_set", nargs="+",
                        default=["A549", "PK1", "IPC298", "MEWO", "GI1", "HS936T"],
                        help="Cell lines to include in LOCO. Leave-one-out iterates over this set. Default: all cell lines.")
    parser.add_argument("--np1_dataset_path", type=str, default=DEFAULT_NP1_DATASET,
                        help="Directory containing NP1 balanced .pt files (F1-F5 per cell line).")
    parser.add_argument("--full_eval_dataset_path", type=str, default=DEFAULT_FULL_DATASET,
                        help="Directory containing full imbalanced .pt files (F1 per cell line) for eval.")
    parser.add_argument("--eval_fulltest", action="store_true",
                        help="np1 mode only: ALSO evaluate on the target's full imbalanced set (train-NP1/test-full). Default off → np1 mode evaluates on target NP1 only.")

    # Eval-only mode
    parser.add_argument("--eval_only", action="store_true",
                        help="Skip training; load checkpoint from --model_dir and evaluate.")
    parser.add_argument("--model_dir", type=str, default=None,
                        help="Path to model checkpoint dir (required for --eval_only).")

    # Shared model/training args (same defaults as main.py)
    parser.add_argument("--biobert_path", type=str, default=BIOMEDBERT)
    parser.add_argument("--save_folder", type=str, default=os.path.join(PROJECT_ROOT, "save"))
    parser.add_argument("--llm_lr", type=float, default=1e-5)
    parser.add_argument("--omics_lr", type=float, default=1e-4)
    parser.add_argument("--geneagg_lr", type=float, default=1e-4)
    parser.add_argument("--pred_lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--early_stop_patience", type=int, default=6)
    parser.add_argument("--lr_factor", type=float, default=0.5)
    parser.add_argument("--lr_patience", type=int, default=3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--dropout_rate", type=float, default=0.1)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--omics_data", type=int, default=0)
    parser.add_argument("--omics_encoder", type=int, default=1)
    parser.add_argument("--geneagg_in", type=int, default=3072)
    parser.add_argument("--geneagg_hidden", type=int, default=768)
    parser.add_argument("--geneagg_ff", type=int, default=1024)
    parser.add_argument("--pred_in", type=int, default=5376)
    parser.add_argument("--pred_hidden", type=int, default=1024)
    parser.add_argument("--freeze_llm", action="store_true")
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--no_save_model", action="store_true")
    parser.add_argument("--save_pred", action="store_true",
                        help="Dump per-pair test predictions (gene_a,gene_b,label,score) in SLBench format.")
    parser.add_argument("--pred_dir", type=str, default=None,
                        help="Root dir for per-pair predictions. Default: results/loco_subset_pred/<result_subdir>.")
    parser.add_argument("--no_kg", action="store_true",
                        help="Ablation: zero out all KG embeddings (entity + neighbor). Keeps GenePT gene embeddings.")
    parser.add_argument("--no_context", action="store_true",
                        help="Ablation: remove ALL context (no LLM text, no omics, no pathway). Gene-only (GenePT+KG). Must train AND eval with this flag.")
    parser.add_argument("--omics_in_kv", action="store_true",
                        help="Inject omics as KV context token instead of FiLM modulation.")
    parser.add_argument("--omics_mode", type=str, default=None,
                        choices=["pathway_tokens", "geneset_tokens", "ge_pathway_selfatt", "pair_pathway_evidence"],
                        help="Pathway omics mode: 'pathway_tokens' uses scalar [P,3] scores; 'geneset_tokens' uses gene-level attention pooling; 'ge_pathway_selfatt' uses GE-only [P,6] stats + text-pathway self-attention fusion; 'pair_pathway_evidence' uses pair-conditioned top-K Reactome evidence tokens.")
    parser.add_argument("--pathway_set", type=str, default="hallmark",
                        choices=["hallmark", "kegg"],
                        help="Pathway gene set (hallmark=P50, kegg=P178). Used when --omics_mode is set.")
    parser.add_argument("--pathway_topk", type=int, default=16,
                        help="Top-K pair-relevant Reactome pathways per sample (pair_pathway_evidence mode).")
    parser.add_argument("--result_tag", type=str, default=None,
                        help="Optional suffix for result dir, e.g. 'kv' → results/LOCO_full_kv/.")
    return parser.parse_args()


# ── Single-target train + eval ───────────────────────────────────────────────

def run_loco_target(args, target_cell_line, device, logger=None):
    """Train and evaluate for a single held-out target cell line."""

    # ── Dataset path: np1 mode uses NP1 folds; full mode uses full-eval folds
    train_dataset_path = args.np1_dataset_path if args.loco_mode == "np1" else args.full_eval_dataset_path

    print(f"\n{'='*60}")
    print(f"LOCO-{args.loco_mode.upper()}  target={target_cell_line}")
    print(f"  train data : {train_dataset_path}")
    print(f"  eval  data : {args.full_eval_dataset_path}")
    print(f"{'='*60}")

    val_loader, train_loader = load_data_LOCO(
        dataset_path=train_dataset_path,
        target_cell_line=target_cell_line,
        all_cell_lines=args.cell_line_set,
        batch_size=args.batch_size,
    )

    # ── Save folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(
        args.save_folder, f"LOCO_{args.loco_mode}", target_cell_line, timestamp
    )
    os.makedirs(save_dir, exist_ok=True)

    log_path = os.path.join(save_dir, "training.log")
    local_logger = setup_logger(f"loco_{target_cell_line}", log_path)
    local_logger.info(f"LOCO mode={args.loco_mode}  target={target_cell_line}")
    local_logger.info("Command-line arguments:")
    for k, v in vars(args).items():
        local_logger.info(f"  {k}: {v}")
    local_logger.info(f"  target_cell_line: {target_cell_line}")

    # ── BioLLM
    biobert = AutoModel.from_pretrained(args.biobert_path).to(device)
    if getattr(args, "use_lora", False):
        biobert = apply_lora(biobert, args, local_logger)
    if getattr(args, "gradient_checkpointing", False):
        enable_gradient_checkpointing(biobert, local_logger)

    # ── Train
    args.scenario = "LOCO"
    args.fold_idx = None
    args.dataset_path = train_dataset_path

    wandb.init(
        project="PromptCASL",
        name=f"LOCO_{args.loco_mode}_{target_cell_line}_{timestamp}",
        config={**vars(args), "target_cell_line": target_cell_line},
        reinit=True,
    )

    train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        bioLLM=biobert,
        args=args,
        save_path=save_dir,
        logger=local_logger,
        omics_data=args.omics_data,
        omics_encoder=args.omics_encoder,
    )
    local_logger.info("Training complete.")

    # ── Eval on full imbalanced dataset
    _eval_loco(args, save_dir, target_cell_line, device, biobert, local_logger)

    return save_dir


def _collect_eval_sets(args, target_cell_line):
    """Return [(test_set_name, dataset), ...] to evaluate on.

    full mode -> target FULL (imbalanced).
    np1  mode -> target NP1 (balanced) by default; ALSO target FULL
                 (the train-NP1/test-full metric) only when --eval_fulltest is set.
    """
    def _load_full():
        p = os.path.join(args.full_eval_dataset_path, f"{target_cell_line}_F1.pt")
        if os.path.exists(p):
            return ("full", torch.load(p, map_location="cpu", weights_only=False))
        print(f"  [eval] full eval .pt not found: {p}")
        return None

    def _load_np1():
        folds = []
        for k in range(1, 6):
            p = os.path.join(args.np1_dataset_path, f"{target_cell_line}_F{k}.pt")
            if os.path.exists(p):
                folds.append(torch.load(p, map_location="cpu", weights_only=False))
        if folds:
            return ("NP1", ConcatDataset(folds) if len(folds) > 1 else folds[0])
        print(f"  [eval] target NP1 .pt not found under {args.np1_dataset_path}")
        return None

    sets = []
    if args.loco_mode == "np1":
        s = _load_np1()                         # default: NP1 balanced test only
        if s:
            sets.append(s)
        if getattr(args, "eval_fulltest", False):   # opt-in: train-NP1/test-full
            s = _load_full()
            if s:
                sets.append(s)
    else:                                       # full mode: test on target full
        s = _load_full()
        if s:
            sets.append(s)
    return sets


def _eval_loco(args, model_dir, target_cell_line, device, bioLLM, logger=None):
    """Load checkpoint and evaluate on the sets from _collect_eval_sets:
    full mode -> target full; np1 mode -> target NP1 (default), plus target full
    only when --eval_fulltest is set (train-NP1/test-full)."""
    try:
        (llm_encoder, gene_agg, gene_ctx_attn, predictor,
         ge_encoder, cnv_encoder, mu_encoder, film_layer, ckpt_args) = load_models_from_log(
            model_dir, bioLLM, scenario="LOCO", fold_idx=1, device=device, single_model=True
        )
    except Exception as e:
        print(f"  [eval] Failed to load checkpoint: {e}")
        if logger:
            logger.warning(f"Eval skipped: {e}")
        return

    eval_sets = _collect_eval_sets(args, target_cell_line)
    if not eval_sets:
        print(f"  [eval] No eval datasets found for {target_cell_line}; skipping.")
        return

    omics_emb_dict = load_omics_emb_dict() if (hasattr(ckpt_args, "omics_data") and ckpt_args.omics_data) else None
    omics_dict     = load_omics_dict()     if (hasattr(ckpt_args, "omics_encoder") and ckpt_args.omics_encoder) else None

    pathway_encoder, pathway_dict, geneset_dict, text_pathway_fusion, pair_builder = _load_pathway_encoder(
        model_dir, ckpt_args, scenario="LOCO", fold_idx=1, device=device, single_model=True
    )
    omics_mode = getattr(ckpt_args, 'omics_mode', None) or None
    # Ablation flags: use what the model was TRAINED with (ckpt_args), fall back to CLI.
    eval_no_context = getattr(ckpt_args, 'no_context', False) or getattr(args, 'no_context', False)
    eval_no_kg      = getattr(ckpt_args, 'no_kg', False)      or getattr(args, 'no_kg', False)

    def _run_eval(ds):
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
        return validate_model(
            loader, llm_encoder, gene_agg, gene_ctx_attn, predictor, device,
            omics_emb_dict=omics_emb_dict, omics_dict=omics_dict,
            ge_encoder=ge_encoder, cnv_encoder=cnv_encoder, mu_encoder=mu_encoder,
            film_layer=film_layer,
            pathway_encoder=pathway_encoder, pathway_dict=pathway_dict,
            omics_mode=omics_mode, geneset_dict=geneset_dict,
            text_pathway_fusion=text_pathway_fusion, pair_builder=pair_builder,
            no_context=eval_no_context, no_kg=eval_no_kg,
            return_preds=getattr(args, "save_pred", False),
        )

    def _save_predictions(test_set_name, metrics):
        """Write per-pair predictions in SL_benchmark format:
        columns = gene_a_id,gene_b_id,gene_a,gene_b,label,score (ids blank; we use symbols)."""
        rec = []
        for gp, lb, sc in zip(metrics["gene_pairs"], metrics["labels"], metrics["probs"]):
            a, b = (str(gp).split("|", 1) + [""])[:2]
            rec.append(("", "", a, b, int(lb), float(sc)))
        pdf = pd.DataFrame(rec, columns=["gene_a_id", "gene_b_id", "gene_a", "gene_b", "label", "score"])
        suffix = "" if str(test_set_name).upper() == "NP1" else "_full"
        result_subdir = f"LOCO_{args.loco_mode}" + (f"_{args.result_tag}" if args.result_tag else "")
        pred_root = args.pred_dir or os.path.join(PROJECT_ROOT, "results", "loco_subset_pred", result_subdir)
        pred_subdir = os.path.join(pred_root, f"MCTL_LOCO_{target_cell_line}{suffix}")
        os.makedirs(pred_subdir, exist_ok=True)
        out = os.path.join(pred_subdir, "test_predictions.csv")
        pdf.to_csv(out, index=False)
        print(f"  Pred saved: {out}  ({len(pdf)} rows)")

    rows = []
    for test_set_name, ds in eval_sets:
        metrics = _run_eval(ds)
        if getattr(args, "save_pred", False):
            _save_predictions(test_set_name, metrics)
        rows.append({
            "Cell Line": target_cell_line,
            "LOCO Mode": args.loco_mode,
            "Test Set": test_set_name,          # "full" (held-out) / "NP1" (balanced)
            "AUC": metrics["roc_auc"],
            "AUPR": metrics["aupr"],
            "F1": metrics["f1"],
            "BACC": metrics["bacc"],
            "Acc": metrics["accuracy"],
            "Recall": metrics["recall"],
            "Val Loss": metrics["loss"],
        })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    result_subdir = f"LOCO_{args.loco_mode}" + (f"_{args.result_tag}" if args.result_tag else "")
    result_dir = os.path.join(PROJECT_ROOT, "results", result_subdir)
    os.makedirs(result_dir, exist_ok=True)
    result_path = os.path.join(result_dir, f"{target_cell_line}.csv")
    if os.path.exists(result_path):
        df.to_csv(result_path, mode="a", header=False, index=False)
    else:
        df.to_csv(result_path, index=False)
    print(f"  Result saved: {result_path}")
    if logger:
        logger.info(f"Eval results: {rows}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_arguments()

    if args.omics_data and args.omics_encoder:
        raise ValueError("--omics_data and --omics_encoder are mutually exclusive.")
    if args.omics_mode in ('pathway_tokens', 'geneset_tokens', 'ge_pathway_selfatt', 'pair_pathway_evidence'):
        if args.omics_data or args.omics_encoder:
            raise ValueError(f"--omics_mode={args.omics_mode} is mutually exclusive with --omics_data and --omics_encoder.")
        if args.omics_mode == 'pair_pathway_evidence':
            args.pathway_set = 'reactome'   # builder reads fixed reactome_* artifacts

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    args.device = device

    targets = [args.target_cell_line] if args.target_cell_line else args.cell_line_set

    if args.eval_only:
        if args.model_dir is None:
            raise ValueError("--model_dir is required with --eval_only.")
        if args.target_cell_line is None:
            raise ValueError("--target_cell_line is required with --eval_only.")
        biobert = AutoModel.from_pretrained(args.biobert_path).to(device)
        _eval_loco(args, args.model_dir, args.target_cell_line, device, biobert, logger=None)
        return

    for target in targets:
        run_loco_target(args, target, device)


if __name__ == "__main__":
    main()
