import torch
import yaml
import pandas as pd
import logging
import argparse
from .train_model import train_model
from datetime import datetime
from transformers import AutoModel
from .data_loader import load_data_SCL, load_data_MCL, load_data_MCTL
from .eval import evaluate_on_folds, evaluate_on_folds_mctl
from src.config import BIOMEDBERT
import os
os.environ["WANDB_MODE"] = "offline"
import wandb
import warnings
warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SCL_CELL_LINES = {"A549", "PK1", "IPC298", "MEWO", "GI1", "HS936T"}
MCTL_CELL_LINES = {"mctl"}  # trigger for the in-main MCTL scenario

# MCL run modes selected via --cell_line -> (cell_line_set, use_gplo).
# The *-gplo mode switches dataset_path to the *_GPLO shards unless --dataset_path is given.
MCL_MODES = {
    "mixed-6cl":      (["A549", "PK1", "IPC298", "MEWO", "GI1", "HS936T"], False),
    "mixed-6cl-gplo": (["A549", "PK1", "IPC298", "MEWO", "GI1", "HS936T"], True),
}
_DEFAULT_DATASET = os.path.join(PROJECT_ROOT, "data/datasets/cell_line_biomedbert_kg50shared")


def load_config(file_path):
    """Load configuration from a YAML file."""
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def setup_logger(name, log_file_path, level=logging.INFO):
    """Set up a logger that writes to a file."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # File handler for logging to a file
    file_handler = logging.FileHandler(log_file_path)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    # Add the file handler to the logger
    logger.addHandler(file_handler)
    return logger

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Training configuration for PromptSL.")
    # Dataset and model paths
    parser.add_argument("--dataset_path", type=str, default=None, help="Dataset .pt dir. If omitted: cell_line_biomedbert_kg50shared, or its *_GPLO variant when --cell_line is a *-gplo mode.")
    parser.add_argument("--biobert_path", type=str, default=BIOMEDBERT, help="Path to the pretrained BiomedBERT model.")
    parser.add_argument("--save_folder", type=str, default=os.path.join(PROJECT_ROOT, "save"), help="Path to save training results.")
    parser.add_argument("--cell_line", type=str, default="mixed-6cl", help="Run mode. MCL: mixed-6cl / mixed-6cl-gplo. SCL: any single cell line. MCTL: mctl.")
    parser.add_argument("--cell_line_set", nargs="+", default=None, help="Source cell lines for MCL/MCTL. If omitted, inferred from --cell_line mode.")
    parser.add_argument("--target_cell_lines", nargs="+", default=["GI1", "HS936T"], help="Target (test) cell lines for MCTL scenarios.")
    parser.add_argument("--negpos", type=int, default=1, help="Negative to positive sample ratio (e.g., 1, 5, 10).")
    # Hyperparameters for model and optimizer
    parser.add_argument("--llm_lr", type=float, default=1e-5, help="Learning rate for LLM encoder.")
    parser.add_argument("--omics_lr", type=float, default=1e-4, help="Learning rate for Omics encoder.")
    parser.add_argument("--geneagg_lr", type=float, default=1e-4, help="Learning rat    e for gene attention aggregator.")
    parser.add_argument("--pred_lr", type=float, default=1e-4, help="Learning rate for predictor.")
    parser.add_argument("--batch_size", type=int, default=24, help="Batch size for training.")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of epochs for training.")
    parser.add_argument("--early_stop_patience", type=int, default=6, help="Early stop patience.")
    parser.add_argument("--lr_factor", type=float, default=0.5, help="Learning rate reducing factor.")
    parser.add_argument("--lr_patience", type=int, default=3, help="Patience for learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay.")
    parser.add_argument("--dropout_rate", type=float, default=0.1, help="Dropout rate.")
    parser.add_argument("--device", type=int, default=0, help="Select the CUDA device index.")
    parser.add_argument("--omics_data", type=int, default=0, help="Use pre-computed AE omics embeddings (mutually exclusive with omics_encoder).")
    parser.add_argument("--omics_encoder", type=int, default=1, help="Use trainable MLP omics encoders (paper default).")
    # Model architecture
    parser.add_argument("--geneagg_in", type=int, default=3072, help="Input dim for gene attention aggregator.")
    parser.add_argument("--geneagg_hidden", type=int, default=768, help="Hidden dim for gene attention aggregator.")
    parser.add_argument("--geneagg_ff", type=int, default=1024, help="FF dim for gene attention aggregator.")
    parser.add_argument("--pred_in", type=int, default=5376, help="Input dim for predictor. pair_ctx(2304) + expressed_geneA/B(768x2) + geneA/B_agg(768x2) = 5376.")
    parser.add_argument("--pred_hidden", type=int, default=1024, help="Hidden dim for predictor.")
    parser.add_argument("--freeze_llm", action="store_true", help="If set, freeze the LLM encoder (default: False, i.e., trainable)")
    # LoRA for parameter-efficient fine-tuning
    parser.add_argument("--use_lora", action="store_true", help="Apply LoRA to the LLM encoder (parameter-efficient fine-tuning).")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank.")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha scaling factor.")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout rate.")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing to trade compute for activation memory (~24x reduction for 24-layer model).")
    parser.add_argument("--no_save_model", action="store_true", help="Skip saving model checkpoints (useful during development to save disk space).")
    parser.add_argument("--no_context", action="store_true", help="Ablation: remove all contextual information (no LLM text, no omics). Gene queries only attend to a dummy zero token — isolates the contribution of gene-only embeddings (GenePT + KG).")
    parser.add_argument("--no_text", action="store_true", help="Ablation: text OFF, pathway ON — sufficiency test (pair_pathway_evidence only). Gene queries read ONLY the pair-conditioned pathway-evidence tokens (+ a dummy zero token); the LLM text branch is dropped. Symmetric to --no_omics (text-only): tests whether the pathway channel alone suffices on conflict pairs.")
    parser.add_argument("--no_kg", action="store_true", help="Ablation: zero out all KG embeddings (entity + neighbor). Keeps GenePT gene embeddings.")
    parser.add_argument("--random_genept", action="store_true", help="Ablation: replace GenePT gene embeddings with a per-gene FIXED random vector (seeded by gene name, scale-matched to real GenePT). Preserves gene identity but removes GenePT's pretrained knowledge -> isolates the value of GenePT's prior. Keeps KG.")
    parser.add_argument("--no_genept", action="store_true", help="Ablation: zero out GenePT gene embeddings entirely. Gene identity comes from KG only -> tests whether KG alone is a sufficient gene representation. Mutually exclusive with --random_genept.")
    parser.add_argument("--no_cross_attn", action="store_true", help="Ablation: drop the gene->context cross-attention (Step 3 in GenePairContextAttention). Gene queries never read context tokens; keeps gene-gene self-attn. Isolates the value of the cross-attention fusion mechanism.")
    parser.add_argument("--concat_context", action="store_true", help="Ablation: replace gene->context cross-attention with masked mean-pooling of context tokens added to each gene query. Tests whether attention beats naive pooling of context. Mutually exclusive with --no_cross_attn.")
    parser.add_argument("--run_tag", type=str, default=None, help="Optional label appended to the save-folder name (and thus the eval result id), e.g. 'ablation_no_kg'. Lets parallel ablation runs stay distinguishable on disk.")
    parser.add_argument("--omics_in_kv", action="store_true", help="Inject omics tokens into cross-attention KV context instead of FiLM modulation.")
    # pathway_tokens mode (mutually exclusive with omics_data / omics_encoder)
    parser.add_argument("--omics_mode", type=str, default=None,
                        choices=["pathway_tokens", "geneset_tokens", "ge_pathway_selfatt", "pair_pathway_evidence"],
                        help="Pathway omics mode: 'pathway_tokens' uses scalar [P,3] scores; 'geneset_tokens' uses gene-level [P,G_max,3] attention pooling; 'ge_pathway_selfatt' uses GE-only [P,6] stats + text-pathway self-attention fusion; 'pair_pathway_evidence' uses pair-conditioned top-K Reactome evidence tokens (forces Reactome).")
    parser.add_argument("--pathway_set", type=str, default="hallmark",
                        choices=["hallmark", "kegg", "reactome"],
                        help="Pathway gene set to use when --omics_mode=pathway_tokens. hallmark=P50, kegg=P178.")
    parser.add_argument("--pathway_topk", type=int, default=16,
                        help="Top-K pair-relevant Reactome pathways per sample (pair_pathway_evidence mode).")
    parser.add_argument("--start_fold", type=int, default=0,
                        help="Start from this fold index (0-4) to resume an interrupted run.")
    parser.add_argument("--resume_dir", type=str, default=None,
                        help="Existing save folder to resume into (reuses its log and CSV). Requires --start_fold > 0.")
    return parser.parse_args()



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
    msg = f"LoRA applied: {trainable:,} trainable / {total:,} total params ({100*trainable/total:.2f}%)"
    print(msg)
    if logger:
        logger.info(msg)
    return model


def enable_gradient_checkpointing(model, logger=None):
    # For PEFT models, must enable input grads first so checkpointing works on frozen base
    if hasattr(model, 'enable_input_require_grads'):
        model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    msg = "Gradient checkpointing enabled."
    print(msg)
    if logger:
        logger.info(msg)


def create_save_folder(device, save_folder, scenario, cell_line, tag=None):
    current_time = datetime.now().strftime(f"%Y%m%d_%H%M%S")
    folder_name = f"{current_time}_{tag}" if tag else current_time
    save_folder_path = os.path.join(save_folder, scenario, cell_line, folder_name)
    os.makedirs(save_folder_path, exist_ok=True)
    return save_folder_path, folder_name

def summarize_cv_results(final_results_path):
    df = pd.read_csv(final_results_path).dropna(how='all')
    if df.empty:
        print("No results found.")
        return
    # Compute mean and std on metric columns only (exclude Fold)
    metric_cols = [c for c in df.columns if c != "Fold"]
    mean_vals = df[metric_cols].mean()
    std_vals = df[metric_cols].std()

    # Append to file with explicit "mean"/"std" label in Fold column
    with open(final_results_path, "a") as f:
        f.write("\n")
        f.write("mean," + ",".join([f"{v:.4f}" for v in mean_vals]) + "\n")
        f.write("std," + ",".join([f"{v:.4f}" for v in std_vals]) + "\n")

    # Prepare wandb log
    log_dict = {
        "mean_acc": mean_vals["Val Accuracy"],
        "mean_f1": mean_vals["Val F1"],
        "mean_auc": mean_vals["Val ROC-AUC"],
        "mean_aupr": mean_vals["Val AUPR"],
        "mean_bacc": mean_vals["Val BACC"],
        "mean_recall": mean_vals["Val Recall"],
        "std_acc": std_vals["Val Accuracy"],
        "std_f1": std_vals["Val F1"],
        "std_auc": std_vals["Val ROC-AUC"],
        "std_aupr": std_vals["Val AUPR"],
        "std_bacc": std_vals["Val BACC"],
        "std_recall": std_vals["Val Recall"],
        "folds": len(df)
    }

    wandb.log(log_dict)
    print("Logged summary metrics to wandb.")
    
def infer_scenario_from_cell_line(cell_line):
    if cell_line in MCL_MODES:
        return "MCL"
    elif cell_line in MCTL_CELL_LINES:
        return "MCTL"
    elif cell_line in SCL_CELL_LINES:
        return "SCL"
    else:
        raise ValueError(f"Unknown cell line: '{cell_line}'. Add it to SCL_CELL_LINES or MCTL_CELL_LINES in main.py.")


def main():
    args = parse_arguments()
    if args.omics_data and args.omics_encoder:
        raise ValueError("--omics_data and --omics_encoder are mutually exclusive. Set at most one to 1.")
    if getattr(args, "random_genept", False) and getattr(args, "no_genept", False):
        raise ValueError("--random_genept and --no_genept are mutually exclusive.")
    if getattr(args, "no_cross_attn", False) and getattr(args, "concat_context", False):
        raise ValueError("--no_cross_attn and --concat_context are mutually exclusive.")
    if args.omics_mode in ('pathway_tokens', 'geneset_tokens', 'ge_pathway_selfatt', 'pair_pathway_evidence'):
        if args.omics_data or args.omics_encoder:
            raise ValueError(f"--omics_mode={args.omics_mode} is mutually exclusive with --omics_data and --omics_encoder.")
        args.omics_data    = 0
        args.omics_encoder = 0
        if args.omics_mode == 'pair_pathway_evidence':
            args.pathway_set = 'reactome'   # builder reads fixed reactome_* artifacts
    args.scenario = infer_scenario_from_cell_line(args.cell_line)

    # Resolve MCL run-mode defaults (explicit --cell_line_set / --dataset_path win)
    if args.cell_line_set is None:
        args.cell_line_set = (list(MCL_MODES[args.cell_line][0])
                              if args.cell_line in MCL_MODES
                              else ["A549", "PK1", "IPC298", "MEWO", "GI1", "HS936T"])
    _use_gplo = args.cell_line in MCL_MODES and MCL_MODES[args.cell_line][1]
    if args.dataset_path is None:
        args.dataset_path = _DEFAULT_DATASET + "_GPLO" if _use_gplo else _DEFAULT_DATASET
    elif _use_gplo and "GPLO" not in os.path.basename(args.dataset_path.rstrip("/")):
        print(f"[warn] --cell_line {args.cell_line} is a GPLO mode but --dataset_path "
              f"does not look like a GPLO dir: {args.dataset_path}")

    # Create save folder
    if args.resume_dir:
        save_folder = args.resume_dir
        print(f"Resuming run in: {save_folder}")
    else:
        tag = f"{args.omics_mode}_{args.pathway_set}" if getattr(args, 'omics_mode', None) else None
        if tag and args.omics_mode == 'pair_pathway_evidence':
            tag = f"{tag}_K{args.pathway_topk}"
        if getattr(args, 'run_tag', None):
            tag = f"{tag}_{args.run_tag}" if tag else args.run_tag
        save_folder, current_time = create_save_folder(args.device, args.save_folder, args.scenario, args.cell_line, tag=tag)
        print(f"Results will be saved in: {save_folder}")
    # Set up logging
    log_file_path = os.path.join(save_folder, "training.log")
    logger = setup_logger("training_logger", log_file_path)
    logger.info("Starting training process.")
    logger.info("Command-line arguments:")
    for k, v in vars(args).items():
        logger.info(f"  {k}: {v}")
    # Set device
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    # Set up wandb
    wandb.init(
        project="PromptCASL",
        config=vars(args)
    )
    # Data loading and training
    if args.scenario in ["SCL", "MCL"]:
        for fold_idx in range(args.start_fold, 5):
            logger.info(f"Running fold {fold_idx + 1}")
            print(args.scenario, args.cell_line, f"Running fold {fold_idx + 1}")
            args.fold_idx = fold_idx
            # Load BioBERT
            biobert = AutoModel.from_pretrained(args.biobert_path).to(device)
            if getattr(args, 'use_lora', False):
                biobert = apply_lora(biobert, args, logger)
            if getattr(args, 'gradient_checkpointing', False):
                enable_gradient_checkpointing(biobert, logger)
            if args.scenario == "SCL":
                val_loader, train_loader = load_data_SCL(args.dataset_path, args.cell_line, args.batch_size, args.negpos, fold_idx)
            elif args.scenario == "MCL":
                val_loader, train_loader = load_data_MCL(args.dataset_path, args.cell_line_set, args.batch_size, args.negpos, fold_idx)
            train_model(
                train_loader=train_loader,
                val_loader=val_loader,
                bioLLM=biobert,
                args=args,
                save_path=save_folder,
                logger=logger,
                omics_data=args.omics_data,
                omics_encoder=args.omics_encoder
            )
            logger.info(f"Fold {fold_idx + 1} training completed successfully.")
        summarize_cv_results(os.path.join(save_folder, "final_results.csv"))
    elif args.scenario == "MCTL":
        args.fold_idx = None
        # Load BioBERT
        biobert = AutoModel.from_pretrained(args.biobert_path).to(device)
        if getattr(args, 'use_lora', False):
            biobert = apply_lora(biobert, args, logger)
        if getattr(args, 'gradient_checkpointing', False):
            enable_gradient_checkpointing(biobert, logger)
        val_loader, train_loader = load_data_MCTL(args.dataset_path, args.target_cell_lines, args.cell_line_set, args.negpos, args.batch_size)
        train_model(
            train_loader=train_loader,
            val_loader=val_loader,
            bioLLM=biobert,
            args=args,
            save_path=save_folder,
            logger=logger,
            omics_data=args.omics_data,
            omics_encoder=args.omics_encoder
        )
    logger.info("Training completed successfully.")
    
    
    ### eval
    base_model_dir = os.path.join(PROJECT_ROOT, "save", args.scenario)
    result_root = os.path.join(PROJECT_ROOT, "results")
    model_ids = [os.path.basename(save_folder)]
    if args.scenario == "MCL":
        evaluate_on_folds(args.cell_line_set, args.cell_line, args.scenario, model_ids, base_model_dir, result_root, device, biobert)
    elif args.scenario == "MCTL":
        evaluate_on_folds_mctl(args.target_cell_lines, "mctl", args.scenario, model_ids, base_model_dir, result_root, device, biobert)
    # SCL: per-fold results already saved to final_results.csv





if __name__ == "__main__":
    main()




