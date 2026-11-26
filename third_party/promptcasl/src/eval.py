import torch
import torch.nn as nn
import os
from transformers import AutoModel, AutoTokenizer
from .data_loader import load_data_SCL, load_data_MCL, load_data_MCTL, GenePairDataset
from torch.utils.data import DataLoader
from .train_model import eval_model, validate_model, load_omics_emb_dict, load_omics_dict
from .model import LLMEncoder, GeneAttentionAggregator, GenePairContextAttention, Predictor, OmicsEncoder, PairFiLM, PathwayTokenEncoder, GeneSetEncoder, GEPathwayTokenEncoder, TextPathwayFusion, PairPathwayEvidenceEncoder
from .pathway_utils import load_pathway_omics_dict, load_geneset_omics_dict, load_ge_pathway_dict_selfatt
from .pair_pathway import PairPathwayEvidenceBuilder
from src.config import BIOMEDBERT
import logging
import re
from types import SimpleNamespace
import warnings
import pandas as pd
warnings.filterwarnings("ignore")

def parse_training_log(log_path):
    args_dict = {}
    with open(log_path, "r") as f:
        lines = f.readlines()
    start = None
    for i, line in enumerate(lines):
        if "Command-line arguments:" in line:
            start = i + 1
            break
    if start is None:
        raise ValueError("Could not find command-line arguments in log file.")
    for line in lines[start:]:
        match = re.match(r".*INFO -\s+(\w+):\s+(.+)", line)
        if match:
            key, value = match.groups()
            if value.isdigit():
                value = int(value)
            else:
                try:
                    value = float(value)
                except ValueError:
                    if value.lower() == "true":
                        value = True
                    elif value.lower() == "false":
                        value = False
            args_dict[key] = value
        else:
            continue
    return args_dict

def load_models_from_log(model_dir, bioLLM, scenario, fold_idx, device, single_model=False):
    log_file = None
    for f in os.listdir(model_dir):
        if f.endswith("training.log"):
            log_file = os.path.join(model_dir, f)
            break
    if log_file is None:
        raise FileNotFoundError("No training.log found in model directory.")
    args_dict = parse_training_log(log_file)
    args = SimpleNamespace(**args_dict)
    if getattr(args, 'use_lora', False):
        from peft import get_peft_model, LoraConfig
        lora_config = LoraConfig(
            r=getattr(args, 'lora_r', 16),
            lora_alpha=getattr(args, 'lora_alpha', 32),
            target_modules=["query", "value"],
            lora_dropout=getattr(args, 'lora_dropout', 0.05),
            bias="none",
        )
        bioLLM = get_peft_model(bioLLM, lora_config)
        print(f"LoRA reconstructed for eval (r={args.lora_r}, alpha={args.lora_alpha})")
    if single_model:
        filename = f"model_{scenario}.pt"
    else:
        filename = f"model_{scenario}_F{fold_idx}.pt"
    model_path = os.path.join(model_dir, filename)
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    llm_encoder = LLMEncoder(bioLLM).to(device)
    gene_agg = GeneAttentionAggregator(args.geneagg_in, args.geneagg_hidden, args.geneagg_ff).to(device)
    dropout = getattr(args, "dropout_rate", 0.1)
    gene_ctx_attn = GenePairContextAttention(
        hidden_dim=args.geneagg_hidden, ff_dim=args.geneagg_ff, dropout=dropout,
        no_cross_attn=getattr(args, "no_cross_attn", False),
        concat_context=getattr(args, "concat_context", False),
    ).to(device)
    predictor = Predictor(args.pred_in, args.pred_hidden).to(device)

    try:
        llm_encoder.load_state_dict(checkpoint["llm_encoder"])
    except (KeyError, RuntimeError):
        print("LLM Encoder state dict keys do not match.")
    gene_agg.load_state_dict(checkpoint["gene_agg"])
    if "gene_ctx_attn" in checkpoint:
        gene_ctx_attn.load_state_dict(checkpoint["gene_ctx_attn"])
    else:
        print("Warning: gene_ctx_attn not found in checkpoint — using fresh weights.")
    predictor.load_state_dict(checkpoint["predictor"])

    llm_encoder.eval()
    gene_agg.eval()
    gene_ctx_attn.eval()
    predictor.eval()

    if "ge_encoder" in checkpoint and "cnv_encoder" in checkpoint and "mu_encoder" in checkpoint:
        print("load omics encoders")
        omics_in_kv = getattr(args, 'omics_in_kv', False)
        omics_out_dim = 768 if omics_in_kv else 256
        ge_encoder  = OmicsEncoder(input_dim=942, output_dim=omics_out_dim).to(device)
        cnv_encoder = OmicsEncoder(input_dim=942, output_dim=omics_out_dim).to(device)
        mu_encoder  = OmicsEncoder(input_dim=942, output_dim=omics_out_dim).to(device)

        ge_encoder.load_state_dict(checkpoint["ge_encoder"])
        cnv_encoder.load_state_dict(checkpoint["cnv_encoder"])
        mu_encoder.load_state_dict(checkpoint["mu_encoder"])

        ge_encoder.eval()
        cnv_encoder.eval()
        mu_encoder.eval()

        film_layer = None
        if "film_layer" in checkpoint:
            film_layer = PairFiLM(omics_dim=768, pair_dim=2304).to(device)
            film_layer.load_state_dict(checkpoint["film_layer"])
            film_layer.eval()
            print("load film_layer")

        return llm_encoder, gene_agg, gene_ctx_attn, predictor, ge_encoder, cnv_encoder, mu_encoder, film_layer, args
    else:
        return llm_encoder, gene_agg, gene_ctx_attn, predictor, None, None, None, None, args

_EVAL_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_pathway_encoder(model_dir, args, scenario, fold_idx, device, single_model=False):
    """Load pathway/geneset encoder from checkpoint if the run used --omics_mode.

    Returns (encoder, pathway_dict, geneset_dict, text_pathway_fusion, pair_builder).
    pair_builder is non-None only for pair_pathway_evidence; text_pathway_fusion is
    non-None for ge_pathway_selfatt and pair_pathway_evidence.
    """
    omics_mode = getattr(args, 'omics_mode', None)
    if not omics_mode or str(omics_mode) == 'None':
        return None, None, None, None, None
    pathway_set = str(getattr(args, 'pathway_set', 'hallmark'))
    filename = f"model_{scenario}.pt" if single_model else f"model_{scenario}_F{fold_idx}.pt"
    checkpoint = torch.load(os.path.join(model_dir, filename), map_location="cpu", weights_only=False)
    # Route by mode FIRST: pair_pathway_evidence and ge_pathway_selfatt both carry a
    # 'text_pathway_fusion' key, so disambiguate before the key-presence checks.
    ckpt_mode = str(checkpoint.get('omics_mode', omics_mode))
    if ckpt_mode == 'pair_pathway_evidence' or str(omics_mode) == 'pair_pathway_evidence':
        pair_builder = PairPathwayEvidenceBuilder(k=getattr(args, 'pathway_topk', 16))
        enc = PairPathwayEvidenceEncoder(
            n_features=pair_builder.n_features, desc_dim=pair_builder.desc_dim, output_dim=768).to(device)
        enc.load_state_dict(checkpoint['pathway_encoder'])
        enc.eval()
        fusion = TextPathwayFusion(hidden_dim=768, nhead=8, ff_dim=512).to(device)
        fusion.load_state_dict(checkpoint['text_pathway_fusion'])
        fusion.eval()
        return enc, None, None, fusion, pair_builder
    if 'text_pathway_fusion' in checkpoint:
        # ge_pathway_selfatt: GEPathwayTokenEncoder + TextPathwayFusion
        json_path = os.path.join(_EVAL_PROJECT_ROOT, f"data/multi_omics/ge_pathway_stats_selfatt_{pathway_set}.json")
        pathway_dict = load_ge_pathway_dict_selfatt(json_path)
        n_pathways = next(iter(pathway_dict.values())).shape[0]
        enc = GEPathwayTokenEncoder(n_stats=6, hidden_dim=128, output_dim=768).to(device)
        enc.load_state_dict(checkpoint['pathway_encoder'])
        enc.eval()
        fusion = TextPathwayFusion(hidden_dim=768, nhead=8, ff_dim=512).to(device)
        fusion.load_state_dict(checkpoint['text_pathway_fusion'])
        fusion.eval()
        return enc, pathway_dict, None, fusion, None
    elif 'pathway_encoder' in checkpoint:
        json_path = os.path.join(_EVAL_PROJECT_ROOT, f"data/multi_omics/pathway_omics_{pathway_set}.json")
        pathway_dict = load_pathway_omics_dict(json_path)
        n_pathways = next(iter(pathway_dict.values())).shape[0]
        enc = PathwayTokenEncoder(n_scores=3, hidden_dim=128, n_pathways=n_pathways, output_dim=768).to(device)
        enc.load_state_dict(checkpoint['pathway_encoder'])
        enc.eval()
        return enc, pathway_dict, None, None, None
    elif 'geneset_encoder' in checkpoint:
        npz_path = os.path.join(_EVAL_PROJECT_ROOT, f"data/multi_omics/pathway_geneset_{pathway_set}.npz")
        geneset_dict, _ = load_geneset_omics_dict(npz_path)
        sample = next(iter(geneset_dict.values()))
        n_pathways, g_max, _ = sample[0].shape
        enc = GeneSetEncoder(n_omics=3, gene_hidden=64, n_pathways=n_pathways, output_dim=768).to(device)
        enc.load_state_dict(checkpoint['geneset_encoder'])
        enc.eval()
        return enc, None, geneset_dict, None, None
    return None, None, None, None, None


def evaluate_on_folds(cell_lines: list, model_cell_line: str, scenario, model_ids: list, base_model_dir: str, result_root: str, device, bioLLM):
    for model_id in model_ids:
        print("eval for model: ", scenario, model_id)
        result_dir = os.path.join(result_root, scenario)
        os.makedirs(result_dir, exist_ok=True)
        result_path = os.path.join(result_dir, f"{model_id}.csv")
        for cell_line in cell_lines:
            print(f"\nEvaluating for {cell_line}\n{'='*50}")
            model_dir = os.path.join(base_model_dir, model_cell_line, model_id)
            model_files = [f for f in os.listdir(model_dir) if f.endswith(".pt") and "F" in f]
            single_model_filename = f"model_{scenario}.pt"
            single_model_path = os.path.join(model_dir, single_model_filename)
            results = []
            
            if not model_files and os.path.exists(single_model_path): ## MCTL
                print(f"Found single model file: {single_model_filename}")
                fold_idx = 1
                llm_encoder, gene_agg, gene_ctx_attn, predictor, ge_encoder, cnv_encoder, mu_encoder, film_layer, args = load_models_from_log(model_dir, bioLLM, scenario, fold_idx, device, single_model=True)
                cell_line_set = list(args.cell_line_set) if hasattr(args, "cell_line_set") else ["A549", "PK1", "IPC298", "MEWO"]
                target_cell_lines = [cell_line]
                eval_loader, _ = load_data_MCTL(args.dataset_path, target_cell_lines, cell_line_set, batch_size=args.batch_size, negpos=1)

                omics_emb_dict = load_omics_emb_dict() if (hasattr(args, "omics_data") and args.omics_data) else None
                omics_dict = load_omics_dict() if (hasattr(args, "omics_encoder") and args.omics_encoder) else None
                pathway_encoder, pathway_dict, geneset_dict, text_pathway_fusion, pair_builder = _load_pathway_encoder(model_dir, args, scenario, fold_idx, device, single_model=True)
                omics_mode = getattr(args, 'omics_mode', None) or None
                val_metrics = validate_model(eval_loader, llm_encoder, gene_agg, gene_ctx_attn, predictor, device,
                                             omics_emb_dict=omics_emb_dict, omics_dict=omics_dict,
                                             ge_encoder=ge_encoder, cnv_encoder=cnv_encoder, mu_encoder=mu_encoder,
                                             film_layer=film_layer,
                                             pathway_encoder=pathway_encoder, pathway_dict=pathway_dict,
                                             omics_mode=omics_mode, geneset_dict=geneset_dict,
                                             text_pathway_fusion=text_pathway_fusion, pair_builder=pair_builder)
        
                
                results.append({
                    "Cell Line": cell_line,
                    "Fold": fold_idx,
                    "Val Loss": val_metrics['loss'],
                    "Acc": val_metrics['accuracy'],
                    "AUC": val_metrics['roc_auc'],
                    "AUPR": val_metrics['aupr'],
                    "F1": val_metrics['f1'],
                    "BACC": val_metrics['bacc'],
                    "Recall": val_metrics['recall']
                })
                df = pd.DataFrame(results)
                # Add an empty row for visual separation
                empty_row = {col: '' for col in df.columns}
                df = pd.concat([df, pd.DataFrame([empty_row])], ignore_index=True)
                
            else: ## MCL
                for model_file in model_files:
                    match = re.search(r'F(\d+)', model_file)
                    if not match:
                        continue
                    fold_idx = int(match.group(1))
                    llm_encoder, gene_agg, gene_ctx_attn, predictor, ge_encoder, cnv_encoder, mu_encoder, film_layer, args = load_models_from_log(model_dir, bioLLM, scenario, fold_idx, device, single_model=False)
                    eval_loader, _ = load_data_SCL(args.dataset_path, cell_line, batch_size=args.batch_size, negpos=1, fold_idx=fold_idx-1)
                    omics_emb_dict = load_omics_emb_dict() if (hasattr(args, "omics_data") and args.omics_data) else None
                    omics_dict = load_omics_dict() if (hasattr(args, "omics_encoder") and args.omics_encoder) else None
                    pathway_encoder, pathway_dict, geneset_dict, text_pathway_fusion, pair_builder = _load_pathway_encoder(model_dir, args, scenario, fold_idx, device, single_model=False)
                    omics_mode = getattr(args, 'omics_mode', None) or None
                    val_metrics = validate_model(eval_loader, llm_encoder, gene_agg, gene_ctx_attn, predictor, device,
                                                 omics_emb_dict=omics_emb_dict, omics_dict=omics_dict,
                                                 ge_encoder=ge_encoder, cnv_encoder=cnv_encoder, mu_encoder=mu_encoder,
                                                 film_layer=film_layer,
                                                 pathway_encoder=pathway_encoder, pathway_dict=pathway_dict,
                                                 omics_mode=omics_mode, geneset_dict=geneset_dict,
                                                 text_pathway_fusion=text_pathway_fusion, pair_builder=pair_builder)

                    results.append({
                        "Cell Line": cell_line,
                        "Fold": fold_idx,
                        "Val Loss": val_metrics['loss'],
                        "Acc": val_metrics['accuracy'],
                        "AUC": val_metrics['roc_auc'],
                        "AUPR": val_metrics['aupr'],
                        "F1": val_metrics['f1'],
                        "BACC": val_metrics['bacc'],
                        "Recall": val_metrics['recall']
                    })
                df = pd.DataFrame(results)
                # Compute mean and std for each metric (excluding 'Cell Line' and 'Fold')
                if not df.empty and all(col in df.columns for col in ["Cell Line", "Fold"]):
                    mean_row = df.drop(columns=["Cell Line", "Fold"]).mean()
                    std_row = df.drop(columns=["Cell Line", "Fold"]).std()
                    mean_row["Cell Line"] = cell_line
                    mean_row["Fold"] = "mean"
                    std_row["Cell Line"] = cell_line
                    std_row["Fold"] = "std"
                    df = pd.concat([df, pd.DataFrame([mean_row, std_row])], ignore_index=True)
            print(df)
            if os.path.exists(result_path):
                print(f"Result file exists, appending: {result_path}")
                df.to_csv(result_path, mode='a', header=False, index=False)
            else:
                df.to_csv(result_path, index=False)


def evaluate_on_folds_mctl(cell_lines: list, model_cell_line: str, scenario, model_ids: list, base_model_dir: str, result_root: str, device, bioLLM, eval_dataset_path=None):
    for model_id in model_ids:
        print("eval for model: ", scenario, model_id)
        result_dir = os.path.join(result_root, scenario)
        os.makedirs(result_dir, exist_ok=True)
        result_path = os.path.join(result_dir, f"{model_id}.csv")
        for cell_line in cell_lines:
            print(f"\nEvaluating for {cell_line}\n{'='*50}")
            model_dir = os.path.join(base_model_dir, model_cell_line, model_id)
            model_files = [f for f in os.listdir(model_dir) if f.endswith(".pt") and "F" in f]
            single_model_filename = f"model_{scenario}.pt"
            single_model_path = os.path.join(model_dir, single_model_filename)
            results = []

            if not model_files and os.path.exists(single_model_path):
                print(f"Found single model file: {single_model_filename}")
                fold_idx = 1
                llm_encoder, gene_agg, gene_ctx_attn, predictor, ge_encoder, cnv_encoder, mu_encoder, film_layer, args = load_models_from_log(model_dir, bioLLM, scenario, fold_idx, device, single_model=True)
                if eval_dataset_path is not None:
                    pt_path = os.path.join(eval_dataset_path, f"{cell_line}_F1.pt")
                    ds = torch.load(pt_path, map_location="cpu", weights_only=False)
                    eval_loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
                else:
                    cell_line_set = ["A549"]  # placeholder; train_loader not used here
                    target_cell_lines = [cell_line]
                    eval_loader, _ = load_data_MCTL(args.dataset_path, [cell_line], cell_line_set, batch_size=args.batch_size, negpos=1)

                omics_emb_dict = load_omics_emb_dict() if (hasattr(args, "omics_data") and args.omics_data) else None
                omics_dict = load_omics_dict() if (hasattr(args, "omics_encoder") and args.omics_encoder) else None
                pathway_encoder, pathway_dict, geneset_dict, text_pathway_fusion, pair_builder = _load_pathway_encoder(model_dir, args, scenario, fold_idx, device, single_model=True)
                omics_mode = getattr(args, 'omics_mode', None) or None
                val_metrics = validate_model(eval_loader, llm_encoder, gene_agg, gene_ctx_attn, predictor, device,
                                             omics_emb_dict=omics_emb_dict, omics_dict=omics_dict,
                                             ge_encoder=ge_encoder, cnv_encoder=cnv_encoder, mu_encoder=mu_encoder,
                                             film_layer=film_layer,
                                             pathway_encoder=pathway_encoder, pathway_dict=pathway_dict,
                                             omics_mode=omics_mode, geneset_dict=geneset_dict,
                                             text_pathway_fusion=text_pathway_fusion, pair_builder=pair_builder)
                results.append({
                    "Cell Line": cell_line, "Fold": fold_idx,
                    "Val Loss": val_metrics['loss'], "Acc": val_metrics['accuracy'],
                    "AUC": val_metrics['roc_auc'], "AUPR": val_metrics['aupr'],
                    "F1": val_metrics['f1'], "BACC": val_metrics['bacc'], "Recall": val_metrics['recall']
                })
                df = pd.DataFrame(results)
                empty_row = {col: '' for col in df.columns}
                df = pd.concat([df, pd.DataFrame([empty_row])], ignore_index=True)
            else:
                print(f"No MCTL model file found in {model_dir}, skipping.")
                continue

            print(df)
            if os.path.exists(result_path):
                print(f"Result file exists, appending: {result_path}")
                df.to_csv(result_path, mode='a', header=False, index=False)
            else:
                df.to_csv(result_path, index=False)


def evaluate_model(cell_lines: list, model_cell_line: str, scenario, model_ids: list, base_model_dir: str, result_root: str, device, bioLLM):
    for model_id in model_ids:
        print("eval for model: ", scenario, model_id)
        result_dir = os.path.join(result_root, scenario)
        os.makedirs(result_dir, exist_ok=True)
        result_path = os.path.join(result_dir, f"{model_id}.csv")
        # cell_lines = ["mixed"]
        for cell_line in cell_lines: # only need eval once for mixed
            print(f"\nEvaluating for {cell_line}\n{'='*50}")
            model_dir = os.path.join(base_model_dir, model_cell_line, model_id)
            model_files = [f for f in os.listdir(model_dir) if f.endswith(".pt") and "F" in f]
            single_model_filename = f"model_{scenario}.pt"
            single_model_path = os.path.join(model_dir, single_model_filename)
            results = []
            
            if not model_files and os.path.exists(single_model_path): ## MCTL
                print(scenario)
                print(f"Found single model file: {single_model_filename}")
                fold_idx = 1
                llm_encoder, gene_agg, gene_ctx_attn, predictor, ge_encoder, cnv_encoder, mu_encoder, film_layer, args = load_models_from_log(model_dir, bioLLM, scenario, fold_idx, device, single_model=True)
                cell_line_set = ["A549"]
                target_cell_lines = [cell_line]
                eval_loader, _ = load_data_MCTL(args.dataset_path, target_cell_lines, cell_line_set, batch_size=args.batch_size, negpos=1)

                omics_emb_dict = load_omics_emb_dict() if (hasattr(args, "omics_data") and args.omics_data) else None
                omics_dict = load_omics_dict() if (hasattr(args, "omics_encoder") and args.omics_encoder) else None
                pathway_encoder, pathway_dict, geneset_dict, text_pathway_fusion, pair_builder = _load_pathway_encoder(model_dir, args, scenario, fold_idx, device, single_model=True)
                omics_mode = getattr(args, 'omics_mode', None) or None
                val_metrics = eval_model(eval_loader, llm_encoder, gene_agg, gene_ctx_attn, predictor, device,
                                         omics_emb_dict=omics_emb_dict, omics_dict=omics_dict,
                                         ge_encoder=ge_encoder, cnv_encoder=cnv_encoder, mu_encoder=mu_encoder,
                                         film_layer=film_layer,
                                         pathway_encoder=pathway_encoder, pathway_dict=pathway_dict,
                                         omics_mode=omics_mode, geneset_dict=geneset_dict,
                                         text_pathway_fusion=text_pathway_fusion, pair_builder=pair_builder,
                                         no_context=getattr(args, "no_context", False),
                                         no_kg=getattr(args, "no_kg", False),
                                         random_genept=getattr(args, "random_genept", False),
                                         no_genept=getattr(args, "no_genept", False),
                                         fold=fold_idx, e_cell_line=cell_line)

                results.append({
                    "Cell Line": cell_line,
                    "Fold": fold_idx,
                    "Val Loss": val_metrics['loss'],
                    "Acc": val_metrics['accuracy'],
                    "AUC": val_metrics['roc_auc'],
                    "AUPR": val_metrics['aupr'],
                    "F1": val_metrics['f1'],
                    "BACC": val_metrics['bacc'],
                    "Recall": val_metrics['recall']
                })
                df = pd.DataFrame(results)
                # Add an empty row for visual separation
                empty_row = {col: '' for col in df.columns}
                df = pd.concat([df, pd.DataFrame([empty_row])], ignore_index=True)

            else: ## MCL
                print(scenario)
                for model_file in model_files:
                    match = re.search(r'F(\d+)', model_file)
                    if not match:
                        continue
                    fold_idx = int(match.group(1))
                    llm_encoder, gene_agg, gene_ctx_attn, predictor, ge_encoder, cnv_encoder, mu_encoder, film_layer, args = load_models_from_log(model_dir, bioLLM, scenario, fold_idx, device, single_model=False)
                    specific_cell_line = "MEWO"
                    eval_loader, _ = load_data_SCL(args.dataset_path, specific_cell_line, batch_size=args.batch_size, negpos=1, fold_idx=fold_idx-1)
                    omics_emb_dict = load_omics_emb_dict() if (hasattr(args, "omics_data") and args.omics_data) else None
                    omics_dict = load_omics_dict() if (hasattr(args, "omics_encoder") and args.omics_encoder) else None
                    pathway_encoder, pathway_dict, geneset_dict, text_pathway_fusion, pair_builder = _load_pathway_encoder(model_dir, args, scenario, fold_idx, device, single_model=False)
                    omics_mode = getattr(args, 'omics_mode', None) or None
                    val_metrics = eval_model(eval_loader, llm_encoder, gene_agg, gene_ctx_attn, predictor, device,
                                             omics_emb_dict=omics_emb_dict, omics_dict=omics_dict,
                                             ge_encoder=ge_encoder, cnv_encoder=cnv_encoder, mu_encoder=mu_encoder,
                                             film_layer=film_layer,
                                             pathway_encoder=pathway_encoder, pathway_dict=pathway_dict,
                                             omics_mode=omics_mode, geneset_dict=geneset_dict,
                                             text_pathway_fusion=text_pathway_fusion, pair_builder=pair_builder,
                                             no_context=getattr(args, "no_context", False),
                                             no_kg=getattr(args, "no_kg", False),
                                             random_genept=getattr(args, "random_genept", False),
                                             no_genept=getattr(args, "no_genept", False),
                                             fold=fold_idx)
        
                    results.append({
                        "Cell Line": cell_line,
                        "Fold": fold_idx,
                        "Val Loss": val_metrics['loss'],
                        "Acc": val_metrics['accuracy'],
                        "AUC": val_metrics['roc_auc'],
                        "AUPR": val_metrics['aupr'],
                        "F1": val_metrics['f1'],
                        "BACC": val_metrics['bacc'],
                        "Recall": val_metrics['recall']
                    })
                df = pd.DataFrame(results)
                # Compute mean and std for each metric (excluding 'Cell Line' and 'Fold')
                if not df.empty and all(col in df.columns for col in ["Cell Line", "Fold"]):
                    mean_row = df.drop(columns=["Cell Line", "Fold"]).mean()
                    std_row = df.drop(columns=["Cell Line", "Fold"]).std()
                    mean_row["Cell Line"] = cell_line
                    mean_row["Fold"] = "mean"
                    std_row["Cell Line"] = cell_line
                    std_row["Fold"] = "std"
                    df = pd.concat([df, pd.DataFrame([mean_row, std_row])], ignore_index=True)
            print(df)
            if os.path.exists(result_path):
                print(f"Result file exists, appending: {result_path}")
                df.to_csv(result_path, mode='a', header=False, index=False)
            else:
                df.to_csv(result_path, index=False)
    
##########################

# Command:  python -m src.eval

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--biobert_path", type=str, default=BIOMEDBERT)
    parser.add_argument("--scenario", type=str, required=True, choices=["MCL", "MCTL"],
                        help="MCL: evaluate on source cell lines; MCTL: evaluate on transfer cell lines.")
    parser.add_argument("--model_id", type=str, required=True, help="Model folder name under save/{scenario}/{model_cell_line}/.")
    parser.add_argument("--model_cell_line", type=str, default=None,
                        help="Sub-dir under save/{scenario}/ holding the model (e.g. mixed-6cl). Defaults: MCL->mixed-6cl, MCTL->mctl.")
    parser.add_argument("--cell_lines", nargs="+", default=None,
                        help="Cell lines to evaluate (default: all cell lines in the study).")
    parser.add_argument("--eval_dataset_path", type=str, default=None,
                        help="Override dataset path for evaluation (e.g. full imbalanced eval set). If not set, uses the path from training log.")
    cli = parser.parse_args()

    device = torch.device(f"cuda:{cli.device}")
    bioLLM = AutoModel.from_pretrained(cli.biobert_path).to(device)

    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    DEFAULT_CELL_LINES = ["A549", "PK1", "IPC298", "MEWO", "GI1", "HS936T"]

    cell_lines      = cli.cell_lines or DEFAULT_CELL_LINES
    model_cell_line = cli.model_cell_line or ("mixed-6cl" if cli.scenario == "MCL" else "mctl")

    kwargs = dict(
        cell_lines=cell_lines,
        model_cell_line=model_cell_line,
        scenario=cli.scenario,
        model_ids=[cli.model_id],
        base_model_dir=os.path.join(PROJECT_ROOT, "save", cli.scenario),
        result_root=os.path.join(PROJECT_ROOT, "results"),
        device=device,
        bioLLM=bioLLM,
    )

    if cli.scenario == "MCL":
        evaluate_on_folds(**kwargs)
    else:
        evaluate_on_folds_mctl(**kwargs, eval_dataset_path=cli.eval_dataset_path)


