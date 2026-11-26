import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam
import numpy as np
import os
import pandas as pd
import pickle
import json
import hashlib
from .model import EarlyStopping, LLMEncoder, GeneAttentionAggregator, GenePairContextAttention, Predictor, OmicsEncoder, PairFiLM, PathwayTokenEncoder, GeneSetEncoder, GEPathwayTokenEncoder, TextPathwayFusion, PairPathwayEvidenceEncoder
from .pathway_utils import load_pathway_omics_dict, load_geneset_omics_dict, load_ge_pathway_dict_selfatt
from .pair_pathway import PairPathwayEvidenceBuilder
from sklearn.metrics import auc, roc_auc_score, f1_score, balanced_accuracy_score, accuracy_score, precision_recall_curve, recall_score
from torch.optim.lr_scheduler import ReduceLROnPlateau

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# GenePT scale (matched to real embeddings: mean~0, std~0.018) for --random_genept.
_RANDOM_GENEPT_STD = 0.018
_random_genept_cache = {}


def _random_genept_vec(gene, dim, device, dtype):
    """Per-gene FIXED random vector, seeded by gene name (deterministic across
    batches/epochs/runs). Preserves gene identity while removing GenePT's
    pretrained knowledge content. Scale-matched to real GenePT embeddings."""
    v = _random_genept_cache.get(gene)
    if v is None:
        seed = int(hashlib.md5(str(gene).encode("utf-8")).hexdigest()[:8], 16)
        g = torch.Generator().manual_seed(seed)
        v = torch.randn(dim, generator=g) * _RANDOM_GENEPT_STD
        _random_genept_cache[gene] = v
    return v.to(device=device, dtype=dtype)


def apply_genept_ablation(geneA_genept, geneB_genept, gene_names,
                          random_genept=False, no_genept=False):
    """Mutate the GenePT halves of a batch in place per ablation flags.
    gene_names = (gene_pair_list, cell_line_list); gene_pair is 'GENE1|GENE2'."""
    if no_genept:
        return torch.zeros_like(geneA_genept), torch.zeros_like(geneB_genept)
    if random_genept:
        gene_pair_list, _ = gene_names
        dim = geneA_genept.shape[-1]
        dev, dt = geneA_genept.device, geneA_genept.dtype
        a_rows, b_rows = [], []
        for gp in gene_pair_list:
            g1, _, g2 = str(gp).partition("|")
            a_rows.append(_random_genept_vec(g1, dim, dev, dt))
            b_rows.append(_random_genept_vec(g2, dim, dev, dt))
        return torch.stack(a_rows, dim=0), torch.stack(b_rows, dim=0)
    return geneA_genept, geneB_genept


def log_print(string, logger):
    print(string)
    logger.info(string)


def log_model_summary(args, llm_encoder, gene_agg, gene_ctx_attn, predictor,
                      ge_encoder=None, cnv_encoder=None, mu_encoder=None,
                      film_layer=None, pathway_encoder=None, logger=None):
    # Only print once — on the first fold
    if getattr(args, 'fold_idx', 0) != 0:
        return

    def n_params(m):
        return sum(p.numel() for p in m.parameters()) / 1e6

    # Omics injection mode
    omics_in_kv = getattr(args, 'omics_in_kv', False)
    if not getattr(args, 'omics_encoder', False):
        omics_mode_str = 'none'
    elif omics_in_kv:
        omics_mode_str = 'omics-in-KV  (942→768, injected as context token)'
    else:
        omics_mode_str = 'FiLM        (942→256×3→768, modulates pair repr)'

    # Pathway mode
    omics_mode = getattr(args, 'omics_mode', None)
    pathway_str = omics_mode if omics_mode else 'none'

    # Total trainable params
    all_modules = [llm_encoder, gene_agg, gene_ctx_attn, predictor]
    for m in [ge_encoder, cnv_encoder, mu_encoder, film_layer, pathway_encoder]:
        if m is not None:
            all_modules.append(m)
    total_trainable = sum(p.numel() for m in all_modules for p in m.parameters() if p.requires_grad) / 1e6

    lines = [
        "=" * 60,
        "MODEL SUMMARY",
        f"  LLM           : BiomedBERT-base  {n_params(llm_encoder):.1f}M  freeze={getattr(args,'freeze_llm',False)}",
        f"  gene_agg      : GeneAttentionAggregator  in={args.geneagg_in} hid={args.geneagg_hidden} ff={args.geneagg_ff}  {n_params(gene_agg):.2f}M",
        f"  gene_ctx_attn : GenePairContextAttention  hid={args.geneagg_hidden} ff={args.geneagg_ff}  {n_params(gene_ctx_attn):.2f}M",
        f"  omics         : {omics_mode_str}",
        f"  pathway       : {pathway_str}",
        f"  predictor     : {args.pred_in}→{args.pred_hidden}→2  {n_params(predictor):.2f}M",
        f"  trainable     : {total_trainable:.1f}M params",
        "=" * 60,
    ]
    for line in lines:
        log_print(line, logger)

def load_omics_emb_dict(pkl_path=None):
    if pkl_path is None:
        pkl_path = os.path.join(PROJECT_ROOT, "data/multi_omics/multi_omics_emb_3_256.pkl")
    with open(pkl_path, "rb") as f:
        records_list = pickle.load(f)
    # Convert list to dict keyed by cell_line_name
    records = {rec["cell_line_name"]: rec for rec in records_list}
    print("Load cell line omics embeddings from AE.")
    return records

def get_omics_emb_for_cell_line(records, cell_line_name):
    rec = records.get(cell_line_name)
    if rec is None:
        raise ValueError(f"Cell line {cell_line_name} not found in omics embeddings.")
    omics_emb = np.concatenate([rec["ge_emb"], rec["cnv_emb"], rec["mu_emb"]], axis=0)  # shape (768,)
    return omics_emb

def load_omics_dict(path=None):
    if path is None:
        path = os.path.join(PROJECT_ROOT, "data/multi_omics/cell_line_omics_data.json")
    with open(path, "r") as f:
        omics_emb_dict = json.load(f)
    return omics_emb_dict

def get_omics_for_cell_line(omics_dict, cell_line_name):
    """
    Returns a dictionary with GE, CNV, MU numpy vectors of shape (942,)
    """
    if cell_line_name not in omics_dict:
        raise ValueError(f"Cell line '{cell_line_name}' not found in omics_dict")
    data = omics_dict[cell_line_name]
    return {
        "GE": np.array(data["GE"], dtype=np.float32),
        "CNV": np.array(data["CNV"], dtype=np.float32),
        "MU": np.array(data["MU"], dtype=np.float32),
    }



def train_model(train_loader, val_loader, bioLLM, args, save_path, logger=None, omics_data=False, omics_encoder=False):
    device = args.device if hasattr(args, 'device') else 'cuda' if torch.cuda.is_available() else 'cpu'

    if omics_data: # load cell line omics embs
        omics_emb_dict = load_omics_emb_dict()
    else:
        omics_emb_dict = None

    no_context = getattr(args, 'no_context', False)
    no_text = getattr(args, 'no_text', False)
    if no_text and getattr(args, 'omics_mode', None) != 'pair_pathway_evidence':
        raise ValueError("--no_text (text-off, pathway-on sufficiency test) requires --omics_mode pair_pathway_evidence")
    no_kg = getattr(args, 'no_kg', False)
    random_genept = getattr(args, 'random_genept', False)
    no_genept = getattr(args, 'no_genept', False)
    omics_in_kv = getattr(args, 'omics_in_kv', False)

    # 1. Instantiate models
    if omics_encoder:
        omics_dict = load_omics_dict()
        if omics_in_kv:
            # KV mode: each encoder outputs 768 to match transformer hidden dim
            ge_encoder  = OmicsEncoder(input_dim=942, output_dim=768).to(device)
            cnv_encoder = OmicsEncoder(input_dim=942, output_dim=768).to(device)
            mu_encoder  = OmicsEncoder(input_dim=942, output_dim=768).to(device)
            film_layer  = None
        else:
            # FiLM mode: each encoder outputs 256, concatenated → 768 for FiLM
            ge_encoder  = OmicsEncoder(input_dim=942, output_dim=256).to(device)
            cnv_encoder = OmicsEncoder(input_dim=942, output_dim=256).to(device)
            mu_encoder  = OmicsEncoder(input_dim=942, output_dim=256).to(device)
            film_layer  = PairFiLM(omics_dim=768, pair_dim=2304).to(device)
    else:
        omics_dict = None
        ge_encoder = cnv_encoder = mu_encoder = film_layer = None

    # ── pathway_tokens / geneset_tokens mode ──────────────────────────────
    omics_mode = getattr(args, 'omics_mode', None)
    pathway_encoder = None
    pathway_dict    = None
    geneset_dict    = None
    text_pathway_fusion = None
    pair_builder    = None
    if omics_mode == 'pathway_tokens':
        pathway_set  = getattr(args, 'pathway_set', 'hallmark')
        pathway_json = os.path.join(PROJECT_ROOT, f"data/multi_omics/pathway_omics_{pathway_set}.json")
        pathway_dict = load_pathway_omics_dict(pathway_json)
        n_pathways   = next(iter(pathway_dict.values())).shape[0]
        pathway_encoder = PathwayTokenEncoder(n_scores=3, hidden_dim=128, n_pathways=n_pathways, output_dim=768).to(device)
        log_print(f"pathway_tokens mode: {pathway_set}, P={n_pathways}", logger)
    elif omics_mode == 'geneset_tokens':
        pathway_set  = getattr(args, 'pathway_set', 'hallmark')
        npz_path     = os.path.join(PROJECT_ROOT, f"data/multi_omics/pathway_geneset_{pathway_set}.npz")
        geneset_dict, _pw_names = load_geneset_omics_dict(npz_path)
        sample = next(iter(geneset_dict.values()))  # (gene_omics [P,G_max,3], gene_mask [P,G_max])
        n_pathways, g_max, _ = sample[0].shape
        pathway_encoder = GeneSetEncoder(
            n_omics=3, gene_hidden=64, n_pathways=n_pathways, output_dim=768
        ).to(device)
        log_print(f"geneset_tokens mode: {pathway_set}, P={n_pathways}, G_max={g_max}", logger)
    elif omics_mode == 'ge_pathway_selfatt':
        pathway_set  = getattr(args, 'pathway_set', 'hallmark')
        pathway_json = os.path.join(PROJECT_ROOT, f"data/multi_omics/ge_pathway_stats_selfatt_{pathway_set}.json")
        pathway_dict = load_ge_pathway_dict_selfatt(pathway_json)
        n_pathways   = next(iter(pathway_dict.values())).shape[0]
        pathway_encoder = GEPathwayTokenEncoder(n_stats=6, hidden_dim=128, output_dim=768).to(device)
        text_pathway_fusion = TextPathwayFusion(hidden_dim=768, nhead=8, ff_dim=512).to(device)
        log_print(f"ge_pathway_selfatt mode: {pathway_set}, P={n_pathways}", logger)
    elif omics_mode == 'pair_pathway_evidence':
        pair_builder = PairPathwayEvidenceBuilder(k=getattr(args, 'pathway_topk', 16))
        pathway_encoder = PairPathwayEvidenceEncoder(
            n_features=pair_builder.n_features, desc_dim=pair_builder.desc_dim, output_dim=768).to(device)
        text_pathway_fusion = TextPathwayFusion(hidden_dim=768, nhead=8, ff_dim=512).to(device)
        log_print(f"pair_pathway_evidence mode: Reactome P={pair_builder.P}, "
                  f"K={pair_builder.k}, F={pair_builder.n_features}", logger)
    # ──────────────────────────────────────────────────────────────────────

    llm_encoder = LLMEncoder(bioLLM).to(device)

    freeze_llm = getattr(args, "freeze_llm", False)
    if freeze_llm:
        log_print("freeze_llm=True: BiomedBERT parameters frozen.", logger)
        for param in llm_encoder.parameters():
            param.requires_grad = False

    gene_agg = GeneAttentionAggregator(args.geneagg_in, args.geneagg_hidden, args.geneagg_ff).to(device)
    gene_ctx_attn = GenePairContextAttention(
        hidden_dim=args.geneagg_hidden, ff_dim=args.geneagg_ff, dropout=args.dropout_rate,
        no_cross_attn=getattr(args, "no_cross_attn", False),
        concat_context=getattr(args, "concat_context", False),
    ).to(device)
    predictor = Predictor(args.pred_in, args.pred_hidden).to(device)

    log_model_summary(
        args, llm_encoder, gene_agg, gene_ctx_attn, predictor,
        ge_encoder=ge_encoder, cnv_encoder=cnv_encoder, mu_encoder=mu_encoder,
        film_layer=film_layer, pathway_encoder=pathway_encoder,
        logger=logger,
    )

    # 2. Optimizer & Scheduler
    param_groups = []
    if not freeze_llm:
        # Filter to only trainable params (handles both full fine-tune and LoRA)
        trainable_llm_params = [p for p in llm_encoder.parameters() if p.requires_grad]
        param_groups.append({'params': trainable_llm_params, 'lr': args.llm_lr})
    if omics_encoder:
        param_groups.append({'params': ge_encoder.parameters(),  'lr': args.omics_lr})
        param_groups.append({'params': cnv_encoder.parameters(), 'lr': args.omics_lr})
        param_groups.append({'params': mu_encoder.parameters(),  'lr': args.omics_lr})
        if not omics_in_kv:
            param_groups.append({'params': film_layer.parameters(),  'lr': args.omics_lr})
    param_groups.append({'params': gene_agg.parameters(), 'lr': args.geneagg_lr})
    param_groups.append({'params': gene_ctx_attn.parameters(), 'lr': args.geneagg_lr})
    param_groups.append({'params': predictor.parameters(), 'lr': args.pred_lr})
    if pathway_encoder is not None:
        param_groups.append({'params': pathway_encoder.parameters(), 'lr': args.omics_lr})
    if text_pathway_fusion is not None:
        param_groups.append({'params': text_pathway_fusion.parameters(), 'lr': args.omics_lr})

    optimizer = Adam(param_groups, weight_decay=args.weight_decay)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=args.lr_patience, factor=args.lr_factor)

    # 3. Early stopping
    save_model = not getattr(args, "no_save_model", False)
    early_stop = EarlyStopping(patience=args.early_stop_patience, verbose=True, save_model=save_model)

    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.num_epochs):
        # if epoch == 2: # for debug
        #     break
        if not getattr(args, "freeze_llm", False):
            llm_encoder.train()
        gene_agg.train()
        gene_ctx_attn.train()
        predictor.train()
        if omics_encoder:
            ge_encoder.train()
            cnv_encoder.train()
            mu_encoder.train()
            if not omics_in_kv:
                film_layer.train()
        if pathway_encoder is not None:
            pathway_encoder.train()
        if text_pathway_fusion is not None:
            text_pathway_fusion.train()
        
        total_loss, total_correct, total_samples = 0, 0, 0
        pbar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")
        for batch in pbar:
            embeddings, labels, gene_names = batch
            
            if omics_emb_dict:
                omics_emb_list = []
                gene_pair_list, cell_line_list = gene_names
                for cell_line in cell_line_list:
                    omics_emb = get_omics_emb_for_cell_line(omics_emb_dict, cell_line)
                    omics_emb_list.append(omics_emb)
                omics_emb_batch = torch.tensor(np.stack(omics_emb_list), dtype=torch.float32).to(device)
            
            if omics_dict:
                ge_list, cnv_list, mu_list = [], [], []
                gene_pair_list, cell_line_list = gene_names
                for cell_line in cell_line_list:
                    omics_data = get_omics_for_cell_line(omics_dict, cell_line)
                    ge_list.append(omics_data["GE"])
                    cnv_list.append(omics_data["CNV"])
                    mu_list.append(omics_data["MU"])
                ge_encoded  = ge_encoder(torch.tensor(np.stack(ge_list),  dtype=torch.float32).to(device))
                cnv_encoded = cnv_encoder(torch.tensor(np.stack(cnv_list), dtype=torch.float32).to(device))
                mu_encoded  = mu_encoder(torch.tensor(np.stack(mu_list),  dtype=torch.float32).to(device))
                if omics_in_kv:
                    omics_tok = torch.stack([ge_encoded, cnv_encoded, mu_encoded], dim=1)  # [B, 3, 768]
                else:
                    omics_summary = torch.cat([ge_encoded, cnv_encoded, mu_encoded], dim=-1)  # [B, 768]

            pp_pad = None
            if pathway_encoder is not None:
                gene_pair_list, cell_line_list = gene_names
                if omics_mode == 'pair_pathway_evidence':
                    pp_numeric, pp_desc, pp_pad = pair_builder.build_batch(gene_pair_list, cell_line_list, device)
                    pathway_tok = pathway_encoder(pp_numeric, pp_desc, pp_pad)  # [B,K,768]
                elif omics_mode == 'geneset_tokens':
                    go_list  = [geneset_dict[cl][0] for cl in cell_line_list]  # [P,G_max,3]
                    gm_list  = [geneset_dict[cl][1] for cl in cell_line_list]  # [P,G_max] bool
                    go_batch = torch.tensor(np.stack(go_list), dtype=torch.float32).to(device)   # [B,P,G_max,3]
                    gm_batch = torch.tensor(np.stack(gm_list), dtype=torch.bool).to(device)      # [B,P,G_max]
                    pathway_tok = pathway_encoder(go_batch, gm_batch)  # [B,P,768]
                else:
                    scores_list = [pathway_dict[cl] for cl in cell_line_list]
                    pathway_scores = torch.tensor(np.stack(scores_list), dtype=torch.float32).to(device)  # [B, P, 3]
                    pathway_tok = pathway_encoder(pathway_scores)  # [B, P, 768]

            (token_emb, attention_mask, geneA_genept, geneA_kg_emb, geneB_genept, geneB_kg_emb, geneA_neighbors_emb, geneB_neighbors_emb) = [x.to(device) for x in embeddings]
            labels = labels.to(device)
            B = labels.size(0)

            if no_kg:
                geneA_kg_emb        = torch.zeros_like(geneA_kg_emb)
                geneB_kg_emb        = torch.zeros_like(geneB_kg_emb)
                geneA_neighbors_emb = torch.zeros_like(geneA_neighbors_emb)
                geneB_neighbors_emb = torch.zeros_like(geneB_neighbors_emb)

            geneA_genept, geneB_genept = apply_genept_ablation(
                geneA_genept, geneB_genept, gene_names,
                random_genept=random_genept, no_genept=no_genept)

            # Step 1: LLM Encoding — full token sequence [B, 512, 768]
            if not no_context and not no_text:
                token_seq = llm_encoder(token_emb, attention_mask)
                text_padding_mask = (attention_mask == 0)

            # Step 2: Gene KG aggregation
            expressed_geneA, geneA_agg_emb = gene_agg(geneA_genept, geneA_neighbors_emb, geneB_neighbors_emb, geneA_kg_emb, geneB_kg_emb)
            expressed_geneB, geneB_agg_emb = gene_agg(geneB_genept, geneB_neighbors_emb, geneA_neighbors_emb, geneB_kg_emb, geneA_kg_emb)

            # Step 3: Build context
            if no_context:
                # Ablation: no text, no omics — gene queries attend only to a dummy zero token
                context_tokens = torch.zeros(B, 1, 768, device=device)
                context_mask = torch.zeros(B, 1, dtype=torch.bool, device=device)
            elif no_text:
                # Ablation: text OFF, pathway ON (sufficiency test) — gene queries read only the pair
                # pathway-evidence tokens; a leading zero token guarantees >=1 valid key (some pairs select
                # 0 pathways -> a fully-padded evidence block would otherwise leave no valid context).
                dummy = torch.zeros(B, 1, 768, device=device)
                context_tokens = torch.cat([dummy, pathway_tok], dim=1)
                context_mask = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=device), pp_pad], dim=1)
            elif omics_mode == 'pair_pathway_evidence':
                context_tokens = text_pathway_fusion(token_seq, pathway_tok, text_padding_mask, pathway_padding_mask=pp_pad)
                context_mask = torch.cat([text_padding_mask, pp_pad], dim=1)
            elif omics_mode == 'ge_pathway_selfatt':
                P_len = pathway_tok.shape[1]
                context_tokens = text_pathway_fusion(token_seq, pathway_tok, text_padding_mask)
                context_mask = torch.cat([text_padding_mask, torch.zeros(B, P_len, dtype=torch.bool, device=device)], dim=1)
            elif pathway_encoder is not None:
                P_len = pathway_tok.shape[1]
                context_tokens = torch.cat([token_seq, pathway_tok], dim=1)  # [B, 512+P, 768]
                context_mask = torch.cat([text_padding_mask, torch.zeros(B, P_len, dtype=torch.bool, device=device)], dim=1)
            elif omics_emb_dict:
                omics_tok = omics_emb_batch.unsqueeze(1)  # [B, 1, 768]
                context_tokens = torch.cat([token_seq, omics_tok], dim=1)
                context_mask = torch.cat([text_padding_mask, torch.zeros(B, 1, dtype=torch.bool, device=device)], dim=1)
            elif omics_dict and omics_in_kv:
                context_tokens = torch.cat([token_seq, omics_tok], dim=1)  # [B, 515, 768]
                context_mask = torch.cat([text_padding_mask, torch.zeros(B, 3, dtype=torch.bool, device=device)], dim=1)
            else:
                context_tokens = token_seq
                context_mask = text_padding_mask

            # Step 4: Gene-pair context attention (gene-gene self-attn + gene-to-text cross-attn + FFN)
            gene_query_ctx = gene_ctx_attn(expressed_geneA, geneA_agg_emb, expressed_geneB, geneB_agg_emb, context_tokens, context_mask)

            # Step 5: Pairwise interaction — symmetric + difference + conjunction
            a = gene_query_ctx[:, 0, :]  # [B, 768]
            b = gene_query_ctx[:, 1, :]  # [B, 768]
            pair_text = torch.cat([a + b, torch.abs(a - b), a * b], dim=-1)  # [B, 2304]

            # Step 5b: FiLM conditioning (only in FiLM mode)
            pair_ctx = film_layer(pair_text, omics_summary) if (omics_dict and not omics_in_kv) else pair_text  # [B, 2304]

            # context-free gene embeddings kept alongside the contextual pair repr
            final_repr = torch.cat([pair_ctx, expressed_geneA, geneA_agg_emb, expressed_geneB, geneB_agg_emb], dim=-1)  # [B, 5376]

            # Step 6: Prediction
            logits = predictor(final_repr)
            # Step 5: Loss and Backprop
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Step 6: Metrics
            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
            pbar.set_postfix({'loss': loss.item()})
        train_loss = total_loss / total_samples

        accuracy = total_correct / total_samples
        
        # Validation
        # omics_emb_dict = omics_emb_dict if omics_emb_dict is not None else None
        val_metrics = validate_model(val_loader, llm_encoder, gene_agg, gene_ctx_attn, predictor, device,
                                     omics_emb_dict=omics_emb_dict, omics_dict=omics_dict,
                                     ge_encoder=ge_encoder if omics_encoder else None,
                                     cnv_encoder=cnv_encoder if omics_encoder else None,
                                     mu_encoder=mu_encoder if omics_encoder else None,
                                     film_layer=film_layer if (omics_encoder and not omics_in_kv) else None,
                                     pathway_encoder=pathway_encoder,
                                     pathway_dict=pathway_dict,
                                     omics_mode=omics_mode,
                                     geneset_dict=geneset_dict,
                                     text_pathway_fusion=text_pathway_fusion,
                                     pair_builder=pair_builder,
                                     no_context=no_context,
                                     no_kg=no_kg,
                                     random_genept=random_genept,
                                     no_genept=no_genept,
                                     no_text=no_text)

        log_print(f"Epoch {epoch + 1}/{args.num_epochs} - Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}, BACC: {val_metrics['bacc']:.4f}, AUC: {val_metrics['roc_auc']:.4f}, AUPR: {val_metrics['aupr']:.4f}, Recall: {val_metrics['recall']:.4f}", logger)

        # Early stopping and checkpoint
        if omics_mode in ('ge_pathway_selfatt', 'pair_pathway_evidence'):
            model_list = [llm_encoder, gene_agg, gene_ctx_attn, predictor, pathway_encoder, text_pathway_fusion]
        elif omics_mode in ('pathway_tokens', 'geneset_tokens'):
            model_list = [llm_encoder, gene_agg, gene_ctx_attn, predictor, pathway_encoder]
        elif omics_encoder:
            if omics_in_kv:
                model_list = [llm_encoder, gene_agg, gene_ctx_attn, predictor, ge_encoder, cnv_encoder, mu_encoder]
            else:
                model_list = [llm_encoder, gene_agg, gene_ctx_attn, predictor, ge_encoder, cnv_encoder, mu_encoder, film_layer]
        else:
            model_list = [llm_encoder, gene_agg, gene_ctx_attn, predictor]
        early_stop(
            val_metrics['accuracy'], val_metrics['f1'], val_metrics['roc_auc'], val_metrics['aupr'], val_metrics['bacc'], val_metrics['recall'],
            model_list, save_path, args, logger
        )
        scheduler.step(val_metrics['loss'])
        if early_stop.early_stop:
            log_print(f"Early stopping at epoch {epoch+1}", logger)
            save_final_results(early_stop.best_metrics, save_path, fold_idx=args.fold_idx)
            break
    # If not early stopped, save final results after all epochs
    if not early_stop.early_stop:
        save_final_results(early_stop.best_metrics, save_path, fold_idx=args.fold_idx)


def validate_model(data_loader, llm_encoder, gene_agg, gene_ctx_attn, predictor, device, omics_emb_dict=None, omics_dict=None, ge_encoder=None, cnv_encoder=None, mu_encoder=None, film_layer=None, pathway_encoder=None, pathway_dict=None, omics_mode=None, geneset_dict=None, text_pathway_fusion=None, pair_builder=None, no_context=False, no_kg=False, random_genept=False, no_genept=False, no_text=False, return_preds=False):
    llm_encoder.eval()
    gene_agg.eval()
    gene_ctx_attn.eval()
    predictor.eval()
    if ge_encoder:
        ge_encoder.eval()
        cnv_encoder.eval()
        mu_encoder.eval()
    if film_layer:
        film_layer.eval()
    if pathway_encoder is not None:
        pathway_encoder.eval()
    if text_pathway_fusion is not None:
        text_pathway_fusion.eval()

    with torch.no_grad():
        criterion = nn.CrossEntropyLoss()
        total_loss, total_samples = 0, 0
        all_labels, all_probs = [], []
        all_gene_pairs, all_cell_lines = [], []

        for batch in data_loader:
            embeddings, labels, gene_names = batch

            if omics_emb_dict:
                omics_emb_list = []
                gene_pair_list, cell_line_list = gene_names
                for cell_line in cell_line_list:
                    omics_emb = get_omics_emb_for_cell_line(omics_emb_dict, cell_line)
                    omics_emb_list.append(omics_emb)
                omics_emb_batch = torch.tensor(np.stack(omics_emb_list), dtype=torch.float32).to(device)

            omics_in_kv = (omics_dict is not None and film_layer is None and ge_encoder is not None)

            if omics_dict:
                ge_list, cnv_list, mu_list = [], [], []
                gene_pair_list, cell_line_list = gene_names
                for cell_line in cell_line_list:
                    omics_data = get_omics_for_cell_line(omics_dict, cell_line)
                    ge_list.append(omics_data["GE"])
                    cnv_list.append(omics_data["CNV"])
                    mu_list.append(omics_data["MU"])
                ge_encoded  = ge_encoder(torch.tensor(np.stack(ge_list),  dtype=torch.float32).to(device))
                cnv_encoded = cnv_encoder(torch.tensor(np.stack(cnv_list), dtype=torch.float32).to(device))
                mu_encoded  = mu_encoder(torch.tensor(np.stack(mu_list),  dtype=torch.float32).to(device))
                if omics_in_kv:
                    omics_tok = torch.stack([ge_encoded, cnv_encoded, mu_encoded], dim=1)  # [B, 3, 768]
                else:
                    omics_summary = torch.cat([ge_encoded, cnv_encoded, mu_encoded], dim=-1)  # [B, 768]

            pp_pad = None
            if pathway_encoder is not None:
                gene_pair_list, cell_line_list = gene_names
                if omics_mode == 'pair_pathway_evidence':
                    pp_numeric, pp_desc, pp_pad = pair_builder.build_batch(gene_pair_list, cell_line_list, device)
                    pathway_tok = pathway_encoder(pp_numeric, pp_desc, pp_pad)
                elif omics_mode == 'geneset_tokens':
                    go_list  = [geneset_dict[cl][0] for cl in cell_line_list]
                    gm_list  = [geneset_dict[cl][1] for cl in cell_line_list]
                    go_batch = torch.tensor(np.stack(go_list), dtype=torch.float32).to(device)
                    gm_batch = torch.tensor(np.stack(gm_list), dtype=torch.bool).to(device)
                    pathway_tok = pathway_encoder(go_batch, gm_batch)
                else:
                    scores_list = [pathway_dict[cl] for cl in cell_line_list]
                    pathway_scores = torch.tensor(np.stack(scores_list), dtype=torch.float32).to(device)
                    pathway_tok = pathway_encoder(pathway_scores)  # [B, P, 768]

            (token_emb, attention_mask, geneA_genept, geneA_kg_emb, geneB_genept, geneB_kg_emb, geneA_neighbors_emb, geneB_neighbors_emb) = [x.to(device) for x in embeddings]
            labels = labels.to(device)
            B = labels.size(0)

            if no_kg:
                geneA_kg_emb        = torch.zeros_like(geneA_kg_emb)
                geneB_kg_emb        = torch.zeros_like(geneB_kg_emb)
                geneA_neighbors_emb = torch.zeros_like(geneA_neighbors_emb)
                geneB_neighbors_emb = torch.zeros_like(geneB_neighbors_emb)

            geneA_genept, geneB_genept = apply_genept_ablation(
                geneA_genept, geneB_genept, gene_names,
                random_genept=random_genept, no_genept=no_genept)

            if not no_context and not no_text:
                token_seq = llm_encoder(token_emb, attention_mask)
                text_padding_mask = (attention_mask == 0)

            expressed_geneA, geneA_agg_emb = gene_agg(geneA_genept, geneA_neighbors_emb, geneB_neighbors_emb, geneA_kg_emb, geneB_kg_emb)
            expressed_geneB, geneB_agg_emb = gene_agg(geneB_genept, geneB_neighbors_emb, geneA_neighbors_emb, geneB_kg_emb, geneA_kg_emb)

            if no_context:
                context_tokens = torch.zeros(B, 1, 768, device=device)
                context_mask = torch.zeros(B, 1, dtype=torch.bool, device=device)
            elif no_text:
                dummy = torch.zeros(B, 1, 768, device=device)
                context_tokens = torch.cat([dummy, pathway_tok], dim=1)
                context_mask = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=device), pp_pad], dim=1)
            elif omics_mode == 'pair_pathway_evidence':
                context_tokens = text_pathway_fusion(token_seq, pathway_tok, text_padding_mask, pathway_padding_mask=pp_pad)
                context_mask = torch.cat([text_padding_mask, pp_pad], dim=1)
            elif omics_mode == 'ge_pathway_selfatt':
                P_len = pathway_tok.shape[1]
                context_tokens = text_pathway_fusion(token_seq, pathway_tok, text_padding_mask)
                context_mask = torch.cat([text_padding_mask, torch.zeros(B, P_len, dtype=torch.bool, device=device)], dim=1)
            elif pathway_encoder is not None:
                P_len = pathway_tok.shape[1]
                context_tokens = torch.cat([token_seq, pathway_tok], dim=1)
                context_mask = torch.cat([text_padding_mask, torch.zeros(B, P_len, dtype=torch.bool, device=device)], dim=1)
            elif omics_emb_dict:
                omics_tok = omics_emb_batch.unsqueeze(1)
                context_tokens = torch.cat([token_seq, omics_tok], dim=1)
                context_mask = torch.cat([text_padding_mask, torch.zeros(B, 1, dtype=torch.bool, device=device)], dim=1)
            elif omics_in_kv:
                context_tokens = torch.cat([token_seq, omics_tok], dim=1)  # [B, 515, 768]
                context_mask = torch.cat([text_padding_mask, torch.zeros(B, 3, dtype=torch.bool, device=device)], dim=1)
            else:
                context_tokens = token_seq
                context_mask = text_padding_mask

            gene_query_ctx = gene_ctx_attn(expressed_geneA, geneA_agg_emb, expressed_geneB, geneB_agg_emb, context_tokens, context_mask)
            a = gene_query_ctx[:, 0, :]
            b = gene_query_ctx[:, 1, :]
            pair_text = torch.cat([a + b, torch.abs(a - b), a * b], dim=-1)  # [B, 2304]
            pair_ctx = film_layer(pair_text, omics_summary) if (omics_dict and not omics_in_kv) else pair_text  # [B, 2304]
            final_repr = torch.cat([pair_ctx, expressed_geneA, geneA_agg_emb, expressed_geneB, geneB_agg_emb], dim=-1)  # [B, 5376]

            logits = predictor(final_repr)

            # Loss
            loss = criterion(logits, labels)

            # Step 6: Collect for metrics
            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)

            probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            all_labels.extend(labels.detach().cpu().numpy())
            all_probs.extend(probs)
            if return_preds:
                gp_, cl_ = gene_names
                all_gene_pairs.extend(list(gp_))
                all_cell_lines.extend(list(cl_))

        avg_loss = total_loss / total_samples

        # Find best threshold by macro F1 (skip 0.0 to avoid all-positive degenerate case)
        all_labels_np = np.array(all_labels)
        all_probs_np = np.array(all_probs)
        best_f1 = 0
        best_threshold = 0.5
        for threshold in [x * 0.05 for x in range(1, 20)]:
            threshold_preds = (all_probs_np > threshold).astype(int)
            f1 = f1_score(all_labels_np, threshold_preds, average='macro')
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        final_preds = (all_probs_np > best_threshold).astype(int)

        accuracy = accuracy_score(all_labels_np, final_preds)
        bacc = balanced_accuracy_score(all_labels_np, final_preds)
        roc_auc = roc_auc_score(all_labels_np, all_probs_np)
        precision, recall, _ = precision_recall_curve(all_labels_np, all_probs_np)
        aupr = auc(recall, precision)
        recall_scr = recall_score(all_labels_np, final_preds, average='macro')

        result = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'bacc': bacc,
            'roc_auc': roc_auc,
            'aupr': aupr,
            'f1': best_f1,
            'recall': recall_scr
        }
        if return_preds:
            result['gene_pairs'] = all_gene_pairs
            result['cell_lines'] = all_cell_lines
            result['labels'] = all_labels_np.tolist()
            result['probs'] = all_probs_np.tolist()
        return result


def eval_model(data_loader, llm_encoder, gene_agg, gene_ctx_attn, predictor, device, omics_emb_dict=None, omics_dict=None, ge_encoder=None, cnv_encoder=None, mu_encoder=None, film_layer=None, pathway_encoder=None, pathway_dict=None, omics_mode=None, geneset_dict=None, text_pathway_fusion=None, pair_builder=None, fold=None, e_cell_line=None, no_context=False, no_kg=False, random_genept=False, no_genept=False):
    llm_encoder.eval()
    gene_agg.eval()
    gene_ctx_attn.eval()
    predictor.eval()
    if ge_encoder:
        ge_encoder.eval()
        cnv_encoder.eval()
        mu_encoder.eval()
    if film_layer:
        film_layer.eval()
    if pathway_encoder is not None:
        pathway_encoder.eval()
    if text_pathway_fusion is not None:
        text_pathway_fusion.eval()

    with torch.no_grad():
        criterion = nn.CrossEntropyLoss()
        total_loss, total_samples = 0, 0
        all_labels, all_probs = [], []
        embedding_records = []
        prediction_records = []

        for batch in tqdm(data_loader, desc=f"Evaluating Fold {fold}"):
            embeddings, labels, gene_names = batch

            if omics_emb_dict:
                omics_emb_list = []
                gene_pair_list, cell_line_list = gene_names
                for cell_line in cell_line_list:
                    omics_emb = get_omics_emb_for_cell_line(omics_emb_dict, cell_line)
                    omics_emb_list.append(omics_emb)
                omics_emb_batch = torch.tensor(np.stack(omics_emb_list), dtype=torch.float32).to(device)

            omics_summary = None
            omics_in_kv = (omics_dict is not None and film_layer is None and ge_encoder is not None)
            if omics_dict:
                ge_list, cnv_list, mu_list = [], [], []
                gene_pair_list, cell_line_list = gene_names
                for cell_line in cell_line_list:
                    omics_data = get_omics_for_cell_line(omics_dict, cell_line)
                    ge_list.append(omics_data["GE"])
                    cnv_list.append(omics_data["CNV"])
                    mu_list.append(omics_data["MU"])
                ge_encoded  = ge_encoder(torch.tensor(np.stack(ge_list),  dtype=torch.float32).to(device))
                cnv_encoded = cnv_encoder(torch.tensor(np.stack(cnv_list), dtype=torch.float32).to(device))
                mu_encoded  = mu_encoder(torch.tensor(np.stack(mu_list),  dtype=torch.float32).to(device))
                if omics_in_kv:
                    omics_tok = torch.stack([ge_encoded, cnv_encoded, mu_encoded], dim=1)  # [B, 3, 768]
                else:
                    omics_summary = torch.cat([ge_encoded, cnv_encoded, mu_encoded], dim=-1)  # [B, 768]

            pp_pad = None
            if pathway_encoder is not None:
                gene_pair_list, cell_line_list = gene_names
                if omics_mode == 'pair_pathway_evidence':
                    pp_numeric, pp_desc, pp_pad = pair_builder.build_batch(gene_pair_list, cell_line_list, device)
                    pathway_tok = pathway_encoder(pp_numeric, pp_desc, pp_pad)
                elif omics_mode == 'geneset_tokens':
                    go_list  = [geneset_dict[cl][0] for cl in cell_line_list]
                    gm_list  = [geneset_dict[cl][1] for cl in cell_line_list]
                    go_batch = torch.tensor(np.stack(go_list), dtype=torch.float32).to(device)
                    gm_batch = torch.tensor(np.stack(gm_list), dtype=torch.bool).to(device)
                    pathway_tok = pathway_encoder(go_batch, gm_batch)
                else:
                    scores_list = [pathway_dict[cl] for cl in cell_line_list]
                    pathway_scores = torch.tensor(np.stack(scores_list), dtype=torch.float32).to(device)
                    pathway_tok = pathway_encoder(pathway_scores)  # [B, P, 768]

            (token_emb, attention_mask, geneA_genept, geneA_kg_emb, geneB_genept, geneB_kg_emb, geneA_neighbors_emb, geneB_neighbors_emb) = [x.to(device) for x in embeddings]
            labels = labels.to(device)
            B = labels.size(0)

            if no_kg:
                geneA_kg_emb        = torch.zeros_like(geneA_kg_emb)
                geneB_kg_emb        = torch.zeros_like(geneB_kg_emb)
                geneA_neighbors_emb = torch.zeros_like(geneA_neighbors_emb)
                geneB_neighbors_emb = torch.zeros_like(geneB_neighbors_emb)

            geneA_genept, geneB_genept = apply_genept_ablation(
                geneA_genept, geneB_genept, gene_names,
                random_genept=random_genept, no_genept=no_genept)

            if not no_context:
                token_seq = llm_encoder(token_emb, attention_mask)
                text_padding_mask = (attention_mask == 0)
            expressed_geneA, geneA_agg_emb = gene_agg(geneA_genept, geneA_neighbors_emb, geneB_neighbors_emb, geneA_kg_emb, geneB_kg_emb)
            expressed_geneB, geneB_agg_emb = gene_agg(geneB_genept, geneB_neighbors_emb, geneA_neighbors_emb, geneB_kg_emb, geneA_kg_emb)

            if no_context:
                context_tokens = torch.zeros(B, 1, 768, device=device)
                context_mask = torch.zeros(B, 1, dtype=torch.bool, device=device)
            elif omics_mode == 'pair_pathway_evidence':
                context_tokens = text_pathway_fusion(token_seq, pathway_tok, text_padding_mask, pathway_padding_mask=pp_pad)
                context_mask = torch.cat([text_padding_mask, pp_pad], dim=1)
            elif omics_mode == 'ge_pathway_selfatt':
                P_len = pathway_tok.shape[1]
                context_tokens = text_pathway_fusion(token_seq, pathway_tok, text_padding_mask)
                context_mask = torch.cat([text_padding_mask, torch.zeros(B, P_len, dtype=torch.bool, device=device)], dim=1)
            elif pathway_encoder is not None:
                P_len = pathway_tok.shape[1]
                context_tokens = torch.cat([token_seq, pathway_tok], dim=1)
                context_mask = torch.cat([text_padding_mask, torch.zeros(B, P_len, dtype=torch.bool, device=device)], dim=1)
            elif omics_emb_dict:
                context_tokens = torch.cat([token_seq, omics_emb_batch.unsqueeze(1)], dim=1)
                context_mask = torch.cat([text_padding_mask, torch.zeros(B, 1, dtype=torch.bool, device=device)], dim=1)
            elif omics_in_kv:
                context_tokens = torch.cat([token_seq, omics_tok], dim=1)  # [B, 515, 768]
                context_mask = torch.cat([text_padding_mask, torch.zeros(B, 3, dtype=torch.bool, device=device)], dim=1)
            else:
                context_tokens = token_seq
                context_mask = text_padding_mask

            gene_query_ctx = gene_ctx_attn(expressed_geneA, geneA_agg_emb, expressed_geneB, geneB_agg_emb, context_tokens, context_mask)
            a = gene_query_ctx[:, 0, :]
            b = gene_query_ctx[:, 1, :]
            pair_text = torch.cat([a + b, torch.abs(a - b), a * b], dim=-1)  # [B, 2304]
            pair_ctx = film_layer(pair_text, omics_summary) if (omics_dict and not omics_in_kv) else pair_text  # [B, 2304]
            final_repr = torch.cat([pair_ctx, expressed_geneA, geneA_agg_emb, expressed_geneB, geneB_agg_emb], dim=-1)  # [B, 5376]

            logits = predictor(final_repr)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)

            probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            all_labels.extend(labels.detach().cpu().numpy())
            all_probs.extend(probs)

            for i in range(labels.size(0)):
                gp, cl = gene_names[0][i], gene_names[1][i]
                prediction_records.append({"gene_pair": gp, "cell_line": cl,
                                           "label": labels[i].item(), "pred_score": float(probs[i])})
                record = {"gene_pair": gp, "cell_line": cl}
                if omics_summary is not None:
                    record["omics_summary"] = omics_summary[i].cpu().numpy()
                record["geneA_ctx"] = a[i].cpu().numpy()
                record["geneB_ctx"] = b[i].cpu().numpy()
                embedding_records.append(record)

        if e_cell_line is not None:
            output_dir = os.path.join(PROJECT_ROOT, "results", "case_study", "materials", e_cell_line)
            os.makedirs(output_dir, exist_ok=True)
            pd.DataFrame(prediction_records).to_csv(os.path.join(output_dir, f"predictions_F{fold}.csv"), index=False)
            with open(os.path.join(output_dir, f"gene_pair_embeddings_F{fold}.pkl"), "wb") as f:
                pickle.dump(embedding_records, f)

        avg_loss = total_loss / total_samples

        all_labels_np = np.array(all_labels)
        all_probs_np = np.array(all_probs)
        best_f1 = 0
        best_threshold = 0.5
        for threshold in [x * 0.05 for x in range(1, 20)]:
            threshold_preds = (all_probs_np > threshold).astype(int)
            f1 = f1_score(all_labels_np, threshold_preds, average='macro')
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        final_preds = (all_probs_np > best_threshold).astype(int)

        accuracy = accuracy_score(all_labels_np, final_preds)
        bacc = balanced_accuracy_score(all_labels_np, final_preds)
        roc_auc = roc_auc_score(all_labels_np, all_probs_np)
        precision, recall, _ = precision_recall_curve(all_labels_np, all_probs_np)
        aupr = auc(recall, precision)
        recall_scr = recall_score(all_labels_np, final_preds, average='macro')

        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'bacc': bacc,
            'roc_auc': roc_auc,
            'aupr': aupr,
            'f1': best_f1,
            'recall': recall_scr
        }
        
        
def save_final_results(best_metrics, save_path, fold_idx=None):
    # Prepare the data for this run as a dictionary
    data = {
        'Fold': [fold_idx if fold_idx is not None else 'N/A'],
        'Val Accuracy': [best_metrics['accuracy']],
        'Val F1': [best_metrics['f1']],
        'Val ROC-AUC': [best_metrics['roc_auc']],
        'Val AUPR': [best_metrics['aupr']],
        'Val BACC': [best_metrics['bacc']],
        'Val Recall': [best_metrics['recall']],
    }

    # Create a DataFrame from the dictionary
    df = pd.DataFrame(data)
    result_file = os.path.join(save_path, 'final_results.csv')
    # Write to CSV (append mode, with header only if the file doesn't exist)
    if os.path.exists(result_file):
        df.to_csv(result_file, mode='a', header=False, index=False)
    else:
        df.to_csv(result_file, mode='w', header=True, index=False)
        
        
        