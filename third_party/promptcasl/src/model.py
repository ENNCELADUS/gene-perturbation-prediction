import torch
import torch.nn as nn
import os

class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0, save_model=True):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.last_best = 0.0
        self.best_metrics = {
            'accuracy': 0,
            'f1': 0,
            'roc_auc': 0,
            'aupr': 0,
            'bacc': 0,
            'recall':0,
        }
        self.save_model = save_model

    def __call__(self, val_accuracy, val_f1, val_roc_auc, aupr, val_bacc, val_recall, model, model_path, args, logger=None):
        score = val_roc_auc

        if self.best_score is None:
            self.best_score = score
            self.best_metrics['accuracy'] = val_accuracy
            self.best_metrics['f1'] = val_f1
            self.best_metrics['roc_auc'] = val_roc_auc
            self.best_metrics['aupr'] = aupr
            self.best_metrics['bacc'] = val_bacc
            self.best_metrics['recall'] = val_recall
            self.save_checkpoint(score, val_accuracy, val_f1, val_roc_auc, aupr, val_bacc, val_recall, model, model_path, args, logger)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_metrics['accuracy'] = val_accuracy
            self.best_metrics['f1'] = val_f1
            self.best_metrics['roc_auc'] = val_roc_auc
            self.best_metrics['aupr'] = aupr
            self.best_metrics['bacc'] = val_bacc
            self.best_metrics['recall'] = val_recall
            self.save_checkpoint(score, val_accuracy, val_f1, val_roc_auc, aupr, val_bacc, val_recall, model, model_path, args, logger)
            self.counter = 0
        return self.best_metrics

    def save_checkpoint(self, score, val_accuracy, val_f1, val_roc_auc, aupr, val_bacc, val_recall, model, model_path, args, logger=None):
        '''Saves only llm_encoder, gene_agg, predictor in a single file.'''
        if self.verbose:
            msg = (f'Validation AUC improved ({self.last_best:.6f} → {score:.6f}). '
                   f'Acc: {val_accuracy:.4f}, F1: {val_f1:.4f}, BACC: {val_bacc:.4f}, '
                   f'AUC: {val_roc_auc:.4f}, AUPR: {aupr:.4f}, Recall: {val_recall:.4f}')
            print(msg)
            if logger:
                logger.info(msg)
        if not self.save_model:
            self.last_best = score
            return
        # Save models
        os.makedirs(model_path, exist_ok=True)
        # model = [llm_encoder, gene_agg, gene_ctx_attn, predictor]              (len=4)
        #       or [llm_encoder, gene_agg, gene_ctx_attn, predictor,
        #           pathway_encoder]                                               (len=5, pathway_tokens/geneset_tokens)
        #       or [llm_encoder, gene_agg, gene_ctx_attn, predictor,
        #           pathway_encoder, text_pathway_fusion]                          (len=6, ge_pathway_selfatt)
        #       or [llm_encoder, gene_agg, gene_ctx_attn, predictor,
        #           ge, cnv, mu]                                                   (len=7, KV mode)
        #       or [llm_encoder, gene_agg, gene_ctx_attn, predictor,
        #           ge, cnv, mu, film]                                             (len=8, FiLM mode)
        model_dict = {
            "llm_encoder": model[0].state_dict(),
            "gene_agg": model[1].state_dict(),
            "gene_ctx_attn": model[2].state_dict(),
            "predictor": model[3].state_dict(),
        }
        # Mode marker — lets eval disambiguate ge_pathway_selfatt vs pair_pathway_evidence
        # (both are len-6 with a text_pathway_fusion key).
        if getattr(args, "omics_mode", None):
            model_dict["omics_mode"] = args.omics_mode
        if len(model) == 5:
            if isinstance(model[4], GeneSetEncoder):
                model_dict["geneset_encoder"] = model[4].state_dict()
            else:
                model_dict["pathway_encoder"] = model[4].state_dict()
        if len(model) == 6:
            model_dict["pathway_encoder"] = model[4].state_dict()
            model_dict["text_pathway_fusion"] = model[5].state_dict()
        if len(model) >= 7:
            model_dict.update({
                "ge_encoder": model[4].state_dict(),
                "cnv_encoder": model[5].state_dict(),
                "mu_encoder": model[6].state_dict(),
            })
        if len(model) == 8:
            model_dict["film_layer"] = model[7].state_dict()

        # Compose filename
        if hasattr(args, "fold_idx") and args.fold_idx is not None and hasattr(args, "scenario"):
            filename = f"model_{args.scenario}_F{args.fold_idx + 1}.pt"
        elif hasattr(args, "scenario"):
            filename = f"model_{args.scenario}.pt"
        else:
            filename = "model.pt"
        torch.save(model_dict, os.path.join(model_path, filename))
        self.last_best = score


#########

class GeneAttentionAggregator(nn.Module):
    def __init__(self, input_dim=3072, hidden_dim=768, ff_dim=1024):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, hidden_dim)
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, geneA_emb, geneA_neighbors, geneB_neighbors, geneA_kg_emb, geneB_kg_emb):
        # geneA_emb: [B, hidden]
        # geneA_neighbors: [B, N, hidden]
        # geneB_neighbors: [B, N, hidden]
        # geneA_kg_emb: [B, hidden]
        # geneB_kg_emb: [B, hidden]

        compressed_geneA = self.proj(geneA_emb)  # [B, hidden]
        q = compressed_geneA.unsqueeze(1)  # [B, 1, hidden]
        geneA_kg_emb_expanded = geneA_kg_emb.unsqueeze(1)  # [B, 1, hidden]
        geneB_kg_emb_expanded = geneB_kg_emb.unsqueeze(1)  # [B, 1, hidden]
        geneA_neighbors = torch.cat([geneA_neighbors, geneA_kg_emb_expanded], dim=1)  # [B, N+1, hidden]
        geneB_neighbors = torch.cat([geneB_neighbors, geneB_kg_emb_expanded], dim=1)  # [B, N+1, hidden]
        k_v = torch.cat([geneA_neighbors, geneB_neighbors], dim=1)  # [B, 2*(N+1), hidden]

        attn_out, _ = self.attn(q, k_v, k_v)  # [B, 1, hidden]
        attn_out = attn_out.squeeze(1)  # [B, hidden]

        x = self.norm1(attn_out + compressed_geneA)
        ff_out = self.ff(x)
        out = self.norm2(x + ff_out)
        return compressed_geneA, out  # [B, hidden], [B, hidden]


class GenePairContextAttention(nn.Module):
    """Gene-pair token fusion + gene-gene self-attention + gene-to-context cross-attention.

    Pipeline:
      1. Fuse GenePT projection + KG aggregation per gene → geneA/B_token [B, 768]
      2. Stack into gene_query [B, 2, 768]
      3. Gene-gene self-attention: A and B see each other before querying context
      4. Cross-attention: gene_query attends to context_tokens (text + omics tokens)
      5. Residual LayerNorm + FFN

    Returns gene_query_ctx [B, 2, 768]; caller extracts a/b and computes pairwise interaction.

    Ablation modes (mutually exclusive):
      no_cross_attn=True  : skip Step 3 entirely — gene queries never read context.
      concat_context=True : replace Step 3's attention with masked mean-pool of the
                            context tokens, added to each gene query (naive fusion).
    """
    def __init__(self, hidden_dim=768, num_heads=4, ff_dim=1024, dropout=0.1,
                 no_cross_attn=False, concat_context=False):
        super().__init__()
        assert not (no_cross_attn and concat_context), \
            "no_cross_attn and concat_context are mutually exclusive"
        self.no_cross_attn = no_cross_attn
        self.concat_context = concat_context
        self.gene_token_norm = nn.LayerNorm(hidden_dim)

        self.gene_self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True, dropout=dropout
        )
        self.norm_self = nn.LayerNorm(hidden_dim)

        self.gene_cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True, dropout=dropout
        )
        self.norm_cross = nn.LayerNorm(hidden_dim)

        # concat_context fusion: project pooled context before adding to gene query
        self.ctx_pool_proj = nn.Linear(hidden_dim, hidden_dim) if concat_context else None

        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
        )
        self.norm_ff = nn.LayerNorm(hidden_dim)

    def forward(self, expressed_geneA, geneA_agg, expressed_geneB, geneB_agg, context_tokens, context_mask):
        # expressed_geneA/B: [B, 768]  — projected GenePT (from GeneAttentionAggregator)
        # geneA/B_agg:       [B, 768]  — KG-aggregated gene embedding
        # context_tokens:    [B, S, 768] — BiomedBERT token sequence + optional omics tokens
        # context_mask:      [B, S]    — True = padding (ignored in cross-attn)

        # Step 1: fuse GenePT + KG per gene, stack into query
        geneA_token = self.gene_token_norm(expressed_geneA + geneA_agg)   # [B, 768]
        geneB_token = self.gene_token_norm(expressed_geneB + geneB_agg)   # [B, 768]
        gene_query = torch.stack([geneA_token, geneB_token], dim=1)        # [B, 2, 768]

        # Step 2: gene-gene self-attention — let A and B see each other before querying context
        sa_out, _ = self.gene_self_attn(gene_query, gene_query, gene_query)
        gene_query = self.norm_self(gene_query + sa_out)                   # [B, 2, 768]

        # Step 3: gene-to-context fusion
        if self.no_cross_attn:
            # Ablation: gene queries never read context tokens
            pass
        elif self.concat_context:
            # Ablation: masked mean-pool context tokens, project, add to each gene query
            valid = (~context_mask).unsqueeze(-1).float()                  # [B, S, 1]
            denom = valid.sum(dim=1).clamp(min=1.0)                        # [B, 1]
            ctx_pooled = (context_tokens * valid).sum(dim=1) / denom       # [B, 768]
            ctx_pooled = self.ctx_pool_proj(ctx_pooled).unsqueeze(1)       # [B, 1, 768]
            gene_query = self.norm_cross(gene_query + ctx_pooled)          # [B, 2, 768]
        else:
            ctx_out, _ = self.gene_cross_attn(
                gene_query, context_tokens, context_tokens,
                key_padding_mask=context_mask
            )
            gene_query = self.norm_cross(gene_query + ctx_out)             # [B, 2, 768]

        # Step 4: FFN
        gene_query = self.norm_ff(gene_query + self.ff(gene_query))        # [B, 2, 768]

        return gene_query


class Predictor(nn.Module):
    def __init__(self, input_dim=768 * 2, hidden_dim=512, output_dim=2, dropout_rate=0.1):
        super(Predictor, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, features):
        return self.mlp(features)


class LLMEncoder(nn.Module):
    def __init__(self, model, model_type="auto"):
        super().__init__()
        self.model = model
        self.model_type = model_type.lower()
        # For auto-detection if not specified
        if model_type == "auto":
            if hasattr(model, "encoder"):
                self.model_type = "bert"
            elif hasattr(model, "layers"):
                self.model_type = "qwen"
            else:
                raise ValueError("Unknown model type. Please specify model_type explicitly.")

    def forward(self, inputs, attention_mask=None):
        """
        For BERT: 
          inputs = token embeddings ([batch, seq_len, hidden_dim])
        For Qwen3: 
          inputs = input_ids ([batch, seq_len])
        """
        if self.model_type == "bert":
            # print("token_emb shape:", token_emb.shape) # [batch_size, seq_len, hidden_size]
            # print("attention_mask shape:", attention_mask.shape) # [batch_size, seq_len]

            # attention_mask: [batch_size, seq_len] with values 0/1
            extended_attention_mask = attention_mask[:, None, None, :]  # [batch, 1, 1, seq_len]
            extended_attention_mask = extended_attention_mask.to(dtype=inputs.dtype)  # float
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0  # 1->0, 0->-10000
            
            encoder_outputs = self.model.encoder(inputs, attention_mask=extended_attention_mask, output_attentions=False)
            return encoder_outputs[0]  # [batch_size, seq_len, hidden_size]
        
        elif self.model_type == "qwen":
            # Ensure inputs (input_ids) are integers and squeeze middle dim if needed
            if not torch.is_floating_point(inputs):
                input_ids = inputs
            else:
                input_ids = inputs.long()
            # Squeeze the middle dimension if present (e.g., [B, 1, S] -> [B, S])
            if input_ids.dim() == 3 and input_ids.shape[1] == 1:
                input_ids = input_ids.squeeze(1)
            input_ids = input_ids.to(next(self.model.parameters()).device)
            outputs = self.model(input_ids)
            return outputs[0]  # [batch_size, seq_len, hidden_size]
        
        else:
            raise ValueError("Unsupported model type: " + str(self.model_type))

class OmicsEncoder(nn.Module):
    def __init__(self, input_dim=942, output_dim=256, dropout_rate=0.1):
        super(OmicsEncoder, self).__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x):
        return self.net(x)


class PairFiLM(nn.Module):
    """FiLM conditioning: modulate pair_text_ctx [B, pair_dim] with omics_summary [B, omics_dim].
    pair_ctx = pair_text + gamma * LN(pair_text) + beta
    """
    def __init__(self, omics_dim=768, pair_dim=2304):
        super().__init__()
        self.film_proj = nn.Linear(omics_dim, pair_dim * 2)
        self.norm = nn.LayerNorm(pair_dim)

    def forward(self, pair_text, omics_summary):
        # pair_text:     [B, pair_dim]
        # omics_summary: [B, omics_dim]
        gamma, beta = self.film_proj(omics_summary).chunk(2, dim=-1)  # each [B, pair_dim]
        return pair_text + gamma * self.norm(pair_text) + beta


class PathwayTokenEncoder(nn.Module):
    """Encode per-pathway omics scores into pathway tokens for cross-attention KV.

    Shared MLP applied independently to each pathway position,
    with learnable pathway identity embeddings (symmetric to GeneSetEncoder).

    Input : [B, P, n_scores]  — n_scores=3 (expr_score, cnv_score, mut_score)
    Output: [B, P, output_dim]
    """
    def __init__(self, n_scores=3, hidden_dim=128, n_pathways=50, output_dim=768, dropout=0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_scores, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pathway_id_emb = nn.Embedding(n_pathways, hidden_dim)
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x):
        # x: [B, P, n_scores]
        B, P, _ = x.shape
        h = self.encoder(x)                                        # [B, P, hidden_dim]
        pw_idx = torch.arange(P, device=x.device)
        h = h + self.pathway_id_emb(pw_idx).unsqueeze(0)          # [B, P, hidden_dim]
        return self.proj(h)                                        # [B, P, output_dim]


class GeneSetEncoder(nn.Module):
    """Gene-level pathway omics encoder with attention pooling and pathway identity embeddings.

    Per-gene shared MLP → masked attention pooling over G_max genes → pathway identity residual
    → projection to output_dim.

    Input : gene_omics [B, P, G_max, n_omics]  — float32
            gene_mask  [B, P, G_max]            — bool, True = padding (excluded from pooling)
    Output: [B, P, output_dim]
    """
    def __init__(self, n_omics=3, gene_hidden=64, n_pathways=50, output_dim=768, dropout=0.1):
        super().__init__()
        self.gene_mlp = nn.Sequential(
            nn.Linear(n_omics, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, gene_hidden),
        )
        self.attn_score = nn.Linear(gene_hidden, 1, bias=False)
        self.pathway_id_emb = nn.Embedding(n_pathways, gene_hidden)
        self.proj = nn.Sequential(
            nn.Linear(gene_hidden, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, gene_omics, gene_mask):
        # gene_omics: [B, P, G_max, n_omics]
        # gene_mask:  [B, P, G_max]  True = padding
        B, P, G, _ = gene_omics.shape

        h = self.gene_mlp(gene_omics)                          # [B, P, G, gene_hidden]

        scores = self.attn_score(h).squeeze(-1)                # [B, P, G]
        scores = scores.masked_fill(gene_mask, float('-inf'))
        weights = torch.softmax(scores, dim=-1)                # [B, P, G]
        pooled = (weights.unsqueeze(-1) * h).sum(dim=2)       # [B, P, gene_hidden]

        pw_idx = torch.arange(P, device=gene_omics.device)
        pooled = pooled + self.pathway_id_emb(pw_idx).unsqueeze(0)  # [B, P, gene_hidden]

        return self.proj(pooled)                               # [B, P, output_dim]


class GEPathwayTokenEncoder(nn.Module):
    """GE-only pathway encoder with 6 distribution stats, no pathway identity embedding.

    Input : [B, P, 6]  — (expr_mean, expr_std, q25, q75, frac_up, frac_dn)
    Output: [B, P, output_dim]
    """
    def __init__(self, n_stats: int = 6, hidden_dim: int = 128, output_dim: int = 768, dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_stats, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, P, n_stats]
        return self.proj(self.encoder(x))   # [B, P, output_dim]


class TextPathwayFusion(nn.Module):
    """1-layer Transformer self-attention fusing text tokens and pathway tokens.

    Concatenates text and pathway tokens, runs a single TransformerEncoderLayer so
    both modalities can attend to each other, then returns the full fused sequence as
    context for gene cross-attention downstream.

    Input:
        text_tokens       [B, S, hidden_dim]  — LLM encoder output
        pathway_tok       [B, P, hidden_dim]  — GEPathwayTokenEncoder output
        text_padding_mask [B, S]              — True = padding token (ignored)
    Output:
        fused  [B, S+P, hidden_dim]
    """
    def __init__(self, hidden_dim: int = 768, nhead: int = 8, ff_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=1)

    def forward(
        self,
        text_tokens: torch.Tensor,
        pathway_tok: torch.Tensor,
        text_padding_mask: torch.Tensor,
        pathway_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # pathway_padding_mask [B,P] (True=pad). None → all pathway tokens valid
        # (back-compat for ge_pathway_selfatt, whose P pathway tokens are always present).
        B, P = pathway_tok.shape[0], pathway_tok.shape[1]
        combined = torch.cat([text_tokens, pathway_tok], dim=1)                          # [B, S+P, D]
        pw_mask  = (torch.zeros(B, P, dtype=torch.bool, device=pathway_tok.device)
                    if pathway_padding_mask is None else pathway_padding_mask)
        src_mask = torch.cat([text_padding_mask, pw_mask], dim=1)                        # [B, S+P]
        return self.transformer(combined, src_key_padding_mask=src_mask)                 # [B, S+P, D]


class PairPathwayEvidenceEncoder(nn.Module):
    """Encode pair-conditioned Reactome evidence tokens (pair_pathway_evidence mode).

    Each of the K selected pathways carries a numeric feature vector O_{p,ij,k} plus a
    FROZEN BiomedBERT description embedding (the generalizable pathway identity anchor —
    there is deliberately NO learnable pathway-id embedding). Only `desc_proj` learns to
    re-space the frozen descriptions; the description bank itself lives in the runtime
    builder and never enters this module's state_dict / the optimizer.

    Input : numeric [B,K,F], desc_emb [B,K,desc_dim] (frozen), pad_mask [B,K] (True=pad)
    Output: tokens  [B,K,output_dim]
    """
    def __init__(self, n_features: int, desc_dim: int = 768, hidden_dim: int = 256,
                 output_dim: int = 768, dropout: float = 0.1):
        super().__init__()
        self.num_mlp = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        self.desc_proj = nn.Linear(desc_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, numeric, desc_emb, pad_mask):
        tok = self.num_mlp(numeric) + self.desc_proj(desc_emb)     # [B,K,output_dim]
        tok = self.norm(tok)
        tok = tok.masked_fill(pad_mask.unsqueeze(-1), 0.0)         # zero padded slots
        return tok

