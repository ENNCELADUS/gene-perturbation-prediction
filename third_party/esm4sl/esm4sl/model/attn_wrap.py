import torch
import torch.nn as nn

from coach_pl.configuration import configurable
from coach_pl.model import MODEL_REGISTRY
from .mlp import MLP, MLP_3D
from .attention import MultiLayerCrossAttention
from .cnn import CellCNN
from .pooling import AttentionPooling, CNNPooling

__all__ = ["AttnWrap"]


def padded_to_variadic(rep: torch.Tensor, mask: torch.Tensor) -> list[torch.Tensor]:
    num_samples = mask.size(0)
    var_reps = []

    for i in range(num_samples):
        valid_num = (~mask[i]).sum().item()
        real_rep = rep[i, :valid_num]
        var_reps.append(real_rep)

    return var_reps


@MODEL_REGISTRY.register()
class AttnWrap(nn.Module):

    @configurable
    def __init__(self, in_channels: int = 1280, kernel_size: int = 3,
                 embed_dim: int = 256, num_heads: int = 4, num_layers: int = 1, 
                 linear_depth: int = 2, dropout: float = 0.1, out_channels: int = 1,
                 pooling: str = 'max', use_cell: bool | None = None) -> None:
        super().__init__()
        # self.gene_conv = nn.Sequential(
        #     nn.Conv1d(in_channels=in_channels, out_channels=in_channels//2, kernel_size=kernel_size, padding=(kernel_size-1)//2),
        #     nn.ReLU(),
        #     nn.Dropout(p=dropout),
        #     nn.Conv1d(in_channels=in_channels//2, out_channels=embed_dim, kernel_size=kernel_size, padding=(kernel_size-1)//2),
        #     nn.ReLU(),
        #     nn.Dropout(p=dropout),
        # )
        self.gene_conv = MLP_3D(in_features=in_channels, out_features=embed_dim,
                                hidden_features=in_channels//2, drop=dropout)

        self.gene_SA = MultiLayerCrossAttention(embed_dim, num_heads, num_layers, linear_depth, dropout)
        self.gene_CA = MultiLayerCrossAttention(embed_dim, num_heads, num_layers, linear_depth, dropout)

        self.use_cell = use_cell
        if use_cell:
            self.cell_conv = CellCNN(feat_dim=embed_dim)
            self.gene_cell_CA = MultiLayerCrossAttention(embed_dim, num_heads, num_layers, linear_depth, dropout)
            self.cell_SA = MultiLayerCrossAttention(embed_dim, num_heads, num_layers, linear_depth, dropout)
            self.cell_CA = MultiLayerCrossAttention(embed_dim, num_heads, num_layers, linear_depth, dropout)

        self.pooling = pooling
        if pooling == "attention":
            self.attn_pool = AttentionPooling(hidden_size=embed_dim, attention_heads=2, dropout_rate=dropout)
        elif pooling == "cnn":
            self.cnn_pool = CNNPooling(hidden_dim=embed_dim, downsampling_dim=embed_dim//2)

        self.head = MLP(in_features=embed_dim*2, out_features=out_channels, norm_layer=nn.BatchNorm1d)

    @classmethod
    def from_config(cls, cfg):
        return {
            "in_channels": cfg.MODEL.IN_CHANNELS,
            "kernel_size": cfg.MODEL.KERNEL_SIZE,
            "embed_dim": cfg.MODEL.EMBED_DIM,
            "num_heads": cfg.MODEL.NUM_HEADS,
            "num_layers": cfg.MODEL.NUM_LAYERS,
            "linear_depth": cfg.MODEL.LINEAR_DEPTH,
            "dropout": cfg.MODEL.DROPOUT,
            "pooling": cfg.MODEL.POOLING,
            "out_channels": cfg.MODEL.OUT_CHANNELS,
            "use_cell": cfg.DATASET.CELL_LINE,
        }

    def forward(self, gene1: torch.Tensor, gene1_mask: torch.Tensor,
                gene2: torch.Tensor, gene2_mask: torch.Tensor,
                cell: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            gene1, gene2: [batch_size (B), seq_len (L), input_dim (Ei)]
            gene1_mask, gene2_mask: [batch_size (B), seq_len (L)]
        
        Returns:
            output: size [1] indicating logit
            all_attn: all attention maps
        """
        # Downsample
        # gene1 = gene1.transpose(1, 2)  # [B, Ei, L]
        gene1 = self.gene_conv(gene1)  # [B, Eo, L]
        # gene1 = gene1.transpose(1, 2)  # [B, L, Eo]
        # gene2 = gene2.transpose(1, 2)  # [B, Ei, L]
        gene2 = self.gene_conv(gene2)  # [B, Eo, L]
        # gene2 = gene2.transpose(1, 2)  # [B, L, Eo]

        if self.use_cell:  # Add cell-line information
            cell = self.cell_conv(cell)
            cell_copy = cell.clone()

            # Layer1
            gene1, attn1 = self.gene_cell_CA(gene1, gene1_mask, cell, None)
            gene2, attn2 = self.gene_cell_CA(gene2, gene2_mask, cell_copy, None)

        # Layer2
        gene1, attn3 = self.gene_SA(gene1, gene1_mask, None, None)
        gene2, attn4 = self.gene_SA(gene2, gene2_mask, None, None)

        # Layer3
        new_gene1, attn5 = self.gene_CA(gene1, gene1_mask, gene2, gene2_mask)
        new_gene2, attn6 = self.gene_CA(gene2, gene2_mask, gene1, gene1_mask)

        if self.pooling == "attention":
            gene1 = self.attn_pool(new_gene1, gene1_mask)
            gene2 = self.attn_pool(new_gene2, gene2_mask)
        elif self.pooling == "cnn":
            gene1 = self.cnn_pool(new_gene1)
            gene2 = self.cnn_pool(new_gene2)
        elif self.pooling == "mean":
            gene1 = padded_to_variadic(new_gene1, gene1_mask)
            gene2 = padded_to_variadic(new_gene2, gene2_mask)
            gene1 = torch.stack([i.mean(0) for i in gene1])  # [B, Eo]
            gene2 = torch.stack([i.mean(0) for i in gene2])  # [B, Eo]
        elif self.pooling == "max":
            gene1 = padded_to_variadic(new_gene1, gene1_mask)
            gene2 = padded_to_variadic(new_gene2, gene2_mask)
            gene1 = torch.stack([i.max(0).values for i in gene1])  # [B, Eo]
            gene2 = torch.stack([i.max(0).values for i in gene2])  # [B, Eo]
        else:
            raise ValueError('Invalid pooling method!')

        gene_embed = torch.cat((gene1, gene2), dim=1)  # [B, 2E]

        output = self.head(gene_embed)  # [B, 1]
        # output = output.squeeze(-1)

        if cell is not None:
            all_attn = [attn1, attn2, attn3, attn4, attn5, attn6]
        else:
            all_attn = [attn3, attn4, attn5, attn6]
        return output, all_attn
