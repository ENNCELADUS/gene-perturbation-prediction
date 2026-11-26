import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, q_dim: int, kv_dim: int, embed_dim: int, num_heads: int = 4, dropout: float = 0.1) -> None:
        super(MultiHeadCrossAttention, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * self.num_heads == self.embed_dim, \
            f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."

        self.scale = self.head_dim ** (-0.5)
        self.dropout = dropout

        self.q_proj = nn.Linear(q_dim, self.embed_dim)
        self.k_proj = nn.Linear(kv_dim, self.embed_dim)
        self.v_proj = nn.Linear(kv_dim, self.embed_dim)

        self.out_proj = nn.Linear(self.embed_dim, q_dim)

        self._reset_parameters()

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        self.q_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.k_proj.weight)
        self.k_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.v_proj.weight)
        self.v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.out_proj.weight)
        self.out_proj.bias.data.fill_(0)

    def forward(self, q: torch.Tensor, q_mask: torch.Tensor | None, 
                kv: torch.Tensor | None, kv_mask: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Perform cross attention between any 2 modals (with masks).

        Args:
            q (Tensor): [B, Lq, D]
            q_mask (Tensor): [B, Lq]
            kv (Tensor): [B, Lk, D]
            kv_mask (Tensor): [B, Lk]

        Returns:
            attn_output (Tensor): Refined q
            attn_weights (Tensor): Attention map
        """
        if kv is None:
            kv = q.clone()
            kv_mask = q_mask.clone()

        bsz, tgt_len, _ = q.size()

        query_states = self.q_proj(q) * self.scale
        key_states = self._shape(self.k_proj(kv), -1, bsz)
        value_states = self._shape(self.v_proj(kv), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))  # [B * N, Lq, Lk]

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"
            )

        if q_mask is not None and kv_mask is not None:
            attn_mask = q_mask.unsqueeze(2) * kv_mask.unsqueeze(1)  # [B, Lq, Lk]
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)  # [B, num_heads, Lq, Lk]
            attn_mask = attn_mask.flatten(0, 1)  # [B * num_heads, Lq, Lk]
            attn_weights.masked_fill_(attn_mask.bool(), float("-inf"))
        elif q_mask is None and kv_mask is not None:
            kv_mask = (
                kv_mask[:, None, None, :].repeat(1, self.num_heads, 1, 1).flatten(0, 1)
            )
            attn_weights.masked_fill_(kv_mask.bool(), float("-inf"))
        elif q_mask is not None and kv_mask is None:
            q_mask = (
                q_mask[:, None, :, None].repeat(1, self.num_heads, 1, 1).flatten(0, 1)
            )
            attn_weights.masked_fill_(q_mask.bool(), float("-inf"))

        attn_weights = attn_weights.softmax(dim=-1)

        attn_probs = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights


class MultiLayerCrossAttention(nn.Module):
    """ Key and Value are the same across all layers. """
    def __init__(self, embed_dim: int, num_heads: int = 4, num_layers: int = 1,
                 linear_depth: int = 2, dropout: float = 0.1) -> None:
        super(MultiLayerCrossAttention, self).__init__()
        self.num_heads = num_heads

        self.layers = nn.ModuleList([
            MultiHeadCrossAttention(q_dim=embed_dim, kv_dim=embed_dim, embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.norms1 = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_layers)])
        self.feed_forwards = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim),  #linear_depth * 
                nn.ReLU(),
                nn.Dropout(dropout),
                # nn.Linear(linear_depth * embed_dim, embed_dim),
                # nn.Dropout(dropout)
            ) for _ in range(num_layers)
        ])
        self.norms2 = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_layers)])

    def forward(self, q: torch.Tensor, q_mask: torch.Tensor | None, 
                kv: torch.Tensor | None, kv_mask: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            q (Tensor): input representations of shape (B, Lq, H)
            q_mask (Tensor): bool mask of shape (B, Lq), with 1 indicating masked positions
            kv (Tensor): input representations of shape (B, Lk, H)
            kv_mask (Tensor): bool mask of shape (B, Lk), with 1 indicating masked positions

        Returns:
            q (Tensor): Output of the whole module
            attn_map (Tensor): Attention map of the last layer
        """
        for attn_layer, norm1, ff_layer, norm2 in zip(self.layers, self.norms1, self.feed_forwards, self.norms2):
            attn_output, attn_map = attn_layer(q, q_mask, kv, kv_mask)  #key_padding_mask=kv_mask, 
            q = norm1(q + attn_output)

            ff_output = ff_layer(q)
            q = norm2(q + ff_output)
        # print(attn_map.shape)
        return q, attn_map
