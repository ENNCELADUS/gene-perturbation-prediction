import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPooling(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        attention_heads: int = 4,
        dropout_rate: float = 0.1,
    ):
        super(AttentionPooling, self).__init__()

        self.attention_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, 1),
            ) for _ in range(attention_heads)
        ])

        self.dropout = nn.Dropout(dropout_rate)

        self.output_layer = nn.Linear(hidden_size * attention_heads, hidden_size)

        self._init_weights()

    def _init_weights(self):
        # 使用Xavier初始化线性层的权重
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)  # 将偏置初始化为0

    def forward(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_size = last_hidden_state.size()  # [batch_size, seq_len, hidden_size]
        attention_embeddings = []

        # 对每个注意力头计算注意力嵌入
        for attention in self.attention_heads:
            # 注意力权重计算
            w = attention(last_hidden_state).view(batch_size, seq_len).squeeze(-1)  # [batch_size, seq_len]

            # 使用mask填充位置
            w = w.masked_fill(attention_mask == 1, float('-inf'))  # 使用mask填充位置
            w = F.softmax(w, dim=-1)  # 在序列长度上应用softmax，形状仍为 [batch_size, seq_len]

            w = self.dropout(w)  # 在每个注意力头的最后添加Dropout

            # 计算加权和，形状为 [batch_size, hidden_size]
            head_embedding = torch.bmm(w.unsqueeze(1), last_hidden_state).squeeze(1)
            attention_embeddings.append(head_embedding)

        # 拼接所有注意力头的输出
        attention_output = torch.cat(attention_embeddings, dim=-1)  # [batch_size, hidden_size * num_attention_heads]

        # 通过输出层
        attention_output = self.output_layer(attention_output)  # [batch_size, hidden_size]

        return attention_output


class CNNPooling(nn.Module):
    def __init__(self, hidden_dim: int, downsampling_dim: int = 256, kernel_size: int = 3, dropout: float = 0.1) -> None:
        super().__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size

        self.conv = nn.Conv1d(hidden_dim, downsampling_dim, kernel_size, padding=(kernel_size - 1) // 2)
        self.activation1 = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(downsampling_dim)
        self.mlp = nn.Linear(downsampling_dim, 1)
        self.activation2 = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs downsampling through CNN.
        x: [B (batch_size), L (protein len), E (hidden dim)]
        D: downsampling_dim
        """
        x = x.transpose(1, 2)  # [B, L, E]
        x = self.conv(x)  # [B, E, D]
        x = self.activation1(x)
        x = self.dropout(x)
        x = self.norm(x)  # [B, E, D]
        x = self.mlp(x)  # [B, E, 1]
        x = self.activation2(x)  # [B, E, 1]
        return x.squeeze(-1)  # [B, E]
