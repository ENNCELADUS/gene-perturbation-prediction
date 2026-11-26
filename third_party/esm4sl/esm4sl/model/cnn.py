import torch
import torch.nn as nn


class CellCNN(nn.Module):
    def __init__(self, feat_dim: int, dropout: float = 0.1):
        super(CellCNN, self).__init__()

        max_pool_size = [2, 2, 6]
        kernel_size = [16, 16, 16]

        in_channels = [6, 16, 32]
        out_channels = [16, 32, 64]

        self.cell_conv = nn.Sequential(
            nn.Conv1d(in_channels=in_channels[0], out_channels=out_channels[0], kernel_size=kernel_size[0]),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.MaxPool1d(max_pool_size[0]),

            nn.Conv1d(in_channels=in_channels[1], out_channels=out_channels[1], kernel_size=kernel_size[1]),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.MaxPool1d(max_pool_size[1]),

            nn.Conv1d(in_channels=in_channels[2], out_channels=out_channels[2], kernel_size=kernel_size[2]),
            nn.ReLU(),
            nn.MaxPool1d(max_pool_size[2]),
        )

        self.cell_linear = nn.Linear(out_channels[2], feat_dim)

        self.init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.cell_conv(x)  # [batch, out_channel, 53]
        x = x.transpose(1, 2)
        x = self.cell_linear(x)  # [batch, 53, 64] or [batch, 53, 128]
        return x

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_normal_(m.weight)
