"""KG-SLomics architecture (RGAT over PrimeKG node features)."""
import torch
import torch.nn as nn
from torch_geometric.nn import RGATConv


class OmicsLayer(nn.Module):
    """4 -> 32 -> 128."""

    def __init__(self, in_channels=4, hidden_channels=32, out_channels=128):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.c_act = nn.LeakyReLU()

    def forward(self, x):
        return self.fc2(self.c_act(self.fc1(x)))


class KGSLomics(nn.Module):
    def __init__(self, num_relations, in_channels=256, hidden_channels=128,
                 out_channels=64, heads=4, dropout=0.4, num_bases=None,
                 omics_dim=4):
        super().__init__()
        self.ccle_layer = OmicsLayer(in_channels=omics_dim)
        self.conv1 = RGATConv(in_channels, hidden_channels // heads,
                              num_relations, heads=heads, num_bases=num_bases)
        self.conv2 = RGATConv(hidden_channels, out_channels // heads,
                              num_relations, heads=heads, concat=True,
                              num_bases=num_bases)
        self.skip_connection = nn.Sequential(
            nn.Linear(in_channels, hidden_channels), nn.LeakyReLU(),
            nn.Linear(hidden_channels, out_channels))
        self.sl_predictor = nn.Sequential(
            nn.Linear(out_channels * 2, out_channels // 4), nn.LeakyReLU(),
            nn.Linear(out_channels // 4, 1))
        self.act = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)

    def encode(self, kg_emb, omics, edge_index, edge_type):
        """kg_emb / omics are already restricted to the sampled subgraph nodes."""
        x_in = torch.cat((kg_emb, self.ccle_layer(omics)), dim=1)
        x = self.conv1(x_in, edge_index, edge_type)
        x = self.dropout(self.act(x))
        x = self.conv2(x, edge_index, edge_type)
        x = x + self.skip_connection(x_in)
        return self.act(x)

    def decode(self, z, edge_label_index):
        u, v = z[edge_label_index[0]], z[edge_label_index[1]]
        x = torch.cat([u * v, torch.abs(u - v)], dim=1)
        return self.sl_predictor(x).squeeze(dim=1)
