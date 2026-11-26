from typing import Optional

import torch
import torch.nn as nn

from coach_pl.configuration import configurable
from coach_pl.model import MODEL_REGISTRY

__all__ = ["MLP"]


@MODEL_REGISTRY.register()
class MLP(nn.Module):

    @configurable
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: Optional[int] = None,
        act_layer: nn.Module = nn.GELU,
        norm_layer: Optional[nn.Module] = nn.BatchNorm1d,
        bias: bool = True,
        drop: float = 0.0,
    ):
        super().__init__()

        if not hidden_features:
            hidden_features = 2 * in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    @classmethod
    def from_config(cls, cfg):
        return {
            "in_features": cfg.MODEL.IN_CHANNELS,
            "out_features": cfg.MODEL.OUT_CHANNELS,
            "hidden_features": cfg.MODEL.HIDDEN_CHANNELS,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inputs:
            x (torch.Tensor): Input tensor of shape [..., in_features]

        Returns:
            torch.Tensor: Output tensor of shape [..., out_features]
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x.squeeze(-1)  # [B]


class MLP_3D(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: Optional[int] = None,
        act_layer: nn.Module = nn.GELU,
        norm_layer: Optional[nn.Module] = nn.BatchNorm1d,
        bias: bool = True,
        drop: float = 0.0,
    ):
        super().__init__()

        if not hidden_features:
            hidden_features = 2 * in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, E]
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = x.transpose(1, 2)
        x = self.fc2(x)
        x = self.drop2(x)
        return x.squeeze(-1)
