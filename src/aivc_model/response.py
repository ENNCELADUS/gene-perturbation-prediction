"""End-to-end response encoding and differentiable mixture pooling."""

import math

import torch
import torch.nn.functional as F
from torch import nn


class ResponseEncoder(nn.Module):
    """Shared per-cell normalized-expression encoder."""

    def __init__(self, input_dim: int = 2000, latent_dim: int = 128) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, latent_dim)
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, expression: torch.Tensor) -> torch.Tensor:
        return self.norm(self.linear(expression))


class TrainableDiagonalGMM(nn.Module):
    """Trainable diagonal Gaussian responsibilities plus bag summaries."""

    def __init__(
        self,
        latent_dim: int,
        n_components: int,
        covariance_floor: float,
        init_scale: float,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.n_components = int(n_components)
        self.covariance_floor = float(covariance_floor)
        self.means = nn.Parameter(
            torch.randn(self.n_components, self.latent_dim) * float(init_scale)
        )
        self.raw_variances = nn.Parameter(
            torch.full((self.n_components, self.latent_dim), 0.54132485)
        )
        self.mixture_logits = nn.Parameter(torch.zeros(self.n_components))

    @property
    def variances(self) -> torch.Tensor:
        return F.softplus(self.raw_variances) + self.covariance_floor

    @property
    def output_dim(self) -> int:
        return 2 * self.n_components + 2 * self.latent_dim + 1

    def _component_log_prob(self, bag: torch.Tensor) -> torch.Tensor:
        variances = self.variances
        diff = bag.unsqueeze(1) - self.means.unsqueeze(0)
        gaussian = -0.5 * (
            diff.square() / variances.unsqueeze(0)
            + variances.log().unsqueeze(0)
            + math.log(2.0 * math.pi)
        ).sum(dim=2)
        return gaussian + self.mixture_logits.log_softmax(dim=0).unsqueeze(0)

    def occupancy(self, bag: torch.Tensor) -> torch.Tensor:
        return self._component_log_prob(bag).softmax(dim=1).mean(dim=0)

    def negative_log_likelihood(self, bag: torch.Tensor) -> torch.Tensor:
        return -torch.logsumexp(self._component_log_prob(bag), dim=1).mean()

    def forward(self, bag: torch.Tensor, control_bag: torch.Tensor) -> torch.Tensor:
        occupancy = self.occupancy(bag)
        control_occupancy = self.occupancy(control_bag)
        mean = bag.mean(dim=0)
        variance = bag.var(dim=0, unbiased=False)
        entropy = -(occupancy * occupancy.clamp_min(1e-8).log()).sum().view(1)
        return torch.cat(
            [occupancy, occupancy - control_occupancy, mean, variance, entropy],
            dim=0,
        )
