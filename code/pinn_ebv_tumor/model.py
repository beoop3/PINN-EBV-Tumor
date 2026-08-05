from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn

from .ode import scale_parameters


class StateEncoder(nn.Module):
    def __init__(self, feature_dim: int = 5, width: int = 128, depth: int = 4) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        size = feature_dim + 1
        for _ in range(depth):
            layers.extend((nn.Linear(size, width), nn.Tanh()))
            size = width
        layers.append(nn.Linear(size, 6))
        self.network = nn.Sequential(*layers)

    def forward(self, time: Tensor, features: Tensor) -> Tensor:
        if time.ndim == 1:
            time = time.unsqueeze(-1)
        return cast(Tensor, self.network(torch.cat((time, features), dim=-1)))


class ParameterHypernetwork(nn.Module):
    def __init__(
        self, feature_dim: int = 5, width: int = 64, depth: int = 3, dropout: float = 0.1
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        size = feature_dim
        for _ in range(depth):
            layers.extend(
                (nn.Linear(size, width), nn.BatchNorm1d(width), nn.ReLU(), nn.Dropout(dropout))
            )
            size = width
        layers.append(nn.Linear(size, 22))
        self.network = nn.Sequential(*layers)

    def forward(self, features: Tensor) -> Tensor:
        return scale_parameters(self.network(features))


class PINNEBVTumor(nn.Module):
    def __init__(
        self,
        feature_dim: int = 5,
        encoder_width: int = 128,
        encoder_depth: int = 4,
        hyper_width: int = 64,
        hyper_depth: int = 3,
        hyper_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = StateEncoder(feature_dim, encoder_width, encoder_depth)
        self.hypernetwork = ParameterHypernetwork(
            feature_dim, hyper_width, hyper_depth, hyper_dropout
        )
        self.risk_head = nn.Linear(6, 1)

    def forward(self, time: Tensor, features: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        states = self.encoder(time, features)
        parameters = self.hypernetwork(features)
        risk = self.risk_head(states).squeeze(-1)
        return states, parameters, risk
