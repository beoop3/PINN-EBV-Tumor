from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from .losses import composite_loss, physics_residual


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_checkpoint(
    path: str | Path, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, seed: int
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "seed": seed,
        },
        temporary,
    )
    os.replace(temporary, destination)


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    features: Tensor,
    time: Tensor,
    observed: Tensor,
    duration: Tensor,
    event: Tensor,
    initial: Tensor,
    weights: dict[str, float] | None = None,
) -> dict[str, float]:
    model.train()
    time = time.detach().clone().requires_grad_(True)
    states, parameters, risk = model(time, features)
    residual = physics_residual(states, time, parameters)
    selected = weights or {}
    losses = composite_loss(
        states,
        observed,
        initial,
        risk,
        duration,
        event,
        residual,
        selected.get("data", 1.0),
        selected.get("ode", 1.0),
        selected.get("boundary", 1.0),
        selected.get("survival", 1.0),
    )
    optimizer.zero_grad(set_to_none=True)
    torch.autograd.backward(losses["total"])
    optimizer.step()
    return {name: float(value.detach()) for name, value in losses.items()}


def restore_checkpoint(
    path: str | Path, model: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device
) -> tuple[int, int]:
    payload: dict[str, Any] = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    seed = int(payload["seed"])
    set_seed(seed)
    return int(payload["epoch"]), seed
