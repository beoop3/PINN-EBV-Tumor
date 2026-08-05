from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import torch
from torch import Tensor


@dataclass(frozen=True)
class Cohort:
    features: Tensor
    duration: Tensor
    event: Tensor
    states: Tensor


class Standardizer:
    def __init__(self) -> None:
        self.mean: np.ndarray | None = None
        self.scale: np.ndarray | None = None

    def fit(self, values: np.ndarray) -> Standardizer:
        self.mean = np.nanmean(values, axis=0)
        self.scale = np.nanstd(values, axis=0)
        self.scale[self.scale == 0] = 1.0
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.mean is None or self.scale is None:
            raise RuntimeError("standardizer has not been fitted")
        filled = np.where(np.isnan(values), self.mean, values)
        return cast(np.ndarray, (filled - self.mean) / self.scale)


def load_cohort(
    path: str | Path,
    feature_columns: list[str],
    time_column: str = "time",
    event_column: str = "event",
) -> Cohort:
    frame = pd.read_csv(path)
    features = frame[feature_columns].to_numpy(dtype=np.float32)
    standardizer = Standardizer().fit(features)
    transformed = standardizer.transform(features)
    state_columns = ["latent", "lytic", "virus", "immune", "tumor", "sensitivity"]
    states = frame[state_columns].to_numpy(dtype=np.float32)
    return Cohort(
        torch.from_numpy(transformed),
        torch.tensor(frame[time_column].to_numpy(), dtype=torch.float32),
        torch.tensor(frame[event_column].to_numpy(), dtype=torch.float32),
        torch.from_numpy(states),
    )


def stratified_split(
    event: Tensor, train_fraction: float = 0.60, validation_fraction: float = 0.15, seed: int = 42
) -> tuple[Tensor, Tensor, Tensor]:
    generator = torch.Generator().manual_seed(seed)
    groups = []
    for value in (0, 1):
        indices = torch.where(event == value)[0]
        groups.append(indices[torch.randperm(indices.numel(), generator=generator)])
    train_parts, validation_parts, test_parts = [], [], []
    for indices in groups:
        first = round(indices.numel() * train_fraction)
        second = first + round(indices.numel() * validation_fraction)
        train_parts.append(indices[:first])
        validation_parts.append(indices[first:second])
        test_parts.append(indices[second:])
    return torch.cat(train_parts), torch.cat(validation_parts), torch.cat(test_parts)
