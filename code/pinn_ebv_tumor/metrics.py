from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def concordance_index(
    duration: NDArray[np.float64], event: NDArray[np.float64], risk: NDArray[np.float64]
) -> float:
    concordant = 0.0
    comparable = 0.0
    for left in range(duration.size):
        for right in range(left + 1, duration.size):
            if duration[left] == duration[right]:
                continue
            early, late = (left, right) if duration[left] < duration[right] else (right, left)
            if event[early] == 0:
                continue
            comparable += 1.0
            if risk[early] == risk[late]:
                concordant += 0.5
            elif risk[early] > risk[late]:
                concordant += 1.0
    return concordant / comparable if comparable else float("nan")


def brier_score(event: NDArray[np.float64], survival_probability: NDArray[np.float64]) -> float:
    return float(np.mean((event - (1.0 - survival_probability)) ** 2))


def rmse(predicted: NDArray[np.float64], observed: NDArray[np.float64]) -> float:
    return float(np.sqrt(np.mean((predicted - observed) ** 2)))


def parameter_recovery(predicted: NDArray[np.float64], observed: NDArray[np.float64]) -> float:
    residual = np.sum((observed - predicted) ** 2)
    total = np.sum((observed - observed.mean(axis=0)) ** 2)
    return float(1.0 - residual / total)
