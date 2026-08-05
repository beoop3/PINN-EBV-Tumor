from __future__ import annotations

import torch
from torch import Tensor

PARAMETER_NAMES = (
    "s_l",
    "r_l",
    "k_l",
    "a_ly",
    "d_l",
    "d_y",
    "p",
    "c_v",
    "k_el",
    "k_ey",
    "k_v",
    "s_e",
    "rho",
    "eta",
    "h",
    "d_e",
    "r_t",
    "k_t",
    "k_et",
    "alpha_0",
    "s_base",
    "gamma",
)

PARAMETER_RANGES = torch.tensor(
    [
        [0.01, 1.0],
        [0.001, 0.05],
        [1e3, 1e6],
        [0.001, 0.1],
        [0.001, 0.02],
        [0.1, 1.0],
        [10.0, 1000.0],
        [1.0, 23.0],
        [1e-6, 1e-3],
        [1e-5, 1e-2],
        [1e-6, 1e-3],
        [0.1, 10.0],
        [0.01, 0.5],
        [0.01, 1.0],
        [1e2, 1e5],
        [0.01, 0.1],
        [0.001, 0.05],
        [1e6, 1e10],
        [1e-8, 1e-5],
        [0.1, 0.5],
        [0.3, 0.7],
        [0.05, 0.5],
    ],
    dtype=torch.float32,
)


def scale_parameters(raw: Tensor, ranges: Tensor | None = None) -> Tensor:
    bounds = PARAMETER_RANGES.to(raw) if ranges is None else ranges.to(raw)
    return bounds[:, 0] + torch.sigmoid(raw) * (bounds[:, 1] - bounds[:, 0])


def radiosensitivity(states: Tensor, parameters: Tensor, epsilon: float = 1e-8) -> Tensor:
    latent = states[..., 0]
    lytic = states[..., 1]
    ratio = latent / (latent + lytic + epsilon)
    return parameters[..., 20] - parameters[..., 21] * torch.sigmoid(ratio)


def ebv_tumor_rhs(states: Tensor, parameters: Tensor, dose: Tensor | float = 0.0) -> Tensor:
    latent, y, v, e, t, _ = states.unbind(dim=-1)
    p = parameters
    d = torch.as_tensor(dose, dtype=states.dtype, device=states.device)
    dl = (
        p[..., 0]
        + p[..., 1] * latent * (1.0 - latent / p[..., 2])
        - p[..., 3] * latent
        - p[..., 4] * latent
        - p[..., 8] * e * latent
    )
    dy = p[..., 3] * latent - p[..., 5] * y - p[..., 9] * e * y
    dv = p[..., 6] * y - p[..., 7] * v - p[..., 10] * e * v
    de = (
        p[..., 11]
        + p[..., 12] * e * (y + p[..., 13] * v) / (y + p[..., 13] * v + p[..., 14])
        - p[..., 15] * e
    )
    s = radiosensitivity(states, parameters)
    dt = (
        p[..., 16] * t * (1.0 - t / p[..., 17])
        - p[..., 18] * e * t
        - p[..., 19] * (1.0 + s) * d * t
    )
    ds = s - states[..., 5]
    return torch.stack((dl, dy, dv, de, dt, ds), dim=-1)


def euler_integrate(
    initial: Tensor, parameters: Tensor, times: Tensor, dose: Tensor | None = None
) -> Tensor:
    values = [initial]
    state = initial
    for index in range(1, times.shape[0]):
        step = times[index] - times[index - 1]
        current_dose = 0.0 if dose is None else dose[index - 1]
        state = state + step * ebv_tumor_rhs(state, parameters, current_dose)
        values.append(state)
    return torch.stack(values, dim=0)
