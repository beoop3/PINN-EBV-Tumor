from __future__ import annotations

import torch
from torch import Tensor

from .ode import ebv_tumor_rhs


def cox_partial_likelihood(risk: Tensor, duration: Tensor, event: Tensor) -> Tensor:
    order = torch.argsort(duration, descending=True)
    ordered_risk = risk[order]
    ordered_event = event[order]
    log_cumulative = torch.logcumsumexp(ordered_risk, dim=0)
    terms = ordered_risk - log_cumulative
    return -(terms * ordered_event).sum() / ordered_event.sum().clamp_min(1.0)


def physics_residual(
    states: Tensor, time: Tensor, parameters: Tensor, dose: Tensor | float = 0.0
) -> Tensor:
    derivatives = []
    for channel in range(states.shape[-1]):
        gradient = torch.autograd.grad(
            states[:, channel].sum(), time, create_graph=True, retain_graph=True
        )[0]
        derivatives.append(gradient.squeeze(-1))
    predicted = torch.stack(derivatives, dim=-1)
    return predicted - ebv_tumor_rhs(states, parameters, dose)


def composite_loss(
    states: Tensor,
    observed: Tensor,
    initial: Tensor,
    risk: Tensor,
    duration: Tensor,
    event: Tensor,
    residual: Tensor,
    data_weight: float = 1.0,
    ode_weight: float = 1.0,
    boundary_weight: float = 1.0,
    survival_weight: float = 1.0,
) -> dict[str, Tensor]:
    data_loss = torch.mean((states - observed) ** 2)
    ode_loss = torch.mean(residual**2)
    boundary_loss = torch.mean((states[duration.argmin()] - initial) ** 2)
    survival_loss = cox_partial_likelihood(risk, duration, event)
    total = (
        data_weight * data_loss
        + ode_weight * ode_loss
        + boundary_weight * boundary_loss
        + survival_weight * survival_loss
    )
    return {
        "total": total,
        "data": data_loss,
        "ode": ode_loss,
        "boundary": boundary_loss,
        "survival": survival_loss,
    }
