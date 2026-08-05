import numpy as np
import torch
from pinn_ebv_tumor.losses import cox_partial_likelihood, physics_residual
from pinn_ebv_tumor.metrics import brier_score, concordance_index, parameter_recovery, rmse
from pinn_ebv_tumor.model import PINNEBVTumor
from pinn_ebv_tumor.ode import (
    PARAMETER_RANGES,
    ebv_tumor_rhs,
    euler_integrate,
    radiosensitivity,
    scale_parameters,
)


def test_parameter_bounds() -> None:
    values = scale_parameters(torch.zeros(4, 22))
    expected = PARAMETER_RANGES.mean(dim=1)
    assert torch.allclose(values, expected.expand_as(values))


def test_rhs_shape_and_finiteness() -> None:
    states = torch.ones(7, 6)
    parameters = scale_parameters(torch.zeros(7, 22))
    result = ebv_tumor_rhs(states, parameters, 2.0)
    assert result.shape == states.shape
    assert torch.isfinite(result).all()


def test_latent_dominance_reduces_sensitivity() -> None:
    parameters = scale_parameters(torch.zeros(2, 22))
    states = torch.ones(2, 6)
    states[0, :2] = torch.tensor([100.0, 1.0])
    states[1, :2] = torch.tensor([1.0, 100.0])
    values = radiosensitivity(states, parameters)
    assert values[0] < values[1]


def test_euler_integrator() -> None:
    initial = torch.ones(2, 6)
    parameters = scale_parameters(torch.zeros(2, 22))
    trajectory = euler_integrate(initial, parameters, torch.linspace(0.0, 0.01, 4))
    assert trajectory.shape == (4, 2, 6)
    assert torch.isfinite(trajectory).all()


def test_model_outputs() -> None:
    model = PINNEBVTumor(
        encoder_width=16, encoder_depth=2, hyper_width=12, hyper_depth=2, hyper_dropout=0.0
    )
    model.eval()
    states, parameters, risk = model(torch.ones(8, 1), torch.randn(8, 5))
    assert states.shape == (8, 6)
    assert parameters.shape == (8, 22)
    assert risk.shape == (8,)


def test_physics_gradient() -> None:
    model = PINNEBVTumor(
        encoder_width=16, encoder_depth=2, hyper_width=12, hyper_depth=2, hyper_dropout=0.0
    )
    model.eval()
    time = torch.linspace(0.0, 1.0, 8).unsqueeze(-1).requires_grad_(True)
    states, parameters, _ = model(time, torch.randn(8, 5))
    residual = physics_residual(states, time, parameters)
    assert residual.shape == (8, 6)


def test_cox_ordering() -> None:
    risk = torch.tensor([3.0, 2.0, 1.0])
    duration = torch.tensor([1.0, 2.0, 3.0])
    event = torch.ones(3)
    assert cox_partial_likelihood(risk, duration, event) < cox_partial_likelihood(
        -risk, duration, event
    )


def test_metrics() -> None:
    duration = np.array([1.0, 2.0, 3.0])
    event = np.ones(3)
    risk = np.array([3.0, 2.0, 1.0])
    assert concordance_index(duration, event, risk) == 1.0
    assert brier_score(event, np.zeros(3)) == 0.0
    assert rmse(np.ones(3), np.ones(3)) == 0.0
    observed = np.arange(12, dtype=float).reshape(6, 2)
    assert parameter_recovery(observed, observed) == 1.0
