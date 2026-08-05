from pathlib import Path

import torch
from pinn_ebv_tumor.model import PINNEBVTumor
from pinn_ebv_tumor.training import atomic_checkpoint, restore_checkpoint, set_seed, train_step


def test_two_step_training_and_checkpoint(tmp_path: Path) -> None:
    set_seed(42)
    model = PINNEBVTumor(
        encoder_width=16, encoder_depth=2, hyper_width=12, hyper_depth=2, hyper_dropout=0.0
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    features = torch.randn(8, 5)
    time = torch.linspace(0.1, 1.0, 8).unsqueeze(-1)
    observed = torch.randn(8, 6)
    duration = torch.linspace(1.0, 8.0, 8)
    event = torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    initial = observed[0]
    first = train_step(model, optimizer, features, time, observed, duration, event, initial)
    second = train_step(model, optimizer, features, time, observed, duration, event, initial)
    assert torch.isfinite(torch.tensor([first["total"], second["total"]])).all()
    path = tmp_path / "model.pt"
    atomic_checkpoint(path, model, optimizer, 2, 42)
    restored = PINNEBVTumor(
        encoder_width=16, encoder_depth=2, hyper_width=12, hyper_depth=2, hyper_dropout=0.0
    )
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-4)
    epoch, seed = restore_checkpoint(path, restored, restored_optimizer, torch.device("cpu"))
    assert (epoch, seed) == (2, 42)
