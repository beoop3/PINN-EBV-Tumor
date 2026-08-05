from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import torch
import yaml

from .data import load_cohort
from .metrics import concordance_index
from .model import PINNEBVTumor
from .training import atomic_checkpoint, set_seed, train_step

LOGGER = logging.getLogger("pinn_ebv_tumor")


def read_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError("configuration root must be a mapping")
    return value


def build_model(config: dict[str, Any]) -> PINNEBVTumor:
    model = config["model"]
    return PINNEBVTumor(
        model["feature_dim"],
        model["encoder_width"],
        model["encoder_depth"],
        model["hyper_width"],
        model["hyper_depth"],
        model["hyper_dropout"],
    )


def train_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/main.yaml")
    parser.add_argument("--output", default="artifacts/model.pt")
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    config = read_config(arguments.config)
    set_seed(int(config["seed"]))
    cohort = load_cohort(
        config["data"]["path"],
        config["data"]["feature_columns"],
        config["data"]["time_column"],
        config["data"]["event_column"],
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    epoch = 0
    for phase in config["training"]["phases"]:
        weights = {
            "data": config["training"]["data_weight"],
            "boundary": config["training"]["boundary_weight"],
            "ode": phase["ode_weight"],
            "survival": phase["survival_weight"],
        }
        for _ in range(int(phase["epochs"])):
            epoch += 1
            values = train_step(
                model,
                optimizer,
                cohort.features.to(device),
                cohort.duration.to(device).unsqueeze(-1),
                cohort.states.to(device),
                cohort.duration.to(device),
                cohort.event.to(device),
                cohort.states[0].to(device),
                weights,
            )
            LOGGER.info("epoch=%d phase=%s loss=%.6f", epoch, phase["name"], values["total"])
    atomic_checkpoint(arguments.output, model, optimizer, epoch, int(config["seed"]))


def evaluate_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/main.yaml")
    parser.add_argument("--weights", required=True)
    arguments = parser.parse_args()
    config = read_config(arguments.config)
    cohort = load_cohort(
        config["data"]["path"],
        config["data"]["feature_columns"],
        config["data"]["time_column"],
        config["data"]["event_column"],
    )
    model = build_model(config)
    payload = torch.load(arguments.weights, map_location="cpu", weights_only=True)
    model.load_state_dict(payload["model"])
    model.eval()
    with torch.no_grad():
        _, _, risk = model(cohort.duration.unsqueeze(-1), cohort.features)
    score = concordance_index(
        cohort.duration.numpy().astype(float),
        cohort.event.numpy().astype(float),
        risk.numpy().astype(float),
    )
    print(json.dumps({"c_index": score}))


def infer_main() -> None:
    evaluate_main()
