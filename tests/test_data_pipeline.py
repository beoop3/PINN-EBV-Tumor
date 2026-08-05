from pathlib import Path

import pandas as pd
import torch
from pinn_ebv_tumor.data import Standardizer, load_cohort, stratified_split


def test_standardizer_training_statistics() -> None:
    values = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).numpy()
    transformed = Standardizer().fit(values).transform(values)
    assert abs(float(transformed.mean())) < 1e-6


def test_cohort_loading(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "ebv_dna": [1.0, 2.0],
            "t_stage": [1.0, 2.0],
            "n_stage": [0.0, 1.0],
            "age": [40.0, 50.0],
            "sex": [0.0, 1.0],
            "time": [12.0, 18.0],
            "event": [1.0, 0.0],
            "latent": [2.0, 3.0],
            "lytic": [1.0, 1.0],
            "virus": [4.0, 3.0],
            "immune": [2.0, 2.0],
            "tumor": [5.0, 4.0],
            "sensitivity": [0.4, 0.5],
        }
    )
    path = tmp_path / "cohort.csv"
    frame.to_csv(path, index=False)
    cohort = load_cohort(path, ["ebv_dna", "t_stage", "n_stage", "age", "sex"])
    assert cohort.features.shape == (2, 5)
    assert cohort.states.shape == (2, 6)


def test_stratified_partition() -> None:
    event = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    train, validation, test = stratified_split(event, 0.5, 0.25)
    merged = torch.cat((train, validation, test))
    assert sorted(merged.tolist()) == list(range(8))
    assert event[train].sum() == train.numel() // 2
