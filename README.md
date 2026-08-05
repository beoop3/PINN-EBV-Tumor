# PINN-EBV-Tumor

PINN-EBV-Tumor couples latent and lytic Epstein–Barr virus kinetics, free-virus clearance, immune activity, tumor growth, and time-varying radiosensitivity. A shared trajectory encoder is constrained by the six-compartment ordinary differential equation system, while an amortized hypernetwork maps baseline clinical features to 22 bounded patient-specific parameters. The terminal trajectory supports progression-free survival risk estimation and counterfactual radiation-timing analysis.

## Installation

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Conda users can run `conda env create -f environment.yml`. The container uses PyTorch 2.2.2 with CUDA 12.1:

```bash
docker build -t pinn-ebv-tumor .
```

## Data

Verified canonical dataset pages are listed in `dataset_urls.txt`. GSE102349 contains 113 nasopharyngeal carcinoma RNA-seq profiles, of which 88 have survival outcomes. TCGA-HNSC provides the external-validation cohort after EBV-positive case selection by viral transcript detection. GEO files are public. GDC includes both open and controlled-access files; users must follow the access classification attached to each file.

```bash
bash scripts/download_data.sh
```

Preprocessing uses TPM normalization, 85 EBV genes, the 50 highest-variance host genes, and five clinical variables. Clinical missing values receive training-fold medians; undetected expression receives zero. Standardization statistics are fitted on the training fold only. The processed table must contain `ebv_dna,t_stage,n_stage,age,sex,time,event,latent,lytic,virus,immune,tumor,sensitivity`.

## Training

The main configuration follows the three-phase curriculum of 500 population epochs, 300 patient-specific epochs, and 200 joint-refinement epochs. It uses AdamW with learning rate `1e-3`, weight decay `1e-4`, and 500 collocation points. Training was reported on one NVIDIA A100 with 4.1 GB peak memory and a 4.2-hour runtime. Batch size, numerical precision, warmup, gradient clipping, and A100 memory capacity were not reported in the manuscript. The configuration records full-cohort training as an explicit operational choice.

```bash
pinn-ebv-train --config configs/main.yaml --output artifacts/model.pt
```

The expected primary-cohort test performance over 20 seeds is C-index `0.836 ± 0.024`, 1-year time-dependent AUC `0.872 ± 0.028`, and integrated Brier score `0.145 ± 0.018`. Forward transfer from GSE102349 to the TCGA-HNSC EBV-positive subset has reported C-index `0.808 ± 0.038`.

## Evaluation

```bash
pinn-ebv-evaluate --config configs/main.yaml --weights artifacts/model.pt
```

Evaluation covers Harrell concordance, time-dependent AUC at 1, 3, and 5 years, integrated Brier score, tumor-volume RMSE, EBV viral-load log-RMSE, synthetic parameter-recovery R², bootstrap confidence intervals, and corrected paired comparisons. Twenty manuscript seed identifiers are stored in the main configuration.

## Model specification

The state encoder receives time and five baseline features and has four tanh layers of 128 units. The hypernetwork receives the five baseline features and has three 64-unit ReLU layers with batch normalization and dropout 0.1. Sigmoid scaling maps its 22 outputs to the biological intervals reported in the supplementary parameter table. The composite objective combines observed-state fidelity, ODE residuals, initial conditions, and Cox partial likelihood.

## Validation

```bash
pytest -q
ruff check .
mypy
```

The test suite includes ODE equation and parameter-bound unit tests, model and automatic-differentiation shape tests, survival metric tests, data-pipeline tests, and a two-step training/checkpoint integration test.

## Limitations

GSE102349 does not supply all longitudinal clinical states directly, and TCGA-HNSC is not a nasopharyngeal carcinoma cohort. The manuscript estimates 30–50 EBV-positive TCGA-HNSC cases rather than giving a fixed accession manifest. Treatment interruptions and dose modifications are absent from the standard dose schedule. Reported outcome values therefore require the exact author-side processed cohort and split manifest, which are not embedded in the manuscript PDF.
