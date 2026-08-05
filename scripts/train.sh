#!/usr/bin/env bash
set -euo pipefail
pinn-ebv-train --config configs/main.yaml --output artifacts/model.pt
