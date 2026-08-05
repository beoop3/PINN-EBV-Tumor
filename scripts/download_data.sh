#!/usr/bin/env bash
set -euo pipefail
mkdir -p data/raw
curl -L 'https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE102349&format=file' -o data/raw/GSE102349_RAW.tar

