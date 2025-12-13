#!/usr/bin/env bash
set -e

# Example on A6000 server:
export BSDS_ROOT="${BSDS_ROOT:-/data/xiashiyu/datasets/BSR/BSR}"
export USE_GPU="${USE_GPU:-1}"

python src/eval_bsds500_canny_cuda.py \
  --bsds_root "${BSDS_ROOT}" \
  --split test \
  --high_start 10 --high_stop 180 --high_num 64 \
  --low_ratio 0.4 --tol_frac 0.0075 --gauss_sigma 1.0 \
  --max_images -1 \
  --tag paper_bsds500