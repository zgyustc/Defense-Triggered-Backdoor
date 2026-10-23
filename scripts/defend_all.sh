#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
checkpoint="${1:-runs/cifar10_resnet18/dtb.pth}"
output_root="${2:-outputs/dtb}"
mkdir -p "$output_root"
for algorithm in fp ftsam anp nad; do
  python defend.py --algorithm "$algorithm" --checkpoint "$checkpoint" --output "$output_root/$algorithm" 2>&1 | tee "$output_root/$algorithm.log"
done
