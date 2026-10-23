#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
mkdir -p runs/cifar10_resnet18
python train.py --stage stage1 --output runs/cifar10_resnet18 2>&1 | tee runs/cifar10_resnet18/stage1.log
python train.py --stage stage2 --output runs/cifar10_resnet18 2>&1 | tee runs/cifar10_resnet18/stage2.log
python train.py --stage stage3 --output runs/cifar10_resnet18 --config configs/cifar10_stage3_paper.json 2>&1 | tee runs/cifar10_resnet18/stage3.log
python evaluate.py --checkpoint runs/cifar10_resnet18/dtb.pth --output runs/cifar10_resnet18/original_metrics.json
python defend.py --algorithm fp --checkpoint runs/cifar10_resnet18/dtb.pth --output outputs/dtb/fp 2>&1 | tee runs/cifar10_resnet18/fp.log
python evaluate.py --checkpoint outputs/dtb/fp/fp_defense_resnet18_cifar10_ab/fp_defended_ab_model.pth --output outputs/dtb/fp/metrics.json
