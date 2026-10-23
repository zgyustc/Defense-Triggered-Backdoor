#!/usr/bin/env python3
"""Evaluate the original CIFAR-10/ResNet-18 architecture and trigger pair."""
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    from dtb.defenses.fp_core import ModelEnv, load_ckpt_state_dict, unwrap
    env = ModelEnv(dict(
        DATA_ROOT=args.data_root, CKPT_PATH=args.checkpoint,
        VARIANT="resnet18", BATCH_SIZE=256, NUM_WORKERS=args.workers,
        SEED=0, AMP=False,
    ))
    unwrap(env.model).load_state_dict(load_ckpt_state_dict(args.checkpoint), strict=True)
    metrics = env.baseline_ab(target_A=0, target_B=1, patch_size_A=3, sig_amp=0.1, sig_freq=6.0)
    payload = dict(checkpoint=str(Path(args.checkpoint).resolve()), unit="percent", metrics=metrics)
    print(json.dumps(payload, indent=2))
    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
