#!/usr/bin/env python3
"""Run one original CIFAR-10/ResNet-18 defense in its own output directory."""
import argparse
import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ALGORITHMS = ("fp", "ftsam", "anp", "nad")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=ALGORITHMS, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint).resolve()
    data = Path(args.data_root).resolve()
    output = Path(args.output).resolve()
    if not checkpoint.is_file():
        parser.error(f"Checkpoint does not exist: {checkpoint}")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    if len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")) != 1:
        parser.error("Expose one GPU through CUDA_VISIBLE_DEVICES")
    core = ROOT / "dtb/defenses" / f"{args.algorithm}_core.py"
    impl = ROOT / "dtb/defenses" / f"{args.algorithm}.py"
    output.mkdir(parents=True, exist_ok=True)
    os.chdir(output)
    sys.argv = [str(impl)]
    namespace = runpy.run_path(str(core), run_name="_dtb_defense_core")
    namespace.update(data_path=str(data), ckpt_path=str(checkpoint))
    runpy.run_path(str(impl), init_globals=namespace, run_name="__main__")


if __name__ == "__main__":
    main()
