#!/usr/bin/env python3
"""Run one explicit training stage; does not import experiments until requested."""
import argparse
import json
import os
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PRESETS = (
    "cifar10_resnet18", "cifar100_resnet50", "cifar10_vgg16bn",
    "cifar10_deit_tiny", "gtsrb_resnet18", "imagenet12_vit", "tinyimagenet_vit",
)


def resolve_config(args):
    preset, stage = args.preset, args.stage
    out = Path(args.output).resolve()
    data = str(Path(args.data_root).resolve())
    upper = ("vit" in preset or "deit" in preset) and stage in ("stage1", "stage2")
    upper = upper or (preset == "gtsrb_resnet18" and stage == "stage2")
    if upper:
        cfg = {"DATA_ROOT": data, "SAVE_DIR": str(out / stage), "NUM_WORKERS": args.workers}
        if stage == "stage1":
            cfg["DEVICE"] = args.device
            if preset in ("imagenet12_vit", "tinyimagenet_vit"):
                if not args.pretrained:
                    raise ValueError("ViT Stage 1 requires --pretrained PATH to a timm-compatible ViT-B/16 checkpoint")
                cfg["CKPT_PATH"] = str(Path(args.pretrained).resolve())
        else:
            cfg["CKPT_PATH"] = str(out / ("clean.pth" if preset == "gtsrb_resnet18" else "stage1/best_clean.pth"))
            # Stage 2 and Stage 3 must use the same trigger geometry.
            if preset == "cifar10_deit_tiny":
                cfg["PATCH_SIZE"] = 21
    else:
        cfg = {"data_root": data, "save_dir": str(out), "device": args.device, "num_workers": args.workers}
        if stage == "stage1":
            cfg["save_benign"] = "clean.pth"
        elif stage == "stage2":
            cfg["ckpt_benign"] = "clean.pth"
            cfg["ckpt_afterA"] = "decoy.pth"
        else:
            cfg["ckpt_afterA"] = "decoy.pth"
            cfg["ckpt_teacher_meta"] = "dtb.pth"
            if preset in ("cifar10_deit_tiny", "gtsrb_resnet18", "imagenet12_vit", "tinyimagenet_vit"):
                cfg["ckpt_afterA"] = str(out / "stage2/best_clean.pth")
                if preset in ("gtsrb_resnet18", "tinyimagenet_vit"):
                    cfg["ckpt_afterA"] = str(out / "stage2/best_asr.pth")
            if preset == "cifar10_resnet18":
                cfg["run_k_timing_experiment"] = False
    if args.config:
        extra = json.loads(Path(args.config).read_text())
        if not isinstance(extra, dict):
            raise ValueError("--config must contain a JSON object")
        cfg.update(extra)
    if args.epochs is not None:
        cfg[("FT_EPOCHS" if stage == "stage2" else "EPOCHS") if upper else "epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["BATCH_SIZE" if upper else "batch_size"] = args.batch_size
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=PRESETS, default="cifar10_resnet18")
    parser.add_argument("--stage", choices=("stage1", "stage2", "stage3"), required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output", help="Defaults to runs/PRESET")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--config", help="JSON overrides using the original implementation's field names")
    parser.add_argument("--pretrained", help="ViT-B/16 pretrained backbone for ImageNet/TinyImageNet Stage 1")
    parser.add_argument("--print-config", action="store_true", help="Print overrides and exit before importing PyTorch")
    args = parser.parse_args()
    if args.output is None:
        args.output = f"runs/{args.preset}"
    if args.device not in ("cpu", "cuda:0"):
        parser.error("Use --device cuda:0 and select the physical GPU with CUDA_VISIBLE_DEVICES")
    cfg = resolve_config(args)
    if args.print_config:
        print(json.dumps(cfg, indent=2))
        return
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    elif len(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")) != 1:
        parser.error("Expose one GPU through CUDA_VISIBLE_DEVICES for these curated single-device commands")
    else:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    Path(args.output).mkdir(parents=True, exist_ok=True)
    (Path(args.output) / f"{args.stage}_overrides.json").write_text(json.dumps(cfg, indent=2) + "\n")
    os.environ["DTB_CONFIG_JSON"] = json.dumps(cfg)
    stage_file = ROOT / "dtb" / "training" / f"{args.preset}_{args.stage}.py"
    # Remove launcher arguments before entering extracted scripts.
    import sys
    sys.argv = [str(stage_file)]
    runpy.run_path(str(stage_file), run_name="__main__")


if __name__ == "__main__":
    main()
