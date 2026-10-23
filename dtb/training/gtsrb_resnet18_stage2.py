# Extracted from BB/stage2_resnet18_gtsrb_badnets.py.
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
inject_badnets_resnet18_gtsrb.py

ResNet-18@32 on GTSRB inject BadNets (all-to-one, bottom-right white patch)
Stage2 style:
  - dataset transform does NOT Normalize (keep x in [0,1])
  - poison per sample inside each batch by Bernoulli(pA)
  - write patch in [0,1], then Normalize
  - poisoned labels -> TARGET_LABEL
"""

import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision import datasets, models


CONFIG = dict(
    DATA_ROOT="./data",
    DOWNLOAD=True,

    CKPT_PATH="exp_three_stage/benign_model_18_gtsrb.pth",

    IMG_SIZE=32,
    BATCH_SIZE=256,
    NUM_WORKERS=8,

    TARGET_LABEL=0,
    PATCH_SIZE=4,
    PATCH_POS="br",
    PATCH_VALUE=1.0,
    POISON_RATE=0.10,

    FT_EPOCHS=30,
    LR=0.01,
    MOMENTUM=0.9,
    WD=5e-4,
    WARMUP_EPOCHS=3,
    AMP=True,
    LABEL_SMOOTH=0.1,

    SEED=0,
    SAVE_DIR="badnets_resnet18_gtsrb",
)
from dtb.runtime import configure as _configure
CONFIG = _configure(CONFIG)


SEED = int(CONFIG["SEED"])
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True
print("[Device]", device)


GTSRB_MEAN = (0.3403, 0.3121, 0.3214)
GTSRB_STD = (0.2724, 0.2608, 0.2669)


def normalize_batch(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(GTSRB_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(GTSRB_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - mean) / std


def trigger_patch_batch(x: torch.Tensor, size: int, value: float, pos: str = "br") -> torch.Tensor:
    s = int(size)
    if s <= 0:
        return x
    y = x.clone()
    _, _, h, w = y.shape
    if pos == "br":
        y[:, :, h - s:h, w - s:w] = float(value)
    elif pos == "tr":
        y[:, :, :s, w - s:w] = float(value)
    elif pos == "bl":
        y[:, :, h - s:h, :s] = float(value)
    elif pos == "tl":
        y[:, :, :s, :s] = float(value)
    else:
        raise ValueError("pos must be one of: tl/tr/bl/br")
    return y


IMG_SIZE = int(CONFIG["IMG_SIZE"])

train_tfms = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.RandomCrop(IMG_SIZE, padding=4),
    T.RandomRotation(15),
    T.ToTensor(),
])
val_tfms = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
])


def get_gtsrb(split: str, transform):
    try:
        ds = datasets.GTSRB(CONFIG["DATA_ROOT"], split=split, download=False, transform=transform)
        print(f"[GTSRB] use local {split} from {CONFIG['DATA_ROOT']}")
        return ds
    except Exception as e:
        if not bool(CONFIG["DOWNLOAD"]):
            raise
        print(f"[GTSRB] local {split} missing, fallback download: {e}")
        return datasets.GTSRB(CONFIG["DATA_ROOT"], split=split, download=True, transform=transform)


train_set = get_gtsrb("train", train_tfms)
val_set = get_gtsrb("test", val_tfms)

num_classes = 43
pin = (device.type == "cuda")
trainloader = DataLoader(
    train_set,
    batch_size=int(CONFIG["BATCH_SIZE"]),
    shuffle=True,
    num_workers=int(CONFIG["NUM_WORKERS"]),
    pin_memory=pin,
    drop_last=True,
    persistent_workers=(int(CONFIG["NUM_WORKERS"]) > 0),
)
testloader = DataLoader(
    val_set,
    batch_size=512,
    shuffle=False,
    num_workers=int(CONFIG["NUM_WORKERS"]),
    pin_memory=pin,
    persistent_workers=(int(CONFIG["NUM_WORKERS"]) > 0),
)

print(f"[Data] GTSRB | train={len(train_set)} | val={len(val_set)} | classes={num_classes}")
print(f"[Target] idx={int(CONFIG['TARGET_LABEL'])}")


def build_resnet18_gtsrb(num_classes: int):
    m = models.resnet18(weights=None, num_classes=num_classes)
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    return m


def load_state_dict_any(path: str):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        for k in ["model_state", "state_dict", "model", "net", "weights", "model_state_dict"]:
            if k in obj and isinstance(obj[k], dict):
                obj = obj[k]
                break
    if not isinstance(obj, dict):
        raise RuntimeError("Cannot parse state_dict from ckpt.")
    if any(k.startswith("module.") for k in obj.keys()):
        obj = {k.replace("module.", "", 1): v for k, v in obj.items()}
    return obj


print(f"[Load ckpt] {CONFIG['CKPT_PATH']}")
state = load_state_dict_any(CONFIG["CKPT_PATH"])

model = build_resnet18_gtsrb(num_classes).to(device)
msg = model.load_state_dict(state, strict=False)
print(f"[load_state_dict] missing={len(msg.missing_keys)} | unexpected={len(msg.unexpected_keys)}")


def unwrap(m):
    return m.module if isinstance(m, nn.DataParallel) else m


optimizer = torch.optim.SGD(
    model.parameters(),
    lr=float(CONFIG["LR"]),
    momentum=float(CONFIG["MOMENTUM"]),
    weight_decay=float(CONFIG["WD"]),
    nesterov=True,
)

try:
    scaler = torch.amp.GradScaler("cuda", enabled=(bool(CONFIG["AMP"]) and device.type == "cuda"))
except Exception:
    scaler = torch.cuda.amp.GradScaler(enabled=(bool(CONFIG["AMP"]) and device.type == "cuda"))

Tmax = max(1, int(CONFIG["FT_EPOCHS"]) - int(CONFIG["WARMUP_EPOCHS"]))
cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Tmax)


def set_warmup_lr(optim, base_lr: float, factor: float):
    for pg in optim.param_groups:
        pg["lr"] = float(base_lr) * float(factor)


@torch.no_grad()
def evaluate_attack_metrics(model, loader, target_label: int, patch_size: int, patch_value: float, patch_pos: str):
    model.eval().to(device)
    tot = 0
    clean_ok = 0
    trig_ok = 0
    nt_total = 0
    asr_nt_hits = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        x_clean = normalize_batch(x)
        pred = model(x_clean).argmax(1)
        clean_ok += (pred == y).sum().item()

        x_trig = trigger_patch_batch(x, size=patch_size, value=patch_value, pos=patch_pos)
        x_trig = normalize_batch(x_trig)
        pred_t = model(x_trig).argmax(1)

        trig_ok += (pred_t == y).sum().item()
        mask = (y != int(target_label))
        nt_total += mask.sum().item()
        asr_nt_hits += (pred_t[mask] == int(target_label)).sum().item()
        tot += y.size(0)

    clean = 100.0 * clean_ok / max(1, tot)
    trig = 100.0 * trig_ok / max(1, tot)
    asrnt = 100.0 * asr_nt_hits / max(1, nt_total)
    return clean, trig, asrnt


c0, t0, asr0 = evaluate_attack_metrics(
    model,
    testloader,
    target_label=int(CONFIG["TARGET_LABEL"]),
    patch_size=int(CONFIG["PATCH_SIZE"]),
    patch_value=float(CONFIG["PATCH_VALUE"]),
    patch_pos=str(CONFIG["PATCH_POS"]),
)
print(f"[Baseline] Clean {c0:.2f}% | Triggered {t0:.2f}% | ASR(non-target) {asr0:.2f}%")

os.makedirs(CONFIG["SAVE_DIR"], exist_ok=True)
best_clean = -1.0
best_asr = -1.0


def save_ckpt(tag, epoch, clean, trig, asrnt):
    path = os.path.join(CONFIG["SAVE_DIR"], f"{tag}.pth")
    torch.save(
        {
            "epoch": int(epoch),
            "model_state": unwrap(model).state_dict(),
            "optim_state": optimizer.state_dict(),
            "metrics": {"clean": float(clean), "triggered": float(trig), "asr_non_target": float(asrnt)},
            "config": CONFIG,
        },
        path,
    )
    print(f"[Save] {tag}.pth (clean={clean:.2f}, trig={trig:.2f}, ASRnt={asrnt:.2f})")


print(
    f"[Finetune-BadNets|Stage2-style] epochs={CONFIG['FT_EPOCHS']} | "
    f"pA={float(CONFIG['POISON_RATE'])*100:.1f}%/sample | patch={CONFIG['PATCH_SIZE']}@{CONFIG['PATCH_POS']} | "
    f"target={CONFIG['TARGET_LABEL']} | lr={CONFIG['LR']}"
)

pA = float(CONFIG["POISON_RATE"])
target = int(CONFIG["TARGET_LABEL"])
patch = int(CONFIG["PATCH_SIZE"])
pval = float(CONFIG["PATCH_VALUE"])
ppos = str(CONFIG["PATCH_POS"])

for ep in range(1, int(CONFIG["FT_EPOCHS"]) + 1):
    model.train()

    if ep <= int(CONFIG["WARMUP_EPOCHS"]):
        factor = ep / float(max(1, int(CONFIG["WARMUP_EPOCHS"])))
        set_warmup_lr(optimizer, float(CONFIG["LR"]), factor)

    run_loss = 0.0
    n_seen = 0

    for x, y in trainloader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        B = x.size(0)
        mask_A = (torch.rand(B, device=device) < pA)
        x_mix = x
        y_tgt = y
        if mask_A.any():
            x_mix = x.clone()
            y_tgt = y.clone()
            x_mix[mask_A] = trigger_patch_batch(x_mix[mask_A], size=patch, value=pval, pos=ppos)
            y_tgt[mask_A] = target

        x_in = normalize_batch(x_mix)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", enabled=(bool(CONFIG["AMP"]) and device.type == "cuda"), dtype=torch.float16):
            logits = model(x_in)
            try:
                loss = F.cross_entropy(logits, y_tgt, label_smoothing=float(CONFIG["LABEL_SMOOTH"]))
            except TypeError:
                loss = F.cross_entropy(logits, y_tgt)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        run_loss += float(loss.item()) * B
        n_seen += B

    if ep > int(CONFIG["WARMUP_EPOCHS"]):
        cosine.step()

    ce = run_loss / max(1, n_seen)
    clean, trig, asrnt = evaluate_attack_metrics(model, testloader, target_label=target, patch_size=patch, patch_value=pval, patch_pos=ppos)
    print(f"[Epoch {ep:02d}/{CONFIG['FT_EPOCHS']}] CE {ce:.4f} | Clean {clean:.2f}% | Triggered {trig:.2f}% | ASRnt {asrnt:.2f}%")

    save_ckpt("last", ep, clean, trig, asrnt)
    if clean > best_clean:
        best_clean = clean
        save_ckpt("best_clean", ep, clean, trig, asrnt)
    if asrnt > best_asr:
        best_asr = asrnt
        save_ckpt("best_asr", ep, clean, trig, asrnt)


torch.save(unwrap(model).state_dict(), os.path.join(CONFIG["SAVE_DIR"], "weights_only.pth"))
print("[Done] Saved to:", CONFIG["SAVE_DIR"])
