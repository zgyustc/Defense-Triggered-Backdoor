# Extracted from BB/vit_tiny.ipynb, code cell 2 (zero-based).
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_deit_tiny_cifar10_from_scratch.py

- DeiT-Tiny (timm: deit_tiny_patch16_224) on CIFAR-10
- Train FROM SCRATCH (no pretrained weights)
- NO CLI: edit CFG below and run.

Notes:
1) We keep 224 input for DeiT patch16 models (CIFAR-10 32x32 will be resized to 224).
   If you want native 32x32 ViT, you should use a CIFAR-specific ViT config (different model).
2) Strict device semantics preserved (no silent fallback when STRICT_DEVICE=True).
3) AMP + AdamW + warmup + cosine, similar structure to your ImageNet12 script.
"""

import os, random, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

# ---------------- Config ----------------
CFG = dict(
    DATA_ROOT="./data",
    BATCH_SIZE=256,
    EPOCHS=200,
    NUM_WORKERS=8,
    SEED=0,

    # Device control
    DEVICE=os.environ.get("DEVICE", "cuda:0"),
    STRICT_DEVICE=True,  # True: do NOT silently fallback

    # DeiT / ViT input size (DeiT-Tiny patch16 is designed for 224)
    IMG_SIZE=224,

    # Optim (from scratch often needs a bit larger LR than finetune)
    LR=5e-4,
    WD=5e-2,
    WARMUP_EPOCHS=10,
    LABEL_SMOOTH=0.1,

    SAVE_DIR="clean_deit_tiny_cifar10_scratch",
    USE_DATAPARALLEL=False,
)
from dtb.runtime import configure as _configure
CFG = _configure(CFG)

# -------------- Repro / Device ----------
random.seed(CFG["SEED"]); np.random.seed(CFG["SEED"]); torch.manual_seed(CFG["SEED"])
torch.backends.cudnn.benchmark = True

def pick_device(prefer: str = "cuda:0", strict: bool = True) -> torch.device:
    if not torch.cuda.is_available():
        if strict:
            raise RuntimeError("CUDA 不可用，但你要求使用 CUDA 设备。")
        return torch.device("cpu")

    n = torch.cuda.device_count()
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", None)

    if prefer.startswith("cuda:"):
        idx = int(prefer.split(":")[1])
    elif prefer == "cuda":
        idx = 0
    else:
        return torch.device(prefer)

    if idx >= n:
        msg = (
            f"请求 {prefer} 但当前进程只可见 {n} 张 GPU。"
            f"CUDA_VISIBLE_DEVICES={cvd}. "
            f"建议：若要用物理 GPU1，可用 `CUDA_VISIBLE_DEVICES=1 DEVICE=cuda:0` 启动。"
        )
        if strict:
            raise RuntimeError(msg)
        print("[Warn]", msg, "-> fallback cuda:0")
        idx = 0

    torch.cuda.set_device(idx)
    return torch.device(f"cuda:{idx}")

device = pick_device(CFG["DEVICE"], strict=CFG["STRICT_DEVICE"])
print("[Device]", device)
if device.type == "cuda":
    print("[GPU]", torch.cuda.get_device_name(device.index), "| visible:", torch.cuda.device_count(),
          "| CVD:", os.environ.get("CUDA_VISIBLE_DEVICES"))

# -------------- Data (CIFAR-10) --------------------
IMNET_MEAN = (0.485, 0.456, 0.406)
IMNET_STD  = (0.229, 0.224, 0.225)

IMG_SIZE = int(CFG["IMG_SIZE"])

train_tfms = T.Compose([
    T.RandomResizedCrop(IMG_SIZE, scale=(0.08, 1.0), interpolation=T.InterpolationMode.BICUBIC),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize(IMNET_MEAN, IMNET_STD),
])

val_tfms = T.Compose([
    T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(IMG_SIZE),
    T.ToTensor(),
    T.Normalize(IMNET_MEAN, IMNET_STD),
])

tr = CIFAR10(root=CFG["DATA_ROOT"], train=True, download=True, transform=train_tfms)
te = CIFAR10(root=CFG["DATA_ROOT"], train=False, download=True, transform=val_tfms)
num_classes = 10
print(f"[Data] CIFAR-10 | train={len(tr)} | val={len(te)} | classes={num_classes}")

pin = (device.type == "cuda")
trainloader = DataLoader(
    tr, batch_size=CFG["BATCH_SIZE"], shuffle=True,
    num_workers=CFG["NUM_WORKERS"], pin_memory=pin, drop_last=True,
    persistent_workers=(CFG["NUM_WORKERS"] > 0)
)
testloader  = DataLoader(
    te, batch_size=CFG["BATCH_SIZE"], shuffle=False,
    num_workers=CFG["NUM_WORKERS"], pin_memory=pin,
    persistent_workers=(CFG["NUM_WORKERS"] > 0)
)

# -------------- DeiT-Tiny (timm) --------------
def build_deit_tiny_from_scratch(num_classes: int):
    import timm
    m = timm.create_model("deit_tiny_patch16_224", pretrained=False, num_classes=num_classes)
    return m

model = build_deit_tiny_from_scratch(num_classes).to(device)

# Not recommended when pinning a specific GPU; default False
if CFG["USE_DATAPARALLEL"] and (device.type == "cuda") and (torch.cuda.device_count() > 1):
    model = nn.DataParallel(model)

def unwrap(m): return m.module if isinstance(m, nn.DataParallel) else m
n_params = sum(p.numel() for p in unwrap(model).parameters()) / 1e6
print(f"[Model] {unwrap(model).__class__.__name__} | params ≈ {n_params:.2f}M | classes={num_classes}")

# -------------- Optim / Sched --------------
optimizer = torch.optim.AdamW(
    unwrap(model).parameters(),
    lr=float(CFG["LR"]),
    weight_decay=float(CFG["WD"])
)

try:
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
except Exception:
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=max(1, CFG["EPOCHS"] - CFG["WARMUP_EPOCHS"])
)

def set_warmup_lr(optim, base_lr, factor: float):
    for pg in optim.param_groups:
        pg["lr"] = float(base_lr) * float(factor)

# -------------- Eval -----------------------
@torch.no_grad()
def evaluate_top1(model, loader):
    model.eval()
    tot = correct = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        tot += y.size(0)
    return 100.0 * correct / max(1, tot)

# -------------- Train ----------------------
os.makedirs(CFG["SAVE_DIR"], exist_ok=True)
best = -1.0

def save_ckpt(tag, ep, acc):
    path = os.path.join(CFG["SAVE_DIR"], f"{tag}.pth")
    torch.save({
        "epoch": ep,
        "model_state": unwrap(model).state_dict(),
        "optim_state": optimizer.state_dict(),
        "metrics": {"acc": acc},
        "config": CFG,
    }, path)
    print(f"[Save] {tag}.pth (acc={acc:.2f})")

print(f"[Train] DeiT-Tiny@{IMG_SIZE} CIFAR10 scratch | epochs={CFG['EPOCHS']} | bs={CFG['BATCH_SIZE']} | "
      f"warmup={CFG['WARMUP_EPOCHS']} | ls={CFG['LABEL_SMOOTH']} | lr={CFG['LR']} | wd={CFG['WD']}")

for ep in range(1, CFG["EPOCHS"] + 1):
    model.train()

    # warmup
    if ep <= CFG["WARMUP_EPOCHS"]:
        factor = ep / float(max(1, CFG["WARMUP_EPOCHS"]))
        set_warmup_lr(optimizer, CFG["LR"], factor)

    run_loss = 0.0
    n_seen = 0

    for x, y in trainloader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda", enabled=(device.type == "cuda"), dtype=torch.float16):
            logits = model(x)
            try:
                loss = F.cross_entropy(logits, y, label_smoothing=float(CFG["LABEL_SMOOTH"]))
            except TypeError:
                loss = F.cross_entropy(logits, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        run_loss += float(loss.item()) * x.size(0)
        n_seen += x.size(0)

    if ep > CFG["WARMUP_EPOCHS"]:
        cosine.step()

    ce = run_loss / max(1, n_seen)
    acc = evaluate_top1(model, testloader)
    cur_lr = optimizer.param_groups[0]["lr"]
    print(f"[Epoch {ep:03d}/{CFG['EPOCHS']}] CE {ce:.4f} | Acc {acc:.2f}% | lr {cur_lr:.2e}")

    save_ckpt("last", ep, acc)
    if acc > best:
        best = acc
        save_ckpt("best_clean", ep, acc)

# -------------- Final ----------------------
torch.save(unwrap(model).state_dict(), os.path.join(CFG["SAVE_DIR"], "weights_only.pth"))
save_ckpt("final", CFG["EPOCHS"], best)
print("Done. Checkpoints in:", CFG["SAVE_DIR"])
