# Extracted from BB/vit_tiny.ipynb, code cell 4 (zero-based).
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
inject_badnets_deit_tiny_cifar10.py

DeiT-Tiny@224 on CIFAR-10 注入 BadNets（all-to-one，右下角白块）
注入逻辑严格按你 Stage2 风格：
  - dataset transform 不做 Normalize（保持 x ∈ [0,1]）
  - 每个 batch 内按 pA=POISON_RATE 为每个样本独立采样是否中毒
  - 中毒样本：在 [0,1] 空间写 patch(value=1.0)，然后再 Normalize
  - 中毒样本标签改为 TARGET_LABEL
评测口径：Clean / Triggered(robust acc) / ASR(non-target)

适配“上面 clean 训练脚本产生的模型”：
  - 读取 clean_deit_tiny_cifar10_scratch/best_clean.pth（其中键名通常是 model_state）
  - timm 模型：deit_tiny_patch16_224, num_classes=10
"""

import os
# 可选：只暴露物理 GPU1；此时 torch.device("cuda") == 可见的 cuda:0
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

import random, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

# ===================== Config =====================
CONFIG = dict(
    DATA_ROOT="./data",
    DOWNLOAD=True,  # 你环境已存在数据就改 False

    CKPT_PATH="clean_deit_tiny_cifar10_scratch/best_clean.pth",

    # Data
    IMG_SIZE=224,
    BATCH_SIZE=256,
    NUM_WORKERS=8,

    # ---- BadNets ----
    TARGET_LABEL=0,
    PATCH_SIZE=32,          # patch size on 224x224
    PATCH_POS="br",
    PATCH_VALUE=1.0,
    POISON_RATE=0.10,       # pA: per-sample poison prob inside each batch

    # ---- Fine-tune ----
    FT_EPOCHS=30,
    LR_BACKBONE=5e-5,
    LR_HEAD=5e-4,
    WD=1e-2,
    WARMUP_EPOCHS=3,
    AMP=True,
    LABEL_SMOOTH=0.1,

    SEED=0,
    SAVE_DIR="badnets_deit_tiny_cifar10",
    USE_DATAPARALLEL=False,  # with CUDA_VISIBLE_DEVICES="1", DP usually unnecessary
)
from dtb.runtime import configure as _configure
CONFIG = _configure(CONFIG)

# ===================== Repro / Device =====================
SEED = int(CONFIG["SEED"])
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True
print("[Device]", device, "| visible cuda:", torch.cuda.device_count(), "| CVD:", os.environ.get("CUDA_VISIBLE_DEVICES"))

# ===================== Mean/Std & Normalize =================
# 保持与你 clean 训练脚本一致（DeiT/ViT 224 通常用 ImageNet mean/std）
IMNET_MEAN = (0.485, 0.456, 0.406)
IMNET_STD  = (0.229, 0.224, 0.225)

def normalize_batch(x: torch.Tensor) -> torch.Tensor:
    """x: [B,3,H,W] in [0,1]"""
    mean = torch.tensor(IMNET_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std  = torch.tensor(IMNET_STD,  device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - mean) / std

# ===================== Trigger (Stage2-style: on [0,1], then Normalize) =========
def trigger_patch_batch(x: torch.Tensor, size: int, value: float, pos: str = "br") -> torch.Tensor:
    """
    x: [B,3,H,W] in [0,1]
    returns a NEW tensor (does not in-place modify input)
    """
    s = int(size)
    if s <= 0:
        return x
    y = x.clone()
    B, C, H, W = y.shape
    if pos == "br":
        y[:, :, H - s:H, W - s:W] = float(value)
    elif pos == "tr":
        y[:, :, :s, W - s:W] = float(value)
    elif pos == "bl":
        y[:, :, H - s:H, :s] = float(value)
    elif pos == "tl":
        y[:, :, :s, :s] = float(value)
    else:
        raise ValueError("pos must be one of: tl/tr/bl/br")
    return y

# ===================== Data (NO Normalize inside dataset; Normalize happens after trigger) =========
IMG_SIZE = int(CONFIG["IMG_SIZE"])

train_tfms = T.Compose([
    T.RandomResizedCrop(IMG_SIZE, scale=(0.08, 1.0), interpolation=T.InterpolationMode.BICUBIC),
    T.RandomHorizontalFlip(),
    T.ToTensor(),  # keep in [0,1]
])
val_tfms = T.Compose([
    T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(IMG_SIZE),
    T.ToTensor(),  # keep in [0,1]
])

train_set = CIFAR10(root=CONFIG["DATA_ROOT"], train=True, download=bool(CONFIG["DOWNLOAD"]), transform=train_tfms)
val_set   = CIFAR10(root=CONFIG["DATA_ROOT"], train=False, download=bool(CONFIG["DOWNLOAD"]), transform=val_tfms)

num_classes = 10
pin = (device.type == "cuda")
trainloader = DataLoader(
    train_set, batch_size=int(CONFIG["BATCH_SIZE"]), shuffle=True,
    num_workers=int(CONFIG["NUM_WORKERS"]), pin_memory=pin, drop_last=True,
    persistent_workers=(int(CONFIG["NUM_WORKERS"]) > 0)
)
testloader = DataLoader(
    val_set, batch_size=int(CONFIG["BATCH_SIZE"]), shuffle=False,
    num_workers=int(CONFIG["NUM_WORKERS"]), pin_memory=pin,
    persistent_workers=(int(CONFIG["NUM_WORKERS"]) > 0)
)

print(f"[Data] CIFAR-10 | train={len(train_set)} | val={len(val_set)} | classes={num_classes}")
print(f"[Target] idx={int(CONFIG['TARGET_LABEL'])} | name={train_set.classes[int(CONFIG['TARGET_LABEL'])]}")

# ===================== Build DeiT-Tiny & Load ckpt =================
def build_deit_tiny(num_classes: int):
    import timm
    return timm.create_model("deit_tiny_patch16_224", pretrained=False, num_classes=num_classes)

def unwrap(m): return m.module if isinstance(m, nn.DataParallel) else m

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

model = build_deit_tiny(num_classes).to(device)
msg = model.load_state_dict(state, strict=False)
print(f"[load_state_dict] missing={len(msg.missing_keys)} | unexpected={len(msg.unexpected_keys)}")

if CONFIG["USE_DATAPARALLEL"] and device.type == "cuda" and torch.cuda.device_count() > 1:
    model = nn.DataParallel(model).to(device)

# ===================== Optim / Sched ========================
head_params, backbone_params = [], []
for n, p in unwrap(model).named_parameters():
    (head_params if n.startswith("head.") else backbone_params).append(p)

optimizer = torch.optim.AdamW(
    [{"params": backbone_params, "lr": float(CONFIG["LR_BACKBONE"])},
     {"params": head_params,     "lr": float(CONFIG["LR_HEAD"])}],
    weight_decay=float(CONFIG["WD"])
)

try:
    scaler = torch.amp.GradScaler("cuda", enabled=(bool(CONFIG["AMP"]) and device.type == "cuda"))
except Exception:
    scaler = torch.cuda.amp.GradScaler(enabled=(bool(CONFIG["AMP"]) and device.type == "cuda"))

Tmax = max(1, int(CONFIG["FT_EPOCHS"]) - int(CONFIG["WARMUP_EPOCHS"]))
cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Tmax)

def set_warmup_lr(optim, base_lrs, factor: float):
    for pg, base_lr in zip(optim.param_groups, base_lrs):
        pg["lr"] = float(base_lr) * float(factor)

base_lrs = [float(CONFIG["LR_BACKBONE"]), float(CONFIG["LR_HEAD"])]

# ===================== Eval (Stage2-style trigger timing) =======================
@torch.no_grad()
def evaluate_attack_metrics(model, loader, target_label: int, patch_size: int, patch_value: float, patch_pos: str):
    """
    loader yields x in [0,1] (NOT normalized), y original labels
    returns: clean_acc(%), triggered_robust_acc(%), ASR_non_target(%)
    """
    model.eval().to(device)
    tot = 0
    clean_ok = 0
    trig_ok = 0
    nt_total = 0
    asr_nt_hits = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        # clean
        x_clean = normalize_batch(x)
        pred = model(x_clean).argmax(1)
        clean_ok += (pred == y).sum().item()

        # triggered: apply patch in [0,1], then normalize
        x_trig = trigger_patch_batch(x, size=patch_size, value=patch_value, pos=patch_pos)
        x_trig = normalize_batch(x_trig)
        pred_t = model(x_trig).argmax(1)

        trig_ok += (pred_t == y).sum().item()

        mask = (y != int(target_label))
        nt_total += mask.sum().item()
        asr_nt_hits += (pred_t[mask] == int(target_label)).sum().item()

        tot += y.size(0)

    clean = 100.0 * clean_ok / max(1, tot)
    trig  = 100.0 * trig_ok  / max(1, tot)
    asrnt = 100.0 * asr_nt_hits / max(1, nt_total)
    return clean, trig, asrnt

# ===================== Baseline ============================
c0, t0, asr0 = evaluate_attack_metrics(
    model, testloader,
    target_label=int(CONFIG["TARGET_LABEL"]),
    patch_size=int(CONFIG["PATCH_SIZE"]),
    patch_value=float(CONFIG["PATCH_VALUE"]),
    patch_pos=str(CONFIG["PATCH_POS"]),
)
print(f"[Baseline] Clean {c0:.2f}% | Triggered {t0:.2f}% | ASR(non-target) {asr0:.2f}%")

# ===================== Fine-tune (inject BadNets, Stage2-style per-batch mix) =========
os.makedirs(CONFIG["SAVE_DIR"], exist_ok=True)
best_clean = -1.0
best_asr = -1.0

def save_ckpt(tag, epoch, clean, trig, asrnt):
    path = os.path.join(CONFIG["SAVE_DIR"], f"{tag}.pth")
    torch.save({
        "epoch": int(epoch),
        "model_state": unwrap(model).state_dict(),
        "optim_state": optimizer.state_dict(),
        "metrics": {"clean": float(clean), "triggered": float(trig), "asr_non_target": float(asrnt)},
        "classes": train_set.classes,
        "config": CONFIG,
    }, path)
    print(f"[Save] {tag}.pth (clean={clean:.2f}, trig={trig:.2f}, ASRnt={asrnt:.2f})")

print(
    f"[Finetune-BadNets|Stage2-style] epochs={CONFIG['FT_EPOCHS']} | "
    f"pA={float(CONFIG['POISON_RATE'])*100:.1f}%/sample | patch={CONFIG['PATCH_SIZE']}@{CONFIG['PATCH_POS']} | "
    f"target={CONFIG['TARGET_LABEL']}({train_set.classes[int(CONFIG['TARGET_LABEL'])]}) | "
    f"lr_backbone={CONFIG['LR_BACKBONE']} lr_head={CONFIG['LR_HEAD']}"
)

pA = float(CONFIG["POISON_RATE"])
target = int(CONFIG["TARGET_LABEL"])
patch = int(CONFIG["PATCH_SIZE"])
pval = float(CONFIG["PATCH_VALUE"])
ppos = str(CONFIG["PATCH_POS"])

for ep in range(1, int(CONFIG["FT_EPOCHS"]) + 1):
    model.train()

    # warmup
    if ep <= int(CONFIG["WARMUP_EPOCHS"]):
        factor = ep / float(max(1, int(CONFIG["WARMUP_EPOCHS"])))
        set_warmup_lr(optimizer, base_lrs, factor)

    run_loss = 0.0
    n_seen = 0

    for x, y in trainloader:
        x = x.to(device, non_blocking=True)  # [B,3,H,W] in [0,1]
        y = y.to(device, non_blocking=True)

        B = x.size(0)
        mask_A = (torch.rand(B, device=device) < pA)

        # Stage2-style: mix batch, only some samples poisoned
        x_mix = x
        y_tgt = y
        if mask_A.any():
            x_mix = x.clone()
            y_tgt = y.clone()
            x_mix[mask_A] = trigger_patch_batch(x_mix[mask_A], size=patch, value=pval, pos=ppos)
            y_tgt[mask_A] = target

        # Normalize after patch (Stage2-consistent)
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

# ===================== Final ===============================
torch.save(unwrap(model).state_dict(), os.path.join(CONFIG["SAVE_DIR"], "weights_only.pth"))
print("[Done] Saved to:", CONFIG["SAVE_DIR"])
