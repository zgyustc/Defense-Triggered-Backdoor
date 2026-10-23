# Extracted from BB/resent18+tinyimagent.ipynb, code cell 4 (zero-based).
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

import random, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


CONFIG = dict(
    DATA_ROOT="./data/tiny-imagenet-200",
    CKPT_PATH="./clean_vit_base_tinyimagenet/best_clean.pth",
    BATCH_SIZE=128,
    NUM_WORKERS=8,

    TARGET_LABEL=1,
    PATCH_SIZE=32,
    PATCH_POS="br",
    PATCH_VALUE=1.0,
    POISON_RATE=0.10,

    FT_EPOCHS=30,
    LR_BACKBONE=5e-5,
    LR_HEAD=5e-4,
    WD=1e-2,
    WARMUP_EPOCHS=3,
    AMP=True,
    LABEL_SMOOTH=0.1,

    SEED=0,
    SAVE_DIR="badnets_vit_base_tinyimagenet",
    USE_DATAPARALLEL=False,
)
from dtb.runtime import configure as _configure
CONFIG = _configure(CONFIG)


SEED = int(CONFIG["SEED"])
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True
print("[Device]", device, "| visible cuda:", torch.cuda.device_count(), "| CVD:", os.environ.get("CUDA_VISIBLE_DEVICES"))


IMNET_MEAN = (0.485, 0.456, 0.406)
IMNET_STD = (0.229, 0.224, 0.225)


def normalize_batch(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMNET_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMNET_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - mean) / std


def trigger_patch_batch(x: torch.Tensor, size: int = 32, value: float = 1.0, pos: str = "br") -> torch.Tensor:
    if size <= 0:
        return x
    y = x.clone()
    _, _, h, w = y.shape
    s = int(size)
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


IMG_SIZE = 224
train_tfms = T.Compose([
    T.RandomResizedCrop(IMG_SIZE, scale=(0.08, 1.0), interpolation=T.InterpolationMode.BICUBIC),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
])
val_tfms = T.Compose([
    T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(IMG_SIZE),
    T.ToTensor(),
])


def maybe_extract_tinyimagenet(data_root: str):
    root = Path(data_root)
    zip_path = root.with_suffix(".zip")
    if root.is_dir():
        return
    if not zip_path.is_file():
        return
    print(f"[TinyImageNet] extracting {zip_path} -> {root.parent}")
    import zipfile
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(root.parent)


def prepare_tinyimagenet_val_for_imagefolder(data_root: str):
    root = Path(data_root)
    val_dir = root / "val"
    images_dir = val_dir / "images"
    ann_path = val_dir / "val_annotations.txt"
    if images_dir.is_dir() and not any(images_dir.iterdir()):
        import shutil
        shutil.rmtree(images_dir)
    if not images_dir.is_dir() or not ann_path.is_file():
        return
    existing_class_dirs = [p for p in val_dir.iterdir() if p.is_dir() and p.name != "images"]
    if existing_class_dirs:
        return
    print(f"[TinyImageNet] reorganizing validation set under {val_dir}")
    with ann_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            img_name, cls_name = parts[0], parts[1]
            src = images_dir / img_name
            dst_dir = val_dir / cls_name
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / img_name
            if src.exists() and not dst.exists():
                src.rename(dst)
    if images_dir.is_dir() and not any(images_dir.iterdir()):
        import shutil
        shutil.rmtree(images_dir)


root = CONFIG["DATA_ROOT"]
maybe_extract_tinyimagenet(root)
prepare_tinyimagenet_val_for_imagefolder(root)
train_set = ImageFolder(os.path.join(root, "train"), transform=train_tfms)
val_set = ImageFolder(os.path.join(root, "val"), transform=val_tfms)
assert set(train_set.class_to_idx) == set(val_set.class_to_idx), "train/val 类别集合不一致"

num_classes = len(train_set.classes)
pin = device.type == "cuda"
trainloader = DataLoader(
    train_set,
    batch_size=CONFIG["BATCH_SIZE"],
    shuffle=True,
    num_workers=CONFIG["NUM_WORKERS"],
    pin_memory=pin,
    drop_last=True,
    persistent_workers=(CONFIG["NUM_WORKERS"] > 0),
)
testloader = DataLoader(
    val_set,
    batch_size=CONFIG["BATCH_SIZE"],
    shuffle=False,
    num_workers=CONFIG["NUM_WORKERS"],
    pin_memory=pin,
    persistent_workers=(CONFIG["NUM_WORKERS"] > 0),
)

print(f"[Data] Tiny-ImageNet | classes={num_classes} | train={len(train_set)} | val={len(val_set)}")
print(f"[Target] idx={CONFIG['TARGET_LABEL']} | name={train_set.classes[CONFIG['TARGET_LABEL']]}")


def build_vit_b16(num_classes: int):
    import timm
    return timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=num_classes)


def unwrap(m):
    return m.module if isinstance(m, nn.DataParallel) else m


print(f"[Load ckpt] {CONFIG['CKPT_PATH']}")
state_obj = torch.load(CONFIG["CKPT_PATH"], map_location="cpu", weights_only=False)
if isinstance(state_obj, dict) and "model_state" in state_obj:
    state = state_obj["model_state"]
elif isinstance(state_obj, dict) and "state_dict" in state_obj:
    state = state_obj["state_dict"]
else:
    state = state_obj
if any(k.startswith("module.") for k in state.keys()):
    state = {k.replace("module.", "", 1): v for k, v in state.items()}

model = build_vit_b16(num_classes).to(device)
msg = model.load_state_dict(state, strict=False)
print(f"[load_state_dict] missing={len(msg.missing_keys)} | unexpected={len(msg.unexpected_keys)}")

if CONFIG["USE_DATAPARALLEL"] and device.type == "cuda" and torch.cuda.device_count() > 1:
    model = nn.DataParallel(model).to(device)


head_params, backbone_params = [], []
for n, p in unwrap(model).named_parameters():
    (head_params if n.startswith("head.") else backbone_params).append(p)

optimizer = torch.optim.AdamW(
    [{"params": backbone_params, "lr": CONFIG["LR_BACKBONE"]},
     {"params": head_params, "lr": CONFIG["LR_HEAD"]}],
    weight_decay=CONFIG["WD"],
)

try:
    scaler = torch.amp.GradScaler("cuda", enabled=(CONFIG["AMP"] and device.type == "cuda"))
except Exception:
    scaler = torch.cuda.amp.GradScaler(enabled=(CONFIG["AMP"] and device.type == "cuda"))

Tmax = max(1, int(CONFIG["FT_EPOCHS"]) - int(CONFIG["WARMUP_EPOCHS"]))
cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Tmax)


def set_warmup_lr(optim, base_lrs, factor: float):
    for pg, base_lr in zip(optim.param_groups, base_lrs):
        pg["lr"] = base_lr * factor


base_lrs = [CONFIG["LR_BACKBONE"], CONFIG["LR_HEAD"]]


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
        mask = y != int(target_label)
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

    if ep <= int(CONFIG["WARMUP_EPOCHS"]):
        factor = ep / float(max(1, int(CONFIG["WARMUP_EPOCHS"])))
        set_warmup_lr(optimizer, base_lrs, factor)

    run_loss = 0.0
    n_seen = 0

    for x, y in trainloader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        bsz = x.size(0)
        mask_A = torch.rand(bsz, device=device) < pA

        x_mix = x
        y_tgt = y
        if mask_A.any():
            x_mix = x.clone()
            y_tgt = y.clone()
            x_mix[mask_A] = trigger_patch_batch(x_mix[mask_A], size=patch, value=pval, pos=ppos)
            y_tgt[mask_A] = target

        x_in = normalize_batch(x_mix)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", enabled=(CONFIG["AMP"] and device.type == "cuda"), dtype=torch.float16):
            logits = model(x_in)
            try:
                loss = F.cross_entropy(logits, y_tgt, label_smoothing=float(CONFIG["LABEL_SMOOTH"]))
            except TypeError:
                loss = F.cross_entropy(logits, y_tgt)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        run_loss += float(loss.item()) * bsz
        n_seen += bsz

    if ep > int(CONFIG["WARMUP_EPOCHS"]):
        cosine.step()

    ce = run_loss / max(1, n_seen)
    clean, trig, asrnt = evaluate_attack_metrics(
        model, testloader, target_label=target, patch_size=patch, patch_value=pval, patch_pos=ppos
    )
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
