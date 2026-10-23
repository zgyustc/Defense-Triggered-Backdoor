# Extracted from BB/vit.ipynb, code cell 2 (zero-based).
# train_vit_imagenet12_clean.py
import os, random, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

# ---------------- Config ----------------
CFG = dict(
    DATA_ROOT="./VIT/imagenet12",
    BATCH_SIZE=256,
    EPOCHS=150,
    NUM_WORKERS=8,
    SEED=0,

    # Device control
    DEVICE=os.environ.get("DEVICE", "cuda:0"),  # set to "cuda:0"
    STRICT_DEVICE=True,                         # True: do NOT silently fallback

    # Finetune hyperparams (ViT)
    LR_BACKBONE=5e-5,
    LR_HEAD=5e-4,
    WD=1e-2,
    WARMUP_EPOCHS=5,
    HEAD_ONLY_EPOCHS=5,
    LABEL_SMOOTH=0.1,

    SAVE_DIR="clean_vit_base_imagenet12",
    CKPT_PATH="./VIT/model.safetensors",

    # DataParallel is NOT recommended when you are pinning a specific GPU
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
        # allow "cpu"
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

IMNET_MEAN = (0.485, 0.456, 0.406)
IMNET_STD  = (0.229, 0.224, 0.225)

# -------------- Data --------------------
IMG_SIZE = 224
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

def _count_images(root_dir: str, exts=(".jpg",".jpeg",".png",".ppm",".bmp",".pgm",".tif",".tiff",".webp")) -> int:
    n = 0
    for dp, _, fns in os.walk(root_dir):
        for fn in fns:
            if fn.lower().endswith(exts):
                n += 1
    return n

def resolve_split_dirs(root: str):
    candidates = [("train","val"), ("train","valid"), ("train","validation"), ("train","test")]
    for a, b in candidates:
        pa, pb = os.path.join(root, a), os.path.join(root, b)
        if os.path.isdir(pa) and os.path.isdir(pb):
            return pa, pb, a, b
    raise FileNotFoundError(
        f"在 {root} 下未找到可用 split 目录（期望 train/val 或 train/test 等）。"
    )

root = CFG["DATA_ROOT"]
train_dir, val_dir, train_name, val_name = resolve_split_dirs(root)

# 关键：提前检查是否真有图片文件，否则 ImageFolder 会报 “Found no valid file ...”
n_train_imgs = _count_images(train_dir)
n_val_imgs   = _count_images(val_dir)
print(f"[Data] split=({train_name},{val_name}) | train_imgs={n_train_imgs} | val_imgs={n_val_imgs}")
if n_train_imgs == 0 or n_val_imgs == 0:
    raise FileNotFoundError(
        f"ImageFolder 在 {train_dir} 或 {val_dir} 中找不到图片文件。"
        f"请确认图片后缀（jpg/jpeg/png/...）与目录结构是否为 class_subdir/*.jpg。"
    )

tr = ImageFolder(train_dir, transform=train_tfms)
te = ImageFolder(val_dir,   transform=val_tfms)

assert set(tr.class_to_idx.keys()) == set(te.class_to_idx.keys()), "train 与 val 的类别集合不一致"
num_classes = len(tr.classes)
assert num_classes > 0, "未在 train/ 下找到类别文件夹"
print(f"[Data] classes={num_classes} | train={len(tr)} | val={len(te)}")

pin = (device.type == "cuda")
trainloader = DataLoader(
    tr, batch_size=CFG["BATCH_SIZE"], shuffle=True,
    num_workers=CFG["NUM_WORKERS"], pin_memory=pin, drop_last=True, persistent_workers=(CFG["NUM_WORKERS"]>0)
)
testloader  = DataLoader(
    te, batch_size=CFG["BATCH_SIZE"], shuffle=False,
    num_workers=CFG["NUM_WORKERS"], pin_memory=pin, persistent_workers=(CFG["NUM_WORKERS"]>0)
)

# -------------- ViT (timm ViT-B/16) --------------
def _load_state_dict_any(ckpt_path: str):
    """兼容 .safetensors / .pth / 各种打包格式，返回纯 state_dict(dict[str,Tensor])"""
    if ckpt_path.endswith(".safetensors"):
        try:
            from safetensors.torch import load_file
        except Exception as e:
            raise RuntimeError("需要安装 safetensors 以读取 .safetensors：pip install safetensors") from e
        sd = load_file(ckpt_path)
    else:
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(sd, dict):
            for k in ["state_dict", "model", "net", "weights", "model_state"]:
                if k in sd and isinstance(sd[k], dict):
                    sd = sd[k]
                    break
    if not isinstance(sd, dict):
        raise RuntimeError("无法从 checkpoint 中解析出 state_dict")
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    return sd

def build_vit_b16_pretrained(num_classes: int, ckpt_path: str):
    import timm
    assert os.path.isfile(ckpt_path), f"找不到预训练权重：{ckpt_path}"

    m = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=num_classes)

    sd_src = _load_state_dict_any(ckpt_path)
    sd_dst = m.state_dict()
    sd_filt, skipped = {}, []

    for k, v in sd_src.items():
        if k.startswith("head."):
            skipped.append((k, "skip head"))
            continue
        if k not in sd_dst:
            skipped.append((k, "not in model"))
            continue
        if sd_dst[k].shape != v.shape:
            skipped.append((k, f"shape {tuple(v.shape)} -> {tuple(sd_dst[k].shape)}"))
            continue
        sd_filt[k] = v

    msg = m.load_state_dict(sd_filt, strict=False)
    matched_ratio = sum(sd_dst[k].numel() for k in sd_filt.keys()) / max(1, sum(p.numel() for p in sd_dst.values()))
    print(f"[ViT] loaded backbone (ignore head), matched_ratio={matched_ratio:.3f}")
    print(f"[ViT] missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)} | skipped={len(skipped)}")
    for i, (k, why) in enumerate(skipped[:8]):
        print(f"  - skipped[{i}]: {k} ({why})")
    return m

model = build_vit_b16_pretrained(num_classes, CFG["CKPT_PATH"]).to(device)

# 不建议在固定 cuda:1 时开启 DP；默认关闭
if CFG["USE_DATAPARALLEL"] and (device.type == "cuda") and (torch.cuda.device_count() > 1):
    model = nn.DataParallel(model)

def unwrap(m): return m.module if isinstance(m, nn.DataParallel) else m
n_params = sum(p.numel() for p in unwrap(model).parameters()) / 1e6
print(f"[Model] {unwrap(model).__class__.__name__} | params ≈ {n_params:.2f}M | classes={num_classes}")

# -------------- Optim / Sched --------------
head_params, backbone_params = [], []
for n, p in unwrap(model).named_parameters():
    (head_params if n.startswith("head.") else backbone_params).append(p)

optimizer = torch.optim.AdamW(
    [{"params": backbone_params, "lr": CFG["LR_BACKBONE"]},
     {"params": head_params,     "lr": CFG["LR_HEAD"]}],
    weight_decay=CFG["WD"]
)

# 新版 AMP（避免你之前看到的 GradScaler warning）
try:
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
except Exception:
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=max(1, CFG["EPOCHS"] - CFG["WARMUP_EPOCHS"])
)

def set_warmup_lr(optim, base_lrs, factor: float):
    for pg, base_lr in zip(optim.param_groups, base_lrs):
        pg["lr"] = base_lr * factor

base_lrs = [CFG["LR_BACKBONE"], CFG["LR_HEAD"]]

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
        "classes": tr.classes,
        "config": CFG,
    }, path)
    print(f"[Save] {tag}.pth (acc={acc:.2f})")

print(f"[Train] epochs={CFG['EPOCHS']} | bs={CFG['BATCH_SIZE']} | warmup={CFG['WARMUP_EPOCHS']} | "
      f"head_only={CFG['HEAD_ONLY_EPOCHS']} | ls={CFG['LABEL_SMOOTH']} | "
      f"lr_backbone={CFG['LR_BACKBONE']} | lr_head={CFG['LR_HEAD']} | wd={CFG['WD']}")

for ep in range(1, CFG["EPOCHS"] + 1):
    model.train()

    # head-only warm start
    if ep <= CFG["HEAD_ONLY_EPOCHS"]:
        for n, p in unwrap(model).named_parameters():
            p.requires_grad = n.startswith("head.")
    else:
        for p in unwrap(model).parameters():
            p.requires_grad = True

    # warmup
    if ep <= CFG["WARMUP_EPOCHS"]:
        factor = ep / float(max(1, CFG["WARMUP_EPOCHS"]))
        set_warmup_lr(optimizer, base_lrs, factor)

    run_loss = 0.0
    n_seen = 0

    for x, y in trainloader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        # autocast
        with torch.autocast(device_type="cuda", enabled=(device.type == "cuda"), dtype=torch.float16):
            logits = model(x)
            try:
                loss = F.cross_entropy(logits, y, label_smoothing=CFG["LABEL_SMOOTH"])
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
    cur_lrs = [pg["lr"] for pg in optimizer.param_groups]
    print(f"[Epoch {ep:03d}/{CFG['EPOCHS']}] CE {ce:.4f} | Acc {acc:.2f}% | "
          f"lr(backbone/head) {cur_lrs[0]:.2e}/{cur_lrs[1]:.2e}")

    save_ckpt("last", ep, acc)
    if acc > best:
        best = acc
        save_ckpt("best_clean", ep, acc)

# -------------- Final ----------------------
torch.save(unwrap(model).state_dict(), os.path.join(CFG["SAVE_DIR"], "weights_only.pth"))
save_ckpt("final", CFG["EPOCHS"], best)
print("Done. Checkpoints in:", CFG["SAVE_DIR"])
