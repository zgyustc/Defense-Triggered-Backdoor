# Extracted from BB/VGG16.ipynb, code cell 4 (zero-based).
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage-2 (Classic): Inject a single BadNets-style backdoor A on CIFAR-10 (VGG16-BN, CIFAR-friendly head).
- No channel selection / no extra regularizers / no unlearning / no importance statistics.
Saves: ./exp_three_stage/afterA_stage2_simple_vgg16bn_cifar10.pth
"""

import os, math, random
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# ==================== Config ====================
@dataclass
class Stage2Cfg:
    data_root: str = "./data"
    save_dir: str = "./exp_three_stage"

    # input: benign model (from your Stage-1 VGG16-BN CIFAR10 script)
    ckpt_benign: str = "benign_model_vgg16bn_cifar10.pth"

    # output: after-A backdoored model
    ckpt_afterA: str = "afterA_stage2_simple_vgg16bn_cifar10.pth"

    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    num_workers: int = 8

    # training hyperparams
    batch_size: int = 256
    epochs: int = 20
    lr: float = 0.05
    momentum: float = 0.9
    weight_decay: float = 5e-4

    # classic poisoning ratio (per-batch)
    poison_rate: float = 0.1

    # A backdoor (BadNets)
    trigger_A_size: int = 3
    trigger_A_value: float = 1.0
    trigger_A_pos: str = "br"
    target_A: int = 0

    # B (eval only)
    sig_amp: float = 0.1
    sig_freq: float = 6.0
    target_B: int = 1

    seed: int = 1

cfg = Stage2Cfg()
from dtb.runtime import configure as _configure
cfg = _configure(cfg)

# CIFAR-10 mean/std
MEAN = (0.4914, 0.4822, 0.4465)
STD  = (0.2023, 0.1994, 0.2010)
norm = transforms.Normalize(MEAN, STD)

def set_seed(s: int):
    random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark = True

set_seed(cfg.seed)

def unwrap(m):
    return m.module if hasattr(m, "module") else m

# ==================== Model (same as Stage-1) ====================
class VGG16BN_CIFAR(nn.Module):
    """
    VGG16-BN backbone (torchvision) + CIFAR-friendly head:
      features -> AdaptiveAvgPool2d(1) -> Linear(512 -> num_classes)
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        base = models.vgg16_bn(weights=None)
        self.features = base.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # [B, 512]
        x = self.classifier(x)
        return x

def build_cifar_vgg16bn(num_classes: int = 10):
    return VGG16BN_CIFAR(num_classes=num_classes)

# ==================== Triggers ====================
def trigger_patch_batch(x: torch.Tensor, size=4, value=1.0, pos="br"):
    # x: [B,3,H,W] in [0,1]
    B, C, H, W = x.shape
    s = int(size)
    if s <= 0:
        return x
    if pos == "br":
        x[:, :, H - s:H, W - s:W] = value
    elif pos == "tr":
        x[:, :, :s, W - s:W] = value
    elif pos == "bl":
        x[:, :, H - s:H, :s] = value
    elif pos == "tl":
        x[:, :, :s, :s] = value
    else:
        raise ValueError("pos must be tl/tr/bl/br")
    return x

def add_sig_trigger_batch(x: torch.Tensor, amp=0.1, freq=6.0):
    # x: [B,3,H,W] in [0,1]
    B, C, H, W = x.shape
    yy, xx = torch.meshgrid(
        torch.linspace(0, 1, H, device=x.device),
        torch.linspace(0, 1, W, device=x.device),
        indexing="ij"
    )
    pattern = amp * torch.sin(2 * math.pi * freq * (xx + yy))
    pattern = pattern.clamp(-0.5, 0.5).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    return (x + pattern).clamp(0.0, 1.0)

# ==================== Data ====================
def make_train_loader() -> DataLoader:
    tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  # keep in [0,1], normalize later
    ])
    ds = datasets.CIFAR10(cfg.data_root, train=True, transform=tf, download=True)
    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(cfg.num_workers > 0),
    )

class EvalView(torch.utils.data.Dataset):
    def __init__(self, base, mode: str):
        self.base, self.mode = base, mode
    def __len__(self): return len(self.base)
    def __getitem__(self, i):
        img, y = self.base[i]          # PIL
        x = transforms.ToTensor()(img) # [0,1]
        if self.mode == "A":
            x = trigger_patch_batch(
                x.unsqueeze(0),
                cfg.trigger_A_size,
                cfg.trigger_A_value,
                cfg.trigger_A_pos
            )[0]
        elif self.mode == "B":
            x = add_sig_trigger_batch(
                x.unsqueeze(0),
                cfg.sig_amp,
                cfg.sig_freq
            )[0]
        x = norm(x)
        return x, y

def make_test_loaders() -> Tuple[DataLoader, DataLoader, DataLoader]:
    base = datasets.CIFAR10(cfg.data_root, train=False, transform=None, download=True)
    T_clean = DataLoader(EvalView(base, "clean"), batch_size=512, shuffle=False,
                         num_workers=cfg.num_workers, pin_memory=True,
                         persistent_workers=(cfg.num_workers > 0))
    T_A = DataLoader(EvalView(base, "A"), batch_size=512, shuffle=False,
                     num_workers=cfg.num_workers, pin_memory=True,
                     persistent_workers=(cfg.num_workers > 0))
    T_B = DataLoader(EvalView(base, "B"), batch_size=512, shuffle=False,
                     num_workers=cfg.num_workers, pin_memory=True,
                     persistent_workers=(cfg.num_workers > 0))
    return T_clean, T_A, T_B

# ==================== Eval: ACC / ASR(A/B) ====================
@torch.no_grad()
def eval_acc_asr_AB(model, T_clean, T_A, T_B, target_A: int, target_B: int, device: str):
    net = unwrap(model).eval().to(device)

    def _acc(loader, target=None):
        hit = tot = 0
        for xb, yb in loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            pred = net(xb).argmax(1)
            if target is None:
                hit += (pred == yb).sum().item()
            else:
                hit += (pred == torch.full_like(yb, target)).sum().item()
            tot += yb.numel()
        return hit / max(1, tot)

    ACC_clean = _acc(T_clean, target=None)
    ASR_A = _acc(T_A, target=target_A)
    ASR_B = _acc(T_B, target=target_B)
    return dict(ACC_clean=ACC_clean, ASR_A=ASR_A, ASR_B=ASR_B)

# ==================== Stage-2: Classic BadNets injection ====================
def train_stage2_afterA():
    os.makedirs(cfg.save_dir, exist_ok=True)
    device = cfg.device

    # 1) load benign model
    model = build_cifar_vgg16bn(num_classes=10).to(device)
    ckpt_path = Path(cfg.save_dir) / cfg.ckpt_benign
    print(f"[Stage2] loading benign model from {ckpt_path}")
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
    model.load_state_dict(sd, strict=True)

    train_loader = make_train_loader()
    T_clean, T_A, T_B = make_test_loaders()

    opt = torch.optim.SGD(
        model.parameters(),
        lr=cfg.lr,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
        nesterov=True
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)

    for ep in range(1, cfg.epochs + 1):
        model.train()
        loss_meter = 0.0
        n = 0

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)  # [B,3,32,32] in [0,1]
            yb = yb.to(device, non_blocking=True)

            B = xb.size(0)
            mask = (torch.rand(B, device=device) < cfg.poison_rate)

            x = xb.clone()
            y = yb.clone()

            if mask.any():
                x[mask] = trigger_patch_batch(
                    x[mask],
                    cfg.trigger_A_size,
                    cfg.trigger_A_value,
                    cfg.trigger_A_pos
                )
                y[mask] = cfg.target_A

            x = norm(x)

            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()

            loss_meter += float(loss.item()) * B
            n += B

        sched.step()

        metrics = eval_acc_asr_AB(model, T_clean, T_A, T_B, cfg.target_A, cfg.target_B, device)
        lr_now = opt.param_groups[0]["lr"]
        print(
            f"[Stage2|ep {ep:02d}] loss={loss_meter/max(1,n):.4f} | lr={lr_now:.5f} | "
            f"ACC_clean={metrics['ACC_clean']*100:.2f}% "
            f"ASR_A={metrics['ASR_A']*100:.2f}% "
            f"ASR_B={metrics['ASR_B']*100:.2f}%"
        )

    # save final
    final_metrics = eval_acc_asr_AB(model, T_clean, T_A, T_B, cfg.target_A, cfg.target_B, device)
    save_path = Path(cfg.save_dir) / cfg.ckpt_afterA
    torch.save(
        {
            "state_dict": model.state_dict(),
            "metrics": final_metrics,
            "cfg": cfg.__dict__,
        },
        save_path
    )
    print(f"[Stage2] saved afterA ckpt -> {save_path}")
    print("[Stage2] Final metrics:", {k: f"{v*100:.2f}%" for k, v in final_metrics.items()})

if __name__ == "__main__":
    train_stage2_afterA()
