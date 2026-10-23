# Extracted from BB/VGG16.ipynb, code cell 2 (zero-based).
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage-1: Benign training on CIFAR-10 (VGG16-BN backbone, CIFAR-friendly head)
Saves: ./exp_three_stage/benign_model_vgg16bn_cifar10.pth

Notes:
- Pure benign training: NO backdoor injection, NO trigger eval, NO ASR logs.
- Designed to be stable (avoid the "collapse to 10%" issue).
"""

import os, json, random
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# ==================== Config ====================
@dataclass
class Cfg:
    data_root: str = "./data"
    seed: int = 1
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"

    num_classes: int = 10
    batch_size: int = 256
    num_workers: int = 8
    epochs: int = 200

    # Optim
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4

    # LR schedule
    warmup_epochs: int = 5

    # Regularization
    label_smoothing: float = 0.0
    grad_clip: float = 0.0  # 0=disable; recommend 5.0 if you want extra safety

    # Augment (mild & stable by default)
    use_randaugment: bool = False
    ra_num_ops: int = 2
    ra_magnitude: int = 7
    random_erasing_p: float = 0.0

    # AMP
    use_amp: bool = False

    # Save
    save_dir: str = "./exp_three_stage"
    save_benign: str = "benign_model_vgg16bn_cifar10.pth"

cfg = Cfg()
from dtb.runtime import configure as _configure
cfg = _configure(cfg)

def set_seed(s: int):
    random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark = True

set_seed(cfg.seed)

# CIFAR-10 mean/std
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)

def make_loaders() -> Tuple[DataLoader, DataLoader]:
    tf_train_list = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ]
    if cfg.use_randaugment:
        tf_train_list.append(
            transforms.RandAugment(num_ops=cfg.ra_num_ops, magnitude=cfg.ra_magnitude)
        )

    tf_train_list += [
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ]

    if cfg.random_erasing_p and cfg.random_erasing_p > 0:
        tf_train_list.append(transforms.RandomErasing(p=cfg.random_erasing_p))

    tf_train = transforms.Compose(tf_train_list)
    tf_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    train_set = datasets.CIFAR10(cfg.data_root, train=True, transform=tf_train, download=True)
    test_set  = datasets.CIFAR10(cfg.data_root, train=False, transform=tf_test, download=True)

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(cfg.num_workers > 0),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=512,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=(cfg.num_workers > 0),
    )
    return train_loader, test_loader

class VGG16BN_CIFAR(nn.Module):
    """
    VGG16-BN backbone (torchvision) + CIFAR-friendly head:
      features -> AdaptiveAvgPool2d(1) -> Linear(512 -> num_classes)
    """
    def __init__(self, num_classes: int = 10, dropout: float = 0.0):
        super().__init__()
        base = models.vgg16_bn(weights=None)
        self.features = base.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if dropout and dropout > 0:
            self.classifier = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(512, num_classes),
            )
        else:
            self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)   # [B, 512]
        x = self.classifier(x)
        return x

@torch.no_grad()
def evaluate_acc(model: nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    hit = tot = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        pred = model(x).argmax(1)
        hit += (pred == y).sum().item()
        tot += y.numel()
    return hit / max(1, tot)

def main():
    os.makedirs(cfg.save_dir, exist_ok=True)
    print("===> Stage-1 Config\n", json.dumps(cfg.__dict__, indent=2))

    device = cfg.device
    train_loader, test_loader = make_loaders()

    model = VGG16BN_CIFAR(num_classes=cfg.num_classes, dropout=0.0).to(device)

    opt = torch.optim.SGD(
        model.parameters(),
        lr=cfg.lr,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
        nesterov=True
    )

    # Warmup + Cosine
    cosine_T = max(1, cfg.epochs - cfg.warmup_epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cosine_T)

    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.use_amp and str(device).startswith("cuda")))

    def set_epoch_lr(epoch: int):
        if cfg.warmup_epochs > 0 and epoch <= cfg.warmup_epochs:
            lr_now = cfg.lr * (epoch / float(cfg.warmup_epochs))
            for pg in opt.param_groups:
                pg["lr"] = lr_now

    best_acc = 0.0
    save_path = os.path.join(cfg.save_dir, cfg.save_benign)

    for ep in range(1, cfg.epochs + 1):
        model.train()
        set_epoch_lr(ep)

        loss_sum = 0.0
        n = 0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                logits = model(x)
                loss = F.cross_entropy(logits, y, label_smoothing=cfg.label_smoothing)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                if cfg.grad_clip and cfg.grad_clip > 0:
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                if cfg.grad_clip and cfg.grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip)
                opt.step()

            loss_sum += float(loss.item()) * x.size(0)
            n += x.size(0)

        if ep > cfg.warmup_epochs:
            scheduler.step()

        acc = evaluate_acc(model, test_loader, device)
        lr_show = opt.param_groups[0]["lr"]
        print(f"[Stage1|epoch {ep:03d}] ACC={acc*100:.2f}% | loss={loss_sum/max(1,n):.4f} | lr={lr_show:.5f}")

        if acc > best_acc:
            best_acc = acc
            torch.save({"state_dict": model.state_dict(), "cfg": cfg.__dict__}, save_path)
            print(f"[save] benign(best so far {best_acc*100:.2f}%) -> {save_path}")

if __name__ == "__main__":
    main()
