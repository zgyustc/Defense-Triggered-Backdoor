# Extracted from BB/test.ipynb, code cell 2 (zero-based).
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage-1: Benign training on CIFAR-10 (ResNet-18, CIFAR stem)
Saves: ./exp_three_stage/benign_model.pth
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

    # 经典 CIFAR 配方（按 batch=256 线性放大初始 LR）
    lr: float = 0.2
    momentum: float = 0.9
    weight_decay: float = 5e-4
    label_smoothing: float = 0.1

    # files
    save_dir: str = "./exp_three_stage"
    save_benign: str = "benign_model_18.pth"

cfg = Cfg()
from dtb.runtime import configure as _configure
cfg = _configure(cfg)

def set_seed(s: int):
    random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    # 为了更高的上限，这里不开 deterministic；需要完全复现可手动打开
    torch.backends.cudnn.benchmark = True
set_seed(cfg.seed)

# ============== Triggers (for eval only) ==============
def trigger_patch(x: torch.Tensor, size=4, value=1.0, pos="br"):
    c, h, w = x.shape
    s = size
    if pos == "br":
        x[:, h - s:h, w - s:w] = value
    elif pos == "tr":
        x[:, :s, w - s:w] = value
    elif pos == "bl":
        x[:, h - s:h, :s] = value
    elif pos == "tl":
        x[:, :s, :s] = value
    else:
        raise ValueError("pos must be tl/tr/bl/br")
    return x

class PoisonedView(torch.utils.data.Dataset):
    """For eval: clean/A/B view after Normalize."""
    def __init__(self, base, mode, size, value, pos, p, normalize):
        self.base, self.mode = base, mode
        self.size, self.value, self.pos, self.p = size, value, pos, p
        self.normalize = normalize
    def __len__(self): return len(self.base)
    def __getitem__(self, i):
        x, y = self.base[i]  # ToTensor -> [0,1]
        if self.mode != "clean":
            x = trigger_patch(x.clone(), self.size, self.value, self.pos)
        x = self.normalize(x)
        return x, y

def make_loaders() -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2023, 0.1994, 0.2010)

    tf_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),   # 训练集在管道内做 Normalize
    ])
    tf_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_base = datasets.CIFAR10(cfg.data_root, train=True,  transform=tf_train, download=True)
    test_base  = datasets.CIFAR10(cfg.data_root, train=False, transform=transforms.ToTensor(),  download=True)

    Ltrain = DataLoader(train_base, batch_size=cfg.batch_size, shuffle=True,
                        num_workers=cfg.num_workers, pin_memory=True, drop_last=True)

    # eval views（注意：这里再做一次 Normalize）
    tf_norm = transforms.Normalize(mean, std)
    test_clean = PoisonedView(test_base, "clean", 0, 0.0, "br", 0.0, tf_norm)
    test_A     = PoisonedView(test_base, "A", 4, 1.0, "br", 1.0, tf_norm)
    test_B     = PoisonedView(test_base, "B", 4, 1.0, "tl", 1.0, tf_norm)

    T_clean = DataLoader(test_clean, batch_size=512, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
    T_A     = DataLoader(test_A,     batch_size=512, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
    T_B     = DataLoader(test_B,     batch_size=512, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
    return Ltrain, T_clean, T_A, T_B

def build_cifar_resnet18(num_classes: int):
    """在 torchvision resnet18 基础上，替换 CIFAR 友好的 stem。"""
    m = models.resnet18(weights=None, num_classes=num_classes)
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    return m

@torch.no_grad()
def evaluate(model, T_clean, T_A, T_B, device):
    model.eval()
    def acc_on(loader, target=None):
        hit = tot = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            pred = logits.argmax(1)
            if target is None:
                hit += (pred == y).sum().item()
            else:
                hit += (pred == target).sum().item()
            tot += y.numel()
        return hit / max(1, tot)
    ACC   = acc_on(T_clean)
    ASR_A = acc_on(T_A, target=0)
    ASR_B = acc_on(T_B, target=1)
    return ACC, ASR_A, ASR_B

def main():
    os.makedirs(cfg.save_dir, exist_ok=True)
    print("===> Stage-1 Config\n", json.dumps(cfg.__dict__, indent=2))

    Ltrain, T_clean, T_A, T_B = make_loaders()
    device = cfg.device

    model = build_cifar_resnet18(cfg.num_classes).to(device)

    # 经典 SGD + Cosine
    opt = torch.optim.SGD(model.parameters(),
                          lr=cfg.lr, momentum=cfg.momentum,
                          weight_decay=cfg.weight_decay, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)

    best_acc = 0.0
    for ep in range(1, cfg.epochs+1):
        model.train()
        loss_meter = 0.0
        for x, y in Ltrain:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y, label_smoothing=cfg.label_smoothing)
            loss.backward()
            opt.step()
            loss_meter += float(loss.item())
        scheduler.step()

        ACC, ASR_A, ASR_B = evaluate(model, T_clean, T_A, T_B, device)
        print(f"[Stage1|epoch {ep:03d}] ACC={ACC*100:.2f}% ASR-A={ASR_A*100:.2f}% ASR-B={ASR_B*100:.2f}% "
              f"| loss={loss_meter/len(Ltrain):.4f} | lr={scheduler.get_last_lr()[0]:.5f}")

        if ACC > best_acc:
            best_acc = ACC
            path = os.path.join(cfg.save_dir, cfg.save_benign)
            torch.save({"state_dict": model.state_dict(), "cfg": cfg.__dict__}, path)
            print(f"[save] benign(best so far {best_acc*100:.2f}%) -> {path}")

if __name__ == "__main__":
    main()
