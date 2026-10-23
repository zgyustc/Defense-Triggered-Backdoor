# Extracted from BB/anbition_denfense-Final-abition_k_3.ipynb, code cell 22 (zero-based).
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NAD (strict aligned to the BackdoorBench/NAD algorithm part you pasted)
- Teacher: clean fine-tuning with CE only (PureCleanModelTrainer behavior, but implemented inline)
- Student: CE + beta3 * AT(activation3), with official adjust_learning_rate schedule
- Trainloader yields: (inputs, labels, original_index, poison_indicator, original_targets)
- Evaluation always prints A + B metrics (even if B does not exist)
- SAVE: save ONLY the final student model state_dict (minimal size, FP-style)
"""

import os, copy, math, random
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
# ---------------- CIFAR constants (for B trigger) ----------------
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2023, 0.1994, 0.2010)
MEAN_T = torch.tensor(CIFAR_MEAN).view(1, 3, 1, 1)
STD_T  = torch.tensor(CIFAR_STD ).view(1, 3, 1, 1)


# ---------------- config ----------------
@dataclass
class NADStrictCfg:
    data_root: str = "./data"
    save_dir: str = "nad_strict_AB"

    # clean subset selection (align choose_index semantics)
    # set either ratio or explicit samples (samples takes priority if not None)
    clean_ratio: Optional[float] = None     # e.g., 0.01
    clean_samples: int = 500               # e.g., 500

    batch_size: int = 64
    num_workers: int = 8
    seed: int = 0

    # teacher FT (clean CE only)
    te_epochs: int = 10
    te_lr: float = 3e-3
    te_wd: float = 1e-4

    # student NAD
    epochs: int = 30
    lr: float = 3e-3
    wd: float = 1e-4
    beta3: float = 1000.0   # BackdoorBench uses args.beta3
    p: float = 2.0          # AT power

    # A/B evaluation definition
    target_A: int = 1
    patch_size_A: int = 3

    target_B: int = 1
    sig_amp: float = 0.1
    sig_freq: float = 6.0

    # saving
    save_name: str = "nad_student_final_ab.pth"


# ---------------- B trigger ----------------
class TriggerSigAB:
    """
    Add sinusoidal pattern in pixel space then re-normalize.
    Input x is normalized (as env.data.testloader provides).
    """
    def __init__(self, amp: float = 0.1, freq: float = 6.0):
        self.amp = float(amp)
        self.freq = float(freq)

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        device, dtype = x.device, x.dtype
        mean = MEAN_T.to(device=device, dtype=dtype)
        std  = STD_T.to(device=device, dtype=dtype)

        if x.dim() == 3:
            x_ = x.unsqueeze(0)
            squeeze = True
        else:
            x_ = x
            squeeze = False

        x_den = x_ * std + mean  # approx pixel [0,1]
        B, C, H, W = x_den.shape

        yy, xx = torch.meshgrid(
            torch.linspace(0, 1, H, device=device, dtype=dtype),
            torch.linspace(0, 1, W, device=device, dtype=dtype),
            indexing="ij",
        )
        pattern = self.amp * torch.sin(2 * math.pi * self.freq * (xx + yy))
        pattern = pattern.clamp(-0.5, 0.5).view(1, 1, H, W)

        x_den = (x_den + pattern).clamp(0.0, 1.0)
        x_norm = (x_den - mean) / std

        return x_norm[0] if squeeze else x_norm


# ---------------- official LR schedule (as in your pasted code) ----------------
def adjust_learning_rate(optimizer: torch.optim.Optimizer, epoch: int, base_lr: float):
    if epoch < 2:
        lr = base_lr
    elif epoch < 20:
        lr = 0.01
    elif epoch < 30:
        lr = 0.0001
    else:
        lr = 0.0001
    for pg in optimizer.param_groups:
        pg["lr"] = lr


# ---------------- AT (strict aligned to your pasted code) ----------------
class AT(nn.Module):
    def __init__(self, p: float):
        super().__init__()
        self.p = float(p)

    def forward(self, fm_s: torch.Tensor, fm_t: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(self.attention_map(fm_s), self.attention_map(fm_t))

    def attention_map(self, fm: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        am = torch.pow(torch.abs(fm), self.p)
        am = torch.sum(am, dim=1, keepdim=True)
        norm = torch.norm(am, dim=(2, 3), keepdim=True)
        am = torch.div(am, norm + eps)
        return am


# ---------------- dataset wrapper to match (inputs, labels, original_index, poison_indicator, original_targets) ----------------
class CleanMetaWrapper(Dataset):
    def __init__(self, base: Dataset, indices: List[int]):
        self.base = base
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i: int):
        base_idx = self.indices[i]
        x, y = self.base[base_idx]
        original_index = base_idx
        poison_indicator = 0  # clean subset
        original_targets = y
        return x, y, original_index, poison_indicator, original_targets


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_index_equiv(n: int, ratio: Optional[float], samples: int, seed: int) -> List[int]:
    rng = np.random.RandomState(seed)
    all_idx = np.arange(n)
    rng.shuffle(all_idx)
    if ratio is not None:
        k = max(1, int(round(n * float(ratio))))
    else:
        k = max(1, int(samples))
    k = min(k, n)
    return all_idx[:k].tolist()


def build_clean_subset_loader(cfg: NADStrictCfg) -> DataLoader:
    """
    Align to BackdoorBench intent:
      - train transform uses augmentation (get_transform(..., train=True))
      - choose_index selects indices deterministically by seed
      - returns meta fields like prepro_cls_DatasetBD_v2 would
    """
    train_tfm = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    ds = datasets.CIFAR10(cfg.data_root, train=True, download=True, transform=train_tfm)
    idx = choose_index_equiv(len(ds), cfg.clean_ratio, cfg.clean_samples, cfg.seed)
    wrapped = CleanMetaWrapper(ds, idx)

    return DataLoader(
        wrapped,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )


@torch.no_grad()
def eval_ab(env: ModelEnv, model, cfg: NADStrictCfg, tag: str):
    trigA = TriggerBadNets(cfg.patch_size_A)
    trigB = TriggerSigAB(cfg.sig_amp, cfg.sig_freq)

    clean = env.eval.clean(model, env.data.testloader)
    _, trigA_acc, asrA = env.eval.metrics_all_to_one(model, env.data.testloader, trigA, cfg.target_A)
    _, trigB_acc, asrB = env.eval.metrics_all_to_one(model, env.data.testloader, trigB, cfg.target_B)

    print(f"[{tag}] Clean {clean:.2f}% | "
          f"A: Trig {trigA_acc:.2f}% ASR_A {asrA:.2f}% | "
          f"B: Trig {trigB_acc:.2f}% ASR_B {asrB:.2f}%")
    return dict(clean=clean, A_trig=trigA_acc, ASR_A=asrA, B_trig=trigB_acc, ASR_B=asrB)


def build_activation3(model: nn.Module, x: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Strictly match the resnet18 branch in your pasted NADModelTrainer:
      features_out = list(model.children())[:-1]
      modelout = nn.Sequential(*features_out)
      activation3 = modelout(inputs)
    """
    m = unwrap(model)
    features_out = list(m.children())[:-1]
    modelout = nn.Sequential(*features_out).to(device)
    return modelout(x)


@torch.no_grad()
def save_final_student(student: nn.Module, save_dir: str, save_name: str):
    """
    Save ONLY the final student model state_dict (minimal file size; avoids optimizer/scaler/teacher).
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, save_name)
    torch.save(unwrap(student).state_dict(), save_path)
    print(f"[Save] final student state_dict saved to: {save_path}")


def run_nad_strict_aligned(env: ModelEnv, cfg: NADStrictCfg, inplace_student: bool = False):
    """
    Strict aligned (algorithm part) to the BackdoorBench/NAD you pasted:
      - teacher: clean fine-tune with CE only (no NAD LR schedule)
      - student: per-epoch adjust_learning_rate, loss = CE + beta3*AT(act3_s, act3_t)
      - training loader yields 5-tuple fields
      - eval always outputs A+B
      - save only final student state_dict
    """
    set_seed(cfg.seed)
    os.makedirs(cfg.save_dir, exist_ok=True)
    device = env.device

    # ---- baseline ----
    eval_ab(env, env.model, cfg, tag="Baseline-AB")

    # ---- shared clean subset loader (BackdoorBench uses one clean subset via choose_index) ----
    trainloader = build_clean_subset_loader(cfg)

    # ---- 1) teacher clean fine-tune (CE only) ----
    teacher = copy.deepcopy(env.model).to(device)
    teacher.train()

    opt_t = torch.optim.SGD(
        unwrap(teacher).parameters(),
        lr=cfg.te_lr,
        momentum=0.9,
        weight_decay=cfg.te_wd,
    )

    scaler_t = env.make_scaler()

    for ep in range(cfg.te_epochs):
        teacher.train().to(device)
        run_loss, n = 0.0, 0

        for inputs, labels, *_ in trainloader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            opt_t.zero_grad(set_to_none=True)
            with env.autocast_ctx():
                logits = teacher(inputs)
                loss = F.cross_entropy(logits, labels)
            scaler_t.scale(loss).backward()
            scaler_t.step(opt_t)
            scaler_t.update()

            run_loss += float(loss.item()) * inputs.size(0)
            n += inputs.size(0)

        print(f"[Teacher FT Ep {ep+1:02d}/{cfg.te_epochs}] CE {run_loss/max(1,n):.4f}")
        eval_ab(env, teacher, cfg, tag=f"Teacher-AB Ep{ep+1:02d}")

    teacher.eval()

    # ---- 2) student NAD (CE + beta3*AT) with official adjust_learning_rate ----
    student = env.model if inplace_student else copy.deepcopy(env.model)
    student = student.to(device)

    opt_s = torch.optim.SGD(
        unwrap(student).parameters(),
        lr=cfg.lr,
        momentum=0.9,
        weight_decay=cfg.wd,
    )

    scaler_s = env.make_scaler()
    criterionAT = AT(cfg.p).to(device)

    for ep in range(cfg.epochs):
        adjust_learning_rate(opt_s, ep, cfg.lr)

        student.train().to(device)
        run_loss, n = 0.0, 0

        for inputs, labels, *_ in trainloader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            opt_s.zero_grad(set_to_none=True)

            with env.autocast_ctx():
                outputs_s = student(inputs)

                activation3_s = build_activation3(student, inputs, device)
                with torch.no_grad():
                    activation3_t = build_activation3(teacher, inputs, device)

                cls_loss = F.cross_entropy(outputs_s, labels)
                at3_loss = criterionAT(activation3_s, activation3_t.detach()) * float(cfg.beta3)
                loss = at3_loss + cls_loss

            scaler_s.scale(loss).backward()
            scaler_s.step(opt_s)
            scaler_s.update()

            run_loss += float(loss.item()) * inputs.size(0)
            n += inputs.size(0)

        cur_lr = opt_s.param_groups[0]["lr"]
        print(f"[NAD Ep {ep+1:02d}/{cfg.epochs}] lr {cur_lr:.6f} | loss {run_loss/max(1,n):.4f}")
        eval_ab(env, student, cfg, tag=f"NAD-AB Ep{ep+1:02d}")

    # ---- save final student (ONLY state_dict) ----
    save_final_student(student, cfg.save_dir, cfg.save_name)
    return student


# ------------------- minimal example -------------------
if __name__ == "__main__":
    # You must define these in your environment (or replace them here)
    # data_path = "..."
    # ckpt_path = "..."

    ENV_CFG = dict(
        DATA_ROOT=data_path,
        CKPT_PATH=ckpt_path,
        VARIANT="resnet18",
        BATCH_SIZE=64,
        NUM_WORKERS=8,
        SEED=0,
        AMP=True,
    )
    env = ModelEnv(ENV_CFG)

    cfg = NADStrictCfg(
        data_root=data_path,
        save_dir="nad_strict_resnet18_cifar10_AB",
        clean_samples=1000,
        batch_size=64,
        num_workers=8,
        seed=0,
        te_epochs=10, te_lr=3e-3, te_wd=1e-4,
        epochs=30, lr=3e-3, wd=1e-4, beta3=1000.0, p=2.0,
        target_A=0, patch_size_A=3,
        target_B=1, sig_amp=0.1, sig_freq=6.0,
        save_name="nad_student_final_ab.pth",
    )

    run_nad_strict_aligned(env, cfg, inplace_student=False)
