# Extracted from BB/vit_tiny.ipynb, code cell 8 (zero-based).
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage-3 (Meta teacher) — DeiT-Tiny (NON-distilled) @ CIFAR-10
A = BadNets patch (all-to-one, y->target_A)
B = SIG sinusoid (teacher-side: keep clean label; defense-side: want y->target_B after D~)

Key points (matching your ViT/ImageNet-12 Stage3 semantics, but adapted to CIFAR-10 + DeiT-Tiny):
1) "Conv channels" -> DeiT/ViT "prunable channels" = blocks.i.mlp.fc1 out_features (hidden neurons)
2) Trigger is applied in [0,1] pixel space, then Normalize(ImageNet mean/std; aligned with Stages 1 and 2)

Dependencies:
  pip install timm

Notes:
- This script assumes your ckpt_afterA is an A-backdoored DeiT-Tiny checkpoint produced by your Stage-2 injector
  (format can be either {"model_state": ...} or a plain state_dict).
- DeiT-Tiny distilled model is NOT used here. Use model_name="deit_tiny_patch16_224" (no "distilled").

Added feature:
- Save per-epoch ckpts (epoch_XXX.pth) including TEST metrics
- After training, select the checkpoint with MINIMUM TEST ASR_B among the LAST 5 epochs,
  and copy it to cfg.ckpt_best_last5_min_asrB under save_dir.
"""

import os, math, random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import CIFAR10

try:
    from torch.func import functional_call
except Exception:
    from torch.nn.utils.stateless import functional_call


# ==================== Config ====================

@dataclass
class Stage3Cfg:
    # ---- data / ckpt ----
    data_root: str = "./data"
    save_dir: str = "stage3_deit_tiny_cifar10"

    # input: Stage-2 afterA ckpt (A injected)
    ckpt_afterA: str = "./badnets_deit_tiny_cifar10/best_clean.pth"

    # output: Stage-3 teacher (meta backdoor)
    ckpt_teacher_meta: str = "afterAB_stage3_meta_deit_tiny_cifar10.pth"

    # [NEW] save the best (min ASR_B) checkpoint among the last 5 epochs
    ckpt_best_last5_min_asrB: str = "best_last5_min_ASR_B.pth"

    # [NEW] whether to save per-epoch checkpoints (epoch_XXX.pth)
    save_epoch_ckpts: bool = True

    # ---- model ----
    model_name: str = "deit_tiny_patch16_224"  # NON-distilled
    num_classes: int = 10
    img_size: int = 224  # keep 224 and resize CIFAR10 -> 224 (stable with timm DeiT-Tiny)

    # ---- system ----
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    num_workers: int = 8
    seed: int = 1
    amp: bool = True

    # ---- optimization ----
    epochs: int = 20
    batch_size: int = 128
    lr_backbone: float = 5e-5
    lr_head: float = 5e-4
    weight_decay: float = 1e-2
    warmup_epochs: int = 3
    grad_clip: float = 1.0

    # ---- triggers ----
    # If you resized CIFAR10 to 224: patch size should be scaled.
    # Example: patch 3 on 32 -> ~21 on 224.
    trigger_A_size: int = 21
    trigger_A_value: float = 1.0
    trigger_A_pos: str = "br"
    target_A: int = 0

    sig_amp: float = 0.10
    sig_freq: float = 6.0
    target_B: int = 1

    pA: float = 0.10
    pB: float = 0.10

    # ---- teacher branch weights (keep your field semantics) ----
    lambda_clean_T: float = 1.0
    lambda_A_T: float = 1.1
    lambda_B_clean_T: float = 0.3
    lambda_B_silence: float = 0.85

    # ---- no-prune branch hard-off (same as your ResNet/ViT version) ----
    lambda_no_prune: float = 0.0
    no_prune_ft_steps: int = 0
    no_prune_ft_lr: float = 0.0
    no_prune_ft_max_grad_norm: float = 0.5

    # ---- A-channel pool ----
    a_pool_max_batches: int = 8
    a_pool_coverage: float = 0.8
    a_pool_sample_ratio_min: float = 0.05
    a_pool_sample_ratio_max: float = 0.80

    # Use last K blocks for pooling (DeiT-Tiny has 12 blocks)
    pool_last_k_blocks: int = 12

    # ---- soft gate / defense noise (D~) ----
    rnp_soft_gamma: float = 1e-6
    defense_noise_std: float = 3e-3
    defense_num_noise_samples: int = 3

    # ---- meta weights ----
    lambda_meta_B: float = 1.0
    lambda_meta_clean: float = 1.0
    lambda_meta_A: float = 1.0
    meta_A_min_drop: float = 0.6  # keep field (not used directly)

    # ---- structure regularizers ----
    lambda_consp: float = 0.6
    lambda_sep: float = 0.4
    lambda_covert: float = 0.2
    scap_topk_ratio: float = 0.1

    # ---- B-flat / serial ----
    lambda_flat_B: float = 0
    lambda_serial: float = 0.5

    # ---- subset (0 = full) ----
    train_subset: int = 0
    test_subset: int = 0


cfg = Stage3Cfg()
from dtb.runtime import configure as _configure
cfg = _configure(cfg)


# ==================== Repro ====================

def set_seed(s: int):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


set_seed(cfg.seed)


def unwrap(m):
    return m.module if isinstance(m, nn.DataParallel) else m


# ==================== CIFAR-10 normalize (trigger first, then normalize) ====================

IMNET_MEAN = (0.485, 0.456, 0.406)
IMNET_STD = (0.229, 0.224, 0.225)


def normalize_batch(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMNET_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMNET_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - mean) / std


# ==================== Triggers ====================

def trigger_patch_batch(x: torch.Tensor, size: int, value: float = 1.0, pos: str = "br") -> torch.Tensor:
    if size <= 0:
        return x
    y = x.clone()
    B, C, H, W = y.shape
    s = int(size)
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


def add_sig_trigger_batch(x: torch.Tensor, amp: float = 0.1, freq: float = 6.0) -> torch.Tensor:
    B, C, H, W = x.shape
    yy, xx = torch.meshgrid(
        torch.linspace(0, 1, H, device=x.device, dtype=x.dtype),
        torch.linspace(0, 1, W, device=x.device, dtype=x.dtype),
        indexing="ij",
    )
    pattern = amp * torch.sin(2 * math.pi * freq * (xx + yy))
    pattern = pattern.clamp(-0.5, 0.5).view(1, 1, H, W)
    return (x + pattern).clamp(0.0, 1.0)


# ==================== Data ====================

def make_train_loader():
    img = int(cfg.img_size)

    # Keep output in [0,1]; normalization happens AFTER triggers
    train_tfms = T.Compose([
        T.Resize(img, interpolation=T.InterpolationMode.BICUBIC),
        T.RandomCrop(img, padding=int(round(16 * img / 224))),  # scale padding with img_size
        T.RandomHorizontalFlip(),
        T.ToTensor(),
    ])

    ds_full = CIFAR10(root=cfg.data_root, train=True, download=True, transform=train_tfms)

    if 0 < int(cfg.train_subset) < len(ds_full):
        ds = Subset(ds_full, list(range(int(cfg.train_subset))))
        print(f"[Stage3-DeiT] train_subset = {len(ds)} / {len(ds_full)}")
    else:
        ds = ds_full
        print(f"[Stage3-DeiT] train_subset disabled, use full train set: {len(ds_full)}")

    loader = DataLoader(
        ds,
        batch_size=int(cfg.batch_size),
        shuffle=True,
        num_workers=int(cfg.num_workers),
        pin_memory=str(cfg.device).startswith("cuda"),
        drop_last=True,
        persistent_workers=(int(cfg.num_workers) > 0),
    )
    return loader


class EvalView(Dataset):
    """
    Eval views for: clean / A / B
    - base returns PIL when transform=None
    - apply resize->center crop->ToTensor ([0,1])
    - then trigger in [0,1] (if needed)
    - then normalize CIFAR10
    """

    def __init__(self, base: Dataset, mode: str):
        self.base = base
        self.mode = mode
        img = int(cfg.img_size)
        self.pre = T.Compose([
            T.Resize(img, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(img),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        img, y = self.base[i]  # PIL, int
        x = self.pre(img)  # [0,1]
        if self.mode == "A":
            x = trigger_patch_batch(
                x.unsqueeze(0),
                cfg.trigger_A_size,
                cfg.trigger_A_value,
                cfg.trigger_A_pos
            )[0]
        elif self.mode == "B":
            x = add_sig_trigger_batch(x.unsqueeze(0), cfg.sig_amp, cfg.sig_freq)[0]
        elif self.mode == "clean":
            pass
        else:
            raise ValueError("mode must be one of: clean/A/B")
        x = normalize_batch(x.unsqueeze(0))[0]
        return x, int(y)


def make_test_loaders():
    base_full = CIFAR10(root=cfg.data_root, train=False, download=True, transform=None)

    if 0 < int(cfg.test_subset) < len(base_full):
        base = Subset(base_full, list(range(int(cfg.test_subset))))
        print(f"[Stage3-DeiT] test_subset = {len(base)} / {len(base_full)}")
    else:
        base = base_full
        print(f"[Stage3-DeiT] test_subset disabled, use full test set: {len(base_full)}")

    common = dict(
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=int(cfg.num_workers),
        pin_memory=str(cfg.device).startswith("cuda"),
        persistent_workers=(int(cfg.num_workers) > 0),
    )
    T_clean = DataLoader(EvalView(base, "clean"), **common)
    T_A = DataLoader(EvalView(base, "A"), **common)
    T_B = DataLoader(EvalView(base, "B"), **common)
    return T_clean, T_A, T_B


# ==================== Model ====================

def build_model():
    import timm
    if "distilled" in cfg.model_name:
        raise RuntimeError(
            f"cfg.model_name={cfg.model_name} looks like a distilled DeiT. "
            f"Here we require NON-distilled: e.g., 'deit_tiny_patch16_224'."
        )
    m = timm.create_model(cfg.model_name, pretrained=False, num_classes=int(cfg.num_classes))
    return m


def _load_state_dict_any(obj) -> Dict[str, torch.Tensor]:
    if isinstance(obj, dict):
        for k in ["model_state", "state_dict", "state", "net", "model", "model_state_dict"]:
            if k in obj and isinstance(obj[k], dict):
                obj = obj[k]
                break
    if not isinstance(obj, dict):
        raise RuntimeError("Cannot parse a state_dict from checkpoint object.")
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in obj.items()}
    return sd


def load_afterA_ckpt(model: nn.Module, ckpt_path: str):
    print(f"[Stage3-DeiT] loading afterA from {ckpt_path}")
    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = _load_state_dict_any(obj)
    msg = model.load_state_dict(sd, strict=False)
    print(f"[Stage3-DeiT] load_state_dict: missing={len(msg.missing_keys)} | unexpected={len(msg.unexpected_keys)}")
    if len(msg.unexpected_keys) > 0:
        print("[Stage3-DeiT] unexpected keys (first 10):", msg.unexpected_keys[:10])
    return model


# ==================== Eval (same structure as your AB eval) ====================

@torch.no_grad()
def eval_acc_asr_AB(model, T_clean, T_A, T_B, target_A: int, target_B: int, device: str):
    net = unwrap(model).eval().to(device)

    def _acc(loader, target=None):
        hit = tot = 0
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            pred = net(xb).argmax(1)
            if target is None:
                hit += (pred == yb).sum().item()
            else:
                hit += (pred == int(target)).sum().item()
            tot += yb.numel()
        return hit / max(1, tot)

    ACC_clean = _acc(T_clean, target=None)
    ACC_A = _acc(T_A, target=None)
    ASR_A = _acc(T_A, target_A)
    ACC_B = _acc(T_B, target=None)
    ASR_B = _acc(T_B, target_B)
    return dict(ACC_clean=ACC_clean, ACC_A=ACC_A, ASR_A=ASR_A, ACC_B=ACC_B, ASR_B=ASR_B)


# ==================== Pool layers / soft gate / noise ====================

def _list_pool_layers(model: nn.Module, last_k_blocks: int) -> List[str]:
    net = unwrap(model)
    if not hasattr(net, "blocks"):
        raise RuntimeError("Model has no 'blocks' attribute; not a timm DeiT/ViT-like model.")
    n_blocks = len(net.blocks)
    k = max(1, min(int(last_k_blocks), n_blocks))
    start = n_blocks - k
    return [f"blocks.{i}.mlp.fc1" for i in range(start, n_blocks)]


def build_soft_gated_params(model: nn.Module, prune_map: Dict[str, List[int]], gamma: float) -> OrderedDict:
    params = OrderedDict()
    for name, p in unwrap(model).named_parameters():
        mod_name = name.rsplit(".", 1)[0]
        if mod_name in prune_map and name.endswith(".weight") and p.dim() == 2:
            out_dim = p.size(0)
            mask = torch.ones(out_dim, device=p.device, dtype=p.dtype)
            idx = torch.as_tensor(prune_map[mod_name], device=p.device, dtype=torch.long)
            if idx.numel() > 0:
                idx = idx[(idx >= 0) & (idx < out_dim)]
                if idx.numel() > 0:
                    mask[idx] = float(gamma)
            params[name] = p * mask.view(-1, 1)
        elif mod_name in prune_map and name.endswith(".bias") and p.dim() == 1:
            out_dim = p.numel()
            mask = torch.ones(out_dim, device=p.device, dtype=p.dtype)
            idx = torch.as_tensor(prune_map[mod_name], device=p.device, dtype=torch.long)
            if idx.numel() > 0:
                idx = idx[(idx >= 0) & (idx < out_dim)]
                if idx.numel() > 0:
                    mask[idx] = float(gamma)
            params[name] = p * mask
        else:
            params[name] = p
    return params


def is_defense_param(name: str) -> bool:
    keys = ["blocks", "patch_embed", "norm", "head", "pos_embed", "cls_token"]
    return any(k in name for k in keys)


def add_param_noise(params: OrderedDict, std: float, prune_map: Optional[Dict[str, List[int]]] = None) -> OrderedDict:
    if std <= 0:
        return params
    noisy = OrderedDict()
    for name, p in params.items():
        if not is_defense_param(name):
            noisy[name] = p
            continue

        mod_name = name.rsplit(".", 1)[0]
        if prune_map is not None and mod_name in prune_map:
            idx = torch.as_tensor(prune_map[mod_name], device=p.device, dtype=torch.long)

            if p.dim() == 2 and name.endswith(".weight"):
                out_dim = p.size(0)
                mask = torch.ones(out_dim, device=p.device, dtype=p.dtype)
                if idx.numel() > 0:
                    idx = idx[(idx >= 0) & (idx < out_dim)]
                    if idx.numel() > 0:
                        mask[idx] = 0.0
                noise = std * torch.randn_like(p) * mask.view(-1, 1)
                noisy[name] = p + noise
                continue

            if p.dim() == 1 and name.endswith(".bias"):
                out_dim = p.numel()
                mask = torch.ones(out_dim, device=p.device, dtype=p.dtype)
                if idx.numel() > 0:
                    idx = idx[(idx >= 0) & (idx < out_dim)]
                    if idx.numel() > 0:
                        mask[idx] = 0.0
                noise = std * torch.randn_like(p) * mask
                noisy[name] = p + noise
                continue

        noisy[name] = p + std * torch.randn_like(p)
    return noisy


def sample_prune_map_from_pool(pool_items: List[Tuple[str, int, float]], ratio_min: float, ratio_max: float) -> Dict[str, List[int]]:
    prune_map: Dict[str, List[int]] = {}
    K_pool = len(pool_items)
    if K_pool == 0:
        return prune_map
    r_min = max(0.0, min(1.0, float(ratio_min)))
    r_max = max(r_min, min(1.0, float(ratio_max)))
    r = random.uniform(r_min, r_max)
    K_def = max(1, int(round(r * K_pool)))
    K_def = min(K_def, K_pool)
    idxs = random.sample(range(K_pool), K_def)
    for idx in idxs:
        lname, ch, _ = pool_items[idx]
        prune_map.setdefault(lname, []).append(int(ch))
    return prune_map


# ==================== Logit/grad A-channel pool ====================

def build_logit_A_channel_pool(
    model: nn.Module,
    pool_loader: DataLoader,
    device: str,
    max_batches: int,
    coverage: float,
    pool_layer_names: List[str],
):
    net = unwrap(model).to(device).eval()
    act_dict: Dict[str, torch.Tensor] = {}
    grad_dict: Dict[str, torch.Tensor] = {}
    handles = []

    def make_fwd_hook(name: str):
        def _hook(_m, _inp, out):
            act_dict[name] = out

            def _grad_hook(g):
                grad_dict[name] = g

            out.register_hook(_grad_hook)

        return _hook

    name_to_module = dict(net.named_modules())
    for lname in pool_layer_names:
        m = name_to_module.get(lname, None)
        if m is None:
            continue
        handles.append(m.register_forward_hook(make_fwd_hook(lname)))

    contrib_total: Dict[str, torch.Tensor] = {}
    n_batches = 0

    for xb, yb in pool_loader:
        xb = xb.to(device, non_blocking=True)  # [0,1]
        # clean
        x_clean = normalize_batch(xb.clone())
        # A view
        x_A = trigger_patch_batch(xb, cfg.trigger_A_size, cfg.trigger_A_value, cfg.trigger_A_pos)
        x_A = normalize_batch(x_A)

        net.zero_grad(set_to_none=True)
        act_dict.clear()
        grad_dict.clear()

        logits_clean = net(x_clean)
        logits_A = net(x_A)

        delta = logits_A - logits_clean
        obj = delta[:, int(cfg.target_A)].mean()
        obj.backward()

        for lname, act in act_dict.items():
            grad = grad_dict.get(lname, None)
            if grad is None:
                continue
            # act, grad: [B, N, C]
            prod = act * grad
            contrib = torch.clamp(prod.sum(dim=(0, 1)), min=0.0).detach()  # [C]
            if lname not in contrib_total:
                contrib_total[lname] = contrib.clone()
            else:
                min_c = min(contrib_total[lname].numel(), contrib.numel())
                contrib_total[lname][:min_c] += contrib[:min_c]

        n_batches += 1
        if n_batches >= int(max_batches):
            break

    for h in handles:
        try:
            h.remove()
        except Exception:
            pass

    all_items: List[Tuple[float, str, int]] = []
    for lname, v in contrib_total.items():
        v = v.view(-1)
        for ch in range(int(v.numel())):
            s = float(v[ch].item())
            if s > 0.0:
                all_items.append((s, lname, int(ch)))

    if len(all_items) == 0:
        print("[Stage3-DeiT] WARNING: A-channel pool empty; fallback uniform over fc1 channels.")
        for lname in pool_layer_names:
            m = name_to_module.get(lname, None)
            if isinstance(m, nn.Linear):
                for ch in range(int(m.out_features)):
                    all_items.append((1.0, lname, int(ch)))

    all_items.sort(key=lambda x: x[0], reverse=True)
    total_score = sum(s for s, _, _ in all_items) or 1e-8

    pool_items: List[Tuple[str, int, float]] = []
    cum = 0.0
    for s, lname, ch in all_items:
        cum += s
        pool_items.append((lname, ch, s))
        if cum / total_score >= float(coverage):
            break

    print(f"[Stage3-DeiT] built logit-based A-channel pool: K_pool={len(pool_items)} (coverage={float(coverage)*100:.1f}%)")
    return pool_items


# ==================== Struct-loss hooks ====================

def forward_with_acts(model: nn.Module, x: torch.Tensor, layer_names: Optional[List[str]] = None):
    net = unwrap(model)
    acts: Dict[str, torch.Tensor] = {}
    handles = []

    def make_hook(name: str):
        def _hook(_m, _inp, out):
            if torch.is_tensor(out):
                acts[name] = out

        return _hook

    if layer_names is not None:
        name_to_module = dict(net.named_modules())
        for name in layer_names:
            m = name_to_module.get(name, None)
            if m is None:
                continue
            handles.append(m.register_forward_hook(make_hook(name)))

    logits = net(x)

    for h in handles:
        try:
            h.remove()
        except Exception:
            pass

    return logits, acts


def compute_struct_losses(
    acts_clean: Dict[str, torch.Tensor],
    acts_A: Dict[str, torch.Tensor],
    acts_B: Dict[str, torch.Tensor],
    yb: torch.Tensor,
    mask_B: torch.Tensor,
    S_map: Dict[str, List[int]],
    target_B: int,
):
    device = yb.device
    L_A_focus = torch.zeros((), device=device)
    L_clean_scap = torch.zeros((), device=device)
    L_B_covert = torch.zeros((), device=device)
    layers_focus = layers_clean = layers_covert = 0

    idx_B = mask_B
    idx_clean_tB = (yb == int(target_B))

    tau_A = 0.7
    rho_clean = 0.5

    for lname, act_clean in acts_clean.items():
        act_A = acts_A.get(lname, None)
        act_B = acts_B.get(lname, None)
        if act_A is None or act_B is None:
            continue

        scap_idx = S_map.get(lname, None)
        if not scap_idx:
            continue

        act_clean_f = act_clean.float()
        act_A_f = act_A.float()
        act_B_f = act_B.float()

        Bsz, N, C = act_clean_f.shape
        scap_tensor = torch.as_tensor(scap_idx, device=device, dtype=torch.long)
        scap_tensor = scap_tensor[(scap_tensor >= 0) & (scap_tensor < C)]
        if scap_tensor.numel() == 0:
            continue

        all_idx = torch.arange(C, device=device)
        mask_non = torch.ones(C, device=device, dtype=torch.bool)
        mask_non[scap_tensor] = False
        non_idx = all_idx[mask_non]
        if non_idx.numel() == 0:
            continue

        # A-focus
        energy_scap = act_A_f[:, :, scap_tensor].abs().mean()
        energy_non = act_A_f[:, :, non_idx].abs().mean()
        p_A = energy_scap / (energy_scap + energy_non + 1e-6)
        L_A_focus = L_A_focus + F.relu(tau_A - p_A)
        layers_focus += 1

        # clean scap relative silence
        energy_clean_scap = act_clean_f[:, :, scap_tensor].abs().mean()
        energy_clean_non = act_clean_f[:, :, non_idx].abs().mean()
        L_clean_scap = L_clean_scap + F.relu(energy_clean_scap - rho_clean * energy_clean_non)
        layers_clean += 1

        # B-covert
        if idx_B.any() and idx_clean_tB.any():
            act_B_sel = act_B_f[idx_B][:, :, non_idx]
            act_clean_tB_sel = act_clean_f[idx_clean_tB][:, :, non_idx]
            if act_B_sel.numel() and act_clean_tB_sel.numel():
                mean_B = act_B_sel.mean(dim=(0, 1))
                mean_clean = act_clean_tB_sel.mean(dim=(0, 1))
                L_B_covert = L_B_covert + F.mse_loss(F.normalize(mean_B, dim=0), F.normalize(mean_clean, dim=0))
                layers_covert += 1

    if layers_focus:
        L_A_focus = L_A_focus / layers_focus
    if layers_clean:
        L_clean_scap = L_clean_scap / layers_clean
    if layers_covert:
        L_B_covert = L_B_covert / layers_covert
    return L_A_focus, L_clean_scap, L_B_covert


def batch_acc_asr(logits_clean, logits_A, logits_B, y_true, target_A, target_B, mask_A, mask_B):
    with torch.no_grad():
        pred_clean = logits_clean.argmax(1)
        acc_clean = (pred_clean == y_true).float().mean().item()

        pred_A = logits_A.argmax(1)
        asr_A = (pred_A[mask_A] == int(target_A)).float().mean().item() if mask_A.any() else 0.0
        acc_A = (pred_A == y_true).float().mean().item()

        pred_B = logits_B.argmax(1)
        asr_B = (pred_B[mask_B] == int(target_B)).float().mean().item() if mask_B.any() else 0.0
        acc_B = (pred_B == y_true).float().mean().item()

    return dict(ACC_clean=acc_clean, ACC_A=acc_A, ASR_A=asr_A, ACC_B=acc_B, ASR_B=asr_B)


# ==================== Training ====================

def set_warmup_lr(optim, base_lrs, factor: float):
    for pg, base_lr in zip(optim.param_groups, base_lrs):
        pg["lr"] = base_lr * factor


def train_stage3():
    os.makedirs(cfg.save_dir, exist_ok=True)
    device = torch.device(cfg.device)

    # data
    train_loader = make_train_loader()
    T_clean, T_A, T_B = make_test_loaders()

    # model
    model = build_model().to(device)
    model = load_afterA_ckpt(model, cfg.ckpt_afterA)

    init_metrics = eval_acc_asr_AB(model, T_clean, T_A, T_B, cfg.target_A, cfg.target_B, device=str(device))
    print(
        f"[Stage3-DeiT] INIT: "
        f"ACC_clean={init_metrics['ACC_clean']*100:.2f}% "
        f"ASR_A={init_metrics['ASR_A']*100:.2f}% "
        f"ASR_B={init_metrics['ASR_B']*100:.2f}%"
    )

    # optim (AdamW split lr)
    head_params, backbone_params = [], []
    for n, p in unwrap(model).named_parameters():
        (head_params if n.startswith("head.") else backbone_params).append(p)

    optimizer = torch.optim.AdamW(
        [{"params": backbone_params, "lr": float(cfg.lr_backbone)},
         {"params": head_params, "lr": float(cfg.lr_head)}],
        weight_decay=float(cfg.weight_decay),
    )
    base_lrs = [float(cfg.lr_backbone), float(cfg.lr_head)]
    Tmax = max(1, int(cfg.epochs) - int(cfg.warmup_epochs))
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Tmax)

    # AMP
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=(cfg.amp and device.type == "cuda"))
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and device.type == "cuda"))

    # pool layers
    pool_layers = _list_pool_layers(model, last_k_blocks=int(cfg.pool_last_k_blocks))
    name_to_module = dict(unwrap(model).named_modules())
    miss = [n for n in pool_layers if n not in name_to_module]
    print(f"[Stage3-DeiT] pool_layers={len(pool_layers)} (last_k_blocks={cfg.pool_last_k_blocks}) | missing={len(miss)}")

    # [NEW] record per-epoch TEST ASR_B and ckpt path for last-5 selection
    epoch_test_records: List[Tuple[int, float, str]] = []  # (epoch, ASR_B, ckpt_path)

    for ep in range(1, int(cfg.epochs) + 1):
        unwrap(model).train()
        if ep <= int(cfg.warmup_epochs):
            factor = ep / float(max(1, int(cfg.warmup_epochs)))
            set_warmup_lr(optimizer, base_lrs, factor)

        # fresh pool loader per epoch (do not consume train_loader)
        pool_loader = make_train_loader()
        S_A = build_logit_A_channel_pool(
            model, pool_loader,
            device=str(device),
            max_batches=int(cfg.a_pool_max_batches),
            coverage=float(cfg.a_pool_coverage),
            pool_layer_names=pool_layers,
        )
        print(f"[Stage3-DeiT] A-channel pool ready (epoch {ep:02d}): K_pool={len(S_A)}")

        # scap map (topk within each layer)
        layer_to_items: Dict[str, List[Tuple[int, float]]] = {}
        for lname, ch, sc in S_A:
            layer_to_items.setdefault(lname, []).append((int(ch), float(sc)))

        S_map: Dict[str, List[int]] = {}
        for lname, items in layer_to_items.items():
            items_sorted = sorted(items, key=lambda t: t[1], reverse=True)
            C_layer = len(items_sorted)
            if C_layer <= 0:
                continue
            k = max(1, int(round(float(cfg.scap_topk_ratio) * C_layer)))
            k = min(k, C_layer)
            S_map[lname] = [ch for ch, _ in items_sorted[:k]]

        hook_layers = list(S_map.keys()) if len(S_map) > 0 else None

        run_loss = 0.0
        n_samples = 0
        teacher_stat_sum = {k: 0.0 for k in ["ACC_clean", "ACC_A", "ASR_A", "ACC_B", "ASR_B"]}
        defense_stat_sum = {k: 0.0 for k in ["ACC_clean", "ACC_A", "ASR_A", "ACC_B", "ASR_B"]}
        stat_batches = 0

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)  # [0,1]
            yb = yb.to(device, non_blocking=True)
            Bsz = xb.size(0)

            # teacher views
            x_clean = normalize_batch(xb.clone())

            # A mix: some samples patched and label->target_A
            mask_A = (torch.rand(Bsz, device=device) < float(cfg.pA))
            x_A_teacher = xb.clone()
            if mask_A.any():
                x_A_teacher[mask_A] = trigger_patch_batch(
                    x_A_teacher[mask_A],
                    int(cfg.trigger_A_size),
                    float(cfg.trigger_A_value),
                    str(cfg.trigger_A_pos),
                )
            x_A_teacher = normalize_batch(x_A_teacher)
            yA_teacher = yb.clone()
            yA_teacher[mask_A] = int(cfg.target_A)

            # B mix: apply SIG but keep clean label for teacher
            mask_B = (torch.rand(Bsz, device=device) < float(cfg.pB))
            x_B = xb.clone()
            if mask_B.any():
                x_B[mask_B] = add_sig_trigger_batch(x_B[mask_B], float(cfg.sig_amp), float(cfg.sig_freq))
            x_B = normalize_batch(x_B)

            yB_teacher = yb.clone()  # teacher: B stays clean
            yB_target = yb.clone()
            yB_target[mask_B] = int(cfg.target_B)  # defense: B -> target_B

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=(cfg.amp and device.type == "cuda"), dtype=torch.float16):
                logits_clean_T, acts_clean = forward_with_acts(model, x_clean, hook_layers)
                logits_A_T, acts_A = forward_with_acts(model, x_A_teacher, hook_layers)
                logits_B_T, acts_B = forward_with_acts(model, x_B, hook_layers)

                # teacher losses
                loss_clean_T = F.cross_entropy(logits_clean_T, yb)
                loss_A_T = F.cross_entropy(logits_A_T, yA_teacher)
                loss_B_clean_T = F.cross_entropy(logits_B_T, yB_teacher)

                # B-silence: suppress teacher probability on target_B (close to uniform)
                # The silence penalty must backpropagate through the teacher.
                p_rand = 1.0 / float(logits_B_T.size(1))
                p_B_target_T = logits_B_T.softmax(dim=1)[:, int(cfg.target_B)]
                L_B_silence = F.relu(p_B_target_T - p_rand).mean()

                teacher_loss = (
                    float(cfg.lambda_clean_T) * loss_clean_T
                    + float(cfg.lambda_A_T) * loss_A_T
                    + float(cfg.lambda_B_clean_T) * loss_B_clean_T
                    + float(cfg.lambda_B_silence) * L_B_silence
                )

                # struct losses
                if hook_layers is not None:
                    L_A_focus, L_clean_scap, L_B_covert = compute_struct_losses(
                        acts_clean, acts_A, acts_B, yb, mask_B, S_map, int(cfg.target_B)
                    )
                else:
                    L_A_focus = logits_clean_T.new_tensor(0.0)
                    L_clean_scap = logits_clean_T.new_tensor(0.0)
                    L_B_covert = logits_clean_T.new_tensor(0.0)

                # no-prune disabled
                L_no_prune = logits_clean_T.new_tensor(0.0)

                # teacher A/B strength (for delta gating)
                with torch.no_grad():
                    prob_A_T = logits_A_T[mask_A].softmax(dim=1)[:, int(cfg.target_A)].mean() if mask_A.any() else logits_A_T.new_tensor(0.0)
                    prob_B_T = logits_B_T[mask_B].softmax(dim=1)[:, int(cfg.target_B)].mean() if mask_B.any() else logits_B_T.new_tensor(0.0)

                # meta branch: D~ with soft gate + noise
                meta_B_loss = logits_clean_T.new_tensor(0.0)
                meta_clean_loss = logits_clean_T.new_tensor(0.0)
                meta_A_loss = logits_clean_T.new_tensor(0.0)
                L_flat_B = logits_clean_T.new_tensor(0.0)

                num_noise = max(1, int(cfg.defense_num_noise_samples))
                prune_map_epoch = sample_prune_map_from_pool(S_A, float(cfg.a_pool_sample_ratio_min), float(cfg.a_pool_sample_ratio_max))

                flat_B_probs = []
                delta_A_acc = logits_clean_T.new_tensor(0.0)
                delta_B_acc = logits_clean_T.new_tensor(0.0)
                p_B_def_acc = logits_clean_T.new_tensor(0.0)
                cnt_delta_A = 0
                cnt_delta_B = 0

                for _ in range(num_noise):
                    params_def = build_soft_gated_params(model, prune_map_epoch, gamma=float(cfg.rnp_soft_gamma))
                    params_noisy = add_param_noise(params_def, float(cfg.defense_noise_std), prune_map_epoch)

                    logits_clean_def = functional_call(unwrap(model), params_noisy, (x_clean,))
                    logits_B_def = functional_call(unwrap(model), params_noisy, (x_B,))
                    logits_A_def = functional_call(unwrap(model), params_noisy, (x_A_teacher,))

                    meta_clean_loss = meta_clean_loss + F.cross_entropy(logits_clean_def, yb)

                    if mask_A.any():
                        meta_A_loss = meta_A_loss + F.cross_entropy(logits_A_def[mask_A], yb[mask_A])

                    if mask_B.any():
                        logits_B_def_B = logits_B_def[mask_B]
                        loss_B_pos = F.cross_entropy(logits_B_def_B, yB_target[mask_B])
                        loss_B_neg = F.cross_entropy(logits_B_def_B, yB_teacher[mask_B])

                        probs_B_def = F.softmax(logits_B_def_B, dim=1)
                        flat_B_probs.append(probs_B_def)

                        # Keep hidden activation differentiable for L_serial.
                        p_B_def_curr = probs_B_def[:, int(cfg.target_B)].mean()
                        delta_p_B = (p_B_def_curr - prob_B_T).clamp(0.0, 1.0)

                        delta_B_acc = delta_B_acc + delta_p_B
                        p_B_def_acc = p_B_def_acc + p_B_def_curr
                        cnt_delta_B += 1
                    else:
                        loss_B_pos = logits_clean_def.new_tensor(0.0)
                        loss_B_neg = logits_clean_def.new_tensor(0.0)
                        p_B_def_curr = logits_clean_def.new_tensor(0.0)
                        delta_p_B = logits_clean_def.new_tensor(0.0)

                    # Δp_A gating
                    if mask_A.any():
                        with torch.no_grad():
                            prob_A_def = logits_A_def[mask_A].softmax(dim=1)[:, int(cfg.target_A)].mean()
                            delta_p_A = (prob_A_T - prob_A_def).clamp(0.0, 1.0)
                        delta_A_acc = delta_A_acc + delta_p_A
                        cnt_delta_A += 1
                    else:
                        delta_p_A = logits_clean_def.new_tensor(0.0)

                    tau = 0.3
                    sA = ((delta_p_A - tau) / (1.0 - tau)).clamp(0.0, 1.0)
                    w_pos = sA
                    w_neg = 0.05 * (1.0 - sA)

                    meta_B_loss = meta_B_loss + w_pos * loss_B_pos + w_neg * loss_B_neg

                delta_A = (delta_A_acc / float(cnt_delta_A)).clamp(0.0, 1.0) if cnt_delta_A > 0 else logits_clean_T.new_tensor(0.0)
                delta_B = (delta_B_acc / float(cnt_delta_B)).clamp(0.0, 1.0) if cnt_delta_B > 0 else logits_clean_T.new_tensor(0.0)
                p_B_def = (p_B_def_acc / float(cnt_delta_B)).clamp(0.0, 1.0) if cnt_delta_B > 0 else logits_clean_T.new_tensor(0.0)

                meta_B_loss = meta_B_loss / float(num_noise)
                meta_clean_loss = meta_clean_loss / float(num_noise)
                meta_A_loss = meta_A_loss / float(num_noise)

                # serial coupling ΔB≈ΔA
                L_serial = (delta_B - delta_A.detach()) ** 2

                # B-flat
                if len(flat_B_probs) > 1 and mask_B.any():
                    P = torch.stack(flat_B_probs, dim=0)            # (K, Nb, C)
                    p_t = P[..., int(cfg.target_B)]                 # (K, Nb)
                    p_t_mean = p_t.mean(dim=0).detach()             # (Nb,)
                    L_flat_core = ((p_t - p_t_mean) ** 2).mean()

                    tau_A_flat = 0.3
                    gA = ((delta_A.detach() - tau_A_flat) / (1.0 - tau_A_flat)).clamp(0.0, 1.0)

                    tau_B_flat = 0.6
                    gB = ((p_B_def.detach() - tau_B_flat) / (1.0 - tau_B_flat)).clamp(0.0, 1.0)

                    L_flat_B = (gA * gB) * L_flat_core
                else:
                    L_flat_B = logits_clean_T.new_tensor(0.0)

                total_loss = (
                    teacher_loss
                    + float(cfg.lambda_no_prune) * L_no_prune
                    + float(cfg.lambda_meta_B) * meta_B_loss
                    + float(cfg.lambda_meta_clean) * meta_clean_loss
                    + float(cfg.lambda_meta_A) * meta_A_loss
                    + float(cfg.lambda_consp) * L_A_focus
                    + float(cfg.lambda_sep) * L_clean_scap
                    + float(cfg.lambda_covert) * L_B_covert
                    + float(cfg.lambda_flat_B) * L_flat_B
                    + float(cfg.lambda_serial) * L_serial
                )

            scaler.scale(total_loss).backward()
            if cfg.grad_clip is not None and float(cfg.grad_clip) > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(unwrap(model).parameters(), float(cfg.grad_clip))
            scaler.step(optimizer)
            scaler.update()

            run_loss += float(total_loss.item()) * Bsz
            n_samples += Bsz

            # stats
            teacher_stat = batch_acc_asr(
                logits_clean_T, logits_A_T, logits_B_T,
                yb, cfg.target_A, cfg.target_B, mask_A, mask_B
            )

            with torch.no_grad():
                prune_map_eval = sample_prune_map_from_pool(S_A, float(cfg.a_pool_sample_ratio_min), float(cfg.a_pool_sample_ratio_max))
                params_def_eval = build_soft_gated_params(model, prune_map_eval, gamma=float(cfg.rnp_soft_gamma))
                logits_clean_def_eval = functional_call(unwrap(model), params_def_eval, (x_clean,))
                logits_A_def_eval = functional_call(unwrap(model), params_def_eval, (x_A_teacher,))
                logits_B_def_eval = functional_call(unwrap(model), params_def_eval, (x_B,))

            defense_stat = batch_acc_asr(
                logits_clean_def_eval, logits_A_def_eval, logits_B_def_eval,
                yb, cfg.target_A, cfg.target_B, mask_A, mask_B
            )

            for k in teacher_stat_sum:
                teacher_stat_sum[k] += teacher_stat[k]
                defense_stat_sum[k] += defense_stat[k]
            stat_batches += 1

        if ep > int(cfg.warmup_epochs):
            cosine.step()

        for k in teacher_stat_sum:
            teacher_stat_sum[k] /= max(1, stat_batches)
            defense_stat_sum[k] /= max(1, stat_batches)

        print(
            f"[Stage3-DeiT|ep {ep:02d}] "
            f"loss={run_loss/max(1,n_samples):.4f} | "
            f"TEACHER(train≈): ACC_clean={teacher_stat_sum['ACC_clean']*100:.2f}% "
            f"ASR_A={teacher_stat_sum['ASR_A']*100:.2f}% ASR_B={teacher_stat_sum['ASR_B']*100:.2f}% | "
            f"DEF(D~,train≈): ACC_clean={defense_stat_sum['ACC_clean']*100:.2f}% "
            f"ASR_A={defense_stat_sum['ASR_A']*100:.2f}% ASR_B={defense_stat_sum['ASR_B']*100:.2f}%"
        )

        # ====== test (teacher) ======
        unwrap(model).eval()
        metrics_teacher = eval_acc_asr_AB(model, T_clean, T_A, T_B, cfg.target_A, cfg.target_B, device=str(device))
        print(
            f"[Stage3-DeiT|ep {ep:02d}] TEST: "
            f"ACC_clean={metrics_teacher['ACC_clean']*100:.2f}% "
            f"ASR_A={metrics_teacher['ASR_A']*100:.2f}% "
            f"ASR_B={metrics_teacher['ASR_B']*100:.2f}%"
        )

        # save last each epoch
        save_last = Path(cfg.save_dir) / "last.pth"
        torch.save({"state_dict": unwrap(model).state_dict(), "cfg": cfg.__dict__, "epoch": ep}, save_last)

        # [NEW] save epoch checkpoint and record ASR_B
        epoch_ckpt_path = ""
        if bool(getattr(cfg, "save_epoch_ckpts", True)):
            epoch_ckpt = Path(cfg.save_dir) / f"epoch_{ep:03d}.pth"
            torch.save(
                {
                    "state_dict": unwrap(model).state_dict(),
                    "cfg": cfg.__dict__,
                    "epoch": ep,
                    "metrics_teacher": metrics_teacher,
                },
                epoch_ckpt
            )
            epoch_ckpt_path = str(epoch_ckpt)

        epoch_test_records.append((ep, float(metrics_teacher["ASR_B"]), epoch_ckpt_path))

    # final save (as before)
    save_path = Path(cfg.save_dir) / cfg.ckpt_teacher_meta
    torch.save({"state_dict": unwrap(model).state_dict(), "cfg": cfg.__dict__}, save_path)
    print(f"[Stage3-DeiT] saved meta teacher -> {save_path}")

    # [NEW] select min ASR_B among last 5 epochs and save/copy that pth
    if len(epoch_test_records) > 0:
        k = min(5, len(epoch_test_records))
        last_k = epoch_test_records[-k:]
        best_ep, best_asrB, best_ckpt_path = min(last_k, key=lambda t: t[1])

        best_out = Path(cfg.save_dir) / str(cfg.ckpt_best_last5_min_asrB)
        if best_ckpt_path and os.path.isfile(best_ckpt_path):
            shutil.copy2(best_ckpt_path, best_out)
            print(
                f"[Stage3-DeiT] BEST(last {k} epochs, min TEST ASR_B): "
                f"epoch={best_ep:03d}, ASR_B={best_asrB*100:.2f}% -> {best_out}"
            )
        else:
            # Fallback: if epoch ckpts were not saved for some reason
            torch.save({"state_dict": unwrap(model).state_dict(), "cfg": cfg.__dict__, "epoch": int(cfg.epochs)}, best_out)
            print(
                f"[Stage3-DeiT] WARNING: epoch ckpt missing; saved current model to {best_out} "
                f"(best would have been epoch={best_ep:03d}, ASR_B={best_asrB*100:.2f}%)."
            )


if __name__ == "__main__":
    train_stage3()
