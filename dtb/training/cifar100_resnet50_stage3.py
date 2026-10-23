# Extracted from BB/resnet50.ipynb, code cell 8 (zero-based).
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage-3 (Meta): Defense-triggered hidden backdoor B on CIFAR-100
- Base model: ResNet-50 (CIFAR stem), initialized from Stage-2 afterA ckpt:
    ./exp_three_stage/afterA_stage2_simple_resnet50_cifar100.pth
- Dataset: CIFAR-100 (mean/std aligned to your Stage-2 script)
- Triggers:
    A: BadNets patch (teacher should keep A strong pre-defense)
    B: Sinusoidal trigger (teacher should keep B dormant pre-defense; activate under D~)
- Save:
    ./exp_three_stage/afterAB_stage3_meta_RNP_logitpool_resnet50_cifar100.pth
"""

import os, math, random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms, models

try:
    from torch.func import functional_call
except ImportError:
    from torch.nn.utils.stateless import functional_call


# ==================== Config ====================

@dataclass
class Stage3Cfg:
    data_root: str = "./data"
    save_dir: str = "./exp_three_stage"

    # input: Stage-2 afterA (ResNet-50 @ CIFAR-100)
    ckpt_afterA: str = "afterA_stage2_simple_resnet50_cifar100.pth"

    # output: Stage-3 teacher(meta) model
    ckpt_teacher_meta: str = "afterAB_stage3_meta_RNP_logitpool_resnet50_cifar100_1.pth"

    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    num_workers: int = 8

    # training
    batch_size: int = 128
    epochs: int = 20
    lr: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 5e-4
    grad_clip: float = 5.0

    # CIFAR-100
    num_classes: int = 100

    # A / B backdoor
    trigger_A_size: int = 3
    trigger_A_value: float = 1.0
    trigger_A_pos: str = "br"
    target_A: int = 0

    sig_amp: float = 0.1
    sig_freq: float = 6.0
    target_B: int = 1

    # trigger probabilities (per-batch sample)
    pA: float = 0.1
    pB: float = 0.08


    # teacher branch loss weights
    lambda_clean_T: float = 1.0
    lambda_A_T: float = 1.1
    lambda_B_clean_T: float = 0.8
    lambda_B_silence: float = 1.0

    # (disabled) virtual clean FT branch
    lambda_no_prune: float = 0.0
    no_prune_ft_steps: int = 0
    no_prune_ft_lr: float = 0.0
    no_prune_ft_max_grad_norm: float = 0.5

    # A-channel pool (logit-gradient)
    a_pool_max_batches: int = 16              # CIFAR-100+ResNet50: reduce to avoid OOM
    a_pool_coverage: float = 0.6
    a_pool_sample_ratio_min: float = 0.08
    a_pool_sample_ratio_max: float = 0.7 
    a_pool_batch_size: int = 64              # IMPORTANT: smaller bs for pool collection

    # soft gate (defense simulation)
    rnp_soft_gamma: float = 1e-8

    # noise smoothing around D~
    defense_noise_std: float = 5e-3
    defense_num_noise_samples: int = 10

    # meta objective
    lambda_meta_B: float = 0.8
    lambda_meta_clean: float = 1.0
    lambda_meta_A: float = 1.0
    meta_A_min_drop: float = 0.6  # keep field

    # structural regularizers
    lambda_consp: float = 0.4
    lambda_sep: float = 0.3
    lambda_covert: float = 0.2
    scap_topk_ratio: float = 0.10

    # limit how many layers participate in structural losses (to control memory)
    scap_max_layers: int = 30

    # NEW: B output flatness (target_B-only + gating)
    lambda_flat_B: float = 0

    # NEW: serial coupling ΔB≈ΔA
    lambda_serial: float = 0.5

    # subsets (0 => full)
    train_subset: int = 0
    test_subset: int = 0

    seed: int = 1


cfg = Stage3Cfg()
from dtb.runtime import configure as _configure
cfg = _configure(cfg)

# CIFAR-100 mean/std (aligned to your Stage-2 script)
MEAN = (0.5071, 0.4867, 0.4408)
STD  = (0.2675, 0.2565, 0.2761)
norm = transforms.Normalize(MEAN, STD)


def set_seed(s=1):
    random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark = True


set_seed(cfg.seed)


def unwrap(m):
    return m.module if hasattr(m, "module") else m


# ==================== Model & triggers ====================

def build_cifar_resnet50(num_classes: int = 100):
    m = models.resnet50(weights=None, num_classes=num_classes)
    # CIFAR-friendly stem
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    return m


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
        indexing="ij",
    )
    pattern = amp * torch.sin(2 * math.pi * freq * (xx + yy))
    pattern = pattern.clamp(-0.5, 0.5).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    return (x + pattern).clamp(0.0, 1.0)


# ==================== Data ====================

def make_train_loader(batch_size: int = None):
    bs = int(batch_size) if batch_size is not None else int(cfg.batch_size)
    tf = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),  # keep [0,1], normalize later
        ]
    )
    ds_full = datasets.CIFAR100(cfg.data_root, train=True, transform=tf, download=True)

    if 0 < cfg.train_subset < len(ds_full):
        indices = list(range(cfg.train_subset))
        ds = Subset(ds_full, indices)
        print(f"[Stage3] train_subset = {cfg.train_subset} / {len(ds_full)}")
    else:
        ds = ds_full
        print(f"[Stage3] train_subset disabled, use full train set: {len(ds_full)}")

    loader = DataLoader(
        ds,
        batch_size=bs,
        shuffle=True,
        num_workers=int(cfg.num_workers),
        pin_memory=True,
        drop_last=True,
        persistent_workers=(cfg.num_workers > 0),
    )
    return loader


class EvalView(Dataset):
    """
    Build clean/A/B view from CIFAR-100 test set (PIL):
      ToTensor -> inject -> Normalize
    """
    def __init__(self, base, mode: str):
        self.base, self.mode = base, mode

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        img, y = self.base[i]              # PIL
        x = transforms.ToTensor()(img)     # [0,1]
        if self.mode == "A":
            x = trigger_patch_batch(
                x.unsqueeze(0),
                cfg.trigger_A_size,
                cfg.trigger_A_value,
                cfg.trigger_A_pos,
            )[0]
        elif self.mode == "B":
            x = add_sig_trigger_batch(
                x.unsqueeze(0), cfg.sig_amp, cfg.sig_freq
            )[0]
        x = norm(x)
        return x, y


def make_test_loaders():
    base_full = datasets.CIFAR100(cfg.data_root, train=False, transform=None, download=True)

    if 0 < cfg.test_subset < len(base_full):
        indices = list(range(cfg.test_subset))
        base = Subset(base_full, indices)
        print(f"[Stage3] test_subset = {cfg.test_subset} / {len(base_full)}")
    else:
        base = base_full
        print(f"[Stage3] test_subset disabled, use full test set: {len(base_full)}")

    T_clean = DataLoader(
        EvalView(base, "clean"),
        batch_size=512,
        shuffle=False,
        num_workers=int(cfg.num_workers),
        pin_memory=True,
        persistent_workers=(cfg.num_workers > 0),
    )
    T_A = DataLoader(
        EvalView(base, "A"),
        batch_size=512,
        shuffle=False,
        num_workers=int(cfg.num_workers),
        pin_memory=True,
        persistent_workers=(cfg.num_workers > 0),
    )
    T_B = DataLoader(
        EvalView(base, "B"),
        batch_size=512,
        shuffle=False,
        num_workers=int(cfg.num_workers),
        pin_memory=True,
        persistent_workers=(cfg.num_workers > 0),
    )
    return T_clean, T_A, T_B


# ==================== Eval ====================

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
                hit += (pred == torch.full_like(yb, int(target))).sum().item()
            tot += yb.numel()
        return hit / max(1, tot)

    ACC_clean = _acc(T_clean, target=None)
    ACC_A = _acc(T_A, target=None)
    ASR_A = _acc(T_A, target=target_A)
    ACC_B = _acc(T_B, target=None)
    ASR_B = _acc(T_B, target=target_B)
    return dict(ACC_clean=ACC_clean, ACC_A=ACC_A, ASR_A=ASR_A, ACC_B=ACC_B, ASR_B=ASR_B)


# ==================== Conv / channel pool helpers ====================

def _list_convs(model: nn.Module):
    model = unwrap(model)
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            yield name, m


def build_soft_gated_params(model: nn.Module, prune_map: Dict[str, List[int]], gamma: float) -> OrderedDict:
    params = OrderedDict()
    for name, p in unwrap(model).named_parameters():
        mod_name = name.rsplit(".", 1)[0]
        if p.dim() == 4 and mod_name in prune_map:
            C_out = p.size(0)
            mask = torch.ones(C_out, device=p.device, dtype=p.dtype)
            idx = torch.as_tensor(prune_map[mod_name], device=p.device, dtype=torch.long)
            if idx.numel() > 0:
                mask[idx] = float(gamma)
            params[name] = p * mask.view(-1, 1, 1, 1)
        else:
            params[name] = p
    return params


def build_logit_A_channel_pool(
    model: nn.Module,
    pool_loader: DataLoader,
    device: str,
    max_batches: int,
    coverage: float,
):
    """
    A-channel pool S_A:
      score channel by positive (activation * gradient) contribution to
      delta_logit(target_A) = logit_A - logit_clean.
    """
    model = unwrap(model).to(device)
    model.eval()

    act_dict: Dict[str, torch.Tensor] = {}
    grad_dict: Dict[str, torch.Tensor] = {}
    handles = []

    def make_fwd_hook(name: str):
        def _hook(_m, _inp, out):
            act_dict[name] = out
            def _grad_hook(grad):
                grad_dict[name] = grad
            out.register_hook(_grad_hook)
        return _hook

    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            handles.append(m.register_forward_hook(make_fwd_hook(name)))

    contrib_total: Dict[str, torch.Tensor] = {}
    n_batches = 0

    for xb, yb in pool_loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        # xb is [0,1]; normalize after injecting trigger
        x_clean = norm(xb.clone())

        x_A = xb.clone()
        x_A = trigger_patch_batch(x_A, cfg.trigger_A_size, cfg.trigger_A_value, cfg.trigger_A_pos)
        x_A = norm(x_A)

        model.zero_grad(set_to_none=True)
        act_dict.clear()
        grad_dict.clear()

        logits_clean = model(x_clean)
        logits_A = model(x_A)

        delta = logits_A - logits_clean
        obj = delta[:, int(cfg.target_A)].mean()
        obj.backward()

        for lname, act in act_dict.items():
            grad = grad_dict.get(lname, None)
            if grad is None:
                continue
            prod = act * grad
            contrib = prod.sum(dim=(0, 2, 3))  # [C]
            contrib = torch.clamp(contrib, min=0.0).detach()
            if lname not in contrib_total:
                contrib_total[lname] = contrib.clone()
            else:
                if contrib_total[lname].shape != contrib.shape:
                    m = min(contrib_total[lname].numel(), contrib.numel())
                    contrib_total[lname][:m] += contrib[:m]
                else:
                    contrib_total[lname] += contrib

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
            score = float(v[ch].item())
            if score > 0.0:
                all_items.append((score, lname, int(ch)))

    if len(all_items) == 0:
        print("[Stage3] WARNING: A-channel pool empty. Fallback to all conv channels.")
        for lname, conv in _list_convs(model):
            C = conv.weight.size(0)
            for ch in range(C):
                all_items.append((1.0, lname, int(ch)))

    all_items.sort(key=lambda x: x[0], reverse=True)
    total_score = sum(s for s, _, _ in all_items)
    total_score = total_score if total_score > 0 else 1e-8

    pool_items: List[Tuple[str, int, float]] = []
    cum = 0.0
    for score, lname, ch in all_items:
        cum += score
        pool_items.append((lname, ch, score))
        if cum / total_score >= float(coverage):
            break

    print(f"[Stage3] built A-channel pool: K_pool={len(pool_items)} / {len(all_items)} (coverage={coverage*100:.1f}%)")
    return pool_items


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


def is_defense_param(name: str) -> bool:
    return any(s in name for s in ["conv", "bn", "downsample", "layer", "fc"])


def add_param_noise(params: OrderedDict, std: float, prune_map: Dict[str, List[int]] = None) -> OrderedDict:
    if std <= 0:
        return params
    noisy = OrderedDict()
    for name, p in params.items():
        if not is_defense_param(name):
            noisy[name] = p
            continue

        if prune_map is not None and p.dim() == 4:
            mod_name = name.rsplit(".", 1)[0]
            if mod_name in prune_map and len(prune_map[mod_name]) > 0:
                C_out = p.size(0)
                mask = torch.ones(C_out, device=p.device, dtype=p.dtype)
                idx = torch.as_tensor(prune_map[mod_name], device=p.device, dtype=torch.long)
                mask[idx] = 0.0
                noise = std * torch.randn_like(p) * mask.view(-1, 1, 1, 1)
                noisy[name] = p + noise
                continue

        noisy[name] = p + std * torch.randn_like(p)
    return noisy


def batch_acc_asr(logits_clean, logits_A, logits_B, y_true, target_A, target_B, mask_A, mask_B):
    with torch.no_grad():
        pred_clean = logits_clean.argmax(1)
        acc_clean = (pred_clean == y_true).float().mean().item()

        pred_A = logits_A.argmax(1)
        acc_A = (pred_A == y_true).float().mean().item()
        asr_A = (pred_A[mask_A] == target_A).float().mean().item() if mask_A.any() else 0.0

        pred_B = logits_B.argmax(1)
        acc_B = (pred_B == y_true).float().mean().item()
        asr_B = (pred_B[mask_B] == target_B).float().mean().item() if mask_B.any() else 0.0

    return dict(ACC_clean=acc_clean, ACC_A=acc_A, ASR_A=asr_A, ACC_B=acc_B, ASR_B=asr_B)


# ==================== hooks & structural losses ====================

def forward_with_acts(model: nn.Module, x: torch.Tensor, layer_names: List[str] = None):
    net = unwrap(model)
    acts: Dict[str, torch.Tensor] = {}
    handles = []

    def make_hook(name: str):
        def _hook(_m, _inp, out):
            if torch.is_tensor(out):
                acts[name] = out
        return _hook

    for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d) and (layer_names is None or name in layer_names):
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

    layers_focus = 0
    layers_clean = 0
    layers_covert = 0

    idx_B = mask_B
    idx_clean_tB = (yb == int(target_B))

    tau_A = 0.7
    rho_clean = 0.5

    for lname, act_clean in acts_clean.items():
        act_A = acts_A.get(lname, None)
        act_B = acts_B.get(lname, None)
        if act_A is None or act_B is None:
            continue

        Bsz, C, H, W = act_clean.shape
        scap_idx = S_map.get(lname, None)
        if scap_idx is None or len(scap_idx) == 0:
            continue

        scap_tensor = torch.as_tensor(scap_idx, device=device, dtype=torch.long)
        all_idx = torch.arange(C, device=device)

        mask_non = torch.ones(C, device=device, dtype=torch.bool)
        mask_non[scap_tensor] = False
        non_idx = all_idx[mask_non]
        if non_idx.numel() == 0:
            continue

        # A-focus
        energy_scap = act_A[:, scap_tensor].abs().mean()
        energy_non = act_A[:, non_idx].abs().mean()
        p_A = energy_scap / (energy_scap + energy_non + 1e-6)
        L_A_focus = L_A_focus + F.relu(tau_A - p_A)
        layers_focus += 1

        # clean scap relative suppression
        energy_clean_scap = act_clean[:, scap_tensor].abs().mean()
        energy_clean_non = act_clean[:, non_idx].abs().mean()
        L_clean_scap = L_clean_scap + F.relu(energy_clean_scap - rho_clean * energy_clean_non)
        layers_clean += 1

        # B-covert on non-scap: match clean(target_B) semantics
        if idx_B.any() and idx_clean_tB.any():
            act_B_sel = act_B[idx_B][:, non_idx]
            act_clean_tB_sel = act_clean[idx_clean_tB][:, non_idx]
            if act_B_sel.numel() == 0 or act_clean_tB_sel.numel() == 0:
                continue

            mean_B = act_B_sel.mean(dim=(0, 2, 3))
            mean_clean_tB = act_clean_tB_sel.mean(dim=(0, 2, 3))

            mean_B_n = F.normalize(mean_B, dim=0)
            mean_clean_n = F.normalize(mean_clean_tB, dim=0)

            L_B_covert = L_B_covert + F.mse_loss(mean_B_n, mean_clean_n)
            layers_covert += 1

    if layers_focus > 0:
        L_A_focus = L_A_focus / layers_focus
    if layers_clean > 0:
        L_clean_scap = L_clean_scap / layers_clean
    if layers_covert > 0:
        L_B_covert = L_B_covert / layers_covert

    return L_A_focus, L_clean_scap, L_B_covert


# ==================== Stage-3 training ====================

def train_stage3_meta():
    os.makedirs(cfg.save_dir, exist_ok=True)
    device = cfg.device

    model = build_cifar_resnet50(num_classes=cfg.num_classes).to(device)
    ckpt_path = Path(cfg.save_dir) / cfg.ckpt_afterA
    print(f"[Stage3] loading afterA from {ckpt_path}")
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
    model.load_state_dict(sd, strict=True)

    train_loader = make_train_loader(batch_size=cfg.batch_size)
    T_clean, T_A, T_B = make_test_loaders()

    init_metrics = eval_acc_asr_AB(model, T_clean, T_A, T_B, cfg.target_A, cfg.target_B, device)
    print(
        f"[Stage3] INIT: "
        f"ACC_clean={init_metrics['ACC_clean']*100:.2f}% "
        f"ASR_A={init_metrics['ASR_A']*100:.2f}% "
        f"ASR_B={init_metrics['ASR_B']*100:.2f}%"
    )

    opt = torch.optim.SGD(
        model.parameters(),
        lr=float(cfg.lr),
        momentum=float(cfg.momentum),
        weight_decay=float(cfg.weight_decay),
        nesterov=True,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(cfg.epochs))

    for ep in range(1, int(cfg.epochs) + 1):
        model.train()
        run_loss = 0.0
        n_samples = 0

        # build A-channel pool each epoch (smaller pool loader to avoid OOM)
        pool_loader = make_train_loader(batch_size=cfg.a_pool_batch_size)
        S_A = build_logit_A_channel_pool(
            model,
            pool_loader,
            device=device,
            max_batches=cfg.a_pool_max_batches,
            coverage=cfg.a_pool_coverage,
        )
        print(f"[Stage3] A-channel pool ready (epoch {ep:02d}): K_pool={len(S_A)}")

        # build scap map (choose top layers by total score, to control memory)
        layer_to_items: Dict[str, List[Tuple[int, float]]] = {}
        layer_score: Dict[str, float] = {}
        for lname, ch, sc in S_A:
            layer_to_items.setdefault(lname, []).append((int(ch), float(sc)))
            layer_score[lname] = layer_score.get(lname, 0.0) + float(sc)

        if len(layer_to_items) > 0:
            top_layers = sorted(layer_score.items(), key=lambda kv: kv[1], reverse=True)[: int(cfg.scap_max_layers)]
            top_layer_names = set([k for k, _ in top_layers])
        else:
            top_layer_names = set()

        S_map: Dict[str, List[int]] = {}
        for lname, items in layer_to_items.items():
            if lname not in top_layer_names:
                continue
            items_sorted = sorted(items, key=lambda t: t[1], reverse=True)
            C_layer = len(items_sorted)
            if C_layer <= 0:
                continue
            k = max(1, int(round(float(cfg.scap_topk_ratio) * C_layer)))
            k = min(k, C_layer)
            S_map[lname] = [ch for ch, _ in items_sorted[:k]]

        hook_layers = list(S_map.keys()) if len(S_map) > 0 else None
        if hook_layers is not None:
            print(f"[Stage3] structural-loss layers: {len(hook_layers)} (max={cfg.scap_max_layers})")

        teacher_stat_sum = {k: 0.0 for k in ["ACC_clean", "ACC_A", "ASR_A", "ACC_B", "ASR_B"]}
        defense_stat_sum = {k: 0.0 for k in ["ACC_clean", "ACC_A", "ASR_A", "ACC_B", "ASR_B"]}
        stat_batches = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            Bsz = xb.size(0)

            # -------- teacher views: clean / A / B --------
            x_clean = norm(xb.clone())

            x_A_teacher = xb.clone()
            mask_A = (torch.rand(Bsz, device=device) < float(cfg.pA))
            if mask_A.any():
                x_A_teacher[mask_A] = trigger_patch_batch(
                    x_A_teacher[mask_A],
                    cfg.trigger_A_size,
                    cfg.trigger_A_value,
                    cfg.trigger_A_pos,
                )
            x_A_teacher = norm(x_A_teacher)
            yA_teacher = yb.clone()
            yA_teacher[mask_A] = int(cfg.target_A)

            x_B = xb.clone()
            mask_B = (torch.rand(Bsz, device=device) < float(cfg.pB))
            if mask_B.any():
                x_B[mask_B] = add_sig_trigger_batch(x_B[mask_B], cfg.sig_amp, cfg.sig_freq)
            x_B = norm(x_B)

            yB_teacher = yb.clone()  # teacher keeps B dormant -> clean labels
            yB_target = yb.clone()
            yB_target[mask_B] = int(cfg.target_B)  # defense-view wants B->target_B

            logits_clean_T, acts_clean = forward_with_acts(model, x_clean, hook_layers)
            logits_A_T, acts_A = forward_with_acts(model, x_A_teacher, hook_layers)
            logits_B_T, acts_B = forward_with_acts(model, x_B, hook_layers)

            # -------- teacher losses --------
            loss_clean_T = F.cross_entropy(logits_clean_T, yb)
            loss_A_T = F.cross_entropy(logits_A_T, yA_teacher)
            loss_B_clean_T = F.cross_entropy(logits_B_T, yB_teacher)

            # B-silence: suppress teacher B->target_B probability toward random
            # The silence penalty must backpropagate through the teacher.
            num_classes = logits_B_T.size(1)
            p_rand = 1.0 / float(num_classes)
            p_B_target_T = logits_B_T.softmax(dim=1)[:, int(cfg.target_B)]
            L_B_silence = F.relu(p_B_target_T - p_rand).mean()

            teacher_loss = (
                float(cfg.lambda_clean_T) * loss_clean_T
                + float(cfg.lambda_A_T) * loss_A_T
                + float(cfg.lambda_B_clean_T) * loss_B_clean_T
                + float(cfg.lambda_B_silence) * L_B_silence
            )

            # structural losses
            if hook_layers is not None:
                L_A_focus, L_clean_scap, L_B_covert = compute_struct_losses(
                    acts_clean, acts_A, acts_B,
                    yb, mask_B, S_map, cfg.target_B
                )
            else:
                L_A_focus = torch.zeros((), device=device)
                L_clean_scap = torch.zeros((), device=device)
                L_B_covert = torch.zeros((), device=device)

            # teacher A/B strengths
            with torch.no_grad():
                prob_A_T = logits_A_T[mask_A].softmax(dim=1)[:, int(cfg.target_A)].mean() if mask_A.any() else torch.tensor(0.0, device=device)
                prob_B_T = logits_B_T[mask_B].softmax(dim=1)[:, int(cfg.target_B)].mean() if mask_B.any() else torch.tensor(0.0, device=device)

            # -------- meta branch (D~) --------
            meta_B_loss = torch.zeros((), device=device)
            meta_clean_loss = torch.zeros((), device=device)
            meta_A_loss = torch.zeros((), device=device)
            L_flat_B = torch.zeros((), device=device)

            num_noise = max(1, int(cfg.defense_num_noise_samples))
            prune_map_epoch = sample_prune_map_from_pool(S_A, cfg.a_pool_sample_ratio_min, cfg.a_pool_sample_ratio_max)

            flat_B_probs = []
            delta_A_acc = torch.zeros((), device=device)
            delta_B_acc = torch.zeros((), device=device)
            p_B_def_acc = torch.zeros((), device=device)
            cnt_delta_A = 0
            cnt_delta_B = 0

            for _ in range(num_noise):
                params_def = build_soft_gated_params(model, prune_map_epoch, gamma=cfg.rnp_soft_gamma)
                params_noisy = add_param_noise(params_def, cfg.defense_noise_std, prune_map_epoch)

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

            delta_A = (delta_A_acc / float(cnt_delta_A)).clamp(0.0, 1.0) if cnt_delta_A > 0 else torch.tensor(0.0, device=device)
            if cnt_delta_B > 0:
                delta_B = (delta_B_acc / float(cnt_delta_B)).clamp(0.0, 1.0)
                p_B_def = (p_B_def_acc / float(cnt_delta_B)).clamp(0.0, 1.0)
            else:
                delta_B = torch.tensor(0.0, device=device)
                p_B_def = torch.tensor(0.0, device=device)

            meta_B_loss = meta_B_loss / float(num_noise)
            meta_clean_loss = meta_clean_loss / float(num_noise)
            meta_A_loss = meta_A_loss / float(num_noise)

            # serial coupling
            L_serial = (delta_B - delta_A.detach()) ** 2

            # flatness on target_B probability
            if len(flat_B_probs) > 1 and mask_B.any():
                P = torch.stack(flat_B_probs, dim=0)          # (K, N_B, C)
                p_t = P[..., int(cfg.target_B)]               # (K, N_B)
                p_t_mean = p_t.mean(dim=0).detach()           # (N_B,)
                L_flat_core = ((p_t - p_t_mean) ** 2).mean()

                tau_A_flat = 0.3
                gA = ((delta_A.detach() - tau_A_flat) / (1.0 - tau_A_flat)).clamp(0.0, 1.0)
                tau_B_flat = 0.6
                gB = ((p_B_def.detach() - tau_B_flat) / (1.0 - tau_B_flat)).clamp(0.0, 1.0)

                gate_flat = gA * gB
                L_flat_B = gate_flat * L_flat_core
            else:
                L_flat_B = torch.zeros((), device=device)

            total_loss = (
                teacher_loss
                + float(cfg.lambda_meta_B) * meta_B_loss
                + float(cfg.lambda_meta_clean) * meta_clean_loss
                + float(cfg.lambda_meta_A) * meta_A_loss
                + float(cfg.lambda_consp) * L_A_focus
                + float(cfg.lambda_sep) * L_clean_scap
                + float(cfg.lambda_covert) * L_B_covert
                + float(cfg.lambda_flat_B) * L_flat_B
                + float(cfg.lambda_serial) * L_serial
            )

            opt.zero_grad(set_to_none=True)
            total_loss.backward()
            if cfg.grad_clip is not None and float(cfg.grad_clip) > 0:
                nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip))
            opt.step()

            run_loss += float(total_loss.item()) * Bsz
            n_samples += Bsz

            # quick train-stats
            teacher_stat = batch_acc_asr(
                logits_clean_T, logits_A_T, logits_B_T,
                yb, int(cfg.target_A), int(cfg.target_B),
                mask_A, mask_B
            )

            with torch.no_grad():
                prune_map_eval = sample_prune_map_from_pool(S_A, cfg.a_pool_sample_ratio_min, cfg.a_pool_sample_ratio_max)
                params_def_eval = build_soft_gated_params(model, prune_map_eval, gamma=cfg.rnp_soft_gamma)
                logits_clean_def_eval = functional_call(unwrap(model), params_def_eval, (x_clean,))
                logits_A_def_eval = functional_call(unwrap(model), params_def_eval, (x_A_teacher,))
                logits_B_def_eval = functional_call(unwrap(model), params_def_eval, (x_B,))

            defense_stat = batch_acc_asr(
                logits_clean_def_eval, logits_A_def_eval, logits_B_def_eval,
                yb, int(cfg.target_A), int(cfg.target_B),
                mask_A, mask_B
            )

            for k in teacher_stat_sum:
                teacher_stat_sum[k] += teacher_stat[k]
                defense_stat_sum[k] += defense_stat[k]
            stat_batches += 1

        sched.step()

        for k in teacher_stat_sum:
            teacher_stat_sum[k] /= max(1, stat_batches)
            defense_stat_sum[k] /= max(1, stat_batches)

        print(
            f"[Stage3|ep {ep:02d}] loss={run_loss/max(1,n_samples):.4f} | "
            f"TEACHER(train-approx): ACC_clean={teacher_stat_sum['ACC_clean']*100:.2f}% "
            f"ASR_A={teacher_stat_sum['ASR_A']*100:.2f}% ASR_B={teacher_stat_sum['ASR_B']*100:.2f}% | "
            f"DEF(D~, train-approx): ACC_clean={defense_stat_sum['ACC_clean']*100:.2f}% "
            f"ASR_A={defense_stat_sum['ASR_A']*100:.2f}% ASR_B={defense_stat_sum['ASR_B']*100:.2f}%"
        )

        # test-set metrics for the published teacher
        model.eval()
        metrics_teacher = eval_acc_asr_AB(model, T_clean, T_A, T_B, cfg.target_A, cfg.target_B, device)
        print(
            f"[Stage3|ep {ep:02d}] TEST-TEACHER: "
            f"ACC_clean={metrics_teacher['ACC_clean']*100:.2f}% "
            f"ASR_A={metrics_teacher['ASR_A']*100:.2f}% "
            f"ASR_B={metrics_teacher['ASR_B']*100:.2f}%"
        )

    # final save (+ final metrics)
    final_metrics = eval_acc_asr_AB(model, T_clean, T_A, T_B, cfg.target_A, cfg.target_B, device)
    save_path = Path(cfg.save_dir) / cfg.ckpt_teacher_meta
    torch.save(
        {
            "state_dict": unwrap(model).state_dict(),
            "metrics": final_metrics,
            "cfg": cfg.__dict__,
        },
        save_path
    )
    print(f"[Stage3] saved teacher(meta) model -> {save_path}")
    print("[Stage3] Final metrics:", {k: f"{v*100:.2f}%" for k, v in final_metrics.items()})


if __name__ == "__main__":
    train_stage3_meta()
