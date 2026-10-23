# Extracted from BB/VGG16.ipynb, code cell 8 (zero-based).
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, math, random, copy, sys
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

# ==================== 配置（VGG16-BN + CIFAR-10） ====================

@dataclass
class Stage3Cfg:
    data_root: str = "./data"
    save_dir: str = "./exp_three_stage"

    # 使用 VGG16-BN Stage-2 生成的 afterA 模型
    ckpt_afterA: str = "afterA_stage2_simple_vgg16bn_cifar10.pth"
    # Stage-3 输出 ckpt 名称（可自行修改）
    ckpt_teacher_meta: str = "afterAB_stage3_meta_vgg16bn_cifar10_RNP_logitpool6-5.pth"

    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    num_workers: int = 8

    batch_size: int = 256
    epochs: int = 15

    lr: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 5e-4

    # A / B 后门
    trigger_A_size: int = 3
    trigger_A_value: float = 1.0
    trigger_A_pos: str = "br"
    target_A: int = 0

    sig_amp: float = 0.1
    sig_freq: float = 6.0
    target_B: int = 1

    # 触发概率
    pA: float = 0.1
    pB: float = 0.1

    # teacher 分支 loss 权重
    lambda_clean_T: float = 1.0
    lambda_A_T: float = 1.1
    lambda_B_clean_T: float = 0.5        # 调低：避免强行压 B→clean
    lambda_B_silence: float = 0.6        # 压制 teacher 视角下 B→target_B

    # D_no_prune：K 步虚拟 clean 微调约束 —— 这里“硬关掉”
    lambda_no_prune: float = 0.0
    no_prune_ft_steps: int = 0
    no_prune_ft_lr: float = 0.0
    no_prune_ft_max_grad_norm: float = 0.5  # 保留字段，但不会用到

    # logit/梯度贡献度的 A-通道池
    a_pool_max_batches: int = 16
    a_pool_coverage: float = 0.8
    a_pool_sample_ratio_min: float = 0.05
    a_pool_sample_ratio_max: float = 0.95

    # soft gate（防御模拟用的软剪枝）
    rnp_soft_gamma: float = 1e-9

    # 噪声：用于围绕 D~ 的 B 输出一致性平滑
    defense_noise_std: float = 5e-3
    defense_num_noise_samples: int = 10

    # meta 目标
    lambda_meta_B: float = 1.0
    lambda_meta_clean: float = 1.0
    lambda_meta_A: float = 1.0
    meta_A_min_drop: float = 0.6        # 保留字段

    # 结构正则：
    # - lambda_consp: A-focus（A 能量集中 scap）
    # - lambda_sep  : clean-scap 相对抑制
    # - lambda_covert: B-covert（B 在 non-scap 上伪装 target_B 语义）
    lambda_consp: float = 0.5
    lambda_sep: float = 0.5
    lambda_covert: float = 0.3
    scap_topk_ratio: float = 0.15

    # B 输出一致性平滑项权重（target_B-only + gate_flat）
    lambda_flat_B: float = 0.05

    # 串行耦合损失 ΔB≈ΔA 的权重
    lambda_serial: float = 0.5

    train_subset: int = 0
    test_subset: int = 0
    grad_clip: float = 5.0


cfg = Stage3Cfg()
from dtb.runtime import configure as _configure
cfg = _configure(cfg)

# CIFAR-10 mean/std
MEAN = (0.4914, 0.4822, 0.4465)
STD  = (0.2023, 0.1994, 0.2010)
norm = transforms.Normalize(MEAN, STD)


def set_seed(s=1):
    random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark = True


set_seed(1)


def unwrap(m):
    return m.module if hasattr(m, "module") else m

# ==================== 模型（VGG16-BN CIFAR 版本） ====================

class VGG16BN_CIFAR(nn.Module):
    """
    VGG16-BN backbone (torchvision) + CIFAR-friendly head:
      features -> AdaptiveAvgPool2d(1) -> Linear(512 -> num_classes)
    与你 Stage-1/Stage-2 中的定义保持一致。
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

# ==================== 触发器 ====================

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

# ==================== 数据 ====================

def make_train_loader():
    tf = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),  # [0,1]
        ]
    )
    ds_full = datasets.CIFAR10(cfg.data_root, train=True, transform=tf, download=True)

    if 0 < cfg.train_subset < len(ds_full):
        indices = list(range(cfg.train_subset))
        ds = Subset(ds_full, indices)
        print(f"[Stage3] train_subset = {cfg.train_subset} / {len(ds_full)}")
    else:
        ds = ds_full
        print(f"[Stage3] train_subset disabled, use full train set: {len(ds_full)}")

    loader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return loader


class EvalView(Dataset):
    def __init__(self, base, mode: str):
        self.base, self.mode = base, mode

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        img, y = self.base[i]
        x = transforms.ToTensor()(img)  # [0,1]
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
    base_full = datasets.CIFAR10(
        cfg.data_root, train=False, transform=None, download=True
    )

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
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    T_A = DataLoader(
        EvalView(base, "A"),
        batch_size=512,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    T_B = DataLoader(
        EvalView(base, "B"),
        batch_size=512,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    return T_clean, T_A, T_B

# ==================== 全集评测 ====================

@torch.no_grad()
def eval_acc_asr_AB(model, T_clean, T_A, T_B, target_A: int, target_B: int, device: str):
    net = unwrap(model).eval().to(device)

    def _acc(loader, target=None):
        hit = tot = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = net(xb)
            pred = logits.argmax(1)
            if target is None:
                hit += (pred == yb).sum().item()
            else:
                hit += (pred == target).sum().item()
            tot += yb.numel()
        return hit / max(1, tot)

    ACC_clean = _acc(T_clean, target=None)
    ACC_A = _acc(T_A, target=None)
    ASR_A = _acc(T_A, target_A)
    ACC_B = _acc(T_B, target=None)
    ASR_B = _acc(T_B, target_B)
    return dict(
        ACC_clean=ACC_clean,
        ACC_A=ACC_A,
        ASR_A=ASR_A,
        ACC_B=ACC_B,
        ASR_B=ASR_B,
    )

# ==================== Conv / 通道池等 ====================

def _list_convs(model: nn.Module):
    model = unwrap(model)
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            yield name, m


def build_soft_gated_params(model: nn.Module,
                            prune_map: Dict[str, List[int]],
                            gamma: float) -> OrderedDict:
    params = OrderedDict()
    for name, p in unwrap(model).named_parameters():
        mod_name = name.rsplit(".", 1)[0]
        if p.dim() == 4 and mod_name in prune_map:
            C_out = p.size(0)
            mask = torch.ones(C_out, device=p.device, dtype=p.dtype)
            idx = torch.as_tensor(
                prune_map[mod_name],
                device=p.device,
                dtype=torch.long,
            )
            if idx.numel() > 0:
                mask[idx] = gamma
            params[name] = p * mask.view(-1, 1, 1, 1)
        else:
            params[name] = p
    return params


def build_logit_A_channel_pool(model: nn.Module,
                               pool_loader: DataLoader,
                               device: str,
                               max_batches: int,
                               coverage: float):
    """
    通道池 S_A：基于“正向 activation×gradient”的 A-敏感通道打分，
    只保留会“增强 A→target_A logit”的通道。
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
            h = m.register_forward_hook(make_fwd_hook(name))
            handles.append(h)

    contrib_total: Dict[str, torch.Tensor] = {}
    n_batches = 0

    for xb, yb in pool_loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        x_clean = norm(xb.clone())
        x_A = xb.clone()
        x_A = trigger_patch_batch(
            x_A,
            cfg.trigger_A_size,
            cfg.trigger_A_value,
            cfg.trigger_A_pos,
        )
        x_A = norm(x_A)

        model.zero_grad(set_to_none=True)
        act_dict.clear()
        grad_dict.clear()

        logits_clean = model(x_clean)
        logits_A = model(x_A)

        # 目标：A 视图相对于 clean 视图的 target_A logit 提升
        delta = logits_A - logits_clean
        obj = delta[:, cfg.target_A].mean()
        obj.backward()

        for lname, act in act_dict.items():
            grad = grad_dict.get(lname, None)
            if grad is None:
                continue

            # 只保留“正向 contribution”的 activation×gradient
            prod = act * grad                    # (B, C, H, W)
            contrib = prod.sum(dim=(0, 2, 3))    # [C]
            contrib = torch.clamp(contrib, min=0.0)
            contrib = contrib.detach()

            if lname not in contrib_total:
                contrib_total[lname] = contrib.clone()
            else:
                if contrib_total[lname].shape != contrib.shape:
                    min_c = min(contrib_total[lname].numel(), contrib.numel())
                    contrib_total[lname][:min_c] += contrib[:min_c]
                else:
                    contrib_total[lname] += contrib

        n_batches += 1
        if n_batches >= max_batches:
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
            if score <= 0.0:
                continue
            all_items.append((score, lname, int(ch)))

    if len(all_items) == 0:
        print("[Stage3] WARNING: logit-based A-channel pool empty, fallback to all conv channels.")
        for lname, conv in _list_convs(model):
            C = conv.weight.size(0)
            for ch in range(C):
                all_items.append((1.0, lname, int(ch)))

    all_items.sort(key=lambda x: x[0], reverse=True)
    total_score = sum(s for s, _, _ in all_items)
    if total_score <= 0:
        total_score = 1e-8

    pool_items: List[Tuple[str, int, float]] = []
    cum = 0.0
    for score, lname, ch in all_items:
        cum += score
        pool_items.append((lname, ch, score))
        if cum / total_score >= coverage:
            break

    K_pool = len(pool_items)
    K_all = len(all_items)
    print(
        f"[Stage3] built logit-based A-channel pool: "
        f"K_pool={K_pool} / {K_all} (coverage={coverage*100:.1f}%)"
    )
    return pool_items


def sample_prune_map_from_pool(
    pool_items: List[Tuple[str, int, float]],
    ratio_min: float,
    ratio_max: float,
) -> Dict[str, List[int]]:
    """
    从 S_A 池中按比例随机采样通道。
    """
    prune_map: Dict[str, List[int]] = {}
    K_pool = len(pool_items)
    if K_pool == 0:
        return prune_map

    r_min = max(0.0, min(1.0, ratio_min))
    r_max = max(r_min, min(1.0, ratio_max))
    r = random.uniform(r_min, r_max)
    K_def = max(1, int(round(r * K_pool)))
    K_def = min(K_def, K_pool)

    idxs = random.sample(range(K_pool), K_def)
    for idx in idxs:
        lname, ch, _ = pool_items[idx]
        prune_map.setdefault(lname, []).append(ch)
    return prune_map


def is_defense_param(name: str) -> bool:
    """
    标记需要加噪/视作“防御参数”的权重名。
    对 VGG 来说，主要是 features.* 和 classifier.*。
    """
    if any(s in name for s in ["conv", "bn", "downsample", "layer", "fc",
                               "features", "classifier"]):
        return True
    return False


def add_param_noise(params: OrderedDict,
                    std: float,
                    prune_map: Dict[str, List[int]] = None) -> OrderedDict:
    """
    参数噪声：用于围绕 D~ 的 B 输出一致性平滑。
    """
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
                idx = torch.as_tensor(prune_map[mod_name],
                                      device=p.device, dtype=torch.long)
                mask[idx] = 0.0

                noise = std * torch.randn_like(p)
                noise = noise * mask.view(-1, 1, 1, 1)
                noisy[name] = p + noise
                continue

        noisy[name] = p + std * torch.randn_like(p)

    return noisy


def batch_acc_asr(
    logits_clean,
    logits_A,
    logits_B,
    y_true,
    target_A,
    target_B,
    mask_A,
    mask_B,
):
    with torch.no_grad():
        pred_clean = logits_clean.argmax(1)
        acc_clean = (pred_clean == y_true).float().mean().item()

        pred_A = logits_A.argmax(1)
        if mask_A.any():
            asr_A = (pred_A[mask_A] == target_A).float().mean().item()
        else:
            asr_A = 0.0
        acc_A = (pred_A == y_true).float().mean().item()

        pred_B = logits_B.argmax(1)
        if mask_B.any():
            asr_B = (pred_B[mask_B] == target_B).float().mean().item()
        else:
            asr_B = 0.0
        acc_B = (pred_B == y_true).float().mean().item()

    return dict(
        ACC_clean=acc_clean,
        ACC_A=acc_A,
        ASR_A=asr_A,
        ACC_B=acc_B,
        ASR_B=asr_B,
    )

# ==================== hook & 结构正则 ====================

def forward_with_acts(model: nn.Module,
                      x: torch.Tensor,
                      layer_names: List[str] = None):
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
            h = m.register_forward_hook(make_hook(name))
            handles.append(h)

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
    """
    结构正则拆成三部分：
    - L_A_focus    : A 的激活能量要集中在每层的 scap 通道中
    - L_clean_scap : clean 视图上 scap 通道相对 non-scap 通道要“安静”（相对约束）
    - L_B_covert   : 在非 scap 通道上，B 触发样本的 mean 激活分布
                     伪装成 "clean & y=target_B" 的语义分布
    """
    device = yb.device

    L_A_focus = torch.zeros((), device=device)
    L_clean_scap = torch.zeros((), device=device)
    L_B_covert = torch.zeros((), device=device)

    layers_focus = 0
    layers_clean = 0
    layers_covert = 0

    idx_B = mask_B
    idx_clean_tB = (yb == target_B)

    # A 能量集中到 scap 通道比例阈值
    tau_A = 0.7
    # clean 上 scap vs non-scap 的相对约束系数
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

        # ----- L_A_focus: A 的能量集中在 scap 通道 -----
        act_A_scap = act_A[:, scap_tensor]
        act_A_non = act_A[:, non_idx]

        energy_scap = act_A_scap.abs().mean()
        energy_non = act_A_non.abs().mean()

        p_A = energy_scap / (energy_scap + energy_non + 1e-6)
        L_A_focus = L_A_focus + F.relu(tau_A - p_A)
        layers_focus += 1

        # ----- L_clean_scap: clean 上 scap 通道相对抑制（相对约束） -----
        act_clean_scap = act_clean[:, scap_tensor]
        act_clean_non = act_clean[:, non_idx]

        energy_clean_scap = act_clean_scap.abs().mean()
        energy_clean_non = act_clean_non.abs().mean()

        L_clean_scap = L_clean_scap + F.relu(
            energy_clean_scap - rho_clean * energy_clean_non
        )
        layers_clean += 1

        # ----- L_B_covert: B 在 non-scap 上伪装成 target_B 语义 -----
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

# ==================== K 步虚拟 clean 微调：D_no_prune 分支 ====================

def run_virtual_clean_ft(
    model: nn.Module,
    x_clean: torch.Tensor,
    yb: torch.Tensor,
    x_B: torch.Tensor,
    yB_teacher: torch.Tensor,
    yB_target: torch.Tensor,
    mask_B: torch.Tensor,
) -> torch.Tensor:
    """
    原始设计是 FOMAML 风格的 inner-loop，
    现在通过 cfg.lambda_no_prune = 0.0、no_prune_ft_steps = 0、
    no_prune_ft_lr = 0.0 来“硬关掉”。
    """
    device = x_clean.device

    if (not mask_B.any() or
        cfg.no_prune_ft_steps <= 0 or
        cfg.no_prune_ft_lr <= 0):
        return torch.zeros((), device=device)

    net = unwrap(model)
    was_training = net.training
    net.eval()

    params_ft = OrderedDict((name, p) for name, p in net.named_parameters())
    L_accum = torch.zeros((), device=device)

    for _ in range(cfg.no_prune_ft_steps):
        logits_clean_ft = functional_call(net, params_ft, (x_clean,))
        loss_clean_ft = F.cross_entropy(logits_clean_ft, yb)

        grads = torch.autograd.grad(
            loss_clean_ft,
            list(params_ft.values()),
            create_graph=False,
        )

        new_params_ft = OrderedDict()
        max_g = cfg.no_prune_ft_max_grad_norm
        lr_inner = cfg.no_prune_ft_lr
        for (name, p), g in zip(params_ft.items(), grads):
            if g is None:
                new_params_ft[name] = p
                continue
            if max_g is not None and max_g > 0:
                g = torch.clamp(g, -max_g, max_g)
            new_params_ft[name] = p - lr_inner * g
        params_ft = new_params_ft

        logits_B_ft = functional_call(net, params_ft, (x_B,))
        logits_B_ft_B = logits_B_ft[mask_B]
        if logits_B_ft_B.numel() == 0:
            continue

        yB_teacher_B = yB_teacher[mask_B]
        yB_target_B = yB_target[mask_B]

        ce_clean_ft = F.cross_entropy(logits_B_ft_B, yB_teacher_B)
        ce_tgt_ft   = F.cross_entropy(logits_B_ft_B, yB_target_B)

        ce_clean_ft = torch.nan_to_num(ce_clean_ft, nan=0.0, posinf=50.0, neginf=-50.0)
        ce_tgt_ft   = torch.nan_to_num(ce_tgt_ft,   nan=0.0, posinf=50.0, neginf=-50.0)

        step_L = F.relu(ce_clean_ft - ce_tgt_ft)
        L_accum = L_accum + step_L

    if was_training:
        net.train()

    if cfg.no_prune_ft_steps > 0:
        L_no_prune = L_accum / float(cfg.no_prune_ft_steps)
    else:
        L_no_prune = torch.zeros((), device=device)

    L_no_prune = torch.clamp(L_no_prune, max=20.0)
    return L_no_prune

# ==================== Stage-3 训练 ====================

def train_stage3_meta():
    os.makedirs(cfg.save_dir, exist_ok=True)
    device = cfg.device

    model = build_cifar_vgg16bn(num_classes=10).to(device)
    ckpt_path = Path(cfg.save_dir) / cfg.ckpt_afterA
    print(f"[Stage3] loading afterA from {ckpt_path}")
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
    model.load_state_dict(sd, strict=True)

    train_loader = make_train_loader()
    T_clean, T_A, T_B = make_test_loaders()

    init_metrics = eval_acc_asr_AB(
        model, T_clean, T_A, T_B,
        cfg.target_A, cfg.target_B, device
    )
    print(
        f"[Stage3] INIT-TEACHER: "
        f"ACC_clean={init_metrics['ACC_clean']*100:.2f}% "
        f"ASR_A={init_metrics['ASR_A']*100:.2f}% "
        f"ASR_B={init_metrics['ASR_B']*100:.2f}%"
    )

    opt = torch.optim.SGD(
        model.parameters(),
        lr=cfg.lr,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)

    for ep in range(1, cfg.epochs + 1):
        model.train()
        run_loss = 0.0
        n_samples = 0

        # 每个 epoch 重建一次 A 通道池
        pool_loader_for_epoch = make_train_loader()
        S_A = build_logit_A_channel_pool(
            model,
            pool_loader_for_epoch,
            device=device,
            max_batches=cfg.a_pool_max_batches,
            coverage=cfg.a_pool_coverage,
        )
        print(f"[Stage3] A-channel pool ready (epoch {ep:02d}): K_pool={len(S_A)}")

        layer_to_items: Dict[str, List[Tuple[int, float]]] = {}
        for lname, ch, sc in S_A:
            layer_to_items.setdefault(lname, []).append((ch, sc))
        S_map: Dict[str, List[int]] = {}
        for lname, items in layer_to_items.items():
            items_sorted = sorted(items, key=lambda t: t[1], reverse=True)
            C_layer = len(items_sorted)
            if C_layer == 0:
                continue
            k = max(1, int(round(cfg.scap_topk_ratio * C_layer)))
            k = min(k, C_layer)
            scap_chs = [ch for ch, _ in items_sorted[:k]]
            S_map[lname] = scap_chs

        hook_layers = list(S_map.keys()) if len(S_map) > 0 else None

        teacher_stat_sum = {k: 0.0 for k in ["ACC_clean", "ACC_A", "ASR_A", "ACC_B", "ASR_B"]}
        defense_stat_sum = {k: 0.0 for k in ["ACC_clean", "ACC_A", "ASR_A", "ACC_B", "ASR_B"]}
        stat_batches = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            Bsz = xb.size(0)

            # -------- teacher 视图：clean / A / B --------
            x_clean = norm(xb.clone())

            x_A_teacher = xb.clone()
            mask_A = (torch.rand(Bsz, device=device) < cfg.pA)
            if mask_A.any():
                x_A_teacher[mask_A] = trigger_patch_batch(
                    x_A_teacher[mask_A],
                    cfg.trigger_A_size,
                    cfg.trigger_A_value,
                    cfg.trigger_A_pos,
                )
            x_A_teacher = norm(x_A_teacher)
            yA_teacher = yb.clone()
            yA_teacher[mask_A] = cfg.target_A

            x_B = xb.clone()
            mask_B = (torch.rand(Bsz, device=device) < cfg.pB)
            if mask_B.any():
                x_B[mask_B] = add_sig_trigger_batch(
                    x_B[mask_B],
                    cfg.sig_amp,
                    cfg.sig_freq,
                )
            x_B = norm(x_B)

            yB_teacher = yb.clone()          # teacher 上 B→干净标签
            yB_target = yb.clone()
            yB_target[mask_B] = cfg.target_B # 防御后希望 B→target_B

            logits_clean_T, acts_clean = forward_with_acts(model, x_clean, hook_layers)
            logits_A_T, acts_A = forward_with_acts(model, x_A_teacher, hook_layers)
            logits_B_T, acts_B = forward_with_acts(model, x_B, hook_layers)

            # ---------- Teacher losses ----------
            loss_clean_T = F.cross_entropy(logits_clean_T, yb)
            loss_A_T = F.cross_entropy(logits_A_T, yA_teacher)
            loss_B_clean_T = F.cross_entropy(logits_B_T, yB_teacher)

            # Teacher 视角下 B-silence：压制 B→target_B 概率（接近随机）
            # The silence penalty must backpropagate through the teacher.
            num_classes = logits_B_T.size(1)
            p_rand = 1.0 / float(num_classes)
            probs_B_T = logits_B_T.softmax(dim=1)
            p_B_target_T = probs_B_T[:, cfg.target_B]
            L_B_silence = F.relu(p_B_target_T - p_rand).mean()

            teacher_loss = (
                cfg.lambda_clean_T * loss_clean_T
                + cfg.lambda_A_T * loss_A_T
                + cfg.lambda_B_clean_T * loss_B_clean_T
                + cfg.lambda_B_silence * L_B_silence
            )

            # 新的结构正则：L_A_focus + L_clean_scap + L_B_covert
            L_A_focus, L_clean_scap, L_B_covert = compute_struct_losses(
                acts_clean, acts_A, acts_B,
                yb, mask_B, S_map, cfg.target_B
            )

            # -------- K 步虚拟 clean 微调分支：现在“硬关掉” --------
            if cfg.lambda_no_prune > 0.0 and cfg.no_prune_ft_steps > 0 and cfg.no_prune_ft_lr > 0.0:
                L_no_prune = run_virtual_clean_ft(
                    model,
                    x_clean,
                    yb,
                    x_B,
                    yB_teacher,
                    yB_target,
                    mask_B,
                )
            else:
                L_no_prune = torch.zeros((), device=device)

            # -------- 预先算 teacher 的 A / B 强度（用于 Δp_A gating + Δ/serial）--------
            with torch.no_grad():
                if mask_A.any():
                    prob_A_T = logits_A_T[mask_A].softmax(dim=1)[:, cfg.target_A].mean()
                else:
                    prob_A_T = torch.tensor(0.0, device=device)

                if mask_B.any():
                    prob_B_T = logits_B_T[mask_B].softmax(dim=1)[:, cfg.target_B].mean()
                else:
                    prob_B_T = torch.tensor(0.0, device=device)

            # -------- meta 分支：D~ 视角的 clean/A/B + B 输出一致性 --------
            meta_B_loss = torch.zeros((), device=device)
            meta_clean_loss = torch.zeros((), device=device)
            meta_A_loss = torch.zeros((), device=device)
            L_flat_B = torch.zeros((), device=device)

            num_noise = max(1, cfg.defense_num_noise_samples)

            # 同一个 prune_map_epoch，模拟“固定剪枝模式下的小步参数扰动”
            prune_map_epoch = sample_prune_map_from_pool(
                S_A,
                cfg.a_pool_sample_ratio_min,
                cfg.a_pool_sample_ratio_max,
            )

            flat_B_probs = []  # 收集各噪声样本下的 B 概率分布

            # ΔA / ΔB / p_B_def 统计
            delta_A_acc = torch.zeros((), device=device)
            delta_B_acc = torch.zeros((), device=device)
            cnt_delta_A = 0
            cnt_delta_B = 0
            p_B_def_acc = torch.zeros((), device=device)

            for _ in range(num_noise):
                # D~：先做 soft gate，再在该点附近加噪声
                params_def = build_soft_gated_params(
                    model, prune_map_epoch, gamma=cfg.rnp_soft_gamma
                )
                params_noisy = add_param_noise(
                    params_def, cfg.defense_noise_std, prune_map_epoch
                )

                logits_clean_def = functional_call(unwrap(model), params_noisy, (x_clean,))
                logits_B_def = functional_call(unwrap(model), params_noisy, (x_B,))
                logits_A_def = functional_call(unwrap(model), params_noisy, (x_A_teacher,))

                # clean：D~ 下保持高 ACC
                meta_clean_loss = meta_clean_loss + F.cross_entropy(
                    logits_clean_def, yb
                )

                # A：D~ 下显式鼓励“回归干净标签”（A 被防御掉）
                if mask_A.any():
                    meta_A_loss = meta_A_loss + F.cross_entropy(
                        logits_A_def[mask_A], yb[mask_A]
                    )

                # B：这里是“剪掉 A 通道就触发 B”
                if mask_B.any():
                    logits_B_def_B = logits_B_def[mask_B]
                    loss_B_pos = F.cross_entropy(
                        logits_B_def_B, yB_target[mask_B]
                    )  # 希望：防御后 B→target_B
                    loss_B_neg = F.cross_entropy(
                        logits_B_def_B, yB_teacher[mask_B]
                    )  # 兜底：防御失败时 B 仍然保持“干净标签"

                    # 收集 B-samples 的 softmax 概率
                    probs_B_def = F.softmax(logits_B_def_B, dim=1)
                    flat_B_probs.append(probs_B_def)

                    # ΔB 统计 & 平滑 gate 用的 p_B_def
                    # Keep hidden activation differentiable for L_serial.
                    p_B_def_curr = probs_B_def[:, cfg.target_B].mean()
                    delta_p_B = (p_B_def_curr - prob_B_T).clamp(0.0, 1.0)
                    delta_B_acc = delta_B_acc + delta_p_B
                    p_B_def_acc = p_B_def_acc + p_B_def_curr
                    cnt_delta_B += 1
                else:
                    zero_B = logits_B_def.new_tensor(0.0)
                    loss_B_pos = zero_B
                    loss_B_neg = zero_B

                # Δp_A（单次防御视角）统计，用于 meta_B gating
                if mask_A.any():
                    with torch.no_grad():
                        prob_A_def = logits_A_def[mask_A].softmax(dim=1)[:, cfg.target_A].mean()
                        delta_p_A = (prob_A_T - prob_A_def).clamp(0.0, 1.0)
                    delta_A_acc = delta_A_acc + delta_p_A
                    cnt_delta_A += 1
                else:
                    delta_p_A = logits_B_def.new_tensor(0.0)

                # 带阈值的 Δp_A gating
                tau = 0.3
                sA = ((delta_p_A - tau) / (1.0 - tau)).clamp(0.0, 1.0)
                w_pos = sA                     # A 被削弱足够多时，B->target_B 强
                w_neg = 0.05 * (1.0 - sA)      # A 没怎么被剪的时候，仅轻微拉回 clean

                meta_B_loss = meta_B_loss + w_pos * loss_B_pos + w_neg * loss_B_neg

            # 完成 noise 采样后，计算 batch 级 ΔA / ΔB / p_B_def
            if cnt_delta_A > 0:
                delta_A = (delta_A_acc / float(cnt_delta_A)).clamp(0.0, 1.0)
            else:
                delta_A = torch.tensor(0.0, device=device)

            if cnt_delta_B > 0:
                delta_B = (delta_B_acc / float(cnt_delta_B)).clamp(0.0, 1.0)
                p_B_def = (p_B_def_acc / float(cnt_delta_B)).clamp(0.0, 1.0)
            else:
                delta_B = torch.tensor(0.0, device=device)
                p_B_def = torch.tensor(0.0, device=device)

            # 归一化 meta loss
            meta_B_loss = meta_B_loss / float(num_noise)
            meta_clean_loss = meta_clean_loss / float(num_noise)
            meta_A_loss = meta_A_loss / float(num_noise)

            # 串行耦合损失 ΔB ≈ ΔA
            L_serial = (delta_B - delta_A.detach())**2

            # B 输出一致性平滑（target_B-only + gate_flat）
            if len(flat_B_probs) > 1 and mask_B.any():
                P = torch.stack(flat_B_probs, dim=0)         # (K, N_B, C)
                p_t = P[..., cfg.target_B]                  # (K, N_B)
                p_t_mean = p_t.mean(dim=0).detach()         # (N_B,)
                L_flat_core = ((p_t - p_t_mean) ** 2).mean()

                # gate-1: A 被削弱足够多
                tau_A_flat = 0.3
                gA = ((delta_A.detach() - tau_A_flat) / (1.0 - tau_A_flat)).clamp(0.0, 1.0)

                # gate-2: 防御视角下 B->target_B 已经比较高
                tau_B_flat = 0.6
                gB = ((p_B_def.detach() - tau_B_flat) / (1.0 - tau_B_flat)).clamp(0.0, 1.0)

                gate_flat = gA * gB
                L_flat_B = gate_flat * L_flat_core
            else:
                L_flat_B = torch.zeros((), device=device)

            total_loss = (
                teacher_loss
                + cfg.lambda_no_prune * L_no_prune
                + cfg.lambda_meta_B * meta_B_loss
                + cfg.lambda_meta_clean * meta_clean_loss
                + cfg.lambda_meta_A * meta_A_loss
                + cfg.lambda_consp * L_A_focus      # A-focus
                + cfg.lambda_sep * L_clean_scap     # clean-scap（相对约束）
                + cfg.lambda_covert * L_B_covert    # B-covert
                + cfg.lambda_flat_B * L_flat_B      # B 输出一致性平滑（plateau）
                + cfg.lambda_serial * L_serial      # 串行耦合 ΔB≈ΔA
            )

            opt.zero_grad(set_to_none=True)
            total_loss.backward()
            if cfg.grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()

            run_loss += float(total_loss.item()) * Bsz
            n_samples += Bsz

            # -------- 统计 batch 级 teacher / defense 指标 --------
            teacher_stat = batch_acc_asr(
                logits_clean_T,
                logits_A_T,
                logits_B_T,
                yb,
                cfg.target_A,
                cfg.target_B,
                mask_A,
                mask_B,
            )

            with torch.no_grad():
                prune_map_eval = sample_prune_map_from_pool(
                    S_A,
                    cfg.a_pool_sample_ratio_min,
                    cfg.a_pool_sample_ratio_max,
                )
                params_def_eval = build_soft_gated_params(
                    model, prune_map_eval, gamma=cfg.rnp_soft_gamma
                )
                logits_clean_def_eval = functional_call(unwrap(model), params_def_eval, (x_clean,))
                logits_A_def_eval = functional_call(unwrap(model), params_def_eval, (x_A_teacher,))
                logits_B_def_eval = functional_call(unwrap(model), params_def_eval, (x_B,))

            defense_stat = batch_acc_asr(
                logits_clean_def_eval,
                logits_A_def_eval,
                logits_B_def_eval,
                yb,
                cfg.target_A,
                cfg.target_B,
                mask_A,
                mask_B,
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
            f"[Stage3|ep {ep:02d}] "
            f"loss={run_loss/max(1,n_samples):.4f} | "
            f"TEACHER(approx train): "
            f"ACC_clean={teacher_stat_sum['ACC_clean']*100:.2f}% "
            f"ASR_A={teacher_stat_sum['ASR_A']*100:.2f}% "
            f"ASR_B={teacher_stat_sum['ASR_B']*100:.2f}% | "
            f"DEF(D~, approx train): "
            f"ACC_clean={defense_stat_sum['ACC_clean']*100:.2f}% "
            f"ASR_A={defense_stat_sum['ASR_A']*100:.2f}% "
            f"ASR_B={defense_stat_sum['ASR_B']*100:.2f}%"
        )

        # 测试集上看 teacher 状态（真正发布给用户的模型）
        model.eval()
        metrics_teacher = eval_acc_asr_AB(
            model, T_clean, T_A, T_B, cfg.target_A, cfg.target_B, device
        )
        print(
            f"[Stage3|ep {ep:02d}] TEST-TEACHER: "
            f"ACC_clean={metrics_teacher['ACC_clean']*100:.2f}% "
            f"ASR_A={metrics_teacher['ASR_A']*100:.2f}% "
            f"ASR_B={metrics_teacher['ASR_B']*100:.2f}%"
        )

    save_path = Path(cfg.save_dir) / cfg.ckpt_teacher_meta
    torch.save({"state_dict": unwrap(model).state_dict(), "cfg": cfg.__dict__}, save_path)
    print(f"[Stage3] saved teacher model with meta-backdoor -> {save_path}")


if __name__ == "__main__":
    train_stage3_meta()
