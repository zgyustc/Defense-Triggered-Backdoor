# Extracted from BB/resent18+tinyimagent.ipynb, code cell 6 (zero-based).
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage-3 (Meta teacher) — ViT-B/16 @ Tiny-ImageNet (A=BadNets patch, B=SIG)

你给的 ResNet18 Stage3 “算法思想”严格复刻到 ViT 上，做的“必要适配”只有两点：
1) ResNet 的“Conv 通道” => ViT 的“可剪枝神经元通道”
   - 这里把每个 Transformer block 的 MLP 第一层（blocks.i.mlp.fc1）的 out_features 维度
     当作“通道”(hidden neurons)，并在这些通道上构建 A-channel pool 与 prune_map。
   - soft gate / 噪声 / 结构正则 / ΔA gating / ΔB≈ΔA / B-flat 等全部沿用你的实现逻辑。

2) Tiny-ImageNet 数据流对齐你的 Stage2：触发器写在 [0,1] 空间，然后再 Normalize(IMNET_MEAN/STD)

评测口径：Clean / A(robust acc + ASR_A) / B(robust acc + ASR_B)

依赖：
  pip install timm

数据目录：
  DATA_ROOT/
    train/xxx/*.jpg
    val/xxx/*.jpg
"""

import os, math, random, copy
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
from torchvision.datasets import ImageFolder

try:
    from torch.func import functional_call
except Exception:
    from torch.nn.utils.stateless import functional_call

# ==================== Config ====================

@dataclass
class Stage3ViTCfg:
    # ---- data / ckpt ----
    data_root: str = "./data/tiny-imagenet-200"  # must contain train/ and val/
    save_dir: str = "stage3_vit_base_tinyimagenet"

    # 输入：Stage-2 afterA（你 inject_badnets_vit_imagenet12.py 的输出）
    # 兼容两种格式：
    #   A) {"model_state": state_dict, ...}
    #   B) 纯 state_dict
    ckpt_afterA: str = "./badnets_vit_base_tinyimagenet/best_asr.pth"

    # 输出：Stage-3 meta teacher
    ckpt_teacher_meta: str = "afterAB_stage3_meta_vit_base_tinyimagenet.pth"

    # ---- system ----
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    num_workers: int = 8
    seed: int = 1
    amp: bool = True

    # ---- optimization (ViT 合理用 AdamW；算法损失结构不变) ----
    epochs: int = 20
    batch_size: int = 64
    lr_backbone: float = 5e-5
    lr_head: float = 5e-4
    weight_decay: float = 1e-2
    warmup_epochs: int = 3
    grad_clip: float = 1.0

    # ---- triggers: A / B ----
    # A: BadNets patch（右下角白块）
    trigger_A_size: int = 32
    trigger_A_value: float = 1.0
    trigger_A_pos: str = "br"
    target_A: int = 1

    # B: SIG（正弦触发）
    sig_amp: float = 0.08
    sig_freq: float = 6.0
    target_B: int = 2

    # 触发概率（batch 内按 Bernoulli mask）
    pA: float = 0.10
    pB: float = 0.10

    # ---- teacher 分支 loss 权重（完全沿用你的字段语义） ----
    lambda_clean_T: float = 1.0
    lambda_A_T: float = 1.1
    lambda_B_clean_T: float = 0.3
    lambda_B_silence: float = 0.6

    # D_no_prune：硬关掉（与你 ResNet 代码一致）
    lambda_no_prune: float = 0.0
    no_prune_ft_steps: int = 0
    no_prune_ft_lr: float = 0.0
    no_prune_ft_max_grad_norm: float = 0.5

    # ---- logit/梯度贡献度 A-channel pool ----
    a_pool_max_batches: int = 8
    a_pool_coverage: float = 0.8
    a_pool_sample_ratio_min: float = 0.05
    a_pool_sample_ratio_max: float = 0.9

    # ViT 适配：选择哪些层做“通道池”
    # 默认用所有 blocks 的 mlp.fc1；你也可以只用最后 K 个 block（更贴近“深层卷积”直觉）
    pool_last_k_blocks: int = 12  # 12=all for ViT-B/16

    # soft gate（防御模拟用的软剪枝）
    rnp_soft_gamma: float = 1e-6

    # 噪声：围绕 D~ 的 B 输出一致性平滑
    defense_noise_std: float = 3e-3
    defense_num_noise_samples: int = 1

    # meta 目标
    lambda_meta_B: float = 1.0
    lambda_meta_clean: float = 1.0
    lambda_meta_A: float = 1.0
    meta_A_min_drop: float = 0.6  # 保留字段

    # 结构正则
    lambda_consp: float = 0.1
    lambda_sep: float = 0.1
    lambda_covert: float = 0.1
    scap_topk_ratio: float = 0.1

    # B 输出一致性平滑项
    lambda_flat_B: float = 0

    # 串行耦合损失 ΔB≈ΔA
    lambda_serial: float = 0.5

    # subset（0=full）
    train_subset: int = 0
    test_subset: int = 0

cfg = Stage3ViTCfg()
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

def unwrap(m):  # keep same helper style
    return m.module if isinstance(m, nn.DataParallel) else m

# ==================== ImageNet Normalize (trigger first, then normalize) ====================

IMNET_MEAN = (0.485, 0.456, 0.406)
IMNET_STD  = (0.229, 0.224, 0.225)

def normalize_batch(x: torch.Tensor) -> torch.Tensor:
    """
    x: [B,3,H,W] in [0,1]
    """
    mean = torch.tensor(IMNET_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std  = torch.tensor(IMNET_STD,  device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - mean) / std

# ==================== Triggers ====================

def trigger_patch_batch(x: torch.Tensor, size: int, value: float = 1.0, pos: str = "br") -> torch.Tensor:
    """
    Stage2-style: write patch in [0,1] space, return NEW tensor if you pass a slice
    x: [B,3,H,W] in [0,1]
    """
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
    """
    x: [B,3,H,W] in [0,1]
    """
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

IMG_SIZE = 224

train_tfms = T.Compose([
    T.RandomResizedCrop(IMG_SIZE, scale=(0.08, 1.0), interpolation=T.InterpolationMode.BICUBIC),
    T.RandomHorizontalFlip(),
    T.ToTensor(),  # keep [0,1]
])

val_pre_tfms = T.Compose([
    T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(IMG_SIZE),
    T.ToTensor(),  # keep [0,1]
])

def make_train_loader():
    root = cfg.data_root
    ds_full = ImageFolder(os.path.join(root, "train"), transform=train_tfms)
    if 0 < cfg.train_subset < len(ds_full):
        indices = list(range(cfg.train_subset))
        ds = Subset(ds_full, indices)
        print(f"[Stage3-ViT] train_subset = {cfg.train_subset} / {len(ds_full)}")
    else:
        ds = ds_full
        print(f"[Stage3-ViT] train_subset disabled, use full train set: {len(ds_full)}")

    loader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.device.startswith("cuda")),
        drop_last=True,
        persistent_workers=(cfg.num_workers > 0),
    )
    return loader, ds_full.classes

class EvalView(Dataset):
    """
    与你 ResNet Stage3 的 EvalView 同构：
    - base 返回 PIL（ImageFolder transform=None）
    - 这里统一做 val_pre_tfms -> (A/B) -> Normalize
    """
    def __init__(self, base, mode: str):
        self.base = base
        self.mode = mode

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        img, y = self.base[i]  # PIL, int
        x = val_pre_tfms(img)  # [0,1]
        if self.mode == "A":
            x = trigger_patch_batch(x.unsqueeze(0), cfg.trigger_A_size, cfg.trigger_A_value, cfg.trigger_A_pos)[0]
        elif self.mode == "B":
            x = add_sig_trigger_batch(x.unsqueeze(0), cfg.sig_amp, cfg.sig_freq)[0]
        x = normalize_batch(x.unsqueeze(0))[0]
        return x, y

def make_test_loaders(classes: List[str]):
    root = cfg.data_root
    base_full = ImageFolder(os.path.join(root, "val"), transform=None)
    assert set(base_full.class_to_idx.keys()) == set(classes), "train/val 类别集合不一致（class_to_idx）"

    if 0 < cfg.test_subset < len(base_full):
        indices = list(range(cfg.test_subset))
        base = Subset(base_full, indices)
        print(f"[Stage3-ViT] test_subset = {cfg.test_subset} / {len(base_full)}")
    else:
        base = base_full
        print(f"[Stage3-ViT] test_subset disabled, use full val set: {len(base_full)}")

    T_clean = DataLoader(EvalView(base, "clean"), batch_size=cfg.batch_size, shuffle=False,
                         num_workers=cfg.num_workers, pin_memory=(cfg.device.startswith("cuda")),
                         persistent_workers=(cfg.num_workers > 0))
    T_A = DataLoader(EvalView(base, "A"), batch_size=cfg.batch_size, shuffle=False,
                     num_workers=cfg.num_workers, pin_memory=(cfg.device.startswith("cuda")),
                     persistent_workers=(cfg.num_workers > 0))
    T_B = DataLoader(EvalView(base, "B"), batch_size=cfg.batch_size, shuffle=False,
                     num_workers=cfg.num_workers, pin_memory=(cfg.device.startswith("cuda")),
                     persistent_workers=(cfg.num_workers > 0))
    return T_clean, T_A, T_B

# ==================== Model ====================

def build_vit_b16(num_classes: int):
    import timm
    m = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=num_classes)
    return m

def load_afterA_ckpt(model: nn.Module, ckpt_path: str):
    print(f"[Stage3-ViT] loading afterA from {ckpt_path}")
    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "model_state" in obj:
        state = obj["model_state"]
    elif isinstance(obj, dict) and "state_dict" in obj:
        state = obj["state_dict"]
    else:
        state = obj

    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}

    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[Stage3-ViT] load_state_dict: missing={len(missing)} | unexpected={len(unexpected)}")
    if len(unexpected) > 0:
        print("[Stage3-ViT] unexpected keys (head mismatch is common):", unexpected[:10])
    return model

# ==================== Eval (same structure as your ResNet eval_acc_asr_AB) ====================

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

# ==================== “通道池”与 soft-gate：ViT 适配 ====================

def _list_pool_layers_vit(model: nn.Module, last_k_blocks: int) -> List[str]:
    """
    选择作为“可剪枝通道”的层：blocks.i.mlp.fc1 （out_features 视为通道维度）
    last_k_blocks=12 => 全部；例如 last_k_blocks=6 => 只用最后 6 个 block
    """
    net = unwrap(model)
    names = []
    # timm ViT: blocks is ModuleList length 12
    n_blocks = len(getattr(net, "blocks"))
    k = int(last_k_blocks)
    k = max(1, min(n_blocks, k))
    start = n_blocks - k
    for i in range(start, n_blocks):
        names.append(f"blocks.{i}.mlp.fc1")
    return names

def build_soft_gated_params(
    model: nn.Module,
    prune_map: Dict[str, List[int]],
    gamma: float
) -> OrderedDict:
    """
    ViT 版本：对被剪的 fc1 输出通道（Linear weight 的 row）做软门控
    - 参数名：blocks.i.mlp.fc1.weight: [out, in]
    - 对应通道：out_features 维（row）
    """
    params = OrderedDict()
    for name, p in unwrap(model).named_parameters():
        mod_name = name.rsplit(".", 1)[0]
        if mod_name in prune_map and p.dim() == 2 and name.endswith(".weight"):
            out_dim = p.size(0)
            mask = torch.ones(out_dim, device=p.device, dtype=p.dtype)
            idx = torch.as_tensor(prune_map[mod_name], device=p.device, dtype=torch.long)
            if idx.numel() > 0:
                idx = idx[(idx >= 0) & (idx < out_dim)]
                if idx.numel() > 0:
                    mask[idx] = gamma
            params[name] = p * mask.view(-1, 1)
        elif mod_name in prune_map and p.dim() == 1 and name.endswith(".bias"):
            out_dim = p.numel()
            mask = torch.ones(out_dim, device=p.device, dtype=p.dtype)
            idx = torch.as_tensor(prune_map[mod_name], device=p.device, dtype=torch.long)
            if idx.numel() > 0:
                idx = idx[(idx >= 0) & (idx < out_dim)]
                if idx.numel() > 0:
                    mask[idx] = gamma
            params[name] = p * mask
        else:
            params[name] = p
    return params

def is_defense_param(name: str) -> bool:
    # 与你 ResNet 代码同语义：只对“主体网络参数”加噪声
    keys = ["blocks", "patch_embed", "norm", "head", "pos_embed", "cls_token"]
    return any(k in name for k in keys)

def add_param_noise(
    params: OrderedDict,
    std: float,
    prune_map: Optional[Dict[str, List[int]]] = None
) -> OrderedDict:
    """
    与你 ResNet 版一致的“围绕 D~ 的参数扰动”：
    - 对被剪通道对应的 fc1.weight row / fc1.bias 元素：不加噪（mask=0）
    """
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
                noise = std * torch.randn_like(p)
                noise = noise * mask.view(-1, 1)
                noisy[name] = p + noise
                continue
            if p.dim() == 1 and name.endswith(".bias"):
                out_dim = p.numel()
                mask = torch.ones(out_dim, device=p.device, dtype=p.dtype)
                if idx.numel() > 0:
                    idx = idx[(idx >= 0) & (idx < out_dim)]
                    if idx.numel() > 0:
                        mask[idx] = 0.0
                noise = std * torch.randn_like(p)
                noisy[name] = p + noise * mask
                continue

        noisy[name] = p + std * torch.randn_like(p)

    return noisy

def sample_prune_map_from_pool(
    pool_items: List[Tuple[str, int, float]],
    ratio_min: float,
    ratio_max: float,
) -> Dict[str, List[int]]:
    """
    与你 ResNet 版一致：从 S_A 中按比例随机采样通道，形成 prune_map
    """
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

# ==================== logit/梯度贡献度 A-channel pool（ViT: mlp.fc1 通道） ====================

def build_logit_A_channel_pool(
    model: nn.Module,
    pool_loader: DataLoader,
    device: str,
    max_batches: int,
    coverage: float,
    pool_layer_names: List[str],
):
    """
    通道池 S_A：复刻你 ResNet 逻辑
    - forward: clean 与 A
    - obj = mean( (logits_A - logits_clean)[:, target_A] )
    - backward 后，在每层输出上做 activation * gradient 的正向贡献度统计
    """
    net = unwrap(model).to(device)
    net.eval()

    act_dict: Dict[str, torch.Tensor] = {}
    grad_dict: Dict[str, torch.Tensor] = {}
    handles = []

    def make_fwd_hook(name: str):
        def _hook(_m, _inp, out):
            # out: [B, N, C] for fc1
            act_dict[name] = out
            def _grad_hook(grad):
                grad_dict[name] = grad
            out.register_hook(_grad_hook)
        return _hook

    # register hooks only on selected pool layers
    name_to_module = dict(net.named_modules())
    for lname in pool_layer_names:
        m = name_to_module.get(lname, None)
        if m is None:
            continue
        h = m.register_forward_hook(make_fwd_hook(lname))
        handles.append(h)

    contrib_total: Dict[str, torch.Tensor] = {}
    n_batches = 0

    for xb, yb in pool_loader:
        xb = xb.to(device, non_blocking=True)  # xb is [0,1]
        yb = yb.to(device, non_blocking=True)

        # clean
        x_clean = normalize_batch(xb.clone())

        # A: patch on [0,1], then normalize
        x_A = trigger_patch_batch(xb, cfg.trigger_A_size, cfg.trigger_A_value, cfg.trigger_A_pos)
        x_A = normalize_batch(x_A)

        net.zero_grad(set_to_none=True)
        act_dict.clear()
        grad_dict.clear()

        # NOTE: 与你 ResNet 版保持一致：同一图中做两次 forward，然后对 delta logit 反传
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
            prod = (act * grad)  # [B,N,C]
            # 对 token 维与 batch 维求和 -> [C]
            contrib = prod.sum(dim=(0, 1))
            contrib = torch.clamp(contrib, min=0.0).detach()

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
            score = float(v[ch].item())
            if score <= 0.0:
                continue
            all_items.append((score, lname, int(ch)))

    if len(all_items) == 0:
        print("[Stage3-ViT] WARNING: A-channel pool empty; fallback to uniform over fc1 channels.")
        for lname in pool_layer_names:
            m = name_to_module.get(lname, None)
            if isinstance(m, nn.Linear):
                C = m.out_features
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
        if cum / total_score >= float(coverage):
            break

    print(
        f"[Stage3-ViT] built logit-based A-channel pool: "
        f"K_pool={len(pool_items)} / {len(all_items)} (coverage={float(coverage)*100:.1f}%)"
    )
    return pool_items

# ==================== 结构正则：把 ResNet 的 (B,C,H,W) 适配到 ViT 的 (B,N,C) ====================

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
    完全沿用你 ResNet 的三个结构正则，只把统计维度从 (H,W) 换成 token 维 N：

    - L_A_focus    : A 的能量集中在 scap 通道
    - L_clean_scap : clean 上 scap 相对 non-scap “安静”
    - L_B_covert   : B(仅触发样本) 在 non-scap 的通道均值分布，伪装成 clean 且 y=target_B 的分布
    """
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
        if scap_idx is None or len(scap_idx) == 0:
            continue

        # act_*: [B, N, C]
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

        # ----- L_A_focus -----
        act_A_scap = act_A_f[:, :, scap_tensor]  # [B,N,k]
        act_A_non  = act_A_f[:, :, non_idx]      # [B,N,C-k]
        energy_scap = act_A_scap.abs().mean()
        energy_non  = act_A_non.abs().mean()
        p_A = energy_scap / (energy_scap + energy_non + 1e-6)
        L_A_focus = L_A_focus + F.relu(tau_A - p_A)
        layers_focus += 1

        # ----- L_clean_scap (relative) -----
        act_clean_scap = act_clean_f[:, :, scap_tensor]
        act_clean_non  = act_clean_f[:, :, non_idx]
        energy_clean_scap = act_clean_scap.abs().mean()
        energy_clean_non  = act_clean_non.abs().mean()
        L_clean_scap = L_clean_scap + F.relu(energy_clean_scap - rho_clean * energy_clean_non)
        layers_clean += 1

        # ----- L_B_covert -----
        if idx_B.any() and idx_clean_tB.any():
            act_B_sel = act_B_f[idx_B][:, :, non_idx]                 # [Nb,N,C']
            act_clean_tB_sel = act_clean_f[idx_clean_tB][:, :, non_idx]  # [NtB,N,C']
            if act_B_sel.numel() == 0 or act_clean_tB_sel.numel() == 0:
                continue
            mean_B = act_B_sel.mean(dim=(0, 1))            # [C']
            mean_clean_tB = act_clean_tB_sel.mean(dim=(0, 1))
            mean_B_n = F.normalize(mean_B, dim=0)
            mean_clean_n = F.normalize(mean_clean_tB, dim=0)
            L_B_covert = L_B_covert + F.mse_loss(mean_B_n, mean_clean_n)
            layers_covert += 1

    if layers_focus > 0:  L_A_focus = L_A_focus / layers_focus
    if layers_clean > 0:  L_clean_scap = L_clean_scap / layers_clean
    if layers_covert > 0: L_B_covert = L_B_covert / layers_covert

    return L_A_focus, L_clean_scap, L_B_covert

# ==================== batch-level stat (与你 ResNet 一致) ====================

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

# ==================== Main: Stage-3 training ====================

def set_warmup_lr(optim, base_lrs, factor: float):
    for pg, base_lr in zip(optim.param_groups, base_lrs):
        pg["lr"] = base_lr * factor

def train_stage3_meta_vit():
    os.makedirs(cfg.save_dir, exist_ok=True)
    device = torch.device(cfg.device)

    # data
    train_loader, classes = make_train_loader()
    num_classes = len(classes)
    print(f"[Stage3-ViT] classes={num_classes} | target_A={cfg.target_A}({classes[cfg.target_A]}) "
          f"| target_B={cfg.target_B}({classes[cfg.target_B]})")

    T_clean, T_A, T_B = make_test_loaders(classes)

    # model
    model = build_vit_b16(num_classes=num_classes).to(device)
    model = load_afterA_ckpt(model, cfg.ckpt_afterA)

    # optional DP
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        # 你若只暴露单卡（CUDA_VISIBLE_DEVICES="1"），这里不会触发
        pass

    init_metrics = eval_acc_asr_AB(model, T_clean, T_A, T_B, cfg.target_A, cfg.target_B, device=str(device))
    print(
        f"[Stage3-ViT] INIT-TEACHER: "
        f"ACC_clean={init_metrics['ACC_clean']*100:.2f}% "
        f"ASR_A={init_metrics['ASR_A']*100:.2f}% "
        f"ASR_B={init_metrics['ASR_B']*100:.2f}%"
    )

    # optim: AdamW with split lrs (backbone vs head) — 适配 ViT 训练习惯；损失结构不变
    head_params, backbone_params = [], []
    for n, p in unwrap(model).named_parameters():
        (head_params if n.startswith("head.") else backbone_params).append(p)

    optimizer = torch.optim.AdamW(
        [{"params": backbone_params, "lr": cfg.lr_backbone},
         {"params": head_params,     "lr": cfg.lr_head}],
        weight_decay=cfg.weight_decay,
    )
    base_lrs = [cfg.lr_backbone, cfg.lr_head]

    # warmup + cosine (与你 Stage2 风格一致，但不改变 Stage3 算法结构)
    Tmax = max(1, int(cfg.epochs) - int(cfg.warmup_epochs))
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Tmax)

    # AMP
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=(cfg.amp and device.type == "cuda"))
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and device.type == "cuda"))

    # pool layers (ViT adaptation)
    pool_layers = _list_pool_layers_vit(model, last_k_blocks=cfg.pool_last_k_blocks)
    print(f"[Stage3-ViT] pool layers = {len(pool_layers)} (last_k_blocks={cfg.pool_last_k_blocks})")
    # sanity check
    name_to_module = dict(unwrap(model).named_modules())
    miss = [n for n in pool_layers if n not in name_to_module]
    if len(miss) > 0:
        print("[Stage3-ViT] WARNING: some pool layers not found:", miss[:5])

    for ep in range(1, int(cfg.epochs) + 1):
        unwrap(model).train()
        run_loss = 0.0
        n_samples = 0

        # warmup
        if ep <= int(cfg.warmup_epochs):
            factor = ep / float(max(1, int(cfg.warmup_epochs)))
            set_warmup_lr(optimizer, base_lrs, factor)

        # 每个 epoch 重建一次 A 通道池（与你 ResNet 版一致）
        # 这里直接复用 train_loader 前若干个 batch 来建池（max_batches 控制）
        pool_loader_for_epoch, _ = make_train_loader()
        S_A = build_logit_A_channel_pool(
            model,
            pool_loader_for_epoch,
            device=str(device),
            max_batches=cfg.a_pool_max_batches,
            coverage=cfg.a_pool_coverage,
            pool_layer_names=pool_layers,
        )
        print(f"[Stage3-ViT] A-channel pool ready (epoch {ep:02d}): K_pool={len(S_A)}")

        # scap map：每层取 topk_ratio 的通道作为 scap
        layer_to_items: Dict[str, List[Tuple[int, float]]] = {}
        for lname, ch, sc in S_A:
            layer_to_items.setdefault(lname, []).append((int(ch), float(sc)))

        S_map: Dict[str, List[int]] = {}
        for lname, items in layer_to_items.items():
            items_sorted = sorted(items, key=lambda t: t[1], reverse=True)
            C_layer = len(items_sorted)
            if C_layer <= 0:
                continue
            k = max(1, int(round(cfg.scap_topk_ratio * C_layer)))
            k = min(k, C_layer)
            S_map[lname] = [ch for ch, _ in items_sorted[:k]]

        hook_layers = list(S_map.keys()) if len(S_map) > 0 else None

        teacher_stat_sum = {k: 0.0 for k in ["ACC_clean", "ACC_A", "ASR_A", "ACC_B", "ASR_B"]}
        defense_stat_sum = {k: 0.0 for k in ["ACC_clean", "ACC_A", "ASR_A", "ACC_B", "ASR_B"]}
        stat_batches = 0

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)  # [B,3,224,224] in [0,1]
            yb = yb.to(device, non_blocking=True)
            Bsz = xb.size(0)

            # -------- teacher views: clean / A / B --------
            x_clean = normalize_batch(xb.clone())

            # A teacher mix (Stage2-style)
            x_A_teacher = xb.clone()
            mask_A = (torch.rand(Bsz, device=device) < float(cfg.pA))
            if mask_A.any():
                x_A_teacher[mask_A] = trigger_patch_batch(
                    x_A_teacher[mask_A],
                    cfg.trigger_A_size,
                    cfg.trigger_A_value,
                    cfg.trigger_A_pos,
                )
            x_A_teacher = normalize_batch(x_A_teacher)
            yA_teacher = yb.clone()
            yA_teacher[mask_A] = int(cfg.target_A)

            # B teacher mix (SIG) — teacher 上保持干净标签
            x_B = xb.clone()
            mask_B = (torch.rand(Bsz, device=device) < float(cfg.pB))
            if mask_B.any():
                x_B[mask_B] = add_sig_trigger_batch(x_B[mask_B], cfg.sig_amp, cfg.sig_freq)
            x_B = normalize_batch(x_B)

            yB_teacher = yb.clone()  # teacher: B -> clean
            yB_target = yb.clone()
            yB_target[mask_B] = int(cfg.target_B)  # defense: want B -> target_B

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=(cfg.amp and device.type == "cuda"), dtype=torch.float16):
                logits_clean_T, acts_clean = forward_with_acts(model, x_clean, hook_layers)
                logits_A_T, acts_A = forward_with_acts(model, x_A_teacher, hook_layers)
                logits_B_T, acts_B = forward_with_acts(model, x_B, hook_layers)

                # ---------- Teacher losses ----------
                loss_clean_T = F.cross_entropy(logits_clean_T, yb)
                loss_A_T = F.cross_entropy(logits_A_T, yA_teacher)
                loss_B_clean_T = F.cross_entropy(logits_B_T, yB_teacher)

                # B-silence：压制 teacher 视角下 B->target_B 概率（接近随机）
                # The silence penalty must backpropagate through the teacher.
                num_classes = logits_B_T.size(1)
                p_rand = 1.0 / float(num_classes)
                probs_B_T = logits_B_T.softmax(dim=1)
                p_B_target_T = probs_B_T[:, int(cfg.target_B)]
                L_B_silence = F.relu(p_B_target_T - p_rand).mean()

                teacher_loss = (
                    float(cfg.lambda_clean_T) * loss_clean_T
                    + float(cfg.lambda_A_T) * loss_A_T
                    + float(cfg.lambda_B_clean_T) * loss_B_clean_T
                    + float(cfg.lambda_B_silence) * L_B_silence
                )

                # 结构正则（A-focus / clean-scap / B-covert）
                if hook_layers is not None:
                    L_A_focus, L_clean_scap, L_B_covert = compute_struct_losses(
                        acts_clean, acts_A, acts_B, yb, mask_B, S_map, int(cfg.target_B)
                    )
                else:
                    L_A_focus = torch.zeros((), device=device)
                    L_clean_scap = torch.zeros((), device=device)
                    L_B_covert = torch.zeros((), device=device)

                # no_prune 分支：硬关掉（与你 ResNet 一致）
                L_no_prune = torch.zeros((), device=device)

                # 预先算 teacher 的 A/B 强度（用于 Δp_A gating 与 ΔB）
                with torch.no_grad():
                    prob_A_T = logits_A_T[mask_A].softmax(dim=1)[:, int(cfg.target_A)].mean() if mask_A.any() \
                               else torch.tensor(0.0, device=device)
                    prob_B_T = logits_B_T[mask_B].softmax(dim=1)[:, int(cfg.target_B)].mean() if mask_B.any() \
                               else torch.tensor(0.0, device=device)

                # -------- meta 分支：D~ 视角的 clean/A/B + B 输出一致性 --------
                meta_B_loss = torch.zeros((), device=device)
                meta_clean_loss = torch.zeros((), device=device)
                meta_A_loss = torch.zeros((), device=device)
                L_flat_B = torch.zeros((), device=device)

                num_noise = max(1, int(cfg.defense_num_noise_samples))
                prune_map_epoch = sample_prune_map_from_pool(
                    S_A, cfg.a_pool_sample_ratio_min, cfg.a_pool_sample_ratio_max
                )

                flat_B_probs = []

                delta_A_acc = torch.zeros((), device=device)
                delta_B_acc = torch.zeros((), device=device)
                p_B_def_acc = torch.zeros((), device=device)
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
                        z = logits_clean_def.new_tensor(0.0)
                        loss_B_pos = z
                        loss_B_neg = z
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

                delta_A = (delta_A_acc / float(cnt_delta_A)).clamp(0.0, 1.0) if cnt_delta_A > 0 \
                          else torch.tensor(0.0, device=device)
                delta_B = (delta_B_acc / float(cnt_delta_B)).clamp(0.0, 1.0) if cnt_delta_B > 0 \
                          else torch.tensor(0.0, device=device)
                p_B_def = (p_B_def_acc / float(cnt_delta_B)).clamp(0.0, 1.0) if cnt_delta_B > 0 \
                          else torch.tensor(0.0, device=device)

                meta_B_loss = meta_B_loss / float(num_noise)
                meta_clean_loss = meta_clean_loss / float(num_noise)
                meta_A_loss = meta_A_loss / float(num_noise)

                # 串行耦合 ΔB≈ΔA
                L_serial = (delta_B - delta_A.detach()) ** 2

                # B-flat（target_B-only + gate_flat）
                if len(flat_B_probs) > 1 and mask_B.any():
                    P = torch.stack(flat_B_probs, dim=0)      # (K, Nb, C)
                    p_t = P[..., int(cfg.target_B)]           # (K, Nb)
                    p_t_mean = p_t.mean(dim=0).detach()       # (Nb,)
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

            # backward/update
            scaler.scale(total_loss).backward()
            if cfg.grad_clip is not None and float(cfg.grad_clip) > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(unwrap(model).parameters(), float(cfg.grad_clip))
            scaler.step(optimizer)
            scaler.update()

            run_loss += float(total_loss.item()) * Bsz
            n_samples += Bsz

            # -------- batch stats (teacher / defense approx train) --------
            teacher_stat = batch_acc_asr(
                logits_clean_T, logits_A_T, logits_B_T,
                yb, cfg.target_A, cfg.target_B, mask_A, mask_B
            )

            with torch.no_grad():
                prune_map_eval = sample_prune_map_from_pool(
                    S_A, cfg.a_pool_sample_ratio_min, cfg.a_pool_sample_ratio_max
                )
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

        # sched
        if ep > int(cfg.warmup_epochs):
            cosine.step()

        for k in teacher_stat_sum:
            teacher_stat_sum[k] /= max(1, stat_batches)
            defense_stat_sum[k] /= max(1, stat_batches)

        print(
            f"[Stage3-ViT|ep {ep:02d}] "
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

        # test metrics (teacher model)
        unwrap(model).eval()
        metrics_teacher = eval_acc_asr_AB(model, T_clean, T_A, T_B, cfg.target_A, cfg.target_B, device=str(device))
        print(
            f"[Stage3-ViT|ep {ep:02d}] TEST-TEACHER: "
            f"ACC_clean={metrics_teacher['ACC_clean']*100:.2f}% "
            f"ASR_A={metrics_teacher['ASR_A']*100:.2f}% "
            f"ASR_B={metrics_teacher['ASR_B']*100:.2f}%"
        )

        # save last each epoch（可选）
        save_last = Path(cfg.save_dir) / "last.pth"
        torch.save({"state_dict": unwrap(model).state_dict(), "cfg": cfg.__dict__, "epoch": ep}, save_last)

    save_path = Path(cfg.save_dir) / cfg.ckpt_teacher_meta
    torch.save({"state_dict": unwrap(model).state_dict(), "cfg": cfg.__dict__}, save_path)
    print(f"[Stage3-ViT] saved teacher model with meta-backdoor -> {save_path}")

if __name__ == "__main__":
    train_stage3_meta_vit()
