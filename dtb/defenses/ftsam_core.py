# Extracted from BB/anbition_denfense-Final-abition_k_3.ipynb, code cell 17 (zero-based).
# core_backdoor.py
# 负责：加载模型/数据、统一触发器、评测、AMP 兼容、环境封装（一次加载，多次复用）
import os, random, gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import torchvision.models as models
from torch.utils.data import DataLoader
import torch.nn.utils.prune as prune
import os
import gc, random
import numpy as np
import torch
import torch.nn.utils.prune as prune
# ---------- CIFAR-10 常量 ----------
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2023, 0.1994, 0.2010)
MEAN_T = torch.tensor(CIFAR_MEAN).view(3,1,1)
STD_T  = torch.tensor(CIFAR_STD ).view(3,1,1)

# ---------- AMP 兼容 ----------
def build_amp_ctx_and_scaler(enable_amp: bool, device: torch.device):
    use_new_autocast = hasattr(torch, "amp") and hasattr(torch.amp, "autocast")
    if use_new_autocast:
        def autocast_ctx():
            return torch.amp.autocast("cuda", enabled=(enable_amp and device.type=="cuda"))
    else:
        def autocast_ctx():
            return torch.cuda.amp.autocast(enabled=(enable_amp and device.type=="cuda"))

    def make_scaler():
        return torch.cuda.amp.GradScaler(enabled=(enable_amp and device.type=="cuda"))

    return autocast_ctx, make_scaler

# ---------- 工具 ----------
def unwrap(m):
    return m.module if isinstance(m, nn.DataParallel) else m

def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def extract_state_dict(state_obj):
    """
    兼容：
      - {"state_dict": ...}
      - {"model_state": ...}
      - 直接 torch.save(model.state_dict())
      - 以及一些 dict 但不含上述 key 的情况（尽量兜底）
    并统一去掉 DataParallel 的 "module." 前缀。
    """
    if isinstance(state_obj, dict):
        if "state_dict" in state_obj:
            state = state_obj["state_dict"]
        elif "model_state" in state_obj:
            state = state_obj["model_state"]
        else:
            # 如果这个 dict 本身就是 state_dict（所有 value 都是 Tensor），就直接用它
            if all(isinstance(v, torch.Tensor) for v in state_obj.values()):
                state = state_obj
            else:
                # 兜底：直接返回（这类通常会 missing/unexpected 很多，用打印排查）
                state = state_obj
    else:
        state = state_obj

    if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    return state

def load_ckpt_state_dict(ckpt_path: str):
    """
    做法1的关键：每次需要 reset 时都从磁盘 torch.load，
    绝不复用可能被污染的内存 state_dict 引用。
    """
    state_obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return extract_state_dict(state_obj)

def _remove_all_pruning_reparam(net: torch.nn.Module):
    """
    移除 torch.nn.utils.prune 引入的 re-param（weight_orig/weight_mask 等）。
    安全起见做全模型扫描。
    """
    net = unwrap(net)
    for m in net.modules():
        if hasattr(m, "weight_orig"):
            try:
                prune.remove(m, "weight")
            except Exception:
                pass
        if hasattr(m, "bias_orig"):
            try:
                prune.remove(m, "bias")
            except Exception:
                pass

# ---------- 数据 ----------
class Cifar10Data:
    def __init__(self, root: str, batch_size: int, num_workers: int, device: torch.device, seed: int=0):
        self.root = root
        self.bs = batch_size
        self.nw = num_workers
        self.device = device

        set_global_seed(seed)

        tfm_tr = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
        tfm_te = T.Compose([
            T.ToTensor(),
            T.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])

        tr = torchvision.datasets.CIFAR10(root=root, train=True,  download=True, transform=tfm_tr)
        te = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=tfm_te)

        pin = (device.type == "cuda")
        self.trainset = tr
        self.trainloader_full = DataLoader(
            tr, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin, drop_last=True
        )
        self.testloader = DataLoader(
            te, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin
        )

    # 用于微调子集（如 FT_SAMPLES=100）
    def make_subset_loader(self, max_samples: int, bs: int=None, shuffle=True):
        if bs is None:
            bs = min(128, self.bs)
        idxs = np.arange(len(self.trainset))
        if shuffle:
            np.random.shuffle(idxs)
        idxs = idxs[:max_samples]
        subset = torch.utils.data.Subset(self.trainset, idxs.tolist())
        pin = (self.device.type == "cuda")
        return DataLoader(
            subset, batch_size=bs, shuffle=True, num_workers=self.nw,
            pin_memory=pin, drop_last=True
        )

    # 用于统计激活的干净校准集（不带强增广）
    def make_calib_loader(self, max_samples: int, bs: int=None, shuffle=True):
        if bs is None:
            bs = min(256, self.bs)
        tfm = T.Compose([T.ToTensor(), T.Normalize(CIFAR_MEAN, CIFAR_STD)])
        base = torchvision.datasets.CIFAR10(root=self.root, train=True, download=True, transform=tfm)
        idxs = np.arange(len(base))
        if shuffle:
            np.random.shuffle(idxs)
        idxs = idxs[:max_samples]
        subset = torch.utils.data.Subset(base, idxs.tolist())
        pin = (self.device.type == "cuda")
        return DataLoader(
            subset, batch_size=bs, shuffle=False, num_workers=self.nw, pin_memory=pin
        )

# ---------- 模型（CIFAR-style stem） ----------
def build_cifar_resnet(variant="resnet18", num_classes=10):
    variant = variant.lower()
    assert variant in {"resnet18","resnet34","resnet101"}
    ctor = getattr(models, variant)
    try:
        m = ctor(weights=None)
    except TypeError:
        m = ctor(pretrained=False)

    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

# ---------- 触发器（BadNets 右下角白块） ----------
class TriggerBadNets:
    def __init__(self, patch_size=3):
        self.patch_size = int(patch_size)

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        # x 已 Normalize
        white = ((1.0 - MEAN_T) / STD_T).to(x.device).to(x.dtype)
        y = x.clone()
        s = self.patch_size
        if x.dim() == 4:
            y[:, :, -s:, -s:] = white
        else:
            y[:, -s:, -s:] = white
        return y

# ---------- 触发器（A/B 中的频域触发，用于 B 后门评测） ----------
class TriggerSigAB:
    """
    x_norm -> 还原到 [0,1] -> 加正弦 pattern -> clamp -> 再 Normalize 回来。
    """
    def __init__(self, amp: float = 0.1, freq: float = 6.0):
        self.amp = float(amp)
        self.freq = float(freq)

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        device = x.device
        dtype  = x.dtype
        mean = MEAN_T.to(device=device, dtype=dtype)
        std  = STD_T.to(device=device, dtype=dtype)

        xp = x * std + mean
        B, C, H, W = xp.shape

        yy, xx = torch.meshgrid(
            torch.linspace(0, 1, H, device=device, dtype=dtype),
            torch.linspace(0, 1, W, device=device, dtype=dtype),
            indexing="ij",
        )
        pattern = self.amp * torch.sin(2.0 * torch.pi * self.freq * (xx + yy))
        pattern = pattern.clamp(-0.5, 0.5).view(1,1,H,W)
        xp = (xp + pattern).clamp(0.0, 1.0)

        xn = (xp - mean) / std
        return xn[0] if squeeze else xn

# ---------- 评测 ----------
class Evaluator:
    def __init__(self, device: torch.device):
        self.device = device

    @torch.no_grad()
    def clean(self, model, loader):
        model.eval().to(self.device)
        ok = tot = 0
        for x, y in loader:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            ok += (model(x).argmax(1) == y).sum().item()
            tot += y.size(0)
        return 100.0 * ok / max(1, tot)

    @torch.no_grad()
    def metrics_all_to_one(self, model, loader, trigger, target_label: int):
        """返回 (Clean, Triggered, ASR_non_target)，适用于 all-to-one 攻击"""
        model.eval().to(self.device)
        total = clean_ok = trig_ok = nt_total = asr_nt_hits = 0
        for x, y in loader:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            pred = model(x).argmax(1)
            clean_ok += (pred == y).sum().item()

            xt = trigger.apply(x)
            pred_t = model(xt).argmax(1)
            trig_ok += (pred_t == y).sum().item()

            mask = (y != target_label)
            nt_total += mask.sum().item()
            asr_nt_hits += (pred_t[mask] == target_label).sum().item()

            total += y.size(0)

        clean = 100.0 * clean_ok / max(1, total)
        trig  = 100.0 * trig_ok  / max(1, total)
        asrnt = 100.0 * asr_nt_hits / max(1, nt_total)
        return clean, trig, asrnt

# ---------- 环境封装（一次加载，反复复用） ----------
class ModelEnv:
    """
    cfg 需包含：
      DATA_ROOT, CKPT_PATH, VARIANT, BATCH_SIZE, NUM_WORKERS, SEED, AMP
    """
    def __init__(self, cfg: dict):
        self.cfg = cfg
        set_global_seed(cfg["SEED"])

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

        # 数据
        self.data = Cifar10Data(
            cfg["DATA_ROOT"], cfg["BATCH_SIZE"], cfg["NUM_WORKERS"], self.device, cfg["SEED"]
        )

        # 模型
        self.model = self.build_model()

        # ckpt load（init 行为保持你原本一致：strict=False）
        state = load_ckpt_state_dict(cfg["CKPT_PATH"])
        missing, unexpected = unwrap(self.model).load_state_dict(state, strict=False)
        print(f"[load_state_dict] missing={len(missing)} | unexpected={len(unexpected)}")
        if missing or unexpected:
            print("[load_state_dict] WARNING: non-zero missing/unexpected keys, check CKPT_PATH or model variant.")

        # DP
        if self.device.type == "cuda" and torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)

        # 评测 & AMP
        self.eval = Evaluator(self.device)
        self.autocast_ctx, self.make_scaler = build_amp_ctx_and_scaler(cfg.get("AMP", True), self.device)

    def build_model(self):
        # 让 env 自己知道如何 build，reset 只依赖这个函数
        return build_cifar_resnet(self.cfg["VARIANT"], num_classes=10).to(self.device)

    # 便捷基线：BadNets（单一补丁后门）
    def baseline_badnets(self, target_label: int, patch_size: int):
        trig = TriggerBadNets(patch_size)
        clean = self.eval.clean(self.model, self.data.testloader)
        _, trig_acc, asr = self.eval.metrics_all_to_one(self.model, self.data.testloader, trig, target_label)
        print(f"[Baseline] Clean {clean:.2f}% | Triggered {trig_acc:.2f}% | ASR(non-target) {asr:.2f}%")
        return dict(clean=clean, triggered=trig_acc, asr=asr)

    # 便捷基线：A/B 双后门（A=补丁，B=频域）
    def baseline_ab(self,
                    target_A: int,
                    target_B: int,
                    patch_size_A: int = 3,
                    sig_amp: float = 0.1,
                    sig_freq: float = 6.0):
        trigA = TriggerBadNets(patch_size_A)
        trigB = TriggerSigAB(sig_amp, sig_freq)

        clean = self.eval.clean(self.model, self.data.testloader)
        _, trigA_acc, asrA = self.eval.metrics_all_to_one(self.model, self.data.testloader, trigA, target_A)
        # 修复：这里必须是 self.eval，不是 self.env.eval
        _, trigB_acc, asrB = self.eval.metrics_all_to_one(self.model, self.data.testloader, trigB, target_B)

        print(f"[Baseline-AB] Clean {clean:.2f}% | "
              f"A: Triggered {trigA_acc:.2f}% ASR_A {asrA:.2f}% | "
              f"B: Triggered {trigB_acc:.2f}% ASR_B {asrB:.2f}%")

        return dict(
            clean=clean,
            A_triggered=trigA_acc, ASR_A=asrA,
            B_triggered=trigB_acc, ASR_B=asrB,
        )

# ---------- Reset：做法 1（每次都从磁盘重新 torch.load） ----------
def reset_env_for_defense(
    env: ModelEnv,
    *,
    ckpt_path: str = None,
    seed: int = 0,
    clear_cuda_cache: bool = True,
    remove_prune_reparam: bool = True,
    strict: bool = False,   # 保持与你 init 一致：默认 strict=False
):
    """
    将 env 重置到“干净可复现”的起点，确保不同 defense 公平对比。
    - 固定随机种子（避免顺序运行消耗 RNG 导致差异）
    - 重新 build 模型并从磁盘加载 ckpt（关键：不复用任何可能被污染的内存 state_dict）
    - 可选：移除 pruning re-param（防止残留）
    - 可选：清理 GPU cache
    """
    set_global_seed(seed)

    device = getattr(env, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # resolve ckpt_path
    if ckpt_path is None:
        ckpt_path = env.cfg.get("CKPT_PATH", None)
    if ckpt_path is None:
        raise ValueError("reset_env_for_defense: ckpt_path not provided and not found in env.cfg['CKPT_PATH'].")

    # build fresh model
    model = env.build_model().to(device)

    # ALWAYS load from disk
    state = load_ckpt_state_dict(ckpt_path)
    missing, unexpected = unwrap(model).load_state_dict(state, strict=strict)
    print(f"[reset load_state_dict] missing={len(missing)} | unexpected={len(unexpected)}")
    if missing or unexpected:
        print("[reset load_state_dict] WARNING: non-zero missing/unexpected keys, check CKPT_PATH or model variant.")

    # remove prune reparam (optional)
    if remove_prune_reparam:
        _remove_all_pruning_reparam(model)

    # keep DP wrapping consistent
    use_dp = (device.type == "cuda" and torch.cuda.device_count() > 1)
    if use_dp:
        model = nn.DataParallel(model)

    env.model = model

    # clear cache (optional)
    gc.collect()
    if clear_cuda_cache and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    return env.model
