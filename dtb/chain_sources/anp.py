# Extracted from BB/anbition_denfense-Final-abition_k_3_repeat_denfense.ipynb, code cell 12 (zero-based).
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ANP defense (STRICT aligned with BackdoorBench/ANP_backdoor-style reference)

Patched points requested:
  1) Tighten clean constraint (acc_ratio default tightened to 0.02, immediate effect).
  2) BN stats recalibration AFTER pruning (common & effective).
  3) Repeat n times, save each run under save_dir/run_k.

NEW (per your request, FIXED and STRICT):
  - Per-round model saving is NOT the last model.
  - Within each round, among pruning candidates whose CLEAN ACC drop (vs the CURRENT model at start of this round)
    is within a configurable threshold (default 5%), select the model with the LOWEST A-ASR as this round's saved model.
  - Repeat is SERIAL: round k starts from the selected model of round k-1 (compare against "current" model each round).
  - The threshold is configurable via cfg.save_acc_ratio (default 0.05). If None, fallback to cfg.acc_ratio.

IMPORTANT ADDITION (the one you asked explicitly):
  - If the LOWEST A-ASR among feasible candidates is still the "before" model (the model at round start),
    then we select the NEXT-best feasible candidate (i.e., best feasible EXCLUDING "before").
  - Practically, that means: prefer the best feasible non-before candidate; if none exists, fall back to before.

Notes:
  - The selection objective is always A-ASR (ASR_A), as requested, regardless of `objective`.
  - Minimal diffs otherwise.
"""

import os
import csv
import math
import copy
import time
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Callable, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, RandomSampler

# ======================= Utils =======================

def unwrap(m: nn.Module) -> nn.Module:
    return m.module if hasattr(m, "module") else m


def seed_everything(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)


def _replace_module_by_name(root: nn.Module, name: str, new_module: nn.Module):
    parts = name.split(".")
    parent = root
    for p in parts[:-1]:
        if p.isdigit():
            parent = parent[int(p)]
        else:
            parent = getattr(parent, p)
    setattr(parent, parts[-1], new_module)


def _get_train_dataset(env: "ModelEnv"):
    if hasattr(env.data, "trainset"):
        return env.data.trainset
    if hasattr(env.data, "trainloader") and hasattr(env.data.trainloader, "dataset"):
        return env.data.trainloader.dataset
    raise AttributeError("env.data has no trainset/trainloader.dataset")


def _get_test_loader(env: "ModelEnv"):
    if hasattr(env.data, "testloader"):
        return env.data.testloader
    raise AttributeError("env.data has no testloader")


# ======================= Noisy BN (aligned) =======================

class _NoisyBNBase:
    def include_noise(self):
        self._is_perturbed = True

    def exclude_noise(self):
        self._is_perturbed = False

    def perturb(self, is_perturbed: bool = True):
        self._is_perturbed = is_perturbed

    def reset(self, rand_init: bool = True, eps: float = 0.2):
        self._eps_anp = float(eps)
        with torch.no_grad():
            if rand_init:
                self.neuron_noise.data.uniform_(-eps, eps)
            else:
                self.neuron_noise.zero_()


class NoisyBatchNorm2d(nn.Module, _NoisyBNBase):
    """
    Strict-aligned flavor:
      - neuron_mask / neuron_noise
      - separate noisy running stats:
          running_mean_noisy, running_var_noisy, num_batches_tracked_noisy
      - forward uses clean stats when exclude_noise; noisy stats when include_noise
      - output is BN(x) * (mask + noise_if_on)
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        if self.track_running_stats:
            self.register_buffer("running_mean", torch.zeros(num_features))
            self.register_buffer("running_var", torch.ones(num_features))
            self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))

            self.register_buffer("running_mean_noisy", torch.zeros(num_features))
            self.register_buffer("running_var_noisy", torch.ones(num_features))
            self.register_buffer("num_batches_tracked_noisy", torch.tensor(0, dtype=torch.long))
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)
            self.register_buffer("num_batches_tracked", None)
            self.register_buffer("running_mean_noisy", None)
            self.register_buffer("running_var_noisy", None)
            self.register_buffer("num_batches_tracked_noisy", None)

        self.neuron_mask = nn.Parameter(torch.ones(num_features))
        self.neuron_noise = nn.Parameter(torch.zeros(num_features))
        self._is_perturbed = False
        self._eps_anp = 0.0

    def _bn_forward(self, x, use_noisy_stats: bool):
        if not self.track_running_stats:
            rm = rv = None
        else:
            if use_noisy_stats:
                rm, rv = self.running_mean_noisy, self.running_var_noisy
                nbt = self.num_batches_tracked_noisy
            else:
                rm, rv = self.running_mean, self.running_var
                nbt = self.num_batches_tracked
            if self.training:
                nbt += 1
        return F.batch_norm(
            x,
            rm, rv,
            self.weight, self.bias,
            self.training,
            self.momentum,
            self.eps,
        )

    def forward(self, x):
        y = self._bn_forward(x, use_noisy_stats=self._is_perturbed)
        gate = self.neuron_mask
        if self._is_perturbed:
            gate = gate + self.neuron_noise
        return y * gate.view(1, -1, 1, 1)


class NoisyBatchNorm1d(nn.Module, _NoisyBNBase):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        if self.track_running_stats:
            self.register_buffer("running_mean", torch.zeros(num_features))
            self.register_buffer("running_var", torch.ones(num_features))
            self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))

            self.register_buffer("running_mean_noisy", torch.zeros(num_features))
            self.register_buffer("running_var_noisy", torch.ones(num_features))
            self.register_buffer("num_batches_tracked_noisy", torch.tensor(0, dtype=torch.long))
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)
            self.register_buffer("num_batches_tracked", None)
            self.register_buffer("running_mean_noisy", None)
            self.register_buffer("running_var_noisy", None)
            self.register_buffer("num_batches_tracked_noisy", None)

        self.neuron_mask = nn.Parameter(torch.ones(num_features))
        self.neuron_noise = nn.Parameter(torch.zeros(num_features))
        self._is_perturbed = False
        self._eps_anp = 0.0

    def _bn_forward(self, x, use_noisy_stats: bool):
        if not self.track_running_stats:
            rm = rv = None
        else:
            if use_noisy_stats:
                rm, rv = self.running_mean_noisy, self.running_var_noisy
                nbt = self.num_batches_tracked_noisy
            else:
                rm, rv = self.running_mean, self.running_var
                nbt = self.num_batches_tracked
            if self.training:
                nbt += 1
        return F.batch_norm(
            x,
            rm, rv,
            self.weight, self.bias,
            self.training,
            self.momentum,
            self.eps,
        )

    def forward(self, x):
        y = self._bn_forward(x, use_noisy_stats=self._is_perturbed)
        if self._is_perturbed:
            y = y + self.neuron_noise.view(1, -1)
        y = y * self.neuron_mask.view(1, -1)
        return y


def build_anp_model_from_base(base: nn.Module) -> nn.Module:
    """
    Replace all BN2d/BN1d with NoisyBN2d/NoisyBN1d and copy parameters+stats.
    Noisy stats are initialized as clean stats.
    """
    base = unwrap(base)
    anp_model = copy.deepcopy(base)

    for name, m in list(anp_model.named_modules()):
        if isinstance(m, nn.BatchNorm2d):
            nb = NoisyBatchNorm2d(
                m.num_features, eps=m.eps, momentum=m.momentum,
                affine=m.affine, track_running_stats=m.track_running_stats,
            )
            with torch.no_grad():
                if m.affine:
                    nb.weight.copy_(m.weight)
                    nb.bias.copy_(m.bias)
                if m.track_running_stats:
                    nb.running_mean.copy_(m.running_mean)
                    nb.running_var.copy_(m.running_var)
                    nb.num_batches_tracked.copy_(m.num_batches_tracked)

                    nb.running_mean_noisy.copy_(m.running_mean)
                    nb.running_var_noisy.copy_(m.running_var)
                    nb.num_batches_tracked_noisy.copy_(m.num_batches_tracked)
            _replace_module_by_name(anp_model, name, nb)

        elif isinstance(m, nn.BatchNorm1d):
            nb = NoisyBatchNorm1d(
                m.num_features, eps=m.eps, momentum=m.momentum,
                affine=m.affine, track_running_stats=m.track_running_stats,
            )
            with torch.no_grad():
                if m.affine:
                    nb.weight.copy_(m.weight)
                    nb.bias.copy_(m.bias)
                if m.track_running_stats:
                    nb.running_mean.copy_(m.running_mean)
                    nb.running_var.copy_(m.running_var)
                    nb.num_batches_tracked.copy_(m.num_batches_tracked)

                    nb.running_mean_noisy.copy_(m.running_mean)
                    nb.running_var_noisy.copy_(m.running_var)
                    nb.num_batches_tracked_noisy.copy_(m.num_batches_tracked)
            _replace_module_by_name(anp_model, name, nb)

    return anp_model


# ======================= ANP primitive ops =======================

def clip_mask(model: nn.Module, lower=0.0, upper=1.0):
    params = [p for n, p in model.named_parameters() if "neuron_mask" in n]
    with torch.no_grad():
        for p in params:
            p.clamp_(lower, upper)


def sign_grad(model: nn.Module):
    noise = [p for n, p in model.named_parameters() if "neuron_noise" in n]
    for p in noise:
        if p.grad is not None:
            p.grad.data = torch.sign(p.grad.data)


def include_noise(model: nn.Module):
    for _, module in model.named_modules():
        if isinstance(module, (NoisyBatchNorm2d, NoisyBatchNorm1d)):
            module.include_noise()


def exclude_noise(model: nn.Module):
    for _, module in model.named_modules():
        if isinstance(module, (NoisyBatchNorm2d, NoisyBatchNorm1d)):
            module.exclude_noise()


def reset(model: nn.Module, rand_init: bool, eps: float):
    for _, module in model.named_modules():
        if isinstance(module, (NoisyBatchNorm2d, NoisyBatchNorm1d)):
            module.reset(rand_init=rand_init, eps=eps)


def mask_train(cfg, model: nn.Module, criterion: nn.Module,
               mask_opt: torch.optim.Optimizer, noise_opt: torch.optim.Optimizer,
               data_loader: DataLoader):
    """
    Strictly aligned with reference mask_train().
    """
    model.train()
    total_correct = 0
    total_loss = 0.0
    nb_samples = 0

    device = cfg.device

    for images, labels in data_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        nb_samples += images.size(0)

        # step 1: adversarial perturbation for neurons (noise ascent)
        if cfg.anp_eps > 0.0:
            reset(model, rand_init=True, eps=cfg.anp_eps)
            for _ in range(cfg.anp_steps):
                noise_opt.zero_grad(set_to_none=True)
                include_noise(model)
                out_noise = model(images)
                loss_noise = -criterion(out_noise, labels)
                loss_noise.backward()
                sign_grad(model)
                noise_opt.step()

        # step 2: update mask (descent)
        mask_opt.zero_grad(set_to_none=True)

        if cfg.anp_eps > 0.0:
            include_noise(model)
            out_noise = model(images)
            loss_rob = criterion(out_noise, labels)
        else:
            loss_rob = 0.0

        exclude_noise(model)
        out_clean = model(images)
        loss_nat = criterion(out_clean, labels)

        loss = cfg.anp_alpha * loss_nat + (1.0 - cfg.anp_alpha) * loss_rob

        pred = out_clean.data.max(1)[1]
        total_correct += pred.eq(labels.view_as(pred)).sum().item()
        total_loss += float(loss.item())

        loss.backward()
        mask_opt.step()
        clip_mask(model)

    loss_avg = total_loss / max(1, len(data_loader))
    acc = float(total_correct) / max(1, nb_samples)
    return loss_avg, acc


def save_mask_scores(state_dict: Dict[str, torch.Tensor], file_name: str):
    mask_values = []
    count = 0
    for name, param in state_dict.items():
        if "neuron_mask" in name:
            for idx in range(param.size(0)):
                neuron_name = ".".join(name.split(".")[:-1])
                mask_values.append(f"{count}\t{neuron_name}\t{idx}\t{param[idx].item():.6f}\n")
                count += 1
    with open(file_name, "w") as f:
        f.write("No\tLayer Name\tNeuron Idx\tMask Score\n")
        f.writelines(mask_values)


def read_mask_scores(file_name: str) -> List[Tuple[str, int, float]]:
    rows = []
    with open(file_name, "r") as f:
        lines = f.readlines()[1:]
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 4:
            continue
        layer = parts[1]
        idx = int(parts[2])
        val = float(parts[3])
        rows.append((layer, idx, val))
    return rows


def pruning(net: nn.Module, neuron: Tuple[str, int, float]):
    state_dict = net.state_dict()
    weight_name = f"{neuron[0]}.weight"
    if weight_name not in state_dict:
        return
    idx = int(neuron[1])
    if idx < 0 or idx >= state_dict[weight_name].shape[0]:
        return
    state_dict[weight_name][idx] = 0.0
    net.load_state_dict(state_dict)


# ======================= Evaluation helpers (patched: force eval) =======================

@torch.no_grad()
def eval_asr_ab(env: "ModelEnv", model: nn.Module,
                trigA, target_A: int,
                trigB, target_B: int) -> Tuple[float, float, float, float, float]:
    """
    Return (clean, RA_A, ASR_A, RA_B, ASR_B), forcing eval() and restoring original mode.
    """
    was_training = model.training
    model.eval()
    try:
        test_loader = _get_test_loader(env)
        clean = float(env.eval.clean(model, test_loader))
        _, raA, asrA = env.eval.metrics_all_to_one(model, test_loader, trigA, target_A)
        _, raB, asrB = env.eval.metrics_all_to_one(model, test_loader, trigB, target_B)
        return clean, float(raA), float(asrA), float(raB), float(asrB)
    finally:
        if was_training:
            model.train()


# ======================= BN recalibration (NEW) =======================

@torch.no_grad()
def bn_recalibrate_after_pruning(model: nn.Module, calib_loader: DataLoader, device: torch.device, num_batches: int = 50):
    """
    Reset BN running stats and re-estimate them by forwarding clean data in train mode.

    - Only updates BatchNorm{1,2}d running_mean/var/num_batches_tracked.
    - Freezes dropout to eval to avoid extra stochasticity during BN calibration.
    - Restores original train/eval state and BN momentums.
    """
    was_training = model.training

    bn_layers: List[nn.Module] = []
    bn_moms: List[Optional[float]] = []
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            bn_layers.append(m)
            bn_moms.append(m.momentum)

    if len(bn_layers) == 0:
        return

    model.train()
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            m.eval()

    for bn in bn_layers:
        if bn.running_mean is not None:
            bn.running_mean.zero_()
        if bn.running_var is not None:
            bn.running_var.fill_(1.0)
        if hasattr(bn, "num_batches_tracked") and bn.num_batches_tracked is not None:
            bn.num_batches_tracked.zero_()
        bn.momentum = None  # cumulative average

    it = 0
    for batch in calib_loader:
        x = batch[0].to(device, non_blocking=True)
        _ = model(x)
        it += 1
        if it >= int(max(1, num_batches)):
            break

    for bn, mom in zip(bn_layers, bn_moms):
        bn.momentum = mom

    if not was_training:
        model.eval()


# ======================= Strict-aligned ANP defense =======================

@dataclass
class ANPCfg:
    ratio: float = 0.02
    index_seed: int = 0

    nb_iter: int = 500
    print_every: int = 50
    batch_size: int = 64
    num_workers: int = 0

    lr: float = 0.1
    anp_eps: float = 0.2
    anp_steps: int = 1
    anp_alpha: float = 0.5

    # legacy: kept (default tightened as your patch)
    acc_ratio: float = 0.02

    # per-round saving threshold (relative CLEAN drop vs current model at round start)
    # If None -> fallback to acc_ratio; recommended default = 0.05
    save_acc_ratio: Optional[float] = 0.05

    pruning_by: str = "number"   # "number" | "threshold"
    pruning_max: float = 1000
    pruning_step: float = 50
    pruning_number: Optional[float] = None

    # BN recalib batches after pruning
    bn_recalib_batches: int = 50

    device: str = "cuda"


class ANPDefenseAligned:
    def __init__(self, env: "ModelEnv", save_dir: str):
        self.env = env
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    # -------- state_dict helpers --------
    def _get_plain_state_dict(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        sd = unwrap(model).state_dict()
        return {k: v.detach().cpu().clone() for k, v in sd.items()}

    def _load_plain_state_dict_inplace(self, model: nn.Module, sd_cpu: Dict[str, torch.Tensor]):
        unwrap(model).load_state_dict(sd_cpu, strict=True)
        return model

    def _choose_index(self, data_len: int, ratio: float, seed: int) -> np.ndarray:
        rs = np.random.RandomState(seed)
        idx = rs.permutation(data_len)
        k = int(max(1, round(float(ratio) * data_len)))
        return idx[:k]

    def _build_clean_val_loader(self, cfg: ANPCfg) -> DataLoader:
        train_ds = _get_train_dataset(self.env)
        data_len = len(train_ds)
        ran_idx = self._choose_index(data_len, cfg.ratio, cfg.index_seed)
        np.savetxt(os.path.join(self.save_dir, "index.txt"), ran_idx, fmt="%d")

        sub = Subset(train_ds, ran_idx.tolist())

        gen = torch.Generator()
        gen.manual_seed(int(cfg.index_seed))

        sampler = RandomSampler(
            data_source=sub,
            replacement=True,
            num_samples=int(cfg.print_every * cfg.batch_size),
            generator=gen,
        )
        loader = DataLoader(
            sub,
            batch_size=int(cfg.batch_size),
            shuffle=False,
            sampler=sampler,
            num_workers=int(cfg.num_workers),
            pin_memory=True,
            drop_last=False,
        )
        return loader

    def _collect_mask_values(self, anp_model: nn.Module) -> List[Tuple[str, int, float]]:
        mask_path = os.path.join(self.save_dir, "mask_values.txt")
        save_mask_scores(anp_model.state_dict(), mask_path)
        mask_values = read_mask_scores(mask_path)
        mask_values = sorted(mask_values, key=lambda x: float(x[2]))
        return mask_values

    @staticmethod
    def _within_clean_drop(clean_acc: float, acc_ref: float, drop_ratio: float) -> bool:
        return float(clean_acc) >= float(acc_ref) * (1.0 - float(drop_ratio))

    @staticmethod
    def _better_by_asr_then_clean(asr: float, clean: float, best_asr: float, best_clean: float) -> bool:
        # primary: ASR asc; tie: clean desc
        if float(asr) < float(best_asr):
            return True
        if float(asr) == float(best_asr) and float(clean) > float(best_clean):
            return True
        return False

    def evaluate_by_number(
        self,
        cfg: ANPCfg,
        model_prune: nn.Module,
        mask_values: List[Tuple[str, int, float]],
        eval_ab_fn: Callable[[nn.Module], Tuple[float, float, float, float, float]],
        acc_ref: float,
        clean_drop_ratio: float,
        save_csv: bool = True,
        bn_calib_loader: Optional[DataLoader] = None,
        bn_calib_device: Optional[torch.device] = None,
    ) -> Dict[str, Any]:
        """
        Returns:
          - best_non_before_model: best feasible pruning candidate (exclude before), or None
          - best_non_before_info: {pruned_total, clean, asrA, asrB}
          - history saved to number_df.csv if save_csv
        """
        nb_max = int(np.ceil(cfg.pruning_max))
        nb_step = int(max(1, np.ceil(cfg.pruning_step)))

        best_non_before_model: Optional[nn.Module] = None
        best_non_before_clean = -float("inf")
        best_non_before_asrA = float("inf")
        best_non_before_asrB = float("inf")
        best_non_before_pruned = None

        rows = []
        pruned_total = 0

        for start in range(0, nb_max + 1, nb_step):
            end = min(start + nb_step, len(mask_values))
            for i in range(start, end):
                pruning(model_prune, mask_values[i])
            pruned_total = end

            if bn_calib_loader is not None and bn_calib_device is not None:
                bn_recalibrate_after_pruning(
                    model_prune, bn_calib_loader, bn_calib_device, num_batches=int(cfg.bn_recalib_batches)
                )

            clean_acc, raA, asrA, raB, asrB = eval_ab_fn(model_prune)

            print(
                f"[ANP][number] pruned={pruned_total:6d} | "
                f"Clean {clean_acc:.2f}% | "
                f"A: ASR {asrA:.2f}% RA {raA:.2f}% | "
                f"B: ASR {asrB:.2f}% RA {raB:.2f}%"
            )

            rows.append((pruned_total, clean_acc, asrA, asrB))

            # selection candidate: must satisfy clean constraint (vs acc_ref for THIS ROUND)
            if self._within_clean_drop(clean_acc, acc_ref, clean_drop_ratio):
                # "non-before" candidates here are all pruning candidates (since before is separate)
                if self._better_by_asr_then_clean(asrA, clean_acc, best_non_before_asrA, best_non_before_clean):
                    best_non_before_asrA = float(asrA)
                    best_non_before_clean = float(clean_acc)
                    best_non_before_asrB = float(asrB)
                    best_non_before_model = copy.deepcopy(model_prune)
                    best_non_before_pruned = int(pruned_total)

        if save_csv:
            out_csv = os.path.join(self.save_dir, "number_df.csv")
            with open(out_csv, "w") as f:
                f.write("number,clean_acc,asrA,asrB\n")
                for (n, ca, aA, aB) in rows:
                    f.write(f"{n},{ca:.6f},{aA:.6f},{aB:.6f}\n")

        return dict(
            best_non_before_model=best_non_before_model,
            best_non_before_info=(
                None if best_non_before_model is None else dict(
                    pruned_total=best_non_before_pruned,
                    clean=best_non_before_clean,
                    asrA=best_non_before_asrA,
                    asrB=best_non_before_asrB,
                )
            ),
        )

    def evaluate_by_threshold(
        self,
        cfg: ANPCfg,
        model_prune: nn.Module,
        mask_values: List[Tuple[str, int, float]],
        eval_ab_fn: Callable[[nn.Module], Tuple[float, float, float, float, float]],
        acc_ref: float,
        clean_drop_ratio: float,
        save_csv: bool = True,
        bn_calib_loader: Optional[DataLoader] = None,
        bn_calib_device: Optional[torch.device] = None,
    ) -> Dict[str, Any]:
        """
        Returns:
          - best_non_before_model: best feasible pruning candidate (exclude before), or None
          - best_non_before_info: {threshold, pruned_total, clean, asrA, asrB}
          - history saved to threshold_df.csv if save_csv
        """
        thresholds = np.arange(0, cfg.pruning_max + cfg.pruning_step, cfg.pruning_step, dtype=float)
        start = 0

        best_non_before_model: Optional[nn.Module] = None
        best_non_before_clean = -float("inf")
        best_non_before_asrA = float("inf")
        best_non_before_asrB = float("inf")
        best_non_before_thr = None
        best_non_before_pruned = None

        rows = []
        for thr in thresholds:
            for idx in range(start, len(mask_values)):
                if float(mask_values[idx][2]) <= float(thr):
                    pruning(model_prune, mask_values[idx])
                    start += 1
                else:
                    break

            pruned_total = start

            if bn_calib_loader is not None and bn_calib_device is not None:
                bn_recalibrate_after_pruning(
                    model_prune, bn_calib_loader, bn_calib_device, num_batches=int(cfg.bn_recalib_batches)
                )

            clean_acc, raA, asrA, raB, asrB = eval_ab_fn(model_prune)

            print(
                f"[ANP][thr] pruned={pruned_total:6d} thr={thr:8.4f} | "
                f"Clean {clean_acc:.2f}% | "
                f"A: ASR {asrA:.2f}% RA {raA:.2f}% | "
                f"B: ASR {asrB:.2f}% RA {raB:.2f}%"
            )

            rows.append((thr, clean_acc, asrA, asrB))

            if self._within_clean_drop(clean_acc, acc_ref, clean_drop_ratio):
                if self._better_by_asr_then_clean(asrA, clean_acc, best_non_before_asrA, best_non_before_clean):
                    best_non_before_asrA = float(asrA)
                    best_non_before_clean = float(clean_acc)
                    best_non_before_asrB = float(asrB)
                    best_non_before_thr = float(thr)
                    best_non_before_pruned = int(pruned_total)
                    best_non_before_model = copy.deepcopy(model_prune)

        if save_csv:
            out_csv = os.path.join(self.save_dir, "threshold_df.csv")
            with open(out_csv, "w") as f:
                f.write("threshold,clean_acc,asrA,asrB\n")
                for (t, ca, aA, aB) in rows:
                    f.write(f"{t:.6f},{ca:.6f},{aA:.6f},{aB:.6f}\n")

        return dict(
            best_non_before_model=best_non_before_model,
            best_non_before_info=(
                None if best_non_before_model is None else dict(
                    threshold=best_non_before_thr,
                    pruned_total=best_non_before_pruned,
                    clean=best_non_before_clean,
                    asrA=best_non_before_asrA,
                    asrB=best_non_before_asrB,
                )
            ),
        )

    def run_ab(
        self,
        cfg: ANPCfg,
        *,
        target_A: int = 0,
        target_B: int = 1,
        patch_size_A: int = 3,
        sig_amp: float = 0.1,
        sig_freq: float = 6.0,
        objective: str = "A",  # kept for compatibility/logging; selection is always by A-ASR
    ) -> Dict[str, Any]:
        """
        Run a single ANP round starting from CURRENT env.model.

        Selection rule (STRICT per your latest request):
          - Define acc_ref = clean(before) for THIS round.
          - Define min_clean_allowed = acc_ref*(1-save_acc_ratio).
          - Collect candidates:
              * "before" (round start)
              * pruning candidates (evaluated along the pruning path) that satisfy clean constraint
          - Choose the candidate with lowest A-ASR among feasible candidates.
          - If that lowest-ASR feasible candidate is still "before", choose the next-best feasible (exclude "before").
            => Equivalent: pick best feasible non-before if any; else fall back to before.

        Save the SELECTED model to:
          - {save_dir}/anp_selected_ab.pth
        And return metrics + selected model state_dict (cpu) for chaining.
        """
        seed_everything(cfg.index_seed)
        device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        cfg.device = str(device)

        clean_drop_ratio = float(cfg.save_acc_ratio) if cfg.save_acc_ratio is not None else float(cfg.acc_ratio)

        # These triggers must exist in your project scope:
        trigA = TriggerBadNets(patch_size_A)
        trigB = TriggerSigAB(sig_amp, sig_freq)

        # snapshot BEFORE state (cpu)
        before_sd_cpu = self._get_plain_state_dict(self.env.model)

        # ================== Baseline (current model = "before") ==================
        clean0, raA0, asrA0, raB0, asrB0 = eval_asr_ab(
            self.env, self.env.model, trigA, target_A, trigB, target_B
        )
        min_clean_allowed = clean0 * (1.0 - clean_drop_ratio)

        print(
            f"[Baseline-AB] Clean {clean0:.2f}% | "
            f"A: RA {raA0:.2f}% ASR_A {asrA0:.2f}% | "
            f"B: RA {raB0:.2f}% ASR_B {asrB0:.2f}%"
        )
        print(
            f"[SelectRule] clean_drop_ratio={clean_drop_ratio:.3f} | "
            f"acc_ref={clean0:.2f}% min_clean={min_clean_allowed:.2f}% | "
            f"objective=A-ASR | (if best is 'before' => pick next-best feasible)"
        )

        baseline_metrics = dict(
            clean=float(clean0),
            A_trig=float(raA0), ASR_A=float(asrA0),
            B_trig=float(raB0), ASR_B=float(asrB0),
        )

        # ================== Build ANP model & loader ==================
        net_anp = build_anp_model_from_base(self.env.model).to(device)
        criterion = nn.CrossEntropyLoss().to(device)

        clean_val_loader = self._build_clean_val_loader(cfg)

        parameters = list(net_anp.named_parameters())
        mask_params = [v for n, v in parameters if "neuron_mask" in n]
        noise_params = [v for n, v in parameters if "neuron_noise" in n]

        if len(mask_params) == 0:
            raise RuntimeError("No neuron_mask found in ANP model. Check BN replacement.")

        mask_opt = torch.optim.SGD(mask_params, lr=cfg.lr, momentum=0.9)
        noise_opt = torch.optim.SGD(
            noise_params, lr=(cfg.anp_eps / max(1, cfg.anp_steps))
        ) if cfg.anp_eps > 0 else None

        # ================== Mask training ==================
        nb_repeat = int(np.ceil(cfg.nb_iter / cfg.print_every))
        print("[ANP-MaskTrain-AB] Iter\tlr\tTime\tTrainLoss\tTrainACC\tCleanACC\tASR_A\tASR_B")
        for i in range(nb_repeat):
            t0 = time.time()
            lr = mask_opt.param_groups[0]["lr"]

            train_loss, train_acc = mask_train(
                cfg, net_anp, criterion, mask_opt, noise_opt, clean_val_loader
            )

            clean_acc, _, asrA, _, asrB = eval_asr_ab(
                self.env, net_anp, trigA, target_A, trigB, target_B
            )

            t1 = time.time()
            print(
                f"{(i+1)*cfg.print_every:6d}\t{lr:.4f}\t{(t1-t0):.1f}\t"
                f"{train_loss:.4f}\t{train_acc:.4f}\t{clean_acc:.2f}\t{asrA:.2f}\t{asrB:.2f}"
            )

        # ================== Collect mask scores & prune on base model ==================
        mask_values = self._collect_mask_values(net_anp)
        print(f"[ANP-AB] collected mask scores: {len(mask_values)} neurons")

        net_prune = copy.deepcopy(unwrap(self.env.model)).to(device)

        def eval_ab_fn(m: nn.Module) -> Tuple[float, float, float, float, float]:
            return eval_asr_ab(self.env, m, trigA, target_A, trigB, target_B)

        # ================== Search best feasible pruning candidate (EXCLUDING 'before') ==================
        if cfg.pruning_by == "threshold":
            res = self.evaluate_by_threshold(
                cfg, net_prune, mask_values, eval_ab_fn,
                acc_ref=float(clean0),
                clean_drop_ratio=float(clean_drop_ratio),
                bn_calib_loader=clean_val_loader,
                bn_calib_device=device,
            )
        else:
            res = self.evaluate_by_number(
                cfg, net_prune, mask_values, eval_ab_fn,
                acc_ref=float(clean0),
                clean_drop_ratio=float(clean_drop_ratio),
                bn_calib_loader=clean_val_loader,
                bn_calib_device=device,
            )

        best_non_before_model = res.get("best_non_before_model", None)
        best_non_before_info = res.get("best_non_before_info", None)

        # ================== Final selection with "if best is before => choose next-best" ==================
        # This implements your exact rule:
        #   - If there exists any feasible non-before candidate, we choose it,
        #     even if before has lower ASR (i.e., before is the "lowest ASR" model).
        #   - If no feasible non-before exists, we must keep before (fallback).
        if best_non_before_model is not None:
            chosen_stage = "prune_best" if best_non_before_info is None else (
                f"prune_best_pruned{int(best_non_before_info.get('pruned_total', -1))}"
                if "pruned_total" in best_non_before_info else "prune_best"
            )
            chosen_model = best_non_before_model
            chosen_reason = "selected best feasible non-before (rule: if best is before -> pick next-best)"
        else:
            chosen_stage = "before"
            # reconstruct chosen model from before state to avoid any side effects
            chosen_model = copy.deepcopy(unwrap(self.env.model)).to(device)
            # ensure it matches before exactly
            unwrap(chosen_model).load_state_dict({k: v.to(device) for k, v in before_sd_cpu.items()}, strict=True)
            chosen_reason = "no feasible non-before candidate; fallback to before"

        # ================== Final evaluation (selected) ==================
        clean_f, raA_f, asrA_f, raB_f, asrB_f = eval_asr_ab(
            self.env, chosen_model, trigA, target_A, trigB, target_B
        )
        print(
            f"[ANP-Final-AB][SELECTED] stage={chosen_stage} | reason={chosen_reason}\n"
            f"  Clean {clean_f:.2f}% | A: RA {raA_f:.2f}% ASR_A {asrA_f:.2f}% | B: RA {raB_f:.2f}% ASR_B {asrB_f:.2f}%"
        )

        final_metrics = dict(
            clean=float(clean_f),
            A_trig=float(raA_f), ASR_A=float(asrA_f),
            B_trig=float(raB_f), ASR_B=float(asrB_f),
        )

        # Save SELECTED model for this round
        selected_path = os.path.join(self.save_dir, "anp_selected_ab.pth")
        torch.save({"state_dict": unwrap(chosen_model).state_dict()}, selected_path)

        selected_sd_cpu = self._get_plain_state_dict(chosen_model)

        return dict(
            baseline=baseline_metrics,
            final=final_metrics,
            selection=dict(
                clean_drop_ratio=float(clean_drop_ratio),
                acc_ref=float(clean0),
                min_clean_allowed=float(min_clean_allowed),
                chosen_stage=str(chosen_stage),
                chosen_reason=str(chosen_reason),
                best_non_before_info=best_non_before_info,
            ),
            objective=objective,
            saved_model=selected_path,
            before_state_dict_cpu=before_sd_cpu,
            selected_state_dict_cpu=selected_sd_cpu,
        )

    # ======================= Repeat n times (SERIAL) =======================

    def run_ab_repeat(
        self,
        cfg: ANPCfg,
        n_rounds: int,
        *,
        target_A: int = 0,
        target_B: int = 1,
        patch_size_A: int = 3,
        sig_amp: float = 0.1,
        sig_freq: float = 6.0,
        objective: str = "A",
        summary_csv_name: str = "anp_defense_summary_ab.csv",
        model_name_prefix: str = "anp_selected_round",
    ) -> Dict[str, object]:
        """
        SERIAL repeat:
          - round k compares clean drop vs CURRENT model at round start (not the very first model).
          - chooses per-round SELECTED model using the rule above,
            saves under save_dir/run_k/, and loads it into env.model for the next round.

        Outputs:
          - save_dir/run_k/anp_selected_ab.pth
          - save_dir/{model_name_prefix}XX.pth
          - save_dir/{summary_csv_name}
        """
        n_rounds = int(max(1, n_rounds))
        base_dir = self.save_dir
        os.makedirs(base_dir, exist_ok=True)

        summary_path = os.path.join(base_dir, summary_csv_name)
        need_header = (not os.path.exists(summary_path)) or (os.path.getsize(summary_path) == 0)

        fieldnames = [
            "round",
            "seed",
            "run_dir",
            "clean_drop_ratio",
            "acc_ref",
            "min_clean_allowed",
            "chosen_stage",
            "clean_before",
            "ASR_A_before",
            "ASR_B_before",
            "clean_after",
            "ASR_A_after",
            "ASR_B_after",
            "saved_model_round",
        ]

        all_runs: List[Dict[str, object]] = []

        with open(summary_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if need_header:
                writer.writeheader()

            for k in range(n_rounds):
                cfg_k = copy.deepcopy(cfg)
                cfg_k.index_seed = int(cfg.index_seed) + k

                run_dir = os.path.join(base_dir, f"run_{k+1:02d}")
                os.makedirs(run_dir, exist_ok=True)

                runner = self.__class__(self.env, save_dir=run_dir)
                print(f"\n========== [ANP][SERIAL-Repeat] {k+1}/{n_rounds} | seed={cfg_k.index_seed} | dir={run_dir} ==========")

                out = runner.run_ab(
                    cfg_k,
                    target_A=target_A, target_B=target_B,
                    patch_size_A=patch_size_A,
                    sig_amp=sig_amp, sig_freq=sig_freq,
                    objective=objective,
                )

                # Load selected model into env.model for next round (SERIAL)
                sel_sd = out["selected_state_dict_cpu"]
                self._load_plain_state_dict_inplace(self.env.model, sel_sd)

                # Save central copy
                model_path = os.path.join(base_dir, f"{model_name_prefix}{k+1:02d}.pth")
                torch.save(unwrap(self.env.model).state_dict(), model_path)

                baseline = out.get("baseline", {}) or {}
                final = out.get("final", {}) or {}
                sel = out.get("selection", {}) or {}

                row = dict(
                    round=k + 1,
                    seed=cfg_k.index_seed,
                    run_dir=run_dir,
                    clean_drop_ratio=sel.get("clean_drop_ratio", ""),
                    acc_ref=sel.get("acc_ref", ""),
                    min_clean_allowed=sel.get("min_clean_allowed", ""),
                    chosen_stage=sel.get("chosen_stage", ""),
                    clean_before=baseline.get("clean", ""),
                    ASR_A_before=baseline.get("ASR_A", ""),
                    ASR_B_before=baseline.get("ASR_B", ""),
                    clean_after=final.get("clean", ""),
                    ASR_A_after=final.get("ASR_A", ""),
                    ASR_B_after=final.get("ASR_B", ""),
                    saved_model_round=model_path,
                )
                writer.writerow(row)
                f.flush()

                out["round"] = k + 1
                out["seed"] = cfg_k.index_seed
                out["run_dir"] = run_dir
                out["saved_model_round"] = model_path
                all_runs.append(out)

        print(f"[ANP][SERIAL-Repeat] Summary CSV: {summary_path}")
        return dict(
            n_rounds=n_rounds,
            objective=objective,
            summary_csv=summary_path,
            runs=all_runs,
        )


# ======================= Minimal runnable example =======================

if __name__ == "__main__":
    # ---- Fill these ----
    CORE_CFG = dict(
        DATA_ROOT=data_path,
        CKPT_PATH=ckpt_path,
        VARIANT="resnet18",
        BATCH_SIZE=64,
        NUM_WORKERS=8,
        SEED=0,
        AMP=False,
    )
    env = ModelEnv(CORE_CFG)

    cfg = ANPCfg(
        ratio=0.05,
        index_seed=0,
        nb_iter=500,
        print_every=200,
        batch_size=64,
        num_workers=0,

        lr=0.10,
        anp_eps=0.2,
        anp_steps=1,
        anp_alpha=0.5,

        acc_ratio=0.02,
        save_acc_ratio=0.05,     # per-round clean drop threshold vs CURRENT round start model

        pruning_by="number",
        pruning_max=400,
        pruning_step=20,
        pruning_number=None,

        bn_recalib_batches=50,

        device="cuda",
    )

    anp = ANPDefenseAligned(env, save_dir="anp_defense_aligned_resnet18_cifar10_ab")

    res = anp.run_ab_repeat(
        cfg,
        n_rounds=10,
        target_A=0, target_B=1,
        patch_size_A=3,
        sig_amp=0.1, sig_freq=6.0,
        objective="A",
        summary_csv_name="anp_defense_summary_ab.csv",
        model_name_prefix="anp_selected_round",
    )

    print("\n================ SUMMARY ================")
    print("Objective used (selection always A-ASR):", res["objective"])
    print("Summary CSV:", res["summary_csv"])
    last = res["runs"][-1]
    print("Last round saved model:", last["saved_model_round"])
    print("Last baseline:", last["baseline"])
    print("Last final:", last["final"])
