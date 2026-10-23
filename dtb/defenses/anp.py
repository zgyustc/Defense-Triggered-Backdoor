# Extracted from BB/anbition_denfense-Final-abition_k_3.ipynb, code cell 16 (zero-based).
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ANP defense (STRICT aligned with BackdoorBench/ANP_backdoor-style reference)

Patched points per your request:
  1) Evaluation wrappers now enforce model.eval() and restore training state (fix BN/Dropout pollution).
  2) Increase anp_alpha default (more emphasis on clean accuracy).
  3) Lower lr and reduce nb_iter (mitigate overfitting to the small clean subset).
  4) Stronger seeding (deterministic cudnn) + fixed generator for RandomSampler (optional but helps reproducibility).

This script is built on your core_backdoor.ModelEnv and keeps EvalView-like A/B evaluation.
"""

import os
import math
import copy
import time
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Callable

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

    # deterministic for reproducibility (may reduce performance)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # If you want stricter determinism (may error on unsupported ops), uncomment:
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
def eval_asr_single(env: "ModelEnv", model: nn.Module, trig, target: int) -> Tuple[float, float, float]:
    """
    Return (clean_acc, RA, ASR) using env's pipeline, forcing eval() and restoring original mode.
    """
    was_training = model.training
    model.eval()
    try:
        test_loader = _get_test_loader(env)
        clean = float(env.eval.clean(model, test_loader))
        _, ra, asr = env.eval.metrics_all_to_one(model, test_loader, trig, target)
        return clean, float(ra), float(asr)
    finally:
        if was_training:
            model.train()


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
    anp_alpha: float = 0.5   # PATCH: increased default
    acc_ratio: float = 0.1

    pruning_by: str = "number"   # "number" | "threshold"
    pruning_max: float = 1000
    pruning_step: float = 50
    pruning_number: Optional[float] = None

    device: str = "cuda"


class ANPDefenseAligned:
    def __init__(self, env: "ModelEnv", save_dir: str):
        self.env = env
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

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

        # PATCH: fixed generator for reproducibility
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

    def evaluate_by_number(
        self,
        cfg: ANPCfg,
        model_prune: nn.Module,
        mask_values: List[Tuple[str, int, float]],
        objective_eval_fn: Callable[[nn.Module], Tuple[float, float]],  # (clean_acc, asr_obj)
        acc_ori: float,
        best_asr: float,
        save_csv: bool = True,
        ab_metrics_fn: Optional[Callable[[nn.Module], Tuple[float, float, float, float]]] = None,  # (asrA, raA, asrB, raB)
    ) -> Tuple[nn.Module, float]:
        nb_max = int(np.ceil(cfg.pruning_max))
        nb_step = int(max(1, np.ceil(cfg.pruning_step)))
        model_best = copy.deepcopy(model_prune)

        rows = []
        for start in range(0, nb_max + 1, nb_step):
            end = min(start + nb_step, len(mask_values))
            for i in range(start, end):
                pruning(model_prune, mask_values[i])

            pruned_total = end

            clean_acc, asr_obj = objective_eval_fn(model_prune)

            if ab_metrics_fn is not None:
                asrA, raA, asrB, raB = ab_metrics_fn(model_prune)
                print(
                    f"[ANP][number] pruned={pruned_total:6d} | "
                    f"Clean {clean_acc:.2f}% | ASR_obj {asr_obj:.2f}% | "
                    f"A: ASR {asrA:.2f}% RA {raA:.2f}% | "
                    f"B: ASR {asrB:.2f}% RA {raB:.2f}%"
                )
            else:
                print(f"[ANP][number] pruned={pruned_total:6d} | Clean {clean_acc:.2f}% | ASR_obj {asr_obj:.2f}%")

            rows.append((pruned_total, clean_acc, asr_obj))

            if abs(clean_acc - acc_ori) / max(1e-6, acc_ori) < cfg.acc_ratio:
                if asr_obj < best_asr:
                    best_asr = asr_obj
                    model_best = copy.deepcopy(model_prune)

        if save_csv:
            out_csv = os.path.join(self.save_dir, "number_df.csv")
            with open(out_csv, "w") as f:
                f.write("number,clean_acc,asr_obj\n")
                for (n, ca, ao) in rows:
                    f.write(f"{n},{ca:.6f},{ao:.6f}\n")

        return model_best, best_asr

    def evaluate_by_threshold(
        self,
        cfg: ANPCfg,
        model_prune: nn.Module,
        mask_values: List[Tuple[str, int, float]],
        objective_eval_fn: Callable[[nn.Module], Tuple[float, float]],  # (clean_acc, asr_obj)
        acc_ori: float,
        best_asr: float,
        save_csv: bool = True,
        ab_metrics_fn: Optional[Callable[[nn.Module], Tuple[float, float, float, float]]] = None,
    ) -> Tuple[nn.Module, float]:
        thresholds = np.arange(0, cfg.pruning_max + cfg.pruning_step, cfg.pruning_step, dtype=float)
        start = 0
        model_best = copy.deepcopy(model_prune)

        rows = []
        for thr in thresholds:
            for idx in range(start, len(mask_values)):
                if float(mask_values[idx][2]) <= float(thr):
                    pruning(model_prune, mask_values[idx])
                    start += 1
                else:
                    break

            pruned_total = start
            clean_acc, asr_obj = objective_eval_fn(model_prune)

            if ab_metrics_fn is not None:
                asrA, raA, asrB, raB = ab_metrics_fn(model_prune)
                print(
                    f"[ANP][thr] pruned={pruned_total:6d} thr={thr:8.4f} | "
                    f"Clean {clean_acc:.2f}% | ASR_obj {asr_obj:.2f}% | "
                    f"A: ASR {asrA:.2f}% RA {raA:.2f}% | "
                    f"B: ASR {asrB:.2f}% RA {raB:.2f}%"
                )
            else:
                print(
                    f"[ANP][thr] pruned={pruned_total:6d} thr={thr:8.4f} | "
                    f"Clean {clean_acc:.2f}% | ASR_obj {asr_obj:.2f}%"
                )

            rows.append((thr, clean_acc, asr_obj))

            if abs(clean_acc - acc_ori) / max(1e-6, acc_ori) < cfg.acc_ratio:
                if asr_obj < best_asr:
                    best_asr = asr_obj
                    model_best = copy.deepcopy(model_prune)

        if save_csv:
            out_csv = os.path.join(self.save_dir, "threshold_df.csv")
            with open(out_csv, "w") as f:
                f.write("threshold,clean_acc,asr_obj\n")
                for (t, ca, ao) in rows:
                    f.write(f"{t:.6f},{ca:.6f},{ao:.6f}\n")

        return model_best, best_asr

    def run_ab(
        self,
        cfg: ANPCfg,
        *,
        target_A: int = 0,
        target_B: int = 1,
        patch_size_A: int = 3,
        sig_amp: float = 0.1,
        sig_freq: float = 6.0,
        objective: str = "A",  # "A" | "B" | "max"
    ):
        seed_everything(cfg.index_seed)
        device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        cfg.device = str(device)

        # These triggers must exist in your environment:
        #   TriggerBadNets, TriggerSigAB
        trigA = TriggerBadNets(patch_size_A)
        trigB = TriggerSigAB(sig_amp, sig_freq)

        # ================== Baseline ==================
        clean0, raA0, asrA0, raB0, asrB0 = eval_asr_ab(
            self.env, self.env.model, trigA, target_A, trigB, target_B
        )
        print(
            f"[Baseline-AB] Clean {clean0:.2f}% | "
            f"A: RA {raA0:.2f}% ASR_A {asrA0:.2f}% | "
            f"B: RA {raB0:.2f}% ASR_B {asrB0:.2f}%"
        )

        baseline_metrics = dict(
            clean=clean0,
            A_trig=raA0, ASR_A=asrA0,
            B_trig=raB0, ASR_B=asrB0,
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

        def objective_eval(m: nn.Module) -> Tuple[float, float]:
            clean, _, asrA, _, asrB = eval_asr_ab(self.env, m, trigA, target_A, trigB, target_B)
            if objective == "B":
                return clean, asrB
            if objective == "max":
                return clean, max(asrA, asrB)
            return clean, asrA

        def ab_metrics_fn(m: nn.Module) -> Tuple[float, float, float, float]:
            _, raA, asrA, raB, asrB = eval_asr_ab(self.env, m, trigA, target_A, trigB, target_B)
            return float(asrA), float(raA), float(asrB), float(raB)

        if objective == "B":
            best_asr = asrB0
        elif objective == "max":
            best_asr = max(asrA0, asrB0)
        else:
            best_asr = asrA0
        acc_ori = clean0

        if cfg.pruning_by == "threshold":
            model_best, best_asr = self.evaluate_by_threshold(
                cfg, net_prune, mask_values, objective_eval,
                acc_ori=acc_ori, best_asr=best_asr,
                ab_metrics_fn=ab_metrics_fn,
            )
        else:
            model_best, best_asr = self.evaluate_by_number(
                cfg, net_prune, mask_values, objective_eval,
                acc_ori=acc_ori, best_asr=best_asr,
                ab_metrics_fn=ab_metrics_fn,
            )

        # ================== Final evaluation ==================
        clean_f, raA_f, asrA_f, raB_f, asrB_f = eval_asr_ab(
            self.env, model_best, trigA, target_A, trigB, target_B
        )
        print(
            f"[ANP-Final-AB] Clean {clean_f:.2f}% | "
            f"A: RA {raA_f:.2f}% ASR_A {asrA_f:.2f}% | "
            f"B: RA {raB_f:.2f}% ASR_B {asrB_f:.2f}%"
        )

        final_metrics = dict(
            clean=clean_f,
            A_trig=raA_f, ASR_A=asrA_f,
            B_trig=raB_f, ASR_B=asrB_f,
        )

        torch.save(
            {"state_dict": unwrap(model_best).state_dict()},
            os.path.join(self.save_dir, "anp_best_ab.pth"),
        )

        return dict(
            baseline=baseline_metrics,
            final=final_metrics,
            objective=objective,
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

    # PATCHED defaults: higher anp_alpha, lower lr, smaller nb_iter
    cfg = ANPCfg(
        ratio=0.05,
        index_seed=0,
        nb_iter=500,          # PATCH: reduced (was 1000)
        print_every=200,
        batch_size=64,
        num_workers=0,

        lr=0.10,              # PATCH: lowered (was 0.15)
        anp_eps=0.2,
        anp_steps=1,
        anp_alpha=0.5,         # PATCH: increased (was 0.3)

        acc_ratio=0.1,

        pruning_by="number",
        pruning_max=400,
        pruning_step=20,
        pruning_number=None,
        device="cuda",
    )

    anp = ANPDefenseAligned(env, save_dir="anp_defense_aligned_resnet18_cifar10_ab")

    res = anp.run_ab(
        cfg,
        target_A=0, target_B=1,
        patch_size_A=3,
        sig_amp=0.1, sig_freq=6.0,
        objective="A",
    )

    print("Baseline metrics:", res["baseline"])
    print("Final metrics:", res["final"])
    print("Objective used:", res["objective"])
