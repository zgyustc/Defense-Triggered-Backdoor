# Extracted from BB/anbition_denfense-Final-abition_k_3.ipynb, code cell 19 (zero-based).
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, math
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset


# ==========================================================
# Label-smoothed CE (same as DSAM/FT-SAM style)
# ==========================================================
def smooth_crossentropy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    smoothing: float = 0.1
) -> torch.Tensor:
    """
    Label-smoothed cross-entropy (per-sample loss).
    predictions: [B, C] logits
    targets:     [B]    int64 labels
    returns:     [B]    per-sample loss
    """
    log_probs = F.log_softmax(predictions, dim=1)
    n_classes = predictions.size(1)
    with torch.no_grad():
        true_dist = torch.zeros_like(log_probs)
        true_dist.fill_(smoothing / (n_classes - 1))
        true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - smoothing)
    return (-true_dist * log_probs).sum(dim=1)


# ==========================================================
# DSAM helpers: ProportionScheduler + SAM (closure-based)
# ==========================================================
class ProportionScheduler:
    """
    Map current LR (from pytorch scheduler) to rho_t in [min_value, max_value].
    This mirrors the DSAM style: rho is tied to LR progress.
    """
    def __init__(
        self,
        pytorch_lr_scheduler,
        max_lr: float,
        min_lr: float,
        max_value: float,
        min_value: float,
    ):
        self.sched = pytorch_lr_scheduler
        self.max_lr = float(max_lr)
        self.min_lr = float(min_lr)
        self.max_value = float(max_value)
        self.min_value = float(min_value)

    def get_value(self) -> float:
        lrs = self.sched.get_last_lr()
        lr = float(lrs[0]) if lrs else self.max_lr
        denom = max(1e-12, self.max_lr - self.min_lr)
        p = (lr - self.min_lr) / denom
        p = max(0.0, min(1.0, p))
        return self.min_value + p * (self.max_value - self.min_value)


class SAM:
    """
    DSAM-style SAM wrapper (closure-driven):
      - set_closure(loss_fn, img, target, autocast_ctx, scaler)
      - step() -> (predictions, loss_tensor)
      - update_rho_t() -> pull rho_t from rho_scheduler
    """
    def __init__(
        self,
        params,
        base_optimizer: torch.optim.Optimizer,
        model: nn.Module,
        rho_scheduler: ProportionScheduler = None,
        adaptive: bool = True,
        sam_alpha: float = 0.0,
    ):
        self.params = [p for p in params]
        self.base_optimizer = base_optimizer
        self.model = model
        self.rho_scheduler = rho_scheduler
        self.adaptive = bool(adaptive)
        self.sam_alpha = float(sam_alpha)  # kept for interface parity

        # runtime closure
        self._loss_fn = None
        self._img = None
        self._target = None
        self._autocast_ctx = nullcontext
        self._scaler = None

        self.rho_t = float(self.rho_scheduler.get_value()) if self.rho_scheduler is not None else 2.0

    @torch.no_grad()
    def update_rho_t(self):
        if self.rho_scheduler is not None:
            self.rho_t = float(self.rho_scheduler.get_value())

    def set_closure(self, loss_fn, img, target, autocast_ctx=None, scaler=None):
        self._loss_fn = loss_fn
        self._img = img
        self._target = target
        self._autocast_ctx = autocast_ctx if autocast_ctx is not None else nullcontext
        self._scaler = scaler  # can be GradScaler(enabled=False) or None

    def zero_grad(self, set_to_none: bool = True):
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def _grad_norm(self) -> torch.Tensor:
        device = None
        norms = []
        for p in self.params:
            if p.grad is None:
                continue
            device = p.grad.device
            if self.adaptive:
                norms.append(((p.abs() + 1e-12) * p.grad).norm(p=2))
            else:
                norms.append(p.grad.norm(p=2))
        if not norms:
            return torch.tensor(0.0, device=device or "cpu")
        return torch.norm(torch.stack(norms), p=2)

    @torch.no_grad()
    def _perturb(self, scale: torch.Tensor):
        e_ws = []
        for p in self.params:
            if p.grad is None:
                e_ws.append(None)
                continue
            if self.adaptive:
                e = (p.abs() + 1e-12) * p.grad * scale
            else:
                e = p.grad * scale
            p.add_(e)
            e_ws.append(e)
        return e_ws

    @torch.no_grad()
    def _restore(self, e_ws):
        for p, e in zip(self.params, e_ws):
            if e is not None:
                p.sub_(e)

    def step(self):
        assert self._loss_fn is not None, "Call set_closure() before step()."

        img = self._img
        target = self._target
        autocast_ctx = self._autocast_ctx if self._autocast_ctx is not None else nullcontext
        scaler = self._scaler

        # ---- first forward/backward at w ----
        self.zero_grad(set_to_none=True)
        with autocast_ctx():
            pred = self.model(img)
            loss = self._loss_fn(pred, target)

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # grad_norm is computed on (possibly) scaled grads.
        grad_norm = self._grad_norm()

        if (not torch.isfinite(grad_norm)) or float(grad_norm.item()) == 0.0:
            # fallback: plain optimizer step
            if scaler is not None:
                scaler.step(self.base_optimizer)
                scaler.update()
            else:
                self.base_optimizer.step()
            return pred.detach(), loss.detach()

        rho = float(self.rho_t)
        scale = (rho / (grad_norm + 1e-12)).to(grad_norm.device)
        e_ws = self._perturb(scale)

        # ---- second forward/backward at w + e ----
        self.zero_grad(set_to_none=True)
        with autocast_ctx():
            pred_adv = self.model(img)
            loss_adv = self._loss_fn(pred_adv, target)

        if scaler is not None:
            scaler.scale(loss_adv).backward()
        else:
            loss_adv.backward()

        # restore + step
        self._restore(e_ws)

        if scaler is not None:
            scaler.step(self.base_optimizer)
            scaler.update()
        else:
            self.base_optimizer.step()

        return pred_adv.detach(), loss_adv.detach()


# ==========================================================
# Optional: class-balanced subset indices (BackdoorBench-like)
# ==========================================================
def choose_by_class_indices(dataset, num_classes: int, ratio: float, seed: int = 0):
    """
    class-balanced subset:
      each class takes floor(ratio * N / num_classes) samples.
    dataset[i] should return (x, y, ...) or (x, y)
    """
    import random
    rnd = random.Random(seed)

    buckets = [[] for _ in range(num_classes)]
    N = len(dataset)
    for i in range(N):
        item = dataset[i]
        y = item[1]
        if isinstance(y, torch.Tensor):
            y = int(y.item())
        buckets[int(y)].append(i)

    per_class = int((N * float(ratio)) / max(1, num_classes))
    idx = []
    for c in range(num_classes):
        lst = buckets[c]
        rnd.shuffle(lst)
        idx.extend(lst[:per_class])

    rnd.shuffle(idx)
    return idx


# ==========================================================
# 1) Strict FT-SAM Defense (NO Fine-Pruning)
#    - Metrics aligned to BackdoorBench meaning:
#      Clean ACC, ASR, RA(where RA == triggered accuracy w.r.t original labels)
#    - We reuse env.eval.metrics_all_to_one to preserve your trigger/transform pipeline.
# ==========================================================
class FTSAMDefense:
    def __init__(self, env: "ModelEnv", save_dir: str):
        self.env = env
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    @torch.no_grad()
    def _save_final_model(self, model: nn.Module, filename: str):
        """
        Save ONLY the final defended model weights (state_dict).
        This keeps file size minimal and matches FP-style saving behavior.
        """
        save_path = os.path.join(self.save_dir, filename)
        torch.save(unwrap(model).state_dict(), save_path)
        print(f"[Save] final model state_dict saved to: {save_path}")

    def _build_ft_loader(
        self,
        *,
        ft_samples: int = None,
        ft_ratio: float = None,
        seed: int = 0,
        batch_size: int = 64,
        num_workers: int = 8,
    ):
        """
        Prefer BackdoorBench-like class-balanced ratio subset if ft_ratio is provided and trainset is accessible.
        Otherwise fallback to env.data.make_subset_loader(ft_samples) if available.
        """
        if ft_ratio is not None:
            # try to find trainset
            trainset = None
            if hasattr(self.env.data, "trainset"):
                trainset = self.env.data.trainset
            elif hasattr(self.env.data, "trainloader") and hasattr(self.env.data.trainloader, "dataset"):
                trainset = self.env.data.trainloader.dataset

            if trainset is None:
                raise RuntimeError("[FT-SAM] ft_ratio provided but cannot locate trainset in env.data.")

            # infer num_classes
            num_classes = getattr(self.env.data, "num_classes", None)
            if num_classes is None:
                net = unwrap(self.env.model)
                if hasattr(net, "fc") and hasattr(net.fc, "out_features"):
                    num_classes = int(net.fc.out_features)
                else:
                    raise RuntimeError("[FT-SAM] Cannot infer num_classes; set env.data.num_classes.")

            idx = choose_by_class_indices(trainset, num_classes=num_classes, ratio=float(ft_ratio), seed=int(seed))
            subset = Subset(trainset, idx)
            return DataLoader(
                subset,
                batch_size=int(batch_size),
                shuffle=True,
                num_workers=int(num_workers),
                pin_memory=True,
                drop_last=False,
            )

        # fallback: sample by count using env helper
        if ft_samples is None:
            raise RuntimeError("[FT-SAM] Provide either ft_ratio or ft_samples.")
        if not hasattr(self.env.data, "make_subset_loader"):
            raise RuntimeError("[FT-SAM] env.data.make_subset_loader(ft_samples) is not available.")
        return self.env.data.make_subset_loader(int(ft_samples))

    def _finetune_smallset(
        self,
        model,
        ft_loader,
        epochs,
        lr,
        wd,
        *,
        rho_max: float = 2.0,
        rho_min: float = 2.0,
        alpha: float = 0.0,
        adaptive: bool = True,
        label_smoothing: float = 0.1,
        eta_min: float = 0.0,
        print_freq: int = 50,
    ):
        device = self.env.device
        autocast_ctx = getattr(self.env, "autocast_ctx", nullcontext)
        make_scaler = getattr(self.env, "make_scaler", lambda: None)
        scaler = make_scaler()

        base_model = unwrap(model)
        base_model.to(device)

        base_optim = torch.optim.SGD(base_model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
        total_steps = max(1, int(epochs) * len(ft_loader))
        lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(base_optim, T_max=total_steps, eta_min=eta_min)

        rho_sched = ProportionScheduler(
            pytorch_lr_scheduler=lr_sched,
            max_lr=lr,
            min_lr=eta_min,
            max_value=rho_max,
            min_value=rho_min,
        )
        optimizer = SAM(
            params=base_model.parameters(),
            base_optimizer=base_optim,
            model=model,
            rho_scheduler=rho_sched,
            adaptive=adaptive,
            sam_alpha=alpha,
        )
        optimizer.update_rho_t()

        for ep in range(1, epochs + 1):
            model.train().to(device)

            sum_loss = 0.0
            sum_correct = 0
            sum_n = 0

            for it, (x, y) in enumerate(ft_loader, start=1):
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True).long()

                def loss_fn(predictions, targets):
                    return smooth_crossentropy(predictions, targets, smoothing=label_smoothing).mean()

                optimizer.set_closure(loss_fn, x, y, autocast_ctx=autocast_ctx, scaler=scaler)
                preds, loss = optimizer.step()

                with torch.no_grad():
                    correct = (preds.argmax(dim=1) == y).sum().item()

                bsz = int(y.size(0))
                sum_loss += float(loss.item()) * bsz
                sum_correct += int(correct)
                sum_n += bsz

                # per-iteration step (match DSAM)
                lr_sched.step()
                optimizer.update_rho_t()

                if print_freq and (it % print_freq == 0):
                    train_loss = sum_loss / max(1, sum_n)
                    train_acc = 100.0 * (sum_correct / max(1, sum_n))
                    cur_lr = lr_sched.get_last_lr()[0]
                    print(
                        f"[FT-SAM][Ep {ep:02d}/{epochs}][{it:04d}/{len(ft_loader)}] "
                        f"loss {train_loss:.4f} acc {train_acc:.2f}% lr {cur_lr:.6f} rho {optimizer.rho_t:.3f}"
                    )

        return model

    def _finetune_smallset_ab(
        self,
        model,
        ft_loader,
        epochs,
        lr,
        wd,
        **kwargs,
    ):
        return self._finetune_smallset(model, ft_loader, epochs, lr, wd, **kwargs)

    # ==========================================================
    # Public API: strict FT-SAM (single backdoor)
    # ==========================================================
    def run(
        self,
        *,
        ft_samples: int = None,
        ft_ratio: float = None,
        ft_epochs: int = 20,
        ft_lr: float = 1e-3,
        ft_weight_decay: float = 1e-4,
        target_label: int = 1,
        patch_size: int = 3,
        # DSAM knobs
        rho_max: float = 2.0,
        rho_min: float = 2.0,
        alpha: float = 0.0,
        adaptive: bool = True,
        label_smoothing: float = 0.1,
        eta_min: float = 0.0,
        print_freq: int = 50,
        # ratio subset knobs
        seed: int = 0,
        ft_batch_size: int = 64,
        ft_num_workers: int = 8,
        # save
        save_name: str = "ftsam_defended_model.pth",
    ):
        trig = TriggerBadNets(patch_size)

        clean0 = self.env.eval.clean(self.env.model, self.env.data.testloader)
        _, ra0, asr0 = self.env.eval.metrics_all_to_one(self.env.model, self.env.data.testloader, trig, target_label)
        print(f"[FT-SAM BASE] Clean {clean0:.2f}% | ASR {asr0:.2f}% | RA {ra0:.2f}%")

        ft_loader = self._build_ft_loader(
            ft_samples=ft_samples,
            ft_ratio=ft_ratio,
            seed=seed,
            batch_size=ft_batch_size,
            num_workers=ft_num_workers,
        )
        print(f"[FT-SAM] fine-tuning on clean subset: "
              f"{'ratio='+str(ft_ratio) if ft_ratio is not None else 'samples='+str(ft_samples)}")

        self.env.model = self._finetune_smallset(
            self.env.model,
            ft_loader,
            ft_epochs,
            ft_lr,
            ft_weight_decay,
            rho_max=rho_max,
            rho_min=rho_min,
            alpha=alpha,
            adaptive=adaptive,
            label_smoothing=label_smoothing,
            eta_min=eta_min,
            print_freq=print_freq,
        )

        clean1 = self.env.eval.clean(self.env.model, self.env.data.testloader)
        _, ra1, asr1 = self.env.eval.metrics_all_to_one(self.env.model, self.env.data.testloader, trig, target_label)
        print(f"[FT-SAM DONE] Clean {clean1:.2f}% | ASR {asr1:.2f}% | RA {ra1:.2f}%")

        # save final
        self._save_final_model(self.env.model, save_name)

        return dict(clean=clean1, ASR=asr1, RA=ra1)

    # ==========================================================
    # Public API: strict FT-SAM (A/B evaluation)
    # ==========================================================
    def run_ab(
        self,
        *,
        ft_samples: int = None,
        ft_ratio: float = None,
        ft_epochs: int = 20,
        ft_lr: float = 1e-3,
        ft_weight_decay: float = 1e-4,
        target_A: int = 0,
        target_B: int = 1,
        patch_size_A: int = 3,
        sig_amp: float = 0.1,
        sig_freq: float = 6.0,
        # DSAM knobs
        rho_max: float = 2.0,
        rho_min: float = 2.0,
        alpha: float = 0.0,
        adaptive: bool = True,
        label_smoothing: float = 0.1,
        eta_min: float = 0.0,
        print_freq: int = 50,
        # ratio subset knobs
        seed: int = 0,
        ft_batch_size: int = 64,
        ft_num_workers: int = 8,
        # save
        save_name: str = "ftsam_defended_ab_model.pth",
    ):
        trigA = TriggerBadNets(patch_size_A)
        trigB = TriggerSigAB(sig_amp, sig_freq)

        clean0 = self.env.eval.clean(self.env.model, self.env.data.testloader)
        _, raA0, asrA0 = self.env.eval.metrics_all_to_one(self.env.model, self.env.data.testloader, trigA, target_A)
        _, raB0, asrB0 = self.env.eval.metrics_all_to_one(self.env.model, self.env.data.testloader, trigB, target_B)
        print(
            f"[FT-SAM-AB BASE] Clean {clean0:.2f}% | "
            f"A: ASR {asrA0:.2f}% RA {raA0:.2f}% | "
            f"B: ASR {asrB0:.2f}% RA {raB0:.2f}%"
        )

        ft_loader = self._build_ft_loader(
            ft_samples=ft_samples,
            ft_ratio=ft_ratio,
            seed=seed,
            batch_size=ft_batch_size,
            num_workers=ft_num_workers,
        )
        print(f"[FT-SAM-AB] fine-tuning on clean subset: "
              f"{'ratio='+str(ft_ratio) if ft_ratio is not None else 'samples='+str(ft_samples)}")

        self.env.model = self._finetune_smallset_ab(
            self.env.model,
            ft_loader,
            ft_epochs,
            ft_lr,
            ft_weight_decay,
            rho_max=rho_max,
            rho_min=rho_min,
            alpha=alpha,
            adaptive=adaptive,
            label_smoothing=label_smoothing,
            eta_min=eta_min,
            print_freq=print_freq,
        )

        clean1 = self.env.eval.clean(self.env.model, self.env.data.testloader)
        _, raA1, asrA1 = self.env.eval.metrics_all_to_one(self.env.model, self.env.data.testloader, trigA, target_A)
        _, raB1, asrB1 = self.env.eval.metrics_all_to_one(self.env.model, self.env.data.testloader, trigB, target_B)
        print(
            f"[FT-SAM-AB DONE] Clean {clean1:.2f}% | "
            f"A: ASR {asrA1:.2f}% RA {raA1:.2f}% | "
            f"B: ASR {asrB1:.2f}% RA {raB1:.2f}%"
        )

        # save final
        self._save_final_model(self.env.model, save_name)

        return dict(
            clean=clean1,
            ASR_A=asrA1, RA_A=raA1,
            ASR_B=asrB1, RA_B=raB1,
        )


# ==========================================================
# 2) Classic Fine-Pruning (statistics/pruning position aligned)
#    - Stats: last conv feature channels (after ReLU) mean activation
#    - Prune: mask classifier input channels (fc.weight[:, pruned_channels] = 0)
# ==========================================================
class FinePruningDefense:
    def __init__(self, env: "ModelEnv", save_dir: str):
        self.env = env
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    @torch.no_grad()
    def _collect_bn_relu_activation(self, model, calib_loader):
        """
        Classic Fine-Pruning:
          collect mean activation per CHANNEL on the last conv feature map (after ReLU).

        For torchvision-style ResNet:
          hook layer4[-1].relu output: [B,C,H,W] -> mean over (B,H,W) => [C]
        Fallback:
          hook the last nn.ReLU found in the network.
        """
        device = self.env.device
        net = unwrap(model).eval().to(device)

        hook_module = None
        if hasattr(net, "layer4") and isinstance(net.layer4, nn.Sequential) and len(net.layer4) > 0:
            last_block = net.layer4[-1]
            if hasattr(last_block, "relu") and isinstance(last_block.relu, nn.Module):
                hook_module = last_block.relu
            else:
                hook_module = net.layer4
        else:
            for _, m in reversed(list(net.named_modules())):
                if isinstance(m, nn.ReLU):
                    hook_module = m
                    break

        if hook_module is None:
            raise RuntimeError("[FP] Cannot find a suitable ReLU/feature module for activation collection.")

        feat_buf = None

        def forward_hook(module, inp, out):
            nonlocal feat_buf
            feat_buf = out

        h = hook_module.register_forward_hook(forward_hook)

        sum_c = None
        count = 0
        try:
            for xb, _ in calib_loader:
                xb = xb.to(device, non_blocking=True)
                _ = net(xb)

                feat = feat_buf
                if feat is None:
                    raise RuntimeError("[FP] Hook did not capture features (feat_buf is None).")

                if feat.dim() == 4:
                    ch = feat.mean(dim=(2, 3))  # [B,C]
                elif feat.dim() == 2:
                    ch = feat  # [B,C]
                else:
                    raise RuntimeError(f"[FP] Unexpected feature dim: {tuple(feat.shape)}")

                if sum_c is None:
                    sum_c = torch.zeros(ch.size(1), dtype=torch.float64, device=device)
                sum_c += ch.sum(dim=0).double()
                count += ch.size(0)
        finally:
            h.remove()

        mean_c = (sum_c / max(1, count)).cpu()  # [C]
        return {"last_conv": mean_c}

    @torch.no_grad()
    def _fine_prune(
        self,
        model,
        bn_means: dict,
        prune_pct_per_layer: float,
        acc_ratio: float,
        target_label: int,
        patch_size: int,
    ):
        device = self.env.device
        net = unwrap(model).eval().to(device)

        if (not bn_means) or ("last_conv" not in bn_means):
            print("[FP] Warning: empty activation stats; skip pruning.")
            return []

        act_vec = bn_means["last_conv"].to(device)  # [C]
        C = int(act_vec.numel())
        if C <= 0:
            print("[FP] Warning: activation dim <= 0; skip pruning.")
            return []

        seq_sort = torch.argsort(act_vec)  # ascending: prune least-activated channels first
        step = max(1, int(math.ceil(C * float(prune_pct_per_layer))))

        # locate classifier linear (prefer net.fc for resnet)
        classifier = None
        if hasattr(net, "fc") and isinstance(net.fc, nn.Linear):
            classifier = net.fc
        else:
            for m in net.modules():
                if isinstance(m, nn.Linear) and int(m.in_features) == C:
                    classifier = m
                    break
            if classifier is None:
                for m in reversed(list(net.modules())):
                    if isinstance(m, nn.Linear):
                        classifier = m
                        break

        if classifier is None:
            raise RuntimeError("[FP] Cannot find a classifier nn.Linear to prune.")
        if int(classifier.in_features) != C:
            raise RuntimeError(
                f"[FP] Mismatch: last_conv channels C={C} but classifier.in_features={classifier.in_features}. "
                "Classic FP expects pruning channels feeding into the classifier."
            )

        base_w = classifier.weight.data.detach().clone()

        trig = TriggerBadNets(patch_size)
        test_loader = self.env.data.testloader
        report = []

        # baseline (no prune)
        mask0 = torch.ones_like(base_w, device=device)
        classifier.weight.data.copy_(base_w * mask0)

        clean0 = self.env.eval.clean(net, test_loader)
        _, ra0, asr0 = self.env.eval.metrics_all_to_one(net, test_loader, trig, target_label)
        report.append(("num_pruned=0", C, 0, [], float(clean0), float(asr0), float(ra0)))

        best_mask = mask0.clone()
        best_num_pruned = 0
        best_clean, best_asr, best_ra = clean0, asr0, ra0

        for num_pruned in range(step, C + 1, step):
            pruned_idx = seq_sort[:num_pruned]
            mask = torch.ones_like(base_w, device=device)
            mask[:, pruned_idx] = 0
            classifier.weight.data.copy_(base_w * mask)

            clean_acc = self.env.eval.clean(net, test_loader)
            _, ra, asr = self.env.eval.metrics_all_to_one(net, test_loader, trig, target_label)

            report.append(
                (f"num_pruned={num_pruned}", C, int(num_pruned), pruned_idx.tolist(),
                 float(clean_acc), float(asr), float(ra))
            )

            if abs(clean_acc - clean0) / max(1e-6, clean0) <= acc_ratio:
                best_mask = mask.clone()
                best_num_pruned = num_pruned
                best_clean, best_asr, best_ra = clean_acc, asr, ra
            else:
                break

        classifier.weight.data.copy_(base_w * best_mask)

        with open(os.path.join(self.save_dir, "fp_prune_report.txt"), "w") as f:
            for tag, dimC, k, idx, acc, asr, ra in report:
                f.write(f"{tag}: C={dimC}, pruned={k}, idx={idx}, Clean={acc:.4f}, ASR={asr:.4f}, RA={ra:.4f}\n")

        print(
            f"[FP] Prune done. Best pruned={best_num_pruned}/{C} | "
            f"Clean {best_clean:.2f}% | ASR {best_asr:.2f}% | RA {best_ra:.2f}%"
        )

        return report


# ==========================================================
# Minimal runnable example
# ==========================================================
if __name__ == "__main__":
    # You must define these in your environment (or replace them here)
    # data_path = "..."
    # ckpt_path = "..."

    CORE_CFG = dict(
        DATA_ROOT=data_path,
        CKPT_PATH=ckpt_path,
        VARIANT="resnet18",
        BATCH_SIZE=64,
        NUM_WORKERS=8,
        SEED=0,
        AMP=True,
    )
    env = ModelEnv(CORE_CFG)

    # baseline (if your env supports AB baseline helper)
    try:
        env.baseline_ab(target_A=0, target_B=1, patch_size_A=3, sig_amp=0.1, sig_freq=6.0)
    except Exception:
        pass

    ft = FTSAMDefense(env, save_dir="ftsam_defense_resnet18_cifar10_ab")
    ft.run_ab(
        # choose ONE: ft_ratio (recommended) OR ft_samples (fallback)
        ft_ratio=0.02,
        ft_samples=1000,

        ft_epochs=5,
        ft_lr=2e-3,
        ft_weight_decay=1e-4,

        target_A=0,
        target_B=1,
        patch_size_A=3,
        sig_amp=0.1,
        sig_freq=6.0,

        rho_max=2.0,
        rho_min=2.0,
        alpha=0.0,
        adaptive=True,
        label_smoothing=0.1,
        eta_min=0.0,
        print_freq=50,

        seed=0,
        ft_batch_size=64,
        ft_num_workers=8,

        # save final defended model here:
        save_name="ftsam_defended_ab_model.pth",
    )
