# Extracted from BB/anbition_denfense-Final-abition_k_3_repeat_denfense.ipynb, code cell 6 (zero-based).
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, math
import csv
import copy
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune


# ==========================================================
# Minimal helper
# ==========================================================
def unwrap(m: nn.Module) -> nn.Module:
    return m.module if hasattr(m, "module") else m


class FinePruningDefense:
    """
    目标：
    1) 每轮保存 defended 模型：fp_defended_ab_model_roundXX.pth
    2) 重复防御 n 次：上一轮输出作为下一轮输入
    3) 仅写一个“总 CSV”记录每一轮各阶段指标（before / after_prune / after_ft），不再产生：
       - fp_prune_report_roundXX.txt
       - fp_defense.log

    修改后的保存逻辑（按你的最新要求）：
    - 每一轮不再固定保存“微调最后一个 epoch 的模型”；
    - 而是在本轮的候选集合中（默认包含：before / after_prune / finetune过程中满足条件的若干epoch-best）
      选出 “Clean ACC 相对本轮起始模型（即进入本轮时的当前模型 before）下降不超过 save_acc_ratio”
      的可行候选里，“A 的 ASR 最低”的那个模型作为本轮最终保存模型，并作为下一轮输入。
    - save_acc_ratio 可调（默认 5%）。
    - 额外规则：若最低 ASR 的可行候选仍然是本轮起始模型（before），则改选“次优”（即在可行候选中排除 before 后的最优）。
      若排除 before 后没有其它可行候选，则仍只能选择 before（兜底）。

    说明：
    - “与当前的模型比较，而不是与最开始的模型比较”这里的“当前”指每一轮开始时的输入模型（before），
      不是第 1 轮最初模型；因此每轮 clean 参考值都会更新为该轮 before 的 clean。
    """

    def __init__(self, env: "ModelEnv", save_dir: str, metrics_csv: str = "fp_defense_total.csv"):
        self.env = env
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.metrics_csv_path = os.path.join(self.save_dir, metrics_csv)
        self._ensure_metrics_csv_header()

    def _ensure_metrics_csv_header(self):
        if os.path.exists(self.metrics_csv_path):
            return
        fieldnames = self._metrics_fieldnames()
        with open(self.metrics_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    def _metrics_fieldnames(self):
        return [
            "timestamp",
            "round",
            "model_path",
            # config
            "calib_samples",
            "prune_pct_per_layer",
            "acc_ratio",
            "save_acc_ratio",
            "ft_samples",
            "ft_epochs",
            "ft_lr",
            "ft_weight_decay",
            "target_A",
            "target_B",
            "patch_size_A",
            "sig_amp",
            "sig_freq",
            # pruning summary
            "feature_dim_D",
            "best_pruned_k",
            # metrics: BEFORE
            "clean_before",
            "A_triggered_before",
            "ASR_A_before",
            "B_triggered_before",
            "ASR_B_before",
            # metrics: AFTER PRUNE
            "clean_after_prune",
            "A_triggered_after_prune",
            "ASR_A_after_prune",
            "B_triggered_after_prune",
            "ASR_B_after_prune",
            # metrics: AFTER FT (NOTE: this is FINAL-SAVED metrics after selection)
            "clean_after_ft",
            "A_triggered_after_ft",
            "ASR_A_after_ft",
            "B_triggered_after_ft",
            "ASR_B_after_ft",
        ]

    def _append_metrics_row(self, row: dict):
        fieldnames = self._metrics_fieldnames()
        for k in fieldnames:
            row.setdefault(k, None)
        with open(self.metrics_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow({k: row.get(k) for k in fieldnames})
            f.flush()

    def _strip_all_pruning_reparam(self, model):
        """
        去掉所有 pruning reparam，避免 weight_orig/weight_mask 进入 state_dict，
        并保证可重复运行防御。
        """
        net = unwrap(model)
        for m in net.modules():
            if hasattr(m, "weight_orig"):
                try:
                    prune.remove(m, "weight")
                except Exception:
                    pass
        return model

    def _get_plain_state_dict(self, model) -> dict:
        """
        返回一个“可直接 load 到原网络”的 plain state_dict：
        - 不包含 weight_orig/weight_mask
        - 若当前 model 里有 pruning reparam，会在 deepcopy 的副本上 remove 后再取 state_dict
        """
        net = copy.deepcopy(unwrap(model))
        for m in net.modules():
            if hasattr(m, "weight_orig"):
                try:
                    prune.remove(m, "weight")
                except Exception:
                    pass
        return net.state_dict()

    def _load_plain_state_dict_inplace(self, model, plain_state_dict: dict):
        """
        把 model 变成“plain 参数结构”并加载 plain_state_dict。
        """
        self._strip_all_pruning_reparam(model)
        unwrap(model).load_state_dict(plain_state_dict, strict=True)
        return model

    def _save_model_state(self, model, filename: str) -> str:
        save_path = os.path.join(self.save_dir, filename)
        net = unwrap(model)
        torch.save(net.state_dict(), save_path)
        print(f"[Save] model state_dict saved to: {save_path}")
        return save_path

    # ==========================================================
    # 1) 统计“最后一层分类器输入特征”的平均激活
    # ==========================================================
    @torch.no_grad()
    def _collect_bn_relu_activation(self, model, calib_loader):
        """
        hook 最后一个顶层 child（对 ResNet18 来说通常是 fc），记录其输入特征 input[0] ∈ R^{B×D}
        对 D 维特征求均值，返回 {"last": activation[D]}（CPU）。
        """
        device = self.env.device
        net = unwrap(model).eval().to(device)

        last_child_name, last_child = list(net.named_children())[-1]
        result_mid = None

        def forward_hook(module, inp, out):
            nonlocal result_mid
            result_mid = inp[0]  # [B, D]

        hook = last_child.register_forward_hook(forward_hook)

        activation = None
        total = 0
        try:
            for xb, _ in calib_loader:
                xb = xb.to(device, non_blocking=True)
                _ = net(xb)
                feat = result_mid  # [B, D]
                if activation is None:
                    activation = torch.zeros(feat.size(1), dtype=torch.float64, device=device)
                activation += feat.sum(dim=0).double()
                total += feat.size(0)
        finally:
            hook.remove()

        activation /= max(1, total)
        return {"last": activation.cpu()}

    # ==========================================================
    # 2) Fine-Pruning 核心算法（无 report 文件输出）
    # ==========================================================
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

        if not bn_means:
            print("[FP] Warning: empty activation stats; skip pruning.")
            return dict(feature_dim_D=0, best_pruned_k=0)

        act_vec = next(iter(bn_means.values())).to(device)  # [D]
        D = int(act_vec.numel())
        if D <= 0:
            print("[FP] Warning: activation dim <= 0; skip pruning.")
            return dict(feature_dim_D=0, best_pruned_k=0)

        seq_sort = torch.argsort(act_vec)  # 升序：越小越“不重要”

        # 找到最后一个 child 里的第一个 Linear
        last_child_name, last_child = list(net.named_children())[-1]
        first_linear = None
        for _, m in last_child.named_modules():
            if isinstance(m, nn.Linear):
                first_linear = m
                break
        if first_linear is None:
            raise RuntimeError("FinePruningDefense: no nn.Linear found in last child; cannot prune.")

        # ----（可重复运行防御时）先把已有的 prune reparam 清掉，并恢复到未剪枝权重 ----
        if hasattr(first_linear, "weight_orig"):
            base_w = first_linear.weight_orig.data.detach().clone()
            try:
                prune.remove(first_linear, "weight")
            except Exception:
                pass
            first_linear.weight.data.copy_(base_w)
        else:
            base_w = first_linear.weight.data.detach().clone()

        # ---- 初始化累积 mask ----
        prune_mask = torch.ones_like(base_w, device=device)

        test_loader = self.env.data.testloader

        # baseline：num_pruned = 0
        first_linear.weight.data.copy_(base_w)
        clean0 = self.env.eval.clean(net, test_loader)

        best_mask = prune_mask.clone()
        best_num_pruned = 0
        best_clean = clean0

        ratio = float(prune_pct_per_layer)
        step = int(math.ceil(D * ratio))  # 注意：不再 max(1, ...)
        if step <= 0:
            if hasattr(first_linear, "weight_orig"):
                try:
                    prune.remove(first_linear, "weight")
                except Exception:
                    pass
                first_linear.weight.data.copy_(base_w)

            print(f"[FP] prune_pct_per_layer={prune_pct_per_layer} => step=0, skip pruning.")
            print(f"[FP] Prune done. Best pruned={best_num_pruned}/{D} | Clean {best_clean:.2f}%")
            return dict(feature_dim_D=D, best_pruned_k=best_num_pruned)

        def _apply_or_update_mask(mask: torch.Tensor):
            if not hasattr(first_linear, "weight_orig"):
                prune.custom_from_mask(first_linear, name="weight", mask=mask.to(device))
            else:
                first_linear.weight_mask.data.copy_(mask.to(device))

        for num_pruned in range(0, D, step):
            # off-by-one：剪 [0 : num_pruned-1]
            if num_pruned >= 2:
                pruned_idx = seq_sort[0: num_pruned - 1]
                if pruned_idx.numel() > 0:
                    prune_mask[:, pruned_idx] = 0
                    _apply_or_update_mask(prune_mask)

            clean_acc = self.env.eval.clean(net, test_loader)

            if abs(clean_acc - clean0) / max(1e-6, clean0) <= acc_ratio:
                if num_pruned >= 2:
                    cur_pruned_idx = seq_sort[0: num_pruned - 1]
                    cur_k = int(cur_pruned_idx.numel())
                else:
                    cur_k = 0

                best_mask = prune_mask.clone()
                best_num_pruned = cur_k
                best_clean = clean_acc
            else:
                break

        # ---- 应用“最后一次满足阈值”的 mask ----
        if hasattr(first_linear, "weight_orig"):
            try:
                prune.remove(first_linear, "weight")
            except Exception:
                pass
        first_linear.weight.data.copy_(base_w)
        if (best_mask == 0).any():
            prune.custom_from_mask(first_linear, name="weight", mask=best_mask.to(device))

        print(f"[FP] Prune done. Best pruned={best_num_pruned}/{D} | Clean {best_clean:.2f}%")
        return dict(feature_dim_D=D, best_pruned_k=best_num_pruned)

    # ==========================================================
    # 4) 小样本微调（A/B 双后门版） + “epoch 内候选”选择
    # ==========================================================
    def _finetune_smallset_ab_with_selection(
        self,
        model,
        ft_loader,
        epochs,
        lr,
        wd,
        target_A,
        target_B,
        patch_size_A,
        sig_amp,
        sig_freq,
        *,
        min_clean_allowed: float,
    ):
        device = self.env.device
        autocast_ctx, make_scaler = self.env.autocast_ctx, self.env.make_scaler
        scaler = make_scaler()
        optim = torch.optim.SGD(unwrap(model).parameters(), lr=lr, momentum=0.9, weight_decay=wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)

        trigA = TriggerBadNets(patch_size_A)
        trigB = TriggerSigAB(sig_amp, sig_freq)

        best_sd_plain = None
        best_epoch = 0
        best_asrA = float("inf")
        best_clean = -float("inf")

        last_metrics = None

        for ep in range(1, epochs + 1):
            model.train().to(device)
            run_loss = n = 0
            for x, y in ft_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                optim.zero_grad(set_to_none=True)
                with autocast_ctx():
                    logits = model(x)
                    loss = F.cross_entropy(logits, y)
                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()
                run_loss += float(loss.item()) * x.size(0)
                n += x.size(0)
            sched.step()
            ce = run_loss / max(1, n)

            clean = float(self.env.eval.clean(model, self.env.data.testloader))
            _, trigA_acc, asrA = self.env.eval.metrics_all_to_one(
                model, self.env.data.testloader, trigA, target_A
            )
            _, trigB_acc, asrB = self.env.eval.metrics_all_to_one(
                model, self.env.data.testloader, trigB, target_B
            )
            trigA_acc, asrA = float(trigA_acc), float(asrA)
            trigB_acc, asrB = float(trigB_acc), float(asrB)

            print(
                f"[FT-AB Ep {ep:02d}/{epochs}] CE {ce:.4f} | Clean {clean:.2f}% | "
                f"A: Trig {trigA_acc:.2f}% ASR_A {asrA:.2f}% | "
                f"B: Trig {trigB_acc:.2f}% ASR_B {asrB:.2f}%"
            )

            last_metrics = dict(
                epoch=ep,
                clean=clean,
                A_triggered=trigA_acc,
                ASR_A=asrA,
                B_triggered=trigB_acc,
                ASR_B=asrB,
            )

            if clean >= float(min_clean_allowed):
                better = False
                if asrA < best_asrA:
                    better = True
                elif asrA == best_asrA and clean > best_clean:
                    better = True

                if better:
                    best_asrA = asrA
                    best_clean = clean
                    best_epoch = ep
                    best_sd_plain = self._get_plain_state_dict(model)

        if best_sd_plain is None:
            best_sd_plain = self._get_plain_state_dict(model)
            best_epoch = epochs
            best_asrA = float(last_metrics["ASR_A"]) if last_metrics else float("inf")
            best_clean = float(last_metrics["clean"]) if last_metrics else -float("inf")

        best_info = dict(
            best_epoch=best_epoch,
            best_clean=float(best_clean),
            best_asrA=float(best_asrA),
        )
        return model, best_sd_plain, best_info, last_metrics

    # ==========================================================
    # A/B：单轮防御 + 保存模型 + 记录到“总 CSV”（一个文件）
    # ==========================================================
    def run_ab(
        self,
        *,
        calib_samples: int,
        prune_pct_per_layer: float,
        ft_samples: int = 100,
        ft_epochs: int = 20,
        ft_lr: float = 1e-3,
        ft_weight_decay: float = 1e-4,
        target_A: int = 0,
        target_B: int = 1,
        patch_size_A: int = 3,
        sig_amp: float = 0.1,
        sig_freq: float = 6.0,
        acc_ratio: float = 0.1,
        save_acc_ratio: float = 0.05,
        round_idx: int = 1,
    ):
        rid = int(round_idx)
        print(f"\n==================== Round {rid:02d} ====================")

        trigA = TriggerBadNets(patch_size_A)
        trigB = TriggerSigAB(sig_amp, sig_freq)

        # ====== snapshot: BEFORE model (plain) ======
        before_sd_plain = self._get_plain_state_dict(self.env.model)

        # -------- BEFORE metrics --------
        clean_before = float(self.env.eval.clean(self.env.model, self.env.data.testloader))
        _, A_trig_before, ASR_A_before = self.env.eval.metrics_all_to_one(
            self.env.model, self.env.data.testloader, trigA, target_A
        )
        _, B_trig_before, ASR_B_before = self.env.eval.metrics_all_to_one(
            self.env.model, self.env.data.testloader, trigB, target_B
        )
        A_trig_before, ASR_A_before = float(A_trig_before), float(ASR_A_before)
        B_trig_before, ASR_B_before = float(B_trig_before), float(ASR_B_before)

        print(
            f"[Before] Clean {clean_before:.2f}% | "
            f"A: Triggered {A_trig_before:.2f}% ASR_A {ASR_A_before:.2f}% | "
            f"B: Triggered {B_trig_before:.2f}% ASR_B {ASR_B_before:.2f}%"
        )

        clean_ref = float(clean_before)
        min_clean_allowed = clean_ref * (1.0 - float(save_acc_ratio))

        # -------- Calib + prune --------
        calib_loader = self.env.data.make_calib_loader(calib_samples)
        print(f"[Calib-AB] collecting last-layer activations on {calib_samples} clean samples ...")
        bn_means = self._collect_bn_relu_activation(self.env.model, calib_loader)

        prune_info = self._fine_prune(
            self.env.model,
            bn_means,
            prune_pct_per_layer,
            acc_ratio=acc_ratio,
            target_label=target_A,
            patch_size=patch_size_A,
        )

        # -------- AFTER PRUNE metrics --------
        clean_after_prune = float(self.env.eval.clean(self.env.model, self.env.data.testloader))
        _, A_trig_after_prune, ASR_A_after_prune = self.env.eval.metrics_all_to_one(
            self.env.model, self.env.data.testloader, trigA, target_A
        )
        _, B_trig_after_prune, ASR_B_after_prune = self.env.eval.metrics_all_to_one(
            self.env.model, self.env.data.testloader, trigB, target_B
        )
        A_trig_after_prune, ASR_A_after_prune = float(A_trig_after_prune), float(ASR_A_after_prune)
        B_trig_after_prune, ASR_B_after_prune = float(B_trig_after_prune), float(ASR_B_after_prune)

        print(
            f"[After-Prune] Clean {clean_after_prune:.2f}% | "
            f"A: Triggered {A_trig_after_prune:.2f}% ASR_A {ASR_A_after_prune:.2f}% | "
            f"B: Triggered {B_trig_after_prune:.2f}% ASR_B {ASR_B_after_prune:.2f}%"
        )

        after_prune_sd_plain = self._get_plain_state_dict(self.env.model)

        # -------- Fine-tune (with epoch-level selection candidates) --------
        ft_loader = self.env.data.make_subset_loader(ft_samples)
        print(
            f"[FT-AB] using {ft_samples} clean samples for fine-tuning. "
            f"(min_clean_allowed={min_clean_allowed:.2f}%, save_acc_ratio={save_acc_ratio:.3f})"
        )

        self.env.model, ft_best_sd_plain, ft_best_info, ft_last_metrics = self._finetune_smallset_ab_with_selection(
            self.env.model,
            ft_loader,
            ft_epochs,
            ft_lr,
            ft_weight_decay,
            target_A,
            target_B,
            patch_size_A,
            sig_amp,
            sig_freq,
            min_clean_allowed=min_clean_allowed,
        )

        if ft_last_metrics is not None:
            print(
                f"[After-FT(last ep)] Ep {ft_last_metrics['epoch']:02d} | "
                f"Clean {ft_last_metrics['clean']:.2f}% | "
                f"A: Triggered {ft_last_metrics['A_triggered']:.2f}% ASR_A {ft_last_metrics['ASR_A']:.2f}% | "
                f"B: Triggered {ft_last_metrics['B_triggered']:.2f}% ASR_B {ft_last_metrics['ASR_B']:.2f}%"
            )

        print(
            f"[FT-BestCandidate] Ep {ft_best_info['best_epoch']:02d} | "
            f"Clean {ft_best_info['best_clean']:.2f}% | ASR_A {ft_best_info['best_asrA']:.2f}%"
        )

        # ==========================================================
        # 本轮最终保存模型选择逻辑（UPDATED）：
        #   1) 候选集合：before / after_prune / ft_best_epXX
        #   2) 仅保留 clean >= clean_before*(1-save_acc_ratio) 的候选
        #   3) 在可行候选里，选 ASR_A 最低（tie: clean 更高优先）
        #   4) 若“最优”仍然是 before（本轮起始模型），则改为选择“次优”：
        #        即在可行候选中排除 before 后，再按相同规则选最优。
        #      若排除 before 后为空，则只能选择 before（兜底）。
        # ==========================================================
        candidates = [
            ("before", clean_before, ASR_A_before, before_sd_plain),
            ("after_prune", clean_after_prune, ASR_A_after_prune, after_prune_sd_plain),
            (f"ft_best_ep{int(ft_best_info['best_epoch']):02d}", float(ft_best_info["best_clean"]), float(ft_best_info["best_asrA"]), ft_best_sd_plain),
        ]

        feasible = [c for c in candidates if float(c[1]) >= float(min_clean_allowed)]
        if len(feasible) == 0:
            feasible = [("before", clean_before, ASR_A_before, before_sd_plain)]

        # primary selection
        feasible.sort(key=lambda x: (float(x[2]), -float(x[1])))
        chosen_stage, chosen_clean, chosen_asrA, chosen_sd = feasible[0]

        # NEW: if best is still the starting model, pick the best among non-before (if exists)
        if chosen_stage == "before":
            feasible_non_before = [c for c in feasible if c[0] != "before"]
            if len(feasible_non_before) > 0:
                feasible_non_before.sort(key=lambda x: (float(x[2]), -float(x[1])))
                chosen_stage, chosen_clean, chosen_asrA, chosen_sd = feasible_non_before[0]
                print("[Select] best feasible is 'before'; switch to next-best feasible (exclude before).")
            else:
                print("[Select] best feasible is 'before' and no other feasible candidates; keep before (fallback).")

        print(
            f"[Select] save_acc_ratio={save_acc_ratio:.3f} | "
            f"clean_ref={clean_ref:.2f}% min_clean={min_clean_allowed:.2f}% | "
            f"chosen={chosen_stage} (Clean {float(chosen_clean):.2f}% | ASR_A {float(chosen_asrA):.2f}%)"
        )

        # restore chosen model (plain), and use as next-round input
        self._load_plain_state_dict_inplace(self.env.model, chosen_sd)

        # -------- FINAL (saved) metrics --------
        clean_after_ft = float(self.env.eval.clean(self.env.model, self.env.data.testloader))
        _, A_trig_after_ft, ASR_A_after_ft = self.env.eval.metrics_all_to_one(
            self.env.model, self.env.data.testloader, trigA, target_A
        )
        _, B_trig_after_ft, ASR_B_after_ft = self.env.eval.metrics_all_to_one(
            self.env.model, self.env.data.testloader, trigB, target_B
        )
        A_trig_after_ft, ASR_A_after_ft = float(A_trig_after_ft), float(ASR_A_after_ft)
        B_trig_after_ft, ASR_B_after_ft = float(B_trig_after_ft), float(ASR_B_after_ft)

        print(
            f"[Final-Saved] (from {chosen_stage}) Clean {clean_after_ft:.2f}% | "
            f"A: Triggered {A_trig_after_ft:.2f}% ASR_A {ASR_A_after_ft:.2f}% | "
            f"B: Triggered {B_trig_after_ft:.2f}% ASR_B {ASR_B_after_ft:.2f}%"
        )

        # -------- Save model (one .pth per round): SAVE THE CHOSEN MODEL --------
        self._strip_all_pruning_reparam(self.env.model)
        model_filename = f"fp_defended_ab_model_round{rid:02d}.pth"
        model_path = self._save_model_state(self.env.model, model_filename)

        # -------- Append ONE row to TOTAL CSV --------
        row = dict(
            timestamp=datetime.now().isoformat(timespec="seconds"),
            round=rid,
            model_path=model_path,
            calib_samples=calib_samples,
            prune_pct_per_layer=prune_pct_per_layer,
            acc_ratio=acc_ratio,
            save_acc_ratio=save_acc_ratio,
            ft_samples=ft_samples,
            ft_epochs=ft_epochs,
            ft_lr=ft_lr,
            ft_weight_decay=ft_weight_decay,
            target_A=target_A,
            target_B=target_B,
            patch_size_A=patch_size_A,
            sig_amp=sig_amp,
            sig_freq=sig_freq,
            feature_dim_D=prune_info.get("feature_dim_D", None),
            best_pruned_k=prune_info.get("best_pruned_k", None),

            clean_before=float(clean_before),
            A_triggered_before=float(A_trig_before),
            ASR_A_before=float(ASR_A_before),
            B_triggered_before=float(B_trig_before),
            ASR_B_before=float(ASR_B_before),

            clean_after_prune=float(clean_after_prune),
            A_triggered_after_prune=float(A_trig_after_prune),
            ASR_A_after_prune=float(ASR_A_after_prune),
            B_triggered_after_prune=float(B_trig_after_prune),
            ASR_B_after_prune=float(ASR_B_after_prune),

            clean_after_ft=float(clean_after_ft),
            A_triggered_after_ft=float(A_trig_after_ft),
            ASR_A_after_ft=float(ASR_A_after_ft),
            B_triggered_after_ft=float(B_trig_after_ft),
            ASR_B_after_ft=float(ASR_B_after_ft),
        )
        self._append_metrics_row(row)
        print(f"[CSV] appended metrics to: {self.metrics_csv_path}")

        return row

    # ==========================================================
    # 重复 n 次：所有轮次指标写入同一个“总 CSV”
    # ==========================================================
    def run_ab_n_times(self, *, n_times: int, **kwargs):
        assert n_times >= 1, "n_times must be >= 1"
        results = []
        for i in range(1, n_times + 1):
            res = self.run_ab(round_idx=i, **kwargs)
            results.append(res)
        print(f"\n[Done] Total CSV: {self.metrics_csv_path}")
        return results


# ------------------- 可选：最小示例 -------------------
if __name__ == "__main__":
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
    try:
        env.baseline_ab(target_A=0, target_B=1, patch_size_A=3, sig_amp=0.1, sig_freq=6.0)
    except AttributeError:
        env.baseline_badnets(target_label=1, patch_size=3)

    fp = FinePruningDefense(env, save_dir="fp_defense_resnet18_cifar10_ab", metrics_csv="fp_defense_total.csv")

    fp.run_ab_n_times(
        n_times=10,
        calib_samples=1000,
        prune_pct_per_layer=0.05,  # 设为 0.0 会严格跳过剪枝
        ft_samples=1000,
        ft_epochs=10,
        ft_lr=2e-3,
        ft_weight_decay=1e-4,
        target_A=0,
        target_B=1,
        patch_size_A=3,
        sig_amp=0.1,
        sig_freq=6.0,
        acc_ratio=0.1,        # pruning 内部阈值（决定 best_pruned_k）
        save_acc_ratio=0.05,  # 本轮最终保存模型 clean 允许下降比例（相对本轮 before）
    )
