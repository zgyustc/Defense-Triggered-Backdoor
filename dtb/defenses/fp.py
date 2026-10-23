# Extracted from BB/anbition_denfense-Final-abition_k_3.ipynb, code cell 6 (zero-based).
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune

class FinePruningDefense:
    def __init__(self, env: "ModelEnv", save_dir: str):
        self.env = env
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

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
    # 2) Fine-Pruning 核心算法（对齐 3 个关键点）
    #    - 剪枝施加方式：prune.custom_from_mask
    #    - 剪枝方式：累积式 prune_mask
    #    - off-by-one：seq_sort[0:num_pruned-1]
    # ==========================================================
    @torch.no_grad()
    def _fine_prune(
        self,
        model,
        bn_means: dict,
        prune_pct_per_layer: float,  # 对齐 once_prune_ratio
        acc_ratio: float,
        target_label: int,
        patch_size: int,
    ):
        device = self.env.device
        net = unwrap(model).eval().to(device)

        if not bn_means:
            print("[FP] Warning: empty activation stats; skip pruning.")
            return []

        act_vec = next(iter(bn_means.values())).to(device)  # [D]
        D = int(act_vec.numel())
        if D <= 0:
            print("[FP] Warning: activation dim <= 0; skip pruning.")
            return []

        seq_sort = torch.argsort(act_vec)  # 升序：越小越“不重要”

        # 找到最后一个 child 里的第一个 Linear
        last_child_name, last_child = list(net.named_children())[-1]
        first_linear = None
        for _, m in last_child.named_modules():
            if isinstance(m, nn.Linear):
                first_linear = m
                break
        if first_linear is None:
            raise RuntimeError(
                "FinePruningDefense: no nn.Linear found in last child; cannot prune."
            )

        # ----（可重复运行防御时）先把已有的 prune reparam 清掉，并恢复到未剪枝权重 ----
        if hasattr(first_linear, "weight_orig"):
            base_w = first_linear.weight_orig.data.detach().clone()
            try:
                prune.remove(first_linear, "weight")
            except Exception:
                pass
            with torch.no_grad():
                first_linear.weight.data.copy_(base_w)
        else:
            base_w = first_linear.weight.data.detach().clone()

        # ---- 初始化累积 mask（对齐参考实现：prune_mask 是“累积式”）----
        prune_mask = torch.ones_like(base_w, device=device)

        trig = TriggerBadNets(patch_size)
        test_loader = self.env.data.testloader

        report = []

        # baseline：num_pruned = 0（不施加 prune.custom_from_mask）
        with torch.no_grad():
            first_linear.weight.data.copy_(base_w)

        clean0 = self.env.eval.clean(net, test_loader)
        _, trig0, asr0 = self.env.eval.metrics_all_to_one(net, test_loader, trig, target_label)
        report.append(("num_pruned=0", D, 0, [], float(clean0), float(asr0)))

        best_mask = prune_mask.clone()
        best_num_pruned = 0
        best_clean, best_asr = clean0, asr0

        # ========= 关键修复：ratio=0 时 step=0，直接跳过剪枝 =========
        ratio = float(prune_pct_per_layer)
        step = int(math.ceil(D * ratio))  # 注意：不再 max(1, ...)
        if step <= 0:
            # 确保没有遗留的 pruning reparam
            if hasattr(first_linear, "weight_orig"):
                try:
                    prune.remove(first_linear, "weight")
                except Exception:
                    pass
                with torch.no_grad():
                    first_linear.weight.data.copy_(base_w)

            # 保存报告（仅 baseline）
            with open(os.path.join(self.save_dir, "fp_prune_report.txt"), "w") as f:
                for tag, C, k, idx, acc, asr in report:
                    f.write(
                        f"{tag}: D={C}, pruned={k}, idx={idx}, ACC={acc:.4f}, ASR={asr:.4f}\n"
                    )

            print(f"[FP] prune_pct_per_layer={prune_pct_per_layer} => step=0, skip pruning.")
            print(
                f"[FP] Prune done. Best pruned={best_num_pruned}/{D} | "
                f"Clean {best_clean:.2f}% | ASR {best_asr:.2f}%"
            )
            return report
        # =========================================================

        # ---- 辅助：避免循环里反复 custom_from_mask 叠加 reparam
        def _apply_or_update_mask(mask: torch.Tensor):
            if not hasattr(first_linear, "weight_orig"):
                prune.custom_from_mask(first_linear, name="weight", mask=mask.to(device))
            else:
                # 只更新 mask（保持“累积式”语义）
                first_linear.weight_mask.data.copy_(mask.to(device))

        # ---- 对齐参考实现：循环从 0 开始；每次用累计 prune_mask；off-by-one 索引 ----
        for num_pruned in range(0, D, step):
            # off-by-one：剪 [0 : num_pruned-1]
            # 注意：num_pruned=1 时 slice 为空，不应触发任何 pruning 行为
            if num_pruned >= 2:
                pruned_idx = seq_sort[0 : num_pruned - 1]
                if pruned_idx.numel() > 0:
                    prune_mask[:, pruned_idx] = 0
                    _apply_or_update_mask(prune_mask)

            # test
            clean_acc = self.env.eval.clean(net, test_loader)
            _, trig_acc, asr = self.env.eval.metrics_all_to_one(net, test_loader, trig, target_label)

            # 记录：这里的“实际剪枝数”是 len(pruned_idx)，对齐 off-by-one 行为
            if num_pruned >= 2:
                cur_pruned_idx = seq_sort[0 : num_pruned - 1]
                cur_k = int(cur_pruned_idx.numel())
                cur_idx_list = cur_pruned_idx.tolist()
            else:
                cur_k = 0
                cur_idx_list = []

            report.append(
                (f"num_pruned={num_pruned}", D, cur_k, cur_idx_list, float(clean_acc), float(asr))
            )

            # 对齐参考实现的选择逻辑：ACC 相对下降在阈值内则接受，否则 break
            if abs(clean_acc - clean0) / max(1e-6, clean0) <= acc_ratio:
                best_mask = prune_mask.clone()
                best_num_pruned = cur_k
                best_clean, best_asr = clean_acc, asr
            else:
                break

        # ---- 应用“最后一次满足阈值”的 mask ----
        if hasattr(first_linear, "weight_orig"):
            try:
                prune.remove(first_linear, "weight")
            except Exception:
                pass
        with torch.no_grad():
            first_linear.weight.data.copy_(base_w)
        if (best_mask == 0).any():
            prune.custom_from_mask(first_linear, name="weight", mask=best_mask.to(device))

        # 保存报告
        with open(os.path.join(self.save_dir, "fp_prune_report.txt"), "w") as f:
            for tag, C, k, idx, acc, asr in report:
                f.write(
                    f"{tag}: D={C}, pruned={k}, idx={idx}, ACC={acc:.4f}, ASR={asr:.4f}\n"
                )

        print(
            f"[FP] Prune done. Best pruned={best_num_pruned}/{D} | "
            f"Clean {best_clean:.2f}% | ASR {best_asr:.2f}%"
        )

        return report

    # ==========================================================
    # 3) 小样本微调（BadNets 单后门版）
    # ==========================================================
    def _finetune_smallset(self, model, ft_loader, epochs, lr, wd, target_label, patch_size):
        device = self.env.device
        autocast_ctx, make_scaler = self.env.autocast_ctx, self.env.make_scaler
        scaler = make_scaler()
        optim = torch.optim.SGD(unwrap(model).parameters(), lr=lr, momentum=0.9, weight_decay=wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)
        trig = TriggerBadNets(patch_size)

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

            clean = self.env.eval.clean(model, self.env.data.testloader)
            _, trig_acc, asr = self.env.eval.metrics_all_to_one(
                model, self.env.data.testloader, trig, target_label
            )
            print(
                f"[FT Ep {ep:02d}/{epochs}] CE {ce:.4f} | Clean {clean:.2f}% | "
                f"Triggered {trig_acc:.2f}% | ASRnt {asr:.2f}%"
            )
        return model

    # ==========================================================
    # 4) 小样本微调（A/B 双后门版）
    # ==========================================================
    def _finetune_smallset_ab(
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
    ):
        device = self.env.device
        autocast_ctx, make_scaler = self.env.autocast_ctx, self.env.make_scaler
        scaler = make_scaler()
        optim = torch.optim.SGD(unwrap(model).parameters(), lr=lr, momentum=0.9, weight_decay=wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)

        trigA = TriggerBadNets(patch_size_A)
        trigB = TriggerSigAB(sig_amp, sig_freq)

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

            clean = self.env.eval.clean(model, self.env.data.testloader)
            _, trigA_acc, asrA = self.env.eval.metrics_all_to_one(
                model, self.env.data.testloader, trigA, target_A
            )
            _, trigB_acc, asrB = self.env.eval.metrics_all_to_one(
                model, self.env.data.testloader, trigB, target_B
            )
            print(
                f"[FT-AB Ep {ep:02d}/{epochs}] CE {ce:.4f} | Clean {clean:.2f}% | "
                f"A: Trig {trigA_acc:.2f}% ASR_A {asrA:.2f}% | "
                f"B: Trig {trigB_acc:.2f}% ASR_B {asrB:.2f}%"
            )
        return model

    # ==========================================================
    # 5) 对外入口：单 BadNets FP 防御
    # ==========================================================
    def run(
        self,
        *,
        calib_samples: int,
        prune_pct_per_layer: float,
        ft_samples: int = 100,
        ft_epochs: int = 20,
        ft_lr: float = 1e-3,
        ft_weight_decay: float = 1e-4,
        target_label: int = 1,
        patch_size: int = 3,
        acc_ratio: float = 0.1,
    ):
        calib_loader = self.env.data.make_calib_loader(calib_samples)
        print(f"[Calib] collecting last-layer activations on {calib_samples} clean samples ...")
        bn_means = self._collect_bn_relu_activation(self.env.model, calib_loader)

        self._fine_prune(
            self.env.model,
            bn_means,
            prune_pct_per_layer,
            acc_ratio=acc_ratio,
            target_label=target_label,
            patch_size=patch_size,
        )

        trig = TriggerBadNets(patch_size)
        clean_p = self.env.eval.clean(self.env.model, self.env.data.testloader)
        _, trig_p, asr_p = self.env.eval.metrics_all_to_one(
            self.env.model, self.env.data.testloader, trig, target_label
        )
        print(
            f"[After-Prune] Clean {clean_p:.2f}% | Triggered {trig_p:.2f}% | "
            f"ASR(non-target) {asr_p:.2f}%"
        )

        ft_loader = self.env.data.make_subset_loader(ft_samples)
        print(f"[FT] using {ft_samples} clean samples for fine-tuning.")
        self.env.model = self._finetune_smallset(
            self.env.model, ft_loader, ft_epochs, ft_lr, ft_weight_decay, target_label, patch_size
        )

        clean_f = self.env.eval.clean(self.env.model, self.env.data.testloader)
        _, trig_f, asr_f = self.env.eval.metrics_all_to_one(
            self.env.model, self.env.data.testloader, trig, target_label
        )
        print(
            f"[After-FT] Clean {clean_f:.2f}% | Triggered {trig_f:.2f}% | "
            f"ASR(non-target) {asr_f:.2f}%"
        )
        return dict(clean=clean_f, triggered=trig_f, asr=asr_f)

    # ==========================================================
    # 6) 对外入口：A/B 双后门版本
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
    ):
        calib_loader = self.env.data.make_calib_loader(calib_samples)
        print(f"[Calib-AB] collecting last-layer activations on {calib_samples} clean samples ...")
        bn_means = self._collect_bn_relu_activation(self.env.model, calib_loader)

        self._fine_prune(
            self.env.model,
            bn_means,
            prune_pct_per_layer,
            acc_ratio=acc_ratio,
            target_label=target_A,
            patch_size=patch_size_A,
        )

        trigA = TriggerBadNets(patch_size_A)
        trigB = TriggerSigAB(sig_amp, sig_freq)

        clean_p = self.env.eval.clean(self.env.model, self.env.data.testloader)
        _, trigA_p, asrA_p = self.env.eval.metrics_all_to_one(
            self.env.model, self.env.data.testloader, trigA, target_A
        )
        _, trigB_p, asrB_p = self.env.eval.metrics_all_to_one(
            self.env.model, self.env.data.testloader, trigB, target_B
        )
        print(
            f"[After-Prune-AB] Clean {clean_p:.2f}% | "
            f"A: Triggered {trigA_p:.2f}% ASR_A {asrA_p:.2f}% | "
            f"B: Triggered {trigB_p:.2f}% ASR_B {asrB_p:.2f}%"
        )

        ft_loader = self.env.data.make_subset_loader(ft_samples)
        print(f"[FT-AB] using {ft_samples} clean samples for fine-tuning.")
        self.env.model = self._finetune_smallset_ab(
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
        )

        clean_f = self.env.eval.clean(self.env.model, self.env.data.testloader)
        _, trigA_f, asrA_f = self.env.eval.metrics_all_to_one(
            self.env.model, self.env.data.testloader, trigA, target_A
        )
        _, trigB_f, asrB_f = self.env.eval.metrics_all_to_one(
            self.env.model, self.env.data.testloader, trigB, target_B
        )
        print(
            f"[After-FT-AB] Clean {clean_f:.2f}% | "
            f"A: Triggered {trigA_f:.2f}% ASR_A {asrA_f:.2f}% | "
            f"B: Triggered {trigB_f:.2f}% ASR_B {asrB_f:.2f}%"
        )
        # ========= 保存最终模型（剪枝+微调后）=========
        save_path = os.path.join(self.save_dir, "fp_defended_ab_model.pth")

        # 去掉 pruning reparam（避免 weight_orig/weight_mask 进入 state_dict）
        net = unwrap(self.env.model)
        for m in net.modules():
            if hasattr(m, "weight_orig"):
                try:
                    prune.remove(m, "weight")
                except Exception:
                    pass

        torch.save(net.state_dict(), save_path)
        print(f"[Save] model state_dict saved to: {save_path}")
        # ===========================================


        return dict(
            clean=clean_f,
            A_triggered=trigA_f,
            ASR_A=asrA_f,
            B_triggered=trigB_f,
            ASR_B=asrB_f,
        )


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

    fp = FinePruningDefense(env, save_dir="fp_defense_resnet18_cifar10_ab")
    fp.run_ab(
        calib_samples=1000,
        prune_pct_per_layer=0.18,  # 现在设置为 0.0 会严格跳过剪枝
        ft_samples=1000,
        ft_epochs=5,
        ft_lr=2e-3,
        ft_weight_decay=1e-4,
        target_A=0,
        target_B=1,
        patch_size_A=3,
        sig_amp=0.1,
        sig_freq=6.0,
        acc_ratio=0.1,
    )
