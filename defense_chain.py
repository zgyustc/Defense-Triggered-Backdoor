# Adapted from BB/multi_defense_chain_runner.py; see docs/changes.md.
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import os
import random
import shutil
import tempfile
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parent


DEFAULT_ALGORITHM_POOL = ["fp", "ftsam", "anp"]


CELL_MAP = {
    "fp": {"notebook": "repeat", "core": 4, "algo": 6},
    "anp": {"notebook": "repeat", "core": 10, "algo": 12},
    "ftsam": {"notebook": "orig", "core": 17, "algo": 19},
}


@dataclass
class RunnerConfig:
    CKPT_PATH: str = "./runs/cifar10_resnet18/dtb.pth"
    DATA_ROOT: str = "./data"
    VARIANT: str = "resnet18"
    BATCH_SIZE: int = 64
    NUM_WORKERS: int = 8
    SEED: int = 0
    NUM_TRIALS: int = 1
    N_ALGORITHMS: int = 3
    ALGORITHM_POOL: List[str] = field(default_factory=lambda: DEFAULT_ALGORITHM_POOL.copy())
    SAVE_DIR: str = "./multi_defense_runs"
    DEVICE: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    AMP: bool = True
    CHAIN_MODE: bool = True
    NO_SAVE: bool = False
    CLEAN_DROP_FROM_ORIGINAL_MAX: float = 4.0
    VERBOSE_JSON: bool = False
    SELECTION_MODE: str = "sequential"

    TARGET_A: int = 0
    TARGET_B: int = 1
    PATCH_SIZE_A: int = 3
    SIG_AMP: float = 0.1
    SIG_FREQ: float = 6.0


@dataclass
class DefenseResult:
    algorithm_name: str
    selected_model_path: str
    metrics_before: Dict[str, float]
    metrics_after: Dict[str, float]
    run_dir: str
    extra: Dict[str, Any]


def unwrap(m: nn.Module) -> nn.Module:
    return m.module if hasattr(m, "module") else m


def _extract_state_dict(obj: Any) -> Dict[str, torch.Tensor]:
    if isinstance(obj, dict):
        for key in ("state_dict", "model_state", "model", "net", "weights"):
            if key in obj and isinstance(obj[key], dict):
                obj = obj[key]
                break
    if not isinstance(obj, dict):
        raise RuntimeError("cannot extract state_dict from checkpoint object")
    if any(k.startswith("module.") for k in obj.keys()):
        obj = {k.replace("module.", "", 1): v for k, v in obj.items()}
    return obj


def _load_state_dict_from_path(path: Path) -> Dict[str, torch.Tensor]:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    return _extract_state_dict(obj)


def _build_namespace(algo_name: str, data_root: str, ckpt_path: str) -> Dict[str, Any]:
    import runpy
    subdir = "chain_sources" if CELL_MAP[algo_name]["notebook"] == "repeat" else "defenses"
    directory = REPO_ROOT / "dtb" / subdir
    ns = runpy.run_path(str(directory / f"{algo_name}_core.py"), run_name="_dtb_chain_core")
    ns.update(data_path=data_root, ckpt_path=ckpt_path)
    return runpy.run_path(str(directory / f"{algo_name}.py"), init_globals=ns, run_name="_dtb_chain_defense")


def _build_env(ns: Dict[str, Any], cfg: RunnerConfig, ckpt_path: str):
    core_cfg = dict(
        DATA_ROOT=cfg.DATA_ROOT,
        CKPT_PATH=ckpt_path,
        VARIANT=cfg.VARIANT,
        BATCH_SIZE=cfg.BATCH_SIZE,
        NUM_WORKERS=cfg.NUM_WORKERS,
        SEED=cfg.SEED,
        AMP=cfg.AMP,
    )
    env = ns["ModelEnv"](core_cfg)
    return env


def _get_env_mean_std(ns: Dict[str, Any], env, cfg: RunnerConfig):
    if "get_env_mean_std" in ns:
        try:
            mean, std = ns["get_env_mean_std"](env, dataset=getattr(env.data, "dataset", "cifar10"))
            if mean is not None and std is not None:
                return tuple(float(x) for x in mean), tuple(float(x) for x in std)
        except Exception:
            pass

    for mean_key, std_key in (("mean", "std"), ("MEAN", "STD")):
        if hasattr(env.data, mean_key) and hasattr(env.data, std_key):
            mean = getattr(env.data, mean_key)
            std = getattr(env.data, std_key)
            try:
                return tuple(float(x) for x in mean), tuple(float(x) for x in std)
            except Exception:
                pass

    if "MEAN_T" in ns and "STD_T" in ns:
        try:
            mean_t = ns["MEAN_T"].view(-1).detach().cpu().tolist()
            std_t = ns["STD_T"].view(-1).detach().cpu().tolist()
            return tuple(float(x) for x in mean_t), tuple(float(x) for x in std_t)
        except Exception:
            pass

    return None, None


def _build_sig_trigger(ns: Dict[str, Any], env, cfg: RunnerConfig):
    trigger_cls = ns["TriggerSigAB"]
    amp = float(cfg.SIG_AMP)
    freq = float(cfg.SIG_FREQ)
    mean, std = _get_env_mean_std(ns, env, cfg)

    ctor_attempts = [
        ((), {"amp": amp, "freq": freq, "mean": mean, "std": std}),
        ((amp, freq), {"mean": mean, "std": std}),
        ((), {"sig_amp": amp, "sig_freq": freq, "mean": mean, "std": std}),
        ((amp,), {"freq": freq, "mean": mean, "std": std}),
        ((amp, freq), {}),
        ((), {"amp": amp, "freq": freq}),
        ((), {"sig_amp": amp, "sig_freq": freq}),
    ]

    for args, kwargs in ctor_attempts:
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        try:
            return trigger_cls(*args, **kwargs)
        except TypeError:
            continue

    return trigger_cls(amp, freq)


def _eval_ab(ns: Dict[str, Any], env, model, cfg: RunnerConfig) -> Dict[str, float]:
    trigA = ns["TriggerBadNets"](int(cfg.PATCH_SIZE_A))
    trigB = _build_sig_trigger(ns, env, cfg)
    clean = float(env.eval.clean(model, env.data.testloader))
    _, a_trig, asr_a = env.eval.metrics_all_to_one(model, env.data.testloader, trigA, int(cfg.TARGET_A))
    _, b_trig, asr_b = env.eval.metrics_all_to_one(model, env.data.testloader, trigB, int(cfg.TARGET_B))
    return {
        "clean": float(clean),
        "A_trig": float(a_trig),
        "ASR_A": float(asr_a),
        "B_trig": float(b_trig),
        "ASR_B": float(asr_b),
    }


def _save_env_model(env, out_path: Path) -> str:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(unwrap(env.model).state_dict(), out_path)
    return str(out_path)


def _load_into_env(env, ckpt_path: Path):
    sd = _load_state_dict_from_path(ckpt_path)
    unwrap(env.model).load_state_dict(sd, strict=True)
    return env.model


def _build_plain_model(ns: Dict[str, Any], cfg: RunnerConfig) -> nn.Module:
    if "build_cifar_resnet" not in ns:
        raise RuntimeError("plain-model builder not found in notebook namespace")
    return ns["build_cifar_resnet"](cfg.VARIANT, num_classes=10).cpu()


def _validate_plain_state_dict(ns: Dict[str, Any], cfg: RunnerConfig, state_dict: Dict[str, torch.Tensor]):
    model = _build_plain_model(ns, cfg)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "selected checkpoint is not chain-compatible with the plain model: "
            f"missing={len(missing)} unexpected={len(unexpected)}"
        )
    return model








def _canonicalize_selected(
    ns: Dict[str, Any],
    cfg: RunnerConfig,
    canonical_path: Path,
    selected_source: Optional[Path] = None,
    fallback_state: Optional[Dict[str, torch.Tensor]] = None,
    state_transform: Optional[Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]] = None,
) -> str:
    if selected_source is not None and selected_source.exists():
        state_dict = _load_state_dict_from_path(selected_source)
    elif fallback_state is not None:
        state_dict = fallback_state
    else:
        raise RuntimeError("no selected checkpoint or fallback state available for chain handoff")

    if state_transform is not None:
        state_dict = state_transform(state_dict)

    model = _validate_plain_state_dict(ns, cfg, state_dict)
    canonical_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), canonical_path)
    return str(canonical_path)


def _eval_canonical(ns, env, checkpoint, cfg):
    model = env.build_model().to(env.device)
    model.load_state_dict(_load_state_dict_from_path(Path(checkpoint)), strict=True)
    return _eval_ab(ns, env, model, cfg)


def _run_fp(cfg: RunnerConfig, input_ckpt_path: str, run_dir: Path) -> DefenseResult:
    ns = _build_namespace("fp", cfg.DATA_ROOT, input_ckpt_path)
    env = _build_env(ns, cfg, input_ckpt_path)
    before = _eval_ab(ns, env, env.model, cfg)

    defense = ns["FinePruningDefense"](env, save_dir=str(run_dir), metrics_csv="fp_defense_total.csv")
    defense.run_ab(
        calib_samples=1000,
        prune_pct_per_layer=0.05,
        ft_samples=1000,
        ft_epochs=10,
        ft_lr=2e-3,
        ft_weight_decay=1e-4,
        target_A=int(cfg.TARGET_A),
        target_B=int(cfg.TARGET_B),
        patch_size_A=int(cfg.PATCH_SIZE_A),
        sig_amp=float(cfg.SIG_AMP),
        sig_freq=float(cfg.SIG_FREQ),
        acc_ratio=0.1,
        save_acc_ratio=0.05,
        round_idx=1,
    )

    src = run_dir / "fp_defended_ab_model_round01.pth"
    canonical = _canonicalize_selected(ns, cfg, run_dir / "selected_model.pth", selected_source=src)
    after = _eval_canonical(ns, env, canonical, cfg)
    return DefenseResult("fp", canonical, before, after, str(run_dir), {"input_model_path": input_ckpt_path, "selected_source": str(src), "metrics_csv": str(run_dir / "fp_defense_total.csv")})




def _run_anp(cfg: RunnerConfig, input_ckpt_path: str, run_dir: Path) -> DefenseResult:
    ns = _build_namespace("anp", cfg.DATA_ROOT, input_ckpt_path)
    env = _build_env(ns, cfg, input_ckpt_path)
    before = _eval_ab(ns, env, env.model, cfg)

    anp_cfg = ns["ANPCfg"]()
    anp_cfg.device = str(cfg.DEVICE)
    anp_cfg.save_acc_ratio = 0.05

    defense = ns["ANPDefenseAligned"](env, save_dir=str(run_dir))
    out = defense.run_ab_repeat(
        anp_cfg,
        n_rounds=1,
        target_A=int(cfg.TARGET_A),
        target_B=int(cfg.TARGET_B),
        patch_size_A=int(cfg.PATCH_SIZE_A),
        sig_amp=float(cfg.SIG_AMP),
        sig_freq=float(cfg.SIG_FREQ),
        objective="A",
        summary_csv_name="anp_defense_summary_ab.csv",
        model_name_prefix="anp_selected_round",
    )

    src = run_dir / "anp_selected_round01.pth"
    canonical = _canonicalize_selected(ns, cfg, run_dir / "selected_model.pth", selected_source=src)
    after = _eval_canonical(ns, env, canonical, cfg)
    return DefenseResult("anp", canonical, before, after, str(run_dir), {"input_model_path": input_ckpt_path, "selected_source": str(src), "summary_csv": str(out["summary_csv"])})














def _run_ftsam(cfg: RunnerConfig, input_ckpt_path: str, run_dir: Path) -> DefenseResult:
    ns = _build_namespace("ftsam", cfg.DATA_ROOT, input_ckpt_path)
    env = _build_env(ns, cfg, input_ckpt_path)
    before = _eval_ab(ns, env, env.model, cfg)

    defense = ns["FTSAMDefense"](env, save_dir=str(run_dir))
    out = defense.run_ab(
        ft_samples=1000,
        ft_epochs=10,
        ft_lr=1e-3,
        ft_weight_decay=1e-4,
        target_A=int(cfg.TARGET_A),
        target_B=int(cfg.TARGET_B),
        patch_size_A=int(cfg.PATCH_SIZE_A),
        sig_amp=float(cfg.SIG_AMP),
        sig_freq=float(cfg.SIG_FREQ),
        seed=int(cfg.SEED),
        ft_batch_size=int(cfg.BATCH_SIZE),
        ft_num_workers=int(cfg.NUM_WORKERS),
        save_name="ftsam_defended_ab_model.pth",
    )

    src = run_dir / "ftsam_defended_ab_model.pth"
    canonical = _canonicalize_selected(ns, cfg, run_dir / "selected_model.pth", selected_source=src)
    after = _eval_canonical(ns, env, canonical, cfg)
    return DefenseResult("ftsam", canonical, before, after, str(run_dir), {"input_model_path": input_ckpt_path, "selected_source": str(src), "metrics": out})






DEFENSE_REGISTRY: Dict[str, Callable[[RunnerConfig, str, Path], DefenseResult]] = {
    "fp": _run_fp,
    "ftsam": _run_ftsam,
    "anp": _run_anp,
}


def _sample_algorithms(pool: List[str], n_algorithms: int, seed: int) -> List[str]:
    if n_algorithms > len(pool):
        raise ValueError(f"n_algorithms={n_algorithms} exceeds pool size={len(pool)}")
    rng = random.Random(seed)
    return rng.sample(pool, n_algorithms)


def _write_json(path: Path, payload: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def _append_summary_rows(summary_path: Path, rows: List[Dict[str, Any]]):
    fieldnames = [
        "trial_id",
        "round_id",
        "algorithm_name",
        "input_model_path",
        "output_model_path",
        "clean_before",
        "asr_a_before",
        "asr_b_before",
        "clean_after",
        "asr_a_after",
        "asr_b_after",
        "status",
        "seed",
        "error_type",
        "error_message",
    ]
    need_header = (not summary_path.exists()) or summary_path.stat().st_size == 0
    with summary_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if need_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _fmt_metric(v: Any) -> str:
    try:
        return f"{float(v):.2f}"
    except Exception:
        return str(v)


def _round_decision_label(*, discard_due_to_clean_drop: bool, is_new_best: bool) -> str:
    if discard_due_to_clean_drop:
        return "skip(clean drop)"
    if is_new_best:
        return "keep(new best)"
    return "keep(not selected)"


def _print_trial_overview(trial_id: int, trial_seed: int, sampled_algorithms: List[str]):
    algo_text = " -> ".join(sampled_algorithms)
    print(f"\n[Trial {trial_id:03d}] seed={trial_seed}")
    print(f"algorithms: {algo_text}")


def _print_round_summary(
    *,
    round_id: int,
    algo_name: str,
    clean_before: float,
    asr_a_before: float,
    asr_b_before: float,
    clean_after: float,
    asr_a_after: float,
    asr_b_after: float,
    decision: str,
    selected_for_next_round_clean: Optional[float],
    selected_for_next_round_asr_b: Optional[float],
):
    best_clean_text = _fmt_metric(selected_for_next_round_clean) if selected_for_next_round_clean is not None else "NA"
    best_asr_b_text = _fmt_metric(selected_for_next_round_asr_b) if selected_for_next_round_asr_b is not None else "NA"
    print(
        f"R{round_id:02d} {algo_name:<6} "
        f"clean {_fmt_metric(clean_before)} -> {_fmt_metric(clean_after)} | "
        f"ASR_A {_fmt_metric(asr_a_before)} -> {_fmt_metric(asr_a_after)} | "
        f"ASR_B {_fmt_metric(asr_b_before)} -> {_fmt_metric(asr_b_after)} | "
        f"{decision} | best_clean={best_clean_text} best_ASR_B={best_asr_b_text}"
    )


def _print_round_failure(round_id: int, algo_name: str, exc_type: str, exc_message: str):
    print(f"R{round_id:02d} {algo_name:<6} failed | {exc_type}: {exc_message}")


def _print_final_summary(all_trials: List[Dict[str, Any]]):
    print("\n[Final Summary]")
    for trial in all_trials:
        print(
            f"trial={trial['trial_id']:03d} status={trial['status']} "
            f"final={trial.get('final_selected_model')} "
            f"algos={','.join(trial.get('sampled_algorithms', []))}"
        )
        records = trial.get("records", [])
        for rec in records:
            if rec.get("status") == "failed":
                print(
                    f"  R{rec['round_id']:02d} {rec['algorithm_name']}: "
                    f"failed ({rec.get('error_type')}: {rec.get('error_message')})"
                )
                continue
            before = rec.get("metrics_before", {})
            after = rec.get("metrics_after", {})
            decision = _round_decision_label(
                discard_due_to_clean_drop=bool(rec.get("discard_due_to_clean_drop", False)),
                is_new_best=bool(rec.get("is_new_best_clean", False)),
            )
            print(
                f"  R{rec['round_id']:02d} {rec['algorithm_name']}: "
                f"clean {_fmt_metric(before.get('clean'))}->{_fmt_metric(after.get('clean'))}, "
                f"ASR_A {_fmt_metric(before.get('ASR_A'))}->{_fmt_metric(after.get('ASR_A'))}, "
                f"ASR_B {_fmt_metric(before.get('ASR_B'))}->{_fmt_metric(after.get('ASR_B'))}, "
                f"{decision}"
            )


def run_multi_defense_chain(cfg: RunnerConfig) -> List[Dict[str, Any]]:
    all_trials: List[Dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="multi_defense_chain_") as tmp_dir:
        save_root = Path(tmp_dir) if cfg.NO_SAVE else Path(cfg.SAVE_DIR)
        save_root.mkdir(parents=True, exist_ok=True)

        for trial_id in range(int(cfg.NUM_TRIALS)):
            trial_seed = int(cfg.SEED) + trial_id
            trial_dir = save_root / f"trial_{trial_id:03d}"
            trial_dir.mkdir(parents=True, exist_ok=True)

            sampled_algorithms = _sample_algorithms(cfg.ALGORITHM_POOL, int(cfg.N_ALGORITHMS), trial_seed)
            if not cfg.NO_SAVE:
                _write_json(trial_dir / "sampled_algorithms.json", {"trial_id": trial_id, "seed": trial_seed, "algorithms": sampled_algorithms})

            current_ckpt = cfg.CKPT_PATH
            round_rows: List[Dict[str, Any]] = []
            chain_records: List[Dict[str, Any]] = []
            final_selected_model: Optional[str] = None
            best_defended_clean: Optional[float] = None
            best_defended_asr_b: Optional[float] = None
            original_clean_reference: Optional[float] = None
            trial_status = "success"

            _print_trial_overview(trial_id, trial_seed, sampled_algorithms)

            for round_id, algo_name in enumerate(sampled_algorithms, start=1):
                round_dir = trial_dir / f"round_{round_id:02d}_{algo_name}"
                round_dir.mkdir(parents=True, exist_ok=True)
                try:
                    input_ckpt_for_round = current_ckpt
                    result = DEFENSE_REGISTRY[algo_name](cfg, current_ckpt, round_dir)
                    clean_before = float(result.metrics_before["clean"])
                    if original_clean_reference is None:
                        original_clean_reference = clean_before
                    result_clean = float(result.metrics_after["clean"])
                    result_asr_b = float(result.metrics_after["ASR_B"])
                    min_clean_allowed = float(original_clean_reference) - float(cfg.CLEAN_DROP_FROM_ORIGINAL_MAX)
                    discard_due_to_clean_drop = (cfg.SELECTION_MODE == "legacy" and result_clean < min_clean_allowed)
                    is_new_best = (
                        cfg.SELECTION_MODE == "sequential" or ((not discard_due_to_clean_drop) and (
                            best_defended_clean is None or
                            best_defended_asr_b is None or
                            (
                                result_clean > best_defended_clean and
                                result_asr_b > best_defended_asr_b
                            )
                        ))
                    )
                    if is_new_best:
                        best_defended_clean = result_clean
                        best_defended_asr_b = result_asr_b

                    if not discard_due_to_clean_drop and is_new_best:
                        final_selected_model = result.selected_model_path
                        current_ckpt = result.selected_model_path
                    else:
                        final_selected_model = input_ckpt_for_round
                        current_ckpt = input_ckpt_for_round
                    chain_records.append(
                        {
                            "round_id": round_id,
                            "algorithm_name": algo_name,
                            "input_model_path": result.extra.get("input_model_path", input_ckpt_for_round),
                            "output_model_path": result.selected_model_path,
                            "selected_for_next_round": current_ckpt,
                            "selected_for_next_round_clean": best_defended_clean,
                            "selected_for_next_round_asr_b": best_defended_asr_b,
                            "is_new_best_clean": is_new_best,
                            "discard_due_to_clean_drop": discard_due_to_clean_drop,
                            "original_clean_reference": float(original_clean_reference),
                            "max_clean_drop_from_original": float(cfg.CLEAN_DROP_FROM_ORIGINAL_MAX),
                            "min_clean_allowed_for_keep": float(min_clean_allowed),
                            "metrics_before": result.metrics_before,
                            "metrics_after": result.metrics_after,
                            "run_dir": result.run_dir,
                            "extra": result.extra,
                        }
                    )
                    row = {
                        "trial_id": trial_id,
                        "round_id": round_id,
                        "algorithm_name": algo_name,
                        "input_model_path": result.extra.get("input_model_path", ""),
                        "output_model_path": result.selected_model_path,
                        "clean_before": result.metrics_before["clean"],
                        "asr_a_before": result.metrics_before["ASR_A"],
                        "asr_b_before": result.metrics_before["ASR_B"],
                        "clean_after": result.metrics_after["clean"],
                        "asr_a_after": result.metrics_after["ASR_A"],
                        "asr_b_after": result.metrics_after["ASR_B"],
                        "status": "success",
                        "seed": trial_seed,
                        "error_type": "",
                        "error_message": "",
                    }
                    round_rows.append(row)
                    decision = _round_decision_label(
                        discard_due_to_clean_drop=discard_due_to_clean_drop,
                        is_new_best=is_new_best,
                    )
                    _print_round_summary(
                        round_id=round_id,
                        algo_name=algo_name,
                        clean_before=float(row["clean_before"]),
                        asr_a_before=float(row["asr_a_before"]),
                        asr_b_before=float(row["asr_b_before"]),
                        clean_after=float(row["clean_after"]),
                        asr_a_after=float(row["asr_a_after"]),
                        asr_b_after=float(row["asr_b_after"]),
                        decision=decision,
                        selected_for_next_round_clean=best_defended_clean,
                        selected_for_next_round_asr_b=best_defended_asr_b,
                    )
                except Exception as exc:
                    trial_status = "failed"
                    tb = traceback.format_exc()
                    print(tb)
                    round_rows.append(
                        {
                            "trial_id": trial_id,
                            "round_id": round_id,
                            "algorithm_name": algo_name,
                            "input_model_path": current_ckpt,
                            "output_model_path": "",
                            "clean_before": "",
                            "asr_a_before": "",
                            "asr_b_before": "",
                            "clean_after": "",
                            "asr_a_after": "",
                            "asr_b_after": "",
                            "status": "failed",
                            "seed": trial_seed,
                            "error_type": type(exc).__name__,
                            "error_message": str(exc),
                        }
                    )
                    chain_records.append(
                        {
                            "round_id": round_id,
                            "algorithm_name": algo_name,
                            "input_model_path": current_ckpt,
                            "status": "failed",
                            "error_type": type(exc).__name__,
                            "error_message": str(exc),
                            "traceback": tb,
                        }
                    )
                    _print_round_failure(round_id, algo_name, type(exc).__name__, str(exc))
                    break

            if not cfg.NO_SAVE:
                _append_summary_rows(trial_dir / "chain_summary.csv", round_rows)

                if final_selected_model is not None and Path(final_selected_model).exists():
                    shutil.copy2(final_selected_model, trial_dir / "final_selected_model.pth")

            trial_meta = {
                "trial_id": trial_id,
                "seed": trial_seed,
                "status": trial_status,
                "start_ckpt_path": cfg.CKPT_PATH,
                "final_selected_model": (
                    str(trial_dir / "final_selected_model.pth")
                    if (not cfg.NO_SAVE and (trial_dir / "final_selected_model.pth").exists())
                    else final_selected_model
                ),
                "sampled_algorithms": sampled_algorithms,
                "records": chain_records,
                "config": asdict(cfg),
            }
            if not cfg.NO_SAVE:
                _write_json(trial_dir / "chain_meta.json", trial_meta)
            all_trials.append(trial_meta)

        if not cfg.NO_SAVE:
            _write_json(save_root / "all_trials_summary.json", all_trials)

    return all_trials


def _parse_args() -> RunnerConfig:
    parser = argparse.ArgumentParser(description="Random multi-defense chain runner")
    parser.add_argument("--ckpt-path", default=RunnerConfig.CKPT_PATH)
    parser.add_argument("--data-root", default=RunnerConfig.DATA_ROOT)
    parser.add_argument("--variant", default=RunnerConfig.VARIANT)
    parser.add_argument("--batch-size", type=int, default=RunnerConfig.BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=RunnerConfig.NUM_WORKERS)
    parser.add_argument("--seed", type=int, default=RunnerConfig.SEED)
    parser.add_argument("--num-trials", type=int, default=RunnerConfig.NUM_TRIALS)
    parser.add_argument("--n-algorithms", type=int, default=RunnerConfig.N_ALGORITHMS)
    parser.add_argument("--algorithm-pool", default=",".join(DEFAULT_ALGORITHM_POOL))
    parser.add_argument("--save-dir", default=RunnerConfig.SAVE_DIR)
    parser.add_argument("--device", default=RunnerConfig.DEVICE)
    parser.add_argument("--target-a", type=int, default=RunnerConfig.TARGET_A)
    parser.add_argument("--target-b", type=int, default=RunnerConfig.TARGET_B)
    parser.add_argument("--patch-size-a", type=int, default=RunnerConfig.PATCH_SIZE_A)
    parser.add_argument("--sig-amp", type=float, default=RunnerConfig.SIG_AMP)
    parser.add_argument("--sig-freq", type=float, default=RunnerConfig.SIG_FREQ)
    parser.add_argument("--clean-drop-from-original-max", type=float, default=RunnerConfig.CLEAN_DROP_FROM_ORIGINAL_MAX)
    parser.add_argument("--no-save", action="store_true", default=RunnerConfig.NO_SAVE)
    parser.add_argument("--verbose-json", action="store_true", default=RunnerConfig.VERBOSE_JSON)
    parser.add_argument("--selection-mode", choices=("sequential", "legacy"), default="sequential",
                        help="sequential passes every round forward; legacy retains the original ACC/hidden-ASR selection")
    parser.add_argument("--list-algorithms", action="store_true")
    args = parser.parse_args()

    if args.list_algorithms:
        print(",".join(DEFAULT_ALGORITHM_POOL))
        raise SystemExit(0)

    return RunnerConfig(
        CKPT_PATH=args.ckpt_path,
        DATA_ROOT=args.data_root,
        VARIANT=args.variant,
        BATCH_SIZE=args.batch_size,
        NUM_WORKERS=args.num_workers,
        SEED=args.seed,
        NUM_TRIALS=args.num_trials,
        N_ALGORITHMS=args.n_algorithms,
        ALGORITHM_POOL=[x.strip() for x in args.algorithm_pool.split(",") if x.strip()],
        SAVE_DIR=args.save_dir,
        DEVICE=args.device,
        TARGET_A=args.target_a,
        TARGET_B=args.target_b,
        PATCH_SIZE_A=args.patch_size_a,
        SIG_AMP=args.sig_amp,
        SIG_FREQ=args.sig_freq,
        CLEAN_DROP_FROM_ORIGINAL_MAX=args.clean_drop_from_original_max,
        NO_SAVE=args.no_save,
        VERBOSE_JSON=args.verbose_json,
        SELECTION_MODE=args.selection_mode,
    )


def main():
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    if len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")) != 1:
        raise ValueError("Expose one GPU through CUDA_VISIBLE_DEVICES")
    cfg = _parse_args()
    missing = [name for name in cfg.ALGORITHM_POOL if name not in DEFENSE_REGISTRY]
    if missing:
        raise ValueError(f"unknown algorithms in pool: {missing}")
    if not Path(cfg.CKPT_PATH).exists():
        raise FileNotFoundError(f"initial ckpt not found: {cfg.CKPT_PATH}")
    cfg.CKPT_PATH = str(Path(cfg.CKPT_PATH).resolve())
    cfg.DATA_ROOT = str(Path(cfg.DATA_ROOT).resolve())
    all_trials = run_multi_defense_chain(cfg)
    _print_final_summary(all_trials)
    if cfg.VERBOSE_JSON:
        print(json.dumps(all_trials, ensure_ascii=False, indent=2))
    if any(trial["status"] != "success" for trial in all_trials):
        raise SystemExit("At least one chain failed; inspect chain_meta.json")


if __name__ == "__main__":
    main()
