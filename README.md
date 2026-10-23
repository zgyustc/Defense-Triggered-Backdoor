# Defense-Triggered Backdoor (DTB)

**When Sanitization Becomes the Trigger: Defense-Triggered Backdoor Attacks**  
Zhiguo Yang · Ruotian Liu · Perley Xu · Wenjie Ruan  
**NeurIPS 2026** · [Read the paper](pdf/neurips_2026.pdf)

---

## Overview

**What if removing one backdoor activates another?**

**Defense-Triggered Backdoor (DTB)** investigates a counterintuitive failure mode of backdoor sanitization. DTB implants two backdoors into one model: an **active decoy backdoor** and a **dormant hidden backdoor**. Sanitization can suppress the decoy while unintentionally **activating the hidden backdoor**.

To achieve this conditional behavior, DTB trains with **differentiable surrogate defenses** that approximate the suppression of decoy-sensitive model channels. Its training objectives couple the decoy's suppression with the hidden backdoor's activation, while keeping normal classification performance and the hidden backdoor's pre-sanitization dormancy in view.

| Model state | Decoy backdoor | Hidden backdoor |
|:--|:--:|:--:|
| **Before sanitization** | Active | Dormant |
| **After sanitization** | Suppressed | Activated |

This repository reproduces the **core CIFAR-10 / ResNet-18 workflow**, including three-stage training, evaluation, and four standalone defenses: **FP, FT-SAM, ANP, and NAD**. It also provides a separate multi-round sanitization runner.

> **Scope:** The released workflow covers the primary reproduction setup, not every dataset, architecture, or defense experiment in the [paper](pdf/neurips_2026.pdf).

## 1. Reproduction Scope

The primary experiment uses **CIFAR-10**, **ResNet-18**, and a **BadNets / SIG** trigger pair:

1. **Train** a clean classifier.
2. **Implant decoy A** using a BadNets patch.
3. **Train DTB:** keep hidden backdoor B (SIG) dormant in the original model while optimizing its activation under surrogate sanitization.
4. **Evaluate:** apply a real defense and measure clean accuracy (ACC) and both backdoor attack success rates (ASRs).

| Component | Reproduction setup |
|:--|:--|
| Dataset / architecture | CIFAR-10 / ResNet-18 |
| Decoy / hidden trigger | BadNets / SIG |
| Training stages | Clean → Decoy → DTB |
| Standalone defenses | FP · FT-SAM · ANP · NAD |
| Multi-round defense pool | FP · FT-SAM · ANP |

## 2. Environment

Use **Python 3.10 or 3.11** as a starting environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run commands from the **repository root**. Expose one physical GPU (the scripts address it as `cuda:0`):

```bash
export CUDA_VISIBLE_DEVICES=0
```

## 3. Main Experiment: CIFAR-10 / ResNet-18

CIFAR-10 is downloaded through `torchvision` when commands run. Alternatively, place an existing `cifar-10-batches-py/` directory under `data/`. Data and experiment outputs are ignored by Git.

### 3.1 Three-Stage Training

```bash
# Stage 1: 200 epochs; produces clean.pth
python train.py --stage stage1 --output runs/cifar10_resnet18

# Stage 2: 20 epochs; clean.pth -> decoy.pth
python train.py --stage stage2 --output runs/cifar10_resnet18

# Stage 3: 20 epochs; decoy.pth -> dtb.pth
python train.py --stage stage3 --output runs/cifar10_resnet18 \
  --config configs/cifar10_stage3_paper.json
```

**Architecture note:** ResNet-18 uses a CIFAR stem (`3×3` first convolution, stride `1`, no max-pooling). Checkpoints must match this implementation.

### 3.2 Evaluate Before and After FP

Evaluate the original DTB checkpoint:

```bash
python evaluate.py --checkpoint runs/cifar10_resnet18/dtb.pth \
  --output runs/cifar10_resnet18/original_metrics.json
```

Apply **FP** and evaluate its sanitized checkpoint:

```bash
python defend.py --algorithm fp \
  --checkpoint runs/cifar10_resnet18/dtb.pth --output outputs/dtb/fp

python evaluate.py \
  --checkpoint outputs/dtb/fp/fp_defense_resnet18_cifar10_ab/fp_defended_ab_model.pth \
  --output outputs/dtb/fp/metrics.json
```

Or run the complete primary workflow, including FP:

```bash
bash scripts/reproduce_cifar10.sh
```

### 3.3 Run the Four Standalone Defenses

Each defense starts **independently** from the same `dtb.pth` checkpoint:

```bash
bash scripts/defend_all.sh runs/cifar10_resnet18/dtb.pth outputs/dtb
```

Run a single defense (`fp`, `ftsam`, `anp`, or `nad`):

```bash
python defend.py --algorithm anp \
  --checkpoint runs/cifar10_resnet18/dtb.pth --output outputs/dtb/anp
```

| Defense | Approach |
|:--|:--|
| **FP** | Pruning and fine-tuning |
| **FT-SAM** | Sharpness-aware fine-tuning |
| **ANP** | Adversarial neuron pruning |
| **NAD** | Attention distillation |

Defense hyperparameters are defined in `dtb/defenses/ALGORITHM.py`, either in a configuration object or the final entry-point block. The launcher supplies absolute data/checkpoint paths, loads the matching implementation, and switches its working directory to `--output`, so relative outputs remain inside that defense's directory.

`evaluate.py` supports plain CIFAR-10 / ResNet-18 checkpoints with strict state-dictionary loading. Each defense also reports its own evaluation metrics. **Standalone and multi-round experiments use separate configurations.**

## 4. Multi-Round Sanitization

The retained multi-round runner supports **FP, FT-SAM, and ANP**. NAD is available through the standalone launcher only.

```bash
python defense_chain.py \
  --ckpt-path runs/cifar10_resnet18/dtb.pth --data-root data \
  --algorithm-pool fp,anp,ftsam \
  --n-algorithms 3 --num-trials 1 --seed 0 \
  --save-dir outputs/chain --selection-mode sequential
```

## 5. Repository Layout

```text
.
├── train.py                 # Three-stage training
├── evaluate.py              # Checkpoint evaluation
├── defend.py                # Standalone defense launcher
├── defense_chain.py         # Multi-round FP / FT-SAM / ANP runner
├── dtb/
│   ├── training/            # Three-stage training implementations
│   ├── defenses/            # FP / FT-SAM / ANP / NAD and matching cores
│   ├── chain_sources/       # Multi-round FP / ANP implementations
│   └── runtime.py           # JSON configuration bridge
├── configs/                 # Main/historical settings; ImageNet-12 classes
├── scripts/                 # Workflows, dataset layout, static checks
├── records/                 # Experiment records
└── pdf/
    └── neurips_2026.pdf    # Paper
```

---

*DTB studies defense-triggered behavior under specific sanitization settings; its activation is not guaranteed for arbitrary defenses or deployment workflows.*
