# Tracked CNN Tuning: CIFAR-10 Hyperparameter Sweep

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Experiment Tracking](https://img.shields.io/badge/tracking-W%26B%20%2F%20MLflow-FFBE00.svg)](https://wandb.ai/)
[![Project](https://img.shields.io/badge/Pipeline-Project%203-brightgreen.svg)]()

> **Concentration Pipeline — Project 3**  
> Systematic, reproducible hyperparameter tuning of a modular CIFAR-10 Convolutional Neural Network with unified experiment tracking across **Weights & Biases (W&B)** and **MLflow**.

---

## 1. Overview

In basic deep learning workflows, models are often trained once in an ad-hoc script or notebook, and results are jotted down manually. In production and applied research, however, model development is an empirical science: we test dozens of learning rates, batch sizes, regularization terms, and network depths, requiring a reproducible and visual way to compare all trials.

This project introduces that discipline:
- **Systematic Sweeps:** Random and grid search across learning rate, batch size, optimizer choice, network depth, dropout, and weight decay.
- **Unified Experiment Tracking:** Automatic, real-time logging to **Weights & Biases (W&B)** and local **MLflow** dashboards with zero manual record-keeping.
- **Strict Reproducibility:** Every run logs its complete configuration, deterministic random seed, and Git commit hash so any checkpoint can be recreated from scratch.
- **Diagnostic Rigor:** Visual evaluation of loss and accuracy curves to diagnose overfitting, underfitting, and learning rate dynamics.

---

## 2. Model Architecture

The core model is a modular `CIFAR10CNN` (`src/models/cnn.py`) designed for flexible hyperparameter exploration:

```
Input (3 x 32 x 32)
   │
   ▼
[ ConvBlock 1: Conv(3->32) -> BN -> ReLU -> Conv(32->32) -> BN -> ReLU -> MaxPool(2x2) ]  --> (16 x 16)
   │
   ▼
[ ConvBlock 2: Conv(32->64) -> BN -> ReLU -> Conv(64->64) -> BN -> ReLU -> MaxPool(2x2) ]  --> (8 x 8)
   │
   ▼
[ ConvBlock 3: Conv(64->128) -> BN -> ReLU -> Conv(128->128) -> BN -> ReLU -> MaxPool(2x2) ] --> (4 x 4)
   │  (Optional ConvBlock 4: 128->256 channels)
   ▼
AdaptiveAvgPool2d((2, 2))  --> (Channels x 2 x 2)
   │
Flatten
   │
Linear(FlattenDim -> 256) -> BatchNorm1d -> ReLU -> Dropout(p)
   │
Linear(256 -> 10)  --> Class Logits (Softmax via CrossEntropyLoss)
```

### Key Architectural Hyperparameters
- **Depth (`num_blocks`):** Configurable from 2 to 4 convolutional blocks.
- **Classifier Dropout (`dropout_rate`):** Swept from `0.1` to `0.5` to regularize the dense projection.
- **Base Channels (`base_channels`):** Base channel scaling (default `32`, doubling per block).
- **Weight Initialization:** He (Kaiming) normal initialization tailored for ReLU activations.

---

## 3. Project Structure

```
tracked-cnn-tuning/
├── .gitignore               # Ignores prd.md, virtual environments, datasets, and tracking caches
├── README.md                # Project documentation and sweep guide
├── requirements.txt         # Core dependencies (torch, torchvision, wandb, mlflow, etc.)
├── train.py                 # Instrumented training script with CLI and config support
├── sweep.py                 # Multi-trial sweep engine (local random search & W&B Sweeps)
├── configs/
│   ├── default.yaml         # Default baseline training configuration
│   └── sweep_config.yaml    # Search space definition for hyperparameter sweeps
├── src/
│   ├── __init__.py
│   ├── data.py              # CIFAR-10 data pipeline (deterministic split + augmentation)
│   ├── tracking.py          # UnifiedTracker interface (W&B + MLflow + local JSONL)
│   ├── utils.py             # Reproducibility seeds, device selection, git commit provenance
│   └── models/
│       ├── __init__.py
│       └── cnn.py           # Modular CIFAR10CNN architecture
└── tests/
    ├── __init__.py
    ├── test_model.py        # Forward pass & parameter count tests
    ├── test_data.py         # Transform & loader validation
    └── test_smoke.py        # Fast pipeline smoke test with synthetic data
```

---

## 4. Setup & Installation

### Prerequisites
- Python 3.10+ (tested on Python 3.12)
- NVIDIA CUDA or AMD ROCm GPU (strongly recommended; falls back to CPU automatically)

### 1. Clone & Navigate
```bash
git clone https://github.com/bryanmsh/tracked-cnn-tuning.git
cd tracked-cnn-tuning
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. (Optional) Configure Experiment Tracking
- **Weights & Biases:** Run `wandb login` with your API key, or set `export WANDB_API_KEY="..."`. If unconfigured, W&B automatically runs in `offline` mode so runs are never interrupted.
- **MLflow:** MLflow runs entirely locally by default, logging to `./mlruns`. You can view the dashboard anytime with:
  ```bash
  mlflow ui --port 5000
  ```

---

## 5. Step 1: Smoke Test & Instrumentation Check

Before launching a full multi-trial sweep, execute a single smoke test to verify end-to-end data loading, forward/backward passes, GPU acceleration, and experiment tracking logging:

```bash
python train.py --smoke_test --tracker both
```

This runs 2 brief epochs over a few mini-batches, logs metrics to both W&B and MLflow, and saves an audit log to `./runs/`.

To train a full baseline model:
```bash
python train.py --lr 0.001 --batch_size 128 --optimizer adam --dropout_rate 0.3 --epochs 15 --tracker both
```

---

## 6. Step 2: Running Hyperparameter Sweeps

The project provides two modes for running sweeps:

### Option A: Local Multi-Trial Sweep (Recommended)
Runs an automated random search across the parameter distribution defined in `configs/sweep_config.yaml`:
```bash
python sweep.py --mode local --count 20 --tracker both
```
At the end of the sweep, a formatted summary leaderboard is printed directly in the terminal, sorted by validation accuracy.

### Option B: Native Weights & Biases Sweep
Launches a managed sweep connected directly to the W&B cloud dashboard:
```bash
python sweep.py --mode wandb --count 20
```

### Swept Hyperparameter Space
| Hyperparameter | Search Range / Values | Impact |
|---|---|---|
| **Learning Rate (`lr`)** | `[0.0003, 0.001, 0.003, 0.01]` | Step size; diagnoses convergence vs. divergence |
| **Batch Size (`batch_size`)** | `[64, 128, 256]` | Gradient variance and throughput |
| **Optimizer (`optimizer`)** | `['adam', 'sgd', 'adamw']` | Adaptive momentum vs. stochastic gradient descent |
| **Weight Decay (`weight_decay`)** | `[0.0, 1e-4, 1e-3, 1e-2]` | L2 regularization against parameter explosion |
| **Dropout Rate (`dropout_rate`)** | `[0.1, 0.3, 0.5]` | Regularization of classification head |
| **Network Depth (`num_blocks`)** | `[2, 3, 4]` | Capacity and spatial feature hierarchy |

---

## 7. Reading Training Curves

The tracking dashboards (W&B / MLflow) allow visual diagnosis of training dynamics:
1. **Overfitting:** Training loss continues dropping while validation loss rises after a specific epoch. *Mitigation:* increase `dropout_rate`, add `weight_decay`, or increase data augmentation.
2. **High Learning Rate:** Loss oscillates erratically, explodes, or fails to decrease below initial baseline. *Mitigation:* lower `lr` by a factor of 10.
3. **Underfitting:** Training loss and validation loss remain flat and close together at a high error rate. *Mitigation:* increase `num_blocks`, decrease dropout, or test `adam` over `sgd`.

---

## 8. Reproducibility & Provenance

Every run strictly logs:
- Exact random seed (`seed=42`) setting Python, NumPy, and PyTorch seeds.
- CUDA deterministic backends (`torch.backends.cudnn.deterministic = True`).
- Active Git commit SHA recorded in the configuration dict.
- Model checkpoints saved to `./checkpoints/` tagged with the run ID.

---

## 9. Running Tests

Run the test suite to verify model mechanics and data transforms:
```bash
python -m unittest discover tests
```
