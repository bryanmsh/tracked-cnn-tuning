"""
Main training script for CIFAR-10 CNN with experiment tracking.
Supports CLI arguments, YAML configs, smoke tests, and unified W&B / MLflow logging.
"""

import argparse
import os
import sys
import time
from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm

from src.data import get_cifar10_dataloaders
from src.models.cnn import CIFAR10CNN
from src.tracking import UnifiedTracker
from src.utils import set_seed, get_device, get_git_commit_hash, save_checkpoint


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_batches: int = 0,
) -> Tuple[float, float]:
    """Train the model for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        if max_batches > 0 and batch_idx >= max_batches:
            break

        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = 100.0 * correct / max(total, 1)
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    max_batches: int = 0,
) -> Tuple[float, float]:
    """Evaluate model performance on validation or test set."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        if max_batches > 0 and batch_idx >= max_batches:
            break

        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    eval_loss = running_loss / max(total, 1)
    eval_acc = 100.0 * correct / max(total, 1)
    return eval_loss, eval_acc


def run_training(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Executes a complete training and evaluation run with experiment tracking.
    """
    seed = config.get("seed", 42)
    set_seed(seed)

    device = get_device()
    print(f"\n[Environment] Using device: {device} | Seed: {seed}")

    git_hash = get_git_commit_hash()
    config["git_commit"] = git_hash

    # Setup tracker
    tracker = UnifiedTracker(
        backend=config.get("tracker", "both"),
        project_name=config.get("project_name", "cifar10-hyperparameter-sweep"),
        run_name=config.get("run_name"),
        config=config,
        tags=config.get("tags", []),
    )

    batch_size = config.get("batch_size", 128)
    data_dir = config.get("data_dir", "./data")
    smoke_test = config.get("smoke_test", False)

    print(f"[Data] Preparing CIFAR-10 data loaders (batch_size={batch_size})...")
    train_loader, val_loader, test_loader = get_cifar10_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        val_split=config.get("val_split", 0.1),
        seed=seed,
        num_workers=0 if os.name == "nt" else 2,  # safer multiprocessing on Windows
    )

    # Build model
    model = CIFAR10CNN(
        num_classes=10,
        num_blocks=config.get("num_blocks", 3),
        base_channels=config.get("base_channels", 32),
        dropout_rate=config.get("dropout_rate", 0.3),
        conv_dropout=config.get("conv_dropout", 0.0),
        fc_dim=config.get("fc_dim", 256),
    ).to(device)

    param_info = model.count_parameters()
    config["total_params"] = param_info["total_params"]
    print(f"[Model] Initialized CIFAR10CNN (blocks={config.get('num_blocks', 3)}, params={param_info['total_params']:,})")

    criterion = nn.CrossEntropyLoss()

    # Optimizer selection
    opt_name = config.get("optimizer", "adam").lower()
    lr = config.get("lr", 0.001)
    weight_decay = config.get("weight_decay", 1e-4)

    if opt_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=config.get("momentum", 0.9),
            weight_decay=weight_decay,
        )
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
    else:  # default adam
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

    epochs = 2 if smoke_test else config.get("epochs", 15)
    max_batches = 5 if smoke_test else 0

    print(f"[Training] Starting training for {epochs} epochs (smoke_test={smoke_test})...")

    best_val_acc = 0.0
    best_val_epoch = 0

    start_train_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        train_loss, train_acc = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            max_batches=max_batches,
        )
        val_loss, val_acc = evaluate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            max_batches=max_batches,
        )
        epoch_duration = time.time() - epoch_start

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            best_val_epoch = epoch

        # Save checkpoint if configured
        if config.get("save_checkpoint", True):
            save_checkpoint(
                state={
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "config": config,
                },
                is_best=is_best,
                checkpoint_dir=config.get("checkpoint_dir", "./checkpoints"),
                filename=f"{tracker.run_name}_latest.pt",
                best_filename=f"{tracker.run_name}_best.pt",
            )

        metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "epoch_duration_sec": epoch_duration,
        }
        tracker.log_metrics(metrics, step=epoch)

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
            f"Time: {epoch_duration:.2f}s"
        )

    total_train_time = time.time() - start_train_time
    print(f"\n[Training Completed] Total time: {total_train_time:.2f}s | Best Val Acc: {best_val_acc:.2f}% (Epoch {best_val_epoch})")

    # Final evaluation on test set
    print("[Evaluation] Evaluating on test set...")
    test_loss, test_acc = evaluate(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device,
        max_batches=max_batches,
    )
    print(f"[Test Results] Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")

    summary = {
        "final_train_loss": train_loss,
        "final_train_acc": train_acc,
        "final_val_loss": val_loss,
        "final_val_acc": val_acc,
        "best_val_acc": best_val_acc,
        "best_val_epoch": best_val_epoch,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "total_train_time_sec": total_train_time,
    }
    tracker.log_summary(summary)
    tracker.finish()

    return summary


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="CIFAR-10 Tracked Training")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd", "adamw"], help="Optimizer")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay (L2 penalty)")
    parser.add_argument("--dropout_rate", type=float, default=0.3, help="Classifier dropout rate")
    parser.add_argument("--conv_dropout", type=float, default=0.0, help="Conv block dropout rate")
    parser.add_argument("--num_blocks", type=int, default=3, help="Number of conv blocks (depth)")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--seed", type=int, default=42, help="Fixed random seed")
    parser.add_argument("--tracker", type=str, default="both", choices=["both", "wandb", "mlflow", "none"], help="Tracking backend")
    parser.add_argument("--project_name", type=str, default="cifar10-hyperparameter-sweep", help="Tracking project name")
    parser.add_argument("--run_name", type=str, default=None, help="Optional run name")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory for CIFAR-10 data")
    parser.add_argument("--smoke_test", action="store_true", help="Run 2 short epochs for smoke testing")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = vars(args)
    run_training(config)
