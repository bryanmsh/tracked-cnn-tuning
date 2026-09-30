"""
Sweep runner supporting both Weights & Biases native sweeps and
standalone local multi-trial sweeps with automatic logging to W&B and MLflow.
"""

import argparse
import copy
import itertools
import os
import random
import yaml
from typing import Dict, Any, List

from train import run_training


def load_yaml_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def sample_parameters(parameters_def: Dict[str, Any]) -> Dict[str, Any]:
    """Sample a single configuration from a sweep parameter dictionary."""
    config: Dict[str, Any] = {}
    for param_name, param_spec in parameters_def.items():
        if "value" in param_spec:
            config[param_name] = param_spec["value"]
        elif "values" in param_spec:
            config[param_name] = random.choice(param_spec["values"])
        elif "min" in param_spec and "max" in param_spec:
            config[param_name] = random.uniform(param_spec["min"], param_spec["max"])
        else:
            raise ValueError(f"Unknown parameter spec format for {param_name}: {param_spec}")
    return config


def run_local_sweep(sweep_config: Dict[str, Any], count: int = 20, tracker: str = "both") -> None:
    """
    Executes a local random sweep across hyperparameter configurations.
    Logs each run to W&B / MLflow / local JSONL.
    """
    params_def = sweep_config.get("parameters", {})
    project_name = sweep_config.get("name", "cifar10-hyperparameter-sweep")

    print(f"\n==========================================")
    print(f"  Starting Local Sweep: {project_name}")
    print(f"  Total planned trials: {count}")
    print(f"  Tracking backend: {tracker}")
    print(f"==========================================\n")

    results: List[Dict[str, Any]] = []

    for trial_idx in range(1, count + 1):
        sampled_config = sample_parameters(params_def)
        sampled_config["tracker"] = tracker
        sampled_config["project_name"] = project_name
        sampled_config["run_name"] = f"sweep_trial_{trial_idx:03d}"
        sampled_config["tags"] = ["sweep", "local_sweep"]

        print(f"\n--- [Trial {trial_idx}/{count}] Running: {sampled_config['run_name']} ---")
        print(f"Config: {sampled_config}")

        try:
            summary = run_training(sampled_config)
            results.append({
                "trial": trial_idx,
                "run_name": sampled_config["run_name"],
                "best_val_acc": summary.get("best_val_acc", 0.0),
                "test_acc": summary.get("test_acc", 0.0),
                "lr": sampled_config.get("lr"),
                "batch_size": sampled_config.get("batch_size"),
                "optimizer": sampled_config.get("optimizer"),
                "dropout_rate": sampled_config.get("dropout_rate"),
                "num_blocks": sampled_config.get("num_blocks"),
                "weight_decay": sampled_config.get("weight_decay"),
            })
        except Exception as e:
            print(f"[Error in Trial {trial_idx}]: {e}")

    # Print summary table sorted by best validation accuracy
    print("\n==========================================")
    print("           SWEEP RESULTS SUMMARY          ")
    print("==========================================")
    sorted_results = sorted(results, key=lambda x: x["best_val_acc"], reverse=True)
    header = f"{'Run':<18} | {'Best Val Acc':<12} | {'Test Acc':<10} | {'LR':<8} | {'Batch':<6} | {'Opt':<6} | {'Blocks':<6}"
    print(header)
    print("-" * len(header))
    for r in sorted_results:
        print(
            f"{r['run_name']:<18} | {r['best_val_acc']:>10.2f}% | {r['test_acc']:>8.2f}% | "
            f"{r['lr']:<8} | {r['batch_size']:<6} | {r['optimizer']:<6} | {r['num_blocks']:<6}"
        )


def run_wandb_sweep(sweep_config: Dict[str, Any], count: int = 20) -> None:
    """Launches and executes a native W&B sweep using wandb.sweep and wandb.agent."""
    try:
        import wandb
    except ImportError:
        raise ImportError("wandb is required for native wandb sweeps. Install it or use --mode local.")

    sweep_id = wandb.sweep(sweep_config, project=sweep_config.get("name", "cifar10-hyperparameter-sweep"))
    print(f"[W&B Sweep] Initialized sweep with ID: {sweep_id}")

    def sweep_train():
        wandb.init()
        config = dict(wandb.config)
        config["tracker"] = "wandb"
        config["run_name"] = wandb.run.name
        run_training(config)

    wandb.agent(sweep_id, function=sweep_train, count=count)


def main():
    parser = argparse.ArgumentParser(description="CIFAR-10 Sweep Runner")
    parser.add_argument("--config", type=str, default="configs/sweep_config.yaml", help="Sweep YAML path")
    parser.add_argument("--mode", type=str, default="local", choices=["local", "wandb"], help="Sweep mode")
    parser.add_argument("--count", type=int, default=20, help="Number of sweep trials to run")
    parser.add_argument("--tracker", type=str, default="both", choices=["both", "wandb", "mlflow", "none"])
    args = parser.parse_args()

    sweep_config = load_yaml_config(args.config)

    if args.mode == "wandb":
        run_wandb_sweep(sweep_config, count=args.count)
    else:
        run_local_sweep(sweep_config, count=args.count, tracker=args.tracker)


if __name__ == "__main__":
    main()
