"""
Unified Experiment Tracking module supporting Weights & Biases (W&B),
local MLflow, and local JSONL logging with zero third-party lock-in.
"""

from typing import Dict, Any, Optional, List
import json
import os
import sys
import time


class UnifiedTracker:
    """
    Unified experiment tracker for W&B and MLflow.
    Handles per-epoch metrics logging, config tracking, and run summaries.
    """

    def __init__(
        self,
        backend: str = "both",  # "both", "wandb", "mlflow", "none"
        project_name: str = "cifar10-hyperparameter-sweep",
        run_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        log_dir: str = "./runs",
    ) -> None:
        self.backend = backend.lower()
        self.project_name = project_name
        self.run_name = run_name or f"run_{int(time.time())}"
        self.config = config or {}
        self.tags = tags or []
        self.log_dir = log_dir

        self.wandb_run = None
        self.mlflow_run = None

        os.makedirs(self.log_dir, exist_ok=True)
        self.local_json_path = os.path.join(self.log_dir, f"{self.run_name}.json")
        self.run_records = {
            "project": self.project_name,
            "run_name": self.run_name,
            "config": self.config,
            "tags": self.tags,
            "history": [],
            "summary": {},
            "start_time": time.time(),
        }

        self._init_trackers()

    def _init_trackers(self) -> None:
        """Initialize selected backends."""
        # 1. Weights & Biases
        if self.backend in ("both", "wandb"):
            try:
                import wandb

                # If no API key is detected and not in terminal login, default to offline
                if not os.environ.get("WANDB_API_KEY") and not os.path.exists(
                    os.path.expanduser("~/.netrc")
                ):
                    os.environ.setdefault("WANDB_MODE", "offline")

                self.wandb_run = wandb.init(
                    project=self.project_name,
                    name=self.run_name,
                    config=self.config,
                    tags=self.tags,
                    reinit=True,
                )
                print(f"[Tracker] W&B initialized (mode={os.environ.get('WANDB_MODE', 'online')})")
            except Exception as e:
                print(f"[Tracker Warning] Failed to initialize W&B: {e}", file=sys.stderr)
                self.wandb_run = None

        # 2. MLflow
        if self.backend in ("both", "mlflow"):
            try:
                import mlflow

                # Store runs locally in ./mlruns
                mlflow_dir = os.path.abspath("./mlruns")
                mlflow.set_tracking_uri(f"file:///{mlflow_dir.replace(os.sep, '/')}")
                mlflow.set_experiment(self.project_name)

                self.mlflow_run = mlflow.start_run(run_name=self.run_name)
                # Log params (flatten nested if needed)
                for k, v in self.config.items():
                    mlflow.log_param(k, v)
                if self.tags:
                    mlflow.set_tags({t: "true" for t in self.tags})
                print(f"[Tracker] MLflow initialized (tracking_uri={mlflow.get_tracking_uri()})")
            except Exception as e:
                print(f"[Tracker Warning] Failed to initialize MLflow: {e}", file=sys.stderr)
                self.mlflow_run = None

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log per-epoch or per-step metrics across active backends."""
        # Record locally
        record = dict(metrics)
        if step is not None:
            record["epoch"] = step
        self.run_records["history"].append(record)

        # W&B
        if self.wandb_run is not None:
            try:
                import wandb
                wandb.log(metrics, step=step)
            except Exception as e:
                print(f"[Tracker Warning] W&B log error: {e}", file=sys.stderr)

        # MLflow
        if self.mlflow_run is not None:
            try:
                import mlflow
                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        mlflow.log_metric(k, float(v), step=step)
            except Exception as e:
                print(f"[Tracker Warning] MLflow log error: {e}", file=sys.stderr)

    def log_summary(self, summary_metrics: Dict[str, Any]) -> None:
        """Log final run summary metrics (e.g. test_acc, best_val_acc)."""
        self.run_records["summary"].update(summary_metrics)

        if self.wandb_run is not None:
            try:
                for k, v in summary_metrics.items():
                    self.wandb_run.summary[k] = v
            except Exception as e:
                print(f"[Tracker Warning] W&B summary error: {e}", file=sys.stderr)

        if self.mlflow_run is not None:
            try:
                import mlflow
                for k, v in summary_metrics.items():
                    if isinstance(v, (int, float)):
                        mlflow.log_metric(f"final_{k}", float(v))
            except Exception as e:
                print(f"[Tracker Warning] MLflow summary error: {e}", file=sys.stderr)

    def log_artifact(self, artifact_path: str) -> None:
        """Log an artifact file (e.g. model checkpoint)."""
        if self.wandb_run is not None:
            try:
                import wandb
                artifact = wandb.Artifact(name=f"{self.run_name}_artifact", type="model")
                artifact.add_file(artifact_path)
                self.wandb_run.log_artifact(artifact)
            except Exception as e:
                print(f"[Tracker Warning] W&B artifact error: {e}", file=sys.stderr)

        if self.mlflow_run is not None:
            try:
                import mlflow
                mlflow.log_artifact(artifact_path)
            except Exception as e:
                print(f"[Tracker Warning] MLflow artifact error: {e}", file=sys.stderr)

    def finish(self) -> None:
        """Finalize run across all trackers and persist local audit file."""
        self.run_records["end_time"] = time.time()
        self.run_records["duration_seconds"] = (
            self.run_records["end_time"] - self.run_records["start_time"]
        )

        with open(self.local_json_path, "w", encoding="utf-8") as f:
            json.dump(self.run_records, f, indent=2)

        master_log = os.path.join(self.log_dir, "runs_summary.jsonl")
        summary_line = {
            "run_name": self.run_name,
            "config": self.config,
            "summary": self.run_records["summary"],
            "duration": self.run_records["duration_seconds"],
            "timestamp": self.run_records["start_time"],
        }
        with open(master_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(summary_line) + "\n")

        if self.wandb_run is not None:
            try:
                import wandb
                wandb.finish()
            except Exception:
                pass
            self.wandb_run = None

        if self.mlflow_run is not None:
            try:
                import mlflow
                mlflow.end_run()
            except Exception:
                pass
            self.mlflow_run = None

        print(f"[Tracker] Finished run '{self.run_name}'. Local audit saved to {self.local_json_path}")
