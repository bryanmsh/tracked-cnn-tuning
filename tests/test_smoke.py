"""
Fast smoke test for training loop, evaluation, and tracking with synthetic CIFAR tensors.
Ensures zero runtime errors before launching live sweeps.
"""

import unittest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.cnn import CIFAR10CNN
from src.tracking import UnifiedTracker
from train import train_one_epoch, evaluate


class TestSmokePipeline(unittest.TestCase):

    def setUp(self):
        # Create small synthetic dataset (16 samples, 3x32x32, 10 classes)
        self.x = torch.randn(16, 3, 32, 32)
        self.y = torch.randint(0, 10, (16,))
        dataset = TensorDataset(self.x, self.y)
        self.loader = DataLoader(dataset, batch_size=4)

        self.model = CIFAR10CNN(num_classes=10, num_blocks=2, base_channels=16)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.device = torch.device("cpu")

    def test_single_train_eval_step(self):
        """Verify train and eval loops run smoothly on synthetic data."""
        loss, acc = train_one_epoch(
            model=self.model,
            dataloader=self.loader,
            criterion=self.criterion,
            optimizer=self.optimizer,
            device=self.device,
        )
        self.assertGreater(loss, 0.0)
        self.assertGreaterEqual(acc, 0.0)
        self.assertLessEqual(acc, 100.0)

        eval_loss, eval_acc = evaluate(
            model=self.model,
            dataloader=self.loader,
            criterion=self.criterion,
            device=self.device,
        )
        self.assertGreater(eval_loss, 0.0)
        self.assertGreaterEqual(eval_acc, 0.0)
        self.assertLessEqual(eval_acc, 100.0)

    def test_tracker_offline_lifecycle(self):
        """Verify UnifiedTracker initializes and finishes without crashing."""
        tracker = UnifiedTracker(
            backend="none",
            project_name="test-project",
            run_name="smoke_test_run",
            config={"lr": 0.001, "batch_size": 4},
        )
        tracker.log_metrics({"train_loss": 1.5, "val_loss": 1.4}, step=1)
        tracker.log_summary({"test_acc": 85.0})
        tracker.finish()
        self.assertTrue(hasattr(tracker, "local_json_path"))


if __name__ == "__main__":
    unittest.main()
