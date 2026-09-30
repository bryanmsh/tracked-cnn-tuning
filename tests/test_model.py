"""
Tests for CIFAR-10 CNN architecture, parameter count, and forward passes.
"""

import unittest
import torch
from src.models.cnn import CIFAR10CNN, build_model


class TestCIFAR10CNN(unittest.TestCase):

    def test_forward_shape_various_depths(self):
        """Verify model output shape is [batch_size, 10] across different depths."""
        batch_size = 4
        x = torch.randn(batch_size, 3, 32, 32)

        for num_blocks in [1, 2, 3, 4]:
            with self.subTest(num_blocks=num_blocks):
                model = CIFAR10CNN(num_classes=10, num_blocks=num_blocks, base_channels=16)
                out = model(x)
                self.assertEqual(out.shape, (batch_size, 10))

    def test_parameter_count(self):
        """Ensure count_parameters returns valid non-zero counts."""
        model = CIFAR10CNN(num_blocks=3, base_channels=32)
        params = model.count_parameters()
        self.assertIn("total_params", params)
        self.assertIn("trainable_params", params)
        self.assertGreater(params["total_params"], 100_000)
        self.assertEqual(params["total_params"], params["trainable_params"])

    def test_build_model_factory(self):
        """Verify factory builder initializes model from config dict."""
        config = {
            "num_classes": 10,
            "num_blocks": 2,
            "base_channels": 16,
            "dropout_rate": 0.4,
            "fc_dim": 128,
        }
        model = build_model(config)
        self.assertEqual(model.num_blocks, 2)
        self.assertEqual(model.dropout_rate, 0.4)
        self.assertEqual(model.fc_dim, 128)


if __name__ == "__main__":
    unittest.main()
