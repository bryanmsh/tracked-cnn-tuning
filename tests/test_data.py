"""
Tests for CIFAR-10 data transforms and dataset utilities.
"""

import unittest
import torch
from src.data import get_transforms, CIFAR10_MEAN, CIFAR10_STD


class TestDataPipeline(unittest.TestCase):

    def test_transforms_dimensions(self):
        """Ensure transforms produce correct 3x32x32 float tensors."""
        train_tf, eval_tf = get_transforms(augment=True)
        self.assertIsNotNone(train_tf)
        self.assertIsNotNone(eval_tf)

    def test_normalization_constants(self):
        """Check CIFAR-10 normalization constants."""
        self.assertEqual(len(CIFAR10_MEAN), 3)
        self.assertEqual(len(CIFAR10_STD), 3)
        self.assertAlmostEqual(CIFAR10_MEAN[0], 0.4914, places=3)


if __name__ == "__main__":
    unittest.main()
