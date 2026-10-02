"""Unit tests for inference and model export utilities."""

import os
import tempfile
import unittest
import torch

from infer import predict_sample, export_to_torchscript, CIFAR10_CLASSES
from src.models.cnn import CIFAR10CNN


class TestInfer(unittest.TestCase):
    def setUp(self):
        self.model = CIFAR10CNN(num_classes=10, num_blocks=2, base_channels=16)
        self.model.eval()
        self.device = torch.device("cpu")

    def test_predict_sample(self):
        dummy_image = torch.randn(3, 32, 32)
        top_k = 3
        preds = predict_sample(self.model, dummy_image, self.device, top_k=top_k)

        self.assertEqual(len(preds), top_k)
        for class_name, prob in preds:
            self.assertIn(class_name, CIFAR10_CLASSES)
            self.assertGreaterEqual(prob, 0.0)
            self.assertLessEqual(prob, 100.0)

    def test_export_to_torchscript(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            export_path = os.path.join(tmpdir, "model_traced.pt")
            saved_path = export_to_torchscript(self.model, export_path, self.device)
            self.assertTrue(os.path.exists(saved_path))

            # Verify loading and running
            loaded_model = torch.jit.load(saved_path)
            dummy_input = torch.randn(2, 3, 32, 32)
            output = loaded_model(dummy_input)
            self.assertEqual(output.shape, (2, 10))


if __name__ == "__main__":
    unittest.main()
