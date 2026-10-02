"""Inference, evaluation, and model export utility for CIFAR-10 CNN."""

import argparse
import os
import sys
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data import get_cifar10_dataloaders
from src.models.cnn import CIFAR10CNN
from src.utils import get_device

CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def load_model_from_checkpoint(
    checkpoint_path: str, device: torch.device
) -> Tuple[CIFAR10CNN, Dict[str, Any]]:
    """Instantiate model using checkpoint config and restore weights."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})

    model = CIFAR10CNN(
        num_classes=10,
        num_blocks=config.get("num_blocks", 3),
        base_channels=config.get("base_channels", 32),
        dropout_rate=config.get("dropout_rate", 0.3),
        conv_dropout=config.get("conv_dropout", 0.0),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print(
        f"[Checkpoint] Loaded weights from '{checkpoint_path}' "
        f"(Blocks: {config.get('num_blocks', 3)}, Val Acc: {checkpoint.get('val_acc', 0.0):.2f}%)",
        flush=True,
    )
    return model, checkpoint


def evaluate_with_confusion_matrix(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    criterion: nn.Module = nn.CrossEntropyLoss(),
) -> Dict[str, Any]:
    """Run full evaluation computing per-class accuracy, precision, and confusion matrix."""
    model.eval()
    num_classes = len(CIFAR10_CLASSES)
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    total_loss = 0.0
    total_samples = 0
    correct = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)

            total_samples += targets.size(0)
            correct += (preds == targets).sum().item()

            for t, p in zip(targets.view(-1), preds.view(-1)):
                confusion_matrix[t.long().item(), p.long().item()] += 1

    overall_acc = (correct / total_samples) * 100.0 if total_samples > 0 else 0.0
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0

    # Per-class metrics
    per_class_acc = {}
    per_class_precision = {}
    per_class_recall = {}

    for i, class_name in enumerate(CIFAR10_CLASSES):
        class_total = confusion_matrix[i, :].sum()
        class_correct = confusion_matrix[i, i]
        pred_total = confusion_matrix[:, i].sum()

        recall = (class_correct / class_total * 100.0) if class_total > 0 else 0.0
        precision = (class_correct / pred_total * 100.0) if pred_total > 0 else 0.0

        per_class_acc[class_name] = recall
        per_class_recall[class_name] = recall
        per_class_precision[class_name] = precision

    return {
        "overall_acc": overall_acc,
        "avg_loss": avg_loss,
        "total_samples": total_samples,
        "confusion_matrix": confusion_matrix,
        "per_class_acc": per_class_acc,
        "per_class_precision": per_class_precision,
        "per_class_recall": per_class_recall,
    }


def plot_confusion_matrix(
    cm: np.ndarray,
    output_path: str = "assets/confusion_matrix.png",
    class_names: List[str] = CIFAR10_CLASSES,
) -> None:
    """Render and save a normalized confusion matrix heatmap."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(9, 8), dpi=300)
    fig.patch.set_facecolor("#ffffff")

    im = ax.imshow(cm_normalized, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title="CIFAR-10 Normalized Confusion Matrix",
        ylabel="True Class",
        xlabel="Predicted Class",
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm_normalized.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm_normalized[i, j] * 100.0
            count = cm[i, j]
            ax.text(
                j,
                i,
                f"{val:.1f}%\n({count})",
                ha="center",
                va="center",
                fontsize=7.5,
                color="white" if cm_normalized[i, j] > thresh else "black",
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Visualization] Saved confusion matrix to {output_path}", flush=True)


def export_to_torchscript(
    model: nn.Module,
    output_path: str = "checkpoints/best_model_traced.pt",
    device: torch.device = torch.device("cpu"),
) -> str:
    """Export model to standalone TorchScript format."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model.eval()
    model.to(device)

    dummy_input = torch.randn(1, 3, 32, 32, device=device)
    traced_model = torch.jit.trace(model, dummy_input)
    traced_model.save(output_path)
    print(f"[Export] Model successfully exported to TorchScript: {output_path}", flush=True)
    return output_path


def export_to_onnx(
    model: nn.Module,
    output_path: str = "checkpoints/best_model.onnx",
    device: torch.device = torch.device("cpu"),
) -> str:
    """Export model to ONNX format with dynamic batching."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model.eval()
    model.to(device)

    dummy_input = torch.randn(1, 3, 32, 32, device=device)
    try:
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
        )
        print(f"[Export] Model successfully exported to ONNX: {output_path}", flush=True)
        return output_path
    except (ModuleNotFoundError, ImportError, Exception) as e:
        print(f"[Export Notice] ONNX export requires 'onnx' and 'onnxscript': {e}", flush=True)
        print("[Export Notice] Automatically exporting to portable TorchScript instead...", flush=True)
        ts_path = output_path.replace(".onnx", "_traced.pt")
        return export_to_torchscript(model, ts_path, device=device)


def predict_sample(
    model: nn.Module,
    image_tensor: torch.Tensor,
    device: torch.device,
    top_k: int = 3,
) -> List[Tuple[str, float]]:
    """Predict class probabilities for a single (3, 32, 32) tensor."""
    model.eval()
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.softmax(logits, dim=1)[0]
        top_probs, top_indices = torch.topk(probs, k=min(top_k, 10))

    results = []
    for prob, idx in zip(top_probs, top_indices):
        results.append((CIFAR10_CLASSES[idx.item()], prob.item() * 100.0))
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CIFAR-10 Inference & Evaluation Tool")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/sweep_trial_014_best.pt",
        help="Path to checkpoint file",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="Directory where CIFAR-10 is stored",
    )
    parser.add_argument(
        "--eval_test",
        action="store_true",
        help="Evaluate model across full test set",
    )
    parser.add_argument(
        "--plot_cm",
        action="store_true",
        help="Generate and save confusion matrix",
    )
    parser.add_argument(
        "--export_onnx",
        action="store_true",
        help="Export checkpoint to ONNX format",
    )
    parser.add_argument(
        "--export_torchscript",
        action="store_true",
        help="Export checkpoint to standalone TorchScript format",
    )
    parser.add_argument(
        "--sample_idx",
        type=int,
        default=None,
        help="Predict test set image at specified index",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device("auto")

    model, ckpt = load_model_from_checkpoint(args.checkpoint, device=device)

    # Export models if requested
    if args.export_onnx:
        onnx_path = args.checkpoint.replace(".pt", ".onnx")
        export_to_onnx(model, output_path=onnx_path, device=device)

    if args.export_torchscript:
        ts_path = args.checkpoint.replace(".pt", "_traced.pt")
        export_to_torchscript(model, output_path=ts_path, device=device)

    # Evaluation mode
    if args.eval_test or args.plot_cm or args.sample_idx is not None:
        _, _, test_loader = get_cifar10_dataloaders(
            data_dir=args.data_dir,
            batch_size=256,
            num_workers=0,
        )

        if args.eval_test or args.plot_cm:
            print("\n[Evaluation] Running comprehensive test set evaluation...", flush=True)
            results = evaluate_with_confusion_matrix(model, test_loader, device=device)

            print("\n==========================================", flush=True)
            print("        EVALUATION RESULTS SUMMARY        ", flush=True)
            print("==========================================", flush=True)
            print(f"Overall Test Accuracy: {results['overall_acc']:.2f}%", flush=True)
            print(f"Average Test Loss:     {results['avg_loss']:.4f}", flush=True)
            print(f"Total Test Samples:    {results['total_samples']}", flush=True)

            print("\n--- Per-Class Performance ---", flush=True)
            print(f"{'Class':<12} | {'Accuracy/Recall':<16} | {'Precision':<12}", flush=True)
            print("-" * 44, flush=True)
            for c in CIFAR10_CLASSES:
                rec = results["per_class_recall"][c]
                prec = results["per_class_precision"][c]
                print(f"{c:<12} | {rec:>14.2f}% | {prec:>10.2f}%", flush=True)

            if args.plot_cm:
                plot_confusion_matrix(results["confusion_matrix"], "assets/confusion_matrix.png")

        if args.sample_idx is not None:
            dataset = test_loader.dataset
            idx = max(0, min(args.sample_idx, len(dataset) - 1))
            img_tensor, true_label = dataset[idx]
            predictions = predict_sample(model, img_tensor, device=device, top_k=3)

            print(f"\n[Sample Prediction] Test Sample #{idx}:", flush=True)
            print(f"True Class: {CIFAR10_CLASSES[true_label]}", flush=True)
            print("Top Predictions:")
            for rank, (cls_name, prob) in enumerate(predictions, 1):
                print(f"  {rank}. {cls_name:<12} ({prob:.2f}%)", flush=True)


if __name__ == "__main__":
    main()
