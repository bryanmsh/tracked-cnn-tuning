"""
CIFAR-10 Convolutional Neural Network architecture.
Configurable network depth, dropout rate, and channel progression.
"""

from typing import Dict, Any, List
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    Standard convolutional block with two Conv2d-BatchNorm-ReLU layers
    followed by MaxPooling and optional spatial dropout.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        conv_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ]
        if conv_dropout > 0.0:
            layers.append(nn.Dropout2d(p=conv_dropout))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CIFAR10CNN(nn.Module):
    """
    Modular CNN for CIFAR-10 classification supporting hyperparameter tuning
    over network depth (number of conv blocks), dropout rate, and channels.
    """

    def __init__(
        self,
        num_classes: int = 10,
        num_blocks: int = 3,
        base_channels: int = 32,
        dropout_rate: float = 0.3,
        conv_dropout: float = 0.0,
        fc_dim: int = 256,
    ) -> None:
        super().__init__()
        assert 1 <= num_blocks <= 4, f"num_blocks must be between 1 and 4, got {num_blocks}"

        self.num_classes = num_classes
        self.num_blocks = num_blocks
        self.base_channels = base_channels
        self.dropout_rate = dropout_rate
        self.fc_dim = fc_dim

        # Channel progression based on depth: e.g. [32, 64, 128, 256]
        channels = [base_channels * (2**i) for i in range(num_blocks)]

        blocks: List[nn.Module] = []
        in_ch = 3
        for out_ch in channels:
            blocks.append(ConvBlock(in_ch, out_ch, conv_dropout=conv_dropout))
            in_ch = out_ch

        self.features = nn.Sequential(*blocks)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((2, 2))

        final_channels = channels[-1]
        flattened_dim = final_channels * 2 * 2

        self.classifier = nn.Sequential(
            nn.Linear(flattened_dim, fc_dim),
            nn.BatchNorm1d(fc_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(fc_dim, num_classes),
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Kaiming normal initialization for conv and linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        pooled = self.adaptive_pool(feat)
        flat = torch.flatten(pooled, 1)
        logits = self.classifier(flat)
        return logits

    def count_parameters(self) -> Dict[str, int]:
        """Return total and trainable parameter count."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total_params": total, "trainable_params": trainable}


def build_model(config: Dict[str, Any]) -> CIFAR10CNN:
    """Helper factory to build CIFAR10CNN from configuration dict."""
    return CIFAR10CNN(
        num_classes=config.get("num_classes", 10),
        num_blocks=config.get("num_blocks", 3),
        base_channels=config.get("base_channels", 32),
        dropout_rate=config.get("dropout_rate", 0.3),
        conv_dropout=config.get("conv_dropout", 0.0),
        fc_dim=config.get("fc_dim", 256),
    )
