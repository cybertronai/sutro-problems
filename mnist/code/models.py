"""Compact baselines and predetermined hyperparameter searches for tiny MNIST.

Inputs are standardized float tensors with shape ``(N, 1, H, W)``. Every model
returns ten unnormalized class logits. Configurations are intentionally fixed
before training, so candidate selection can use validation data without tuning
the search against test accuracy.
"""

from __future__ import annotations

from typing import Any

from torch import nn


class TinyConvNet(nn.Module):
    """Keep position information, which global pooling would discard at 3x3."""

    def __init__(
        self,
        image_size: int,
        width: int,
        depth: int,
        dropout: float,
        pooling: str = "none",
    ) -> None:
        super().__init__()
        if pooling not in {"none", "max"}:
            raise ValueError(f"Unknown pooling mode: {pooling!r}")
        if pooling != "none" and image_size < 9:
            raise ValueError("Pooling is disabled for images smaller than 9x9")

        blocks: list[nn.Module] = []
        channels = 1
        spatial_size = image_size
        for layer in range(depth):
            blocks.extend(
                [
                    nn.Conv2d(channels, width, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(width),
                    nn.GELU(),
                ]
            )
            channels = width
            # Pool once, after the second convolution (or the only convolution).
            # Odd 9x9 images become 4x4; the head still sees absolute position.
            if pooling == "max" and layer == min(1, depth - 1):
                blocks.append(nn.MaxPool2d(2))
                spatial_size //= 2

        self.features = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(width * spatial_size * spatial_size, width * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width * 4, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def build_model(config: dict[str, Any], image_size: int) -> nn.Module:
    """Construct a model without applying softmax or doing normalization."""
    if image_size <= 0:
        raise ValueError("image_size must be positive")
    architecture = config["architecture"]
    if architecture == "linear":
        return nn.Sequential(nn.Flatten(), nn.Linear(image_size**2, 10))

    width = int(config["width"])
    depth = int(config["depth"])
    dropout = float(config["dropout"])
    if width <= 0 or depth <= 0:
        raise ValueError("width and depth must be positive")
    if not 0 <= dropout < 1:
        raise ValueError("dropout must lie in [0, 1)")

    if architecture == "mlp":
        layers: list[nn.Module] = [nn.Flatten()]
        input_features = image_size**2
        for _ in range(depth):
            layers.extend(
                [
                    nn.Linear(input_features, width),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            input_features = width
        layers.append(nn.Linear(width, 10))
        return nn.Sequential(*layers)

    if architecture == "cnn":
        return TinyConvNet(
            image_size=image_size,
            width=width,
            depth=depth,
            dropout=dropout,
            pooling=config.get("pooling", "none"),
        )
    raise ValueError(f"Unknown architecture: {architecture!r}")


def candidate_configs(tier: str) -> list[dict[str, Any]]:
    """Return a new, reproducible 18-trial validation search for each tier.

    ``depth`` counts hidden linear layers for MLPs and convolutions for CNNs.
    All candidates assume AdamW and the same learning-rate schedule supplied by
    the runner. The linear model is a sanity control, not an architecture prior.
    """
    tier = tier.lower().replace("_", "").replace("-", "").replace(" ", "")
    if tier in {"small", "mnistsmall", "small3x3", "3x3"}:
        # architecture, width, depth, dropout, lr, weight_decay, batch_size, pool
        settings = [
            ("linear", 0, 0, 0.0, 0.01, 1e-3, 64, "none"),
            ("mlp", 64, 2, 0.0, 0.003, 1e-3, 64, "none"),
            ("mlp", 128, 2, 0.0, 0.003, 1e-3, 64, "none"),
            ("mlp", 256, 2, 0.0, 0.003, 1e-3, 64, "none"),
            ("mlp", 128, 3, 0.0, 0.001, 1e-3, 64, "none"),
            ("mlp", 256, 3, 0.1, 0.001, 1e-3, 64, "none"),
            ("mlp", 128, 2, 0.1, 0.003, 1e-3, 64, "none"),
            ("mlp", 128, 2, 0.2, 0.003, 1e-3, 64, "none"),
            ("mlp", 128, 2, 0.0, 0.001, 1e-3, 64, "none"),
            ("mlp", 128, 2, 0.0, 0.003, 1e-2, 64, "none"),
            ("mlp", 128, 2, 0.1, 0.003, 1e-3, 128, "none"),
            ("cnn", 16, 2, 0.0, 0.001, 1e-3, 64, "none"),
            ("cnn", 32, 2, 0.0, 0.001, 1e-3, 64, "none"),
            ("cnn", 32, 2, 0.1, 0.001, 1e-3, 64, "none"),
            ("cnn", 32, 2, 0.2, 0.003, 1e-3, 64, "none"),
            ("cnn", 32, 3, 0.1, 0.001, 1e-3, 64, "none"),
            ("cnn", 32, 2, 0.1, 0.003, 1e-2, 64, "none"),
            ("cnn", 32, 2, 0.1, 0.001, 1e-3, 128, "none"),
        ]
        name, epochs = "small", 120
    elif tier in {"medium", "mnistmedium", "medium9x9", "9x9"}:
        settings = [
            ("linear", 0, 0, 0.0, 0.01, 1e-3, 128, "none"),
            ("mlp", 128, 2, 0.1, 0.001, 1e-3, 128, "none"),
            ("mlp", 256, 2, 0.1, 0.001, 1e-3, 128, "none"),
            ("mlp", 256, 2, 0.2, 0.003, 1e-3, 128, "none"),
            ("mlp", 256, 3, 0.1, 0.001, 1e-3, 128, "none"),
            ("mlp", 512, 2, 0.2, 0.001, 1e-3, 128, "none"),
            ("cnn", 16, 2, 0.1, 0.001, 1e-3, 128, "max"),
            ("cnn", 32, 2, 0.1, 0.001, 1e-3, 128, "max"),
            ("cnn", 32, 2, 0.1, 0.003, 1e-3, 128, "max"),
            ("cnn", 32, 2, 0.2, 0.001, 1e-3, 128, "max"),
            ("cnn", 48, 2, 0.1, 0.001, 1e-3, 128, "max"),
            ("cnn", 32, 3, 0.1, 0.001, 1e-3, 128, "max"),
            ("cnn", 32, 3, 0.2, 0.003, 1e-3, 128, "max"),
            ("cnn", 32, 2, 0.1, 0.001, 1e-2, 128, "max"),
            ("cnn", 32, 2, 0.1, 0.001, 1e-3, 256, "max"),
            ("cnn", 16, 2, 0.1, 0.001, 1e-3, 128, "none"),
            ("cnn", 32, 2, 0.1, 0.001, 1e-3, 128, "none"),
            ("cnn", 32, 3, 0.2, 0.001, 1e-3, 128, "none"),
        ]
        name, epochs = "medium", 60
    else:
        raise ValueError(f"No tuning search defined for tier {tier!r}")

    configs = []
    for index, values in enumerate(settings):
        architecture, width, depth, dropout, lr, weight_decay, batch_size, pool = values
        config = dict(
            trial_id=f"{name}-{index:02d}-{architecture}",
            architecture=architecture,
            width=width,
            depth=depth,
            dropout=dropout,
            lr=lr,
            weight_decay=weight_decay,
            batch_size=batch_size,
            epochs=epochs if architecture != "linear" else min(epochs, 80),
        )
        if architecture == "cnn":
            config["pooling"] = pool
        configs.append(config)
    return configs
