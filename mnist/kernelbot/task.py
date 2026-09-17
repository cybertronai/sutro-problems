"""Types for MNIST energy leaderboards."""
from typing import Tuple

import torch

# (train_images (N, 1, H, W) float32 in [0, 1], train_labels (N,) int64, test_images (Q, 1, H, W) float32)
input_t = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
# predicted labels (Q,), integer dtype, on the same device
output_t = torch.Tensor
