#!POPCORN leaderboard mnist-medium-2pct
#!POPCORN gpu A100

"""512-filter CG pair, ported from mnist/submissions/medium-cg-pair-20260916 (@jurajselep).

Two ridge systems fitted by 300 Jacobi-preconditioned conjugate-gradient
iterations and fused by z-scores:

  1. random convolutional features - 512 frozen 3x3 filters from PCG64 seed 0,
     ReLU, 3x3 mean pooling, mean-centred, ridge lambda 1e-3 x mean diagonal
  2. an RBF kernel on the arcsine-transformed pixels, gamma 0.3, lambda 1e-2

Targets are one-versus-rest +1/-1. Reported at 98.12% on MNIST-medium (the 2%
band) and about 260 ms per call on an A100. Every constant below comes from the
submission's config.json; the filters are a seeded random initialization, not a
trained model.

The port is a literal transcription of that submission's learner.py, with the
device taken from the input tensors and the config inlined so this file stands
alone. It is slow on a CPU (minutes per call) but runs there for dry runs.

Harness 1.2.0 release. Draws now arrive as (N, D) features -- the draw's images
whitened onto D = 60 principal directions and secretly rotated -- and there is
no pixel grid, so the convolutional half of the pair has nothing to convolve
over. On that input the file substitutes the same idea without the lattice:
4,608 frozen random ReLU features (the width the 512 filters plus 3x3 pooling
produced), drawn from the same PCG64(0) stream, times the same ridge and the
same CG solver. The arcsine transform is skipped -- the release is signed and
already standardized -- and the RBF bandwidth is set from the data
(gamma = 3.0 / mean pairwise squared distance, which reproduces the upstream
gamma = 0.3 on arcsine pixels, where that mean is about 10) rather than from a
constant tuned for pixel scale. On a pixel release (release_dims: 0) the file
behaves exactly as before.
"""

import torch
import torch.nn.functional as F

FILTERS = 512
FILTER_SEED = 0
CONV_BATCH_SIZE = 128
RIDGE_LAMBDA = 1e-3
RBF_GAMMA = 0.3
RBF_LAMBDA = 1e-2
ITERATIONS = 300
# 512 filters x 3x3 pooled cells = 4,608: the feature width the convolutional
# half produces on a 9x9 image, reused as the width of the flat random features.
# (Spelled as a literal because module scope may hold constants only.)
FLAT_FEATURES = 4608
# gamma * (mean pairwise squared distance) of the upstream pixel setting, used
# to set gamma on a release whose scale is not the pixel scale.
RBF_BANDWIDTH = 3.0

torch.backends.cuda.matmul.allow_tf32 = False

_filters = {}


def frozen_filters(device):
    """numpy PCG64(0): standard_normal((512, 9)) / 3, then standard_normal(512) * 0.1."""
    key = str(device)
    if key not in _filters:
        import numpy as np

        generator = np.random.Generator(np.random.PCG64(FILTER_SEED))
        weights = (generator.standard_normal((FILTERS, 9)) / 3).astype("float32")
        biases = (generator.standard_normal(FILTERS) * 0.1).astype("float32")
        _filters[key] = (
            torch.as_tensor(weights, device=device).view(FILTERS, 1, 3, 3),
            torch.as_tensor(biases, device=device),
        )
    return _filters[key]


_projection = {}


def frozen_projection(device, width):
    """The same PCG64(0) stream, shaped for a flat release: (4608, D) / sqrt(D)."""
    key = (str(device), width)
    if key not in _projection:
        import math

        import numpy as np

        generator = np.random.Generator(np.random.PCG64(FILTER_SEED))
        weights = (
            generator.standard_normal((FLAT_FEATURES, width)) / math.sqrt(width)
        ).astype("float32")
        biases = (generator.standard_normal(FLAT_FEATURES) * 0.1).astype("float32")
        _projection[key] = (
            torch.as_tensor(weights, device=device),
            torch.as_tensor(biases, device=device),
        )
    return _projection[key]


def cg(matrix, rhs):
    """The source recurrence, fixed iteration count, no convergence guards."""
    inverse_diagonal = 1.0 / matrix.diagonal()
    solution = torch.zeros_like(rhs)
    residual = rhs.clone()
    preconditioned = residual * inverse_diagonal[:, None]
    direction = preconditioned.clone()
    rz = (residual * preconditioned).sum(0)
    for _ in range(ITERATIONS):
        product = matrix @ direction
        alpha = rz / (direction * product).sum(0)
        solution = solution + alpha * direction
        residual = residual - alpha * product
        preconditioned = residual * inverse_diagonal[:, None]
        next_rz = (residual * preconditioned).sum(0)
        direction = preconditioned + (next_rz / rz) * direction
        rz = next_rz
    return solution


def features(pixels, weights, biases):
    chunks = [
        F.avg_pool2d(
            F.relu(F.conv2d(chunk.reshape(-1, 1, 9, 9), weights, biases, padding=1)), 3
        ).flatten(1)
        for chunk in pixels.split(CONV_BATCH_SIZE)
    ]
    return chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=0)


def flat_features(values, weights, biases):
    """The lattice-free stand-in: one frozen random ReLU layer of the same width."""
    return torch.relu(values @ weights.T + biases)


def rbf_gamma(values):
    """RBF_BANDWIDTH / mean pairwise squared distance, from a fixed 1,000-row slice."""
    sample = values[:1000]
    return RBF_BANDWIDTH / (torch.cdist(sample, sample) ** 2).mean().clamp_min(1e-6)


def zscore(scores):
    """PyTorch's default sample standard deviation, as in the source."""
    return (scores - scores.mean(1, keepdim=True)) / scores.std(1, keepdim=True)


@torch.no_grad()
def custom_kernel(data):
    train_x, train_y, test_x = data
    device = train_x.device
    spatial = train_x.dim() == 4
    x = train_x.reshape(train_x.shape[0], -1)
    xq = test_x.reshape(test_x.shape[0], -1)

    if spatial:
        # A pixel release: the upstream arcsine transform and the 3x3 filters.
        u = torch.asin(torch.sqrt(x.clamp(0, 1)))
        uq = torch.asin(torch.sqrt(xq.clamp(0, 1)))
        weights, biases = frozen_filters(device)
        gamma = RBF_GAMMA
    else:
        # A linear release: already signed and standardized, and no lattice.
        u, uq = x, xq
        weights, biases = frozen_projection(device, x.shape[1])
        gamma = rbf_gamma(u)

    targets = torch.full((train_y.shape[0], 10), -1.0, device=device, dtype=torch.float32)
    targets[torch.arange(train_y.shape[0], device=device), train_y] = 1.0

    if spatial:
        p, pq = features(u, weights, biases), features(uq, weights, biases)
    else:
        p, pq = flat_features(u, weights, biases), flat_features(uq, weights, biases)
    mean = p.mean(0)
    p = p - mean
    pq = pq - mean

    gram = p.T @ p
    gram.diagonal().add_(RIDGE_LAMBDA * gram.diagonal().mean())
    rhs = p.T @ targets

    kernel = torch.exp(-gamma * torch.cdist(u, u) ** 2)
    kernel.diagonal().add_(RBF_LAMBDA)
    query_kernel = torch.exp(-gamma * torch.cdist(uq, u) ** 2)

    ridge_scores = pq @ cg(gram, rhs)
    rbf_scores = query_kernel @ cg(kernel, targets)
    return (zscore(ridge_scores) + zscore(rbf_scores)).argmax(1)
