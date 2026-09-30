"""Non-neural kernel ridge classifier, refitted from the supplied labels."""

import torch
import triton
import triton.language as tl

torch.backends.cuda.matmul.allow_tf32 = False

GAMMA, RIDGE, STEPS = 0.02, 0.1, 16
RANK = 256
KIND = "rbf"
NORMALIZE = True
USE_TRITON = True
METRIC_UPDATES, METRIC_SAMPLES, METRIC_BLEND = 1, 512, 0.05


@triton.jit
def _product(K, P, PART, N: tl.constexpr, SPLITS: tl.constexpr,
             BM: tl.constexpr, BK: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    split = tl.program_id(1)
    cols = tl.arange(0, 16)
    acc = tl.zeros((BM, 16), tl.float32)
    for block in range(tl.cdiv(N, BK * SPLITS)):
        inner = (block * SPLITS + split) * BK + tl.arange(0, BK)
        a = tl.load(K + rows[:, None] * N + inner[None, :],
                    (rows[:, None] < N) & (inner[None, :] < N), 0)
        b = tl.load(P + inner[:, None] * 10 + cols[None, :],
                    (inner[:, None] < N) & (cols[None, :] < 10), 0)
        acc = tl.dot(a, b, acc, input_precision="tf32x3")
    tl.store(PART + split * N * 16 + rows[:, None] * 16 + cols[None, :],
             acc, rows[:, None] < N)


@triton.jit
def _product_reduce(PART, P, OUT, N: tl.constexpr, RIDGE: tl.constexpr,
                    SPLITS: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, col = index // 10, index % 10
    splits = tl.arange(0, SPLITS)
    partial = tl.load(PART + splits[:, None] * N * 16 + row[None, :] * 16 + col[None, :],
                      row[None, :] < N, 0)
    p = tl.load(P + index, index < N * 10, 0)
    tl.store(OUT + index, tl.sum(partial, 0) + RIDGE * p, index < N * 10)


def product(k, direction, ridge):
    if not USE_TRITON:
        return k @ direction + ridge * direction
    n = k.shape[0]
    partial = torch.empty((8, n, 16), device=k.device)
    out = torch.empty_like(direction)
    _product[(triton.cdiv(n, 32), 8)](k, direction, partial, n, 8, 32, 64)
    _product_reduce[(triton.cdiv(n * 10, 256),)](partial, direction, out, n, ridge, 8, 256)
    return out


def kernel(x, z, gamma, kind):
    distance = (x.square().sum(1)[:, None] + z.square().sum(1)[None, :] - 2 * (x @ z.T)).clamp_min_(0)
    if kind == "laplacian":
        distance.sqrt_()
    return distance.mul_(-gamma).exp_()


def solve(k, targets, ridge, steps):
    if RANK:
        m = min(RANK, len(k))
        landmark = k[:m, :m].clone()
        landmark.diagonal().add_(1e-4)
        factor = torch.linalg.cholesky(landmark)
        u = torch.linalg.solve_triangular(factor, k[:, :m].T.contiguous(), upper=False).T.contiguous()
        small = u.T @ u
        small.diagonal().add_(ridge)
        factor = torch.linalg.cholesky(small)
        v = torch.linalg.solve_triangular(factor, u.T.contiguous(), upper=False).T.contiguous()

    def precondition(residual):
        if not RANK:
            return residual
        return (residual - v @ (v.T @ residual)) / ridge

    weights = torch.zeros_like(targets)
    residual = targets.clone()
    direction = precondition(residual).clone()
    norm = (residual * direction).sum(0)
    for _ in range(steps):
        applied = product(k, direction, ridge)
        alpha = norm / (direction.mul(applied).sum(0).clamp_min(1e-20))
        weights.add_(direction * alpha)
        residual.sub_(applied * alpha)
        preconditioned = precondition(residual)
        next_norm = (residual * preconditioned).sum(0)
        direction = preconditioned + direction * (next_norm / norm.clamp_min(1e-20))
        norm = next_norm
    return weights


def prepare(train_x, test_x):
    x, q = train_x.float(), test_x.float()
    if NORMALIZE:
        x = x / x.norm(dim=1, keepdim=True).clamp_min(1e-6) * (x.shape[1] ** 0.5)
        q = q / q.norm(dim=1, keepdim=True).clamp_min(1e-6) * (q.shape[1] ** 0.5)
    return x, q


def gradient_metric(x, transform, weights, gamma, kind, samples, blend):
    z = x @ transform
    points = z[:samples]
    factor = kernel(points, z, gamma, kind)
    if kind == "laplacian":
        distance = (points.square().sum(1)[:, None] + z.square().sum(1)[None] - 2 * (points @ z.T)).clamp_min_(0)
        factor = factor / distance.sqrt_().clamp_min_(1e-6)
        index = torch.arange(len(points), device=x.device)
        factor[index, index] = 0
    scores = factor @ weights
    weighted_points = (weights[:, :, None] * z[:, None, :]).reshape(len(z), -1)
    gradients = (factor @ weighted_points).reshape(len(points), 10, -1) - scores[:, :, None] * points[:, None, :]
    gradients = gradients @ transform.T
    gradients = gradients.reshape(-1, x.shape[1])
    metric = gradients.T @ gradients
    metric = metric * (x.shape[1] / metric.trace().clamp_min(1e-12))
    metric = (1 - blend) * metric + blend * torch.eye(x.shape[1], device=x.device)
    values, vectors = torch.linalg.eigh(metric)
    return vectors * values.clamp_min(1e-6).sqrt()[None, :]


def classify(train_x, train_y, test_x):
    with torch.no_grad():
        x, q = prepare(train_x, test_x)
        targets = torch.zeros((len(x), 10), device=x.device)
        targets.scatter_(1, train_y[:, None], 1.0)
        # ponytail: quadratic storage for feasibility; use landmarks if bandwidth dominates.
        transform = torch.eye(x.shape[1], device=x.device)
        for update in range(METRIC_UPDATES + 1):
            z = x @ transform if METRIC_UPDATES else x
            k = kernel(z, z, GAMMA, KIND)
            k.diagonal().fill_(1.0)
            weights = solve(k, targets, RIDGE, STEPS)
            if update < METRIC_UPDATES:
                transform = gradient_metric(x, transform, weights, GAMMA, KIND, METRIC_SAMPLES, METRIC_BLEND)
        q = q @ transform if METRIC_UPDATES else q
        return torch.cat([
            (kernel(chunk, z, GAMMA, KIND) @ weights).argmax(1)
            for chunk in q.split(2000)
        ])


def self_check():
    global USE_TRITON, RANK
    original = USE_TRITON
    torch.manual_seed(782)
    x = torch.randn(128, 12, device="cuda")
    targets = torch.randn(128, 10, device="cuda")
    k = kernel(x, x, 0.1, "rbf")
    rotation, _ = torch.linalg.qr(torch.randn(12, 12, device="cuda"))
    torch.testing.assert_close(kernel(x @ rotation, x @ rotation, 0.1, "rbf"), k,
                               atol=1e-5, rtol=1e-5)
    actual = solve(k, targets, 0.5, 64)
    expected = torch.linalg.solve(k + 0.5 * torch.eye(128, device="cuda"), targets)
    torch.testing.assert_close(actual, expected, atol=0.003, rtol=0.003)
    original_rank = RANK
    try:
        RANK = 0
        torch.testing.assert_close(solve(k, targets, 0.5, 64), expected, atol=0.003, rtol=0.003)
    finally:
        RANK = original_rank
    try:
        USE_TRITON = True
        for n in (128, 1003):
            matrix = torch.randn(n, n, device="cuda")
            direction = torch.randn(n, 10, device="cuda")
            torch.testing.assert_close(product(matrix, direction, 0.1), matrix @ direction + 0.1 * direction,
                                       atol=0.001, rtol=0.001)
        torch.testing.assert_close(solve(k, targets, 0.5, 64), expected, atol=0.003, rtol=0.003)
    finally:
        USE_TRITON = original
    # Check the supervised metric against a small autograd Jacobian, not just itself.
    small = torch.randn(17, 4, device="cuda")
    coefficients = torch.randn(17, 10, device="cuda")
    transform = torch.randn(4, 4, device="cuda") / 2
    actual = gradient_metric(small, transform, coefficients, 0.2, "rbf", 5, 0.05)
    points = small[:5].clone().requires_grad_()
    scores = kernel(points @ transform, small @ transform, 0.2, "rbf") @ coefficients
    gradients = torch.stack([torch.autograd.grad(scores[:, c].sum(), points, retain_graph=True)[0]
                             for c in range(10)], 1).reshape(-1, 4)
    expected = gradients.T @ gradients
    expected = 0.95 * expected * (4 / expected.trace()) + 0.05 * torch.eye(4, device="cuda")
    torch.testing.assert_close(actual @ actual.T, expected, atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    self_check()
