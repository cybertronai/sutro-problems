#!POPCORN leaderboard mnist-a100-5
#!POPCORN gpu A100
#!POPCORN function custom_kernel
"""Yaroslav's shorter-training Ladder, difficulty 5.

Derived from @SethTS's submissions/ladder-fast-20260929/ladder_d5.py.
Uses 8400 updates instead of 9000: 6.7% fewer training updates. The model,
loss, batch size, learning rate and proportional decay schedule are unchanged.
All fitting happens inside the call; parameters and optimizer state are reset.
See README.md for development comparisons and official scorer measurements.
"""
import math
import os

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = False

STEPS, BATCH = 8400, 1000
DIMS = (60, 1000, 500, 250, 250, 250, 10)
NOISE, INPUT_SCALE, RECONSTRUCTION = 0.3, 0.6, 2000.0
LR, EPOCHS_CONFIG, DECAY_START_CONFIG = 0.006, 150, 100
EPS, COMBINATOR_STD = 1e-10, 0.025
SEED = 11
WARPS, TILE = 8, 2048  # warps per Triton program; each program holds all rows and TILE // rows columns
CACHE = {}
# Combinator parameter rows: W1[i, o] = 2i + o, b1 6-7, W2[j, o] = 8 + 2j + o, b2 12-13, W3[j] = 14 + j, b3 16.
WEIGHT_ROWS = (0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 14, 15)


@triton.jit
def _leaky(x):
    return tl.where(x > 0, x, 0.1 * x)


@triton.jit
def _column_norm(u, rows_ok, n, eps):
    """Biased batch statistics over the rows of a [rows, columns] tile; returns (normalised, rstd)."""
    mean = tl.sum(u, 0) / n
    centred = tl.where(rows_ok, u - mean[None, :], 0.0)
    rstd = 1.0 / tl.sqrt(tl.sum(centred * centred, 0) / n + eps)
    return centred * rstd[None, :], rstd


@triton.jit
def _decoder_forward(u_ptr, lat_ptr, p_ptr, out_ptr, n, s, eps, ROWS: tl.constexpr, COLS: tl.constexpr):
    cols = tl.program_id(0) * COLS + tl.arange(0, COLS)
    rows = tl.arange(0, ROWS)
    rows_ok = (rows < n)[:, None]
    ok = rows_ok & (cols < s)[None, :]
    index = rows[:, None] * s + cols[None, :]
    v, _ = _column_norm(tl.load(u_ptr + index, mask=ok, other=0.0), rows_ok, n, eps)
    l = tl.load(lat_ptr + index, mask=ok, other=0.0)
    col_ok = cols < s
    p = p_ptr + cols
    x2 = v * l
    a0 = v * tl.load(p, col_ok)[None, :] + l * tl.load(p + 2 * s, col_ok)[None, :] + x2 * tl.load(p + 4 * s, col_ok)[None, :] + tl.load(p + 6 * s, col_ok)[None, :]
    a1 = v * tl.load(p + s, col_ok)[None, :] + l * tl.load(p + 3 * s, col_ok)[None, :] + x2 * tl.load(p + 5 * s, col_ok)[None, :] + tl.load(p + 7 * s, col_ok)[None, :]
    h0, h1 = _leaky(a0), _leaky(a1)
    c0 = h0 * tl.load(p + 8 * s, col_ok)[None, :] + h1 * tl.load(p + 10 * s, col_ok)[None, :] + tl.load(p + 12 * s, col_ok)[None, :]
    c1 = h0 * tl.load(p + 9 * s, col_ok)[None, :] + h1 * tl.load(p + 11 * s, col_ok)[None, :] + tl.load(p + 13 * s, col_ok)[None, :]
    out = _leaky(c0) * tl.load(p + 14 * s, col_ok)[None, :] + _leaky(c1) * tl.load(p + 15 * s, col_ok)[None, :] + tl.load(p + 16 * s, col_ok)[None, :]
    tl.store(out_ptr + index, out, mask=ok)


@triton.jit
def _decoder_backward(u_ptr, lat_ptr, p_ptr, g_ptr, du_ptr, dlat_ptr, dp_ptr, n, s, eps,
                      ROWS: tl.constexpr, COLS: tl.constexpr):
    cols = tl.program_id(0) * COLS + tl.arange(0, COLS)
    rows = tl.arange(0, ROWS)
    rows_ok = (rows < n)[:, None]
    col_ok = cols < s
    ok = rows_ok & col_ok[None, :]
    index = rows[:, None] * s + cols[None, :]
    v, rstd = _column_norm(tl.load(u_ptr + index, mask=ok, other=0.0), rows_ok, n, eps)
    l = tl.load(lat_ptr + index, mask=ok, other=0.0)
    g = tl.load(g_ptr + index, mask=ok, other=0.0)
    p = p_ptr + cols
    w00, w01 = tl.load(p, col_ok)[None, :], tl.load(p + s, col_ok)[None, :]
    w10, w11 = tl.load(p + 2 * s, col_ok)[None, :], tl.load(p + 3 * s, col_ok)[None, :]
    w20, w21 = tl.load(p + 4 * s, col_ok)[None, :], tl.load(p + 5 * s, col_ok)[None, :]
    u00, u01 = tl.load(p + 8 * s, col_ok)[None, :], tl.load(p + 9 * s, col_ok)[None, :]
    u10, u11 = tl.load(p + 10 * s, col_ok)[None, :], tl.load(p + 11 * s, col_ok)[None, :]
    z0, z1 = tl.load(p + 14 * s, col_ok)[None, :], tl.load(p + 15 * s, col_ok)[None, :]
    x2 = v * l
    a0 = v * w00 + l * w10 + x2 * w20 + tl.load(p + 6 * s, col_ok)[None, :]
    a1 = v * w01 + l * w11 + x2 * w21 + tl.load(p + 7 * s, col_ok)[None, :]
    h0, h1 = _leaky(a0), _leaky(a1)
    c0 = h0 * u00 + h1 * u10 + tl.load(p + 12 * s, col_ok)[None, :]
    c1 = h0 * u01 + h1 * u11 + tl.load(p + 13 * s, col_ok)[None, :]
    dc0 = g * z0 * tl.where(c0 > 0, 1.0, 0.1)
    dc1 = g * z1 * tl.where(c1 > 0, 1.0, 0.1)
    da0 = (dc0 * u00 + dc1 * u01) * tl.where(a0 > 0, 1.0, 0.1)
    da1 = (dc0 * u10 + dc1 * u11) * tl.where(a1 > 0, 1.0, 0.1)
    dx2 = da0 * w20 + da1 * w21
    dv = da0 * w00 + da1 * w01 + dx2 * l
    tl.store(dlat_ptr + index, da0 * w10 + da1 * w11 + dx2 * v, mask=ok)
    dv = tl.where(ok, dv, 0.0)
    du = rstd[None, :] * (dv - (tl.sum(dv, 0) / n)[None, :] - v * (tl.sum(dv * v, 0) / n)[None, :])
    tl.store(du_ptr + index, du, mask=ok)
    q = dp_ptr + cols
    tl.store(q, tl.sum(da0 * v, 0), col_ok)
    tl.store(q + s, tl.sum(da1 * v, 0), col_ok)
    tl.store(q + 2 * s, tl.sum(da0 * l, 0), col_ok)
    tl.store(q + 3 * s, tl.sum(da1 * l, 0), col_ok)
    tl.store(q + 4 * s, tl.sum(da0 * x2, 0), col_ok)
    tl.store(q + 5 * s, tl.sum(da1 * x2, 0), col_ok)
    tl.store(q + 6 * s, tl.sum(da0, 0), col_ok)
    tl.store(q + 7 * s, tl.sum(da1, 0), col_ok)
    tl.store(q + 8 * s, tl.sum(dc0 * h0, 0), col_ok)
    tl.store(q + 9 * s, tl.sum(dc1 * h0, 0), col_ok)
    tl.store(q + 10 * s, tl.sum(dc0 * h1, 0), col_ok)
    tl.store(q + 11 * s, tl.sum(dc1 * h1, 0), col_ok)
    tl.store(q + 12 * s, tl.sum(dc0, 0), col_ok)
    tl.store(q + 13 * s, tl.sum(dc1, 0), col_ok)
    tl.store(q + 14 * s, tl.sum(g * _leaky(c0), 0), col_ok)
    tl.store(q + 15 * s, tl.sum(g * _leaky(c1), 0), col_ok)
    tl.store(q + 16 * s, tl.sum(g, 0), col_ok)


@triton.jit
def _encoder_forward(raw_ptr, beta_ptr, gamma_ptr, counter_ptr, z_ptr, h_ptr, n, d, eps, noise, layer,
                     TOP: tl.constexpr, ROWS: tl.constexpr, COLS: tl.constexpr):
    """Both streams (labelled rows 0..n-1, unlabelled n..2n-1), normalised separately, plus noise and activation."""
    cols = tl.program_id(0) * COLS + tl.arange(0, COLS)
    rows = tl.arange(0, ROWS)
    rows_ok = (rows < n)[:, None]
    col_ok = cols < d
    ok = rows_ok & col_ok[None, :]
    beta = tl.load(beta_ptr + cols, col_ok)[None, :]
    seed = tl.load(counter_ptr) * 16 + layer + 1234
    for stream in range(2):
        index = (stream * n + rows[:, None]) * d + cols[None, :]
        xhat, _ = _column_norm(tl.load(raw_ptr + index, mask=ok, other=0.0), rows_ok, n, eps)
        z = xhat + noise * tl.randn(seed, index)
        tl.store(z_ptr + index, z, mask=ok)
        if TOP:
            h = (z + beta) * tl.load(gamma_ptr + cols, col_ok)[None, :]
        else:
            h = tl.maximum(z + beta, 0.0)
        tl.store(h_ptr + index, h, mask=ok)


@triton.jit
def _encoder_backward(raw_ptr, z_ptr, beta_ptr, gamma_ptr, dz_ptr, dh_ptr, draw_ptr, dbeta_ptr, dgamma_ptr,
                      n, d, eps, TOP: tl.constexpr, ROWS: tl.constexpr, COLS: tl.constexpr):
    cols = tl.program_id(0) * COLS + tl.arange(0, COLS)
    rows = tl.arange(0, ROWS)
    rows_ok = (rows < n)[:, None]
    col_ok = cols < d
    ok = rows_ok & col_ok[None, :]
    beta = tl.load(beta_ptr + cols, col_ok)[None, :]
    dbeta = tl.zeros((COLS,), tl.float32)
    dgamma = tl.zeros((COLS,), tl.float32)
    for stream in range(2):
        index = (stream * n + rows[:, None]) * d + cols[None, :]
        xhat, rstd = _column_norm(tl.load(raw_ptr + index, mask=ok, other=0.0), rows_ok, n, eps)
        pre = tl.load(z_ptr + index, mask=ok, other=0.0) + beta
        dh = tl.load(dh_ptr + index, mask=ok, other=0.0)
        if TOP:
            dpre = dh * tl.load(gamma_ptr + cols, col_ok)[None, :]
            dgamma += tl.sum(dh * pre, 0)
        else:
            dpre = tl.where(pre > 0, dh, 0.0)
        dbeta += tl.sum(dpre, 0)
        dz = tl.where(ok, dpre + tl.load(dz_ptr + index, mask=ok, other=0.0), 0.0)
        draw = rstd[None, :] * (dz - (tl.sum(dz, 0) / n)[None, :] - xhat * (tl.sum(dz * xhat, 0) / n)[None, :])
        tl.store(draw_ptr + index, draw, mask=ok)
    tl.store(dbeta_ptr + cols, dbeta, col_ok)
    if TOP:
        tl.store(dgamma_ptr + cols, dgamma, col_ok)


def _columns(n):
    return max(1, TILE // triton.next_power_of_2(n))


def _launch(width, n):
    return (triton.cdiv(width, _columns(n)),)


class Decode(torch.autograd.Function):
    """out = combinator(lateral, normalize(u)), unit-wise, for u and lateral of shape (rows, units)."""

    @staticmethod
    def forward(ctx, u, lateral, p):
        n, s = u.shape
        out = torch.empty_like(u)
        _decoder_forward[_launch(s, n)](u, lateral, p, out, n, s, EPS, ROWS=triton.next_power_of_2(n), COLS=_columns(n), num_warps=WARPS)
        ctx.save_for_backward(u, lateral, p)
        return out

    @staticmethod
    def backward(ctx, g):
        u, lateral, p = ctx.saved_tensors
        n, s = u.shape
        du, dlat, dp = torch.empty_like(u), torch.empty_like(lateral), torch.empty_like(p)
        _decoder_backward[_launch(s, n)](u, lateral, p, g.contiguous(), du, dlat, dp, n, s, EPS,
                                      ROWS=triton.next_power_of_2(n), COLS=_columns(n), num_warps=WARPS)
        return du, dlat, dp


class Encode(torch.autograd.Function):
    """(z, h) for raw of shape (2, rows, units): z = normalize(raw) per stream + noise, h its activation."""

    @staticmethod
    def forward(ctx, raw, beta, gamma, counter, layer):
        _, n, d = raw.shape
        z, h = torch.empty_like(raw), torch.empty_like(raw)
        top = gamma is not None
        _encoder_forward[_launch(d, n)](raw, beta, gamma if top else beta, counter, z, h, n, d, EPS, NOISE, layer,
                                     TOP=top, ROWS=triton.next_power_of_2(n), COLS=_columns(n), num_warps=WARPS)
        ctx.top = top
        ctx.save_for_backward(raw, z, beta, gamma if top else beta)
        return z, h

    @staticmethod
    def backward(ctx, dz, dh):
        raw, z, beta, gamma = ctx.saved_tensors
        _, n, d = raw.shape
        dz = torch.zeros_like(raw) if dz is None else dz.contiguous()
        dh = torch.zeros_like(raw) if dh is None else dh.contiguous()
        draw, dbeta, dgamma = torch.empty_like(raw), torch.empty_like(beta), torch.empty_like(gamma)
        _encoder_backward[_launch(d, n)](raw, z, beta, gamma, dz, dh, draw, dbeta, dgamma, n, d, EPS,
                                      TOP=ctx.top, ROWS=triton.next_power_of_2(n), COLS=_columns(n), num_warps=WARPS)
        return draw, dbeta, dgamma if ctx.top else None, None, None


def loss(params, counter, x_labelled, y_labelled, x_unlabelled):
    h = torch.stack((x_labelled, x_unlabelled))  # the two streams, normalised separately
    h = h + NOISE * torch.randn_like(h)
    lateral = [h[1]]
    last = len(DIMS) - 2
    for layer, weight in enumerate(params["encoder"]):
        z, h = Encode.apply(h @ weight, params["beta"][layer], params["gamma"] if layer == last else None,
                            counter, layer)
        lateral.append(z[1])
    reconstruction = Decode.apply(F.softmax(h[1], dim=-1), lateral[-1], params["combinators"][-1])
    for layer in range(last, -1, -1):
        reconstruction = Decode.apply(reconstruction @ params["decoder"][layer], lateral[layer],
                                      params["combinators"][layer])
    return F.cross_entropy(h[0], y_labelled) + RECONSTRUCTION * F.mse_loss(reconstruction, x_unlabelled)


def encode(params, layer, z):
    """The clean encoder's activation after normalisation, for calibration and prediction."""
    beta = params["beta"][layer]
    if layer == len(DIMS) - 2:
        return (z + beta) * params["gamma"]
    return F.relu(z + beta)


class Trainer:
    def __init__(self, n, pool_n, device):
        self.device = device
        self.steps_per_epoch = math.ceil(n / BATCH)
        self.epochs = max(1, math.ceil(STEPS / self.steps_per_epoch))
        self.total = self.epochs * self.steps_per_epoch
        decay_start = min(self.epochs, max(1, round(self.epochs * DECAY_START_CONFIG / EPOCHS_CONFIG)))
        factor = [1.0 if self.epochs <= decay_start else
                  max(0.0, min(1.0, (self.epochs - e) / (self.epochs - decay_start))) for e in range(self.epochs)]
        self.schedule = (LR * torch.tensor(factor, device=device)).repeat_interleave(self.steps_per_epoch)
        self.n, self.pool_n = n, pool_n
        self.x = torch.zeros(n, DIMS[0], device=device)
        self.y = torch.zeros(n, dtype=torch.long, device=device)
        self.pool = torch.zeros(pool_n, DIMS[0], device=device)
        self.labelled = torch.zeros(self.total, BATCH, dtype=torch.long, device=device)
        self.unlabelled = torch.zeros(self.total, BATCH, dtype=torch.long, device=device)
        self.counter = torch.zeros(1, dtype=torch.long, device=device)
        zeros = lambda *shape: torch.zeros(*shape, device=device, requires_grad=True)
        self.params = {
            "encoder": [zeros(a, b) for a, b in zip(DIMS[:-1], DIMS[1:])],
            "beta": [zeros(b) for b in DIMS[1:]],
            "gamma": zeros(DIMS[-1]),
            "decoder": [zeros(b, a) for a, b in zip(DIMS[:-1], DIMS[1:])],
            "combinators": [zeros(17, s) for s in DIMS],
        }
        self.flat = (self.params["encoder"] + self.params["beta"] + [self.params["gamma"]] + self.params["decoder"]
                     + self.params["combinators"])
        self.lr = torch.tensor(LR, device=device)
        self.optimizer = torch.optim.Adam(self.flat, lr=self.lr, betas=(0.9, 0.999), eps=1e-8,
                                          fused=True, capturable=True)
        self.initialise()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                self.optimizer.zero_grad(set_to_none=True)
                self.step()
        torch.cuda.current_stream().wait_stream(side)
        self.graph = torch.cuda.CUDAGraph()
        self.optimizer.zero_grad(set_to_none=True)
        with torch.cuda.graph(self.graph):
            self.step()

    def step(self):
        labelled = self.labelled.index_select(0, self.counter).squeeze(0)
        unlabelled = self.unlabelled.index_select(0, self.counter).squeeze(0)
        self.lr.copy_(self.schedule.index_select(0, self.counter).squeeze(0))
        loss(self.params, self.counter, self.x[labelled], self.y[labelled], self.pool[unlabelled]).backward()
        self.optimizer.step()
        with torch.no_grad():
            self.counter.add_(1)

    @torch.no_grad()
    def initialise(self):
        generator = torch.Generator(device=self.device)
        generator.manual_seed(SEED)
        normal = lambda p, std: p.copy_(torch.randn(p.shape, generator=generator, device=self.device) * std)
        for weight in self.params["encoder"] + self.params["decoder"]:
            normal(weight, weight.shape[0] ** -0.5)
        for p in self.params["combinators"]:
            p.zero_()
            p[list(WEIGHT_ROWS)] = torch.randn(len(WEIGHT_ROWS), p.shape[1], generator=generator,
                                               device=self.device) * COMBINATOR_STD
        for p in self.params["beta"]:
            p.zero_()
        self.params["gamma"].fill_(1.0)
        for p in self.flat:
            state = self.optimizer.state.get(p)
            if state:
                state["exp_avg"].zero_()
                state["exp_avg_sq"].zero_()
                state["step"].zero_()
        self.counter.zero_()
        # Each epoch's shuffles, drawn at once: labelled rows padded to full minibatches with a second
        # shuffle, unlabelled rows the first steps_per_epoch * BATCH of a shuffle of train and test rows.
        epochs, spe = self.epochs, self.steps_per_epoch
        order = torch.rand(epochs, self.n, generator=generator, device=self.device).argsort(1)
        if spe * BATCH > self.n:
            extra = torch.rand(epochs, self.n, generator=generator, device=self.device).argsort(1)
            order = torch.cat((order, extra[:, :spe * BATCH - self.n]), 1)
        self.labelled.copy_(order.reshape(self.total, BATCH))
        recon = torch.rand(epochs, self.pool_n, generator=generator, device=self.device).argsort(1)
        self.unlabelled.copy_(recon[:, :spe * BATCH].reshape(self.total, BATCH))
        torch.cuda.manual_seed(SEED + 1)

    @torch.no_grad()
    def predict(self, q):
        """Calibrate BatchNorm on the training rows (minibatch means and unbiased variances, averaged over
        one pass, deeper layers fed minibatch-normalised activations), then run the clean encoder on q."""
        params = self.params
        batches = self.n // BATCH
        generator = torch.Generator(device=self.device)
        generator.manual_seed(1)
        rows = torch.randperm(self.n, generator=generator, device=self.device)[:batches * BATCH]
        h = self.x[rows].reshape(batches, BATCH, DIMS[0])
        stats = []
        for layer, weight in enumerate(params["encoder"]):
            raw = h @ weight
            variance, mean = torch.var_mean(raw, dim=1, unbiased=False, keepdim=True)
            stats.append((mean.mean(0), variance.mean(0) * BATCH / (BATCH - 1)))
            h = encode(params, layer, (raw - mean) * torch.rsqrt(variance + EPS))
        h = q
        for layer, weight in enumerate(params["encoder"]):
            mean, variance = stats[layer]
            h = encode(params, layer, (h @ weight - mean) * torch.rsqrt(variance + EPS))
        return h


def ladder(train_x, train_y, test_x):
    # Triton compiles on first use and caches what it builds; the sandboxed worker's home may not be writable.
    os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/triton-cache")
    device = train_x.device
    x = train_x.reshape(train_x.shape[0], -1).float() * INPUT_SCALE
    q = test_x.reshape(test_x.shape[0], -1).float() * INPUT_SCALE
    key = (tuple(x.shape), tuple(q.shape), str(device), STEPS, BATCH)
    if key not in CACHE:
        CACHE.clear()
        CACHE[key] = Trainer(x.shape[0], x.shape[0] + q.shape[0], device)
    trainer = CACHE[key]
    with torch.no_grad():
        trainer.x.copy_(x)
        trainer.y.copy_(train_y.long())
        trainer.pool.copy_(torch.cat((x, q)))
    trainer.initialise()
    for _ in range(trainer.total):
        trainer.graph.replay()
    return trainer.predict(q).argmax(1)


def custom_kernel(train_x, train_y, test_x):
    return ladder(train_x, train_y, test_x)
