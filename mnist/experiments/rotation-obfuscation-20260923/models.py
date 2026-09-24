"""Learners used both by honest entrants and by attackers (numpy in, numpy out)."""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(4)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def _seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed % (2 ** 31))


class MLP(nn.Module):
    def __init__(self, d_in, width=512, depth=2):
        super().__init__()
        layers, d = [], d_in
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.ReLU()]
            d = width
        layers += [nn.Linear(d, 10)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class CNN(nn.Module):
    """Small spatial model for 9x9 inputs: three 3x3 conv layers with BatchNorm,
    a 128-unit head with dropout.  This is the 'spatial prior' an attacker wants
    to bring in; it is also the honest reference on true pixels."""
    def __init__(self, c1=32, c2=64, c3=64):
        super().__init__()
        self.c1 = nn.Conv2d(1, c1, 3, padding=1); self.b1 = nn.BatchNorm2d(c1)
        self.c2 = nn.Conv2d(c1, c2, 3, padding=1); self.b2 = nn.BatchNorm2d(c2)
        self.c3 = nn.Conv2d(c2, c3, 3, padding=1); self.b3 = nn.BatchNorm2d(c3)
        self.fc1 = nn.Linear(c3 * 81, 128)
        self.drop = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.b1(self.c1(x)))
        x = F.relu(self.b2(self.c2(x)))
        x = F.relu(self.b3(self.c3(x)))
        return self.fc2(self.drop(F.relu(self.fc1(x.flatten(1)))))


def _shift_augment(xb, rng):
    """With probability 1/2 shift the whole batch by one pixel (zero fill)."""
    if rng.random() < 0.5:
        return xb
    dx, dy = int(rng.integers(-1, 2)), int(rng.integers(-1, 2))
    if dx == 0 and dy == 0:
        return xb
    out = torch.zeros_like(xb)
    h, w = xb.shape[-2:]
    ys, yd = (slice(0, h - dy), slice(dy, h)) if dy >= 0 else (slice(-dy, h), slice(0, h + dy))
    xs, xd = (slice(0, w - dx), slice(dx, w)) if dx >= 0 else (slice(-dx, w), slice(0, w + dx))
    out[..., yd, xd] = xb[..., ys, xs]
    return out


def _train(model, x, y, epochs, batch, lr, seed, augment=False, x_val=None):
    _seed(seed)
    rng = np.random.default_rng(seed)
    x = torch.as_tensor(x, dtype=torch.float32).to(DEVICE)
    y = torch.as_tensor(y, dtype=torch.long).to(DEVICE)
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * ((len(x) + batch - 1) // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(len(x), device=DEVICE)
        for i in range(0, len(x), batch):
            idx = perm[i:i + batch]
            xb, yb = x[idx], y[idx]
            if augment:
                xb = _shift_augment(xb, rng)
            loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
    model.eval()
    return model


@torch.no_grad()
def predict(model, x, batch=2048):
    x = torch.as_tensor(x, dtype=torch.float32).to(DEVICE)
    out = []
    for i in range(0, len(x), batch):
        out.append(model(x[i:i + batch]).argmax(1).cpu())
    return torch.cat(out).numpy()


class Standardizer:
    def __init__(self, x):
        self.mu = x.mean(0, keepdims=True)
        # floor the scale: near-dead coordinates (the 9x9 border) must not be amplified 1e6x at test time
        self.sd = np.maximum(x.std(0, keepdims=True), 1e-2)

    def __call__(self, x):
        return (x - self.mu) / self.sd


def mlp_fit_predict(z_train, y_train, z_test, seed=0, epochs=30, width=512):
    """Honest permutation/rotation-invariant learner on released features."""
    t = time.perf_counter()
    st = Standardizer(z_train)
    model = _train(MLP(z_train.shape[1], width), st(z_train), y_train, epochs, 128, 2e-3, seed)
    pred = predict(model, st(z_test))
    return pred, time.perf_counter() - t


def logreg_fit_predict(z_train, y_train, z_test, C=1.0):
    from sklearn.linear_model import LogisticRegression
    t = time.perf_counter()
    st = Standardizer(z_train)
    clf = LogisticRegression(C=C, max_iter=1000)
    clf.fit(st(z_train), y_train)
    return clf.predict(st(z_test)), time.perf_counter() - t


def cnn_fit(imgs_train, y_train, seed=0, epochs=30, augment=True, standardize=True):
    """imgs: (n, 9, 9) float.  Returns (model, standardizer)."""
    flat = imgs_train.reshape(len(imgs_train), -1)
    st = Standardizer(flat) if standardize else None
    xin = (st(flat) if st else flat).reshape(-1, 1, 9, 9)
    model = _train(CNN(), xin, y_train, epochs, 128, 2e-3, seed, augment=augment)
    return model, st


def cnn_predict(model, st, imgs):
    flat = imgs.reshape(len(imgs), -1)
    xin = (st(flat) if st else flat).reshape(-1, 1, 9, 9)
    return predict(model, xin)


def cnn_fit_predict(imgs_train, y_train, imgs_test, seed=0, epochs=30, augment=True):
    t = time.perf_counter()
    model, st = cnn_fit(imgs_train, y_train, seed, epochs, augment)
    return cnn_predict(model, st, imgs_test), time.perf_counter() - t


def adapter_fit_predict(model, st, z_train, y_train, z_test, h0, c0, epochs=40, lr=5e-3, seed=0, nonneg=1.0):
    """Frozen offline CNN + learned linear adapter x_hat = z H^T + c (attacker-side).

    The attacker never recovers a basis explicitly: H, c are fitted on the 10k
    released labelled rows by cross-entropy through the frozen pixel-space CNN,
    with early stopping on the attacker's own 1k validation split and a penalty on
    negative reconstructed pixels.  Returns (pred_test, best_val_acc)."""
    _seed(seed)
    model = model.to(DEVICE).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    n = len(z_train)
    perm = np.random.default_rng(seed).permutation(n)
    va, tr = perm[:1000], perm[1000:]
    zt = torch.as_tensor(z_train, dtype=torch.float32, device=DEVICE)
    yt = torch.as_tensor(y_train, dtype=torch.long, device=DEVICE)
    ze = torch.as_tensor(z_test, dtype=torch.float32, device=DEVICE)
    mu = torch.as_tensor(st.mu, dtype=torch.float32, device=DEVICE)
    sd = torch.as_tensor(st.sd, dtype=torch.float32, device=DEVICE)
    h = torch.nn.Parameter(torch.as_tensor(np.asarray(h0), dtype=torch.float32, device=DEVICE))
    c = torch.nn.Parameter(torch.as_tensor(np.asarray(c0), dtype=torch.float32, device=DEVICE))
    opt = torch.optim.Adam([h, c], lr=lr)

    def forward(z):
        x_hat = z @ h.T + c
        return model(((x_hat - mu) / sd).reshape(-1, 1, 9, 9)), x_hat

    best = (-1.0, None, None)
    tr_t = torch.as_tensor(tr, device=DEVICE)
    for ep in range(epochs):
        order = tr_t[torch.randperm(len(tr_t), device=DEVICE)]
        for i in range(0, len(order), 128):
            idx = order[i:i + 128]
            logits, x_hat = forward(zt[idx])
            loss = F.cross_entropy(logits, yt[idx]) + nonneg * torch.relu(-x_hat).pow(2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        with torch.no_grad():
            val = (forward(zt[va])[0].argmax(1) == yt[va]).float().mean().item()
        if val > best[0]:
            best = (val, h.detach().clone(), c.detach().clone())
    with torch.no_grad():
        h.copy_(best[1]); c.copy_(best[2])
        pred = torch.cat([forward(ze[i:i + 2048])[0].argmax(1).cpu() for i in range(0, len(ze), 2048)]).numpy()
    return pred, best[0] * 100
