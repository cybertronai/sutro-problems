"""Research adoption of @likelytobelaura PR118, pinned 436b80bb40e4c346792ddad71670c7a5dbc8c5e9.
Changes: 2000 updates, cross-teaching every400 from update400, both groups sample all queries.
Retains labelled tangent jitter, RMSNorm/GELU, Muon+AdamW, EMA, signed-power/hub readout.
Joint train/query graph readout: @sjbaebae. Muon coefficients: Keller Jordan et al.
Trains from scratch per call; only train_x/train_y/test_x; no query labels or fitted data.
"""
import os

import torch
import torch.nn.functional as F

torch.backends.cuda.matmul.allow_tf32 = True
E, W, DEPTH, D, C = 8, 256, 3, 60, 10
STEPS, BL, BU = 2000, 2048, 1024
LR_MUON, LR_ADAM, MOM, WD, WARM = 0.04, 0.01, 0.95, 0.1, 0.05
MIXUP, DROP, EMA, SCALE, NS = 0.5, 0.1, 0.998, 2.0, 3
TAN, TAN_K, TAN_R = 1.0, 10, 4
START, EVERY, COVER = 0.2, 400, 0.9
KNN, GK, TEMP, ALPHA, GSTEPS, POW, HUB, HUBK = 5, 5, 20.0, 0.9, 3, 0.25, 0.3, 10
_S = {}


def _orth(g):
    tall = g.shape[-2] > g.shape[-1]
    x = g.bfloat16().mT if tall else g.bfloat16()
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(NS):
        a = x @ x.mT
        x = 3.4445 * x + (-4.7750 * a + 2.0315 * a @ a) @ x
    return (x.mT if tall else x).float()


def _forward(P, x, drop=None):
    """x (E, B, 60) -> logits (E, B, 10), last hidden (E, B, W)."""
    h = x * SCALE
    for i in range(DEPTH):
        u = torch.baddbmm(P[DEPTH + i][:, None], h, P[i].mT)
        h = F.gelu(u * torch.rsqrt(u.float().pow(2).mean(-1, keepdim=True) + 1e-6).to(u.dtype) * P[2 * DEPTH + i][:, None])
        if drop is not None:
            h = h * (drop[i] < 1 - DROP).to(h.dtype) * (1 / (1 - DROP))
    return torch.baddbmm(P[-1][:, None], h, P[-2].mT), h


def _loss(P, tx, ty, il, lam, drop, tan, tr, qx=None, uy=None, uw=None, iu=None):
    lam = torch.maximum(lam, 1 - lam)
    x, y = tx[il] + (tan[il] @ tr[..., None])[..., 0], ty[il]  # random step along the row's local tangents
    x = lam[..., None] * x + (1 - lam[..., None]) * x.roll(1, 1)  # partner = next row of the random batch
    if qx is not None:
        x = torch.cat([x, qx[iu]], 1)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = _forward(P, x, drop)[0].float()
    lp = logits[:, :BL].reshape(-1, C).log_softmax(-1)
    lam = lam.reshape(-1)
    loss = -(lam * lp.gather(1, y.reshape(-1, 1))[:, 0] + (1 - lam) * lp.gather(1, y.roll(1, 1).reshape(-1, 1))[:, 0]).mean()
    if qx is not None:
        lu = logits[:, BL:].reshape(-1, C).log_softmax(-1)
        loss = loss - (lu.gather(1, uy.gather(1, iu).reshape(-1, 1))[:, 0] * uw.gather(1, iu).reshape(-1)).mean()
    return loss * E


def _update(P, G, bufs, ema, sched, decay, t):
    lr = LR_MUON * sched.index_select(0, t.view(1))[0]
    look = []
    for g, buf in zip(G[:DEPTH], bufs):
        buf.mul_(MOM).add_(g)
        look.append(g.add(buf, alpha=MOM))
    us = [_orth(look[0])] + list(_orth(torch.cat(look[1:])).split(E))  # square matrices: one batched call
    for p, u in zip(P[:DEPTH], us):
        p.mul_(1 - lr * WD).sub_(u * (lr * max(1.0, p.shape[-2] / p.shape[-1]) ** 0.5))
    w = 1 - decay.index_select(0, t.view(1))[0]
    for e_, p in zip(ema, P):
        e_.lerp_(p, w)


def _step(S, pseudo):
    P = S["P"]
    il = torch.randint(0, S["tx"].shape[0], (E, BL), device="cuda")
    a = torch.full((E, BL), MIXUP, device="cuda")
    ga, gb = torch._standard_gamma(a), torch._standard_gamma(a)
    lam = ga / (ga + gb)  # Beta(MIXUP, MIXUP), graph-safe
    n = BL + (BU if pseudo else 0)
    tr = torch.randn(E, BL, TAN_R, device="cuda")
    drop = [torch.rand(E, n, W, device="cuda", dtype=torch.bfloat16) for _ in range(DEPTH)]
    for p in P:
        p.grad = None
    if pseudo:
        iu = S["pool"].gather(1, torch.randint(0, S["pool"].shape[1], (E, BU), device="cuda"))
        loss = S["loss"](P, S["tx"], S["ty"], il, lam, drop, S["tan"], tr, S["qx"], S["uy"], S["uw"], iu)
    else:
        loss = S["loss"](P, S["tx"], S["ty"], il, lam, drop, S["tan"], tr)
    loss.backward()
    with torch.no_grad():
        S["lr"].copy_(LR_ADAM * S["sched"].index_select(0, S["t"].view(1))[0])
        S["adam"].step()
        S["update"]([p.detach() for p in P], [p.grad for p in P], S["bufs"], S["ema"], S["sched"], S["decay"], S["t"])
        S["t"].add_(1)


@torch.no_grad()
def _readout(S, members, hub=False):
    """Label propagation over the kNN graph of labelled (clamped) + query rows, on the EMA embeddings of `members`.
    Returns (probs (N, 10), member probs (M, N, 10))."""
    P = [p[members] for p in S["ema"]]
    tx, ty, qx = S["tx"], S["ty"], S["qx"]
    m = P[0].shape[0]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        _, ht = _forward(P, tx[None].expand(m, -1, -1))
        lq, hq = _forward(P, qx[None].expand(m, -1, -1))
    mu = ht.float().mean(1, keepdim=True)
    emb = lambda h: F.normalize((h.float() - mu).sign() * (h.float() - mu).abs().pow(POW), dim=-1)  # signed power
    zt = emb(ht).permute(1, 0, 2).reshape(len(tx), -1).bfloat16()
    zq = emb(hq).permute(1, 0, 2).reshape(len(qx), -1).bfloat16()
    s = (zq @ torch.cat([zt, zq]).T).float()
    s[:, len(zt):].fill_diagonal_(-1e9)
    if hub:
        s -= HUB * s.topk(HUBK, dim=0).values.mean(0)  # hub correction: each row's mean similarity to its nearest queries
    sv, si = s[:, :len(zt)].topk(KNN, dim=1)
    prior = torch.zeros(len(qx), C, device="cuda").scatter_add_(1, ty[si], (TEMP * sv / m).softmax(1))
    gv, gi = s.topk(GK, dim=1)
    gw = (TEMP * gv / m).softmax(1)
    yt = F.one_hot(ty, C).float()
    p = prior
    for _ in range(GSTEPS):
        p = ALPHA * (gw[..., None] * torch.cat([yt, p])[gi]).sum(1) + (1 - ALPHA) * prior
    return p, lq.float().softmax(-1)


@torch.no_grad()
def _cross_teach(S):
    """Each half's readout labels the query rows; its surest COVER fraction teaches the other half."""
    for h in (0, 1):
        p, mp = _readout(S, slice(h * E // 2, (h + 1) * E // 2))
        lab = p.argmax(-1)
        score = mp.mean(0).gather(1, lab[:, None])[:, 0]
        w = torch.zeros_like(score).scatter_(0, score.topk(int(COVER * len(lab))).indices, 1.0)
        other = slice((1 - h) * E // 2, (2 - h) * E // 2)
        S["uy"][other] = lab
        S["uw"][other] = w


@torch.no_grad()
def _tangents(tx, ty):
    """(N, 60, TAN_R): per labelled row, the top TAN_R principal directions of the offsets to its TAN_K nearest
    same-class rows, each scaled to the rms offset along it."""
    sim = tx @ tx.T - 0.5 * tx.pow(2).sum(1)  # ranks rows by -distance^2 / 2 (row term dropped)
    sim.masked_fill_(ty[:, None] != ty[None], -float("inf")).fill_diagonal_(-float("inf"))
    Dm = tx[sim.topk(TAN_K, dim=1).indices] - tx[:, None]  # (N, k, 60) offsets to same-class neighbours
    V = torch.linalg.eigh(Dm @ Dm.mT)[1][..., -TAN_R:]  # top eigenvectors of each k x k Gram matrix
    return Dm.mT @ V / TAN_K ** 0.5


def _reset(S):
    """Fresh weights, optimizer state, EMA and pseudo-labels for every call: nothing learned carries over."""
    g = S["gen"]
    g.manual_seed(0)
    P, tx = S["P"], S["tx"]
    with torch.no_grad():
        P[0].copy_(F.normalize(tx[torch.randint(0, len(tx), (E * W,), generator=g, device="cuda")].view(E, W, D), dim=-1))
        for p in P[1:DEPTH]:
            p.copy_(torch.randn(p.shape, generator=g, device="cuda") * W ** -0.5)
        for p in P[DEPTH:2 * DEPTH]:
            p.zero_()
        for p in P[2 * DEPTH:3 * DEPTH]:
            p.fill_(1.0)
        P[-2].copy_(torch.randn(P[-2].shape, generator=g, device="cuda") * 0.5 * W ** -0.5)
        P[-1].zero_()
        for e_, p in zip(S["ema"], P):
            e_.copy_(p)
        for b in S["bufs"]:
            b.zero_()
        for st in S["adam"].state.values():
            for v in st.values():
                v.zero_()
        S["t"].zero_()
        S["uy"].zero_()
        S["uw"].zero_()
        S["pool"].copy_(torch.randperm(S["qx"].shape[0], generator=g, device="cuda")[None].repeat(E,1))


def _build(n_train, n_test):
    cuda = dict(device="cuda")
    shapes = [(E, W, D)] + [(E, W, W)] * (DEPTH - 1) + [(E, W)] * (2 * DEPTH) + [(E, C, W), (E, C)]
    P = [torch.zeros(s, **cuda, requires_grad=True) for s in shapes]
    lr = torch.tensor(LR_ADAM, **cuda)
    t = torch.arange(STEPS + 1, **cuda, dtype=torch.float32)
    warm = max(1, int(WARM * STEPS))
    return {"P": P, "lr": lr, "t": torch.zeros((), dtype=torch.long, **cuda), "gen": torch.Generator(**cuda),
            "sched": torch.clamp((t + 1) / warm, max=1) * 0.5 * (1 + torch.cos(torch.pi * t / STEPS)),
            "decay": torch.clamp((1 + t) / (10 + t), max=EMA),
            "ema": [torch.zeros_like(p) for p in P], "bufs": [torch.zeros_like(p) for p in P[:DEPTH]],
            "adam": torch.optim.AdamW(P[DEPTH:], lr=lr, betas=(0.9, 0.99), weight_decay=WD, fused=True, capturable=True),
            "tx": torch.zeros(n_train, D, **cuda), "ty": torch.zeros(n_train, dtype=torch.long, **cuda),
            "tan": torch.zeros(n_train, D, TAN_R, **cuda),
            "qx": torch.zeros(n_test, D, **cuda), "uy": torch.zeros(E, n_test, dtype=torch.long, **cuda),
            "uw": torch.zeros(E, n_test, **cuda), "pool": torch.zeros(E, n_test, dtype=torch.long, **cuda),
            "loss": torch.compile(_loss, fullgraph=True, dynamic=False),
            "update": torch.compile(_update, fullgraph=True, dynamic=False)}


def _capture(S):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):  # compile, autotune, create optimizer state
        for _ in range(3):
            _step(S, False)
            _step(S, True)
    torch.cuda.current_stream().wait_stream(side)
    for name, pseudo in (("labelled", False), ("pseudo", True)):
        S[name] = torch.cuda.CUDAGraph()
        with torch.cuda.graph(S[name]):
            _step(S, pseudo)


def _load(S, train_x, train_y, test_x):
    S["tx"].copy_(train_x)
    S["ty"].copy_(train_y)
    S["qx"].copy_(test_x)
    S["tan"].copy_(_tangents(S["tx"], S["ty"]) * TAN)


def classify(train_x, train_y, test_x):
    if "S" not in _S:
        uid = os.getuid()
        os.environ.setdefault("TRITON_CACHE_DIR", f"/tmp/triton-{uid}")  # the sandbox's home is read-only
        os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", f"/tmp/inductor-{uid}")
        S = _build(train_x.shape[0], test_x.shape[0])
        _load(S, train_x, train_y, test_x)
        _reset(S)
        _capture(S)
        _S["S"] = S
    S = _S["S"]
    _load(S, train_x, train_y, test_x)
    _reset(S)
    first = int(START * STEPS)
    for step in range(STEPS):
        if step >= first and (step - first) % EVERY == 0:
            _cross_teach(S)
        S["pseudo" if step >= first else "labelled"].replay()
    return _readout(S, slice(None), hub=True)[0].argmax(-1)


if __name__ == "__main__":
    import mnist

    print(mnist.score(classify, difficulty=5))
