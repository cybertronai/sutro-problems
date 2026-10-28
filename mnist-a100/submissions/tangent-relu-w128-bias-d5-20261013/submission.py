"""Cross-taught Muon ReLU-MLP ensemble (width 128) with local tangent jitter and joint train+query label propagation,
for mnist-a100 difficulty 5.

Eight MLPs (64-128-128-128-10: inputs zero-padded 60 -> 64 so the first GEMM is 16-byte aligned; hidden layers
x W^T + b -> RMSNorm without gain -> ReLU -> dropout 0.05) train together, batched over members, for 2,200
updates. Dropout masks come from a counter hash of (element, update, layer) computed inside the compiled step. Muon
(3-step Newton-Schulz, one batched call for every square matrix) updates every hidden matrix; AdamW for the hidden
biases and the head is its own small compiled graph. Labelled rows are jittered along their local tangents (per row, the top 4 principal
directions of the offsets to its 10 nearest same-class labelled rows, scaled to the rms offset; a fresh N(0, 1) step
along each per update) and then mixed up (Beta(0.5, 0.5)). The members form two halves of four, and the query rows
are split once into two disjoint pools, one per half. From 20% of training, every 200 steps each half reads out the
query rows (EMA weights: centred, signed-power (0.25), normalised last-hidden embeddings -> 5-nearest-labelled-
neighbour prior -> three steps of propagation over the kNN graph of labelled rows, clamped to their labels, and query
rows) and the 90% of rows its members are most sure of (mean member probability of the label) become hard
pseudo-labels for the OTHER half, which trains on them only within its own pool, so every vote is on rows the voters
never trained on. The returned labels come from the same readout over all eight members, with each member's
embedding widened by its middle hidden layer (both normalised, concatenated), 10 neighbours, and hub-corrected
similarities (each row's mean similarity to its 10 nearest query rows, x0.3, subtracted).

The training steps are CUDA graphs (loss/backward, the Muon+EMA update and the head AdamW compiled by Inductor)
captured on the warm-up call; every call recomputes the tangents and re-initialises all weights, optimizer state, EMA
and pseudo-labels from scratch.
"""
import os

import torch
import torch.nn.functional as F

torch.backends.cuda.matmul.allow_tf32 = True
E, W, DEPTH, DIN, D, C = 8, 128, 3, 60, 64, 10  # inputs zero-padded DIN -> D so the first-layer GEMM is 16-byte aligned
STEPS, BL, BU = 2200, 2048, 1024
LR_MUON, LR_ADAM, MOM, WD, WARM = 0.04, 0.01, 0.95, 0.1, 0.05
MIXUP, DROP, EMA, SCALE, NS = 0.5, 0.05, 0.998, 2.0, 3
TAN, TAN_K, TAN_R = 1.0, 10, 4
MID = 1.0  # final readout: weight of the middle hidden layer in each member's embedding
START, EVERY, COVER = 0.2, 200, 0.9
KNN, GK, TEMP, ALPHA, GSTEPS, POW, HUB, HUBK = 5, 5, 20.0, 0.9, 3, 0.25, 0.3, 10
FK = 10  # neighbours in the final readout (cross-teaching readouts keep KNN = GK)
_S = {}


def _orth(g):
    tall = g.shape[-2] > g.shape[-1]
    x = g.bfloat16().mT if tall else g.bfloat16()
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(NS):
        a = x @ x.mT
        x = 3.4445 * x + (-4.7750 * a + 2.0315 * a @ a) @ x
    return (x.mT if tall else x).float()


def _keep(shape, seed, layer):
    """Dropout keep-mask from a counter hash (murmur3 finalizer) of (element index, step, layer): no RNG kernel and no
    mask tensor, so Inductor fuses it into the activation; `seed` is the step counter, so every update draws anew."""
    # wrapping int32 arithmetic (signed constants = the murmur3 ones mod 2^32); logical shifts via masks
    s = seed.to(torch.int32) * -2048144777 + layer * -1028477379
    x = (torch.arange(shape[0] * shape[1] * shape[2], device="cuda", dtype=torch.int32).view(shape) + s) * -1640531535
    x = (x ^ ((x >> 16) & 0xFFFF)) * -2048144789
    x = (x ^ ((x >> 13) & 0x7FFFF)) * -1028477387
    return ((x ^ ((x >> 16) & 0xFFFF)) & 0x7FFFFFFF) >= int(DROP * 2 ** 31)


def _forward(P, x, drop=None):
    """x (E, B, 60) -> logits (E, B, 10), last hidden (E, B, W)."""
    h = x * SCALE
    for i in range(DEPTH):
        u = torch.baddbmm(P[DEPTH + i][:, None], h, P[i].mT)  # hidden bias; gains P[2*DEPTH:3*DEPTH] stay unused at 1
        h = F.relu(u * torch.rsqrt(u.float().pow(2).mean(-1, keepdim=True) + 1e-6).to(u.dtype))  # RMSNorm, no gain
        if drop is not None:
            h = h * _keep(h.shape, drop, i).to(h.dtype) * (1 / (1 - DROP))
        if i == DEPTH - 2:
            mid = h
    return torch.baddbmm(P[-1][:, None], h, P[-2].mT), h, mid  # also the middle hidden layer, for the final readout


def _loss(P, tx, ty, il, lam, drop, tan, tr, qx=None, uy=None, uw=None, iu=None):
    lam = torch.maximum(lam, 1 - lam)
    x, y = tx[il] + (tan[il] * tr[..., None, :]).sum(-1), ty[il]  # random step along the row's local tangents
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


def _adam_params(P):
    """Parameters updated by AdamW: the hidden biases and the head."""
    return list(P[DEPTH:2 * DEPTH]) + list(P[-2:])


def _adam(P, G, mom, sched, t):
    """Head AdamW (betas 0.9/0.99, eps 1e-8, decoupled weight decay) as one small compiled graph; = torch.optim.AdamW."""
    la = LR_ADAM * sched.index_select(0, t.view(1))[0]
    n = (t + 1).float()  # AdamW step count (1 on the first update), for bias correction
    for p, g, (m, v) in zip(P, G, mom):
        p.mul_(1 - la * WD)
        m.lerp_(g, 0.1)
        v.mul_(0.99).addcmul_(g, g, value=0.01)
        p.sub_(la / (1 - 0.9 ** n) * m / (v.sqrt() / (1 - 0.99 ** n).sqrt() + 1e-8))


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
    drop = S["t"]  # step counter: seeds the in-kernel dropout hash
    for p in P:
        p.grad = None
    if pseudo:
        iu = S["pool"].gather(1, torch.randint(0, S["pool"].shape[1], (E, BU), device="cuda"))
        loss = S["loss"](P, S["tx"], S["ty"], il, lam, drop, S["tan"], tr, S["qx"], S["uy"], S["uw"], iu)
    else:
        loss = S["loss"](P, S["tx"], S["ty"], il, lam, drop, S["tan"], tr)
    loss.backward()
    with torch.no_grad():
        A = _adam_params(P)
        S["adam"]([p.detach() for p in A], [p.grad for p in A], S["mom"], S["sched"], S["t"])
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
        _, ht, mt = _forward(P, tx[None].expand(m, -1, -1))
        lq, hq, mq = _forward(P, qx[None].expand(m, -1, -1))
    pw = lambda h, mu: F.normalize((h.float() - mu).sign() * (h.float() - mu).abs().pow(POW), dim=-1)  # signed power
    mu, mm = ht.float().mean(1, keepdim=True), mt.float().mean(1, keepdim=True)
    if MID and hub:  # final readout only: last + middle hidden layer, each normalised, middle weighted by MID
        ht = torch.cat([pw(ht, mu), MID * pw(mt, mm)], -1)
        hq = torch.cat([pw(hq, mu), MID * pw(mq, mm)], -1)
        emb = lambda h: F.normalize(h, dim=-1)
    else:
        emb = lambda h: pw(h, mu)
    zt = emb(ht).permute(1, 0, 2).reshape(len(tx), -1).bfloat16()
    zq = emb(hq).permute(1, 0, 2).reshape(len(qx), -1).bfloat16()
    st, sq = (zq @ zt.T).float(), (zq @ zq.T).float()  # query x train, query x query (symmetric)
    sq.fill_diagonal_(-1e9)
    if hub:  # hub correction: each row's mean similarity to its nearest queries, read row-wise
        st -= HUB * (zt @ zq.T).float().topk(HUBK, dim=1).values.mean(1)
        sq -= HUB * sq.topk(HUBK, dim=1).values.mean(1)
    k = FK if hub else KNN  # final readout may use more neighbours; cross-teaching keeps KNN = GK
    sv, si = st.topk(k, dim=1)
    prior = torch.zeros(len(qx), C, device="cuda").scatter_add_(1, ty[si], (TEMP * sv / m).softmax(1))
    qv, qi = sq.topk(k, dim=1)  # graph neighbours = top k of the two blocks' top-k candidates
    gv, j = torch.cat([sv[:, :k], qv], 1).topk(k, dim=1)
    gi = torch.cat([si[:, :k], qi + len(zt)], 1).gather(1, j)
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
        for mv in S["mom"]:
            for v in mv:
                v.zero_()
        S["t"].zero_()
        S["uy"].zero_()
        S["uw"].zero_()
        S["pool"].copy_(torch.randperm(S["qx"].shape[0], generator=g, device="cuda").view(2, -1).repeat_interleave(E // 2, 0))


def _build(n_train, n_test):
    cuda = dict(device="cuda")
    shapes = [(E, W, D)] + [(E, W, W)] * (DEPTH - 1) + [(E, W)] * (2 * DEPTH) + [(E, C, W), (E, C)]
    P = [torch.zeros(s, **cuda, requires_grad=True) for s in shapes]
    t = torch.arange(STEPS + 1, **cuda, dtype=torch.float32)
    warm = max(1, int(WARM * STEPS))
    return {"P": P, "t": torch.zeros((), dtype=torch.long, **cuda), "gen": torch.Generator(**cuda),
            "sched": torch.clamp((t + 1) / warm, max=1) * 0.5 * (1 + torch.cos(torch.pi * t / STEPS)),
            "decay": torch.clamp((1 + t) / (10 + t), max=EMA),
            "ema": [torch.zeros_like(p) for p in P], "bufs": [torch.zeros_like(p) for p in P[:DEPTH]],
            "mom": [(torch.zeros_like(p), torch.zeros_like(p)) for p in _adam_params(P)],  # Adam moments (biases + head)
            "adam": torch.compile(_adam, fullgraph=True, dynamic=False),
            "tx": torch.zeros(n_train, D, **cuda), "ty": torch.zeros(n_train, dtype=torch.long, **cuda),
            "tan": torch.zeros(n_train, D, TAN_R, **cuda),
            "qx": torch.zeros(n_test, D, **cuda), "uy": torch.zeros(E, n_test, dtype=torch.long, **cuda),
            "uw": torch.zeros(E, n_test, **cuda), "pool": torch.zeros(E, n_test // 2, dtype=torch.long, **cuda),
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
    S["tx"][:, :DIN].copy_(train_x)  # padding columns stay zero (buffers are built with zeros)
    S["ty"].copy_(train_y)
    S["qx"][:, :DIN].copy_(test_x)
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
