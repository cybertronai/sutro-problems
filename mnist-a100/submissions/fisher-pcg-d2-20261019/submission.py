"""MNIST difficulty 2: RBF kernel ridge on a compressed training set in a closed-form learned metric. No backprop;
plain PyTorch (cuBLAS, cuSOLVER, ATen, PyTorch's flash-attention kernel); the whole call is one CUDA graph.

Credits: our RFM/kernel-PCG d2 entry (rfm-pcg-d2, built on @islamborghini's kernel-PCG) rebuilt around the method of
@sigkillme0's #125 d3 entry, re-implemented here in plain PyTorch: the LDA + QDA Fisher-information metric, the
log-radius feature, k-means compression of the training set into hard rows plus weighted centroids, the
FITC-style preconditioner, the extrapolated last PCG step and the untimed warm-up replays.

Per call: (1) rows scaled toward norm sqrt(60) (power 0.6). (2) Metric M = 0.8 F^0.75 + 0.196 LDA + 0.004 I:
LDA = S_w^-1 with the between-class term x3; F = input-space Fisher information (2048 rows) of the tempered softmax
of class Gaussians whose covariances are shrunk 0.3 toward (I + LDA)^-1/2; rows z = x L (M = L L^T), then put on
the sphere of the mean radius plus a log-radius coordinate. (3) Class-restricted k-means (24 per class, 1 Lloyd
step); the 4096 rows of lowest margin (own vs other-class centre) kept, the rest replaced by count-weighted
centroids; (W^1/2 K W^1/2 + 0.1 I) b = W^1/2 Y on the 4336 points (RBF gamma 0.02) by 5 PCG steps (early stopping
regularises) + an extrapolated direction, preconditioned by Nystrom on the centres plus a diagonal (Woodbury, one
Cholesky and one triangular solve, on a high-priority side stream). (4) Labels: argmax of the test scores. test_x
is used without labels (row scaling and features only).

    python mnist.py kk_d2_final.py:kc_d2 --difficulty 2
"""
import gc
import math

import torch
import torch.nn.functional as F

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
F32, F64, F16 = torch.float32, torch.float64, torch.float16

P = {  # metric, compression, kernel ridge / PCG (see the docstring)
    "NPOW": 0.6, "LDA_S": 3.0, "SHR": 0.7, "QT": 0.15, "QROWS": 2048, "UW0": 0.8, "UW1": 0.196,
    "NS_ITERS": 4, "BLEND": 0.02, "KPC": 24, "LLOYD": 1, "NHARD": 4096, "GAMMA": 0.02, "RIDGE": 0.1, "JIT": 1e-4,
    "STEPS": 5, "VSC": 256.0,
}
WARM = 64  # replays ending the warm-up call (an A100 SXM4 runs the first calls ~20% slower)
_GRAPHS = {}
_ST = {}
_C = {}


def const(key, fn):
    """input-independent tensors, made once outside the captured graph (no kernels on replay)"""
    if key not in _C:
        _C[key] = fn()
    return _C[key]


def eye(n, s=1.0, dt=F32):
    return const(("eye", n, s, dt), lambda: torch.eye(n, device="cuda", dtype=dt) * s)


def _stream(k):
    """a high-priority side stream: its latency-bound factorisation chain is the critical path"""
    if k not in _ST:
        _ST[k] = torch.cuda.Stream(priority=-1)
    return _ST[k]


def chol(a):
    return torch.linalg.cholesky_ex(a)[0]


def sqrtm(a, iters):
    """coupled Newton-Schulz square root as the sign iteration on [[0, A/c], [I, 0]] (2 GEMMs per step)"""
    n = len(a)
    c = a.norm()
    x = torch.cat([torch.cat([const(("z", n), lambda: torch.zeros(n, n, device="cuda")), a / c], 1),
                   const(("iz", n), lambda: torch.cat([torch.eye(n, device="cuda"), torch.zeros(n, n, device="cuda")], 1))], 0)
    i15 = eye(2 * n, 1.5)
    for _ in range(iters):
        x = x @ torch.addmm(i15, x, x, alpha=-0.5)
    return x[:n, n:] * c.sqrt()


def pow34(m, P):
    """trace-normalised ~M^0.75 with an identity blend (our AGOP post-processing)"""
    d = len(m)
    sq = sqrtm(m, P["NS_ITERS"])
    m = sq @ sqrtm(sq, P["NS_ITERS"])
    m = m + m.T
    return torch.addcmul(eye(d, P["BLEND"]), m, (1 - P["BLEND"]) * d / m.trace())


def lda(raw, y1h, cnt, s):
    """S_w^-1 with the between-class (Woodbury) term stretched by s, trace d (whitened rows: S_w = I - A A^T,
    A = class means * sqrt(class frequency))"""
    n, d = raw.shape
    A = (y1h.T @ raw).mul_((cnt * n).rsqrt()[:, None]).T
    C = 1 - 1 / n + 1e-3
    H = torch.addmm(eye(10), A.T, A, alpha=-1 / C)
    B = torch.addmm(eye(d, 1 / C), A, torch.cholesky_solve(A.T, chol(H)), alpha=s / C ** 2)
    return B.mul_(d / B.trace())


def qda_fisher(x, y1h, cnt, Bm, P):
    """Class Gaussians (shrunk covariances) -> tempered softmax of their log-likelihoods at QROWS rows: returns the
    input-space Fisher information sum_r [sum_c p_rc g_rc g_rc^T - gbar_r gbar_r^T], g_rc = -P_c (x_r - mu_c)."""
    n, d = x.shape
    Nm = torch.cholesky_inverse(chol(torch.add(eye(d, 0.5), Bm, alpha=0.5)))
    xa = torch.cat([x, const(("ones", n), lambda: torch.ones(n, 1, device="cuda"))], 1)
    S = (xa.T @ (y1h[:, :, None] * x[:, None, :]).view(n, 10 * d)).view(d + 1, 10, d)
    rc = 1.0 / cnt
    mu = S[d] * rc[:, None]
    sig = torch.baddbmm(S[:d].transpose(0, 1) * rc[:, None, None], mu[:, :, None], mu[:, None, :], beta=P["SHR"], alpha=-P["SHR"])
    L = chol(sig.add_(Nm, alpha=1 - P["SHR"]))
    Pc = torch.cholesky_inverse(L)
    R = P["QROWS"]
    Dl = x[:R][None] - mu[:, None]
    J = torch.bmm(Dl, Pc)  # = -g
    lg = (Dl * J).sum(2).add_(L.diagonal(dim1=1, dim2=2).log().sum(1)[:, None], alpha=2.0).mul_(-0.5 * P["QT"])
    p = torch.softmax(lg.T, 1)
    G = J.transpose(0, 1)  # (R, 10, d) view
    K = torch.bmm(p[:, None, :], G).squeeze(1)
    return torch.addmm((J * p.T[:, :, None]).view(-1, d).T @ J.view(-1, d), K.T, K, alpha=-1)


def features(xx, n, T):
    """rows projected to the sphere of the mean train radius, plus a log-radius coordinate; the mean train squared
    norm is kept. One spare zero row at the end (padding vertex for HKNN)."""
    A = xx @ T
    Q = A.norm(dim=1, keepdim=True)
    Pm = Q[:n].square().mean()
    lq = (Q.clamp_min(1e-6) / Pm.sqrt()).log_()
    s = (1 + 0.25 * lq[:n].square().mean()).rsqrt()  # mean train |row|^2 = Pm (1 + mean lq^2 / 4)
    out = torch.empty(len(A) + 1, A.shape[1] + 1, device=A.device)
    out[-1] = 0
    torch.mul(A, (Pm.sqrt() * s) / Q.clamp_min(1e-6), out=out[:-1, :-1])
    torch.mul(lq, 0.5 * Pm.sqrt() * s, out=out[:-1, -1:])
    return out


def class_slots(lab, valid, cap, ncls=10):
    """table (ncls, cap) of row indices grouped by class (rows in order; unused slots = len(lab))"""
    n = len(lab)
    oh = F.one_hot(lab, ncls).T.float()
    if valid is not None:
        oh = oh * valid
    rank = (oh.cumsum(1) * oh).sum(0)
    slot = torch.where((rank > 0) & (rank <= cap), lab * cap + rank.long() - 1, ncls * cap)
    tbl = torch.full((ncls * cap + 1,), n, dtype=torch.long, device=lab.device)
    tbl.scatter_(0, slot, const(("ar", n), lambda: torch.arange(n, device="cuda")))
    return tbl[:ncls * cap]


def compress(z, zhl, lab, P, hook):
    """class-restricted k-means (KPC per class, LLOYD steps, seeded with each class's first rows); fp16 assignment
    scores; centroid sums from one fp16 one-hot GEMM against [z_hi, z_lo, 1] (exact counts, ~fp32 sums)"""
    n, d = z.shape
    kpc = P["KPC"]
    m = 10 * kpc
    C = z[class_slots(lab, None, kpc)]
    zh = zhl[:, :d]
    own = lab[:, None, None].expand(n, 1, kpc)
    ar = const(("ar", m), lambda: torch.arange(m, device="cuda"))
    base = lab * kpc

    def scores(C):
        return torch.addmm(C.square().sum(1), zh, C.half().T, beta=-0.5, out_dtype=F32).view(n, 10, kpc)

    hist = [C]
    for _ in range(P["LLOYD"]):
        a = scores(C).gather(1, own).squeeze(1).argmax(1).add_(base)
        s = torch.mm((a[:, None] == ar).half().T, zhl, out_dtype=F32)
        cnt = s[:, 2 * d:]
        C = torch.where(cnt > 0, (s[:, :d] + s[:, d:2 * d]) / cnt.clamp_min(1), C)
        hist.append(C)
        if len(hist) == 2:
            hook(C)  # side-stream work on the centres after one Lloyd step (preconditioner landmarks)
    S = scores(C)
    jo, ja = S.gather(1, own).squeeze(1).max(1)
    oth = S.masked_fill(F.one_hot(lab, 10).bool()[:, :, None], float("-inf")).amax((1, 2))
    marg = (jo - oth) / (z.square().sum(1) - jo - oth).clamp_min(1e-6)
    hard = marg.argsort()[:P["NHARD"]]
    oh = (ja.add_(base)[:, None] == ar).half().index_fill_(0, hard, 0.0)
    s = torch.mm(oh.T, zhl, out_dtype=F32)
    cnt = s[:, 2 * d]
    Q = torch.where(cnt[:, None] > 0, (s[:, :d] + s[:, d:2 * d]) / cnt.clamp_min(1)[:, None], C)
    pts = torch.cat([z[hard], Q])
    plab = torch.cat([lab[hard], const(("cl", kpc), lambda: torch.arange(10, device="cuda").repeat_interleave(kpc))])
    w = torch.cat([const(("one", P["NHARD"]), lambda: torch.ones(P["NHARD"], device="cuda")), cnt])
    return pts, plab, w, hist


def rbf32(a, b, g, bn=None):
    """exp(-g |a - b|^2) by one fp32 GEMM (few kernels; the operands here are small)"""
    an = a.square().sum(1, keepdim=True)
    bn = an.T if bn is None else bn
    return torch.addmm(bn, a, b.T, beta=-g, alpha=2 * g).sub_(an, alpha=g).exp_()


def kaug(a, g, la, left):
    """one side of the fp16 exponent GEMM: la - g|a|^2 as a hi/lo pair; the cross term 2g a.b from fp16 coordinates"""
    al = torch.addcmul(la, a.square().sum(1), const(("mg", g), lambda: torch.tensor(-g, device="cuda")))
    ah = al.half()
    n, d = a.shape
    out = torch.zeros(n, -(-(d + 4) // 8) * 8, device=a.device, dtype=F16)
    if left:
        out[:, :d] = a
        out[:, d:d + 2] = 1
        out[:, d + 2] = ah
        torch.sub(al, ah.float(), out=out[:, d + 3])
    else:
        torch.mul(a, 2 * g, out=out[:, :d])
        out[:, d] = ah
        torch.sub(al, ah.float(), out=out[:, d + 1])
        out[:, d + 2:d + 4] = 1
    return out


def flash_scores(zq, zk, val, g, vsc):
    """K(zq, zk) @ val (RBF exp(-g|.|^2)) by the fp16 flash-attention kernel with the log-sum-exp trick (our d2/d3
    entries): q = [s, 1, 1], k = [s, hi, lo] with hi + lo = -|s|^2/2, s = sqrt(2g) z rounded once"""
    nq, dim = zq.shape
    nk = len(zk)
    hd = -(-(dim + 2) // 8) * 8
    m = val.shape[1]
    q = torch.zeros(nq, hd, device=zq.device, dtype=F16)
    torch.mul(zq, math.sqrt(2 * g), out=q[:, :dim])
    q[:, dim:dim + 2] = 1
    qn = q[:, :dim].float().square().sum(1).mul_(0.5)
    k = torch.zeros(nk, hd, device=zq.device, dtype=F16)
    torch.mul(zk, math.sqrt(2 * g), out=k[:, :dim])
    kn = k[:, :dim].float().square().sum(1)
    torch.mul(kn, -0.5, out=k[:, dim])
    torch.addcmul(k[:, dim].float(), kn, const(("h", 0), lambda: torch.tensor(0.5, device="cuda")), out=k[:, dim + 1]).neg_()
    sv = val.abs().amax(0, keepdim=True).clamp_min_(1e-30)
    v = torch.zeros(nk, hd, device=zq.device, dtype=F16)
    torch.mul(val, vsc / sv, out=v[:, :m])
    o, lse = torch.ops.aten._scaled_dot_product_flash_attention(q[None, None], k[None, None], v[None, None], scale=1.0)[:2]
    return o[0, 0, :, :m] * (lse[0, 0] - qn).exp_().mul_(1 / vsc)[:, None] * sv


def fitc_a(lm, g, jit):
    """inverse landmark factor L^-1, L L^T = Kmm + jit I (k-means centres after one Lloyd step)"""
    Kmm = rbf32(lm, lm, g)
    Kmm.diagonal().add_(jit)
    L = chol(Kmm)
    return torch.linalg.solve_triangular(L, eye(len(lm)), upper=False)


def fitc_k(pts, w, sw, lm, g, ridge, jit):
    """FITC-style preconditioner for A = W^1/2 K W^1/2 + ridge I (#125 idea, stable form), main-stream part:
    Kw = W^1/2 Knm and 1/D, D = ridge + w (1 - max_j Kij^2) (#125's cheap stand-in for the Nystrom residual)"""
    Kc = rbf32(pts, lm, g, lm.square().sum(1))
    dinv = torch.addcmul(const(("r", ridge), lambda: torch.tensor(ridge, device="cuda")), w,
                         (1 - Kc.amax(1).square_() / (1 + jit)).clamp_min_(0)).reciprocal_()
    return Kc.mul_(sw[:, None]), dinv


def fitc_b(Kw, dinv, Li):
    """side-stream part: Phi = Kw L^-T, A ~ D + Phi Phi^T; Woodbury with S = I + Phi^T D^-1 Phi = R R^T (>= I,
    fp32): A^-1 ~ D^-1 - Q Q^T, Q^T = R^-1 (D^-1 Phi)^T (one triangular solve, no explicit S^-1). Returns Q^T."""
    PhiT = Li @ Kw.T
    PdT = PhiT * dinv
    return torch.linalg.solve_triangular(chol(torch.addmm(eye(len(Li)), PhiT, PdT.T)), PdT, upper=False)


def pcg_t(mvt, prect, t, steps):
    """PCG with vectors stored transposed (classes, n) and scaled directions d = p / rho (our d3 flash):
    g = d.(A d), r -= (A d) / g, d' = d + z' / rho', x = sum_k d_k / g_k; then the next direction with the last
    step length (#125): x += d_S (rho_S / rho_{S-1}) / g_{S-1}."""
    m, n = t.shape
    D = t.new_empty(steps + 1, m, n)
    G = t.new_empty(steps + 1, m, 1)
    r = t.clone()
    z = prect(r)
    rho = torch.bmm(r[:, None, :], z[:, :, None])[:, 0]
    d = torch.div(z, rho, out=D[0])
    for k in range(steps):
        ap = mvt(d)
        g = torch.bmm(d[:, None, :], ap[:, :, None], out=G[k][:, :, None])[:, 0]
        r.addcdiv_(ap, g, value=-1)
        z = prect(r)
        rn = torch.bmm(r[:, None, :], z[:, :, None])[:, 0]
        d = torch.addcdiv(d, z, rn, out=D[k + 1])
        if k == steps - 1:
            torch.mul(g, rho / rn, out=G[steps])
        rho = rn
    return (D / G).sum(0)


def run(train_x, train_y, test_x, P):
    g = P["GAMMA"]
    with torch.no_grad():
        lab = train_y.long()
        n, dim = train_x.shape
        raw = torch.cat([train_x, test_x]).float()
        xx = raw * raw.norm(dim=1, keepdim=True).clamp_min_(1e-6).pow_(-P["NPOW"]).mul_(dim ** (P["NPOW"] / 2))
        x = xx[:n]
        y1h = F.one_hot(lab, 10).float()
        cnt = y1h.sum(0)
        Bm = lda(raw[:n], y1h, cnt, P["LDA_S"])
        Fm = qda_fisher(x, y1h, cnt, Bm, P)
        U = torch.add(eye(dim, 1 - P["UW0"] - P["UW1"]), pow34(Fm, P), alpha=P["UW0"]).add_(Bm, alpha=P["UW1"])
        za = features(xx, n, chol(U))
        zt = za[:n]
        zh = zt.half()
        zhl = torch.cat([zh, (zt - zh.float()).half(), const(("ones16", n), lambda: torch.ones(n, 1, device="cuda", dtype=F16))], 1)
        main = torch.cuda.current_stream()
        side = _stream("side")
        fa = {}

        def hook(lm):  # the landmark factor on a side stream while the k-means finishes
            side.wait_stream(main)
            with torch.cuda.stream(side):
                fa["L"] = fitc_a(lm, g, P["JIT"])
        pts, plab, w, hist = compress(zt, zhl, lab, P, hook)
        sw = w.sqrt()
        lw = w.clamp_min(1e-30).log_().mul_(0.5)
        Kw, dinv = fitc_k(pts, w, sw, hist[1], g, P["RIDGE"], P["JIT"])
        side.wait_stream(main)
        with torch.cuda.stream(side):  # the rest of the preconditioner while the main stream builds K
            QT = fitc_b(Kw, dinv, fa["L"])
        Km = torch.mm(kaug(pts, g, lw, True), kaug(pts, g, lw, False).T, out_dtype=F32)
        Km = torch.exp(Km, out=torch.empty_like(Km, dtype=F16))
        Km.diagonal().add_(P["RIDGE"])
        main.wait_stream(side)
        dr = dinv[None]

        def prect(r):
            return torch.addmm(r * dr, r @ QT.T, QT, alpha=-1)

        def mvt(d):
            return torch.mm(d.half(), Km, out_dtype=F32)
        t = (plab == const(("c10", 0), lambda: torch.arange(10, device="cuda")[:, None])) * sw
        alpha = pcg_t(mvt, prect, t, P["STEPS"]).mul_(sw)
        return flash_scores(za[n:2 * n], pts, alpha.T, g, P["VSC"]).argmax(1)


def kc_d2(train_x, train_y, test_x):
    """Entry point. The first call for a shape runs run() eagerly twice, captures it as one CUDA graph and replays
    it WARM times (no torch.compile); later calls copy their inputs in and replay it (a full refit per call)."""
    key = (tuple(train_x.shape), tuple(test_x.shape), train_x.dtype, train_y.dtype, str(train_x.device))
    if key not in _GRAPHS:
        static = [train_x.clone(), train_y.clone(), test_x.clone()]
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                run(*static, P)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = run(*static, P)
        _GRAPHS[key] = (graph, static, out)
        gc.collect()
        gc.freeze()  # keep warm-up objects out of every later collection (wall clock)
        for _ in range(WARM):  # untimed: sustained clocks before the first timed call
            graph.replay()
        torch.cuda.synchronize()
    graph, static, out = _GRAPHS[key]
    static[0].copy_(train_x)
    static[1].copy_(train_y)
    static[2].copy_(test_x)
    graph.replay()
    return out.clone()


if __name__ == "__main__":
    import mnist
    print(mnist.score(kc_d2, difficulty=2), "ms per call")
