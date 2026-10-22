"""MNIST difficulty 3: kernel ridge on a compressed training set in a closed-form learned metric, then HKNN
on the low-margin test rows. No backprop; plain PyTorch (cuBLAS, cuSOLVER, ATen, PyTorch's flash-attention
kernel); the whole call is one CUDA graph.

Credits: our RFM/HKNN d3 learner (rfm-hknn-d3; partial row normalisation, learned-metric RBF ridge with Nystrom
PCG, HKNN local hulls of Vincent & Bengio 2002) and @islamborghini's kernel-PCG, rebuilt around ideas of
@sigkillme0's #125 d3 entry, re-implemented here in plain PyTorch: the LDA + QDA Fisher-information metric with a
rank-48 subspace and a natural-gradient step, the log-radius feature, k-means compression into hard rows plus
weighted centroids, the FITC-style preconditioner, the extrapolated last PCG step, per-row HKNN temperature.

Per call: (1) rows scaled toward norm sqrt(60) (power 0.6). (2) Metric: LDA (between-class term x3); class
Gaussians shrunk 0.3 toward (I + LDA)^-1/2; their tempered-softmax Fisher information F at 2048 rows;
M = 0.8 F^0.75 + 0.196 LDA + 0.004 I; rank-48 factor; one natural-gradient step (QDA cross-entropy); rows ->
sphere of the mean radius + a log-radius coordinate.
(3) Class-restricted k-means (24 per class, 1 Lloyd step); the 4096 rows of lowest margin kept, the rest replaced
by count-weighted centroids; (W^1/2 K W^1/2 + 0.1 I) b = W^1/2 Y on the 4336 points (RBF gamma 0.02) by 5 PCG
steps + an extrapolated direction; preconditioner: Nystrom on the centres plus a diagonal, applied by Woodbury.
(4) Test scores; up to 512 rows with margin < 0.2 get HKNN hull distances for their top 3 classes (70 nearest
train or confident test vertices per class), added as probabilities x 0.15 e^3. test_x is used without labels.

    python mnist.py kd3_k.py:kc_d3 --difficulty 3
"""
import gc
import math

import torch
import torch.nn.functional as F

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
F32, F64, F16 = torch.float32, torch.float64, torch.float16

P = {  # see the docstring
    "NPOW": 0.6, "LDA_S": 3.0, "SHR": 0.7, "QT": 0.15, "QROWS": 2048, "UW0": 0.8, "UW1": 0.196, "RANKD": 48,
    "NAT": 0.2, "NATD": 0.1, "NS_ITERS": 4, "BLEND": 0.02,
    "KPC": 24, "LLOYD": 1, "NHARD": 4096, "GAMMA": 0.02, "RIDGE": 0.1, "JIT": 1e-4, "STEPS": 5,
    "CONF": 0.2, "NU": 512, "HK_K": 70, "HK_LAM": 3.0, "HK_T": 2.0, "HK_W": 0.15, "HK_C": 3, "CAP": 2048,
    "RSHIFT": 3.0, "VSC": 256.0,
}
WARM = 64  # untimed replays ending the warm-up call (A100 clocks)
_GRAPHS = {}
_ST = {}
_C = {}


def const(key, fn):
    """input-independent tensors, made once outside the captured graph"""
    if key not in _C:
        _C[key] = fn()
    return _C[key]


def eye(n, s=1.0, dt=F32):
    return const(("eye", n, s, dt), lambda: torch.eye(n, device="cuda", dtype=dt) * s)


def _stream(k):
    """a high-priority side stream (latency-bound factorisations)"""
    if k not in _ST:
        _ST[k] = torch.cuda.Stream(priority=-1)
    return _ST[k]


def chol(a):
    return torch.linalg.cholesky_ex(a)[0]


def cholb(a):
    """small factor via the batched path (a batch of two with I; single potrf has a high fixed cost)"""
    return chol(torch.stack([a, eye(len(a), dt=a.dtype)]))[0]


def sqrtm(a, iters):
    """coupled Newton-Schulz square root (sign iteration on [[0, A/c], [I, 0]])"""
    n = len(a)
    c = a.norm()
    x = torch.cat([torch.cat([const(("z", n), lambda: torch.zeros(n, n, device="cuda")), a / c], 1),
                   const(("iz", n), lambda: torch.cat([torch.eye(n, device="cuda"), torch.zeros(n, n, device="cuda")], 1))], 0)
    i15 = eye(2 * n, 1.5)
    for _ in range(iters):
        x = x @ torch.addmm(i15, x, x, alpha=-0.5)
    return x[:n, n:] * c.sqrt()


def pow34(m, P):
    """trace-normalised ~M^0.75 with an identity blend"""
    d = len(m)
    sq = sqrtm(m, P["NS_ITERS"])
    m = sq @ sqrtm(sq, P["NS_ITERS"])
    m = m + m.T
    return torch.addcmul(eye(d, P["BLEND"]), m, (1 - P["BLEND"]) * d / m.trace())


def lda(raw, y1h, cnt, s):
    """B = S_w^-1 with the between-class (Woodbury) term stretched by s, trace d (whitened rows: S_w = I - A A^T,
    A = class means * sqrt(class frequency)), and Nm = (I/2 + B/2)^-1, also by Woodbury (10x10 factors only)"""
    n, d = raw.shape
    A = (y1h.T @ raw).mul_((cnt * n).rsqrt()[:, None]).T
    C = 1 - 1 / n + 1e-3
    AA = A.T @ A
    H = torch.add(eye(10), AA, alpha=-1 / C)
    B = torch.addmm(eye(d, 1 / C), A, torch.cholesky_solve(A.T, cholb(H)), alpha=s / C ** 2)
    k = d / B.trace()
    b = 0.5 + 0.5 * k / C  # I/2 + B/2 = b I + (k s / 2C^2) A H^-1 A^T
    K10 = AA + H * (b / (0.5 * k * s / C ** 2))
    Nm = torch.addmm(eye(d), A, torch.cholesky_solve(A.T, cholb(K10)), alpha=-1).div_(b)
    return B.mul_(k), Nm


def moments(x, y1h):
    """per-class sums of x x^T and of x (one GEMM)"""
    n, d = x.shape
    xa = torch.cat([x, const(("ones", n), lambda: torch.ones(n, 1, device="cuda"))], 1)
    return (xa.T @ (y1h[:, :, None] * x[:, None, :]).view(n, 10 * d)).view(d + 1, 10, d)


def qda_fisher(x, y1h, cnt, Nm, S, P):
    """shrunk class Gaussians -> Fisher information of their tempered softmax at QROWS rows; CE gradient"""
    n, d = x.shape
    rc = 1.0 / cnt
    mu = S[d] * rc[:, None]
    sig = torch.baddbmm(S[:d].transpose(0, 1) * rc[:, None, None], mu[:, :, None], mu[:, None, :], beta=P["SHR"], alpha=-P["SHR"])
    L = chol(sig.add_(Nm, alpha=1 - P["SHR"]))
    Li = torch.linalg.solve_triangular(L, eye(d).expand(10, d, d), upper=False)
    Pc = Li.transpose(1, 2) @ Li
    R = P["QROWS"]
    Dl = x[:R][None] - mu[:, None]
    J = torch.bmm(Dl, Pc)  # = -g
    lg = (Dl * J).sum(2).add_(L.diagonal(dim1=1, dim2=2).log().sum(1)[:, None], alpha=2.0).mul_(-0.5 * P["QT"])
    p = torch.softmax(lg.T, 1)
    G = J.transpose(0, 1)  # (R, 10, d) view
    K = torch.bmm(p[:, None, :], G).squeeze(1)
    diff = torch.bmm((y1h[:R] - p)[:, None, :], G).squeeze(1)
    Fm = torch.addmm((J * p.T[:, :, None]).view(-1, d).T @ J.view(-1, d), K.T, K, alpha=-1)
    return Fm, diff


def subspace(U, x, k):
    """U^-1-orthonormal basis of span(U^3 x[:k]^T) (~ the top-k eigenspace of U)"""
    A = U.double()
    C2 = A @ (A @ x[:k].T.double())
    Fm = A @ C2
    B = C2.T @ Fm
    B = B + B.T
    B.diagonal().add_(1e-12 * B.trace())
    return torch.linalg.solve_triangular(cholb(B), Fm.T, upper=False).T.float().mul_(math.sqrt(2.0))


def nat_pre(xr, diff, Fm, damp):
    """natstep's V-free parts: E = x_R^T diff, damped Fisher inverse"""
    d = len(Fm)
    return xr.T @ diff, torch.cholesky_inverse(chol(torch.addcmul(Fm, eye(d), Fm.trace() * (damp / d))))


def natstep(B, E, Fi, V, step):
    """one natural-gradient step on V (QDA cross-entropy gradient), scale preserved"""
    K = Fi @ V
    L0 = E @ K
    N = B @ V
    L1 = N @ (V.T @ K)
    Ld = torch.addcmul(L0, (N * L0).sum() / (N * L1).sum(), L1, value=-1)
    Nn = torch.addcmul(V, (V.square().sum() / Ld.square().sum()).sqrt(), Ld, value=-step)
    return Nn.mul_(((V * N).sum() / (Nn * (B @ Nn)).sum()).sqrt())


def features(xx, n, T):
    """rows on the sphere of the mean train radius + a log-radius coordinate; a zero row appended"""
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
    """(ncls, cap) table of row indices grouped by class (unused slots = len(lab))"""
    n = len(lab)
    oh = F.one_hot(lab, ncls).T.float()
    if valid is not None:
        oh = oh * valid
    rank = (oh.cumsum(1) * oh).sum(0)
    slot = torch.where((rank > 0) & (rank <= cap), lab * cap + rank.long() - 1, ncls * cap)
    tbl = torch.full((ncls * cap + 1,), n, dtype=torch.long, device=lab.device)
    tbl.scatter_(0, slot, const(("ar", n), lambda: torch.arange(n, device="cuda")))
    return tbl[:ncls * cap]


def compress(z, zhl, lab, P, hook, seeds):
    """class-restricted k-means, then the hard rows + count-weighted centroids of the rest"""
    n, d = z.shape
    kpc = P["KPC"]
    m = 10 * kpc
    C = z[seeds]
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
        cnt = s[:, 2 * d:2 * d + 1]
        C = torch.where(cnt > 0, (s[:, :d] + s[:, d:2 * d]) / cnt.clamp_min(1), C)
        hist.append(C)
        if len(hist) == 2:
            hook(C)  # preconditioner landmarks
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
    """exp(-g |a - b|^2) by one fp32 GEMM"""
    an = a.square().sum(1, keepdim=True)
    bn = an.T if bn is None else bn
    return torch.addmm(bn, a, b.T, beta=-g, alpha=2 * g).sub_(an, alpha=g).exp_()


def kaug(a, g, la, left):
    """one side of the fp16 exponent GEMM: la - g|a|^2 as a hi/lo pair, cross term 2g a.b in fp16"""
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
    """K(zq, zk) @ val by the flash-attention kernel: q = [s, 1, 1], k = [s, hi, lo], hi + lo = -|s|^2/2"""
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


def fitc_k(pts, w, sw, lm, g, ridge, jit):
    """FITC-style preconditioner, main-stream part: Kw = W^1/2 Knm, 1/D, D = ridge + w (1 - max_j Kij^2)"""
    Kc = rbf32(pts, lm, g, lm.square().sum(1))
    dinv = torch.addcmul(const(("r", ridge), lambda: torch.tensor(ridge, device="cuda")), w,
                         (1 - Kc.amax(1).square_() / (1 + jit)).clamp_min_(0)).reciprocal_()
    return Kc.mul_(sw[:, None]), dinv


def fitc_b(Kw, dinv, Kmm):
    """side stream: A ~ D + Kw Kmm^-1 Kw^T; by Woodbury A^-1 ~ D^-1 - Q Q^T, Q^T = R^-1 (D^-1 Kw)^T,
    R R^T = Kmm + Kw^T D^-1 Kw"""
    DK = Kw * dinv[:, None]
    return torch.linalg.solve_triangular(chol(torch.addmm(Kmm, Kw.T, DK)), DK.T, upper=False)


def pcg_t(mvt, prect, t, steps):
    """PCG, vectors transposed (classes, n), directions d = p / rho; then the next direction with the last
    step length (#125)"""
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


def hknn(zq, zv, vaug, yv, valid, cls, P):
    """HKNN: squared distance from each query to the ridge-regularised affine hull of the HK_K nearest valid
    vertices of each class in cls"""
    nq, C = cls.shape
    nv = len(yv)
    dim = zv.shape[1]
    K, lam, cap = P["HK_K"], P["HK_LAM"], P["CAP"]
    tbl = class_slots(yv, valid, cap)
    qa = kaug(zq, 1.0, zq.new_zeros(()).expand(nq), True)
    D = torch.mm(qa, vaug[tbl].T).view(nq, 10, cap)  # fp16; padding vertex at distance^2 > 60000
    sv, si = D.gather(1, cls[:, :, None].expand(nq, C, cap)).topk(K, 2, largest=False, sorted=False)
    nb = tbl.view(10, cap)[cls].gather(2, si)
    nbz = zv.index_select(0, nb.view(-1)).view(nq * C, K, dim)
    m = nbz.mean(1, keepdim=True)
    vh = (nbz - m).mul_(1024.0).half()
    r = (zq[:, None, :] - m.view(nq, C, dim)).view(nq * C, dim, 1)
    # min_a |r - V^T a|^2 + lam |a|^2 = lam r^T G^-1 r (G = V^T V + lam I) = lam (c - p^2), p the last pivot of
    # the Cholesky factor of [[G, r], [r^T, c]] (c > r^T G^-1 r); lam drops out of the readout
    gb = torch.empty(nq * C, dim + 1, dim + 1, device=zq.device)
    torch.bmm(vh.transpose(1, 2), vh, out_dtype=F32, out=gb[:, :dim, :dim])
    gb[:, :dim, :dim].mul_(2.0 ** -20).add_(eye(dim, lam))
    gb[:, dim:, :dim] = r.transpose(1, 2)
    c = r.square().sum(1, keepdim=True).mul_(1.0 / lam).add_(1.0)
    gb[:, dim:, dim:] = c
    return (c[:, 0, 0] - chol(gb)[:, dim, dim].square()).view(nq, C)  # reads the lower triangle only


def hk_probs(d, t):
    """class probabilities from hull distances; temperature t x each row's nearest distance (#125)"""
    mn = d.min(1, keepdim=True).values
    return torch.softmax((mn - d) / (t * mn).clamp_min(1e-12), 1)


def run(train_x, train_y, test_x, P):
    g = P["GAMMA"]
    with torch.no_grad():
        lab = train_y.long()
        n, dim = train_x.shape
        raw = torch.cat([train_x, test_x]).float()
        xx = raw * raw.norm(dim=1, keepdim=True).clamp_min_(1e-6).pow_(-P["NPOW"]).mul_(dim ** (P["NPOW"] / 2))
        x = xx[:n]
        main = torch.cuda.current_stream()
        side = _stream("side")
        y1h = F.one_hot(lab, 10).float()
        cnt = y1h.sum(0)
        side.wait_stream(main)
        with torch.cuda.stream(side):  # label-only work in parallel with LDA
            B = x.T @ x
            S = moments(x, y1h)
            seeds = class_slots(lab, None, P["KPC"])
        Bm, Nm = lda(raw[:n], y1h, cnt, P["LDA_S"])
        main.wait_stream(side)
        Fm, diff = qda_fisher(x, y1h, cnt, Nm, S, P)
        side.wait_stream(main)
        with torch.cuda.stream(side):
            E, Fi = nat_pre(x[:P["QROWS"]], diff, Fm, P["NATD"])
        U = torch.add(eye(dim, 1 - P["UW0"] - P["UW1"]), pow34(Fm, P), alpha=P["UW0"]).add_(Bm, alpha=P["UW1"])
        T = subspace(U, x, P["RANKD"])
        main.wait_stream(side)
        T = natstep(B, E, Fi, T, P["NAT"])
        za = features(xx, n, T)  # (2n + 1, 49): train, test, padding
        zt = za[:n]
        zh = zt.half()
        zhl = torch.cat([zh, (zt - zh.float()).half(), const(("pad1", n), lambda: torch.eye(1, 6, device="cuda", dtype=F16).expand(n, 6))], 1)
        fa = {}

        def hook(lm):  # landmark kernel (side stream)
            side.wait_stream(main)
            with torch.cuda.stream(side):
                fa["K"] = rbf32(lm, lm, g)
                fa["K"].diagonal().add_(P["JIT"])
        pts, plab, w, hist = compress(zt, zhl, lab, P, hook, seeds)
        lm = hist[1]
        sw = w.sqrt()
        lw = w.clamp_min(1e-30).log_().mul_(0.5)
        Kw, dinv = fitc_k(pts, w, sw, lm, g, P["RIDGE"], P["JIT"])
        side.wait_stream(main)
        with torch.cuda.stream(side):
            QT = fitc_b(Kw, dinv, fa["K"])
        ka = kaug(pts, g, lw, True)
        Km = torch.mm(ka, kaug(pts, g, lw, False).T, out_dtype=F32)
        Km = torch.exp(Km, out=torch.empty_like(Km, dtype=F16))
        Km.diagonal().add_(P["RIDGE"])
        vaug = kaug(za, 1.0, za.new_zeros(()).expand(len(za)), False).mul_(-1.0)  # HKNN vertex operand
        vaug[-1:, za.shape[1]:za.shape[1] + 1] = 60000.0  # padding vertex: never a neighbour
        main.wait_stream(side)
        dr = dinv[None]

        QTh = QT.half()

        def prect(r):  # fp16 products, fp32 accumulation
            t = torch.mm(r.half(), QTh.T, out_dtype=F32).half()
            return torch.mm(t, QTh, out_dtype=F32).mul_(-1).addcmul_(r, dr)

        def mvt(d):
            return torch.mm(d.half(), Km, out_dtype=F32)
        t = (plab == const(("c10", 0), lambda: torch.arange(10, device="cuda")[:, None])) * sw
        alpha = pcg_t(mvt, prect, t, P["STEPS"]).mul_(sw)
        s = flash_scores(za[n:2 * n], pts, alpha.T, g, P["VSC"])
        m1, labp = s.max(1)
        marg = m1 - s.scatter(1, labp[:, None], float("-inf")).amax(1)
        low = marg < P["CONF"]  # only these can change; the first NU (spare slots: row 0)
        rk = low.cumsum(0)
        uidx = torch.zeros(P["NU"] + 1, dtype=torch.long, device=s.device).scatter_(
            0, torch.where(low & (rk <= P["NU"]), rk - 1, P["NU"]), const(("ar", n), lambda: torch.arange(n, device="cuda")))[:P["NU"]]
        su = s[uidx].mul_(math.exp(-P["RSHIFT"]))  # HKNN weight: HK_W e^RSHIFT score units
        cls = su.topk(P["HK_C"], 1).indices
        yall = torch.cat([lab, labp])
        valid = torch.cat([const(("t", n), lambda: torch.ones(n, dtype=torch.bool, device="cuda")), marg >= P["CONF"]])
        pr = hk_probs(hknn(za[n:2 * n][uidx], za, vaug, yall, valid, cls, P), P["HK_T"])
        su.scatter_add_(1, cls, pr.mul_(P["HK_W"]))
        res = labp.clone()
        res[uidx] = torch.where(valid[n:][uidx], labp[uidx], su.argmax(1))
        return res


def kc_d3(train_x, train_y, test_x):
    """First call per shape: run() eagerly twice, capture it as one CUDA graph, replay it WARM times. Later
    calls copy their inputs in and replay it (a full refit per call)."""
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
        gc.freeze()  # warm-up objects out of later collections
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
    print(mnist.score(kc_d3, difficulty=3), "ms per call")
