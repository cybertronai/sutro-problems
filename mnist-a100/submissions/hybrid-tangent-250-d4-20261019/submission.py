"""MNIST difficulty 4: two RFM kernel teachers distilled into a small MLP ensemble with local tangent jitter.

Credits: teacher = our d3 flash entry (RBF kernel ridge in an AGOP metric, Nystrom-PCG, HKNN of Vincent & Bengio)
on @islamborghini's kernel-pcg; cyclic row orders and a third AGOP update as in @islamborghini's rfm-ensemble d4.
Soft teacher targets and AGOP^0.5 student inputs: our ladder-kernel d4. Students: RMSNorm ReLU MLPs, local tangent
jitter and cross-teaching from @likelytobelaura's merged d5 entries; zero-init head as in modded-nanogpt.
@sigkillme0's neural d4 PR first combined kernel-metric features, MLP ensembles and neighbour jitter at d4.
Plain PyTorch; no hand-written kernels.

Per call, from scratch:
 1. Teacher: two RFM models on cyclic shifts of the training rows (AGOP metric fits on 3000, 6000, 10000
    rows; kernel ridge on all; HKNN on the 512 lowest-margin test rows); softmax(scores / 0.3) averaged.
 2. Student inputs: member 0's AGOP^0.5 coordinates; per row, the rms-scaled top-4 tangents of the offsets to its
    10 nearest rows of the (true/teacher) class.
 3. Students: 3 MLPs 60-256-256-10 (RMSNorm, ReLU, no hidden bias, zero-init head), 250 Adam steps on 1024
    labelled + 1024 test rows (teacher targets, weight 0.3); rows move 1.2 N(0, 1) along each tangent, + N(0, 0.2^2).
 4. Cross-teaching: after 80 and 170 steps the test targets become 0.3 teacher + 0.7 students.
 5. Prediction: 0.75 students + 0.25 teacher, then HKNN (weight 1, margin 0.3) in member 0's metric.
test_x is used without labels. One CUDA graph is captured in the warm-up call; each call retrains from scratch.
"""
import gc
import math

import torch
import torch.nn.functional as F

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

GAMMA, RIDGE, RANK, BLEND, NPOW = 0.02, 0.1, 256, 0.02, 0.6
FITS, SAMPLES, B_STEPS = ((3000, 12), (6000, 12), (10000, 12)), 512, 13  # metric fits: rows, CG steps
CONF, NU, TEMP, TMEMB = 0.2, 512, 0.3, 2
HK_K, HK_LAM, HK_T, HK_W, HK_C, HK_M = 70, 8.0, 2.0, 3.0, 3, 2048
NS_ITERS, SHIFT, VSC, GRAMS = 7, 3.0, 256.0, 40
M, W, STEPS, BL, BU, LR, DECAY = 3, 256, 250, 1024, 1024, 0.015, 0.3  # students
B1, B2, TW, TJ, NOISE, KNB, NT, BETA, CHUNK = 0.9, 0.99, 0.3, 1.2, 0.2, 10, 4, 0.25, 10
XT, XW = (8, 17), 0.7  # cross-teaching chunks and student weight
DIN, DOUT = 64, 16
F32, BF = torch.float32, torch.bfloat16
_ST, _AUX, STATE = {}, {}, {}


def chol(a):
    return torch.linalg.cholesky_ex(a)[0]


def rownorm(x):
    return x * (x.shape[1] ** 0.5 / x.norm(dim=1, keepdim=True).clamp_min(1e-6)) ** NPOW


def aug(a, b, shift=0.0):
    sa = a.square().sum(1, keepdim=True)
    sb = b.square().sum(1, keepdim=True)
    A = torch.cat([a, sa, torch.ones_like(sa)], 1)
    B = torch.cat([(2 * GAMMA) * b, torch.full_like(sb, -GAMMA), shift - GAMMA * sb], 1).T.contiguous()
    return A, B


def kern(a, b):
    A, B = aug(a, b)
    return (A @ B).exp_()


def h3_ops(z, b=None):
    A, B = aug(z, z if b is None else b, SHIFT)
    Ah, Bh = A.half(), B.half()
    Al, Bl = (A - Ah.float()).half(), (B - Bh.float()).half()
    return torch.cat([Ah, Ah, Al], 1), torch.cat([Bh, Bl, Bh], 0)


def _flash(q, k, v):
    return torch.ops.aten._scaled_dot_product_flash_attention(q, k, v, scale=1.0)[:2]


class Flash:
    """c (K + RIDGE I) products on z[:nk] by flash attention; z[nk:] are test queries"""

    def __init__(self, z, nk, c):
        s = (z * math.sqrt(2 * GAMMA)).half()
        qn = s.float().square().sum(1) * 0.5
        nq, dim = s.shape
        hd = -(-(dim + 2) // 8) * 8
        q = s.new_zeros(nq, hd)
        q[:, :dim] = s
        q[:, dim:dim + 2] = 1
        k = s.new_zeros(nk, hd)
        k[:, :dim] = s[:nk]
        hi = (-qn[:nk]).half()
        k[:, dim] = hi
        k[:, dim + 1] = (hi.float() + qn[:nk]).neg_().half()
        self.k = k.view(1, 1, nk, hd)
        self.vb = s.new_zeros(nk, hd)
        self.v = self.vb.view(1, 1, nk, hd)
        self.q, self.qt = q[:nk][None, None], q[nk:][None, None]
        self.qc = qn - (SHIFT - math.log(VSC))
        self.nk, self.lam, self.el = nk, RIDGE * c, None

    def mv(self, d):
        m = d.shape[0]
        torch.mul(d.T, VSC, out=self.vb[:self.nk, :m])
        o, lse = _flash(self.q, self.k, self.v)
        if self.el is None:
            self.el = (lse[0, 0, :self.nk] - self.qc[:self.nk]).exp_().div_(self.lam)[None, :]
        return torch.addcmul(d, o[0, 0, :, :m].T, self.el, out=torch.empty_like(d))

    def last(self, w, d):
        """last CG step and test scores c K[nk:, :nk] (w + a d)"""
        n, m = self.nk, d.shape[0]
        sw = w.abs().amax(1, keepdim=True).clamp_min(1e-30)
        torch.mul(d.T, VSC, out=self.vb[:n, :m])
        torch.mul(w.T, (VSC / sw).T, out=self.vb[:n, m:2 * m])
        o, _ = _flash(self.q, self.k, self.v)
        ap = torch.addcmul(d, o[0, 0, :, :m].T, self.el, out=torch.empty_like(d))
        a = 1.0 / (d * ap).sum(1, keepdim=True).mul_(self.lam)
        o, lse = _flash(self.qt, self.k, self.v)
        t = o[0, 0, :, :2 * m] * (lse[0, 0] - self.qc[n:]).exp_()[:, None]
        return torch.addcmul(t[:, m:] * sw.T, t[:, :m], a.T)


def _stream(k):
    if k not in _ST:
        _ST[k] = torch.cuda.Stream()
    return _ST[k]


def tri_inv(L):
    # L^-1: diagonal blocks on 4 streams, merged by GEMMs
    m = L.shape[0]
    b = m // 4
    cur = torch.cuda.current_stream()
    eye = torch.eye(b, device=L.device)
    out = L.new_zeros(m, m)
    sts = []
    for p in range(4):
        st = _stream(p)
        st.wait_stream(cur)
        with torch.cuda.stream(st):
            i = slice(p * b, (p + 1) * b)
            out[i, i] = torch.linalg.solve_triangular(L[i, i], eye, upper=False)
        sts.append(st)
    for p in (0, 2):
        st = sts[p]
        st.wait_stream(sts[p + 1])
        with torch.cuda.stream(st):
            i, j = slice(p * b, (p + 1) * b), slice((p + 1) * b, (p + 2) * b)
            out[j, i] = (out[j, j] @ (L[j, i] @ out[i, i])).neg_()
    cur.wait_stream(sts[0])
    cur.wait_stream(sts[2])
    h = 2 * b
    out[h:, :h] = (out[h:, h:] @ (L[h:, :h] @ out[:h, :h])).neg_()
    return out


def split2n(a, s):
    k = a.shape[1]
    x = a.new_empty(a.shape[0], 2 * k, dtype=torch.half)
    torch.mul(a, s, out=x[:, :k])
    torch.add(x[:, :k], a, alpha=-s, out=x[:, k:])
    return x


def mm_hln(x, b, sb):
    k = b.shape[0]
    bs = b * sb
    bh = bs.half()
    bl = (bs - bh.float()).half()
    o = torch.mm(x, torch.cat([bh, -bh]), out_dtype=F32)
    return torch.addmm(o, x[:, :k], bl, out_dtype=F32, out=o)


def precond(z, lam, jit, c):
    """Nystrom preconditioner (I - v v^T); returns (2^29 v, 2^-29 v^T)"""
    cur, st = torch.cuda.current_stream(), _stream("w")
    st.wait_stream(cur)
    with torch.cuda.stream(st):
        lm = kern(z[:RANK], z[:RANK]) * c
        lm.diagonal().add_(jit)
        lit = tri_inv(chol(lm)).T
    xu = split2n(kern(z, z[:RANK]) * c, 2.0 ** 11)
    cur.wait_stream(st)
    u = mm_hln(xu, lit, 2.0 ** 11)
    xu = split2n(u, 2.0 ** -9)
    n = len(z)
    h, lo = xu[:, :RANK], xu[:, RANK:]
    hs = h.view(GRAMS, n // GRAMS, RANK)
    small = torch.bmm(hs.mT, hs, out_dtype=F32).sum(0)
    hl = torch.mm(h.T, lo, out_dtype=F32)
    small.sub_(hl).sub_(hl.T)
    small.diagonal().add_(lam * 2.0 ** 26)
    v = mm_hln(xu, tri_inv(chol(small)).T, 2.0 ** 29)
    return v, torch.mul(v.T, 2.0 ** -58, out=v.new_empty(v.shape[1], v.shape[0]))


def pcg(mv, sc, t, steps, v, vt, last=None):
    """Nystrom-PCG for (K + lam I) w = t, a few steps (early stopping)"""
    n, m = t.shape
    D = t.new_empty(steps, m, n)
    G = t.new_empty(steps, m, 1)
    r = t.T.contiguous()
    z = torch.addmm(r, r @ v, vt, alpha=-1)
    d = torch.div(z, (r * z).sum(1, keepdim=True), out=D[0])
    for k in range(steps):
        if last is not None and k == steps - 1:
            return last((D[:k] / G[:k]).sum(0).div_(sc), d)
        ap = mv(d)
        torch.sum(d * ap, 1, keepdim=True, out=G[k])
        if k == steps - 1:
            break
        r.addcdiv_(ap, G[k], value=-1)
        z = torch.addmm(r, r @ v, vt, alpha=-1)
        d = torch.addcdiv(d, z, (r * z).sum(1, keepdim=True), out=D[k + 1])
    return (D / G).sum(0).div_(sc).T


def sqrtm(a):
    c = a.norm()
    n = len(a)
    x = torch.zeros(2 * n, 2 * n, device=a.device)
    x[:n, n:] = a / c
    x[n:, :n] = torch.eye(n, device=a.device)
    i15 = 1.5 * torch.eye(2 * n, device=a.device)
    for _ in range(NS_ITERS):
        x = x @ torch.addmm(i15, x, x, alpha=-0.5)
    return x[:n, n:] * c.sqrt()


def agop(transform, z, w, fac):
    """AGOP M of the fit: factor of blend(M^0.75); keeps blend(M^0.5)'s (student inputs)"""
    s = fac.shape[0]
    wp = torch.cat([w, (w[:, :, None] * z[:, None, :]).reshape(len(z), -1)], 1)
    sc = wp.abs().amax(0, keepdim=True).clamp_min(1e-30)
    r = torch.mm(fac, (wp / sc).half(), out_dtype=F32) * sc
    c = w.shape[1]
    g = r[:, c:].reshape(s, c, -1) - r[:, :c, None] * z[:s, None, :]
    dim = transform.shape[0]
    g = (g @ transform.T).reshape(-1, dim)
    sq = sqrtm(g.T @ g)

    def fin(m):
        m = (m + m.T) / 2
        m = m * (dim / m.trace().clamp_min(1e-12))
        return chol((1 - BLEND) * m + BLEND * torch.eye(dim, device=z.device))
    _AUX["half"] = fin(sq)
    return fin(sq @ sqrtm(sq))


def hknn(zq, zv, yv, valid, cls):
    """squared distances to ridge-regularised affine hulls of HK_K nearest vertices per class"""
    nq, C = cls.shape
    dim = zq.shape[1]
    qa = torch.cat([zq, zq.square().sum(1, keepdim=True), torch.ones_like(zq[:, :1])], 1)
    va = torch.cat([-2 * zv, torch.ones_like(zv[:, :1]), zv.square().sum(1, keepdim=True)], 1)
    D = (qa @ va.T).half().clamp_min_(0)
    cd, ci = D.topk(HK_M, 1, largest=False)
    ok = valid[ci][:, None, :] & (yv[ci][:, None, :] == cls[:, :, None])
    sv, si = cd[:, None, :].masked_fill(~ok, float("inf")).topk(HK_K, 2, largest=False)
    nb = ci.gather(1, si.reshape(nq, -1)).reshape(nq, C, HK_K)
    bad = torch.isinf(sv)
    nb = torch.where(bad, nb[:, :, :1], nb)
    nbz = zv[nb].reshape(nq * C, HK_K, dim)
    m = nbz.mean(1, keepdim=True)
    v = nbz - m
    r = (zq[:, None, :].expand(nq, C, dim).reshape(nq * C, 1, dim) - m).mT
    eye = torch.eye(dim, device=zq.device) * HK_LAM
    vs = v * 1024.0
    vh = vs.half()
    vl = (vs - vh.float()).half()
    g = torch.bmm(vh.mT, vh, out_dtype=F32)
    cc = torch.bmm(vh.mT, vl, out_dtype=F32)
    g.add_(cc).add_(cc.mT)
    lc = torch.linalg.cholesky_ex(torch.add(eye[None], g, alpha=2.0 ** -20))[0]
    out = (torch.cholesky_solve(r, lc).square().sum((1, 2)) * HK_LAM ** 2).reshape(nq, C)
    return out.masked_fill(bad[:, :, 0], 1e30)


def member(x, q, y, ty):
    """one RFM kernel model: HKNN-corrected test scores"""
    c = math.exp(SHIFT)
    lam, jit, diag = RIDGE * c, 1e-4 * c, c * (1.0 + RIDGE)
    n = len(x)
    transform = torch.eye(x.shape[1], device=x.device)
    main, side = torch.cuda.current_stream(), _stream("side")
    for m, st in FITS:
        z = x[:m] @ transform
        side.wait_stream(main)
        with torch.cuda.stream(side):
            v, vt = precond(z, lam, jit, c)
        if m > 3000:
            op = Flash(z, m, c)
            fac = torch.mm(*h3_ops(z[:SAMPLES], z)).exp_()
            fac.diagonal().fill_(c)
            main.wait_stream(side)
            w = pcg(op.mv, op.lam, y[:m], st, v, vt)
        else:
            A3, B3 = h3_ops(z)
            kz = torch.mm(A3, B3).exp_()
            kz.diagonal().fill_(diag)
            main.wait_stream(side)
            w = pcg(lambda d: torch.mm(d.half(), kz.T, out_dtype=F32), 1.0, y[:m], st, v, vt)
            fac = kz[:SAMPLES].clone()
            fac.diagonal().fill_(c)
            del kz
        transform = agop(transform, z, w, fac)
    za = torch.cat([x, q]) @ transform
    side.wait_stream(main)
    with torch.cuda.stream(side):
        v, vt = precond(za[:n], lam, jit, c)
    op = Flash(za, n, c)
    main.wait_stream(side)
    _AUX["za"] = za
    return hk_fix(pcg(op.mv, op.lam, y, B_STEPS, v, vt, op.last), za, ty, CONF, HK_W)


def hk_fix(s, za, ty, conf, w):
    """add w x HKNN probabilities to the NU lowest-margin rows with margin < conf"""
    n = len(ty)
    top = s.topk(2, 1).values
    marg = top[:, 0] - top[:, 1]
    keep = marg >= conf
    uidx = marg.topk(NU, largest=False).indices
    cls = s[uidx].topk(HK_C, 1).indices
    valid = torch.cat([torch.ones(n, dtype=torch.bool, device=s.device), keep])
    d = hknn(za[n:][uidx], za, torch.cat([ty, s.argmax(1)]), valid, cls)
    dn = d - d.min(1, keepdim=True).values
    pr = torch.softmax(-dn / (HK_T * d.min(1).values.mean().clamp_min(1e-12)), 1)
    add = torch.zeros_like(s[uidx]).scatter_(1, cls, pr) * (w * (~keep[uidx]).float())[:, None]
    return s.index_add(0, uidx, add)


def neighbours(Z, lab, chunk=4000):
    """KNB nearest same-label rows: fp16 GEMM of |z_i - z_j|^2 + 1e4 [label_i != label_j]"""
    sq = Z.square().sum(1, keepdim=True)
    oh = F.one_hot(lab, 10).float()
    A = torch.cat([Z, sq, torch.ones_like(sq), 1e4 * oh], 1).half()
    B = torch.cat([-2 * Z, torch.ones_like(sq), sq, 1 - oh], 1).half()
    out = []
    for i in range(0, len(Z), chunk):
        d = torch.mm(A[i:i + chunk], B.T)
        d.diagonal(i).fill_(float("inf"))
        out.append(d.topk(KNB, 1, largest=False).indices)
    return torch.cat(out)


def tangents(Z, nb):
    """per row: top-NT tangents of the offsets to its KNB neighbours (subspace iteration), rms-scaled"""
    O = Z[nb] - Z[:, None]
    G = O @ O.mT
    G = G / G.diagonal(dim1=1, dim2=2).sum(1).clamp_min(1e-12)[:, None, None]
    Q = G[:, :, :NT]
    for _ in range(4):
        Q = G @ Q
        L = chol(Q.mT @ Q + 1e-6 * torch.eye(NT, device=Z.device))
        Q = torch.linalg.solve_triangular(L, Q.mT, upper=False).mT
    return (Q.mT @ O) / math.sqrt(KNB)


def teacher(tx, ty, vx):
    with torch.no_grad():
        p = 0
        for i in range(TMEMB):
            o = i * (len(tx) // TMEMB)
            rx, ry = tx.roll(o, 0), ty.roll(o, 0).long()
            x, q = rownorm(rx.float()), rownorm(vx.float())
            s = member(x, q, F.one_hot(ry, 10).float(), ry)
            p = p + torch.softmax(s / TEMP, 1) / TMEMB
            if i == 0:
                feat, za = torch.cat([x, q]) @ _AUX["half"], _AUX["za"]
        feat = feat / feat[:len(tx)].square().mean().sqrt()
        nb = neighbours(feat, torch.cat([ty.long(), p.argmax(1)]))
        feat = F.pad(feat, (0, DIN - feat.shape[1]))
        return p, feat, tangents(feat, nb), za


def fwd_bwd(Ws, bo, x, tgt, wrow):
    """MLP (bf16 matmuls, fp32 RMSNorm, ReLU, output bias) and its hand-written backward"""
    L = len(Ws)
    acts, ys, rs = [x.to(BF)], [], []
    h = acts[0]
    for i in range(L):
        z = torch.bmm(h, Ws[i]).float()
        if i < L - 1:
            r = torch.rsqrt(z.square().mean(-1, keepdim=True) + 1e-6)
            ys.append(z * r)
            rs.append(r)
            h = F.relu(ys[-1]).to(BF)
            acts.append(h)
    z = z + bo.float()
    p = torch.softmax(z.masked_fill(torch.arange(DOUT, device=x.device) >= 10, -1e30), -1)
    g = (p - tgt) * wrow
    gW, gb = [None] * L, g.sum(1, keepdim=True)
    for i in range(L - 1, -1, -1):
        gd = g.to(BF)
        gW[i] = torch.bmm(acts[i].mT, gd).float()
        if i:
            gy = torch.bmm(gd, Ws[i].mT).float() * (ys[i - 1] > 0)
            y = ys[i - 1]
            g = rs[i - 1] * (gy - y * (gy * y).mean(-1, keepdim=True))
    return gW + [gb]


class Students:
    """M MLPs; weights, Adam moments and a bf16 copy in flat buffers"""

    def __init__(self, n, nq, dev):
        self.n, self.nq = n, nq
        dims = [DIN, W, W, DOUT]
        self.shapes = [(M, a, b) for a, b in zip(dims[:-1], dims[1:])] + [(M, 1, DOUT)]
        self.offs = [0]
        for s in self.shapes:
            self.offs.append(self.offs[-1] + math.prod(s))
        z = lambda *s, **k: torch.zeros(*s, device=dev, **k)
        self.flat, self.mflat, self.vflat = z(self.offs[-1]), z(self.offs[-1]), z(self.offs[-1])
        self.hflat = z(self.offs[-1], dtype=BF)
        self.Z, self.TG, self.TT, self.PT = z(n + nq, DIN), z(n + nq, 10), z(n + nq, NT, DIN), z(nq, 10)
        frac = torch.arange(STEPS, device=dev, dtype=F32) / STEPS
        self.lrs = LR * ((1 - frac) / DECAY).clamp(max=1.0)
        t = torch.arange(1, STEPS + 1, device=dev, dtype=F32)
        self.bc1, self.bc2 = 1 - B1 ** t, 1 - B2 ** t
        self.counter = z((), dtype=torch.long)
        self.fn = torch.compile(self._step, dynamic=False)

    def views(self, flat):
        return [flat[a:b].view(s) for a, b, s in zip(self.offs[:-1], self.offs[1:], self.shapes)]

    @torch.no_grad()
    def reset(self):
        for i, w in enumerate(self.views(self.flat)):
            if i < 2:
                w.copy_(torch.randn_like(w) / math.sqrt(60 if i == 0 else w.shape[1]))
            else:
                w.zero_()
        self.hflat.copy_(self.flat)
        self.mflat.zero_()
        self.vflat.zero_()
        self.counter.zero_()

    def _step(self, counter, Z, TG, TT, lrs, bc1, bc2, flat, mflat, vflat, hflat):
        at = counter.clamp_max(STEPS - 1).view(1)
        lr, c1, c2 = lrs.index_select(0, at), bc1.index_select(0, at), bc2.index_select(0, at)
        r = torch.rand(M, BL + BU, device=Z.device)
        il = (r[:, :BL] * self.n).long().clamp_max(self.n - 1)
        iu = self.n + (r[:, BL:] * self.nq).long().clamp_max(self.nq - 1)
        ia = torch.cat([il, iu], 1)
        x = Z[ia] + TJ * (torch.randn(M, BL + BU, NT, 1, device=Z.device) * TT[ia]).sum(2)
        x = x + NOISE * torch.randn_like(x)
        wrow = torch.cat([torch.full((BL,), 1.0 / BL, device=Z.device),
                          torch.full((BU,), TW / BU, device=Z.device)])[None, :, None]
        hv = self.views(hflat)
        g = torch.cat([t.reshape(-1) for t in fwd_bwd(hv[:3], hv[3], x, F.pad(TG[ia], (0, DOUT - 10)), wrow)])
        mflat.mul_(B1).add_(g, alpha=1 - B1)
        vflat.mul_(B2).addcmul_(g, g, value=1 - B2)
        flat.sub_(lr * (mflat / c1) / ((vflat / c2).sqrt() + 1e-8))
        hflat.copy_(flat)
        counter.add_(1)

    def step(self):
        self.fn(self.counter, self.Z, self.TG, self.TT, self.lrs, self.bc1, self.bc2, self.flat, self.mflat,
                self.vflat, self.hflat)

    @torch.no_grad()
    def predict(self):
        hv = self.views(self.hflat)
        h = self.Z[self.n:].expand(M, -1, -1).to(BF)
        for i in range(3):
            z = torch.bmm(h, hv[i]).float()
            if i < 2:
                h = F.relu(z * torch.rsqrt(z.square().mean(-1, keepdim=True) + 1e-6)).to(BF)
        return torch.softmax((z + hv[3].float())[..., :10], -1).mean(0)


def capture(fn):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = fn()
    return g, out


def hybrid_d4(train_x, train_y, test_x):
    key = (tuple(train_x.shape), tuple(test_x.shape), str(train_x.device))
    if key not in STATE:
        import torch._inductor.config as icfg
        icfg.compile_threads = 1
        static = [train_x.clone(), train_y.clone(), test_x.clone()]
        n = len(train_x)
        stu = Students(n, len(test_x), train_x.device)

        def run():
            ty = static[1].long()
            pt, feat, tt, za = teacher(*static)
            with torch.no_grad():
                stu.Z.copy_(feat)
                stu.TT.copy_(tt)
                stu.TG[:n].copy_(F.one_hot(ty, 10).float())
                stu.TG[n:].copy_(pt)
                stu.PT.copy_(pt)
                stu.reset()
            for c in range(STEPS // CHUNK):
                if c in XT:
                    with torch.no_grad():
                        stu.TG[n:].copy_((1 - XW) * stu.PT + XW * stu.predict())
                for _ in range(CHUNK):
                    stu.step()
            with torch.no_grad():
                return hk_fix((1 - BETA) * stu.predict() + BETA * pt, za, ty, 0.3, 1.0).argmax(1)
        STATE[key] = (static,) + capture(run)
        gc.collect()
        gc.freeze()
    static, g, out = STATE[key]
    for b, v in zip(static, (train_x, train_y, test_x)):
        b.copy_(v)
    g.replay()
    return out.clone()
