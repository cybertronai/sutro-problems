"""D4 Fisher teachers and tangent-jittered students, derived from spartanhaden PR138.
Credits: sigkillme0 PR125 Fisher/compression/PCG; spartanhaden RFM/HKNN and PR121/130/138;
islamborghini kernel-PCG; Vincent & Bengio HKNN; likelytobelaura tangent jitter;
modded-nanogpt zero head. sjbaebae: width128, one teacher, four-way FP32 split-K gradients
with BF16 gradient rounding and separate reduction/Adam. spartanhaden: class-wise neighbours,
without-replacement row walk, labelled-heavy batches, no isotropic noise or cross-teaching.
Every call trains from scratch.
"""
import gc
import math
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
NPOW, LDA_S, SHR, QT, QROWS, UW0, UW1, RANKD, NAT, NATD, NS_ITERS, BLEND = 0.6, 3, 0.7, 0.15, 2048, 0.8, 0.196, 48, 0.2, 0.1, 4, 0.02
KPC, NHARD, GAMMA, RIDGE, JIT, PSTEPS = 24, 4096, 0.02, 0.1, 1e-4, 5
FTM, TEMP, CONF, NU = 1, 0.25, 0.2, 512
HK_K, HK_LAM, HK_T, HK_W, HK_C, CAP = 70, 8.0, 1.5, 3.0, 3, 2560
M, W, STEPS, BL, BU, LR, DECAY = 3, 128, 220, 1280, 256, 0.024, 0.3
B1, B2, TW, TJ, KNB, NT, BETA, CHUNK = 0.8, 0.88, 0.3, 1.2, 10, 4, 0.25, 10
DIN, DOUT = 64, 16
NCAP, NBG = 2400, 20
F32, BF = torch.float32, torch.bfloat16
_ST, STATE = {}, {}
def chol(a):
 return torch.linalg.cholesky_ex(a)[0]
def eye(n, s=1.0):
 return torch.eye(n, device="cuda") * s
def _stream(k):
 if k not in _ST:
  _ST[k] = torch.cuda.Stream()
 return _ST[k]
def sqrtm(a):
 c = a.norm()
 n = len(a)
 x = torch.zeros(2 * n, 2 * n, device=a.device)
 x[:n, n:] = a / c
 x[n:, :n] = eye(n)
 for _ in range(NS_ITERS):
  x = x @ torch.addmm(eye(2 * n, 1.5), x, x, alpha=-0.5)
 return x[:n, n:] * c.sqrt()
def metric(m, lda):
 m = m + m.T
 d = len(m)
 m = torch.addcmul(eye(d, BLEND), m, (1 - BLEND) * d / m.trace())
 return torch.add(eye(d, 1 - UW0 - UW1), m, alpha=UW0).add_(lda, alpha=UW1)
def lda(raw, y1h, cnt):
 n, d = raw.shape
 A = (y1h.T @ raw).mul_((cnt * n).rsqrt()[:, None]).T
 C = 1 - 1 / n + 1e-3
 H = torch.addmm(eye(10), A.T, A, alpha=-1 / C)
 B = torch.addmm(eye(d, 1 / C), A, torch.cholesky_solve(A.T, chol(H)), alpha=LDA_S / C ** 2)
 return B.mul_(d / B.trace())
def qda_fisher(x, y1h, cnt, Bm):
 n, d = x.shape
 Nm = torch.cholesky_inverse(chol(torch.add(eye(d, 0.5), Bm, alpha=0.5)))
 xa = torch.cat([x, torch.ones_like(x[:, :1])], 1)
 S = (xa.T @ (y1h[:, :, None] * x[:, None, :]).view(n, 10 * d)).view(d + 1, 10, d)
 rc = 1.0 / cnt
 mu = S[d] * rc[:, None]
 sig = torch.baddbmm(S[:d].transpose(0, 1) * rc[:, None, None], mu[:, :, None], mu[:, None, :], beta=SHR, alpha=-SHR)
 L = chol(sig.add_(Nm, alpha=1 - SHR))
 Pc = torch.cholesky_inverse(L)
 Dl = x[:QROWS][None] - mu[:, None]
 J = torch.bmm(Dl, Pc)
 lg = (Dl * J).sum(2).add_(L.diagonal(dim1=1, dim2=2).log().sum(1)[:, None], alpha=2.0).mul_(-0.5 * QT)
 p = torch.softmax(lg.T, 1)
 G = J.transpose(0, 1)
 K = torch.bmm(p[:, None, :], G).squeeze(1)
 diff = torch.bmm((y1h[:QROWS] - p)[:, None, :], G).squeeze(1)
 return torch.addmm((J * p.T[:, :, None]).view(-1, d).T @ J.view(-1, d), K.T, K, alpha=-1), diff
def subspace(U, x):
 A = U.double()
 C2 = A @ (A @ x[:RANKD].T.double())
 Fm = A @ C2
 B = C2.T @ Fm
 B = B + B.T
 B.diagonal().add_(1e-12 * B.trace())
 return torch.linalg.solve_triangular(chol(B), Fm.T, upper=False).T.float().mul_(math.sqrt(2.0))
def natstep(B, E, Fi, V):
 K = Fi @ V
 L0 = E @ K
 N = B @ V
 L1 = N @ (V.T @ K)
 Ld = torch.addcmul(L0, (N * L0).sum() / (N * L1).sum(), L1, value=-1)
 Nn = torch.addcmul(V, (V.square().sum() / Ld.square().sum()).sqrt(), Ld, value=-NAT)
 return Nn.mul_(((V * N).sum() / (Nn * (B @ Nn)).sum()).sqrt())
def features(xx, n, T):
 A = xx @ T
 Q = A.norm(dim=1, keepdim=True).clamp_min(1e-6)
 Pm = Q[:n].square().mean()
 lq = (Q / Pm.sqrt()).log()
 s = Pm.sqrt() * (1 + 0.25 * lq[:n].square().mean()).rsqrt()
 return torch.cat([A * (s / Q), lq * (0.5 * s)], 1)
def compress(z, lab, side):
 n, d = z.shape
 m = 10 * KPC
 oh = F.one_hot(lab, 10).T.float()
 rank = (oh.cumsum(1) * oh).sum(0).long() - 1
 seed = torch.zeros(m + 1, dtype=torch.long, device=z.device)
 seed.scatter_(0, torch.where(rank < KPC, lab * KPC + rank, m), torch.arange(n, device=z.device))
 zh = z.half()
 zhl = torch.cat([zh, (z - zh.float()).half(), torch.ones_like(zh[:, :1])], 1)
 own = lab[:, None, None].expand(n, 1, KPC)
 ar = torch.arange(m, device=z.device)
 base = lab * KPC
 def scores(C):
  return torch.addmm(C.square().sum(1), zh, C.half().T, beta=-0.5, out_dtype=F32).view(n, 10, KPC)
 def centres(oh, C):
  s = torch.mm(oh.T, zhl, out_dtype=F32)
  cnt = s[:, 2 * d:]
  return torch.where(cnt > 0, (s[:, :d] + s[:, d:2 * d]) / cnt.clamp_min(1), C), cnt[:, 0]
 C = z[seed[:m]]
 a = scores(C).gather(1, own).squeeze(1).argmax(1).add_(base)
 C = centres((a[:, None] == ar).half(), C)[0]
 side.wait_stream(torch.cuda.current_stream())
 with torch.cuda.stream(side):
  Kmm = rbf(C, C)
  Kmm.diagonal().add_(JIT)
  Li = torch.linalg.solve_triangular(chol(Kmm), eye(m), upper=False)
 S = scores(C)
 jo, ja = S.gather(1, own).squeeze(1).max(1)
 oth = S.masked_fill(F.one_hot(lab, 10).bool()[:, :, None], float("-inf")).amax((1, 2))
 marg = (jo - oth) / (z.square().sum(1) - jo - oth).clamp_min(1e-6)
 hard = marg.argsort()[:NHARD]
 Q, cnt = centres((ja.add_(base)[:, None] == ar).half().index_fill_(0, hard, 0.0), C)
 w = torch.cat([torch.ones_like(hard, dtype=F32), cnt])
 return torch.cat([z[hard], Q]), torch.cat([lab[hard], ar // KPC]), w, C, Li
def aug(a, b, g, la=0.0, lb=0.0):
 A = torch.cat([a, torch.ones_like(a[:, :1]), la - g * a.square().sum(1, keepdim=True)], 1)
 B = torch.cat([2 * g * b, lb - g * b.square().sum(1, keepdim=True), torch.ones_like(b[:, :1])], 1)
 Ah, Bh = A.half(), B.half()
 return torch.mm(torch.cat([Ah, Ah, (A - Ah.float()).half()], 1), torch.cat([Bh, (B - Bh.float()).half(), Bh], 1).T,
     out_dtype=F32)
def rbf(a, b):
 return aug(a, b, GAMMA).exp_()
def flash_scores(zq, zk, val):
 q, k = (zq * math.sqrt(2 * GAMMA)).half(), (zk * math.sqrt(2 * GAMMA)).half()
 qn = q.float().square().sum(1) * 0.5
 kn = k.float().square().sum(1, keepdim=True) * -0.5
 sv = val.abs().amax(0, keepdim=True)
 hd = -(-(q.shape[1] + 2) // 8) * 8
 q, k, v = (F.pad(t, (0, hd - t.shape[1]))[None, None] for t in (
  torch.cat([q, torch.ones_like(q[:, :2])], 1), torch.cat([k, kn.half(), (kn - kn.half().float()).half()], 1),
  (val * (256 / sv)).half()))
 o, lse = torch.ops.aten._scaled_dot_product_flash_attention(q, k, v, scale=1.0)[:2]
 return o[0, 0, :, :10] * (lse[0, 0] - qn).exp_()[:, None] * (sv / 256)
def fitc(pts, w, sw, lm, Li):
 Kc = rbf(pts, lm)
 PhiT = Li @ (Kc * sw[:, None]).T
 dinv = (RIDGE + w * (1 - Kc.square().amax(1) / (1 + JIT)).clamp_min_(0)).reciprocal_()
 PdT = PhiT * dinv
 S = torch.addmm(eye(len(lm)), PhiT, PdT.T)
 return PdT.T, (torch.cholesky_inverse(chol(S + S.T)) * 2) @ PdT, dinv
def pcg_t(mvt, prect, t):
 D = t.new_empty(PSTEPS + 1, *t.shape)
 G = t.new_empty(PSTEPS + 1, len(t), 1)
 r = t.clone()
 z = prect(r)
 rho = torch.bmm(r[:, None, :], z[:, :, None])[:, 0]
 d = torch.div(z, rho, out=D[0])
 for k in range(PSTEPS):
  ap = mvt(d)
  g = torch.bmm(d[:, None, :], ap[:, :, None], out=G[k][:, :, None])[:, 0]
  r.addcdiv_(ap, g, value=-1)
  z = prect(r)
  rn = torch.bmm(r[:, None, :], z[:, :, None])[:, 0]
  d = torch.addcdiv(d, z, rn, out=D[k + 1])
  if k == PSTEPS - 1:
   torch.mul(g, rho / rn, out=G[PSTEPS])
  rho = rn
 return (D / G).sum(0)
def fit(train_x, lab, test_x, i):
 n, dim = train_x.shape
 raw = torch.cat([train_x, test_x]).float()
 xx = raw * raw.norm(dim=1, keepdim=True).clamp_min_(1e-6).pow_(-NPOW).mul_(dim ** (NPOW / 2))
 x = xx[:n]
 main, side = torch.cuda.current_stream(), _stream(i)
 side.wait_stream(main)
 with torch.cuda.stream(side):
  B = x.T @ x
 y1h = F.one_hot(lab, 10).float()
 cnt = y1h.sum(0)
 Bm = lda(raw[:n], y1h, cnt)
 Fm, diff = qda_fisher(x, y1h, cnt, Bm)
 side.wait_stream(main)
 with torch.cuda.stream(side):
  E = x[:QROWS].T @ diff
  Fi = torch.cholesky_inverse(chol(torch.addcmul(Fm, eye(dim), Fm.trace() * (NATD / dim))))
 sq = sqrtm(Fm)
 U = metric(sq @ sqrtm(sq), Bm)
 T = subspace(U, x)
 main.wait_stream(side)
 za = features(xx, n, natstep(B, E, Fi, T))
 pts, plab, w, lm, Li = compress(za[:n], lab, side)
 sw = w.sqrt()
 side.wait_stream(main)
 with torch.cuda.stream(side):
  Pd, PdS, dinv = fitc(pts, w, sw, lm, Li)
 lw = w.clamp_min(1e-30).log()[:, None] * 0.5
 Km = aug(pts, pts, GAMMA, lw, lw)
 Km = torch.exp(Km, out=torch.empty_like(Km, dtype=torch.half))
 Km.diagonal().add_(RIDGE)
 main.wait_stream(side)
 def prect(r):
  return torch.addmm(r * dinv, r @ Pd, PdS, alpha=-1)
 t = (plab == torch.arange(10, device=lab.device)[:, None]) * sw
 alpha = pcg_t(lambda d: torch.mm(d.half(), Km, out_dtype=F32), prect, t).mul_(sw)
 return flash_scores(za[n:], pts, alpha.T), za, xx @ chol(metric(sq, Bm))
def hknn(zq, zv, yv, valid, cls):
 nq, C = cls.shape
 dim = zq.shape[1]
 n = len(zv)
 oh = F.one_hot(yv, 10).T.float() * valid
 rank = (oh.cumsum(1) * oh).sum(0)
 slot = torch.where((rank > 0) & (rank <= CAP), yv * CAP + rank.long() - 1, 10 * CAP)
 tbl = torch.full((10 * CAP + 1,), n, device=zv.device).scatter_(0, slot, torch.arange(n, device=zv.device))[:-1]
 D = aug(zq, F.pad(zv, (0, 0, 0, 1), value=30.0)[tbl], 1.0).view(nq, 10, CAP)
 nb = tbl.view(10, CAP)[cls].gather(2, D.gather(1, cls[:, :, None].expand(nq, C, CAP)).topk(HK_K, 2).indices)
 nbz = zv[nb].reshape(nq * C, HK_K, dim)
 m = nbz.mean(1, keepdim=True)
 v = nbz - m
 r = (zq[:, None, :].expand(nq, C, dim).reshape(nq * C, 1, dim) - m).mT
 lc = chol(torch.baddbmm(eye(dim, HK_LAM)[None], v.mT, v))
 return (torch.cholesky_solve(r, lc).square().sum((1, 2)) * HK_LAM ** 2).reshape(nq, C)
def hk_fix(s, za, ty, conf, w):
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
_CF = {}
def cc(f):
 if f not in _CF:
  _CF[f] = torch.compile(f, dynamic=False)
 return _CF[f]
def blkmin(d):
 return d.view(len(d), -1, NBG).amin(2)
def blkget(d, bi):
 blk = (bi[:, :, None] * NBG + torch.arange(NBG, device=d.device)).flatten(1)
 return d.gather(1, blk), blk
def neighbours(Z, lab):
 N = len(Z)
 dev = Z.device
 oh = F.one_hot(lab, 10).T.float()
 rank = (oh.cumsum(1) * oh).sum(0)
 slot = torch.where(rank <= NCAP, lab * NCAP + rank.long() - 1, 10 * NCAP)
 tbl = torch.full((10 * NCAP + 1,), N, device=dev).scatter_(0, slot, torch.arange(N, device=dev))[:-1]
 Zp = F.pad(Z, (0, 0, 0, 1), value=30.0)[tbl]
 sq = Zp.square().sum(1, keepdim=True)
 A = torch.cat([Zp, sq, torch.ones_like(sq)], 1).half().view(10, NCAP, -1)
 B = torch.cat([-2 * Zp, torch.ones_like(sq), sq], 1).half().view(10, NCAP, -1)
 d = torch.bmm(A, B.mT)
 d.diagonal(dim1=1, dim2=2).fill_(float("inf"))
 d = d.view(10 * NCAP, NCAP)
 v, blk = cc(blkget)(d, cc(blkmin)(d).topk(KNB, 1, largest=False).indices)
 best = blk.gather(1, v.topk(KNB, 1, largest=False).indices)
 cls = (torch.arange(10 * NCAP, device=dev) // NCAP)[:, None] * NCAP
 out = torch.arange(N + 1, device=dev)[:, None].repeat(1, KNB)
 out[tbl] = tbl[cls + best]
 return out[:N]
def tangents(Z, nb):
 O = Z[nb] - Z[:, None]
 G = (O[:, :, None, :] * O[:, None, :, :]).sum(-1)
 G = G / G.diagonal(dim1=1, dim2=2).sum(1).clamp_min(1e-12)[:, None, None]
 Q = G[:, :, :NT]
 for _ in range(2):
  Q = (G[:, :, :, None] * Q[:, None, :, :]).sum(2)
  cols = []
  for j in range(NT):
   q = Q[:, :, j]
   for p in cols:
    q = q - (p * q).sum(1, keepdim=True) * p
   cols.append(q * torch.rsqrt(q.square().sum(1, keepdim=True) + 1e-6))
  Q = torch.stack(cols, 2)
 return (Q[:, :, :, None] * O[:, :, None, :]).sum(1) / math.sqrt(KNB)
def teacher(tx, ty, vx):
 with torch.no_grad():
  main, ps = torch.cuda.current_stream(), []
  for i in range(FTM):
   _stream(f"m{i}").wait_stream(main)
   with torch.cuda.stream(_stream(f"m{i}")):
    o = i * (len(tx) // FTM)
    ry = ty.roll(o, 0).long()
    s, za, feat = fit(tx.roll(o, 0), ry, vx, i)
    ps.append(torch.softmax(hk_fix(s, za, ry, CONF, HK_W) / TEMP, 1) / FTM)
    if i == 0:
     za0, feat0 = za, feat / feat[:len(tx)].square().mean().sqrt()
     fz = F.pad(feat0, (0, DIN - feat0.shape[1]))
     tt = cc(tangents)(fz, neighbours(feat0, torch.cat([ty.long(), ps[0].argmax(1)])))
  for i in range(FTM):
   main.wait_stream(_stream(f"m{i}"))
  return sum(ps), fz, tt, za0
@triton.jit
def _bf(x):
 return x.to(tl.bfloat16).to(tl.float32)
@triton.jit(do_not_specialize=['z'])
def k_rows(Z, TT, U, N, TG, HW, H0, H1, H2, G1, G2, G3, GB, n, nq, z, M: tl.constexpr, R: tl.constexpr,
  BL: tl.constexpr, W: tl.constexpr, TW: tl.constexpr, TJ: tl.constexpr, K: tl.constexpr = 64,
  E: tl.constexpr = 16):
 m = tl.program_id(1)
 o = m * R + tl.program_id(0) * 16 + tl.arange(0, 16)
 c, w, d = tl.arange(0, K), tl.arange(0, W), tl.arange(0, E)
 oc, ow, od = o[:, None] * W + c, o[:, None] * W + w, o[:, None] * E + d
 r = o % R
 lab = r < BL
 q = m * 4 + tl.where(lab, 0, 2)
 s = tl.where(lab, n, nq)
 a = (tl.load(U + q) * (n // 2)).to(tl.int32) * 2 + 1
 a = tl.where(a % 5 == 0, a + 2, a)
 i = (a * (z * tl.where(lab, BL, R - BL) + tl.where(lab, r, r - BL)) + (tl.load(U + q + 1) * s).to(tl.int32)) % s + tl.where(lab, 0, n)
 x = tl.load(Z + i[:, None] * K + c).to(tl.float32)
 for k in tl.range(4, loop_unroll_factor=1):
  x += TJ * tl.load(N + o * 4 + k)[:, None] * tl.load(TT + i[:, None] * 256 + k * K + c).to(tl.float32)
 h0 = x.to(tl.bfloat16)
 tl.store(H0 + o[:, None] * K + c, h0)
 W1 = HW + m * K * W + c[:, None] * W
 W2 = HW + M * K * W + m * W * W
 W3 = HW + M * (K + W) * W + m * W * E
 z1 = _bf(tl.dot(h0, tl.load(W1 + w)))
 r1 = 1 / tl.sqrt(tl.sum(z1 * z1, 1) / W + 1e-6)[:, None]
 a = tl.zeros_like(z1)
 for k in tl.range(W // K, loop_unroll_factor=1):
  h = tl.maximum(_bf(tl.dot(h0, tl.load(W1 + k * K + c))) * r1, 0).to(tl.bfloat16)
  tl.store(H1 + oc + k * K, h)
  a = tl.dot(h, tl.load(W2 + (k * K + c)[:, None] * W + w), a)
 z2 = _bf(a)
 r2 = 1 / tl.sqrt(tl.sum(z2 * z2, 1) / W + 1e-6)[:, None]
 y = z2 * r2
 h = tl.maximum(y, 0).to(tl.bfloat16)
 tl.store(H2 + ow, h)
 z = _bf(tl.dot(h, tl.load(W3 + w[:, None] * E + d))) + tl.load(HW + M * (80 + W) * W + m * E + d).to(tl.float32)
 z = tl.where(d < 10, z, -1e30)
 p = tl.exp(z - tl.max(z, 1)[:, None])
 g = (p / tl.sum(p, 1)[:, None] - tl.load(TG + i[:, None] * 10 + d, d < 10, 0)) * tl.where(lab, 1 / BL, TW / (R - BL))[:, None]
 tl.atomic_add(GB + m * E + d, tl.sum(g, 0))
 g = g.to(tl.bfloat16)
 tl.store(G3 + od, g)
 g = tl.where(z2 > 0, _bf(tl.dot(g, tl.load(W3 + d[:, None] + w * E))), 0)
 tl.store(G2 + ow, (r2 * (g - y * tl.sum(g * y, 1)[:, None] / W)).to(tl.bfloat16))
 tl.debug_barrier()
 a = tl.zeros_like(z1)
 for k in tl.range(W // K, loop_unroll_factor=1):
  a = tl.dot(tl.load(G2 + oc + k * K), tl.trans(tl.load(W2 + w[:, None] * W + k * K + c)), a)
 z1 = _bf(tl.dot(h0, tl.load(W1 + w)))
 y = z1 * r1
 g = tl.where(z1 > 0, _bf(a), 0)
 tl.store(G1 + ow, (r1 * (g - y * tl.sum(g * y, 1)[:, None] / W)).to(tl.bfloat16))
@triton.jit
def partial_tile(A,B,P,O,LA:tl.constexpr,LB:tl.constexpr,N:tl.constexpr,R:tl.constexpr,S:tl.constexpr,K:tl.constexpr,BASE:tl.constexpr):
 i,j=tl.arange(0,K),tl.arange(0,N)
 acc=tl.zeros((K,N),tl.float32)
 for k in range((tl.program_id(2)+BASE)*(R//S),(tl.program_id(2)+BASE+1)*(R//S),K):
  acc=tl.dot(tl.trans(tl.load(A+(k+i)[:,None]*LA+i)),tl.load(B+(k+i)[:,None]*LB+j),acc)
 tl.store(P+O+i[:,None]*LB+j,acc)
@triton.jit
def partials(H0,H1,H2,G1,G2,G3,P,M:tl.constexpr,R:tl.constexpr,W:tl.constexpr,S:tl.constexpr,F:tl.constexpr,K:tl.constexpr=64,E:tl.constexpr=16,BASE:tl.constexpr=0):
 t,m,C=tl.program_id(0),tl.program_id(1),W//K
 p=P+(tl.program_id(2)+BASE)*F
 i,j,q=t//C-1,t%C,t-C-C*C
 if t<C:
  partial_tile(H0+m*R*K,G1+m*R*W+t*K,p,m*K*W+t*K,K,W,K,R,S,K,BASE)
 elif t<C+C*C:
  partial_tile(H1+m*R*W+i*K,G2+m*R*W+j*K,p,M*K*W+m*W*W+i*K*W+j*K,W,W,K,R,S,K,BASE)
 else:
  partial_tile(H2+m*R*W+q*K,G3+m*R*E,p,M*(K+W)*W+m*W*E+q*K*E,W,E,E,R,S,K,BASE)
@triton.jit(do_not_specialize=['lr','c1','c2'])
def finish(P,GB,FL,MF,VF,HW,lr,c1,c2,F:tl.constexpr,BSTART:tl.constexpr,S:tl.constexpr,B1:tl.constexpr,B2:tl.constexpr,B:tl.constexpr=256):
 o=tl.program_id(0)*B+tl.arange(0,B)
 g=tl.zeros((B,),tl.float32)
 for s in tl.static_range(S):
  g+=tl.load(P+s*F+o,o<BSTART,0)
 g=g.to(tl.bfloat16).to(tl.float32)
 g=tl.where(o<BSTART,g,tl.load(GB+o-BSTART,(o>=BSTART)&(o<F),0))
 m=B1*tl.load(MF+o,o<F,0)+(1-B1)*g
 v=B2*tl.load(VF+o,o<F,0)+(1-B2)*g*g
 w=tl.load(FL+o,o<F,0)-lr*(m/c1)/(tl.sqrt(v/c2)+1e-8)
 tl.store(MF+o,m,o<F);tl.store(VF+o,v,o<F);tl.store(FL+o,w,o<F);tl.store(HW+o,w.to(tl.bfloat16),o<F)
 tl.store(GB+o-BSTART,0,(o>=BSTART)&(o<F))
class Students:
 def __init__(self, n, nq, dev):
  self.n, self.nq = n, nq
  dims = [DIN, W, W, DOUT]
  self.shapes = [(M, a, b) for a, b in zip(dims[:-1], dims[1:])] + [(M, 1, DOUT)]
  self.offs = [0]
  for s in self.shapes:
   self.offs.append(self.offs[-1] + math.prod(s))
  z = lambda *s, **k: torch.zeros(*s, device=dev, **k)
  self.flat, self.mflat, self.vflat = (z(self.offs[-1]) for _ in "fmv")
  self.hflat = z(self.offs[-1], dtype=BF)
  self.Z, self.TG, self.TT = z(n + nq, DIN, dtype=BF), z(n + nq, 10), z(n + nq, NT, DIN, dtype=BF)
  R = BL + BU
  self.U, self.N, self.GB = z(M, 4), z(CHUNK, M, R, NT), z(M, DOUT)
  self.ws = [z(M, R, k, dtype=BF) for k in (DIN, W, W, W, W, DOUT)]
  self.partial=torch.empty((4,self.flat.numel()),device=dev)
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
  self.GB.zero_()
  self.s = 0
 def chunk(self):
  R=BL+BU
  self.U.uniform_()
  self.N.normal_()
  for z in range(CHUNK):
   s=min(self.s,STEPS-1)
   k_rows[(R//16,M)](self.Z,self.TT,self.U,self.N[z],self.TG,self.hflat,*self.ws,self.GB,self.n,self.nq,z,M,R,BL,W,TW,TJ,num_warps=2,num_stages=1)
   partials[(W//64+(W//64)**2+W//64,M,4)](*self.ws,self.partial,M,R,W,4,self.flat.numel(),num_warps=4,num_stages=3)
   finish[(triton.cdiv(self.flat.numel(),256),)](self.partial,self.GB,self.flat,self.mflat,self.vflat,self.hflat,LR*min(1,(1-s/STEPS)/DECAY),1-B1**(s+1),1-B2**(s+1),self.flat.numel(),M*(80+W)*W,4,B1,B2,num_warps=4)
   self.s+=1
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
    stu.reset()
   for _ in range(STEPS // CHUNK):
    stu.chunk()
   with torch.no_grad():
    return hk_fix((1 - BETA) * stu.predict() + BETA * pt, za, ty, 0.3, 1.0).argmax(1)
  STATE[key] = (static,) + capture(run) + (stu,)
  gc.collect()
  gc.freeze()
 static, g, out, _ = STATE[key]
 for b, v in zip(static, (train_x, train_y, test_x)):
  b.copy_(v)
 g.replay()
 return out.clone()
