#!/usr/bin/env python3
"""Frozen spatial-computer v4 lowering of the 512-filter, 300-iteration CG pair.

Uses explicit FP32 scalar arithmetic, six Newton square-root steps, truncated
arcsine series and polynomial exponential. Both systems use Jacobi
preconditioning. Pixels are clamped to [0,1]. This numerical implementation
requires its own accuracy evaluation; it is not bitwise equivalent to PyTorch.
"""
import math, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'grid-mlp-scoring-20260912'))
from affine import ref as R, ins as I, loop as L, make_program
f32 = np.float32
def raw(v): return int(np.array(v, dtype=f32).view(np.uint32))
C = 10; D = 81; PAD = 121; NR = 6


def filters(F, seed=0):
    g = np.random.default_rng(seed); return (g.standard_normal((F, 9)) / 3).astype(f32), (g.standard_normal(F) * 0.1).astype(f32)


# scalar slots
(S_T, S_U, S_V, S_ACC, S_A, S_B, S_COND, S_Y, S_Q, S_S, S_S2, S_R, S_ALPHA, S_BETA, S_RZ, S_RZN, S_MX, S_SD, S_LAB, S_BEST) = range(20)
(K_ZERO, K_ONE, K_HALF, K_TWO, K_TINY, K_C0, K_C3, K_C5, K_C7, K_C9, K_PI2, K_THR, K_GAM, K_LAM, K_LAMK, K_INVN, K_INV9, K_INVC, K_EXPN, K_NEG1) = range(20)
K_CLASS = 20   # 10 class literals
NK = 30


def s(i): return R('s', i)
def k(i): return R('k', i)


def clamp01(x):
    return [I('cmp', s(S_COND), x, k(K_ZERO)),
            I('select', x, s(S_COND), k(K_ZERO), x),
            I('cmp', s(S_COND), k(K_ONE), x),
            I('select', x, s(S_COND), k(K_ONE), x)]


def nr_sqrt(dst, src):
    """dst <- sqrt(src) by NR: y = src + tiny; NR x (q = src / y; q = y + q; y = half q)."""
    y, q = s(S_Y), s(S_Q); b = [I('add', y, src, k(K_TINY))]
    for _ in range(NR): b += [I('div', q, src, y), I('add', q, y, q), I('mul', y, k(K_HALF), q)]
    return b + [I('copy', dst, y)]


def asin_series(dst, x):
    """dst <- x + x^3/6 + 3x^5/40 + 5x^7/112 + 35x^9/1152 (Horner in x^2), for |x| <= 0.71."""
    t, u = s(S_T), s(S_U)
    return [I('mul', u, x, x), I('mul', t, k(K_C9), u), I('add', t, t, k(K_C7)), I('mul', t, t, u), I('add', t, t, k(K_C5)), I('mul', t, t, u), I('add', t, t, k(K_C3)), I('mul', t, t, u), I('add', t, t, k(K_ONE)), I('mul', dst, t, x)]


def asin_sqrt_inplace(region, var):
    """region[var] <- arcsin(sqrt(region[var])): s = sqrt(x); if s <= 0.7: series(s) else pi/2 - series(sqrt(1 - s^2))."""
    x = R(region, **{var: 1}); ss, s2, a, b = s(S_S), s(S_S2), s(S_A), s(S_B)
    body = clamp01(x) + nr_sqrt(ss, x) + asin_series(a, ss) + [I('mul', s2, ss, ss), I('sub', s2, k(K_ONE), s2)] + nr_sqrt(s2, s2) + asin_series(b, s2) + \
           [I('sub', b, k(K_PI2), b), I('cmp', s(S_COND), k(K_THR), ss), I('select', x, s(S_COND), b, a)]
    return L(var, D, body)


def exp_neg(dst, src):
    """dst <- exp(-src) for src >= 0: t = -src / 2^8; e = 1 + t + t^2/2 + t^3/6 + t^4/24; square 8 times."""
    t, e, u = s(S_T), s(S_U), s(S_V)
    b = [I('mul', t, src, k(K_EXPN)), I('mul', u, t, k(K_INVC)), I('add', u, u, k(K_C3)), I('mul', u, u, t), I('add', u, u, k(K_HALF)), I('mul', u, u, t), I('add', u, u, k(K_ONE)), I('mul', u, u, t), I('add', e, u, k(K_ONE))]
    for _ in range(8): b.append(I('mul', e, e, e))
    return b + [I('copy', dst, e)]


def build(N, Q, F=512, T=300, seed=0, gamma=0.3, lam_ridge=1e-3, lam_k=1e-2):
    NF = 9 * F; W, Bf = filters(F, seed); NB = (NF + 95) // 96
    BLK = 64; assert NF % BLK == 0 and N % 16 == 0 and NF % 16 == 0
    regions = [('s', 20), ('k', NK), ('pad', PAD), ('w', 9 * F), ('bias', F), ('pj', C), ('acc', 16 * C), ('rowa', BLK), ('rowb', BLK), ('gb', BLK * BLK), ('ui', D),
               ('x', N * D), ('xq', Q * D), ('labels', N), ('mean', NF), ('diag', NF), ('diagk', N), ('p', NF * C), ('ap', NF * C), ('rr', NF * C), ('zz', NF * C), ('X', NF * C), ('B', NF * C),
               ('pk', N * C), ('apk', N * C), ('rk', N * C), ('zk', N * C), ('A', N * C), ('Y', N * C), ('sc1', Q * C), ('sc2', Q * C), ('out', Q),
               ('G', NF * NF), ('Phi', N * NF), ('PhiQ', Q * NF), ('K', N * N), ('KQ', Q * N)]
    body = []
    for name, words in regions:
        if name in ('k', 'w', 'bias', 'x', 'labels', 'pad'): continue   # xq, ui, pj, acc zeroed: first real writes come late   # ui is zeroed too: its first real write is late and would exhaust the initialization proof
        body.append(L('init', words, [I('set', R(name, init=1), 0)]))
    consts = [raw(0), raw(1), raw(.5), raw(2), raw(1e-6), raw(1 / 24), raw(1 / 6), raw(3 / 40), raw(5 / 112), raw(35 / 1152), raw(math.pi / 2), raw(0.7), raw(gamma), raw(lam_ridge), raw(lam_k), raw(1 / N), raw(1 / 9), raw(1 / 24), raw(-1 / 256), raw(-1)]
    consts += list(range(C)); assert len(consts) == NK
    for i, v in enumerate(consts): body.append(I('set', k(i), v))
    for f in range(F):
        for t in range(9): body.append(I('set', R('w', f * 9 + t), int(W[f, t].view(np.uint32))))
        body.append(I('set', R('bias', f), int(Bf[f].view(np.uint32))))
    body.append(L('pz', PAD, [I('set', R('pad', pz=1), raw(0))]))
    body.append(L('rx', N * D, [I('recv', R('x', rx=1))])); body.append(L('rl', N, [I('recv', R('labels', rl=1))]))
    # --- features of one image staged at x[off + n*D]: pad interior <- arcsin(sqrt(pixels)); feat[f*9 + cy*3 + cx] = mean_{iy,ix} relu(conv)
    def features(nvar, dst_region, xoff, src='x'):
        b = [L('cp', 9, [L('cq', 9, [I('copy', R('pad', 12, cp=11, cq=1), R(src, xoff, **{nvar: D}, cp=9, cq=1))])])]
        # arcsin(sqrt) on the interior of pad (rows 1..9, cols 1..9)
        b.append(L('ay', 9, [L('ax', 9, [*asin_sqrt_inplace_cell()])]))
        conv = [I('copy', s(S_ACC), R('bias', fl=1))]
        for t, (dy, dx) in enumerate([(a, c) for a in range(3) for c in range(3)]):
            conv += [I('mul', s(S_T), R('w', t, fl=9), R('pad', dy * 11 + dx, cy=33, iy=11, cx=3, ix=1)), I('add', s(S_ACC), s(S_ACC), s(S_T))]
        conv += [I('cmp', s(S_COND), s(S_ACC), k(K_ZERO)), I('select', s(S_T), s(S_COND), k(K_ZERO), s(S_ACC)), I('mul', s(S_T), s(S_T), k(K_INV9)),
                 I('add', R(dst_region, **{nvar: NF}, fl=9, cy=3, cx=1), R(dst_region, **{nvar: NF}, fl=9, cy=3, cx=1), s(S_T))]
        b.append(L('fl', F, [L('cy', 3, [L('cx', 3, [L('iy', 3, [L('ix', 3, conv)])])])]))
        return b
    def asin_sqrt_inplace_cell():
        x = R('pad', 12, ay=11, ax=1); ss, s2, a, bb = s(S_S), s(S_S2), s(S_A), s(S_B)
        return clamp01(x) + nr_sqrt(ss, x) + asin_series(a, ss) + [I('mul', s2, ss, ss), I('sub', s2, k(K_ONE), s2)] + nr_sqrt(s2, s2) + asin_series(bb, s2) + [I('sub', bb, k(K_PI2), bb), I('cmp', s(S_COND), k(K_THR), ss), I('select', x, s(S_COND), bb, a)]
    body.append(L('n1', N, features('n1', 'Phi', 0)))
    # pixels of the training rows are now arcsin(sqrt) only inside pad; redo the transform in place in x for the kernel (cheap: 81 per row)
    body.append(L('n2', N, [asin_sqrt_inplace_x('n2', 0)]))
    # --- feature mean, Y targets (+1/-1)
    body.append(L('mn', N, [L('mf', NF, [I('add', R('mean', mf=1), R('mean', mf=1), R('Phi', mn=NF, mf=1))])]))
    body.append(L('mm', NF, [I('mul', R('mean', mm=1), R('mean', mm=1), k(K_INVN))]))
    body.append(L('yn', N, [L('yc', C, [I('cmp', s(S_COND), R('labels', yn=1), R('k', K_CLASS, yc=1), predicate='eq'), I('select', R('Y', yn=C, yc=1), s(S_COND), k(K_ONE), k(K_NEG1))])]))
    # --- tiled Gram of the centred features: 96x96 resident blocks on diagonal bands t = bj - bi (48 affine loops over bi,
    #     one per band, so the scorer sees 48 leaves per operand instead of 1,176), B = Phi_c^T Y accumulated on band 0
    NB = NF // BLK
    for t in range(NB):
        rowb = [L('ra', BLK, [I('sub', R('rowa', ra=1), R('Phi', bd=BLK, gn=NF, ra=1), R('mean', bd=BLK, ra=1))]),
                L('rb', BLK, [I('sub', R('rowb', rb=1), R('Phi', t * BLK, bd=BLK, gn=NF, rb=1), R('mean', t * BLK, bd=BLK, rb=1))]),
                L('gi', BLK, [L('gj', BLK, [I('mul', s(S_T), R('rowa', gi=1), R('rowb', gj=1)), I('add', R('gb', gi=BLK, gj=1), R('gb', gi=BLK, gj=1), s(S_T))])])]
        if t == 0:
            rowb.append(L('bi', BLK, [L('bc', C, [I('mul', s(S_T), R('rowa', bi=1), R('Y', gn=C, bc=1)), I('add', R('B', bd=BLK * C, bi=C, bc=1), R('B', bd=BLK * C, bi=C, bc=1), s(S_T))])]))
        band = [L('gn', N, rowb),
                L('oi', BLK, [L('oj', BLK, [I('copy', R('G', t * BLK, bd=BLK * NF + BLK, oi=NF, oj=1), R('gb', oi=BLK, oj=1)), I('copy', R('G', t * BLK * NF, bd=BLK * NF + BLK, oi=1, oj=NF), R('gb', oi=BLK, oj=1))])]),
                L('zi', BLK * BLK, [I('set', R('gb', zi=1), raw(0))])]
        body.append(L('bd', NB - t, band))
    # ridge: G += lam * mean(diag) I
    body.append(I('set', s(S_ACC), raw(0))); body.append(L('td', NF, [I('add', s(S_ACC), s(S_ACC), R('G', td=NF + 1))]))
    body.append(I('mul', s(S_U), s(S_ACC), k(K_LAM)))
    body.append(I('set', s(S_V), raw(1.0 / NF))); body.append(I('mul', s(S_U), s(S_U), s(S_V)))
    body.append(L('rd', NF, [I('add', R('G', rd=NF + 1), R('G', rd=NF + 1), s(S_U)), I('div', R('diag', rd=1), k(K_ONE), R('G', rd=NF + 1))]))
    # --- CG on G X = B (X: NF x C), Jacobi: r = B (X = 0), z = diag r, p = z, rz = r.z
    body += cg_block('G', NF, 'X', 'B', 'p', 'ap', 'rr', 'zz', 'diag', T)
    # --- kernel matrix K[i,j] = exp(-gamma |u_i - u_j|^2), + lam on the diagonal
    body.append(L('ki', N, [L('cu', D, [I('copy', R('ui', cu=1), R('x', ki=D, cu=1))]),
                            L('kj', N, [I('set', s(S_ACC), raw(0)), L('kd', D, [I('sub', s(S_T), R('ui', kd=1), R('x', kj=D, kd=1)), I('mul', s(S_T), s(S_T), s(S_T)), I('add', s(S_ACC), s(S_ACC), s(S_T))]),
                                        I('mul', s(S_ACC), s(S_ACC), k(K_GAM)), *exp_neg(R('K', ki=N, kj=1), s(S_ACC))])]))
    body.append(L('kl', N, [I('add', R('K', kl=N + 1), R('K', kl=N + 1), k(K_LAMK))]))
    body.append(L('kdg', N, [I('div', R('diagk', kdg=1), k(K_ONE), R('K', kdg=N + 1))]))
    body += cg_block('K', N, 'A', 'Y', 'pk', 'apk', 'rk', 'zk', 'diagk', T)
    # --- queries: receive all, features -> PhiQ, arcsin pixels, KQ, scores
    body.append(L('rq', Q * D, [I('recv', R('xq', rq=1))]))
    body.append(L('q1', Q, features('q1', 'PhiQ', 0, src='xq')))
    body.append(L('q2', Q, [asin_sqrt_inplace_x('q2', 0, region='xq')]))
    body.append(L('s1', Q, [L('s1c', C, [I('set', s(S_ACC), raw(0)), L('s1f', NF, [I('sub', s(S_T), R('PhiQ', s1=NF, s1f=1), R('mean', s1f=1)), I('mul', s(S_T), s(S_T), R('X', s1c=1, s1f=C)), I('add', s(S_ACC), s(S_ACC), s(S_T))]), I('copy', R('sc1', s1=C, s1c=1), s(S_ACC))])]))
    body.append(L('qi', Q, [L('cu2', D, [I('copy', R('ui', cu2=1), R('xq', qi=D, cu2=1))]),
                            L('qj', N, [I('set', s(S_ACC), raw(0)), L('qd', D, [I('sub', s(S_T), R('ui', qd=1), R('x', qj=D, qd=1)), I('mul', s(S_T), s(S_T), s(S_T)), I('add', s(S_ACC), s(S_ACC), s(S_T))]),
                                        I('mul', s(S_ACC), s(S_ACC), k(K_GAM)), *exp_neg(R('KQ', qi=N, qj=1), s(S_ACC))])]))
    body.append(L('s2', Q, [L('s2c', C, [I('set', s(S_ACC), raw(0)), L('s2n', N, [I('mul', s(S_T), R('KQ', s2=N, s2n=1), R('A', s2c=1, s2n=C)), I('add', s(S_ACC), s(S_ACC), s(S_T))]), I('copy', R('sc2', s2=C, s2c=1), s(S_ACC))])]))
    # --- z-scores of each member's 10 scores, sum, first strict max
    qb = []
    for reg in ('sc1', 'sc2'):
        qb += [I('set', s(S_MX), raw(0)), L('zm', C, [I('add', s(S_MX), s(S_MX), R(reg, zq=C, zm=1))]), I('set', s(S_V), raw(0.1)), I('mul', s(S_MX), s(S_MX), s(S_V)), I('set', s(S_SD), raw(0)),
               L('zv', C, [I('sub', s(S_T), R(reg, zq=C, zv=1), s(S_MX)), I('mul', s(S_T), s(S_T), s(S_T)), I('add', s(S_SD), s(S_SD), s(S_T))]), I('set', s(S_V), raw(1 / 9)), I('mul', s(S_SD), s(S_SD), s(S_V)), *nr_sqrt(s(S_SD), s(S_SD)),
               L('zs', C, [I('sub', R(reg, zq=C, zs=1), R(reg, zq=C, zs=1), s(S_MX)), I('div', R(reg, zq=C, zs=1), R(reg, zq=C, zs=1), s(S_SD))])]
    qb += [L('zsum', C, [I('add', R('sc1', zq=C, zsum=1), R('sc1', zq=C, zsum=1), R('sc2', zq=C, zsum=1))]), I('copy', s(S_BEST), R('sc1', zq=C)), I('copy', s(S_LAB), k(K_CLASS))]
    for c in range(1, C): qb += [I('cmp', s(S_COND), s(S_BEST), R('sc1', c, zq=C)), I('select', s(S_BEST), s(S_COND), R('sc1', c, zq=C), s(S_BEST)), I('select', s(S_LAB), s(S_COND), k(K_CLASS + c), s(S_LAB))]
    qb.append(I('copy', R('out', zq=1), s(S_LAB)))
    body.append(L('zq', Q, qb)); body.append(L('so', Q, [I('send', R('out', so=1))]))
    return make_program(regions, body, {'learner': 'CG pair: arcsin(sqrt) pixels, random 3x3 conv (F filters, ReLU, 3x3 mean pool) ridge + RBF kernel ridge, Jacobi CG T iterations each, z-sum', 'F': F, 'T': T, 'gamma': gamma, 'lam_ridge': lam_ridge, 'lam_k': lam_k, 'N': N, 'Q': Q, 'grid_revision': 1, 'pixel_clamp': [0, 1], 'sqrt_newton_steps': NR, 'kernel_jacobi': True, 'numeric_scope': 'Ordered FP32 multiply/add; no FMA; approximate sqrt/asin/exp; independent grid accuracy required'})


def asin_sqrt_inplace_x(nvar, xoff, region='x'):
    x = R(region, xoff, **{nvar: D}, ax2=1); ss, s2, a, bb = s(S_S), s(S_S2), s(S_A), s(S_B)
    return L('ax2', D, clamp01(x) + nr_sqrt(ss, x) + asin_series(a, ss) + [I('mul', s2, ss, ss), I('sub', s2, k(K_ONE), s2)] + nr_sqrt(s2, s2) + asin_series(bb, s2) + [I('sub', bb, k(K_PI2), bb), I('cmp', s(S_COND), k(K_THR), ss), I('select', x, s(S_COND), bb, a)])


def cg_block(M, n, X, Bv, p, ap, rr, zz, diag, T):
    """Jacobi CG for M X = B with M (n x n) remote, X/B/p/ap/r/z of shape n x C stored row-major [i*C + c]."""
    b = [L('c0', n * C, [I('copy', R(rr, c0=1), R(Bv, c0=1))])]
    if diag: b.append(L('c1', n, [L('c1c', C, [I('mul', R(zz, c1=C, c1c=1), R(rr, c1=C, c1c=1), R(diag, c1=1))])]))
    else: b.append(L('c1', n * C, [I('copy', R(zz, c1=1), R(rr, c1=1))]))
    b.append(L('c2', n * C, [I('copy', R(p, c2=1), R(zz, c2=1))]))
    b.append(I('set', s(S_RZ), raw(0)))
    # per-class rz kept in a small region? use the 'k' trick: store the C values of rz/rzn in 'ap' row 0? -> simpler: keep rz as C scalars in region 'sc' -- we use dedicated scalars via loops over C in 'gb' scratch (first 2C words of gb are free after the Gram)
    RZ = lambda cc=None, **kw: R('gb', 0, **kw) if cc is None else R('gb', cc)
    RZN = lambda **kw: R('gb', C, **kw)
    ALPHA = lambda **kw: R('gb', 2 * C, **kw)
    b.append(L('rz0', C, [I('set', RZ(rz0=1), raw(0))]))
    b.append(L('rz1', n, [L('rz1c', C, [I('mul', s(S_T), R(rr, rz1=C, rz1c=1), R(zz, rz1=C, rz1c=1)), I('add', RZ(rz1c=1), RZ(rz1c=1), s(S_T))])]))
    it = [L('bl', n // 16, [L('za', 16 * C, [I('set', R('acc', za=1), raw(0))]),
                            L('mvj', n, [L('pc', C, [I('copy', R('pj', pc=1), R(p, mvj=C, pc=1))]),
                                         L('ib', 16, [I('copy', s(S_U), R(M, bl=16 * n, ib=n, mvj=1)),
                                                      L('mvc', C, [I('mul', s(S_T), s(S_U), R('pj', mvc=1)), I('add', R('acc', ib=C, mvc=1), R('acc', ib=C, mvc=1), s(S_T))])])]),
                            L('wa', 16 * C, [I('copy', R(ap, bl=16 * C, wa=1), R('acc', wa=1))])]),
          L('pa0', C, [I('set', ALPHA(pa0=1), raw(0))]),
          L('pa', n, [L('pac', C, [I('mul', s(S_T), R(p, pa=C, pac=1), R(ap, pa=C, pac=1)), I('add', ALPHA(pac=1), ALPHA(pac=1), s(S_T))])]),
          L('al', C, [I('div', ALPHA(al=1), RZ(al=1), ALPHA(al=1))]),
          L('up', n, [L('upc', C, [I('mul', s(S_T), ALPHA(upc=1), R(p, up=C, upc=1)), I('add', R(X, up=C, upc=1), R(X, up=C, upc=1), s(S_T)), I('mul', s(S_T), ALPHA(upc=1), R(ap, up=C, upc=1)), I('sub', R(rr, up=C, upc=1), R(rr, up=C, upc=1), s(S_T))])])]
    if diag: it.append(L('zj', n, [L('zjc', C, [I('mul', R(zz, zj=C, zjc=1), R(rr, zj=C, zjc=1), R(diag, zj=1))])]))
    else: it.append(L('zj', n * C, [I('copy', R(zz, zj=1), R(rr, zj=1))]))
    it += [L('rn0', C, [I('set', RZN(rn0=1), raw(0))]),
           L('rn', n, [L('rnc', C, [I('mul', s(S_T), R(rr, rn=C, rnc=1), R(zz, rn=C, rnc=1)), I('add', RZN(rnc=1), RZN(rnc=1), s(S_T))])]),
           L('be', C, [I('div', ALPHA(be=1), RZN(be=1), RZ(be=1)), I('copy', RZ(be=1), RZN(be=1))]),
           L('pu', n, [L('puc', C, [I('mul', s(S_T), ALPHA(puc=1), R(p, pu=C, puc=1)), I('add', R(p, pu=C, puc=1), R(zz, pu=C, puc=1), s(S_T))])])]
    b.append(L('cg', T, it))
    return b


def tape_words(train_pixels, train_labels, test_pixels):
    return np.concatenate((np.ascontiguousarray(train_pixels, dtype=f32).reshape(-1).view(np.uint32),
                           np.asarray(train_labels, dtype=np.uint32),
                           np.ascontiguousarray(test_pixels, dtype=f32).reshape(-1).view(np.uint32)))


if __name__ == '__main__':
    import argparse, json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n-train', type=int, default=10000)
    parser.add_argument('--n-test', type=int, default=10000)
    parser.add_argument('--filters', type=int, default=512)
    parser.add_argument('--iterations', type=int, default=300)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    document = build(args.n_train, args.n_test, args.filters, args.iterations)
    args.output.write_text(json.dumps(document, separators=(',', ':')) + '\n')
