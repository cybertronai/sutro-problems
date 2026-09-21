#!/usr/bin/env python3
"""Grid lowering (affine IL, ISA v4) of the 3% two-mixture bag: two Gaussian-mixture QDA members on square-rooted
pixels, their class log-scores standardized and summed, first strict maximum.

Member m: PCA basis of dimension K by `rounds` of subspace iteration on the global covariance of the first NPB
samples (fixed literal start, division-only Gram-Schmidt, max-|entry| column scaling, as in the merged 5% entry);
projection; then per class a k-component Gaussian mixture fitted by k-means initialisation and EM, each component
covariance shrunk toward the class covariance (lam) and by a trace ridge (shrink), inverted by Gauss-Jordan without
pivoting, its log-determinant from the pivots through the halving/doubling log series.

Two ISA obstacles and their legal replacements, both declared:
  per-class passes   selecting a class's rows is a gather; every class pass instead runs over all N rows with a
                     compare/select mask, so each of the ten classes pays for the whole training set.
  k-means counts     accumulating into `centroid[assign[n]]` is an indexed write; the assignment is stored as a
                     value and each component accumulates a masked pass instead.

python bag_ir.py validate | score [n_train n_test k1 K1 k2 K2 npb]
"""
import json, math, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / 'submissions' / 'grid-mlp-scoring-20260912'))
from affine import ref as R, ins as I, loop as L, make_program, expand, Program
f32 = np.float32
def raw(v): return int(np.array(v, dtype=f32).view(np.uint32))
D = 81; CLS = 10; NR = 6; EXPSH = 8; LOG_STEPS = 40; SERIES = 8; LN2 = float(np.float32(0.6931471805599453))
SHRINK, LAM, EMSTEPS, KMSTEPS = 0.05, 0.5, 10, 10

(S_T, S_U, S_V, S_ACC, S_A, S_B, S_C, S_CD, S_Y, S_Q, S_Z, S_Z2, S_TERM, S_KK, S_W, S_LD, S_PIV, S_FCT, S_MX,
 S_SUM, S_BEST, S_LAB, S_CNT, S_DEN, S_D2, S_MIN, S_ARG, S_SD, S_MEAN, S_E, S_G, S_TMP) = range(32)
NS = 32
(KZ, KONE, KHALF, KTWO, KHI, KLO, KLN2, KTINY, KEXPN, KE3, KE4, KSHR, KKEEP, KLAM, KOLAM, KEPSJ, KNEG, KBIG, KTEN) = range(19)
KODD = 19; KCLS = KODD + SERIES - 1; NK = KCLS + CLS


def s_(i): return R('s', i)
def k_(i): return R('k', i)


def nr_sqrt(dst, src):
    y, q = s_(S_Y), s_(S_Q); b = [I('add', y, src, k_(KTINY))]
    for _ in range(NR): b += [I('div', q, src, y), I('add', q, y, q), I('mul', y, k_(KHALF), q)]
    return b + [I('copy', dst, y)]


def log_body(src, dst):
    u, kk, z, z2, term, acc, t, v, cd = s_(S_U), s_(S_KK), s_(S_Z), s_(S_Z2), s_(S_TERM), s_(S_ACC), s_(S_T), s_(S_V), s_(S_CD)
    b = [I('copy', u, src), I('set', kk, raw(0)),
         L('lhi', LOG_STEPS, [I('cmp', cd, k_(KHI), u), I('mul', t, u, k_(KHALF)), I('select', u, cd, t, u), I('add', t, kk, k_(KONE)), I('select', kk, cd, t, kk)]),
         L('llo', LOG_STEPS, [I('cmp', cd, u, k_(KLO)), I('mul', t, u, k_(KTWO)), I('select', u, cd, t, u), I('sub', t, kk, k_(KONE)), I('select', kk, cd, t, kk)]),
         I('sub', t, u, k_(KONE)), I('add', v, u, k_(KONE)), I('div', z, t, v), I('mul', z2, z, z), I('copy', term, z), I('copy', acc, z)]
    for m in range(1, SERIES): b += [I('mul', term, term, z2), I('div', t, term, k_(KODD + m - 1)), I('add', acc, acc, t)]
    return b + [I('mul', t, kk, k_(KLN2)), I('mul', v, k_(KTWO), acc), I('add', dst, t, v)]


def exp_neg(dst, src):
    t, e = s_(S_T), s_(S_U)
    b = [I('mul', t, src, k_(KEXPN)), I('mul', e, t, k_(KE4)), I('add', e, e, k_(KE3)), I('mul', e, e, t), I('add', e, e, k_(KHALF)),
         I('mul', e, e, t), I('add', e, e, k_(KONE)), I('mul', e, e, t), I('add', e, e, k_(KONE))]
    for _ in range(EXPSH): b.append(I('mul', e, e, e))
    return b + [I('copy', dst, e)]


def build(n_train=10000, n_test=10000, k1=16, K1=30, k2=32, K2=40, npb=2000, rounds=3, seed=0, nmem=2, order=None, insize=None):
    N, Q = n_train, n_test
    members = [('A', k1, K1), ('B', k2, K2)][:nmem]
    regions = [('s', NS), ('k', NK), ('jval', max(k1, k2)), ('clsval', CLS), ('kval', max(K1, K2)), ('gm', D * (D + 1) // 2), ('gs', D), ('S', D * D), ('ui', D), ('sc', 2 * CLS * Q if False else CLS * Q), ('sc2', CLS * Q)]
    for tag, k, K in members:   # the per-component working set first: it is read once per sample per component
        regions += [(f'Sk{tag}', K * K), (f'mu{tag}', k * K), (f'clcov{tag}', K * K), (f'Pw{tag}', K * K), (f'll{tag}', k), (f'ld{tag}', k), (f'pi{tag}', k),
                    (f'cs{tag}', k * K), (f'cc{tag}', k), (f'cnt{tag}', k), (f'den{tag}', K), (f'm{tag}', D), (f'P{tag}', k * K * K), (f'W{tag}', D * K), (f'V{tag}', D * K),
                    (f'cov{tag}', K * K), (f'as{tag}', N), (f'r{tag}', N * k), (f'z{tag}', N * K), (f'zq{tag}', Q * K)]
    regions += [('x', N * D), ('q', Q * D), ('labels', N), ('out', Q)]
    if insize: regions += [('xin', N * insize * insize), ('qin', Q * insize * insize)]
    if order:   # placement only: region order fixes which words get the near cells
        rank = {n: i for i, n in enumerate(order)}
        regions = sorted(regions, key=lambda rw: rank.get(rw[0], len(rank)))
    body = []
    # with in-program downsampling, x and q are first written by a triple-nested loop the initialization
    # proof cannot walk cheaply, so they are zeroed explicitly and only the recv-written buffers skip it
    lit = ('k', 'labels', 'jval', 'clsval', 'kval') + (('xin', 'qin') if insize else ('x', 'q'))
    for name, words in regions:
        if name in lit: continue
        body.append(L('init', words, [I('set', R(name, init=1), 0)]))
    cs = {KZ: 0., KONE: 1., KHALF: .5, KTWO: 2., KHI: 1.5, KLO: .75, KLN2: LN2, KTINY: 1e-6, KEXPN: -1. / (1 << EXPSH),
          KE3: 1 / 6, KE4: 1 / 24, KSHR: SHRINK, KKEEP: 1 - SHRINK, KLAM: LAM, KOLAM: 1 - LAM, KEPSJ: 1e-4, KNEG: -1., KBIG: 1e30, KTEN: 10.}
    for i, v in cs.items(): body.append(I('set', k_(i), raw(v)))
    for m in range(1, SERIES): body.append(I('set', k_(KODD + m - 1), raw(2 * m + 1)))
    for c in range(CLS): body.append(I('set', k_(KCLS + c), c))
    for j in range(max(k1, k2)): body.append(I('set', R('jval', j), raw(float(j))))
    for c in range(CLS): body.append(I('set', R('clsval', c), c))
    for i in range(max(K1, K2)): body.append(I('set', R('kval', i), raw(float(i))))
    if insize:
        side = int(round(D ** 0.5)); f = insize // side
        assert side * side == D and f * side == insize, 'insize must be an integer multiple of the model side'
        body.append(L('rx', N * insize * insize, [I('recv', R('xin', rx=1))]))
        body.append(L('rl', N, [I('recv', R('labels', rl=1))]))
        body.append(L('rq', Q * insize * insize, [I('recv', R('qin', rq=1))]))
        inv = raw(1.0 / (f * f))
        for src, dst, cnt in (('xin', 'x', N), ('qin', 'q', Q)):
            avg = [I('set', s_(S_ACC), raw(0))]
            for aa in range(f):
                for bb in range(f):
                    avg.append(I('add', s_(S_ACC), s_(S_ACC),
                                 R(src, aa * insize + bb, dn=insize * insize, dr=f * insize, dc=f)))
            avg += [I('set', s_(S_T), inv), I('mul', R(dst, dn=D, dr=side, dc=1), s_(S_ACC), s_(S_T))]
            body.append(L('dn', cnt, [L('dr', side, [L('dc', side, avg)])]))
    else:
        body.append(L('rx', N * D, [I('recv', R('x', rx=1))])); body.append(L('rl', N, [I('recv', R('labels', rl=1))]))
        body.append(L('rq', Q * D, [I('recv', R('q', rq=1))]))
    body.append(L('sx', N * D, nr_sqrt(R('x', sx=1), R('x', sx=1))))
    body.append(L('sq', Q * D, nr_sqrt(R('q', sq=1), R('q', sq=1))))
    # global mean and packed second moments of the first npb samples
    body.append(L('nb', npb, [L('gsl', D, [I('add', R('gs', gsl=1), R('gs', gsl=1), R('x', nb=D, gsl=1))]),
                              L('gi', D, [L('gj', D, [I('mul', s_(S_T), R('x', nb=D, gi=1), R('x', nb=D, gj=1)), I('add', R('S', gi=D, gj=1), R('S', gi=D, gj=1), s_(S_T))])])]))
    body.append(I('set', s_(S_CNT), raw(float(npb))))
    body.append(L('gmn', D, [I('div', R('gs', gmn=1), R('gs', gmn=1), s_(S_CNT))]))
    body.append(L('si', D, [L('sj', D, [I('div', s_(S_U), R('S', si=D, sj=1), s_(S_CNT)), I('mul', s_(S_V), R('gs', si=1), R('gs', sj=1)), I('sub', R('S', si=D, sj=1), s_(S_U), s_(S_V))])]))
    rng = np.random.default_rng(seed)
    for tag, k, K in members:
        W0 = rng.standard_normal((D, K)).astype(f32)
        for i in range(D):
            for j in range(K): body.append(I('set', R(f'W{tag}', i * K + j), int(W0[i, j].view(np.uint32))))
        body.append(L('cm', D, [I('copy', R(f'm{tag}', cm=1), R('gs', cm=1))]))
        for _ in range(rounds):
            body.append(L('vi', D, [L('vk', K, [I('set', s_(S_ACC), raw(0)),
                                                L('vj', D, [I('mul', s_(S_T), R('S', vi=D, vj=1), R(f'W{tag}', vj=K, vk=1)), I('add', s_(S_ACC), s_(S_ACC), s_(S_T))]),
                                                I('copy', R(f'V{tag}', vi=K, vk=1), s_(S_ACC))])]))
            for j in range(K):
                for i in range(j):
                    body += [I('set', s_(S_ACC), raw(0)),
                             L('gr', D, [I('mul', s_(S_T), R(f'V{tag}', i, gr=K), R(f'V{tag}', j, gr=K)), I('add', s_(S_ACC), s_(S_ACC), s_(S_T))]),
                             I('div', s_(S_C), s_(S_ACC), R(f'den{tag}', i)),
                             L('gu', D, [I('mul', s_(S_T), s_(S_C), R(f'V{tag}', i, gu=K)), I('sub', R(f'V{tag}', j, gu=K), R(f'V{tag}', j, gu=K), s_(S_T))])]
                body += [I('set', s_(S_ACC), raw(0)),
                         L('gd', D, [I('mul', s_(S_T), R(f'V{tag}', j, gd=K), R(f'V{tag}', j, gd=K)), I('add', s_(S_ACC), s_(S_ACC), s_(S_T))]),
                         I('copy', R(f'den{tag}', j), s_(S_ACC))]
            for j in range(K):
                body += [I('set', s_(S_MX), raw(0)),
                         L('sa', D, [I('sub', s_(S_T), k_(KZ), R(f'V{tag}', j, sa=K)), I('cmp', s_(S_CD), R(f'V{tag}', j, sa=K), k_(KZ)), I('select', s_(S_A), s_(S_CD), s_(S_T), R(f'V{tag}', j, sa=K)),
                                     I('cmp', s_(S_C), s_(S_MX), s_(S_A)), I('select', s_(S_MX), s_(S_C), s_(S_A), s_(S_MX))]),
                         L('sd', D, [I('div', R(f'V{tag}', j, sd=K), R(f'V{tag}', j, sd=K), s_(S_MX))])]
            body.append(L('cw', D * K, [I('copy', R(f'W{tag}', cw=1), R(f'V{tag}', cw=1))]))
        # projections
        for src, dst, cnt in (('x', f'z{tag}', N), ('q', f'zq{tag}', Q)):
            body.append(L('pn', cnt, [L('pc', D, [I('sub', R('ui', pc=1), R(src, pn=D, pc=1), R(f'm{tag}', pc=1))]),
                                      L('pk', K, [I('set', s_(S_ACC), raw(0)),
                                                  L('pi', D, [I('mul', s_(S_T), R('ui', pi=1), R(f'W{tag}', pi=K, pk=1)), I('add', s_(S_ACC), s_(S_ACC), s_(S_T))]),
                                                  I('copy', R(dst, pn=K, pk=1), s_(S_ACC))])]))
        # per class (affine loop), masked k-means and EM with affine component and iteration loops
        zt = f'z{tag}'; sreg = 'sc' if tag == 'A' else 'sc2'
        mask = [I('cmp', s_(S_E), R('labels', n=1), R('clsval', c=1), predicate='eq')]
        cls_body = [I('set', s_(S_CNT), raw(0)), L('zc', K, [I('set', R(f'cs{tag}', zc=1), raw(0))]), L('zk', K * K, [I('set', R(f'clcov{tag}', zk=1), raw(0))])]
        cls_body.append(L('n', N, mask + [I('select', s_(S_T), s_(S_E), k_(KONE), k_(KZ)), I('add', s_(S_CNT), s_(S_CNT), s_(S_T)),
                                          L('ck', K, [I('select', s_(S_T), s_(S_E), R(zt, n=K, ck=1), k_(KZ)), I('add', R(f'cs{tag}', ck=1), R(f'cs{tag}', ck=1), s_(S_T))])]))
        cls_body.append(L('mk', K, [I('div', R(f'cs{tag}', mk=1), R(f'cs{tag}', mk=1), s_(S_CNT))]))
        cls_body.append(L('n', N, mask + [L('ci', K, [I('sub', s_(S_A), R(zt, n=K, ci=1), R(f'cs{tag}', ci=1)), I('select', s_(S_A), s_(S_E), s_(S_A), k_(KZ)),
                                                      L('cj', K, [I('sub', s_(S_B), R(zt, n=K, cj=1), R(f'cs{tag}', cj=1)), I('mul', s_(S_T), s_(S_A), s_(S_B)),
                                                                  I('add', R(f'clcov{tag}', ci=K, cj=1), R(f'clcov{tag}', ci=K, cj=1), s_(S_T))])])]))
        cls_body.append(L('cv', K * K, [I('div', R(f'clcov{tag}', cv=1), R(f'clcov{tag}', cv=1), s_(S_CNT))]))
        cls_body += [I('set', s_(S_ACC), raw(0)), L('tr', K, [I('add', s_(S_ACC), s_(S_ACC), R(f'clcov{tag}', tr=K + 1))]),
                     I('set', s_(S_U), raw(1.0 / K)), I('mul', s_(S_ACC), s_(S_ACC), s_(S_U)), I('mul', s_(S_ACC), s_(S_ACC), k_(KSHR)),
                     L('sh', K * K, [I('mul', R(f'clcov{tag}', sh=1), R(f'clcov{tag}', sh=1), k_(KKEEP))]),
                     L('sd2', K, [I('add', R(f'clcov{tag}', sd2=K + 1), R(f'clcov{tag}', sd2=K + 1), s_(S_ACC))])]
        # k-means: centroids from fixed sample offsets, then KMSTEPS masked rounds
        cls_body.append(L('j', k, [L('ic', K, [I('copy', R(f'mu{tag}', j=K, ic=1), R(zt, j=7 * K, ic=1))])]))
        assign = [I('set', s_(S_MIN), k_(KBIG) if False else raw(1e30)), I('set', s_(S_ARG), raw(0)),
                  L('j', k, [I('set', s_(S_D2), raw(0)),
                             L('kd', K, [I('sub', s_(S_T), R(zt, n=K, kd=1), R(f'mu{tag}', j=K, kd=1)), I('mul', s_(S_T), s_(S_T), s_(S_T)), I('add', s_(S_D2), s_(S_D2), s_(S_T))]),
                             I('cmp', s_(S_CD), s_(S_D2), s_(S_MIN)), I('select', s_(S_MIN), s_(S_CD), s_(S_D2), s_(S_MIN)), I('select', s_(S_ARG), s_(S_CD), R('jval', j=1), s_(S_ARG))]),
                  I('copy', R(f'as{tag}', n=1), s_(S_ARG))]
        km_round = [L('n', N, assign),
                    L('j', k, [I('set', R(f'cc{tag}', j=1), raw(0)), L('z7', K, [I('set', R(f'cs{tag}', j=K, z7=1), raw(0))]),
                               L('n', N, mask + [I('cmp', s_(S_C), R(f'as{tag}', n=1), R('jval', j=1), predicate='eq'),
                                                 I('select', s_(S_T), s_(S_E), k_(KONE), k_(KZ)), I('select', s_(S_T), s_(S_C), s_(S_T), k_(KZ)),
                                                 I('add', R(f'cc{tag}', j=1), R(f'cc{tag}', j=1), s_(S_T)),
                                                 L('ak', K, [I('mul', s_(S_U), s_(S_T), R(zt, n=K, ak=1)), I('add', R(f'cs{tag}', j=K, ak=1), R(f'cs{tag}', j=K, ak=1), s_(S_U))])]),
                               I('add', s_(S_DEN), R(f'cc{tag}', j=1), k_(KTINY)),
                               L('nm', K, [I('div', R(f'mu{tag}', j=K, nm=1), R(f'cs{tag}', j=K, nm=1), s_(S_DEN))])])]
        cls_body.append(L('km', KMSTEPS, km_round))
        cls_body.append(L('n', N, [L('j', k, [I('cmp', s_(S_C), R(f'as{tag}', n=1), R('jval', j=1), predicate='eq'), I('select', R(f'r{tag}', n=k, j=1), s_(S_C), k_(KONE), k_(KZ))])]))
        # M step: weighted means and covariances, shrinkage, Gauss-Jordan inverse, log-determinant, weight
        mstep = [L('j', k, [I('set', s_(S_CNT), raw(0)), L('z5', K, [I('set', R(f'cs{tag}', j=K, z5=1), raw(0))]), L('z6', K * K, [I('set', R(f'Sk{tag}', z6=1), raw(0))]),
                 L('n', N, mask + [I('select', s_(S_G), s_(S_E), R(f'r{tag}', n=k, j=1), k_(KZ)), I('add', s_(S_CNT), s_(S_CNT), s_(S_G)),
                                   L('wk', K, [I('mul', s_(S_T), s_(S_G), R(zt, n=K, wk=1)), I('add', R(f'cs{tag}', j=K, wk=1), R(f'cs{tag}', j=K, wk=1), s_(S_T))])]),
                 I('add', s_(S_DEN), s_(S_CNT), k_(KTINY)),
                 L('wm', K, [I('div', R(f'mu{tag}', j=K, wm=1), R(f'cs{tag}', j=K, wm=1), s_(S_DEN))]),
                 L('n', N, mask + [I('select', s_(S_G), s_(S_E), R(f'r{tag}', n=k, j=1), k_(KZ)),
                                   L('si2', K, [I('sub', s_(S_A), R(zt, n=K, si2=1), R(f'mu{tag}', j=K, si2=1)), I('mul', s_(S_A), s_(S_A), s_(S_G)),
                                                L('sj2', K, [I('sub', s_(S_B), R(zt, n=K, sj2=1), R(f'mu{tag}', j=K, sj2=1)), I('mul', s_(S_T), s_(S_A), s_(S_B)),
                                                             I('add', R(f'Sk{tag}', si2=K, sj2=1), R(f'Sk{tag}', si2=K, sj2=1), s_(S_T))])])]),
                 L('nc', K * K, [I('div', R(f'Sk{tag}', nc=1), R(f'Sk{tag}', nc=1), s_(S_DEN)), I('mul', R(f'Sk{tag}', nc=1), R(f'Sk{tag}', nc=1), k_(KOLAM)),
                                 I('mul', s_(S_T), R(f'clcov{tag}', nc=1), k_(KLAM)), I('add', R(f'Sk{tag}', nc=1), R(f'Sk{tag}', nc=1), s_(S_T))]),
                 L('ridge', K, [I('add', R(f'Sk{tag}', ridge=K + 1), R(f'Sk{tag}', ridge=K + 1), k_(KEPSJ))]),
                 L('pw', K * K, [I('set', R(f'Pw{tag}', pw=1), raw(0))]), L('pd', K, [I('set', R(f'Pw{tag}', pd=K + 1), raw(1.0))]),
                 I('set', s_(S_LD), raw(0))]
                 + sum([[I('copy', s_(S_PIV), R(f'Sk{tag}', p_ * K + p_)), *log_body(s_(S_PIV), s_(S_W)), I('add', s_(S_LD), s_(S_LD), s_(S_W)),
                         L('gj', K, [I('div', R(f'Sk{tag}', p_ * K, gj=1), R(f'Sk{tag}', p_ * K, gj=1), s_(S_PIV)), I('div', R(f'Pw{tag}', p_ * K, gj=1), R(f'Pw{tag}', p_ * K, gj=1), s_(S_PIV))]),
                         L('ei', K, [I('copy', s_(S_FCT), R(f'Sk{tag}', p_, ei=K)), I('set', s_(S_T), raw(float(p_))), I('cmp', s_(S_C), R('kval', ei=1), s_(S_T), predicate='eq'), I('select', s_(S_FCT), s_(S_C), k_(KZ), s_(S_FCT)),
                                     L('ge', K, [I('mul', s_(S_T), s_(S_FCT), R(f'Sk{tag}', p_ * K, ge=1)), I('sub', R(f'Sk{tag}', ei=K, ge=1), R(f'Sk{tag}', ei=K, ge=1), s_(S_T)),
                                                 I('mul', s_(S_T), s_(S_FCT), R(f'Pw{tag}', p_ * K, ge=1)), I('sub', R(f'Pw{tag}', ei=K, ge=1), R(f'Pw{tag}', ei=K, ge=1), s_(S_T))])])] for p_ in range(K)], [])
                 + [L('cp', K * K, [I('copy', R(f'P{tag}', j=K * K, cp=1), R(f'Pw{tag}', cp=1))]),
                    I('copy', R(f'ld{tag}', j=1), s_(S_LD)), I('add', s_(S_TMP), s_(S_CNT), k_(KTINY)), I('copy', R(f'pi{tag}', j=1), s_(S_TMP))])]
        nll_j = [I('set', s_(S_ACC), raw(0)),
                 L('qi', K, [I('sub', s_(S_A), R(zt, n=K, qi=1), R(f'mu{tag}', j=K, qi=1)), I('set', s_(S_B), raw(0)),
                             L('qj', K, [I('sub', s_(S_T), R(zt, n=K, qj=1), R(f'mu{tag}', j=K, qj=1)), I('mul', s_(S_T), s_(S_T), R(f'P{tag}', j=K * K, qi=K, qj=1)), I('add', s_(S_B), s_(S_B), s_(S_T))]),
                             I('mul', s_(S_T), s_(S_A), s_(S_B)), I('add', s_(S_ACC), s_(S_ACC), s_(S_T))]),
                 I('mul', s_(S_TMP), s_(S_ACC), k_(KHALF)), I('mul', s_(S_U), R(f'ld{tag}', j=1), k_(KHALF)), I('add', s_(S_TMP), s_(S_TMP), s_(S_U)),
                 *log_body(R(f'pi{tag}', j=1), s_(S_V)), I('sub', s_(S_TMP), s_(S_TMP), s_(S_V)), I('copy', R(f'll{tag}', j=1), s_(S_TMP))]
        estep = [L('n', N, mask + [L('j', k, nll_j), I('copy', s_(S_MIN), R(f'll{tag}', 0)),
                                   L('j', k, [I('cmp', s_(S_C), R(f'll{tag}', j=1), s_(S_MIN)), I('select', s_(S_MIN), s_(S_C), R(f'll{tag}', j=1), s_(S_MIN))]),
                                   I('set', s_(S_SUM), raw(0)),
                                   L('j', k, [I('sub', s_(S_A), R(f'll{tag}', j=1), s_(S_MIN)), *exp_neg(s_(S_G), s_(S_A)), I('copy', R(f'r{tag}', n=k, j=1), s_(S_G)), I('add', s_(S_SUM), s_(S_SUM), s_(S_G))]),
                                   L('j', k, [I('div', s_(S_G), R(f'r{tag}', n=k, j=1), s_(S_SUM)), I('select', R(f'r{tag}', n=k, j=1), s_(S_E), s_(S_G), k_(KZ))])])]
        cls_body.append(L('em', EMSTEPS, mstep + estep)); cls_body += mstep
        nll_q = [x_ for x_ in nll_j]
        nll_q = [I('set', s_(S_ACC), raw(0)),
                 L('qi', K, [I('sub', s_(S_A), R(f'zq{tag}', qn=K, qi=1), R(f'mu{tag}', j=K, qi=1)), I('set', s_(S_B), raw(0)),
                             L('qj', K, [I('sub', s_(S_T), R(f'zq{tag}', qn=K, qj=1), R(f'mu{tag}', j=K, qj=1)), I('mul', s_(S_T), s_(S_T), R(f'P{tag}', j=K * K, qi=K, qj=1)), I('add', s_(S_B), s_(S_B), s_(S_T))]),
                             I('mul', s_(S_T), s_(S_A), s_(S_B)), I('add', s_(S_ACC), s_(S_ACC), s_(S_T))]),
                 I('mul', s_(S_TMP), s_(S_ACC), k_(KHALF)), I('mul', s_(S_U), R(f'ld{tag}', j=1), k_(KHALF)), I('add', s_(S_TMP), s_(S_TMP), s_(S_U)),
                 *log_body(R(f'pi{tag}', j=1), s_(S_V)), I('sub', s_(S_TMP), s_(S_TMP), s_(S_V)), I('copy', R(f'll{tag}', j=1), s_(S_TMP))]
        cls_body.append(L('qn', Q, [L('j', k, nll_q), I('copy', s_(S_MIN), R(f'll{tag}', 0)),
                                    L('j', k, [I('cmp', s_(S_C), R(f'll{tag}', j=1), s_(S_MIN)), I('select', s_(S_MIN), s_(S_C), R(f'll{tag}', j=1), s_(S_MIN))]),
                                    I('set', s_(S_SUM), raw(0)),
                                    L('j', k, [I('sub', s_(S_A), R(f'll{tag}', j=1), s_(S_MIN)), *exp_neg(s_(S_G), s_(S_A)), I('add', s_(S_SUM), s_(S_SUM), s_(S_G))]),
                                    *log_body(s_(S_SUM), s_(S_V)), I('sub', s_(S_TMP), s_(S_V), s_(S_MIN)), I('copy', R(sreg, qn=CLS, c=1), s_(S_TMP))]))
        body.append(L('c', CLS, cls_body))
    # standardize each member's ten scores per query, sum, first strict maximum
    qb = []
    for reg in ('sc', 'sc2')[:nmem]:
        qb += [I('set', s_(S_MEAN), raw(0)), L('zm', CLS, [I('add', s_(S_MEAN), s_(S_MEAN), R(reg, zq=CLS, zm=1))]),
               I('set', s_(S_U), raw(0.1)), I('mul', s_(S_MEAN), s_(S_MEAN), s_(S_U)), I('set', s_(S_SD), raw(0)),
               L('zv', CLS, [I('sub', s_(S_T), R(reg, zq=CLS, zv=1), s_(S_MEAN)), I('mul', s_(S_T), s_(S_T), s_(S_T)), I('add', s_(S_SD), s_(S_SD), s_(S_T))]),
               I('set', s_(S_U), raw(1 / 9)), I('mul', s_(S_SD), s_(S_SD), s_(S_U)), *nr_sqrt(s_(S_SD), s_(S_SD)),
               L('zs', CLS, [I('sub', R(reg, zq=CLS, zs=1), R(reg, zq=CLS, zs=1), s_(S_MEAN)), I('div', R(reg, zq=CLS, zs=1), R(reg, zq=CLS, zs=1), s_(S_SD))])]
    if nmem == 2:
        qb += [L('su', CLS, [I('add', R('sc', zq=CLS, su=1), R('sc', zq=CLS, su=1), R('sc2', zq=CLS, su=1))])]
    qb += [
           I('copy', s_(S_BEST), R('sc', zq=CLS)), I('copy', s_(S_LAB), k_(KCLS))]
    for c in range(1, CLS): qb += [I('cmp', s_(S_CD), s_(S_BEST), R('sc', c, zq=CLS)), I('select', s_(S_BEST), s_(S_CD), R('sc', c, zq=CLS), s_(S_BEST)), I('select', s_(S_LAB), s_(S_CD), k_(KCLS + c), s_(S_LAB))]
    qb.append(I('copy', R('out', zq=1), s_(S_LAB)))
    body.append(L('zq', Q, qb)); body.append(L('so', Q, [I('send', R('out', so=1))]))
    meta = {'learner': 'two-mixture bag (3% candidate) lowered to the affine IL', 'members': [{'k': k, 'K': K, 'tag': t} for t, k, K in members],
            'declared_deviations': ['masked per-class passes instead of gathers (each class scans all rows)', 'masked k-means accumulation instead of indexed writes',
                                    'PCA basis by subspace iteration instead of an eigensolver', 'k-means centroids initialised from fixed sample offsets']}
    return make_program(regions, body, meta)


def mirror(x, y, q, k1=16, K1=30, k2=32, K2=40, npb=2000, rounds=3, seed=0, nmem=2):
    """float64 reference of exactly the algorithm the program runs (masked passes have no effect in numpy)."""
    N, Q = len(x), len(q); u = np.sqrt(np.clip(x, 0, None)).astype(np.float64); uq = np.sqrt(np.clip(q, 0, None)).astype(np.float64)
    m = u[:npb].mean(0); S = (u[:npb].T @ u[:npb]) / npb - np.outer(m, m)
    rng = np.random.default_rng(seed); scores = []
    for k, K in (((k1, K1), (k2, K2))[:nmem]):
        W = rng.standard_normal((D, K))
        for _ in range(rounds):
            V = S @ W; den = np.zeros(K)
            for j in range(K):
                for i in range(j): V[:, j] -= (V[:, i] @ V[:, j]) / den[i] * V[:, i]
                den[j] = V[:, j] @ V[:, j]
            V = V / np.abs(V).max(0, keepdims=True); W = V
        z = (u - m) @ W; zq = (uq - m) @ W; sc = np.zeros((Q, CLS))
        for c in range(CLS):
            sel = y == c; zc = z[sel]; nc = len(zc)
            mu_c = zc.mean(0); cov_c = ((zc - mu_c).T @ (zc - mu_c)) / nc
            cov_c = (1 - SHRINK) * cov_c + SHRINK * np.trace(cov_c) / K * np.eye(K)
            mu = np.stack([z[(j * 7) % N] for j in range(k)])
            for _ in range(KMSTEPS):
                a = ((zc[:, None, :] - mu[None]) ** 2).sum(2).argmin(1)
                for j in range(k):
                    msk = a == j; cnt = msk.sum(); mu[j] = zc[msk].sum(0) / (cnt + 1e-6)
            Rw = np.zeros((nc, k)); Rw[np.arange(nc), ((zc[:, None, :] - mu[None]) ** 2).sum(2).argmin(1)] = 1
            for em in range(EMSTEPS + 1):
                Nj = Rw.sum(0) + 1e-6; mu = (Rw.T @ zc) / Nj[:, None]; P = []; ld = []
                for j in range(k):
                    d = zc - mu[j]; Sj = (d * Rw[:, j:j + 1]).T @ d / Nj[j]
                    Sj = (1 - LAM) * Sj + LAM * cov_c + 1e-4 * np.eye(K)
                    P.append(np.linalg.inv(Sj)); ld.append(np.linalg.slogdet(Sj)[1])
                P = np.stack(P); ld = np.array(ld)
                if em == EMSTEPS: break
                nll = np.stack([0.5 * np.einsum('ni,ij,nj->n', zc - mu[j], P[j], zc - mu[j]) + 0.5 * ld[j] - math.log(Nj[j]) for j in range(k)], 1)
                e = np.exp(-(nll - nll.min(1, keepdims=True))); Rw = e / e.sum(1, keepdims=True)
            nllq = np.stack([0.5 * np.einsum('ni,ij,nj->n', zq - mu[j], P[j], zq - mu[j]) + 0.5 * ld[j] - math.log(Nj[j]) for j in range(k)], 1)
            mn = nllq.min(1); sc[:, c] = np.log(np.exp(-(nllq - mn[:, None])).sum(1)) - mn
        scores.append((sc - sc.mean(1, keepdims=True)) / sc.std(1, ddof=1, keepdims=True))
    return sum(scores[:nmem]).argmax(1)


def execute(document, tape_in, return_memory=False):
    program = Program(document); mem = np.zeros(program.words + 1, dtype=np.uint32); fmem = mem.view(f32); tape = iter(tape_in); out = []
    for op in expand(document):
        c = op[0]
        if c == 'set': mem[op[1]] = np.uint32(op[2])
        elif c == 'recv': mem[op[1]] = np.uint32(next(tape))
        elif c == 'send': out.append(int(mem[op[1]]))
        elif c == 'copy': mem[op[1]] = mem[op[2]]
        elif c == 'add': fmem[op[1]] = fmem[op[2]] + fmem[op[3]]
        elif c == 'sub': fmem[op[1]] = fmem[op[2]] - fmem[op[3]]
        elif c == 'mul': fmem[op[1]] = fmem[op[2]] * fmem[op[3]]
        elif c == 'div': fmem[op[1]] = fmem[op[2]] / fmem[op[3]]
        elif c == 'cmp': mem[op[1]] = np.uint32(1 if fmem[op[2]] < fmem[op[3]] else 0)
        elif c == 'cmp_eq': mem[op[1]] = np.uint32(1 if mem[op[2]] == mem[op[3]] else 0)
        elif c == 'select': mem[op[1]] = mem[op[3]] if mem[op[2]] != 0 else mem[op[4]]
        else: raise ValueError(c)
    if return_memory: return out, {n: fmem[b:b + w].copy() for n, (b, w) in program.regions.items()}
    return out


def tape_words(train, labels, test):
    return (list(np.ascontiguousarray(train, dtype=f32).reshape(-1).view(np.uint32)) + [int(v) for v in labels]
            + list(np.ascontiguousarray(test, dtype=f32).reshape(-1).view(np.uint32)))


if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'score':
        sys.path.insert(0, str(HERE)); import score as SC
        def _nocache(program):
            counts = {n_: np.zeros(program.words + 1, dtype=np.int64) for n_ in ('reads', 'writes', 'input_destinations', 'output_sources')}
            def charge(n_, operand, scope):
                first, h = program.histogram(operand, scope); counts[n_][first:first + len(h)] += h
            for node, scope, mult in program.leaves:
                if not mult: continue
                if node['op'] == 'recv': charge('input_destinations', node['dst'], scope)
                elif node['op'] == 'send': charge('output_sources', node['src'][0], scope)
                else:
                    for src in node.get('src', []): charge('reads', src, scope)
                    charge('writes', node['dst'], scope)
            return counts, len(program.leaves)
        SC.histogram_counts = _nocache
        a = [int(v) for v in sys.argv[2:]] or [10000, 10000, 16, 30, 32, 40, 2000]
        t0 = time.time(); doc = build(*a); print('built', round(time.time() - t0), 's', flush=True)
        r = SC.score(doc)
        print({k: r[k] for k in ('energy_mj', 'time_ms', 'energy_fj', 'cycles', 'total_executed_instructions', 'peak_allocated_scratch_bytes', 'memory_tiles', 'time_to_score_seconds')}, a, flush=True)
        (HERE / 'evidence' / f'bag_ir_score_N{a[0]}_k{a[2]}_{a[4]}.json').write_text(json.dumps({k: v for k, v in r.items() if not isinstance(v, (dict, list))}, indent=1))
    else:
        sys.path.insert(0, str(HERE.parents[1])); from mnist.code import data as ds
        n, qn = 300, 40; RAW = HERE.parents[0] / 'data/raw'
        PIX = ds.read_idx(RAW / 'train-images-idx3-ubyte.gz', 60000, True); LAB = ds.read_idx(RAW / 'train-labels-idx1-ubyte.gz', 60000, False)
        order = np.random.Generator(np.random.PCG64(20261121)).permutation(60000); tr, te = order[:n], order[10000:10000 + qn]
        xr = ds.area_resize(PIX[tr].astype(f32) / f32(255), 9).reshape(n, 81); qr = ds.area_resize(PIX[te].astype(f32) / f32(255), 9).reshape(qn, 81)
        doc = build(n, qn, 2, 6, 3, 8, 100); t0 = time.time(); out = np.array(execute(doc, tape_words(xr, LAB[tr], qr)))
        ref = mirror(xr, LAB[tr], qr, 2, 6, 3, 8, 100)
        print(f'reduced {n}/{qn}: program correct {(out == LAB[te]).sum()}/{qn}, mirror correct {(ref == LAB[te]).sum()}/{qn}, labels agree {(out == ref).sum()}/{qn}; executor {time.time() - t0:.0f}s', flush=True)
