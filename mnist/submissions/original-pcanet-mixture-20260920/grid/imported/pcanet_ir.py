#!/usr/bin/env python3
"""Grid lowering (affine IL, ISA v4) of PCANet on the original 28x28 MNIST.

Two PCA filter banks learned from image patches, the second stage's responses binarised and packed into an L2-bit
code, block histograms of that code, and a ridge on the histogram vector.  No gradients anywhere: every stage is
either a fixed linear map or a counting pass.

ISA obstacles and their legal replacements, all declared:
  block histogram  counting into bin number `code` is an indexed write, which the ISA has no instruction for.  The
                   code is instead carried as a one-hot vector over its 2^L2 values, built two multiplies per node
                   down a binary tree from the L2 sign bits, and the histogram accumulates the whole one-hot.  This
                   is the same construction the context-tree program uses, and it is what makes PCANet expensive.
  eigenvectors     no eigensolver, so both filter banks come from subspace iteration with division-only
                   Gram-Schmidt, exactly as in the merged 5% entry.
  patch statistics the filter-bank covariance is accumulated over the first NPB images rather than a random
                   subsample, which would be a gather; the draw is already a uniform permutation.
  ridge            the primal normal equations are solved by Jacobi-preconditioned conjugate gradients, which is
                   cheaper than the dual whenever the feature count is below the row count.

python pcanet_ir.py validate | score [N Q L1 L2 k blk stride]
"""
import json, math, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / 'submissions' / 'grid-mlp-scoring-20260912'))
from affine import ref as R, ins as I, loop as L, make_program, expand, Program
f32 = np.float32
def raw(v): return int(np.array(v, dtype=f32).view(np.uint32))
CLS = 10; NR = 6

(S_T, S_U, S_V, S_ACC, S_A, S_B, S_C, S_CD, S_Y, S_Q, S_MX, S_MEAN, S_BIT, S_NB,
 S_RZ, S_RZN, S_AL, S_BEST, S_LAB, S_TMP) = range(20)
NS = 20
(K_Z, K_ONE, K_HALF, K_TWO, K_TINY, K_LAM, K_INVKK, K_NEG, K_INVB, K_INVDF) = range(10)
K_CLS = 10; NK = K_CLS + CLS


def s_(i): return R('s', i)
def k_(i): return R('k', i)


def nr_sqrt(dst, src):
    """Newton-Raphson square root, forced exact at zero.

    Without the final select this returns TINY / 2^NR for a zero input, about 1.6e-8.  Every other lowering here
    survives that, because a uniform offset cancels once patches or features are centred.  PCANet does not: it
    thresholds its stage-two responses at zero, and the offset meets true zeros at the padding boundary, which
    flips the sign bit for a majority of background pixels."""
    y, q = s_(S_Y), s_(S_Q)
    b = [I('add', y, src, k_(K_TINY))]
    for _ in range(NR): b += [I('div', q, src, y), I('add', q, y, q), I('mul', y, k_(K_HALF), q)]
    return b + [I('cmp', s_(S_CD), k_(K_Z), src), I('select', dst, s_(S_CD), y, k_(K_Z))]


def build(N=10000, Q=10000, L1=8, L2=6, kk=5, blk=14, stride=14, IN=28, npb=2000, rounds=3, seed=0, order=None, features_only=False, dead_words=0):
    PAD = kk // 2; PS = IN + 2 * PAD; PP = PS * PS; KK = kk * kk; M2 = 1 << L2
    # Overlapping blocks are the same loop with a different origin stride: the block index already
    # multiplies a pitch, so only that pitch and the block count change. Measured at 60,000 rows the
    # overlap is worth 0.15 points and is the difference between clearing 1 % error and not.
    assert (IN - blk) % stride == 0, 'block grid must tile the image at this stride'
    NBS = (IN - blk) // stride + 1; NB = NBS * NBS
    DF = L1 * NB * M2
    base = PAD * PS + PAD
    NODES = 2 * M2 - 1
    off = [(1 << d) - 1 for d in range(L2 + 1)]
    LMAX = max(L1, L2)
    regions = [('s', NS), ('k', NK), ('ind', NODES), ('resp', L2), ('m1', L1 * PP), ('hist', DF),
               ('cov', KK * KK), ('W1', KK * L1), ('W2', KK * L2), ('V', KK * LMAX), ('den', LMAX), ('pv', KK),
               ('FtF', DF * DF), ('FtY', DF * CLS), ('W', DF * CLS), ('p', DF * CLS), ('ap', DF * CLS),
               ('rr', DF * CLS), ('zz', DF * CLS), ('gb', 3 * CLS), ('sc', Q * CLS),
               ('F', N * DF), ('Fq', Q * DF), ('px', N * PP), ('pq', Q * PP),
               ('x', N * IN * IN), ('xq', Q * IN * IN), ('labels', N), ('Y', N * CLS), ('out', Q)]
    if dead_words:   # a dead block the size of the classifier's working set, to price the fusion penalty
        regions = regions + [('dead', dead_words)]
    if order:   # placement only: region order fixes which words get the near cells
        rank = {n: i for i, n in enumerate(order)}
        regions = sorted(regions, key=lambda rw: rank.get(rw[0], len(rank)))
    body = []
    lit = ('k', 'x', 'xq', 'labels', 'dead')
    for name, words in regions:
        if name in lit: continue
        body.append(L('init', words, [I('set', R(name, init=1), 0)]))
    cs = {K_Z: 0.0, K_ONE: 1.0, K_HALF: 0.5, K_TWO: 2.0, K_TINY: 1e-6, K_LAM: 1e-3, K_INVKK: 1.0 / KK,
          K_NEG: -1.0, K_INVB: 1.0 / (blk * blk), K_INVDF: 1.0 / DF}
    for i, v in cs.items(): body.append(I('set', k_(i), raw(v)))
    for c in range(CLS): body.append(I('set', k_(K_CLS + c), c))
    body.append(L('rx', N * IN * IN, [I('recv', R('x', rx=1))]))
    body.append(L('rl', N, [I('recv', R('labels', rl=1))]))
    body.append(L('rq', Q * IN * IN, [I('recv', R('xq', rq=1))]))
    for src, dst, cnt in (('x', 'px', N), ('xq', 'pq', Q)):
        body.append(L('pn', cnt, [L('pr', IN, [L('pc', IN, nr_sqrt(
            R(dst, base, pn=PP, pr=PS, pc=1), R(src, pn=IN * IN, pr=IN, pc=1)))])]))
    body.append(L('yn', N, [L('yc', CLS, [
        I('cmp', s_(S_CD), R('labels', yn=1), R('k', K_CLS, yc=1), predicate='eq'),
        I('select', R('Y', yn=CLS, yc=1), s_(S_CD), k_(K_ONE), k_(K_Z))])]))

    def patch_centred(reg, nvar, npitch, roff, coff, rowpitch, target, extra=None):
        """target[0..KK) <- the kk x kk patch at (roff, coff), with its own mean removed"""
        b = [I('set', s_(S_ACC), raw(0))]
        co = {nvar: npitch, roff: rowpitch, coff: 1, **(extra or {})}   # the row and column may be split across loops
        for a in range(kk):
            for c in range(kk):
                b.append(I('add', s_(S_ACC), s_(S_ACC), R(reg, a * rowpitch + c, **co)))
        b += [I('mul', s_(S_MEAN), s_(S_ACC), k_(K_INVKK))]
        for a in range(kk):
            for c in range(kk):
                b.append(I('sub', R(target, a * kk + c), R(reg, a * rowpitch + c, **co), s_(S_MEAN)))
        return b

    def subspace(covreg, Wreg, ncols):
        """top `ncols` directions of a KK x KK symmetric matrix, by subspace iteration"""
        b = []
        rng = np.random.default_rng(seed)
        W0 = rng.standard_normal((KK, ncols)).astype(f32)
        for i in range(KK):
            for j in range(ncols):
                b.append(I('set', R(Wreg, i * ncols + j), int(W0[i, j].view(np.uint32))))
        for _ in range(rounds):
            b.append(L('vi', KK, [L('vk', ncols, [I('set', s_(S_ACC), raw(0)),
                L('vj', KK, [I('mul', s_(S_T), R(covreg, vi=KK, vj=1), R(Wreg, vj=ncols, vk=1)),
                             I('add', s_(S_ACC), s_(S_ACC), s_(S_T))]),
                I('copy', R('V', vi=LMAX, vk=1), s_(S_ACC))])]))
            for j in range(ncols):
                for i in range(j):
                    b += [I('set', s_(S_ACC), raw(0)),
                          L('gr', KK, [I('mul', s_(S_T), R('V', i, gr=LMAX), R('V', j, gr=LMAX)),
                                       I('add', s_(S_ACC), s_(S_ACC), s_(S_T))]),
                          I('div', s_(S_C), s_(S_ACC), R('den', i)),
                          L('gu', KK, [I('mul', s_(S_T), s_(S_C), R('V', i, gu=LMAX)),
                                       I('sub', R('V', j, gu=LMAX), R('V', j, gu=LMAX), s_(S_T))])]
                b += [I('set', s_(S_ACC), raw(0)),
                      L('gd', KK, [I('mul', s_(S_T), R('V', j, gd=LMAX), R('V', j, gd=LMAX)),
                                   I('add', s_(S_ACC), s_(S_ACC), s_(S_T))]),
                      I('copy', R('den', j), s_(S_ACC))]
            for j in range(ncols):
                b += [I('set', s_(S_MX), raw(0)),
                      L('sa', KK, [I('sub', s_(S_T), k_(K_Z), R('V', j, sa=LMAX)),
                                   I('cmp', s_(S_CD), R('V', j, sa=LMAX), k_(K_Z)),
                                   I('select', s_(S_A), s_(S_CD), s_(S_T), R('V', j, sa=LMAX)),
                                   I('cmp', s_(S_C), s_(S_MX), s_(S_A)),
                                   I('select', s_(S_MX), s_(S_C), s_(S_A), s_(S_MX))]),
                      I('add', s_(S_MX), s_(S_MX), k_(K_TINY)),
                      L('sd', KK, [I('div', R('V', j, sd=LMAX), R('V', j, sd=LMAX), s_(S_MX))])]
            b.append(L('cw', KK, [L('cwj', ncols, [I('copy', R(Wreg, cw=ncols, cwj=1), R('V', cw=LMAX, cwj=1))])]))
        return b

    # ---- stage one filter bank
    body.append(L('c1n', npb, [L('c1r', IN, [L('c1c', IN,
        patch_centred('px', 'c1n', PP, 'c1r', 'c1c', PS, 'pv') +
        [L('oi', KK, [L('oj', KK, [I('mul', s_(S_T), R('pv', oi=1), R('pv', oj=1)),
                                   I('add', R('cov', oi=KK, oj=1), R('cov', oi=KK, oj=1), s_(S_T))])])])])]))
    body += subspace('cov', 'W1', L1)

    def stage1_maps(src, nvar, npitch):
        """m1[l] <- the padded stage-one response maps of one image"""
        return [L('mz', L1 * PP, [I('set', R('m1', mz=1), raw(0))]),
                L('s1r', IN, [L('s1c', IN, patch_centred(src, nvar, npitch, 's1r', 's1c', PS, 'pv') +
                    [L('s1l', L1, [I('set', s_(S_ACC), raw(0)),
                                   L('s1p', KK, [I('mul', s_(S_T), R('pv', s1p=1), R('W1', s1p=L1, s1l=1)),
                                                 I('add', s_(S_ACC), s_(S_ACC), s_(S_T))]),
                                   I('copy', R('m1', base, s1l=PP, s1r=PS, s1c=1), s_(S_ACC))])])])]

    # ---- stage two filter bank, from the stage-one maps of the same npb images
    body.append(L('cz', KK * KK, [I('set', R('cov', cz=1), raw(0))]))   # the covariance buffer is reused
    body.append(L('c2n', npb, stage1_maps('px', 'c2n', PP) + [
        L('c2l', L1, [L('c2r', IN, [L('c2c', IN,
            patch_centred('m1', 'c2l', PP, 'c2r', 'c2c', PS, 'pv') +
            [L('o2i', KK, [L('o2j', KK, [I('mul', s_(S_T), R('pv', o2i=1), R('pv', o2j=1)),
                                         I('add', R('cov', o2i=KK, o2j=1), R('cov', o2i=KK, o2j=1), s_(S_T))])])])])])]))
    body += subspace('cov', 'W2', L2)

    def feature_pass(src, nvar, npitch, dst, cnt):
        """one image -> its L1 * NB * 2^L2 histogram vector"""
        inner = patch_centred('m1', 'fl', PP, 'frr', 'fcc', PS, 'pv', extra={'fbr': stride * PS, 'fbc': stride}) + [
            L('f2j', L2, [I('set', s_(S_ACC), raw(0)),
                          L('f2p', KK, [I('mul', s_(S_T), R('pv', f2p=1), R('W2', f2p=L2, f2j=1)),
                                        I('add', s_(S_ACC), s_(S_ACC), s_(S_T))]),
                          I('copy', R('resp', f2j=1), s_(S_ACC))]),
            I('set', R('ind', 0), raw(1.0))]
        for d in range(L2):
            inner += [I('cmp', s_(S_CD), k_(K_Z), R('resp', d)),
                      I('select', s_(S_BIT), s_(S_CD), k_(K_ONE), k_(K_Z)),
                      I('sub', s_(S_NB), k_(K_ONE), s_(S_BIT)),
                      L(f'oh{d}', 1 << d, [
                          I('mul', R('ind', off[d + 1], **{f'oh{d}': 2}), R('ind', off[d], **{f'oh{d}': 1}), s_(S_NB)),
                          I('mul', R('ind', off[d + 1] + 1, **{f'oh{d}': 2}), R('ind', off[d], **{f'oh{d}': 1}), s_(S_BIT))])]
        inner.append(L('acc', M2, [
            I('mul', s_(S_TMP), R('ind', off[L2], acc=1), k_(K_INVB)),
            I('add', R('hist', fl=NB * M2, fbr=NBS * M2, fbc=M2, acc=1),
              R('hist', fl=NB * M2, fbr=NBS * M2, fbc=M2, acc=1), s_(S_TMP))]))
        return [L('fn', cnt, stage1_maps(src, 'fn', npitch) + [
            L('hz', DF, [I('set', R('hist', hz=1), raw(0))]),
            L('fl', L1, [L('fbr', NBS, [L('frr', blk, [L('fbc', NBS, [L('fcc', blk, inner)])])])]),
            L('wf', DF, [I('copy', R(dst, fn=DF, wf=1), R('hist', wf=1))])])]

    body += feature_pass('px', 'fn', PP, 'F', N)
    body += feature_pass('pq', 'fn', PP, 'Fq', Q)

    if features_only:
        return body, regions, dict(N=N, Q=Q, L1=L1, L2=L2, kk=kk, blk=blk, NB=NB, NBS=NBS, DF=DF, M2=M2,
                                   PS=PS, PP=PP, KK=KK, IN=IN, base=base, npb=npb, off=off)
    # ---- primal ridge: (F^T F + lam I) W = F^T Y, by Jacobi conjugate gradients
    body.append(L('gi', DF, [L('gj', DF, [I('set', s_(S_ACC), raw(0)),
        L('gn', N, [I('mul', s_(S_T), R('F', gn=DF, gi=1), R('F', gn=DF, gj=1)),
                    I('add', s_(S_ACC), s_(S_ACC), s_(S_T))]),
        I('copy', R('FtF', gi=DF, gj=1), s_(S_ACC))])]))
    body += [I('set', s_(S_ACC), raw(0)),
             L('tr', DF, [I('add', s_(S_ACC), s_(S_ACC), R('FtF', tr=DF + 1))]),
             I('mul', s_(S_ACC), s_(S_ACC), k_(K_INVDF)), I('mul', s_(S_ACC), s_(S_ACC), k_(K_LAM)),
             I('add', s_(S_ACC), s_(S_ACC), k_(K_TINY)), I('copy', s_(S_V), s_(S_ACC))]
    body.append(L('ri', DF, [I('add', R('FtF', ri=DF + 1), R('FtF', ri=DF + 1), s_(S_V))]))
    body.append(L('bi', DF, [L('bc', CLS, [I('set', s_(S_ACC), raw(0)),
        L('bn', N, [I('mul', s_(S_T), R('F', bn=DF, bi=1), R('Y', bn=CLS, bc=1)),
                    I('add', s_(S_ACC), s_(S_ACC), s_(S_T))]),
        I('copy', R('FtY', bi=CLS, bc=1), s_(S_ACC))])]))
    return body, regions, dict(N=N, Q=Q, L1=L1, L2=L2, kk=kk, blk=blk, NB=NB, NBS=NBS, DF=DF, M2=M2,
                               PS=PS, PP=PP, KK=KK, IN=IN, base=base, npb=npb, off=off)


def finish(body, regions, P, cgit=200):
    """Jacobi conjugate gradients on the normal equations, then the first strict maximum."""
    DF = P['DF']
    RZ = lambda **kw: R('gb', 0, **kw)
    RZN = lambda **kw: R('gb', CLS, **kw)
    AL = lambda **kw: R('gb', 2 * CLS, **kw)
    body.append(L('c0', DF * CLS, [I('copy', R('rr', c0=1), R('FtY', c0=1))]))
    body.append(L('c1', DF, [L('c1c', CLS, [I('div', R('zz', c1=CLS, c1c=1), R('rr', c1=CLS, c1c=1), R('FtF', c1=DF + 1))])]))
    body.append(L('c2', DF * CLS, [I('copy', R('p', c2=1), R('zz', c2=1))]))
    body.append(L('z0', CLS, [I('set', RZ(z0=1), raw(0))]))
    body.append(L('z1', DF, [L('z1c', CLS, [I('mul', s_(S_T), R('rr', z1=CLS, z1c=1), R('zz', z1=CLS, z1c=1)),
                                            I('add', RZ(z1c=1), RZ(z1c=1), s_(S_T))])]))
    it = [L('mv', DF, [L('mvc', CLS, [I('set', s_(S_ACC), raw(0)),
                                      L('mvj', DF, [I('mul', s_(S_T), R('FtF', mv=DF, mvj=1), R('p', mvj=CLS, mvc=1)),
                                                    I('add', s_(S_ACC), s_(S_ACC), s_(S_T))]),
                                      I('copy', R('ap', mv=CLS, mvc=1), s_(S_ACC))])]),
          L('a0', CLS, [I('set', AL(a0=1), raw(0))]),
          L('a1', DF, [L('a1c', CLS, [I('mul', s_(S_T), R('p', a1=CLS, a1c=1), R('ap', a1=CLS, a1c=1)),
                                      I('add', AL(a1c=1), AL(a1c=1), s_(S_T))])]),
          L('a2', CLS, [I('add', AL(a2=1), AL(a2=1), k_(K_TINY)), I('div', AL(a2=1), RZ(a2=1), AL(a2=1))]),
          L('up', DF, [L('upc', CLS, [I('mul', s_(S_T), AL(upc=1), R('p', up=CLS, upc=1)),
                                      I('add', R('W', up=CLS, upc=1), R('W', up=CLS, upc=1), s_(S_T)),
                                      I('mul', s_(S_T), AL(upc=1), R('ap', up=CLS, upc=1)),
                                      I('sub', R('rr', up=CLS, upc=1), R('rr', up=CLS, upc=1), s_(S_T))])]),
          L('zj', DF, [L('zjc', CLS, [I('div', R('zz', zj=CLS, zjc=1), R('rr', zj=CLS, zjc=1), R('FtF', zj=DF + 1))])]),
          L('n0', CLS, [I('set', RZN(n0=1), raw(0))]),
          L('n1', DF, [L('n1c', CLS, [I('mul', s_(S_T), R('rr', n1=CLS, n1c=1), R('zz', n1=CLS, n1c=1)),
                                      I('add', RZN(n1c=1), RZN(n1c=1), s_(S_T))])]),
          L('be', CLS, [I('add', RZ(be=1), RZ(be=1), k_(K_TINY)), I('div', AL(be=1), RZN(be=1), RZ(be=1)),
                        I('copy', RZ(be=1), RZN(be=1))]),
          L('pu', DF, [L('puc', CLS, [I('mul', s_(S_T), AL(puc=1), R('p', pu=CLS, puc=1)),
                                      I('add', R('p', pu=CLS, puc=1), R('zz', pu=CLS, puc=1), s_(S_T))])])]
    body.append(L('cg', cgit, it))
    qb = [L('qc', CLS, [I('set', s_(S_ACC), raw(0)),
                        L('qj', DF, [I('mul', s_(S_T), R('Fq', qn=DF, qj=1), R('W', qj=CLS, qc=1)),
                                     I('add', s_(S_ACC), s_(S_ACC), s_(S_T))]),
                        I('copy', R('sc', qn=CLS, qc=1), s_(S_ACC))]),
          I('copy', s_(S_BEST), R('sc', qn=CLS)), I('copy', s_(S_LAB), k_(K_CLS))]
    for c in range(1, CLS):
        qb += [I('cmp', s_(S_CD), s_(S_BEST), R('sc', c, qn=CLS)),
               I('select', s_(S_BEST), s_(S_CD), R('sc', c, qn=CLS), s_(S_BEST)),
               I('select', s_(S_LAB), s_(S_CD), k_(K_CLS + c), s_(S_LAB))]
    qb.append(I('copy', R('out', qn=1), s_(S_LAB)))
    body.append(L('qn', P['Q'], qb))
    body.append(L('so', P['Q'], [I('send', R('out', so=1))]))
    meta = {'learner': 'PCANet (two PCA filter banks, binary hashing, block histograms, primal ridge) lowered to the affine IL',
            'config': P, 'cg_iterations': cgit,
            'declared_deviations': [
                'the block histogram counts through a one-hot over 2^L2 bins instead of an indexed write',
                'both filter banks from subspace iteration instead of an eigensolver',
                'filter-bank covariance over the first npb images instead of a random subsample',
                'primal normal equations by Jacobi conjugate gradients instead of a factorisation',
                'non-overlapping blocks; the paper overlaps them by half']}
    return make_program(regions, body, meta)


def build_program(**kw):
    cgit = kw.pop('cgit', 200)
    features_only = kw.pop('features_only', False)
    body, regions, P = build(**kw, features_only=features_only)
    if features_only:
        body.append(L('so', P['N'] * P['DF'], [I('send', R('F', so=1))]))
        meta = {'learner': 'PCANet front end only: two filter banks, hashing and block histograms, no classifier',
                'config': P, 'declared_deviations': ['the one-hot histogram, the subspace-iteration filter banks',
                                                     'this program emits features, so it prices the front end alone']}
        return make_program(regions, body, meta)
    return finish(body, regions, P, cgit)


def mirror(x, y, q, L1=8, L2=6, kk=5, blk=14, stride=None, IN=28, npb=2000, rounds=3, seed=0, cgit=200, lam=1e-3, features_only=False):
    """float64 reference of exactly the arithmetic the program runs."""
    N, Q = len(x), len(q); KK = kk * kk; PAD = kk // 2; M2 = 1 << L2
    stride = stride or blk
    NBS = (IN - blk) // stride + 1; NB = NBS * NBS; DF = L1 * NB * M2
    xs = np.sqrt(np.clip(x, 0, None)).reshape(N, IN, IN)
    qs = np.sqrt(np.clip(q, 0, None)).reshape(Q, IN, IN)

    def pad(a):
        n = len(a); p = np.zeros((n, IN + 2 * PAD, IN + 2 * PAD)); p[:, PAD:PAD + IN, PAD:PAD + IN] = a; return p

    def patches(p):
        n = len(p); out = np.empty((n, IN, IN, KK))
        for a in range(kk):
            for c in range(kk):
                out[:, :, :, a * kk + c] = p[:, a:a + IN, c:c + IN]
        return out - out.mean(-1, keepdims=True)

    def subspace_ref(cov, nc):
        rng = np.random.default_rng(seed); W = rng.standard_normal((KK, nc)).astype(f32).astype(np.float64)
        den = np.zeros(nc)
        for _ in range(rounds):
            V = cov @ W
            for j in range(nc):
                for i in range(j):
                    V[:, j] -= (V[:, i] @ V[:, j]) / den[i] * V[:, i]
                den[j] = V[:, j] @ V[:, j]
            for j in range(nc):
                V[:, j] /= np.abs(V[:, j]).max() + 1e-6
            W = V.copy()
        return W

    px, pq = pad(xs), pad(qs)
    P1 = patches(px[:npb]).reshape(-1, KK)
    W1 = subspace_ref(P1.T @ P1, L1)

    def maps(p):
        return np.einsum('nhwp,pl->nlhw', patches(p), W1)

    M1 = maps(px[:npb])
    P2 = patches(pad(M1.reshape(-1, IN, IN))).reshape(-1, KK)
    W2 = subspace_ref(P2.T @ P2, L2)

    def feats(p):
        n = len(p); m1 = maps(p)
        B = np.einsum('nhwp,pj->nhwj', patches(pad(m1.reshape(-1, IN, IN))), W2).reshape(n, L1, IN, IN, L2)
        code = np.zeros((n, L1, IN, IN), dtype=np.int64)
        for d in range(L2):
            code = code * 2 + (B[..., d] > 0)
        F = np.zeros((n, L1, NBS, NBS, M2))
        for br in range(NBS):
            for bc in range(NBS):
                blkv = code[:, :, br * stride:br * stride + blk, bc * stride:bc * stride + blk].reshape(n, L1, -1)
                for i in range(n):
                    for l in range(L1):
                        F[i, l, br, bc] = np.bincount(blkv[i, l], minlength=M2)
        return F.reshape(n, DF) / (blk * blk)

    F, Fq = feats(px), feats(pq)
    if features_only: return F, Fq        # so the front end can be checked without the classifier
    Y = np.zeros((N, CLS)); Y[np.arange(N), y] = 1.0
    A0 = F.T @ F
    lam_eff = lam * np.trace(A0) / DF + 1e-6
    A = A0 + lam_eff * np.eye(DF); Bv = F.T @ Y
    d = 1.0 / np.diag(A)
    X = np.zeros((DF, CLS)); Rr = Bv.copy(); Zz = Rr * d[:, None]; Pp = Zz.copy(); rz = (Rr * Zz).sum(0)
    for _ in range(cgit):
        Ap = A @ Pp
        al = rz / ((Pp * Ap).sum(0) + 1e-6)
        X += al * Pp; Rr -= al * Ap
        Zz = Rr * d[:, None]; rzn = (Rr * Zz).sum(0)
        Pp = Zz + (rzn / (rz + 1e-6)) * Pp; rz = rzn
    return (Fq @ X).argmax(1)


def execute(document, tape_in, return_memory=False):
    program = Program(document); mem = np.zeros(program.words + 1, dtype=np.uint32); fmem = mem.view(f32)
    tape = iter(tape_in); out = []
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
        a = [int(v) for v in sys.argv[2:]] or [10000, 10000, 8, 6, 5, 14, 14]
        kw = dict(N=a[0], Q=a[1], L1=a[2], L2=a[3], kk=a[4], blk=a[5], stride=a[6])
        t0 = time.time(); doc = build_program(**kw); print('built', round(time.time() - t0), 's', flush=True)
        r = SC.score(doc)
        keep = {kk_: r[kk_] for kk_ in ('energy_mj', 'time_ms', 'total_executed_instructions', 'peak_allocated_scratch_bytes', 'memory_tiles')}
        print('SCORE', json.dumps(keep), kw, flush=True)
        (HERE / 'evidence' / f'pcanet_ir_score_L{a[2]}_{a[3]}.json').write_text(json.dumps({'args': kw, **keep}, indent=1))
    else:
        sys.path.insert(0, str(HERE.parents[1])); from mnist.code import data as ds
        n, qn = 120, 24; RAW = HERE.parents[0] / 'data/raw'
        PIX = ds.read_idx(RAW / 'train-images-idx3-ubyte.gz', 60000, True)
        LAB = ds.read_idx(RAW / 'train-labels-idx1-ubyte.gz', 60000, False)
        o = np.random.Generator(np.random.PCG64(20261121)).permutation(60000)
        tr, te = o[:n], o[10000:10000 + qn]
        xr = (PIX[tr].astype(f32) / f32(255)).reshape(n, 784)
        qr = (PIX[te].astype(f32) / f32(255)).reshape(qn, 784)
        kw = dict(N=n, Q=qn, L1=2, L2=3, kk=3, blk=14, stride=14, npb=40, cgit=80)
        doc = build_program(**kw)
        t0 = time.time(); out = np.array(execute(doc, tape_words(xr, LAB[tr], qr)))
        ref = mirror(xr, LAB[tr], qr, L1=2, L2=3, kk=3, blk=14, npb=40, cgit=80)
        print(f'reduced {n}/{qn} L1=2 L2=3 k=3: program correct {(out == LAB[te]).sum()}/{qn}, '
              f'mirror correct {(ref == LAB[te]).sum()}/{qn}, labels agree {(out == ref).sum()}/{qn}; '
              f'executor {time.time() - t0:.0f}s', flush=True)
