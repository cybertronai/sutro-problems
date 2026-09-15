#!/usr/bin/env python3
"""PCA-QDA for MNIST-medium lowered to the spatial-computer affine IL (ISA v4), plus an
executor that runs the expanded program on a tape so the lowering can be checked bitwise
against reference.py.

Program: zero every region (charged) except those fully written before any read, set
literals (including the 81xK start basis W0), receive train pixels and labels, then

  pass A   for the first NP samples: stage the 81 pixels near, Newton-Raphson sqrt in
           place (6 steps), write the sqrt pixels back, accumulate the global sums and
           the 3,321 packed second moments in the processor tile;
  basis    m = sum/NP, S = M/NP - m m^T; ROUNDS rounds of V = S W, division-only
           Gram-Schmidt, max-|entry| column scaling, W = V;
  pass B1  every sample: stage (sqrt if not yet stored), subtract m, project onto the K
           basis columns (resident), store z, masked class counts and sums;
  pass B2  (stat_blocks passes) re-read z, packed K-dim products, masked per-class second
           moments accumulated in a resident block, copied out;
  params   per class: mean, shrunk covariance, Gauss-Jordan inverse, log-determinant,
           prior, packed precision;
  queries  receive all queries into the dead training buffer, stage/sqrt/project each
           into z; then one resident class at a time (packed precision, mean, kappa copied
           near): quadratic score of every stored query, running best score and label;
           send the labels.

Only set/recv/send/copy/add/sub/mul/div/cmp/select are used; there is no data-dependent
control flow.
"""
import sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'grid-mlp-scoring-20260912'))
sys.path.insert(0, str(HERE))
from affine import ref as R, ins as I, loop as L, make_program, expand, Program
import reference as REF

f32 = np.float32
C, D, K = REF.C, REF.D, REF.K
LOG_STEPS, SERIES_TERMS = REF.LOG_STEPS, REF.SERIES_TERMS
TRI81 = REF.TRI81; NT81 = len(TRI81)
TRI = REF.TRI; NT = len(TRI)


def raw(value):
    return int(np.array(value, dtype=f32).view(np.uint32))


# scalar and constant slots
(S_T, S_A, S_COND, S_U, S_K, S_Z, S_Z2, S_TERM, S_ACC, S_V, S_W, S_LD, S_FCT, S_PIV, S_Y, S_Q, S_F, S_MX, S_C2) = range(19)
NS = 19
(K_ZERO, K_ONE, K_HALF, K_TWO, K_HI, K_LO, K_LN2, K_N, K_NP, K_TINY) = range(10)
K_CLASS = 10                     # 10 raw integer class literals
K_ODD = 20                       # series divisors 3, 5, ..., 15 as FP32 literals
K_KEEP, K_SHR, K_KD = 27, 28, 29      # 1-shrink, shrink, K as FP32 (the ridge divides the trace by K exactly as the reference)
NK = 30


def s(i): return R('s', i)
def k(i): return R('k', i)


def log_body(src_slot, dst_slot):
    """FP32 log of scalar s[src] into s[dst] using scalars U,K,Z,Z2,TERM,ACC,T,V,COND."""
    u, kk, z, z2, term, acc, t, v, cond = (s(S_U), s(S_K), s(S_Z), s(S_Z2), s(S_TERM), s(S_ACC), s(S_T), s(S_V), s(S_COND))
    body = [I('copy', u, s(src_slot)), I('set', kk, raw(0))]
    body.append(L('lg_hi', LOG_STEPS, [I('cmp', cond, k(K_HI), u), I('mul', t, u, k(K_HALF)), I('select', u, cond, t, u),
                                       I('add', t, kk, k(K_ONE)), I('select', kk, cond, t, kk)]))
    body.append(L('lg_lo', LOG_STEPS, [I('cmp', cond, u, k(K_LO)), I('mul', t, u, k(K_TWO)), I('select', u, cond, t, u),
                                       I('sub', t, kk, k(K_ONE)), I('select', kk, cond, t, kk)]))
    body += [I('sub', t, u, k(K_ONE)), I('add', v, u, k(K_ONE)), I('div', z, t, v), I('mul', z2, z, z),
             I('copy', term, z), I('copy', acc, z)]
    for m in range(1, SERIES_TERMS):
        body += [I('mul', term, term, z2), I('div', t, term, k(K_ODD + m - 1)), I('add', acc, acc, t)]
    body += [I('mul', t, kk, k(K_LN2)), I('mul', v, k(K_TWO), acc), I('add', s(dst_slot), t, v)]
    return body


def sqrt_in_place(region, var):
    """Newton-Raphson sqrt of region[var] in place: y = x + tiny; NR_ITERATIONS x (q = x/y; q = y+q; y = half*q)."""
    x = R(region, **{var: 1}); y, q = s(S_Y), s(S_Q)
    body = [I('add', y, x, k(K_TINY))]
    for _ in range(REF.NR_ITERATIONS):
        body += [I('div', q, x, y), I('add', q, y, q), I('mul', y, k(K_HALF), q)]
    body.append(I('copy', x, y))
    return L(var, D, body)


def build_program(n_train=10000, n_test=10000, n_basis=REF.NP, stat_blocks=2, region_order=None):
    N, Q, NP = n_train, n_test, n_basis
    assert 0 < NP <= N
    BS = (NT + stat_blocks - 1) // stat_blocks
    NXQ = max(N, Q)
    regions = [('s', NS), ('k', NK), ('xs', D), ('zs', K), ('m', D), ('Wn', D * K), ('prod', BS), ('momb', C * BS),
               ('kapb', 1), ('mub', K), ('pkb', NT), ('gm', NT81), ('d', K), ('den', K), ('gsum', D), ('cnt', C),
               ('sum', C * K), ('kap', C), ('mu', C * K), ('pk', C * NT), ('best', Q), ('lab', Q), ('mom', C * NT),
               ('Sk', K * K), ('Pw', K * K), ('S', D * D), ('V', D * K), ('z', NXQ * K), ('x', NXQ * D), ('labels', N)]
    if region_order is not None:
        rank = {n: i for i, n in enumerate(region_order)}
        regions.sort(key=lambda r: rank.get(r[0], len(rank)))
    skip_init = ('k', 'xs', 'x', 'labels', 'Wn')      # fully written before any read (literals, staging copies, recv)
    body = []
    for region, words in regions:
        if region in skip_init: continue
        body.append(L('init', words, [I('set', R(region, init=1), 0)]))
    consts = [raw(0), raw(1), raw(.5), raw(2), raw(1.5), raw(.75), int(REF.LN2.view(np.uint32)), raw(N), raw(NP), int(REF.TINY.view(np.uint32))]
    consts += list(range(C)) + [raw(2 * m + 1) for m in range(1, SERIES_TERMS)] + [raw(1.0 - float(REF.SHRINK)), int(REF.SHRINK.view(np.uint32)), raw(K)]
    assert len(consts) == NK
    for i, v in enumerate(consts):
        body.append(I('set', k(i), v))
    W0 = REF.initial_basis()
    for i in range(D):
        for kk in range(K):
            body.append(I('set', R('Wn', i * K + kk), int(W0[i, kk].view(np.uint32))))
    body.append(L('recv_x', N * D, [I('recv', R('x', recv_x=1))]))
    body.append(L('recv_l', N, [I('recv', R('labels', recv_l=1))]))
    # --- pass A: sqrt features of the first NP samples (stored back), global sums and packed second moments ---
    sa = [L('cx', D, [I('copy', R('xs', cx=1), R('x', na=D, cx=1))]), sqrt_in_place('xs', 'sq'),
          L('wb', D, [I('copy', R('x', na=D, wb=1), R('xs', wb=1))]),
          L('gs', D, [I('add', R('gsum', gs=1), R('gsum', gs=1), R('xs', gs=1))])]
    t0 = 0
    for i in range(D):
        sa.append(L('pj', D - i, [I('mul', s(S_T), R('xs', i), R('xs', i, pj=1)), I('add', R('gm', t0, pj=1), R('gm', t0, pj=1), s(S_T))]))
        t0 += D - i
    body.append(L('na', NP, sa))
    # --- mean and covariance of the sqrt pixels ---
    body.append(L('mf', D, [I('div', R('m', mf=1), R('gsum', mf=1), k(K_NP))]))
    for t, (i, j) in enumerate(TRI81):
        body += [I('div', s(S_U), R('gm', t), k(K_NP)), I('mul', s(S_V), R('m', i), R('m', j)), I('sub', R('S', i * D + j), s(S_U), s(S_V))]
        if i != j: body.append(I('copy', R('S', j * D + i), R('S', i * D + j)))
    # --- subspace iteration ---
    for _ in range(REF.ROUNDS):
        body.append(L('vi', D, [L('vk', K, [I('set', s(S_ACC), raw(0)),
                                           L('vj', D, [I('mul', s(S_T), R('S', vi=D, vj=1), R('Wn', vj=K, vk=1)), I('add', s(S_ACC), s(S_ACC), s(S_T))]),
                                           I('copy', R('V', vi=K, vk=1), s(S_ACC))])]))
        for j in range(K):
            for i in range(j):
                body += [I('set', s(S_ACC), raw(0)),
                         L('gr', D, [I('mul', s(S_T), R('V', i, gr=K), R('V', j, gr=K)), I('add', s(S_ACC), s(S_ACC), s(S_T))]),
                         I('div', s(S_F), s(S_ACC), R('den', i)),
                         L('gu', D, [I('mul', s(S_T), s(S_F), R('V', i, gu=K)), I('sub', R('V', j, gu=K), R('V', j, gu=K), s(S_T))])]
            body += [I('set', s(S_ACC), raw(0)),
                     L('gd', D, [I('mul', s(S_T), R('V', j, gd=K), R('V', j, gd=K)), I('add', s(S_ACC), s(S_ACC), s(S_T))]),
                     I('copy', R('den', j), s(S_ACC))]
        for j in range(K):
            body += [I('set', s(S_MX), raw(0)),
                     L('sa', D, [I('sub', s(S_T), k(K_ZERO), R('V', j, sa=K)), I('cmp', s(S_COND), R('V', j, sa=K), k(K_ZERO)),
                                 I('select', s(S_A), s(S_COND), s(S_T), R('V', j, sa=K)),
                                 I('cmp', s(S_C2), s(S_MX), s(S_A)), I('select', s(S_MX), s(S_C2), s(S_A), s(S_MX))]),
                     L('sd', D, [I('div', R('V', j, sd=K), R('V', j, sd=K), s(S_MX))])]
        body.append(L('cw', D * K, [I('copy', R('Wn', cw=1), R('V', cw=1))]))
    # --- pass B1: projection of every sample, masked class counts and sums ---
    def projection(var, x_offset, need_sqrt):
        b = [L('cx', D, [I('copy', R('xs', cx=1), R('x', x_offset, **{var: D}, cx=1))])]
        if need_sqrt: b.append(sqrt_in_place('xs', 'sq'))
        b.append(L('cm', D, [I('sub', R('xs', cm=1), R('xs', cm=1), R('m', cm=1))]))
        for kk in range(K):
            b += [I('set', s(S_ACC), raw(0)),
                  L('pi', D, [I('mul', s(S_T), R('xs', pi=1), R('Wn', kk, pi=K)), I('add', s(S_ACC), s(S_ACC), s(S_T))]),
                  I('copy', R('zs', kk), s(S_ACC))]
        b.append(L('cz', K, [I('copy', R('z', x_offset // D * K, **{var: K}, cz=1), R('zs', cz=1))]))
        return b
    def class_sums(var, lab_offset):
        b = []
        for c in range(C):
            b += [I('cmp', s(S_COND), R('labels', lab_offset, **{var: 1}), k(K_CLASS + c), predicate='eq'),
                  I('select', s(S_T), s(S_COND), k(K_ONE), k(K_ZERO)), I('add', R('cnt', c), R('cnt', c), s(S_T)),
                  L('sf', K, [I('select', s(S_T), s(S_COND), R('zs', sf=1), k(K_ZERO)), I('add', R('sum', c * K, sf=1), R('sum', c * K, sf=1), s(S_T))])]
        return b
    body.append(L('nb1', NP, projection('nb1', 0, False) + class_sums('nb1', 0)))
    if N > NP:
        body.append(L('nb2', N - NP, projection('nb2', NP * D, True) + class_sums('nb2', NP)))
    # --- pass B2: masked packed second moments of z, stat_blocks resident blocks ---
    for blk in range(stat_blocks):
        t_lo, t_hi = blk * BS, min(NT, (blk + 1) * BS)
        sb = [L('cz', K, [I('copy', R('zs', cz=1), R('z', nc=K, cz=1))])]
        t = 0
        for i in range(K):
            cnt = K - i
            lo, hi = max(t, t_lo), min(t + cnt, t_hi)
            if lo < hi:
                sb.append(L('pj', hi - lo, [I('mul', R('prod', lo - t_lo, pj=1), R('zs', i), R('zs', i + (lo - t), pj=1))]))
            t += cnt
        for c in range(C):
            sb += [I('cmp', s(S_COND), R('labels', nc=1), k(K_CLASS + c), predicate='eq'),
                   L('st', t_hi - t_lo, [I('select', s(S_T), s(S_COND), R('prod', st=1), k(K_ZERO)),
                                         I('add', R('momb', c * BS, st=1), R('momb', c * BS, st=1), s(S_T))])]
        body.append(L('nc', N, sb))
        for c in range(C):
            body.append(L('mc', t_hi - t_lo, [I('copy', R('mom', c * NT + t_lo, mc=1), R('momb', c * BS, mc=1))]))
            body.append(L('mz', t_hi - t_lo, [I('set', R('momb', c * BS, mz=1), raw(0))]))
    # --- per-class parameters (K dims) ---
    for c in range(C):
        cnt = R('cnt', c)
        body.append(L('mf', K, [I('div', R('mu', c * K, mf=1), R('sum', c * K, mf=1), cnt)]))
        for t, (i, j) in enumerate(TRI):
            body += [I('div', s(S_U), R('mom', c * NT + t), cnt), I('mul', s(S_V), R('mu', c * K + i), R('mu', c * K + j)),
                     I('sub', R('Sk', i * K + j), s(S_U), s(S_V))]
            if i != j: body.append(I('copy', R('Sk', j * K + i), R('Sk', i * K + j)))
        body.append(I('set', s(S_ACC), raw(0)))
        for i in range(K): body.append(I('add', s(S_ACC), s(S_ACC), R('Sk', i * K + i)))
        body += [I('div', s(S_U), s(S_ACC), k(K_KD)), I('mul', s(S_U), k(K_SHR), s(S_U))]      # ridge = shrink * (tr / K) in S_U
        for t, (i, j) in enumerate(TRI):
            body.append(I('mul', s(S_V), k(K_KEEP), R('Sk', i * K + j)))
            if i == j: body.append(I('add', s(S_V), s(S_V), s(S_U)))
            body.append(I('copy', R('Sk', i * K + j), s(S_V)))
            if i != j: body.append(I('copy', R('Sk', j * K + i), s(S_V)))
        body.append(L('pw', K * K, [I('set', R('Pw', pw=1), raw(0))]))
        for i in range(K): body.append(I('set', R('Pw', i * K + i), raw(1)))
        body.append(I('set', s(S_LD), raw(0)))
        for p in range(K):
            body.append(I('copy', s(S_PIV), R('Sk', p * K + p)))
            body += log_body(S_PIV, S_W)
            body.append(I('add', s(S_LD), s(S_LD), s(S_W)))
            body.append(L('gj', K, [I('div', R('Sk', p * K, gj=1), R('Sk', p * K, gj=1), s(S_PIV)),
                                    I('div', R('Pw', p * K, gj=1), R('Pw', p * K, gj=1), s(S_PIV))]))
            for i in range(K):
                if i == p: continue
                body.append(I('copy', s(S_FCT), R('Sk', i * K + p)))
                body.append(L('ge', K, [I('mul', s(S_T), s(S_FCT), R('Sk', p * K, ge=1)), I('sub', R('Sk', i * K, ge=1), R('Sk', i * K, ge=1), s(S_T)),
                                        I('mul', s(S_T), s(S_FCT), R('Pw', p * K, ge=1)), I('sub', R('Pw', i * K, ge=1), R('Pw', i * K, ge=1), s(S_T))]))
        body.append(I('div', s(S_U), cnt, k(K_N)))
        body += log_body(S_U, S_W)
        body += [I('mul', s(S_T), k(K_HALF), s(S_LD)), I('sub', R('kap', c), s(S_W), s(S_T))]
        for t, (i, j) in enumerate(TRI):
            if i == j: body.append(I('copy', R('pk', c * NT + t), R('Pw', i * K + i)))
            else: body.append(I('add', R('pk', c * NT + t), R('Pw', i * K + j), R('Pw', i * K + j)))
    # --- queries: receive all, sqrt and project each into z ---
    body.append(L('recv_q', Q * D, [I('recv', R('x', recv_q=1))]))
    body.append(L('qp', Q, projection('qp', 0, True)))
    # --- resident-class scoring of all stored queries ---
    for c in range(C):
        body.append(L('cpk', NT, [I('copy', R('pkb', cpk=1), R('pk', c * NT, cpk=1))]))
        body.append(L('cmu', K, [I('copy', R('mub', cmu=1), R('mu', c * K, cmu=1))]))
        body.append(I('copy', R('kapb', 0), R('kap', c)))
        qb = [L('df', K, [I('sub', R('d', df=1), R('z', qs=K, df=1), R('mub', df=1))]), I('set', s(S_ACC), raw(0))]
        t0 = 0
        for i in range(K):
            cnt = K - i
            qb.append(I('set', s(S_A), raw(0)))
            qb.append(L('qj', cnt, [I('mul', s(S_T), R('pkb', t0, qj=1), R('d', i, qj=1)), I('add', s(S_A), s(S_A), s(S_T))]))
            qb += [I('mul', s(S_T), R('d', i), s(S_A)), I('add', s(S_ACC), s(S_ACC), s(S_T))]
            t0 += cnt
        qb += [I('mul', s(S_T), k(K_HALF), s(S_ACC)), I('sub', s(S_V), R('kapb', 0), s(S_T))]
        if c == 0:
            qb += [I('copy', R('best', qs=1), s(S_V)), I('copy', R('lab', qs=1), k(K_CLASS))]
        else:
            qb += [I('cmp', s(S_COND), R('best', qs=1), s(S_V)), I('select', R('best', qs=1), s(S_COND), s(S_V), R('best', qs=1)),
                   I('select', R('lab', qs=1), s(S_COND), k(K_CLASS + c), R('lab', qs=1))]
        body.append(L('qs', Q, qb))
    body.append(L('send_q', Q, [I('send', R('lab', send_q=1))]))
    metadata = {
        'algorithm': 'ordered-FP32 PCA-QDA: Newton-Raphson sqrt features, K-dimensional basis by subspace iteration on the global covariance of the first NP samples (fixed literal start, division-only Gram-Schmidt, max-|entry| scaling), quadratic discriminant analysis with trace shrinkage on the projected features',
        'classes': C, 'features': D, 'basis_dims': K, 'basis_samples': NP, 'rounds': REF.ROUNDS, 'sqrt_steps': REF.NR_ITERATIONS,
        'train_examples': N, 'test_examples': Q, 'shrinkage': float(REF.SHRINK), 'stat_blocks': stat_blocks,
        'training': 'pass A: sqrt of the first NP samples stored back, global sums and 3,321 packed second moments; m, S; ROUNDS x (V = S W, Gram-Schmidt without normalization, column max-abs scaling); pass B1: every sample sqrt (if not stored), minus m, projected onto the resident basis, z stored, masked class counts/sums; pass B2: masked packed second moments of z in resident blocks; per class mean, S = M/count - mu mu^T, S <- (1-shrink) S + shrink (tr S / K) I, Gauss-Jordan inverse without pivoting, kappa = log(count/N) - 0.5 sum log(pivots)',
        'log': f'{LOG_STEPS} halving and {LOG_STEPS} doubling compare/select steps into [0.75, 1.5], then k*ln2 + 2*atanh((x-1)/(x+1)) with the series to z^{2*SERIES_TERMS-1}',
        'prediction': 'each query sqrt, minus m, projected into z; per class (resident packed precision, mean, kappa): d = z - mu; s = sum_i d_i * sum_{j>=i} P\'_ij d_j; score = kappa - 0.5*s; running best per query, first strict maximum wins; labels sent last',
        'tape': 'train pixels FP32 bits (image/255, 9x9 area resize), train labels uint32, test pixels FP32 bits',
        'normalization': 'none beyond /255; the square-root feature map and max-abs basis scaling are fixed parts of the learner, including its trace shrinkage',
        'initialization': 'all scratch explicitly zeroed (charged) except literals, recv targets and staging buffers written before their first read',
        'reference': 'reference.py in this directory (bitwise equal parameters and labels under the executor at reduced size: validate_reduced.py)',
        'variant': {'stat_blocks': stat_blocks, 'region_order': region_order},
    }
    return make_program(regions, body, metadata)


# ------------------------------------------------------------------ executor
def execute(document, tape_in, return_memory=False):
    """Run the expanded program.  tape_in: list of raw uint32 words.  Returns output raw words (and memory)."""
    program = Program(document)
    mem = np.zeros(program.words + 1, dtype=np.uint32)
    fmem = mem.view(f32)
    tape = iter(tape_in); out = []
    for op in expand(document):
        code = op[0]
        if code == 'set': mem[op[1]] = np.uint32(op[2])
        elif code == 'recv': mem[op[1]] = np.uint32(next(tape))
        elif code == 'send': out.append(int(mem[op[1]]))
        elif code == 'copy': mem[op[1]] = mem[op[2]]
        elif code == 'add': fmem[op[1]] = fmem[op[2]] + fmem[op[3]]
        elif code == 'sub': fmem[op[1]] = fmem[op[2]] - fmem[op[3]]
        elif code == 'mul': fmem[op[1]] = fmem[op[2]] * fmem[op[3]]
        elif code == 'div': fmem[op[1]] = fmem[op[2]] / fmem[op[3]]
        elif code == 'cmp': mem[op[1]] = np.uint32(1 if fmem[op[2]] < fmem[op[3]] else 0)
        elif code == 'cmp_eq': mem[op[1]] = np.uint32(1 if mem[op[2]] == mem[op[3]] else 0)
        elif code == 'select': mem[op[1]] = mem[op[3]] if mem[op[2]] != 0 else mem[op[4]]
        else: raise ValueError(code)
    if return_memory:
        return out, {name: fmem[base:base + words].copy() for name, (base, words) in program.regions.items()}
    return out


def tape_words(train_pixels, train_labels, test_pixels):
    """Tape: FP32 bits of the 9x9 train pixels (/255), uint32 labels, FP32 test pixels."""
    return (list(np.ascontiguousarray(train_pixels, dtype=f32).reshape(-1).view(np.uint32)) +
            [int(v) for v in train_labels] +
            list(np.ascontiguousarray(test_pixels, dtype=f32).reshape(-1).view(np.uint32)))


def compiled_source(document):
    """Emit a numeric C executor directly from the validated affine loop tree.

    This executes the same scalar instructions and loop order as execute(); it
    does not compute spatial costs. Compile with GCC, -O0 -ffp-contract=off
    -fno-fast-math -frounding-math -fexcess-precision=standard -mfpmath=sse.
    Every arithmetic result is stored as float32 before the next instruction.
    """
    import hashlib
    import json
    program = Program(document)
    digest = hashlib.sha256(json.dumps(document, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    n_input = program.instructions.get('recv', 0)
    n_output = program.instructions.get('send', 0)
    lines = [r'''#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <fenv.h>
#include <xmmintrin.h>
typedef union { uint32_t u; float f; } Word;
static Word *m;
static uint32_t *tape, *output;
static size_t ti, oi;
''']
    counter = 0

    def address(operand, scope):
        base = program.regions[operand['region']][0] + operand['offset']
        return '(' + str(base) + ''.join(f'+({coefficient})*{scope[var]}'
                                         for var, coefficient in operand['coefficients'].items()) + ')'

    def emit(body, scope):
        nonlocal counter
        for node in body:
            if 'loop' in node:
                var = f'v{counter}'; counter += 1
                first, stop = node['start'], node['start'] + node['count']
                lines.append(f'for (int64_t {var}={first}; {var}<{stop}; ++{var}) {{')
                emit(node['body'], {**scope, node['loop']: var})
                lines.append('}')
                continue
            op = node['op']
            src = [address(operand, scope) for operand in node.get('src', [])]
            dst = address(node['dst'], scope) if 'dst' in node else None
            if op == 'set': expression = f'm[{dst}].u=UINT32_C({node["imm"]});'
            elif op == 'recv': expression = f'm[{dst}].u=tape[ti++];'
            elif op == 'send': expression = f'output[oi++]=m[{src[0]}].u;'
            elif op == 'copy': expression = f'm[{dst}].u=m[{src[0]}].u;'
            elif op in ('add', 'sub', 'mul', 'div'):
                symbol = {'add': '+', 'sub': '-', 'mul': '*', 'div': '/'}[op]
                expression = f'm[{dst}].f=m[{src[0]}].f{symbol}m[{src[1]}].f;'
            elif op == 'cmp' and node.get('predicate', 'lt') == 'lt':
                expression = f'm[{dst}].u=(m[{src[0]}].f<m[{src[1]}].f);'
            elif op == 'cmp' and node.get('predicate') == 'eq':
                expression = f'm[{dst}].u=(m[{src[0]}].u==m[{src[1]}].u);'
            elif op == 'select':
                expression = f'm[{dst}].u=m[{src[0]}].u?m[{src[1]}].u:m[{src[2]}].u;'
            else: raise ValueError(f'Unsupported compiled-executor opcode: {op}')
            lines.append(expression)

    # Bound compiler function size without expanding any affine loop.
    chunks = []
    for first in range(0, len(document['body']), 64):
        name = f'chunk{len(chunks)}'; chunks.append(name)
        lines.append(f'static void {name}(void) {{')
        emit(document['body'][first:first + 64], {})
        lines.append('}')
    lines.append(f'''int main(int argc, char **argv) {{
if (argc!=4 || sizeof(float)!=4 || FLT_RADIX!=2 || FLT_MANT_DIG!=24) return 2;
uint32_t endian=1; if (*(unsigned char *)&endian!=1) return 3;
if (fesetround(FE_TONEAREST)) return 4;
_mm_setcsr(_mm_getcsr() & ~UINT32_C(0x8040)); /* disable FTZ and DAZ */
m=malloc({program.words + 1}*sizeof(Word));
tape=malloc({max(1, n_input)}*sizeof(uint32_t));
output=malloc({max(1, n_output)}*sizeof(uint32_t));
if (!m || !tape || !output) return 5;
for (size_t i=0;i<{program.words + 1};++i) m[i].u=UINT32_C(0x7fc00001);
FILE *f=fopen(argv[1],"rb"); if (!f) return 6;
if (fread(tape,4,{n_input},f)!={n_input} || fgetc(f)!=EOF) return 7;
fclose(f);
''')
    lines.extend(f'{name}();' for name in chunks)
    lines.append(f'''if (ti!={n_input} || oi!={n_output}) return 8;
f=fopen(argv[2],"wb"); if (!f) return 9;
if (fwrite(output,4,{n_output},f)!={n_output} || fclose(f)) return 10;
f=fopen(argv[3],"wb"); if (!f) return 11;
if (fwrite(m,4,{program.words + 1},f)!={program.words + 1} || fclose(f)) return 12;
puts("{digest}");
free(m); free(tape); free(output); return 0;
}}
''')
    return '\n'.join(lines)


def execute_compiled(document, tape_in, executable, workdir=None):
    """Run a compiled_source executable and return labels plus final regions."""
    import hashlib
    import json
    import subprocess
    import tempfile
    program = Program(document)
    digest = hashlib.sha256(json.dumps(document, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    with tempfile.TemporaryDirectory(prefix='pca-qda-execute-', dir=workdir) as folder:
        folder = Path(folder)
        tape_path, output_path, memory_path = (folder / name for name in ('tape.bin', 'output.bin', 'memory.bin'))
        np.asarray(tape_in, dtype='<u4').tofile(tape_path)
        result = subprocess.run([str(Path(executable).resolve()), str(tape_path), str(output_path), str(memory_path)],
                                check=True, capture_output=True, text=True)
        if result.stdout.strip() != digest:
            raise ValueError('Compiled executor program hash differs from requested document')
        output = np.fromfile(output_path, dtype='<u4')
        memory = np.fromfile(memory_path, dtype='<u4')
        if len(output) != program.instructions.get('send', 0) or len(memory) != program.words + 1:
            raise ValueError('Compiled executor output or memory length differs')
        memory = memory.view(f32)
        return output.tolist(), {name: memory[base:base + words].copy()
                                 for name, (base, words) in program.regions.items()}


SUBMITTED = dict(stat_blocks=3, region_order=['s', 'k', 'xs', 'zs', 'd', 'kapb', 'mub', 'pkb', 'm', 'Wn', 'prod', 'momb', 'gm', 'den', 'gsum', 'cnt', 'sum', 'kap', 'mu', 'pk', 'best', 'lab', 'mom', 'Sk', 'Pw', 'S', 'V', 'z', 'x', 'labels'])
"""The submitted placement: three resident blocks for the K-dim class moments (the 8,200-word table does
not fit the processor's tile next to the basis), the per-query difference and the resident class
parameters nearest the processor (they are read 820 and 40 times per query and class), then the
basis and the training accumulators. The source authors selected this fixed placement in a sweep."""


def build_submitted(n_train=10000, n_test=10000):
    return build_program(n_train, n_test, **SUBMITTED)


if __name__ == '__main__':
    import argparse, gzip, hashlib, json, platform
    from score import score
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--emit-c', type=Path, help='Write a numeric C executor instead of scoring')
    parser.add_argument('--n-train', type=int, default=10000)
    parser.add_argument('--n-test', type=int, default=10000)
    parser.add_argument('--n-basis', type=int, default=REF.NP)
    args = parser.parse_args()
    if args.emit_c is not None:
        args.emit_c.parent.mkdir(parents=True, exist_ok=True)
        args.emit_c.write_text(compiled_source(build_program(args.n_train, args.n_test, args.n_basis, **SUBMITTED)))
        raise SystemExit(0)
    if (args.n_train, args.n_test, args.n_basis) != (10000, 10000, REF.NP):
        parser.error('Reduced dimensions apply only to --emit-c')
    out = HERE / 'grid'; out.mkdir(exist_ok=True)
    doc = build_submitted(); res = score(doc)
    res['software'] = {'python': sys.version, 'numpy': np.__version__, 'platform': platform.platform()}
    shared = HERE.parent / 'grid-mlp-scoring-20260912'
    res['source_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (shared / 'affine.py', shared / 'score.py', Path(__file__), HERE / 'reference.py')}
    program_bytes = (json.dumps(doc, separators=(',', ':')) + '\n').encode()
    (out / 'program.spatial.json.gz').write_bytes(gzip.compress(program_bytes, mtime=0))
    res['program_file_sha256'] = hashlib.sha256(program_bytes).hexdigest()
    (out / 'grid-score.json').write_text(json.dumps(res, indent=2) + '\n')
    print(json.dumps({k: res[k] for k in ('energy_mj', 'time_ms', 'energy_fj', 'cycles', 'total_executed_instructions', 'peak_allocated_scratch_bytes', 'memory_tiles', 'max_tile_scratch_words', 'time_to_score_seconds')}, indent=1))
