#!/usr/bin/env python3
"""Generate and execute MNIST-small QDA in spatial-computer affine IL (ISA v4).

The submitted lowering uses unnormalized area-resized pixels, a masked pass
over class statistics, Gauss-Jordan inversion, and polynomial query scores.
Scratch initialization is charged; regions fully overwritten before use skip
zeroing. The executor checks output-label equality with reference.py, not
bitwise equality of intermediate floating-point computations. Only existing
set/recv/send/copy/add/sub/mul/div/cmp/select operations are used.
"""
import sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'grid-mlp-scoring-20260912'))
from affine import ref as R, ins as I, loop as L, make_program, expand, Program

f32 = np.float32
C, D = 10, 9
TRI = [(i, j) for i in range(D) for j in range(i, D)]
NT = len(TRI)
LOG_STEPS, SERIES_TERMS = 40, 8
LN2 = f32(0.6931471805599453)


def raw(value):
    return int(np.array(value, dtype=f32).view(np.uint32))


# scalar and constant slots
S_T, S_A, S_COND, S_BEST, S_LABEL, S_U, S_K, S_Z, S_Z2, S_TERM, S_ACC, S_V, S_W, S_LD, S_FCT, S_PIV, S_LAB = range(17)
K_ZERO, K_ONE, K_FOUR, K_HALF, K_TWO, K_HI, K_LO, K_LN2, K_N = range(9)
K_CLASS = 9                      # 10 raw integer class literals
K_ODD = 19                       # 1/(2m+1) divisors as FP32 literals 3,5,...,15
NK = K_ODD + SERIES_TERMS - 1


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


def build_qda(n_train=1000, n_test=1000, stage_x=False, stage_label=False, poly=False, normalize=True, region_order=None, skip_init=()):
    """stage_x: copy each training sample's pixels to a near buffer before the masked pass;
    stage_label: copy the sample's label to a near scalar; poly: score queries with the
    expanded polynomial (shared query products, per-class A', b, c) instead of d^T P d."""
    N, Q = n_train, n_test
    regions = [('s', 17), ('k', NK), ('q', D), ('d', D), ('sc', C), ('xs', D), ('mu', C * D), ('pk', C * NT), ('kap', C)]
    if poly: regions += [('pb', C * D), ('qq', NT)]
    regions += [('cnt', C), ('sum', C * D), ('mom', C * NT), ('S', D * D), ('Pw', D * D), ('prod', NT),
               ('x', N * D), ('labels', N)]
    if region_order is not None:
        # placement is nearest-first in declaration order: put the densest regions first
        rank = {n: i for i, n in enumerate(region_order)}
        regions.sort(key=lambda r: rank.get(r[0], len(rank)))
    body = []
    for region, words in regions:
        if region in skip_init: continue          # fully written before any read; the scorer's initialization proof accepts it
        body.append(L('init', words, [I('set', R(region, init=1), 0)]))
    consts = [raw(0), raw(1), raw(4), raw(.5), raw(2), raw(1.5), raw(.75), int(LN2.view(np.uint32)), raw(N)]
    consts += list(range(C)) + [raw(2 * m + 1) for m in range(1, SERIES_TERMS)]
    for i, v in enumerate(consts):
        body.append(I('set', k(i), v))
    body.append(L('recv_x', N * D, [I('recv', R('x', recv_x=1))]))
    body.append(L('recv_l', N, [I('recv', R('labels', recv_l=1))]))
    if normalize:
        v = R('x', norm_i=1)
        body.append(L('norm_i', N * D, [I('mul', v, v, k(K_FOUR)), I('sub', v, v, k(K_HALF))]))
    # --- masked statistics pass ---
    sample = []
    t0 = 0
    if stage_x:
        sample.append(L('cx', D, [I('copy', R('xs', cx=1), R('x', n=D, cx=1))]))
        xi = lambda i: R('xs', i); xj = lambda i: R('xs', i, pj=1); xf = R('xs', sf=1)
    else:
        xi = lambda i: R('x', i, n=D); xj = lambda i: R('x', i, n=D, pj=1); xf = R('x', n=D, sf=1)
    if stage_label:
        sample.append(I('copy', s(S_LAB), R('labels', n=1))); lab = s(S_LAB)
    else:
        lab = R('labels', n=1)
    for i in range(D):
        cnt = D - i
        sample.append(L('pj', cnt, [I('mul', R('prod', t0, pj=1), xi(i), xj(i))]))
        t0 += cnt
    for c in range(C):
        cb = [I('cmp', s(S_COND), lab, k(K_CLASS + c), predicate='eq'),
              I('select', s(S_T), s(S_COND), k(K_ONE), k(K_ZERO)), I('add', R('cnt', c), R('cnt', c), s(S_T))]
        cb.append(L('sf', D, [I('select', s(S_T), s(S_COND), xf, k(K_ZERO)),
                              I('add', R('sum', c * D, sf=1), R('sum', c * D, sf=1), s(S_T))]))
        cb.append(L('st', NT, [I('select', s(S_T), s(S_COND), R('prod', st=1), k(K_ZERO)),
                               I('add', R('mom', c * NT, st=1), R('mom', c * NT, st=1), s(S_T))]))
        sample += cb
    body.append(L('n', N, sample))
    # --- per-class parameters ---
    for c in range(C):
        cnt = R('cnt', c)
        body.append(L('mf', D, [I('div', R('mu', c * D, mf=1), R('sum', c * D, mf=1), cnt)]))
        for t, (i, j) in enumerate(TRI):
            body += [I('div', s(S_U), R('mom', c * NT + t), cnt), I('mul', s(S_V), R('mu', c * D + i), R('mu', c * D + j)),
                     I('sub', R('S', i * D + j), s(S_U), s(S_V))]
            if i != j: body.append(I('copy', R('S', j * D + i), R('S', i * D + j)))
        body.append(L('pw', D * D, [I('set', R('Pw', pw=1), raw(0))]))
        for i in range(D): body.append(I('set', R('Pw', i * D + i), raw(1)))
        body.append(I('set', s(S_LD), raw(0)))
        for p in range(D):
            body.append(I('copy', s(S_PIV), R('S', p * D + p)))
            body += log_body(S_PIV, S_W)
            body.append(I('add', s(S_LD), s(S_LD), s(S_W)))
            body.append(L('gj', D, [I('div', R('S', p * D, gj=1), R('S', p * D, gj=1), s(S_PIV)),
                                    I('div', R('Pw', p * D, gj=1), R('Pw', p * D, gj=1), s(S_PIV))]))
            for i in range(D):
                if i == p: continue
                body.append(I('copy', s(S_FCT), R('S', i * D + p)))
                body.append(L('ge', D, [I('mul', s(S_T), s(S_FCT), R('S', p * D, ge=1)), I('sub', R('S', i * D, ge=1), R('S', i * D, ge=1), s(S_T)),
                                        I('mul', s(S_T), s(S_FCT), R('Pw', p * D, ge=1)), I('sub', R('Pw', i * D, ge=1), R('Pw', i * D, ge=1), s(S_T))]))
        body.append(I('div', s(S_U), cnt, k(K_N)))
        body += log_body(S_U, S_W)
        body += [I('mul', s(S_T), k(K_HALF), s(S_LD)), I('sub', R('kap', c), s(S_W), s(S_T))]
        for t, (i, j) in enumerate(TRI):
            if i == j: body.append(I('copy', R('pk', c * NT + t), R('Pw', i * D + i)))
            else: body.append(I('add', R('pk', c * NT + t), R('Pw', i * D + j), R('Pw', i * D + j)))
        if poly:
            # b_c = P mu (9), c_c = kappa - 0.5 * mu^T b ; pk becomes the coefficients of q_i q_j: -0.5 P_ii, -P_ij
            for i in range(D):
                body.append(I('set', s(S_A), raw(0)))
                body.append(L('pb', D, [I('mul', s(S_T), R('Pw', i * D, pb=1), R('mu', c * D, pb=1)), I('add', s(S_A), s(S_A), s(S_T))]))
                body.append(I('copy', R('pb', c * D + i), s(S_A)))
            body.append(I('set', s(S_ACC), raw(0)))
            body.append(L('pc', D, [I('mul', s(S_T), R('mu', c * D, pc=1), R('pb', c * D, pc=1)), I('add', s(S_ACC), s(S_ACC), s(S_T))]))
            body += [I('mul', s(S_T), k(K_HALF), s(S_ACC)), I('sub', R('kap', c), R('kap', c), s(S_T))]
            for t, (i, j) in enumerate(TRI):
                body += [I('mul', s(S_T), k(K_HALF), R('pk', c * NT + t)), I('sub', R('pk', c * NT + t), k(K_ZERO), s(S_T))]
    # --- queries ---
    qb = [L('recv_q', D, [I('recv', R('q', recv_q=1))])]
    if normalize:
        v = R('q', nq=1)
        qb.append(L('nq', D, [I('mul', v, v, k(K_FOUR)), I('sub', v, v, k(K_HALF))]))
    if poly:
        t0 = 0
        for i in range(D):
            cnt = D - i
            qb.append(L('qj', cnt, [I('mul', R('qq', t0, qj=1), R('q', i), R('q', i, qj=1))]))
            t0 += cnt
    for c in range(C):
        if poly:
            qb.append(I('copy', s(S_ACC), R('kap', c)))
            qb.append(L('qt', NT, [I('mul', s(S_T), R('pk', c * NT, qt=1), R('qq', qt=1)), I('add', s(S_ACC), s(S_ACC), s(S_T))]))
            qb.append(L('qb', D, [I('mul', s(S_T), R('pb', c * D, qb=1), R('q', qb=1)), I('add', s(S_ACC), s(S_ACC), s(S_T))]))
            qb.append(I('copy', R('sc', c), s(S_ACC)))
        else:
          qb.append(L('df', D, [I('sub', R('d', df=1), R('q', df=1), R('mu', c * D, df=1))]))
          qb.append(I('set', s(S_ACC), raw(0)))
          t0 = 0
          for i in range(D):
            cnt = D - i
            qb.append(I('set', s(S_A), raw(0)))
            qb.append(L('qj', cnt, [I('mul', s(S_T), R('pk', c * NT + t0, qj=1), R('d', i, qj=1)), I('add', s(S_A), s(S_A), s(S_T))]))
            qb += [I('mul', s(S_T), R('d', i), s(S_A)), I('add', s(S_ACC), s(S_ACC), s(S_T))]
            t0 += cnt
          qb += [I('mul', s(S_T), k(K_HALF), s(S_ACC)), I('sub', R('sc', c), R('kap', c), s(S_T))]
        if c == 0:
            qb += [I('copy', s(S_BEST), R('sc', 0)), I('copy', s(S_LABEL), k(K_CLASS))]
        else:
            qb += [I('cmp', s(S_COND), s(S_BEST), R('sc', c)), I('select', s(S_BEST), s(S_COND), R('sc', c), s(S_BEST)),
                   I('select', s(S_LABEL), s(S_COND), k(K_CLASS + c), s(S_LABEL))]
    qb.append(I('send', s(S_LABEL)))
    body.append(L('query', Q, qb))
    metadata = {
        'algorithm': 'ordered-FP32 quadratic discriminant analysis (per-class Gaussian, full covariance, no shrinkage, class-frequency prior, log-determinant)',
        'classes': C, 'features': D, 'train_examples': N, 'test_examples': Q,
        'training': 'one masked pass in supplied order: counts, sums, packed second moments via cmp(eq)/select; mean = sum/count; S = M/count - mu mu^T; Gauss-Jordan inverse without pivoting; kappa = log(count/N) - 0.5*sum log(pivots)',
        'log': f'{LOG_STEPS} halving and {LOG_STEPS} doubling compare/select steps into [0.75, 1.5], then k*ln2 + 2*atanh((x-1)/(x+1)) with the series to z^{2*SERIES_TERMS-1}',
        'prediction': ('per query: share upper-triangle query products across classes; per class: constant + sum packed quadratic coefficients times query products + sum linear coefficients times query; first strict maximum wins' if poly else 'per query and class: d = q - mu; s = sum_i d_i * sum_{j>=i} P\'_ij d_j with P\' the packed upper triangle (off-diagonals doubled); score = kappa - 0.5*s; first strict maximum wins'),
        'tape': 'train pixels FP32 bits, train labels uint32, test pixels FP32 bits (one query streamed at a time)',
        'normalization': 'x*float32(4)-float32(0.5)' if normalize else 'none; tape supplies area-resized float32 pixels/255',
        'initialization': 'scratch zeroing and literals charged; regions in variant.skip_init fully written before any read; no learned or random initial state',
        'reference': 'reference.py in this directory (label-equal under the executor in spatial_program.py on all 11 official draws)',
        'variant': {'stage_x': stage_x, 'stage_label': stage_label, 'poly': poly, 'normalize': normalize, 'region_order': region_order, 'skip_init': list(skip_init)},
    }
    return make_program(regions, body, metadata)


# ------------------------------------------------------------------ executor
def execute(document, tape_in):
    """Run the expanded program.  tape_in: list of raw uint32 words.  Returns output raw words."""
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
    return out


def tape_words(train_pixels, train_labels, test_pixels):
    """Tape: FP32 bits of un-normalized 3x3 train pixels, uint32 labels, FP32 test pixels."""
    return (list(np.ascontiguousarray(train_pixels, dtype=f32).reshape(-1).view(np.uint32)) +
            [int(v) for v in train_labels] +
            list(np.ascontiguousarray(test_pixels, dtype=f32).reshape(-1).view(np.uint32)))


SUBMITTED = dict(poly=True, normalize=False, stage_x=True, skip_init=('k', 'xs', 'prod', 'labels', 'x'),
                 region_order=['s', 'k', 'xs', 'q', 'qq', 'prod', 'sc', 'cnt', 'sum', 'mom', 'pk', 'kap', 'pb', 'Pw', 'S', 'mu', 'labels', 'x', 'd'])
"""The submitted (v2) lowering: no input normalization (QDA is affine-invariant, so the
statistics are taken on the raw pixels/255), polynomial scoring with the query products
shared across classes, each training sample staged into a near buffer, and regions
declared in order of scratch-access density so the placement puts the hottest words
nearest the processor; the five regions that are fully written before any read (k, xs, prod,
labels, x) skip the charged zeroing, which the scorer's initialization proof accepts.  It emits
exactly the frozen predictions on all 11 draws."""


def build_submitted(n_train=1000, n_test=1000):
    return build_qda(n_train, n_test, **SUBMITTED)


if __name__ == '__main__':
    """Build the submitted program, score it with the shared scorer, write grid/program.spatial.json and grid/grid-score.json."""
    import hashlib, json, platform
    from score import score
    out = HERE / 'grid'
    out.mkdir(exist_ok=True)
    doc = build_submitted()
    res = score(doc)
    res['software'] = {'python': sys.version, 'numpy': np.__version__, 'platform': platform.platform()}
    shared = HERE.parent / 'grid-mlp-scoring-20260912'
    res['source_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (shared / 'affine.py', shared / 'score.py', Path(__file__))}
    program_file = out / 'program.spatial.json'
    program_file.write_text(json.dumps(doc, indent=2) + '\n')
    res['program_file_sha256'] = hashlib.sha256(program_file.read_bytes()).hexdigest()
    (out / 'grid-score.json').write_text(json.dumps(res, indent=2) + '\n')
    print(json.dumps({k: res[k] for k in ('energy_mj', 'time_ms', 'energy_fj', 'cycles', 'total_executed_instructions', 'peak_allocated_scratch_bytes', 'time_to_score_seconds')}, indent=1))
