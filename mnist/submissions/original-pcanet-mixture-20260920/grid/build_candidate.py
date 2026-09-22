"""Experimental complete PCANet grid port; NOT an A100-equivalent result.

Keeps the task shape, histogram geometry and mixture hyperparameters. Arithmetic
uses the spatial ISA's ordered FP32 operations. Qualification is separate from
scoring and must never be inferred from the A100 predictions.
"""
from pathlib import Path
import argparse, copy, gzip, hashlib, json, sys
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE / 'shared'), str(HERE / 'imported')]
import numpy as np
from affine import ref as R, ins as I, loop as L, make_program
import pcanet_ir as P
import bag_ir as B
import fuse as F


def nodes(body):
    for node in body:
        yield node
        if 'loop' in node:
            yield from nodes(node['body'])


def unit_columns(vreg, pitch, cols, rows, module):
    """Normalize after max-entry scaling, so the norm is in [1,sqrt(rows)]."""
    acc = module.s_(module.S_ACC)
    temp = module.s_(module.S_T)
    norm = module.s_(module.S_MX)
    return [L('unit_col', cols, [I('set', acc, module.raw(0)),
        L('unit_row', rows, [I('mul', temp, R(vreg, unit_row=pitch, unit_col=1),
                              R(vreg, unit_row=pitch, unit_col=1)),
                            I('add', acc, acc, temp)]),
        *module.nr_sqrt(norm, acc),
        L('unit_div', rows, [I('div', R(vreg, unit_div=pitch, unit_col=1),
                              R(vreg, unit_div=pitch, unit_col=1), norm)])])]


def repeat_subspace(body, rounds, norms):
    result = []
    i = stage = 0
    while i < len(body):
        if body[i].get('loop') != 'vi':
            result.append(body[i]); i += 1; continue
        end = i
        while body[end].get('loop') != 'cw':
            end += 1
        block = body[i:end] + norms[stage] + [body[end]]
        result.append(L('subspace_iteration', rounds, block))
        stage += 1; i = end + 1
    assert stage == len(norms)
    return result


def build(N=60000, Q=10000, side=28, kernel=7, block=14, stride=7,
          L1=8, L2=5, K=100, components=8, filter_rounds=128, pca_rounds=256):
    P.NR = B.NR = 20
    B.KMSTEPS = B.EMSTEPS = 8
    nb = ((side - block) // stride + 1) ** 2
    dim = L1 * nb * (1 << L2)
    assert K <= dim and N >= 7 * (components - 1) + 1
    samples1 = min(N, 255)
    front = P.build_program(N=N, Q=Q, L1=L1, L2=L2, kk=kernel,
                            blk=block, stride=stride, IN=side,
                            npb=samples1, rounds=1, features_only=True)
    # Pixel normalization is already /255. Remove the old sqrt pixel map.
    for n in front['body']:
        if n.get('loop') == 'pn':
            leaf = n['body'][0]['body'][0]
            last = leaf['body'][-1]
            # The source of nr_sqrt's opening add is the original pixel.
            original = leaf['body'][0]['src'][0]
            leaf['body'] = [I('copy', last['dst'], original)]
        if n.get('op') == 'set' and n.get('dst') == P.k_(P.K_INVB):
            n['imm'] = P.raw(1)  # integer counts, normalized once below
    # W2 uses exactly the first 200,000 centered patches of the first 2,000
    # first-stage planes, as in the submitted learner (or all reduced data).
    cap = min(200000, min(N, 250) * L1 * side * side)
    whole_images, rem = divmod(cap, L1 * side * side)
    rebuilt = []
    for n in front['body']:
        if n.get('loop') != 'c2n':
            rebuilt.append(n); continue
        full = copy.deepcopy(n); full['count'] = whole_images
        rebuilt.append(full)
        if rem:
            partial = copy.deepcopy(n)
            partial['start'] = whole_images; partial['count'] = 1
            planes, pixels = divmod(rem, side * side)
            layer = partial['body'][-1]
            assert layer['loop'] == 'c2l'
            layer['count'] = planes
            if pixels:
                row_count, columns = divmod(pixels, side)
                tail = copy.deepcopy(layer)
                tail['start'] = planes; tail['count'] = 1
                row = tail['body'][0]; row['count'] = row_count
                if columns:
                    lastrow = copy.deepcopy(row)
                    lastrow['start'] = row_count; lastrow['count'] = 1
                    lastrow['body'][0]['count'] = columns
                    tail['body'].append(lastrow)
                partial['body'].append(tail)
            rebuilt.append(partial)
    front['body'] = repeat_subspace(rebuilt, filter_rounds, [
        unit_columns('V', max(L1, L2), L1, kernel * kernel, P),
        unit_columns('V', max(L1, L2), L2, kernel * kernel, P)])

    B.D = dim
    head = B.build(N, Q, components, K, components, K, N, rounds=1, nmem=1)
    head['body'] = repeat_subspace(head['body'], pca_rounds,
                                  [unit_columns('VA', K, K, dim, B)])
    # The head receives counts from the front. Use one global normalization,
    # followed by sqrt with an exact-zero result.
    denom = float(L1 * nb * block * block)
    for n in head['body']:
        if n.get('loop') in ('sx', 'sq'):
            var = n['loop']; reg = 'x' if var == 'sx' else 'q'
            dst = R(reg, **{var: 1})
            tmp = B.s_(B.S_TMP)
            n['body'] = [I('set', tmp, B.raw(denom)), I('div', dst, dst, tmp),
                         *B.nr_sqrt(tmp, dst),
                         I('cmp', B.s_(B.S_CD), B.k_(B.KZ), dst),
                         I('select', dst, B.s_(B.S_CD), tmp, B.k_(B.KZ))]
    # Fuse using a single label input; the historical fusion read labels twice.
    front['body'] = [n for n in front['body'] if n.get('loop') not in ('rl', 'yn')]
    regions, body, dropped = F.fuse(front, head)
    # Dead front-end label/Y regions were removed by the fuser. Input order is
    # images, queries, labels, consistent with the composed program's prologue.
    cfg = dict(N=N, Q=Q, side=side, kernel=kernel, block=block, stride=stride,
               L1=L1, L2=L2, blocks=nb, features=dim, K=K, components=components,
               filter_rounds=filter_rounds, pca_rounds=pca_rounds,
               kmeans_steps=8, em_steps=8, sqrt_steps=20,
               first_bank_patches=samples1*side*side, second_bank_patches=cap)
    return make_program(regions, body, {
        'status': 'experimental; numeric qualification required',
        'config': cfg,
        'tape_order': 'train images FP32 /255, test images FP32 /255, train labels uint32',
        'output': 'one uint32 predicted class per query; no query-label input',
        'differences_from_A100': [
            'Ordered FP32 covariance/reductions instead of FP64 filter covariance and CUDA reductions.',
            'Orthonormal subspace iteration instead of library eigendecomposition.',
            'Fixed global row offsets 7*j initialize components, rather than per-class seeded randperm.',
            'Gauss-Jordan inverses and primitive log/exp/sqrt approximations replace library operations.',
            'No A100 accuracy claim transfers to this grid program.'],
        'dropped_unused_front_regions': dropped})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reduced', action='store_true')
    args = parser.parse_args()
    kw = dict(N=60,Q=8,side=6,kernel=3,block=4,stride=1,L1=2,L2=2,
              K=4,components=2,filter_rounds=3,pca_rounds=3) if args.reduced else {}
    doc = build(**kw)
    data = json.dumps(doc, sort_keys=True, separators=(',', ':')).encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(gzip.compress(data,mtime=0))
    print(json.dumps({'program_sha256':hashlib.sha256(data).hexdigest(),
                      'json_bytes':len(data),'scratch_words':sum(r['words'] for r in doc['regions']),
                      'config':doc['metadata']['config']}),flush=True)
