#!/usr/bin/env python3
"""Fuse any feature-producing front end with any bag_ir head into one program.

fused_ir.py did this by hand for the scattering front end. The machinery turned out to be generic, so
this is the same thing driven by the documents themselves: affine-IL operands are plain dicts carrying a
region name, so a front end's regions can be renamed out of the head's way, its feature output pointed at
the head's input buffer, its trailing `send` dropped and the head's two `recv` loops dropped.

Three things are not obvious and each cost a failed build:

  * The initialization verifier walks the body until every word is proved initialized and gives up after
    2*words + 100,000 visits. BOTH programs' prologues - zero-inits, constants, receives - must precede
    either program's arithmetic, or the walk burns its budget on the front end's convolutions and never
    reaches the head's constants.
  * The feature buffer must stay explicitly zeroed even though the front end fills it completely: it is
    written by a nest the verifier cannot walk cheaply, the same reason bag_ir zeroes x and q under
    `insize`.
  * A front end built to emit features may still DECLARE the regions of the classifier it is not running
    (PCANet's features_only path declares a 1,024 x 1,024 normal-equations matrix it never touches).
    Those are found by looking for regions nothing but their own zero-init refers to, and dropped.

python fuse.py scatter|pcanet|pcanet_qda [N Q]
"""
import os, copy, json, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[0] / 'submissions' / 'grid-mlp-scoring-20260912'))
from affine import make_program
import bag_ir as BG
f32 = np.float32


def walk(node, fn):
    if isinstance(node, list):
        for n in node: walk(n, fn)
    elif isinstance(node, dict):
        if 'region' in node: fn(node)
        elif 'loop' in node: walk(node['body'], fn)
        else:
            for key in ('dst', 'src'):
                if key in node: walk(node[key], fn)
    return node


def loop_vars(body, acc=None):
    acc = set() if acc is None else acc
    if isinstance(body, list):
        for n in body: loop_vars(n, acc)
    elif isinstance(body, dict) and 'loop' in body:
        acc.add(body['loop']); loop_vars(body['body'], acc)
    return acc


def has_op(node, op):
    if isinstance(node, list): return any(has_op(n, op) for n in node)
    if not isinstance(node, dict): return False
    if node.get('op') == op: return True
    return has_op(node['body'], op) if 'loop' in node else False


def is_init(b):
    """a top-level loop whose whole body is one immediate store: a zero-init of one region"""
    if not (isinstance(b, dict) and 'loop' in b): return None
    inner = b['body'][0] if isinstance(b.get('body'), list) and len(b['body']) == 1 else None
    if isinstance(inner, dict) and inner.get('op') == 'set' and isinstance(inner.get('dst'), dict):
        return inner['dst'].get('region')
    return None


def fuse(front, head, feat=('F', 'Fq'), prefix='f_'):
    fbody = [b for b in front['body'] if not has_op(b, 'send')]        # the front end no longer sends
    hbody = [b for b in head['body'] if not has_op(b, 'recv')] \
        + [b for b in head['body'] if has_op(b, 'recv')]               # keep the head's label recv, ordered below
    hbody = [b for b in head['body']]                                  # (restored; the recv filter is by loop var)
    hrecv_drop = set()
    for b in head['body']:
        if isinstance(b, dict) and 'loop' in b and has_op(b, 'recv'):
            # the head receives features and labels; only the feature receives go away
            regs = set()
            walk(b['body'], lambda r: regs.add(r['region']))
            if regs & {'x', 'q'}: hrecv_drop.add(id(b))
    hbody = [b for b in head['body'] if id(b) not in hrecv_drop]

    # which front-end regions does anything other than a zero-init touch?
    used = set()
    for b in fbody:
        if is_init(b): continue
        walk(b, lambda r: used.add(r['region']))
    fdrop = {r['name'] for r in front['regions']
             if r['name'] not in used and r['name'] not in feat}       # declared but never really used
    fbody = [b for b in fbody if is_init(b) not in fdrop]

    LV = {v: prefix + v for v in loop_vars(fbody)}
    RM = {feat[0]: 'x', feat[1]: 'q'}
    for r in front['regions']:
        if r['name'] not in RM: RM[r['name']] = prefix + r['name']

    def rename(node):
        def fn(ref):
            ref['region'] = RM.get(ref['region'], ref['region'])
            co = ref.get('coefficients') or {}
            ref['coefficients'] = {LV.get(kk, kk): vv for kk, vv in co.items()}
        walk(node, fn)
        def relabel(n):
            if isinstance(n, list):
                for m in n: relabel(m)
            elif isinstance(n, dict) and 'loop' in n:
                n['loop'] = LV.get(n['loop'], n['loop']); relabel(n['body'])
        relabel(node); return node

    fbody = [rename(copy.deepcopy(b)) for b in fbody]
    fregions = [(RM[r['name']], r['words']) for r in front['regions']
                if r['name'] not in fdrop and r['name'] not in feat]   # x and q come from the head
    regions = fregions + [(r['name'], r['words']) for r in head['regions']]

    def is_prologue_shaped(b):
        if not isinstance(b, dict): return False
        if 'loop' not in b: return b.get('op') == 'set'
        return is_init(b) is not None or has_op(b, 'recv')

    def split(bs):
        """the prologue is the LEADING RUN of setup nodes, not every setup-shaped node anywhere.

        PCANet re-seeds its subspace iteration with a loop whose body is a single immediate store - the
        same shape as a zero-init - in the middle of its computation. Filtering the whole body for that
        shape hoists that re-seed to the front, and the filter bank then comes out wrong while every
        stage before it still matches. Taking a prefix cannot make that mistake."""
        i = 0
        while i < len(bs) and is_prologue_shaped(bs[i]): i += 1
        return bs[:i], bs[i:]

    fpro, frest = split(fbody); hpro, hrest = split(hbody)
    body = fpro + hpro + frest + hrest
    return regions, body, sorted(fdrop)


def build(kind, N=10000, Q=10000, k=8, K=60, npb=2000, npb_front=2000, order=None):
    if kind == 'scatter':
        import scatter_sub_ir as SC
        keep = json.load(open(HERE / 'evidence' / 'paths_P108.json'))
        DF = len(keep) * 16
        front = SC.build(N, Q, keep=keep, single_canvas=True)
        label = f'subsampled scattering, {len(keep)} paths'
    else:
        import pcanet_ir as P
        DF = 1024
        front = P.build_program(N=N, Q=Q, L1=8, L2=5, kk=7, blk=14, stride=14,
                                npb=min(npb_front, N), features_only=True)
        label = 'PCANet, 8 filters 5 bits 4 blocks'
    BG.D = DF
    head = BG.build(N, Q, k, K, k, K, npb, nmem=1)
    regions, body, dropped = fuse(front, head)
    if order:
        rank = {n: i for i, n in enumerate(order)}
        regions = sorted(regions, key=lambda rw: rank.get(rw[0], len(rank)))
    meta = {'learner': f'FUSED {label} ({DF} features) + mixture QDA k={k} K={K}',
            'config': {'N': N, 'Q': Q, 'DF': DF, 'k': k, 'K': K, 'dropped_unused_regions': dropped},
            'tape_order': 'train images, test images, train labels',
            'declared_deviations': front['metadata'].get('declared_deviations', [])
                                   + head['metadata'].get('declared_deviations', [])}
    return make_program(regions, body, meta)


if __name__ == '__main__':
    import split_phase as SP, score as SC_
    from affine import Program
    SP.nocache(SC_)
    kind = sys.argv[1]
    a = [int(v) for v in sys.argv[2:]] or [60000, 10000]
    k, K = (1, 100) if kind == 'pcanet_qda' else (8, 60 if kind == 'pcanet' else 100)
    # a FUSE_K in the environment overrides the per-kind default, so the same fusion can be scored at a
    # smaller projection without editing this table
    if os.environ.get('FUSE_K'): K = int(os.environ['FUSE_K'])
    base = 'pcanet' if kind.startswith('pcanet') else 'scatter'
    t0 = time.time(); doc = build(base, a[0], a[1], k=k, K=K)
    print(f'built {time.time()-t0:.0f}s; dropped unused front-end regions: {doc["metadata"]["config"]["dropped_unused_regions"]}', flush=True)
    order = SP.density_order(doc, Program)
    r = SC_.score(build(base, a[0], a[1], k=k, K=K, order=order))
    keep = {kk: r[kk] for kk in ('energy_mj', 'time_ms', 'total_executed_instructions', 'peak_allocated_scratch_bytes', 'memory_tiles')}
    print(f'FUSE {kind} N={a[0]} Q={a[1]} k={k} K={K}  {keep["energy_mj"]:10.3f} mJ  tiles {keep["memory_tiles"]}', flush=True)
    (HERE / 'evidence' / f'fuse_{kind}_N{a[0]}.json').write_text(json.dumps({'kind': kind, 'args': a, 'k': k, 'K': K, **keep}, indent=1))
