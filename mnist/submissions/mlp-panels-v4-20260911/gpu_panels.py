"""Triton implementation of the two selected panel policies.

Baseline definitions are extracted verbatim from the frozen original runner.
GPU SIMD broadcasts implement panel reuse; v4 physical distances do not map to
GPU addresses. The X cache is real cross-kernel storage, filled inside hidden
with one CTA and a barrier, avoiding an additional per-minibatch kernel launch.
"""
import ast
import hashlib
from pathlib import Path
import textwrap

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FROZEN = ROOT/'mnist/submissions/mlp60-affine-20260911/gpu_benchmark.py'
BASELINE_HASH = '9e85ccd67c25ddac738a7d65e67c04212e7564e4718a5d3a26d7875eccc20eb4'
BASELINE_NAMES = ('normalize_kernel','initialize_kernel','hidden_kernel','delta2_kernel',
                  'delta1_kernel','update_kernel','inference_hidden_kernel','inference_output_kernel')

PANEL_SOURCE = '''
@triton.jit(do_not_specialize=['batch_start'])
def hidden_cached(X, XC, P, H, batch_start, PANELS: tl.constexpr):
    off = tl.arange(0, 512)
    v = tl.load(X + batch_start * 9 + off, off < 270, other=0.)
    tl.store(XC + off, v, off < 270)
    tl.debug_barrier()
    if PANELS:
        rows = tl.arange(0, 32)
        for panel in tl.static_range(8):
            cols = panel * 4 + tl.arange(0, 4)
            acc = tl.full((32, 4), 0., tl.float32)
            for f in tl.static_range(9):
                a = tl.load(XC + rows * 9 + f, rows < 30, other=0.)
                b = tl.load(P + f * 32 + cols)
                acc = acc + a[:, None] * b[None, :]
            acc = acc + tl.load(P + 288 + cols)[None, :]
            tl.store(H + rows[:, None] * 32 + cols[None, :],
                     tl.where(acc > 0., acc, 0.), rows[:, None] < 30)
    else:
        idx = tl.arange(0, 1024)
        row = idx // 32
        col = idx % 32
        acc = tl.full((1024,), 0., tl.float32)
        for f in tl.static_range(9):
            a = tl.load(XC + row * 9 + f, idx < 960, other=0.)
            b = tl.load(P + f * 32 + col, idx < 960, other=0.)
            acc = acc + a * b
        acc = acc + tl.load(P + 288 + col, idx < 960, other=0.)
        tl.store(H + idx, tl.where(acc > 0., acc, 0.), idx < 960)

@triton.jit(do_not_specialize=['batch_start'])
def output_panel(H, P, TARGET, D2, batch_start):
    rows = tl.program_id(0) * 8 + tl.arange(0, 8)
    cols = tl.arange(0, 16)
    acc = tl.full((8, 16), 0., tl.float32)
    for k in tl.static_range(32):
        a = tl.load(H + rows * 32 + k, rows < 30, other=0.)
        b = tl.load(P + 320 + k * 10 + cols, cols < 10, other=0.)
        acc = acc + a[:, None] * b[None, :]
    acc = acc + tl.load(P + 640 + cols, cols < 10, other=0.)[None, :]
    valid = (rows[:, None] < 30) & (cols[None, :] < 10)
    target = tl.load(TARGET + (batch_start + rows[:, None]) * 10 + cols[None, :], valid, other=0.)
    tl.store(D2 + rows[:, None] * 10 + cols[None, :], acc - target, valid)

@triton.jit
def backward_panel(H, P, D2, D1):
    rows = tl.arange(0, 32)
    hidden = tl.program_id(0) * 4 + tl.arange(0, 4)
    acc = tl.full((32, 4), 0., tl.float32)
    for k in tl.static_range(10):
        a = tl.load(D2 + rows * 10 + k, rows < 30, other=0.)
        b = tl.load(P + 320 + hidden * 10 + k)
        acc = acc + a[:, None] * b[None, :]
    addresses = rows[:, None] * 32 + hidden[None, :]
    active = tl.load(H + addresses, rows[:, None] < 30, other=0.) > 0.
    tl.store(D1 + addresses, tl.where(active, acc, 0.), rows[:, None] < 30)

@triton.jit
def update_panels(XC, H, D1, D2, P, STEP: tl.constexpr):
    panel = tl.program_id(0)
    if panel < 8:
        features = tl.arange(0, 16)
        hidden = panel * 4 + tl.arange(0, 4)
        grad = tl.full((16, 4), 0., tl.float32)
        bias = tl.full((4,), 0., tl.float32)
        for row in tl.static_range(30):
            a = tl.load(XC + row * 9 + features, features < 9, other=0.)
            b = tl.load(D1 + row * 32 + hidden)
            grad = grad + a[:, None] * b[None, :]
            bias = bias + b
        address = features[:, None] * 32 + hidden[None, :]
        old = tl.load(P + address, features[:, None] < 9, other=0.)
        tl.store(P + address, old - STEP * grad, features[:, None] < 9)
        old_bias = tl.load(P + 288 + hidden)
        tl.store(P + 288 + hidden, old_bias - STEP * bias)
    else:
        hidden2 = (panel - 8) * 8 + tl.arange(0, 8)
        cols2 = tl.arange(0, 16)
        grad2 = tl.full((8, 16), 0., tl.float32)
        bias2 = tl.full((16,), 0., tl.float32)
        for row2 in tl.static_range(30):
            a2 = tl.load(H + row2 * 32 + hidden2)
            b2 = tl.load(D2 + row2 * 10 + cols2, cols2 < 10, other=0.)
            grad2 = grad2 + a2[:, None] * b2[None, :]
            if panel == 8:
                bias2 = bias2 + b2
        address2 = 320 + hidden2[:, None] * 10 + cols2[None, :]
        old2 = tl.load(P + address2, cols2[None, :] < 10, other=0.)
        tl.store(P + address2, old2 - STEP * grad2, cols2[None, :] < 10)
        if panel == 8:
            old_bias2 = tl.load(P + 640 + cols2, cols2 < 10, other=0.)
            tl.store(P + 640 + cols2, old_bias2 - STEP * bias2, cols2 < 10)

@triton.jit
def inference_hidden_panels(Q, P, H):
    rows = tl.program_id(0) * 30 + tl.arange(0, 32)
    valid = tl.arange(0, 32) < 30
    # Retain each query feature in registers across the eight column panels.
    q0 = tl.load(Q + rows * 9 + 0, valid, other=0.)
    q1 = tl.load(Q + rows * 9 + 1, valid, other=0.)
    q2 = tl.load(Q + rows * 9 + 2, valid, other=0.)
    q3 = tl.load(Q + rows * 9 + 3, valid, other=0.)
    q4 = tl.load(Q + rows * 9 + 4, valid, other=0.)
    q5 = tl.load(Q + rows * 9 + 5, valid, other=0.)
    q6 = tl.load(Q + rows * 9 + 6, valid, other=0.)
    q7 = tl.load(Q + rows * 9 + 7, valid, other=0.)
    q8 = tl.load(Q + rows * 9 + 8, valid, other=0.)
    for panel in tl.static_range(8):
        cols = panel * 4 + tl.arange(0, 4)
        acc = tl.full((32, 4), 0., tl.float32)
        b0 = tl.load(P + 0 * 32 + cols)
        acc = acc + q0[:, None] * b0[None, :]
        b1 = tl.load(P + 1 * 32 + cols)
        acc = acc + q1[:, None] * b1[None, :]
        b2 = tl.load(P + 2 * 32 + cols)
        acc = acc + q2[:, None] * b2[None, :]
        b3 = tl.load(P + 3 * 32 + cols)
        acc = acc + q3[:, None] * b3[None, :]
        b4 = tl.load(P + 4 * 32 + cols)
        acc = acc + q4[:, None] * b4[None, :]
        b5 = tl.load(P + 5 * 32 + cols)
        acc = acc + q5[:, None] * b5[None, :]
        b6 = tl.load(P + 6 * 32 + cols)
        acc = acc + q6[:, None] * b6[None, :]
        b7 = tl.load(P + 7 * 32 + cols)
        acc = acc + q7[:, None] * b7[None, :]
        b8 = tl.load(P + 8 * 32 + cols)
        acc = acc + q8[:, None] * b8[None, :]
        acc = acc + tl.load(P + 288 + cols)[None, :]
        tl.store(H + rows[:, None] * 32 + cols[None, :], tl.where(acc > 0., acc, 0.), valid[:, None])

@triton.jit
def inference_hidden_cached(Q, P, H):
    rows = tl.program_id(0) * 4 + tl.arange(0, 4)
    cols = tl.arange(0, 32)
    acc = tl.full((4, 32), 0., tl.float32)
    for f in tl.static_range(9):
        a = tl.load(Q + rows * 9 + f, rows < 600, other=0.)
        b = tl.load(P + f * 32 + cols)
        acc = acc + a[:, None] * b[None, :]
    acc = acc + tl.load(P + 288 + cols)[None, :]
    tl.store(H + rows[:, None] * 32 + cols[None, :], tl.where(acc > 0., acc, 0.), rows[:, None] < 600)

@triton.jit
def inference_output_panels(H, P, SCORES, OUT):
    rows = tl.program_id(0) * 30 + tl.arange(0, 32)
    valid_rows = tl.arange(0, 32) < 30
    cols = tl.arange(0, 16)
    acc = tl.full((32, 16), 0., tl.float32)
    for h in tl.static_range(32):
        a = tl.load(H + rows * 32 + h, valid_rows, other=0.)
        b = tl.load(P + 320 + h * 10 + cols, cols < 10, other=0.)
        acc = acc + a[:, None] * b[None, :]
    acc = acc + tl.load(P + 640 + cols, cols < 10, other=0.)[None, :]
    tl.store(SCORES + rows[:, None] * 10 + cols[None, :], acc, valid_rows[:, None] & (cols[None, :] < 10))
    masked = tl.where(cols[None, :] < 10, acc, float('-inf'))
    best = tl.max(masked, axis=1)
    winner = tl.min(tl.where((cols[None, :] < 10) & (masked == best[:, None]), cols[None, :], 2147483647), axis=1)
    tl.store(OUT + rows, winner, valid_rows)
'''


def generated_source():
    source = FROZEN.read_text()
    assert hashlib.sha256(source.encode()).hexdigest()==BASELINE_HASH
    lines = source.splitlines()
    tree = ast.parse(source)
    definitions = []
    for name in BASELINE_NAMES:
        node = next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name)
        start = min([node.lineno]+[d.lineno for d in node.decorator_list])
        definition = textwrap.dedent('\n'.join(lines[start-1:node.end_lineno]))
        assert ast.dump(ast.parse(definition).body[0])==ast.dump(node)
        definitions.append(definition)
    return 'import triton\nimport triton.language as tl\n\n'+'\n\n'.join(definitions)+PANEL_SOURCE
