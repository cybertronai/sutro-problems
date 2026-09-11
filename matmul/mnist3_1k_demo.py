"""MNIST-small practice run (Yaroslav's Sep-10 ask, msg 6774): the Tier 1
3x3 demo scaled to 1000 train / 1000 test, with artifacts for the A100
transpile-and-measure leg.

Extends mnist3_demo.py (same model boundary: on-chip movement only,
int8 cells, wrap-mod-256 semantics). Produces:

  - mnist3_1k_ir.txt          straight-line dally IR (9 inputs, 10 logit outputs)
  - mnist3_1k_fixture.bin     dally-eval fixture (1000 instances + expected
                              outputs from this module's Python reference,
                              so `dally-eval run` is itself the bit-exact
                              check of Python-vs-Rust)
  - mnist3_1k_inputs.bin      raw per-instance input bytes (for the CUDA
                              harness)
  - mnist3_1k_expected.bin    raw per-instance expected output bytes (10 per
                              instance; for the CUDA bit-exactness check)

Honesty note carried over: the IR's 10 output cells hold mod-256 wrapped
logits. We report both the pre-wrap integer argmax accuracy and the
IR-exact accuracy (argmax over the wrapped bytes read as signed int8).
"""
from __future__ import annotations

import os
import struct

import numpy as np

from mnist3_demo import (
    build_inference_ir,
    downsample9,
    load_mnist,
    quantize_weights,
    static_cost,
    train_linear,
)

HERE = os.path.dirname(os.path.abspath(__file__))
N_TRAIN = 1000
N_TEST = 1000


def ref_execute(ir_text: str, inputs: np.ndarray) -> np.ndarray:
    """Python reference executor, matching dally-eval 8-bit semantics:
    add/mul wrap mod 256; every cell is a byte."""
    lines = [l for l in ir_text.splitlines() if l.strip()]
    in_cells = [int(x) for x in lines[0].split(",")]
    out_cells = [int(x) for x in lines[-1].split(",")]
    n = inputs.shape[0]
    outputs = np.zeros((n, len(out_cells)), dtype=np.uint8)
    mem = {}
    for row in range(n):
        mem.clear()
        for j, c in enumerate(in_cells):
            mem[c] = int(inputs[row, j]) & 0xFF
        for l in lines[1:-1]:
            parts = l.split(None, 1)
            op = parts[0]
            args = [int(x) for x in parts[1].split(",")]
            if op == "set":
                mem[args[0]] = args[1] & 0xFF
            elif op == "copy":
                mem[args[0]] = mem[args[1]]
            elif op == "mul":
                mem[args[0]] = (mem[args[1]] * mem[args[2]]) & 0xFF
            elif op == "add":
                mem[args[0]] = (mem[args[1]] + mem[args[2]]) & 0xFF
            else:
                raise ValueError(f"op {op} not supported by this demo's IR")
        for j, c in enumerate(out_cells):
            outputs[row, j] = mem[c]
    return outputs


def quantize_weights_int8safe(qW: np.ndarray, tr_q: np.ndarray, target: int = 120) -> np.ndarray:
    """Rescale the full-range int8 weights down until every train-set logit
    fits in signed int8 (|logit| <= target), so the IR's mod-256 wrap never
    fires and wrapped-byte argmax equals integer argmax. Scale is fitted on
    train only; main() asserts the test batch also stays within 127.

    This is the fix for the discovered wrap hazard: full-range weights give
    logits up to ~3300, and argmax over mod-256 wrapped bytes scores 9.6%.
    Weights drop to about +-5, costing a few accuracy points.
    """
    s = 1.0
    while True:
        q2 = np.round(qW * s).astype(np.int32)
        logits = tr_q @ q2[:9, :] + q2[9, :]
        if np.abs(logits).max() <= target:
            return q2
        s *= 0.9


def main() -> None:
    tr_i, tr_l, te_i, te_l = load_mnist()
    tr = downsample9(tr_i)
    te = downsample9(te_i)
    mu = tr.reshape(len(tr), -1).mean(axis=0)
    tr_q = np.clip(np.round((tr.reshape(len(tr), -1) - mu) / 4.0), -31, 31).astype(np.int32)
    te_q = np.clip(np.round((te.reshape(len(te), -1) - mu) / 4.0), -31, 31).astype(np.int32)
    tr_y, te_y = tr_l.astype(np.int64), te_l.astype(np.int64)

    Wf = train_linear(tr_q[:N_TRAIN].astype(np.float64), tr_y[:N_TRAIN])
    qW_full = quantize_weights(Wf)
    qW = quantize_weights_int8safe(qW_full, tr_q[:N_TRAIN])
    Lte = te_q[:N_TEST] @ qW[:9, :] + qW[9, :]
    assert np.abs(Lte).max() <= 127, (
        f"test logits exceed int8: {np.abs(Lte).max()} (rescale tighter)"
    )

    # float baseline
    Xt = np.hstack([te_q[:N_TEST].astype(np.float64), np.ones((N_TEST, 1))])
    acc_f = ((Xt @ Wf).argmax(axis=1) == te_y[:N_TEST]).mean()
    # quantized-int argmax (pre-wrap)
    pred_q = (te_q[:N_TEST] @ qW[:9, :] + qW[9, :]).argmax(axis=1)
    acc_q = (pred_q == te_y[:N_TEST]).mean()

    ir, _ = build_inference_ir(qW, 9)
    cost = static_cost(ir)

    # IR-exact: run the wrapped-byte semantics and argmax over signed int8
    in_bytes = (te_q[:N_TEST] & 0xFF).astype(np.uint8)
    outs = ref_execute(ir, in_bytes)
    pred_ir = outs.astype(np.int8).astype(np.int64).argmax(axis=1)
    acc_ir = (pred_ir == te_y[:N_TEST]).mean()

    energy_j = cost * 1e-15
    print("MNIST-small practice run (3x3, 1000/1000)")
    print(f"  float accuracy     : {acc_f*100:.1f}%")
    print(f"  int8-IR accuracy   : {acc_q*100:.1f}% (pre-wrap integer argmax)")
    print(f"  IR-exact accuracy  : {acc_ir*100:.1f}% (mod-256 wrapped byte argmax)")
    print(f"  IR ops             : {len([l for l in ir.splitlines() if l.strip()]) - 2}")
    print(f"  static cost        : {cost} distance-units")
    print(f"  on-chip energy     : {energy_j*1e9:.3f} nJ per inference (1 fJ/unit)")
    print(f"  synthetic batch J  : {energy_j * N_TEST * 1e3:.6f} mJ per {N_TEST}-example eval batch")

    with open(os.path.join(HERE, "mnist3_1k_ir.txt"), "w") as f:
        f.write(ir)
    with open(os.path.join(HERE, "mnist3_1k_inputs.bin"), "wb") as f:
        f.write(in_bytes.tobytes())
    with open(os.path.join(HERE, "mnist3_1k_expected.bin"), "wb") as f:
        f.write(outs.tobytes())
    # dally-eval fixture: u32 n, u32 width, u64 ref cost, inputs,
    # u32 out_words, expected
    out_words = N_TEST * outs.shape[1]
    with open(os.path.join(HERE, "mnist3_1k_fixture.bin"), "wb") as f:
        f.write(struct.pack("<IIQ", N_TEST, 9, cost))
        f.write(in_bytes.tobytes())
        f.write(struct.pack("<I", out_words))
        f.write(outs.tobytes())
    print("  written: mnist3_1k_ir.txt, mnist3_1k_fixture.bin, "
          "mnist3_1k_inputs.bin, mnist3_1k_expected.bin")


if __name__ == "__main__":
    main()
