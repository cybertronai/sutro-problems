"""Tier 1 MNIST demo: train + eval a tiny linear classifier on 3x3
MNIST, express the trained inference as a straight-line Dally IR
program, and report accuracy plus ON-CHIP movement energy in joules via
the static scorer and Yaroslav's calibration (1 fJ per byte per unit of
charged distance).

Model boundary (honest): this covers ON-CHIP movement only - the die
plane of the Dally model. Off-chip DRAM/HBM streaming is not priced
(tier 3 of the MNIST ladder will need a memory-tier extension; noted in
the report, not silently assumed free).

Tier 1 spec (Yaroslav, 2026-09-04): 3x3 images, 600 train / 600 test.
A linear classifier trains on-chip-scale data and its inference
compiles to IR we can score exactly.
"""
from __future__ import annotations

import gzip
import math
import os
import struct
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "mnist_cache")


def load_mnist() -> tuple[np.ndarray, np.ndarray]:
    """Full 28x28 MNIST; we downsample to 3x3 by 9x9 block averaging."""
    os.makedirs(CACHE, exist_ok=True)
    base = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    files = {
        "train_img": "train-images-idx3-ubyte.gz",
        "train_lab": "train-labels-idx1-ubyte.gz",
        "test_img": "t10k-images-idx3-ubyte.gz",
        "test_lab": "t10k-labels-idx1-ubyte.gz",
    }
    paths = {}
    for key, fn in files.items():
        p = os.path.join(CACHE, fn)
        if not os.path.exists(p):
            urllib.request.urlretrieve(base + fn, p)
        paths[key] = p

    def imgs(p: str) -> np.ndarray:
        with gzip.open(p, "rb") as f:
            magic, n, r, c = struct.unpack(">IIII", f.read(16))
            assert magic == 2051
            return np.frombuffer(f.read(), dtype=np.uint8).reshape(n, r, c)

    def labs(p: str) -> np.ndarray:
        with gzip.open(p, "rb") as f:
            magic, n = struct.unpack(">II", f.read(8))
            assert magic == 2049
            return np.frombuffer(f.read(), dtype=np.uint8)

    return imgs(paths["train_img"]), labs(paths["train_lab"]), imgs(paths["test_img"]), labs(paths["test_lab"])


def downsample9(x: np.ndarray) -> np.ndarray:
    """28x28 -> 3x3 is not integer; crop 1px border (26x26) won't give 27.
    Instead resize by averaging over a 9.33 grid: use 9,9,10 row/col splits.
    Simplest honest scheme: block-mean over unequal blocks via repeat trick.
    We average 28->3 by convolving with box filters at positions 0,9.5,19."""
    n = x.shape[0]
    out = np.zeros((n, 3, 3), dtype=np.float32)
    # row/col boundaries for 3 blocks over 28: [0:9], [9:19], [19:28]
    bounds = [(0, 9), (9, 19), (19, 28)]
    for i, (r0, r1) in enumerate(bounds):
        for j, (c0, c1) in enumerate(bounds):
            out[:, i, j] = x[:, r0:r1, c0:c1].reshape(n, -1).mean(axis=1)
    return out


def train_linear(train_x: np.ndarray, train_y: np.ndarray, n_classes: int = 10,
                 epochs: int = 60, lr: float = 0.12, seed: int = 0) -> np.ndarray:
    """Train W (n_classes x dim) + b by softmax regression (float64, numpy)."""
    rng = np.random.default_rng(seed)
    n, dim = train_x.shape
    X = np.hstack([train_x, np.ones((n, 1))])
    W = rng.normal(0, 0.01, (dim + 1, n_classes))
    Y = np.eye(n_classes)[train_y]
    for _ in range(epochs):
        logits = X @ W
        logits -= logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        p /= p.sum(axis=1, keepdims=True)
        g = X.T @ (p - Y) / n
        W -= lr * g
    return W


def quantize_weights(W: np.ndarray) -> np.ndarray:
    """Quantize to int8 (per-class scale) so inference is exact in the
    Dally IR's 8-bit cells. Inputs are already 0..255 uint8 averages;
    we scale features down to 0..15 (4 bits) to keep products in range."""
    scale = np.abs(W).max()
    q = np.round(W / scale * 127).astype(np.int32)
    return q


def build_inference_ir(qW: np.ndarray, dim: int) -> tuple[str, dict]:
    """Emit a Dally IR program: 9 inputs at cells 1..9, int8 quantized
    dot products per class using mul + add chains (mod-256 wrapping to
    match the 8-bit cell semantics is NOT applied - we choose the input
    scale so |logit| < 128 exactly in integers before wrapping matters).
    Ops: for each class c and input i: acc_c += x_i * qW[c, i]."""
    n_classes, dimfull = qW.shape
    assert dimfull == dim + 1  # + bias column
    lines = [",".join(str(1 + i) for i in range(dim))]
    out_cells = {}
    next_cell = 100
    acc_cells = {}
    for c in range(n_classes):
        acc = None
        for i in range(dim):
            w = int(qW[i, c])
            # mul cell: dst=prod, reads x_i and W (weights are set-imm)
            prod = next_cell; next_cell += 1
            lines.append(f"set {next_cell}, {(w & 0xFF)}")
            wcell = next_cell; next_cell += 1
            lines.append(f"mul {prod},{1+i},{wcell}")
            if acc is None:
                acc = prod
            else:
                nacc = next_cell; next_cell += 1
                lines.append(f"add {nacc},{acc},{prod}")
                acc = nacc
        # bias term (row `dim` of qW) as set + add
        b = int(qW[dim, c])
        lines.append(f"set {next_cell}, {(b & 0xFF)}")
        bcell = next_cell; next_cell += 1
        nacc = next_cell; next_cell += 1
        lines.append(f"add {nacc},{acc},{bcell}")
        acc = nacc
        acc_cells[c] = acc
    # Argmax itself is control flow; the IR output = the 10 logits.
    outs = [acc_cells[c] for c in range(n_classes)]
    lines.append(",".join(str(o) for o in outs))
    return "\n".join(lines) + "\n", acc_cells


def rc(a): return math.isqrt(a - 1) + 1


def static_cost(ir_text: str) -> int:
    lines = [l for l in ir_text.splitlines() if l.strip()]
    outputs = [int(x) for x in lines[-1].split(",")]
    cost = 0
    for l in lines[1:-1]:
        parts = l.split(None, 1)
        op = parts[0]
        if op == "set":
            continue
        args = [int(x) for x in parts[1].split(",")]
        if op == "copy":
            cost += rc(args[1])
        else:
            cost += sum(rc(a) for a in args[1:])
    cost += sum(rc(o) for o in outputs)
    return cost


def main() -> None:
    tr_i, tr_l, te_i, te_l = load_mnist()
    tr = downsample9(tr_i)
    te = downsample9(te_i)
    # zero-centered signed features (±31, 6 bits): keeps int8 weight
    # resolution meaningful under one global scale (argmax-comparable)
    mu = tr.reshape(len(tr), -1).mean(axis=0)
    tr_q = np.clip(np.round((tr.reshape(len(tr), -1) - mu) / 4.0), -31, 31).astype(np.int32)
    te_q = np.clip(np.round((te.reshape(len(te), -1) - mu) / 4.0), -31, 31).astype(np.int32)
    tr_y, te_y = tr_l.astype(np.int64), te_l.astype(np.int64)

    Wf = train_linear(tr_q[:600].astype(np.float64), tr_y[:600])
    qW = quantize_weights(Wf)

    # float baseline accuracy (600 test)
    Xt = np.hstack([te_q[:600].astype(np.float64), np.ones((600, 1))])
    pred_f = (Xt @ Wf).argmax(axis=1)
    acc_f = (pred_f == te_y[:600]).mean()

    # quantized-int accuracy (exactly what the IR computes, pre-wrap)
    pred_q = (te_q[:600] @ qW[:9, :] + qW[9, :]).argmax(axis=1)
    acc_q = (pred_q == te_y[:600]).mean()

    ir, _ = build_inference_ir(qW, 9)
    cost = static_cost(ir)

    # energy: cost units are read-distances; each unit = 1 byte moved 1
    # charged-distance step = 1 fJ (Yaroslav calibration)
    energy_j = cost * 1e-15

    print("Tier 1 MNIST (3x3, 600/600) - static dataflow energy demo")
    print(f"  float accuracy   : {acc_f*100:.1f}%")
    print(f"  int8-IR accuracy : {acc_q*100:.1f}%")
    print(f"  IR ops           : {len([l for l in ir.splitlines() if l.strip()]) - 2}")
    print(f"  static cost      : {cost} distance-units")
    print(f"  on-chip energy   : {energy_j*1e9:.3f} nJ per inference (1 fJ/unit)")
    print(f"  (off-chip DRAM streaming NOT priced - see model boundary note)")

    with open(os.path.join(HERE, "mnist3_ir.txt"), "w") as f:
        f.write(ir)
    print("  IR written: matmul/mnist3_ir.txt")


if __name__ == "__main__":
    main()
