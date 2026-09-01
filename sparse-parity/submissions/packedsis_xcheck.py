"""Cross-check packedsis.generate_packed_sis (row-order variant) against a
numpy emulation: last-eligible static pivots, F clustered in cols 0..15,
row-order walk state, coefficient-weight-ordered walk with optional g2
pair restriction, weight-k last-wins capture, row-order unscatter."""
import os
import sys
from itertools import combinations
from random import Random

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

import numpy as np  # noqa: E402
import mask_sparse_parity as mp  # noqa: E402
from packedsis import generate_packed_sis  # noqa: E402


def emulate(cap, seed, X, y, g2=None, n=32, m=18, k=5):
    G = n - m
    rng = Random(seed)
    F = sorted(rng.sample(range(16), G))
    S = sorted(set(range(n)) - set(F))
    rows = np.concatenate([X, y[:, None]], axis=1).astype(np.uint8)
    piv = {}
    usedset = set()
    for c in S:                                  # last eligible row wins
        cand = [r for r in range(m) if rows[r, c] and r not in usedset]
        pr = cand[-1] if cand else None
        piv[c] = pr
        if pr is None:
            continue
        usedset.add(pr)
        flip = [r for r in range(m) if r != pr and rows[r, c]]
        for r in flip:
            rows[r] ^= rows[pr]
    # row-order state: bit r = row r's y value / column-f value
    wS = rows[:, n].copy()
    avec = np.zeros(G, dtype=np.uint8)
    knobs = [rows[:, f].copy() for f in F]
    out = np.zeros(n, dtype=np.uint8)

    def capture():
        if int(wS.sum()) + int(avec.sum()) == k:
            out[:] = 0
            for j, c in enumerate(S):
                out[c] = wS[piv[c]] if piv[c] is not None else 0
            for j, c in enumerate(F):
                out[c] = avec[j]

    capture()
    sets = [()]
    for w in range(1, cap + 1):
        if w == 2 and g2 is not None:
            sets.extend(combinations(range(g2), 2))
        else:
            sets.extend(combinations(range(G), w))
    prev = frozenset()
    for cur in sets:
        cur_s = frozenset(cur)
        for j in sorted(cur_s.symmetric_difference(prev)):
            wS = wS ^ knobs[j]
            avec[j] ^= 1
        prev = cur_s
        capture()
    return out


def main():
    cases = ((1, None, 0), (2, 8, 0), (2, None, 0), (3, None, 0),
             (2, 8, 13), (3, None, 13))
    inputs, masks, _meta = mp.mask_suite()
    for cap, g2, seed in cases:
        ir = generate_packed_sis(cap=cap, seed=seed, g2=g2)
        run, cost, _ = mp._compile_vector(ir, mp.OP_CAP)
        o = run(inputs)
        mism = 0
        for i in range(inputs.shape[0]):
            X = inputs[i][:576].reshape(18, 32)
            y = inputs[i][576:]
            e = emulate(cap, seed, X, y, g2=g2)
            if not np.array_equal(o[i], e):
                mism += 1
                if mism < 3:
                    print("mismatch at", i, list(o[i]), list(e))
        print(f"cap={cap} g2={g2} seed={seed}: "
              f"mismatches {mism}/{len(inputs)}, "
              f"recovery {(o == masks).all(axis=1).mean():.4f}, cost {cost}")
        if mism:
            raise AssertionError(f"{mism} emulator mismatches")


if __name__ == "__main__":
    main()
