"""Half-output approx50 medium sparse-parity IR generator.

Reuses Sung Jae's packed-column candidate-check decoder
(``ge_medium_packed.py``) but only emits real predictions for the
first half of the test rows. The remaining ``m_test/2`` outputs are
all aliased to a single cell that holds an arbitrary constant
(``set X, 0``). Per-row prediction cost is roughly halved at the
expense of getting only ~75% of test rows correct on average:

  - rows 0..31: actual XOR over the recovered secret mask,
  - rows 32..63: constant 0; matches y_test[j] with probability 1/2.

Designed to pass ``score_medium_approx50`` (per-seed threshold 50%)
and to *fail* the strict ``score_medium`` (which requires 100%).
"""
from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from pathlib import Path
import importlib.util


def _load_official():
    path = Path(__file__).resolve().parents[1] / "sparse_parity.py"
    spec = importlib.util.spec_from_file_location("_official_sparse_parity", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _generate(spec) -> str:
    candidates = list(combinations(range(spec.n_bits), spec.k_secret))
    half = spec.m_test // 2

    tmp = ("tmp_gate",)
    acc = ("acc",)
    w_ind = ("w_ind",)
    ymask = ("ymask",)
    zero_const = ("zero_const",)

    x_tr = lambda i, c: ("xtr", i, c)
    y_tr = lambda i: ("y", i)
    col_mask = lambda c: ("cm", c)
    secret_mask = lambda c: ("s", c)
    x_te = lambda j, c: ("xte", j, c)

    names = []
    counts = defaultdict(int)

    def add_name(name) -> None:
        if name not in names:
            names.append(name)

    for name in (tmp, acc, w_ind, ymask, zero_const):
        add_name(name)
    for c in range(spec.n_bits):
        add_name(col_mask(c))
        add_name(secret_mask(c))

    train_names = [
        x_tr(i, c)
        for i in range(spec.m_train)
        for c in range(spec.n_bits)
    ] + [y_tr(i) for i in range(spec.m_train)]
    # Only need m_test/2 output aliases for the real-prediction half;
    # the other half are all aliased to zero_const.
    output_aliases = train_names[:half]

    for name in train_names:
        add_name(name)
    for j in range(spec.m_test):
        for c in range(spec.n_bits):
            add_name(x_te(j, c))

    def count_pack(dest, bit_at) -> None:
        counts[bit_at(0)] += 1
        for i in range(1, spec.m_train):
            counts[bit_at(i)] += 1
            counts[w_ind] += 1
            counts[tmp] += 1
            counts[dest] += 1
            counts[tmp] += 1

    count_pack(ymask, y_tr)
    for c in range(spec.n_bits):
        count_pack(col_mask(c), lambda i, c=c: x_tr(i, c))

    seen_secret_column = [False] * spec.n_bits
    for cand in candidates:
        counts[col_mask(cand[0])] += 1
        counts[col_mask(cand[1])] += 1
        counts[tmp] += 1
        counts[col_mask(cand[2])] += 1
        counts[tmp] += 1
        counts[ymask] += 1
        for c in cand:
            counts[w_ind] += 1
            if seen_secret_column[c]:
                counts[secret_mask(c)] += 1
            else:
                seen_secret_column[c] = True

    for j in range(half):
        counts[secret_mask(0)] += 1
        counts[x_te(j, 0)] += 1
        for c in range(1, spec.n_bits):
            counts[secret_mask(c)] += 1
            counts[x_te(j, c)] += 1
            counts[acc] += 1
            counts[tmp] += 1
        counts[acc] += 1
        counts[output_aliases[j]] += 1
    # zero_const is read once for every "lazy" output position
    counts[zero_const] += spec.m_test - half

    ordered = sorted(names, key=lambda name: (-counts[name], str(name)))
    addr = {name: idx + 1 for idx, name in enumerate(ordered)}
    at = lambda name: addr[name]

    inputs = (
        [at(x_tr(i, c)) for i in range(spec.m_train) for c in range(spec.n_bits)]
        + [at(y_tr(i)) for i in range(spec.m_train)]
        + [at(x_te(j, c)) for j in range(spec.m_test) for c in range(spec.n_bits)]
    )
    outputs = (
        [at(output_aliases[j]) for j in range(half)]
        + [at(zero_const)] * (spec.m_test - half)
    )

    lines = [",".join(map(str, inputs))]
    lines.append(f"set {at(zero_const)},0")

    def pack(dest, bit_at) -> None:
        lines.append(f"copy {at(dest)},{at(bit_at(0))}")
        for i in range(1, spec.m_train):
            lines.append(f"set {at(w_ind)},{1 << i}")
            lines.append(f"mul {at(tmp)},{at(bit_at(i))},{at(w_ind)}")
            lines.append(f"or {at(dest)},{at(tmp)}")

    pack(ymask, y_tr)
    for c in range(spec.n_bits):
        pack(col_mask(c), lambda i, c=c: x_tr(i, c))

    seen_secret_column = [False] * spec.n_bits
    for cand in candidates:
        lines.append(
            f"xor {at(tmp)},{at(col_mask(cand[0]))},{at(col_mask(cand[1]))}"
        )
        lines.append(f"xor {at(tmp)},{at(col_mask(cand[2]))}")
        lines.append(f"cmp {at(w_ind)},{at(tmp)},{at(ymask)},eq")
        for c in cand:
            if seen_secret_column[c]:
                lines.append(f"or {at(secret_mask(c))},{at(w_ind)}")
            else:
                lines.append(f"copy {at(secret_mask(c))},{at(w_ind)}")
                seen_secret_column[c] = True

    # Only predict for the first `half` test rows; the remaining outputs
    # all reuse the zero_const cell for "free" lazy 0s.
    for j in range(half):
        lines.append(f"and {at(acc)},{at(secret_mask(0))},{at(x_te(j, 0))}")
        for c in range(1, spec.n_bits):
            lines.append(f"and {at(tmp)},{at(secret_mask(c))},{at(x_te(j, c))}")
            lines.append(f"xor {at(acc)},{at(tmp)}")
        lines.append(f"copy {at(output_aliases[j])},{at(acc)}")

    lines.append(",".join(map(str, outputs)))
    return "\n".join(lines)


def generate_half_packed_approx50() -> str:
    sp = _load_official()
    return _generate(sp.MEDIUM)


def main() -> None:
    sp = _load_official()
    ir = generate_half_packed_approx50()
    cost = sp.score_medium_approx50(ir)
    ops = len(ir.splitlines()) - 2

    here = Path(__file__).resolve().parent
    ir_path = here / "half_packed_approx50.ir"
    ir_path.write_text(ir + "\n")
    print(f"  {ir_path.name:<28}  cost={cost:>9,}  ops={ops:>6,}  -> {ir_path}")


if __name__ == "__main__":
    main()
