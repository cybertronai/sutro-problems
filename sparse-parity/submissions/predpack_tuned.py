"""Tuned grouped-candidate medium sparse-parity IR."""
from __future__ import annotations

from collections import defaultdict
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
    tmp = ("tmp_gate",)
    pair = ("pair",)
    acc = ("acc",)
    w_ind = ("w_ind",)
    ymask = ("ymask",)

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

    for name in (tmp, pair, acc, w_ind, ymask):
        add_name(name)
    for c in range(spec.n_bits):
        add_name(col_mask(c))
        add_name(secret_mask(c))

    train_names = [x_tr(i, c) for i in range(spec.m_train) for c in range(spec.n_bits)] + [
        y_tr(i) for i in range(spec.m_train)
    ]
    for name in train_names:
        add_name(name)
    for j in range(spec.m_test):
        for c in range(spec.n_bits):
            add_name(x_te(j, c))

    dead_output_aliases = [w_ind, ymask] + [col_mask(c) for c in range(spec.n_bits)]
    output_aliases = dead_output_aliases + train_names[: spec.m_test - len(dead_output_aliases) - 1] + [tmp]

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
    for a in range(spec.n_bits - 2):
        for b in range(a + 1, spec.n_bits - 1):
            counts[col_mask(a)] += 1
            counts[col_mask(b)] += 1
            for c in range(b + 1, spec.n_bits):
                counts[pair] += 1
                counts[col_mask(c)] += 1
                counts[tmp] += 1
                counts[ymask] += 1
                for secret_col in (a, b, c):
                    # tmp holds the candidate indicator after cmp.
                    counts[tmp] += 1
                    if seen_secret_column[secret_col]:
                        counts[secret_mask(secret_col)] += 1
                    else:
                        seen_secret_column[secret_col] = True

    for j in range(spec.m_test):
        counts[secret_mask(0)] += 1
        counts[x_te(j, 0)] += 1
        for c in range(1, spec.n_bits):
            counts[secret_mask(c)] += 1
            counts[x_te(j, c)] += 1
            counts[acc] += 1
            counts[tmp] += 1
        counts[acc] += 1
        counts[output_aliases[j]] += 1

    ordered = sorted(names, key=lambda name: (-counts[name], str(name)))
    addr = {name: idx + 1 for idx, name in enumerate(ordered)}
    at = lambda name: addr[name]

    inputs = (
        [at(x_tr(i, c)) for i in range(spec.m_train) for c in range(spec.n_bits)]
        + [at(y_tr(i)) for i in range(spec.m_train)]
        + [at(x_te(j, c)) for j in range(spec.m_test) for c in range(spec.n_bits)]
    )
    outputs = [at(output_aliases[j]) for j in range(spec.m_test)]

    lines = [",".join(map(str, inputs))]

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
    for a in range(spec.n_bits - 2):
        for b in range(a + 1, spec.n_bits - 1):
            lines.append(f"xor {at(pair)},{at(col_mask(a))},{at(col_mask(b))}")
            for c in range(b + 1, spec.n_bits):
                lines.append(f"xor {at(tmp)},{at(pair)},{at(col_mask(c))}")
                lines.append(f"cmp {at(tmp)},{at(tmp)},{at(ymask)},eq")
                for secret_col in (a, b, c):
                    if seen_secret_column[secret_col]:
                        lines.append(f"or {at(secret_mask(secret_col))},{at(tmp)}")
                    else:
                        lines.append(f"copy {at(secret_mask(secret_col))},{at(tmp)}")
                        seen_secret_column[secret_col] = True

    for j in range(spec.m_test):
        last_output_to_tmp = output_aliases[j] == tmp
        lines.append(f"and {at(acc)},{at(secret_mask(0))},{at(x_te(j, 0))}")
        for c in range(1, spec.n_bits):
            lines.append(f"and {at(tmp)},{at(secret_mask(c))},{at(x_te(j, c))}")
            dest = tmp if (last_output_to_tmp and c == spec.n_bits - 1) else acc
            lines.append(f"xor {at(dest)},{at(acc)},{at(tmp)}" if dest == tmp else f"xor {at(acc)},{at(tmp)}")
        if not last_output_to_tmp:
            lines.append(f"copy {at(output_aliases[j])},{at(acc)}")

    lines.append(",".join(map(str, outputs)))
    return "\n".join(lines)


def generate_predpack_tuned_medium() -> str:
    return _generate(_load_official().MEDIUM)


def main() -> None:
    sp = _load_official()
    ir = generate_predpack_tuned_medium()
    cost = sp.score_medium(ir)
    ops = len(ir.splitlines()) - 2
    Path(__file__).with_name("predpack_tuned_medium.ir").write_text(ir + "\n")
    Path(__file__).with_name("predpack_tuned.md").write_text(
        f"""# Sparse parity - tuned pair-XOR reuse (medium)

**Author:** [@sjbaebae](https://github.com/sjbaebae)
**Date:** 2026-05-08
**Problem:** sparse parity (n=8, k=3, 8 train / 64 test)
**Cost:** {cost:,}
**IR:** [`predpack_tuned_medium.ir`](predpack_tuned_medium.ir)
**Generator:** [`predpack_tuned.py`](predpack_tuned.py)
**Method:** `generate_predpack_tuned_medium` (packed candidate check with pair-XOR reuse and address/liveness tuning)

## Algorithm

The medium task has 3 hidden bits among 8 total bits, 8 training examples, and
64 test examples. The IR first packs each training column into an 8-bit mask and
packs the training labels into `ymask`.

For each candidate triple `(a, b, c)`, the decoder checks:

```text
col[a] xor col[b] xor col[c] == ymask
```

If the comparison is true, the candidate contributes to the recovered secret
mask cells. Instead of recomputing the first two XORs for every triple, triples
are grouped by `(a, b)`: `pair = col[a] xor col[b]` is computed once and reused
for every later column `c`.

After the secret mask is recovered, the IR predicts all 64 test rows by gating
the input bits with the recovered mask and XORing the selected bits.

## Tuning

Relative to `predpack_medium.ir`, this version keeps the candidate indicator in
`tmp_gate`, aliases early output cells to dead decode scratch (`w_ind`, `ymask`,
and `cm[0..7]`), and leaves the final prediction in `tmp_gate`. These are
placement/liveness changes only; the algorithm remains a seed-independent
packed-column GF(2) decoder.

Opcode counts:

| Opcode | Count |
| --- | ---: |
| `and` | 512 |
| `cmp` | 56 |
| `copy` | 80 |
| `mul` | 63 |
| `or` | 223 |
| `set` | 63 |
| `xor` | 525 |

## Comparison

| Submission | Cost | IR ops | Notes |
| --- | ---: | ---: | --- |
| `baseline_medium.ir` | 816,251 | 16,457 | Upstream try-each-candidate baseline |
| `ge_medium.ir` | 473,046 | 8,590 | Upstream GF(2) Gaussian elimination baseline |
| `ge_medium_packed.ir` | 16,084 | 1,558 | Packed GF(2) candidate check |
| `predpack_medium.ir` | 15,960 | 1,523 | Pair-XOR reuse |
| `predpack_tuned_medium.ir` | {cost:,} | {ops:,} | Pair-XOR reuse plus address/liveness tuning |

The IR uses only official v3 operations: `set`, `copy`, `xor`, `and`, `or`,
`mul`, and `cmp`.
"""
    )
    print(f"predpack_tuned_medium.ir cost={cost:,} ops={ops:,}")


if __name__ == "__main__":
    main()
