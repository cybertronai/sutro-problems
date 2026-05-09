"""Prediction-packed medium sparse-parity IR generator.

This remains an exhaustive, seed-independent decoder for official v3
``score_medium``.  Training rows are packed into 8-bit column masks.  Candidate
testing is grouped by the first two columns so the shared pair XOR is computed
once and reused for every third column in that group.

Packing the 64 test rows into masks was explored but is not emitted here: the
v3 cost of scalar output unpacking is higher than the saved prediction reads.
"""
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

    train_names = [
        x_tr(i, c)
        for i in range(spec.m_train)
        for c in range(spec.n_bits)
    ] + [y_tr(i) for i in range(spec.m_train)]
    output_aliases = train_names[:spec.m_test]

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
                    counts[w_ind] += 1
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
                lines.append(f"cmp {at(w_ind)},{at(tmp)},{at(ymask)},eq")
                for secret_col in (a, b, c):
                    if seen_secret_column[secret_col]:
                        lines.append(f"or {at(secret_mask(secret_col))},{at(w_ind)}")
                    else:
                        lines.append(f"copy {at(secret_mask(secret_col))},{at(w_ind)}")
                        seen_secret_column[secret_col] = True

    for j in range(spec.m_test):
        lines.append(f"and {at(acc)},{at(secret_mask(0))},{at(x_te(j, 0))}")
        for c in range(1, spec.n_bits):
            lines.append(f"and {at(tmp)},{at(secret_mask(c))},{at(x_te(j, c))}")
            lines.append(f"xor {at(acc)},{at(tmp)}")
        lines.append(f"copy {at(output_aliases[j])},{at(acc)}")

    lines.append(",".join(map(str, outputs)))
    return "\n".join(lines)


def generate_predpack_medium() -> str:
    sp = _load_official()
    return _generate(sp.MEDIUM)


def main() -> None:
    sp = _load_official()
    ir = generate_predpack_medium()
    cost = sp.score_medium(ir)
    ops = len(ir.splitlines()) - 2

    here = Path(__file__).resolve().parent
    ir_path = here / "predpack_medium.ir"
    report_path = here / "predpack_medium.md"

    ir_path.write_text(ir + "\n")
    report_path.write_text(
        "# Sparse parity - pair-XOR reuse (medium)\n\n"
        "**Author:** [@sjbaebae](https://github.com/sjbaebae)\n"
        "**Date:** 2026-05-08\n"
        "**Problem:** sparse parity (n=8, k=3, 8 train / 64 test)\n"
        f"**Cost:** {cost:,}\n"
        "**IR:** [`predpack_medium.ir`](predpack_medium.ir)\n"
        "**Generator:** [`predpack_medium.py`](predpack_medium.py)\n"
        "**Method:** `generate_predpack_medium` (packed candidate check with shared pair-XOR reuse)\n\n"
        "## Algorithm\n\n"
        "This is a generated official v3 IR for arbitrary `score_medium` seeds. "
        "It keeps the packed 8-row training column masks from "
        "`ge_medium_packed.py`, but groups candidate triples by their first two "
        "columns.  The shared two-column XOR is computed once per `(a, b)` "
        "pair and reused across all valid third columns `c`, reducing decode "
        "work from 56 pair XORs to 21 pair XORs.\n\n"
        "Packing the 64 prediction rows into bit masks was tested, but the "
        "required scalar output unpacking costs more than it saves under the "
        "v3 read model.  The emitted IR therefore uses direct scalar "
        "prediction from the recovered secret mask.\n\n"
        "## Cost breakdown\n\n"
        f"| section | ops |\n| --- | ---: |\n| total IR body | {ops:,} |\n\n"
        "The pair-XOR grouping reduces decode work relative to "
        "`ge_medium_packed.ir`, lowering cost from 16,084 to 15,960.\n"
    )
    print(f"{ir_path.name} cost={cost:,} ops={ops:,}")


if __name__ == "__main__":
    main()
