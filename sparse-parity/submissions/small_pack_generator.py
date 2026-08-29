"""Small sparse-parity official-v3 IR generator.

This file owns only the ``small_pack_*`` artifacts.  It keeps the best variant
found here plus a packed-column GF(2) variant used for comparison.
"""
from __future__ import annotations

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


def generate_small_pack_best() -> str:
    """Best small variant found: row decode, low test placement."""
    sp = _load_official()
    spec = sp.SMALL
    candidates = list(combinations(range(spec.n_bits), spec.k_secret))

    pred_base = 1
    tmp, parity, one = 1, 2, 3
    y_base = 4
    x_base = 8
    x_te_base = 20
    ind_base = x_te_base + spec.n_bits * spec.m_test
    stage_base = ind_base + len(candidates)
    acc = 4
    gated = 5

    pred = lambda j: pred_base + j
    y_tr = lambda i: y_base + i
    x_tr = lambda i, c: x_base + i * spec.n_bits + c
    x_te = lambda j, c: x_te_base + j * spec.n_bits + c
    ind = lambda t: ind_base + t
    secret = lambda c: 1 + c

    inputs = (
        [x_tr(i, c) for i in range(spec.m_train) for c in range(spec.n_bits)]
        + [y_tr(i) for i in range(spec.m_train)]
        + [x_te(j, c) for j in range(spec.m_test) for c in range(spec.n_bits)]
    )
    outputs = [pred(j) for j in range(spec.m_test)]

    lines = [",".join(map(str, inputs))]
    lines.append(f"set {one},1")

    for t_idx, cand in enumerate(candidates):
        lines.append(f"xor {tmp},{y_tr(0)},{x_tr(0, cand[0])}")
        lines.append(f"xor {parity},{tmp},{x_tr(0, cand[1])}")
        lines.append(f"xor {ind(t_idx)},{parity},{one}")
        for i in range(1, spec.m_train):
            lines.append(f"xor {tmp},{y_tr(i)},{x_tr(i, cand[0])}")
            lines.append(f"xor {parity},{tmp},{x_tr(i, cand[1])}")
            lines.append(f"xor {parity},{one}")
            lines.append(f"and {ind(t_idx)},{parity}")

    lines.extend(
        [
            f"copy {secret(0)},{ind(0)}",
            f"or {secret(0)},{ind(1)}",
            f"copy {secret(1)},{ind(0)}",
            f"or {secret(1)},{ind(2)}",
            f"copy {secret(2)},{ind(1)}",
            f"or {secret(2)},{ind(2)}",
        ]
    )

    reserved = {secret(c) for c in range(spec.n_bits)} | {acc, gated}
    deferred = [j for j in range(spec.m_test) if pred(j) in reserved]
    direct = [j for j in range(spec.m_test) if pred(j) not in reserved]

    def emit_pred(j: int, dest: int) -> None:
        lines.append(f"and {acc},{secret(0)},{x_te(j, 0)}")
        for c in range(1, spec.n_bits):
            lines.append(f"and {gated},{secret(c)},{x_te(j, c)}")
            lines.append(f"xor {acc},{gated}")
        lines.append(f"copy {dest},{acc}")

    # Consume low-address test cells 20..34 before direct outputs can overwrite
    # any of those addresses.
    for pos, j in enumerate(deferred):
        emit_pred(j, stage_base + pos)
    for j in direct:
        emit_pred(j, pred(j))
    for pos, j in enumerate(deferred):
        lines.append(f"copy {pred(j)},{stage_base + pos}")

    lines.append(",".join(map(str, outputs)))
    return "\n".join(lines)


def generate_small_pack_column_gf2() -> str:
    """Packed-column GF(2) comparison variant."""
    sp = _load_official()
    spec = sp.SMALL

    pred = lambda j: 1 + j
    tmp, acc, gated = 1, 4, 5
    y_base = 4
    x_base = 8
    x_te_base = 20
    after_inputs = x_te_base + spec.n_bits * spec.m_test
    weights = [after_inputs, after_inputs + 1, after_inputs + 2]
    col = [after_inputs + 3, after_inputs + 4, after_inputs + 5]
    y_mask = after_inputs + 6
    ind = [after_inputs + 7, after_inputs + 8, after_inputs + 9]
    stage_base = after_inputs + 10

    y_tr = lambda i: y_base + i
    x_tr = lambda i, c: x_base + i * spec.n_bits + c
    x_te = lambda j, c: x_te_base + j * spec.n_bits + c

    inputs = (
        [x_tr(i, c) for i in range(spec.m_train) for c in range(spec.n_bits)]
        + [y_tr(i) for i in range(spec.m_train)]
        + [x_te(j, c) for j in range(spec.m_test) for c in range(spec.n_bits)]
    )
    lines = [",".join(map(str, inputs))]
    for dest, value in zip(weights, [2, 4, 8]):
        lines.append(f"set {dest},{value}")

    def pack(dest: int, addrs: list[int]) -> None:
        lines.append(f"copy {dest},{addrs[0]}")
        for addr, weight in zip(addrs[1:], weights):
            lines.append(f"mul {tmp},{addr},{weight}")
            lines.append(f"xor {dest},{tmp}")

    for c in range(spec.n_bits):
        pack(col[c], [x_tr(i, c) for i in range(spec.m_train)])
    pack(y_mask, [y_tr(i) for i in range(spec.m_train)])

    for idx, (a, b) in enumerate(combinations(range(spec.n_bits), spec.k_secret)):
        lines.append(f"xor {tmp},{col[a]},{col[b]}")
        lines.append(f"cmp {ind[idx]},{tmp},{y_mask},eq")

    lines.extend(
        [
            f"copy 1,{ind[0]}",
            f"or 1,{ind[1]}",
            f"copy 2,{ind[0]}",
            f"or 2,{ind[2]}",
            f"copy 3,{ind[1]}",
            f"or 3,{ind[2]}",
        ]
    )

    def emit_pred(j: int, dest: int) -> None:
        lines.append(f"and {acc},1,{x_te(j, 0)}")
        lines.append(f"and {gated},2,{x_te(j, 1)}")
        lines.append(f"xor {acc},{gated}")
        lines.append(f"and {gated},3,{x_te(j, 2)}")
        lines.append(f"xor {acc},{gated}")
        lines.append(f"copy {dest},{acc}")

    deferred = [0, 1, 2, 3, 4]
    direct = [j for j in range(spec.m_test) if j not in deferred]
    for pos, j in enumerate(deferred):
        emit_pred(j, stage_base + pos)
    for j in direct:
        emit_pred(j, pred(j))
    for pos, j in enumerate(deferred):
        lines.append(f"copy {pred(j)},{stage_base + pos}")

    lines.append(",".join(str(pred(j)) for j in range(spec.m_test)))
    return "\n".join(lines)


def main() -> None:
    sp = _load_official()
    variants = [
        ("small_pack_best.ir", generate_small_pack_best()),
        ("small_pack_column_gf2.ir", generate_small_pack_column_gf2()),
    ]
    for name, ir in variants:
        Path(__file__).with_name(name).write_text(ir + "\n")
        print(f"{name} cost={sp.score_small(ir):,} ops={len(ir.splitlines()) - 2:,}")


if __name__ == "__main__":
    main()
