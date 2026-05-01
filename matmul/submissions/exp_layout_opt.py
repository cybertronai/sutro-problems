#!/usr/bin/env python3
"""
Experiment: optimized scratchpad address layout for tiled 16x16 matmul

Hypothesis: moving tmp from address 49 (cost 7) to address 1 (cost 1) and
shifting sA/sB/sC up by one slot each reduces cost significantly, since tmp
is read ~4,000 times in the inner loop while the sA/sB/sC cost increase is
modest.

Baseline layout (generate_tiled_16x16):
  tmp at 49       cost 7
  sA  at 1..16    cost 1-4
  sB  at 17..32   cost 5-6
  sC  at 33..48   cost 6-7

Opt1 layout (this file):
  tmp at 1        cost 1
  sA  at 2..17    cost 2-5
  sB  at 18..33   cost 5-6
  sC  at 34..49   cost 6-7
  bulk unchanged  at 50..817

Usage:
    cd ~/dev/research/sutro/sutro-problems
    python3 matmul/submissions/exp_layout_opt.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from matmul import score_16x16, generate_tiled_16x16
import math


def generate_tiled_16x16_opt1() -> str:
    """Tiled 16x16 with tmp at address 1, sA/sB/sC shifted to 2-49.

    Layout:
      tmp at 1        (was 49)
      sA  at 2..17    (was 1..16)
      sB  at 18..33   (was 17..32)
      sC  at 34..49   (was 33..48)
      A bulk: 50..305   (unchanged)
      B bulk: 306..561  (unchanged)
      C bulk: 562..817  (unchanged)
    """
    n, T = 16, 4
    tmp = 1
    sA = lambda ii, kk: 2 + ii * T + kk
    sB = lambda kk, jj: 2 + T * T + kk * T + jj
    sC = lambda ii, jj: 2 + 2 * T * T + ii * T + jj

    A_base = 50
    B_base = A_base + n * n
    C_base = B_base + n * n
    A_at = lambda i, j: A_base + i * n + j
    B_at = lambda i, j: B_base + i * n + j
    C_at = lambda i, j: C_base + i * n + j

    inputs = ([A_at(i, j) for i in range(n) for j in range(n)] +
              [B_at(i, j) for i in range(n) for j in range(n)])
    outputs = [C_at(i, j) for i in range(n) for j in range(n)]

    lines = [",".join(map(str, inputs))]
    nb = n // T
    for bi in range(nb):
        for bj in range(nb):
            for bk in range(nb):
                for ii in range(T):
                    for kk in range(T):
                        lines.append(f"copy {sA(ii,kk)},{A_at(bi*T+ii, bk*T+kk)}")
                for kk in range(T):
                    for jj in range(T):
                        lines.append(f"copy {sB(kk,jj)},{B_at(bk*T+kk, bj*T+jj)}")
                for ii in range(T):
                    for jj in range(T):
                        for kk in range(T):
                            lines.append(f"mul {tmp},{sA(ii,kk)},{sB(kk,jj)}")
                            if bk == 0 and kk == 0:
                                lines.append(f"copy {sC(ii,jj)},{tmp}")
                            else:
                                lines.append(f"add {sC(ii,jj)},{tmp}")
            for ii in range(T):
                for jj in range(T):
                    lines.append(f"copy {C_at(bi*T+ii, bj*T+jj)},{sC(ii,jj)}")

    lines.append(",".join(map(str, outputs)))
    return "\n".join(lines)


def addr_cost(addr):
    return math.isqrt(addr - 1) + 1


def analyze_layout(name, tmp_addr, sA_range, sB_range, sC_range):
    """Print read costs for each scratchpad region."""
    n, T, nb = 16, 4, 4
    inner_iters = T * T * T * nb * nb * nb  # total inner loop iterations
    adds = inner_iters - T * T * nb * nb    # subtracts first-iter copies

    sA_cost = sum(addr_cost(2 + ii*T + kk) for ii in range(T) for kk in range(T))
    sB_cost = sum(addr_cost(sB_range + kk*T + jj) for kk in range(T) for jj in range(T))
    sC_cost = sum(addr_cost(sC_range + ii*T + jj) for ii in range(T) for jj in range(T))

    print(f"  {name}: tmp@{tmp_addr}(cost={addr_cost(tmp_addr)})  "
          f"sA_sum={sA_cost}  sB_sum={sB_cost}  sC_sum={sC_cost}")


if __name__ == "__main__":
    baseline_ir = generate_tiled_16x16()
    opt1_ir     = generate_tiled_16x16_opt1()

    baseline_cost = score_16x16(baseline_ir)
    opt1_cost     = score_16x16(opt1_ir)

    print("Layout analysis:")
    analyze_layout("baseline", tmp_addr=49, sA_range=1,  sB_range=17, sC_range=33)
    analyze_layout("opt1    ", tmp_addr=1,  sA_range=2,  sB_range=18, sC_range=34)

    print(f"\nResults:")
    print(f"  baseline (tiled_16x16):  {baseline_cost:>10,}")
    print(f"  opt1 (tmp@1):            {opt1_cost:>10,}")
    print(f"  improvement:             {baseline_cost - opt1_cost:>+10,}  "
          f"({(baseline_cost - opt1_cost) / baseline_cost * 100:.1f}%)")

    # Save opt1 IR
    ir_path = Path(__file__).parent / "tiled_16x16_opt1.ir"
    ir_path.write_text(opt1_ir + "\n")
    print(f"\n  Saved: {ir_path}")
