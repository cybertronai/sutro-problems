"""Sweep (n_bits, k_sparse, n_samples) and report Dally-cost vs accuracy
for two prototype solvers. Run: ``python3 explore.py``.

Reference points from ../matmul:
  baseline_4x4  cost   1,316
  baseline_16x16 cost 340,704
  tiled_16x16   cost 133,783
"""
from __future__ import annotations

from sparse_parity import Cost, generate, solve_bruteforce, solve_gf2

CONFIGS = [
    # (label,         n,   k,  m_samples)
    ("user-target",   11,  3,   12),
    ("tiny",           8,  2,    8),
    ("medium",        16,  3,   16),
    ("medium-k4",     20,  4,   24),
    ("large",         32,  3,   32),
]

SOLVERS = [
    ("bruteforce", solve_bruteforce),
    ("gf2",        solve_gf2),
]

SEEDS = (0, 1, 2, 3, 4)


def main() -> None:
    print(f"{'label':<14}{'n':>4}{'k':>3}{'m':>5}  {'solver':<11}"
          f"{'cost':>12}  {'reads':>10}  acc")
    print("-" * 72)
    for label, n, k, m in CONFIGS:
        for solver_name, solver in SOLVERS:
            costs, reads, correct = [], [], 0
            for seed in SEEDS:
                X, y, secret = generate(n, k, m, seed=seed)
                cost = Cost(m=m, n=n)
                try:
                    found = solver(X, y, k, cost)
                except Exception as exc:
                    found = f"err: {exc}"
                if isinstance(found, list) and sorted(found) == secret:
                    correct += 1
                costs.append(cost.read_cost)
                reads.append(cost.reads)
            avg_cost = sum(costs) / len(costs)
            avg_reads = sum(reads) / len(reads)
            acc = correct / len(SEEDS)
            print(
                f"{label:<14}{n:>4}{k:>3}{m:>5}  {solver_name:<11}"
                f"{avg_cost:>12,.0f}  {avg_reads:>10,.0f}"
                f"  {acc:.0%}"
            )


if __name__ == "__main__":
    main()
