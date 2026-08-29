"""GF(2) Gaussian-elimination submission for the medium sparse-parity
instance (n=8, k=3, m_train=8, m_test=64).

Same algorithm as ``ge_small.py`` (branchless GF(2) row-reduction with
full pivot tracking and a brute-force fallback for the rare rank-deficient
cases) — generalized to ``n=8``, ``k=3``, ``m=8``, ``m_test=64``. The
``_generate_ge(spec)`` helper is shared with the small submission.

For ``n=8``, ``k=3`` the brute-force fallback enumerates the
``C(8, 3) = 56`` weight-``k`` candidates; each column appears in
``C(7, 2) = 21`` of them.

Run ``python ge_medium.py`` to regenerate ``ge_medium.ir`` and print
its cost.
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

import sparse_parity  # noqa: E402
from sparse_parity import MEDIUM, score_medium  # noqa: E402
from ge_small import _generate_ge  # noqa: E402


def generate_ge_medium() -> str:
    """GE-based predictor IR for the medium instance."""
    return _generate_ge(MEDIUM)


if __name__ == "__main__":
    ir = generate_ge_medium()
    cost = score_medium(ir)
    out = os.path.join(_HERE, "ge_medium.ir")
    with open(out, "w") as f:
        f.write(ir)
        f.write("\n")
    n_ops = len(ir.splitlines()) - 2
    print(f"  ge_medium.ir      cost={cost:>9,}  ops={n_ops:>5,}  -> {out}")
