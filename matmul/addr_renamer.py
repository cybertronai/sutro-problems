#!/usr/bin/env python3
"""
addr_renamer.py — reusable address-renaming pass for Dally-model IRs.

Usage as a module:

    from addr_renamer import rename_optimal, count_reads, optimal_remap

    new_ir, info = rename_optimal(ir_text)
    # info = {'reads': {...}, 'remap': {...}, 'orig_cost_estimate': N,
    #         'remapped_cost_estimate': N}

The renamer:

  1. Counts every READ in the IR (operands of add/sub/mul/copy and the exit
     line; writes are free, input placement is free).
  2. Sorts cells by descending read count and assigns the cheapest addresses
     (1, 2, 3, ...) in that order.  Ties are broken by original address asc
     for determinism.
  3. Rewrites every appearance of each cell (input line, operand lists, exit
     line) with its new address.

Because cost(addr) = ceil(sqrt(addr)) is strictly nondecreasing in addr,
greedy "highest-count → lowest-addr" is provably optimal among all
permutations *for a fixed schedule of operations*.  This is the
structure-preserving lower bound; closing it requires changing the
algorithm itself.

Caveats:

  * Cells that are written but never read (and not in the exit list) have
    zero reads, so they get no entry in `reads` and stay at their original
    address.  In practice every dest is either an operand of a later op or
    an output, so this rarely matters; the renamer here defaults
    write-only cells to keep their original address.
  * Input addresses that are actually used will show up as reads, so they
    get remapped consistently.  The simulator only requires input
    addresses to be distinct after remapping; greedy assignment guarantees
    distinctness because every cell with reads gets a unique rank.

Thin wrapper around the existing `lower_bound.py` analyzer in SutroAna.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

# Pull in the canonical implementation from the SutroAna toolbox so we
# don't drift.  Both this module and lower_bound.py share helpers.
_SUTRO_ANA = Path("/Users/yaroslavvb/Library/CloudStorage/Dropbox/git0/SutroAna")
if str(_SUTRO_ANA) not in sys.path:
    sys.path.insert(0, str(_SUTRO_ANA))

from lower_bound import (  # noqa: E402
    count_reads,
    optimal_remap,
    rewrite_ir,
    compute_cost,
    analyze,
    _cost,
)


def rename_optimal(ir: str) -> tuple[str, dict]:
    """Apply the optimal address rename to `ir`.

    Returns (new_ir, info) where info has keys:
      reads, remap, orig_cost_estimate, remapped_cost_estimate, gap
    """
    result = analyze(ir)
    info = {
        "reads": result["reads"],
        "remap": result["remap"],
        "orig_cost_estimate": result["current_cost"],
        "remapped_cost_estimate": result["lower_bound"],
        "gap": result["gap"],
    }
    return result["remapped_ir"], info


__all__ = [
    "rename_optimal",
    "count_reads",
    "optimal_remap",
    "rewrite_ir",
    "compute_cost",
    "analyze",
    "_cost",
]
