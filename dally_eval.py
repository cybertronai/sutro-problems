"""Thin wrapper around the dally-eval Rust engine for fast IR scoring.

The engine lives at ~/sutro/dally-eval (github.com/cybertronai/dally-eval);
it scores the same Bill Dally read-cost model as this repo's Python
evaluators, bit-exactly, ~500x faster on batches. This shim shells out
to its `verify`/`run` CLI so Python callers get native-speed scoring
with a transparent fallback to the local Python evaluator when the
binary is unavailable.

Usage:
    import dally_eval

    dally_eval.static_cost(ir_text)          # -> int (or None)
    dally_eval.available()                  # -> bool

Set DALLY_EVAL_BIN to point at the binary; the default picks the
release build in the sibling checkout.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from typing import Optional

_DEFAULT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    os.pardir, "dally-eval", "target", "release", "dally-eval",
)


def _binary() -> Optional[str]:
    cand = os.environ.get("DALLY_EVAL_BIN", os.path.abspath(_DEFAULT))
    if os.path.isfile(cand) and os.access(cand, os.X_OK):
        return cand
    return shutil.which("dally-eval")


def available() -> bool:
    return _binary() is not None


def static_cost(ir: str) -> Optional[int]:
    """Static read-cost of an IR program via the Rust engine.

    Returns None when the engine binary is not built (callers should
    fall back to the native Python scorer). Raises ValueError on
    malformed IR.
    """
    binary = _binary()
    if binary is None:
        return None
    proc = subprocess.run(
        [binary, "verify"],
        input=ir,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if proc.returncode != 0:
        raise ValueError(f"dally-eval rejected IR: {proc.stderr.strip()}")
    return json.loads(proc.stdout)["cost"]
