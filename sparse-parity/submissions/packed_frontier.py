"""Generators for the compact packed-scan sparse-parity records.

The 40/60/80% circuits keep every lower-weight coefficient mask, then take a
fixed subset of the next BRGC-filtered weight class.  The 40% circuit skips one
low-yield state inside its prefix; 60/80 use plain prefixes.  Compact
exact-weight/control flow, per-band interval-order biases, reverse traversal of
the frozen nonzero paths, and a post-allocation no-op pass reduce energy
without changing the evaluator.
"""
from __future__ import annotations

import hashlib
import random
import sys
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional

HERE = Path(__file__).resolve().parent
SPARSE_PARITY = HERE.parent
if str(SPARSE_PARITY) not in sys.path:
    sys.path.insert(0, str(SPARSE_PARITY))

import mask_sparse_parity as mp  # noqa: E402
import packed_sparse_parity as packed  # noqa: E402


class Config(NamedTuple):
    cap: int
    top_weight_states: Optional[int]
    walk_order_bias: Optional[float]
    omitted_top_index: Optional[int] = None
    two_opt_seed: Optional[int] = None
    reverse_nonzero_path: bool = False


CONFIGS: Dict[int, Config] = {
    40: Config(2, 55, 3.0, 47, None, True),
    60: Config(3, 19, 8.0, None, 67, True),
    80: Config(3, 269, 1.0, None, 92, True),
}


def states_for_config(config: Config) -> Optional[List[int]]:
    """Return the deterministic state subset, or None for a complete cap."""
    if config.top_weight_states is None:
        return None
    states: List[int] = []
    selected = 0
    seen = 0
    for state in packed.bounded_weight_gray_states(14, config.cap):
        weight = packed._bit_count(state)
        if weight < config.cap:
            states.append(state)
        elif selected < config.top_weight_states:
            omit = config.omitted_top_index
            if omit is None or seen != omit:
                states.append(state)
                selected += 1
            seen += 1
    if selected != config.top_weight_states:
        raise AssertionError((selected, config))
    if config.two_opt_seed is not None:
        rng = random.Random(config.two_opt_seed)
        for _step in range(50_000):
            left = rng.randrange(1, len(states) - 1)
            right = rng.randrange(left + 1, len(states))
            before = packed._bit_count(states[left - 1] ^ states[left])
            after = packed._bit_count(states[left - 1] ^ states[right])
            if right + 1 < len(states):
                before += packed._bit_count(states[right] ^ states[right + 1])
                after += packed._bit_count(states[left] ^ states[right + 1])
            if after < before:
                states[left:right + 1] = reversed(states[left:right + 1])
    if config.reverse_nonzero_path:
        states[1:] = reversed(states[1:])
    return states


def generate_packed_frontier(target: int) -> str:
    """Generate one of the 40%, 60%, or 80% record candidates."""
    try:
        config = CONFIGS[target]
    except KeyError as exc:
        raise ValueError(f"unsupported target: {target}") from exc
    return packed.generate_packed_scan(
        config.cap,
        walk_states=states_for_config(config),
        compact_predicates=True,
        compact_flow=True,
        walk_order_bias=config.walk_order_bias,
    )


def main() -> None:
    for target in CONFIGS:
        ir = generate_packed_frontier(target)
        result = mp.evaluate_mask(ir)
        output = HERE / f"packedfrontier{target}_mask32.ir"
        output.write_text(ir + "\n", encoding="utf-8")
        digest = hashlib.sha256((ir + "\n").encode()).hexdigest()
        print(
            f"{target:>3}%  cost={result.cost:>7,}  "
            f"recovery={result.recovery:.10f}  "
            f"lines={len(ir.splitlines()):>6,}  sha256={digest}"
        )


if __name__ == "__main__":
    main()
