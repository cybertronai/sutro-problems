"""Exact two-tier allocation using integral min-cost flow."""
from __future__ import annotations

import argparse
import itertools
import json
import random
import time
from pathlib import Path

import numpy as np
from ortools.graph.python import min_cost_flow

from core import DEFAULT_IR, assignment_score, color_tiers, cost, lifetimes, load, save


def optimize_pair(lives, tiers, low, high, rng=None):
    selected = [(v, s, e, w) for v, (s, e, w) in lives.items()
                if w and tiers[v] in (low, high)]
    if not selected:
        return None
    times = sorted({x for _, s, e, _ in selected for x in (s, e)})
    index = {t: k for k, t in enumerate(times)}
    n = len(times)
    delta = np.zeros(n, dtype=np.int64)
    for _, s, e, _ in selected:
        delta[index[s]] += 1
        delta[index[e]] -= 1
    occupancy = np.cumsum(delta)[:-1]
    cap_low, cap_high = 2 * low - 1, 2 * high - 1
    lower = np.maximum(0, occupancy - cap_high)
    if np.any(lower > cap_low):
        return None
    mcf = min_cost_flow.SimpleMinCostFlow()
    mcf.add_arcs_with_capacity_and_unit_cost(
        np.arange(n - 1), np.arange(1, n), cap_low - lower,
        np.zeros(n - 1, dtype=np.int64))
    costs = [-(w * 100_000_000) + rng.randrange(100) if rng else -w
             for _, _, _, w in selected]
    job_arcs = mcf.add_arcs_with_capacity_and_unit_cost(
        np.array([index[s] for _, s, _, _ in selected]),
        np.array([index[e] for _, _, e, _ in selected]),
        np.ones(len(selected), dtype=np.int64), np.array(costs, dtype=np.int64))
    mcf.set_node_supply(0, cap_low)
    mcf.set_node_supply(n - 1, -cap_low)
    status = mcf.solve()
    if status != mcf.OPTIMAL:
        return None
    flows = mcf.flows(job_arcs)
    updated = {v: low if flow else high for (v, _, _, _), flow in zip(selected, flows)}
    improvement = sum(w * (tiers[v] - updated[v]) for v, _, _, w in selected)
    return updated, improvement


def improve(p, rounds=2, seed=0, max_seconds=120, allow_ties=False, checkpoint=None):
    lives = lifetimes(p)
    tiers = {v: cost(a) for v, a in p.assignment.items()}
    rng = random.Random(seed)
    start = time.monotonic()
    initial = assignment_score(p)
    running = initial
    history = []
    for round_i in range(rounds):
        choices = list(itertools.combinations(sorted(set(tiers.values())), 2))
        rng.shuffle(choices)
        changed = 0
        for lo, hi in choices:
            if time.monotonic() - start > max_seconds:
                break
            result = optimize_pair(lives, tiers, lo, hi, rng if allow_ties else None)
            if result is None:
                continue
            new_tiers, gain = result
            if gain < 0:
                raise AssertionError(f"Exact pair allocator worsened feasible incumbent: {gain}")
            if gain > 0 or allow_ties:
                tiers.update(new_tiers)
            if gain > 0:
                running -= gain
                changed += 1
                history.append({"round": round_i, "tiers": [lo, hi], "gain": gain,
                                "score": running})
                p.assignment = color_tiers(p, tiers)
                assert assignment_score(p) == running
                print(json.dumps(history[-1]), flush=True)
                if checkpoint:
                    save(p, checkpoint, {"method": "exact pair-tier min-cost flow", "history": history})
        print(f"round={round_i} initial={initial} current={running} improvements={changed} elapsed={time.monotonic()-start:.2f}", flush=True)
        if (not changed and not allow_ties) or time.monotonic() - start > max_seconds:
            break
    p.assignment = color_tiers(p, tiers)
    assert assignment_score(p) == running
    return p, history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ir", nargs="?", type=Path, default=DEFAULT_IR)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--seconds", type=float, default=180)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ties", action="store_true")
    a = ap.parse_args()
    p, history = improve(load(a.ir), a.rounds, a.seed, a.seconds, a.ties, "pair")
    print(save(p, "pair_final", {"source": str(a.ir), "seed": a.seed, "history": history}), flush=True)


if __name__ == "__main__":
    main()
