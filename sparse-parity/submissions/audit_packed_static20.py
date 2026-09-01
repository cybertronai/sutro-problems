"""Reproduce the fixed-suite audit for the packed static 20% record."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import mask_sparse_parity as mp
import packed_sparse_parity as packed


KEYS = [f"packed-static-20-audit-{letter}" for letter in "abcdefghij"]
OUT = HERE / "packed_static20_audit.json"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    ir = packed.generate_packed_static20().rstrip("\n") + "\n"
    committed = (HERE / "packedstatic20_mask32.ir").read_text()
    assert ir == committed
    run, cost, _ = mp._compile_vector(ir, mp.OP_CAP)
    suites = []
    successes = []
    for key in KEYS:
        inputs, targets, meta = mp.mask_suite(
            suite_key=key, n_secrets=256, repetitions=8
        )
        count = int(np.all(run(inputs) == targets, axis=1).sum())
        successes.append(count)
        suites.append({
            "key": key,
            "inputs_sha256": sha256(inputs.tobytes()),
            "targets_sha256": sha256(targets.tobytes()),
            "meta_sha256": sha256(repr(meta).encode()),
            "successes": count,
            "instances": len(targets),
        })
    payload = {
        "suite_version": mp.SUITE_VERSION,
        "ir_sha256": sha256(ir.encode()),
        "cost": cost,
        "lines": len(ir.splitlines()),
        "suites": suites,
        "summary": {
            "total_successes": sum(successes),
            "total_instances": 20480,
            "min_recovery": min(successes) / 2048,
            "mean_recovery": sum(successes) / 20480,
            "max_recovery": max(successes) / 2048,
        },
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
