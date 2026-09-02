"""Audit packed-frontier recovery on fixed final-sized suites."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPARSE_PARITY = HERE.parent
if str(SPARSE_PARITY) not in sys.path:
    sys.path.insert(0, str(SPARSE_PARITY))

import mask_sparse_parity as mp  # noqa: E402

try:  # Support both direct execution and package imports from tests.
    from .packed_frontier import CONFIGS, generate_packed_frontier
except ImportError:  # pragma: no cover - exercised by direct script use
    from packed_frontier import CONFIGS, generate_packed_frontier  # type: ignore


SEARCH_SUITES = (
    "packed-record-audit-v1-00",
    "packed-record-audit-v1-01",
    "packed-record-audit-v1-02",
    "packed-record-audit-v1-03",
    "packed-record-audit-v1-04",
    "packed-record-audit-v1-05",
    "packed-record-audit-v1-06",
    "packed-record-audit-v1-07",
    "packed-record-audit-v1-08",
    "packed-record-audit-v1-09",
    "npow-packed-frontier-a-20260901",
    "npow-packed-frontier-b-20260901",
)
GATE_SUITES = (
    "arbor-packed-merge-v1-00",
    "arbor-packed-merge-v1-01",
    "arbor-packed-merge-v1-02",
    "arbor-packed-merge-v1-03",
)
SUITES = SEARCH_SUITES + GATE_SUITES
OUTPUT = HERE / "packed_frontier_audit.json"


def suite_hash(inputs, masks, metadata) -> str:
    payload = inputs.tobytes() + masks.tobytes() + repr(metadata).encode()
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    compiled = {}
    for target in CONFIGS:
        ir = generate_packed_frontier(target)
        run, cost, input_count = mp._compile_vector(ir, mp.OP_CAP)
        compiled[target] = (run, cost, input_count)

    result = {
        "suite_version": mp.SUITE_VERSION,
        "n_secrets": mp.FINAL_SECRETS,
        "repetitions": mp.FINAL_REPS,
        "suites": [],
    }
    for key in SUITES:
        inputs, masks, metadata = mp.mask_suite(
            suite_key=key,
            n_secrets=mp.FINAL_SECRETS,
            repetitions=mp.FINAL_REPS,
        )
        record = {
            "key": key,
            "instances": len(masks),
            "sha256": suite_hash(inputs, masks, metadata),
            "targets": {},
        }
        for target, (run, cost, input_count) in compiled.items():
            if input_count != inputs.shape[1]:
                raise AssertionError((input_count, inputs.shape))
            outputs = run(inputs)
            successes = int((outputs == masks).all(axis=1).sum())
            record["targets"][str(target)] = {
                "cost": cost,
                "successes": successes,
                "denominator": len(masks),
                "recovery": successes / len(masks),
            }
        result["suites"].append(record)

    result["summary"] = {}
    for target in compiled:
        draws = [
            suite["targets"][str(target)]
            for suite in result["suites"]
        ]
        total_successes = sum(draw["successes"] for draw in draws)
        total_instances = sum(draw["denominator"] for draw in draws)
        result["summary"][str(target)] = {
            "draws": len(draws),
            "instances_per_draw": draws[0]["denominator"],
            "total_successes": total_successes,
            "total_instances": total_instances,
            "min_recovery": min(draw["recovery"] for draw in draws),
            "mean_recovery": total_successes / total_instances,
            "max_recovery": max(draw["recovery"] for draw in draws),
        }

    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
