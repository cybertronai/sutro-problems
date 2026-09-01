"""Reproduce the packed-record adjudication audit.

Unlike ``suite_key=None``, every suite below has a stable public key.  The
generated JSON records hashes of the exact input/target arrays and reports
integer successes with denominators, so record-selection claims can be
audited rather than relying on unrepeatable random draws.

Run from ``sparse-parity``::

    python3 submissions/audit_packed_records.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import mask_sparse_parity as mp
import packedsis
import packedwalk
import septwalk


N_DRAWS = 10
N_SECRETS = 256
REPETITIONS = 8
SUITE_KEYS = [f"packed-record-audit-v1-{i:02d}" for i in range(N_DRAWS)]
OUT = Path(__file__).with_name("packed_records_audit.json")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _normalized_ir(ir: str) -> str:
    return ir.rstrip("\n") + "\n"


def candidates():
    return [
        ("packedsis_cap1_s3", "generate_packed_sis(cap=1, seed=3)",
         packedsis.generate_packed_sis(cap=1, seed=3), None),
        *[
            (f"packedsis_cap2_s13_g{g2}",
             f"generate_packed_sis(cap=2, seed=13, g2={g2})",
             packedsis.generate_packed_sis(cap=2, seed=13, g2=g2),
             "packedsis_pcap2_mask32.ir" if g2 == 8 else None)
            for g2 in (0, 2, 4, 6, 8)
        ],
        ("packedwalk_cap1_s0", "packedwalk.generate(1, 1, seed=0)",
         packedwalk.generate(1, 1, seed=0), None),
        ("packedwalk_cap2_s0", "packedwalk.generate(1, 2, seed=0)",
         packedwalk.generate(1, 2, seed=0), None),
        ("packedwalk_cap2_s5", "packedwalk.generate(1, 2, seed=5)",
         packedwalk.generate(1, 2, seed=5),
         "packedwalk1_cap2_s5_mask32.ir"),
        ("packedwalk_cap3_s0", "packedwalk.generate(1, 3, seed=0)",
         packedwalk.generate(1, 3, seed=0), None),
        ("packedsis_cap3_s13", "generate_packed_sis(cap=3, seed=13)",
         packedsis.generate_packed_sis(cap=3, seed=13),
         "packedsis_cap3_s13_mask32.ir"),
        ("septwalk_cap3", "septwalk.generate_staged(weight_cap=3)",
         septwalk.generate_staged(weight_cap=3),
         "septwalk_wcap3_mask32.ir"),
        ("septwalk_cap5", "septwalk.generate_staged(weight_cap=5)",
         septwalk.generate_staged(weight_cap=5), "septwalk_mask32.ir"),
    ]


def main() -> None:
    compiled = []
    for name, call, generated, committed_name in candidates():
        ir = _normalized_ir(generated)
        if committed_name is not None:
            committed = Path(__file__).with_name(committed_name).read_text()
            if ir != _normalized_ir(committed):
                raise AssertionError(
                    f"{name} does not regenerate {committed_name} byte-identically")
        run, cost, n_inputs = mp._compile_vector(ir, mp.OP_CAP)
        if n_inputs != mp._n_inputs(mp.MASK32):
            raise AssertionError(f"{name} has {n_inputs} inputs")
        compiled.append({
            "name": name,
            "call": call,
            "committed_ir": committed_name,
            "ir_sha256": _sha256(ir.encode()),
            "cost": cost,
            "run": run,
            "draws": [],
        })

    suites = []
    for key in SUITE_KEYS:
        inputs, targets, meta = mp.mask_suite(
            suite_key=key, n_secrets=N_SECRETS, repetitions=REPETITIONS)
        suites.append({
            "suite_key": key,
            "n_instances": int(len(inputs)),
            "inputs_shape": list(inputs.shape),
            "inputs_dtype": str(inputs.dtype),
            "inputs_sha256": _sha256(inputs.tobytes()),
            "targets_shape": list(targets.shape),
            "targets_dtype": str(targets.dtype),
            "targets_sha256": _sha256(targets.tobytes()),
            "meta_sha256": _sha256(repr(meta).encode()),
        })
        for candidate in compiled:
            output = candidate["run"](inputs)
            successes = int(np.all(output == targets, axis=1).sum())
            candidate["draws"].append({
                "suite_key": key,
                "successes": successes,
                "n_instances": int(len(targets)),
                "recovery": successes / len(targets),
            })

    records = []
    for candidate in compiled:
        candidate.pop("run")
        recoveries = [draw["recovery"] for draw in candidate["draws"]]
        candidate["summary"] = {
            "n_draws": len(recoveries),
            "n_instances_per_draw": N_SECRETS * REPETITIONS,
            "total_successes": sum(d["successes"] for d in candidate["draws"]),
            "total_instances": sum(d["n_instances"] for d in candidate["draws"]),
            "min_recovery": min(recoveries),
            "mean_recovery": sum(recoveries) / len(recoveries),
            "max_recovery": max(recoveries),
        }
        records.append(candidate)

    payload = {
        "audit_version": "packed-record-audit-v1",
        "suite_version": mp.SUITE_VERSION,
        "n_secrets_per_draw": N_SECRETS,
        "repetitions_per_secret": REPETITIONS,
        "suites": suites,
        "candidates": records,
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {OUT}")
    for record in records:
        summary = record["summary"]
        print(record["name"], record["cost"],
              f"min={summary['min_recovery']:.4f}",
              f"mean={summary['mean_recovery']:.4f}",
              f"max={summary['max_recovery']:.4f}",
              f"n={summary['total_instances']}")


if __name__ == "__main__":
    main()
