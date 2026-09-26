"""Run the unchanged official scorer with a bounded Modal budget.

Usage: python verify_modal.py --runs 3 --output /tmp/mnist-a100-verification
Only this controller imports Modal; fast_mlp.py is the standalone submission.
"""

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
if modal.is_local():
    PORT = HERE.parents[1]
    sys.path.insert(0, str(PORT))
    import mnist
    from run_modal import image
else:
    image = None

app = modal.App("mnist-a100-graph-submission-20260926", image=image)


@app.function(gpu="A100-80GB", cpu=(1, 2), memory=(4096, 8192),
              timeout=600, startup_timeout=120, retries=0, max_containers=1,
              single_use_containers=True, scaledown_window=2)
def verify(source: str) -> dict:
    import os
    import platform
    import subprocess
    import torch

    entry = Path("/root/entry/fast_mlp.py")
    entry.parent.mkdir(exist_ok=True)
    entry.write_text(source)
    started = time.monotonic()
    hardware = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"],
        capture_output=True, text=True, timeout=20).stdout.strip()
    result = subprocess.run(
        [sys.executable, "/root/mnist.py", f"{entry}:fast_mlp", "--difficulty", "1"],
        capture_output=True, text=True, timeout=550,
        env={**os.environ, "MNIST_SANDBOX": "required"})
    return {
        "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr,
        "elapsed_seconds": time.monotonic() - started, "hardware": hardware,
        "python": platform.python_version(), "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "scorer_sha256": hashlib.sha256(Path("/root/mnist.py").read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(entry.read_bytes()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=3, choices=range(1, 4))
    parser.add_argument("--output", type=Path, default=HERE / "rerun-evidence")
    args = parser.parse_args()
    source = (HERE / "fast_mlp.py").read_text()
    mnist.check_source(source.encode(), "fast_mlp", "fast_mlp.py")
    flags = mnist.review_flags(source.encode())
    print(f"source: {len(source.encode())} bytes; review flags: {flags}", flush=True)
    evidence = args.output
    evidence.mkdir(parents=True, exist_ok=True)
    ledger_path = evidence / "budget.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {
        "cap_usd": 5.0, "build_reserve_usd": 0.50,
        "per_call_reserve_usd": 0.80,
        "rate_upper_bound_usd_per_second": 0.001,
        "reservation_basis": "600s function + 120s startup + 80s shutdown allowance, at $0.001/s; separate $0.50 CPU image-build reserve",
        "pricing_url": "https://modal.com/pricing", "attempts": []}
    for _ in range(args.runs):
        total_reserved = round(ledger["build_reserve_usd"] + (len(ledger["attempts"]) + 1) * ledger["per_call_reserve_usd"], 2)
        if total_reserved > ledger["cap_usd"]:
            raise RuntimeError("The next run would exceed the reserved $5 budget")
        number = len(ledger["attempts"]) + 1
        record = {"attempt": number, "reserved_usd": ledger["per_call_reserve_usd"],
                  "started_utc": datetime.now(timezone.utc).isoformat(), "status": "reserved"}
        ledger["attempts"].append(record)
        ledger["total_reserved_usd"] = total_reserved
        ledger_path.write_text(json.dumps(ledger, indent=2) + "\n")
        started = time.monotonic()
        with modal.enable_output(), app.run():
            record["app_id"] = app.app_id
            ledger_path.write_text(json.dumps(ledger, indent=2) + "\n")
            result = verify.remote(source)
        result.update(app_id=record["app_id"], attempt=number,
                      completed_utc=datetime.now(timezone.utc).isoformat(),
                      controller_seconds=time.monotonic() - started)
        (evidence / f"run-{number}.json").write_text(json.dumps(result, indent=2) + "\n")
        assert result["source_sha256"] == hashlib.sha256(source.encode()).hexdigest()
        assert result["scorer_sha256"] == hashlib.sha256((PORT / "mnist.py").read_bytes()).hexdigest()
        print(result["stdout"], flush=True)
        if result["stderr"]:
            print(result["stderr"], file=sys.stderr, flush=True)
        record.update(status="passed" if result["returncode"] == 0 else "failed",
                      controller_seconds=result["controller_seconds"],
                      function_seconds=result["elapsed_seconds"])
        ledger_path.write_text(json.dumps(ledger, indent=2) + "\n")
        if result["returncode"]:
            raise SystemExit(result["returncode"])


if __name__ == "__main__":
    main()
