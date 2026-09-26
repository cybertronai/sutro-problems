"""Conservative app-lifetime budget guard. No Modal calls or GPU launches.

Reserve BEFORE entering app.run()/creating an app. One canonical ledger covers
every phase. While an app is active or ambiguous its full reservation remains
held. After provider evidence says the app is stopped with zero tasks, charge
two fully occupied workers for the entire elapsed interval through that check
and release only the unused reservation. This is an upper-envelope accounting
guard, not a provider invoice or account-wide billing limit. Keep $2 unallocated
for image-building/other overhead; use one shared max-two-container pool.

Worker contract: A100-40GB or T4, cpu=(4,4), memory=(16384,16384), retries=0, at most
MAX_CONTAINERS (8) containers, min_containers=buffer_containers=0; no region or
non-preemptible premium.  Each reservation records the GPU it ran on and is charged
at that GPU's worker rate.
Use each returned work_deadline_unix in workers and the local controller.
Stop the app at that deadline; shutdown_grace_seconds is reserved for shutdown.
Do not call finish_app merely because a local context/RPC ended.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from decimal import Decimal, ROUND_CEILING
import fcntl
import json
import os
from pathlib import Path
import tempfile
import time

MAX_CONTAINERS = 8  # upper bound; each reservation declares its own container count
CONTINGENCY_USD = Decimal("2")

# Per-worker upper-envelope rates, from https://modal.com/pricing (read 2026-09-24):
#   A100-40GB  $0.000583/s   T4  $0.000164/s
#   CPU        $0.0000131 per physical core-second  x 4 cores  = $0.0000524/s
#   memory     $0.00000222 per GiB-second           x 16 GiB   = $0.00003552/s
# A100-40GB: 0.000583 + 0.0000524 + 0.00003552 = 0.00067122
# T4:        0.000164 + 0.0000524 + 0.00003552 = 0.00025192
# The earlier 0.00065316 / 0.00023420 figures assumed an 8 GiB worker; this study
# reserves memory=(16384,16384), so both rates are recomputed for 16 GiB.
WORKER_RATES = {"A100-40GB": Decimal("0.00067122"), "T4": Decimal("0.00025192")}
DEFAULT_GPU = "A100-40GB"
WORKER_USD_PER_SECOND = WORKER_RATES[DEFAULT_GPU]   # most expensive worker
APP_USD_PER_SECOND = MAX_CONTAINERS * WORKER_USD_PER_SECOND
PRICING_SOURCE = "https://modal.com/pricing"
_MICRO = Decimal(1_000_000)


class BudgetError(ValueError):
    pass


def _micro(value):
    return int((Decimal(str(value)) * _MICRO).to_integral_value(rounding=ROUND_CEILING))


AMENDABLE_POLICY_FIELDS = {"max_containers", "app_usd_per_second"}


def _rate(gpu):
    """Per-second rate of one fully busy worker of this GPU type."""
    if gpu not in WORKER_RATES:
        raise BudgetError(f"gpu must be one of {sorted(WORKER_RATES)}, got {gpu!r}")
    return WORKER_RATES[gpu]


def _cost(seconds, containers=None, gpu=DEFAULT_GPU):
    """Upper-envelope cost of ``containers`` fully busy ``gpu`` workers for ``seconds``."""
    containers = MAX_CONTAINERS if containers is None else containers
    if isinstance(containers, bool) or not isinstance(containers, int) or not 1 <= containers <= MAX_CONTAINERS:
        raise BudgetError(f"containers must be an integer in 1..{MAX_CONTAINERS}")
    return _micro(Decimal(str(seconds)) * containers * _rate(gpu))


def _row_containers(ledger, row):
    return int(row.get("containers", ledger["policy"]["max_containers"]))


def _row_gpu(ledger, row):
    """A row without an explicit GPU predates the two-rate ledger: charge the A100 rate."""
    return str(row.get("gpu", DEFAULT_GPU))


def _policy(budget):
    cap = _micro(budget)
    if cap <= _micro(CONTINGENCY_USD):
        raise BudgetError("Explicit total budget must exceed the $2 contingency")
    return {"total_budget_microusd": cap, "contingency_microusd": _micro(CONTINGENCY_USD),
            "max_containers": MAX_CONTAINERS, "worker_usd_per_second": str(WORKER_USD_PER_SECOND),
            "app_usd_per_second": str(APP_USD_PER_SECOND),
            "worker_rates_usd_per_second": {name: str(rate) for name, rate in sorted(WORKER_RATES.items())},
            "gpu": sorted(WORKER_RATES),
            "cpu": [4, 4], "memory_mib": [16384, 16384], "pricing_source": PRICING_SOURCE}


@contextmanager
def _locked(path):
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_name(path.name + ".lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield path
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _write(path, ledger):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as f:
            temporary = Path(f.name)
            json.dump(ledger, f, indent=2, sort_keys=True, allow_nan=False)
            f.write("\n"); f.flush(); os.fsync(f.fileno())
        os.replace(temporary, path)
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _read(path, total_budget_usd=None):
    if not path.exists():
        if total_budget_usd is None:
            raise BudgetError("No ledger; supply the explicitly authorized total budget when reserving")
        return {"schema_version": 1, "policy": _policy(total_budget_usd), "apps": {}, "blocked_reason": None}
    try:
        ledger = json.loads(path.read_text())
        if ledger["schema_version"] != 1:
            raise BudgetError("Unsupported ledger schema")
        stored = Decimal(ledger["policy"]["total_budget_microusd"]) / _MICRO
        if ledger["policy"] != _policy(stored):
            raise BudgetError("Ledger policy altered; refusing to reset it")
        if total_budget_usd is not None and ledger["policy"] != _policy(total_budget_usd):
            raise BudgetError("Budget changed; refusing to increase/reset the existing ledger")
        for key, row in ledger["apps"].items():
            if key != row["reservation_id"] or row["status"] not in ("reserved", "running", "stopped"):
                raise BudgetError("Invalid reservation identity/status")
            containers = _row_containers(ledger, row)
            gpu = _row_gpu(ledger, row)
            if row["reserved_microusd"] != _cost(row["work_seconds"] + row["shutdown_grace_seconds"], containers, gpu):
                raise BudgetError("Reservation amount altered")
            if row["status"] == "stopped":
                if not row["verified_stopped"] or row["remaining_tasks"] != 0:
                    raise BudgetError("Stopped reservation lacks termination evidence")
                if row["charged_upper_microusd"] != _cost(row["elapsed_seconds"], containers, gpu):
                    raise BudgetError("Charge amount altered")
        return ledger
    except BudgetError:
        raise
    except (KeyError, TypeError, ValueError, OSError) as e:
        raise BudgetError("Invalid budget ledger; no reset allowed") from e


def _summary(ledger):
    charged = sum(x.get("charged_upper_microusd", 0) for x in ledger["apps"].values() if x["status"] == "stopped")
    held = sum(x["reserved_microusd"] for x in ledger["apps"].values() if x["status"] != "stopped")
    available = ledger["policy"]["total_budget_microusd"] - ledger["policy"]["contingency_microusd"] - charged - held
    return {"charged_upper_usd": charged / 1e6, "active_reserved_usd": held / 1e6,
            "available_worker_usd": available / 1e6,
            "active_ids": [k for k, x in ledger["apps"].items() if x["status"] != "stopped"],
            "blocked_reason": ledger.get("blocked_reason")}


def read_ledger(ledger_path):
    """Read validated ledger and current charge/reservation summary."""
    with _locked(ledger_path) as path:
        ledger = _read(path)
        return {**ledger, "summary": _summary(ledger)}


def reserve_app(ledger_path, reservation_id, *, total_budget_usd, work_seconds,
                shutdown_grace_seconds=60, containers=None, gpu=DEFAULT_GPU):
    """Atomically reserve before app creation; return worker/controller deadlines.

    work_seconds covers app creation/startup plus useful work, NOT just kernels.
    Only one active app is allowed; its one function pool must cap at ``containers``
    GPUs (default MAX_CONTAINERS) of type ``gpu``, and the reservation charges that
    many fully busy workers of that type for the whole reserved lifetime.
    Reusing an ID, changing a cap, or any unresolved app fails closed.
    """
    if not isinstance(reservation_id, str) or not reservation_id.strip():
        raise BudgetError("A unique nonempty reservation_id is required")
    for value in (work_seconds, shutdown_grace_seconds):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise BudgetError("Work and shutdown durations must be positive integer seconds")
    if shutdown_grace_seconds < 30:
        raise BudgetError("Reserve at least 30 seconds for verified shutdown")
    with _locked(ledger_path) as path:
        ledger = _read(path, total_budget_usd)
        s = _summary(ledger)
        if s["blocked_reason"] or s["active_ids"] or reservation_id in ledger["apps"]:
            raise BudgetError("Prior app unresolved, ledger blocked, or reservation ID reused")
        containers = MAX_CONTAINERS if containers is None else containers
        gpu = str(gpu)
        _rate(gpu)                      # fail closed on an unpriced GPU
        reserved = _cost(work_seconds + shutdown_grace_seconds, containers, gpu)
        charged = sum(x.get("charged_upper_microusd", 0) for x in ledger["apps"].values() if x["status"] == "stopped")
        available = _micro(total_budget_usd) - _micro(CONTINGENCY_USD) - charged
        if reserved > available:
            raise BudgetError(f"Reservation ${reserved/1e6:.6f} exceeds available ${available/1e6:.6f}")
        now = time.time()
        row = {"reservation_id": reservation_id, "status": "reserved", "app_id": None,
               "started_at_unix": now, "started_monotonic": time.monotonic(),
               "work_seconds": work_seconds, "shutdown_grace_seconds": shutdown_grace_seconds,
               "work_deadline_unix": now + work_seconds,
               "shutdown_deadline_unix": now + work_seconds + shutdown_grace_seconds,
               "reserved_microusd": reserved, "reserved_usd": reserved / 1e6,
               "containers": containers, "gpu": gpu,
               "worker_usd_per_second": str(_rate(gpu))}
        ledger["apps"][reservation_id] = row
        _write(path, ledger)
        return dict(row)


def attach_app(ledger_path, reservation_id, app_id):
    """Record the actual Modal app ID immediately after creation; never overwrite."""
    if not isinstance(app_id, str) or not app_id.startswith("ap-"):
        raise BudgetError("A real Modal app ID is required")
    with _locked(ledger_path) as path:
        ledger = _read(path); row = ledger["apps"][reservation_id]
        if row["status"] == "stopped" or row["app_id"] not in (None, app_id):
            raise BudgetError("Cannot replace or reactivate a recorded app")
        if any(x["app_id"] == app_id for k, x in ledger["apps"].items() if k != reservation_id):
            raise BudgetError("App ID already attached to another reservation")
        row.update({"app_id": app_id, "status": "running"})
        _write(path, ledger)
        return dict(row)


def finish_app(ledger_path, reservation_id, *, verified_stopped, remaining_tasks,
               app_state, evidence, outcome="completed"):
    """Charge through the verification moment and release unused reservation.

    Caller must just have observed provider app_state='stopped' and zero tasks.
    evidence is a short description/path to saved provider evidence. Ambiguous
    creation with no known app ID remains reserved and cannot be settled here.
    An observed lifetime beyond reserved shutdown time blocks future launches.
    """
    if verified_stopped is not True or remaining_tasks != 0 or app_state != "stopped" or not evidence:
        raise BudgetError("Require verified stopped app, zero tasks and saved evidence")
    with _locked(ledger_path) as path:
        ledger = _read(path); row = ledger["apps"][reservation_id]
        if row["status"] == "stopped" or not row["app_id"]:
            raise BudgetError("Already settled or actual app identity is unknown")
        elapsed = max(0., time.time() - row["started_at_unix"], time.monotonic() - row["started_monotonic"])
        charge = _cost(elapsed, _row_containers(ledger, row), _row_gpu(ledger, row))
        row.update({"status": "stopped", "verified_stopped": True, "remaining_tasks": 0,
                    "app_state": app_state, "evidence": str(evidence), "outcome": outcome,
                    "finished_at_unix": time.time(), "elapsed_seconds": elapsed,
                    "charged_upper_microusd": charge, "charged_upper_usd": charge / 1e6})
        if charge > row["reserved_microusd"]:
            ledger["blocked_reason"] = "Verified lifetime exceeded its reservation; reconcile before any further launch"
        _write(path, ledger)
        return {**row, "summary": _summary(ledger)}


def amend_policy(ledger_path, reason):
    """Adopt the current module policy (e.g. a new MAX_CONTAINERS) in an existing ledger.

    Refuses while any reservation is unresolved. Only max_containers and the derived
    app rate may change; the total budget and worker rate are immutable. Historical
    rows keep the container count they were charged under (backfilled explicitly)
    and every recorded amount is re-verified before the amendment is written.
    """
    if not isinstance(reason, str) or not reason.strip():
        raise BudgetError("An amendment reason is required")
    with _locked(ledger_path) as path:
        if not path.exists():
            raise BudgetError("No ledger to amend")
        ledger = json.loads(path.read_text())
        old_policy = dict(ledger["policy"])
        stored = Decimal(old_policy["total_budget_microusd"]) / _MICRO
        new_policy = _policy(stored)
        changed = {k for k in set(old_policy) | set(new_policy) if old_policy.get(k) != new_policy.get(k)}
        if not changed:
            return {**ledger, "summary": _summary(ledger)}
        if not changed <= AMENDABLE_POLICY_FIELDS:
            raise BudgetError(f"Only {sorted(AMENDABLE_POLICY_FIELDS)} may be amended; found {sorted(changed)}")
        if any(row["status"] != "stopped" for row in ledger["apps"].values()):
            raise BudgetError("Cannot amend the policy while a reservation is unresolved")
        for row in ledger["apps"].values():
            row.setdefault("containers", int(old_policy["max_containers"]))
            row.setdefault("gpu", DEFAULT_GPU)
        ledger["policy"] = new_policy
        ledger.setdefault("policy_amendments", []).append({
            "at_unix": time.time(), "reason": reason,
            "previous_policy": old_policy, "new_policy": new_policy})
        _write(path, ledger)
        verified = _read(path)          # fail closed if anything no longer verifies
        return {**verified, "summary": _summary(verified)}


def self_test():
    """Exercise money boundaries, unresolved apps, evidence gates, GPU rates and corruption."""
    from unittest.mock import patch
    with tempfile.TemporaryDirectory() as temp, patch("time.time", return_value=1000.), patch("time.monotonic", return_value=2000.):
        p = Path(temp) / "ledger.json"
        def rejected(fn):
            try: fn()
            except BudgetError: return
            raise AssertionError("Expected BudgetError")
        a = reserve_app(p, "one", total_budget_usd=20, work_seconds=600)
        assert a["reserved_microusd"] == _cost(660, MAX_CONTAINERS, DEFAULT_GPU)
        assert a["containers"] == MAX_CONTAINERS and a["gpu"] == DEFAULT_GPU
        rejected(lambda: reserve_app(p, "two", total_budget_usd=20, work_seconds=60))
        attach_app(p, "one", "ap-test-one")
        rejected(lambda: finish_app(p, "one", verified_stopped=False, remaining_tasks=0, app_state="stopped", evidence="test"))
        rejected(lambda: finish_app(p, "one", verified_stopped=True, remaining_tasks=1, app_state="stopped", evidence="test"))
        with patch("time.time", return_value=1100.), patch("time.monotonic", return_value=2100.):
            settled = finish_app(p, "one", verified_stopped=True, remaining_tasks=0, app_state="stopped", evidence="test")
        assert settled["charged_upper_microusd"] == _cost(100, MAX_CONTAINERS, DEFAULT_GPU)
        assert settled["summary"]["active_reserved_usd"] == 0
        rejected(lambda: reserve_app(p, "one", total_budget_usd=20, work_seconds=60))
        rejected(lambda: reserve_app(p, "two", total_budget_usd=50, work_seconds=60))
        rejected(lambda: reserve_app(p, "two", total_budget_usd=20, work_seconds=100000))
        rejected(lambda: reserve_app(p, "two", total_budget_usd=20, work_seconds=60, containers=0))
        rejected(lambda: reserve_app(p, "two", total_budget_usd=20, work_seconds=60, containers=MAX_CONTAINERS + 1))
        rejected(lambda: reserve_app(p, "two", total_budget_usd=20, work_seconds=60, gpu="H100"))
        # A T4 reservation is charged at the T4 rate, not the A100 one.
        c = reserve_app(p, "two", total_budget_usd=20, work_seconds=60, containers=2, gpu="T4")
        assert c["reserved_microusd"] == _cost(120, 2, "T4") and c["gpu"] == "T4"
        assert c["reserved_microusd"] < _cost(120, 2, "A100-40GB")
        attach_app(p, "two", "ap-test-two")
        with patch("time.time", return_value=1200.), patch("time.monotonic", return_value=2200.):
            b = finish_app(p, "two", verified_stopped=True, remaining_tasks=0, app_state="stopped", evidence="test")
        assert b["charged_upper_microusd"] == _cost(200, 2, "T4")
        assert b["summary"]["blocked_reason"], "an overrun beyond the reservation must block"
        rejected(lambda: reserve_app(p, "three", total_budget_usd=20, work_seconds=60))
        j = json.loads(p.read_text()); j["apps"]["two"]["gpu"] = "A100-40GB"; p.write_text(json.dumps(j))
        rejected(lambda: read_ledger(p))                 # re-pricing a settled row is tamper
        j["apps"]["two"]["gpu"] = "T4"; j["apps"]["one"]["reserved_microusd"] -= 1
        p.write_text(json.dumps(j))
        rejected(lambda: read_ledger(p))
    assert _rate("A100-40GB") == Decimal("0.00067122") and _rate("T4") == Decimal("0.00025192")
    assert _cost(1, 2) == 1343 and _cost(1, 8) == 5370 and _cost(1) == _cost(1, MAX_CONTAINERS)
    assert _cost(1, 8, "T4") == 2016
    # A full 8-worker A100 hour is well over the whole $20 cap: the guard matters.
    assert _cost(3600, 8, "A100-40GB") > _micro(Decimal("19"))
    print("PASS: per-GPU upper-envelope rates, reservation/ID/cap guards, termination "
          "evidence, refund after stop, overrun lock and tamper detection")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--status", type=Path)
    parser.add_argument("--amend", type=Path, help="adopt the current MAX_CONTAINERS policy in an existing ledger")
    parser.add_argument("--reason", default="")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    elif args.status:
        print(json.dumps(read_ledger(args.status), indent=2))
    elif args.amend:
        print(json.dumps(amend_policy(args.amend, args.reason)["summary"], indent=2))
    else:
        parser.error("Use --self-test, --status PATH or --amend PATH --reason TEXT; launches must call reserve_app explicitly")
