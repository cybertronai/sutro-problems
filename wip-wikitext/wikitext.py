"""Energy-efficient character-level language modelling on WikiText-103.

Defines:

* ``CharModel`` — streaming next-character API. By construction the model
  emits its distribution for position ``i`` *before* it has been told the
  ground-truth character at position ``i``, so within-stream future-peeking
  is impossible.
* ``evaluate(model, stream)`` — char-accuracy runner. Plain correctness
  metric; no energy accounting.
* ``EnergyMeter`` — reads ``nvmlDeviceGetTotalEnergyConsumption``
  (Volta+) at start/end of a measured block and reports
  ``E_run − idle_watts × duration`` in joules. Idle subtraction is the
  only correction; everything else is the raw NVML counter. On hosts
  without NVML the meter reports ``available=False`` and an energy of
  ``None`` — the eval still runs for development, but submissions must
  use a host where NVML is exposed (Lambda On-Demand A100, RunPod
  Secure, etc.).
* ``load_wikitext103(data_dir, split)`` — load the raw WikiText-103
  splits (``wiki.{train,valid,test}.raw``) as a single string.

Energy is measured around the **training** phase (per the v0 design
note: training-only). Eval is *not* energy-accounted.
"""
from __future__ import annotations

import os
import signal
import threading
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator


class BudgetExceededError(RuntimeError):
    """Raised inside ``EnergyMeter.measure()`` when running net energy
    crosses the configured ``e_max_joules`` budget. The submission is
    disqualified; partial duration / energy are still recorded on the
    yielded ``Measurement``.
    """


# ---------------------------------------------------------------------------
# CharModel API
# ---------------------------------------------------------------------------

class CharModel(ABC):
    """Streaming next-character model.

    The runner drives a single loop::

        model.reset()
        for true_char in stream:
            dist = model.predict()         # P(c | observed_so_far)
            argmax(dist) ?== true_char     # scored
            model.observe(true_char)       # commit, then advance

    The model never receives a character before emitting that
    character's distribution. Future-peeking is structurally impossible.
    """

    @abstractmethod
    def reset(self) -> None:
        """Clear streaming context (not trained parameters)."""

    @abstractmethod
    def predict(self) -> dict[str, float]:
        """Return P(next_char | chars_observed_so_far).

        Only entries with nonzero probability need be returned. The
        runner takes ``argmax`` over this dict; ties broken by dict
        insertion order.
        """

    @abstractmethod
    def observe(self, char: str) -> None:
        """Commit a single ground-truth character to the model's history."""


# ---------------------------------------------------------------------------
# Streaming evaluator
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    n_chars: int
    n_correct: int
    duration_s: float

    @property
    def accuracy(self) -> float:
        return self.n_correct / max(1, self.n_chars)

    def __str__(self) -> str:
        return (f"chars={self.n_chars:,}  "
                f"acc={self.accuracy:.4f}  "
                f"eval_duration={self.duration_s:.1f}s")


def evaluate(
    model: CharModel,
    stream: Iterable[str],
    *,
    progress_every: int = 0,
    total: int | None = None,
) -> EvalResult:
    """Score ``model`` on ``stream`` by greedy-argmax char-accuracy.

    A single ``model.reset()`` is issued at the start; the model then
    sees the entire stream as one long sequence and is responsible for
    its own context-window management.

    ``progress_every``: if > 0, print one line every N characters with
    throughput, running accuracy, and ETA (if ``total`` is set or the
    stream is sized).
    """
    if total is None and hasattr(stream, "__len__"):
        total = len(stream)  # type: ignore[arg-type]

    n_chars = 0
    n_correct = 0
    model.reset()
    t0 = time.monotonic()
    for true_char in stream:
        dist = model.predict()
        if dist:
            pred_char = max(dist, key=lambda c: dist[c])
            if pred_char == true_char:
                n_correct += 1
        n_chars += 1
        model.observe(true_char)

        if progress_every and n_chars % progress_every == 0:
            elapsed = time.monotonic() - t0
            chars_per_s = n_chars / max(1e-9, elapsed)
            acc = n_correct / n_chars
            if total:
                pct = 100.0 * n_chars / total
                remaining = max(0, total - n_chars) / max(1e-9, chars_per_s)
                eta = f"  eta={remaining:6.0f}s"
                head = f"{n_chars:>10,}/{total:,} ({pct:5.1f}%)"
            else:
                eta = ""
                head = f"{n_chars:>10,}"
            print(f"  eval {head}  acc={acc:.4f}  "
                  f"{chars_per_s:7.0f} char/s{eta}", flush=True)

    return EvalResult(
        n_chars=n_chars,
        n_correct=n_correct,
        duration_s=time.monotonic() - t0,
    )


# ---------------------------------------------------------------------------
# Energy measurement (NVML)
# ---------------------------------------------------------------------------

@dataclass
class Measurement:
    energy_joules: float | None = None
    duration_s: float = 0.0
    budget_exceeded: bool = False

    def __str__(self) -> str:
        e = (f"{self.energy_joules:,.1f} J"
             if self.energy_joules is not None else "energy: not measured")
        dq = "  [BUDGET EXCEEDED]" if self.budget_exceeded else ""
        return f"{e}   duration={self.duration_s:.1f}s{dq}"


class EnergyMeter:
    """``nvmlDeviceGetTotalEnergyConsumption``-based energy accountant.

    Usage::

        meter = EnergyMeter()
        with meter.measure() as m:
            train_my_model()
        print(m.energy_joules, "J")

    Idle baseline: ``idle_watts * duration_s`` is subtracted from the
    raw NVML delta. Default 50 W is conservative for an A100 80GB at
    rest; calibrate per host for production runs.

    On hosts without NVML (CPU-only laptops), ``available`` is ``False``
    and ``Measurement.energy_joules`` is ``None``. Submissions to the
    leaderboard must run on a host where ``available`` is ``True``.

    If ``e_max_joules`` is set, a watchdog thread polls NVML every
    ``poll_interval_s`` seconds during ``measure()``; when the running
    net energy crosses the budget, ``BudgetExceededError`` is raised
    into the main thread via SIGUSR1. The watchdog is a no-op on hosts
    without NVML and on non-main-thread callers (signal install fails);
    those callers hit the wall-clock hard floor instead.
    """

    def __init__(
        self,
        *,
        gpu_index: int = 0,
        idle_watts: float = 50.0,
        e_max_joules: float | None = None,
        poll_interval_s: float = 0.25,
    ):
        self.gpu_index = gpu_index
        self.idle_watts = idle_watts
        self.e_max_joules = e_max_joules
        self.poll_interval_s = poll_interval_s
        self.available = False
        self._handle = None
        self._pynvml = None
        try:
            import pynvml  # type: ignore[import-not-found]
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            # Probe the energy counter; if unsupported, fall back.
            pynvml.nvmlDeviceGetTotalEnergyConsumption(self._handle)
            self._pynvml = pynvml
            self.available = True
        except Exception:
            self.available = False

    @contextmanager
    def measure(self) -> Iterator[Measurement]:
        m = Measurement()
        e0: int | None = None
        if self.available and self._pynvml is not None:
            e0 = self._pynvml.nvmlDeviceGetTotalEnergyConsumption(self._handle)
        t0 = time.monotonic()

        watchdog_stop = threading.Event()
        watchdog_thread: threading.Thread | None = None
        prev_handler = None
        budget_armed = (
            self.e_max_joules is not None
            and self.available
            and self._pynvml is not None
            and e0 is not None
        )

        if budget_armed:
            try:
                prev_handler = signal.signal(
                    signal.SIGUSR1, _make_budget_signal_handler(m, self.e_max_joules)
                )
            except (ValueError, OSError):
                # Not on main thread, or signal not available — fall back
                # to os._exit() inside the watchdog.
                prev_handler = None

            main_pid = os.getpid()
            handler_installed = prev_handler is not None
            e_max = float(self.e_max_joules)  # type: ignore[arg-type]
            pynvml = self._pynvml
            handle = self._handle
            idle_w = self.idle_watts
            poll = self.poll_interval_s

            def _watchdog() -> None:
                while not watchdog_stop.wait(poll):
                    try:
                        e_now = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
                    except Exception:
                        return
                    duration = time.monotonic() - t0
                    e_run_j = (e_now - e0) / 1000.0
                    e_net_j = max(0.0, e_run_j - duration * idle_w)
                    if e_net_j > e_max:
                        m.budget_exceeded = True
                        m.energy_joules = e_net_j
                        m.duration_s = duration
                        if handler_installed:
                            os.kill(main_pid, signal.SIGUSR1)
                        else:
                            # Best-effort hard kill when we can't raise
                            # into the main thread cleanly.
                            os._exit(124)
                        return

            watchdog_thread = threading.Thread(target=_watchdog, daemon=True)
            watchdog_thread.start()

        try:
            yield m
        finally:
            watchdog_stop.set()
            if watchdog_thread is not None:
                watchdog_thread.join(timeout=2.0)
            if prev_handler is not None:
                try:
                    signal.signal(signal.SIGUSR1, prev_handler)
                except (ValueError, OSError):
                    pass
            # If the watchdog already filled these in, leave them — the
            # post-fire NVML read can race the kill and produce a smaller
            # delta. Otherwise compute as usual.
            if not m.budget_exceeded:
                m.duration_s = time.monotonic() - t0
                if self.available and self._pynvml is not None and e0 is not None:
                    e1 = self._pynvml.nvmlDeviceGetTotalEnergyConsumption(self._handle)
                    e_run_j = (e1 - e0) / 1000.0  # NVML returns millijoules
                    e_idle_j = m.duration_s * self.idle_watts
                    m.energy_joules = max(0.0, e_run_j - e_idle_j)


def _make_budget_signal_handler(m: Measurement, e_max: float | None):
    def _handler(signum, frame):  # noqa: ARG001
        raise BudgetExceededError(
            f"training energy budget exceeded "
            f"(e_max={e_max:,.0f} J, used≈{m.energy_joules or 0:,.0f} J)"
        )
    return _handler


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_wikitext103(data_dir: Path | str, split: str = "test") -> str:
    """Return one of the WikiText-103 raw splits as a single string.

    Expects ``data_dir`` to contain ``wiki.train.raw``, ``wiki.valid.raw``,
    ``wiki.test.raw``. The historical ``s3.amazonaws.com/research.metamind.io``
    URL no longer resolves; fetch via the HuggingFace mirror instead — see
    ``RUNBOOK.md`` step 2 for the snippet that materialises these files.
    """
    valid = {"train", "valid", "test"}
    if split not in valid:
        raise ValueError(f"split must be one of {sorted(valid)}; got {split!r}")
    p = Path(data_dir) / f"wiki.{split}.raw"
    if not p.exists():
        raise FileNotFoundError(
            f"WikiText-103 raw file not found: {p}\n"
            f"Fetch via HuggingFace (`datasets.load_dataset('Salesforce/wikitext', "
            f"'wikitext-103-raw-v1')`) and write wiki.{{train,valid,test}}.raw "
            f"into {data_dir} — see RUNBOOK.md step 2."
        )
    return p.read_text(encoding="utf-8")
