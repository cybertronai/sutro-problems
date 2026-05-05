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

import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator


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

    def __str__(self) -> str:
        e = (f"{self.energy_joules:,.1f} J"
             if self.energy_joules is not None else "energy: not measured")
        return f"{e}   duration={self.duration_s:.1f}s"


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
    """

    def __init__(self, *, gpu_index: int = 0, idle_watts: float = 50.0):
        self.gpu_index = gpu_index
        self.idle_watts = idle_watts
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
        try:
            yield m
        finally:
            m.duration_s = time.monotonic() - t0
            if self.available and self._pynvml is not None and e0 is not None:
                e1 = self._pynvml.nvmlDeviceGetTotalEnergyConsumption(self._handle)
                e_run_j = (e1 - e0) / 1000.0  # NVML returns millijoules
                e_idle_j = m.duration_s * self.idle_watts
                m.energy_joules = max(0.0, e_run_j - e_idle_j)


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
