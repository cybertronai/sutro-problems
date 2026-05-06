"""Tests for wikitext.py and the n-gram baseline.

Runs with ``python3 -m pytest test_wikitext.py`` or ``python3 test_wikitext.py``
(falls back to a hand-rolled runner when pytest is not installed).
"""
from __future__ import annotations

from pathlib import Path
import time

from wikitext import BudgetExceededError, CharModel, EnergyMeter, evaluate, load_wikitext103
from baseline_ngram import NGramModel


# ---------------------------------------------------------------------------
# CharModel API contract
# ---------------------------------------------------------------------------

class _ConstantModel(CharModel):
    """Always predicts a single fixed char with probability 1.0."""

    def __init__(self, ch: str):
        self.ch = ch
        self.observed: list[str] = []

    def reset(self) -> None:
        self.observed = []

    def predict(self) -> dict[str, float]:
        return {self.ch: 1.0}

    def observe(self, char: str) -> None:
        self.observed.append(char)


def test_evaluate_counts_correct_argmax() -> None:
    """Constant predictor of 'a' on stream 'aaa' should get 3/3."""
    r = evaluate(_ConstantModel("a"), "aaa")
    assert r.n_chars == 3
    assert r.n_correct == 3
    assert r.accuracy == 1.0


def test_evaluate_counts_wrong_argmax() -> None:
    """Constant predictor of 'a' on stream 'bcd' should get 0/3."""
    r = evaluate(_ConstantModel("a"), "bcd")
    assert r.n_chars == 3
    assert r.n_correct == 0
    assert r.accuracy == 0.0


def test_evaluate_streaming_order() -> None:
    """observe() must be called *after* predict() at every step.

    A model that records the (#predicts seen, #observes seen) tuple
    every time predict is called should always have predicts == observes
    + 1, never seeing future characters.
    """
    seen_at_predict: list[tuple[int, int]] = []

    class _Recorder(CharModel):
        def __init__(self) -> None:
            self.n_pred = 0
            self.n_obs = 0

        def reset(self) -> None:
            self.n_pred = 0
            self.n_obs = 0

        def predict(self) -> dict[str, float]:
            seen_at_predict.append((self.n_pred, self.n_obs))
            self.n_pred += 1
            return {"x": 1.0}

        def observe(self, char: str) -> None:
            del char
            self.n_obs += 1

    evaluate(_Recorder(), "abcd")
    # At each predict, n_pred == n_obs (we haven't yet observed the
    # char we're about to predict). Future-peeking would manifest as
    # n_obs > n_pred at some point.
    for n_pred, n_obs in seen_at_predict:
        assert n_pred == n_obs


# ---------------------------------------------------------------------------
# Energy meter
# ---------------------------------------------------------------------------

def test_energy_meter_fallback_when_no_nvml() -> None:
    """On hosts without NVML, energy_joules must be None, not a fudged proxy."""
    meter = EnergyMeter()
    if meter.available:
        # CI may run on a GPU host; this test only validates the
        # fallback branch when NVML is absent.
        return
    with meter.measure() as m:
        sum(range(1000))
    assert m.energy_joules is None
    assert m.duration_s >= 0


class _FakeNvml:
    """Fake pynvml that synthesizes a monotonic mJ counter at a fixed power."""

    def __init__(self, watts: float):
        self._t0 = time.monotonic()
        self._watts = watts

    def nvmlDeviceGetTotalEnergyConsumption(self, handle):  # noqa: ARG002
        elapsed_s = time.monotonic() - self._t0
        return int(elapsed_s * self._watts * 1000.0)  # mJ


def _patch_meter_with_fake_nvml(meter: EnergyMeter, watts: float) -> None:
    meter._pynvml = _FakeNvml(watts)
    meter._handle = object()
    meter.available = True


def test_energy_budget_killswitch_fires() -> None:
    """Watchdog must raise BudgetExceededError once net energy > e_max."""
    # 200 W, idle 0 W, budget 5 J → fires ~25 ms in (plus one poll).
    meter = EnergyMeter(e_max_joules=5.0, poll_interval_s=0.02, idle_watts=0.0)
    _patch_meter_with_fake_nvml(meter, watts=200.0)

    raised = False
    m = None
    try:
        with meter.measure() as m:
            time.sleep(2.0)  # plenty of time for the watchdog to fire
    except BudgetExceededError:
        raised = True

    assert raised, "watchdog should have raised BudgetExceededError"
    assert m is not None
    assert m.budget_exceeded
    assert m.energy_joules is not None and m.energy_joules >= 5.0
    # Should fire well before the 2 s sleep would naturally end.
    assert m.duration_s < 1.0


def test_energy_budget_within_limit_does_not_fire() -> None:
    """A short, low-energy block under budget must complete cleanly."""
    meter = EnergyMeter(e_max_joules=1000.0, poll_interval_s=0.02, idle_watts=0.0)
    _patch_meter_with_fake_nvml(meter, watts=100.0)

    with meter.measure() as m:
        time.sleep(0.1)  # 100 W * 0.1 s = 10 J, well under 1000 J

    assert not m.budget_exceeded
    assert m.energy_joules is not None
    assert m.energy_joules < 1000.0


def test_energy_budget_no_op_without_nvml() -> None:
    """e_max_joules on a host without NVML is a no-op; nothing crashes."""
    meter = EnergyMeter(e_max_joules=1.0, poll_interval_s=0.02)
    if meter.available:
        return  # only meaningful when NVML is absent
    with meter.measure() as m:
        time.sleep(0.05)
    assert not m.budget_exceeded
    assert m.energy_joules is None


# ---------------------------------------------------------------------------
# n-gram baseline
# ---------------------------------------------------------------------------

def test_ngram_trains_and_predicts() -> None:
    m = NGramModel(n=3)
    m.train("abcabcabcabc")
    m.reset()
    # After observing 'a', argmax of P(. | 'a') in 'abcabc...' should be 'b'.
    m.observe("a")
    dist = m.predict()
    assert max(dist, key=lambda c: dist[c]) == "b"


def test_ngram_accuracy_on_repeating_pattern() -> None:
    """A 3-cycle pattern is perfectly predictable from a 3-gram after warmup."""
    m = NGramModel(n=3)
    train = "abc" * 200
    test = "abc" * 50
    m.train(train)
    r = evaluate(m, test)
    # The first few chars (before context fills) may miss; the rest hit.
    assert r.accuracy > 0.9


def test_ngram_backoff_to_unigram() -> None:
    """Predicting on an out-of-distribution starting char falls back."""
    m = NGramModel(n=3)
    m.train("aaaa")
    m.reset()
    # No context yet — should return the unigram distribution.
    dist = m.predict()
    assert "a" in dist
    assert dist["a"] == 1.0


def test_tiny_fixture_ngram_smoke() -> None:
    """The committed fixture exercises the local evaluator path."""
    data_dir = Path(__file__).parent / "fixtures" / "tiny"
    train = load_wikitext103(data_dir, "train")
    test = load_wikitext103(data_dir, "test")
    m = NGramModel(n=3)
    m.train(train)
    r = evaluate(m, test)
    assert r.n_chars == len(test)
    assert r.accuracy > 0.65


# ---------------------------------------------------------------------------
# Standalone runner (works without pytest)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys, traceback

    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception:
            failures += 1
            print(f"FAIL  {t.__name__}")
            traceback.print_exc()
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
