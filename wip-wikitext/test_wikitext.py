"""Tests for wikitext.py and the n-gram baseline.

Runs with ``python3 -m pytest test_wikitext.py`` or ``python3 test_wikitext.py``
(falls back to a hand-rolled runner when pytest is not installed).
"""
from __future__ import annotations

from wikitext import CharModel, EnergyMeter, evaluate
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
