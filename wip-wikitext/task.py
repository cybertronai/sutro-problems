"""Task constants — the wikitext leaderboard rules.

These values define the **task** (the leaderboard contract), not any
single submission. Submitters MUST NOT vary them. ``submit.py`` reads
from here and forwards the values to ``run_eval.py`` inside the
container; any future re-evaluator does the same.

The flags on ``run_eval.py`` (``--max-test-chars``, ``--e-max-joules``)
are the *mechanism* that enforces these values; this module is the
*policy* that picks them.
"""
from __future__ import annotations

# Hardware platform. Energy numbers are only comparable across submissions
# when measured on the same SKU; the README pins this.
INSTANCE_TYPE: str = "gpu_1x_a100_sxm4"

# Test-stream length scored by the runner. 60K is the v0 standard:
# ~2 min eval on A100, ±0.4–1.3pp 95% CI on accuracy. See NOTES.md.
TEST_CHARS: int = 60_000

# Fixed-budget framing: hard cap on training energy. A run that crosses
# this is killed by the EnergyMeter watchdog and reported DISQUALIFIED.
# 100 kJ ≈ 5 min × 329 W avg net on the pinned A100 SXM4 (NOTES.md:181).
E_MAX_JOULES: float | None = 100_000.0

# Fixed-floor framing: minimum char-accuracy a submission must reach to
# be eligible for the minimize-energy ranking. ``None`` until calibrated.
ACC_MIN: float | None = None
