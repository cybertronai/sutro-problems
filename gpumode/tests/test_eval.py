"""Unit tests for the parts of the harness that decide whether a run is fair.

    python -m pytest gpumode/tests/test_eval.py -q

No GPU, no dataset download: every test here is arithmetic, parsing or
bookkeeping.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import eval as harness  # noqa: E402
import make_bands  # noqa: E402
from utils import combine, required_correct, stats, timing_plausible  # noqa: E402


# ------------------------------------------------------------------ accuracy rule

def test_required_correct_matches_the_published_bands():
    # 11 draws x 10,000 queries, the numbers printed in every task.yml
    assert required_correct(110000, 200) == 107800
    assert required_correct(110000, 300) == 106700
    assert required_correct(110000, 500) == 104500
    assert required_correct(110000, 800) == 101200
    assert required_correct(110000, 1200) == 96800


def test_required_correct_rounds_up_and_uses_integers():
    # 3 * 0.6667 = 2.0001 correct: two is not enough
    assert required_correct(3, 3333) == 3
    # exact multiples must not be inflated by float error
    assert required_correct(10000, 200) == 9800
    assert required_correct(110000, 0) == 110000
    assert required_correct(110000, 10000) == 0
    for total in (1, 7, 9999, 110000):
        for error_bp in (0, 1, 160, 200, 1500, 9999, 10000):
            exact = -(-(total * (10000 - error_bp)) // 10000)
            assert required_correct(total, error_bp) == exact
            assert (exact - 1) * 10000 < total * (10000 - error_bp) <= exact * 10000


def test_required_correct_rejects_nonsense_bands():
    with pytest.raises(ValueError):
        required_correct(100, -1)
    with pytest.raises(ValueError):
        required_correct(100, 10001)


def test_aggregate_rule_is_over_all_draws_not_per_draw():
    # one bad draw can be paid for by the others; that is the intent
    per_draw = [9600, 9450, 9500]
    assert sum(per_draw) >= required_correct(30000, 500)
    assert min(per_draw) < required_correct(10000, 500)


# ------------------------------------------------------------------ seeds

def test_combine_is_kernelbots_cantor_pairing():
    def reference(a, b):
        return int(a + (a + b) * (a + b + 1) // 2)

    for a in (0, 1, 101, 202, 65535):
        for b in (0, 3, 20260922, 2**40 + 7):
            assert combine(a, b) == reference(a, b)


def test_combine_hides_the_public_seed_behind_a_large_secret():
    secret = 2**40 + 12345
    combined = {combine(public, secret) for public in (101, 202, 303)}
    assert all(value > secret for value in combined)
    assert len(combined) == 3  # distinct public seeds stay distinct


# ------------------------------------------------------------------ draws

def make_pool(count=60000, classes=10, side=3):
    rng = np.random.default_rng(7)
    images = rng.random((count, 1, side, side), dtype=np.float32)
    labels = (np.arange(count) % classes).astype(np.int64)
    return images, labels


def universes_for(pool, seed=4242):
    return harness.split_universes(len(pool[1]), seed, harness.UNIVERSE_SALT)


def test_draw_is_disjoint_deterministic_and_pulled_from_the_pool():
    pool = make_pool()
    universes = universes_for(pool)
    visible, truth = harness.make_draw(pool, 12345, 50, 40, universes)
    again, truth_again = harness.make_draw(pool, 12345, 50, 40, universes)
    assert visible[0].shape == (50, 1, 3, 3)
    assert visible[1].shape == (50,)
    assert visible[2].shape == (40, 1, 3, 3)
    assert truth.shape == (40,)
    assert np.array_equal(visible[0], again[0]) and np.array_equal(truth, truth_again)
    # no test image is also a training image
    train_rows = {row.tobytes() for row in visible[0]}
    assert not any(row.tobytes() in train_rows for row in visible[2])


def test_test_images_are_never_shown_with_a_label_in_any_draw():
    """The hole that made a pool-memoization table pay: every draw used to
    re-split the same 60,000 rows, so a test image of draw 7 had probably
    already arrived, labelled, in the training half of draw 2."""
    pool = make_pool(count=2000)
    universes = universes_for(pool, seed=99)
    seen_with_a_label = set()
    queried = set()
    for step in range(12):
        visible, _ = harness.make_draw(pool, 99 + 13 * step, 300, 300, universes)
        seen_with_a_label.update(row.tobytes() for row in visible[0])
        queried.update(row.tobytes() for row in visible[2])
    assert seen_with_a_label and queried
    assert not (seen_with_a_label & queried)


def test_universes_split_the_pool_in_half_and_depend_on_the_secret():
    first = harness.split_universes(1000, 12345, harness.UNIVERSE_SALT)
    same = harness.split_universes(1000, 12345, harness.UNIVERSE_SALT)
    other = harness.split_universes(1000, 12346, harness.UNIVERSE_SALT)
    assert len(first[0]) == len(first[1]) == 500
    assert not set(first[0]) & set(first[1])
    assert np.array_equal(first[0], same[0])  # deterministic within a run
    assert not np.array_equal(first[0], other[0])  # a different secret, a different split


def test_label_permutation_is_secret_consistent_and_per_draw():
    images, labels = make_pool()
    pool = (images, labels)
    universes = universes_for(pool)
    visible, truth = harness.make_draw(pool, 999, 200, 200, universes)
    rows = np.random.default_rng([999, harness.DRAW_SALT])
    train_rows = rows.choice(universes[0], 200, replace=False)
    test_rows = rows.choice(universes[1], 200, replace=False)
    # the same permutation maps the true labels of both halves
    mapping = {}
    for true_label, shown in zip(labels[train_rows], visible[1]):
        mapping.setdefault(int(true_label), int(shown))
        assert mapping[int(true_label)] == int(shown)
    for true_label, shown in zip(labels[test_rows], truth):
        assert mapping[int(true_label)] == int(shown)
    assert sorted(mapping.values()) == list(range(10))  # a permutation, not a collapse
    # a different draw uses a different mapping, so memorized labels go stale
    other, _ = harness.make_draw(pool, 1000, 200, 200, universes)
    assert not np.array_equal(visible[1][:50], other[1][:50])


def test_draw_refuses_to_overflow_its_half_of_the_pool():
    pool = make_pool(count=60000)
    universes = universes_for(pool)
    with pytest.raises(ValueError):
        harness.make_draw(pool, 1, 40000, 30000, universes)


# ------------------------------------------------------------------ the linear release

RELEASE_DIMS = 60


def release_pool(count=24000, classes=10, side=9):
    """A synthetic 81-pixel pool whose covariance has full rank."""
    return make_pool(count=count, classes=classes, side=side)


def rows_of(seed, universes, n_train, n_test):
    """The pool rows ``make_draw`` will pick for this seed, recomputed here."""
    rng = np.random.default_rng([int(seed), harness.DRAW_SALT])
    return (
        rng.choice(universes[0], n_train, replace=False),
        rng.choice(universes[1], n_test, replace=False),
    )


def test_release_dims_zero_is_the_1_1_1_pixel_release():
    """The default must be bit-identical to the pre-1.2.0 draw: the pool rows
    themselves, same shape, same dtype, no arithmetic in between."""
    pool = make_pool()
    universes = universes_for(pool)
    visible, truth = harness.make_draw(pool, 12345, 50, 40, universes)
    explicit, explicit_truth = harness.make_draw(pool, 12345, 50, 40, universes, release_dims=0)
    train_rows, test_rows = rows_of(12345, universes, 50, 40)
    assert visible[0].shape == (50, 1, 3, 3) and visible[0].dtype == np.float32
    assert visible[2].shape == (40, 1, 3, 3) and visible[2].dtype == np.float32
    assert np.array_equal(visible[0], pool[0][train_rows])  # the pool rows, untouched
    assert np.array_equal(visible[2], pool[0][test_rows])
    assert np.array_equal(visible[0], explicit[0]) and np.array_equal(visible[2], explicit[2])
    assert np.array_equal(visible[1], explicit[1]) and np.array_equal(truth, explicit_truth)
    assert harness.DEFAULTS["release_dims"] == 0  # an old case line means the old behaviour


def test_released_arrays_are_flat_float32_of_the_requested_width():
    pool = release_pool()
    universes = universes_for(pool)
    visible, truth = harness.make_draw(pool, 7, 4000, 1500, universes, release_dims=RELEASE_DIMS)
    assert visible[0].shape == (4000, RELEASE_DIMS) and visible[0].dtype == np.float32
    assert visible[2].shape == (1500, RELEASE_DIMS) and visible[2].dtype == np.float32
    assert visible[1].shape == (4000,) and truth.shape == (1500,)
    assert len(visible) == 3  # no fourth element carrying the map


def test_released_training_rows_are_white():
    """z = Q W (x - mu) with W fitted on these very rows: zero mean, identity
    covariance. If either drifts, the whitening is not exact and the covariance
    channel the rotation study measured is back."""
    pool = release_pool(count=24000)
    universes = universes_for(pool)
    visible, _ = harness.make_draw(pool, 99, 10000, 1000, universes, release_dims=RELEASE_DIMS)
    z = visible[0].astype(np.float64)
    covariance = np.cov(z, rowvar=False)
    assert np.abs(z.mean(0)).max() < 0.05
    assert np.abs(covariance - np.eye(RELEASE_DIMS)).max() < 0.05


def test_both_halves_go_through_the_same_map():
    """Refit the map from the training rows and reproduce the released test rows
    exactly. Fitting the halves separately would fail this."""
    pool = release_pool()
    universes = universes_for(pool)
    seed = 4321
    visible, _ = harness.make_draw(pool, seed, 4000, 1000, universes, release_dims=RELEASE_DIMS)
    train_rows, test_rows = rows_of(seed, universes, 4000, 1000)
    mu, whitener, rotation = harness.release_map(pool[0][train_rows], seed, RELEASE_DIMS)
    transform = rotation @ whitener
    assert np.allclose(
        harness.apply_release(pool[0][train_rows], mu, transform), visible[0], atol=1e-3
    )
    assert np.allclose(
        harness.apply_release(pool[0][test_rows], mu, transform), visible[2], atol=1e-3
    )
    # and the test half is not white on its own account: it was not fitted
    test_covariance = np.cov(visible[2].astype(np.float64), rowvar=False)
    assert np.abs(test_covariance - np.eye(RELEASE_DIMS)).max() > 1e-6


def test_the_rotation_is_orthogonal_and_changes_with_the_seed():
    first = harness.haar_rotation(RELEASE_DIMS, 11)
    again = harness.haar_rotation(RELEASE_DIMS, 11)
    other = harness.haar_rotation(RELEASE_DIMS, 12)
    identity = np.eye(RELEASE_DIMS)
    assert np.abs(first @ first.T - identity).max() < 1e-10
    assert np.array_equal(first, again)  # deterministic within a draw
    assert np.abs(first - other).max() > 0.01  # a different draw, a different basis
    assert np.abs(first - identity).max() > 0.5  # it really does rotate


def test_the_same_rows_look_different_under_a_different_draw_seed():
    """A submission cannot recognize a row across draws, and no fixed inverse
    exists, because Q is redrawn from the secret seed on every draw."""
    pool = release_pool()
    universes = universes_for(pool)
    rows = rows_of(1, universes, 4000, 200)[0]
    first = harness.apply_release(
        pool[0][rows], *map_parts(harness.release_map(pool[0][rows], 1, RELEASE_DIMS))
    )
    second = harness.apply_release(
        pool[0][rows], *map_parts(harness.release_map(pool[0][rows], 2, RELEASE_DIMS))
    )
    assert np.abs(first - second).max() > 0.5


def map_parts(fitted):
    mu, whitener, rotation = fitted
    return mu, rotation @ whitener


def test_the_release_round_trips_to_the_top_k_pixel_projection():
    """The map is invertible *given the map*: undoing it returns the projection
    of the pixels onto the top-k principal subspace. That is what a scorer could
    do and a submission cannot."""
    pool = release_pool()
    universes = universes_for(pool)
    seed = 202
    rows = rows_of(seed, universes, 4000, 500)[0]
    pixels = pool[0][rows].reshape(len(rows), -1).astype(np.float64)
    mu, whitener, rotation = harness.release_map(pool[0][rows], seed, RELEASE_DIMS)
    z = harness.apply_release(pool[0][rows], mu, rotation @ whitener).astype(np.float64)
    recovered = mu + (rotation.T @ z.T).T @ np.linalg.pinv(whitener).T
    # the same projection, computed directly from the whitener's own row space
    basis = np.linalg.svd(whitener, full_matrices=False)[2]
    projected = mu + (pixels - mu) @ basis.T @ basis
    assert np.abs(recovered - projected).max() < 1e-3


def test_a_rank_deficient_draw_is_rejected_rather_than_divided_by_zero():
    images, labels = release_pool(count=4000)
    images = images.copy()
    images[:, :, :, 4:] = 0.0  # only 36 of the 81 pixels still vary
    with pytest.raises(ValueError) as error:
        harness.release_map(images[:2000], 5, RELEASE_DIMS)
    assert "rank deficient" in str(error.value)


def test_release_dims_outside_the_pixel_count_is_rejected():
    pool = release_pool(count=4000)
    with pytest.raises(ValueError):
        harness.release_map(pool[0][:2000], 5, 82)
    with pytest.raises(ValueError):
        harness.release_map(pool[0][:2000], 5, 0)
    with pytest.raises(ValueError):  # fewer rows than directions
        harness.release_map(pool[0][:40], 5, RELEASE_DIMS)


def test_the_label_permutation_is_untouched_by_the_release():
    pool = release_pool()
    universes = universes_for(pool)
    pixels, pixel_truth = harness.make_draw(pool, 77, 4000, 500, universes)
    released, released_truth = harness.make_draw(
        pool, 77, 4000, 500, universes, release_dims=RELEASE_DIMS
    )
    assert np.array_equal(pixels[1], released[1])
    assert np.array_equal(pixel_truth, released_truth)


# ------------------------------------------------------------------ run_case with a stub child

class StubChild:
    """Stands in for the submission's process: records what it was handed."""

    def __init__(self):
        self.staged = []
        self.commands = []

    def call(self, command, *args, timeout_s=None):
        self.commands.append(command)
        if command == "stage":
            self.staged.append(tuple(array.shape for array in args[0]))
            return 0.4, 0.5
        if command == "probe":
            return (np.zeros(0, dtype=np.int64), 0.0, 0.1, 0.4), 0.2
        if command == "load":
            return True, 0.1
        if command == "untimed":
            self.staged.append(tuple(array.shape for array in args[0]))
            return np.zeros(args[0][2].shape[0], dtype=np.int64), 1.0
        if command == "timed":
            count = self.staged[-1][2][0]
            predictions = np.zeros(count, dtype=np.int64)
            return (predictions, 10.0, 10.2, 0.4), 11.0
        raise AssertionError(command)


def run_case_with_stub(release_dims, n=300):
    pool = release_pool(count=4000)
    pools = {("mnist", 9): pool, ("fashion", 9): release_pool(count=4000, side=9)}
    case = dict(
        harness.DEFAULTS,
        size=9,
        release_dims=release_dims,
        train=n,
        test=n,
        error_bp=10000,
        draws=3,
        holdout_draws=1,
        holdout_min_bp=10000,
    )
    child = StubChild()
    report = harness.run_case(child, pools, case, timed_draws=3, holdout=True, floor_bp=10000)
    return child, report


def test_run_case_hands_the_child_the_released_shapes():
    child, _ = run_case_with_stub(RELEASE_DIMS)
    assert child.staged  # warm-up, calibration, every timed and hold-out draw
    for train_shape, label_shape, test_shape in child.staged:
        assert train_shape == (300, RELEASE_DIMS)
        assert test_shape == (300, RELEASE_DIMS)
        assert label_shape == (300,)


def test_run_case_still_hands_over_pixels_when_release_dims_is_zero():
    child, _ = run_case_with_stub(0)
    for train_shape, _, test_shape in child.staged:
        assert train_shape == (300, 1, 9, 9) and test_shape == (300, 1, 9, 9)


def test_no_report_key_or_value_mentions_the_release_map():
    _, report = run_case_with_stub(RELEASE_DIMS)
    text = json.dumps({key: str(value) for key, value in report.items()}).lower()
    for secret in ("mu", "whiten", "rotation", "haar", "eigen", "release_map", "map"):
        assert secret not in text.split() and f'"{secret}"' not in text
    assert not any(
        word in key.lower()
        for key in report
        for word in ("mu", "whiten", "rot", "haar", "eigen", "map", "release")
    )


def test_every_band_case_asks_for_the_linear_release():
    config = json.loads((HERE.parent / "bands.json").read_text())
    assert config["defaults"]["release_dims"] == 60
    for band in config["bands"]:
        task = (HERE.parent / band["name"] / "task.yml").read_text()
        cases = [json.loads(line[4:]) for line in task.splitlines()
                 if line.startswith("  - {") and "error_bp" in line]
        assert cases and all(fields["release_dims"] == 60 for fields in cases)


# ------------------------------------------------------------------ submission source

def write_submission(tmp_path, body):
    path = tmp_path / "submission.py"
    path.write_text(body)
    return path


def test_source_cap_rejects_an_embedded_dataset(tmp_path):
    case = dict(harness.DEFAULTS)
    path = write_submission(tmp_path, "TABLE = '" + "a" * 30000 + "'\n")
    with pytest.raises(harness.Failure) as error:
        harness.check_submission_source(case, path)
    assert "over the" in str(error.value)


def test_source_cap_rejects_one_oversized_literal(tmp_path):
    case = dict(harness.DEFAULTS, max_source_bytes=1_000_000)
    path = write_submission(tmp_path, "TABLE = '" + "a" * 30000 + "'\n")
    with pytest.raises(harness.Failure) as error:
        harness.check_submission_source(case, path)
    assert "literal" in str(error.value)


def test_source_cap_accepts_every_shipped_submission():
    case = dict(harness.DEFAULTS)
    for path in sorted((HERE.parent / "submissions").glob("*.py")):
        harness.check_submission_source(case, path)
    harness.check_submission_source(case, HERE.parent / "submission.py")


# ------------------------------------------------------------------ timing gate

def test_timing_gate_accepts_an_honest_call():
    assert timing_plausible(3.30, 3.45, 9.10) is None
    assert timing_plausible(0.21, 0.55, 4.00, 3.60) is None  # sub-millisecond call
    assert timing_plausible(260.0, 261.2, 275.0, 1.5) is None


def test_timing_gate_catches_a_patched_timer():
    assert "less than half" in timing_plausible(0.0, 260.0, 280.0)
    assert timing_plausible(1.0, 260.0, 280.0) is not None


def test_timing_gate_bounds_the_device_clock_from_below_with_the_parents():
    # Both of the child's clocks scaled by the same constant: every ratio test
    # between them still passes, and only the parent's clock notices.
    reason = timing_plausible(0.008, 0.010, 70.0, 0.7)
    assert reason is not None and "the parent measured" in reason
    # the same call reported honestly is fine
    assert timing_plausible(68.8, 69.0, 70.0, 0.7) is None
    # and the overhead really is subtracted: a short call behind a slow pipe
    assert timing_plausible(3.0, 3.2, 40.0, 36.0) is None


def test_timing_gate_catches_work_outside_the_timed_window():
    # half the work moved to an unsynchronized side stream
    assert timing_plausible(40.0, 200.0, 220.0) is not None


def test_timing_gate_catches_a_device_time_longer_than_the_wall_clock():
    assert "exceeds the child" in timing_plausible(12.0, 5.0, 30.0)


def test_timing_gate_trusts_the_parent_clock_over_the_child():
    # a child that under-reports its own wall clock is still bounded by the parent
    assert timing_plausible(100.0, 100.0, 50.0) is not None
    assert timing_plausible(float("nan"), 10.0, 20.0) is not None


# ------------------------------------------------------------------ cases

def write_cases(tmp_path, text):
    path = tmp_path / "cases.txt"
    path.write_text(text)
    return path


def test_read_cases_fills_defaults_and_combines_the_seed(tmp_path):
    path = write_cases(tmp_path, "size: 9; train: 10000; test: 10000; error_bp: 500; seed: 202\n")
    case = harness.read_cases(path, 20260922)[0]
    assert case["seed"] == combine(202, 20260922)
    assert case["draws"] == harness.DEFAULTS["draws"]
    assert case["max_call_ms"] == harness.DEFAULTS["max_call_ms"]
    assert case["spec"].startswith("size: 9")


def test_read_cases_rejects_unknown_fields_and_junk(tmp_path):
    with pytest.raises(ValueError):
        harness.read_cases(write_cases(tmp_path, "size: 9; sneaky: 1\n"), None)
    with pytest.raises(ValueError):
        harness.read_cases(write_cases(tmp_path, "size: nine\n"), None)
    with pytest.raises(ValueError):
        harness.read_cases(write_cases(tmp_path, "\n\n"), None)


def test_read_cases_rejects_a_release_dims_no_draw_can_serve(tmp_path):
    """An organizer's typo has to fail as a bad case file (exit 113), not as the
    submission failing validation, which is what a raise inside ``run_case``
    would report."""
    with pytest.raises(ValueError, match="release_dims"):
        harness.read_cases(write_cases(tmp_path, "size: 9; release_dims: 90\n"), None)
    with pytest.raises(ValueError, match="release_dims"):
        harness.read_cases(write_cases(tmp_path, "size: 9; release_dims: -1\n"), None)
    with pytest.raises(ValueError, match="whitening fit"):
        harness.read_cases(write_cases(tmp_path, "size: 9; release_dims: 60; train: 50\n"), None)
    # the two legal ends stay legal
    assert harness.read_cases(write_cases(tmp_path, "size: 9; release_dims: 0\n"), None)
    assert harness.read_cases(write_cases(tmp_path, "size: 9; release_dims: 81\n"), None)


def test_eval_exits_113_on_a_bad_release_dims(tmp_path):
    """End to end, because the exit code is the contract with KernelBot: 113 is
    "the case file is wrong", 112 is "the submission failed"."""
    cases = write_cases(tmp_path, "size: 9; release_dims: 90; train: 200; test: 200\n")
    environment = {**os.environ, "POPCORN_FD": "1", "POPCORN_SEED": "7"}
    finished = subprocess.run(
        [sys.executable, "eval.py", "test", str(cases)],
        cwd=HERE.parent,
        env=environment,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert finished.returncode == harness.EXIT_BAD_CASES
    assert "release_dims" in finished.stderr


# ------------------------------------------------------------------ statistics

def test_stats_reports_nanoseconds_like_kernelbot():
    values = [3.0e6, 3.2e6, 2.8e6, 3.1e6]
    result = stats(values)
    assert result["runs"] == 4
    assert result["best"] == 2.8e6 and result["worst"] == 3.2e6
    assert abs(result["mean"] - 3.025e6) < 1
    assert abs(result["median"] - 3.05e6) < 1
    assert result["err"] == pytest.approx(result["std"] / 2)


# ------------------------------------------------------------------ generated files

def test_generated_problem_folders_are_up_to_date():
    config = json.loads((HERE.parent / "bands.json").read_text())
    for relative, contents in make_bands.render(config).items():
        assert (HERE.parent / relative).read_text() == contents, f"{relative} is stale"


def test_readme_shows_the_generated_band_table():
    table = (HERE.parent / "bands.md").read_text().strip()
    assert table in (HERE.parent / "README.md").read_text()


def test_every_band_case_parses_and_keeps_its_threshold():
    config = json.loads((HERE.parent / "bands.json").read_text())
    for band in config["bands"]:
        task = (HERE.parent / band["name"] / "task.yml").read_text()
        cases = [json.loads(line[4:]) for line in task.splitlines()
                 if line.startswith("  - {") and "error_bp" in line]
        assert len(cases) == 2  # one test case, one ranked case
        for fields in cases:
            assert fields["error_bp"] == band["error_bp"]
            assert set(fields) <= set(harness.DEFAULTS)


# ------------------------------------------------------------------ process isolation

def run_script(tmp_path, body, environment=None):
    """Run a small program in its own interpreter and return its stdout."""
    import subprocess

    script = tmp_path / "probe.py"
    script.write_text(f"import sys\nsys.path.insert(0, {str(HERE.parent)!r})\n" + body)
    env = dict(os.environ)
    env.pop("POPCORN_SEED", None)
    env.update(environment or {})
    result = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, env=env, timeout=120
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def test_network_guard_survives_raw_sockets_and_a_reloaded_module(tmp_path):
    # The monkeypatch alone missed both of these: socket.socket subclasses the
    # C type _socket.socket, and reloading socket rebuilds clean functions.
    output = run_script(
        tmp_path,
        """
from utils import install_network_guard

install_network_guard()
results = []

import _socket
try:
    raw = _socket.socket(); raw.settimeout(1); raw.connect(("127.0.0.1", 9))
    results.append("raw:REACHED")
except BaseException as error:
    results.append("raw:" + type(error).__name__)

import importlib, socket
importlib.reload(socket)
try:
    socket.create_connection(("127.0.0.1", 9), timeout=1)
    results.append("reload:REACHED")
except BaseException as error:
    results.append("reload:" + type(error).__name__)

try:
    open("/tmp/train-images-idx3-ubyte.gz", "rb")
    results.append("dataset:REACHED")
except BaseException as error:
    results.append("dataset:" + type(error).__name__)

print(";".join(results))
""",
    )
    assert output == "raw:NetworkDisabled;reload:NetworkDisabled;dataset:DatasetFileDenied"


def test_the_secret_seed_is_removed_from_the_process_environment(tmp_path):
    # os.environ.pop does not rewrite /proc/<pid>/environ, so the evaluator
    # re-execs itself with the secret handed over on a pipe instead.
    output = run_script(
        tmp_path,
        """
import os
import eval as harness

payload = harness.scrub_secret_environment()
print("secret=%s in_environ=%s" % (payload.get("secret"), "POPCORN_SEED" in os.environ))
""",
        {"POPCORN_SEED": "20260922"},
    )
    assert output == "secret=20260922 in_environ=False"


# ------------------------------------------------------------------ the secret seed of a ranked run

def run_main_and_capture(tmp_path, case_line, secret=None):
    """Run eval.main() far enough to see which seed it used, and return its keys.

    ``scrub_secret_environment`` is stubbed because its real implementation
    re-execs the interpreter, which would replace this probe with the shipped
    eval.py; what is under test is what main() does with what it hands back.
    """
    (tmp_path / "submission.py").write_text("def custom_kernel(data):\n    return data[2]\n")
    (tmp_path / "cases.txt").write_text(case_line + "\n")
    body = """
import json
import os
import eval as harness

seen = {}


def stop(cases, cache, consume=False):
    seen["seed"] = cases[0]["seed"]
    raise harness.Failure("stop here")


harness.load_pools = stop
harness.scrub_secret_environment = lambda: json.loads(os.environ.get("PROBE_SECRETS", "{}"))
read_fd, write_fd = os.pipe()
os.environ["POPCORN_FD"] = str(write_fd)
sys.argv = ["eval.py", "test", "cases.txt"]
code = harness.main()
try:
    os.close(write_fd)  # main() closes it through PopcornOutput
except OSError:
    pass
with os.fdopen(read_fd) as handle:
    lines = handle.read().splitlines()
print(json.dumps({"code": code, "lines": lines, "seed": seen.get("seed")}))
"""
    script = tmp_path / "probe.py"
    script.write_text(f"import sys\nsys.path.insert(0, {str(HERE.parent)!r})\n" + body)
    env = dict(os.environ)
    env.pop("POPCORN_SEED", None)
    env["PROBE_SECRETS"] = json.dumps({"secret": secret} if secret else {})
    import subprocess

    result = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, env=env,
        cwd=tmp_path, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


CASE_LINE = "size: 9; train: 10; test: 10; draws: 1; seed: 202"


def test_a_ranked_run_without_popcorn_seed_still_draws_a_secret_seed(tmp_path):
    # KernelBot's participant-visible run -- the one whose time is published --
    # is submitted with seed=None, so POPCORN_SEED is not in the environment.
    # Falling back to the public case seed would make every draw, every label
    # permutation and every hold-out position reproducible offline.
    first = run_main_and_capture(tmp_path, CASE_LINE)
    second = run_main_and_capture(tmp_path, CASE_LINE)
    assert "system.seed_source: random" in first["lines"]
    assert first["seed"] != 202 and second["seed"] != 202
    assert first["seed"] != second["seed"]
    assert first["code"] == harness.EXIT_VALIDATE_FAIL  # load_pools was stubbed out


def test_a_supplied_popcorn_seed_is_used_and_reported(tmp_path):
    result = run_main_and_capture(tmp_path, CASE_LINE, secret="20260922")
    assert "system.seed_source: popcorn" in result["lines"]
    assert result["seed"] == combine(202, 20260922)


# ------------------------------------------------------------------ module-level inertness

def test_module_level_code_is_rejected(tmp_path):
    # KernelBot compiles a python submission by running it, before eval.py and
    # outside every guard this harness installs.
    case = dict(harness.DEFAULTS)
    for body in (
        "import os\nos.system('curl http://example.com')\n",
        "import mnist_data\nTABLE = mnist_data.load_pool\n",
        "LABELS = open('train-labels-idx1-ubyte').read()\n",
    ):
        path = write_submission(tmp_path, body)
        with pytest.raises(harness.Failure) as error:
            harness.check_submission_source(case, path)
        assert "module level" in str(error.value)


def test_module_level_imports_definitions_and_constants_are_allowed(tmp_path):
    case = dict(harness.DEFAULTS)
    path = write_submission(
        tmp_path,
        '"""doc."""\n'
        "import torch\n"
        "from task import input_t\n"
        "C, D = 10, 81\n"
        "MASK = (1 << 40) - 1\n"
        "SHAPES = {'x': [1, 2, 3]}\n"
        "torch.backends.cuda.matmul.allow_tf32 = False\n"
        "torch.set_float32_matmul_precision('highest')\n"
        "class Net:\n    pass\n"
        "def custom_kernel(data):\n    return data[2]\n",
    )
    harness.check_submission_source(case, path)


# ------------------------------------------------------------------ mode time budget

def test_every_command_deadline_is_clamped_to_the_mode_budget():
    # A fixed per-command deadline can run the mode timeout out, and KernelBot
    # then records a bare TIMEOUT with no check line.
    child = harness.Child.__new__(harness.Child)
    child.deadline = time.perf_counter() + 10.0
    assert child.budget(None, "load") == pytest.approx(10.0, abs=0.5)
    assert child.budget(150.0, "untimed") == pytest.approx(10.0, abs=0.5)
    assert child.budget(2.0, "timed") == pytest.approx(2.0, abs=0.5)
    child.deadline = time.perf_counter() - 1.0
    with pytest.raises(harness.Failure):
        child.budget(150.0, "untimed")


def test_mode_deadlines_come_from_the_case_fields():
    case = dict(harness.DEFAULTS, test_timeout=300, benchmark_timeout=600, ranked_timeout=1200)
    now = time.perf_counter()
    assert harness.mode_deadline(case, "test") - now == pytest.approx(270, abs=1)
    assert harness.mode_deadline(case, "benchmark") - now == pytest.approx(570, abs=1)
    assert harness.mode_deadline(case, "leaderboard") - now == pytest.approx(1170, abs=1)


def test_task_yml_timeouts_and_case_timeouts_agree():
    config = json.loads((HERE.parent / "bands.json").read_text())
    for band in config["bands"]:
        task = (HERE.parent / band["name"] / "task.yml").read_text()
        top = {
            line.split(":")[0]: int(line.split(":")[1])
            for line in task.splitlines()
            if line.startswith(("test_timeout", "benchmark_timeout", "ranked_timeout"))
        }
        cases = [json.loads(line[4:]) for line in task.splitlines()
                 if line.startswith("  - {") and "error_bp" in line]
        for fields in cases:
            for key, value in top.items():
                assert fields[key] == value


# ------------------------------------------------------------------ per-band templates

def test_each_band_ships_a_template_naming_its_own_board():
    config = json.loads((HERE.parent / "bands.json").read_text())
    shared = (HERE.parent / "submission.py").read_text().splitlines()
    for band in config["bands"]:
        template = (HERE.parent / band["name"] / "submission.py").read_text().splitlines()
        assert template[0] == f"#!POPCORN leaderboard {band['name']}"
        assert template[1:] == shared[1:]
        assert f'Python: "submission.py"' in (HERE.parent / band["name"] / "task.yml").read_text()


def test_make_bands_check_notices_a_stale_readme(tmp_path):
    config = json.loads((HERE.parent / "bands.json").read_text())
    rendered = make_bands.render(config)
    assert "README.md" in rendered
    assert make_bands.README_END in rendered["README.md"]
