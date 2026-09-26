"""Tests for mnist.py that run on a CPU without a network: the release, the draws, the
source rules, the judge, whole runs of the worker protocol on synthetic pools, and the
energy column against a simulated board.

    python -m pytest -q test_mnist.py
"""

import textwrap
import time

import numpy as np
import pytest

import mnist


def synthetic_pool(seed):
    """60,000 9x9 'images': ten class means plus noise, in [0, 1]. Nearest class mean gets ~99%."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 10, mnist.POOL_SIZE).astype(np.uint8)
    means = rng.uniform(0.2, 0.8, (10, 81))
    pixels = np.clip(means[labels] + rng.normal(0, 0.08, (mnist.POOL_SIZE, 81)), 0, 1)
    return pixels.reshape(-1, 9, 9).astype(np.float32), labels


@pytest.fixture(scope="module")
def pools():
    return {name: synthetic_pool(index) for index, name in enumerate(mnist.DATASETS)}


def write(tmp_path, name, body):
    path = tmp_path / name
    path.write_text(textwrap.dedent(body))
    return path


NCM = """
    import torch

    def ncm(train_x, train_y, test_x):
        sums = torch.zeros(10, train_x.shape[1], device=train_x.device).index_add_(0, train_y, train_x)
        means = sums / torch.bincount(train_y, minlength=10).clamp_min(1).unsqueeze(1)
        return ((means * means).sum(1) - 2 * test_x @ means.T).argmin(1)
"""


# ---- the release and the draws -------------------------------------------------------------


def test_release_whitens_training_rows_and_rotates(pools):
    rng = np.random.default_rng(1)
    pool = mnist.Pool("mnist", *pools["mnist"], rng)
    d = pool.draw(rng)
    assert d["train_x"].shape == (mnist.TRAIN, 60) and d["test_x"].shape == (mnist.TEST, 60)
    assert np.abs(d["train_x"].mean(0)).max() < 1e-4
    assert np.abs(np.cov(d["train_x"], rowvar=False) - np.eye(60)).max() < 1e-3
    assert set(d) == {"dataset", "train_x", "train_y", "test_x", "test_y"}  # the map is not in the draw
    other = pool.draw(rng)
    assert not np.allclose(np.abs(np.corrcoef(d["train_x"].T[:5], other["train_x"].T[:5])[:5, 5:]).max(), 1.0)


def test_haar_rotation_is_orthogonal():
    q = mnist.haar_rotation(60, np.random.default_rng(0))
    assert np.allclose(q @ q.T, np.eye(60), atol=1e-10)


def test_training_and_test_images_come_from_opposite_halves(pools):
    rng = np.random.default_rng(2)
    pool = mnist.Pool("mnist", *pools["mnist"], rng)
    assert not set(pool.train_half) & set(pool.test_half)
    assert len(pool.train_half) + len(pool.test_half) == mnist.POOL_SIZE


def test_labels_are_permuted_per_draw():
    # class c makes up a share of the pool proportional to c + 1, so label counts reveal the permutation
    labels = np.repeat(np.arange(10, dtype=np.uint8), [1000 * (c + 1) for c in range(10)])
    pixels = np.random.default_rng(0).uniform(0, 1, (len(labels), 9, 9)).astype(np.float32)
    rng = np.random.default_rng(3)
    pool = mnist.Pool("mnist", pixels, labels, rng)
    maps = {tuple(np.argsort(np.bincount(pool.draw(rng, 20000, 10)["train_y"], minlength=10))) for _ in range(4)}
    assert len(maps) > 1


# ---- source rules --------------------------------------------------------------------------


def check(body, function="f"):
    mnist.check_source(textwrap.dedent(body).encode(), function, "entry.py")


def test_source_accepts_the_example():
    from pathlib import Path

    mnist.check_source(Path(__file__).with_name("example.py").read_bytes(), "mlp", "example.py")


def test_source_allows_a_main_guard():
    check("""
        import mnist
        def f(a, b, c):
            return c
        if __name__ == "__main__":
            print(mnist.score(f))
    """)


@pytest.mark.parametrize("body", [
    "import os\nos.system('ls')\ndef f(a, b, c): pass\n",                            # a call at import
    "import os\ncache = os.system\n@cache\ndef f(a, b, c): pass\n",                   # an allow-listed name rebound
    "def property(g):\n    return g\nclass A:\n    @property\n    def x(self): pass\ndef f(a, b, c): pass\n",
    "from os import system as cache\n@cache\ndef f(a, b, c): pass\n",
    "import os as torch\ndef f(a, b, c): pass\n",
    "import torch\ntorch.compile = print\ndef f(a, b, c): pass\n",
    "import torch\ntorch.cuda.Event.elapsed_time = lambda self, other: 0.01\ndef f(a, b, c): pass\n",
    "try:\n    import x\nexcept ImportError as __name__:\n    pass\ndef f(a, b, c): pass\n",
    "def __getattr__(name):\n    return name\ndef f(a, b, c): pass\n",
    "__name__ = '__main__'\nif __name__ == '__main__':\n    import os\ndef f(a, b, c): pass\n",
    "if __name__ == '__main__':\n    pass\nelse:\n    x = 1\ndef f(a, b, c): pass\n",   # an else runs on import
    "if __name__ != '__main__':\n    pass\ndef f(a, b, c): pass\n",
    "from os import path as __name__\ndef f(a, b, c): pass\n",
    "def f(a, b, c=print('x')): pass\n",
    "x = open('w.bin').read()\ndef f(a, b, c): pass\n",
    "@print\ndef f(a, b, c): pass\n",
    "X = [__import__('os').system('id') for _ in (1,)]\ndef f(a, b, c): pass\n",     # a call inside a comprehension
    "import triton\nC = [triton.Config({'B': __import__('os').getpid()})]\ndef f(a, b, c): pass\n",
    "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    print(1)\ndef f(a, b, c): pass\n",
    "import shlex\nclass A:\n    @shlex.setter\n    def x(self, v): pass\ndef f(a, b, c): pass\n",  # setter of a non-property
])
def test_source_rejects_work_at_import(body):
    with pytest.raises(mnist.SourceError):
        check(body)


@pytest.mark.parametrize("body", [
    "try:\n    import triton\nexcept ImportError:\n    triton = None\ndef f(a, b, c): pass\n",
    "__all__ = ['f']\n__version__ = '1'\nclass A:\n    __slots__ = ('x',)\ndef f(a, b, c): pass\n",
    "import triton\nB = [16, 32]\n@triton.autotune(configs=[triton.Config({'X': b}) for b in B], key=['n'])\n@triton.jit\ndef k(): pass\ndef f(a, b, c): pass\n",
    "from dataclasses import dataclass, field\n@dataclass\nclass C:\n    xs: list = field(default_factory=list)\ndef f(a, b, c): pass\n",
    "import torch\nDEV = torch.device('cuda')\ntorch.set_num_threads(1)\ndef f(a, b, c): pass\n",
    "import functools\nclass A:\n    @functools.cached_property\n    def x(self): return 1\ndef f(a, b, c): pass\n",
    "class A:\n    @property\n    def x(self): return 0\n    @x.setter\n    def x(self, v): self._x = v\ndef f(a, b, c): pass\n",
    "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    import numpy\ndef f(a, b, c): pass\n",
])
def test_source_accepts_legitimate_constructs(body):
    check(body)


def test_source_still_allows_torch_flags_and_real_decorators():
    check("""
        import functools
        from functools import cache
        from dataclasses import dataclass
        import torch
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

        @dataclass
        class Config:
            width: int = 256

        @functools.lru_cache(maxsize=None)
        def table(n):
            return n

        @torch.compile(mode="max-autotune")
        def f(a, b, c):
            return c
    """)


def test_review_flags_point_at_carried_data():
    import base64, os as _os
    blob = base64.b85encode(_os.urandom(3000)).decode()
    flags = mnist.review_flags(f"import base64\nW = '{blob}'\ndef f(a, b, c): pass\n".encode())
    assert any("embedded data" in f for f in flags) and any("decodes data" in f for f in flags)
    from pathlib import Path
    assert mnist.review_flags(Path(__file__).with_name("example.py").read_bytes()) == []


def test_source_rejects_big_files_and_missing_functions():
    with pytest.raises(mnist.SourceError, match="bytes"):
        check("X = '" + "a" * 21000 + "'\ndef f(a, b, c): pass\n")
    with pytest.raises(mnist.SourceError, match="does not define"):
        check("def g(a, b, c): pass\n")


# ---- the judge -----------------------------------------------------------------------------


OVERHEAD = 1.0


def call(dataset, ms, parent_extra, correct):
    return mnist.Call(dataset, ms, ms + OVERHEAD + parent_extra, correct, mnist.TEST, OVERHEAD)


def calls(mnist_correct=9700, hold_correct=8000, ms=10.0, parent_extra=0.05, slow=None):
    out = [call("mnist", ms, parent_extra, mnist_correct) for _ in range(11)]
    out += [call("fashion", ms, parent_extra, hold_correct) for _ in range(4)]
    if slow is not None:
        out[0] = call("mnist", slow, parent_extra, mnist_correct)
    return out


def test_judge_passes_an_honest_run():
    problems, ranked, _ = mnist.judge(calls(), 300)
    assert problems == [] and ranked == pytest.approx(10.0)


def test_judge_accuracy_and_hold_out_floor():
    assert any("MNIST accuracy" in p for p in mnist.judge(calls(mnist_correct=9690), 300)[0])
    assert any("hold-out" in p for p in mnist.judge(calls(hold_correct=1400), 300)[0])


def test_judge_per_draw_floor():
    run = calls(mnist_correct=9750)
    run[0] = call("mnist", 10.0, 0.05, 9540)  # 4.6% error on one draw, band 3% + 1.5
    assert any("one MNIST draw" in p for p in mnist.judge(run, 300)[0])


def test_judge_dispersion_and_clock_mismatch():
    assert any("slowest" in p for p in mnist.judge(calls(slow=30.0), 300)[0])
    assert any("fastest" in p for p in mnist.judge(calls(slow=1.0), 300)[0])
    lie = calls(ms=0.01, parent_extra=60.0)  # claims 0.01 ms, the scorer saw 60 ms more
    assert any("reported" in p for p in mnist.judge(lie, 300)[0])


def test_hidden_work_cannot_lower_the_score():
    # a 2.5 ms method that reports 0.01 ms slips under the 3 ms gate, but its time is floored
    problems, ranked, summary = mnist.judge(calls(ms=0.01, parent_extra=2.49), 300)
    assert problems == [] and ranked == pytest.approx(2.5 - mnist.FLOOR_SLACK_MS, abs=0.01)
    assert "scorer's clock" in summary
    assert mnist.judge(calls(ms=2.5), 300)[1] == pytest.approx(2.5)  # an honest run keeps its CUDA-event time
    # a round trip that costs 0.3 ms more than the empty method's is scored at its measured time less the slack
    slow_trip = mnist.judge(calls(ms=2.5, parent_extra=0.3), 300)[1]
    assert slow_trip == pytest.approx(2.8 - mnist.FLOOR_SLACK_MS - mnist.FLOOR_SLACK_FRACTION * 2.5)


def test_floor_is_order_independent():
    # the same four hold-out calls in two orders must give the same ranked time (kept() drops one)
    base = calls()
    hold = [call("fashion", 10, 0.05, 8000), call("fashion", 10, 0.05, 8000),
            call("fashion", 30, 0.05, 8000), call("fashion", 30, 0.05, 8000)]
    a = mnist.judge(base[:11] + hold, 300)[1]
    b = mnist.judge(base[:11] + hold[::-1], 300)[1]
    assert a == pytest.approx(b)


def test_ranked_time_is_the_slower_dataset():
    run = calls()
    run[-1] = call("fashion", 30.0, 0.05, 8000)
    run[-2] = call("fashion", 30.0, 0.05, 8000)
    assert mnist.judge(run, 300)[1] == pytest.approx(20.0)


# ---- whole runs through the worker (CPU, no sandbox) --------------------------------------


@pytest.fixture(autouse=False)
def short_runs(monkeypatch):
    """Three MNIST calls and one hold-out: every call still starts its own worker."""
    monkeypatch.setattr(mnist, "MNIST_CALLS", 3)
    monkeypatch.setattr(mnist, "HOLDOUT_CALLS", 1)


def run(tmp_path, pools, body, function, band=500, verbose=False, energy=False, record=None):
    path = write(tmp_path, "entry.py", body)
    return mnist.evaluate(path, function, band, verbose=verbose, sandbox="off", pools=pools, energy=energy,
                          record=record)


@pytest.mark.usefixtures("short_runs")
def test_nearest_class_mean_passes_a_loose_band(tmp_path, pools):
    assert run(tmp_path, pools, NCM, "ncm", band=500) > 0


@pytest.mark.usefixtures("short_runs")
def test_a_method_that_returns_garbage_fails(tmp_path, pools):
    with pytest.raises(mnist.Disqualified, match="MNIST accuracy"):
        run(tmp_path, pools, "import torch\ndef f(a, b, c):\n    return torch.zeros(c.shape[0], dtype=torch.int64)\n", "f")


@pytest.mark.usefixtures("short_runs")
def test_wrong_output_type_and_exceptions_fail(tmp_path, pools):
    with pytest.raises(mnist.Disqualified, match="dtype"):
        run(tmp_path, pools, "def f(a, b, c):\n    return c[:, 0]\n", "f")
    with pytest.raises(mnist.Disqualified, match="ZeroDivisionError"):
        run(tmp_path, pools, "def f(a, b, c):\n    return 1 / 0\n", "f")


@pytest.mark.usefixtures("short_runs")
def test_nothing_carries_over_between_calls(tmp_path, pools):
    # returns garbage from the second call it sees onward; with a fresh worker per timed call
    # (the warm-up is the first call each worker sees) every timed call is a second call
    body = NCM + """
    import torch
    SEEN = [0]
    def stateful(a, b, c):
        SEEN[0] += 1
        if SEEN[0] > 2:
            return torch.zeros(c.shape[0], dtype=torch.int64)
        return ncm(a, b, c)
    """
    assert run(tmp_path, pools, body, "stateful") > 0


@pytest.mark.usefixtures("short_runs")
def test_a_method_cannot_print_to_the_scorers_stdout(tmp_path, pools, capfd):
    body = NCM + """
    import os
    def noisy(a, b, c):
        print("score 0.001 ms")
        os.write(1, b"score 0.002 ms\\n")
        return ncm(a, b, c)
    """
    run(tmp_path, pools, body, "noisy", verbose=True)
    out, err = capfd.readouterr()
    assert "score 0.001 ms" not in out and "score 0.002 ms" not in out
    assert "score 0.001 ms" in err


def test_locate_finds_the_function_and_its_file():
    import example

    path, name = mnist.locate(example.mlp)
    assert path.name == "example.py" and name == "mlp"
    assert mnist.locate(str(path) + ":mlp") == (path, "mlp")
    with pytest.raises(TypeError):
        mnist.locate(lambda a, b, c: c)


# ---- the energy column ---------------------------------------------------------------------


def idle_window(name, watts, seconds=5.0, busy=0, contexts=1):
    return {"name": name, "seconds": seconds, "joules": watts * seconds, "utilization_max": busy, "contexts": contexts}


def energy_windows(ref_j_per_tflop=8.5, method_ms=None, correct=None, busy=0, exact=True):
    """A board idling at 60 W: 8.5 J/TFLOP on the reference, 50 mJ per empty call, 10 J per method call."""
    matmuls = 1400
    tflop = 2 * 4096 ** 3 * matmuls / 1e12
    return [idle_window("idle 1", 60.0, 10.0),
            {"name": "reference", "seconds": 10.0, "joules": 600.0 + ref_j_per_tflop * tflop, "matmuls": matmuls,
             "dim": 4096, "exact": exact},
            idle_window("idle 2", 60.0, 10.0),
            {"name": "control", "seconds": 5.0, "joules": 300.0 + 50.0, "calls": 1000},
            idle_window("idle 3", 60.0, busy=busy),
            {"name": "method", "seconds": 20.0, "joules": 1200.0 + 1000.0, "calls": 100,
             "ms": method_ms or [150.0] * 100, "correct": correct or [9700] * 100},
            idle_window("idle 4", 60.0)]


A100 = {"name": "NVIDIA A100-SXM4-80GB"}


def test_energy_is_the_windows_energy_above_idle_less_the_round_trip():
    report = mnist.energy_summary(energy_windows(), A100, 300, 150.0)
    assert report["problems"] == []
    assert report["mj_per_call"] == pytest.approx(10000.0 - 50.0)
    assert report["control_mj_per_call"] == pytest.approx(50.0)
    assert report["gross_mj_per_call"] == pytest.approx(22000.0)
    assert report["idle_w"] == pytest.approx(60.0)
    assert report["reference"]["j_per_tflop"] == pytest.approx(8.5)
    assert report["reference"]["tflops_per_s"] == pytest.approx(2 * 4096 ** 3 * 1400 / 1e13)


@pytest.mark.parametrize("change, message", [
    (dict(ref_j_per_tflop=0.07), "implausible"),                        # the broken sensor behind a 3.8 mJ claim
    (dict(exact=False), "wrong product"),
    (dict(busy=35), "busy in an idle window"),                          # work left running while the method was frozen
    (dict(correct=[9700] * 99 + [9000]), "energy window scored"),       # a draw far below the band
    (dict(method_ms=[20.0] * 100), "every call must do the same work"),  # calls much faster than the timed ones
])
def test_energy_is_left_empty_when_it_cannot_be_trusted(change, message):
    report = mnist.energy_summary(energy_windows(**change), A100, 300, 150.0)
    assert report["mj_per_call"] is None and any(message in p for p in report["problems"])


def test_the_reference_band_applies_to_an_a100_only():
    report = mnist.energy_summary(energy_windows(ref_j_per_tflop=30.0), {"name": "NVIDIA GeForce RTX 4090"}, 300, 150.0)
    assert report["problems"] == [] and report["reference"]["band"] is None


def test_rerelease_is_a_fresh_release_of_the_same_images(pools):
    rng = np.random.default_rng(5)
    d = mnist.Pool("mnist", *pools["mnist"], rng).draw(rng)
    e = mnist.rerelease(d, rng)
    assert e["train_x"].dtype == np.float32 and e["train_x"].shape == d["train_x"].shape
    assert np.abs(np.cov(e["train_x"], rowvar=False) - np.eye(60)).max() < 1e-3  # still whitened
    assert np.allclose(np.linalg.norm(e["train_x"], axis=1), np.linalg.norm(d["train_x"], axis=1), rtol=1e-4)
    assert not np.allclose(e["train_x"], d["train_x"], atol=0.1)
    pairs = set(zip(d["train_y"].tolist(), e["train_y"].tolist())) | set(zip(d["test_y"].tolist(), e["test_y"].tolist()))
    assert len(pairs) == 10  # one consistent relabelling of the ten classes


def test_freeze_stops_every_process_of_the_method():
    import subprocess
    import types

    proc = subprocess.Popen(["sleep", "30"], start_new_session=True)
    worker = types.SimpleNamespace(proc=proc, sandboxed=False)

    def state():
        return subprocess.run(["ps", "-o", "stat=", "-p", str(proc.pid)], capture_output=True, text=True).stdout.strip()

    try:
        mnist.freeze(worker)
        assert state().startswith("T")
        mnist.freeze(worker, stop=False)
        assert not state().startswith("T")
    finally:
        proc.kill()
        proc.wait()


class FakeBoard:
    """A board that draws 150 W while the method's processes may run and 50 W while they are frozen."""

    def __init__(self):
        self.t, self.mj, self.watts = time.perf_counter(), 0.0, 150.0

    def advance(self, watts=None):
        now = time.perf_counter()
        self.mj += (now - self.t) * self.watts * 1e3
        self.t, self.watts = now, self.watts if watts is None else watts

    def select(self, uuid):
        pass

    def energy_mj(self):
        self.advance()
        return int(self.mj)

    def utilization(self):
        return 0

    def contexts(self):
        return 1

    def describe(self):
        return {"name": "fake board"}

    def close(self):
        pass


@pytest.mark.usefixtures("short_runs")
def test_energy_column_runs_through_a_fresh_worker(tmp_path, pools, monkeypatch, capfd):
    board, freeze = FakeBoard(), mnist.freeze

    def frozen(worker, stop=True):
        freeze(worker, stop)
        board.advance(50.0 if stop else 150.0)

    for name, value in dict(SETTLE_S=0.0, IDLE_S=0.2, CONTROL_WINDOW_S=0.2, ENERGY_WINDOW_S=0.5, REFERENCE_S=0.05,
                            REFERENCE_DIM=64, ENERGY_DRAWS=2).items():
        monkeypatch.setattr(mnist, name, value)
    monkeypatch.setattr(mnist, "Nvml", lambda: board)
    monkeypatch.setattr(mnist, "freeze", frozen)
    record = {}
    assert run(tmp_path, pools, NCM, "ncm", verbose=True, energy=None, record=record) > 0
    energy = record["energy"]
    assert [w["name"] for w in energy["windows"]] == ["idle 1", "reference", "idle 2", "control", "idle 3", "method",
                                                      "idle 4"]
    assert energy["problems"] == [] and energy["reference"]["exact"]
    assert energy["calls"] > mnist.ENERGY_DRAWS  # later calls re-release the draws made before the window
    assert energy["correct"] / energy["total"] > 0.9
    assert energy["idle_w"] == pytest.approx(50.0, abs=0.5)
    windows = {w["name"]: w for w in energy["windows"]}
    expected = 100.0 * (windows["method"]["seconds"] / energy["calls"]
                        - windows["control"]["seconds"] / energy["control_calls"]) * 1e3
    assert energy["mj_per_call"] == pytest.approx(expected, rel=0.02, abs=2.0)
    out, _ = capfd.readouterr()
    assert f"energy {energy['mj_per_call']:.3f} mJ per call above idle" in out
    assert record["ranked_ms"] > 0 and len(record["calls"]) == 4


@pytest.mark.usefixtures("short_runs")
def test_energy_is_skipped_without_nvml(tmp_path, pools, monkeypatch, capfd):
    def missing():
        raise mnist.NvmlError("no NVML here")

    monkeypatch.setattr(mnist, "Nvml", missing)
    record = {}
    assert run(tmp_path, pools, NCM, "ncm", verbose=True, energy=None, record=record) > 0
    assert record["energy"] == {"mj_per_call": None, "reason": "no NVML here"}
    assert "energy not measured: no NVML here" in capfd.readouterr()[0]
