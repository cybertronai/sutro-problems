"""Tests for mnist.py that run on a CPU without a network: the release, the draws, the
source rules, the judge, and whole runs of the worker protocol on synthetic pools.

    python -m pytest -q test_mnist.py
"""

import textwrap

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


def run(tmp_path, pools, body, function, band=500, verbose=False):
    path = write(tmp_path, "entry.py", body)
    return mnist.evaluate(path, function, band, verbose=verbose, sandbox="off", pools=pools)


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
