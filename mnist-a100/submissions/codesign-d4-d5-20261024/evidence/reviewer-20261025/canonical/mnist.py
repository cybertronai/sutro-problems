#!/usr/bin/env python3
"""MNIST on an A100: learn from 10,000 labelled images, then label 10,000 more, fast.

    import mnist

    def my_method(train_x, train_y, test_x):
        # train_x (10000, 60) float32, train_y (10000,) int64 in 0..9, test_x (10000, 60) float32,
        # all on the GPU. Train from scratch here: every call runs in a fresh process.
        ...
        return labels  # (10000,) integer tensor on the GPU

    if __name__ == "__main__":
        print(mnist.score(my_method, difficulty=1))  # ms per call; raises mnist.Disqualified

Or from the shell, with a clean scorer process (how official times are taken):

    python mnist.py my_method.py[:function] --difficulty 1

A difficulty is an error band. To pass it, the mean test error over the MNIST
calls must be at most the band. The score is the mean time per call on the
slower of the two datasets (MNIST and the hold-out), timed with CUDA events and
floored by this process's own clock. The bands are the error a Ladder network
reaches when trained for 24,000 steps on only N labelled images (DIFFICULTY).
Your method always gets all 10,000.

What a run does (the protections of sutro-mnist-medium/3.0.0, the popcorn3
KernelBot harness, tightened after a red-team review of this port):

* Every timed call runs in a fresh worker process that imports your file, gets
  one untimed warm-up call on a foreign dataset (compile, autotune and capture
  CUDA graphs there), then the timed call, and is killed. Nothing in memory
  carries over to the next call. The worker gets train_x, train_y and test_x
  and sends back label bytes only; the labels, the draws and the scoring stay
  in this process.
* As root on Linux x86-64 (for example on Modal) the worker runs as an
  unprivileged user (uid 65534) under a seccomp filter: no network, no access
  to this process or the dataset files, and every file it leaves in /tmp is
  deleted before the next call. Elsewhere it runs without the sandbox, with a
  warning: such a time is for your information only, and files can persist.
* Each call is a fresh draw: 10,000 training and 10,000 test images, disjoint,
  from the 60,000-image MNIST training set, box-averaged 28x28 -> 9x9. Training
  and test images come from opposite halves of the pool, split once per run.
  The ten class labels are secretly permuted per draw.
* The draw arrives released, not as pixels: z = Q W (x - mu). W whitens onto
  the top 60 principal directions of that draw's training rows, and Q is a
  Haar-random rotation, fresh for every draw. The release hides coordinates,
  not identity: rotation-invariant statistics survive, so a method holding the
  60,000-image pool could re-identify the images. The sandbox keeps the pool
  out of reach and the 20,480-byte limit keeps it out of the file.
* 11 MNIST calls and 4 hold-out calls run in a secret order. The hold-out is
  Fashion-MNIST or KMNIST, released and label-permuted the same way, and must
  score at least 15%. The ranked time is the slower of the two datasets' means,
  so recognising MNIST instead of learning it gains nothing. The warm-ups use
  the foreign dataset that is not this run's hold-out.
* Calls must do the same work: within a dataset, the slowest call may take at
  most 2x the median + 2 ms, the fastest at least half the median - 2 ms.
* Time is taken twice. The worker times the call with CUDA events after
  flushing the 40 MB L2 (256 MB written). This process times the whole round
  trip, less what the same trip costs with an empty method (measured in each
  worker before your file is imported). A CUDA-event time far below that clock
  fails, and a dataset's time is never less than this clock's mean less
  0.1 ms + 0.5%, so hiding work from the events cannot lower a score.
* No MNIST draw may be more than 1.5 points worse than the band. Each timed
  call is limited to 60 s, each warm-up to 180 s.
* Energy is a column beside the score, not the score. After a passing run, one
  more fresh worker runs the method back to back on fresh MNIST draws for 20 s,
  and this process reads the board's NVML energy counter at the edges of that
  window and of idle windows in which every process of the method is frozen
  (SIGSTOP). The column is the board's energy per call above idle, less what
  the same round trip costs with an empty method. One call cannot be read on
  its own: the A100's counter moves every 100 ms. Before the method is
  imported, the same worker runs an FP32 matmul whose energy shows whether the
  board's power telemetry is plausible; if it is not, or the method behaves
  differently in the window, the column is left empty and the score stands.
* Your file is at most 20,480 bytes, so it cannot carry the pool or a large
  trained model. At import it may only import, define functions and classes,
  assign constants and set torch flags; the names the allow-lists trust
  (torch, cache, property, ...) cannot be rebound. A final
  `if __name__ == "__main__":` block is allowed, because the worker never runs
  it. Review flags (large high-entropy literals, decoders, network imports)
  are printed for a human to read; they reject nothing. A method small enough
  to fit can still carry a compact prior trained offline, so records are read
  before they stand.
"""

import ast
import ctypes
import dataclasses
import gc
import gzip
import hashlib
import json
import math
import mmap
import os
import platform
import select
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import time
import traceback
import urllib.request
import warnings
from pathlib import Path

VERSION = "mnist-a100/1.2.0 (protections of sutro-mnist-medium/3.0.0, tightened; energy column)"

# difficulty: (band, as the most mean test error allowed in basis points;
#              labelled images the Ladder network needs to reach it at 24,000 steps)
DIFFICULTY = {
    1: (540, 200),
    2: (340, 532),
    3: (270, 1414),
    4: (230, 3761),
    5: (190, 10000),
}

TRAIN, TEST, DIMS, SIDE = 10000, 10000, 60, 9
MNIST_CALLS, HOLDOUT_CALLS = 11, 4
DRAW_SLACK_BP = 150            # no MNIST draw may be worse than the band plus this
HOLDOUT_ERROR_BP = 8500        # the hold-out calls must be at least 15% correct
DISPERSION = 2.0               # slowest call <= 2x median + 2 ms, fastest >= median / 2 - 2 ms
MAX_CALL_MS, WARMUP_MAX_CALL_MS, LATER_WARMUP_MAX_CALL_MS = 60000, 180000, 90000
RUN_MAX_S = 2700  # a whole run; each call starts a worker, so budgets per call add up
MAX_SOURCE_BYTES = 20480
DATASETS = ("mnist", "fashion", "kmnist")
FOREIGN = ("fashion", "kmnist")

# Timing gate. "Unexplained" time is this process's measurement of a call, less
# the protocol's calibrated cost, less the method's CUDA-event time; on an A100
# it averages under 1 ms, with rare scheduling hiccups up to ~20 ms. Over one
# dataset's calls, after dropping the largest fifth (at least one), its mean may
# be at most GROUP_SLACK_MS + GROUP_SLACK_FRACTION of the mean call. Any single
# call may be at most CALL_SLACK_MS + CALL_SLACK_FRACTION of itself.
GROUP_SLACK_MS, GROUP_SLACK_FRACTION = 3.0, 0.15
CALL_SLACK_MS, CALL_SLACK_FRACTION = 50.0, 0.5
# The ranked time of a dataset is at least this process's mean measurement over the same calls
# less FLOOR_SLACK_MS + FLOOR_SLACK_FRACTION of the reported mean, so work hidden from the CUDA
# events (a side stream, a re-recorded start event, a patched timer) gains at most that much. An
# honest method whose round trip costs more than the empty method's is scored at its measured time
# less the slack (A100 runs of the popcorn3 harness left 0-0.7 ms unexplained on 25-250 ms calls).
FLOOR_SLACK_MS, FLOOR_SLACK_FRACTION = 0.1, 0.005
CALIBRATION_CALLS, CALIBRATION_DISCARD = 16, 6
L2_FLUSH_BYTES = 256 << 20
# Energy, the column beside the ranked time. The A100's energy counter moves every ~100 ms, so a
# call of tens of ms cannot be read on its own (thirty isolated 74 ms matmul bursts read 8.1-15.4 J
# against 16.5 J, energy/results/probe-a100-80gb.json). The method runs back to back instead, and
# the window's energy above idle is split over its calls.
ENERGY_WINDOW_S, ENERGY_MIN_CALLS = 20.0, 3  # the method's window: back-to-back calls on fresh MNIST draws
ENERGY_DRAWS = 32          # draws made before the window; later calls get one again under a fresh Q and labels
CONTROL_WINDOW_S = 5.0     # empty calls: what the round trip itself costs per call, subtracted
SETTLE_S, IDLE_S = 3.0, 5.0  # each idle window, after the board settles, with the method frozen
REFERENCE_S, REFERENCE_DIM = 5.0, 4096  # FP32 matmul of constant operands, TF32 off, before the import
# What a healthy board reads on that reference, in J above idle per 10^12 FLOPs and TFLOP/s: six
# A100s (four SXM4-80GB on Modal, two SXM4-40GB) read 8.1-8.8 J at 18.5-19.0 TFLOP/s, and fifteen
# more on Modal (eleven 80GB PCIe, four SXM4-80GB) 7.2-9.2 J at 17.2-18.4 TFLOP/s (energy/); the
# broken sensor behind a published 3.8 mJ claim read about 0.07 J. Other GPUs are reported unchecked.
REFERENCE_BANDS = {"NVIDIA A100": ((6.0, 11.0), (15.0, 23.0))}
WORKER_START_S, IMPORT_S, STAGE_S = 180, 180, 60
MAX_HEADER = 1 << 16
WORKER_FLAG = "--mnist-a100-worker"


class Disqualified(Exception):
    """The method broke a rule. ``problems`` lists every rule it broke."""

    def __init__(self, summary, problems):
        self.summary, self.problems = summary, list(problems)
        super().__init__("\n".join(([summary] if summary else []) + self.problems))


# ============================================================================
# Public API
# ============================================================================


def score(method, difficulty=1, *, verbose=True, energy=None):
    """Time ``method`` at a difficulty from 1 (loosest) to 5 (tightest).

    ``method(train_x, train_y, test_x) -> labels`` must be a function defined at
    the top level of a .py file (it is re-imported from that file in a worker
    process); ``"file.py:function"`` works too. Returns the ranked time in
    milliseconds per call. Raises ``Disqualified`` when a rule is broken.
    A passing run then measures the energy column where the GPU's energy
    counter can be read (about 80 s more); ``energy=False`` skips it.
    """
    path, function = locate(method)
    return evaluate(path, function, band_bp(difficulty), verbose=verbose, energy=energy)


def band_bp(difficulty):
    if difficulty not in DIFFICULTY:
        raise ValueError(f"difficulty must be one of {sorted(DIFFICULTY)}, not {difficulty!r}")
    band = DIFFICULTY[difficulty][0]
    if not band:
        raise RuntimeError(f"the band for difficulty {difficulty} has not been set yet")
    return band


def draw(dataset="mnist", seed=None):
    """One released draw for development, as numpy arrays:
    (train_x (10000, 60) f32, train_y (10000,) i64, test_x (10000, 60) f32, test_y (10000,) i64)."""
    import numpy as np

    rng = np.random.default_rng(seed)
    pixels, labels = load_pool(dataset)
    d = Pool(dataset, pixels, labels, rng).draw(rng)
    return d["train_x"], d["train_y"], d["test_x"], d["test_y"]


def locate(method):
    """(path of the .py file, name of the top-level function) for a function or "file.py:function"."""
    if isinstance(method, (str, os.PathLike)):
        text = str(method)
        path, _, function = text.rpartition(":") if text.rpartition(":")[0].endswith(".py") else (text, "", "")
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"no method file at {path}")
        if not function:
            tree = ast.parse(path.read_bytes())
            names = [n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            if len(names) != 1:
                raise ValueError(f"{path} defines {len(names)} top-level functions; name one as {path}:function")
            function = names[0]
        return path.resolve(), function
    module = sys.modules.get(getattr(method, "__module__", None) or "")
    source = getattr(module, "__file__", None)
    names = [name for name, value in vars(module).items() if value is method] if module else []
    if not callable(method) or not source or not source.endswith(".py") or not names:
        raise TypeError("score() needs a function defined at the top level of a .py file "
                        "(not a lambda, a nested function or a notebook cell); the worker imports it from that file")
    name = getattr(method, "__name__", None)
    return Path(source).resolve(), name if name in names else names[0]


def evaluate(path, function, band, verbose=True, sandbox=None, pools=None, energy=None, record=None):
    """Score ``function`` from the file ``path`` against a band in basis points. ``energy=False``
    skips the energy column; a dict passed as ``record`` receives every call and energy window."""
    import numpy as np

    say = print if verbose else (lambda *args, **kwargs: None)
    keep = record.update if record is not None else (lambda **fields: None)
    name = Path(path).name
    source = Path(path).read_bytes()
    try:
        check_source(source, function, name)
    except SourceError as error:
        say(f"disqualified: {error}", flush=True)
        raise Disqualified(f"{name} was not run", [str(error)]) from None
    policy = sandbox or os.environ.get("MNIST_SANDBOX", "auto")
    reason = sandbox_available()
    sandboxed = policy != "off" and reason is None
    if policy == "required" and reason:
        raise RuntimeError(f"cannot sandbox the method: {reason}")
    raw = pools or {dataset: load_pool(dataset) for dataset in DATASETS}
    rng = np.random.default_rng(np.random.SeedSequence(int.from_bytes(os.urandom(16), "little")))
    pools = {dataset: Pool(dataset, *raw[dataset], rng) for dataset in DATASETS}
    order = list(FOREIGN)
    rng.shuffle(order)  # the warm-up dataset is never this run's hold-out
    warmup, holdout = order
    sequence = ["mnist"] * MNIST_CALLS + [holdout] * HOLDOUT_CALLS
    rng.shuffle(sequence)
    if sandboxed:
        not_dumpable()
    say(f"{name}:{function} at {band / 100:.2f}% error: {len(sequence)} timed calls, each in a fresh process", flush=True)
    for flag in review_flags(source):
        say(f"  review flag: {flag}", flush=True)

    keep(version=VERSION, file=name, function=function, source_sha256=hashlib.sha256(source).hexdigest(),
         band_bp=band, sandboxed=sandboxed, holdout=holdout)

    def announce(hello):
        keep(device=hello.get("device"), torch=hello.get("torch"))
        say(f"  {hello.get('device')}, torch {hello.get('torch')}, sandbox "
            + ("on" if sandboxed else f"OFF ({reason or 'MNIST_SANDBOX=off'}): not an official time"), flush=True)

    calls, started = [], time.monotonic()
    try:
        for index, dataset in enumerate(sequence):
            if time.monotonic() - started > RUN_MAX_S:
                raise Failure(f"the run took more than {RUN_MAX_S // 60} minutes")
            call = timed_call(source, function, sandboxed, pools[warmup].draw(rng), pools[dataset].draw(rng),
                              announce if index == 0 else None,
                              (WARMUP_MAX_CALL_MS if index == 0 else LATER_WARMUP_MAX_CALL_MS) / 1e3)
            say(f"  call {index + 1:2d}/{len(sequence)}: {call.ms:,.3f} ms, scorer's clock "
                f"{max(0.0, call.parent_ms - call.overhead_ms):,.3f} ms", flush=True)
            calls.append(call)
    except Failure as error:
        say(f"disqualified: {error}", flush=True)
        keep(calls=[dataclasses.asdict(c) for c in calls], problems=[str(error)], ranked_ms=None)
        raise Disqualified("", [str(error)]) from None
    problems, ranked_ms, summary = judge(calls, band)
    say(summary + ("" if problems else f"; score {ranked_ms:.3f} ms"), flush=True)
    for problem in problems:
        say(f"disqualified: {problem}", flush=True)
    keep(calls=[dataclasses.asdict(c) for c in calls], problems=problems, summary=summary,
         ranked_ms=None if problems else ranked_ms)
    if problems:
        raise Disqualified(summary, problems)
    if energy is not False:
        mnist_ms = _median([c.ms for c in calls if c.dataset == "mnist"])
        keep(energy=energy_column(source, function, sandboxed, pools, rng, warmup, band, mnist_ms, say))
    return ranked_ms


def timed_call(source, function, sandboxed, warm, d, announce=None, warmup_s=WARMUP_MAX_CALL_MS / 1e3):
    """One fresh worker: start it, calibrate the protocol, import the method, run an untimed
    warm-up on a foreign draw, then the timed call on ``d``; the worker is killed afterwards."""
    import numpy as np

    if sandboxed:
        reap_sandbox()
    worker = Worker(sandboxed, source)
    try:
        hello, _ = worker.request({"cmd": "hello", "buffer": str(worker.buffer_path),
                                   "buffer_bytes": worker.buffer_bytes}, timeout_s=WORKER_START_S, what="starting")
        if sandboxed and not hello.get("sandboxed"):
            raise RuntimeError(f"could not sandbox the method: {hello.get('sandbox_error')}")
        if announce:
            announce(hello)
        overhead = calibrate(worker)
        reply, _ = worker.request({"cmd": "load", "function": function}, timeout_s=IMPORT_S,
                                  what="importing the method file")
        if not reply.get("ok"):
            raise Failure(f"importing the method file failed:\n{reply.get('error')}")
        worker.call(warm, warmup_s, label="the untimed warm-up on a")
        ms, parent_ms, preds = worker.call(d, MAX_CALL_MS / 1e3 + 5)
        if ms > MAX_CALL_MS:
            raise Failure(f"a call took {ms:,.0f} ms; the limit is {MAX_CALL_MS:,} ms")
        correct = int(np.count_nonzero(preds.astype(np.int64) == d["test_y"]))
        return Call(d["dataset"], ms, parent_ms, correct, TEST, overhead)
    finally:
        worker.close()


def not_dumpable():
    """Defence in depth: a non-dumpable process's /proc entries are root-only."""
    if sys.platform == "linux":
        try:
            ctypes.CDLL(None).prctl(4, 0, 0, 0, 0)  # PR_SET_DUMPABLE
        except Exception:  # noqa: BLE001
            pass


# ============================================================================
# Data: the pools, the draws and the release
# ============================================================================

# dataset -> mirrors, and file -> sha256 (the md5s match torchvision's records).
# Only the 60,000-image training split of each is used.
SOURCES = {
    "mnist": {
        "mirrors": ("https://ossci-datasets.s3.amazonaws.com/mnist/", "https://storage.googleapis.com/cvdf-datasets/mnist/"),
        "images": ("train-images-idx3-ubyte.gz", "440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609"),
        "labels": ("train-labels-idx1-ubyte.gz", "3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c"),
    },
    "fashion": {
        "mirrors": ("https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/",
                    "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"),
        "images": ("train-images-idx3-ubyte.gz", "3aede38d61863908ad78613f6a32ed271626dd12800ba2636569512369268a84"),
        "labels": ("train-labels-idx1-ubyte.gz", "a04f17134ac03560a47e3764e11b92fc97de4d1bfaf8ba1a3aa29af54cc90845"),
    },
    "kmnist": {
        "mirrors": ("https://codh.rois.ac.jp/kmnist/dataset/kmnist/",),
        "images": ("train-images-idx3-ubyte.gz", "51467d22d8cc72929e2a028a0428f2086b092bb31cfb79c69cc0a90ce135fde4"),
        "labels": ("train-labels-idx1-ubyte.gz", "e38f9ebcd0f3ebcdec7fc8eabdcdaef93bb0df8ea12bee65224341c8183d8e17"),
    },
}
POOL_SIZE = 60000


def cache_dir():
    """Downloads are kept where only this user can read them (the sandbox user cannot)."""
    path = Path(os.environ.get("MNIST_CACHE_DIR", Path.home() / ".cache" / "sutro-mnist"))
    path.mkdir(parents=True, exist_ok=True)
    os.chmod(path, 0o700)
    return path


def fetch(dataset, kind):
    """The verified gzip bytes of one file, from the cache or a mirror."""
    filename, digest = SOURCES[dataset][kind]
    cached = cache_dir() / f"{dataset}-{filename}"
    if cached.exists():
        data = cached.read_bytes()
        if hashlib.sha256(data).hexdigest() == digest:
            return data
    errors = []
    for mirror in SOURCES[dataset]["mirrors"]:
        try:
            request = urllib.request.Request(mirror + filename, headers={"User-Agent": "mnist-a100/1"})
            with urllib.request.urlopen(request, timeout=120) as response:
                data = response.read()
        except Exception as error:  # noqa: BLE001 - try the next mirror
            errors.append(f"{mirror}: {error}")
            continue
        if hashlib.sha256(data).hexdigest() != digest:
            errors.append(f"{mirror}: sha256 mismatch")
            continue
        partial = cached.with_suffix(".part")
        partial.write_bytes(data)
        os.chmod(partial, 0o600)
        partial.replace(cached)
        return data
    raise RuntimeError(f"could not load {dataset} {filename}: " + "; ".join(errors))


def read_idx(payload, images):
    import numpy as np

    header = 16 if images else 8
    fields = struct.unpack(">IIII" if images else ">II", payload[:header])
    if fields != ((2051, POOL_SIZE, 28, 28) if images else (2049, POOL_SIZE)):
        raise ValueError(f"unexpected IDX header {fields}")
    values = np.frombuffer(payload, dtype=np.uint8, offset=header)
    return values.reshape(POOL_SIZE, 28, 28) if images else values.copy()


def area_weights(input_size, output_size):
    """Exact overlap of unit input pixels with output bins, rows summing to 1."""
    import numpy as np

    left = np.arange(output_size, dtype=np.int64)[:, None] * input_size
    right = left + input_size
    pixel_left = np.arange(input_size, dtype=np.int64)[None, :] * output_size
    pixel_right = pixel_left + output_size
    overlap = np.maximum(0, np.minimum(right, pixel_right) - np.maximum(left, pixel_left))
    return (overlap / input_size).astype(np.float32)


def load_pool(dataset):
    """(pixels (60000, 9, 9) float32 in [0, 1], labels (60000,) uint8): uint8 / 255, exact box-area
    averaging 28x28 -> 9x9, clipped to [0, 1], as in the Sutro MNIST tiers."""
    import numpy as np

    images = read_idx(gzip.decompress(fetch(dataset, "images")), images=True)
    labels = read_idx(gzip.decompress(fetch(dataset, "labels")), images=False)
    weights = area_weights(28, SIDE)
    pixels = np.matmul(np.matmul(weights, images.astype(np.float32) / np.float32(255)), weights.T)
    np.clip(pixels, 0.0, 1.0, out=pixels)
    return np.ascontiguousarray(pixels, dtype=np.float32), labels


def haar_rotation(dim, rng):
    """A Haar-uniform orthogonal matrix: QR of a Gaussian matrix, column signs fixed by diag(R)."""
    import numpy as np

    q, r = np.linalg.qr(rng.standard_normal((dim, dim)))
    return q * np.sign(np.diag(r))[None, :]


def fit_whitener(train_rows, dims=DIMS):
    """(mu, W): exact PCA whitening onto the top ``dims`` principal directions of the rows,
    in float64, no variance floor (a floor would leak the always-dark border through the
    bottom eigenvectors). Eigenvector signs are pinned, so the fit is deterministic."""
    import numpy as np

    flat = np.asarray(train_rows, dtype=np.float64).reshape(len(train_rows), -1)
    dim = flat.shape[1]
    mu = flat.mean(0)
    eigenvalues, vectors = np.linalg.eigh(np.cov(flat - mu, rowvar=False))
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    signs = np.sign(vectors[np.argmax(np.abs(vectors), axis=0), np.arange(dim)])
    signs[signs == 0] = 1.0
    vectors = vectors * signs[None, :]
    top = np.arange(dim - dims, dim)  # eigh returns ascending eigenvalues
    eigenvalues, vectors = eigenvalues[top], vectors[:, top]
    if eigenvalues.min() <= 1e-10:
        raise ValueError(f"the training rows are rank deficient: direction {dims} has variance {eigenvalues.min():.3e}")
    return mu, (vectors / np.sqrt(eigenvalues)[None, :]).T


def apply_release(images, mu, transform):
    import numpy as np

    flat = np.asarray(images, dtype=np.float64).reshape(len(images), -1)
    return np.ascontiguousarray((flat - mu) @ transform.T, dtype=np.float32)


class Pool:
    """One dataset, split in half once per run: every draw's training images come from one
    half and its test images from the other, so no image is asked about after it was shown."""

    def __init__(self, name, pixels, labels, rng):
        self.name, self.pixels, self.labels = name, pixels, labels
        order = rng.permutation(len(labels))
        self.train_half, self.test_half = order[: len(order) // 2], order[len(order) // 2:]

    def draw(self, rng, n_train=TRAIN, n_test=TEST):
        """A draw with its own secret label permutation and its own secret release z = Q W (x - mu),
        fitted on the draw's training rows. The map never leaves this function."""
        import numpy as np

        train = rng.choice(self.train_half, n_train, replace=False)
        test = rng.choice(self.test_half, n_test, replace=False)
        relabel = rng.permutation(10).astype(np.int64)
        mu, whitener = fit_whitener(self.pixels[train])
        transform = haar_rotation(DIMS, rng) @ whitener
        out = {"dataset": self.name,
               "train_x": apply_release(self.pixels[train], mu, transform), "train_y": relabel[self.labels[train]],
               "test_x": apply_release(self.pixels[test], mu, transform), "test_y": relabel[self.labels[test]]}
        del mu, whitener, transform
        return out


def rerelease(d, rng):
    """The same images under a fresh secret rotation and label permutation, in about a millisecond:
    R z is again a release of the draw (R Q is Haar when R is). Energy windows use it once they
    have used every draw made for them, so that no two calls ever see the same inputs."""
    import numpy as np

    turn = haar_rotation(DIMS, rng).T.astype(np.float32)
    relabel = rng.permutation(10).astype(np.int64)
    return {"dataset": d["dataset"], "train_x": d["train_x"] @ turn, "train_y": relabel[d["train_y"]],
            "test_x": d["test_x"] @ turn, "test_y": relabel[d["test_y"]]}


# ============================================================================
# Source rules
# ============================================================================

# Calls allowed at module scope: they configure torch and read nothing.
MODULE_CALLS = frozenset({"torch.set_float32_matmul_precision", "torch.set_default_dtype", "torch.set_grad_enabled",
                          "torch.manual_seed", "torch.use_deterministic_algorithms", "torch.set_num_threads",
                          "torch.set_num_interop_threads", "torch.set_printoptions", "torch.set_flush_denormal"})
# Decorators allowed on module-level functions and classes, bare or called with static arguments.
DECORATORS = frozenset({"torch.compile", "torch.no_grad", "torch.inference_mode", "torch.jit.script", "triton.jit",
                        "triton.autotune", "triton.heuristics", "functools.cache", "functools.lru_cache", "cache",
                        "lru_cache", "staticmethod", "classmethod", "property", "dataclass", "dataclasses.dataclass",
                        "functools.cached_property", "cached_property"})
STATIC_CALLS = frozenset({"triton.Config", "dataclasses.field", "field", "torch.device"})
# Class-body methods Python calls implicitly while the module is still being imported.
CLASS_HOOKS = frozenset({"__init_subclass__", "__class_getitem__", "__set_name__", "__mro_entries__"})
# Names the allow-lists trust. Rebinding one (cache = print, def property(f): ...) would turn an
# allowed decorator into user code that runs at import, so none of them may be rebound, and an
# import may bind one only to the real thing.
RESERVED = frozenset(name.split(".")[0] for name in DECORATORS | MODULE_CALLS | STATIC_CALLS)
FROM_IMPORTS = {"cache": "functools", "lru_cache": "functools", "cached_property": "functools",
                "dataclass": "dataclasses", "field": "dataclasses"}
# Dunder metadata a module or class may assign (data, never rebinding __name__ or a hook).
DUNDER_DATA = frozenset({"__all__", "__version__", "__author__", "__doc__", "__slots__", "__constants__"})
# Allow-listed dotted names and their prefixes, which a torch flag assignment may not overwrite.
PROTECTED = frozenset(".".join(name.split(".")[:i]) for name in DECORATORS | MODULE_CALLS | STATIC_CALLS
                      for i in range(2, name.count(".") + 2))
_STATIC_NODES = (ast.Constant, ast.Name, ast.Attribute, ast.Subscript, ast.Slice, ast.Starred, ast.Tuple, ast.List,
                 ast.Dict, ast.Set, ast.UnaryOp, ast.BinOp, ast.BoolOp, ast.Compare, ast.IfExp, ast.JoinedStr,
                 ast.FormattedValue, ast.keyword, ast.expr_context, ast.operator, ast.unaryop, ast.boolop, ast.cmpop,
                 ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp, ast.comprehension)


class SourceError(Exception):
    pass


def dotted(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return ".".join(reversed(parts))


def _static(node, aliases=None):
    """True when evaluating ``node`` cannot run the method's code: constants, names, attribute
    reads, lambdas (not called), and allow-listed calls (``aliases`` maps a bound name to the
    dotted origin an ``import ... as`` gave it, so ``field(...)`` is seen as ``dataclasses.field``)."""
    if node is None:
        return True
    if isinstance(node, ast.Lambda):
        return all(_static(d, aliases) for d in node.args.defaults + node.args.kw_defaults)
    if isinstance(node, ast.Call):
        name = (aliases or {}).get(dotted(node.func), dotted(node.func))
        return name in STATIC_CALLS and all(_static(c, aliases) for c in node.args + [k.value for k in node.keywords])
    if not isinstance(node, _STATIC_NODES):
        return False
    return all(_static(child, aliases) for child in ast.iter_child_nodes(node))


def _decorator_ok(node, aliases=None, setters=frozenset()):
    if isinstance(node, ast.Call):
        name = (aliases or {}).get(dotted(node.func), dotted(node.func))
        return name in DECORATORS and all(_static(c, aliases) for c in node.args + [k.value for k in node.keywords])
    if isinstance(node, ast.Attribute) and node.attr in {"setter", "getter", "deleter"}:
        return isinstance(node.value, ast.Name) and node.value.id in setters  # @x.setter for a property x defined here
    return (aliases or {}).get(dotted(node), dotted(node)) in DECORATORS


def _signature_ok(args, aliases=None):
    annotations = [a.annotation for a in args.posonlyargs + args.args + args.kwonlyargs]
    annotations += [args.vararg and args.vararg.annotation, args.kwarg and args.kwarg.annotation]
    return (all(_static(d, aliases) for d in args.defaults + args.kw_defaults)
            and all(_static(a, aliases) for a in annotations if a is not None))


def _name_ok(name):
    """A name a method file may bind at module or class level: allow-listed dunder metadata, no other
    dunder (never rebind __name__), nothing the allow-lists trust."""
    if name.startswith("__"):
        return name in DUNDER_DATA
    return name not in RESERVED


def _pure_constant(value):
    """A value built only from constants and constant containers/operators: safe to evaluate."""
    if isinstance(value, ast.Constant):
        return True
    if isinstance(value, (ast.Tuple, ast.List, ast.Set)):
        return all(_pure_constant(e) for e in value.elts)
    if isinstance(value, ast.Dict):
        return all(_pure_constant(e) for e in value.keys + value.values if e is not None)
    if isinstance(value, (ast.UnaryOp, ast.BinOp, ast.BoolOp)):
        return all(_pure_constant(c) for c in ast.iter_child_nodes(value)
                   if isinstance(c, (ast.expr,)))
    return False


def _torch_flag(target):
    """``torch.backends.cudnn.benchmark = True``-style targets; they may only receive constants."""
    name = dotted(target)
    return bool(name) and name.startswith("torch.") and name not in PROTECTED


def _name_target(target, value):
    """A simple name target: a RESERVED name may only be set to None (an optional-import fallback)."""
    if target.id in RESERVED:
        return isinstance(value, ast.Constant) and value.value is None
    return _name_ok(target.id)


def _target_ok(target, value):
    if isinstance(target, ast.Name):
        return _name_target(target, value)
    if isinstance(target, (ast.Tuple, ast.List)):
        return all(isinstance(t, ast.Name) and _name_ok(t.id) and t.id not in RESERVED for t in target.elts)
    return _torch_flag(target) and _pure_constant(value)


def _main_guard(node):
    """``if __name__ == "__main__":`` with no else: its body never runs when the worker imports the file."""
    test = node.test
    if node.orelse or not isinstance(test, ast.Compare) or len(test.ops) != 1 or not isinstance(test.ops[0], ast.Eq):
        return False
    sides = [test.left, test.comparators[0]]
    return (any(isinstance(s, ast.Name) and s.id == "__name__" for s in sides)
            and any(isinstance(s, ast.Constant) and s.value == "__main__" for s in sides))


NODE_PROSE = {ast.If: 'an `if` other than `if __name__ == "__main__":`', ast.For: "a loop",
              ast.While: "a loop", ast.With: "a `with` block", ast.AugAssign: "an augmented assignment",
              ast.Assert: "an assert", ast.Delete: "a `del`", ast.Raise: "a `raise`", ast.Return: "a return",
              ast.AsyncFor: "a loop", ast.AsyncWith: "a `with` block"}


def _type_checking(test):
    """``if TYPE_CHECKING:`` / ``if typing.TYPE_CHECKING:`` -- its body still runs under the checker."""
    return dotted(test) in ("TYPE_CHECKING", "typing.TYPE_CHECKING")


def _check_statement(node, where, filename, aliases, setters=frozenset()):
    def reject(what):
        move = ("; move the work inside your function, and put a scoring call such as "
                "print(mnist.score(...)) under `if __name__ == \"__main__\":`"
                if isinstance(node, (ast.Expr, ast.If, ast.For, ast.While, ast.With)) else "")
        raise SourceError(f"{filename} line {node.lineno}: {what} at {where} level. At import a method file may "
                          f"only import, define functions and classes, assign constants and set torch flags{move}")

    if isinstance(node, (ast.Import, ast.ImportFrom)):
        for alias in node.names:
            bound = alias.asname or alias.name.split(".")[0]
            if bound.startswith("__"):
                reject(f"an import that binds the dunder name {bound}")
            if bound in RESERVED:
                real = (alias.name.split(".")[0] == bound if isinstance(node, ast.Import)
                        else FROM_IMPORTS.get(bound) == node.module and alias.name == bound)
                if not real:
                    reject(f"an import that rebinds {bound}, which the scorer's allow-list trusts")
        return
    if isinstance(node, ast.Pass):
        return
    if isinstance(node, ast.Expr):
        if isinstance(node.value, ast.Constant):
            return  # docstring
        if isinstance(node.value, ast.Call) and dotted(node.value.func) in MODULE_CALLS:
            if all(_static(c, aliases) for c in node.value.args + [k.value for k in node.value.keywords]):
                return
        name = dotted(node.value.func) if isinstance(node.value, ast.Call) else None
        reject(f"a call to {name}" if name else "a call")
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        if node.name in RESERVED or (where == "module" and node.name.startswith("__")):
            reject(f"a function named {node.name}, which the scorer's allow-list trusts")
        bad = next((d for d in node.decorator_list if not _decorator_ok(d, aliases, setters)), None)
        if bad is not None:
            raise SourceError(f"{filename} line {getattr(bad, 'lineno', node.lineno)}: the decorator "
                              f"{dotted(bad.func if isinstance(bad, ast.Call) else bad)} is not on the allow-list "
                              f"{sorted(DECORATORS)}")
        if not _signature_ok(node.args, aliases) or not _static(node.returns, aliases):
            reject("a default value or annotation that runs code")
        return
    if isinstance(node, ast.ClassDef):
        if not _name_ok(node.name):
            reject(f"a class named {node.name}, which the scorer's allow-list trusts")
        if node.keywords or not all(_static(b, aliases) for b in node.bases):
            reject("a class with keywords or computed bases")
        if not all(_decorator_ok(d, aliases) for d in node.decorator_list):
            reject("a class decorator that is not on the allow-list")
        # properties defined here, so a later @x.setter is allowed
        props = {m.name for m in node.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))
                 and any(_decorator_ok(d, aliases) and dotted(d) == "property" for d in m.decorator_list)}
        for statement in node.body:
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)) and statement.name in CLASS_HOOKS:
                reject(f"{statement.name}, which runs when the class is subscripted or subclassed,")
            _check_statement(statement, "class", filename, aliases, props)
        return
    if isinstance(node, ast.Assign):
        if all(_target_ok(t, node.value) for t in node.targets) and _static(node.value, aliases):
            return
        target = ", ".join(dotted(t) or "?" for t in node.targets)
        reject(f"an assignment to {target} that runs code or rebinds a protected name")
    if isinstance(node, ast.AnnAssign):
        if _target_ok(node.target, node.value) and _static(node.value, aliases) and _static(node.annotation, aliases):
            return
        reject(f"an assignment to {dotted(node.target)} that runs code or rebinds a protected name")
    if isinstance(node, ast.If) and where == "module" and _main_guard(node):
        return
    if isinstance(node, ast.If) and _type_checking(node.test) and not node.orelse:
        for statement in node.body:  # the body is still fully checked; nothing may run there
            _check_statement(statement, where, filename, aliases, setters)
        return
    if isinstance(node, ast.Try) and where == "module":
        if not all(h.type is None or _static(h.type, aliases) for h in node.handlers):
            reject("an exception handler that runs code")
        if not all(h.name is None or _name_ok(h.name) for h in node.handlers):
            reject("an exception handler that binds a protected name")
        for block in (node.body, node.orelse, node.finalbody, *[h.body for h in node.handlers]):
            for statement in block:
                _check_statement(statement, "module", filename, aliases, setters)
        return
    reject(NODE_PROSE.get(type(node), type(node).__name__))


def _import_aliases(tree):
    """{bound name: dotted origin} for imports whose origin matters to the allow-lists, so
    ``from dataclasses import field as fld`` lets ``fld(...)`` read as ``dataclasses.field``."""
    aliases = {}
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    aliases[alias.asname] = alias.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                aliases[alias.asname or alias.name] = f"{node.module}.{alias.name}"
    return aliases


def check_source(source, function, filename="the method file", max_bytes=MAX_SOURCE_BYTES):
    """Raise SourceError unless ``source`` is small, inert at import and defines ``function``."""
    if len(source) > max_bytes:
        raise SourceError(f"{filename} is {len(source):,} bytes; the limit is {max_bytes:,}, "
                          "so a method cannot carry a trained model or the dataset with it")
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        raise SourceError(f"{filename} does not parse: {error}") from None
    aliases = _import_aliases(tree)
    for node in tree.body:
        _check_statement(node, "module", filename, aliases)
    if not any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == function for n in tree.body):
        raise SourceError(f"{filename} does not define {function}() at the top level")


# Advisory review flags (popcorn3 tools/score_submissions.py): what a human should read before a
# record stands. They reject nothing.
BLOB_BYTES, BLOB_ENTROPY, HEX_FRACTION, NUMERIC_LITERALS = 512, 5.2, 0.95, 400
DECODERS = {"base64", "binascii", "zlib", "lzma", "bz2", "gzip", "pickle", "marshal", "struct", "codecs"}
REACH_OUT = {"socket", "urllib", "http", "requests", "subprocess", "ctypes", "multiprocessing", "shutil"}
DYNAMIC_CALLS = {"eval", "exec", "compile", "__import__", "open", "getattr"}


def review_flags(source):
    """Static hints that a file carries data (a string of 512+ characters at 5.2+ bits per character,
    or almost only hex; prose and code sit at 4.1-4.8, base64 near 6.0), many numeric literals, a
    decoder, a reach outside the process, or dynamic code."""
    tree = ast.parse(source)
    blobs, numbers, imports, calls = [], 0, set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, (str, bytes)) and len(node.value) >= BLOB_BYTES:
            text = "".join((node.value if isinstance(node.value, str) else node.value.decode("latin-1")).split())
            counts = [text.count(c) for c in set(text)]
            bits = -sum(c / len(text) * math.log2(c / len(text)) for c in counts) if text else 0.0
            if bits >= BLOB_ENTROPY or sum(c in "0123456789abcdefABCDEF" for c in text) >= HEX_FRACTION * max(1, len(text)):
                blobs.append(len(node.value))
        elif isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            numbers += 1
        elif isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in DYNAMIC_CALLS:
            calls.add(node.func.id)
    flags = []
    if blobs:
        flags.append(f"embedded data? {len(blobs)} high-entropy literal(s), {sum(blobs):,} characters")
    if numbers >= NUMERIC_LITERALS:
        flags.append(f"embedded weights? {numbers:,} numeric literals")
    for kind, found in (("decodes data", imports & DECODERS), ("reaches outside the process", imports & REACH_OUT)):
        if found:
            flags.append(f"{kind}: imports {', '.join(sorted(found))}")
    if calls:
        flags.append(f"dynamic code or files: calls {', '.join(sorted(calls))}")
    if len(source) > 0.75 * MAX_SOURCE_BYTES:
        flags.append(f"{len(source):,} bytes, close to the {MAX_SOURCE_BYTES:,}-byte limit")
    return flags


# ============================================================================
# Sandbox (measured on a Modal A100-80GB for sutro-mnist-medium/3.0.0: as uid
# 65534 the worker cannot read the parent's memory, environment or file
# descriptors, or root-only files; with the filter it cannot open an IPv4 or
# IPv6 socket, from Python, ctypes or a subprocess; CUDA, torch.compile, Triton
# and load_inline all still work)
# ============================================================================

SANDBOX_UID = SANDBOX_GID = 65534  # "nobody" on Debian/Ubuntu
WORLD_WRITABLE = ("/tmp", "/var/tmp", "/run/lock", "/dev/shm")
_AUDIT_ARCH_X86_64 = 0xC000003E
_LD_W_ABS, _JEQ, _JGE, _RET = 0x20, 0x15, 0x35, 0x06
_ALLOW, _KILL = 0x7FFF0000, 0x80000000
_SYS_SOCKET = 41
# io_uring_setup/enter/register (a second way to open sockets), ptrace, process_vm_readv/writev, bpf,
# and SysV IPC and POSIX message queues (kernel objects that outlive a worker and are not files):
# shmget/shmat/shmctl/shmdt, semget/semop/semctl/semtimedop, msgget/msgsnd/msgrcv/msgctl, mq_*.
_SYS_DENIED = (425, 426, 427, 101, 310, 311, 321, 29, 30, 31, 67, 64, 65, 66, 220, 68, 69, 70, 71,
               240, 241, 242, 243, 244, 245)


def seccomp_program():
    """BPF: x86-64 only; socket() only for AF_UNIX; a few syscalls denied."""
    head = [(_LD_W_ABS, 0, 0, 4), (_JEQ, 1, 0, _AUDIT_ARCH_X86_64), (_RET, 0, 0, _KILL), (_LD_W_ABS, 0, 0, 0)]
    checks = [(_JGE, 0x40000000, "deny"), (_JEQ, _SYS_SOCKET, "socket")] + [(_JEQ, nr, "deny") for nr in _SYS_DENIED]
    allow = len(head) + len(checks)
    socket, deny = allow + 1, allow + 5
    program = list(head)
    for index, (op, value, target) in enumerate(checks, start=len(head)):
        program.append((op, (deny if target == "deny" else socket) - index - 1, 0, value))
    program += [(_RET, 0, 0, _ALLOW),
                (_LD_W_ABS, 0, 0, 16), (_JEQ, 1, 0, 1), (_RET, 0, 0, 0x00050000 | 13), (_RET, 0, 0, _ALLOW),
                (_RET, 0, 0, 0x00050000 | 1)]
    assert len(program) == deny + 1
    return program


def install_seccomp():
    """Install seccomp_program() on every thread of this process, irrevocably."""
    raw = b"".join(struct.pack("HBBI", *instruction) for instruction in seccomp_program())
    buffer = ctypes.create_string_buffer(raw, len(raw))

    class SockFprog(ctypes.Structure):
        _fields_ = [("len", ctypes.c_ushort), ("filter", ctypes.c_void_p)]

    fprog = SockFprog(len(raw) // 8, ctypes.cast(buffer, ctypes.c_void_p))
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(38, 1, 0, 0, 0) != 0:  # PR_SET_NO_NEW_PRIVS
        raise OSError(ctypes.get_errno(), "prctl(PR_SET_NO_NEW_PRIVS) failed")
    if libc.syscall(317, 1, 1, ctypes.byref(fprog)) != 0:  # seccomp(SET_MODE_FILTER, FLAG_TSYNC, &prog)
        raise OSError(ctypes.get_errno(), "seccomp(SECCOMP_SET_MODE_FILTER) failed")


def sandbox_available():
    """Why the worker cannot be sandboxed here, or None when it can."""
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        return f"the sandbox needs Linux x86-64, this is {platform.system()} {platform.machine()}"
    if os.geteuid() != 0:
        return "the sandbox needs root, to start the worker as another user"
    return None


def sandbox_pids(uid=SANDBOX_UID):
    """Every process running as ``uid``."""
    pids = []
    for entry in filter(str.isdigit, os.listdir("/proc")):
        try:
            with open(f"/proc/{entry}/status") as status:
                uids = next(line for line in status if line.startswith("Uid:")).split()[1:]
        except (OSError, StopIteration):
            continue
        if str(uid) in uids:
            pids.append(int(entry))
    return pids


def freeze(worker, stop=True):
    """Stop (SIGSTOP) or continue every process of the method: the worker's session and, sandboxed,
    every process of the sandbox user, so none of its threads runs while idle power is measured."""
    sig = signal.SIGSTOP if stop else signal.SIGCONT
    try:
        os.killpg(worker.proc.pid, sig)
    except OSError:
        pass
    for pid in sandbox_pids() if worker.sandboxed else ():
        try:
            os.kill(pid, sig)
        except OSError:
            pass


def reap_sandbox(uid=SANDBOX_UID):
    """Kill every process running as ``uid`` and delete every file it owns in the
    world-writable directories, so nothing a method leaves behind survives the run."""
    for _ in range(5):
        pids = sandbox_pids(uid)
        if not pids:
            break
        for pid in pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
        time.sleep(0.05)
    for kind, flag in (("shm", "-m"), ("msg", "-q"), ("sem", "-s")):
        try:
            rows = Path(f"/proc/sysvipc/{kind}").read_text().splitlines()[1:]
        except OSError:
            continue
        for row in rows:
            fields = row.split()
            uid_column = {"shm": 7, "msg": 9, "sem": 4}[kind]
            if len(fields) > uid_column and fields[uid_column] == str(uid):
                subprocess.run(["ipcrm", flag, fields[1]], capture_output=True)
    for root in WORLD_WRITABLE:
        for directory, subdirs, files in os.walk(root, topdown=True) if os.path.isdir(root) else ():
            for name in list(subdirs) + files:
                path = os.path.join(directory, name)
                try:
                    if os.lstat(path).st_uid != uid:
                        continue
                except OSError:
                    continue
                if name in subdirs:
                    subdirs.remove(name)
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    try:
                        os.unlink(path)
                    except OSError:
                        pass


# ============================================================================
# The scorer's side of a run
# ============================================================================


class Failure(Exception):
    pass


@dataclasses.dataclass
class Call:
    dataset: str
    ms: float         # CUDA events in the worker: the ranked number
    parent_ms: float  # this process's clock: staging, the call and the reply
    correct: int
    total: int
    overhead_ms: float  # what the same round trip costs with an empty method, in this worker


class Worker:
    """The process that imports and runs the method file."""

    def __init__(self, sandboxed, source):
        self.sandboxed, self.proc, self.buffer, self.dir, self.io_dir, self.log = sandboxed, None, None, None, None, None
        self.to_write = self.from_read = None
        try:
            self._start(source)
        except BaseException:
            self.close()
            raise

    def _start(self, source):
        sandboxed = self.sandboxed
        # Inputs reach the worker through a file both processes map. It is read-only to the
        # worker and only ever holds the current draw, written inside the timed window.
        self.io_dir = Path(tempfile.mkdtemp(prefix="mnist-a100-io-"))
        os.chmod(self.io_dir, 0o711)
        self.buffer_path = self.io_dir / "inputs"
        self.buffer_bytes = TRAIN * (DIMS * 4 + 8) + TEST * DIMS * 4
        with open(self.buffer_path, "wb") as handle:
            handle.truncate(self.buffer_bytes)
        os.chmod(self.buffer_path, 0o644)
        with open(self.buffer_path, "r+b") as handle:
            self.buffer = mmap.mmap(handle.fileno(), self.buffer_bytes)
        self.dir = Path(tempfile.mkdtemp(prefix="mnist-a100-worker-"))
        shutil.copy(Path(__file__).resolve(), self.dir / "mnist.py")
        (self.dir / "submission.py").write_bytes(source)  # the bytes check_source saw
        caches = {"HOME": "", "TMPDIR": "tmp", "XDG_CACHE_HOME": "cache", "TRITON_CACHE_DIR": "cache/triton",
                  "TORCHINDUCTOR_CACHE_DIR": "cache/inductor", "TORCH_EXTENSIONS_DIR": "cache/torch_extensions",
                  "CUDA_CACHE_PATH": "cache/nv"}
        env = {k: v for k, v in os.environ.items() if not k.startswith(("MNIST_", "SUTRO_", "POPCORN_"))}
        for key, sub in caches.items():
            (self.dir / sub).mkdir(parents=True, exist_ok=True)
            env[key] = str(self.dir / sub)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        extra = {}
        if sandboxed:
            for path in [self.dir, *self.dir.rglob("*")]:
                os.chown(path, SANDBOX_UID, SANDBOX_GID)
            extra = dict(user=SANDBOX_UID, group=SANDBOX_GID, extra_groups=[])
        os.chmod(self.dir, 0o700)
        to_read, self.to_write = os.pipe()
        self.from_read, from_write = os.pipe()
        # -u: the method's prints reach stderr even though the worker is killed, not exited.
        # -E -s: no PYTHON* variables and no user site-packages for the sandboxed interpreter.
        flags = ["-u", "-E", "-s"] if sandboxed else ["-u"]
        command = [sys.executable, *flags, str(self.dir / "mnist.py"), WORKER_FLAG,
                   str(to_read), str(from_write), "1" if sandboxed else "0"]
        # The method's prints go to a file of this worker's own, copied to the scorer's stderr when the
        # worker is closed: never to the scorer's stdout, and never into a file a later worker can see.
        self.log = tempfile.TemporaryFile()
        try:
            self.proc = subprocess.Popen(command, pass_fds=(to_read, from_write), cwd=self.dir, env=env,
                                         stdin=subprocess.DEVNULL, stdout=self.log, stderr=self.log,
                                         start_new_session=True, **extra)
        finally:
            os.close(to_read)
            os.close(from_write)
        os.set_blocking(self.to_write, False)
        os.set_blocking(self.from_read, False)

    def _died(self, what):
        return f"the method's process exited ({self.proc.poll()}) while {what}"

    def _write(self, data, deadline, what):
        view = memoryview(data).cast("B")
        while view:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise Failure(f"timed out while {what}")
            if not select.select([], [self.to_write], [], remaining)[1]:
                continue
            try:
                view = view[os.write(self.to_write, view[: 1 << 20]):]
            except BlockingIOError:
                continue
            except BrokenPipeError:
                raise Failure(self._died(what)) from None

    def _read(self, count, deadline, what):
        buffer = bytearray(count)
        view, got = memoryview(buffer), 0
        while got < count:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise Failure(f"timed out while {what}")
            if not select.select([self.from_read], [], [], remaining)[0]:
                continue
            try:
                n = os.readv(self.from_read, [view[got:]])
            except BlockingIOError:
                continue
            if n == 0:
                raise Failure(self._died(what))
            got += n
        return buffer

    def request(self, header, timeout_s=60.0, max_reply=1 << 20, what="waiting for the method"):
        deadline = time.monotonic() + timeout_s
        body = json.dumps(header).encode()
        self._write(struct.pack("<II", len(body), 0) + body, deadline, what)
        head_len, payload_len = struct.unpack("<II", self._read(8, deadline, what))
        if head_len > MAX_HEADER or payload_len > max_reply:
            raise Failure(f"the method's process sent a malformed reply while {what}")
        try:
            reply = json.loads(bytes(self._read(head_len, deadline, what)))
        except ValueError:
            raise Failure(f"the method's process sent a malformed reply while {what}") from None
        payload = self._read(payload_len, deadline, what)
        if not isinstance(reply, dict):
            raise Failure(f"the method's process sent a malformed reply while {what}")
        return reply, payload

    def call(self, d, timeout_s, null=False, label="a", collect=True):
        """One timed call: (CUDA-event ms, this process's ms, predicted labels as uint8). Energy
        windows pass ``collect=False``: their calls run back to back, without collecting garbage."""
        import numpy as np

        arrays = (d["train_x"], d["train_y"].astype(np.int64), d["test_x"])
        what = f"running the method on {label} {d['dataset']} draw"
        if collect:
            self.request({"cmd": "gc"}, STAGE_S, what="collecting garbage")
            gc.collect()
        # The timed window opens before the draw exists anywhere the worker can see it and
        # closes when the predictions are back: copying the inputs in, the call, the reply.
        started = time.perf_counter()
        offset = 0
        for array in arrays:
            self.buffer[offset:offset + array.nbytes] = memoryview(array).cast("B")
            offset += array.nbytes
        reply, preds = self.request({"cmd": "go", "null": null}, timeout_s, max_reply=TEST, what=what)
        parent_ms = (time.perf_counter() - started) * 1e3
        if not reply.get("ok"):
            raise Failure(f"the method failed on {label} {d['dataset']} draw:\n{reply.get('error')}")
        ms = reply.get("ms")
        if not isinstance(ms, (int, float)) or not math.isfinite(ms) or ms < 0 or len(preds) != TEST:
            raise Failure("the method's process sent a malformed result")
        return float(ms), parent_ms, np.frombuffer(bytes(preds), dtype=np.uint8)

    def close(self):
        if self.proc is not None:
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)
            except OSError:
                pass
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
        for fd in (self.to_write, self.from_read):
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
        if self.sandboxed:
            reap_sandbox()
        if self.log is not None:
            self.log.seek(0)
            text = self.log.read()[-20000:].decode("utf-8", "replace")
            if text.strip():
                sys.stderr.write(text if text.endswith("\n") else text + "\n")
                sys.stderr.flush()
            self.log.close()
        if self.buffer is not None:
            self.buffer.close()
        for directory in (self.dir, self.io_dir):
            if directory is not None:
                shutil.rmtree(directory, ignore_errors=True)


def _median(values):
    ordered = sorted(values)
    middle = len(ordered) // 2
    return ordered[middle] if len(ordered) % 2 else 0.5 * (ordered[middle - 1] + ordered[middle])


def required_correct(total, error_bp):
    """ceil(total * (10000 - error_bp) / 10000), in integers."""
    return -(-(total * (10000 - error_bp)) // 10000)


def calibrate(worker):
    """What the protocol costs with a method that does nothing, before the method is imported:
    copying the inputs in, the L2 flush, the events, the reply."""
    import numpy as np

    d = {"dataset": "calibration", "train_x": np.zeros((TRAIN, DIMS), np.float32),
         "train_y": np.zeros(TRAIN, np.int64), "test_x": np.zeros((TEST, DIMS), np.float32)}
    samples = []
    for index in range(CALIBRATION_CALLS):
        ms, parent_ms, _ = worker.call(d, 60, null=True)
        if index >= CALIBRATION_DISCARD:
            samples.append(parent_ms - ms)
    return _median(samples)


def judge(calls, band):
    """Every rule a run must pass: (problems, ranked ms, one-line summary)."""
    problems = []
    mnist = [c for c in calls if c.dataset == "mnist"]
    hold = [c for c in calls if c.dataset != "mnist"]

    def unexplained(c):
        return c.parent_ms - c.overhead_ms - c.ms

    def kept(group):  # all but the calls with the most unexplained time (a fifth, at least one)
        return sorted(group, key=unexplained)[: len(group) - max(1, len(group) // 5)] if len(group) >= 3 else group

    def mean(values):
        values = list(values)
        return sum(values) / len(values)

    for index, c in enumerate(calls):
        if unexplained(c) > CALL_SLACK_MS + CALL_SLACK_FRACTION * c.ms or c.ms > c.parent_ms + 1.0:
            problems.append(f"call {index + 1} ({c.dataset}): the method reported {c.ms:.3f} ms but the scorer "
                            f"measured {c.parent_ms - c.overhead_ms:.3f} ms beyond the protocol's own cost")
    for group, name in ((mnist, "MNIST"), (hold, "hold-out")):
        if len(group) >= 3:
            reported, measured = mean(c.ms for c in kept(group)), mean(c.parent_ms - c.overhead_ms for c in kept(group))
            if measured - reported > GROUP_SLACK_MS + GROUP_SLACK_FRACTION * reported:
                problems.append(f"{name} calls: the method reported {reported:.3f} ms on average but the scorer "
                                f"measured {measured:.3f} ms beyond the protocol's own cost")
    correct, total = sum(c.correct for c in mnist), sum(c.total for c in mnist)
    need = required_correct(total, band)
    if correct < need:
        problems.append(f"MNIST accuracy {correct / total:.3%} ({correct:,}/{total:,}); the band needs {need:,} correct")
    per_draw_floor = required_correct(TEST, band + DRAW_SLACK_BP)
    weakest = min(c.correct for c in mnist)
    if weakest < per_draw_floor and len(mnist) > 1:
        problems.append(f"one MNIST draw scored {weakest:,}/{TEST:,}; no draw may fall below {per_draw_floor:,}")
    hold_correct, hold_total = sum(c.correct for c in hold), sum(c.total for c in hold)
    if hold and hold_correct < required_correct(hold_total, HOLDOUT_ERROR_BP):
        problems.append(f"hold-out accuracy {hold_correct / hold_total:.2%} ({hold_correct:,}/{hold_total:,}); at least "
                        f"{required_correct(hold_total, HOLDOUT_ERROR_BP):,} needed. The method does not appear to "
                        "learn from the data it is given")
    for group, name in ((mnist, "MNIST"), (hold, "hold-out")):
        if len(group) >= 3:
            times = [c.ms for c in group]
            median = _median(times)
            if max(times) > DISPERSION * median + 2.0:
                problems.append(f"the slowest {name} call took {max(times):.3f} ms, more than {DISPERSION:g}x the "
                                f"median {median:.3f} ms; every call must do the same work")
            if min(times) < median / DISPERSION - 2.0:
                problems.append(f"the fastest {name} call took {min(times):.3f} ms, less than 1/{DISPERSION:g} of the "
                                f"median {median:.3f} ms; every call must do the same work")

    def group_ms(group):
        """The mean reported time, floored by this process's clock: hiding work never lowers it."""
        reported = mean(c.ms for c in group)
        floor = mean(c.parent_ms - c.overhead_ms for c in group) - FLOOR_SLACK_MS - FLOOR_SLACK_FRACTION * reported
        return max(reported, floor), floor > reported

    mnist_ms, mnist_floored = group_ms(mnist)
    hold_ms, hold_floored = group_ms(hold) if hold else (0.0, False)
    summary = f"MNIST {correct / total:.2%} ({correct:,}/{total:,}), {mnist_ms:.3f} ms/call"
    if hold:
        summary += (f"; hold-out ({'+'.join(sorted({c.dataset for c in hold}))}) {hold_correct / hold_total:.2%}, "
                    f"{hold_ms:.3f} ms/call")
    if mnist_floored or hold_floored:
        summary += " (floored by the scorer's clock)"
    return problems, max(mnist_ms, hold_ms), summary


# ============================================================================
# The energy column: this process reads the board's NVML counter, the method
# never does
# ============================================================================


class NvmlError(Exception):
    pass


class EnergyUnavailable(Exception):
    pass


class Nvml:
    """The board's cumulative energy counter and a few checks, through the driver's NVML library
    (loaded with ctypes; the image has no pynvml). A read of the counter takes ~3 ms, and a tight
    loop of them raised idle power by ~20 W on an A100, so it is read only at window edges."""

    def __init__(self):
        try:
            self.lib = ctypes.CDLL("libnvidia-ml.so.1")
        except OSError:
            raise NvmlError("no NVML here (libnvidia-ml.so.1, from an NVIDIA driver on Linux)") from None
        self.lib.nvmlErrorString.restype = ctypes.c_char_p
        self._check(self.lib.nvmlInit_v2(), "nvmlInit_v2")
        self.handle = None

    def _check(self, code, name):
        if code:
            raise NvmlError(f"{name} failed: {self.lib.nvmlErrorString(code).decode()}")

    def select(self, uuid):
        """The board that is the worker's CUDA device, by UUID; without one, the only board there is."""
        handle, count = ctypes.c_void_p(), ctypes.c_uint()
        if uuid and self.lib.nvmlDeviceGetHandleByUUID(uuid.encode(), ctypes.byref(handle)) == 0:
            self.handle = handle
            return
        self._check(self.lib.nvmlDeviceGetCount_v2(ctypes.byref(count)), "nvmlDeviceGetCount_v2")
        if count.value != 1:
            raise NvmlError(f"cannot tell which of {count.value} GPUs the method runs on")
        self._check(self.lib.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(handle)), "nvmlDeviceGetHandleByIndex_v2")
        self.handle = handle

    def energy_mj(self):
        value = ctypes.c_ulonglong()
        self._check(self.lib.nvmlDeviceGetTotalEnergyConsumption(self.handle, ctypes.byref(value)),
                    "nvmlDeviceGetTotalEnergyConsumption")
        return value.value

    def utilization(self):
        """Percent of the last sample period (1/6 s to 1 s) in which a kernel ran."""
        rates = (ctypes.c_uint * 2)()
        self._check(self.lib.nvmlDeviceGetUtilizationRates(self.handle, rates), "nvmlDeviceGetUtilizationRates")
        return rates[0]

    def contexts(self):
        """How many processes hold a compute context on the board, or None when it cannot say."""
        function = (getattr(self.lib, "nvmlDeviceGetComputeRunningProcesses_v3", None)
                    or getattr(self.lib, "nvmlDeviceGetComputeRunningProcesses_v2", None))
        if function is None:
            return None
        count = ctypes.c_uint(0)
        code = function(self.handle, ctypes.byref(count), None)
        return 0 if code == 0 else count.value if code == 7 else None  # 7: NVML_ERROR_INSUFFICIENT_SIZE

    def describe(self):
        def text(name, size, *args):
            buffer = ctypes.create_string_buffer(size)
            return buffer.value.decode() if getattr(self.lib, name)(*args, buffer, size) == 0 else None

        limit = ctypes.c_uint()
        known = self.lib.nvmlDeviceGetPowerManagementLimit(self.handle, ctypes.byref(limit)) == 0
        return {"name": text("nvmlDeviceGetName", 96, self.handle), "uuid": text("nvmlDeviceGetUUID", 96, self.handle),
                "vbios": text("nvmlDeviceGetVbiosVersion", 32, self.handle),
                "driver": text("nvmlSystemGetDriverVersion", 80), "power_limit_w": limit.value / 1e3 if known else None}

    def close(self):
        self.lib.nvmlShutdown()


def energy_column(source, function, sandboxed, pools, rng, warmup, band, mnist_ms, say):
    """Measure the energy column after a passing run and print it. Returns energy_summary()'s record,
    or one with ``mj_per_call`` None and the ``reason`` it could not be measured."""
    say(f"energy: one more fresh process runs the method back to back on fresh MNIST draws for "
        f"{ENERGY_WINDOW_S:g} s, between idle windows in which it is frozen", flush=True)
    try:
        windows, device = measure_energy(source, function, sandboxed, pools, rng, warmup, mnist_ms)
    except (EnergyUnavailable, NvmlError, Failure, RuntimeError, OSError) as error:
        say(f"energy not measured: {error}", flush=True)
        return {"mj_per_call": None, "reason": str(error)}
    report = energy_summary(windows, device, band, mnist_ms)
    ref = report["reference"]
    check = (f"a healthy {device.get('name')} reads {ref['band'][0][0]:g}-{ref['band'][0][1]:g}"
             if ref["band"] else "unchecked on this GPU")
    say(f"  {device.get('name')}: idle {report['idle_w']:.1f} W with the method frozen; telemetry reference "
        f"{ref['tflops_per_s']:.1f} TFLOP/s at {ref['j_per_tflop']:.2f} J/TFLOP above idle ({check})", flush=True)
    say(f"  the round trip alone: {report['control_mj_per_call']:,.1f} mJ per call over "
        f"{report['control_calls']:,} empty calls, subtracted", flush=True)
    say(f"  the method: {report['calls']:,} calls in {report['seconds']:.1f} s, MNIST "
        f"{report['correct'] / report['total']:.2%}, {report['ms_median']:,.3f} ms per call, "
        f"{report['active_w']:.1f} W on average", flush=True)
    for problem in report["problems"]:
        say(f"energy not measured: {problem}", flush=True)
    if report["mj_per_call"] is not None:
        say(f"energy {report['mj_per_call']:.3f} mJ per call above idle", flush=True)
    return report


def measure_energy(source, function, sandboxed, pools, rng, warmup, mnist_ms):
    """One fresh worker and seven windows: idle, reference, idle, control, [import and warm-up],
    idle, method, idle. Each window records its seconds and the joules the counter moved across it.
    The reference and the control run before the method is imported."""
    import numpy as np

    board = Nvml()  # before any worker starts, so a machine without NVML says so at once
    worker, windows = None, []
    try:
        if sandboxed:
            reap_sandbox()
        worker = Worker(sandboxed, source)
        hello, _ = worker.request({"cmd": "hello", "buffer": str(worker.buffer_path),
                                   "buffer_bytes": worker.buffer_bytes}, timeout_s=WORKER_START_S, what="starting")
        if sandboxed and not hello.get("sandboxed"):
            raise EnergyUnavailable(f"could not sandbox the method: {hello.get('sandbox_error')}")
        board.select(hello.get("uuid"))
        device = board.describe()

        def measure(name, run):
            t0, e0 = time.perf_counter(), board.energy_mj()
            detail = run()
            t1, e1 = time.perf_counter(), board.energy_mj()
            windows.append({"name": name, "seconds": t1 - t0, "joules": (e1 - e0) / 1e3, **detail})

        def idle(name):
            freeze(worker)
            try:
                time.sleep(SETTLE_S)

                def wait():
                    busy, end = [], time.perf_counter() + IDLE_S
                    while time.perf_counter() < end:
                        time.sleep(max(0.0, min(0.25, end - time.perf_counter())))
                        busy.append(board.utilization())
                    return {"utilization_max": max(busy, default=0), "contexts": board.contexts()}

                measure(name, wait)
            finally:
                freeze(worker, stop=False)

        def reference():
            reply, _ = worker.request({"cmd": "reference", "seconds": REFERENCE_S, "dim": REFERENCE_DIM},
                                      timeout_s=REFERENCE_S + 120, what="running the telemetry reference")
            if not reply.get("ok"):
                raise EnergyUnavailable(f"the telemetry reference failed:\n{reply.get('error')}")
            return {"matmuls": reply["count"], "dim": REFERENCE_DIM, "exact": reply["exact"]}

        empty = {"dataset": "control", "train_x": np.zeros((TRAIN, DIMS), np.float32),
                 "train_y": np.zeros(TRAIN, np.int64), "test_x": np.zeros((TEST, DIMS), np.float32)}

        def control():
            calls, end = 0, time.perf_counter() + CONTROL_WINDOW_S
            while calls < ENERGY_MIN_CALLS or time.perf_counter() < end:
                worker.call(empty, 60, null=True, collect=False)
                calls += 1
            return {"calls": calls}

        idle("idle 1")
        measure("reference", reference)
        idle("idle 2")
        measure("control", control)
        reply, _ = worker.request({"cmd": "load", "function": function}, timeout_s=IMPORT_S,
                                  what="importing the method file")
        if not reply.get("ok"):
            raise EnergyUnavailable(f"importing the method file failed:\n{reply.get('error')}")
        worker.call(pools[warmup].draw(rng), LATER_WARMUP_MAX_CALL_MS / 1e3, label="the untimed warm-up on a")
        # As many fresh draws as the window should need, up to ENERGY_DRAWS, made before it starts.
        wanted = ENERGY_MIN_CALLS + math.ceil(ENERGY_WINDOW_S * 1e3 / max(mnist_ms, 1.0))
        draws = [pools["mnist"].draw(rng) for _ in range(min(ENERGY_DRAWS, wanted))]

        def method():
            ms, correct = [], []
            end = time.perf_counter() + ENERGY_WINDOW_S
            while len(ms) < ENERGY_MIN_CALLS or time.perf_counter() < end:
                index = len(ms)
                d = draws[index] if index < len(draws) else rerelease(draws[index % len(draws)], rng)
                call_ms, _, preds = worker.call(d, MAX_CALL_MS / 1e3 + 5, collect=False)
                ms.append(call_ms)
                correct.append(int(np.count_nonzero(preds.astype(np.int64) == d["test_y"])))
            return {"calls": len(ms), "ms": ms, "correct": correct}

        idle("idle 3")
        measure("method", method)
        idle("idle 4")
        return windows, device
    finally:
        if worker is not None:
            worker.close()
        board.close()


def energy_summary(windows, device, band, mnist_ms):
    """The column from the windows: the method's energy per call above the idle power measured
    around its window, less the empty round trip's, with every reason not to trust it in
    ``problems`` (then ``mj_per_call`` is None)."""
    w = {x["name"]: x for x in windows}

    def watts(name):
        return w[name]["joules"] / w[name]["seconds"]

    def net_j(name, before, after):
        return w[name]["joules"] - (watts(before) + watts(after)) / 2 * w[name]["seconds"]

    ref, empty, run = w["reference"], w["control"], w["method"]
    flops = 2 * ref["dim"] ** 3 * ref["matmuls"]
    j_per_tflop, tflops = net_j("reference", "idle 1", "idle 2") / (flops / 1e12), flops / 1e12 / ref["seconds"]
    control_mj = net_j("control", "idle 2", "idle 3") * 1e3 / empty["calls"]
    net_mj = net_j("method", "idle 3", "idle 4") * 1e3 / run["calls"]
    median_ms = _median(run["ms"])
    bands = next((b for prefix, b in REFERENCE_BANDS.items() if str(device.get("name")).startswith(prefix)), None)
    idles = [x for x in windows if x["name"].startswith("idle")]
    floor = required_correct(TEST, band + DRAW_SLACK_BP)
    problems = []
    if any(x["seconds"] <= 0 or x["joules"] < 0 for x in windows):
        problems.append("the board's energy counter went backwards")
    if not ref["exact"]:
        problems.append("the telemetry reference computed a wrong product")
    if bands and not (bands[0][0] <= j_per_tflop <= bands[0][1] and bands[1][0] <= tflops <= bands[1][1]):
        problems.append(f"the board's power telemetry is implausible: the reference read {j_per_tflop:.2f} J/TFLOP "
                        f"above idle at {tflops:.1f} TFLOP/s, where a healthy {device.get('name')} reads "
                        f"{bands[0][0]:g}-{bands[0][1]:g} J/TFLOP at {bands[1][0]:g}-{bands[1][1]:g} TFLOP/s")
    if any(x["utilization_max"] for x in idles):
        problems.append("the GPU was busy in an idle window, with the method's processes frozen")
    if any((x["contexts"] or 0) > 1 for x in idles):
        problems.append("another process held a CUDA context on the GPU")
    if min(run["correct"]) < floor:
        problems.append(f"a draw in the energy window scored {min(run['correct']):,}/{TEST:,}; "
                        f"no timed MNIST draw may fall below {floor:,}")
    if not mnist_ms / DISPERSION - 2.0 <= median_ms <= DISPERSION * mnist_ms + 2.0:
        problems.append(f"calls in the energy window took {median_ms:,.3f} ms (median) against {mnist_ms:,.3f} ms "
                        "in the timed calls; every call must do the same work")
    if net_mj - control_mj <= 0:
        problems.append("the method's window read no energy above idle")
    return {"mj_per_call": None if problems else net_mj - control_mj, "problems": problems,
            "net_mj_before_control": net_mj, "control_mj_per_call": control_mj, "control_calls": empty["calls"],
            "gross_mj_per_call": run["joules"] * 1e3 / run["calls"], "idle_w": (watts("idle 3") + watts("idle 4")) / 2,
            "active_w": watts("method"), "calls": run["calls"], "seconds": run["seconds"], "ms_median": median_ms,
            "timed_mnist_ms_median": mnist_ms, "correct": sum(run["correct"]), "total": run["calls"] * TEST,
            "reference": {"tflops_per_s": tflops, "j_per_tflop": j_per_tflop, "exact": ref["exact"], "band": bands},
            "device": device, "windows": windows}


# ============================================================================
# The worker's side: runs as the sandbox user, imports the method on request
# ============================================================================


def worker_main(read_fd, write_fd, sandboxed):
    status = {}
    if sandboxed:
        try:
            install_seccomp()  # first, before torch starts threads or the method runs
            status["sandboxed"] = True
        except Exception as error:  # noqa: BLE001 - the scorer refuses to continue
            status["sandbox_error"] = repr(error)

    def read_exact(count):
        buffer = bytearray(count)
        view, got = memoryview(buffer), 0
        while got < count:
            n = os.readv(read_fd, [view[got:]])
            if n == 0:
                sys.exit(0)
            got += n
        return buffer

    def reply(header, payload=b""):
        body = json.dumps(header).encode()
        data = memoryview(struct.pack("<II", len(body), len(payload)) + body + payload)
        while data:
            data = data[os.write(write_fd, data):]

    import numpy as np
    import torch

    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    # Captured before the method is imported. This does not make the CUDA-event number trustworthy
    # (the method shares this process: it can patch torch.cuda.Event or reach this frame), which is
    # why judge() floors every dataset's time with the scorer's own clock.
    Event, synchronize, perf_counter = torch.cuda.Event, torch.cuda.synchronize, time.perf_counter
    sync = synchronize if cuda else (lambda: None)
    flush = torch.empty(L2_FLUSH_BYTES // 4, dtype=torch.int32, device=device) if cuda else None
    integer = (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)
    inputs, method, buffer = None, None, None

    def null_method(train_x, train_y, test_x):
        return torch.zeros(test_x.shape[0], dtype=torch.int64, device=test_x.device)

    def check_output(out, test_x):
        if type(out) is not torch.Tensor:
            return f"the method returned {type(out).__name__}, not a plain torch.Tensor"
        if tuple(out.shape) != (test_x.shape[0],):
            return f"the method returned shape {tuple(out.shape)}, expected ({test_x.shape[0]},)"
        if out.dtype not in integer:
            return f"the method returned dtype {out.dtype}, expected an integer dtype"
        if out.device != test_x.device:
            return f"the method returned a tensor on {out.device}, expected {test_x.device}"
        return None

    while True:
        head_len, payload_len = struct.unpack("<II", read_exact(8))
        header = json.loads(bytes(read_exact(head_len)))
        read_exact(payload_len)
        command = header.get("cmd")
        try:
            if command == "hello":
                with open(header["buffer"], "rb") as handle:
                    buffer = mmap.mmap(handle.fileno(), header["buffer_bytes"], prot=mmap.PROT_READ)
                uuid = getattr(torch.cuda.get_device_properties(0), "uuid", None) if cuda else None
                reply({"ok": True, "device": torch.cuda.get_device_name() if cuda else "cpu (not an A100 time)",
                       "torch": torch.__version__, "uuid": f"GPU-{uuid}" if uuid is not None else None, **status})
            elif command == "reference":
                # The telemetry reference, only before the method is imported: FP32 matmuls of constant
                # operands (every product entry is exactly 1), TF32 off, for about header["seconds"].
                if method is not None:
                    raise RuntimeError("the reference runs only before the method is imported")
                dim = int(header["dim"])
                value = 2.0 ** -((dim.bit_length() - 1) // 2)  # dim a power of 4
                tf32, torch.backends.cuda.matmul.allow_tf32 = torch.backends.cuda.matmul.allow_tf32, False
                a = torch.full((dim, dim), value, dtype=torch.float32, device=device)
                c = torch.mm(a, a)
                sync()
                count, end = 0, perf_counter() + float(header["seconds"])
                while count == 0 or perf_counter() < end:
                    for _ in range(8):
                        torch.mm(a, a, out=c)
                    sync()
                    count += 8
                exact = bool((c == 1).all())
                del a, c
                torch.backends.cuda.matmul.allow_tf32 = tf32
                if cuda:
                    torch.cuda.empty_cache()
                reply({"ok": True, "count": count, "exact": exact})
            elif command == "load":
                import submission

                method = getattr(submission, header["function"])
                if not callable(method):
                    raise TypeError(f"{header['function']} is not callable")
                reply({"ok": True})
            elif command == "gc":
                gc.collect()
                reply({"ok": True})
            elif command == "go":
                a, b = TRAIN * DIMS * 4, TRAIN * 8
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")  # the mapping is read-only; we only copy out of it
                    sources = (torch.from_numpy(np.frombuffer(buffer, np.float32, TRAIN * DIMS, 0).reshape(TRAIN, DIMS)),
                               torch.from_numpy(np.frombuffer(buffer, np.int64, TRAIN, a)),
                               torch.from_numpy(np.frombuffer(buffer, np.float32, TEST * DIMS, a + b).reshape(TEST, DIMS)))
                if inputs is None:  # the same three tensors on every call, so CUDA graphs work
                    inputs = tuple(torch.empty(t.shape, dtype=t.dtype, device=device) for t in sources)
                for target, source in zip(inputs, sources):
                    target.copy_(source)
                del sources
                fn = null_method if header.get("null") else method
                sync()
                if flush is not None:
                    flush.zero_()
                sync()
                if cuda:
                    start, end = Event(enable_timing=True), Event(enable_timing=True)
                started = perf_counter()
                if cuda:
                    start.record()
                out = fn(*inputs)
                if cuda:
                    end.record()
                sync()
                wall_ms = (perf_counter() - started) * 1e3
                ms = start.elapsed_time(end) if cuda else wall_ms
                problem = check_output(out, inputs[2])
                if problem:
                    reply({"ok": False, "error": problem})
                    continue
                wide = out.to(torch.int64)
                labels = torch.where((wide >= 0) & (wide <= 9), wide, 255).to(torch.uint8).cpu().numpy()
                reply({"ok": True, "ms": ms}, labels.tobytes())
            else:
                reply({"ok": False, "error": f"unknown command {command!r}"})
        except BaseException:  # noqa: BLE001 - report the method's errors to the scorer
            reply({"ok": False, "error": traceback.format_exc(limit=6)[-4000:]})


# ============================================================================
# Command line
# ============================================================================


def main(argv=None):
    import argparse

    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == WORKER_FLAG:
        return worker_main(int(argv[1]), int(argv[2]), argv[3] == "1")
    parser = argparse.ArgumentParser(description="Score a method file: python mnist.py my_method.py[:function] --difficulty 1")
    parser.add_argument("method", nargs="?", help="file.py or file.py:function")
    parser.add_argument("--difficulty", type=int, default=1, choices=sorted(DIFFICULTY))
    parser.add_argument("--band", type=int, help="test a band in basis points instead of a difficulty (2000 = 20%%)")
    parser.add_argument("--download", action="store_true", help="fetch and verify the three datasets, then exit")
    parser.add_argument("--no-energy", action="store_true", help="skip the energy column (about 80 s after a pass)")
    parser.add_argument("--json", type=Path, metavar="PATH", help="also write every call and energy window to PATH")
    args = parser.parse_args(argv)
    if args.download:
        for name in DATASETS:
            load_pool(name)
        print(f"datasets ready in {cache_dir()}")
        return 0
    if not args.method:
        parser.error("name a method file")
    not_dumpable()
    record = {}
    try:
        path, function = locate(args.method)
        if args.band is not None and not 0 < args.band < 10000:
            parser.error("--band must be between 1 and 9999 basis points")
        evaluate(path, function, band_bp(args.difficulty) if args.band is None else args.band,
                 energy=False if args.no_energy else None, record=record)
    except Disqualified:
        return 1
    except (OSError, ValueError, TypeError, RuntimeError, SyntaxError) as error:
        print(f"error: {error}", flush=True)
        return 2
    finally:
        if args.json and record:
            args.json.write_text(json.dumps(record, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
