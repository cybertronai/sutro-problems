"""Data/protocol layer for the release-cutoffs study (gpumode harness 1.2.0).

A copy of mnist/experiments/release-ladder-20260924/study.py with nine levels
from 100 to 10,000 labels in place of five from 500; the draw protocol, the
seeds and the release are unchanged, so a level both studies share gets
bit-identical arrays.

THE TASK BEING MEASURED
-----------------------
The competition-candidate release of the MNIST-medium time leaderboard.  A
learner is handed, for one dataset seed and one training level ``N``:

  * ``train_z``  float32 ``(N, 60)``      ``= Q W (x - mu)`` for its training images
  * ``train_y``  uint8   ``(N,)``         labels 0-9
  * ``query_z``  float32 ``(10000, 60)``  the same map applied to 10,000 query images

``mu`` is the mean of that level's ``N`` training rows, ``W`` is exact PCA
whitening onto their top 60 principal directions (no variance floor) and ``Q``
is a secret per-draw Haar rotation.  Pixels, pixel coordinates, the image
lattice and the map itself are unavailable by design: the released coordinates
are an arbitrary orthogonal basis of a whitened 60-dimensional subspace.

This module owns every dataset decision.  It is imported by ``plan.py``,
``runner.py`` and ``score.py``.  It NEVER hands out query labels except through
:func:`query_labels`, which is scoring-only: it refuses when a learner module is
loaded in the process AND when the process entry point is the runner, the
planner or a learner script (the runner executes as ``__main__``, so the module
check alone cannot see it).  This is defence in depth, not a sandbox -- the
structural guarantee is that no learner process ever holds a query label: the
container is shipped three arrays and nothing else.

No Modal calls, no GPU, no network access beyond the one-time MNIST gz download
that ``harness_mnist_data`` performs into ``MNIST_POOL_CACHE``.

The released arrays are a function of the controller's LAPACK, not only of the
seed: ``release_map`` diagonalises the covariance, and ``W`` changes in the last
bits with the BLAS thread count, the BLAS build and the CPU.  Every released
array therefore comes from ONE process -- this module -- and is shipped to the
learner rather than recomputed next to it.  Run planning, dispatch and scoring
with the same interpreter and the same OPENBLAS_NUM_THREADS/OMP_NUM_THREADS, or
``score.verify_record``'s regeneration check may refuse a good result.

Draw protocol (implemented exactly here, and re-implemented from the same
``release.py`` primitives inside the Modal container by ``runner.py``)::

    pool                      = harness_mnist_data.load_pool(cache, 'mnist', 9)
    train_universe, test_universe = release.split_universes(60000, seed, UNIVERSE_SALT)
    train_order               = default_rng([seed, DRAW_SALT]).permutation(train_universe)
    train_rows(n)             = train_order[:n]                      # nested prefixes
    query_rows                = default_rng([seed, DRAW_SALT + 1]).choice(
                                    test_universe, 10000, replace=False)
    mu, W, Q                  = release.release_map(pixels[train_rows(n)], seed, 60)
    A                         = Q @ W
    train_z                   = release.apply_release(pixels[train_rows(n)], mu, A)
    query_z                   = release.apply_release(pixels[query_rows],    mu, A)

``Q`` is identical across the levels of one seed (it depends only on the seed);
``mu`` and ``W`` are refitted on each level's ``N`` rows, exactly as the harness
would at ``train=N``.  No label permutation is applied: the harness's per-draw
relabelling (``LABEL_SALT``) is a bijection of the ten classes applied to the
training labels a learner sees and to the truth it is scored against, so query
accuracy is invariant to it.  Omitting it keeps the scorer simple and changes no
measured number.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import sys
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:  # importable from any working directory
    sys.path.insert(0, str(ROOT))

import release  # noqa: E402  (needs ROOT on sys.path)

# ---------------------------------------------------------------- protocol ---
STUDY = "release-cutoffs-20260925-full-batches"
HARNESS_VERSION = "gpumode harness 1.2.0"

DEV_SEEDS = [2026092491, 2026092492]
FINAL_SEEDS = list(range(2026092401, 2026092412))   # 2026092401 .. 2026092411 (11)
LEVELS = [100, 178, 316, 562, 1000, 1778, 3162, 5623, 10000]   # round(100 * 10**(i/4)), i=0..8
LEVELS_EXACT = [100.0 * 10.0 ** (i / 4.0) for i in range(9)]
QUERY_COUNT = 10000
RELEASE_DIMS = 60

POOL_COUNT = 60000
IMAGE_SIZE = 9
PIXEL_COUNT = IMAGE_SIZE * IMAGE_SIZE               # 81
UNIVERSE_HALF = POOL_COUNT // 2                     # 30000 train-universe rows
DEFAULT_TIME_BUDGET_SECONDS = 1200
DEFAULT_LEARNER_SEED = 11

POOL_CACHE = Path(os.environ.get("MNIST_POOL_CACHE", "/tmp/gpumode-pool"))
SOURCE_FILES = {
    "train_images": ("train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
    "train_labels": ("train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
}

RAW_DIR = ROOT / "raw"
POOL_PIXELS_PATH = RAW_DIR / "pool_pixels.npy"
POOL_LABELS_PATH = RAW_DIR / "pool_labels.npy"
DATA_MANIFEST_PATH = RAW_DIR / "data_manifest.json"
PROTOCOL_DRAFT_PATH = ROOT / "protocol.draft.json"

PIXEL_DESCRIPTION = (
    "harness_mnist_data.load_pool(cache,'mnist',9): uint8 -> float32/255; exact "
    "separable box-area averaging 28->9; np.clip to [0,1]; stored flattened "
    "row-major to 81 pixels (p = row*9 + col)"
)
DRAW_RULE = (
    "train_universe,test_universe = split_universes(60000, seed, UNIVERSE_SALT); "
    "train_order = default_rng([seed, DRAW_SALT]).permutation(train_universe); "
    "train_rows = train_order[:n] (nested prefixes across levels); "
    "query_rows = default_rng([seed, DRAW_SALT+1]).choice(test_universe, 10000, replace=False)"
)
RELEASE_RULE = (
    "mu,W,Q = release_map(pixels[train_rows(n)], seed, 60); A = Q @ W; "
    "z = apply_release(x, mu, A) in float64, returned float32; Q is shared by all "
    "levels of a seed, mu and W are refitted per level"
)

_STAGE_RE = re.compile(r"^[a-z][a-z0-9-]*$")
_CANDIDATE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.+-]*$")
DEVICE_KINDS = ("gpu", "cpu")

# Which dataset seeds each stage may touch.  Model selection ('smoke'/'dev')
# must never be performed on a final seed, otherwise the final freeze protects
# nothing.  The '-ensemble' stages are written by score.py from already saved
# per-candidate logits; they are bound to the same seeds as the stage they
# combine, and for final seeds score.py additionally requires every constituent
# job to appear unchanged in the final prediction freeze.
STAGE_SEEDS = {
    "smoke": tuple(DEV_SEEDS),
    "smoke-ensemble": tuple(DEV_SEEDS),
    "dev": tuple(DEV_SEEDS),
    "dev-ensemble": tuple(DEV_SEEDS),
    "final": tuple(FINAL_SEEDS),
    "final-ensemble": tuple(FINAL_SEEDS),
}
ENSEMBLE_STAGE = {"smoke": "smoke-ensemble", "dev": "dev-ensemble",
                  "final": "final-ensemble"}


def allowed_seeds(stage):
    """Seeds a stage may use, or ``None`` when the stage is not seed-restricted."""
    return STAGE_SEEDS.get(str(stage))


# ------------------------------------------------------------------ helpers ---
def utc() -> str:
    """Current UTC timestamp as an ISO-8601 string with a trailing Z."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def file_hash(path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha(path) -> str:
    """sha256 of the raw bytes of a file."""
    return file_hash(path, "sha256")


def ahash(array) -> str:
    """sha256 of C-order little-endian array contents."""
    array = np.asarray(array)
    array = np.ascontiguousarray(array.astype(array.dtype.newbyteorder("<"), copy=False))
    return hashlib.sha256(array.tobytes()).hexdigest()


def write_json(path, obj) -> Path:
    """Atomically write ``obj`` as JSON (indent=2, sort_keys=True, allow_nan=False)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n"
    handle, temporary = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".",
                                         suffix=".part")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(text)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return path


def _save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    with temporary.open("wb") as stream:
        np.save(stream, array, allow_pickle=False)
    os.replace(temporary, path)


def levels() -> list:
    """Exact geometric training budgets alongside the rounded integers used."""
    return [{"index": i, "exact": LEVELS_EXACT[i], "n": int(LEVELS[i])}
            for i in range(len(LEVELS))]


def _protocol_constants() -> dict:
    return {
        "task": "MNIST-medium release (gpumode harness 1.2.0), 9x9 pixels, 60 released dims",
        "harness": HARNESS_VERSION,
        "harness_eval_sha256": release.HARNESS_EVAL_SHA256,
        "source_pool": "official MNIST 60,000-example training split",
        "pool_count": POOL_COUNT,
        "image_size": IMAGE_SIZE,
        "pixel_count": PIXEL_COUNT,
        "release_dims": RELEASE_DIMS,
        "pixels": PIXEL_DESCRIPTION,
        "draw_rule": DRAW_RULE,
        "release_rule": RELEASE_RULE,
        "label_permutation": (
            "not applied; the harness's per-draw relabelling is a bijection of the "
            "ten classes applied to both halves, so query accuracy is invariant to it"
        ),
        "salts": {"UNIVERSE_SALT": int(release.UNIVERSE_SALT),
                  "DRAW_SALT": int(release.DRAW_SALT),
                  "RELEASE_SALT": int(release.RELEASE_SALT)},
        "dev_seeds": list(DEV_SEEDS),
        "final_seeds": list(FINAL_SEEDS),
        "levels": list(LEVELS),
        "levels_exact": list(LEVELS_EXACT),
        "query_count": QUERY_COUNT,
        "per_fit_time_budget_seconds": DEFAULT_TIME_BUDGET_SECONDS,
    }


# --------------------------------------------------------------- preparation ---
def _build_pool() -> tuple:
    import harness_mnist_data

    pixels, labels = harness_mnist_data.load_pool(POOL_CACHE, "mnist", IMAGE_SIZE)
    if pixels.shape != (POOL_COUNT, 1, IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError(f"unexpected pool shape {pixels.shape}")
    flat = np.ascontiguousarray(pixels.reshape(POOL_COUNT, PIXEL_COUNT), dtype=np.float32)
    pool_labels = np.ascontiguousarray(labels, dtype=np.int64)
    if pool_labels.shape != (POOL_COUNT,) or np.any(pool_labels > 9) or np.any(pool_labels < 0):
        raise ValueError("invalid MNIST class labels in the pool")
    if not np.isfinite(flat).all() or flat.min() < 0.0 or flat.max() > 1.0:
        raise ValueError("pool pixels are not finite values in [0,1]")
    return flat, pool_labels


def _source_specs() -> dict:
    specs = {}
    for key, (filename, expected_md5) in SOURCE_FILES.items():
        path = POOL_CACHE / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing canonical MNIST source {path}")
        actual_md5 = file_hash(path, "md5")
        if actual_md5 != expected_md5:
            raise ValueError(f"{path}: expected MD5 {expected_md5}, found {actual_md5}")
        specs[key] = {"path": str(path), "md5": expected_md5, "sha256": sha(path),
                      "bytes": path.stat().st_size}
    return specs


def _array_spec(path: Path, array: np.ndarray) -> dict:
    return {"path": str(Path(path).relative_to(ROOT)), "shape": list(array.shape),
            "dtype": str(array.dtype), "sha256": ahash(array),
            "file_sha256": sha(path), "bytes": Path(path).stat().st_size}


def prepare(force: bool = False) -> dict:
    """Build (or verify) ``raw/pool_pixels.npy``, ``raw/pool_labels.npy`` and the manifest.

    ``pool_pixels.npy`` holds pixels ONLY; it is the single array baked into the
    Modal image, so the container can recompute a draw's release from the seed
    without ever seeing a label it was not given.
    """
    for name in ("raw", "results", "predictions", "plans", "logs", "research"):
        (ROOT / name).mkdir(parents=True, exist_ok=True)
    outputs_exist = all(p.exists() for p in (POOL_PIXELS_PATH, POOL_LABELS_PATH,
                                             DATA_MANIFEST_PATH))
    if outputs_exist and not force:
        manifest = json.loads(DATA_MANIFEST_PATH.read_text())
        _verify_manifest(manifest)
        return manifest

    pixels, labels = _build_pool()
    _save_npy(POOL_PIXELS_PATH, pixels)
    _save_npy(POOL_LABELS_PATH, labels)
    manifest = {
        "schema_version": 1,
        "created_at_utc": utc(),
        "study": STUDY,
        "arrays": {"pool_pixels": _array_spec(POOL_PIXELS_PATH, pixels),
                   "pool_labels": _array_spec(POOL_LABELS_PATH, labels)},
        "class_histogram": np.bincount(labels, minlength=10).tolist(),
        "sources": _source_specs(),
        "pool_cache": str(POOL_CACHE),
        "generator_sha256": sha(ROOT / "harness_mnist_data.py"),
        "release_sha256": sha(ROOT / "release.py"),
        "protocol": _protocol_constants(),
        "software": {"python": platform.python_version(), "numpy": np.__version__},
        "query_labels_materialised": False,
        "notes": (
            "pool_pixels holds the canonical 9x9 pixels flattened to 81 features and is "
            "the only data array shipped to a container; pool_labels never leaves this "
            "host except as the per-job train_y slice. The release map is applied inside "
            "job_arrays(), never stored."
        ),
    }
    write_json(DATA_MANIFEST_PATH, manifest)
    _reset_cache()
    return manifest


def _verify_manifest(manifest: dict) -> None:
    if manifest.get("protocol") != _protocol_constants():
        raise ValueError("Protocol constants changed since raw/data_manifest.json was written")
    if manifest.get("generator_sha256") != sha(ROOT / "harness_mnist_data.py"):
        raise ValueError("harness_mnist_data.py changed since the manifest was written")
    if manifest.get("release_sha256") != sha(ROOT / "release.py"):
        raise ValueError("release.py changed since the manifest was written")
    expected = {"pool_pixels": (POOL_PIXELS_PATH, np.dtype("float32"), (POOL_COUNT, PIXEL_COUNT)),
                "pool_labels": (POOL_LABELS_PATH, np.dtype("int64"), (POOL_COUNT,))}
    for name, (path, dtype, shape) in expected.items():
        spec = manifest["arrays"][name]
        value = np.load(path)
        if value.dtype != dtype or value.shape != shape:
            raise ValueError(f"{name}: expected {dtype} {shape}, found {value.dtype} {value.shape}")
        if ahash(value) != spec["sha256"]:
            raise ValueError(f"{name}: content hash does not match raw/data_manifest.json")


# ------------------------------------------------------------------- loading ---
_CACHE: dict = {}


def _reset_cache() -> None:
    _CACHE.clear()


def _pool() -> tuple:
    if "pool" not in _CACHE:
        for path in (POOL_PIXELS_PATH, POOL_LABELS_PATH):
            if not path.exists():
                raise FileNotFoundError(f"{path} missing; run study.prepare() first")
        pixels = np.load(POOL_PIXELS_PATH)
        labels = np.load(POOL_LABELS_PATH)
        if pixels.shape != (POOL_COUNT, PIXEL_COUNT) or pixels.dtype != np.float32:
            raise ValueError("pool_pixels has an unexpected shape/dtype")
        if labels.shape != (POOL_COUNT,) or labels.dtype != np.int64:
            raise ValueError("pool_labels has an unexpected shape/dtype")
        _CACHE["pool"] = (pixels, labels)
    return _CACHE["pool"]


def pool_pixels() -> np.ndarray:
    """Read-only view of the (60000,81) float32 canonical pixel pool."""
    view = _pool()[0].view()
    view.flags.writeable = False
    return view


# --------------------------------------------------------------------- draws ---
def draw_rows(seed):
    """``(train_order (30000,), query_rows (10000,))`` for one dataset seed.

    ``train_order`` is the full shuffled training universe; level ``n`` uses its
    first ``n`` entries, so the levels are nested prefixes.  ``query_rows`` comes
    from the disjoint test universe and is the same for every level.
    """
    seed = int(seed)
    train_universe, test_universe = release.split_universes(
        POOL_COUNT, seed, release.UNIVERSE_SALT)
    train_order = np.random.default_rng([seed, release.DRAW_SALT]).permutation(train_universe)
    query_rows = np.random.default_rng([seed, release.DRAW_SALT + 1]).choice(
        test_universe, QUERY_COUNT, replace=False)
    return np.ascontiguousarray(train_order), np.ascontiguousarray(query_rows)


def _check_n(n) -> int:
    n = int(n)
    if not 1 <= n <= UNIVERSE_HALF:
        raise ValueError(f"n must satisfy 1 <= n <= {UNIVERSE_HALF}, got {n}")
    if n <= RELEASE_DIMS:
        raise ValueError(f"n must exceed release_dims={RELEASE_DIMS} to fit the whitener")
    return n


def train_rows(seed, n) -> np.ndarray:
    return draw_rows(seed)[0][:_check_n(n)].copy()


def query_rows(seed) -> np.ndarray:
    return draw_rows(seed)[1].copy()


def release_transform(seed, n):
    """``(mu, whitener, rotation, A)`` for one (seed, level); SECRET, diagnostics only."""
    pixels = _pool()[0]
    rows = train_rows(seed, n)
    mu, whitener, rotation = release.release_map(pixels[rows], int(seed), RELEASE_DIMS)
    return mu, whitener, rotation, rotation @ whitener


def job_arrays(seed, n) -> dict:
    """Exactly the three arrays a learner may see.  Never returns query labels."""
    n = _check_n(n)
    pixels, labels = _pool()
    order, query = draw_rows(seed)
    train = order[:n]
    if query.shape[0] != QUERY_COUNT:
        raise AssertionError("query rows must contain exactly 10,000 entries")
    if np.intersect1d(train, query).size != 0:
        raise AssertionError("train and query rows overlap")
    mu, whitener, rotation = release.release_map(pixels[train], int(seed), RELEASE_DIMS)
    transform = rotation @ whitener
    train_z = release.apply_release(pixels[train], mu, transform)
    query_z = release.apply_release(pixels[query], mu, transform)
    train_y = np.ascontiguousarray(labels[train], dtype=np.uint8)
    del mu, whitener, rotation, transform
    return {"train_z": train_z, "train_y": train_y, "query_z": query_z}


ARRAY_NAMES = ("train_z", "train_y", "query_z")


def make_job(stage, seed, n, candidate_id, config, learner_seed=DEFAULT_LEARNER_SEED,
             time_budget_seconds=DEFAULT_TIME_BUDGET_SECONDS, device_kind="gpu",
             extra=None) -> dict:
    """Fully specified unit of work, hashed against the exact arrays the learner sees."""
    if not _STAGE_RE.match(str(stage)):
        raise ValueError(f"stage must be lowercase alphanumeric/hyphen, got {stage!r}")
    if not _CANDIDATE_RE.match(str(candidate_id)):
        raise ValueError(f"candidate_id has unsupported characters: {candidate_id!r}")
    if str(device_kind) not in DEVICE_KINDS:
        raise ValueError(f"device_kind must be one of {DEVICE_KINDS}, got {device_kind!r}")
    permitted = allowed_seeds(stage)
    if permitted is not None and int(seed) not in permitted:
        raise ValueError(
            f"stage {str(stage)!r} may only use seeds {list(permitted)}, got {int(seed)}; "
            "model-selection stages must never touch a final seed")
    if permitted is None and int(seed) in FINAL_SEEDS:
        raise ValueError(
            f"stage {str(stage)!r} is not a protocol stage and may not use final "
            f"seed {int(seed)}")
    if not isinstance(config, dict):
        raise TypeError("config must be a dict")
    budget = int(time_budget_seconds)
    if not 0 < budget <= DEFAULT_TIME_BUDGET_SECONDS:
        raise ValueError(f"time_budget_seconds must be in 1..{DEFAULT_TIME_BUDGET_SECONDS}")
    arrays = job_arrays(seed, n)
    job = {
        "id": f"{stage}-{candidate_id}-s{int(seed)}-n{int(n)}",
        "stage": str(stage),
        "seed": int(seed),
        "n": int(n),
        "candidate_id": str(candidate_id),
        "config": copy.deepcopy(config),
        "learner_seed": int(learner_seed),
        "time_budget_seconds": budget,
        "device_kind": str(device_kind),
        "release_dims": RELEASE_DIMS,
        "query_count": QUERY_COUNT,
        "train_rows_sha256": ahash(train_rows(seed, n)),
        "query_rows_sha256": ahash(query_rows(seed)),
        "input_sha256": {name: ahash(arrays[name]) for name in ARRAY_NAMES},
    }
    if extra:
        overlap = set(extra) & set(job)
        if overlap:
            raise ValueError(f"extra may not override job keys: {sorted(overlap)}")
        job.update(copy.deepcopy(extra))
    return job


# --------------------------------------------------------- scoring-only labels ---
FORBIDDEN_MODULES = ("kernels", "neural", "runner", "pmnist_learners", "ladder_model")
# A script run as ``python runner.py`` is ``__main__``, never ``runner`` in
# sys.modules, so FORBIDDEN_MODULES alone cannot see it: the entry point is
# checked too.  (``score.py`` and the research scorers are the intended callers;
# they are not on this list.)
FORBIDDEN_ENTRYPOINTS = ("runner.py", "plan.py", "kernels.py", "neural.py",
                         "probe_neural.py", "pmnist_learners.py", "ladder_model.py")


def query_labels(seed) -> np.ndarray:
    """SCORING ONLY. The single function in this study that returns query labels.

    Never import or call this from a learner, the runner, or anything that will be
    serialised into a Modal container.  ``score.py`` is the only caller in the
    measurement path.
    """
    for forbidden in FORBIDDEN_MODULES:
        if forbidden in sys.modules:
            raise RuntimeError(
                f"query_labels() refused: module {forbidden!r} is loaded in this process; "
                "query labels are scoring-only and must never reach a learner")
    entrypoint = Path(getattr(sys.modules.get("__main__"), "__file__", "") or "").name
    if entrypoint in FORBIDDEN_ENTRYPOINTS:
        raise RuntimeError(
            f"query_labels() refused: this process was started as {entrypoint}; "
            "query labels are scoring-only and must never reach a learner or the runner")
    _, labels = _pool()
    return np.ascontiguousarray(labels[query_rows(seed)], dtype=np.uint8)


# ----------------------------------------------------------- selection rule ---
def selection_rule() -> dict:
    """The frozen selection rule.  Written before any dev result exists.

    Changing anything here after a dev result has been scored invalidates the
    study: protocol.draft.json records this dict verbatim with its own hash.
    """
    return {
        "frozen_before_any_dev_result": True,
        "dev_evidence": (
            "Mean query error over the dev seeds 2026092491 and 2026092492 for every "
            "candidate, at the two endpoint levels N=500 and N=10000 as a minimum; "
            "interior levels are run on dev only when the budget allows."
        ),
        "eligibility": (
            "A candidate is eligible at a level only if every dev fit at that level "
            "finished untruncated inside the 1200 s per-fit budget (GPU candidates on one "
            "A100-40GB; CPU-only kernel candidates within 1200 s wall on the controller "
            "CPU) and used no forbidden input."
        ),
        "finalists": (
            "Three finalists at most: (1) the eligible recipe with the lowest mean dev "
            "error at N=500, (2) the eligible recipe with the lowest mean dev error at "
            "N=10000, and (3) the best two-member ensemble of dev candidates -- included "
            "only if it beats BOTH single finalists at at least one endpoint. If (1) and "
            "(2) are the same recipe, that recipe is one finalist. An ensemble finalist "
            "must be written into selection.json as "
            "final_ensemble={members,weights,temperatures} before the final freeze; "
            "score.py refuses to build any other final ensemble, and refuses to build one "
            "at all once the freeze has been scored."
        ),
        "final_measurement": (
            "Every finalist is retrained from scratch at all five levels "
            "(500, 1057, 2236, 4729, 10000) on all eleven final seeds "
            "2026092401..2026092411 with learner seed 11; every prediction is frozen "
            "(predictions/final_freeze.json) before score.py reads any final query label."
        ),
        "thresholds": (
            "The threshold at each level is the lowest POOLED error among the finalists "
            "at that level (pooled = total misclassified queries over all eleven final "
            "seeds divided by 110,000), i.e. a per-level best, and the recipe attaining it "
            "is named alongside it."
        ),
        "sota_recipe": (
            "The single recipe with the lowest mean of its five per-level pooled errors "
            "is named 'the SOTA recipe' for this release."
        ),
        "ties": (
            "Two recipes are tied at a level (or in the five-level mean) when they are "
            "within the LARGER of 0.05 percentage points and the observed across-draw "
            "standard error at that level (sem_error_pct = sd over the final seeds / "
            "sqrt(count)), and the tie goes to the cheaper recipe: lower mean fit wall "
            "seconds, and a CPU-only recipe beats a GPU recipe at equal error."
        ),
        "no_test_guided_changes": (
            "No hyperparameter, epoch count, member count, ensemble weight or temperature "
            "is changed after any final query label has been read."
        ),
        "budget_fallback": (
            "If the projected final cost of a finalist exceeds the remaining authorized "
            "allowance (the $20 cap minus the $2 contingency minus everything already "
            "charged in budget-ledger.json), the next-best eligible candidate whose "
            "projected cost fits is used instead, and the substitution is disclosed in the "
            "report. If even that does not fit, the final stage is run on a reduced but "
            "pre-declared subset of the eleven final seeds (a prefix of the seed list, "
            "never a subset chosen after seeing results), the reduction is disclosed, and "
            "the thresholds are reported with the smaller seed count and its standard error."
        ),
        "reporting": (
            "Five thresholds are reported as pooled error percentages with the ACROSS-DRAW "
            "standard error over the final seeds (score.py's sem_error_pct = sd/sqrt(count)) "
            "and the per-seed spread. The within-draw binomial standard error "
            "(pooled_binomial_se_pct) is kept only as a diagnostic: the 10,000 queries of "
            "one draw are classified by one model fitted on one training prefix, so they "
            "are not independent trials and the binomial figure understates the real "
            "uncertainty (roughly 2x at N=500). Each threshold is stated as 'achievable', "
            "not as a bound, because a better recipe may exist, and because the "
            "min-over-finalists rule carries a winner's-curse bias of about one standard "
            "error the report says so explicitly."
        ),
    }


# ------------------------------------------------------------- protocol draft ---
def protocol_draft(manifest: dict = None) -> dict:
    """Protocol document with the data hashes filled in from the manifest."""
    if manifest is None and DATA_MANIFEST_PATH.exists():
        manifest = json.loads(DATA_MANIFEST_PATH.read_text())
    manifest = manifest or {}
    arrays = manifest.get("arrays", {})
    return {
        "schema_version": 1,
        "status": "draft",
        "drafted_at_utc": utc(),
        "study": STUDY,
        "task": (
            "gpumode harness 1.2.0 competition-candidate release of MNIST-medium: "
            "z = Q W (x - mu) in 60 dims, 10,000 queries per draw"
        ),
        "question": (
            "Best achievable ('state-of-the-art') accuracy on the release, under 1200 s "
            "on one A100-40GB per fit, at five geometric training levels from 500 to "
            "10,000 examples, and five accuracy thresholds derived from it."
        ),
        "source": {
            "pool": "official MNIST 60,000-example training split (the official test split is never used)",
            "files": manifest.get("sources", {}),
            "pixels": PIXEL_DESCRIPTION,
            "loader": "harness_mnist_data.py (verbatim copy of gpumode/mnist_data.py)",
            "loader_sha256": manifest.get("generator_sha256"),
        },
        "release": {
            "module": "release.py (verbatim copy of the harness release map)",
            "harness": HARNESS_VERSION,
            "harness_eval_sha256": release.HARNESS_EVAL_SHA256,
            "release_sha256": manifest.get("release_sha256"),
            "rule": RELEASE_RULE,
            "dims": RELEASE_DIMS,
            "q_shared_across_levels": True,
            "map_refitted_per_level": ["mu", "W"],
            "spatial_structure": "unavailable by design; no pixel lattice is recoverable from z",
            "delivery": (
                "the release is computed once on the controller and the three released "
                "arrays are shipped to the learner process; the pixel pool, the dataset "
                "labels and the map (mu, W, Q) are never present in a container, so the "
                "map cannot be refitted and z cannot be inverted back to pixels. No "
                "numerical agreement between controller and container is assumed: "
                "release_map is an eigendecomposition and LAPACK is not bit-reproducible "
                "across BLAS builds or CPU targets"
            ),
        },
        "draws": {
            "rule": DRAW_RULE,
            "nesting": "training prefixes are nested across levels within a dataset seed",
            "query_count": QUERY_COUNT,
            "disjointness": (
                "train and query rows come from the two halves of split_universes, so "
                "query rows are disjoint from every training prefix at every level"
            ),
            "label_permutation": _protocol_constants()["label_permutation"],
            "dev_seeds": list(DEV_SEEDS),
            "final_seeds": list(FINAL_SEEDS),
            "stage_seed_binding": (
                "'smoke'/'dev' jobs may only use dev seeds and 'final' jobs only final "
                "seeds; study.make_job refuses other combinations and score.py refuses to "
                "read query labels for a job whose stage and seed disagree"
            ),
        },
        "levels": levels(),
        "learner_contract": {
            "input_allowlist": list(ARRAY_NAMES),
            "train_z": f"float32 (n,{RELEASE_DIMS}) released features",
            "train_y": "uint8 (n,) labels 0-9",
            "query_z": f"float32 ({QUERY_COUNT},{RELEASE_DIMS}) released features",
            "forbidden": [
                "query labels", "the release map (mu, W, Q)", "the dataset pixels",
                "external data", "pretrained weights", "MNIST-specific constants",
            ],
            "from_scratch": "every draw is trained from scratch; nothing carries across draws",
            "transductive_rule": (
                "unlabeled use of query_z is allowed (the harness hands the query features "
                "to the learner) and must be recorded per candidate as "
                "'uses_query_features_unlabeled'"
            ),
            "api": "fit_predict(train_z, train_y, query_z, config, seed, device, deadline_unix)",
            "modules": {"kernels.py": "kernel learners", "neural.py": "families mlp/ladder/vat"},
        },
        "compute": {
            "gpu": ["A100-40GB", "T4"],
            "per_fit_time_budget_seconds": DEFAULT_TIME_BUDGET_SECONDS,
            "budget_note": (
                "wall clock on one A100-40GB including inference; learners stop gracefully "
                "at the deadline. CPU-only kernel candidates run on the local controller "
                "(free) and must still finish inside 1200 s to be eligible."
            ),
            "modal_cap_usd": 20,
            "required_metrics": ["fit_wall_seconds", "truncated", "uses_query_features_unlabeled"],
            "container_image": (
                "ghcr.io/ab-10/wikitext-bench@sha256:"
                "95de89319ba89c53a91d5440a5db4ff46f68b05031062c0a760ec3caa48dc42f"
            ),
        },
        "outputs": {
            "predictions": (
                f"predictions/<job_id>.npz with logits float32 ({QUERY_COUNT},10) and "
                f"labels uint8 ({QUERY_COUNT},)"
            ),
            "tie_break": "labels = argmax over logits with ties resolved to the lowest class index",
            "results": "results/<job_id>.json per the study-wide contract",
        },
        "selection_rule": selection_rule(),
        "prediction_freeze": (
            "final scoring requires predictions/final_freeze.json covering every planned "
            "final job id; score.py refuses to score an unfrozen or altered final run, "
            "requires every planned job to have a result even for a --job-id spot check "
            "(which writes results/scores_final_filtered.json, never the canonical table), "
            "marks the freeze 'scored' once final labels have been read and then refuses "
            "any re-freeze; superseding an unscored freeze needs --refreeze and archives "
            "the previous file. A 'final-ensemble' recipe is built only from constituent "
            "final jobs that appear unchanged in that freeze, only while the freeze is "
            "still unscored, and only with the members, weights and temperatures declared "
            "in selection.json; scoring the derived 'final-ensemble' stage requires the "
            "same freeze."
        ),
        "data_manifest": {
            "path": str(DATA_MANIFEST_PATH.relative_to(ROOT)),
            "sha256": sha(DATA_MANIFEST_PATH) if DATA_MANIFEST_PATH.exists() else None,
            "pool_pixels_sha256": arrays.get("pool_pixels", {}).get("sha256"),
            "pool_labels_sha256": arrays.get("pool_labels", {}).get("sha256"),
        },
        "software": {
            "preparation": manifest.get("software", {}),
            "local_python": "/tmp/penv/bin/python (3.11, numpy 1.26.4, torch 2.2.2 CPU, scipy 1.11.4)",
            "remote": "PyTorch 2.5.1+cu124, numpy 2.2.6, scipy 1.15.3",
        },
        "query_labels_materialised": False,
    }


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Prepare the release-cutoffs pool")
    parser.add_argument("--prepare", action="store_true", help="build or verify the pool arrays")
    parser.add_argument("--force", action="store_true", help="rebuild even when outputs exist")
    parser.add_argument("--write-protocol-draft", action="store_true")
    arguments = parser.parse_args(argv)
    manifest = prepare(force=arguments.force) if (arguments.prepare or arguments.force) else None
    if manifest is not None:
        for name, spec in sorted(manifest["arrays"].items()):
            print(f"{name}: {spec['dtype']} {tuple(spec['shape'])} sha256={spec['sha256']}")
        print(f"data_manifest.json sha256={sha(DATA_MANIFEST_PATH)}")
    if arguments.write_protocol_draft:
        write_json(PROTOCOL_DRAFT_PATH, protocol_draft(manifest))
        print(f"wrote {PROTOCOL_DRAFT_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
