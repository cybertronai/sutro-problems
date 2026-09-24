"""Data/protocol layer for the permutation-invariant MNIST-medium cutoff study.

Task: "MNIST-medium, permutation-invariant".  The source pool is the official
60,000-example MNIST *training* split.  Pixels are converted uint8 -> float32/255,
resized 28->9 with the canonical exact separable area filter, clipped to [0,1] and
flattened row-major to 81 features (feature ``f = row*9 + col``).  One fixed
feature permutation is applied to every image so learners only ever see
``x[:, perm]``; learners never receive ``perm``, its inverse, or any coordinates.

This module owns every dataset decision.  It is imported by the runner, by the
learners' job builders and by ``score.py``.  It NEVER hands out query labels
except through :func:`query_labels`, which is scoring-only and refuses to run
inside a learner/runner process.

No Modal calls, no GPU, no network access.
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

import canonical_data  # noqa: E402  (needs ROOT on sys.path)

# ---------------------------------------------------------------- protocol ---
FEATURE_PERMUTATION_SEED = 20260923
DEV_SEEDS = [2026092301, 2026092302, 2026092303]
FINAL_SEEDS = list(range(2026092001, 2026092012))  # 2026092001 .. 2026092011
LEVELS = [1000, 1778, 3162, 5623, 10000]           # round(1000*10**(i/4)), i=0..4
QUERY_START = 10000
QUERY_COUNT = 10000

POOL_COUNT = 60000
IMAGE_SIZE = 9
FEATURE_COUNT = IMAGE_SIZE * IMAGE_SIZE           # 81
DEFAULT_TIME_BUDGET_SECONDS = 1200

SOURCE_FILES = {
    "train_images": ("train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
    "train_labels": ("train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
}

RAW_DIR = ROOT / "raw"
SOURCE_DIR = RAW_DIR / "source"
POOL_IMAGES_PATH = RAW_DIR / "pool_images.npy"
POOL_LABELS_PATH = RAW_DIR / "pool_labels.npy"
PERMUTATION_PATH = RAW_DIR / "feature_permutation.npy"
DATA_MANIFEST_PATH = RAW_DIR / "data_manifest.json"
PROTOCOL_DRAFT_PATH = ROOT / "protocol.draft.json"

RESIZE_DESCRIPTION = (
    "uint8 -> float32/255; canonical_data.area_resize exact separable box-area "
    "averaging 28->9; np.clip to [0,1]; flatten row-major to 81 features (f=row*9+col)"
)
DRAW_RULE = (
    "order = Generator(PCG64(SeedSequence(seed).spawn(2)[0])).permutation(60000); "
    "train = order[:n] (nested prefixes across levels); query = order[10000:20000]"
)

_STAGE_RE = re.compile(r"^[a-z][a-z0-9-]*$")
_CANDIDATE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.+-]*$")
# Which dataset seeds each stage may touch.  Model selection ('smoke'/'dev') must
# never be performed on a final seed, otherwise the final freeze protects nothing.
STAGE_SEEDS = {
    "smoke": tuple(DEV_SEEDS),
    "dev": tuple(DEV_SEEDS),
    "final": tuple(FINAL_SEEDS),
}


def allowed_seeds(stage: str):
    """Seeds a stage may use, or ``None`` when the stage is not seed-restricted."""
    return STAGE_SEEDS.get(str(stage))


# ------------------------------------------------------------------ helpers ---
def utc() -> str:
    """Current UTC timestamp as an ISO-8601 string with a trailing Z."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha(path) -> str:
    """sha256 of the raw bytes of a file."""
    return canonical_data.file_hash(Path(path), "sha256")


def ahash(array: np.ndarray) -> str:
    """sha256 of C-order little-endian array contents (== canonical_data.array_hash)."""
    return canonical_data.array_hash(np.asarray(array))


def write_json(path, obj) -> Path:
    """Atomically write ``obj`` as JSON (indent=2, sort_keys=True, allow_nan=False)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n"
    handle, temporary = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".part")
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
    """Exact geometric training budgets alongside the rounded integers actually used."""
    return [
        {"index": i, "exact": float(1000.0 * 10.0 ** (i / 4.0)), "n": int(LEVELS[i])}
        for i in range(len(LEVELS))
    ]


def _protocol_constants() -> dict:
    return {
        "task": "MNIST-medium, permutation-invariant",
        "source_pool": "official MNIST 60,000-example training split",
        "pool_count": POOL_COUNT,
        "image_size": IMAGE_SIZE,
        "feature_count": FEATURE_COUNT,
        "feature_layout": "row-major, feature f = row*9 + col, before permutation",
        "feature_permutation_seed": FEATURE_PERMUTATION_SEED,
        "feature_permutation_rule": (
            "np.random.Generator(np.random.PCG64(20260923)).permutation(81); "
            "learners see x[:, perm] and never receive perm or any coordinates"
        ),
        "resize": RESIZE_DESCRIPTION,
        "draw_rule": DRAW_RULE,
        "dev_seeds": list(DEV_SEEDS),
        "final_seeds": list(FINAL_SEEDS),
        "levels": list(LEVELS),
        "levels_exact": [entry["exact"] for entry in levels()],
        "query_start": QUERY_START,
        "query_count": QUERY_COUNT,
        "per_fit_time_budget_seconds": DEFAULT_TIME_BUDGET_SECONDS,
    }


# --------------------------------------------------------------- preparation ---
def _feature_permutation() -> np.ndarray:
    rng = np.random.Generator(np.random.PCG64(FEATURE_PERMUTATION_SEED))
    return rng.permutation(FEATURE_COUNT).astype(np.int64)


def _source_specs() -> dict:
    specs = {}
    for key, (filename, expected_md5) in SOURCE_FILES.items():
        path = SOURCE_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing canonical MNIST source {path}")
        actual_md5 = canonical_data.file_hash(path, "md5")
        if actual_md5 != expected_md5:
            raise ValueError(f"{path}: expected MD5 {expected_md5}, found {actual_md5}")
        specs[key] = {
            "path": str(path.relative_to(ROOT)),
            "md5": expected_md5,
            "sha256": sha(path),
            "bytes": path.stat().st_size,
        }
    return specs


def _build_pool() -> tuple:
    images = canonical_data.read_idx(SOURCE_DIR / SOURCE_FILES["train_images"][0], POOL_COUNT, True)
    labels = canonical_data.read_idx(SOURCE_DIR / SOURCE_FILES["train_labels"][0], POOL_COUNT, False)
    values = images.astype(np.float32) / np.float32(255)
    resized = canonical_data.area_resize(values, IMAGE_SIZE)
    np.clip(resized, 0.0, 1.0, out=resized)
    pool_images = np.ascontiguousarray(resized.reshape(POOL_COUNT, FEATURE_COUNT), dtype=np.float32)
    pool_labels = np.ascontiguousarray(labels, dtype=np.uint8)
    if np.any(pool_labels > 9):
        raise ValueError("Invalid MNIST class label in the pool")
    return pool_images, pool_labels


def prepare(force: bool = False) -> dict:
    """Build (or verify) the unpermuted pool, the feature permutation and the manifest.

    Idempotent: when every output already exists and ``force`` is false the arrays
    are re-hashed and checked against ``raw/data_manifest.json`` instead of rebuilt.
    """
    for name in ("raw", "raw/source", "results", "predictions", "plans", "logs"):
        (ROOT / name).mkdir(parents=True, exist_ok=True)
    sources = _source_specs()
    outputs_exist = all(p.exists() for p in (POOL_IMAGES_PATH, POOL_LABELS_PATH,
                                             PERMUTATION_PATH, DATA_MANIFEST_PATH))
    if outputs_exist and not force:
        manifest = json.loads(DATA_MANIFEST_PATH.read_text())
        _verify_manifest(manifest, sources)
        return manifest

    pool_images, pool_labels = _build_pool()
    permutation = _feature_permutation()
    _save_npy(POOL_IMAGES_PATH, pool_images)
    _save_npy(POOL_LABELS_PATH, pool_labels)
    _save_npy(PERMUTATION_PATH, permutation)
    manifest = {
        "schema_version": 1,
        "created_at_utc": utc(),
        "study": "pmnist-medium-cutoffs-20260923",
        "arrays": {
            "pool_images": _array_spec(POOL_IMAGES_PATH, pool_images),
            "pool_labels": _array_spec(POOL_LABELS_PATH, pool_labels),
            "feature_permutation": _array_spec(PERMUTATION_PATH, permutation),
        },
        "class_histogram": np.bincount(pool_labels, minlength=10).tolist(),
        "sources": sources,
        "generator_sha256": sha(Path(canonical_data.__file__)),
        "protocol": _protocol_constants(),
        "software": {"python": platform.python_version(), "numpy": np.__version__},
        "test_labels_created": False,
        "notes": (
            "pool_images holds UNPERMUTED canonical 9x9 features; the permutation is "
            "applied only inside job_arrays(). Query labels are never materialised here."
        ),
    }
    write_json(DATA_MANIFEST_PATH, manifest)
    _reset_cache()
    return manifest


def _array_spec(path: Path, array: np.ndarray) -> dict:
    return {
        "path": str(Path(path).relative_to(ROOT)),
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sha256": ahash(array),
        "file_sha256": sha(path),
        "bytes": Path(path).stat().st_size,
    }


def _verify_manifest(manifest: dict, sources: dict) -> None:
    if manifest.get("sources") != sources:
        raise ValueError("raw/data_manifest.json disagrees with the on-disk MNIST sources")
    if manifest.get("generator_sha256") != sha(Path(canonical_data.__file__)):
        raise ValueError("canonical_data.py changed since the manifest was written")
    if manifest.get("protocol") != _protocol_constants():
        raise ValueError("Protocol constants changed since raw/data_manifest.json was written")
    arrays = {
        "pool_images": (POOL_IMAGES_PATH, np.dtype("float32"), (POOL_COUNT, FEATURE_COUNT)),
        "pool_labels": (POOL_LABELS_PATH, np.dtype("uint8"), (POOL_COUNT,)),
        "feature_permutation": (PERMUTATION_PATH, np.dtype("int64"), (FEATURE_COUNT,)),
    }
    for name, (path, dtype, shape) in arrays.items():
        spec = manifest["arrays"][name]
        value = np.load(path)
        if value.dtype != dtype or value.shape != shape:
            raise ValueError(f"{name}: expected {dtype} {shape}, found {value.dtype} {value.shape}")
        if ahash(value) != spec["sha256"]:
            raise ValueError(f"{name}: content hash does not match raw/data_manifest.json")
    permutation = np.load(PERMUTATION_PATH)
    if not np.array_equal(np.sort(permutation), np.arange(FEATURE_COUNT)):
        raise ValueError("feature_permutation is not a bijection of 0..80")
    if not np.array_equal(permutation, _feature_permutation()):
        raise ValueError("feature_permutation does not match the seeded rule")


# ------------------------------------------------------------------- loading ---
_CACHE: dict = {}


def _reset_cache() -> None:
    _CACHE.clear()


def _pool() -> tuple:
    if "pool" not in _CACHE:
        for path in (POOL_IMAGES_PATH, POOL_LABELS_PATH, PERMUTATION_PATH):
            if not path.exists():
                raise FileNotFoundError(f"{path} missing; run study.prepare() first")
        images = np.load(POOL_IMAGES_PATH)
        labels = np.load(POOL_LABELS_PATH)
        permutation = np.load(PERMUTATION_PATH)
        if images.shape != (POOL_COUNT, FEATURE_COUNT) or images.dtype != np.float32:
            raise ValueError("pool_images has an unexpected shape/dtype")
        if labels.shape != (POOL_COUNT,) or labels.dtype != np.uint8:
            raise ValueError("pool_labels has an unexpected shape/dtype")
        if permutation.shape != (FEATURE_COUNT,) or permutation.dtype != np.int64:
            raise ValueError("feature_permutation has an unexpected shape/dtype")
        _CACHE["pool"] = (images, labels, permutation)
    return _CACHE["pool"]


def pool_images() -> np.ndarray:
    """Read-only view of the UNPERMUTED (60000,81) float32 canonical pool."""
    images = _pool()[0]
    view = images.view()
    view.flags.writeable = False
    return view


def feature_permutation() -> np.ndarray:
    """Read-only copy of the fixed 81-feature permutation (scoring/diagnostics only)."""
    permutation = _pool()[2].copy()
    permutation.flags.writeable = False
    return permutation


# --------------------------------------------------------------------- draws ---
def draw_order(seed: int) -> np.ndarray:
    """Per-dataset-seed permutation of the 60,000 pool rows (int64)."""
    child = np.random.SeedSequence(int(seed)).spawn(2)[0]
    return np.random.Generator(np.random.PCG64(child)).permutation(POOL_COUNT).astype(np.int64)


def train_indices(seed: int, n: int) -> np.ndarray:
    n = int(n)
    if n < 1 or n > QUERY_START:
        raise ValueError(f"n must satisfy 1 <= n <= {QUERY_START}, got {n}")
    return draw_order(seed)[:n].copy()


def query_indices(seed: int) -> np.ndarray:
    return draw_order(seed)[QUERY_START:QUERY_START + QUERY_COUNT].copy()


def job_arrays(seed: int, n: int) -> dict:
    """Exactly the three arrays a learner may see: permuted pixels and train labels."""
    n = int(n)
    if n not in LEVELS and not (1 <= n <= QUERY_START):
        raise ValueError(f"n={n} is neither a protocol level nor <= {QUERY_START}")
    images, labels, permutation = _pool()
    order = draw_order(seed)
    train = order[:n]
    query = order[QUERY_START:QUERY_START + QUERY_COUNT]
    if query.shape[0] != QUERY_COUNT:
        raise AssertionError("query slice must contain exactly 10,000 rows")
    if np.intersect1d(train, query).size != 0:
        raise AssertionError("train and query indices overlap")
    train_x = np.ascontiguousarray(images[train][:, permutation], dtype=np.float32)
    query_x = np.ascontiguousarray(images[query][:, permutation], dtype=np.float32)
    train_y = np.ascontiguousarray(labels[train], dtype=np.uint8)
    return {"train_x": train_x, "train_y": train_y, "query_x": query_x}


def make_job(stage: str, seed: int, n: int, candidate_id: str, config: dict,
             learner_seed: int, time_budget_seconds: int = DEFAULT_TIME_BUDGET_SECONDS) -> dict:
    """Fully specified unit of work, hashed against the exact arrays the learner sees."""
    if not _STAGE_RE.match(str(stage)):
        raise ValueError(f"stage must be lowercase alphanumeric/hyphen, got {stage!r}")
    if not _CANDIDATE_RE.match(str(candidate_id)):
        raise ValueError(f"candidate_id has unsupported characters: {candidate_id!r}")
    permitted = allowed_seeds(stage)
    if permitted is not None and int(seed) not in permitted:
        raise ValueError(
            f"stage {str(stage)!r} may only use seeds {list(permitted)}, got {int(seed)}; "
            "model-selection stages must never touch a final seed"
        )
    if permitted is None and int(seed) in FINAL_SEEDS:
        raise ValueError(
            f"stage {str(stage)!r} is not a protocol stage and may not use final seed {int(seed)}"
        )
    if not isinstance(config, dict):
        raise TypeError("config must be a dict")
    arrays = job_arrays(seed, n)
    return {
        "id": f"{stage}-{candidate_id}-s{int(seed)}-n{int(n)}",
        "stage": str(stage),
        "seed": int(seed),
        "n": int(n),
        "candidate_id": str(candidate_id),
        "config": copy.deepcopy(config),
        "learner_seed": int(learner_seed),
        "time_budget_seconds": int(time_budget_seconds),
        "train_indices_sha256": ahash(train_indices(seed, n)),
        "query_indices_sha256": ahash(query_indices(seed)),
        "input_sha256": {name: ahash(value) for name, value in sorted(arrays.items())},
    }


# --------------------------------------------------------- scoring-only labels ---
def query_labels(seed: int) -> np.ndarray:
    """SCORING ONLY. The single function in this study that returns query labels.

    Never import or call this from a learner, a runner, or anything that will be
    serialised into a Modal container.  ``score.py`` is the only caller.
    """
    for forbidden in ("learners", "runner"):
        if forbidden in sys.modules:
            raise RuntimeError(
                f"query_labels() refused: module {forbidden!r} is loaded in this process; "
                "query labels are scoring-only and must never reach a learner"
            )
    _, labels, _ = _pool()
    return np.ascontiguousarray(labels[query_indices(seed)], dtype=np.uint8)


# ------------------------------------------------------------- protocol draft ---
def protocol_draft(manifest: dict = None) -> dict:
    """Human-readable protocol document; data hashes filled from the manifest."""
    if manifest is None and DATA_MANIFEST_PATH.exists():
        manifest = json.loads(DATA_MANIFEST_PATH.read_text())
    manifest = manifest or {}
    arrays = manifest.get("arrays", {})
    return {
        "schema_version": 1,
        "status": "draft",
        "drafted_at_utc": utc(),
        "study": "pmnist-medium-cutoffs-20260923",
        "task": "MNIST-medium, permutation-invariant (9x9, 10,000 queries)",
        "question": (
            "Best achievable permutation-invariant accuracy under 1200 s on one "
            "A100-40GB per fit, and a five-level ladder from 1,000 to 10,000 training examples."
        ),
        "source": {
            "pool": "official MNIST 60,000-example training split (no official test split is used)",
            "files": manifest.get("sources", {}),
            "generator_sha256": manifest.get("generator_sha256"),
            "generator": "canonical_data.py (verbatim copy of mnist/code/data.py)",
        },
        "features": {
            "resize": RESIZE_DESCRIPTION,
            "dtype": "float32 in [0,1]",
            "shape": [FEATURE_COUNT],
            "permutation_seed": FEATURE_PERMUTATION_SEED,
            "permutation_rule": (
                "perm = np.random.Generator(np.random.PCG64(20260923)).permutation(81); "
                "every train and query image is presented as x[:, perm]"
            ),
            "permutation_disclosure": (
                "Learners never receive perm, its inverse, pixel coordinates, or any "
                "spatial metadata. Recovering topology from the data itself is allowed "
                "and must be declared by the candidate."
            ),
        },
        "draws": {
            "rule": DRAW_RULE,
            "nesting": "training prefixes are nested across levels within a dataset seed",
            "query_slice": [QUERY_START, QUERY_START + QUERY_COUNT],
            "query_count": QUERY_COUNT,
            "disjointness": "query rows are disjoint from every training prefix (n <= 10000)",
            "dev_seeds": list(DEV_SEEDS),
            "final_seeds": list(FINAL_SEEDS),
            "stage_seed_binding": (
                "'smoke' and 'dev' jobs may only use dev seeds and 'final' jobs only final "
                "seeds; study.make_job refuses other combinations and score.py refuses to "
                "read query labels for a job whose stage and seed disagree"
            ),
            "final_seed_note": (
                "the same eleven seeds as mnist/experiments/cutoff-calibration-20260920, "
                "enabling a paired comparison against its spatial CNN results"
            ),
        },
        "levels": levels(),
        "learner_contract": {
            "input_allowlist": ["train_x", "train_y", "query_x"],
            "train_x": "float32 (n,81) permuted pixels",
            "train_y": "uint8 (n,) labels 0-9",
            "query_x": "float32 (10000,81) permuted pixels",
            "forbidden": [
                "query labels", "the feature permutation", "pixel coordinates",
                "the official MNIST test split", "any pretrained weights or external data",
            ],
            "transductive_rule": (
                "unlabeled use of query_x is allowed and must be recorded per candidate as "
                "'uses_query_images_unlabeled'"
            ),
            "api": "learners.fit_predict(train_x, train_y, query_x, config, seed, deadline_unix, device)",
        },
        "compute": {
            "gpu": "A100-40GB",
            "per_fit_time_budget_seconds": DEFAULT_TIME_BUDGET_SECONDS,
            "budget_note": "wall clock on one GPU including inference; learners stop gracefully",
            "required_metrics": ["epochs_completed", "truncated", "fit_wall_seconds"],
            "container_image": (
                "ghcr.io/ab-10/wikitext-bench@sha256:"
                "95de89319ba89c53a91d5440a5db4ff46f68b05031062c0a760ec3caa48dc42f"
            ),
        },
        "outputs": {
            "predictions": "predictions/<job_id>.npz with logits float32 (10000,10) and labels uint8 (10000,)",
            "tie_break": "labels = argmax over logits with ties resolved to the lowest class index",
            "results": "results/<job_id>.json per the study-wide contract",
        },
        "selection_rule": (
            "to be frozen in selection.json before final dispatch; score.py refuses to "
            "create a final freeze while selection.json is absent"
        ),
        "prediction_freeze": (
            "final scoring requires predictions/final_freeze.json covering every planned "
            "final job id; score.py refuses to score an unfrozen or altered final run, "
            "requires every planned job to have a result even for a --job-id spot check "
            "(which writes results/scores_final_filtered.json, never the canonical table), "
            "marks the freeze 'scored' once final labels have been read and then refuses "
            "any re-freeze; superseding an unscored freeze needs --refreeze and archives "
            "the previous file"
        ),
        "data_manifest": {
            "path": str(DATA_MANIFEST_PATH.relative_to(ROOT)),
            "sha256": sha(DATA_MANIFEST_PATH) if DATA_MANIFEST_PATH.exists() else None,
            "pool_images_sha256": arrays.get("pool_images", {}).get("sha256"),
            "pool_labels_sha256": arrays.get("pool_labels", {}).get("sha256"),
            "feature_permutation_sha256": arrays.get("feature_permutation", {}).get("sha256"),
        },
        "software": {
            "preparation": manifest.get("software", {}),
            "local_python": "/tmp/pmnist-env/bin/python (3.11, numpy 1.26.4, torch 2.2.2 CPU)",
            "remote": "PyTorch 2.5.1+cu124, numpy 2.2.6",
        },
        "test_labels_created": False,
    }


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Prepare the permutation-invariant MNIST-medium pool")
    parser.add_argument("--prepare", action="store_true", help="build or verify the pool arrays")
    parser.add_argument("--force", action="store_true", help="rebuild even when outputs exist")
    parser.add_argument("--write-protocol-draft", action="store_true",
                        help="refresh protocol.draft.json from the data manifest")
    arguments = parser.parse_args(argv)
    manifest = prepare(force=arguments.force) if (arguments.prepare or arguments.force) else None
    if manifest is not None:
        for name, spec in sorted(manifest["arrays"].items()):
            print(f"{name}: {spec['dtype']} {tuple(spec['shape'])} sha256={spec['sha256']}")
    if arguments.write_protocol_draft:
        write_json(PROTOCOL_DRAFT_PATH, protocol_draft(manifest))
        print(f"wrote {PROTOCOL_DRAFT_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
