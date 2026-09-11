"""Download verified MNIST and create reproducible competition tiers.

Run ``python -m mnist.code.data --output mnist/data``. Only NumPy is required.
Small and medium use disjoint samples from the official training split; large
uses both complete official splits. Use ``--profile reference-20260910`` only to
reproduce the earlier 1,000/10,000-example reference experiments.
Pixels use exact separable box-area averaging, including fractional boundary
pixels when 28 is not divisible by the output resolution.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import struct
import urllib.request

import numpy as np


DEFAULT_SEED = 20260910
DEFAULT_PROFILE = "competition-v2"
REFERENCE_PROFILE = "reference-20260910"
MEDIUM_ERROR_PROFILE = "medium-error-targets-v1"
SOURCE_BASE = "https://ossci-datasets.s3.amazonaws.com/mnist/"
SOURCES = {
    "train_images": (
        "train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"
    ),
    "train_labels": (
        "train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"
    ),
    "test_images": (
        "t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3"
    ),
    "test_labels": (
        "t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c"
    ),
}
TIERS = {
    "small": {
        "size": 3, "train_count": 600, "test_count": 600,
        "train_source": "train", "test_source": "train",
        "train_offset": 0, "test_offset": 6000,
    },
    "medium": {
        "size": 9, "train_count": 6000, "test_count": 6000,
        "train_source": "train", "test_source": "train",
        "train_offset": 0, "test_offset": 6000,
    },
    "large": {
        "size": 28, "train_count": 60000, "test_count": 10000,
        "train_source": "train", "test_source": "test",
        "train_offset": 0, "test_offset": 0,
    },
}
REFERENCE_TIERS = {
    name: {
        **tier,
        "train_count": {"small": 1000, "medium": 10000, "large": 60000}[name],
        "test_count": {"small": 1000, "medium": 10000, "large": 10000}[name],
        "test_source": "test", "test_offset": 0,
    }
    for name, tier in TIERS.items()
}
MEDIUM_ERROR_TIERS = {name: dict(tier) for name, tier in TIERS.items()}
MEDIUM_ERROR_TIERS['medium'].update(train_count=10000, test_count=10000, test_offset=10000)
PROFILES = {DEFAULT_PROFILE: TIERS, REFERENCE_PROFILE: REFERENCE_TIERS,
            MEDIUM_ERROR_PROFILE: MEDIUM_ERROR_TIERS}


def file_hash(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_source(raw_dir: Path, filename: str, expected_md5: str) -> Path:
    """Cache a canonical gzip file, verifying both cached and new downloads."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    destination = raw_dir / filename
    if destination.exists():
        if file_hash(destination, "md5") != expected_md5:
            raise ValueError(f"Cached MNIST source has an invalid MD5: {destination}")
        return destination
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        request = urllib.request.Request(
            SOURCE_BASE + filename, headers={"User-Agent": "sutro-mnist/1.0"}
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            with temporary.open("wb") as stream:
                for block in iter(lambda: response.read(1024 * 1024), b""):
                    stream.write(block)
        actual_md5 = file_hash(temporary, "md5")
        if actual_md5 != expected_md5:
            raise ValueError(
                f"MNIST source {filename}: expected MD5 {expected_md5}, "
                f"received {actual_md5}"
            )
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def read_idx(path: Path, expected_count: int, images: bool) -> np.ndarray:
    """Read gzip IDX, rejecting wrong magic, dimensions, and payload length."""
    with gzip.open(path, "rb") as stream:
        payload = stream.read()
    header_size = 16 if images else 8
    if len(payload) < header_size:
        raise ValueError(f"Truncated IDX header in {path}")
    fields = struct.unpack(">IIII" if images else ">II", payload[:header_size])
    expected_fields = (
        (2051, expected_count, 28, 28) if images else (2049, expected_count)
    )
    if fields != expected_fields:
        raise ValueError(f"Unexpected IDX header {fields} in {path}")
    expected_length = expected_count * (28 * 28 if images else 1)
    if len(payload) - header_size != expected_length:
        raise ValueError(f"Unexpected IDX payload length in {path}")
    values = np.frombuffer(payload, dtype=np.uint8, offset=header_size)
    if images:
        return values.reshape(expected_count, 28, 28)
    if np.any(values > 9):
        raise ValueError(f"Invalid MNIST class label in {path}")
    return values.astype(np.int64)


def area_weights(input_size: int, output_size: int) -> np.ndarray:
    """Return normalized exact overlap of unit input pixels and output bins."""
    if input_size < 1 or output_size < 1 or output_size > input_size:
        raise ValueError("Sizes must satisfy 1 <= output_size <= input_size")
    # Integer-scaled endpoints avoid boundary rounding before the division.
    left = np.arange(output_size, dtype=np.int64)[:, None] * input_size
    right = left + input_size
    pixel_left = np.arange(input_size, dtype=np.int64)[None, :] * output_size
    pixel_right = pixel_left + output_size
    overlap = np.maximum(0, np.minimum(right, pixel_right) - np.maximum(left, pixel_left))
    return (overlap / input_size).astype(np.float32)


def area_resize(images: np.ndarray, size: int) -> np.ndarray:
    """Resize an N,H,W array by box-area averaging; return float32 N,size,size."""
    images = np.asarray(images)
    if images.ndim != 3:
        raise ValueError("Expected images with shape N,H,W")
    if images.shape[1:] == (size, size):
        return np.ascontiguousarray(images, dtype=np.float32)
    height_weights = area_weights(images.shape[1], size)
    width_weights = area_weights(images.shape[2], size)
    values = images.astype(np.float32, copy=False)
    result = np.matmul(np.matmul(height_weights, values), width_weights.T)
    return np.ascontiguousarray(result, dtype=np.float32)


def source_permutations(
    train_count: int = 60000, test_count: int = 10000, seed: int = DEFAULT_SEED
) -> tuple[np.ndarray, np.ndarray]:
    """Independently shuffle official splits, preserving reference-era ordering."""
    train_seed, test_seed = np.random.SeedSequence(seed).spawn(2)
    train_rng = np.random.Generator(np.random.PCG64(train_seed))
    test_rng = np.random.Generator(np.random.PCG64(test_seed))
    return (
        train_rng.permutation(train_count).astype(np.int64),
        test_rng.permutation(test_count).astype(np.int64),
    )


def array_hash(array: np.ndarray) -> str:
    """Hash canonical C-order little-endian contents, independent of ZIP metadata."""
    canonical = np.ascontiguousarray(array.astype(array.dtype.newbyteorder("<"), copy=False))
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def prepare(
    output_dir: Path, seed: int = DEFAULT_SEED, profile: str = DEFAULT_PROFILE
) -> dict:
    """Create NPZ tiers and a manifest identifying the profile and source rows.

    The competition profile reserves official-training permutation rows
    [0,6000) for train and [6000,12000) for test. Small uses the first 600
    examples of each allocation, so the smaller tiers are nested without
    train/test leakage between them. Large keeps all 60,000/10,000 examples.
    The reference profile reproduces the original official-split prefixes.
    """
    if profile not in PROFILES:
        raise ValueError(f"Unknown MNIST profile {profile!r}; choose from {tuple(PROFILES)}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw"
    raw_arrays = {}
    manifest = {
        "format_version": 2,
        "dataset": "MNIST",
        "profile": profile,
        "seed": seed,
        "sampling": {
            "algorithm": "numpy.random.PCG64",
            "seed_derivation": "SeedSequence(seed).spawn(2): child 0 official train, child 1 official test",
            "selection": (
                "Without replacement; small/medium train use official-training permutation "
                "prefixes and test use the declared per-tier offsets; large uses both "
                "complete official splits"
                if profile != REFERENCE_PROFILE else
                "Without replacement; nested prefixes of independent official-split permutations"
            ),
            "source_splits": {
                "train": "Official MNIST 60,000-example training split",
                "test": "Official MNIST 10,000-example test split",
            },
            "offsets": "Per-tier offsets into the seeded permutation of each source split",
            "indices": "Zero-based rows of the original official split, before resizing",
        },
        "resize": {
            "method": "Exact separable box-area overlap averaging",
            "boundary": "Pixel i covers [i,i+1); output j covers [j*28/size,(j+1)*28/size)",
            "normalization": "Convert uint8 to float32 and divide by 255 before resizing",
            "implementation": "Integer-scaled overlap endpoints, float32 weights and matrix multiplication",
            "pixel_dtype": "float32",
            "layout": "N,1,H,W",
            "range": [0.0, 1.0],
        },
        "sources": {},
        "tiers": {},
    }
    for name, (filename, expected_md5) in SOURCES.items():
        path = download_source(raw_dir, filename, expected_md5)
        manifest["sources"][name] = {
            "url": SOURCE_BASE + filename,
            "path": str(Path("raw") / filename),
            "md5": expected_md5,
            "sha256": file_hash(path),
            "bytes": path.stat().st_size,
        }
        raw_arrays[name] = read_idx(
            path, 60000 if name.startswith("train") else 10000,
            images=name.endswith("images"),
        )
    train_order, test_order = source_permutations(seed=seed)
    source_orders = {"train": train_order, "test": test_order}
    for name, tier in PROFILES[profile].items():
        arrays = {}
        for split in ("train", "test"):
            source = tier[f"{split}_source"]
            offset = tier[f"{split}_offset"]
            count = tier[f"{split}_count"]
            indices = source_orders[source][offset:offset + count].copy()
            source_pixels = raw_arrays[f"{source}_images"][indices].astype(np.float32)
            source_pixels /= np.float32(255)
            images = area_resize(source_pixels, tier["size"])
            # Float32 reductions can produce a value one ULP beyond endpoints.
            np.clip(images, 0.0, 1.0, out=images)
            arrays[f"{split}_images"] = images[:, None, :, :]
            arrays[f"{split}_labels"] = raw_arrays[f"{source}_labels"][indices]
            arrays[f"{split}_indices"] = indices
        destination = output_dir / f"{name}.npz"
        with destination.with_suffix(".npz.part").open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        destination.with_suffix(".npz.part").replace(destination)
        manifest["tiers"][name] = {
            **tier,
            "path": destination.name,
            "sha256": file_hash(destination),
            "bytes": destination.stat().st_size,
            "arrays": {
                key: {
                    "shape": list(value.shape), "dtype": str(value.dtype),
                    "sha256_c_order_little_endian": array_hash(value),
                }
                for key, value in arrays.items()
            },
            "class_histograms": {
                split: np.bincount(arrays[f"{split}_labels"], minlength=10).tolist()
                for split in ("train", "test")
            },
        }
        print(f"Prepared {name}: {tier['train_count']} train / {tier['test_count']} test, "
              f"{tier['size']}x{tier['size']} -> {destination}", flush=True)
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("mnist/data"))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--profile", choices=tuple(PROFILES), default=DEFAULT_PROFILE)
    arguments = parser.parse_args()
    prepare(arguments.output, arguments.seed, arguments.profile)


if __name__ == "__main__":
    main()
