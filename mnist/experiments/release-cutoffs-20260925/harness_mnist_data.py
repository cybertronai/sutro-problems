"""Standalone MNIST / Fashion-MNIST loader for the MNIST-medium time leaderboard.

Trimmed from ``mnist/code/data.py`` of cybertronai/sutro-problems so that this
directory runs with no dependency on that repository. The pixel pipeline is
bit-for-bit the one the competition tiers were built with: convert uint8 to
float32, divide by 255, then resize 28x28 -> size x size by exact separable
box-area averaging, then clip to [0, 1].

Fashion-MNIST is used only for the leaderboard hold-out check. Its files have
the same IDX layout as MNIST and are fetched from the zalandoresearch GitHub
repository; the MD5s below were verified by download on 2026-09-22.
"""

from __future__ import annotations

import gzip
import hashlib
import struct
import urllib.request
from pathlib import Path

import numpy as np

MNIST_BASE = "https://ossci-datasets.s3.amazonaws.com/mnist/"
FASHION_BASE = (
    "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"
)

# key -> (local filename, remote url, md5, expected record count, is_images)
SOURCES = {
    "mnist_train_images": (
        "train-images-idx3-ubyte.gz",
        MNIST_BASE + "train-images-idx3-ubyte.gz",
        "f68b3c2dcbeaaa9fbdd348bbdeb94873",
        60000,
        True,
    ),
    "mnist_train_labels": (
        "train-labels-idx1-ubyte.gz",
        MNIST_BASE + "train-labels-idx1-ubyte.gz",
        "d53e105ee54ea40749a09fcbcd1e9432",
        60000,
        False,
    ),
    "fashion_train_images": (
        "fashion-train-images-idx3-ubyte.gz",
        FASHION_BASE + "train-images-idx3-ubyte.gz",
        "8d4fb7e6c68d591d4c3dfef9ec88bf0d",
        60000,
        True,
    ),
    "fashion_train_labels": (
        "fashion-train-labels-idx1-ubyte.gz",
        FASHION_BASE + "train-labels-idx1-ubyte.gz",
        "25c81989df183df01b3e8a0aad5dffbe",
        60000,
        False,
    ),
}

# The official MNIST test split is deliberately absent: medium draws both the
# train and the test half from the official 60,000-example training split.


def file_hash(path: Path, algorithm: str = "md5") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_source(raw_dir: Path, key: str) -> Path:
    """Cache one canonical gzip file, verifying cached and freshly fetched bytes."""
    filename, url, expected_md5, _, _ = SOURCES[key]
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    destination = raw_dir / filename
    if destination.exists():
        if file_hash(destination) != expected_md5:
            raise ValueError(f"Cached source has an invalid MD5: {destination}")
        return destination
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "sutro-mnist/1.0"})
        with urllib.request.urlopen(request, timeout=120) as response:
            with temporary.open("wb") as stream:
                for block in iter(lambda: response.read(1024 * 1024), b""):
                    stream.write(block)
        actual = file_hash(temporary)
        if actual != expected_md5:
            raise ValueError(f"{filename}: expected MD5 {expected_md5}, received {actual}")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def read_idx(path: Path, expected_count: int, images: bool) -> np.ndarray:
    """Read a gzip IDX file, rejecting wrong magic, dimensions and payload length."""
    with gzip.open(path, "rb") as stream:
        payload = stream.read()
    header_size = 16 if images else 8
    if len(payload) < header_size:
        raise ValueError(f"Truncated IDX header in {path}")
    fields = struct.unpack(">IIII" if images else ">II", payload[:header_size])
    expected_fields = (2051, expected_count, 28, 28) if images else (2049, expected_count)
    if fields != expected_fields:
        raise ValueError(f"Unexpected IDX header {fields} in {path}")
    expected_length = expected_count * (28 * 28 if images else 1)
    if len(payload) - header_size != expected_length:
        raise ValueError(f"Unexpected IDX payload length in {path}")
    values = np.frombuffer(payload, dtype=np.uint8, offset=header_size)
    if images:
        return values.reshape(expected_count, 28, 28)
    if np.any(values > 9):
        raise ValueError(f"Invalid class label in {path}")
    return values.astype(np.int64)


def area_weights(input_size: int, output_size: int) -> np.ndarray:
    """Normalized exact overlap between unit input pixels and output bins."""
    if input_size < 1 or output_size < 1 or output_size > input_size:
        raise ValueError("Sizes must satisfy 1 <= output_size <= input_size")
    left = np.arange(output_size, dtype=np.int64)[:, None] * input_size
    right = left + input_size
    pixel_left = np.arange(input_size, dtype=np.int64)[None, :] * output_size
    pixel_right = pixel_left + output_size
    overlap = np.maximum(0, np.minimum(right, pixel_right) - np.maximum(left, pixel_left))
    return (overlap / input_size).astype(np.float32)


def area_resize(images: np.ndarray, size: int) -> np.ndarray:
    """Resize an N,H,W array by box-area averaging; returns float32 N,size,size."""
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


def load_pool(raw_dir: Path, dataset: str, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (pixels (60000,1,size,size) float32 in [0,1], labels (60000,) int64)."""
    images_key, labels_key = f"{dataset}_train_images", f"{dataset}_train_labels"
    images_path = download_source(raw_dir, images_key)
    labels_path = download_source(raw_dir, labels_key)
    raw_images = read_idx(images_path, SOURCES[images_key][3], images=True)
    labels = read_idx(labels_path, SOURCES[labels_key][3], images=False)
    pixels = raw_images.astype(np.float32)
    pixels /= np.float32(255)
    resized = area_resize(pixels, size)
    np.clip(resized, 0.0, 1.0, out=resized)
    return np.ascontiguousarray(resized[:, None, :, :]), labels
