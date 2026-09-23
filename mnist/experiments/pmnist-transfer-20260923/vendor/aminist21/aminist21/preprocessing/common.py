"""Canonical resize arithmetic vendored from sutro-problems mnist/code/data.py.

Resize functions are kept byte-identical to the original implementation.
"""
from __future__ import annotations
import gzip, hashlib, struct, urllib.request
from pathlib import Path
import numpy as np

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

def array_hash(array: np.ndarray) -> str:
    """Hash canonical C-order little-endian contents, independent of ZIP metadata."""
    canonical = np.ascontiguousarray(array.astype(array.dtype.newbyteorder("<"), copy=False))
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()

