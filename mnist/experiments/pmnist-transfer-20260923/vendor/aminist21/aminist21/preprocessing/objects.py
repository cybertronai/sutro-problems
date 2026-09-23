"""Prepare three ten-class photographic datasets for procedure-transfer runs.

Use ``python tools/build_data.py --datasets fashion_mnist,cifar10,svhn``.
Only official training splits are used.
The output pools contain all original rows (deduplication belongs to the
shared experiment protocol), SHA256 hashes of native image bytes, and 9x9
single-channel float32 inputs. NumPy is required; SVHN also requires SciPy.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import tarfile
import urllib.request

import numpy as np

from .common import area_resize, array_hash, file_hash, read_idx

DEFAULT_ROOT = Path("data")
NAMES = ("fashion_mnist", "cifar10", "svhn")
FASHION_BASE = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"
SOURCES = {
    "fashion-images.gz": {
        "url": FASHION_BASE + "train-images-idx3-ubyte.gz",
        "md5": "8d4fb7e6c68d591d4c3dfef9ec88bf0d",
        "checksum_reference": "https://github.com/zalandoresearch/fashion-mnist#Get-the-Data",
    },
    "fashion-labels.gz": {
        "url": FASHION_BASE + "train-labels-idx1-ubyte.gz",
        "md5": "25c81989df183df01b3e8a0aad5dffbe",
        "checksum_reference": "https://github.com/zalandoresearch/fashion-mnist#Get-the-Data",
    },
    "cifar-10-binary.tar.gz": {
        "url": "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz",
        "download_urls": [
            "https://data.brainchip.com/dataset-mirror/cifar10/cifar-10-binary.tar.gz",
            "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz",
        ],
        "md5": "c32a1d4ab5d03f1284b67883e8d87530",
        "checksum_reference": "https://www.cs.toronto.edu/~kriz/cifar.html",
    },
    "svhn-train_32x32.mat": {
        "url": "http://ufldl.stanford.edu/housenumbers/train_32x32.mat",
        "md5": "e26dedcc434d2e4c54c9b2d4a06d8373",
        "checksum_reference": "https://docs.pytorch.org/vision/main/_modules/torchvision/datasets/svhn.html",
    },
}


def download(raw_dir: Path, filename: str) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / filename
    spec = SOURCES[filename]
    if path.exists():
        if file_hash(path, "md5") != spec["md5"]:
            raise ValueError(f"Cached source checksum mismatch: {path}")
        return path
    part = path.with_suffix(path.suffix + ".part")
    print(f"Downloading {filename} from {spec['url']}", flush=True)
    try:
        urls = spec.get("download_urls", [spec["url"]])
        for index, download_url in enumerate(urls):
            try:
                request = urllib.request.Request(download_url, headers={"User-Agent": "aminist-21-validation/1.0"})
                with urllib.request.urlopen(request, timeout=120) as response, part.open("wb") as stream:
                    for block in iter(lambda: response.read(1024 * 1024), b""):
                        stream.write(block)
                break
            except OSError:
                part.unlink(missing_ok=True)
                if index == len(urls) - 1:
                    raise
        actual = file_hash(part, "md5")
        if actual != spec["md5"]:
            raise ValueError(f"Checksum mismatch for {filename}: {actual}")
        part.replace(path)
        path.with_suffix(path.suffix + ".retrieval.json").write_text(json.dumps({
            "upstream_url": spec["url"], "retrieved_from": download_url,
            "md5": actual, "sha256": file_hash(path),
            "identity": "Downloaded bytes match the original release's pinned MD5",
        }, indent=2) + "\n")
    finally:
        part.unlink(missing_ok=True)
    print(f"Verified {filename}", flush=True)
    return path


def read_source(name: str, root: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    raw = root / "raw"
    if name == "fashion_mnist":
        files = ("fashion-images.gz", "fashion-labels.gz")
        with ThreadPoolExecutor(max_workers=2) as executor:
            paths = list(executor.map(lambda filename: download(raw, filename), files))
        native = read_idx(paths[0], 60000, images=True)
        labels = read_idx(paths[1], 60000, images=False)
        metadata = {
            "display_name": "Fashion-MNIST",
            "dataset_page": "https://github.com/zalandoresearch/fashion-mnist",
            "description": "Grayscale photographs of clothing products; no handwriting or MNIST examples.",
            "class_names": ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"],
            "source_order": "Original official training IDX row order",
            "label_mapping": "Original labels 0..9 unchanged",
        }
    elif name == "cifar10":
        files = ("cifar-10-binary.tar.gz",)
        path = download(raw, files[0])
        batches = []
        with tarfile.open(path, "r:gz") as archive:
            for batch in range(1, 6):
                member = archive.extractfile(f"cifar-10-batches-bin/data_batch_{batch}.bin")
                if member is None:
                    raise ValueError(f"Missing CIFAR-10 batch {batch}")
                values = np.frombuffer(member.read(), dtype=np.uint8)
                if values.size != 10000 * 3073:
                    raise ValueError(f"Invalid CIFAR-10 batch {batch} length")
                batches.append(values.reshape(10000, 3073))
        values = np.concatenate(batches)
        labels = values[:, 0].astype(np.int64)
        native = np.ascontiguousarray(values[:, 1:].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1))
        metadata = {
            "display_name": "CIFAR-10 (grayscale)",
            "dataset_page": "https://www.cs.toronto.edu/~kriz/cifar.html",
            "description": "Color photographs of animals and vehicles, converted to grayscale.",
            "class_names": ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"],
            "source_order": "Concatenated official data_batch_1.bin through data_batch_5.bin, preserving row order",
            "label_mapping": "Original labels 0..9 unchanged",
        }
    elif name == "svhn":
        from scipy.io import loadmat

        files = ("svhn-train_32x32.mat",)
        path = download(raw, files[0])
        values = loadmat(path)
        native = np.ascontiguousarray(values["X"].transpose(3, 0, 1, 2))
        labels = values["y"].reshape(-1).astype(np.int64) % 10
        if native.shape != (73257, 32, 32, 3):
            raise ValueError(f"Unexpected SVHN shape {native.shape}")
        metadata = {
            "display_name": "SVHN (grayscale)",
            "dataset_page": "http://ufldl.stanford.edu/housenumbers/",
            "description": "Cropped color photographs of house-number digits, converted to grayscale; no handwriting.",
            "class_names": [str(label) for label in range(10)],
            "source_order": "Original official train_32x32.mat fourth-axis row order",
            "label_mapping": "Original label 10 (digit zero) remapped to 0; labels 1..9 unchanged",
        }
    else:
        raise ValueError(f"Unknown dataset {name}")
    metadata["sources"] = [
        {"filename": filename, **SOURCES[filename], "sha256": file_hash(raw / filename), "bytes": (raw / filename).stat().st_size}
        for filename in files
    ]
    return native, labels, metadata


def prepare(name: str, root: Path = DEFAULT_ROOT) -> dict:
    root = Path(root)
    native, labels, metadata = read_source(name, root)
    if native.dtype != np.uint8 or labels.dtype != np.int64:
        raise ValueError("Unexpected raw source dtypes")
    if not np.array_equal(np.unique(labels), np.arange(10)):
        raise ValueError("Expected exactly ten labels 0..9")
    count = len(native)
    images = np.empty((count, 1, 9, 9), dtype=np.float32)
    for start in range(0, count, 2048):
        values = native[start:start + 2048].astype(np.float32) / np.float32(255)
        if values.ndim == 4:
            values = (values * np.array([0.299, 0.587, 0.114], dtype=np.float32)).sum(axis=-1)
        images[start:start + len(values), 0] = area_resize(values, 9)
    example_hashes = np.asarray([
        hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()
        for image in native
    ], dtype="S64")
    source_indices = np.arange(count, dtype=np.int64)
    if not np.all(np.isfinite(images)) or images.min() < 0 or images.max() > 1.000001:
        raise ValueError("Images must be finite and within [0,1]")
    pool_dir = root / "raw_pools"
    pool_dir.mkdir(parents=True, exist_ok=True)
    destination = pool_dir / f"{name}.npz"
    part = destination.with_suffix(".npz.part")
    with part.open("wb") as stream:
        np.savez_compressed(stream, images=images, labels=labels,
                            example_hashes=example_hashes, source_indices=source_indices)
    part.replace(destination)
    metadata.update({
        "name": name,
        "format_version": 1,
        "source_split": "Official training split only",
        "pool_size": count,
        "class_counts": np.bincount(labels, minlength=10).tolist(),
        "native_shape": list(native.shape[1:]),
        "native_dtype": str(native.dtype),
        "image_shape": list(images.shape),
        "image_dtype": str(images.dtype),
        "labels_dtype": str(labels.dtype),
        "preprocessing": {
            "normalization": "Convert uint8 to float32 and divide by 255",
            "grayscale": "RGB luminance 0.299 R + 0.587 G + 0.114 B" if native.ndim == 4 else "Already grayscale; no change",
            "resize": "mnist.code.data.area_resize: exact separable box-area averaging to 9x9",
            "inversion": "None; preserve original intensity polarity",
            "additional_transforms": "None",
        },
        "example_hash_definition": "SHA256 of C-order native uint8 image bytes only; native RGB represented H,W,C before hashing; excludes labels",
        "exact_native_duplicate_rows": count - len(np.unique(example_hashes)),
        "deduplication": "Not applied here; all original rows retained for common experiment protocol",
        "array_sha256": {"images": array_hash(images), "labels": array_hash(labels),
                          "example_hashes": array_hash(example_hashes), "source_indices": array_hash(source_indices)},
        "pool_sha256": file_hash(destination),
    })
    (pool_dir / f"{name}.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Prepared {name}: {count} rows, class counts {metadata['class_counts']}, "
          f"exact native duplicates {metadata['exact_native_duplicate_rows']}", flush=True)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--names", nargs="+", choices=NAMES, default=list(NAMES))
    args = parser.parse_args()
    with ThreadPoolExecutor(max_workers=len(args.names)) as executor:
        list(executor.map(lambda name: prepare(name, args.root), args.names))


if __name__ == "__main__":
    main()
