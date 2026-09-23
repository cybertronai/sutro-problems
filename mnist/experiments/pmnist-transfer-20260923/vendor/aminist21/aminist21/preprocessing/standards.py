"""Prepare MNIST, related EMNIST controls, KMNIST, USPS and recovered QMNIST.

Use ``python tools/build_data.py --datasets all``. Completed source files are
checked before reuse. Additional EMNIST IDX
members are fetched from the official ZIP by checked HTTP byte ranges.
Pools retain original rows and per-image hashes; no deduplication or sampling
occurs here. In particular QMNIST's recovered 50k are a new repartitioned
learning task, not a held-out evaluation of a pretrained MNIST checkpoint.
"""

from __future__ import annotations

import argparse
import bz2
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import zipfile
import zlib

import numpy as np

from .common import SOURCES as MNIST_SOURCES, SOURCE_BASE as MNIST_BASE
from .common import area_resize, array_hash, download_source, file_hash
from . import sources as previous

DEFAULT_ROOT = Path("data")
NAMES = ("mnist", "kmnist", "emnist_letters_aj", "emnist_letters_kt",
         "emnist_balanced_aj", "emnist_digits", "emnist_mnist", "usps", "qmnist_recovered")
NEW_EMNIST_SPLITS = ("balanced", "digits", "mnist")
# Pinned after CRC32/length-checked extraction from the official NIST ZIP.
PINNED_NEW_EMNIST_SHA256 = {
    "emnist-balanced-train-images-idx3-ubyte.gz": "74295e75e4e9fa2c102147460ee7eb763653d5132d883a84fb88932bfb8d2db4",
    "emnist-balanced-train-labels-idx1-ubyte.gz": "b46d41e5156a0a8aaaa4d18979fdd1588850fa8571018acffd5d93b325a7d6ed",
    "emnist-digits-train-images-idx3-ubyte.gz": "f7c95004d14d81af89522e67d8f0c781dfb3bd544181c81bfe1eb6b41c35b726",
    "emnist-digits-train-labels-idx1-ubyte.gz": "6638a6ff5fe2eefd9cca2471995b46c82986b8e4b2a70d2f5ce8e05d7edb319e",
    "emnist-mnist-train-images-idx3-ubyte.gz": "5f9942f441031c0f1f2e4d162059fd6e19ea808b34c328b6d16cab0ed24c78f2",
    "emnist-mnist-train-labels-idx1-ubyte.gz": "00c6d3c2342fdffb1711e0b5656cac4ce5862d2826399d383a9bac07d612e9f8",
}


def cached_source(filename: str, root: Path) -> Path:
    for path in (root / "raw" / filename,):
        if path.exists():
            previous._verify(path, previous.SOURCES.get(filename, {}))
            return path
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    if filename.startswith("emnist-"):
        download_new_emnist(root, [filename.split("-")[1]])
        path = raw / filename
        previous._verify(path, {})
        return path
    return previous._download(raw, filename)


def source_info(path: Path) -> dict:
    source = previous.SOURCES.get(path.name, {
        "url": previous.EMNIST_URL, "member": "gzip/" + path.name,
    })
    return {"filename": path.name, **source, "sha256": file_hash(path), "bytes": path.stat().st_size}


def download_new_emnist(root: Path, splits: list[str]) -> None:
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    members = [f"gzip/emnist-{split}-train-{kind}-idx{dim}-ubyte.gz"
               for split in splits for kind, dim in (("images", 3), ("labels", 1))]
    with previous._HTTPRangeReader(previous.EMNIST_URL) as remote, zipfile.ZipFile(remote) as archive:
        for member in members:
            info = archive.getinfo(member)
            path = raw / Path(member).name
            if not path.exists():
                print(f"Downloading official EMNIST ZIP member {member}", flush=True)
                part = path.with_suffix(path.suffix + ".part")
                try:
                    part.write_bytes(archive.read(member))  # Also verifies ZIP CRC32.
                    part.replace(path)
                finally:
                    part.unlink(missing_ok=True)
            data = path.read_bytes()
            if len(data) != info.file_size or zlib.crc32(data) != info.CRC:
                raise ValueError(f"Invalid EMNIST member CRC32/length: {path}")
            expected = PINNED_NEW_EMNIST_SHA256.get(path.name)
            if expected and file_hash(path) != expected:
                raise ValueError(f"Invalid EMNIST member SHA256: {path}")
            sidecar = {"url": previous.EMNIST_URL, "member": member,
                       "archive_bytes": remote.size, "archive_etag": remote.etag,
                       "member_crc32": f"{info.CRC:08x}", "bytes": len(data),
                       "sha256": file_hash(path),
                       "verification": "Official ZIP central-directory CRC32 and size; pinned SHA256 when available"}
            path.with_suffix(path.suffix + ".json").write_text(json.dumps(sidecar, indent=2) + "\n")


def read_usps_native(path: Path) -> tuple[np.ndarray, np.ndarray]:
    lines = bz2.decompress(path.read_bytes()).decode("ascii").splitlines()
    images = np.zeros((len(lines), 256), dtype="<f4")
    labels = np.empty(len(lines), dtype=np.int64)
    for row, line in enumerate(lines):
        fields = line.split()
        labels[row] = int(fields[0]) - 1
        for field in fields[1:]:
            index, value = field.split(":")
            images[row, int(index) - 1] = float(value)
    if images.min() < -1 or images.max() > 1:
        raise ValueError("USPS pixels outside [-1,1]")
    return images.reshape(-1, 16, 16), labels


def read_source(name: str, root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    metadata = {"name": name, "class_names": [str(i) for i in range(10)],
                "source_split": "Official training split only",
                "orientation": "Original orientation unchanged",
                "label_mapping": "Original labels 0..9 unchanged"}
    if name == "mnist":
        paths = [download_source(root / "raw", *MNIST_SOURCES[f"train_{kind}"])
                 for kind in ("images", "labels")]
        native, labels = [previous._idx(path) for path in paths]
        metadata.update(display_name="MNIST", dataset_page="http://yann.lecun.com/exdb/mnist/",
                        relationship="Original MNIST control; no weights reused",
                        sources=[{"filename": path.name, "url": MNIST_BASE + path.name,
                                  "md5": MNIST_SOURCES[f"train_{kind}"][1],
                                  "sha256": file_hash(path), "bytes": path.stat().st_size}
                                 for kind, path in zip(("images", "labels"), paths)])
        indices = np.arange(len(labels), dtype=np.int64)
    elif name == "kmnist":
        paths = [cached_source(f"kmnist-train-{kind}.npz", root) for kind in ("imgs", "labels")]
        arrays = []
        for path in paths:
            with np.load(path, allow_pickle=False) as archive:
                arrays.append(archive["arr_0"])
        native, labels = arrays
        metadata.update(display_name="KMNIST", dataset_page="https://github.com/rois-codh/kmnist",
                        class_names=["お", "き", "す", "つ", "な", "は", "ま", "や", "れ", "を"],
                        sources=[source_info(path) for path in paths],
                        relationship="Independent Japanese cursive character source; MNIST-format images")
        indices = np.arange(len(labels), dtype=np.int64)
    elif name.startswith("emnist_"):
        split = "letters" if name.startswith("emnist_letters_") else "balanced" if name == "emnist_balanced_aj" else name.removeprefix("emnist_")
        paths = []
        for kind, dim in (("images", 3), ("labels", 1)):
            filename = f"emnist-{split}-train-{kind}-idx{dim}-ubyte.gz"
            paths.append(cached_source(filename, root) if split == "letters" else root / "raw" / filename)
        all_native, all_labels = [previous._idx(path) for path in paths]
        lower, upper = (1, 10) if name == "emnist_letters_aj" else (11, 20) if name == "emnist_letters_kt" else (10, 19) if name == "emnist_balanced_aj" else (0, 9)
        keep = (all_labels >= lower) & (all_labels <= upper)
        indices = np.flatnonzero(keep).astype(np.int64)
        native = np.ascontiguousarray(all_native[keep].transpose(0, 2, 1))
        labels = all_labels[keep].astype(np.int64) - lower
        classes = list("KLMNOPQRST") if name == "emnist_letters_kt" else list("ABCDEFGHIJ") if name.endswith("_aj") else [str(i) for i in range(10)]
        metadata.update(display_name={"emnist_letters_aj": "EMNIST Letters A–J", "emnist_letters_kt": "EMNIST Letters K–T", "emnist_balanced_aj": "EMNIST Balanced A–J", "emnist_digits": "EMNIST Digits", "emnist_mnist": "EMNIST MNIST"}[name],
                        dataset_page="https://www.nist.gov/itl/products-and-services/emnist-dataset",
                        class_names=classes, sources=[source_info(path) for path in paths],
                        orientation="Transpose each raw IDX image (swap height and width) before hashing and resizing",
                        label_mapping=f"Official {split} labels {lower}..{upper} selected before experiments; subtract {lower}",
                        relationship="Related NIST handwriting control; not an independent MNIST-family source")
        if split != "letters":
            for source, path in zip(metadata["sources"], paths):
                source.update(json.loads(path.with_suffix(path.suffix + ".json").read_text()))
    elif name == "usps":
        paths = [cached_source(filename, root) for filename in ("usps.bz2", "usps.t.bz2")]
        parts = [read_usps_native(path) for path in paths]
        native = np.concatenate([part[0] for part in parts])
        labels = np.concatenate([part[1] for part in parts])
        indices = np.arange(len(labels), dtype=np.int64)
        metadata.update(display_name="USPS", dataset_page="https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html#usps",
                        sources=[source_info(path) for path in paths],
                        source_split="Concatenate official training (7,291) then test (2,007), repartitioned for fresh training",
                        source_index_ranges={"official_train": [0, 7291], "official_test": [7291, 9298]},
                        label_mapping="LIBSVM labels 1..10 represent digits 0..9; subtract 1",
                        relationship="Independent US postal handwriting, smaller than required 20,000 pool")
    elif name == "qmnist_recovered":
        paths = [cached_source(filename, root) for filename in ("qmnist-test-images-idx3-ubyte.gz", "qmnist-test-labels-idx2-int.gz")]
        native = previous._idx(paths[0])[10000:60000]
        labels = previous._idx(paths[1])[10000:60000, 0]
        indices = np.arange(10000, 60000, dtype=np.int64)
        metadata.update(display_name="QMNIST recovered 50k", dataset_page="https://github.com/facebookresearch/qmnist",
                        sources=[source_info(path) for path in paths],
                        source_split="Official QMNIST test rows [10000,60000), repartitioned into a new learning task",
                        source_index_ranges={"official_qmnist_test": [10000, 60000]},
                        relationship="Additional recovered NIST examples; excludes the reconstructed MNIST test10k; related source",
                        evaluation_scope="Freshly trained procedure with new train/evaluation draws; not held-out evaluation of prior MNIST weights")
    else:
        raise ValueError(f"Unknown dataset {name}")
    return native, labels.astype(np.int64), indices, metadata


def prepare(name: str, root: Path = DEFAULT_ROOT) -> dict:
    root = Path(root)
    if name.startswith("emnist_"):
        split = "letters" if name.startswith("emnist_letters_") else "balanced" if name == "emnist_balanced_aj" else name[len("emnist_"):]
        expected = [root / "raw" / f"emnist-{split}-train-{kind}-idx{dim}-ubyte.gz" for kind, dim in (("images", 3), ("labels", 1))]
        if not all(path.exists() for path in expected):
            download_new_emnist(root, [split])
        for path in expected:
            previous._verify(path, {})
            checksum = PINNED_NEW_EMNIST_SHA256.get(path.name)
            if checksum and file_hash(path) != checksum:
                raise ValueError(f"Invalid EMNIST member SHA256: {path}")
    native, labels, indices, metadata = read_source(name, root)
    if not np.array_equal(np.unique(labels), np.arange(10)):
        raise ValueError(f"Expected ten classes: {name}")
    count = len(labels)
    images = np.empty((count, 1, 9, 9), dtype=np.float32)
    for start in range(0, count, 2048):
        values = native[start:start + 2048].astype(np.float32)
        values = (values + 1) / 2 if name == "usps" else values / np.float32(255)
        images[start:start + len(values), 0] = area_resize(values, 9)
    np.clip(images, 0, 1, out=images)
    hashes = np.asarray([hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest()
                         for row in native], dtype="S64")
    arrays = {"images": images, "labels": labels, "example_hashes": hashes, "source_indices": indices}
    pool = root / "raw_pools"
    pool.mkdir(parents=True, exist_ok=True)
    path = pool / f"{name}.npz"
    part = path.with_suffix(".npz.part")
    with part.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    part.replace(path)
    metadata.update(format_version=1, pool_size=count, class_counts=np.bincount(labels, minlength=10).tolist(),
                    native_shape=list(native.shape[1:]), native_dtype=str(native.dtype),
                    image_shape=list(images.shape), image_dtype=str(images.dtype), labels_dtype=str(labels.dtype),
                    example_hash_definition="SHA256 native C-order image bytes, excluding labels; EMNIST orientation corrected; USPS little-endian float32 in original [-1,1], all others uint8",
                    exact_native_duplicate_rows=count - len(np.unique(hashes)),
                    deduplication="Not applied here; common protocol decides after inspecting all pools",
                    preprocessing={"normalization": "Source float32 [-1,1] mapped by (x+1)/2" if name == "usps" else "uint8 converted to float32 and divided by 255",
                                   "resize": "mnist.code.data.area_resize exact box-area averaging to9x9, clip roundoff to [0,1]",
                                   "orientation": metadata["orientation"], "inversion": "None"},
                    array_sha256={key: array_hash(value) for key, value in arrays.items()}, pool_sha256=file_hash(path))
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    print(f"Prepared {name}: {count} rows, {metadata['exact_native_duplicate_rows']} exact native duplicates", flush=True)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--names", nargs="+", choices=NAMES, default=list(NAMES))
    args = parser.parse_args()
    new_names = [name for name in args.names if name in ("emnist_balanced_aj", "emnist_digits", "emnist_mnist")]
    def new_work() -> None:
        splits = ["balanced" if name == "emnist_balanced_aj" else name.removeprefix("emnist_") for name in new_names]
        download_new_emnist(args.root, splits)
        for name in new_names:
            prepare(name, args.root)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(new_work) if new_names else None
        for name in args.names:
            if name not in new_names:
                prepare(name, args.root)
        if future:
            future.result()


if __name__ == "__main__":
    main()
