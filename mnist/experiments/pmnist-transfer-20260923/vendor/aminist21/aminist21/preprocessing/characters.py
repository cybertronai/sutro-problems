"""Prepare independent-script and font character pools for procedure transfer.

Run from any directory with Python 3.10+, numpy and Pillow.  Each pool contains
images float32 [N,1,9,9], labels int64 [N], native example_hashes S64 [N], and
source_indices int64 [N].  No fitting, class search, or deduplication is done
here; the experiment runner deduplicates native images before drawing splits.
All ten-class choices were fixed before examining training results.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import urllib.request
import zipfile

import numpy as np
from PIL import Image

from .common import area_resize, array_hash, file_hash

DEFAULT_ROOT = Path("data")
NAMES = ("k49_10", "kannada_digits", "dig_mnist", "devanagari_digits", "madbase", "notmnist_large", "omniglot_gurmukhi10")
K49_IDS = [0, 1, 2, 5, 7, 9, 10, 11, 15, 18]
KANNADA_BASE = "https://raw.githubusercontent.com/vinayprabhu/Kannada_MNIST/master/data/output_tensors/MNIST_format/"
SOURCES = {
    **{name: "https://codh.rois.ac.jp/kmnist/dataset/k49/" + name for name in ("k49-train-imgs.npz", "k49-train-labels.npz", "k49_classmap.csv")},
    **{name: KANNADA_BASE + name for name in ("X_kannada_MNIST_train.npz", "y_kannada_MNIST_train.npz", "X_dig_MNIST.npz", "y_dig_MNIST.npz")},
    "devanagari.zip": "https://archive.ics.uci.edu/static/public/389/devanagari+handwritten+character+dataset.zip",
    "MAHDBase_TrainingSet.rar": "https://datacenter.aucegypt.edu/shazeem/Files/MAHDBase_TrainingSet.rar",
    "notMNIST_large.tar.gz": "http://yaroslavvb.com/upload/notMNIST/notMNIST_large.tar.gz",
    "omniglot-images_evaluation.zip": "https://raw.githubusercontent.com/brendenlake/omniglot/master/python/images_evaluation.zip",
}
PINNED_SHA256 = {
    "k49-train-imgs.npz": "1c42adc463ed8efe598cf002f6ecafbca6d8c38f0551f7a35e0b23755fca1c8d",
    "k49-train-labels.npz": "fbfef4750ce9aa70b6072f0bca7daa9e80e2d79b379d5ae923265a63396260e4",
    "k49_classmap.csv": "95f5a2cfbfba1721f566059cf77c22808898b63044a5f9a08de9d8a2950eed84",
    "X_kannada_MNIST_train.npz": "705210ec1a9e83be08defa8cc25b91a5b88232ffee1e4a70fcba49b2dde798ab",
    "y_kannada_MNIST_train.npz": "462e31f9c9b78ac0e920d6dbdc7ee3e8260ed39fb4d3a48cf45dca0e03a1c415",
    "X_dig_MNIST.npz": "19b485746106c3cb4f0aaac2ae5aa798eea87a51c4555a2e87a1ac3b930fa864",
    "y_dig_MNIST.npz": "20f3cbae89db160a1a0537c7010de96f356fcf7b27b45b5990b140c66d6ad897",
    "devanagari.zip": "752e6a2e6b0d1b475375e6a9db937918bbbf7c6b7d137dbe234264029fd3b88d",
    "MAHDBase_TrainingSet.rar": "ae2d5bcd658320eaf054de1ba46aac187a0470830d685258a0a2b0659c8d2eb8",
    "notMNIST_large.tar.gz": "1e8261b75484b520dc76027f5c22af50fe608e3bb16277caec086eb928f9b61c",
    "omniglot-images_evaluation.zip": "1f61a8f3366785b057fc117d9228e78a16e3d976c8953b2a10fcc74cf0609cee",
}


def download(raw: Path, filename: str) -> Path:
    path = raw / filename
    if path.exists():
        if file_hash(path) != PINNED_SHA256[filename]:
            raise ValueError(f"Source SHA256 mismatch: {path}")
        return path
    partial = path.with_suffix(path.suffix + ".part")
    print("Downloading", filename, flush=True)
    try:
        if filename == "notMNIST_large.tar.gz":
            # The creator's server rejects urllib's default user agent (406).
            subprocess.run(["curl", "-L", "--fail", "--retry", "2", "--max-time", "180",
                            "--output", str(partial), SOURCES[filename]], check=True)
        else:
            request = urllib.request.Request(SOURCES[filename], headers={"User-Agent": "sutro-procedure-transfer/1.0"})
            with urllib.request.urlopen(request, timeout=120) as source, partial.open("wb") as target:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    target.write(block)
        if file_hash(partial) != PINNED_SHA256[filename]:
            raise ValueError(f"Source SHA256 mismatch: {filename}")
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)
    return path


def save_pool(name, native, labels, metadata, root, sources, source_indices=None):
    native = np.asarray(native, dtype=np.uint8)
    labels = np.asarray(labels, dtype=np.int64)
    assert native.ndim == 3 and len(native) == len(labels)
    assert set(np.unique(labels)) == set(range(10))
    images = np.empty((len(labels), 1, 9, 9), dtype=np.float32)
    for start in range(0, len(labels), 4096):
        images[start:start + 4096, 0] = area_resize(native[start:start + 4096].astype(np.float32) / 255.0, 9)
    np.clip(images, 0.0, 1.0, out=images)
    hashes = np.asarray([hashlib.sha256(row.tobytes()).hexdigest() for row in native], dtype="S64")
    if source_indices is None:
        source_indices = np.arange(len(labels), dtype=np.int64)
    else:
        source_indices = np.asarray(source_indices, dtype=np.int64)
    output = root / "raw_pools"
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / f"{name}.npz", images=images, labels=labels,
                        example_hashes=hashes, source_indices=source_indices)
    metadata.update({
        "name": name, "count": len(labels), "class_counts": np.bincount(labels, minlength=10).tolist(),
        "sources": [{"filename": f, "url": SOURCES[f], "sha256": file_hash(root / "raw" / f),
                     "bytes": (root / "raw" / f).stat().st_size} for f in sources],
        "preprocessing": {"native_shape": list(native.shape[1:]), "native_dtype": "uint8",
                          "resize": "mnist.code.data.area_resize: exact separable box-area averaging",
                          "output_shape": [1, 9, 9], "pixel_range": [0, 1],
                          "pixel_dtype": "float32", "polarity": "dark background, bright ink",
                          "native_hash": "sha256 of canonical bright-ink uint8 image bytes in C order",
                          "deduplication": "deferred to unified experiment pool preparation"},
        "images_sha256": array_hash(images), "labels_sha256": array_hash(labels),
        "example_hashes_sha256": array_hash(hashes), "source_indices_sha256": array_hash(source_indices),
        "npz_sha256": file_hash(output / f"{name}.npz"),
    })
    (output / f"{name}.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    print("Prepared", name, len(labels), "class counts", metadata["class_counts"], flush=True)
    return metadata


def prepare_k49(root):
    files = ["k49-train-imgs.npz", "k49-train-labels.npz", "k49_classmap.csv"]
    paths = [download(root / "raw", f) for f in files]
    native, original_labels = [np.load(p)["arr_0"] for p in paths[:2]]
    selected = np.flatnonzero(np.isin(original_labels, K49_IDS))
    labels = np.searchsorted(K49_IDS, original_labels[selected])
    classmap = {int(row["index"]): row["char"] for row in csv.DictReader(paths[2].open())}
    assert len(selected) == 60000
    return save_pool("k49_10", native[selected], labels, {
        "title": "Kuzushiji-49: fixed 10-class subset", "kind": "handwritten Japanese hiragana",
        "homepage": "https://github.com/rois-codh/kmnist", "class_names": [classmap[i] for i in K49_IDS],
        "original_class_ids": K49_IDS, "selection_policy": "Fixed IDs before experiments; 6,000 official training examples per selected class. These ten characters are disjoint from KMNIST's ten characters.",
        "pool_policy": "Selected classes from official training split only; official test withheld.",
        "original_split_counts": {"selected_train": 60000, "selected_test": 10000},
        "source_index_definition": "row index in k49-train-imgs.npz",
        "caveat": "K49 extends KMNIST; these are related source collections and cannot be counted as independent corpora.",
        "license": "CC BY-SA 4.0",
    }, root, files, selected)


def prepare_kannada(root):
    files = ["X_kannada_MNIST_train.npz", "y_kannada_MNIST_train.npz"]
    native, labels = [np.load(download(root / "raw", f))["arr_0"] for f in files]
    assert native.shape == (60000, 28, 28)
    return save_pool("kannada_digits", native, labels, {
        "title": "Kannada-MNIST", "kind": "handwritten Kannada digits", "homepage": "https://github.com/vinayprabhu/Kannada_MNIST",
        "class_names": [chr(0x0CE6 + i) for i in range(10)], "original_class_ids": list(range(10)),
        "selection_policy": "All ten native digit classes.", "pool_policy": "Official training split only; official test withheld.",
        "original_split_counts": {"train": 60000, "test": 10000},
        "source_index_definition": "row index in X_kannada_MNIST_train.npz",
        "caveat": "MNIST format but independently collected Kannada handwriting; unrelated to NIST samples.",
    }, root, files)


def prepare_dig(root):
    files = ["X_dig_MNIST.npz", "y_dig_MNIST.npz"]
    native, labels = [np.load(download(root / "raw", f))["arr_0"] for f in files]
    assert native.shape == (10240, 28, 28)
    return save_pool("dig_mnist", native, labels, {
        "title": "DiG-MNIST Kannada digits", "kind": "handwritten Kannada digits (challenging secondary collection)",
        "homepage": "https://github.com/vinayprabhu/Kannada_MNIST",
        "class_names": [chr(0x0CE6 + i) for i in range(10)], "original_class_ids": list(range(10)),
        "selection_policy": "All ten native digit classes in the separately released DiG-MNIST collection.",
        "pool_policy": "Complete secondary DiG-MNIST pool, separate from Kannada-MNIST main train/test.",
        "source_index_definition": "row index in X_dig_MNIST.npz",
        "caveat": "Cannot support 10,000+10,000 draws; use at most 5,120+5,120 examples before deduplication. Fresh training measures procedure portability, not the original Kannada-to-DiG shift task.",
    }, root, files)


def prepare_devanagari(root):
    filename = "devanagari.zip"
    path = download(root / "raw", filename)
    native, labels = [], []
    with zipfile.ZipFile(path) as archive:
        names = sorted(n for n in archive.namelist() if re.search(r"/(Train|Test)/digit_[0-9]/[^/]+\.png$", n))
        for member in names:
            native.append(np.asarray(Image.open(io.BytesIO(archive.read(member))).convert("L")))
            labels.append(int(member.split("/digit_")[1][0]))
    assert len(native) == 20000
    return save_pool("devanagari_digits", native, labels, {
        "title": "Devanagari handwritten digits (UCI DHCD)", "kind": "handwritten Devanagari digits",
        "homepage": "https://archive.ics.uci.edu/dataset/389/devanagari+handwritten+character+dataset",
        "class_names": [chr(0x0966 + i) for i in range(10)], "original_class_ids": [f"digit_{i}" for i in range(10)],
        "selection_policy": "All ten digit classes; 36 consonant classes excluded.",
        "pool_policy": "Combined official train (17,000) and test (3,000), because training-only pool is below 20,000.",
        "original_split_counts": {"selected_train": 17000, "selected_test": 3000},
        "source_index_definition": "row in lexicographically sorted digit PNG archive member names",
        "caveat": "Native 32x32 includes its original 2-pixel border; direct area resize retains it. Deduplication may reduce total below 20,000.",
        "license": "CC BY 4.0",
    }, root, [filename])


def prepare_madbase(root):
    filename = "MAHDBase_TrainingSet.rar"
    path = download(root / "raw", filename)
    directory = root / "raw" / "madbase_extracted"
    marker = directory / ".complete"
    if not marker.exists():
        listing = subprocess.run(["bsdtar", "-tf", str(path)], check=True, capture_output=True, text=True).stdout.splitlines()
        if any(Path(n).is_absolute() or ".." in Path(n).parts for n in listing):
            raise ValueError("Unsafe source archive path")
        directory.mkdir(parents=True, exist_ok=True)
        subprocess.run(["bsdtar", "-xf", str(path), "-C", str(directory)], check=True)
        marker.touch()
    names = sorted(directory.rglob("*.bmp"))
    native, labels = [], []
    for path in names:
        native.append(255 - np.asarray(Image.open(path).convert("L")))
        labels.append(int(re.search(r"digit([0-9])\.bmp$", path.name).group(1)))
    assert len(native) == 60000
    return save_pool("madbase", native, labels, {
        "title": "MADBase Arabic handwritten digits", "kind": "handwritten Arabic-Indic digits",
        "homepage": "https://datacenter.aucegypt.edu/shazeem/",
        "class_names": [chr(0x0660 + i) for i in range(10)], "original_class_ids": list(range(10)),
        "selection_policy": "All ten native digit classes.", "pool_policy": "Official training split only; official test withheld.",
        "original_split_counts": {"train": 60000, "test": 10000},
        "source_index_definition": "row in lexicographically sorted original BMP relative paths",
        "native_transform": "Invert white-background BMP pixel values (255 - grayscale).",
        "caveat": "Official split is writer-disjoint, but fresh draws from official training pool are image-disjoint, not writer-disjoint.",
    }, root, [filename])


def prepare_notmnist(root):
    filename = "notMNIST_large.tar.gz"
    path = download(root / "raw", filename)
    native, labels, source_indices, errors = [], [], [], []
    ordinal = 0
    with tarfile.open(path, "r|gz") as archive:
        for member in archive:
            if not member.isfile() or not member.name.endswith(".png"):
                continue
            parts = Path(member.name).parts
            if len(parts) != 3 or parts[1] not in "ABCDEFGHIJ":
                continue
            try:
                array = np.asarray(Image.open(io.BytesIO(archive.extractfile(member).read())).convert("L"))
                if array.shape != (28, 28):
                    raise ValueError(f"unexpected shape {array.shape}")
                native.append(array)
                labels.append(ord(parts[1]) - ord("A"))
                source_indices.append(ordinal)
            except (OSError, ValueError) as error:
                errors.append({"member": member.name, "error": str(error)})
            ordinal += 1
    assert len(native) > 500000
    return save_pool("notmnist_large", native, labels, {
        "title": "notMNIST large A–J", "kind": "font-rendered Latin letters (not handwriting)",
        "homepage": "http://yaroslavvb.blogspot.com/2011/09/notmnist-dataset.html",
        "class_names": list("ABCDEFGHIJ"), "original_class_ids": list("ABCDEFGHIJ"),
        "selection_policy": "All ten native letter classes.", "pool_policy": "Creator's large pool only; small pool withheld.",
        "source_index_definition": "ordinal PNG member in original tar archive, including unreadable members",
        "source_png_count": ordinal, "unreadable_pngs": errors,
        "download_note": "Creator's original HTTP URL; its HTTPS certificate was expired during acquisition.",
        "caveat": "Many exact duplicate glyphs and visually near-identical fonts; runner deduplicates exact native images. Font-level leakage remains possible under random image draws.",
    }, root, [filename], source_indices)


def prepare_omniglot(root):
    filename = "omniglot-images_evaluation.zip"
    path = download(root / "raw", filename)
    native, labels = [], []
    classes = [f"character{i:02d}" for i in range(1, 11)]
    with zipfile.ZipFile(path) as archive:
        names = sorted(n for n in archive.namelist() if "/Gurmukhi/" in n and n.endswith(".png") and Path(n).parts[-2] in classes)
        for member in names:
            native.append(255 - np.asarray(Image.open(io.BytesIO(archive.read(member))).convert("L")))
            labels.append(classes.index(Path(member).parts[-2]))
    assert len(native) == 200
    return save_pool("omniglot_gurmukhi10", native, labels, {
        "title": "Omniglot Gurmukhi characters 01–10", "kind": "handwritten Gurmukhi characters",
        "homepage": "https://github.com/brendenlake/omniglot", "class_names": classes, "original_class_ids": classes,
        "selection_policy": "Fixed first ten character IDs in Gurmukhi evaluation alphabet before experiments.",
        "pool_policy": "200 images from one alphabet in official evaluation archive; 20 samples per character.",
        "source_index_definition": "row in lexicographically sorted selected PNG member paths",
        "native_transform": "Invert white-background PNG pixel values (255 - grayscale).",
        "caveat": "Cannot support 10,000+10,000 draws; run as explicitly smaller 100+100 example exception. Ten fixed classes, not standard Omniglot episodes.",
        "license": "MIT (official repository)",
    }, root, [filename])


PREPARERS = dict(zip(NAMES, (prepare_k49, prepare_kannada, prepare_dig, prepare_devanagari, prepare_madbase, prepare_notmnist, prepare_omniglot)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--names", nargs="+", choices=NAMES, default=list(NAMES))
    args = parser.parse_args()
    (args.root / "raw").mkdir(parents=True, exist_ok=True)
    for name in args.names:
        PREPARERS[name](args.root)


if __name__ == "__main__":
    main()
