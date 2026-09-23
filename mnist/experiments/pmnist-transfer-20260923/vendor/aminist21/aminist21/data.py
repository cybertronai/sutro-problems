"""Checksum-verified release assets and fixed draws. No upstream download at evaluation time."""
from pathlib import Path
import ssl
import urllib.request
import certifi
import numpy as np
from .common import ROOT, DATASETS, SEEDS, read, sha, array_sha, split_indices


def manifest():
    result = read(ROOT/"datasets/manifest.json")
    if [row["id"] for row in result["datasets"]] != DATASETS:
        raise ValueError("Dataset order differs from the v1 fifteen-number contract")
    return result


def fetch(data_dir, names=DATASETS):
    root = Path(data_dir)/"pools"
    root.mkdir(parents=True, exist_ok=True)
    for spec in manifest()["datasets"]:
        if spec["id"] not in names:
            continue
        target = root/spec["filename"]
        if target.exists():
            if sha(target) != spec["sha256"]:
                raise ValueError(f"Checksum mismatch: {target}; remove that file explicitly to download again")
            print(f"Verified {spec['id']}", flush=True)
            continue
        part = target.with_suffix(target.suffix+".part")
        try:
            request = urllib.request.Request(spec["url"], headers={"User-Agent":"aminist21-validation/1.0"})
            with urllib.request.urlopen(request, timeout=120, context=ssl.create_default_context(cafile=certifi.where())) as response, part.open("wb") as stream:
                for block in iter(lambda: response.read(1024*1024), b""):
                    stream.write(block)
            if part.stat().st_size != spec["bytes"] or sha(part) != spec["sha256"]:
                raise ValueError(f"Downloaded asset differs: {spec['id']}")
            part.replace(target)
        finally:
            part.unlink(missing_ok=True)
        print(f"Downloaded and verified {spec['id']}", flush=True)


def load_pool(name, data_dir, verify=True):
    spec = next(row for row in manifest()["datasets"] if row["id"] == name)
    path = Path(data_dir)/"pools"/spec["filename"]
    if verify and sha(path) != spec["sha256"]:
        raise ValueError(f"Dataset checksum mismatch: {name}")
    with np.load(path, allow_pickle=False) as archive:
        expected = {"images", "labels", "example_hashes", "source_indices", "pool_indices"}
        if set(archive.files) != expected or len(archive.files) != len(expected):
            raise ValueError(f"Unexpected or duplicate dataset archive fields: {name}")
        arrays = {key: archive[key] for key in archive.files}
    x, y, native = arrays["images"], arrays["labels"], arrays["example_hashes"]
    if x.dtype != np.float32 or x.shape != (len(y), 1, 9, 9) or y.dtype != np.int64 or y.shape != (len(y),):
        raise ValueError(f"Invalid tensor contract: {name}")
    if native.dtype != np.dtype('S64') or native.shape != (len(y),):
        raise ValueError(f"Invalid native-image hashes: {name}")
    if any(arrays[key].dtype != np.int64 or arrays[key].shape != (len(y),) for key in ('source_indices','pool_indices')):
        raise ValueError(f"Invalid source indices: {name}")
    if not np.isfinite(x).all() or x.min() < 0 or x.max() > 1 or set(np.unique(y)) != set(range(10)):
        raise ValueError(f"Invalid pixels/labels: {name}")
    if len(y) != spec["count"] or len(np.unique(native)) != len(y):
        raise ValueError(f"Uncurated or incomplete pool: {name}")
    if verify:
        for key, expected in spec["array_sha256"].items():
            if array_sha(arrays[key]) != expected:
                raise ValueError(f"Array checksum mismatch: {name}/{key}")
    return arrays


def learner_inputs(pool, draw):
    fit, query = split_indices(len(pool["labels"]), SEEDS[draw])
    return {"train_images": pool["images"][fit].copy(),
            "train_labels": pool["labels"][fit].copy(),
            "test_images": pool["images"][query].copy()}, fit, query
