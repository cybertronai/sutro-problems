from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ["kmnist", "emnist_letters_aj", "emnist_letters_kt", "emnist_balanced_aj",
            "emnist_digits", "emnist_mnist", "qmnist_recovered", "k49_10",
            "kannada_digits", "devanagari_digits", "madbase", "notmnist_large",
            "fashion_mnist", "svhn", "cifar10"]
SEEDS = list(range(20261101, 20261112))


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha(array):
    value = np.ascontiguousarray(array.astype(array.dtype.newbyteorder("<"), copy=False))
    return hashlib.sha256(value.tobytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+".part")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n")
    temporary.replace(path)


def split_indices(count, seed):
    if count < 20000:
        raise ValueError("Every v1 task requires at least 20,000 curated examples")
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed).spawn(2)[0]))
    order = rng.permutation(count)
    return order[:10000], order[10000:20000]

