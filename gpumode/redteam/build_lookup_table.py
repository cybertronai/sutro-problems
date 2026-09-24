"""Offline table builder for the label-lookup red-team submissions.

Not a submission. Run once, on the attacker's own machine, before the
competition starts:

    python build_lookup_table.py --raw <dir with the official idx.gz files>

It downsamples the *public* 60,000-image MNIST training split to 9x9 with the
harness's own ``mnist_data.area_resize``, fingerprints every image with a
40-bit hash of its float32 bit pattern, and prints a base64 blob that maps
fingerprint -> public MNIST label. The blob is pasted into
``label-lookup.py`` / ``label-lookup_fallback.py``.

Nothing here needs the network at evaluation time and nothing here is secret:
MNIST-medium draws both of its halves from the public training split, so every
test image a submission is ever shown is in this table.
"""

from __future__ import annotations

import argparse
import base64
import sys
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import mnist_data  # noqa: E402

HASH_SEED = 0xC0FFEE
HASH_BITS = 40


def hash_constants(pixels: int) -> np.ndarray:
    """Odd 64-bit multipliers, one per pixel; the same numbers live in the exploit."""
    rng = np.random.default_rng(HASH_SEED)
    return rng.integers(1, 2**62, size=pixels, dtype=np.int64) * 2 + 1


def fingerprints(images: np.ndarray) -> np.ndarray:
    """Hash the exact float32 bit pattern of each image.

    The harness builds its pixels with this same code path, so the bits a
    submission receives are identical to the bits hashed here.
    """
    flat = np.ascontiguousarray(images.reshape(images.shape[0], -1))
    bits = flat.view(np.uint32).astype(np.int64)
    keys = (bits * hash_constants(bits.shape[1])).sum(axis=1)
    return keys & ((1 << HASH_BITS) - 1)


def build(raw_dir: str, size: int = 9) -> str:
    images, labels = mnist_data.load_pool(raw_dir, "mnist", size)
    keys = fingerprints(images)
    if np.unique(keys).size != keys.size:
        raise SystemExit("fingerprint collision; widen HASH_BITS")
    order = np.argsort(keys)
    sorted_keys = keys[order].astype(np.uint64)
    sorted_labels = labels[order].astype(np.uint8)
    deltas = np.empty(sorted_keys.size, dtype=np.uint32)
    deltas[0] = sorted_keys[0]
    deltas[1:] = np.diff(sorted_keys).astype(np.uint32)
    packed = (sorted_labels[0::2] << 4) | sorted_labels[1::2]
    blob = zlib.compress(deltas.tobytes() + packed.tobytes(), 9)
    return base64.b64encode(blob).decode("ascii")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True)
    parser.add_argument("--size", type=int, default=9)
    parser.add_argument("--out", default="-")
    args = parser.parse_args()
    text = build(args.raw, args.size)
    if args.out == "-":
        print(text)
    else:
        Path(args.out).write_text(text)


if __name__ == "__main__":
    main()
