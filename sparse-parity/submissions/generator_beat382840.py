"""Regenerate the frozen 382840 IR byte-identically from the upstream generator.

The record in ``submissions/packedscan_beat382840_mask32.ir`` is produced by the
upstream generator ``packed_sparse_parity.generate_packed_scan`` from
``cybertronai/sutro-problems`` with the two layout-compaction flags ON.
Shipped defaults are OFF, i.e. the upstream 80%/100% records were submitted
without these optimizations. Cost delta vs the shipped 100% record
(``generate_packed_scan(5)`` = 392666) is -9826 (-2.5024%).

Upstream code is NOT vendored here; this wrapper only calls it and compares
the result with the frozen file.

Usage (Git Bash, from ``sparse-parity``):
    python submissions/generator_beat382840.py \
        --upstream-dir . \
        --out /tmp/regen.ir \
        [--compare submissions/packedscan_beat382840_mask32.ir]
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

FROZEN_DEFAULT = Path(__file__).with_name("packedscan_beat382840_mask32.ir")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--compare", type=Path, default=FROZEN_DEFAULT)
    ap.add_argument("--weight-cap", type=int, default=5)
    ap.add_argument("--walk-order-bias", type=float, default=1.0)
    args = ap.parse_args()

    sys.path.insert(0, str(args.upstream_dir.resolve()))
    import packed_sparse_parity as psp  # noqa: E402

    t0 = time.perf_counter()
    ir = psp.generate_packed_scan(
        args.weight_cap,
        compact_predicates=True,
        compact_flow=True,
        walk_order_bias=args.walk_order_bias,
    )
    gen_seconds = round(time.perf_counter() - t0, 3)

    out: Path = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    # newline="\n" so the regenerated file is byte-comparable with the frozen
    # file after LF normalization.
    with out.open("w", newline="\n", encoding="utf-8") as f:
        # The frozen file carries one trailing newline (the upstream generator
        # string has none); write it so the files are byte-identical.
        f.write(ir)
        if not ir.endswith("\n"):
            f.write("\n")

    lf_text = ir.replace("\r\n", "\n")
    if not lf_text.endswith("\n"):
        lf_text += "\n"
    lf_sha = hashlib.sha256(lf_text.encode()).hexdigest()
    print(f"generated: lines={ir.count(chr(10))} seconds={gen_seconds}")
    print(f"sha256_lf_normalized={lf_sha}")

    ok = True
    if args.compare and args.compare.exists():
        with args.compare.open("r", newline="") as f:
            frozen = f.read()
        same_lf = frozen.replace("\r\n", "\n") == lf_text
        print(f"compare_lf_equal={same_lf} ({args.compare})")
        ok = ok and same_lf
    print("GENERATE:", "OK" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
