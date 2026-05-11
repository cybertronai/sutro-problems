"""Local helper that materialises WikiText-103 raw splits from the
public GCS mirror at ``gs://wikitext-103-raw-v1``.

Used by ``fetch_data.py`` to populate ``wikitext-103-raw-v1/wiki.*.raw``,
which the Dockerfile then ``COPY``s into ``/data`` of the prebuilt
``ghcr.io/ab-10/wikitext-bench`` image. Not invoked by ``submit.py`` —
the Modal path uses ``Image.from_registry(...)`` and the splits are
already baked into the registry image.
"""
from __future__ import annotations

GCS_BUCKET_URL = "https://storage.googleapis.com/wikitext-103-raw-v1"

# Parquet shards in gs://wikitext-103-raw-v1, mirrored from the
# Salesforce/wikitext HuggingFace export. Single ``text: string`` column;
# one row per line of the original .raw file.
SPLIT_SHARDS: dict[str, tuple[str, ...]] = {
    "wiki.train.raw": (
        "train-00000-of-00002.parquet",
        "train-00001-of-00002.parquet",
    ),
    "wiki.valid.raw": ("validation-00000-of-00001.parquet",),
    "wiki.test.raw": ("test-00000-of-00001.parquet",),
}


def bake(out_dir: str = "/data") -> None:
    """Download the parquet shards and write
    ``wiki.{train,valid,test}.raw`` files into ``out_dir``."""
    import io
    import time
    import urllib.request
    from pathlib import Path

    import pyarrow.parquet as pq

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for fname, shards in SPLIT_SHARDS.items():
        rows: list[str] = []
        for shard in shards:
            url = f"{GCS_BUCKET_URL}/{shard}"
            t0 = time.monotonic()
            print(f"[bake_wikitext] GET {url}", flush=True)
            with urllib.request.urlopen(url, timeout=120) as resp:
                buf = resp.read()
            print(
                f"[bake_wikitext]   read {len(buf):,} bytes "
                f"in {time.monotonic() - t0:.1f}s",
                flush=True,
            )
            rows.extend(pq.read_table(io.BytesIO(buf))["text"].to_pylist())
        path = out / fname
        path.write_text("\n".join(rows), encoding="utf-8")
        print(
            f"[bake_wikitext] wrote {path} "
            f"({path.stat().st_size:,} bytes, {len(rows):,} rows)",
            flush=True,
        )
