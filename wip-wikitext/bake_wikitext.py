"""Image-build hook: bake WikiText-103 raw splits into /data.

Lives in its own module so Modal's ``Image.run_function`` can resolve
the callable as ``bake_wikitext:bake`` without importing ``submit.py``
(which depends on ``task`` and other harness files not present in the
build container at this layer).
"""
from __future__ import annotations


def bake() -> None:
    """Materialize WikiText-103 raw splits into /data inside the image.
    Runs once per image build, layer-cached by Modal across every
    subsequent submission."""
    from pathlib import Path
    from datasets import load_dataset
    out = Path("/data")
    out.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
    for hf_split, fname in [
        ("train", "wiki.train.raw"),
        ("validation", "wiki.valid.raw"),
        ("test", "wiki.test.raw"),
    ]:
        (out / fname).write_text("\n".join(ds[hf_split]["text"]), encoding="utf-8")
