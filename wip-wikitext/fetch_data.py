"""Fetch WikiText-103 raw splits via HuggingFace and write
``wiki.{train,valid,test}.raw`` files (the layout ``load_wikitext103``
expects).

Used at container start when ``/data/wiki.train.raw`` is not already
present. The historical ``s3.amazonaws.com/research.metamind.io`` URL
no longer resolves — see NOTES.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

from datasets import load_dataset

out = Path(sys.argv[1] if len(sys.argv) > 1 else "/data")
out.mkdir(parents=True, exist_ok=True)

ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
for hf_split, fname in [
    ("train", "wiki.train.raw"),
    ("validation", "wiki.valid.raw"),
    ("test", "wiki.test.raw"),
]:
    (out / fname).write_text("\n".join(ds[hf_split]["text"]), encoding="utf-8")
    print(f"wrote {out / fname}")
