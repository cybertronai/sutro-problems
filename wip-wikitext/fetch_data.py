"""Fetch WikiText-103 raw splits from the public GCS mirror at
gs://wikitext-103-raw-v1 and write ``wiki.{train,valid,test}.raw`` files
(the layout ``load_wikitext103`` expects).

Used at container start when ``/data/wiki.train.raw`` is not already
present. The historical ``s3.amazonaws.com/research.metamind.io`` URL
no longer resolves; the prior HuggingFace fetch hung inside Modal's
builder — see NOTES.md.
"""
from __future__ import annotations

import sys

from bake_wikitext import bake

if __name__ == "__main__":
    bake(sys.argv[1] if len(sys.argv) > 1 else "/data")
