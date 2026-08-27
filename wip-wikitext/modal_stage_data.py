"""Stage WikiText-103 into a Modal Volume — one-time, no GPU.

The Lambda harness fetches WikiText-103 fresh per run inside
entrypoint.sh. On Modal we use a persistent modal.Volume so submissions
re-mount the staged splits instead of re-downloading the dataset every
time.

Data layout in the volume:
    /data/wiki.train.raw
    /data/wiki.valid.raw
    /data/wiki.test.raw

Anti-cheat note: README says the test set must NOT be present during
training. With a single volume that mixes splits, training sees test.
Two follow-ups for a real submission run:

  (a) Split into two volumes (train+valid vs. test only) and mount
      train+valid at train time, test at eval time, or
  (b) Use a single volume but enforce non-readability of wiki.test.raw
      inside the train function (subprocess + filesystem perms, or
      pop the path before user code runs).

For staging itself, one volume is fine — we just need the bytes on disk.

Usage:
    modal run modal_stage_data.py             # download + commit
    modal run modal_stage_data.py::ls_main    # list staged files
"""
from __future__ import annotations

import modal

VOLUME_NAME = "wikitext-103-raw"

image = modal.Image.debian_slim().pip_install("datasets==3.2.0")

app = modal.App("wikitext-stage-data", image=image)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


@app.function(volumes={"/data": volume}, timeout=1800)
def stage() -> dict:
    """Download WikiText-103 raw splits via HuggingFace and write the
    three .raw files into the mounted volume. Commits the volume on
    success so subsequent runs see the data.
    """
    from pathlib import Path
    from datasets import load_dataset

    out = Path("/data")
    out.mkdir(parents=True, exist_ok=True)

    existing = {p.name: p.stat().st_size for p in out.iterdir() if p.is_file()}
    if existing:
        print(f"[stage] volume already has files: {existing}")

    print("[stage] downloading Salesforce/wikitext-103-raw-v1 ...")
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")

    written = {}
    for hf_split, fname in [
        ("train", "wiki.train.raw"),
        ("validation", "wiki.valid.raw"),
        ("test", "wiki.test.raw"),
    ]:
        path = out / fname
        text = "\n".join(ds[hf_split]["text"])
        path.write_text(text, encoding="utf-8")
        size = path.stat().st_size
        written[fname] = size
        print(f"[stage] wrote {path} ({size:,} bytes, {len(text):,} chars)")

    volume.commit()
    print("[stage] volume committed")
    return {"written": written, "volume": VOLUME_NAME}


@app.function(volumes={"/data": volume}, timeout=120)
def ls_volume() -> dict:
    """Read-back probe: list what's actually in the volume."""
    from pathlib import Path
    out = Path("/data")
    if not out.exists():
        return {"exists": False}
    files = {}
    for p in sorted(out.rglob("*")):
        if p.is_file():
            files[str(p.relative_to(out))] = p.stat().st_size
    return {"exists": True, "files": files}


@app.local_entrypoint()
def main() -> None:
    import json
    result = stage.remote()
    print("===== STAGED =====")
    print(json.dumps(result, indent=2))


@app.local_entrypoint()
def ls_main() -> None:
    import json
    print(json.dumps(ls_volume.remote(), indent=2))
