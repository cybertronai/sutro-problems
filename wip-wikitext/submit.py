#!/usr/bin/env python3
"""End-to-end submission runner for the wikitext energy benchmark on Modal.

What it does:
  1. Imports the user's submission file locally as a precheck (catches
     SyntaxError / missing torch / missing `train` before any cloud spend).
  2. Defines a Modal App with an inline image (PyTorch + nvidia-ml-py +
     datasets) and a single A100-40GB function.
  3. The remote function: stages WikiText-103 onto a persistent Modal
     Volume (cached across runs), verifies NVML, writes the user's
     submission to disk, runs run_eval, and returns the result dict.
  4. Saves the result JSON to submissions/ and appends a row to the
     Record History table in README.md.

Setup (once):
  pip install modal
  modal token new      # opens browser, writes ~/.modal.toml

Usage:
  python3 submit.py path/to/my_submission.py
  python3 submit.py example_submission.py --yes
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import modal

import task  # task-pinned constants — single source of truth

HERE = Path(__file__).resolve().parent

# Modal A100-40GB list price as of 2026-05: $2.10/hr.
# A typical run is ~10 min (image cold-start + verify_nvml + WikiText-103
# fetch on first run + ~5 min training capped by E_MAX_JOULES + ~2 min
# eval). After the first run the dataset is cached on a Modal Volume,
# so subsequent runs are ~7 min. We size the estimate for a *first* run.
EST_RUNTIME_MIN = 10
EST_RATE_USD_PER_HR = 2.10
EST_COST_USD = round(EST_RUNTIME_MIN * EST_RATE_USD_PER_HR / 60, 2)

# task.INSTANCE_TYPE is the leaderboard-pinned hardware string. The
# "modal:" prefix is informational; Modal's gpu= kwarg takes the bare
# SKU. Strip and validate so a future change to task.INSTANCE_TYPE
# (different provider, different GPU) fails loudly here instead of
# silently launching on the wrong hardware.
_provider, _, MODAL_GPU = task.INSTANCE_TYPE.partition(":")
if _provider != "modal" or not MODAL_GPU:
    sys.exit(
        f"task.INSTANCE_TYPE = {task.INSTANCE_TYPE!r} not understood by "
        f"submit.py — expected 'modal:<gpu>' (e.g. 'modal:A100-40GB')."
    )


def _load_dotenv() -> None:
    """Load KEY=VALUE pairs from the nearest .env on the way up to the
    repo root. Doesn't overwrite anything already in os.environ — so an
    explicit ``export FOO=bar`` still wins. Modal auth normally lives in
    ``~/.modal.toml`` (written by ``modal token new``); .env is here for
    CI-style overrides via ``MODAL_TOKEN_ID`` / ``MODAL_TOKEN_SECRET``.
    """
    for d in (HERE, *HERE.parents):
        env = d / ".env"
        if env.is_file():
            for line in env.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                k, v = k.strip(), v.strip().strip('"').strip("'")
                os.environ.setdefault(k, v)
            return


_load_dotenv()


# ---------------------------------------------------------------------------
# Modal app definition
# ---------------------------------------------------------------------------
#
# Harness files are added to /workspace inside the container. The user's
# submission is *not* baked in — it's passed as bytes to the function
# call, so we don't rebuild the image per submission. Image-level deps
# (torch, nvidia-ml-py, datasets) are cached by Modal and reused.

HARNESS_FILES = (
    "wikitext.py",
    "baseline_ngram.py",
    "baseline_transformer.py",
    "run_eval.py",
    "verify_nvml.py",
    "fetch_data.py",
    "task.py",
)

app = modal.App("wikitext-bench")

# CUDA 12.4 PyTorch wheels match the driver Modal exposes on its A100
# fleet. nvidia-ml-py is the NVML binding EnergyMeter uses; datasets is
# how fetch_data.py pulls WikiText-103 from the HuggingFace mirror.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "nvidia-ml-py==12.560.30",
        "datasets==3.2.0",
    )
    .workdir("/workspace")
)
for _f in HARNESS_FILES:
    image = image.add_local_file(str(HERE / _f), f"/workspace/{_f}")

# WikiText-103 cached across runs. ~750 MB of raw text; first run pays
# the ~1 min HuggingFace fetch, all subsequent runs reuse the volume.
data_volume = modal.Volume.from_name("wikitext-103-data", create_if_missing=True)


@app.function(
    image=image,
    gpu=MODAL_GPU,
    # Hard wall-clock cap. Training is bounded at ~5 min by
    # task.E_MAX_JOULES (NVML watchdog), plus image cold-start +
    # data fetch + eval ≈ <15 min realistic. 30 min gives 2× safety.
    timeout=30 * 60,
    volumes={"/data": data_volume},
)
def run_submission(submission_bytes: bytes, submission_name: str) -> dict:
    """Run a submission end-to-end on the pinned Modal GPU and return
    the result dict that submit.py will save and record."""
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path

    workspace = Path("/workspace")
    os.chdir(workspace)

    # Stage WikiText-103 onto the volume on first run; reuse otherwise.
    train_raw = Path("/data") / "wiki.train.raw"
    if not train_raw.exists():
        print("[modal] fetching WikiText-103 (one-time, cached on volume) ...")
        subprocess.run(
            [sys.executable, "fetch_data.py", "/data"], check=True
        )
        data_volume.commit()
    else:
        print("[modal] WikiText-103 already cached on volume")

    # NVML probe — bail before training cycles if the energy counter
    # isn't exposed on this host.
    print("[modal] verifying NVML energy counter ...")
    nvml = subprocess.run(
        [sys.executable, "verify_nvml.py"], capture_output=True, text=True
    )
    print(nvml.stdout)
    if nvml.returncode != 0:
        raise RuntimeError(
            f"verify_nvml.py failed (rc={nvml.returncode}). "
            f"stderr:\n{nvml.stderr}"
        )
    # verify_nvml prints a JSON summary as its final stdout line.
    nvml_summary = json.loads(nvml.stdout.strip().splitlines()[-1])

    # Drop the user's submission next to the harness files. Keep the
    # original stem so run_eval.py's submission_name (== Path(...).stem)
    # propagates into the result JSON and the README record row.
    if submission_name in {Path(f).stem for f in HARNESS_FILES} | {"submission"}:
        raise RuntimeError(
            f"submission name {submission_name!r} collides with a harness "
            f"file; rename your submission file."
        )
    sub_path = workspace / f"{submission_name}.py"
    sub_path.write_bytes(submission_bytes)

    # Pull task constants from the in-image task.py (single source of
    # truth — submitters do not get to vary these).
    sys.path.insert(0, str(workspace))
    import importlib
    task_mod = importlib.import_module("task")
    test_chars = task_mod.TEST_CHARS
    e_max = task_mod.E_MAX_JOULES

    eval_args = [
        sys.executable, "run_eval.py",
        "--data-dir", "/data",
        "--submission", str(sub_path),
        "--results-json", "/tmp/result.json",
        "--max-test-chars", str(test_chars),
    ]
    if e_max is not None:
        eval_args += ["--e-max-joules", str(e_max)]

    print(f"[modal] running submission "
          f"(TEST_CHARS={test_chars} E_MAX={e_max}) ...")
    rc = subprocess.run(eval_args).returncode

    # run_eval exits 2 on the energy-budget DQ path *with* a written
    # result.json — that's a valid leaderboard outcome and we ship it.
    # Anything else missing the JSON is a harness failure.
    rj = Path("/tmp/result.json")
    if not rj.exists():
        raise RuntimeError(f"run_eval.py failed (rc={rc}); no result.json written")

    result = json.loads(rj.read_text())
    result["_nvml"] = nvml_summary
    return result


# ---------------------------------------------------------------------------
# Local orchestration
# ---------------------------------------------------------------------------

def check_submission_imports(submission_path: Path) -> None:
    """Import the submission file in a child process. Fails fast on
    SyntaxError, missing module deps (typo'd imports, missing torch),
    and ``train`` not being defined — all of which would otherwise only
    surface mid-run on the Modal host, after Modal billing started.
    """
    snippet = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('s', {str(submission_path)!r})\n"
        "mod = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(mod)\n"
        "fn = getattr(mod, 'train', None)\n"
        "assert callable(fn), 'submission must define train(train_text, valid_text=None) -> CharModel'\n"
    )
    r = subprocess.run(
        [sys.executable, "-c", snippet], capture_output=True, text=True, cwd=HERE,
    )
    if r.returncode != 0:
        msg = (r.stderr or r.stdout or "").strip()
        sys.exit(
            f"submission failed to import locally: {submission_path}\n"
            f"{msg}\n"
            "(Note: transformer-based submissions need `pip install torch` "
            "locally to validate; CPU is fine.)"
        )


def save_result(result: dict, submission_path: Path) -> Path:
    out_dir = HERE / "submissions"
    out_dir.mkdir(exist_ok=True)
    sub_name = submission_path.stem
    date = result["date_utc"][:10]
    out_path = out_dir / f"{sub_name}_{date}.json"
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    return out_path


def save_nvml_artifact(result: dict, submission_path: Path) -> Path | None:
    """Mirror the Lambda-era ``submissions/<sub>_<date>.nvml.json``
    artifact, sourced from the embedded ``_nvml`` field that the Modal
    function returns. Returns the path written, or None if absent.
    """
    nvml = result.get("_nvml")
    if not nvml:
        return None
    out_dir = HERE / "submissions"
    out_dir.mkdir(exist_ok=True)
    sub_name = submission_path.stem
    date = result["date_utc"][:10]
    out_path = out_dir / f"{sub_name}_{date}.nvml.json"
    out_path.write_text(json.dumps(nvml, indent=2) + "\n")
    return out_path


def append_record(result: dict, json_relpath: str) -> None:
    """Append one row to the Record History table in README.md.

    Replaces the placeholder dash row if present, otherwise appends.
    Disqualified rows render their accuracy cell as ``DQ`` so they
    don't pollute the leaderboard sort.
    """
    readme = HERE / "README.md"
    text = readme.read_text()
    energy = result.get("training_energy_J")
    energy_cell = f"{energy:>10,.0f}" if energy is not None else "         —"
    if result.get("disqualified"):
        acc_cell = "      DQ"
    else:
        acc_cell = f"{result['test_char_accuracy']:.4f}"
    row = (
        f"| {result['date_utc'][:10]} "
        f"| {energy_cell} "
        f"| {acc_cell} "
        f"| {result['submission']} "
        f"| [json]({json_relpath}) "
        f"| @you |\n"
    )
    placeholder = "| —    |          — |        — | —      | —          | —           |\n"
    if placeholder in text:
        text = text.replace(placeholder, row, 1)
    else:
        text = text.rstrip() + "\n" + row
    readme.write_text(text)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("submission", type=Path,
                   help="Python file exposing train(train_text, valid_text=None) -> CharModel")
    p.add_argument("--yes", action="store_true",
                   help="Skip cost confirmation prompt")
    args = p.parse_args()

    if not args.submission.exists():
        sys.exit(f"submission file not found: {args.submission}")

    print(f"╭─ Modal {MODAL_GPU} wikitext submission ───────")
    print(f"│  submission:    {args.submission}")
    print(f"│  est. runtime:  ~{EST_RUNTIME_MIN} min  (cold start + train + eval)")
    print(f"│  est. cost:     ~${EST_COST_USD:.2f}  (@ ${EST_RATE_USD_PER_HR:.2f}/hr)")
    print(f"╰───────────────────────────────────────────────")

    if not args.yes and input("proceed? [y/N] ").strip().lower() != "y":
        sys.exit("aborted")

    check_submission_imports(args.submission)

    submission_bytes = args.submission.read_bytes()
    submission_name = args.submission.stem

    print(f"[modal] launching {MODAL_GPU} ...")
    with app.run():
        result = run_submission.remote(submission_bytes, submission_name)

    out_path = save_result(result, args.submission)
    save_nvml_artifact(result, args.submission)
    append_record(result, json_relpath=f"submissions/{out_path.name}")

    if result.get("disqualified"):
        e_max = result.get("e_max_joules")
        e_at_kill = result.get("training_energy_J")
        dur = result.get("training_duration_s")
        print(f"[done] DISQUALIFIED — submission exceeded the training "
              f"energy budget.")
        print(f"       reason   = {result.get('reason', 'unknown')}")
        if e_max is not None:
            print(f"       e_max    = {e_max:,.0f} J")
        if e_at_kill is not None:
            print(f"       at_kill  = {e_at_kill:,.0f} J")
        if dur is not None:
            print(f"       duration = {dur:.1f} s")
        print(f"       result   = {out_path}")
        # Non-zero exit so CI scripts can distinguish DQ from a clean
        # leaderboard submission, even though the harness ran cleanly.
        return 2

    print(f"[done] {out_path}")
    energy = result.get("training_energy_J")
    if energy is not None:
        print(f"       energy = {energy:,.0f} J")
    else:
        print(f"       energy = NOT MEASURED")
    print(f"       acc    = {result['test_char_accuracy']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
