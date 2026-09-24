#!/usr/bin/env python3
"""Run a submission through this harness the way KernelBot would.

    # on one Modal A100, exactly as the leaderboard will
    python run_modal.py --band mnist-medium-5pct --submission submissions/pca_qda.py \
        --mode leaderboard --seed 20260922 --output results/pca-qda-5pct.json

    # the identical pipeline on this machine's CPU, no Modal account needed
    python run_modal.py --band mnist-medium-5pct --submission submissions/ncm_baseline.py \
        --mode test --local --case draws=1 --case train=2000 --case test=2000

What it replicates from KernelBot (src/libkernelbot/run_eval.py):
  * task.yml "files" are copied into a scratch directory, with the chosen
    submission written as submission.py;
  * the case lines are rendered as "k: v; k: v", one per line, and benchmark or
    leaderboard modes keep only the last line because ranking_by is "last";
  * eval.py is run as "python eval.py <mode> <cases-file>" with POPCORN_FD
    pointing at a pipe and POPCORN_SEED holding the secret seed, under the
    task.yml timeout for that mode;
  * "key: value" lines from the pipe become the result dictionary, and
    check == pass decides success;
  * --mode leaderboard runs test, then benchmark, then leaderboard, each in a
    fresh process, and stops at the first failure.

The Modal image follows src/runners/modal_runner.py's CUDA image: an
nvidia/cuda base with add_python="3.13", numpy ~= 2.3 and torch == 2.12.0. The
rest of that image (CUTLASS, cuDSL, nsight) is not needed to run a PyTorch
submission and is left out so the image builds in a couple of minutes.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

CUDA_TAG = "13.3.0-devel-ubuntu24.04"
TORCH_PIN = "torch==2.12.0"
NUMPY_PIN = "numpy~=2.3"
GPU = "A100"  # KernelBot's consts.ModalGPU.A100 is this exact string
MODAL_TIMEOUT = 1800
# Deny container egress. Set SUTRO_ALLOW_NETWORK=1 only to let the container
# download MNIST on a first run; a scored run must keep this on.
BLOCK_NETWORK = os.environ.get("SUTRO_ALLOW_NETWORK", "") != "1"

BAKED_POOL = "/opt/sutro-pool"
_MNIST = "https://ossci-datasets.s3.amazonaws.com/mnist/"
_FASHION = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"
# The names match mnist_data.SOURCES, which re-checks every MD5 on load.
BAKED_SOURCES = (
    ("train-images-idx3-ubyte.gz", _MNIST + "train-images-idx3-ubyte.gz"),
    ("train-labels-idx1-ubyte.gz", _MNIST + "train-labels-idx1-ubyte.gz"),
    ("fashion-train-images-idx3-ubyte.gz", _FASHION + "train-images-idx3-ubyte.gz"),
    ("fashion-train-labels-idx1-ubyte.gz", _FASHION + "train-labels-idx1-ubyte.gz"),
)
CAPACITY_RETRY_SECONDS = 40 * 60


# ------------------------------------------------------------------ task.yml

def load_task(band_dir: Path) -> dict:
    """Read the generated task.yml.

    PyYAML when it is installed; otherwise a small reader for the subset the
    generator emits (scalars, ``|`` blocks, and lists of one-line JSON objects).
    """
    text = (band_dir / "task.yml").read_text()
    try:
        import yaml

        return yaml.safe_load(text)
    except ImportError:
        pass
    task: dict = {}
    key = None
    block_indent = None
    for line in text.splitlines():
        if block_indent is not None:
            if line.strip() == "" or line.startswith(" " * block_indent):
                task[key] = task[key] + line[block_indent:] + "\n"
                continue
            block_indent = None
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if line.startswith("  - "):
            if not isinstance(task.get(key), list):
                task[key] = []
            task[key].append(json.loads(line[4:]))
            continue
        if line.startswith("  ") and key is not None and isinstance(task.get(key), dict):
            name, _, value = stripped.partition(":")
            task[key][name.strip()] = json.loads(value.strip())
            continue
        name, _, value = stripped.partition(":")
        key = name.strip()
        value = value.strip()
        if value == "|":
            task[key] = ""
            block_indent = 2
        elif value == "":
            task[key] = {}
        else:
            task[key] = json.loads(value) if value[0] in '"[{0123456789-' else value
    return task


def collect_sources(band_dir: Path, task: dict, submission: Path) -> dict[str, str]:
    sources = {}
    for entry in task["files"]:
        if entry["source"] == "@SUBMISSION@":
            sources[entry["name"]] = submission.read_text()
        else:
            sources[entry["name"]] = (band_dir / entry["source"]).read_text()
    return sources


def build_test_string(cases: list[dict], overrides: dict[str, int]) -> str:
    out = ""
    for case in cases:
        merged = {**case, **overrides}
        out += "; ".join(f"{k}: {v}" for k, v in merged.items()) + "\n"
    return out


# ------------------------------------------------------------------ evaluation

def run_one(work: Path, mode: str, cases_text: str, timeout: int, seed: int, env_extra: dict,
            pool_bytes: dict | None = None):
    """One eval.py process; mirrors run_eval.run_program.

    eval.py runs in its own session so that, when ``timeout`` passes, the whole
    process group is killed: a submission process that outlived eval.py would
    otherwise keep the GPU and could hold the result pipe open. The pipe is
    drained by a thread from the start, so a chatty result channel cannot fill
    the pipe buffer and the read can never block this function past the
    timeout.
    """
    cases_path = work / f"cases-{mode}.txt"
    cases_path.write_text(cases_text)
    env = os.environ.copy()
    env.update(env_extra)
    pool_dir = None
    if pool_bytes:
        # The pool is materialized immediately before this process starts and
        # eval.py deletes it as soon as the arrays are in RAM -- which happens
        # before any submission process exists. Nothing is on disk while the
        # submission runs.
        pool_dir = Path(tempfile.mkdtemp(prefix="sutro-pool-"))
        os.chmod(pool_dir, 0o700)
        for name, payload in pool_bytes.items():
            (pool_dir / name).write_bytes(payload)
        env["MNIST_POOL_CACHE"] = str(pool_dir)
        env["MNIST_POOL_CONSUME"] = "1"
    read_fd, write_fd = os.pipe()
    env["POPCORN_FD"] = str(write_fd)
    env["POPCORN_SEED"] = str(seed)
    chunks: list[str] = []

    def drain():
        with os.fdopen(read_fd, "r") as pipe:
            chunks.append(pipe.read())

    reader = threading.Thread(target=drain, daemon=True)
    reader.start()
    started = time.perf_counter()
    try:
        process = subprocess.Popen(
            [sys.executable, "eval.py", mode, str(cases_path)],
            cwd=work,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            pass_fds=[write_fd],
            start_new_session=True,
        )
    finally:
        os.close(write_fd)  # the child holds its own copy; EOF now follows the children
    try:
        stdout, stderr = process.communicate(timeout=timeout)
        code = process.returncode
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        stdout, stderr = process.communicate()
        code = -1
    duration = time.perf_counter() - started
    if pool_dir is not None:
        shutil.rmtree(pool_dir, ignore_errors=True)
    reader.join(timeout=10)
    raw = "".join(chunks)
    result = {}
    for line in raw.splitlines():
        key, _, value = line.partition(":")
        if key or value:
            result[key.strip()] = value.strip()
    return {
        "mode": mode,
        "exit_code": code,
        "duration_s": round(duration, 2),
        "passed": result.get("check") == "pass",
        "result": result,
        "stdout": (stdout or "")[-8000:],
        "stderr": (stderr or "")[-8000:],
    }


def compile_submission(work: Path, env_extra: dict, timeout: int = 180) -> dict:
    """KernelBot compiles a Python submission by running it once, before eval.py.

    run_eval.run_pytorch_script executes ``python3 submission.py`` in the work
    directory ahead of the evaluator, outside every guard eval.py installs. The
    module-level inertness check makes that step inert, and running it here is
    what keeps the hosted sequence honest in our own runs.
    """
    env = {key: value for key, value in os.environ.items() if not key.startswith("POPCORN_")}
    env.update(env_extra)
    started = time.perf_counter()
    try:
        done = subprocess.run(
            [sys.executable, "submission.py"],
            cwd=work,
            capture_output=True,
            text=True,
            env=env,
            timeout=timeout,
            start_new_session=True,
        )
        code, stdout, stderr = done.returncode, done.stdout, done.stderr
    except subprocess.TimeoutExpired as error:
        code, stdout, stderr = -1, error.stdout or "", error.stderr or ""
    return {
        "mode": "compile",
        "exit_code": code,
        "duration_s": round(time.perf_counter() - started, 2),
        "passed": code == 0,
        "result": {"check": "pass" if code == 0 else "fail"},
        "stdout": (stdout or "")[-8000:],
        "stderr": (stderr or "")[-8000:],
    }


def evaluate(sources: dict[str, str], task: dict, mode: str, overrides: dict, seed: int,
             env_extra: dict, work_root: str | None = None,
             pool_bytes: dict | None = None) -> dict:
    """Mirrors run_eval.run_pytorch_script + run_evaluation for one submission."""
    tests = build_test_string(task.get("tests", []), overrides)
    benchmarks = build_test_string(task.get("benchmarks", []), overrides)
    ranked = benchmarks.splitlines(keepends=True)[-1] if task.get("ranking_by", "last") == "last" \
        else benchmarks
    timeouts = {
        "test": task.get("test_timeout", 180),
        "benchmark": task.get("benchmark_timeout", 180),
        "profile": task.get("benchmark_timeout", 180),
        "leaderboard": task.get("ranked_timeout", 180),
    }
    sequence = {
        "test": ["test"],
        "benchmark": ["benchmark"],
        "profile": ["profile"],
        "leaderboard": ["test", "benchmark", "leaderboard"],
    }[mode]

    runs = {}
    with tempfile.TemporaryDirectory(dir=work_root) as directory:
        work = Path(directory)
        for name, contents in sources.items():
            (work / name).write_text(contents)
        runs["compile"] = compile_submission(work, env_extra)
        if not runs["compile"]["passed"]:
            return {"runs": runs, "passed": False}
        for step in sequence:
            cases_text = tests if step == "test" else ranked
            runs[step] = run_one(
                work, step, cases_text, timeouts[step], seed, env_extra, pool_bytes
            )
            if not runs[step]["passed"]:
                break
    return {"runs": runs, "passed": all(run["passed"] for run in runs.values())}


# ------------------------------------------------------------------ modal

# The image mirrors src/runners/modal_runner.py's cuda_image where it matters
# for a PyTorch submission: the same CUDA base, the same add_python, the same
# numpy and torch pins. CUTLASS, cuDSL, nsight and the PCH volume are omitted.
try:
    import modal as _modal
except ImportError:  # --local does not need Modal installed
    _modal = None

if _modal is not None:
    image = (
        _modal.Image.from_registry(f"nvidia/cuda:{CUDA_TAG}", add_python="3.13")
        .run_commands("ln -sf $(which python) /usr/local/bin/python3")
        .apt_install("git", "curl")
        .uv_pip_install("wheel~=0.45", NUMPY_PIN, "PyYAML")
        .uv_pip_install(TORCH_PIN)
        .run_commands(
            # The container has no egress while a submission runs, so the pools
            # are fetched at build time. The evaluator verifies their MD5s and
            # deletes this directory once the arrays are in RAM, which is why a
            # submission cannot read the public labels off the container disk.
            f"mkdir -p {BAKED_POOL}",
            *[
                f"curl -sSfL -o {BAKED_POOL}/{name} {url}"
                for name, url in BAKED_SOURCES
            ],
        )
        .add_local_dir(str(HERE), "/root/harness")
        .add_local_python_source("run_modal")
    )
    app = _modal.App("sutro-mnist-medium-time", image=image)

    # block_network is the only real network defence: the in-process guard in
    # utils.install_network_guard cannot stop "subprocess curl", which is a
    # fresh interpreter without the audit hook. The datasets are baked into the
    # image, so the container needs no egress of its own.
    @app.function(
        gpu=GPU, timeout=MODAL_TIMEOUT, max_containers=1, block_network=BLOCK_NETWORK
    )
    def remote_evaluate(sources: dict, task: dict, mode: str, overrides: dict, seed: int,
                        env_extra: dict | None = None) -> dict:
        """Runs inside the container: the same evaluate() this file uses locally."""
        # Lift the baked pool into RAM and remove it from the image's filesystem
        # before anything else happens in this container.
        baked = Path(BAKED_POOL)
        pool_bytes = {path.name: path.read_bytes() for path in sorted(baked.glob("*.gz"))}
        shutil.rmtree(baked, ignore_errors=True)
        payload = evaluate(
            sources, task, mode, overrides, seed, env_extra or {},
            work_root="/tmp", pool_bytes=pool_bytes,
        )
        try:
            import torch

            payload["gpu"] = torch.cuda.get_device_name(0)
            payload["torch"] = torch.__version__
        except Exception as error:  # pragma: no cover
            payload["gpu"] = f"unknown: {error!r}"
        return payload


def run_on_modal(sources, task, mode, overrides, seed, env_extra=None):
    """One A100, one container, patient about capacity."""
    if _modal is None:
        raise SystemExit("modal is not installed; use --local or pip install modal")
    deadline = time.time() + CAPACITY_RETRY_SECONDS
    delay = 30
    while True:
        try:
            with _modal.enable_output(), app.run():
                return remote_evaluate.remote(
                    sources, task, mode, overrides, seed, env_extra or {}
                )
        except Exception as error:
            message = str(error).lower()
            transient = any(
                word in message
                for word in ("capacity", "unavailable", "resource", "try again",
                             "no gpu", "429", "temporarily")
            )
            if not transient or time.time() > deadline:
                raise
            print(f"[retry in {delay}s] {error}", file=sys.stderr)
            time.sleep(delay)
            delay = min(delay * 2, 300)


# ------------------------------------------------------------------ cli

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--band", default="mnist-medium-5pct")
    parser.add_argument("--submission", default="submissions/ncm_baseline.py")
    parser.add_argument("--mode", default="test",
                        choices=["test", "benchmark", "leaderboard", "profile"])
    parser.add_argument("--seed", type=int, default=20260922, help="the secret POPCORN_SEED")
    parser.add_argument("--output", type=Path, help="write the full result JSON here")
    parser.add_argument("--local", action="store_true",
                        help="run on this machine instead of Modal (CPU is fine)")
    parser.add_argument("--device", default=None,
                        help="MNIST_EVAL_DEVICE for --local; defaults to cpu")
    parser.add_argument("--pool-cache", default=os.environ.get("MNIST_POOL_CACHE"),
                        help="directory of verified idx.gz files, for offline local runs")
    parser.add_argument("--env", action="append", default=[], metavar="KEY=VALUE",
                        help="extra environment variable for the evaluator process; "
                             "forwarded into the Modal container too")
    parser.add_argument("--case", action="append", default=[], metavar="KEY=VALUE",
                        help="override a case field, e.g. --case draws=2")
    args = parser.parse_args()

    if args.output and args.output.exists():
        print(f"refusing to overwrite {args.output}", file=sys.stderr)
        return 2

    band_dir = HERE / args.band
    task = load_task(band_dir)
    submission = Path(args.submission)
    if not submission.is_absolute():
        submission = HERE / submission
    sources = collect_sources(band_dir, task, submission)
    overrides = {}
    for item in args.case:
        key, _, value = item.partition("=")
        overrides[key.strip()] = int(value)

    env_extra = {}
    for item in args.env:
        key, _, value = item.partition("=")
        env_extra[key.strip()] = value

    started = time.time()
    if args.local:
        env = {"MNIST_EVAL_DEVICE": args.device or "cpu"}
        if args.pool_cache:
            env["MNIST_POOL_CACHE"] = str(Path(args.pool_cache).resolve())
        env.update(env_extra)
        payload = evaluate(sources, task, args.mode, overrides, args.seed, env)
        payload["gpu"] = env["MNIST_EVAL_DEVICE"]
    else:
        payload = run_on_modal(sources, task, args.mode, overrides, args.seed, env_extra)

    payload.update(
        band=args.band,
        submission=str(submission.relative_to(HERE)) if submission.is_relative_to(HERE)
        else str(submission),
        mode=args.mode,
        seed=args.seed,
        overrides=overrides,
        env=env_extra,
        where="local" if args.local else "modal",
        wall_s=round(time.time() - started, 1),
    )

    for step, run in payload["runs"].items():
        verdict = "pass" if run["passed"] else "FAIL"
        print(f"[{step}] {verdict} exit={run['exit_code']} {run['duration_s']}s")
        for key, value in run["result"].items():
            print(f"    {key}: {value}")
        if not run["passed"] and run["stderr"]:
            print(run["stderr"][-2000:], file=sys.stderr)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote {args.output}")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
