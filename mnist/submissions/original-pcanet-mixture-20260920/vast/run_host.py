"""Run the frozen K=100 validation on one already-provisioned A100 container.

This does not contact Vast.ai or provision resources. Run separately on each
host with Python from pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel. Only missing
or mismatched pinned Python dependencies are installed. No CUDA context is
created by this runner's preflight; validate.py owns the first context.
"""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import signal
import subprocess
import sys
import time


PACKAGE = Path(__file__).resolve().parents[1]
PINS = {"numpy": "2.1.2", "nvidia-ml-py": "13.610.43", "ninja": "1.11.1.1"}
TORCH_PROBE = """
import json
import torch
assert not torch.cuda.is_initialized(), 'Import unexpectedly initialized CUDA'
info = {'version': torch.__version__, 'cuda': torch.version.cuda,
        'cudnn': torch.backends.cudnn.version(),
        'cuda_initialized': torch.cuda.is_initialized()}
assert not info['cuda_initialized'], 'Version query initialized CUDA'
print(json.dumps(info))
"""


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def inventory():
    result = {}
    for name in ("torch", *PINS):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def console(text):
    try:
        sys.stdout.write(text)
        sys.stdout.flush()
    except BrokenPipeError:
        # Preserve local evidence even if the SSH client's output pipe closes.
        pass


def tee(command, log, env):
    console("$ " + shlex.join(command) + "\n")
    log.write(("$ " + shlex.join(command) + "\n").encode())
    log.flush()
    process = subprocess.Popen(command, cwd=PACKAGE, env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        for line in iter(process.stdout.readline, b""):
            log.write(line)
            log.flush()
            console(line.decode("utf-8", errors="replace"))
        return process.wait()
    except BaseException:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        raise
    finally:
        process.stdout.close()


def capture(command, env):
    result = subprocess.run(command, cwd=PACKAGE, env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return {"command": command, "returncode": result.returncode,
            "stdout": result.stdout, "stderr": result.stderr}


def torch_info(env):
    record = capture([sys.executable, "-c", TORCH_PROBE], env)
    if record["returncode"]:
        return None, record
    return json.loads(record["stdout"]), record


def matching_torch(info):
    return info is not None and info["version"].split("+")[0] == "2.5.1" and info["cuda"] == "12.4"


def dependencies(log, env, no_install):
    before = inventory()
    info, probe = torch_info(env) if before["torch"] else (None, None)
    installs = []
    if not matching_torch(info):
        if no_install:
            raise RuntimeError("PyTorch 2.5.1 built for CUDA 12.4 is required; dependency installation is disabled")
        command = [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "--no-input",
                   "--upgrade", "--force-reinstall", "torch==2.5.1+cu124",
                   "--index-url", "https://download.pytorch.org/whl/cu124"]
        installs.append(command)
        if tee(command, log, env):
            raise RuntimeError("Installation of pinned CUDA PyTorch failed")
    current = inventory()
    missing = [f"{name}=={version}" for name, version in PINS.items() if current[name] != version]
    if missing:
        if no_install:
            raise RuntimeError("Missing or mismatched pinned dependencies: " + ", ".join(missing))
        command = [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "--no-input", *missing]
        installs.append(command)
        if tee(command, log, env):
            raise RuntimeError("Installation of pinned Python dependencies failed")
    final = inventory()
    info, final_probe = torch_info(env)
    if not matching_torch(info) or any(final[name] != version for name, version in PINS.items()):
        raise RuntimeError("Installed Python environment does not match the pinned versions")
    return {"before": before, "after": final, "installation_commands": installs,
            "initial_torch_probe": probe, "final_torch_probe": final_probe, "torch": info}


def preflight(env):
    record = {"started_utc": utc(), "hostname": platform.node(), "platform": platform.platform(),
              "python": platform.python_version(), "python_executable": sys.executable,
              "package_path": str(PACKAGE),
              "environment": {key: env.get(key) for key in ("MAX_JOBS", "TORCH_CUDA_ARCH_LIST",
                              "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES", "CUDA_HOME", "CXX")}}
    smi = capture(["nvidia-smi", "--query-gpu=index,name,uuid,pci.bus_id,driver_version,memory.total,power.limit",
                   "--format=csv,noheader,nounits"], env)
    record["nvidia_smi_query"] = smi
    if smi["returncode"]:
        raise RuntimeError("nvidia-smi could not query the allocated GPU")
    rows = list(csv.reader(line for line in smi["stdout"].splitlines() if line.strip()))
    if len(rows) != 1 or rows[0][0].strip() != "0" or rows[0][1].strip() != "NVIDIA A100-SXM4-40GB":
        raise RuntimeError("Exactly one exposed NVIDIA A100-SXM4-40GB at index 0 is required; " + smi["stdout"].strip())
    names = ("index", "name", "uuid", "pci_bus_id", "driver", "memory_mib", "power_limit_w")
    record["gpu"] = dict(zip(names, (value.strip() for value in rows[0])))
    visible = env.get("CUDA_VISIBLE_DEVICES")
    if visible is not None and visible not in ("0", record["gpu"]["uuid"]):
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be unset, 0, or the sole exposed GPU UUID so CUDA and NVML address the same device")
    compiler = env.get("CXX") or shutil.which("c++") or shutil.which("g++")
    if not compiler:
        raise RuntimeError("A C++ compiler is required for the unchanged CUDA extension")
    record["cxx"] = capture([compiler, "--version"], env)
    nvcc = shutil.which("nvcc")
    if nvcc is None and env.get("CUDA_HOME"):
        candidate = Path(env["CUDA_HOME"]) / "bin" / "nvcc"
        nvcc = str(candidate) if candidate.is_file() else None
    if nvcc is None:
        raise RuntimeError("CUDA 12.4 nvcc is required; use the CUDA development image")
    record["nvcc"] = capture([nvcc, "--version"], env)
    if record["nvcc"]["returncode"] or "release 12.4" not in record["nvcc"]["stdout"]:
        raise RuntimeError("CUDA compiler must be release 12.4 to match the frozen environment")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("generated/vast-rerun"))
    parser.add_argument("--raw", type=Path, default=Path("raw"))
    parser.add_argument("--no-install", action="store_true", help="Fail instead of installing missing/mismatched pinned packages")
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else PACKAGE / args.output
    raw = args.raw if args.raw.is_absolute() else PACKAGE / args.raw
    if output.exists():
        parser.error(f"Refusing to overwrite existing results: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(output.name + ".runner")
    staging.mkdir(exist_ok=False)
    console(f"Runner log while work is active: {staging / 'run.log'}\n")
    env = os.environ.copy()
    env.update(MAX_JOBS="2", TORCH_CUDA_ARCH_LIST="8.0", PYTHONUNBUFFERED="1")
    command = [sys.executable, "-u", str(PACKAGE / "validate.py"), "--dimensions", "100",
               "--rounds", "4", "--repeats", "60", "--idle-seconds", "10",
               "--raw", str(raw), "--output", str(output)]
    execution = {"started_utc": utc(), "command": command, "returncode": None,
                 "validation_started": False, "provider": "vast.ai",
                 "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    source_before = {name: hashlib.sha256((PACKAGE / name).read_bytes()).hexdigest()
                     for name in ("model.py", "cuda_kernel.py", "data.py", "validate.py")}
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    failed = False
    with (staging / "run.log").open("xb") as log:
        try:
            record = preflight(env)
            write_json(staging / "preflight.json", record)
            record["dependencies"] = dependencies(log, env, args.no_install)
            record["source_sha256"] = source_before
            record["completed_utc"] = utc()
            write_json(staging / "preflight.json", record)
            execution["validation_started"] = True
            execution["returncode"] = tee(command, log, env)
            failed = execution["returncode"] != 0
        except (Exception, KeyboardInterrupt) as exc:
            failed = True
            execution["runner_error"] = f"{type(exc).__name__}: {exc}"
            message = execution["runner_error"] + "\n"
            console(message)
            log.write(message.encode())
        finally:
            execution["completed_utc"] = utc()
            source_after = {name: hashlib.sha256((PACKAGE / name).read_bytes()).hexdigest() for name in source_before}
            execution["source_unchanged"] = source_before == source_after
            if not execution["source_unchanged"]:
                failed = True
                execution["runner_error"] = "Frozen sources changed during the run"
            # validate.py itself creates this directory. Only create it here if
            # validation never started, so preflight failures remain inspectable.
            output.mkdir(parents=True, exist_ok=True)
            write_json(staging / "execution.json", execution)
    for path in staging.iterdir():
        path.replace(output / path.name)
    staging.rmdir()
    if not failed:
        result = subprocess.run([sys.executable, str(PACKAGE / "verify_results.py"),
                                 "--results", str(output), "--raw", str(raw)],
                                cwd=PACKAGE, env=env, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
        (output / "verification.json").write_text(result.stdout)
        if result.stderr:
            (output / "verification-stderr.txt").write_text(result.stderr)
        failed = result.returncode != 0
        console("Offline verification: " + ("FAILED" if failed else "passed") + "\n")
    console(f"Evidence saved in {output}\n")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
