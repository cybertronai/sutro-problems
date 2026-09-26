"""Score a method on a Modal A100-80GB, the way official times are taken.

    pip install modal && modal setup                            # once
    python run_modal.py example.py:mlp --difficulty 1           # one run
    python run_modal.py example.py:mlp --difficulty 1 --runs 3  # three containers; a record takes the median

Each run is its own container, running `python mnist.py FILE:FUNCTION --difficulty N`
as root: the scorer is a clean process and the method runs sandboxed (uid 65534,
seccomp, no network). The image follows popcorn3's KernelBot image: CUDA 13.3,
Python 3.13, torch 2.12.0.
"""

import argparse
import re
import statistics
import sys
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
image = (
    modal.Image.from_registry("nvidia/cuda:13.3.0-devel-ubuntu24.04", add_python="3.13")
    .uv_pip_install("numpy~=2.3", "ninja~=1.11")
    .uv_pip_install("torch==2.12.0")
    .add_local_file(HERE / "mnist.py", "/root/mnist.py", copy=True)
    .run_commands("python /root/mnist.py --download")  # into /root/.cache, unreadable to the sandbox user
)
app = modal.App("mnist-a100", image=image)


# One run per container, so --runs N really samples N hosts.
@app.function(gpu="A100-80GB", timeout=7200, max_containers=8, single_use_containers=True)
def remote_score(filename: str, source: str, function: str, options: list) -> dict:
    import os
    import subprocess

    os.makedirs("/root/entry", exist_ok=True)
    path = f"/root/entry/{filename}"
    Path(path).write_text(source)
    run = subprocess.run([sys.executable, "/root/mnist.py", f"{path}:{function}", *options],
                         capture_output=True, text=True, env={**os.environ, "MNIST_SANDBOX": "required"})
    return {"returncode": run.returncode, "stdout": run.stdout, "stderr": run.stderr[-6000:]}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("method", help="file.py:function")
    parser.add_argument("--difficulty", type=int, default=1, choices=range(1, 6))
    parser.add_argument("--runs", type=int, default=1)
    args = parser.parse_args()
    sys.path.insert(0, str(HERE))
    import mnist

    try:
        path, function = mnist.locate(args.method)
        source = path.read_bytes()
        mnist.check_source(source, function, path.name)  # fail here, not after paying for a container
    except (OSError, ValueError, TypeError, mnist.SourceError) as error:
        parser.error(str(error))
    job = (path.name, source.decode(), function, ["--difficulty", str(args.difficulty)])
    scores = []
    with modal.enable_output(), app.run():
        for index, result in enumerate(remote_score.starmap([job] * args.runs)):
            print(f"--- run {index + 1}:\n{result['stdout'].rstrip()}")
            if result["returncode"] not in (0, 1):  # a crash or a setup error: show the tail of stderr
                print(result["stderr"][-3000:])
            found = re.search(r"; score ([0-9.]+) ms$", result["stdout"], re.M)
            if found and result["returncode"] == 0:
                scores.append(float(found.group(1)))
    if args.runs > 1:
        print(f"passed {len(scores)} of {args.runs} runs" +
              (f"; median {statistics.median(scores):.3f} ms" if scores else ""))
    return 0 if len(scores) == args.runs else 1


if __name__ == "__main__":
    sys.exit(main())
