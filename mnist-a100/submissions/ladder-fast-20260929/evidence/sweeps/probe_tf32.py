"""One Modal A100-80GB container: TF32 tensor-core matmuls vs full FP32, at the best batch/lr settings so far.

    modal run probe_tf32.py      # ~6-8 min, ~$0.30

Same six draws as probe_sched.py / probe_batch.py (seeds 301-306). TF32 is set before each capture.
"""
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
image = (
    modal.Image.from_registry("nvidia/cuda:13.3.0-devel-ubuntu24.04", add_python="3.13")
    .uv_pip_install("numpy~=2.3")
    .uv_pip_install("torch==2.12.0")
    .add_local_file(HERE / "mnist.py", "/root/mnist.py", copy=True)
    .run_commands("python /root/mnist.py --download")
    .add_local_file(HERE / "ladder_tb.py", "/root/ladder_tb.py")
)
app = modal.App("sutro-ladder-probe", image=image)

CONFIGS = [(500, 3250, 0.006), (1000, 1625, 0.008), (2000, 813, 0.008)]


@app.function(gpu="A100-80GB", timeout=1500, single_use_containers=True)
def probe():
    import sys
    sys.path.insert(0, "/root")
    import torch, mnist, ladder_tb as L

    dev = torch.device("cuda")
    draws = [[torch.from_numpy(a).to(dev) for a in mnist.draw("mnist", s)] for s in range(301, 307)]
    results = {"gpu": torch.cuda.get_device_name()}
    for batch, steps, lr in CONFIGS:
        for tf32 in (False, True):
            torch.backends.cuda.matmul.allow_tf32 = tf32
            L.BATCH, L.STEPS = batch, steps
            L.CACHE.clear()
            L.ladder(*draws[0][:3])  # build and capture with this precision
            trainer = next(iter(L.CACHE.values()))
            trainer.schedule.mul_(lr / L.LR)
            errs, secs = [], []
            for tx, ty, qx, qy in draws:
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                labels = L.ladder(tx, ty, qx)
                end.record(); torch.cuda.synchronize()
                errs.append((labels != qy).float().mean().item() * 100)
                secs.append(start.elapsed_time(end) / 1000)
            key = f"batch {batch} lr {lr} {'TF32' if tf32 else 'FP32'}"
            results[key] = (sum(errs) / len(errs), errs, sum(secs) / len(secs))
            print(f"{key:28s} mean {results[key][0]:.3f}%  {results[key][2]:.2f} s/call  draws {[round(e, 2) for e in errs]}",
                  flush=True)
    return results


@app.local_entrypoint()
def main():
    print(probe.remote())
