"""One Modal A100-80GB container: bigger minibatches, fewer steps, same number of epochs (162, as 6,500 steps at 250).

    modal run probe_batch.py      # ~8-10 min, ~$0.40

Same six draws as probe_sched.py (seeds 301-306). The schedule keeps the recipe's shape and is rescaled in place
for each learning rate, so one captured graph serves every rate at a batch size.
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

CONFIGS = [(250, 6500, (0.004,)), (500, 3250, (0.004, 0.006, 0.008)), (1000, 1625, (0.004, 0.008, 0.012)),
           (2000, 813, (0.008, 0.012, 0.016))]


@app.function(gpu="A100-80GB", timeout=1500, single_use_containers=True)
def probe():
    import sys
    sys.path.insert(0, "/root")
    import torch, mnist, ladder_tb as L

    dev = torch.device("cuda")
    draws = [[torch.from_numpy(a).to(dev) for a in mnist.draw("mnist", s)] for s in range(301, 307)]
    results = {}
    for batch, steps, rates in CONFIGS:
        L.BATCH, L.STEPS = batch, steps
        L.ladder(*draws[0][:3])  # build and capture
        trainer = next(iter(L.CACHE.values()))
        base = trainer.schedule.clone()  # built at L.LR = 0.002
        for lr in rates:
            trainer.schedule.copy_(base * (lr / L.LR))
            errs, secs = [], []
            for tx, ty, qx, qy in draws:
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                labels = L.ladder(tx, ty, qx)
                end.record(); torch.cuda.synchronize()
                errs.append((labels != qy).float().mean().item() * 100)
                secs.append(start.elapsed_time(end) / 1000)
            key = f"batch {batch} steps {trainer.total} lr {lr}"
            results[key] = (sum(errs) / len(errs), errs, sum(secs) / len(secs))
            print(f"{key:32s} mean {results[key][0]:.3f}%  {results[key][2]:.2f} s/call  draws {[round(e, 2) for e in errs]}",
                  flush=True)
    return results


@app.local_entrypoint()
def main():
    print(probe.remote())
