"""One Modal A100-80GB container: how many steps each difficulty needs at batch 1,000, TF32, lr 0.008.

    modal run probe_epochs.py      # ~8-10 min, ~$0.50

Short and middle budgets on the six draws of the earlier probes (seeds 301-306); the long budgets that decide
difficulty 5 on eleven (301-311). Targets: dev mean <= band - 0.15 (2.55 / 2.15 / 1.75%).
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

CONFIGS = [(600, (0.008,), 6), (900, (0.008,), 6), (1200, (0.008,), 6), (2400, (0.008,), 6), (3600, (0.008,), 6),
           (6000, (0.008,), 11), (9000, (0.008, 0.006), 11)]


@app.function(gpu="A100-80GB", timeout=1800, single_use_containers=True)
def probe():
    import sys
    sys.path.insert(0, "/root")
    import torch, mnist, ladder_tb as L

    torch.backends.cuda.matmul.allow_tf32 = True
    dev = torch.device("cuda")
    draws = [[torch.from_numpy(a).to(dev) for a in mnist.draw("mnist", s)] for s in range(301, 312)]
    results = {"gpu": torch.cuda.get_device_name()}
    L.BATCH = 1000
    for steps, rates, n_draws in CONFIGS:
        L.STEPS = steps
        L.CACHE.clear()
        L.ladder(*draws[0][:3])  # build and capture
        trainer = next(iter(L.CACHE.values()))
        base = trainer.schedule.clone()
        for lr in rates:
            trainer.schedule.copy_(base * (lr / L.LR))
            errs, secs = [], []
            for tx, ty, qx, qy in draws[:n_draws]:
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                labels = L.ladder(tx, ty, qx)
                end.record(); torch.cuda.synchronize()
                errs.append((labels != qy).float().mean().item() * 100)
                secs.append(start.elapsed_time(end) / 1000)
            key = f"steps {trainer.total} lr {lr} ({n_draws} draws)"
            results[key] = (sum(errs) / len(errs), errs, sum(secs) / len(secs))
            print(f"{key:34s} mean {results[key][0]:.3f}%  {results[key][2]:.2f} s/call  draws {[round(e, 2) for e in errs]}",
                  flush=True)
    return results


@app.local_entrypoint()
def main():
    print(probe.remote())
