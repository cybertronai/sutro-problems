"""One Modal A100-80GB container: loss-proportional sampling of labelled rows on the retuned Ladder (batch 1,000, TF32).

    modal run probe_samp.py      # ~10-12 min, ~$0.50

Same six draws as the earlier probes (seeds 301-306). Scores are each row's latest training cross-entropy,
recorded as a side effect of the step. mix_t is a per-step table: 0 for the first 10% (scores warm up), then
mix0, then (annealed variants) linear back to 0 between 60% and 90%, uniform after.
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
    .add_local_file(HERE / "ladder_samp.py", "/root/ladder_samp.py")
)
app = modal.App("sutro-ladder-probe", image=image)

CONFIGS = [(1200, 0.008), (2400, 0.008), (9000, 0.006)]


def mix_table(total, mix0, anneal):
    import numpy as np
    f = np.arange(total) / total
    m = np.where(f < 0.1, 0.0, mix0)
    if anneal:
        m = np.where(f >= 0.6, mix0 * np.clip((0.9 - f) / 0.3, 0, 1), m)
    return m


@app.function(gpu="A100-80GB", timeout=1800, single_use_containers=True)
def probe():
    import sys
    sys.path.insert(0, "/root")
    import numpy as np, torch, mnist, ladder_samp as L

    torch.backends.cuda.matmul.allow_tf32 = True
    dev = torch.device("cuda")
    draws = [[torch.from_numpy(a).to(dev) for a in mnist.draw("mnist", s)] for s in range(301, 307)]
    results = {"gpu": torch.cuda.get_device_name()}
    L.BATCH = 1000

    def run(key, trainer, sched):
        trainer.schedule.copy_(sched)
        errs, secs = [], []
        for tx, ty, qx, qy in draws:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            labels = L.ladder(tx, ty, qx)
            end.record(); torch.cuda.synchronize()
            errs.append((labels != qy).float().mean().item() * 100)
            secs.append(start.elapsed_time(end) / 1000)
        results[key] = (sum(errs) / len(errs), errs, sum(secs) / len(secs))
        print(f"{key:26s} mean {results[key][0]:.3f}%  {results[key][2]:.2f} s/call  draws {[round(e, 2) for e in errs]}",
              flush=True)

    for steps, lr in CONFIGS:
        L.STEPS = steps
        L.CACHE.clear()
        L.SAMPLE = False
        L.ladder(*draws[0][:3])
        tr = next(iter(L.CACHE.values()))
        sched = tr.schedule.clone() * (lr / L.LR)
        run(f"steps {tr.total} epoch-shuffle", tr, sched)
        L.CACHE.clear()
        L.SAMPLE = True
        L.ladder(*draws[0][:3])
        tr = next(iter(L.CACHE.values()))
        T = tr.total
        for name, mix0, anneal in (("iid-uniform", 0.0, False), ("mix50-anneal", 0.5, True),
                                   ("mix90-anneal", 0.9, True), ("mix50-hold", 0.5, False)):
            tr.mix.copy_(torch.tensor(mix_table(T, mix0, anneal), dtype=torch.float32, device=dev))
            run(f"steps {T} {name}", tr, sched)
    return results


@app.local_entrypoint()
def main():
    print(probe.remote())
