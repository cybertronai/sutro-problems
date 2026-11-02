"""D6 / d5 fresh-split runner on Modal A100-80GB (official image recipe: CUDA 13.3, torch 2.12.0).
Split s = mnist.Pool from default_rng(SEED + s) and its 11 draws (one official run's worth). Each job runs in a fresh
subprocess with empty caches: one kmnist warm-up call, then 11 calls per split; returns wrong counts and ms per call.
Only our own method files in bio/d6_probe/modal/ run (plus the scorer's mnist.py for data).

    cd upstream/sutro_bench && uv run --with modal modal run bio/d6_probe/d6_modal.py --specs 'f.py:fn=name' \
        --seed 1025000 --splits 0:12 --per 2 --out bio/d6_probe/d6_modal.jsonl
spec = file:fn[|json module overrides]=name, ';'-separated; --per = splits per container.
"""
import json
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
MNIST = HERE.parent.parent.parent / "sutro-problems" / "mnist-a100" / "mnist.py"
FILES = sorted(p.name for p in (HERE / "modal").glob("*.py"))

image = (
    modal.Image.from_registry("nvidia/cuda:13.3.0-devel-ubuntu24.04", add_python="3.13")
    .uv_pip_install("numpy~=2.3", "ninja~=1.11")
    .uv_pip_install("torch==2.12.0")
    .add_local_file(MNIST, "/root/mnist.py", copy=True)
    .run_commands("python /root/mnist.py --download")
)
for f in FILES:
    image = image.add_local_file(HERE / "modal" / f, f"/root/subs/{f}", copy=True)
app = modal.App("d6-splits", image=image)

CHILD = r"""
import importlib.util, json, sys, time
import numpy as np, torch
sys.path.insert(0, "/root")
import mnist
path, fn = @FILE@.split(":")
s = importlib.util.spec_from_file_location("cand", "/root/subs/" + path)
mod = importlib.util.module_from_spec(s); sys.modules["cand"] = mod; s.loader.exec_module(mod)
for k, v in json.loads(@CFG@).items():
    setattr(mod, k, tuple(v) if isinstance(v, list) else v)
m = getattr(mod, fn)
cu = lambda a: torch.from_numpy(a).cuda()
w = [cu(a) for a in mnist.draw("kmnist", seed=10999)]
t0 = time.time(); m(*w[:3]); torch.cuda.synchronize(); warm = time.time() - t0
pix, lab = mnist.load_pool("mnist")
out = []
for sp in @SPLITS@:
    rng = np.random.default_rng(@SEED@ + sp)
    pool = mnist.Pool("mnist", pix, lab, rng)
    wrong, ms = [], []
    for _ in range(11):
        d = pool.draw(rng)
        tx, ty, vx = (cu(d[k]) for k in ("train_x", "train_y", "test_x"))
        torch.cuda.synchronize(); t = time.perf_counter(); p = m(tx, ty, vx); torch.cuda.synchronize()
        ms.append(round((time.perf_counter() - t) * 1e3, 2))
        wrong.append(int((p.view(-1).long().cpu().numpy() != d["test_y"]).sum()))
    out.append(dict(split=sp, wrong=wrong, ms=ms))
    print("SPLIT", json.dumps(out[-1]), flush=True)
print("RESULT", json.dumps(dict(warm=round(warm, 1), out=out)), flush=True)
"""


@app.function(gpu="A100-80GB", timeout=2400, max_containers=8)
def job(spec: dict):
    import os, subprocess, sys, tempfile
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,power.limit", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    d = tempfile.mkdtemp()
    env = dict(os.environ, TRITON_CACHE_DIR=f"{d}/t", TORCHINDUCTOR_CACHE_DIR=f"{d}/i")
    code = (CHILD.replace("@FILE@", repr(spec["file"])).replace("@CFG@", repr(json.dumps(spec["cfg"])))
            .replace("@SPLITS@", repr(spec["splits"])).replace("@SEED@", str(spec["seed"])))
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, cwd=d, timeout=2300)
    res = next((json.loads(l[7:]) for l in r.stdout.splitlines() if l.startswith("RESULT ")), None)
    return dict(name=spec["name"], gpu=gpu, res=res, err=None if res else (r.stdout[-1500:] + r.stderr[-3000:]))


@app.local_entrypoint()
def main(specs: str, seed: int = 1025000, splits: str = "0:12", per: int = 2, out: str = "bio/d6_probe/d6_modal.jsonl"):
    lo, hi = map(int, splits.split(":"))
    jobs = []
    for sp in specs.split(";"):
        fs, name = sp.rsplit("=", 1)
        f, cfg = (fs.split("|", 1) + ["{}"])[:2]
        for a in range(lo, hi, per):
            jobs.append(dict(name=name, file=f, cfg=json.loads(cfg), seed=seed, splits=list(range(a, min(hi, a + per)))))
    with open(out, "a") as fh:
        for r in job.map(jobs, order_outputs=False):
            if r["res"] is None:
                print("FAILED", r["name"], r["err"])
                continue
            for o in r["res"]["out"]:
                fh.write(json.dumps(dict(name=r["name"], seed=seed, gpu=r["gpu"], warm=r["res"]["warm"], **o)) + "\n")
                print(r["name"], o["split"], f"{sum(o['wrong']) / 1100:.3f}%", f"med {sorted(o['ms'])[5]:.0f} ms", r["gpu"])
