"""One bounded, paired development probe, not an official score.

Compare baseline and changed MLPs on six public-seeded MNIST draws. Runs
unsandboxed in one A100 process; only the unchanged official scorer may
produce submission scores. No probe data or learned state enters submissions.
"""
import hashlib
import json
import sys
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
if modal.is_local():
    PORT = HERE.parents[2]
    sys.path.insert(0, str(PORT))
    from run_modal import image
else:
    image = None

app = modal.App("mnist-a100-improved-mlp-dev-20260930", image=image)


@app.function(gpu="A100-80GB", cpu=(2, 4), memory=(4096, 8192), timeout=600,
              max_containers=1, single_use_containers=True)
def probe(sources):
    import statistics
    import time
    import types
    import torch

    sys.path.insert(0, "/root")
    import mnist

    torch.set_num_threads(1)
    device = torch.device("cuda")
    modules = {}
    for name, source in sources.items():
        module = types.ModuleType(name)
        exec(compile(source, name + ".py", "exec"), module.__dict__)
        modules[name] = module
    results = {name: [] for name in modules}
    seeds = list(range(9401, 9407))
    warm = [torch.from_numpy(a).to(device) for a in mnist.draw("fashion", 9400)]
    for name, module in modules.items():
        torch.backends.cuda.matmul.allow_tf32 = name != "baseline_d1"
        torch.backends.cudnn.allow_tf32 = name != "baseline_d1"
        module.custom_kernel(*warm[:3])
        torch.cuda.synchronize()
    for seed in seeds:
        draw = [torch.from_numpy(a).to(device) for a in mnist.draw("mnist", seed)]
        for name, module in modules.items():
            torch.backends.cuda.matmul.allow_tf32 = name != "baseline_d1"
            torch.backends.cudnn.allow_tf32 = name != "baseline_d1"
            torch.cuda.synchronize()
            start = time.perf_counter()
            prediction = module.custom_kernel(*draw[:3])
            torch.cuda.synchronize()
            ms = (time.perf_counter() - start) * 1000
            errors = int((prediction != draw[3]).sum().item())
            row = {"seed": seed, "errors": errors, "n_test": len(prediction), "ms": ms}
            results[name].append(row)
            print(name, row, flush=True)
    return {
        "kind": "paired public-seed development probe; unsandboxed; not official",
        "device": torch.cuda.get_device_name(), "torch": str(torch.__version__),
        "seeds": seeds,
        "sources_sha256": {k: hashlib.sha256(v.encode()).hexdigest() for k, v in sources.items()},
        "calls": results,
        "summary": {name: {"error_percent": sum(r["errors"] for r in rows) / sum(r["n_test"] for r in rows) * 100,
                           "mean_ms": statistics.mean(r["ms"] for r in rows)} for name, rows in results.items()},
    }


def main():
    base = HERE.parents[1] / "baselines-20260930"
    sources = {"baseline_d1": (base / "baseline_d1.py").read_text(),
               "mlp_d1": (HERE.parent / "mlp_d1.py").read_text(),
               "baseline_d2": (base / "baseline_d2.py").read_text(),
               "mlp_d2": (HERE / "dev-mlp-d2-k8-s400.py").read_text()}
    with modal.enable_output(), app.run():
        result = probe.remote(sources)
        result["app_id"] = app.app_id
    (HERE / "dev-mlp-probe.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
