"""Bounded Modal runner for original pMNIST and five Aminist transfer tasks."""
from __future__ import annotations
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time
import zlib

import modal
import numpy as np

ROOT = Path(__file__).resolve().parent
SUITE = Path(os.environ.get("AMINIST21_ROOT", "/Users/yaroslavvb/git/aminist-21-validation"))
sys.path.insert(0, str(SUITE))
from aminist21.common import sha, array_sha, write, read, utc, SEEDS

IMAGE_REF = "ghcr.io/ab-10/wikitext-bench@sha256:95de89319ba89c53a91d5440a5db4ff46f68b05031062c0a760ec3caa48dc42f"
NAMES = ["kmnist", "emnist_letters_aj", "qmnist_recovered", "fashion_mnist", "cifar10"]
CONFIG = {"seed":11,"permutation_seed":20260923,"epochs":150,"decay_start":100,
          "batch_size":100,"learning_rate":0.002,"device":"cuda","cuda_graph":True,
          "independent_reconstruction_stream":True,"checkpoint_selection":"last_fixed_epoch"}
app = modal.App("pmnist-ladder-transfer-20260923")
image = (modal.Image.from_registry(IMAGE_REF).pip_install("numpy==2.2.6")
         .env({"PYTHONPATH":"/root/pmnist:/root/suite"})
         .add_local_dir(ROOT, "/root/pmnist", ignore=["results", "research", "__pycache__", "*.log", "*.pdf"])
         .add_local_dir(SUITE/"aminist21", "/root/suite/aminist21", ignore=["__pycache__"]))


def pack(array):
    return {"bytes":zlib.compress(array.tobytes(),1),"shape":array.shape,"dtype":str(array.dtype)}


@app.function(image=image,gpu="A100-40GB",cpu=4,memory=8192,timeout=1800,
              min_containers=0,max_containers=4,buffer_containers=0,scaledown_window=2,retries=0)
def fit_remote(payload, provenance, config, deadline):
    import torch
    from train import fit
    from aminist21.worker import invoke
    if set(payload) != {"train_images","train_labels","test_images"}:
        raise ValueError("Forbidden payload fields")
    if time.time() >= deadline-60:
        raise TimeoutError("Absolute study deadline reached before fit")
    for name, expected in provenance["sources"].items():
        if sha(Path("/root/pmnist")/name) != expected:
            raise ValueError("Source changed after freeze: "+name)
    arrays = {key:np.frombuffer(zlib.decompress(row["bytes"]), dtype=row["dtype"]).reshape(row["shape"]).copy()
              for key,row in payload.items()}
    before = {key:array_sha(value) for key,value in arrays.items()}
    if before != provenance["input_sha256"]:
        raise ValueError("Input hashes changed")
    started = time.monotonic()
    predictions, details, artifacts = fit(**arrays, config=config, deadline=deadline)
    if before != {key:array_sha(value) for key,value in arrays.items()}:
        raise ValueError("Input array modified")
    metadata = {"completed_at_utc":utc(), "adapter":"train.py:train_predict",
                "adapter_source_sha256":sha("/root/pmnist/train.py"),"config":config,
                "input_sha256":before,"predictions_sha256":array_sha(predictions),
                "adapter_wall_seconds":time.monotonic()-started,"query_labels_supplied":False,
                "metadata":details,"provenance":provenance,"container_image":IMAGE_REF,
                "hardware":{"gpu":torch.cuda.get_device_name()},
                "software":{"python":platform.python_version(),"numpy":np.__version__,
                            "torch":str(torch.__version__),"cuda":torch.version.cuda}}
    return {"predictions":predictions,"metadata":metadata,**artifacts}


def source_hashes():
    return {name:sha(ROOT/name) for name in ["model.py","train.py","run_study.py"]}


def save_result(result, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target.with_suffix(".npz"), predictions=result["predictions"])
    np.savez_compressed(target.with_name(target.stem+"-logits.npz"), logits=result["logits"])
    target.with_suffix(".pt").write_bytes(result["checkpoint"])
    meta = result["metadata"]
    meta["prediction_file_sha256"] = sha(target.with_suffix(".npz"))
    meta["checkpoint_sha256"] = sha(target.with_suffix(".pt"))
    meta["logits_file_sha256"] = sha(target.with_name(target.stem+"-logits.npz"))
    write(target, meta)
    print("SAVED", target, "seconds", round(meta["adapter_wall_seconds"],2), flush=True)


def official_inputs():
    folder = ROOT.parent/"official-mnist-sample-curve-20260922/raw/source"
    specs = {"train-images-idx3-ubyte.gz":"f68b3c2dcbeaaa9fbdd348bbdeb94873",
             "train-labels-idx1-ubyte.gz":"d53e105ee54ea40749a09fcbcd1e9432",
             "t10k-images-idx3-ubyte.gz":"9fb629c4189551a2d022fa330f9573f3"}
    content = {}
    for name, expected in specs.items():
        blob = (folder/name).read_bytes()
        if hashlib.md5(blob).hexdigest() != expected:
            raise ValueError("Canonical MNIST checksum mismatch")
        content[name] = gzip.decompress(blob)
    train = np.frombuffer(content["train-images-idx3-ubyte.gz"],np.uint8,offset=16).reshape(60000,1,28,28)
    query = np.frombuffer(content["t10k-images-idx3-ubyte.gz"],np.uint8,offset=16).reshape(10000,1,28,28)
    labels = np.frombuffer(content["train-labels-idx1-ubyte.gz"],np.uint8,offset=8).astype(np.int64)
    return {"train_images":train.astype(np.float32)/255,"train_labels":labels,
            "test_images":query.astype(np.float32)/255}


@app.local_entrypoint()
def main(mode: str = "smoke"):
    # Shared hard deadlines and no retries bound this task's own GPU usage.
    limits = {"smoke":300, "official":1800, "transfer":3000, "all":3000}
    if mode not in limits:
        raise ValueError(mode)
    output = ROOT/"results"/mode
    output.mkdir(parents=True, exist_ok=True)
    execution_path = output/"execution.json"
    if execution_path.exists():
        raise RuntimeError("A launch record already exists; review before any explicit recovery run")
    started = time.time()
    deadline = started+limits[mode]
    write(execution_path,{"started_at_utc":utc(),"app_id":app.app_id,"deadline_unix":deadline,
          "max_gpu_workers":4 if mode in ("transfer","all") else 1,"mode":mode,
          "status":"running","resource_rate_upper_usd_per_second":0.00065316,
          "nominal_max_worker_cost_usd":limits[mode]*(4 if mode in ("transfer","all") else 1)*0.00065316,
          "cost_caveat":"Resource estimate, not provider invoice; includes 4 CPU cores and 8GiB RAM per GPU."})
    sources = source_hashes()
    (output/"source").mkdir(exist_ok=True)
    for name in sources:
        shutil.copy2(ROOT/name, output/"source"/name)
    config = dict(CONFIG)
    if mode == "smoke":
        rng = np.random.default_rng(9823)
        arrays = {"train_images":rng.random((200,1,9,9),dtype=np.float32),
                  "train_labels":rng.integers(0,10,200,dtype=np.int64),
                  "test_images":rng.random((100,1,9,9),dtype=np.float32)}
        config.update(epochs=2,decay_start=1)
        provenance = {"dataset":"synthetic","draw":0,"sources":sources,
                      "input_sha256":{k:array_sha(v) for k,v in arrays.items()}}
        save_result(fit_remote.remote({k:pack(v) for k,v in arrays.items()},provenance,config,deadline), output/"smoke.json")
    elif mode == "official":
        arrays = official_inputs()
        provenance = {"dataset":"official_mnist_60000_10000","draw":0,"sources":sources,
                      "input_sha256":{k:array_sha(v) for k,v in arrays.items()}}
        write(output/"plan.json",{"config":config,"sources":sources,"provenance":provenance,
              "frozen_at_utc":utc(),"test_labels_supplied":False,"test_scoring":"after all transfer predictions frozen"})
        save_result(fit_remote.remote({k:pack(v) for k,v in arrays.items()},provenance,config,deadline), output/"mnist.json")
    else:
        from aminist21.data import load_pool, learner_inputs
        from aminist21.suite import freeze_predictions
        if mode == "all":
            original_output = output/"official"
            original_output.mkdir(exist_ok=True)
            arrays = official_inputs()
            original_provenance = {"dataset":"official_mnist_60000_10000","draw":0,"sources":sources,
                                  "input_sha256":{k:array_sha(v) for k,v in arrays.items()}}
            write(original_output/"plan.json",{"config":config,"sources":sources,"provenance":original_provenance,
                  "frozen_at_utc":utc(),"test_labels_supplied":False,"test_scoring":"after all transfer predictions frozen"})
            output = output/"transfer"
            output.mkdir(exist_ok=True)
        plan = {"adapter":"train.py:train_predict","adapter_source_sha256":sources["train.py"],
                "config":config,"sources":sources,"dataset_manifest_sha256":sha(SUITE/"datasets/manifest.json"),
                "datasets":NAMES,"draws":list(range(11)),"seeds":SEEDS,"train_count":10000,
                "query_count":10000,"fresh_weights_per_draw":True,"query_labels_supplied":False,
                "selection":"Five representative datasets selected before training/scoring; partial v1 suite"}
        write(output/"plan.json",plan)
        def jobs():
            if mode == "all":
                yield ({k:pack(v) for k,v in arrays.items()},original_provenance,config,deadline)
            for name in NAMES:
                pool = load_pool(name, SUITE/"data")
                for draw in range(11):
                    transfer_arrays, fit_idx, query_idx = learner_inputs(pool, draw)
                    provenance = {"dataset":name,"draw":draw,"sources":sources,
                        "dataset_seed":SEEDS[draw],"input_sha256":{k:array_sha(v) for k,v in transfer_arrays.items()},
                        "train_indices_sha256":array_sha(fit_idx),"query_indices_sha256":array_sha(query_idx)}
                    yield ({k:pack(v) for k,v in transfer_arrays.items()},provenance,config,deadline)
        for result in fit_remote.starmap(jobs(),order_outputs=False):
            p = result["metadata"]["provenance"]
            if p["dataset"] == "official_mnist_60000_10000":
                save_result(result,original_output/"mnist.json")
            else:
                save_result(result,output/p["dataset"]/f"draw-{p['draw']:02d}.json")
        freeze_predictions(output,NAMES,list(range(11)))
    execution = read(execution_path)
    execution.update(status="complete",finished_at_utc=utc(),elapsed_seconds=time.time()-started)
    write(execution_path,execution)
