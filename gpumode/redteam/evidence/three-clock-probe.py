import os, sys, time
os.environ["MNIST_EVAL_DEVICE"] = "cpu"
sys.path.insert(0, "/tmp/rtprobe")
import numpy as np, eval as E

def main():
    cache = "/Users/yaroslavvb/git/sutro-problems/output/gpumode-cache/raw"
    case = dict(E.DEFAULTS); case["seed"] = 202; case["spec"] = "probe"
    pools = E.load_pools([case], cache, need_holdout=False)
    pool = pools[("mnist", 9)]
    for name in sys.argv[1:]:
        import shutil
        shutil.copy(name, "/tmp/rtprobe/submission.py")
        os.chdir("/tmp/rtprobe")
        child = E.Child("cpu")
        warm, _ = E.make_draw(pool, case["seed"] + E.WARMUP_OFFSET, 10000, 10000)
        child.call("untimed", warm)
        for i in (1, 2):
            vis, truth = E.make_draw(pool, case["seed"] + 13 * i, 10000, 10000)
            (pred, dev_ms, child_ms), parent_ms = child.call("timed", vis)
            print(f"{os.path.basename(name)} call{i}: device={dev_ms:.4f} ms  child_wall={child_ms:.3f} ms  "
                  f"parent_wall={parent_ms:.3f} ms  correct={(pred==truth).sum()}  "
                  f"gate={E.timing_plausible(dev_ms, child_ms, parent_ms)}")
        child.stop()

if __name__ == "__main__":
    main()
