"""ctypes interface for the serial-FP32 numerical grid specialization."""
from __future__ import annotations

import ctypes
from pathlib import Path
import subprocess
import numpy as np

HERE = Path(__file__).resolve().parent
LIBRARY = HERE / "grid_pair.so"
CALLBACK = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.POINTER(ctypes.c_float),
                          ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p)


def build(output=LIBRARY):
    command = ["g++", "-std=c++17", "-O3", "-march=native", "-fopenmp",
               "-ffp-contract=off", "-fno-fast-math", "-fno-associative-math",
               "-fPIC", "-shared", str(HERE / "grid_pair.cpp"), "-o", str(output)]
    subprocess.run(command, check=True)
    return command


def run(x, labels, q, weights, biases, *, iterations=300, gamma=.3,
        lam_ridge=.001, lam_k=.01, threads=16, verbose=True,
        snapshot=None, library=LIBRARY):
    """Return int64 labels and FP32 summed standardized scores.

    ``snapshot(name, array)`` runs synchronously for each completed stage. Arrays
    view C++ storage and must be copied if retained after callback return. No
    arrays are copied unless the callback requests it. This keeps full-size
    qualification memory bounded while permitting all-stage parity checks.
    """
    x = np.ascontiguousarray(x, dtype=np.float32).reshape(-1, 81)
    labels = np.ascontiguousarray(labels, dtype=np.int64).reshape(-1)
    q = np.ascontiguousarray(q, dtype=np.float32).reshape(-1, 81)
    weights = np.ascontiguousarray(weights, dtype=np.float32).reshape(-1, 9)
    biases = np.ascontiguousarray(biases, dtype=np.float32).reshape(-1)
    if len(x) != len(labels) or len(weights) != len(biases):
        raise ValueError("inconsistent shapes")
    pred = np.empty(len(q), dtype=np.int64)
    scores = np.empty((len(q), 10), dtype=np.float32)
    lib = ctypes.CDLL(str(library))
    fp = ctypes.POINTER(ctypes.c_float)
    ip = ctypes.POINTER(ctypes.c_int64)
    lib.cg_grid_run.argtypes = [fp, ip, fp, fp, fp, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float,
        ctypes.c_int, ctypes.c_int, ip, fp, CALLBACK, ctypes.c_void_p]
    lib.cg_grid_run.restype = ctypes.c_int
    lib.cg_grid_last_error.restype = ctypes.c_char_p
    errors = []

    def emit(name, pointer, rows, cols, context):
        if snapshot is not None:
            try:
                snapshot(name.decode(), np.ctypeslib.as_array(pointer, shape=(rows * cols,)).reshape(rows, cols))
            except Exception as exc:
                errors.append(exc)

    callback = CALLBACK(emit) if snapshot is not None else CALLBACK()
    status = lib.cg_grid_run(x.ctypes.data_as(fp), labels.ctypes.data_as(ip),
        q.ctypes.data_as(fp), weights.ctypes.data_as(fp), biases.ctypes.data_as(fp),
        len(x), len(q), len(weights), iterations, gamma, lam_ridge, lam_k,
        threads, int(verbose), pred.ctypes.data_as(ip), scores.ctypes.data_as(fp),
        callback, None)
    if status:
        raise RuntimeError(lib.cg_grid_last_error().decode())
    if errors:
        raise errors[0]
    return pred, scores


if __name__ == "__main__":
    print(" ".join(build()))
