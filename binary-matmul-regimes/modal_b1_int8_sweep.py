#!/usr/bin/env python3
"""Sweep native A100 B1 AND+POPC GEMM against cuBLAS INT8 GEMM.

The comparison is functionally exact for {0,1} inputs:

    sum_k int8(A[m,k]) * int8(B[k,n])
      == popcount(A_bits[m,:] AND B_bits[:,n])

Inputs and outputs are allocated on the GPU before every measured window.  The
main B1 path assumes packed resident operands; separate paths include packing a
dynamic left operand or both operands.  Energy is GPU-board energy above a
paired loaded-idle baseline, read from NVML's cumulative energy counter.
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
CUTLASS_COMMIT = "afa1772203677c5118fcd82537a9c8fefbcc7008"  # v3.8.0
CUTLASS_WRAPPER_LOCAL = HERE / "cutlass_b1_wrapper.cu"
CUTLASS_WRAPPER_REMOTE = Path("/opt/b1-benchmark/cutlass_b1_wrapper.cu")
# Modal re-imports this module from /root in a remote container. At local image
# construction time use the repository file; in the built container the copied
# remote path already exists.
CUTLASS_WRAPPER_IMAGE_SOURCE = (
    CUTLASS_WRAPPER_LOCAL
    if CUTLASS_WRAPPER_LOCAL.exists()
    else CUTLASS_WRAPPER_REMOTE
)
IMAGE_REF = "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel"

image = (
    modal.Image.from_registry(IMAGE_REF)
    .apt_install("build-essential", "curl")
    .pip_install(
        "cupy-cuda12x==13.6.0",
        "nvidia-ml-py==13.580.82",
    )
    .run_commands(
        "mkdir -p /opt/cutlass && "
        f"curl -fsSL https://github.com/NVIDIA/cutlass/archive/{CUTLASS_COMMIT}.tar.gz "
        "| tar -xz --strip-components=1 -C /opt/cutlass"
    )
    .env({"LD_LIBRARY_PATH": "/usr/local/cuda/lib64"})
    .workdir("/workspace")
    .add_local_file(
        str(CUTLASS_WRAPPER_IMAGE_SOURCE),
        remote_path=str(CUTLASS_WRAPPER_REMOTE),
        copy=True,
    )
)
app = modal.App("a100-b1-vs-int8-regime-sweep")

PACK_CUDA = r"""
extern "C" __global__ void pack_rows_u8_to_b1(
    const signed char *__restrict__ src,
    unsigned *__restrict__ dst,
    long long logical_elements) {
  long long bit = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  unsigned active = __ballot_sync(0xffffffffu, bit < logical_elements);
  unsigned value = 0;
  if (bit < logical_elements) {
    value = static_cast<unsigned>(src[bit] & 1);
  }
  unsigned packed = __ballot_sync(active, value != 0);
  if ((threadIdx.x & 31) == 0 && bit < logical_elements) {
    dst[bit >> 5] = packed;
  }
}
"""


def _shape_plan(quick: bool) -> list[dict]:
    """Aligned shapes chosen to expose scale, batch, K, and output regimes."""
    if quick:
        return [
            {"family": "square", "m": 256, "n": 256, "k": 256},
            {"family": "square", "m": 2048, "n": 2048, "k": 2048},
            {"family": "square", "m": 8192, "n": 8192, "k": 8192},
            {"family": "batch", "m": 64, "n": 8192, "k": 8192},
            {"family": "output_bound", "m": 8192, "n": 8192, "k": 256},
        ]

    plan: list[dict] = []
    for size in (256, 512, 1024, 2048, 4096, 8192, 16384):
        plan.append({"family": "square", "m": size, "n": size, "k": size})
    for batch in (32, 64, 128, 256, 512, 1024, 2048):
        plan.append({"family": "batch", "m": batch, "n": 8192, "k": 8192})
    for inner in (256, 512, 1024, 2048, 4096, 8192, 16384, 32768):
        plan.append({"family": "k_sweep", "m": 4096, "n": 4096, "k": inner})
    plan.append({"family": "output_bound", "m": 8192, "n": 8192, "k": 256})
    return plan


@app.function(
    image=image,
    gpu="A100-40GB",
    min_containers=0,
    max_containers=1,
    buffer_containers=0,
    scaledown_window=2,
    single_use_containers=True,
    retries=0,
    timeout=900,
    startup_timeout=240,
)
def run_benchmark(
    *,
    quick: bool = False,
    measure_energy: bool = True,
    timing_target_s: float = 0.08,
    energy_trial_s: float = 3.0,
    energy_trials: int = 3,
    idle_s: float = 3.0,
) -> dict:
    import ctypes
    import glob
    import math
    import os
    import shutil
    import statistics
    import subprocess
    import tempfile
    import time
    from datetime import datetime, timezone

    import cupy as cp
    import pynvml
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if torch.cuda.get_device_capability(0) != (8, 0):
        raise RuntimeError(
            f"SM80 A100 required, got {torch.cuda.get_device_capability(0)}"
        )
    if not hasattr(torch, "_int_mm"):
        raise RuntimeError("this PyTorch build does not expose torch._int_mm")

    torch.manual_seed(20260901)
    torch.cuda.manual_seed_all(20260901)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    device_properties = torch.cuda.get_device_properties(0)
    device_memory_bytes = int(device_properties.total_memory)
    # Modal may satisfy an A100-40GB request with an 80GB SXM4.  The two
    # products have different HBM2(e) bandwidths, so select the matching
    # published roofline constant from the hardware that actually arrived.
    hbm_bandwidth_gb_s = 2039 if device_memory_bytes >= 60 * 1024**3 else 1555
    stream = torch.cuda.current_stream(device)
    stream_ptr = int(stream.cuda_stream)

    # Compile the pinned, official CUTLASS implementation to a tiny shared object.
    build_dir = Path(tempfile.mkdtemp(prefix="cutlass-b1-build-"))
    source_path = build_dir / "cutlass_b1_wrapper.cu"
    library_path = build_dir / "libcutlass_b1.so"
    shutil.copyfile(CUTLASS_WRAPPER_REMOTE, source_path)

    nvcc_candidates = [
        shutil.which("nvcc"),
        *glob.glob(
            "/usr/local/lib/python*/site-packages/nvidia/cuda_nvcc/bin/nvcc"
        ),
    ]
    nvcc = next((item for item in nvcc_candidates if item), None)
    if not nvcc:
        raise RuntimeError(f"nvcc not found; candidates={nvcc_candidates}")
    cuda_include_dirs = sorted(
        set(
            glob.glob(
                "/usr/local/lib/python*/site-packages/nvidia/cuda_runtime/include"
            )
            + glob.glob(
                "/usr/local/lib/python*/site-packages/nvidia/cuda_nvcc/include"
            )
        )
    )
    cuda_library_dirs = sorted(
        set(
            glob.glob(
                "/usr/local/lib/python*/site-packages/nvidia/cuda_runtime/lib"
            )
        )
    )
    command = [
        nvcc,
        "-std=c++17",
        "-O3",
        "--shared",
        "-Xcompiler=-fPIC",
        "-lineinfo",
        "-gencode=arch=compute_80,code=sm_80",
        "-I/opt/cutlass/include",
        str(source_path),
        "-o",
        str(library_path),
    ]
    for path in cuda_include_dirs:
        command.insert(-3, f"-I{path}")
    for path in cuda_library_dirs:
        command.insert(-3, f"-L{path}")
    compilation = subprocess.run(command, capture_output=True, text=True)
    if compilation.returncode != 0:
        raise RuntimeError(
            "CUTLASS wrapper compilation failed\n"
            f"command={' '.join(command)}\n"
            f"stdout={compilation.stdout}\n"
            f"stderr={compilation.stderr}"
        )

    library = ctypes.CDLL(str(library_path))
    variant_names = (
        "128x256_k1024",
        "256x128_k1024",
        "128x128_k1024",
        "64x128_k1024",
        "128x64_k1024",
        "64x64_k1024",
        "128x256_k512",
        "256x128_k512",
        "64x256_k512",
        "256x64_k512",
        "128x128_k512",
        "64x128_k512",
        "128x64_k512",
        "64x64_k512",
    )
    variants = {}
    for name in variant_names:
        function = getattr(library, f"cutlass_b1_{name}")
        function.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        function.restype = ctypes.c_int
        variants[name] = function
    library.cutlass_b1_build_description.restype = ctypes.c_char_p
    build_description = library.cutlass_b1_build_description().decode()

    pack_kernel = cp.RawKernel(PACK_CUDA, "pack_rows_u8_to_b1")

    def pointer(array) -> ctypes.c_void_p:
        return ctypes.c_void_p(int(array.data.ptr))

    def pack(
        source_cp: cp.ndarray,
        destination: cp.ndarray,
        logical_elements: int,
    ) -> None:
        threads = 256
        blocks = (logical_elements + threads - 1) // threads
        with cp.cuda.ExternalStream(stream_ptr):
            pack_kernel(
                (blocks,),
                (threads,),
                (source_cp, destination, logical_elements),
            )

    def launch(
        function,
        a: cp.ndarray,
        b: cp.ndarray,
        d: cp.ndarray,
        m: int,
        n: int,
        k: int,
    ) -> None:
        status = function(
            pointer(a),
            pointer(b),
            pointer(d),
            m,
            n,
            k,
            ctypes.c_void_p(stream_ptr),
        )
        if status != 0:
            raise RuntimeError(f"CUTLASS launch failed with status {status}")

    def event_duration(runner, repetitions: int) -> float:
        torch.cuda.synchronize(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(stream)
        runner(repetitions)
        end.record(stream)
        end.synchronize()
        return start.elapsed_time(end) / 1000.0

    def calibrated_timing(runner, *, target_s: float = timing_target_s) -> dict:
        repetitions = 1
        duration = event_duration(runner, repetitions)
        while duration < min(0.02, target_s / 2) and repetitions < 131_072:
            repetitions *= 2
            duration = event_duration(runner, repetitions)
        projected = max(
            2,
            min(131_072, math.ceil(target_s * repetitions / max(duration, 1e-9))),
        )
        samples = [event_duration(runner, projected) / projected for _ in range(3)]
        return {
            "repetitions_per_trial": projected,
            "trials": samples,
            "seconds_per_call_median": statistics.median(samples),
            "seconds_per_call_mean": statistics.mean(samples),
            "seconds_per_call_stdev": statistics.stdev(samples),
        }

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)

    def energy_millijoules() -> int:
        return int(pynvml.nvmlDeviceGetTotalEnergyConsumption(handle))

    def loaded_idle_window() -> dict:
        torch.cuda.synchronize(device)
        e0 = energy_millijoules()
        t0 = time.monotonic()
        time.sleep(idle_s)
        duration = time.monotonic() - t0
        e1 = energy_millijoules()
        raw = (e1 - e0) / 1000.0
        return {
            "duration_s": duration,
            "raw_gpu_energy_J": raw,
            "average_gpu_power_W": raw / duration,
        }

    energy_results: list[dict] = []

    def measure_energy_stage(
        label: str,
        shape: dict,
        runner,
        seconds_per_call: float,
        *,
        regime: str = "ordinary",
    ) -> None:
        if not measure_energy:
            return
        repetitions = max(2, math.ceil(energy_trial_s / seconds_per_call))
        # A hard cap bounds launch-list construction for extremely small kernels.
        repetitions = min(repetitions, 524_288)
        runner(min(repetitions, 8))
        torch.cuda.synchronize(device)
        time.sleep(2.0)
        idle_before = loaded_idle_window()
        trials = []
        for trial in range(energy_trials):
            torch.cuda.synchronize(device)
            e0 = energy_millijoules()
            t0 = time.monotonic()
            runner(repetitions)
            torch.cuda.synchronize(device)
            duration = time.monotonic() - t0
            e1 = energy_millijoules()
            trials.append(
                {
                    "trial": trial + 1,
                    "repetitions": repetitions,
                    "duration_s": duration,
                    "raw_gpu_energy_J": (e1 - e0) / 1000.0,
                }
            )
        time.sleep(2.0)
        idle_after = loaded_idle_window()
        idle_watts = statistics.mean(
            [
                idle_before["average_gpu_power_W"],
                idle_after["average_gpu_power_W"],
            ]
        )
        idle_range = sorted(
            [
                idle_before["average_gpu_power_W"],
                idle_after["average_gpu_power_W"],
            ]
        )
        total_repetitions = sum(item["repetitions"] for item in trials)
        total_duration = sum(item["duration_s"] for item in trials)
        total_raw = sum(item["raw_gpu_energy_J"] for item in trials)
        total_net = max(0.0, total_raw - idle_watts * total_duration)
        low_net = max(0.0, total_raw - idle_range[1] * total_duration)
        high_net = max(0.0, total_raw - idle_range[0] * total_duration)
        record = {
            "label": label,
            "regime": regime,
            "shape": dict(shape),
            "repetitions": total_repetitions,
            "seconds_per_call_wall": total_duration / total_repetitions,
            "raw_gpu_energy_J_per_call": total_raw / total_repetitions,
            "idle_adjusted_gpu_energy_J_per_call": total_net / total_repetitions,
            "idle_adjusted_gpu_energy_J_per_call_baseline_range": [
                low_net / total_repetitions,
                high_net / total_repetitions,
            ],
            "idle_baseline_W": idle_watts,
            "idle_baseline_range_W": idle_range,
            "idle_before": idle_before,
            "idle_after": idle_after,
            "trials": trials,
        }
        energy_results.append(record)
        print(
            f"[energy] {label} {shape['m']}x{shape['n']}x{shape['k']} "
            f"{1e3 * record['seconds_per_call_wall']:.4f} ms, "
            f"{record['idle_adjusted_gpu_energy_J_per_call']:.6f} J",
            flush=True,
        )

    def theoretical(shape: dict) -> dict:
        m, n, k = shape["m"], shape["n"], shape["k"]
        pair_contributions = m * n * k
        int8_bytes = m * k + k * n + 4 * m * n
        b1_bytes = (m * k + k * n) // 8 + 4 * m * n
        bandwidth = hbm_bandwidth_gb_s * 1e9
        int8_peak = 624e12
        b1_peak = 4_992e12
        operations = 2 * pair_contributions
        return {
            "pair_contributions": pair_contributions,
            "nvidia_operation_count": operations,
            "minimum_bytes_int8_s32": int8_bytes,
            "minimum_bytes_b1_s32": b1_bytes,
            "minimum_byte_ratio_int8_over_b1": int8_bytes / b1_bytes,
            "int8_compute_floor_s": operations / int8_peak,
            "int8_hbm_floor_s": int8_bytes / bandwidth,
            "int8_roofline_floor_s": max(
                operations / int8_peak, int8_bytes / bandwidth
            ),
            "b1_compute_floor_s": operations / b1_peak,
            "b1_hbm_floor_s": b1_bytes / bandwidth,
            "b1_roofline_floor_s": max(
                operations / b1_peak, b1_bytes / bandwidth
            ),
        }

    timing_results = []
    energy_shapes = {
        (2048, 2048, 2048),
        (8192, 8192, 8192),
        (8192, 8192, 256),
    }

    for shape in _shape_plan(quick):
        m, n, k = shape["m"], shape["n"], shape["k"]
        if m % 16 or n % 8 or k % 256:
            raise AssertionError(f"unaligned benchmark shape: {shape}")

        a8 = torch.randint(0, 2, (m, k), device=device, dtype=torch.int8)
        b8_storage = torch.randint(0, 2, (n, k), device=device, dtype=torch.int8)
        b8 = b8_storage.t()
        a8_cp = cp.from_dlpack(a8)
        b8_storage_cp = cp.from_dlpack(b8_storage)
        c8 = torch.empty((m, n), device=device, dtype=torch.int32)
        a1 = cp.empty((m, k // 32), dtype=cp.uint32)
        b1_col = cp.empty((n, k // 32), dtype=cp.uint32)
        c1 = cp.empty((m, n), dtype=cp.int32, order="F")
        pack(a8_cp, a1, m * k)
        pack(b8_storage_cp, b1_col, n * k)
        torch.cuda.synchronize(device)

        def int8_runner(repetitions: int) -> None:
            for _ in range(repetitions):
                torch._int_mm(a8, b8, out=c8)

        candidate_timings = {}
        candidate_errors = {}
        for name, function in variants.items():
            def candidate_runner(repetitions: int, fn=function) -> None:
                for _ in range(repetitions):
                    launch(fn, a1, b1_col, c1, m, n, k)

            try:
                candidate_runner(1)
                candidate_timings[name] = event_duration(candidate_runner, 3) / 3
            except RuntimeError as exc:
                candidate_errors[name] = str(exc)
        if not candidate_timings:
            raise RuntimeError(f"no CUTLASS variant accepted shape {shape}: {candidate_errors}")
        best_variant = min(candidate_timings, key=candidate_timings.get)
        best_function = variants[best_variant]

        def b1_runner(repetitions: int) -> None:
            for _ in range(repetitions):
                launch(best_function, a1, b1_col, c1, m, n, k)

        def pack_a_runner(repetitions: int) -> None:
            for _ in range(repetitions):
                pack(a8_cp, a1, m * k)

        def pack_b_runner(repetitions: int) -> None:
            for _ in range(repetitions):
                pack(b8_storage_cp, b1_col, n * k)

        def b1_pack_a_runner(repetitions: int) -> None:
            for _ in range(repetitions):
                pack(a8_cp, a1, m * k)
                launch(best_function, a1, b1_col, c1, m, n, k)

        def b1_pack_both_runner(repetitions: int) -> None:
            for _ in range(repetitions):
                pack(a8_cp, a1, m * k)
                pack(b8_storage_cp, b1_col, n * k)
                launch(best_function, a1, b1_col, c1, m, n, k)

        # Initialize all paths and validate identical 0/1 integer semantics.
        int8_runner(1)
        b1_runner(1)
        torch.cuda.synchronize(device)
        sample_points = sorted(
            set(
                [
                    (0, 0),
                    (min(17, m - 1), min(31, n - 1)),
                    (m - 1, n - 1),
                ]
            )
        )
        validation = []
        for row, col in sample_points:
            observed_int8 = int(c8[row, col].item())
            observed_b1 = int(c1[row, col].item())
            validation.append(
                {
                    "row": row,
                    "col": col,
                    "int8": observed_int8,
                    "b1": observed_b1,
                    "equal": observed_int8 == observed_b1,
                }
            )
        if not all(item["equal"] for item in validation):
            raise RuntimeError(f"B1/INT8 validation failed for {shape}: {validation}")

        timings = {
            "int8": calibrated_timing(int8_runner),
            "b1_prepacked": calibrated_timing(b1_runner),
            "pack_dynamic_a_only": calibrated_timing(pack_a_runner),
            "pack_static_b_only": calibrated_timing(pack_b_runner),
            "b1_including_dynamic_a_pack": calibrated_timing(b1_pack_a_runner),
            "b1_including_both_packs": calibrated_timing(b1_pack_both_runner),
        }
        int8_s = timings["int8"]["seconds_per_call_median"]
        b1_s = timings["b1_prepacked"]["seconds_per_call_median"]
        record = {
            **shape,
            "best_cutlass_variant": best_variant,
            "cutlass_variant_probe_seconds": candidate_timings,
            "cutlass_variant_probe_errors": candidate_errors,
            "validation": validation,
            "timings": timings,
            "speedup_b1_prepacked_over_int8": int8_s / b1_s,
            "speedup_b1_pack_a_over_int8": int8_s
            / timings["b1_including_dynamic_a_pack"]["seconds_per_call_median"],
            "speedup_b1_pack_both_over_int8": int8_s
            / timings["b1_including_both_packs"]["seconds_per_call_median"],
            "effective_int8_TOPS": 2 * m * n * k / int8_s / 1e12,
            "effective_b1_TOPS": 2 * m * n * k / b1_s / 1e12,
            "resident_bytes": {
                "int8_operands_s32_output": m * k + k * n + 4 * m * n,
                "b1_operands_s32_output": (m * k + k * n) // 8 + 4 * m * n,
            },
            "theory": theoretical(shape),
        }
        timing_results.append(record)
        print(
            f"[timing] {shape['family']:<12} {m:>5}x{n:<5}x{k:<5} "
            f"INT8={1e3 * int8_s:>8.4f} ms "
            f"B1={1e3 * b1_s:>8.4f} ms "
            f"speedup={int8_s / b1_s:>5.2f}x ({best_variant})",
            flush=True,
        )

        if (m, n, k) in energy_shapes:
            measure_energy_stage(
                "int8",
                shape,
                int8_runner,
                timings["int8"]["seconds_per_call_median"],
            )
            measure_energy_stage(
                "b1_prepacked",
                shape,
                b1_runner,
                timings["b1_prepacked"]["seconds_per_call_median"],
            )
            measure_energy_stage(
                "b1_including_dynamic_a_pack",
                shape,
                b1_pack_a_runner,
                timings["b1_including_dynamic_a_pack"][
                    "seconds_per_call_median"
                ],
            )

        del a8, b8_storage, b8, a8_cp, b8_storage_cp, c8, a1, b1_col, c1
        torch.cuda.empty_cache()
        cp.get_default_memory_pool().free_all_blocks()

    # Cache-capacity experiment: one static 8 MiB packed weight fits in the
    # A100's 40 MiB L2; eight rotating packed weights (64 MiB total) do not.
    cache_shape = {"family": "weight_residency", "m": 64, "n": 8192, "k": 8192}
    m, n, k = cache_shape["m"], cache_shape["n"], cache_shape["k"]
    a8 = torch.randint(0, 2, (m, k), device=device, dtype=torch.int8)
    a8_cp = cp.from_dlpack(a8)
    a1 = cp.empty((m, k // 32), dtype=cp.uint32)
    pack(a8_cp, a1, m * k)
    c8 = torch.empty((m, n), device=device, dtype=torch.int32)
    c1 = cp.empty((m, n), dtype=cp.int32, order="F")
    b8_storages = [
        torch.randint(0, 2, (n, k), device=device, dtype=torch.int8)
        for _ in range(8)
    ]
    b8_views = [item.t() for item in b8_storages]
    b8_storage_cp = [cp.from_dlpack(item) for item in b8_storages]
    b1_banks = [cp.empty((n, k // 32), dtype=cp.uint32) for _ in range(8)]
    for source_cp, destination in zip(b8_storage_cp, b1_banks):
        pack(source_cp, destination, n * k)
    torch.cuda.synchronize(device)

    cache_candidate_timings = {}
    cache_candidate_errors = {}
    for name, function in variants.items():
        def cache_candidate(repetitions: int, fn=function) -> None:
            for _ in range(repetitions):
                launch(fn, a1, b1_banks[0], c1, m, n, k)

        try:
            cache_candidate_timings[name] = event_duration(cache_candidate, 3) / 3
        except RuntimeError as exc:
            cache_candidate_errors[name] = str(exc)
    if not cache_candidate_timings:
        raise RuntimeError(f"no CUTLASS variant accepted cache shape: {cache_candidate_errors}")
    cache_variant = min(cache_candidate_timings, key=cache_candidate_timings.get)
    cache_function = variants[cache_variant]

    def int8_hot(repetitions: int) -> None:
        for _ in range(repetitions):
            torch._int_mm(a8, b8_views[0], out=c8)

    def b1_hot(repetitions: int) -> None:
        for _ in range(repetitions):
            launch(cache_function, a1, b1_banks[0], c1, m, n, k)

    def int8_rotating8(repetitions: int) -> None:
        for index in range(repetitions):
            torch._int_mm(a8, b8_views[index & 7], out=c8)

    def b1_rotating8(repetitions: int) -> None:
        for index in range(repetitions):
            launch(cache_function, a1, b1_banks[index & 7], c1, m, n, k)

    cache_timings = {
        "int8_hot_static_weight": calibrated_timing(int8_hot),
        "b1_hot_static_weight": calibrated_timing(b1_hot),
        "int8_rotating8_weights": calibrated_timing(int8_rotating8),
        "b1_rotating8_weights": calibrated_timing(b1_rotating8),
    }
    cache_result = {
        **cache_shape,
        "best_cutlass_variant": cache_variant,
        "cutlass_variant_probe_seconds": cache_candidate_timings,
        "cutlass_variant_probe_errors": cache_candidate_errors,
        "timings": cache_timings,
        "single_weight_bytes": {"int8": n * k, "b1": n * k // 8},
        "eight_weight_bank_bytes": {"int8": 8 * n * k, "b1": n * k},
        "a100_l2_bytes": 40 * 1024 * 1024,
        "hot_speedup": cache_timings["int8_hot_static_weight"]
        ["seconds_per_call_median"]
        / cache_timings["b1_hot_static_weight"]["seconds_per_call_median"],
        "rotating8_speedup": cache_timings["int8_rotating8_weights"]
        ["seconds_per_call_median"]
        / cache_timings["b1_rotating8_weights"]["seconds_per_call_median"],
    }
    print(
        "[cache] 64x8192x8192 "
        f"hot={cache_result['hot_speedup']:.2f}x "
        f"rotating8={cache_result['rotating8_speedup']:.2f}x",
        flush=True,
    )
    if measure_energy:
        for label, runner in (
            ("int8_hot_static_weight", int8_hot),
            ("b1_hot_static_weight", b1_hot),
            ("int8_rotating8_weights", int8_rotating8),
            ("b1_rotating8_weights", b1_rotating8),
        ):
            measure_energy_stage(
                label,
                cache_shape,
                runner,
                cache_timings[label]["seconds_per_call_median"],
                regime="weight_residency",
            )

    # Inspect the linked cubin as an independent implementation audit.
    tool_candidates = [
        shutil.which("cuobjdump"),
        *glob.glob(
            "/usr/local/lib/python*/site-packages/nvidia/cuda_nvcc/bin/cuobjdump"
        ),
    ]
    cuobjdump = next((item for item in tool_candidates if item), None)
    sass_audit = {"cuobjdump": cuobjdump, "bmma_lines": [], "error": ""}
    if cuobjdump:
        audit = subprocess.run(
            [cuobjdump, "--dump-sass", str(library_path)],
            capture_output=True,
            text=True,
        )
        sass_audit["returncode"] = audit.returncode
        sass_audit["error"] = audit.stderr.strip()
        sass_audit["bmma_lines"] = [
            line.strip()
            for line in audit.stdout.splitlines()
            if "BMMA" in line.upper()
        ][:80]

    hardware_query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version,power.limit,memory.total,"
            "compute_cap",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
    )
    gpu_name = pynvml.nvmlDeviceGetName(handle)
    if isinstance(gpu_name, bytes):
        gpu_name = gpu_name.decode()
    result = {
        "benchmark": "native A100 B1 AND+POPC versus INT8 GEMM regime sweep",
        "date_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "semantics": {
            "input_domain": "identical pseudo-random values in {0,1}",
            "int8": "signed INT8 multiply-accumulate into S32",
            "b1": "packed B1 AND then POPC into S32",
            "equivalence": "for 0/1 values, a*b equals a AND b exactly",
            "output_layout": {
                "int8": "row-major",
                "b1": "column-major",
            },
        },
        "measurement": {
            "timing": "CUDA events around synchronized aggregate launch windows",
            "timing_target_s": timing_target_s,
            "timing_trials": 3,
            "energy_enabled": measure_energy,
            "energy_counter": "nvmlDeviceGetTotalEnergyConsumption",
            "energy_quantity": "GPU board energy above paired loaded-idle baseline",
            "energy_trial_target_s": energy_trial_s,
            "energy_trials": energy_trials,
            "idle_window_s": idle_s,
            "included_in_prepacked_paths": [
                "kernel launches",
                "input reads",
                "tensor-core computation",
                "S32 output writes",
            ],
            "excluded_from_prepacked_paths": [
                "allocation and initialization",
                "host-device transfer",
                "bit packing and layout conversion",
                "Modal startup",
                "CPU, cooling, and facility energy",
            ],
        },
        "implementation": {
            "b1_library": "NVIDIA CUTLASS 3.8.0",
            "cutlass_commit": CUTLASS_COMMIT,
            "build_description": build_description,
            "instruction_shape": "m16n8k256",
            "operation": "AND+POPC",
            "candidate_threadblock_shapes": list(variant_names),
            "wrapper_compile_command": command,
            "wrapper_compiler_stdout": compilation.stdout.strip(),
            "wrapper_compiler_stderr": compilation.stderr.strip(),
            "sass_audit": sass_audit,
        },
        "hardware": {
            "gpu": gpu_name,
            "requested_modal_gpu": "A100-40GB",
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "gpu_memory_bytes": device_memory_bytes,
            "power_limit_W": pynvml.nvmlDeviceGetPowerManagementLimit(handle)
            / 1000.0,
            "driver": pynvml.nvmlSystemGetDriverVersion(),
            "torch": str(torch.__version__),
            "torch_cuda": str(torch.version.cuda),
            "cupy": str(cp.__version__),
            "image_ref": IMAGE_REF,
            "nvidia_smi_query": hardware_query.stdout.strip(),
        },
        "theory_constants": {
            "a100_dense_int8_peak_TOPS": 624,
            "a100_dense_b1_peak_TOPS": 4992,
            "a100_hbm_bandwidth_GB_per_s": hbm_bandwidth_gb_s,
            "a100_l2_MiB": 40,
            "operation_count_convention": "2*M*N*K for both INT8 and B1",
        },
        "timing_results": timing_results,
        "weight_residency_result": cache_result,
        "energy_results": energy_results,
        "quick": quick,
    }
    pynvml.nvmlShutdown()
    return result


def print_summary(result: dict) -> None:
    print("\nA100 native B1 AND+POPC vs INT8")
    print(f"GPU: {result['hardware']['gpu']}")
    print("family          M      N      K     INT8 ms     B1 ms   B1 speedup")
    for row in result["timing_results"]:
        int8_ms = 1e3 * row["timings"]["int8"]["seconds_per_call_median"]
        b1_ms = 1e3 * row["timings"]["b1_prepacked"]["seconds_per_call_median"]
        print(
            f"{row['family']:<12} {row['m']:>6} {row['n']:>6} {row['k']:>6} "
            f"{int8_ms:>11.4f} {b1_ms:>9.4f} "
            f"{row['speedup_b1_prepacked_over_int8']:>10.2f}x"
        )
    cache = result["weight_residency_result"]
    print(
        f"\n64x8192x8192 static-weight speedup: {cache['hot_speedup']:.2f}x; "
        f"rotating-eight speedup: {cache['rotating8_speedup']:.2f}x"
    )


@app.local_entrypoint()
def main(
    output: str = str(HERE / "a100_b1_vs_int8_results.json"),
    quick: bool = False,
    energy: bool = True,
    timing_target_s: float = 0.08,
    energy_trial_s: float = 3.0,
    energy_trials: int = 3,
    idle_s: float = 3.0,
) -> None:
    destination = Path(output).expanduser().resolve()
    result = run_benchmark.remote(
        quick=quick,
        measure_energy=energy,
        timing_target_s=timing_target_s,
        energy_trial_s=energy_trial_s,
        energy_trials=energy_trials,
        idle_s=idle_s,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print_summary(result)
    print(f"\nwrote {destination}")
