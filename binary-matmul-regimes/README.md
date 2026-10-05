# Native A100 B1 versus INT8 report

This directory contains the standalone report, the matched-semantics A100
benchmark, raw measurements, and reproducible derived tables.

## Result boundary

- GPU actually supplied: NVIDIA A100-SXM4-80GB (SM80, 400 W limit)
- Inputs: identical pseudorandom values in `{0, 1}`
- INT8: signed INT8 multiply-accumulate into S32 via `torch._int_mm`
- B1: packed AND plus population count into S32 via NVIDIA CUTLASS 3.8.0
- Native instruction verified in SASS: `BMMA.168256.AND.POPC`
- Timing: three CUDA-event trials after calibration
- Energy: NVML cumulative GPU-board energy above paired loaded-idle baselines

The prepacked paths include kernel launch, operand reads, computation, and S32
output writes. They exclude allocation, initialization, host transfer, packing,
Modal startup, CPU, cooling, and facility energy. Separate rows measure dynamic
A packing and packing of both operands.

## Files

- `index.html`: published report
- `a100_b1_vs_int8_results.json`: raw benchmark output
- `derived.json`: calculated ratios, rooflines, and Dally calibration
- `results.csv`: machine-readable timing sweep
- `build_report_data.py`: recreates the two derived files from raw JSON
- `modal_b1_int8_sweep.py`: Modal benchmark harness
- `cutlass_b1_wrapper.cu`: pinned CUTLASS SM80 wrapper

Rebuild the derived data with:

```bash
python3 build_report_data.py
```

The raw JSON retained the 1,555 GB/s roofline constant for the requested
A100-40GB. Modal supplied an A100-80GB, so `build_report_data.py` correctly uses
that product's published 2,039 GB/s bandwidth in `derived.json`. The raw timing
and energy measurements are unaffected.

The bandwidth is no longer taken from the raw constant. It is selected from
`hardware.gpu_memory_bytes` by `canonical_hbm_bandwidth_GB_per_s()`, which
returns 2,039 GB/s at or above the 80GB part's capacity and 1,555 GB/s below it,
so a future run on the requested 40GB part derives its own correct roofline.
`derived.json` records both the derived and the raw constants under
`roofline.raw_json_bandwidth_GB_per_s` and `roofline.raw_json_bandwidth_matches_hardware`.
The raw JSON is left unmodified as the measurement record; every per-shape
`*_hbm_floor_s` and `*_roofline_floor_s` it contains was computed at 1,555 GB/s
and is therefore superseded by `derived.json` `per_shape_roofline`, which
recomputes the same byte and operation counts at the derived bandwidth.

## Roofline ceiling versus measurement

`roofline.square_roofline_speedup_ceiling_at_8192` is a theoretical ceiling built
from the dense peak TOPS and the bandwidth constant, capped by the 8x peak
ratio. It is not a measurement. The observed value for the same shape is
`roofline.square_measured_speedup_at_8192` (6.74x). The two are separate keys
precisely so the 8x figure cannot be read as an achieved result.

## Do not compare with `a100-grid-energy-report` at equal weight

`a100-grid-energy-report` reports 2.944 ms and 0.648 J for its B1 stage at
8192^3, while this track reports 0.37083 ms and 0.12269 J for the same nominal
shape. These are two independent records, not two values of one quantity:

- different hardware. That report ran on `NVIDIA A100-SXM4-40GB`
  (`gpu_memory_bytes` 42,405,855,232, HBM2); this one ran on
  `NVIDIA A100-SXM4-80GB` (85,094,825,984, HBM2e).
- different energy protocol. That report used five 5 s trials with a 5 s idle
  window; this one uses three 3 s trials with a 3 s idle window.
- different B1 kernel. That report's stage is recorded as "native SM80 WMMA
  BMMA"; this one uses CUTLASS 3.8.0 `OpAndPopc` on `mma.sync m16n8k256`,
  with the per-shape fragment and threadblock shape chosen per form.

Neither number is corrected here and neither is preferred. Reconciling them
requires a new A100 run, not an edit.

## Weight-residency caveat

The eight-rotating-weights panel reports 5.45x less energy. Its original
explanation ("neither bank fits L2, so both paths are bandwidth-bound") is not
supported by the data and is left uncorrected rather than rewritten.
`derived.json` `cache.hbm_floor_consistency` now emits each weight-residency
call's HBM byte floor beside its measured time: `int8_rotating8_weights`
measures 0.0572 ms against a 0.2646 ms floor for the 512 MiB bank, i.e. 4.6x
faster than streaming that bank from HBM can allow, and essentially equal to
its own single-weight hot time. Whatever produced that number, it is not an
HBM-streaming measurement, so the bandwidth-bound reading does not follow.
