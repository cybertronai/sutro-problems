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
