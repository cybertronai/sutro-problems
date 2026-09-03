# Dally-Eval Execution Engines: Python vs. Rust CPU vs. Rust GPU (LDS)

This document formalizes the three execution engines of the **Bill Dally / Sutro Evaluator** and provides the architecture mapping between hardware physics, benchmark challenges, and search engines.

---

## 1. The Three Execution Engines

The Sutro benchmark suite supports three distinct execution implementations:

1. **Python Reference Engine (`mask_sparse_parity.py`)**:
   - The authoritative ground-truth specification.
   - Vectorized via NumPy, but bounded by Python interpreter overhead.
   - Evaluates ~300 to 15,000 instances/sec.

2. **Rust CPU Rayon Engine (`dally_eval::CpuRunner`)**:
   - High-throughput multi-threaded CPU execution.
   - Bit-exact with Python reference.
   - Measured speedup: **61× to 72× over Python** (up to 935,000 instances/sec / 15.5 Billion ops/sec).

3. **Rust GPU LDS Engine (`dally_eval::LdsRunner`)**:
   - Hardware-accelerated execution on AMD Radeon RX 6900 XT.
   - **LDS Shared Memory**: Memory cells live in on-chip Local Data Share SRAM (64 KB/CU), eliminating VRAM roundtrips.
   - Transient VRAM footprint: only ~120–150 MiB.
   - Executes massive parallel batches across GPU compute units.

---

## 2. Benchmark Comparison Table (Across All 3 Engines)

*Measured on AMD Ryzen 9 9950X CPU & AMD Radeon RX 6900 XT GPU (16 GB):*

| Benchmark Program | Ops / Instance | Python Reference | Rust CPU (Rayon) | Rust GPU (LDS Runner) | GPU vs. CPU Speedup | Net Speedup vs. Python |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **20% packedsis** | 15,390 | 15.2k inst/s | 1,093k inst/s | 658k inst/s | 0.60x | 43x |
| **40% packedwalk** | 17,815 | 13.7k inst/s | 944k inst/s | 629k inst/s | 0.67x | 46x |
| **60% packedsis** | 30,555 | 7.0k inst/s | 538k inst/s | 485k inst/s | 0.90x | 69x |
| **80% weightscan** | 266,185 | 937 inst/s | 66k inst/s | 25k inst/s | 0.38x | 27x |
| **100% weightscan**| 709,513 | 313 inst/s | 23k inst/s | 10k inst/s | 0.43x | 31x |

*GPU column: measured at 100k-instance batches (the GPU's best case, amortizing dispatch overhead). CPU column: same batch size. All engines bit-exact on the golden fixtures; all runs under `training.slice` on the RX 6900 XT. The CPU engine leads at every program size on this host — the GPU path's value is its 768 MiB-bounded multi-tenant envelope and headroom on machines where CPU cores are contended.*

*Additional measurement (sparse-parity siswalk program, 73,293 ops, the search-workload shape): CPU 63-86k inst/s vs GPU LDS 160-165k inst/s — here the GPU WINS ~2x, because the program's cells fit entirely in LDS while its op stream is mid-sized. The crossover is program-shape-dependent: short-cell programs favor LDS, long-op programs favor CPU.*

---

## 3. The Tripartite Architecture Matrix

```text
┌────────────────────────────────┬────────────────────────────────┬────────────────────────────────┐
│   BILL DALLY HARDWARE MODEL    │   SUTRO BENCHMARK CHALLENGE    │   SYSTEM-1 MCTS SEARCH ENGINE  │
├────────────────────────────────┼────────────────────────────────┼────────────────────────────────┤
│ Physical 2D Silicon Geometry:  │ Mathematical Matrix Product:   │ Semantic Value Action Space:   │
│ - Processor at origin (0, 0)   │ - Compute C = A * B (4x4, 16x) │ - DAG tokens: InputA, InputB,  │
│ - Memory cells on 2D half-plane│ - 64 multiplications           │   Product, Accumulator, Output │
│ - Distance = ceil(sqrt(addr))  │ - 48 additions                 │ - MCTS explores valid DAG      │
│   (isqrt(addr - 1) + 1)        │ - 16 output declarations       │   topological orderings        │
├────────────────────────────────┼────────────────────────────────┼────────────────────────────────┤
│ Wire-Length Energy Physics:    │ The Competition Frontiers:     │ Liveness Interval Analysis:    │
│ - Operand read cost: sqrt(addr)│ - Naive baseline: 1,309 ops    │ - Derives [t_birth, t_death]   │
│ - Writes & arithmetic: 0 cost  │ - Human record (Juraj): 681 ops│   for every intermediate value │
│ - Output write: charged 1 read │ - Theoretical floor: 474 ops   │ - Identifies non-overlapping   │
│ - 1 movement unit = 1 fJ       │ - Target: Beat 681 -> 474      │   temporal lifetimes           │
├────────────────────────────────┼────────────────────────────────┼────────────────────────────────┤
│ Locality Invariant:            │ Scoring & Verification:        │ Yaroslav Bulatov Coloring:     │
│ - Rearrangement Inequality:    │ - Formal C = A * B verifier    │ - Rearrangement optimal mapping│
│   Hot values MUST be placed in │ - Exact integer bit-exactness  │ - Highest read counts assigned │
│   cold lowest physical addrs   │ - Static cost = sum(sqrt(addr))│   lowest addresses (1, 2, 3..) │
└────────────────────────────────┴────────────────────────────────┴────────────────────────────────┘
```

---

## 4. Measurement provenance

All numbers measured 2026-09-02 on the host described above, harnesses:
`dally-eval/examples/cross-engine-bench.rs` (the five record programs)
and `dally-eval/benches/gpu-throughput.rs` (the siswalk search shape,
including an occupancy sweep showing tiling invariance: lanes 32/16/8/4
within noise). Python column: warm 1,024-instance batches of the
reference evaluator. Bit-exactness: every GPU row re-verified against
CPU outputs on live batches during the same runs.
