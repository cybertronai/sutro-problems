# D3 Fisher-PCG with native local-hull CG

A latency candidate derived from [Fisher-PCG D2](../fisher-pcg-d2-20261019/README.md) by @spartanhaden, using @sigkillme0’s learned Fisher/compression method and @islamborghini’s kernel-PCG work. The HKNN affine-hull readout comes from the published hybrid-tangent work. Existing credits remain in the source.

The learner fits a full-dimensional Fisher metric, compresses training data to 6144 hard rows plus 240 weighted class landmarks, and performs eight extrapolated FITC-preconditioned CG updates. Flash attention evaluates test scores; local affine hulls correct the 384 lowest-margin queries, with 70 neighbors per candidate class and a 2048-neighbor prescreen. All fitting is repeated per call; no test labels are used. This is an alternative implementation/budget to the published D3 record, not an isolated kernel-only comparison against it.

The execution change replaces each local 61-dimensional Cholesky solve with native FP32 CG: one 64-thread CTA per system, a shared-memory matrix with padded stride 65, at most 48 iterations, and a recursive residual threshold of 1e-6 relative to the initial residual. Neighbor selection, compensated Gram construction, regularization and voting remain unchanged. This is a numerical approximation, not a claim of bitwise equivalence.

## Canonical author results

| Run | Ranked ms | MNIST error | Above-idle energy mJ |
|---|---:|---:|---:|
| 1 | 4.990809 | 2.604545% | 858.428614 |
| 2 | 4.660437 | 2.533636% | 766.636973 |
| 3 | 5.568000 | 2.601818% | 839.981355 |

**All three independent canonical sandbox evaluations pass.** Median ranked latency is **4.990809 ms**, median energy **839.981355 mJ**, and pooled MNIST error **2.580000%** (321486/330000 correct). Each run contains 11 MNIST calls and four foreign hold-out calls and independently meets the judge. Three distinct A100-SXM4 GPU UUIDs were observed. All original returned outcomes are included; no agent-initiated qualification replacements were launched.

The listed independent reviewer record is 5.179753 ms / 320.165897 mJ. This author group is approximately **3.65% faster in median latency**, but uses substantially **more energy** and has higher pooled error. Different draws and GPU hosts mean this is not a paired comparison. Independent reviewer replication remains outstanding; this submission does not edit the leaderboard or claim an accepted record.

[Raw records](evidence/official/) include complete timings and energy windows. `verify_records.py` independently rebuilds the canonical judge, ranked score, summary, source identity and energy-window measurements. All checks pass for all three records. These are saved-record verification checks, not a second execution group. Scorer and runner hashes match the tracked versions and are in `verification.json`.

Source SHA256: `321a606267643768e00a9fb760e9864a0a0a9d37c0e83e2045d9443f33dc4cfb`; **20027 bytes**. The CUDA/PTX source is separately maintained under [development](development/), and its exact bytes are embedded in the single-file submission to satisfy the 20480-byte contract. NVRTC compiles it during normal warmup; fitting data and trained parameters are never cached between calls. Only input-independent code, scratch and graph state persist.

```bash
python run_modal.py submissions/native-hull-cg-d3-20261019/submission.py:kc_d3 --difficulty 3 --runs 3
cd submissions/native-hull-cg-d3-20261019
python verify_records.py evidence/official submission.py --difficulty 3 --function kc_d3 --record-filename classical-native-hull-d3-20261019.py --output evidence/official/verification.json
```

## Development and numerical evidence

The included eleven-input fresh paired development comparison measured the PyTorch hull baseline at **2.524545% / 4.677300 ms**, versus native hull-CG at **2.524545% / 4.457801 ms**. There were **zero changed warm predictions across 110000 paired queries**. These are warm complete-call development measurements, not canonical qualification. The baseline here uses the same learner and differs only in the hull solve.

On actual 1152 x 61 x 61 regularized hull systems, FP64 reference checking gave relative solution error about **8.47e-7** and residual about **1.06e-6**, with finite outputs. Those checks precede graph capture and are not part of the submitted classifier. The unpadded native prototype was slower (4.787536 versus 4.714052 ms); stride padding and coalesced loading produced the retained version. Combining a separate Gram/exp fusion gained only ~0.046 ms beyond CG in the fresh screen and changed 16 predictions; it is not included. Native PCG vector fusion and wider duplicate-RHS batches were also rejected as slower; their summaries are included.

The standalone packaging smoke draw produced **2.81% error for both implementations** with zero changed predictions, at 4.549643 versus 4.767757 ms. That single draw is not a qualification claim. Its source was subsequently changed only to correct prose attribution; executable AST identity was checked. Exact final-source canonical results above determine qualification.
