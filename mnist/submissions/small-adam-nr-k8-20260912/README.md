# NR-K8 Adam MLP: MNIST-small, 67% target

**Review request: this PR adds an Adam optimizer path to the grid scorer; scorer approval is requested as a merge condition (grid columns are provisional until then).**

Submission date: September 13, 2026 (UTC). Contributor: OpenCode.

This submission reports **7,409 / 11,000 = 67.35%** mean accuracy with sample
standard deviation **1.76 pp** over eleven independently sampled datasets,
meeting the 67% target (7,370 required). Each draw uses 1,000 training and
1,000 disjoint test images resized to 3x3 from the official 60,000-image
training split. The learner is a 9-32-10 MLP trained with NR-K8 Adam.

## Accuracy evidence

Draws use `Generator(PCG64(seed)).permutation(60000)`; rows 0-999 are training
and rows 1000-1999 are test. All 11 prediction vectors were written and hashed
before any evaluation-label slice was opened (two-phase freeze/score).

| Draw | Dataset seed | Correct / total | Accuracy |
| ---: | ---: | ---: | ---: |
| 0 | 20261201 | 675 / 1,000 | 67.5% |
| 1 | 20261202 | 693 / 1,000 | 69.3% |
| 2 | 20261203 | 688 / 1,000 | 68.8% |
| 3 | 20261204 | 670 / 1,000 | 67.0% |
| 4 | 20261205 | 636 / 1,000 | 63.6% |
| 5 | 20261206 | 669 / 1,000 | 66.9% |
| 6 | 20261207 | 676 / 1,000 | 67.6% |
| 7 | 20261208 | 676 / 1,000 | 67.6% |
| 8 | 20261209 | 676 / 1,000 | 67.6% |
| 9 | 20261210 | 698 / 1,000 | 69.8% |
| 10 | 20261211 | 652 / 1,000 | 65.2% |

Exact mean: `7409 / 11000 = 0.6735454545454546`. Sample SD: `1.7642922866484638 pp`
(ddof=1). Target: 7,370 correct; met. Seeds, train/test indices and input
hashes are in `draw_manifest.json`; prediction hashes in
`prediction_manifest.json`.

## Frozen learner: NR-K8 Adam

9-32-10 MLP, batch 25, lr 0.2, 100 epochs, fixed supplied sample order, fresh
seed-101 initialization per draw, ordered FP32 arithmetic, no state transfer.
Per minibatch, gradients come from pre-update weights and are summed in
ascending order. `step = lr / batch = 0.008`. With `t` the global minibatch
index starting at 1:

```
m = 0.9*m + 0.1*g
v = 0.999*v + 0.001*g*g
mhat = m / (1 - 0.9**t)
vhat = v / (1 - 0.999**t)
y = vhat + 1e-6;  repeat 8x: y = 0.5*(y + vhat/y)     # Newton-Raphson sqrt
p = p - step * mhat / (y + 1e-8)                       # IEEE division
```

The Newton-Raphson lowering uses only existing v4 ops (`div`, `add`, `mul`).
The procedure (width 32, 100 epochs, K=8) was frozen on disjoint pilot seeds
(20261006-15, 67.47%) before the official draws were evaluated. This is a
defined optimizer variant, not exact-sqrt Adam.

## Measured A100 cost

| Accuracy | Energy on A100 | Time on A100 |
| --- | ---: | ---: |
| 67.35% +/- 1.76 pp | 2,560 mJ | 110 ms |

Cost values use two significant figures in the table row; exact values are in the JSON files.

Medians of three cyclic rounds of CUDA-graph replays on an A100-SXM4-40GB
(session 2; session 1: 2,483 mJ / 107.1 ms). Each replay resets parameters,
optimizer moments and step state, normalizes inputs, runs all 4,000 minibatches
and produces 1,000 predictions. GPU parameters, scores and predictions are
bitwise equal to the ordered CPU reference (eager and graph; canonical,
changed-labels, changed-queries; all 11 official draws re-verified).
Scope excludes transfers, allocation, JIT, graph capture and cold start; NVML
board energy is idle-adjusted with paired idle measurements.

**Disclosure:** a same-container interleave shows this advantage over the
published H32 row is largely the fused runtime. Under matched execution
(H32 learner through the same runtime) H32 costs 44.5 ms / 1,062 mJ versus
106.9 ms / 2,424 mJ here. This entry does not claim accuracy superiority
(+0.27 pp; the quoted +/- 1.76 pp is a sample SD, not a standard error).

## Grid-model cost (provisional pending scorer review)

| Energy in grid model | Time in grid model | Peak scratch | Time to score |
| ---: | ---: | ---: | ---: |
| 0.185228620462 mJ | 1,798.760521 ms | 128,352 bytes | 0.123 s |

Exact totals: 185,228,620,462 fJ and 1,798,760,521 cycles for the serialized
schedule under spatial-computer revision `01a0bd5e0d2564825b0f53dd766f763c82dbc7c0`.
The score comes from an extension of `grid-mlp-scoring-20260912` that adds the
Adam optimizer path; the SGD path reproduces the official score exactly
(0.242990182992 mJ / 3073.508456 ms, byte-identical program), and the Adam path
uses only existing instruction-set operations. The extension, exact generated
program, scorer diffs and score are in `grid-scorer/`. **These columns are
provisional until the scorer extension is reviewed; the Adam program's state
initialization, bias correction, Newton-Raphson arithmetic and memory/tape
accounting are open for review.**

## Reproduce and audit

```sh
python run.py prepare        # official draws, indices and input hashes
python run.py freeze         # train + hash predictions (no evaluation labels)
python run.py score          # verify hashes first, then count
python reference.py          # ordered FP32 learner (portable)
modal run gpu_benchmark.py   # one fused A100 session
```

`verify.py` re-checks the frozen manifests, the per-draw input hashes and, when
present, the A100 arrays against the ordered CPU reference.

Evidence: [accuracy](accuracy.json), [draw manifest](draw_manifest.json),
[prediction manifest](prediction_manifest.json), [protocol](protocol.json),
[GPU results](results/gpu_results.json), [grid score](grid-scorer/program/grid-score.json),
[submission metadata](submission.json).
## SHA-256 manifest
- `PROPOSED-ROW.md` `93860cbddff567c5cd5213bc546f04349488c24b247b687ae5a3f6607be975a1`
- `REPRODUCE.md` `4aa09bc248e69a467d43337747cac25839076c37ab2c2e71def56364edf3bf57`
- `evidence/accuracy/accuracy.json` `f04833e04855b1b58ddaee0da71d27bdb1e445fb48021a76a1991c62eebdec77`
- `evidence/accuracy/draw_manifest.json` `c0417be944c3490df8260c617561d5b6ee14f4d9ee1dcf7ca92f777d3c31be17`
- `evidence/accuracy/evaluation_freeze.json` `98b0fdeec83858ffcd125cc0c7b59099f1f2186cabcad2b0c72d324fcb79b87f`
- `evidence/accuracy/prediction_manifest.json` `abdd678498089dc1177f66d40abb93ee8f7c77ff1a73a45c34e5622e6110fef0`
- `evidence/accuracy/predictions/draw-00.npy` `5a858a186d5cfc1c86904e5abb8876b3399de8e56a5f9066a0f61620a0819d61`
- `evidence/accuracy/predictions/draw-01.npy` `e79fbb0f7cdbb502ef7913b40092b903f635e479372ecafecefcd76c579e422f`
- `evidence/accuracy/predictions/draw-02.npy` `5230dc0f6618d043e05f7caad9068734bfaec576014c880e27c9a0429f2c035d`
- `evidence/accuracy/predictions/draw-03.npy` `8438a7b0f31b1e4219ee7b966286f5e5f7030078f914ed9aff1c553002d26d93`
- `evidence/accuracy/predictions/draw-04.npy` `3cdf3ebfb2ab01995f910032b45772334e8297d363c33e420e735533c862a8d0`
- `evidence/accuracy/predictions/draw-05.npy` `58c124053114980f8a78302bcd5b3ebbc23faabd1e224a475e9063273146b23d`
- `evidence/accuracy/predictions/draw-06.npy` `ef5d27d0603935514474551546de47210546dc5f1893fb95959f341d29958bb1`
- `evidence/accuracy/predictions/draw-07.npy` `4c22477a4a40d33af3a07d0e529a7b75311fbbefccfe5b72ad740ccd3d28779c`
- `evidence/accuracy/predictions/draw-08.npy` `e7a95db775bfc39e7cda08cfb492a6c033ae216ff357cf820da773a659db6c79`
- `evidence/accuracy/predictions/draw-09.npy` `9700a9d1cd430e566b870f7e50820f00b12e89489afad0f64b2674015b02cd0c`
- `evidence/accuracy/predictions/draw-10.npy` `720a1e27d8c78d6850470f456669114901dc4ebaac92457c2fbc33080f5ab5d0`
- `generated/adam11-expected.npz` `03bf5731294b0b9d9dc7014210e6ce04b4120c52a78064cdeb8d5abda4d07bae`
- `generated/adam11-payload.npz` `4e62ca44b1a06c7bc28b35399d53ef74786a150447c657bd682c2da590ea7556`
- `gpu_benchmark.py` `d0b3e747b1179d43c65b26b12f501781bfdb842e5172d17b9ca377f79d94fab9`
- `grid-scorer/affine.diff` `f7eb921098c338ad6d1138e43171a38d7ee526266cb206f6c7aab6aeacc54962`
- `grid-scorer/model_ir.diff` `e5d98366cfbd524618f19c5fbd6e67d7250d87e0b7a6a671b411cd7ca0256527`
- `grid-scorer/program/grid-score.json` `4b74cf93f83c0faf50815b2f3267d1b6fa883fedf12e2a33f4b5785c19a62b78`
- `grid-scorer/program/program.spatial.json` `6e7e44a21ab0169a05318f165f8010fb32b17376200c46341da9ddef11d110bf`
- `grid-scorer/score.diff` `d5b51da2bf87ce885437dde5d5a0ca3bd142cbb239c653bb1771c3e6305857fc`
- `protocol.json` `aca1a305008256641bc31a3ec53190b27d79f55159b565525ed68a8413f8cc3b`
- `reference.py` `16d24a82e6d2a010dd045a7ddfabcca6a99b2c16d9ed8da928115ec5178d7809`
- `results/adam-results-session1.json` `6e4e3ca1d848dbe369533ca3421ff662c27038532507900a7e22aa48435c4dea`
- `results/adam-results-session2.json` `0f09cb57e9bfe54e4dfbb55f51b08068767075bf0bd566637973d5f155f6b868`
- `results/gpu_results.json` `0f09cb57e9bfe54e4dfbb55f51b08068767075bf0bd566637973d5f155f6b868`
- `results/gpu_results_session1.json` `6e4e3ca1d848dbe369533ca3421ff662c27038532507900a7e22aa48435c4dea`
- `run.py` `d5c6e2f1c8714f126ad8d6a47915eec4cf331b8c8bdce0ed7b09a610ae7734d7`
- `sources/adam-confirm.json` `89003057d7c1806c2d568a46c8be2431ee90f4d9357f3e7c3eedfaeee0a81b76`
- `sources/adam-nr-k.json` `8bc738f30a0cda954054cec9bfcb5f64c7ebbc1da9cb2446ee16d0e2f9867295`
- `sources/adam_confirm.py` `dd89343bc938368c581ef67a895f68bd3be319ef1d0174e615f8148f65f9666a`
- `sources/adam_evidence_v2.py` `7a88f11ae4ae9c7b7c1f55338126ae666bfd6223ff4802d32051685b9e83a852`
- `sources/adam_k8_final.py` `c729df76fcd84e931ee750b01c43f15e52b0c47ffa5fd2e8c139f565f984f77f`
- `sources/adam_nr_k.py` `04de65992595b62069b76204f1fea8856658b336858c8d10febc273ceb5bbc58`
- `sources/adam_nr_screen.py` `0c02655fe7b3438ed6ef18e7f9d06e8528c7cda2eb22d8766905b4e9cb14a947`
- `sources/make_payload_adam.py` `70664627ef1b5a33908802c24082abdc03d98d46d2668e1954f7973660401a7e`
- `sources/make_payload_adam11.py` `324af8da824c4661d6cef650ad7c74acee66de346e3c2aeb92cfbcc20d701a4d`
- `sources/modal_adam.py` `d0b3e747b1179d43c65b26b12f501781bfdb842e5172d17b9ca377f79d94fab9`
- `sources/optimizer_screen.py` `ab3bd2cecd3dfcfd6630c6ad5c7cce5f2101efbe3bc24701e7f8e59fb9a8c85a`
- `sources/scorer-adam/affine.py` `8ac38bf9fccbef87ba05c4dd6a4e5d9901dceae4e71baad0ddd5794c22298d45`
- `sources/scorer-adam/model_ir.py` `cc039abc33a731a471c4caf9e35d9c1799ea5246501ae9bea962406a91bcf910`
- `sources/scorer-adam/score.py` `41b47384ec525718a4546f98f5ad25a4d82dd8bdfb97a762e451476cae29eaf8`
- `sources/stage2.json` `2763f9144a345719cd12c39af6cd495c6527b7b826dff4591e305509d7f4b31d`
- `sources/stage2.py` `1967be37af97cf0cc6d6dee8aea38fb00e87d9e42f3bfec9c44ebdcbdb7d6f21`
- `sources/sweep.json` `ca4e90cc75d47067f306c00e3b3da9c10c48a237b7241202f653b2274baaa934`
- `sources/sweep.py` `e2dcc9e3983bd6e185e777473e784fa5a4f2c718c372db83c91c1fb0c280577c`
- `submission.json` `46f5ba4fa69cdfcb560cd82990e72bd92bff4f3da88d7f6019255c78eeb0e35d`
- `verify.py` `58cb20b0ba165c3c5ee37c9156dc1e879014e399e3143b2e4190d952ee7099f5`
