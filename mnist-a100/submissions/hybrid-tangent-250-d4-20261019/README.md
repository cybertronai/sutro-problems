# D4 tangent hybrid with 250 student updates

Reducing the published [tangent hybrid](../hybrid-tangent-d4-20261019/README.md) from 400 to 250 Adam updates, with cross-teaching moved from chunks (13, 27) to (8, 17), gives **77.010242 ms/call** and **11,464.572860 mJ/call** medians across three fresh canonical sandboxed Modal evaluations. All three runs pass difficulty 4 (MNIST error ≤2.30%); pooled MNIST error is **2.148182%** (322,911/330,000 correct).

These are author execution results, not independent reviewer execution. Independent review is pending; this PR does not change the leaderboard. No failed qualification run was replaced or retried. Three distinct recorded GPU UUIDs are included.

| Run | MNIST error | Margin, labels | Ranked ms | Above-idle mJ |
| - | -: | -: | -: | -: |
| [1](evidence/official/run-1.json) | 2.160909% | 153 | 76.822557 | 11464.572860 |
| [2](evidence/official/run-2.json) | 2.051818% | 273 | 77.038087 | 11928.423796 |
| [3](evidence/official/run-3.json) | 2.231818% | 75 | 77.010242 | 11388.266687 |

Compared with the published hybrid's independent reviewer medians (97.047379 ms, 14,754.131810 mJ), this author group is 20.65% faster and uses 22.30% less above-idle energy. These are separate draws and hosts, not a paired accuracy comparison. The parent's pooled reviewer error was 2.083333%; reducing training trades some accuracy margin for speed.

## Exact change and credits

The algorithm is @spartanhaden's hybrid, unchanged apart from student update count, proportional cross-teaching schedule, and corresponding method documentation. The initial source SHA-256 is `102363cfe2825eee42b5ee641731655679825e7087c3d83e19c3340d93859306`. This submission's exact tested source is 20,458 bytes, SHA-256 `ec90dfb48793017577eaeac212b7357ff19bfef950dfb884025921b53198e608`; all three raw records identify those same bytes under their execution filename `hybrid-d4-steps250-20261019.py`. Packaging renames the file to `submission.py` without changing its contents.

Retained components: two RFM teachers with three AGOP metric updates, FlashAttention kernel products, Nyström-preconditioned CG, HKNN; three 60–256–256–10 RMSNorm/ReLU students; 1,024 labelled plus 1,024 pseudo-labelled query rows per update; local tangent jitter and isotropic noise; Adam and the original fractional LR anneal; teacher/student probability mixture and final HKNN. Every call refits the teachers and resets/retrains students. Only graphs, scratch tensors and compiled code persist. Query labels are never used.

The parent credits @islamborghini's kernel-PCG/RFM ensemble, Radhakrishnan et al. RFM, Vincent and Bengio HKNN, @likelytobelaura's tangent cross-teaching/ReLU ensembles, @sjbaebae's graph-readout work and modded-nanogpt's zero head initialization. All original source credits are retained. This is a training-budget optimization of their learner, not a new learner or kernel implementation.

## Validation and reproduction

Before qualification, eleven fresh paired development input draws, each with three stochastic retrainings, gave 2.214242% mean error and 75.396596 ms warm full-call wall time at 250 updates; the unchanged 400-update parent gave 2.115758% and 92.884268 ms. Those development timings exclude sandbox overhead and are separate from the canonical results above. The 200-update alternative had weaker mean accuracy margin (2.274848%); it was not substituted into qualification. Repeated stochastic trainings on one input draw are not independent input draws.

From `mnist-a100/`, run the exact method with the unchanged runner and scorer:

```sh
python run_modal.py submissions/hybrid-tangent-250-d4-20261019/submission.py:hybrid_d4 --difficulty 4 --runs 3 --json /tmp/hybrid250-d4-verification
```

The saved [raw records](evidence/official/) include all initial outcomes, complete call data, energy windows, GPU identities and source hashes. [Verification](evidence/official/verification.json) reconstructs every verdict, ranked time, summary and energy value through the repository's canonical `judge` and `energy_summary`. The verifier is adapted from the parent's published reviewer verifier; it audits saved evidence and does not constitute another fresh execution group.

From this submission directory:

```sh
python verify_records.py evidence/official submission.py --difficulty 4 --function hybrid_d4 --record-filename hybrid-d4-steps250-20261019.py --output evidence/official/verification.json
```

The scorer SHA-256 is `1bc6d8d96a776565bc70f3ccdb91be7de57187d4495b2b0437fac98f5e120064` and runner SHA-256 is `1180b843cf6c32414774da90c835a8b7298d2db8719989237a793ca1c62a10ab`, byte-identical to the repository versions used for this audit. Compilation/graph capture happens during canonical warm-up; ranked measurements cover complete subsequent calls, including teacher fitting, training and readout. The unchanged warm-up and timing checks passed in every run. Source review can be performed by comparing this file against the linked parent; no custom native-kernel experiments are included.
