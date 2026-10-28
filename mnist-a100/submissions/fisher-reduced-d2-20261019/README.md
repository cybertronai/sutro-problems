# Research archive: D2 Fisher-PCG with reduced fitting work

**Research inclusion; D2 qualification remains unresolved.** The existing author group below passed 3/3 (median **3.029522 ms**). The initial independent reviewer group passed **2/3**: its third complete run scored **106,259/110,000**, one label below the **106,260** required for D2. All 45 timed calls are retained; missing energy on that failed score is not the reason for nonqualification. There is no qualifying reviewer-group median or leaderboard promotion. Original source, author evidence and upstream attribution are preserved. See [reviewer evidence](reviewer-evidence.json).

This derives directly from [the published Fisher-PCG D2](../fisher-pcg-d2-20261019/README.md) by @spartanhaden, itself a PyTorch implementation of @sigkillme0’s Fisher/compression method. All original source attribution is retained. This is a parameter-budget change, not a new kernel or learning algorithm.

The retained hard rows decrease from 4096 to 3072, and the rows used to estimate the Fisher metric decrease from 2048 to 1024. Five PCG steps and all other program operations remain unchanged. Only those two configuration literals and corresponding documentation differ from the parent. Everything is fitted anew per call; test labels are never used.

## Three independent author sandbox evaluations

| Run | Ranked ms | MNIST error | Energy mJ above idle |
|---|---:|---:|---:|
| 1 | 2.781440 | 3.246364% | 295.933230 |
| 2 | 3.086730 | 3.250000% | 430.983288 |
| 3 | 3.029522 | 3.370000% | 408.781601 |

All three pass the unchanged canonical sandbox judge. Median ranked latency is **3.029522 ms**; median energy is **408.781601 mJ**. Pooled observed MNIST error is **3.288788%** (319147/330000 correct). Each run independently qualifies, including all four foreign hold-out calls. All initial outcomes are included; there were no qualification reruns, replacements, or infrastructure preemptions. Three distinct A100-SXM4 GPU UUIDs were observed. Energy is higher than the published parent's listed 208 mJ; this is a latency candidate, not an energy improvement. The listed parent is approximately 3.2 ms, so the latency gain is modest and subject to independent reviewer replication. No leaderboard change is requested before review.

These are author-executed evaluations, not independent reviewer executions. Complete raw records and energy samples are in [evidence/official](evidence/official/). The saved-record verifier independently rebuilds the judge outputs and energy-window measurements; all checks pass. Scorer and runner hashes match the tracked versions and are recorded in verification.json. Source SHA256: `65a14f4024d8af73ea3b7c088aeec44591715095b5cf2f76bcaf3a5c1d693a91` (16400 bytes).

```bash
python run_modal.py submissions/fisher-reduced-d2-20261019/submission.py:kc_d2 --difficulty 2 --runs 3
cd submissions/fisher-reduced-d2-20261019
python verify_records.py evidence/official submission.py --difficulty 2 --function kc_d2 --record-filename classical-d2-reduced-20261019.py --output evidence/official/verification.json
```

## Development evidence

Both eleven-input paired development screens are included under evidence. They are warm complete-call tests, not canonical qualification. The refined screen measured the original parent at 3.129091% / 2.700487 ms versus this candidate at 3.300000% / 2.505534 ms. Reducing further to 2048 hard rows failed the accuracy target in the earlier screen. More PCG steps at 3072 rows did not recover useful accuracy. These learner-budget changes trade accuracy margin for less complete-call work; no numerical-equivalence claim is made.
