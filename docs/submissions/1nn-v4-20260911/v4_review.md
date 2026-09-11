# Independent review of the tape-machine scorer

**No blocking correctness or model-accounting defect found for the submitted MNIST-small program.** The scorer is submission-owned; its FP32 interpretation, tape encoding and occupied-cell Area remain explicit conventions, not official benchmark certification.

Reviewed scorer SHA-256: `0187db7f3bb9757ff7b5fc4ac375beab5bf7ec8fbb76584b03c6b7436a9ea19c`.
Reviewed generator-run result SHA-256: `7078bcc4adfd9e4dfbcced08ea0b045972d17d2eba6795a1dbb480c6eff67bc3`.
Model-spec revision: `26abcca402de647381d31286d42dfbb7a001763d`.

## Independent results

All **13 checks passed**. The review reconstructed the input tape directly from the three canonical learner arrays and compared all bytes with the saved input tape. It compared every output word and all 600 model predictions with the independently reviewed CPU reference. It recomputed the score from per-address read/write counts in `placement.csv`, using the published distance formulas rather than the scorer's closed-form helper:

- Energy: **1,875,974,400 fJ = 1.8759744 µJ**.
- Time: **1,682,197,200 ps = 1.6821972 ms**.
- Charged reads: **22,316,400**; charged writes: **11,159,400**.
- Allocated scratch: **6,014 words = 24,056 bytes**.
- Occupied-cell area: **6,014 µm²** under the declared convention; enclosing rectangle: **12,012 µm²**.
- All **11,400 input words** are consumed and **600 output words** produced.

All 6,014 coordinates are unique, inside the machine bounds, and match the declared half-diamond placement. Every persistent training word has exactly 600 charged reads and zero charged writes: its initialization is through uncharged `recv`, and each test query reads it once. The operation counts match the fixed 600 × 600 × 9 computation, totaling **11,171,400 instructions**.

Independent edge-case checks verified that aliased multiplication charges its source address twice before overwriting it; `select` charges the condition and both alternatives; distinct read/write propagation times are applied correctly; far-away tape operations still have zero modeled charge; input overread and unconsumed input are rejected; and equal-distance toy examples select the first training label.

The final result's source hash and replay-aware metadata match the final source. The expanded instruction archive contains exactly 11,171,400 newline-terminated instructions and has uncompressed SHA-256 `f6f657cc159160050e73d1df8ae169b313b795f3d1b1a45a72ad4f757383ef2b`, matching the emission record. Full text replay is a separate validation procedure; this independent review did not repeat the entire long-running interpretation.

## Scope and remaining limitations

The input-tape loader checks shapes/ranges but does not itself authenticate canonical array hashes. For this retained run, the independently reconstructed tape matches the canonical arrays exactly, and the separate dataset audit establishes all six canonical identities. Reproduction should retain those identity checks rather than treating any 600/600 archive as canonical.

The interpreter supports the subset of v4 used by this program. Its boolean comparisons, raw-word copies/selections and FP32 arithmetic implement the submission's stated convention correctly on the finite, bounded MNIST inputs. This does not validate every possible v4 program, undefined exceptional arithmetic, or an official numeric interpretation that the specification has not supplied.

Tape I/O costs are excluded by the model. The reported modeled time and energy therefore do not include physical end-to-end data movement, and occupied scratch cells do not represent complete chip area. The host scoring timer is separately labeled and excludes input loading, placement generation, output writing, and independent verification.

[Machine-readable independent checks](v4_review.json) · [Rules and ambiguities](ambiguities.md)
