# Build Notes: Scheduled Dally Language & Engine PRs

**Author:** Andy Zhang (`@zh4ngx`)  
**Repository:** [cybertronai/sutro-problems](https://github.com/cybertronai/sutro-problems)  
**TL;DR:** Only **PR #48** (Language RFC) and **PR #58** (Tier 1 MNIST dataflow demo) need human review; the other 5 are automated benchmarks and search runs. Built via high-level architectural steering (no line-by-line prompting) using GLM-5.3 for Rust code and Gemini/GPT-5.6 for coordination and proofs (~1.8M tokens, 100% local compute on a 9950X + RX 6900 XT).

---

## 1. Reviewer Triage

Yaroslav asked for calibration on where human attention is actually needed vs. automated search artifacts:

- **Needs Human Review:**
  - **[PR #48](https://github.com/cybertronai/sutro-problems/pull/48) (Scheduled Dally Language RFC):** The core language design. Addresses the 27 TB scaling wall via hierarchical loop nests, closed-form $O(1)$ symbolic scoring, and physical grounding ($1\text{ fJ/byte}$, $1\ \mu\text{m}$, $c/160$).
  - **[PR #58](https://github.com/cybertronai/sutro-problems/pull/58) (Tier 1 MNIST Static Dataflow Demo):** Implements static streaming dataflow (rejecting full SRAM materialization). Bit-exact integer quantization, 53.0% accuracy, static cost of 4,625 distance units (~0.005 nJ per inference).

- **Reference / Skim Only:**
  - **[PR #49](https://github.com/cybertronai/sutro-problems/pull/49):** 3-engine execution comparison (Rust CPU vs. Python vs. GPU LDS).
  - **[PR #51](https://github.com/cybertronai/sutro-problems/pull/51), [#52](https://github.com/cybertronai/sutro-problems/pull/52), [#55](https://github.com/cybertronai/sutro-problems/pull/55), [#56](https://github.com/cybertronai/sutro-problems/pull/56):** 4×4 matmul search runs, access heatmaps, and the 858 search floor.

---

## 2. Build Footprint

- **Human Role:** Problem framing, physical constraints, and test gates. Zero line-by-line prompting.
- **Models:**
  - **GLM-5.3 (Zhipu AI):** Primary implementation work (Rust AST, interpreters, scorers, search scripts, and PR writeups).
  - **Gemini 3.7 Flash:** High-level task coordination, test verification across repos.
  - **GPT-5.6 Sol (OpenAI):** Accumulator lifetime analysis and mathematical lower-bound proofs.
- **Compute & Tokens:**
  - **Tokens:** ~1.8M total tokens across the 7 PRs.
  - **Local Compute:** ~14 hours continuous CPU (Rayon) and GPU (CubeCL) local search runs on an AMD Ryzen 9 9950X + Radeon RX 6900 XT ($0 cloud spend).
- **Verification:** 100% bit-exact parity across Python `matmul.score_4x4`, Rust `dally-eval`, and numpy across 5 reference test matrices before submitting.

---

## 3. Timeline

- **Sep 2:** Yaroslav suggests a scalable, cheat-proof 2D grid model.
- **Sep 3:** SDL designed to avoid 27 TB trace files; $O(1)$ closed-form scorer and execution benchmarks submitted (**PR #48**, **PR #49**).
- **Sep 4:** Following Yaroslav's guidance against full SRAM storage, added physical constants and built the 3×3 MNIST static streaming demo (**PR #58**).
- **Sep 4:** Automated matmul search runs exploring accumulator reuse and staging settled at 858 (**PR #51**, **PR #52**, **PR #55**, **PR #56**).
