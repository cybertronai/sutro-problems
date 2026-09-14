# Provenance of “Chebyshev” in the Sutro Grid VM design

Date: 2026-09-05  
Scope: Telegram history, `cybertronai/sutro-problems`, PR #48, and local DSH session traces

## Executive finding

“Chebyshev” is not supported by the recorded design requirements or by the project’s earlier physical model. The evidence instead consistently names Bill Dally’s two-dimensional **Manhattan** model.

The first located use of “Chebyshev” in this design lineage is commit [`4f4d3e2`](https://github.com/cybertronai/sutro-problems/commit/4f4d3e2e7fef347710ca8ec234ca00659170164f), the AI-assisted Grid VM report committed by Yaroslav on 2026-09-04. Section 5 changes the metric to `max(x,y)`, and decision D3 recommends “native 2-D Chebyshev.” The commit is authored and committed by Yaroslav and has `Co-Authored-By: Claude Fable 5`; its message calls the document a synthesis of a 14-agent design run.

The most likely mechanism is visible in the report itself. The synthesis optimized for two mathematical conveniences:

1. An `L∞` square in one quadrant contains roughly `c²` cells, resembling the legacy rank-to-radius law `ceil(sqrt(a))`.
2. `max` of affine coordinates is piecewise affine and therefore convenient for symbolic summation.

That is a shell-counting and scorer-convenience argument, not a physical-routing argument. It missed a stronger fact already implemented in the project: the upper-half-plane **Manhattan** shell at radius `c` has exactly `2c-1` cells, exactly matching the legacy `ceil(sqrt(a))` shell. No Chebyshev substitution is needed.

Local agents then propagated the term rather than challenging its physical premise. `sutro-06-ds` treated D3 as a potentially valid future metric and objected only to record continuity. `sutro-07-ds` received that conclusion as an input, wrote it into `docs/pr48-response-draft.md`, and PR #48 commit `e9e4d38` plus Andy’s posted comment repeated “native 2D Chebyshev” as a proposed v2 metric. This turned a one-document model error into an apparently accepted design option.

**Recommended correction:** use Manhattan distance (`L1`) everywhere. Keep the canonical one-dimensional address surface for v1, defined as an exact upper-half-plane Manhattan embedding. If a later spatial division permits native `(x,y)` placement, it must also use Manhattan distance and must be versioned because native 2-D placement changes the optimization surface and leaderboard baselines.

## Confidence and limits

| Finding | Confidence | Basis |
|---|---:|---|
| The prior and requested model was Manhattan, not Chebyshev | Very high | Telegram, earlier project docs, code, and Dally’s CACM description all agree |
| The first located repository occurrence is `4f4d3e2` | Very high | `git log -S'Chebyshev' --all` finds `4f4d3e2` first, then PR #48 commit `e9e4d38` |
| Fable’s design synthesis introduced the term | High | Commit attribution, linked Fable design sessions, no Telegram occurrence, and the report’s self-contained shell/scorability rationale |
| The exact internal chain of thought was “match `c²` growth, then prefer `max`” | High as a reconstruction, not directly observable | Section 5 explicitly gives those two reasons; private model reasoning was not available |
| Chebyshev is physically wrong for this benchmark | Very high | It contradicts the explicitly selected Dally Manhattan model and underprices rectilinear diagonal displacement by up to 2× |

This report does not claim access to Claude’s hidden reasoning. “Hallucination mechanism” below means the best evidence-based reconstruction from the generated text and surrounding records.

## Evidence chronology

### 1. The preexisting requirement is Manhattan

The Telegram database contains 2,588 messages from 2026-02-09 through 2026-09-05. A case-insensitive database-wide search finds **zero** messages containing `Chebyshev`.

Relevant messages from Yaroslav include:

| UTC timestamp | Message ID | Evidence |
|---|---:|---|
| 2026-03-30 18:52:23 | 1275 | Links Dally’s CACM article while discussing formalization of the metric |
| 2026-03-30 19:39:56 | 1277 | “Bill Dally advocates a model based on Manhattan distance…” |
| 2026-04-27 17:41:02 | 1659 | Describes evaluation on Bill Dally’s “2d grid” model |
| 2026-04-27 20:25:32 | 1660 | Proposes implementing algorithms “in the Manhattan distance model directly” with manual 2-D placement |
| 2026-09-02 22:16:18 | 2749 | “we want Bill Dally’s model” |
| 2026-09-02 22:16:19 | 2750 | “2D grid” |
| 2026-09-04 00:17:27 | 2768 | Calls the Manhattan grid a physically realizable processor |
| 2026-09-04 00:18:40 | 2769 | Gives the physical constants: 1 fJ per byte moved, 1 micron pitch, and `c/160`, compared with Dally’s 256 MB CACM example |
| 2026-09-04 00:19:40 | 2770 | Says these conversions provide energy, time, and chip size for an idealized 2-D grid algorithm in silicon |

The same thread records that Yaroslav was using Fable in parallel:

- Message 2752, 2026-09-02: “Also having fable brainstorm this in parallel.”
- Message 2755, 2026-09-03: links a “fable 5.1 brainstorming session on how to score.”
- Message 2784, 2026-09-05: says he is asking Fable to take Andy’s report, inject firmer choices, and produce a new design; links the Grid VM report as his current thinking.

The local implementation agrees with the chat. `simplified-dally-model/README.md` says Dally models data movement on a Manhattan grid and charges reads by Manhattan distance. `simplified_explicit_communication_model_figure.py` maps every positive address into an upper-half-plane Manhattan shell. `bytedmd/docs/manhattan-diamond.md` states the same model.

Dally’s article is also explicit: [“Two-dimensional Manhattan distance models on-chip communication”](https://cacm.acm.org/opinion/on-the-model-of-computation-point/).

### 2. PR #48 initially preserved the legacy metric

PR [#48](https://github.com/cybertronai/sutro-problems/pull/48) opened on 2026-09-03 as “RFC: the Scheduled Dally Language.” Its original commit `295f0843` says:

- Existing reads cost `ceil(sqrt(addr))`.
- The score is the sum of that function over source addresses.
- Small expansions are cross-validated bit-exactly against the existing scorer.

The original PR contains no `Chebyshev` occurrence. It uses two-coordinate placement syntax, but does not define `L∞` as the physical metric.

### 3. `4f4d3e2` is the injection point

Commit `4f4d3e2e7fef347710ca8ec234ca00659170164f`, dated 2026-09-04 18:14:35 -0700, adds `docs/grid-vm-competition-design.html`. Its metadata is:

- Author and committer: Yaroslav Bulatov
- Co-author trailer: Claude Fable 5
- Commit description: “Synthesis of a 14-agent adversarially-reviewed design run”

The exact change is in [Section 5, lines 233–240](https://github.com/cybertronai/sutro-problems/blob/4f4d3e2e7fef347710ca8ec234ca00659170164f/docs/grid-vm-competition-design.html#L233-L240):

> Cells are 2-D; distance is Chebyshev, d(x,y) = max(x,y). Roughly c² cells lie within cost c, reproducing the 1-D model’s ceil(sqrt(a)) shells… The payoff: max of affine coordinates is piecewise-affine…

The choice is repeated in [decision D3, line 453](https://github.com/cybertronai/sutro-problems/blob/4f4d3e2e7fef347710ca8ec234ca00659170164f/docs/grid-vm-competition-design.html#L453):

> native 2-D Chebyshev with canonical `lin` legacy embedding

No reference in the report supports changing Dally’s Manhattan metric to Chebyshev. The Dally reference at line 477 points in the opposite direction.

### 4. How local sessions laundered the term

#### `sutro-06-ds`

Archive: `~/.dsh/sessions/--home-andy-sutro--/session-681888ea-f6f0-4c69-beee-c4abf8fe4789/session.jsonl.zstd`

- Session title event `seq=3` identifies it as `sutro-06-ds`.
- Tool result `seq=3034` ingests the report’s Section 5 verbatim, including the `max(x,y)`, `c²`, and piecewise-affine rationale.
- The later synthesis accepts the scorer convenience as real and frames D3 only as a migration problem: native 2-D would break trace cross-validation and rebaseline records.
- Its recommended compromise is `lin(a)` for v1 and native Chebyshev for a v2 spatial division.

That response caught a genuine compatibility problem but did not compare the proposed norm with the project’s physical source model. In other words, it challenged **when** to adopt Chebyshev, not **whether** Chebyshev belonged at all.

#### `sutro-07-ds`

Archive: `~/.dsh/sessions/--home-andy-sutro--/session-fd8a7081-ea70-4aa3-90fd-c54734e60165/session.jsonl.zstd`

- Session title event `seq=3` identifies it as `sutro-07-ds`.
- Its initial task, events `seq=4` and `seq=8`, already states as an input that the agent should “push back on native 2D Chebyshev … for v1” and reserve it for v2.
- The session therefore does not independently evaluate the metric. It turns the inherited conclusion into a polished response.
- Tool call `seq=12768` writes `docs/pr48-response-draft.md`, which says Chebyshev’s “payoff is real” and proposes it for v2.

This is a textbook echo chain: generated premise, uncritical synthesis, distilled task, polished draft.

### 5. Propagation into PR #48

PR branch commit `e9e4d38890903d23b1c2fc1bdcbecc81d239e971`, dated later on 2026-09-04, is the second repository commit found by `git log -S'Chebyshev' --all`.

It adds §8.3, which says native Chebyshev should be designated for v2. Andy’s posted PR comment on 2026-09-05 02:59:26Z repeats that conclusion in item 5. The PR thus echoes the Grid VM report rather than originating the term.

The current PR text also contains a second metric-layer error in §8.1:

> each edge is charged `ceil(sqrt(distance))` once

For a native 2-D geometric edge, the distance is already a physical route length. It should be charged directly. `ceil(sqrt(a))` converts a **linear address rank** `a` into the radius of its packed 2-D shell. Applying another square root to a geometric distance conflates rank with route length.

## Reconstructed hallucination mechanism

The generated report appears to have optimized a mathematical representation in this order:

1. Start from the legacy scalar law `cost(a)=ceil(sqrt(a))`.
2. Seek a native 2-D ball with `Theta(c²)` capacity so rank `a` corresponds to radius `Theta(sqrt(a))`.
3. Choose a first-quadrant square because `max(x,y)≤c` describes it.
4. Notice that `max` of affine forms is piecewise affine, making exact symbolic scoring attractive.
5. Promote that convenient geometry to the physical distance metric without checking it against Dally, the Telegram requirements, or the existing Manhattan embedding.

The report itself exposes the mistake by admitting that its shell boundary has `2c+1` cells while the legacy shell has `2c-1`. The existing Manhattan construction has no mismatch.

For radius `c≥1`:

- Legacy addresses with `ceil(sqrt(a))=c`:
  
  `c²-(c-1)² = 2c-1` cells.

- Upper-half-plane Manhattan shell with `y>0`:
  
  `{(x,y): |x|+y=c}` has `x=-(c-1), …, c-1`, hence exactly `2c-1` cells.

- First-quadrant Chebyshev shell:
  
  `{(x,y): max(x,y)=c}` has `2c+1` cells.

Thus the original Manhattan model is not only physically intended. It is the exact discrete shell realization of the legacy scoring law.

## Physical analysis

For points `p=(x₁,y₁)` and `q=(x₂,y₂)`, let `dx=|x₁-x₂|` and `dy=|y₁-y₂|`.

| Metric | Formula | Geometry | Fit to this model |
|---|---|---|---|
| Chebyshev (`L∞`) | `max(dx,dy)` | Squares; king moves allowed | Wrong for the selected rectilinear routing model |
| Euclidean (`L2`) | `sqrt(dx²+dy²)` | Circles; straight diagonal permitted | Geometric lower bound, but not the route length of the model |
| Manhattan (`L1`) | `dx+dy` | Diamonds; horizontal plus vertical routing | The metric explicitly selected by Dally and Sutro’s prior docs |

The norms obey

`L∞ ≤ L2 ≤ L1 ≤ 2L∞`

and

`L2 ≤ L1 ≤ sqrt(2)L2`.

For diagonal displacement `(d,d)`, Chebyshev charges `d`, Euclidean charges `sqrt(2)d`, and Manhattan charges `2d`. Saying Chebyshev provides “cost-free diagonal routing” should be read precisely: after paying for the larger coordinate, displacement along the smaller coordinate adds no cost. It can underprice a rectilinear route by a factor of two.

Modern physical routing has additional effects such as vias, repeaters, congestion, shielding, and layer-dependent resistance/capacitance. The benchmark deliberately abstracts those away. Within that abstraction, orthogonal horizontal and vertical segments make route length Manhattan. Chebyshev is not a harmless alternative norm because the energy and delay constants are calibrated per unit of route length.

### Why the physical constants require Manhattan route length

The September 4 calibration says:

- cell pitch: 1 micron;
- movement energy: 1 fJ per byte-micron;
- effective propagation speed: `c/160`.

Under Manhattan routing, a byte traveling from `p` to `q` traverses

`wire_length = (dx+dy) × 1 micron`.

The movement term is therefore

`E_move = byte_count × (dx+dy) × 1 fJ`,

and propagation delay is proportional to the same route length. This is dimensionally coherent. Replacing `dx+dy` by `max(dx,dy)` silently changes the assumed wire length while retaining constants calibrated for rectilinear distance.

These constants remain approximations, not foundry-validated laws. Manhattan makes the conversion internally coherent; it does not by itself prove that 1 fJ per byte-micron is accurate for every process or account for endpoints, arithmetic, clocks, or leakage.

### Separate unit error found in the Grid VM report

Section 5 writes propagation speed as `1.875 µm/ns` while calling it `c/160`. The conversion is:

`c/160 ≈ 1,873.7 µm/ns`,

so one micron takes approximately `0.000534 ns = 0.534 ps`. The report’s `1.875 µm/ns` is smaller by a factor of about 1,000. This should be corrected independently of the metric change.

## Mathematical and scorer analysis

### Legacy one-dimensional scoring

The existing scorer uses

`d(a)=ceil(sqrt(a)) = isqrt(a-1)+1`, for `a≥1`.

The exact prefix sum is

`F(n)=q(q+1)(4q-1)/6 + (n-q²)(q+1)`, where `q=floor(sqrt(n))`.

This gives exact constant-time sums over contiguous address ranges. More general affine address forms over multidimensional iteration boxes require floor-sum or banded lattice methods; they should not be described categorically as pure `O(1)` polynomials.

### Exact Manhattan embedding of legacy addresses

Define `k=ceil(sqrt(a))` and `t=a-(k-1)²-1`, where `0≤t≤2k-2`. One valid orientation is

`x=-(k-1)+t`,  
`y=k-|x|`.

Then `y>0` and

`|x|+y=k=ceil(sqrt(a))`.

Alternating the orientation of successive shells produces the project’s existing spiral but does not change distance. This `lin(a)` is an exact isometric embedding of every legacy core-to-address cost into the upper-half-plane Manhattan grid.

### Native 2-D Manhattan scoring

For an origin-anchored cell with nonnegative affine coordinates, Manhattan cost is simply `x(i)+y(i)`, an affine form. Its sum over a rectangular affine loop domain is elementary.

For general affine endpoints, cost is

`|X(i)|+|Y(i)|`,

where `X` and `Y` are affine coordinate differences. Partitioning the iteration domain by the sign hyperplanes `X=0` and `Y=0` makes the cost affine on each region. Exact sums are weighted lattice-point sums over rational polytopes, the same Ehrhart/floor-sum family already accepted for the schedule scorer.

Therefore Manhattan does **not** sacrifice closed-form affine scorability. In the common origin-anchored nonnegative case it is simpler than Chebyshev.

### Chebyshev scoring

Chebyshev cost is `max(|X(i)|,|Y(i)|)`. It is also piecewise affine after partitioning by signs and comparisons such as `|X|≥|Y|`. It is scoreable, but scoreability is not a reason to prefer it physically. Both `abs` and `max` lead to finite polyhedral partitions; Manhattan needs no `max` partition when coordinates are nonnegative.

### Euclidean scoring

Euclidean cost is `sqrt(X(i)²+Y(i)²)`. Even when `X` and `Y` are affine, the summand is generally not piecewise affine or an Ehrhart quasi-polynomial. Exact symbolic sums usually require enumeration, special cases, or a separately specified certified approximation. It is less suitable for the project’s execution-free exact scorer.

### Compatibility versus native placement

Two decisions must not be conflated:

1. **Metric:** Manhattan versus Chebyshev versus Euclidean.
2. **Addressing surface:** legacy scalar addresses versus native `(x,y)` placement.

The metric should be Manhattan in both versions. Keeping scalar addresses in v1 preserves every current score through the exact `lin(a)` embedding. Allowing arbitrary native 2-D placements later still changes the feasible placement set and therefore requires a versioned leaderboard, even though the norm remains Manhattan.

## Required corrections

### A. `docs/grid-vm-competition-design.html`, Section 5

Replace the current distance paragraph with:

> **Distance metric.** Cells occupy a two-dimensional rectilinear grid. The distance between cells `p=(x₁,y₁)` and `q=(x₂,y₂)` is Manhattan distance, `d₁(p,q)=|x₁-x₂|+|y₁-y₂|`. This is the route length in the simplified Dally model: horizontal and vertical wire segments are priced by their total length.
>
> Legacy address `a≥1` has the normative upper-half-plane embedding `lin(a)=(x,y)`: let `k=ceil(sqrt(a))`, `t=a-(k-1)²-1`, `x=-(k-1)+t`, and `y=k-|x|`. Then `y>0` and `d₁((0,0),lin(a))=k`. Each radius-`k` shell contains exactly `2k-1` cells, so this reproduces every existing `ceil(sqrt(a))` score exactly.
>
> For native affine coordinates, Manhattan cost is a sum of absolute affine forms. The judge partitions the iteration domain at sign changes and evaluates the resulting affine lattice sums using the normative floor-sum/Ehrhart evaluator. No square root is applied to an already geometric distance.

If alternating spiral orientation is important for continuity with the visualizer, specify the parity-dependent `x` orientation from `simplified_explicit_communication_model_figure.py`; it is distance-equivalent to the compact formula above.

### B. Decision D3

Preferred v1 wording:

> **D3, Addressing surface:** canonical scalar addresses in v1, interpreted by the exact upper-half-plane Manhattan `lin(a)` embedding. Native 2-D Manhattan placement belongs in a separately versioned spatial division because it changes the feasible placement set and rebaselines records.

If native 2-D placement is wanted immediately, use:

> **D3, Addressing surface:** native 2-D Manhattan coordinates, with the canonical `lin(a)` embedding as an exact legacy subset. Launch a new leaderboard; do not compare native-placement scores directly with existing scalar-address records.

Do not use Chebyshev in either option.

### C. Energy and time equations

Use `d₁(p,q)` directly:

`E_move = Σ_reads bytes(r)·d₁(src(r),dst(r))·(1 fJ/byte-micron)·(1 micron/grid-step)`.

For propagation, replace `1.875 µm/ns` with approximately `1,875 µm/ns` if the intended speed is `c/160`. Keep any 1 ns sequential issue term separate from flight time.

### D. PR #48 text

1. In §8.1, replace “each edge is charged `ceil(sqrt(distance))` once” with:

   > Each edge is charged once by its Manhattan route length, `|Δx|+|Δy|`. For a legacy scalar address `a`, the normative `lin(a)` embedding makes that route length exactly `ceil(sqrt(a))`; no second square root is applied.

2. Replace §8.3 with:

   > ### 8.3 Metric continuity and native spatial placement
   >
   > The physical metric is Manhattan distance throughout. For v1, the normative addressing surface remains scalar: address `a` is interpreted through the upper-half-plane `lin(a)` embedding, whose Manhattan radius is exactly `ceil(sqrt(a))`. This preserves bit-exact scores for the existing sparse-parity, matmul, and Tier-1 calibration corpus.
   >
   > A later native spatial division may expose explicit `(x,y)` placement, but it also uses Manhattan distance. Native placement enlarges the feasible layout space and therefore gets a separately versioned leaderboard. Chebyshev distance is not part of the Dally model.

3. Correct the already-posted PR comment with a short follow-up rather than leaving item 5 as the apparent conclusion.

## Ready-to-post correction for PR #48

> Correction to my earlier item 5 and to §8.3: I accepted “native 2D Chebyshev” as a possible v2 metric without checking it against the source model. That was wrong. Yaroslav’s March 30 and September 4 messages, Dally’s CACM article, and the existing simplified-Dally implementation all specify Manhattan distance.
>
> The shell argument does not require Chebyshev. In the existing upper-half-plane Manhattan layout, radius `c` contains exactly `2c-1` cells, matching the addresses with `ceil(sqrt(a))=c` exactly. Chebyshev’s first-quadrant shell has `2c+1` cells and underprices diagonal rectilinear routes by as much as 2×.
>
> I will use Manhattan distance throughout. V1 keeps scalar addresses through the exact Manhattan `lin(a)` embedding so existing scores remain bit-identical. If we later expose native `(x,y)` placement, it will also use Manhattan distance and will get a versioned leaderboard because the larger placement surface rebaselines records. I will also remove §8.1’s `ceil(sqrt(distance))` wording: the square root maps linear address rank to grid radius; it is not applied to an already geometric edge length.

## Ready-to-send note to Yaroslav

> I traced the Chebyshev reference. It first appears in the Fable-assisted Grid VM report (`4f4d3e2`), not in our Telegram requirements or the earlier Dally docs, and our agents then echoed it into PR #48. The shell rationale is mistaken: the existing upper-half-plane Manhattan shell already has exactly `2c-1` cells and reproduces `ceil(sqrt(a))` exactly. I’m correcting the RFC to Manhattan throughout; native 2D, if we add it, should be versioned Manhattan rather than Chebyshev.

## Action checklist

- [ ] Replace Chebyshev with Manhattan in Grid VM Section 5.
- [ ] Replace D3 with the scalar-v1/native-Manhattan-later decision above.
- [ ] Remove `ceil(sqrt(distance))` from PR #48’s native edge semantics.
- [ ] Replace PR #48 §8.3 with the Manhattan continuity wording.
- [ ] Post the PR correction comment.
- [ ] Correct `c/160` from `1.875 µm/ns` to approximately `1,875 µm/ns`.
- [ ] Add scorer tests for `lin(a)` over shell boundaries and for native diagonal displacement, including `(0,0)→(d,d)` costing `2d`.
- [ ] Keep legacy and native-placement leaderboards distinct even though both use `L1`.

## Reproducibility notes

The forensic checks used read-only queries or local git object inspection:

- Telegram schema: `messages(id, topic_id, date, sender, text, reply_to)`.
- Database-wide term query: `lower(text) like '%chebyshev%'`, returning zero rows through `2026-09-05T01:24:48Z`.
- Git provenance: `git log --all --reverse -S'Chebyshev' -- docs`, returning `4f4d3e2` and then `e9e4d38`.
- Commit contents inspected directly from `4f4d3e2`, independent of the current dirty worktree.
- PR metadata and comments read through `gh pr view 48`.
- DSH archives decompressed read-only with `zstd`; session identity and relevant event sequence numbers are recorded above.

No GitHub comment, branch update, push, or Telegram message was performed as part of this investigation.
