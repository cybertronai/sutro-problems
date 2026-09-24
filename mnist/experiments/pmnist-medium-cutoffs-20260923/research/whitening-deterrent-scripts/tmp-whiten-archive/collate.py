"""Collect every measurement into ROOT/research/whitening-dense-results.json + .md."""
import glob
import json
import os
import platform
import sys
import time

import numpy as np

ROOT = "/Users/yaroslavvb/git/sutro-problems/mnist/experiments/pmnist-medium-cutoffs-20260923"
sys.path.insert(0, ROOT)
RESEARCH = os.path.join(ROOT, "research")
os.makedirs(RESEARCH, exist_ok=True)

VARIANT_LABEL = {
    "permuted": "permuted pixels (current protocol)",
    "w2": "whitened+rotated, ZCA eps=1e-2 (81 feat)",
    "w4": "whitened+rotated, ZCA eps=1e-4 (81 feat)",
    "wexact64": "whitened+rotated, exact, top-64 components",
    "zca_norot": "ZCA whitened eps=1e-2, permuted, NO rotation",
    "rot_only": "random rotation only, no whitening",
}
VARIANT_ORDER = ["permuted", "rot_only", "w2", "w4", "wexact64"]
ATTACK_ORDER = VARIANT_ORDER + ["zca_norot"]
METHOD_ORDER = ["kernel_ridge:arccos1", "kernel_ridge:rbf", "mlp-standardize"]
METHOD_LABEL = {
    "kernel_ridge:arccos1": "kernel ridge, arc-cosine depth 3",
    "kernel_ridge:rbf": "kernel ridge, RBF",
    "mlp-standardize": "MLP 1024-1024, standardized",
}


def load(pattern):
    out = []
    for path in sorted(glob.glob(pattern)):
        out.append(json.load(open(path)))
    return out


def main():
    dense = load("/tmp/whiten/results/classical-*.json") + \
        load("/tmp/whiten/results/mlp-*.json")
    attacks = load("/tmp/whiten/results/topoattack-*.json")
    ica = load("/tmp/whiten/results/ica-*.json")
    eps_cv = json.load(open("/tmp/whiten/eps_cv.json"))

    table = {}
    for row in dense:
        table[(row["method"], row["variant"], row["n"])] = row

    lines = []
    lines.append("# Whitening / rotation vs the permutation: dense-method cost "
                 "and attack resistance\n")
    lines.append("Study `pmnist-medium-cutoffs-20260923`. All numbers below are "
                 "**single-draw, single-seed CPU pilots** on dev seed 2026092301 "
                 "with learner seed 11: one training draw, one query set of 10,000 "
                 "rows, no repetitions and no error bars. The binomial standard "
                 "error on a 10,000-row query set at 3% error is 0.17 pp, so treat "
                 "gaps under ~0.4 pp (N=10,000) and ~1.0 pp (N=1,000) as noise. "
                 "The eleven FINAL seeds 2026092001..2026092011 were never touched; "
                 "no file belonging to the live study was modified.\n")
    lines.append("Variants (all share the study's draws exactly: "
                 "`order = PCG64(SeedSequence(seed).spawn(2)[0]).permutation(60000)`, "
                 "train = order[:N], query = order[10000:20000]):\n")
    lines.append("- **permuted pixels** - the current protocol, `x[:, perm]`.")
    lines.append("- **random rotation only** - `z = P Q (x - mean)`, `Q` Haar "
                 "orthogonal (QR of a PCG64 Gaussian, seed 20260923). No whitening.")
    lines.append("- **whitened+rotated, eps** - `z = P Q U diag(1/sqrt(lambda+eps)) "
                 "U^T (x - mean)`, pool covariance. eps is in variance units of "
                 "[0,1] pixels; eps=1e-2 is the value chosen by the train-only CV "
                 "sweep at the bottom of this file.")
    lines.append("- **exact, top-64** - drop the 17 near-null eigen-directions and "
                 "whiten the rest exactly; releases 64 features, so its covariance "
                 "is exactly the identity and there is no low-variance tail to leak."
                 " The `mlp` family hard-codes 81 inputs, so it cannot run on this "
                 "variant without a protocol change (marked `-`).\n")
    lines.append("## Dense methods: query error (%) on dev seed 2026092301, "
                 "single draw, learner seed 11\n")
    header = "| method | N | " + " | ".join(VARIANT_LABEL[v] for v in VARIANT_ORDER) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (2 + len(VARIANT_ORDER)))
    for method in METHOD_ORDER:
        for n in (1000, 10000):
            cells = []
            for variant in VARIANT_ORDER:
                row = table.get((method, variant, n))
                cells.append("%.2f" % row["query_error_pct"] if row else "-")
            lines.append("| %s | %d | %s |" % (METHOD_LABEL[method], n, " | ".join(cells)))

    lines.append("\n## Deltas vs the permuted-pixel protocol (percentage points; "
                 "positive = the preprocessing hurts the dense learner)\n")
    lines.append(header)
    lines.append("|" + "---|" * (2 + len(VARIANT_ORDER)))
    for method in METHOD_ORDER:
        for n in (1000, 10000):
            base = table.get((method, "permuted", n))
            cells = []
            for variant in VARIANT_ORDER:
                row = table.get((method, variant, n))
                if row is None or base is None:
                    cells.append("-")
                elif variant == "permuted":
                    cells.append("0 (ref)")
                else:
                    cells.append("%+.2f" % (row["query_error_pct"]
                                            - base["query_error_pct"]))
            lines.append("| %s | %d | %s |" % (METHOD_LABEL[method], n, " | ".join(cells)))

    lines.append("\n## Pixel-topology attack (second order), dev seed 2026092301\n")
    lines.append("| variant | N | recover_layout | adjacency precision | chance | "
                 "exact placement | cnn-09 (1 member, 25 epochs) error |")
    lines.append("|---|---|---|---|---|---|---|")
    for row in sorted(attacks, key=lambda r: (r["n"], ATTACK_ORDER.index(r["variant"]))):
        lines.append("| %s | %d | %s | %.1f%% | %.1f%% | %.1f%% | %s |" % (
            VARIANT_LABEL[row["variant"]], row["n"],
            "ok" if row.get("recover_layout_failed") is None else "CRASHED",
            100 * row["adjacency_precision"],
            100 * row["chance_adjacency_precision"],
            100 * row["exact_placement_fraction_best_dihedral"],
            ("%.2f%%" % row["cnn_query_error_pct"]) if "cnn_query_error_pct" in row
            else "-"))

    if ica:
        lines.append("\n## ICA attack on the whitened variant "
                     "(FastICA, 20,000 unlabelled rows)\n")
        lines.append("| transform | eps | components | contrast | mean peak-energy of the "
                     "composite map | random-rotation control (null) | ZCA control "
                     "(ceiling) | matched energy on active pixels | pixels matched "
                     ">0.9 | recover_layout on the sources |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for row in ica:
            lines.append("| %s | %g | %d | %s | %.3f | %.3f | %s | %.3f | %d/%d | %s |" % (
                row["variant"].get("method", "zca"), row["variant"]["epsilon"],
                row["ica_components"], row["ica_fun"],
                row["mean_peak_energy_fraction"],
                row.get("control_random_rotation", float("nan")),
                ("%.3f" % row["control_zca_whitening"])
                if "control_zca_whitening" in row else "-",
                row.get("matching_matched_energy_mean", float("nan")),
                row.get("matching_matched_above_0.9", -1),
                row.get("matching_n_matched", -1),
                "ok" if row.get("recover_layout_failed") is None else "CRASHED"))
        lines.append("\nWhy `recover_layout` cannot be chained after ICA: every "
                     "FastICA output is exactly white (`Cov(S) = I`) and correlation "
                     "is invariant to per-coordinate affine rescaling, so no amount "
                     "of post-hoc rescaling of the sources can put second-order "
                     "neighbour structure back. The second-order attack is "
                     "structurally unavailable downstream of ICA; what ICA can give "
                     "an attacker is the pixel *basis*, measured by the matched-energy "
                     "column. `matched energy` is a Hungarian one-to-one matching of "
                     "sources to the 64 active pixels on the row-normalised energy of "
                     "the composite map `W_ica A`; 1.0 means every source is exactly "
                     "one pixel. The ZCA control is the ceiling for any white basis "
                     "(no rotation of whitened data can reproduce raw pixels, because "
                     "raw pixels are correlated).")
        lines.append("\nRead of the ICA numbers: off-the-shelf FastICA on 20,000 "
                     "unlabelled rows recovers a real but partial piece of the pixel "
                     "basis -- matched energy 0.39-0.41 against a 0.086-0.108 null "
                     "and a 0.84-0.94 ceiling, with 15-17 of the 64 active pixels "
                     "isolated above 0.9 in about 25 s. It is not a working attack "
                     "(the recovered basis is too noisy to hand the lattice search a "
                     "usable input, and the 81-component run collapses onto the 17 "
                     "near-null quantisation directions, matched energy 0.037), but "
                     "it is not nothing either. The reason plain ICA stalls is that "
                     "its model is wrong for this data: ICA insists the sources are "
                     "independent, pixels are strongly dependent, and no white basis "
                     "can be the pixel basis. The method designed for exactly that "
                     "mismatch -- topographic ICA (Hyvarinen, Hoyer & Inki 2001), "
                     "which models dependence between *neighbouring* components and "
                     "recovers a 2-D topography from natural image patches -- is the "
                     "attack this defence should be assumed to face. It was not run "
                     "here. Non-negative ICA (Plumbley) is a second untested route: "
                     "the pixels are non-negative and 44% zero on the median active "
                     "pixel, so the released cloud is a linear image of a shifted "
                     "orthant whose 81 extreme rays are the pixel directions.")

    lines.append("\n## Train-only 5-fold CV used to choose epsilon "
                 "(arc-cosine depth 3, N=1000)\n")
    lines.append("| variant | CV error (%) | chosen lambda |")
    lines.append("|---|---|---|")
    for row in eps_cv["rows"]:
        lines.append("| %s | %.2f | %g |" % (row["variant"], row["cv_error_pct"],
                                             row["chosen_lambda"]))

    lines.append("\nScripts: `research/whitening-dense-scripts/`. The attack side "
                 "of the same question (topographic ICA and friends) was worked in "
                 "parallel by a second agent and lands in "
                 "`research/whitening-attack-results.*`.")
    lines.append("\n## Which ingredient does the work\n")
    lines.append("- **Whitening alone is not a defence.** ZCA whitening filters are "
                 "localised centre-surround, so after whitening-without-rotation each "
                 "coordinate still essentially *is* its pixel: the attack keeps 44.4% "
                 "adjacency precision against a 4.4% chance rate.")
    lines.append("- **The rotation alone is a full defence against this attack, and "
                 "it is free.** Rotation without whitening puts the attack at chance "
                 "(3.5% vs 4.4%) while costing the RBF kernel exactly nothing (a "
                 "rotation preserves pairwise distances) and the arc-cosine kernel "
                 "+0.23 pp at N=1,000 / +0.14 pp at N=10,000, both inside the "
                 "single-draw noise. It *helps* the standardized MLP (-2.01 pp at "
                 "N=1,000, -0.43 pp at N=10,000): a rotation spreads each pixel's "
                 "signal over all 81 coordinates, which suits per-feature "
                 "standardisation, Gaussian input noise and dropout better than the "
                 "raw sparse pixels do. That is a change to the task, not a free "
                 "lunch -- it shifts which dense recipe wins -- and one draw is not "
                 "enough to size it.")
    lines.append("- **Whitening on top of the rotation buys no extra security.** An "
                 "attacker can whiten the released data himself: `Q x` and `Q W x` "
                 "differ by a linear map he can estimate from the sample covariance, "
                 "so both leave him with exactly the same residual problem (recover "
                 "an unknown orthogonal factor from higher-order statistics). What "
                 "whitening does change is what the *learner* sees, and there it is "
                 "a straight cost: +0.9 to +1.2 pp for the kernels at eps=1e-2 and "
                 "+1.3 to +4.1 pp at eps=1e-4 or with exact whitening.")
    lines.append("- **The variance floor is a security/accuracy dial with the wrong "
                 "shape.** Large eps keeps accuracy but leaves the released "
                 "covariance far from the identity (at eps=1e-2 only the top handful "
                 "of directions are whitened at all, so the low-variance tail is "
                 "still second-order identifiable); small eps whitens properly but "
                 "amplifies ~17 directions of area-resize quantisation noise to unit "
                 "variance and costs 3-4 pp at N=1,000. Since the rotation already "
                 "does the security work, there is no reason to pay this.\n")
    lines.append("\n## Notes on the method configurations\n")
    lines.append("- Kernel ridge uses the study's `plans/classical_candidates.json` "
                 "grids: lambda in {1e-7,1e-6,1e-5,1e-4,1e-3}, `cv_subsample=4000`, "
                 "arc-cosine depth 3.")
    lines.append("- `classical.fit_predict` applies the study-wide `4x-0.5` map to "
                 "kernel inputs and it cannot be disabled. For the arc-cosine kernel "
                 "this is harmless: the kernel is homogeneous and the gram matrix is "
                 "renormalised by its mean diagonal, so only the -0.5 shift matters, "
                 "and on whitened features (per-feature sd 1, scaled to 4) that shift "
                 "is negligible. Measured directly in the CV table below: "
                 "pre-descaling the inputs so that `4x-0.5` reproduces z exactly "
                 "changes the CV error by 0.1-0.2 pp at every eps.")
    lines.append("- The RBF gamma grid IS in `4x-0.5` pixel units and does not "
                 "transfer: whitened features have ~27x larger mean squared pairwise "
                 "distance, so the study's gammas would give an all-zero kernel. The "
                 "grid used on every non-pixel variant is the study grid PLUS a "
                 "scale-matched copy (study gamma times the ratio of mean squared "
                 "pairwise distances); the existing train-only CV chooses between "
                 "them and always picked the scale-matched values.")
    lines.append("- MLP: widths [1024,1024], dropout 0.3, input noise 0.3, lr 0.002, "
                 "weight decay 0.01, batch 128, warmup 0.1, 1 member, "
                 "`normalization='standardize'`. 300 epochs at N=1,000. At N=10,000 "
                 "the requested 150 epochs needs ~24 min per fit on this CPU, so "
                 "**60 epochs** were used instead (~7 min); the N=10,000 MLP numbers "
                 "are therefore under-trained in absolute terms but identical across "
                 "variants, which is what the comparison needs.")
    lines.append("- Topology attack: `topology.recover_layout` verbatim. On every "
                 "rotated variant it raises `ValueError: matrix contains invalid "
                 "numeric entries` - the mutual-kNN graph of a near-identity partial "
                 "correlation matrix is disconnected, so the hop distances are "
                 "infinite. The reported adjacency precision then comes from the "
                 "decisive stage run on its own (`topology._search`, the QAP that on "
                 "real pixels reaches the true layout exactly) from 9 starts, and the "
                 "cnn-09 fit is given that layout. Both choices favour the attacker.")
    lines.append("- `adjacency precision` = of the feature pairs the recovered layout "
                 "puts on adjacent lattice cells, the fraction whose pixel-space "
                 "synthesis filters (columns of `A_inv`) peak on adjacent pixels. "
                 "`chance` is the density of adjacent pairs among the peak-pixel "
                 "assignment actually realised, so it is above 4.4% when the "
                 "delocalised filters share peak pixels.\n")
    markdown = "\n".join(lines) + "\n"
    payload = {
        "study": "pmnist-medium-cutoffs-20260923",
        "artifact": "research/whitening-dense-results.json",
        "written_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "dataset_seed": 2026092301,
        "seed_class": "dev (final seeds 2026092001..11 were never touched)",
        "pilot_disclosure": (
            "Every number here is a SINGLE draw on ONE dev seed with ONE learner "
            "seed (11). No repetitions, no error bars. Differences below roughly "
            "0.4 pp at N=10000 and 1.0 pp at N=1000 are inside the binomial noise "
            "of a 10,000-row query set and should not be read as real."),
        "variant_definitions": VARIANT_LABEL,
        "dense_results": dense,
        "topology_attack": attacks,
        "ica_attack": ica,
        "epsilon_selection": eps_cv,
        "markdown": markdown,
    }
    json.dump(payload, open(os.path.join(RESEARCH, "whitening-dense-results.json"), "w"),
              indent=2, sort_keys=True, default=str)
    open(os.path.join(RESEARCH, "whitening-dense-results.md"), "w").write(markdown)
    print(markdown)


if __name__ == "__main__":
    main()
