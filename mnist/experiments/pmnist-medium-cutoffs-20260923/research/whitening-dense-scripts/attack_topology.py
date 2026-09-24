"""Run the existing second-order pixel-topology attack against each variant.

For every variant we report
  * whether topology.recover_layout places features on lattice cells that agree
    with where those features actually live in pixel space, and
  * the end-to-end consequence: a quick cnn-09 fit on the unpermuted images.

"Where a feature actually lives" is defined through the released map z = A(x-mu):
coordinate j of z is *synthesised* back into pixel space by column j of A_inv, so
its peak pixel is argmax_p A_inv[p, j]^2.  For the permuted-pixel protocol each
column of A_inv is a single pixel and the peak pixel is exact ground truth; after
a random rotation the columns are delocalised and the peak pixel is only the
best single-pixel summary of a spread-out filter.

Metric: of the 144 feature pairs that recover_layout puts on adjacent lattice
cells, what fraction have adjacent peak pixels?  Chance is 144/3240 = 4.4%.

usage: python attack_topology.py <variant> <n> [--cnn]
"""
import json
import os
import sys
import time

import numpy as np

ROOT = "/Users/yaroslavvb/git/sutro-problems/mnist/experiments/pmnist-medium-cutoffs-20260923"
sys.path.insert(0, ROOT)

import study      # noqa: E402
import topology   # noqa: E402
import whiten     # noqa: E402

sys.path.insert(0, "/tmp/whiten")
from run_dense import VARIANTS, arrays_for, query_labels, SEED  # noqa: E402

OUT = "/tmp/whiten/results"
os.makedirs(OUT, exist_ok=True)
GRID = 9
ADJ = topology.ADJACENCY


def peak_pixels(variant, transform):
    if variant == "permuted":
        # feature j is pixel perm[j] exactly
        return np.asarray(study.feature_permutation()), np.ones(81)
    report = whiten.filter_peaks(transform)
    return report["peak_pixel"], report["peak_energy_fraction"]


def layout_report(layout, pixel_of):
    """Agreement between the recovered lattice and the features' pixel locations."""
    layout = np.asarray(layout, dtype=np.int64)
    n = layout.shape[0]
    # adjacency implied by the recovered layout
    recovered_adjacent = ADJ[np.ix_(layout, layout)] > 0
    rows, cols = pixel_of // GRID, pixel_of % GRID
    d2 = (rows[:, None] - rows[None, :]) ** 2 + (cols[:, None] - cols[None, :]) ** 2
    true_adjacent = (d2 == 1)
    upper = np.triu(np.ones((n, n), dtype=bool), 1)
    pairs = int((recovered_adjacent & upper).sum())
    hits = int((recovered_adjacent & true_adjacent & upper).sum())
    chance = float(((true_adjacent & upper).sum()) / upper.sum())
    # best exact placement over the 8 dihedral images (only meaningful when the
    # features really are pixels)
    best = 0.0
    for relabel in topology.DIHEDRAL:
        best = max(best, float((relabel[layout] == pixel_of).mean()))
    return {
        "recovered_adjacent_pairs": pairs,
        "adjacent_pairs_with_adjacent_peak_pixels": hits,
        "adjacency_precision": hits / max(pairs, 1),
        "chance_adjacency_precision": chance,
        "exact_placement_fraction_best_dihedral": best,
        "distinct_peak_pixels": int(np.unique(pixel_of).size),
    }


def qap_only_attack(train_x, seconds=60.0, n_random_starts=8, seed=0):
    """The QAP half of the attack, run directly on the released features.

    ``recover_layout``'s front end (variance screen -> sqrt(clip(x)) -> mutual-kNN
    -> geodesics -> MDS) assumes non-negative, spatially smooth pixels and simply
    crashes on whitened data: the mutual-kNN graph of a near-identity partial
    correlation matrix is disconnected, the hop distances are infinite and the
    assignment sees NaNs.  So that the failure is measured rather than merely
    observed, this runs the *decisive* stage -- maximise sum_ij S_ij adj(pi_i,pi_j)
    over layouts, the step that on real pixels reaches the true layout's objective
    exactly -- on the same similarity matrix the pipeline would have used, from
    random starts.
    """
    similarity = np.clip(topology.partial_correlation(np.asarray(train_x, np.float64)),
                         0.0, None)
    np.fill_diagonal(similarity, 0.0)
    rng = np.random.default_rng(seed)
    starts = [np.arange(81, dtype=np.int64)]
    starts += [rng.permutation(81).astype(np.int64) for _ in range(n_random_starts)]
    layout, value, search = topology._search(similarity, starts, time.time(), seconds)
    return np.asarray(layout, dtype=np.int64), float(value), search


def main():
    variant, n = sys.argv[1], int(sys.argv[2])
    want_cnn = "--cnn" in sys.argv
    arrays, transform = arrays_for(variant, n)
    if arrays["train_x"].shape[1] != 81:
        raise SystemExit("the topology attack needs 81 features")
    pixel_of, peak_fraction = peak_pixels(variant, transform)

    started = time.time()
    failure = None
    try:
        layout, details = topology.recover_layout(
            np.ascontiguousarray(arrays["train_x"]), max_seconds=120.0,
            return_details=True)
        details = {k: (float(v) if isinstance(v, (int, float, np.floating)) else str(v))
                   for k, v in details.items() if not isinstance(v, dict)}
    except Exception as error:                        # the attack does not survive
        failure = f"{type(error).__name__}: {error}"
        details = {}
        layout = None
        print("%-12s n=%-6d recover_layout FAILED: %s" % (variant, n, failure),
              flush=True)
    layout_seconds = time.time() - started

    qap_layout, qap_value, qap_search = qap_only_attack(arrays["train_x"])
    if layout is None:
        layout = qap_layout          # so the downstream CNN still has something
    record = {
        "variant": variant, "n": int(n), "seed": SEED,
        "layout_seconds": layout_seconds,
        "recover_layout_failed": failure,
        "recover_details": details,
        "peak_energy_fraction_mean": float(np.mean(peak_fraction)),
        "peak_energy_fraction_median": float(np.median(peak_fraction)),
        "qap_only": {"objective": qap_value,
                     **{"report_" + k: v
                        for k, v in layout_report(qap_layout,
                                                  np.asarray(pixel_of)).items()}},
        **layout_report(layout, np.asarray(pixel_of)),
    }
    if transform is not None:
        record["transform"] = {k: transform[k] for k in
                               ("epsilon", "method", "rotation_seed", "n_released",
                                "condition_number")}
    print("%-12s n=%-6d adjacency precision %5.1f%%  (chance %.1f%%)  "
          "exact placement %5.1f%%  peak-energy %.3f"
          % (variant, n, 100 * record["adjacency_precision"],
             100 * record["chance_adjacency_precision"],
             100 * record["exact_placement_fraction_best_dihedral"],
             record["peak_energy_fraction_mean"]), flush=True)

    if want_cnn:
        import learners
        if failure is not None:
            # learners._fit_topo_cnn calls recover_layout itself and therefore
            # dies with the same error.  Hand the attacker the best layout its
            # own QAP stage could find and run the frozen cnn-09 recipe on it,
            # so the end-to-end number is an upper bound on what the attack buys.
            topology.recover_layout = (                      # noqa: E731
                lambda train_x, orient=False, max_seconds=120.0,
                return_details=False, _layout=layout:
                (_layout, {"substituted": True}) if return_details else _layout)
            record["cnn_layout_source"] = "qap_only (recover_layout crashed)"
        else:
            record["cnn_layout_source"] = "recover_layout"
        config = {"family": "topo_cnn", "member_seeds": [101], "epochs": 25}
        started = time.time()
        out = learners.fit_predict(arrays["train_x"], arrays["train_y"],
                                   arrays["query_x"], config, seed=11,
                                   deadline_unix=time.time() + 3600.0, device="cpu")
        truth = query_labels()
        record["cnn_query_error_pct"] = 100.0 * float((out["labels"] != truth).mean())
        record["cnn_wall_seconds"] = time.time() - started
        record["cnn_config"] = config
        print("   topo_cnn(1 member, 25 epochs) error %.2f%%  (%.0fs)"
              % (record["cnn_query_error_pct"], record["cnn_wall_seconds"]), flush=True)

    path = os.path.join(OUT, f"topoattack-{variant}-n{n}.json")
    json.dump(record, open(path, "w"), indent=2, sort_keys=True, default=str)
    print("   -> " + path, flush=True)


if __name__ == "__main__":
    main()
