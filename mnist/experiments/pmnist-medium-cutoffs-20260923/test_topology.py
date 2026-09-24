"""Real-data tests for pixel-topology recovery from permuted 9x9 MNIST.

The data here is built straight from ``raw/source`` with
``canonical_data.read_idx`` / ``canonical_data.area_resize`` so the tests run
before ``study.py`` exists.  A known random feature permutation is applied and
``topology.recover_layout`` must put every informative pixel back on its true
lattice cell.
"""

from __future__ import annotations

import functools
from pathlib import Path
import sys
import time
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import canonical_data  # noqa: E402
import topology  # noqa: E402

SOURCE_IMAGES = ROOT / "raw" / "source" / "train-images-idx3-ubyte.gz"
POOL_SIZE = 60000
# A pixel is called informative when its variance clears this bar; the test
# contract requires exact placement for those (>= 95% as documented fallback).
REPORT_VARIANCE = 1e-4
REQUIRED_FRACTION = 0.95


@functools.lru_cache(maxsize=1)
def canonical_pool():
    """All 60,000 official training digits as canonical flat 9x9 features."""
    images = canonical_data.read_idx(SOURCE_IMAGES, POOL_SIZE, images=True)
    values = images.astype(np.float32) / np.float32(255)
    resized = canonical_data.area_resize(values, 9)
    np.clip(resized, 0.0, 1.0, out=resized)
    return np.ascontiguousarray(resized.reshape(POOL_SIZE, 81))


def draw(n, seed):
    """``n`` canonical rows plus a known random feature permutation."""
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed)))
    rows = rng.permutation(POOL_SIZE)[:n]
    permutation = rng.permutation(81).astype(np.int64)
    canonical = canonical_pool()[rows]
    return canonical, canonical[:, permutation], permutation


def grid_of_true_positions(layout, permutation):
    """9x9 render: the true grid index of whichever feature landed in each cell."""
    inverse = np.empty(81, dtype=np.int64)
    inverse[layout] = np.arange(81)
    truth = permutation[inverse]
    return "\n".join(" ".join(f"{v:3d}" for v in truth[r * 9:(r + 1) * 9])
                     for r in range(9))


class RecoverLayoutOnRealData(unittest.TestCase):
    maxDiff = None

    def _check(self, n, seed):
        canonical, permuted, permutation = draw(n, seed)
        started = time.time()
        # Default (orient=False): the layout is recovered up to the eight
        # symmetries of the square, which is all a permutation-invariant
        # statistic can determine and all a CNN needs.  Scoring therefore picks
        # the best dihedral image -- no external prior is consulted.
        raw_layout, details = topology.recover_layout(permuted, return_details=True)
        elapsed = time.time() - started
        self.assertFalse(details["used_external_orientation_prior"])

        self.assertEqual(raw_layout.dtype, np.int64)
        self.assertEqual(raw_layout.shape, (81,))
        self.assertTrue(np.array_equal(np.sort(raw_layout), np.arange(81)),
                        "layout must be a bijection onto 0..80")

        variance = permuted.var(axis=0)
        informative = variance >= REPORT_VARIANCE
        scored = [(float((relabel[raw_layout][informative]
                          == permutation[informative]).mean()), index)
                  for index, relabel in enumerate(topology.DIHEDRAL)]
        best_fraction, best_index = max(scored)
        layout = topology.DIHEDRAL[best_index][raw_layout]
        correct = layout[informative] == permutation[informative]
        fraction = float(correct.mean())
        self.assertEqual(fraction, best_fraction)
        wrong = np.where(informative & (layout != permutation))[0]
        report = ", ".join(
            f"feature {int(j)} (true cell r{int(permutation[j]) // 9}c{int(permutation[j]) % 9}, "
            f"var {variance[j]:.2e}) placed at r{int(layout[j]) // 9}c{int(layout[j]) % 9}"
            for j in wrong)

        print(f"\n[N={n} seed={seed}] recover_layout {elapsed:.1f}s, "
              f"{int(informative.sum())} informative pixels, "
              f"exact fraction (best dihedral image) {fraction:.4f}, "
              f"n_active={details['n_active']}, "
              f"stress={details['embedding_stress']:.3f}, "
              f"qap={details['qap_objective']:.3f}, "
              f"winning start={details['winning_start_index']}, "
              f"truncated={details['search_truncated']}")
        print("true grid index occupying each recovered cell "
              "(0..80 in reading order means perfect recovery):")
        print(grid_of_true_positions(layout, permutation))
        if wrong.size:
            print("misplaced informative pixels:", report)

        self.assertGreaterEqual(
            fraction, REQUIRED_FRACTION,
            f"N={n}: only {fraction:.4f} of informative pixels recovered exactly; {report}")
        self.assertEqual(
            fraction, 1.0,
            f"N={n}: exact-recovery fraction {fraction:.4f} < 1.0; {report}")

        # unpermute must reconstruct the canonical image on every informative pixel
        images = topology.unpermute(permuted, layout)
        self.assertEqual(images.shape, (n, 1, 9, 9))
        self.assertEqual(images.dtype, np.float32)
        flat = images.reshape(n, 81)
        active_cells = permutation[informative]
        np.testing.assert_array_equal(flat[:, active_cells],
                                      canonical[:, active_cells])
        return fraction, elapsed

    def test_recovery_10000(self):
        self._check(10000, 2026092301)

    def test_recovery_1000(self):
        self._check(1000, 2026092301)


class LayoutHelpers(unittest.TestCase):
    def test_unpermute_round_trip_is_exact(self):
        canonical, permuted, permutation = draw(64, 2026092399)
        images = topology.unpermute(permuted, permutation)
        np.testing.assert_array_equal(images.reshape(64, 81), canonical)

    def test_unpermute_rejects_non_bijection(self):
        with self.assertRaises(ValueError):
            topology.unpermute(np.zeros((3, 81), dtype=np.float32), np.zeros(81, dtype=np.int64))

    def test_dihedral_relabelings_are_a_group_of_eight(self):
        maps = topology.dihedral_relabelings()
        self.assertEqual(len(maps), 8)
        seen = {tuple(int(v) for v in m) for m in maps}
        self.assertEqual(len(seen), 8)
        for relabel in maps:
            self.assertTrue(np.array_equal(np.sort(relabel), np.arange(81)))

    def test_diagnostics_reports_expected_keys(self):
        _, permuted, permutation = draw(1000, 2026092398)
        report = topology.diagnostics(permuted, permutation)
        for key in ("n_active", "assignment_cost", "embedding_stress",
                    "qap_objective", "is_bijection", "orientation_scores"):
            self.assertIn(key, report)
        self.assertTrue(report["is_bijection"])
        self.assertEqual(len(report["orientation_scores"]), 8)

    def test_recover_layout_is_deterministic(self):
        _, permuted, _ = draw(1000, 2026092397)
        first = topology.recover_layout(permuted)
        second = topology.recover_layout(permuted)
        np.testing.assert_array_equal(first, second)

    def test_recover_layout_rejects_bad_shapes(self):
        with self.assertRaises(ValueError):
            topology.recover_layout(np.zeros((10, 80), dtype=np.float32))

    def test_orientation_prior_is_opt_in_and_declared(self):
        """The MNIST mean/std prior is pool-wide data: never used by default."""
        _, permuted, permutation = draw(1000, 2026092301)
        plain, details = topology.recover_layout(permuted, return_details=True)
        self.assertFalse(details["used_external_orientation_prior"])
        self.assertIsNone(details["orientation_index"])
        oriented, oriented_details = topology.recover_layout(
            permuted, orient=True, return_details=True)
        self.assertTrue(oriented_details["used_external_orientation_prior"])
        self.assertIsNotNone(oriented_details["orientation_index"])
        relabel = topology.DIHEDRAL[oriented_details["orientation_index"]]
        np.testing.assert_array_equal(oriented, relabel[plain])
        informative = permuted.var(axis=0) >= REPORT_VARIANCE
        np.testing.assert_array_equal(oriented[informative], permutation[informative])

    def test_details_describe_the_winning_start_and_truncation(self):
        _, permuted, _ = draw(1000, 2026092397)
        layout, details = topology.recover_layout(permuted, return_details=True)
        winner = details["winning_start"]
        self.assertEqual(details["n_graph_edges"], winner["n_graph_edges"])
        self.assertEqual(details["graph_components"], winner["graph_components"])
        self.assertEqual(details["embedding_stress"], winner["embedding_stress"])
        self.assertEqual(details["assignment_cost"], winner["assignment_cost"])
        self.assertLessEqual(details["assignment_cost_min"], details["assignment_cost"])
        self.assertFalse(details["search_truncated"])
        self.assertEqual(details["ils_iterations_completed"],
                         details["ils_iterations_planned"])
        # A clock-starved search must say so rather than quietly returning a
        # different (load-dependent) layout with no evidence in the record.
        _, starved = topology.recover_layout(permuted, max_seconds=0.0,
                                             return_details=True)
        self.assertTrue(starved["search_truncated"])
        self.assertEqual(starved["anneals_completed"], 0)
        self.assertEqual(starved["ils_iterations_completed"], 0)


def _without_scipy(function):
    """Run ``function(module)`` with a freshly imported, scipy-less topology."""
    import builtins
    import importlib

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "scipy" or name.startswith("scipy."):
            raise ImportError("scipy is not installed in the Modal image")
        return real_import(name, *args, **kwargs)

    saved = {name: module for name, module in sys.modules.items()
             if name == "topology" or name == "scipy" or name.startswith("scipy.")}
    for name in list(sys.modules):
        if name == "topology" or name == "scipy" or name.startswith("scipy."):
            del sys.modules[name]
    builtins.__import__ = blocked
    try:
        module = importlib.import_module("topology")
        return function(module)
    finally:
        builtins.__import__ = real_import
        for name in list(sys.modules):
            if name == "topology" or name == "scipy" or name.startswith("scipy."):
                del sys.modules[name]
        sys.modules.update(saved)


class WorksWithoutScipy(unittest.TestCase):
    """The Modal image ships torch + numpy only -- topology must not need scipy."""

    def test_primitives_match_scipy(self):
        from scipy.optimize import linear_sum_assignment
        from scipy.sparse.csgraph import connected_components, shortest_path
        rng = np.random.default_rng(0)
        for _ in range(25):
            rows, extra = int(rng.integers(1, 12)), int(rng.integers(0, 4))
            cost = rng.normal(size=(rows, rows + extra))
            for matrix in (cost, cost.T):
                expected_r, expected_c = linear_sum_assignment(matrix)
                got_r, got_c = topology._assignment_numpy(matrix)
                np.testing.assert_array_equal(got_r, expected_r)
                self.assertAlmostEqual(float(matrix[got_r, got_c].sum()),
                                       float(matrix[expected_r, expected_c].sum()), 9)
            size = 20
            graph = (rng.random((size, size)) < 0.08).astype(float)
            graph = np.maximum(graph, graph.T)
            np.fill_diagonal(graph, 0.0)
            expected = shortest_path(graph, directed=False, unweighted=True)
            np.testing.assert_array_equal(topology._bfs_layers(graph), expected)
            self.assertEqual(topology._component_count(graph),
                             int(connected_components(graph, directed=False)[0]))

    def test_import_and_recovery_without_scipy(self):
        _, permuted, permutation = draw(1000, 2026092397)
        expected = topology.recover_layout(permuted)

        def run(module):
            self.assertFalse(module.HAVE_SCIPY)
            started = time.time()
            layout, details = module.recover_layout(permuted, return_details=True)
            print(f"\n[no scipy] recover_layout {time.time() - started:.1f}s "
                  f"(scipy_available={details['scipy_available']})")
            return layout

        np.testing.assert_array_equal(_without_scipy(run), expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
