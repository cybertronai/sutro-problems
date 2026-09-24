"""Unit tests for whiten.py (the whitened + randomly-rotated task variant).

Run:  /tmp/pmnist-env/bin/python -m unittest test_whiten -v

Nothing here reads query labels; the only dataset seed used is the dev seed
2026092301 and only its *indices* are touched.
"""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import study      # noqa: E402
import whiten     # noqa: E402

DEV_SEED = 2026092301
EPSILON = 1e-3

_POOL = None
_TRANSFORM = None


def pool():
    global _POOL
    if _POOL is None:
        _POOL = np.asarray(study.pool_images())
    return _POOL


def transform():
    global _TRANSFORM
    if _TRANSFORM is None:
        _TRANSFORM = whiten.fit_transform(pool(), epsilon=EPSILON,
                                          rotation_seed=20260923, method="zca")
    return _TRANSFORM


class Algebra(unittest.TestCase):
    def test_rotation_is_orthogonal(self):
        q = whiten.random_rotation(81, 20260923)
        self.assertLess(np.abs(q @ q.T - np.eye(81)).max(), 1e-12)
        self.assertAlmostEqual(abs(np.linalg.det(q)), 1.0, places=10)

    def test_transform_is_invertible(self):
        t = transform()
        self.assertLess(np.abs(t["A_inv"] @ t["A"] - np.eye(81)).max(), 1e-8)
        self.assertLess(np.abs(t["A"] @ t["A_inv"] - np.eye(81)).max(), 1e-8)

    def test_apply_invert_roundtrip(self):
        sample = pool()[:512]
        recovered = whiten.invert(whiten.apply(sample, transform()), transform())
        self.assertLess(np.abs(recovered - sample).max(), 1e-4)  # float32 release

    def test_whitened_pool_covariance_is_identity_up_to_epsilon(self):
        """With eps>0 the released covariance is Q diag(l/(l+eps)) Q^T, not I.

        The eigenvalues of the released covariance are exactly l/(l+eps) and are
        rotation-invariant; the deviation from I is entirely the shrunk tail.
        """
        t = transform()
        z = whiten.apply(pool(), t).astype(np.float64)
        covariance = np.cov(z, rowvar=False)
        spectrum = np.linalg.eigvalsh(covariance)
        expected = np.sort(t["eigenvalues"] / (t["eigenvalues"] + EPSILON))
        self.assertLess(np.abs(spectrum - expected).max(), 1e-3)
        off = np.abs(covariance - np.diag(np.diag(covariance))).max()
        shrunk = int((expected < 0.99).sum())
        print("\n  eps=%g: released covariance spectrum in [%.4f, %.4f]; "
              "%d/81 directions shrunk below 0.99; max |offdiag| = %.3f"
              % (EPSILON, spectrum.min(), spectrum.max(), shrunk, off))
        # the strong directions really are whitened
        strong = t["eigenvalues"] > 100 * EPSILON
        achieved = t["eigenvalues"] / (t["eigenvalues"] + EPSILON)
        self.assertGreater(achieved[strong].min(), 0.99)
        self.assertGreater(np.diag(covariance).mean(), 0.5)

    def test_exact_whitening_gives_identity_covariance(self):
        """eps=0 on the retained subspace: covariance is the identity to 1e-6."""
        t = whiten.fit_transform(pool(), epsilon=0.0, rotation_seed=20260923,
                                 n_components=64)
        z = whiten.apply(pool(), t).astype(np.float64)
        covariance = np.cov(z, rowvar=False)
        deviation = np.abs(covariance - np.eye(64)).max()
        print("\n  eps=0, k=64: max |Cov - I| = %.2e" % deviation)
        self.assertLess(deviation, 1e-6)

    def test_reduced_rank_right_inverse(self):
        t = whiten.fit_transform(pool(), epsilon=0.0, rotation_seed=1, n_components=64)
        self.assertEqual(t["A"].shape, (64, 81))
        self.assertEqual(t["A_inv"].shape, (81, 64))
        self.assertLess(np.abs(t["A"] @ t["A_inv"] - np.eye(64)).max(), 1e-8)

    def test_determinism(self):
        first = whiten.fit_transform(pool(), epsilon=EPSILON, rotation_seed=7)
        second = whiten.fit_transform(pool(), epsilon=EPSILON, rotation_seed=7)
        self.assertEqual(first["A_sha256"], second["A_sha256"])
        self.assertTrue(np.array_equal(first["A"], second["A"]))
        other = whiten.fit_transform(pool(), epsilon=EPSILON, rotation_seed=8)
        self.assertNotEqual(other["A_sha256"], first["A_sha256"])

    def test_eigenvalue_bookkeeping(self):
        t = transform()
        self.assertEqual(t["eigenvalues"].shape, (81,))
        self.assertTrue(np.all(np.diff(t["eigenvalues"]) >= -1e-15))   # ascending
        self.assertGreaterEqual(t["eigenvalues_below"][1e-4], 8)       # black frame
        self.assertGreater(t["condition_number"], 1.0)
        print("\n  eigenvalues in [%.3e, %.3e]; %d below 1e-4; cond(A)=%.1f"
              % (t["eigenvalue_min"], t["eigenvalue_max"],
                 t["eigenvalues_below"][1e-4], t["condition_number_A"]))


class Draws(unittest.TestCase):
    def test_job_arrays_mirrors_study_draws(self):
        reference = study.job_arrays(DEV_SEED, 1000)
        arrays = whiten.job_arrays(DEV_SEED, 1000, transform())
        self.assertEqual(arrays["train_x"].shape, (1000, 81))
        self.assertEqual(arrays["query_x"].shape, (10000, 81))
        self.assertEqual(arrays["train_x"].dtype, np.float32)
        self.assertTrue(np.array_equal(arrays["train_y"], reference["train_y"]))
        # never any query labels
        self.assertEqual(set(arrays), {"train_x", "train_y", "query_x"})
        order = study.draw_order(DEV_SEED)
        pixels = pool()[order[:1000]]
        self.assertEqual(
            np.abs(whiten.apply(pixels, transform()) - arrays["train_x"]).max(), 0.0)

    def test_job_arrays_train_query_disjoint(self):
        train = study.train_indices(DEV_SEED, 10000)
        query = study.query_indices(DEV_SEED)
        self.assertEqual(np.intersect1d(train, query).size, 0)
        arrays = whiten.job_arrays(DEV_SEED, 10000, transform())
        self.assertEqual(arrays["train_x"].shape[0], 10000)
        self.assertEqual(arrays["query_x"].shape[0], 10000)
        self.assertEqual(arrays["train_y"].shape, (10000,))

    def test_nested_prefixes(self):
        small = whiten.job_arrays(DEV_SEED, 1000, transform())
        large = whiten.job_arrays(DEV_SEED, 10000, transform())
        self.assertTrue(np.array_equal(small["train_x"], large["train_x"][:1000]))
        self.assertTrue(np.array_equal(small["query_x"], large["query_x"]))

    def test_rejects_bad_n(self):
        with self.assertRaises(ValueError):
            whiten.job_arrays(DEV_SEED, 0, transform())
        with self.assertRaises(ValueError):
            whiten.job_arrays(DEV_SEED, 10001, transform())


class SecondOrderStructure(unittest.TestCase):
    """The point of the exercise: the cheap attack's input becomes a constant."""

    def test_rotation_destroys_neighbour_correlations(self):
        rows = np.asarray(study.draw_order(DEV_SEED)[:20000])
        permutation = study.feature_permutation()
        permuted = pool()[rows][:, permutation]
        whitened = whiten.apply(pool()[rows], transform())

        before = whiten.neighbour_correlation_report(permuted, permutation)
        after = whiten.neighbour_correlation_report(whitened, permutation)
        print("\n  permuted pixels : mean |r| neighbour %.4f  non-neighbour %.4f "
              "(ratio %.1f)"
              % (before["mean_abs_corr_neighbour"], before["mean_abs_corr_non_neighbour"],
                 before["mean_abs_corr_neighbour"] / before["mean_abs_corr_non_neighbour"]))
        print("  whitened+rotated: mean |r| neighbour %.4f  non-neighbour %.4f "
              "(ratio %.3f)"
              % (after["mean_abs_corr_neighbour"], after["mean_abs_corr_non_neighbour"],
                 after["mean_abs_corr_neighbour"] / after["mean_abs_corr_non_neighbour"]))

        # pixels: neighbours are 5x more correlated than non-neighbours
        self.assertGreater(before["mean_abs_corr_neighbour"],
                           2.0 * before["mean_abs_corr_non_neighbour"])
        # whitened + rotated: the two are indistinguishable, which is the claim.
        # (They are not zero: with eps>0 the shrunk tail leaves residual
        # correlation, but it is spread isotropically by Q and knows nothing
        # about which pairs are neighbours.)
        gap = abs(after["mean_abs_corr_neighbour"] - after["mean_abs_corr_non_neighbour"])
        self.assertLess(gap, 0.1 * after["mean_abs_corr_non_neighbour"])
        self.assertLess(after["mean_abs_corr_neighbour"],
                        0.2 * before["mean_abs_corr_neighbour"])

    def test_exact_whitening_reaches_the_sampling_noise_floor(self):
        """With eps=0 there is no residual correlation at all, neighbour or not."""
        rows = np.asarray(study.draw_order(DEV_SEED)[:20000])
        permutation = study.feature_permutation()
        t = whiten.fit_transform(pool(), epsilon=0.0, rotation_seed=20260923)
        report = whiten.neighbour_correlation_report(whiten.apply(pool()[rows], t),
                                                     permutation)
        print("\n  whitened(eps=0)+rotated: mean |r| neighbour %.4f  "
              "non-neighbour %.4f  (1/sqrt(M) = %.4f)"
              % (report["mean_abs_corr_neighbour"],
                 report["mean_abs_corr_non_neighbour"], 1.0 / np.sqrt(len(rows))))
        noise_floor = 3.0 / np.sqrt(len(rows))
        self.assertLess(report["mean_abs_corr_neighbour"], noise_floor)
        self.assertLess(report["mean_abs_corr_non_neighbour"], noise_floor)

    def test_filters_are_delocalised(self):
        report = whiten.filter_peaks(transform())
        print("\n  synthesis-filter peak-pixel energy fraction: mean %.3f median %.3f "
              "(uniform reference %.4f)"
              % (report["mean_peak_energy_fraction"],
                 report["median_peak_energy_fraction"], report["uniform_reference"]))
        self.assertLess(report["mean_peak_energy_fraction"], 0.25)
        self.assertAlmostEqual(report["uniform_reference"], 1.0 / 81, places=12)

    def test_rotate_only_is_orthogonal_and_delocalising(self):
        """A rotation with no whitening: norms preserved, filters delocalised.

        This is the variant the measurements end up recommending -- it defeats the
        pixel-topology attack (which needs per-coordinate second-order structure,
        not the covariance spectrum) while leaving every rotation-invariant kernel
        exactly unchanged.
        """
        t = whiten.fit_transform(pool(), epsilon=0.0, rotation_seed=20260923,
                                 method="rotate_only")
        self.assertLess(np.abs(t["A"] @ t["A"].T - np.eye(81)).max(), 1e-12)
        rows = np.asarray(study.draw_order(DEV_SEED)[:2000])
        pixels = pool()[rows]
        z = whiten.apply(pixels, t)
        centred = pixels - t["mean"][None, :]
        self.assertLess(np.abs(np.linalg.norm(z, axis=1)
                               - np.linalg.norm(centred, axis=1)).max(), 1e-4)
        report = whiten.filter_peaks(t)
        self.assertLess(report["mean_peak_energy_fraction"], 0.25)

    def test_zca_without_rotation_keeps_the_pixel_correspondence(self):
        """Whitening alone is NOT a defence: ZCA filters stay pixel-centred."""
        t = whiten.fit_transform(pool(), epsilon=1e-2, rotation_seed=1,
                                 method="zca", rotation="none")
        report = whiten.filter_peaks(t)
        rotated = whiten.filter_peaks(
            whiten.fit_transform(pool(), epsilon=1e-2, rotation_seed=1, method="zca"))
        print("\n  ZCA peak-energy without rotation %.3f, with rotation %.3f"
              % (report["mean_peak_energy_fraction"],
                 rotated["mean_peak_energy_fraction"]))
        self.assertGreater(report["mean_peak_energy_fraction"], 0.8)
        self.assertLess(rotated["mean_peak_energy_fraction"], 0.25)

    def test_permuted_pixel_filters_are_single_pixels(self):
        """Control for the diagnostic: with no whitening each filter IS a pixel."""
        identity = whiten.fit_transform(pool(), epsilon=1.0, rotation_seed=1,
                                        rotation="none")
        report = whiten.filter_peaks(identity)
        # eps=1.0 >> every pool eigenvalue, so W ~ I and A ~ P: a pure permutation.
        self.assertGreater(report["mean_peak_energy_fraction"], 0.9)
        self.assertEqual(report["distinct_peak_pixels"], 81)


if __name__ == "__main__":
    unittest.main(verbosity=2)
