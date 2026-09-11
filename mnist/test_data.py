"""Meaningful checks of MNIST sampling, IDX validation, and area resizing."""

import contextlib
import gzip
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock

import numpy as np

from mnist.data import (
    DEFAULT_PROFILE,
    REFERENCE_PROFILE,
    SOURCES,
    area_resize,
    area_weights,
    prepare,
    read_idx,
    source_permutations,
)


class AreaResizeTests(unittest.TestCase):
    def test_constant_preservation(self):
        inputs = np.full((2, 28, 28), 0.37, dtype=np.float32)
        for size in (3, 9, 28):
            result = area_resize(inputs, size)
            self.assertEqual(result.shape, (2, size, size))
            self.assertEqual(result.dtype, np.float32)
            np.testing.assert_allclose(result, 0.37, atol=1e-7)

    def test_mean_preservation_for_fractional_resize(self):
        inputs = np.random.default_rng(9).random((5, 28, 28), dtype=np.float32)
        for size in (3, 9):
            result = area_resize(inputs, size)
            np.testing.assert_allclose(
                result.mean(axis=(1, 2)), inputs.mean(axis=(1, 2)), atol=1e-7
            )

    def test_known_fractional_overlap(self):
        # 3 -> 2 bins each have width 1.5, so center pixel contributes 1/3.
        expected_weights = np.array([[2 / 3, 1 / 3, 0], [0, 1 / 3, 2 / 3]])
        np.testing.assert_allclose(area_weights(3, 2), expected_weights, atol=1e-7)
        inputs = np.arange(9, dtype=np.float32).reshape(1, 3, 3)
        expected = np.array([[[4 / 3, 8 / 3], [16 / 3, 20 / 3]]])
        np.testing.assert_allclose(area_resize(inputs, 2), expected, atol=1e-6)

    def test_identity_resize_preserves_pixels(self):
        inputs = np.arange(2 * 28 * 28, dtype=np.float32).reshape(2, 28, 28)
        np.testing.assert_array_equal(area_resize(inputs, 28), inputs)

    def test_invalid_shapes_and_sizes(self):
        for size in (0, 29):
            with self.assertRaises(ValueError):
                area_resize(np.zeros((1, 28, 28)), size)
        with self.assertRaises(ValueError):
            area_resize(np.zeros((1, 1, 28, 28)), 3)


class SamplingTests(unittest.TestCase):
    def test_reference_permutation_order_is_stable(self):
        train, test = source_permutations()
        np.testing.assert_array_equal(
            train[:12],
            [49472, 754, 48691, 16382, 18835, 51786, 5569, 34192, 29130,
             33342, 50766, 51126],
        )
        np.testing.assert_array_equal(
            test[:12],
            [7702, 4381, 5508, 5393, 7602, 8508, 3699, 6858, 7687,
             7644, 7496, 7594],
        )

    def test_deterministic_nested_without_replacement(self):
        train, test = source_permutations()
        repeat_train, repeat_test = source_permutations()
        np.testing.assert_array_equal(train, repeat_train)
        np.testing.assert_array_equal(test, repeat_test)
        for order, expected_count in ((train, 60000), (test, 10000)):
            self.assertEqual(order.dtype, np.int64)
            self.assertEqual(len(np.unique(order)), expected_count)
            self.assertEqual(int(order.min()), 0)
            self.assertEqual(int(order.max()), expected_count - 1)
            self.assertTrue(set(order[:1000]).issubset(set(order[:10000])))
        different_train, different_test = source_permutations(seed=17)
        self.assertFalse(np.array_equal(train, different_train))
        self.assertFalse(np.array_equal(test, different_test))

    def test_independent_split_rng_and_source_provenance(self):
        train_indices, test_indices = source_permutations(100, 100)
        self.assertFalse(np.array_equal(train_indices, test_indices))
        # Equal numeric row indices in different official splits identify
        # different examples. Keep each index tied to its source split.
        train_source = np.arange(100) + 1000
        test_source = np.arange(100) + 2000
        train_examples = train_source[train_indices[:20]]
        test_examples = test_source[test_indices[:20]]
        np.testing.assert_array_equal(train_examples - 1000, train_indices[:20])
        np.testing.assert_array_equal(test_examples - 2000, test_indices[:20])
        self.assertFalse(set(train_examples).intersection(test_examples))
        changed_train, same_test = source_permutations(101, 100)
        self.assertEqual(len(changed_train), 101)
        np.testing.assert_array_equal(test_indices, same_test)


class PreparedDatasetTests(unittest.TestCase):
    """Exercise preparation with full-size synthetic IDX downloads, offline."""

    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name)
        cls.raw_dir = cls.root / "raw"
        cls.raw_dir.mkdir()
        for name, (filename, _) in SOURCES.items():
            is_train = name.startswith("train")
            count = 60000 if is_train else 10000
            with gzip.open(cls.raw_dir / filename, "wb", compresslevel=1) as stream:
                if name.endswith("images"):
                    stream.write(struct.pack(">IIII", 2051, count, 28, 28))
                    # Source splits occupy distinct intensity ranges. Chunking
                    # avoids keeping the entire raw fixture in memory.
                    for start in range(0, count, 1000):
                        pixels = cls.intensities(np.arange(start, start + 1000), is_train)
                        images = np.broadcast_to(pixels[:, None, None], (1000, 28, 28))
                        stream.write(images.tobytes())
                else:
                    stream.write(struct.pack(">II", 2049, count))
                    stream.write(cls.labels(np.arange(count), is_train).astype(np.uint8).tobytes())
        cls.manifests = {
            profile: cls.prepare_fixture(cls.root / profile, profile)
            for profile in (DEFAULT_PROFILE, REFERENCE_PROFILE)
        }

    @staticmethod
    def intensities(indices, is_train):
        return (indices % 128 + (0 if is_train else 128)).astype(np.uint8)

    @staticmethod
    def labels(indices, is_train):
        return ((indices + (0 if is_train else 3)) % 10).astype(np.int64)

    @classmethod
    def prepare_fixture(cls, destination, profile=DEFAULT_PROFILE):
        with mock.patch(
            "mnist.data.download_source",
            side_effect=lambda raw_dir, filename, expected_md5: cls.raw_dir / filename,
        ), contextlib.redirect_stdout(io.StringIO()):
            return prepare(destination, profile=profile)

    def assert_tier(self, profile, tier, size, train_indices, test_indices, test_source):
        manifest = self.manifests[profile]
        metadata = manifest["tiers"][tier]
        self.assertEqual(metadata["train_source"], "train")
        self.assertEqual(metadata["test_source"], test_source)
        with np.load(self.root / profile / f"{tier}.npz", allow_pickle=False) as arrays:
            for split, expected_indices, source in (
                ("train", train_indices, "train"), ("test", test_indices, test_source)
            ):
                with self.subTest(profile=profile, tier=tier, split=split):
                    indices = arrays[f"{split}_indices"]
                    self.assertEqual(indices.dtype, np.int64)
                    np.testing.assert_array_equal(indices, expected_indices)
                    self.assertEqual(len(np.unique(indices)), len(indices))
                    self.assertEqual(metadata[f"{split}_count"], len(indices))
                    labels = arrays[f"{split}_labels"]
                    self.assertEqual(labels.dtype, np.int64)
                    np.testing.assert_array_equal(labels, self.labels(indices, source == "train"))
                    self.assertEqual(
                        metadata["class_histograms"][split],
                        np.bincount(labels, minlength=10).tolist(),
                    )
                    images = arrays[f"{split}_images"]
                    self.assertEqual(images.shape, (len(indices), 1, size, size))
                    self.assertEqual(images.dtype, np.float32)
                    expected_pixels = self.intensities(indices, source == "train").astype(np.float32) / 255
                    # Every pixel of each fixture image has one known value;
                    # its min and max jointly check all pixels after resizing.
                    np.testing.assert_allclose(images.min(axis=(1, 2, 3)), expected_pixels, atol=2e-7)
                    np.testing.assert_allclose(images.max(axis=(1, 2, 3)), expected_pixels, atol=2e-7)

    def test_competition_counts_sources_and_disjoint_nested_samples(self):
        train, test = source_permutations()
        self.assert_tier(DEFAULT_PROFILE, "small", 3, train[:600], train[6000:6600], "train")
        self.assert_tier(DEFAULT_PROFILE, "medium", 9, train[:6000], train[6000:12000], "train")
        with np.load(self.root / DEFAULT_PROFILE / "small.npz") as small:
            with np.load(self.root / DEFAULT_PROFILE / "medium.npz") as medium:
                for split in ("train", "test"):
                    np.testing.assert_array_equal(small[f"{split}_indices"], medium[f"{split}_indices"][:600])
                # Checking the larger allocations rules out all cross-tier
                # train/test overlap for the two reduced-resolution tasks.
                self.assertEqual(
                    np.intersect1d(medium["train_indices"], medium["test_indices"]).size, 0
                )

    def test_large_preserves_complete_official_splits(self):
        train, test = source_permutations()
        self.assert_tier(DEFAULT_PROFILE, "large", 28, train, test, "test")
        np.testing.assert_array_equal(np.sort(train), np.arange(60000))
        np.testing.assert_array_equal(np.sort(test), np.arange(10000))

    def test_reference_profile_preserves_original_counts_and_official_sources(self):
        train, test = source_permutations()
        self.assert_tier(REFERENCE_PROFILE, "small", 3, train[:1000], test[:1000], "test")
        self.assert_tier(REFERENCE_PROFILE, "medium", 9, train[:10000], test, "test")
        # Large is identical across profiles, including all canonical arrays.
        self.assertEqual(
            self.manifests[DEFAULT_PROFILE]["tiers"]["large"]["arrays"],
            self.manifests[REFERENCE_PROFILE]["tiers"]["large"]["arrays"],
        )

    def test_manifest_persists_profile_and_per_tier_source_provenance(self):
        for profile, manifest in self.manifests.items():
            with self.subTest(profile=profile):
                self.assertEqual(manifest["format_version"], 2)
                self.assertEqual(manifest["profile"], profile)
                self.assertEqual(
                    json.loads((self.root / profile / "manifest.json").read_text()), manifest
                )
                for tier in ("small", "medium", "large"):
                    metadata = manifest["tiers"][tier]
                    self.assertEqual(metadata["train_offset"], 0)
                    self.assertEqual(
                        metadata["test_offset"],
                        6000 if profile == DEFAULT_PROFILE and tier != "large" else 0,
                    )

    def test_repeated_preparation_has_identical_array_hashes(self):
        repeated = self.prepare_fixture(self.root / "repeat")
        self.assertEqual(repeated["profile"], DEFAULT_PROFILE)
        for tier in ("small", "medium", "large"):
            self.assertEqual(
                repeated["tiers"][tier]["arrays"],
                self.manifests[DEFAULT_PROFILE]["tiers"][tier]["arrays"],
            )

    def test_unknown_profile_fails_before_creating_files(self):
        destination = self.root / "invalid"
        with self.assertRaisesRegex(ValueError, "Unknown MNIST profile"):
            prepare(destination, profile="typo")
        self.assertFalse(destination.exists())


class IDXTests(unittest.TestCase):
    def test_labels_validate_length_classes_and_header(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.gz"
            cases = (
                (struct.pack(">II", 2049, 3) + bytes([2, 4, 9]), False),
                (struct.pack(">II", 2049, 3) + bytes([2, 4]), True),
                (struct.pack(">II", 2049, 3) + bytes([2, 4, 10]), True),
                (struct.pack(">II", 2051, 3) + bytes([2, 4, 9]), True),
                (b"short", True),
            )
            for payload, expect_error in cases:
                with gzip.open(path, "wb") as stream:
                    stream.write(payload)
                if expect_error:
                    with self.assertRaises(ValueError):
                        read_idx(path, 3, images=False)
                else:
                    np.testing.assert_array_equal(read_idx(path, 3, images=False), [2, 4, 9])


if __name__ == "__main__":
    unittest.main()
