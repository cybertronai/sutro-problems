"""Competition score correctness and malformed submission rejection tests."""

from contextlib import redirect_stdout
from fractions import Fraction
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from mnist.code.evaluate import (
    ACCURACY_TARGETS_PATH,
    accuracy_target_status,
    load_accuracy_target,
    load_predictions,
    main,
    score_predictions,
)


class ScoringTests(unittest.TestCase):
    def test_known_confusion_and_absent_classes(self):
        labels = np.array([0, 0, 1, 1, 1, 2], dtype=np.int64)
        predictions = np.array([0, 1, 1, 2, 1, 0], dtype=np.int32)
        result = score_predictions(predictions, labels)
        self.assertEqual(result["correct"], 3)
        self.assertEqual(result["total"], 6)
        self.assertEqual(result["accuracy"], 0.5)
        matrix = np.array(result["confusion_matrix"])
        expected = np.zeros((10, 10), dtype=np.int64)
        expected[:3, :3] = [[1, 1, 0], [0, 2, 1], [1, 0, 0]]
        np.testing.assert_array_equal(matrix, expected)
        self.assertEqual(result["class_totals"], [2, 3, 1, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(result["per_class_accuracy"][:3], [0.5, 2 / 3, 0.0])
        self.assertEqual(result["per_class_accuracy"][3:], [None] * 7)
        json.dumps(result, allow_nan=False)

    def test_perfect_unsigned_integer_predictions(self):
        labels = np.arange(10, dtype=np.uint8)
        result = score_predictions(labels.astype(np.uint64), labels)
        self.assertEqual(result["accuracy"], 1.0)
        self.assertEqual(result["per_class_accuracy"], [1.0] * 10)

    def test_invalid_count(self):
        with self.assertRaisesRegex(ValueError, "count"):
            score_predictions(np.array([0, 1]), np.array([0]))

    def test_invalid_shape_empty_values_and_dtypes(self):
        invalid = (
            np.array([], dtype=np.int64),
            np.array(0),
            np.array([[0, 1]]),
            np.array([-1, 1]),
            np.array([0, 10]),
            np.array([0.0, 1.0]),
            np.array([0.0, np.nan]),
            np.array([0.0, np.inf]),
            np.array([False, True]),
            np.array(["0", "1"]),
            np.array([0, 1], dtype=object),
            np.array([0, 2**64 - 1], dtype=np.uint64),
        )
        valid = np.array([0, 1])
        for value in invalid:
            with self.subTest(value=value, argument="predictions"):
                with self.assertRaises(ValueError):
                    score_predictions(value, valid)
            with self.subTest(value=value, argument="labels"):
                with self.assertRaises(ValueError):
                    score_predictions(valid, value)


class AccuracyTargetTests(unittest.TestCase):
    def test_small_exact_boundary_and_rounded_false_positive(self):
        below = accuracy_target_status(359, 600, "60")
        at = accuracy_target_status(360, 600, "60")
        self.assertEqual(f"{100 * 359 / 600:.2g}", "60")
        self.assertEqual(below["required_correct"], 360)
        self.assertFalse(below["meets_accuracy_target"])
        self.assertTrue(at["meets_accuracy_target"])
        self.assertEqual(at["accuracy_target_percent"], 60.0)
        self.assertFalse(accuracy_target_status(308, 600, "60")["meets_accuracy_target"])
        for count in (374, 371, 377):
            self.assertTrue(accuracy_target_status(count, 600, "60")["meets_accuracy_target"])

    def test_medium_exact_decimal_boundary(self):
        below = accuracy_target_status(5888, 6000, "98.14")
        at = accuracy_target_status(5889, 6000, "98.14")
        self.assertEqual(below["required_correct"], 5889)
        self.assertFalse(below["meets_accuracy_target"])
        self.assertTrue(at["meets_accuracy_target"])
        self.assertEqual(at["accuracy_target_percent"], 98.14)

    def test_large_exact_boundary(self):
        below = accuracy_target_status(9799, 10000, "98")
        at = accuracy_target_status(9800, 10000, "98")
        self.assertEqual(below["required_correct"], 9800)
        self.assertFalse(below["meets_accuracy_target"])
        self.assertTrue(at["meets_accuracy_target"])
        self.assertEqual(at["accuracy_target_percent"], 98.0)

    def test_published_targets_and_canonical_required_counts(self):
        expected = {"small": "60", "medium": "98", "large": "98"}
        self.assertEqual(json.loads(ACCURACY_TARGETS_PATH.read_text()), expected)
        for tier, total, required in (("small", 600, 360),
                                      ("medium", 6000, 5880),
                                      ("large", 10000, 9800)):
            with self.subTest(tier=tier):
                target = load_accuracy_target(tier)
                self.assertEqual(target, expected[tier])
                result = accuracy_target_status(required, total, target)
                self.assertEqual(result["required_correct"], required)
                self.assertTrue(result["meets_accuracy_target"])
                self.assertFalse(accuracy_target_status(required - 1, total, target)["meets_accuracy_target"])

    def test_generic_fraction_and_high_precision_decimal(self):
        exact = accuracy_target_status(1, 3, Fraction(100, 3))
        just_above = accuracy_target_status(1, 3, "33.3333333333333333333333333333333333333334")
        self.assertEqual(exact["required_correct"], 1)
        self.assertTrue(exact["meets_accuracy_target"])
        self.assertEqual(just_above["required_correct"], 2)
        self.assertFalse(just_above["meets_accuracy_target"])
        self.assertTrue(accuracy_target_status(0, 3, "0")["meets_accuracy_target"])
        self.assertFalse(accuracy_target_status(2, 3, "100")["meets_accuracy_target"])
        self.assertTrue(accuracy_target_status(3, 3, "100")["meets_accuracy_target"])

    def test_invalid_target_or_count_rejected(self):
        for value in ("nan", "inf", "-1", "100.01", "invalid", "1/0", 60.0, None):
            with self.subTest(target=value), self.assertRaises(ValueError):
                accuracy_target_status(1, 3, value)
        for correct, total in ((-1, 3), (4, 3), (0, 0), (1.0, 3), (True, 3)):
            with self.subTest(correct=correct, total=total), self.assertRaises(ValueError):
                accuracy_target_status(correct, total, "60")

    def test_shared_target_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "targets.json"
            path.write_text(json.dumps({"small": "60", "medium": "98.14"}))
            with patch("mnist.code.evaluate.ACCURACY_TARGETS_PATH", path):
                self.assertEqual(load_accuracy_target("small"), "60")
                self.assertEqual(load_accuracy_target("medium"), "98.14")
                with self.assertRaisesRegex(ValueError, "No confirmed"):
                    load_accuracy_target("unconfigured")
                path.write_text(json.dumps({"small": 60}))
                with self.assertRaisesRegex(ValueError, "percentage string"):
                    load_accuracy_target("small")


class PredictionFileTests(unittest.TestCase):
    def test_npy_npz_and_missing_key(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            predictions = np.array([7, 1, 9], dtype=np.int64)
            np.save(directory / "submission.npy", predictions)
            np.savez(directory / "submission.npz", predictions=predictions)
            np.savez(directory / "missing.npz", wrong_key=predictions)
            for extension in ("npy", "npz"):
                np.testing.assert_array_equal(
                    load_predictions(directory / f"submission.{extension}"), predictions
                )
            with self.assertRaisesRegex(ValueError, "predictions"):
                load_predictions(directory / "missing.npz")
            with self.assertRaisesRegex(ValueError, "npy"):
                load_predictions(directory / "submission.csv")

    def test_cli_reads_tier_and_writes_json(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            np.savez(directory / "small.npz", test_labels=np.array([1, 9, 9]))
            predictions_path = directory / "submission.npy"
            np.save(predictions_path, np.array([1, 9, 1]))
            output_path = directory / "scores" / "score.json"
            targets = directory / "targets.json"
            targets.write_text(json.dumps({"small": "60"}))
            stdout = io.StringIO()
            with redirect_stdout(stdout), patch("mnist.code.evaluate.ACCURACY_TARGETS_PATH", targets):
                main([
                    "--predictions", str(predictions_path), "--tier", "small",
                    "--data-dir", str(directory), "--output", str(output_path),
                ])
            result = json.loads(output_path.read_text())
            self.assertEqual(result, json.loads(stdout.getvalue()))
            self.assertEqual(result["tier"], "small")
            self.assertEqual(result["correct"], 2)
            self.assertEqual(result["total"], 3)
            self.assertEqual(result["accuracy_target_percent"], 60.0)
            self.assertEqual(result["required_correct"], 2)
            self.assertTrue(result["meets_accuracy_target"])

    def test_cli_below_target_is_valid_successful_score(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            np.savez(directory / "small.npz", test_labels=np.array([1, 9, 9]))
            predictions = directory / "submission.npy"
            np.save(predictions, np.array([1, 1, 1]))
            targets = directory / "targets.json"
            targets.write_text(json.dumps({"small": "60"}))
            stdout = io.StringIO()
            with redirect_stdout(stdout), patch("mnist.code.evaluate.ACCURACY_TARGETS_PATH", targets):
                self.assertIsNone(main([
                    "--predictions", str(predictions), "--tier", "small",
                    "--data-dir", str(directory),
                ]))
            result = json.loads(stdout.getvalue())
            self.assertEqual(result["correct"], 1)
            self.assertEqual(result["required_correct"], 2)
            self.assertFalse(result["meets_accuracy_target"])


if __name__ == "__main__":
    unittest.main()
