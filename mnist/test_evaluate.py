"""Competition score correctness and malformed submission rejection tests."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from mnist.evaluate import load_predictions, main, score_predictions


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
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                main([
                    "--predictions", str(predictions_path), "--tier", "small",
                    "--data-dir", str(directory), "--output", str(output_path),
                ])
            result = json.loads(output_path.read_text())
            self.assertEqual(result, json.loads(stdout.getvalue()))
            self.assertEqual(result["tier"], "small")
            self.assertEqual(result["correct"], 2)
            self.assertEqual(result["total"], 3)


if __name__ == "__main__":
    unittest.main()
