"""Score one integer digit prediction per example in an MNIST tier's test set."""

from __future__ import annotations

import argparse
from fractions import Fraction
import json
from pathlib import Path

import numpy as np


ACCURACY_TARGETS_PATH = Path(__file__).resolve().parents[1] / "doc" / "accuracy_targets.json"


def accuracy_target_status(correct: int, total: int, target_percent: str | Fraction) -> dict:
    """Check the classification threshold using exact rational arithmetic.

    Decimal strings from the target configuration are converted directly to
    fractions, never through a binary floating-point percentage. The minimum
    integer correct count is the ceiling of total * target_percent / 100.
    This status concerns classification accuracy only, not complete compliance
    with the benchmark's model, tape, scoring, or hardware requirements.
    """
    if type(correct) is not int or type(total) is not int or not 0 <= correct <= total or total <= 0:
        raise ValueError("Accuracy target counts must satisfy 0 <= correct <= total and total > 0")
    if not isinstance(target_percent, (str, Fraction)):
        raise ValueError("Accuracy target must be an exact decimal string or fraction")
    try:
        percent = Fraction(target_percent)
    except (ValueError, ZeroDivisionError) as error:
        raise ValueError("Accuracy target must be a finite percentage between 0 and 100") from error
    if not 0 <= percent <= 100:
        raise ValueError("Accuracy target must be a finite percentage between 0 and 100")
    denominator = percent.denominator * 100
    required = (total * percent.numerator + denominator - 1) // denominator
    return {
        "accuracy_target_percent": float(percent),
        "required_correct": required,
        "meets_accuracy_target": correct >= required,
    }


def load_accuracy_target(tier: str) -> str:
    """Read the selected tier's current percentage from the shared rules file."""
    configured = json.loads(ACCURACY_TARGETS_PATH.read_text())
    if not isinstance(configured, dict):
        raise ValueError("Accuracy target configuration must map tiers to percentage strings")
    value = configured.get(tier)
    if value is None:
        raise ValueError(f"No confirmed accuracy target is configured for MNIST-{tier}")
    if not isinstance(value, str):
        raise ValueError(f"Accuracy target for MNIST-{tier} must be a percentage string")
    return value


def _digit_vector(value: np.ndarray, name: str) -> np.ndarray:
    vector = np.asarray(value)
    if vector.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    if not len(vector):
        raise ValueError(f"{name} must be nonempty")
    if not np.issubdtype(vector.dtype, np.integer):
        raise ValueError(f"{name} must have an integer dtype; got {vector.dtype}")
    if np.any(vector < 0) or np.any(vector > 9):
        raise ValueError(f"{name} must contain only digit labels 0 through 9")
    return vector.astype(np.int64, copy=False)


def score_predictions(predictions: np.ndarray, labels: np.ndarray) -> dict:
    """Return accuracy and a true-class-row/predicted-class-column confusion matrix.

    Inputs must be nonempty one-dimensional integer arrays of equal length,
    containing only labels 0 through 9. Floating-point labels and booleans are
    rejected even when their values could be converted to valid integers.
    Absent true classes have ``None`` per-class accuracy, serialized as JSON null.
    """
    predictions = _digit_vector(predictions, "predictions")
    labels = _digit_vector(labels, "labels")
    if len(predictions) != len(labels):
        raise ValueError(
            f"Prediction count {len(predictions)} does not match label count {len(labels)}"
        )
    confusion = np.bincount(labels * 10 + predictions, minlength=100).reshape(10, 10)
    class_totals = confusion.sum(axis=1)
    class_correct = np.diag(confusion)
    correct = int(class_correct.sum())
    total = len(labels)
    return {
        "accuracy": correct / total,
        "correct": correct,
        "total": total,
        "classes": list(range(10)),
        "confusion_matrix": confusion.tolist(),
        "confusion_matrix_orientation": "rows=true class, columns=predicted class",
        "class_totals": class_totals.tolist(),
        "per_class_accuracy": [
            int(hits) / int(count) if count else None
            for hits, count in zip(class_correct, class_totals)
        ],
    }


def load_predictions(path: Path) -> np.ndarray:
    """Read an NPY vector or an NPZ archive containing a ``predictions`` vector."""
    path = Path(path)
    if path.suffix.lower() not in (".npy", ".npz"):
        raise ValueError("Predictions must be a .npy file or a .npz archive")
    loaded = np.load(path, allow_pickle=False)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        with loaded as archive:
            if "predictions" not in archive:
                raise ValueError("Prediction NPZ archive must contain a 'predictions' array")
            return archive["predictions"]
    return loaded


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Valid predictions below the accuracy target still produce a successful "
               "JSON score with meets_accuracy_target=false. This flag checks only "
               "classification accuracy, not complete benchmark compliance.",
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--tier", choices=("small", "medium", "large"), required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("mnist/data"))
    parser.add_argument("--error-target", choices=("2", "3", "5", "8", "12"),
                        help="MNIST-medium error-rate target in percent; select explicitly for the new 10000/10000 profile")
    parser.add_argument("--output", type=Path, help="Also save the JSON score to this path")
    arguments = parser.parse_args(argv)
    try:
        predictions = load_predictions(arguments.predictions)
        with np.load(arguments.data_dir / f"{arguments.tier}.npz", allow_pickle=False) as archive:
            labels = archive["test_labels"]
        result = {"tier": arguments.tier, **score_predictions(predictions, labels)}
        if arguments.error_target is not None and arguments.tier != "medium":
            raise ValueError("The five error-rate targets currently apply to MNIST-medium")
        target = (Fraction(100)-Fraction(arguments.error_target) if arguments.error_target is not None
                  else load_accuracy_target(arguments.tier))
        result.update(accuracy_target_status(result["correct"], result["total"], target))
        if arguments.error_target is not None:
            result["error_target_percent"] = float(arguments.error_target)
            result["error_rate_percent"] = 100 * (result["total"]-result["correct"]) / result["total"]
        serialized = json.dumps(result, indent=2, allow_nan=False) + "\n"
        if arguments.output is not None:
            arguments.output.parent.mkdir(parents=True, exist_ok=True)
            arguments.output.write_text(serialized)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    print(serialized, end="")


if __name__ == "__main__":
    main()
