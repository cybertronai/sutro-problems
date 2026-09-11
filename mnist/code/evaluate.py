"""Score one integer digit prediction per example in an MNIST tier's test set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--tier", choices=("small", "medium", "large"), required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("mnist/data"))
    parser.add_argument("--output", type=Path, help="Also save the JSON score to this path")
    arguments = parser.parse_args(argv)
    try:
        predictions = load_predictions(arguments.predictions)
        with np.load(arguments.data_dir / f"{arguments.tier}.npz", allow_pickle=False) as archive:
            labels = archive["test_labels"]
        result = {"tier": arguments.tier, **score_predictions(predictions, labels)}
        serialized = json.dumps(result, indent=2, allow_nan=False) + "\n"
        if arguments.output is not None:
            arguments.output.parent.mkdir(parents=True, exist_ok=True)
            arguments.output.write_text(serialized)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    print(serialized, end="")


if __name__ == "__main__":
    main()
