"""Fresh-process adapter host. Receives three allowed arrays, never query labels."""
import argparse
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import numpy as np
from .common import array_sha, sha, utc, write


def resolve_adapter(spec):
    module_name, function = spec.rsplit(":", 1)
    path = Path(module_name)
    if path.suffix == ".py":
        resolved = path.resolve()
        sys.path.insert(0, str(resolved.parent))
        loader = importlib.util.spec_from_file_location("aminist21_external_adapter", resolved)
        module = importlib.util.module_from_spec(loader)
        loader.loader.exec_module(module)
    else:
        module = importlib.import_module(module_name)
    return getattr(module, function), Path(module.__file__).resolve()


def invoke(adapter, arrays, config):
    if set(arrays) != {"train_images", "train_labels", "test_images"}:
        raise ValueError("Learner inputs must contain exactly three arrays; evaluation labels forbidden")
    function, source = resolve_adapter(adapter)
    before = {key: array_sha(value) for key, value in arrays.items()}
    started = time.perf_counter()
    value = function(**arrays, config=dict(config))
    elapsed = time.perf_counter()-started
    predictions, metadata = value if isinstance(value, tuple) and len(value) == 2 else (value, {})
    predictions = np.asarray(predictions)
    if predictions.shape != (len(arrays["test_images"]),) or predictions.dtype.kind not in "iu":
        raise ValueError("Adapter must return one integer class per query image")
    if np.any(predictions < 0) or np.any(predictions > 9):
        raise ValueError("Predictions must be in 0..9")
    predictions = predictions.astype(np.int64)
    if before != {key: array_sha(value) for key, value in arrays.items()}:
        raise ValueError("Adapter mutated an input array")
    result = {"completed_at_utc": utc(), "adapter": adapter,
              "adapter_source_sha256": sha(source), "config": config,
              "input_sha256": before, "predictions_sha256": array_sha(predictions),
              "adapter_wall_seconds": elapsed, "query_labels_supplied": False,
              "metadata": metadata}
    return predictions, result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with np.load(args.input, allow_pickle=False) as archive:
        if len(archive.files) != 3 or set(archive.files) != {"train_images","train_labels","test_images"}:
            raise ValueError("Input archive must contain exactly three unique allowed arrays")
        arrays = {key: archive[key].copy() for key in archive.files}
    predictions, metadata = invoke(args.adapter, arrays, json.loads(Path(args.config).read_text()))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output.with_suffix(".npz"), predictions=predictions)
    metadata["prediction_file_sha256"] = sha(output.with_suffix(".npz"))
    write(output, metadata)


if __name__ == "__main__":
    main()
