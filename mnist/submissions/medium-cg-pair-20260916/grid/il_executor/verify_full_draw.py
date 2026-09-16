#!/usr/bin/env python3
"""Execute the full frozen IL on one label-free qualification input archive."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

from compiler import compile_shared, document_digest, run_shared


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def ah(array):
    return hashlib.sha256(memoryview(np.ascontiguousarray(array)).cast("B")).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--draw", type=int, default=0)
    p.add_argument("--affine-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    sys.path.insert(0, str(args.affine_dir))
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    doc_path = args.root / "program.json"
    doc = json.loads(doc_path.read_text())
    stem = f"draw_{args.draw:02d}"
    input_path = args.root / "inputs" / (stem + ".npz")
    reference_path = args.root / "evaluation" / (stem + ".json")
    reference = json.loads(reference_path.read_text())
    with np.load(input_path, allow_pickle=False) as source:
        if set(source.files) != {"x", "y", "q"}:
            raise ValueError("Input archive must contain only training pixels, training labels, and query pixels")
        x, y, q = source["x"], source["y"], source["q"]
    tape = np.concatenate([np.ascontiguousarray(x, dtype=np.float32).reshape(-1).view(np.uint32),
                           np.ascontiguousarray(y, dtype=np.uint32).reshape(-1),
                           np.ascontiguousarray(q, dtype=np.float32).reshape(-1).view(np.uint32)])
    print(json.dumps({"stage": "compile_and_validate", "program_sha256": document_digest(doc),
                      "tape_words": len(tape)}), flush=True)
    started = time.perf_counter()
    library = compile_shared(doc, output / "build")
    compile_elapsed = time.perf_counter() - started
    print(json.dumps({"stage": "execute", "compile_and_validation_seconds": compile_elapsed}), flush=True)
    result = run_shared(library, tape, document_digest(doc))
    print(json.dumps({"stage": "compare", "execution_seconds": result["elapsed_seconds"]}), flush=True)
    memory, out = result["memory"], result["output"]
    regions, position = {}, 1
    for region in doc["regions"]:
        regions[region["name"]] = (position, region["words"])
        position += region["words"]
    comparisons = {}
    skipped = []
    for name, snapshot in reference["snapshots"].items():
        target = "sc2" if name == "sc2_normalized" else name
        if target not in regions:
            skipped.append(name)
            continue
        base, words = regions[target]
        array = memory[base:base+words]
        observed = ah(array)
        comparisons[target] = {"words": words, "sha256": observed,
                               "reference_sha256": snapshot["sha256"],
                               "sha256_equal": observed == snapshot["sha256"],
                               "all_finite": bool(np.isfinite(array.view(np.float32)).all())}
    predictions_path = args.root / "evaluation" / reference["predictions"]["filename"]
    predictions = np.load(predictions_path, allow_pickle=False)
    if sha(predictions_path) != reference["predictions"]["file_sha256"]:
        raise ValueError("Reference predictions changed")
    differences = int(np.count_nonzero(out.astype(np.int64) != predictions))
    comparisons["out"] = {"words": len(out), "different_words": differences,
                           "sha256_int64": ah(out.astype(np.int64)),
                           "reference_sha256_int64": reference["predictions"]["array_sha256"],
                           "sha256_equal": ah(out.astype(np.int64)) == reference["predictions"]["array_sha256"]}
    report = {"draw": args.draw, "shape": doc["metadata"], "program_file_sha256": sha(doc_path),
              "program_canonical_sha256": result["program_sha256"],
              "compiler_source_sha256": sha(Path(__file__).parent / "compiler.py"),
              "verifier_source_sha256": sha(__file__),
              "input_file_sha256": sha(input_path), "tape_sha256": ah(tape),
              "reference_record_sha256": sha(reference_path),
              "reference_run_fingerprint": reference["run_fingerprint"],
              "test_labels_read": False, "compiler_build": json.loads((output / "build/build.json").read_text()),
              "compile_and_validation_seconds": compile_elapsed, "execution_seconds": result["elapsed_seconds"],
              "scratch_words_including_zero": len(memory), "scratch_sha256": ah(memory),
              "regions": comparisons, "intermediate_snapshots_not_in_final_scratch": skipped,
              "all_compared_regions_bitwise_equal": all(r["sha256_equal"] for r in comparisons.values()),
              "all_compared_float_regions_finite": all(r.get("all_finite", True) for r in comparisons.values())}
    (output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    # Keep only compact prediction and hash evidence, never a full scratch dump.
    np.save(output / "output.npy", out, allow_pickle=False)
    print(json.dumps(report), flush=True)
    if not report["all_compared_regions_bitwise_equal"] or not report["all_compared_float_regions_finite"]:
        raise AssertionError("Full generic IL and qualification executor disagree or contain nonfinite floats")


if __name__ == "__main__":
    main()
