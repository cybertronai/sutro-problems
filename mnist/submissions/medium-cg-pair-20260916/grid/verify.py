"""Verify saved grid evidence without executing the full spatial scorer.

Reconstruct datasets and check predictions, frozen sources, numerical
conformance, instruction/access counts, tape costs and memory occupancy.
Per-address scratch costs, placement hashes and read-initialization order
remain the responsibility of the recorded unmodified project scorer.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
from mnist.code import data


MODEL_SPEC = "01a0bd5e0d2564825b0f53dd766f763c82dbc7c0"
SCORER_PATH = "mnist/submissions/grid-mlp-scoring-20260912"
SCORER_SOURCES = {"score.py", "affine.py", "model_ir.py"}
SCORING_METHOD = (
    "Direct score.score(document); unmodified repository scorer "
    "with its default histogram cache"
)
FULL_SHAPE = {"N": 10000, "Q": 10000, "F": 512, "T": 300}
ARITY = {
    "set": 0, "recv": 0, "send": 1, "copy": 1, "add": 2,
    "sub": 2, "mul": 2, "cmp": 2, "select": 3, "div": 2,
}


def check(condition, message):
    if not condition:
        raise ValueError(message)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_hash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def verify_frozen_program(evidence, manifest):
    """Bind the source, program and input manifest to the frozen run."""
    identity = manifest["identity"]
    for name, expected in identity["source_sha256"].items():
        check(file_hash(HERE / name) == expected, f"Frozen grid source changed: {name}")

    program_bytes = gzip.decompress((evidence / "program.json.gz").read_bytes())
    check(
        hashlib.sha256(program_bytes).hexdigest() == identity["program_file_sha256"],
        "Frozen program.json bytes differ",
    )
    program = json.loads(program_bytes)
    canonical = hashlib.sha256(
        json.dumps(program, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    check(canonical == identity["program_canonical_sha256"], "Canonical program differs")
    check(
        file_hash(evidence / "input_manifest.json") == identity["input_manifest_sha256"],
        "Input manifest changed",
    )
    check(
        file_hash(evidence / "conformance/parity.json") == identity["parity_report_sha256"],
        "Initial numerical parity report changed",
    )
    return program, program_bytes, canonical


def verify_input_datasets(raw_dir, inputs):
    """Recreate every frozen split and compare its arrays with A100 inputs."""
    with gzip.open(HERE.parent / "evidence/a100/croatia.json.gz", "rt") as stream:
        a100 = json.load(stream)
    check(
        file_hash(ROOT / "mnist/code/data.py") == a100["dataset"]["data_helper_sha256"],
        "Canonical data helper changed",
    )
    for kind in ("train_images", "train_labels"):
        path = raw_dir / data.SOURCES[kind][0]
        check(
            data.file_hash(path, "md5") == data.SOURCES[kind][1],
            f"Canonical {kind} changed: {path}",
        )
        check(
            file_hash(path) == a100["dataset"]["source_sha256"][kind],
            f"Raw {kind} SHA-256 differs: {path}",
        )

    pixels = data.read_idx(raw_dir / data.SOURCES["train_images"][0], 60000, True)
    labels = data.read_idx(raw_dir / data.SOURCES["train_labels"][0], 60000, False)
    check(len(inputs) == 11, "Expected all 11 input datasets")

    def resize(rows):
        normalized = pixels[rows].astype(np.float32) / np.float32(255)
        return data.area_resize(normalized, 9).reshape(10000, 81)

    for index, record in enumerate(inputs):
        context = f"Input draw {index:02d}"
        check(
            record["draw"] == index and record["seed"] == 2026091600 + index,
            f"{context}: unexpected draw or seed",
        )
        order = np.random.Generator(np.random.PCG64(record["seed"])).permutation(60000)
        train, query = order[:10000], order[10000:20000]
        arrays = {
            "train_indices": train,
            "query_indices": query,
            "train_pixels": resize(train),
            "train_labels": labels[train],
            "query_pixels": resize(query),
            "test_labels": labels[query],
        }
        check(
            {key: array_hash(value) for key, value in arrays.items()} == record["array_sha256"],
            f"{context}: reconstructed arrays differ; pin OPENBLAS_CORETYPE=Haswell",
        )
        check(
            record["array_sha256"]
            == a100["qualification"]["draws"][index]["dataset"]["array_sha256"],
            f"{context}: grid and A100 inputs differ",
        )


def verify_saved_accuracy(evidence, raw_dir):
    """Independently rescore saved predictions and validate their score argmaxes."""
    output = subprocess.check_output(
        [
            sys.executable, str(HERE / "verify_accuracy.py"),
            "--results-dir", str(evidence), "--raw-dir", str(raw_dir),
        ],
        text=True,
    )
    accuracy = json.loads(output)
    check(accuracy == read_json(evidence / "accuracy.json"), "Saved accuracy report differs")
    return accuracy


def verify_score_provenance(evidence, program, program_bytes, canonical):
    """Check that the saved result comes from the unchanged project scorer."""
    score = read_json(evidence / "grid-score.json")
    check(
        score["program_sha256"] == canonical and score["configuration"] == program["metadata"],
        "Saved grid score belongs to another program",
    )
    score_run = read_json(evidence / "score-run.json")
    check(score_run["method"] == SCORING_METHOD, "Unexpected scoring method")
    check(
        score_run["score_file_sha256"] == file_hash(evidence / "grid-score.json"),
        "Scoring run result hash differs",
    )
    check(
        score_run["program_file_sha256"] == hashlib.sha256(program_bytes).hexdigest(),
        "Scoring run program hash differs",
    )
    check(score_run["scorer_repository_path"] == SCORER_PATH, "Unexpected scorer path")
    check(set(score_run["source_sha256"]) == SCORER_SOURCES, "Unexpected scorer source set")
    for name, expected in score_run["source_sha256"].items():
        check(file_hash(ROOT / SCORER_PATH / name) == expected, f"Scoring source changed: {name}")
    return score


def verify_model(program, score):
    """Check the full-size model and declared serialized schedule."""
    check(program["format"] == "sutro-serial-spatial-affine-v4/1", "Wrong program format")
    check(program["arithmetic"] == "fp32-rne; cmp=lt; select=raw-word", "Wrong program arithmetic")
    check(
        program["placement"] == "nearest-tile-nearest-legal-cell; reserve-bottom-stage",
        "Wrong program placement",
    )
    check(
        program["model_spec_commit"] == score["model_spec_commit"] == MODEL_SPEC,
        "Wrong spatial model revision",
    )
    check(score["configuration"] == program["metadata"], "Frozen configuration differs")
    check(
        all(program["metadata"][key] == value for key, value in FULL_SHAPE.items()),
        "Scored program is not full-size",
    )
    check(score["compute_processor"] == [125, 0], "Wrong compute processor")
    check(score["stage_local_coordinate"] == [64, 31], "Wrong tape stage cell")
    check(
        score["max_simultaneous_instructions"]
        == score["max_simultaneous_outstanding_accesses"] == 1,
        "Unexpected schedule concurrency",
    )
    check(
        score["score_kind"] == "exact static counts of a globally serialized legal schedule",
        "Unexpected score schedule",
    )
    check(
        score["numeric_execution_checked_by_scorer"] is False,
        "Static scorer unexpectedly claims to check numerical execution",
    )


def count_program(program):
    """Walk the affine program without expanding loops or allocating addresses."""
    regions = {row["name"]: row["words"] for row in program["regions"]}
    check(len(regions) == len(program["regions"]), "Duplicate scratch region names")
    logical, accesses = Counter(), Counter()
    histograms = set()
    written_intervals = defaultdict(list)

    def operand_range(operand, scope):
        low = high = operand["offset"]
        for name, coefficient in operand["coefficients"].items():
            start, count = scope[name]
            left, right = coefficient * start, coefficient * (start + count - 1)
            low += min(left, right)
            high += max(left, right)
        region = operand["region"]
        check(
            0 <= low <= high < regions[region],
            f"Operand range [{low}, {high}] exceeds scratch region {region}",
        )
        return low, high

    def charge(kind, operand, scope, multiplicity):
        accesses[kind] += multiplicity
        histograms.add((
            operand["region"], operand["offset"],
            tuple(sorted(operand["coefficients"].items())), tuple(scope.items()),
        ))
        low, high = operand_range(operand, scope)
        if kind not in ("writes", "input_destinations"):
            return
        # These writes cover a contiguous range only if each added stride is
        # no wider than the contiguous interval already accumulated.
        width = 1
        strides = sorted(
            (abs(coefficient), scope[name][1])
            for name, coefficient in operand["coefficients"].items()
            if coefficient and scope[name][1] > 1
        )
        for stride, count in strides:
            if stride > width:
                return  # Do not count uncertain ranges toward write coverage.
            width += stride * (count - 1)
        check(width == high - low + 1, f"Write coverage calculation differs: {operand['region']}")
        written_intervals[operand["region"]].append((low, high + 1))

    def walk(body, scope, multiplicity):
        for node in body:
            if "loop" in node:
                name = node["loop"]
                check(name not in scope, f"Shadowed loop variable: {name}")
                check(type(node["count"]) is int and node["count"] >= 0, f"Invalid loop count: {name}")
                nested_scope = {**scope, name: (node["start"], node["count"])}
                walk(node["body"], nested_scope, multiplicity * node["count"])
                continue
            op = node["op"]
            check(op in ARITY and len(node.get("src", [])) == ARITY[op], f"Invalid opcode/arity: {op}")
            logical[op] += multiplicity
            if not multiplicity:
                continue
            if op == "recv":
                charge("input_destinations", node["dst"], scope, multiplicity)
            elif op == "send":
                charge("output_sources", node["src"][0], scope, multiplicity)
            else:
                for operand in node.get("src", []):
                    charge("reads", operand, scope, multiplicity)
                charge("writes", node["dst"], scope, multiplicity)

    walk(program["body"], {}, 1)
    return {
        "regions": regions,
        "logical": logical,
        "accesses": accesses,
        "histogram_count": len(histograms),
        "written_intervals": written_intervals,
    }


def verify_instruction_counts(counts, score):
    logical = counts["logical"]
    check(dict(logical) == score["logical_instructions"], "Logical instruction counts differ")
    executed = dict(logical)
    executed["copy"] = executed.get("copy", 0) + logical["recv"] + logical["send"]
    check(executed == score["executed_instructions"], "Expanded instruction counts differ")
    check(sum(executed.values()) == score["total_executed_instructions"], "Executed instruction total differs")
    check(counts["histogram_count"] == score["histograms_computed"], "Default cached histogram count differs")
    expected_components = set(counts["accesses"]) | {"input_stage_and_tape", "output_stage_and_tape"}
    check(set(score["components"]) == expected_components, "Unexpected score cost components")
    for kind, total in counts["accesses"].items():
        check(
            score["components"][kind]["scratch_accesses"] == total,
            f"Scratch access total differs: {kind}",
        )


def verify_tape_costs(logical, score):
    """Recompute tape port assignment, stage energy and blocking latency."""
    active_processors = {(125, 0)}
    for kind, opcode in (("input", "recv"), ("output", "send")):
        total = logical[opcode]
        check(score[f"{kind}_tape_words"] == total, f"{kind.capitalize()} tape count differs")
        ports = [total // 250 + int(port < total % 250) for port in range(250)]
        check(score["per_port_words"][kind] == ports, f"{kind.capitalize()} tape port assignment differs")
        energy = cycles = 0
        for port, count in enumerate(ports):
            if count:
                active_processors.add((port, 0))
            links = abs(port - 125)
            # Stage [64,31] is 32 local steps from its processor. Tape adds
            # 64 entry hops, max(50, 2*32) local hops and 2 blocking cycles.
            scratch_energy = 256 * links + 64
            if links == 0:
                scratch_cycles = 1
            elif kind == "input":
                scratch_cycles = 2 * links + 1
            else:
                scratch_cycles = links + 2
            energy += count * (scratch_energy + 128)
            cycles += count * (scratch_cycles + 2)
        expected = {
            "stage_scratch_accesses": total,
            "tape_instructions": total,
            "energy_fj": energy,
            "cycles": cycles,
        }
        check(
            score["components"][f"{kind}_stage_and_tape"] == expected,
            f"{kind.capitalize()} tape stage costs differ",
        )
    check(logical["recv"] == 1630000 and logical["send"] == 10000, "Qualification tape sizes differ")
    check(score["instruction_issuing_processors"] == len(active_processors), "Instruction processor count differs")


def verify_scratch(counts, score):
    """Check capacity, final write coverage and occupancy without per-word arrays.

    Final coverage does not prove read-initialization order; the full scorer
    checks that every read follows initialization.
    """
    regions = counts["regions"]
    words = sum(regions.values())
    check(
        score["program_scratch_words"] == words and score["stage_scratch_words"] == 250,
        "Scratch allocation differs",
    )
    check(score["peak_allocated_scratch_words"] == words + 250 <= 384000000, "Scratch capacity differs")
    check(score["peak_allocated_scratch_bytes"] == 4 * (words + 250), "Scratch bytes differ")
    coverage = 0
    for name, size in regions.items():
        end = 0
        for low, high in sorted(counts["written_intervals"][name]):
            check(low <= end, f"Uncovered final scratch interval in {name} at word {end}")
            end = max(end, high)
        check(end == size, f"Incomplete final writes to scratch region {name}")
        coverage += end
    check(score["peak_initialized_program_words"] == coverage == words, "Initialized scratch total differs")

    remaining, occupancy = words, {}
    tiles = sorted(
        ((x, y) for x in range(250) for y in range(125)),
        key=lambda tile: (abs(tile[0] - 125) + tile[1], tile[1], tile[0]),
    )
    for tile in tiles:
        if not remaining:
            break
        assigned = min(remaining, 12288 - int(tile[1] == 0))
        occupancy[tile] = assigned
        remaining -= assigned
    for port in range(250):
        occupancy[(port, 0)] = occupancy.get((port, 0), 0) + 1
    check(remaining == 0 and max(occupancy.values()) <= 12288, "Memory tile capacity exceeded")
    check(score["memory_tiles"] == len(occupancy), "Memory tile count differs")
    check(score["max_tile_scratch_words"] == max(occupancy.values()), "Peak tile occupancy differs")


def verify_score_totals(score):
    energy = sum(row["energy_fj"] for row in score["components"].values())
    cycles = sum(row["cycles"] for row in score["components"].values())
    check(score["energy_fj"] == score["word_node_hops"] == energy, "Energy component sum differs")
    check(score["cycles"] == score["time_ns"] == cycles, "Cycle component sum differs")
    check(
        score["energy_mj"] == energy / 10**12 and score["time_ms"] == cycles / 10**6,
        "Model display unit conversion differs",
    )


def verify_score_accounting(program, score):
    """Audit saved static counts independently, without importing scorer code."""
    verify_model(program, score)
    counts = count_program(program)
    verify_instruction_counts(counts, score)
    verify_tape_costs(counts["logical"], score)
    verify_scratch(counts, score)
    verify_score_totals(score)


def verify_reduced_conformance(evidence):
    for name in ("parity", "parity_medium", "parity_wide"):
        report = read_json(evidence / f"conformance/{name}.json")
        check(
            report["all_compared_regions_bitwise_equal"] is True,
            f"Reduced numerical parity failed: {name}",
        )
    for name in ("tiny", "edges", "grid"):
        report = read_json(evidence / f"conformance/interpreter-{name}.json")
        check(report["pass"] is True, f"Interpreter parity failed: {name}")


def verify_full_conformance(evidence, manifest, inputs, canonical):
    """Bind the full-size numerical comparison to frozen sources and results."""
    report = read_json(evidence / "conformance/full-draw.json")
    check(
        report["all_compared_regions_bitwise_equal"] and report["all_compared_float_regions_finite"],
        "Full-size numerical parity failed",
    )
    check(
        report["program_canonical_sha256"] == canonical
        and report["program_file_sha256"] == manifest["identity"]["program_file_sha256"],
        "Full-size conformance program differs",
    )
    check(
        report["reference_run_fingerprint"] == manifest["fingerprint"]
        and report["reference_record_sha256"] == file_hash(evidence / "draw_00.json"),
        "Full-size conformance reference differs",
    )
    check(
        report["compiler_source_sha256"] == file_hash(HERE / "il_executor/compiler.py")
        and report["verifier_source_sha256"] == file_hash(HERE / "il_executor/verify_full_draw.py"),
        "Independent numerical executor source differs",
    )
    check(report["input_file_sha256"] == inputs[0]["file_sha256"], "Independent numerical executor input differs")
    check(
        all(report["shape"][key] == value for key, value in FULL_SHAPE.items()),
        "Independent numerical execution was not full-size",
    )
    reference = read_json(evidence / "draw_00.json")
    check(
        len(report["regions"]) == 26
        and report["regions"]["out"]["different_words"] == 0
        and report["regions"]["out"]["words"] == 10000,
        "Full-size conformance report is missing comparisons",
    )
    for name, item in report["regions"].items():
        check(item["sha256_equal"] is True, f"Independent numerical hash mismatch: {name}")
        if name == "out":
            check(
                item["sha256_int64"] == reference["predictions"]["array_sha256"],
                "Independent numerical predictions differ",
            )
        else:
            source_name = "sc2_normalized" if name == "sc2" else name
            check(
                item["sha256"] == reference["snapshots"][source_name]["sha256"]
                and item["all_finite"],
                f"Independent numerical region differs or is nonfinite: {name}",
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    args = parser.parse_args()
    check(np.__version__ == "2.1.2", "Use NumPy 2.1.2 and OPENBLAS_CORETYPE=Haswell")

    evidence = HERE / "evidence"
    manifest = read_json(evidence / "run_manifest.json")
    inputs = read_json(evidence / "input_manifest.json")
    program, program_bytes, canonical = verify_frozen_program(evidence, manifest)
    verify_input_datasets(args.raw_dir, inputs)
    accuracy = verify_saved_accuracy(evidence, args.raw_dir)
    score = verify_score_provenance(evidence, program, program_bytes, canonical)
    verify_score_accounting(program, score)
    verify_reduced_conformance(evidence)
    verify_full_conformance(evidence, manifest, inputs, canonical)

    report = {
        "verified": True,
        "program_sha256": canonical,
        "frozen_source_hashes_verified": True,
        "all11_input_datasets_match_a100": True,
        "accuracy_and_argmax_verified": True,
        "score_source_hashes_and_arithmetic_verified": True,
        "reduced_numerical_parity_verified": True,
        "full_size_numerical_parity_verified": True,
        "total_correct": accuracy["total_correct"],
        "total_predictions": 110000,
        "meets_2_percent_target": accuracy["meets_2_percent_target"],
        "grid_energy_mj": score["energy_mj"],
        "grid_time_ms": score["time_ms"],
        "grid_peak_scratch_bytes": score["peak_allocated_scratch_bytes"],
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
