#!/usr/bin/env python3
"""Exact scratch-read distance histogram and CDF for the frozen small grid trace.

Run from any directory; outputs are written alongside this script. The affine
loop counts are exact: no sampling or full billion-instruction expansion.
"""
from __future__ import annotations

from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
SCORER = HERE.parent.parent / "grid-mlp-scoring-20260912"
sys.path.insert(0, str(SCORER))
from affine import Program  # noqa: E402
from score import (  # noqa: E402
    COMPUTE, PITCH, PORTS, STAGE_DISTANCE, histogram_counts, placement,
)

COMPONENTS = (
    "normal_operands", "output_copy_sources", "input_stage_sources",
    "output_send_sources",
)
# Half-open intervals: eight grid hops near the computational working set,
# then powers of two to retain the sparse, far-away input-staging tail.
BIN_EDGES = [*range(32, 257, 8), 512, 1024, 2048, 4096, 8192, 16384]


def collect_read_counts(document):
    """Return exact distance -> read-count Counters by scratch-read source.

    One sample is one executed source operand, including repeated sources and
    both select candidates. Distance is the one-way routed length in grid-node
    hops: 128 * processor-link count + owning-core-to-cell Manhattan distance.
    Tape-port transport and scratch writes are outside this distribution.
    """
    program = Program(document)
    _, _, local_distances, links, _ = placement(program.words)
    address_counts, _ = histogram_counts(program)
    distances = PITCH * links + local_distances
    result = {name: Counter() for name in COMPONENTS}
    for name, source in (("normal_operands", "reads"),
                         ("output_copy_sources", "output_sources")):
        for distance, count in zip(distances[1:], address_counts[source][1:]):
            if count:
                result[name][int(distance)] += int(count)
    n_input = program.instructions.get("recv", 0)
    for port in range(PORTS):
        count = n_input // PORTS + (port < n_input % PORTS)
        if count:
            distance = PITCH * (abs(port - COMPUTE[0]) + COMPUTE[1]) + STAGE_DISTANCE
            result["input_stage_sources"][distance] += count
    n_output = program.instructions.get("send", 0)
    if n_output:
        # send reads its local stage, then transports the word to the tape.
        result["output_send_sources"][STAGE_DISTANCE] = n_output
    return result


def combine_counts(components):
    total = Counter()
    for counts in components.values():
        total.update(counts)
    return total


def quantile(counts, numerator, denominator):
    """Smallest integer distance whose exact count CDF reaches the fraction."""
    total = sum(counts.values())
    cumulative = 0
    for distance, count in sorted(counts.items()):
        cumulative += count
        if cumulative * denominator >= total * numerator:
            return distance
    raise ValueError("Empty distribution or invalid quantile")


def binned_counts(counts):
    if min(counts) < BIN_EDGES[0] or max(counts) >= BIN_EDGES[-1]:
        raise ValueError("Frozen-submission bins do not cover this distribution")
    return [sum(count for distance, count in counts.items() if lo <= distance < hi)
            for lo, hi in zip(BIN_EDGES, BIN_EDGES[1:])]


def write_data(components, summary):
    counts = combine_counts(components)
    total = sum(counts.values())
    cumulative = 0
    with (HERE / "read_distance.csv").open("w", newline="") as output:
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(["distance_grid_hops", *COMPONENTS, "reads",
                         "cumulative_reads", "cumulative_fraction", "read_energy_fj"])
        for distance, count in sorted(counts.items()):
            cumulative += count
            writer.writerow([distance, *(components[name].get(distance, 0) for name in COMPONENTS),
                             count, cumulative, format(cumulative / total, ".17g"),
                             2 * distance * count])
    bins = binned_counts(counts)
    assert sum(bins) == total
    with (HERE / "read_distance_bins.csv").open("w", newline="") as output:
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(["lower_grid_hops_inclusive", "upper_grid_hops_exclusive",
                         "reads", "fraction"])
        for lo, hi, count in zip(BIN_EDGES, BIN_EDGES[1:], bins):
            writer.writerow([lo, hi, count, format(count / total, ".17g")])
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def plot(counts):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, PercentFormatter

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "svg.hashsalt": "mnist-small-read-distance"})
    distances = np.array(sorted(counts))
    weights = np.array([counts[int(distance)] for distance in distances], dtype=np.int64)
    total = int(weights.sum())
    cumulative = np.cumsum(weights) / total
    blue, red = "#286d9e", "#bf493d"
    fig, (ax_h, ax_c) = plt.subplots(1, 2, figsize=(13.2, 5.7))
    fig.subplots_adjust(left=.067, right=.98, bottom=.19, top=.77, wspace=.23)
    fig.suptitle("MNIST-small · H32 MLP · grid scratch reads", fontsize=17, x=.067, ha="left", y=.97)
    fig.text(.067, .89, f"{total:,} reads per complete training + prediction task   |   "
             "one-way routed distance in grid-node hops (1 hop = 1 µm)", color="#444444", fontsize=11)

    bins = np.array(binned_counts(counts))
    edges = np.array(BIN_EDGES)
    ax_h.bar(edges[:-1], bins, width=np.diff(edges), align="edge", color=blue,
             edgecolor="white", linewidth=.65, zorder=3)
    ax_h.set_yscale("log")
    ax_h.set_ylim(1, 8e9)
    ax_h.set_ylabel("Reads per bin (log scale)")
    ax_h.set_title("Histogram · full distance range", loc="left", pad=12, fontweight="bold")
    ax_h.grid(axis="y", alpha=.18, zorder=0)
    ax_h.axvline(256, color="#666666", linewidth=.8, linestyle=":")
    ax_h.text(.60, .42, "18,772 reads beyond 256 hops\n0.000925% of all reads\nInput staging only",
              transform=ax_h.transAxes, fontsize=9, color="#444444", linespacing=1.6)

    # Resolve the two adjacent hot distances that share the first 8-hop bin.
    inset = ax_h.inset_axes([.48, .62, .49, .32])
    close = distances <= 128
    inset.bar(distances[close], weights[close] / total * 100, width=1, color=blue)
    inset.set_xlim(30, 129)
    inset.set_ylim(0, 32)
    inset.set_xticks([32, 64, 96, 128])
    inset.set_yticks([0, 15, 30], ["0%", "15%", "30%"])
    inset.tick_params(labelsize=8)
    inset.set_title("Nearby reads · 1-hop bins", fontsize=9, pad=6)
    inset.grid(axis="y", alpha=.15)

    # The CDF uses every exact distance, not the plotted histogram's bins.
    x = np.r_[31, distances, BIN_EDGES[-1]]
    y = np.r_[0, cumulative, 1]
    ax_c.step(x, y, where="post", color=red, linewidth=2)
    ax_c.set_ylim(0, 1.04)
    ax_c.yaxis.set_major_formatter(PercentFormatter(1))
    ax_c.set_ylabel("Fraction of reads at distance ≤ x")
    ax_c.set_title("CDF · exact, unbinned counts", loc="left", pad=12, fontweight="bold")
    ax_c.grid(alpha=.18)
    for numerator, denominator in ((1, 2), (9, 10), (99, 100)):
        q = quantile(counts, numerator, denominator)
        cdf = sum(count for distance, count in counts.items() if distance <= q) / total
        ax_c.plot(q, cdf, "o", color=red, markersize=4)
    ax_c.text(.42, .22, "Median    33 hops\n90%        57 hops\n99%        99 hops\nMaximum   16,032 hops",
              transform=ax_c.transAxes, fontsize=11, linespacing=1.8,
              bbox={"boxstyle": "round,pad=.7", "facecolor": "#faf5f3", "edgecolor": "none"})
    for axis in (ax_h, ax_c):
        axis.set_xscale("log", base=2)
        axis.set_xlim(31, 18000)
        axis.set_xticks([32, 64, 128, 256, 1024, 4096, 16384])
        axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
        axis.tick_params(axis="x", labelsize=9)
        axis.set_xlabel("One-way distance (grid-node hops; log scale)", labelpad=9)
    fig.text(.067, .065, "Histogram: 8-hop bins over [32, 256); doubling-width bins over [256, 16,384). "
             "Bar height is count per bin, not density.", fontsize=9, color="#555555")
    fig.text(.067, .027, "Includes input-copy sources and final output scratch reads. "
             "Excludes writes and tape-link transfers. CDF weights reads equally, not by energy.",
             fontsize=9, color="#555555")
    fig.savefig(HERE / "read_distance.png", dpi=180, facecolor="white")
    fig.savefig(HERE / "read_distance.svg", facecolor="white", metadata={"Date": None})
    svg = HERE / "read_distance.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)


def main():
    program_path = SCORER / "small60/program.spatial.json"
    score_path = SCORER / "small60/grid-score.json"
    document = json.loads(program_path.read_text())
    frozen = json.loads(score_path.read_text())
    sha256 = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    # Refuse to silently plot a changed program, placement, or counting source.
    assert sha256(program_path) == frozen["program_file_sha256"]
    for name, digest in frozen["source_sha256"].items():
        assert sha256(SCORER / name) == digest, name
    components = collect_read_counts(document)
    counts = combine_counts(components)
    totals = {name: sum(value.values()) for name, value in components.items()}
    source = frozen["components"]
    assert totals == {
        "normal_operands": source["reads"]["scratch_accesses"],
        "output_copy_sources": source["output_sources"]["scratch_accesses"],
        "input_stage_sources": frozen["input_tape_words"],
        "output_send_sources": frozen["output_tape_words"],
    }
    total = sum(counts.values())
    distance_sum = sum(distance * count for distance, count in counts.items())
    # All legal cells have d >= 32, making the 50 fJ local floor inactive.
    # Remove recv's tape+stage-write charge and add send's local scratch read.
    expected_energy = (source["reads"]["energy_fj"] + source["output_sources"]["energy_fj"]
                       + source["input_stage_and_tape"]["energy_fj"]
                       - 128 * frozen["input_tape_words"] + 64 * frozen["output_tape_words"])
    assert 2 * distance_sum == expected_energy
    summary = {
        "scope": "One complete task; every executed scratch source operand read, including tape staging copies and final send scratch reads. Excludes writes and tape-link transfers. Same fixed schedule for all 11 draws.",
        "distance_definition": "One-way routed grid-node hops = 128 * Manhattan processor links + owning-core-to-cell Manhattan distance; 1 hop = 1 micrometre. Not direct geometric core-to-cell displacement.",
        "cdf_definition": "Fraction of read count at distance <= x, using exact per-distance counts; not energy weighted.",
        "model_spec_commit": frozen["model_spec_commit"],
        "program_file_sha256": sha256(program_path),
        "grid_score_file_sha256": sha256(score_path),
        "scorer_source_sha256": frozen["source_sha256"],
        "component_reads": totals,
        "total_reads": total,
        "minimum_grid_hops": min(counts),
        "maximum_grid_hops": max(counts),
        "distinct_distances": len(counts),
        "sum_one_way_grid_hops": distance_sum,
        "mean_grid_hops": distance_sum / total,
        "quantile_grid_hops": {name: quantile(counts, num, den) for name, num, den in
                               (("p50", 1, 2), ("p90", 9, 10), ("p95", 19, 20),
                                ("p99", 99, 100), ("p99.9", 999, 1000),
                                ("p99.99", 9999, 10000), ("p99.999", 99999, 100000))},
        "reads_beyond_256_grid_hops": sum(count for distance, count in counts.items() if distance > 256),
        "scratch_read_energy_fj": 2 * distance_sum,
        "total_grid_energy_fj": frozen["energy_fj"],
        "bin_intervals": "[lower, upper); count per bin, not density. Inset uses one-hop bins centred on integer distance.",
        "bin_edges_grid_hops": BIN_EDGES,
        "validation": "Program and scorer source hashes match frozen score; component read totals match; twice the weighted distance sum exactly matches the frozen scratch-read energy components; bins preserve total read count.",
    }
    write_data(components, summary)
    plot(counts)
    print(json.dumps({key: summary[key] for key in
                      ("total_reads", "quantile_grid_hops", "scratch_read_energy_fj")}, indent=2))


if __name__ == "__main__":
    main()
