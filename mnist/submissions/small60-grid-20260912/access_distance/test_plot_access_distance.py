"""Check compressed read counts against independently lowered tiny traces."""
from collections import Counter
import json
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
SCORER = HERE.parent.parent / "grid-mlp-scoring-20260912"
sys.path.insert(0, str(SCORER))

from affine import Program, expand, ins, loop, make_program, ref
from model_ir import build_mlp
from score import placement
from plot_access_distance import collect_read_counts


def expanded_read_counts(document):
    """Lower tape operations without using the scorer's schedule generator.

    Compute path length from processor and cell coordinates, independently of
    access energies. In particular, send's outgoing tape link is not a read.
    """
    program = Program(document)
    owners, cells, _, _, _ = placement(program.words)
    counts = {name: Counter() for name in (
        "normal_operands", "output_copy_sources", "input_stage_sources",
        "output_send_sources",
    )}

    def core(tile):
        return -16000 + 128 * int(tile[0]) + 64, 128 * int(tile[1]) + 64

    def charge(category, issuer, owner, cell):
        start, end = core(issuer), core(owner)
        point = (-16000 + 128 * int(owner[0]) + int(cell[0]),
                 128 * int(owner[1]) + 1 + int(cell[1]))
        mesh = abs(start[0] - end[0]) + abs(start[1] - end[1])
        local = abs(end[0] - point[0]) + abs(end[1] - point[1])
        counts[category][mesh + local] += 1

    incoming = outgoing = 0
    for opcode, *arguments in expand(document):
        if opcode == "recv":
            # Receive writes the stage; copying into program memory reads it.
            port = incoming % 250
            incoming += 1
            charge("input_stage_sources", (125, 0), (port, 0), (64, 31))
        elif opcode == "send":
            source = arguments[0]
            charge("output_copy_sources", (125, 0), owners[source], cells[source])
            port = outgoing % 250
            outgoing += 1
            charge("output_send_sources", (port, 0), (port, 0), (64, 31))
        elif opcode != "set":
            for source in arguments[1:]:
                charge("normal_operands", (125, 0), owners[source], cells[source])
    return counts


def combined(counts):
    result = Counter()
    for values in counts.values():
        result.update(values)
    return result


class ReadDistanceTests(unittest.TestCase):
    def test_tiny_mlp_matches_independent_expansion(self):
        document = build_mlp(features=3, width=4, n_train=4, n_test=3,
                             batch=2, epochs=2, learning_rate=.1, seed=101)
        actual = collect_read_counts(document)
        self.assertEqual(actual, expanded_read_counts(document))
        self.assertEqual(sum(combined(actual).values()), 7897)

    def test_port_wrap_and_remote_source_match_independent_expansion(self):
        document = make_program([("x", 12301)], [
            loop("i", 12301, [ins("set", ref("x", i=1), 0)]),
            loop("k", 253, [ins("recv", ref("x", 12000, k=1)),
                            ins("send", ref("x", 12000, k=1))]),
            ins("copy", ref("x", 12300), ref("x", 12252)),
            ins("send", ref("x", 12300)),
        ])
        actual = collect_read_counts(document)
        self.assertEqual(actual, expanded_read_counts(document))
        self.assertEqual(sum(combined(actual).values()), 762)
        self.assertEqual(actual["output_send_sources"], Counter({32: 254}))
        self.assertEqual(max(actual["input_stage_sources"]), 16032)
        self.assertGreater(max(actual["output_copy_sources"]), 128)

    def test_production_count_and_energy_match_frozen_score(self):
        document = json.loads((SCORER / "small60/program.spatial.json").read_text())
        score = json.loads((SCORER / "small60/grid-score.json").read_text())
        actual = collect_read_counts(document)
        self.assertEqual({name: sum(values.values()) for name, values in actual.items()}, {
            "normal_operands": 2029672000,
            "output_copy_sources": 1000,
            "input_stage_sources": 19000,
            "output_send_sources": 1000,
        })
        reads = combined(actual)
        self.assertEqual(sum(reads.values()), 2029693000)
        self.assertEqual((min(reads), max(reads)), (32, 16032))
        # All scratch distances exceed the energy floor. Remove tape transport
        # and recv's stage write, and add send's local scratch read alone.
        components = score["components"]
        expected_energy = (components["reads"]["energy_fj"]
                           + components["output_sources"]["energy_fj"]
                           + components["input_stage_and_tape"]["energy_fj"]
                           - 128 * score["input_tape_words"]
                           + 64 * score["output_tape_words"])
        self.assertEqual(2 * sum(d * count for d, count in reads.items()), expected_energy)
        self.assertEqual(expected_energy, 174613318112)


if __name__ == "__main__":
    unittest.main(verbosity=2)
