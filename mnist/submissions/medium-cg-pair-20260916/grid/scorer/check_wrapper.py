#!/usr/bin/env python3
"""Check no-cache counts and scores against shared cached and expanded counts."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from score_program import compare_scores, histogram_counts_nocache, load_shared, score_uncached, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shared-scorer', type=Path, required=True)
    parser.add_argument('--reduced-program', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    shared = load_shared(args.shared_scorer)
    from affine import Program, expand, ins, loop, make_program, ref

    # More than 250 I/O words exercises tape wrap. The second region is remote.
    # Reversed affine coefficients, duplicated sources, broadcast dimensions,
    # empty loops and immediate-only writes cover histogram edge conditions.
    toy = make_program([('near', 13000), ('far', 300)], [
        loop('z', 13000, [ins('set', ref('near', z=1), 0)]),
        loop('in', 300, [ins('recv', ref('far', **{'in': 1}))]),
        loop('repeat', 3, [loop('i', 300, [
            ins('add', ref('near', i=1), ref('far', 299, i=-1), ref('far', 299, i=-1)),
            ins('copy', ref('near', 500), ref('near', i=1)),
        ])]),
        loop('empty', 0, [ins('copy', ref('near'), ref('far'))]),
        loop('out', 300, [ins('send', ref('near', out=1))]),
    ], {'test': 'negative strides, repeats, broadcasts, zero loops, tape wrap and remote memory'})
    program = Program(toy)
    cached_counts, _ = shared.histogram_counts(program)
    uncached_counts, _ = histogram_counts_nocache(program)
    expanded = {name: Counter() for name in cached_counts}
    for instruction in expand(toy):
        opcode, *operands = instruction
        if opcode == 'recv':
            expanded['input_destinations'][operands[0]] += 1
        elif opcode == 'send':
            expanded['output_sources'][operands[0]] += 1
        else:
            expanded['writes'][operands[0]] += 1
            if opcode != 'set':
                expanded['reads'].update(operands[1:])
    for name in cached_counts:
        np.testing.assert_array_equal(cached_counts[name], uncached_counts[name])
        expected = np.zeros(program.words + 1, dtype=np.int64)
        for address, count in expanded[name].items():
            expected[address] = count
        np.testing.assert_array_equal(uncached_counts[name], expected)
    report = {
        'toy_cached_uncached_and_expanded_address_counts_equal': True,
        'toy_score': compare_scores(shared.score(toy), score_uncached(shared, toy)),
    }
    reduced = json.loads(args.reduced_program.read_text())
    result = score_uncached(shared, reduced)
    report['reduced_program'] = {
        'program_file_sha256': sha256(args.reduced_program),
        'program_sha256': result['program_sha256'],
        'configuration': reduced['metadata'],
        'program_scratch_words': result['program_scratch_words'],
        'energy_fj': result['energy_fj'],
        'cycles': result['cycles'],
        'comparison': compare_scores(shared.score(reduced), result),
    }
    report['source_sha256'] = {Path(path).name: sha256(path)
                               for path in (Path(__file__), Path(__file__).with_name('score_program.py'),
                                            Path(shared.__file__), Path(shared.__file__).with_name('affine.py'),
                                            Path(shared.__file__).with_name('model_ir.py'))}
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
