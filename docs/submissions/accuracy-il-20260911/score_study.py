"""Score every frozen MLP configuration and record repeatable host timings.

No test labels or learned parameters are loaded. Seed-only initialization literals
are part of the IL; all three seeds are checked for identical cost counts.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
from pathlib import Path
import statistics
import sys
import numpy as np
from il import score
from mlp_il import build_mlp

HERE = Path(__file__).resolve().parent


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HERE)
    parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 3:
        parser.error('At least three timing samples are required')
    args.output.mkdir(parents=True, exist_ok=True)
    shortlist = json.loads((HERE/'shortlist.json').read_text())
    rows = []
    for config in sorted(shortlist['candidates'], key=lambda c: (c['width'], c['epochs'])):
        path = args.output/(config['id'] + '-s101.il.json')
        document = build_mlp(config['width'], config['epochs'], config['learning_rate'], 101)
        write(path, document)
        samples = [score(document) for _ in range(args.repeats)]
        result = samples[0]
        fields = ['instructions', 'total_instructions', 'charged_reads', 'charged_writes',
                  'time_ticks_0_2_ps', 'energy_fj', 'area_um2_occupied_cells']
        for sample in samples[1:]:
            assert all(result[k] == sample[k] for k in fields)
        for seed in (102, 103):
            other = score(build_mlp(config['width'], config['epochs'], config['learning_rate'], seed))
            assert all(result[k] == other[k] for k in fields)
        result.update(config_id=config['id'], width=config['width'], epochs=config['epochs'],
                      learning_rate=config['learning_rate'], seed=101, program_file=path.name,
                      program_json_bytes=path.stat().st_size, all_three_seed_costs_equal=True,
                      static_score_seconds=[s['time_to_score_seconds'] for s in samples])
        result['median_static_score_seconds'] = statistics.median(result['static_score_seconds'])
        write(path.with_name(config['id'] + '.score.json'), result)
        rows.append(result)
        print(f"{config['id']}: {result['total_instructions']} instructions; "
              f"{result['median_static_score_seconds']:.2g}s static scoring", flush=True)
    baseline = json.loads((HERE/'1nn.il.json').read_text())
    baseline_samples = [score(baseline) for _ in range(args.repeats)]
    b = baseline_samples[0]
    assert (b['time_ticks_0_2_ps'], b['energy_fj']) == (8410986000, 1875974400)
    b['static_score_seconds'] = [s['time_to_score_seconds'] for s in baseline_samples]
    b['median_static_score_seconds'] = statistics.median(b['static_score_seconds'])
    b['program_json_bytes'] = (HERE/'1nn.il.json').stat().st_size
    result = {'purpose': 'Exact costs and static-scoring feasibility for the frozen accuracy shortlist',
              'software': {'python': sys.version, 'numpy': np.__version__, 'platform': platform.platform()},
              'shortlist_sha256': hashlib.sha256((HERE/'shortlist.json').read_bytes()).hexdigest(),
              'source_sha256': {name: hashlib.sha256((HERE/name).read_bytes()).hexdigest()
                                for name in ['score_study.py','il.py','mlp_il.py','accuracy_study.py']},
              'timing_repeats': args.repeats, 'baseline': b, 'configurations': rows,
              'timing_scope': b['time_to_score_scope'],
              'a100_time_ps': None, 'a100_energy_fj': None}
    write(args.output/'scoring_results.json', result)


if __name__ == '__main__':
    main()
