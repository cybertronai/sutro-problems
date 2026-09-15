#!/usr/bin/env python3
"""Recompute an energy_benchmark.py result without a GPU: python energy_verify.py FILE.json."""
import argparse
import hashlib
import json
import statistics

import numpy as np


def analyze(doc):
    assert doc['measurement_complete'] and not doc['monitor_errors']
    assert doc['positive_control_output_verified']
    plan = doc['plan']
    assert hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(',', ':')).encode()).hexdigest() == doc['plan_sha256']
    assert doc['ptx_sha256'] == '38d36146e4303b7acd640ee6c04ba642c751182c4fb9501b9fbccd52333cba58'
    for kind, correct in [('original', 7465), ('fresh', 7474)]:
        draws = doc['validation'][kind]
        assert len(draws) == 11 and sum(r['matches'] for r in draws) == 11000
        assert sum(r['correct'] for r in draws) == correct
        assert all(len(r['graph_replays_equal_eager']) == 3 and all(r['graph_replays_equal_eager']) for r in draws)
    owner = doc['pid_ownership']
    assert not owner['before_context']
    allocated = owner['after_128MiB_allocation']
    assert len(allocated) == 1 and allocated[0]['pid'] == owner['nvml_pid']
    assert allocated[0]['memory_bytes'] - sum(p['memory_bytes'] for p in owner['before_probe']) >= 128*1024**2
    assert allocated[0]['memory_bytes'] - sum(p['memory_bytes'] for p in owner['after_release']) >= 128*1024**2
    assert doc['process_samples']
    assert all(p['pid'] == owner['nvml_pid'] for row in doc['process_samples'] for p in row['processes'])
    columns = {name: index for index, name in enumerate(doc['trace_columns'])}
    trace = np.array(sorted(doc['trace'], key=lambda row: row[columns['time_s']]), dtype=np.float64)
    times, power = trace[:, columns['time_s']], trace[:, columns['power_w']]
    assert np.isfinite(trace).all() and (power >= 0).all()
    assert np.all(np.diff(times) > 0)
    intervals = {}
    for row in doc['intervals']:
        a, b = row['start']['time_s'], row['end']['time_s']
        energy = (row['end']['energy_mj'] - row['start']['energy_mj'])/1000
        assert b > a and energy >= 0 and times[0] <= a < b <= times[-1]
        target = np.r_[a, times[(times > a) & (times < b)], b]
        integral = float(np.trapezoid(np.interp(target, times, power), target))
        intervals[row['name']] = {**row, 'seconds': b-a, 'counter_j': energy, 'power_j': integral}
    rounds = []
    for comparison in doc['comparisons']:
        before, active, after = [intervals[comparison[k]] for k in ('before', 'active', 'after')]
        count = comparison['nominal_tasks']
        for baseline in (before, after):
            assert baseline['seconds'] >= plan['idle_seconds']
            settle = intervals[baseline['name'] + '-settle']
            assert settle['seconds'] >= plan['settle_seconds'] and settle['end']['time_s'] <= baseline['start']['time_s']
        row = {'kind': comparison['kind'], 'name': active['name'], 'replays': active['replays'],
               'nominal_tasks': count, 'cuda_ms': active['cuda_ms_per_replay'],
               'wall_ms': active['seconds']*1000/count}
        for meter in ('counter', 'power'):
            idle = statistics.mean(part[meter+'_j']/part['seconds'] for part in (before, after))
            row[meter] = {'active_w': active[meter+'_j']/active['seconds'], 'idle_w': idle,
                          'gross_mj': active[meter+'_j']*1000/count,
                          'net_mj': (active[meter+'_j']-idle*active['seconds'])*1000/count}
        if row['kind'] == 'matmul':
            row['tflops_s'] = 2*plan['positive_control']['matrix_dimension']**3*active['replays']/active['seconds']/1e12
        rounds.append(row)
    assert [r['kind'] for r in rounds] == plan['sequence'] + ['matmul']
    active = [r for r in rounds if r['kind'] == 'ptx']
    assert all(r['replays'] == plan['ptx_replays'] and r['nominal_tasks'] == plan['ptx_replays'] for r in active)
    control = rounds[-1]
    sanity = all(control[m]['active_w'] - control[m]['idle_w'] >= plan['positive_control']['minimum_above_idle_w']
                 for m in ('counter', 'power'))
    result = {'telemetry_sanity_passed': sanity,
              'ptx_cuda_ms_median': statistics.median(r['cuda_ms'] for r in active), 'rounds': rounds}
    for meter in ('counter', 'power'):
        values = [r[meter]['net_mj'] for r in active]
        result[meter] = {'net_mj_median': statistics.median(values), 'net_mj_range': [min(values), max(values)],
                         'net_mj_sample_sd': statistics.stdev(values),
                         'gross_mj_median': statistics.median(r[meter]['gross_mj'] for r in active)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('result')
    args = parser.parse_args()
    with open(args.result) as stream:
        doc = json.load(stream)
    result = analyze(doc)
    assert result == doc['summary'], 'Saved summary differs from raw reconstruction'
    assert result['telemetry_sanity_passed'], 'Positive-control telemetry sanity check failed'
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
