"""Independent CPU-only reconstruction of the two frozen initial reviewer sets.

No learner import, Modal call, replacement, repair, or mutation of evidence.
Canonical judge/energy_summary are replayed alongside separately written literal
integer, timing-floor, dispersion, and energy arithmetic.
"""
import argparse
from collections import Counter
import datetime
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import statistics
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SCORER_SHA = '1bc6d8d96a776565bc70f3ccdb91be7de57187d4495b2b0437fac98f5e120064'
RUNNER_SHA = '1180b843cf6c32414774da90c835a8b7298d2db8719989237a793ca1c62a10ab'


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path): return json.loads(Path(path).read_bytes())
def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def require(condition, message):
    if not condition: raise ValueError(message)
def close(a, b): return isinstance(a, (int, float)) and math.isfinite(a) and abs(a-b) <= max(1e-9, abs(b)*1e-12)
def mean(values): return sum(values)/len(values)
def threshold(n, bp): return (n*(10000-bp)+9999)//10000


def literal_score(calls, band, holdout):
    """Reimplement critical acceptance/count/clock arithmetic without judge()."""
    groups = {'mnist': [c for c in calls if c['dataset'] == 'mnist'],
              'holdout': [c for c in calls if c['dataset'] == holdout]}
    checks = {'exact15calls': len(calls) == 15, 'exact11MNIST': len(groups['mnist']) == 11,
              'exact4foreign': len(groups['holdout']) == 4,
              'only_declared_datasets': all(c['dataset'] in ['mnist', holdout] for c in calls),
              'foreign_valid': holdout in ['fashion', 'kmnist']}
    literal = []
    for index, c in enumerate(calls, 1):
        valid = (type(c['correct']) is int and type(c['total']) is int and c['total'] == 10000
                 and 0 <= c['correct'] <= 10000 and all(type(c[k]) in [int, float] and math.isfinite(c[k])
                     and c[k] >= 0 for k in ['ms', 'parent_ms', 'overhead_ms']))
        corrected = c['parent_ms'] - c['overhead_ms']
        unexplained = corrected - c['ms']
        clock_ok = unexplained <= 50 + .5*c['ms'] and c['ms'] <= c['parent_ms'] + 1
        checks[f'call{index}_valid_counts_clocks'] = valid and clock_ok and c['ms'] <= 60000
        literal.append({'index': index, **c, 'parent_less_protocol_ms': corrected,
                        'unexplained_ms': unexplained, 'maximum_unexplained_ms': 50 + .5*c['ms'],
                        'clock_gate_passed': clock_ok})
    reconstructed = {}
    for name, rows in groups.items():
        if not rows: continue
        correct = sum(c['correct'] for c in rows); total = sum(c['total'] for c in rows)
        needed = threshold(total, band if name == 'mnist' else 8500)
        per_draw = threshold(10000, band + 150) if name == 'mnist' else None
        times = [c['ms'] for c in rows]; median = statistics.median(times)
        kept = sorted(enumerate(rows), key=lambda x: x[1]['parent_ms']-x[1]['overhead_ms']-x[1]['ms'])[:len(rows)-max(1, len(rows)//5)] if len(rows) >= 3 else list(enumerate(rows))
        kept_reported = mean([c['ms'] for _, c in kept])
        kept_parent = mean([c['parent_ms']-c['overhead_ms'] for _, c in kept])
        reported = mean(times); corrected = mean([c['parent_ms']-c['overhead_ms'] for c in rows])
        floor = corrected - .1 - .005*reported
        checks[name+'_accuracy'] = correct >= needed
        checks[name+'_per_draw'] = name != 'mnist' or min(c['correct'] for c in rows) >= per_draw
        checks[name+'_trimmed_parent_clock'] = kept_parent-kept_reported <= 3 + .15*kept_reported
        checks[name+'_dispersion'] = max(times) <= 2*median+2 and min(times) >= median/2-2
        reconstructed[name] = {'correct': correct, 'total': total, 'required_correct': needed,
            'accuracy_margin_labels': correct-needed, 'accuracy_pct': correct/total*100,
            'per_draw_floor_correct': per_draw, 'weakest_correct': min(c['correct'] for c in rows),
            'reported_ms_mean': reported, 'parent_less_protocol_ms_mean': corrected,
            'parent_floor_ms': floor, 'floor_applied': floor > reported, 'ranked_group_ms': max(reported, floor),
            'median_reported_ms': median, 'min_reported_ms': min(times), 'max_reported_ms': max(times),
            'trimmed_kept_group_indices': [i+1 for i, _ in kept], 'trimmed_reported_mean_ms': kept_reported,
            'trimmed_parent_less_protocol_mean_ms': kept_parent}
    return {'checks': checks, 'literal_calls': literal, 'groups': reconstructed,
            'unqualified_ranked_ms': max(v['ranked_group_ms'] for v in reconstructed.values()) if reconstructed else None}


def literal_energy(energy, band, timed_mnist_median):
    if not energy or not energy.get('windows'):
        return {'valid': False, 'reason': 'No complete energy-window record; energy is optional', 'recorded': energy}
    windows = energy['windows']; w = {x['name']: x for x in windows}
    require(len(w) == len(windows) == 7 and set(w) == {'idle 1', 'reference', 'idle 2', 'control', 'idle 3', 'method', 'idle 4'}, 'Unexpected energy window inventory')
    watts = {k: v['joules']/v['seconds'] for k, v in w.items()}
    net = lambda name, before, after: w[name]['joules']-(watts[before]+watts[after])/2*w[name]['seconds']
    method = w['method']; ref = w['reference']; control = w['control']
    require(type(method['calls']) is int and method['calls'] > 0 and len(method['ms']) == len(method['correct']) == method['calls'], 'Incomplete energy-call vectors')
    require(type(control['calls']) is int and control['calls'] > 0, 'Invalid energy control count')
    flops = 2*ref['dim']**3*ref['matmuls']; tflops = flops/1e12/ref['seconds']
    j_per_tf = net('reference', 'idle 1', 'idle 2')/(flops/1e12)
    control_mj = net('control', 'idle 2', 'idle 3')*1000/control['calls']
    method_mj = net('method', 'idle 3', 'idle 4')*1000/method['calls']
    result = method_mj-control_mj; method_median = statistics.median(method['ms'])
    checks = {'finite_nonnegative_windows': all(math.isfinite(x['seconds']) and math.isfinite(x['joules']) and x['seconds'] > 0 and x['joules'] >= 0 for x in windows),
              'exact_matmul': ref['exact'] is True, 'reference_j_per_tflop': 6 <= j_per_tf <= 11,
              'reference_tflops': 15 <= tflops <= 23,
              'idle_utilization_zero': all(x['utilization_max'] == 0 for x in windows if x['name'].startswith('idle')),
              'idle_contexts_at_most_one': all((x['contexts'] or 0) <= 1 for x in windows if x['name'].startswith('idle')),
              'energy_correct_vector': all(type(x) is int and threshold(10000, band+150) <= x <= 10000 for x in method['correct']),
              'energy_call_times': all(type(x) in [int, float] and math.isfinite(x) and x >= 0 for x in method['ms']),
              'energy_dispersion': timed_mnist_median/2-2 <= method_median <= 2*timed_mnist_median+2,
              'positive_net_energy': result > 0}
    return {'valid': all(checks.values()), 'checks': checks, 'rebuilt_mj_per_call': result,
            'method_net_mj_before_control': method_mj, 'control_mj_per_call': control_mj,
            'reference_tflops': tflops, 'reference_j_per_tflop': j_per_tf, 'energy_calls': method['calls'],
            'energy_correct': sum(method['correct']), 'energy_total': method['calls']*10000,
            'energy_median_ms': method_median, 'energy_window_names': [x['name'] for x in windows],
            'GPU_device': energy['device']}


def audit_role(role, canonical):
    folder = HERE/role; planpath = folder/'measurement-plan.json'; plan = load(planpath)
    source_path = Path(plan['command'][3].rsplit(':', 1)[0]); source = source_path.read_bytes()
    checks = {'required_runs3': plan['required_runs'] == 3, 'source_identity': sha(source_path) == plan['source_sha256'],
              'stock_runner': plan['stock_runner'] is True and '--no-energy' not in plan['command'],
              'source_frozen_before_launch': plan['source_frozen_before_launch'] is True}
    for name, expected in plan['input_sha256'].items(): checks['input_pin:'+name] = sha(name) == expected
    canonical.check_source(source, plan['function'], source_path.name)
    controller_path = folder/'modal/controller-status.json'; controller = load(controller_path)
    checks.update(controller_terminal=controller.get('running') is False,
                  controller_plan=controller['measurement_plan_sha256'] == sha(planpath),
                  controller_command=controller['command'] == plan['command'])
    rows = []
    for number in range(1, 4):
        path = folder/f'modal/run-{number}.json'
        if not path.exists():
            rows.append({'run': number, 'present': False, 'score_passed': False, 'energy_valid': False,
                         'failure': 'No raw record after controller termination'}); continue
        result = load(path); record = result.get('record'); stderr = result.get('stderr', '')
        row = {'run': number, 'path': str(path.relative_to(ROOT)), 'raw_sha256': sha(path), 'present': True,
               'returncode': result.get('returncode'), 'score_passed': False, 'energy_valid': False,
               'stderr_bytes': len(stderr.encode()), 'stderr_sha256': hashlib.sha256(stderr.encode()).hexdigest(),
               'stderr_tail': stderr[-1500:], 'stderr_exception_markers': re.findall(r'Traceback|RuntimeError|OutOfMemoryError|CUDA error|SIGKILL|Killed', stderr)}
        if record is None:
            row['failure'] = 'Process returned no saved scorer record'; rows.append(row); continue
        band = {4: 230, 5: 190}[plan['difficulty']]
        headers = {'successful_return': result.get('returncode') == 0, 'source_sha256': record['source_sha256'] == sha(source_path),
                   'file': record['file'] == source_path.name, 'function': record['function'] == plan['function'],
                   'version': record['version'] == canonical.VERSION, 'band': record['band_bp'] == band,
                   'sandboxed': record['sandboxed'] is True, 'A100': str(record.get('device')).startswith('NVIDIA A100')}
        literal = literal_score(record.get('calls', []), band, record['holdout'])
        row.update(record_headers={key: record.get(key) for key in ['version', 'file', 'function', 'source_sha256', 'band_bp', 'sandboxed', 'holdout', 'device', 'torch']},
                   header_checks=headers, literal_reconstruction=literal, record_problems=record.get('problems'))
        complete = all(literal['checks'][key] for key in ['exact15calls', 'exact11MNIST', 'exact4foreign', 'only_declared_datasets', 'foreign_valid'])
        if complete:
            calls = [canonical.Call(**c) for c in record['calls']]
            problems, ranked, summary = canonical.judge(calls, band)
            parity = {'literal_rank_vs_canonical': close(literal['unqualified_ranked_ms'], ranked),
                      'record_problems_exact': problems == record['problems'], 'record_summary_exact': summary == record['summary'],
                      'record_rank_exact': record['ranked_ms'] == (None if problems else ranked),
                      'literal_pass_agrees_with_canonical': all(literal['checks'].values()) == (not problems)}
            row.update(canonical_problems=problems, canonical_ranked_ms=ranked, parity_checks=parity)
            row['score_passed'] = all(headers.values()) and all(literal['checks'].values()) and not problems and all(parity.values())
            energy = record.get('energy'); rebuilt = literal_energy(energy, band, statistics.median(c.ms for c in calls if c.dataset == 'mnist'))
            if energy and energy.get('windows'):
                canonical_energy = canonical.energy_summary(energy['windows'], energy['device'], band,
                    statistics.median(c.ms for c in calls if c.dataset == 'mnist'))
                energy_parity = {'canonical_record_exact': json.loads(json.dumps(canonical_energy)) == energy,
                    'literal_mj_exact': close(rebuilt['rebuilt_mj_per_call'], canonical_energy['net_mj_before_control']-canonical_energy['control_mj_per_call']),
                    'literal_validity_agrees': rebuilt['valid'] == (not canonical_energy['problems'] and canonical_energy['mj_per_call'] is not None)}
                rebuilt['parity_checks'] = energy_parity; rebuilt['canonical_problems'] = canonical_energy['problems']
                row['energy_valid'] = rebuilt['valid'] and all(energy_parity.values())
                row['GPU_uuid'] = energy['device'].get('uuid')
            row['energy_reconstruction'] = rebuilt
        rows.append(row)
    logpath = folder/'modal/controller.log'; log = logpath.read_text(errors='replace')
    observed = [{'line': i+1, 'text': line} for i, line in enumerate(log.splitlines()) if re.search(r'preempt|lost|Traceback|Error|cancel|terminat|retry|fail|passed .* runs|App completed|App stopped|Stopping app|https://modal.com/apps', line, re.I)]
    passed = sum(r['score_passed'] for r in rows); energy_valid = sum(r['energy_valid'] for r in rows)
    issues = [k for k, v in checks.items() if not v]
    if controller.get('returncode') != (0 if passed == 3 else 1): issues.append('controller returncode disagrees with saved score outcomes')
    uuids = [r.get('GPU_uuid') for r in rows if r.get('GPU_uuid')]
    return {'role': role, 'function': plan['function'], 'difficulty': plan['difficulty'], 'source_sha256': sha(source_path),
            'measurement_plan_sha256': sha(planpath), 'preflight_checks': checks, 'passed_runs': passed,
            'energy_valid_runs': energy_valid, 'required_runs': 3, 'complete_raw_runs': sum(r['present'] for r in rows),
            'median_ranked_ms': statistics.median(r['canonical_ranked_ms'] for r in rows) if passed == 3 else None,
            'median_energy_mj': statistics.median(r['energy_reconstruction']['rebuilt_mj_per_call'] for r in rows) if passed == 3 and energy_valid == 3 else None,
            'GPU_uuids': uuids, 'distinct_physical_GPU_ids': len(set(x.lower().removeprefix('gpu-') for x in uuids)),
            'GPU_ids_scope': 'Physical IDs observed in retained energy records, not proof of randomized hardware assignment or an added score gate.',
            'controller': controller, 'controller_sha256': sha(controller_path), 'controller_log_sha256': sha(logpath),
            'infrastructure_log_events': observed, 'issues': issues, 'runs': rows}


def terminal():
    try: return all(load(HERE/role/'modal/controller-status.json').get('running') is False for role in ['d4', 'd5_244'])
    except (OSError, json.JSONDecodeError): return False


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--wait-seconds', type=int, default=0)
    args = parser.parse_args(); deadline = time.monotonic()+args.wait_seconds; previous = None
    while not terminal():
        state = {role: {'raw_records': len(list((HERE/role/'modal').glob('run-*.json')))} for role in ['d4', 'd5_244']}
        if state != previous: print(json.dumps({'utc': now(), 'waiting_for_initial_groups': state}), flush=True); previous = state
        if time.monotonic() >= deadline:
            print(json.dumps({'status': 'PENDING_CONTROLLERS', 'actual_final_audit_written': False}), flush=True); return 2
        time.sleep(min(20, max(.01, deadline-time.monotonic())))
    require(sha(ROOT/'canonical/mnist.py') == SCORER_SHA and sha(ROOT/'canonical/run_modal.py') == RUNNER_SHA, 'Frozen canonical hashes changed')
    spec = importlib.util.spec_from_file_location('independent_oct25_canonical', ROOT/'canonical/mnist.py')
    canonical = importlib.util.module_from_spec(spec); sys.modules[spec.name] = canonical; spec.loader.exec_module(canonical)
    roles = {}
    for role in ['d4', 'd5_244']:
        try:
            roles[role] = audit_role(role, canonical)
        except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError) as error:
            # An incomplete/corrupt setup or worker outcome is evidence, not a
            # reason to wait for a replacement or silently omit the whole set.
            raws = sorted((HERE/role/'modal').glob('run-*.json'))
            roles[role] = {'role': role, 'passed_runs': 0, 'energy_valid_runs': 0,
                'complete_raw_runs': len(raws), 'median_ranked_ms': None, 'median_energy_mj': None,
                'distinct_physical_GPU_ids': 0, 'issues': [type(error).__name__+': '+str(error)],
                'raw_sha256': {str(p.relative_to(ROOT)): sha(p) for p in raws},
                'scope': 'Audit could not fully reconstruct this retained terminal set; no replacement requested.'}
    report = {'format': 'sutro-pr140-actual-independent-audit-v1', 'checked_utc': now(),
              'status': 'PASS' if all(r['passed_runs'] == 3 and not r['issues'] for r in roles.values()) else 'FAIL',
              'reviewed_head': '91cf2542e7229755edd2741efe0c1d68698ef2a2',
              'scorer_sha256': SCORER_SHA, 'runner_sha256': RUNNER_SHA, 'independent_auditor_sha256': sha(Path(__file__)),
              'roles': roles, 'all_initial_outcomes_retained': True,
              'energy_separate_from_score': True, 'scope': 'Canonical judge/energy_summary replay plus independent literal integer/count/clock-floor/dispersion/energy arithmetic. No PR-authored verifier or root aggregate trusted; no GPU/Modal call, learner import, rerun, replacement or evidence mutation.',
              'paid_runs_launched_or_replaced': 0}
    output = HERE/'actual-final-independent-audit.json'
    with output.open('x') as stream: stream.write(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'status': report['status'], 'output': str(output), 'sha256': sha(output),
                      'roles': {key: {k: value[k] for k in ['passed_runs', 'energy_valid_runs', 'median_ranked_ms', 'median_energy_mj', 'distinct_physical_GPU_ids', 'issues']} for key, value in roles.items()}}, indent=2), flush=True)
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__': raise SystemExit(main())
