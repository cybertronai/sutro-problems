"""CPU-only, independent PR140 D5 source/record audit; never invokes submitted code or GPUs."""
import ast
from collections import Counter
from dataclasses import dataclass
import datetime
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE / 'source/mnist-a100/submissions/codesign-d4-d5-20261024'
CANONICAL = HERE.parent / 'canonical'
HEAD = '91cf2542e7229755edd2741efe0c1d68698ef2a2'
LATEST = 'd5_244/ladder-w480-s315-population-tb20-dec50-lr720-cut50-ph2-initboth-d5-20261024.py'
LATEST_SHA = 'a245b278c9f425ec951942caf13e4a14b7123717a428bc9862b13491f42fdc29'


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path): return json.loads(Path(path).read_text())
def close(x, y): assert math.isfinite(x) and abs(x-y) < 1e-8


def literal_judge(calls, band=190):
    """Fresh implementation of published judge arithmetic, without calling author verifier."""
    groups = [[c for c in calls if c['dataset'] == 'mnist'], [c for c in calls if c['dataset'] != 'mnist']]
    problems = []
    unexplained = lambda c: c['parent_ms']-c['overhead_ms']-c['ms']
    mean = statistics.mean
    for i, c in enumerate(calls):
        if unexplained(c) > 50+.5*c['ms'] or c['ms'] > c['parent_ms']+1:
            problems.append({'gate': 'per_call_clock', 'call': i+1})
    details = []
    for name, group in zip(['mnist', 'holdout'], groups):
        if len(group) >= 3:
            kept = sorted(group, key=unexplained)[:len(group)-max(1, len(group)//5)]
            reported = mean(c['ms'] for c in kept)
            measured = mean(c['parent_ms']-c['overhead_ms'] for c in kept)
            if measured-reported > 3+.15*reported:
                problems.append({'gate': 'group_clock', 'dataset_group': name})
            med = statistics.median(c['ms'] for c in group)
            if max(c['ms'] for c in group) > 2*med+2:
                problems.append({'gate': 'slow_dispersion', 'dataset_group': name})
            if min(c['ms'] for c in group) < med/2-2:
                problems.append({'gate': 'fast_dispersion', 'dataset_group': name})
        reported = mean(c['ms'] for c in group)
        floor = mean(c['parent_ms']-c['overhead_ms'] for c in group)-.1-.005*reported
        details.append({'group': name, 'calls': len(group), 'reported_mean_ms': reported,
                        'parent_minus_overhead_mean_ms': mean(c['parent_ms']-c['overhead_ms'] for c in group),
                        'clock_floor_ms': floor, 'ranked_group_ms': max(reported, floor),
                        'clock_floor_applied': floor > reported,
                        'correct': sum(c['correct'] for c in group), 'total': sum(c['total'] for c in group)})
    mnist, hold = groups
    correct = sum(c['correct'] for c in mnist); total = sum(c['total'] for c in mnist)
    need = -(-(total*(10000-band))//10000)
    if correct < need: problems.append({'gate': 'mean_mnist_accuracy', 'shortfall': need-correct})
    per_draw = -(-(10000*(10000-band-150))//10000)
    if min(c['correct'] for c in mnist) < per_draw:
        problems.append({'gate': 'per_draw_mnist_accuracy'})
    if sum(c['correct'] for c in hold) < -(-(sum(c['total'] for c in hold)*1500)//10000):
        problems.append({'gate': 'holdout_accuracy'})
    return {'problems': problems, 'ranked_ms': max(g['ranked_group_ms'] for g in details), 'groups': details,
            'correct': correct, 'total': total, 'required_correct': need, 'margin_labels': correct-need,
            'minimum_mnist_draw_correct': min(c['correct'] for c in mnist),
            'error_pct': (total-correct)*100/total}


def module():
    spec = importlib.util.spec_from_file_location('frozen_pr140_mnist_for_crosscheck', CANONICAL/'mnist.py')
    result = importlib.util.module_from_spec(spec); sys.modules[spec.name] = result; spec.loader.exec_module(result)
    return result


def audit_group(name, source, folder, canonical):
    source = ROOT/source; paths = sorted((ROOT/folder).glob('run-*.json'))
    assert [p.name for p in paths] == ['run-1.json', 'run-2.json', 'run-3.json']
    canonical.check_source(source.read_bytes(), 'ladder_kernel', source.name)
    rows = []
    for path in paths:
        saved = load(path); record = saved['record']; calls = record['calls']
        assert record['source_sha256'] == sha(source) and record['function'] == 'ladder_kernel' and record['file'] == source.name
        assert record['band_bp'] == 190 and record['sandboxed'] is True and 'A100' in record['device'] and '80GB' in record['device']
        assert record['version'] == canonical.VERSION
        assert len(calls) == 15 and Counter(c['dataset'] for c in calls) == Counter({'mnist': 11, record['holdout']: 4})
        assert record['holdout'] in ['fashion', 'kmnist'] and 'energy' not in record
        assert all(type(c['correct']) is int and c['total'] == 10000 and 0 <= c['correct'] <= 10000
                   and all(math.isfinite(c[k]) for k in ['ms', 'parent_ms', 'overhead_ms']) and 0 < c['ms'] <= 60000 for c in calls)
        own = literal_judge(calls)
        problems, ranked, _ = canonical.judge([canonical.Call(**c) for c in calls], 190)
        assert bool(problems) == bool(own['problems']) and problems == record['problems']
        close(ranked, own['ranked_ms'])
        passed = not own['problems']
        assert saved['returncode'] == (0 if passed else 1)
        if passed: close(record['ranked_ms'], own['ranked_ms'])
        else: assert record['ranked_ms'] is None
        rows.append({'file': str(path.relative_to(ROOT)), 'file_sha256': sha(path), 'device': record['device'],
                     'torch': record['torch'], 'holdout': record['holdout'], 'passed': passed,
                     'saved_problems': problems, **own})
    passed = sum(r['passed'] for r in rows); correct = sum(r['correct'] for r in rows); total = sum(r['total'] for r in rows)
    return {'name': name, 'source': str(source.relative_to(ROOT)), 'source_sha256': sha(source),
            'source_bytes': source.stat().st_size, 'passed_runs': passed, 'required_runs': 3,
            'complete_scored_calls': 45, 'rows': rows, 'pooled_correct': correct, 'pooled_total': total,
            'pooled_error_pct': (total-correct)*100/total, 'minimum_margin_labels': min(r['margin_labels'] for r in rows),
            'qualifying_author_median_ms': statistics.median(r['ranked_ms'] for r in rows) if passed == 3 else None,
            'energy_median_mj': None, 'energy_scope': 'Not measured; author canonical invocation uses --no-energy.'}


def kernel_partition_check():
    rows = []
    for sms in [58, 108, 132]:
        for width, (bc, sub, warps, blocks_per_sm, sb) in {480:(16,16,4,3,64), 60:(4,16,2,4,64), 256:(16,16,2,4,64)}.items():
            batch = 2500; ncb = width//bc
            chunks = max(1, (sms*blocks_per_sm-(4 if blocks_per_sm > 1 else 0))//ncb)
            row_size = math.ceil(math.ceil(batch/chunks)/sb)*sb
            nrs = math.ceil(batch/row_size); nprog = ncb*nrs; chunk = math.ceil(17*width/nprog)
            du = np.zeros((batch, width), np.int16); ws = np.zeros((nrs, 20, width), np.int16); dp = np.zeros(17*width, np.int16)
            for pid in range(nprog):
                col = pid % ncb; rs = pid//ncb; lo, hi = rs*row_size, min((rs+1)*row_size, batch)
                du[lo:hi, col*bc:(col+1)*bc] += 1
                ws[rs, :, col*bc:(col+1)*bc] += 1
                dp[pid*chunk:min((pid+1)*chunk, 17*width)] += 1
                assert row_size % sb == row_size % sub == 0
            assert np.all(du == 1) and np.all(ws == 1) and np.all(dp == 1) and width % bc == 0
            rows.append({'sms': sms, 'width': width, 'batch': batch, 'row_size': row_size, 'row_segments': nrs,
                         'programs': nprog, 'parameter_chunk': chunk, 'data_writes_exactly_once': True,
                         'workspace_terms_exactly_once': True, 'parameter_writes_exactly_once': True})
    return rows


def combinator_formula_check():
    rng = np.random.default_rng(261025140)
    u, lat, p, go = rng.normal(size=(7, 3)), rng.normal(size=(7, 3)), rng.normal(size=(17, 3)), rng.normal(size=(7, 3))
    leak = lambda x: np.where(x > 0, x, .1*x)
    def terms(u, lat, p):
        normalized = (u-u.mean(0))/np.sqrt(u.var(0)+1e-10); a = normalized*p[0]+lat*p[2]+normalized*lat*p[4]+p[6]
        b = normalized*p[1]+lat*p[3]+normalized*lat*p[5]+p[7]
        c, d = leak(a), leak(b); e, f = c*p[8]+d*p[10]+p[12], c*p[9]+d*p[11]+p[13]
        return normalized, a, b, c, d, e, f
    def forward(u, lat, p):
        _, _, _, _, _, e, f = terms(u, lat, p)
        return leak(e)*p[14]+leak(f)*p[15]+p[16]
    v, a, b, c, d, e, f = terms(u, lat, p)
    n, o = go*np.where(e > 0, 1., .1), go*np.where(f > 0, 1., .1)
    j = (n*p[14]*p[8]+o*p[15]*p[9])*np.where(a > 0, 1., .1)
    k = (n*p[14]*p[10]+o*p[15]*p[11])*np.where(b > 0, 1., .1)
    dv = j*p[0]+k*p[1]+(j*p[4]+k*p[5])*lat
    dl = j*p[2]+k*p[3]+(j*p[4]+k*p[5])*v
    du = (dv-dv.mean(0)-(v-v.mean(0))*(dv*v).mean(0))/np.sqrt(u.var(0)+1e-10)
    dp = np.stack([j*v,k*v,j*lat,k*lat,j*v*lat,k*v*lat,j,k,
                   n*c*p[14],o*c*p[15],n*d*p[14],o*d*p[15],n*p[14],o*p[15],n*e,o*f,go]).sum(1)
    errors = []
    for argument, analytic in enumerate([du, dl, dp]):
        variable = [u, lat, p][argument]; numeric = np.zeros_like(variable)
        for index in np.ndindex(variable.shape):
            original = variable[index]; variable[index] = original+1e-6; high = (forward(u, lat, p)*go).sum()
            variable[index] = original-1e-6; low = (forward(u, lat, p)*go).sum(); variable[index] = original
            numeric[index] = (high-low)/2e-6
        error = float(np.max(np.abs(numeric-analytic))); assert error < 2e-7; errors.append(error)
    v, lat = rng.normal(size=(1000,)), rng.normal(size=(1000,))
    p = np.array([.1,-.1,.9,-.9,0,0,0,0,1,0,0,1,0,0,1/1.01,-1/1.01,0])
    a, b = leak(v*p[0]+lat*p[2]), leak(v*p[1]+lat*p[3])
    out = leak(a*p[8]+b*p[10])*p[14]+leak(a*p[9]+b*p[11])*p[15]
    identity_error = float(np.max(np.abs(out-(.1*v+.9*lat)))); assert identity_error < 1e-14
    return {'finite_difference_max_absolute_errors_du_dlat_dp': errors,
            'near_identity_1000_random_pairs_max_absolute_error': identity_error,
            'scope': 'Independent CPU FP64 mathematical formula check. Not a Triton execution or BF16 bitwise parity test.'}


def main():
    canonical = module(); source = ROOT/LATEST; manifest = load(ROOT/'manifest.json')
    assert sha(source) == LATEST_SHA and source.stat().st_size == 20439
    assert sha(CANONICAL/'mnist.py') == manifest['scorer_sha256'] == '1bc6d8d96a776565bc70f3ccdb91be7de57187d4495b2b0437fac98f5e120064'
    assert sha(CANONICAL/'run_modal.py') == manifest['runner_sha256'] == '1180b843cf6c32414774da90c835a8b7298d2db8719989237a793ca1c62a10ab'
    canonical.check_source(source.read_bytes(), 'ladder_kernel', source.name)
    configs = [('latest_initboth', LATEST, 'd5_244/official'),
               ('historical_width512', 'd5/ladder-label-topdown-width512-s420-d5-20261022.py', 'd5/official'),
               ('earlier_decoder_combo315', 'research/ladder-w480-s315-population-tb20-dec50-lr720-cut50-d5-20261024.py', 'research/d5-decoder-combo315-official-20261024'),
               ('earlier_parallel_phase2', 'research/ladder-w480-s315-population-tb20-dec50-lr720-cut50-ph2-d5-20261024.py', 'research/d5-decoder-combo315-phase2-official-20261024')]
    groups = [audit_group(*config, canonical) for config in configs]
    logpath = ROOT/'d5_244/official/modal-launch.log.gz'; logbytes = gzip.decompress(logpath.read_bytes()); log = logbytes.decode()
    assert log.count('Container terminated due to preemption.') == 1 and 'restarted with the same input' in log
    assert sum(line.startswith('--- run ') for line in log.splitlines()) == 3 and 'passed 3 of 3 runs; median 243.411 ms' in log
    infrastructure = {'log_sha256': sha(logpath), 'uncompressed_log_sha256': hashlib.sha256(logbytes).hexdigest(),
                      'modal_app_id': 'ap-6WMzFM880X287hG8casCaN', 'preemptions_logged': 1, 'completed_logical_records': 3,
                      'agent_initiated_replacement_disclosed': False,
                      'scope': 'The retained log supports one automatic preemption restart with identical remote-function arguments, not an identical tensor draw: the scorer seeds a fresh run from os.urandom. No completed failure is present/replaced in this saved logical group. Partial work on a preempted physical attempt is not recoverable from returned records; absence of omitted historical attempts cannot be proved from final files alone.'}
    gates = {}
    for name in ['d5-comb-parallel-phase2-gate-a100-modal.json', 'd5-comb-parallel-phase2-gate-l4-modal.json']:
        item = load(ROOT/'research'/name); raw = json.loads(item['stdout'].split('RESULT_JSON ',1)[1])
        assert item['returncode'] == 0 and raw == item['result'] and raw['passed'] is True
        assert [r['width'] for r in raw['rows']] == [60,256,480] and all(r['changing_replays'] == 20 for r in raw['rows'])
        gates[name] = {'file_sha256': sha(ROOT/'research'/name), 'gpu': raw['gpu'], 'max_relative_gradient_errors_by_width': {str(r['width']):r['max_relative_gradient_errors'] for r in raw['rows']}, 'bitwise_exact': False}
    packaging = load(ROOT/'research/d5-decoder-init-packaging-a100-modal.json'); parity = load(ROOT/'research/d5-decoder-init-numeric-parity-a100-modal.json')
    assert packaging['passed'] is False and parity['passed'] is True and packaging['source_sha256'] == parity['source_sha256'] == LATEST_SHA
    for result in [packaging, parity]:
        for row in result['rows']:
            assert row['returncode'] == 0 and row['result'] == json.loads(row['stdout'].split('RESULT_JSON ',1)[1])
            assert row['result']['repeat_stable'] is True
    assert packaging['rows'][0]['result']['parameter_sha256'] != packaging['rows'][1]['result']['parameter_sha256']
    assert parity['rows'][0]['result']['parameter_sha256'] == parity['rows'][1]['result']['parameter_sha256']
    assert parity['rows'][0]['result']['prediction_sha256'] == parity['rows'][1]['result']['prediction_sha256']
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    author = {'format':'sutro-pr140-independent-d5-author-record-audit-v1','checked_utc':now,'reviewed_head':HEAD,
              'status':'PASS_AUTHOR_ARITHMETIC_NOT_INDEPENDENT_QUALIFICATION','latest_source_sha256':LATEST_SHA,
              'method':'Independent literal arithmetic for every accuracy/timing/dispersion/clock-floor gate, then cross-check against frozen canonical judge; author verify_records.py is not imported or executed.',
              'groups':groups,'infrastructure':infrastructure,'retained_history':{
                  'cancelled_batches':load(ROOT/'research/d5-244-post-completion-cancellation.json'),
                  'initial_packaging_parity_failed_retained':True,'later_exact_trained_parameter_prediction_parity_supported_by_saved_hashes':True,
                  'initial_parameter_exactness_scope':'The later summary reports initial exactness; exported row hashes independently demonstrate trained parameters/predictions, not full initial tensor arrays.',
                  'earlier_discrepancy_unexplained':True,'two_failed_groups_use_different_source_hashes_from_latest':True,
                  'no_permanent_source_ban_in_published_rules':True},
              'latest_qualifying_author_median_ms':groups[0]['qualifying_author_median_ms'],'energy_median_mj':None,
              'blockers':[],'qualification_scope':'Author records only. Saved calls contain canonical correct-count/timing summaries, not truth/prediction arrays to recount labels. Root owns one fresh independent three-run canonical group; no new GPU calls here.'}
    partitions = kernel_partition_check(); formula = combinator_formula_check()
    review = {'format':'sutro-pr140-independent-d5-source-review-v1','checked_utc':now,'reviewed_head':HEAD,
              'status':'PASS_STATIC_SOURCE_PENDING_FRESH_REVIEWER_GROUP','source':LATEST,'source_sha256':LATEST_SHA,
              'source_bytes':20439,'limit_bytes':20480,'remaining_bytes':41,'function':'ladder_kernel',
              'canonical_source_check':'PASS','advisory_review_flags':canonical.review_flags(source.read_bytes()),
              'published_rule_pins':{p.name:sha(p) for p in [CANONICAL/'mnist.py',CANONICAL/'run_modal.py',CANONICAL/'README.md']},
              'api_and_truth_boundary':{'arguments':['train_x','train_y','test_x'],'returned_labels':'argmax/guard integer tensor over test rows on same GPU','hidden_test_truth_access_found':False,'dataset_or_network_access_found':False,'external_io':'Only /tmp compiler-cache environment defaults are set. No file reads, network/process calls, scorer imports, prediction artifacts or pool access.','transduction':'Teacher, conditional decoder, hard-query sampling and hidden hull use unlabeled query coordinates plus train labels/predicted labels; no hidden query truth.'},
              'initialization':{'fixed_seed':11,'encoder_decoder_rng':'Every original encoder and decoder Gaussian draw is consumed; decoder values are then copied from fresh encoder transposes scaled by sqrt(input_width/output_width). No persistent tying after initialization.','combinator_rng':'Original random denoiser draws are consumed before the explicit17coefficient initializer overwrites them.','near_identity':'The fixed coefficients algebraically implement0.1*v+0.9*latent, a mathematical generative initialization, not frozen trained weights.','trained_offline_payload_found':False,'credit':'Explicitly inspired by sigkillme0 PR127 commit19d2a3faba642814aee9c12ed1a118a927b796d2, whose reset uses the same transpose/scaling idea and piecewise-linear near-identity initializer.'},
              'per_call_reset':{'teacher_cache':'Copies current train_x/train_y/test_x into graph buffers, replays all RFM fitting and returns cloned current outputs. No fitted teacher state skips work.','ladder_inputs':'Refreshes x, class-conditioned tangent bank, one-hot train labels, pooled coordinates, teacher probabilities and hard-query indices from current inputs.','learned_state':'A.R resets encoders/decoders/combinators/biases/gain and Adam exp_avg/exp_avg_sq/step; sampling history/counter, sampling/noise tables and RNG are recreated before315updates.','retained_cache_scope':'Allocations, compiled graphs, optimizer objects, fixed schedules and CUDA streams only; learned state is overwritten.','gradients':'Each captured update clears parameter grads; side-stream matmul backward joins the caller stream before returning.'},
              'custom_kernel_review':{'phase1':'One owner per(row segment,column block) writes its data gradients and20workspace terms:17parameter partials plus dv,dv*normalized_input,normalized_input sums.','phase2':'Contiguous pid*ceil(17*S/NPROG) chunks cover each of17*S parameter outputs once. Masked padded row reduction sums partials; each feature block reduces three BN sums over all row segments, then writes only its own input-row segment.','tail_safety':'Supported widths60/256/480 divide BC exactly; row masks bound2500rows, padded row-segment loads use NR masks, and phase launches are ordered on the same stream. M=1 is a fixed source constant.','cpu_partition_checks':partitions,'cpu_mathematical_gradient_and_initializer_checks':formula,'author_gpu_gradient_gates':gates,'scope':'Independent static indexing/formula review plus CPU checks; author supplied L4/A100 execution evidence is supporting data, not independently rerun GPU parity.'},
              'parity_and_history':author['retained_history'],
              'attribution':['Rasmus','SethTS','spartanhaden','sigkillme0','islamborghini','likelytobelaura','Vincent/Bengio','HazyResearch/TK','sjbaebae'],
              'documentation_notes':['The early implementation paragraph describes the historical width512/420-update candidate; the later D5_244 section correctly describes the selected width480/315 source. This could be clearer, but is not a compliance blocker.','Author median243.411ms is not a per-host maximum; the275.808ms SXM logical run is retained. Cross-host/draw changes prevent assigning the whole speedup to one kernel.','One Modal restart is transparently retained; same input means the remote job arguments, not the scorer entropy/tensor draw.'],
              'blockers':[],'disposition':'Source review approves the latest candidate for root-owned unchanged initial independent three-run canonical qualification. Publication/merge/record promotion still depends on root review and those independent outcomes; energy is separate.'}
    for name, value in [('d5-author-audit.json',author),('d5-source-review.json',review)]:
        value['audit_program_sha256'] = sha(Path(__file__))
        with (HERE/name).open('x') as stream:stream.write(json.dumps(value,indent=2)+'\n')
    print(json.dumps({'latest_author':{k:groups[0][k] for k in ['passed_runs','qualifying_author_median_ms','pooled_correct','pooled_total','pooled_error_pct','minimum_margin_labels']},
                      'historical_group_pass_counts':{g['name']:g['passed_runs'] for g in groups},
                      'cpu_formula':formula,'source_review_sha256':sha(HERE/'d5-source-review.json'),'author_audit_sha256':sha(HERE/'d5-author-audit.json')},indent=2))


if __name__ == '__main__': main()
