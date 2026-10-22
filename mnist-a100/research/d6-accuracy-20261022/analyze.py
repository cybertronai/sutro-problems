"""CPU-only audit of frozen D6 development records; no learner execution."""
import argparse
import array
import ast
import base64
import datetime
import hashlib
import gzip
import json
import math
from pathlib import Path
import statistics
import struct
import zipfile
import zlib

ROOT = Path(__file__).resolve().parent
EXPECTED_ARMS = 19
TOTAL = 10000


def sha(path):
    return hashlib.sha256(read_bytes(path)).hexdigest()


def existing(path):
    return path if path.exists() else Path(str(path)+'.gz') if Path(str(path)+'.gz').exists() else path


def read_bytes(path):
    path = existing(path)
    data = path.read_bytes()
    return gzip.decompress(data) if path.suffix == '.gz' else data


def load(path):
    return json.loads(read_bytes(path))


def labels_ok(values):
    return isinstance(values, list) and len(values) == TOTAL and all(type(v) is int and 0 <= v < 10 for v in values)


def finite_time(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def hexhash(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def audit_input_manifest(root, plan):
    """Read local immutable NPZ archives without importing NumPy or executing a learner."""
    errors, archives = [], []
    path_value = plan.get('input_manifest')
    if not path_value:
        return None, ['No shared input manifest attached'], []
    path = root/path_value
    if not path.exists():
        return None, ['Shared input manifest missing'], []
    if sha(path) != plan.get('input_manifest_sha256'):
        errors.append('Shared input manifest SHA mismatch')
    manifest = load(path)
    if manifest.get('format') != 'sutro-d6-input-manifest-v1' or not isinstance(manifest.get('dict_name'),str) or not manifest['dict_name']:
        errors.append('Invalid shared input manifest format/Dict identity')
    if manifest.get('scorer_sha256') != plan['scorer_sha256']:
        errors.append('Manifest scorer SHA mismatch')
    if manifest.get('source_release_sha256') != plan['scorer_sha256']:
        errors.append('Manifest canonical source release SHA mismatch')
    release_version = None
    for node in ast.parse((root/'canonical/mnist.py').read_text()).body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='VERSION' for t in node.targets):
            release_version = ast.literal_eval(node.value)
    required = [f'fashion:{plan["warmup_seed"]}'] + [f'mnist:{seed}' for seed in plan['draw_seeds']]
    for key in required:
        entry = manifest.get('entries', {}).get(key)
        if not entry:
            errors.append('Missing bundle '+key)
            continue
        local = manifest.get('local_archives', {}).get(key)
        ae = []
        arrays = {}
        dataset,seed = key.split(':')
        expected_metadata = {'format':'npz-arrays-v1','key':f'v1:{plan["scorer_sha256"]}:{dataset}:{seed}',
                             'dataset':dataset,'seed':int(seed),'scorer_sha256':plan['scorer_sha256'],
                             'source_release_sha256':plan['scorer_sha256'],'release_version':release_version}
        ae.extend('Frozen bundle metadata mismatch: '+name for name,value in expected_metadata.items() if entry.get(name)!=value)
        if not hexhash(entry.get('npz_sha256')) or type(entry.get('npz_bytes')) is not int or not 0<entry['npz_bytes']<5*1024*1024:
            ae.append('Invalid frozen archive byte hash/size metadata')
        if not hexhash(entry.get('train_sha256')) or not hexhash(entry.get('query_sha256')) or set(entry.get('arrays',{}))!={'tx','ty','qx','qy'}:
            ae.append('Invalid frozen train/query/array metadata')
        for name in ('tx','ty','qx','qy'):
            meta = entry.get('arrays',{}).get(name,{})
            expected_shape = [TOTAL,60] if name in ('tx','qx') else [TOTAL]
            expected_dtype = '<f4' if name in ('tx','qx') else '<i8'
            if meta.get('shape')!=expected_shape or meta.get('dtype')!=expected_dtype or not hexhash(meta.get('sha256')):
                ae.append('Invalid canonical array shape/dtype/hash metadata: '+name)
        if not local or not (root/local).exists():
            ae.append('Missing local archive')
        else:
            archive = root/local
            if sha(archive) != entry['npz_sha256'] or archive.stat().st_size != entry['npz_bytes']:
                ae.append('NPZ byte hash/size mismatch')
            try:
                with zipfile.ZipFile(archive) as zipped:
                    if set(zipped.namelist()) != {name+'.npy' for name in ('tx','ty','qx','qy')}:
                        raise ValueError('NPZ must contain exactly four canonical arrays')
                    for name in ('tx','ty','qx','qy'):
                        npy = zipped.read(name+'.npy')
                        if npy[:6] != b'\x93NUMPY':
                            raise ValueError('Invalid NPY magic')
                        size_width = 2 if npy[6] == 1 else 4
                        start = 8 + size_width
                        length = int.from_bytes(npy[8:start], 'little')
                        header = ast.literal_eval(npy[start:start+length].decode().strip())
                        payload = npy[start+length:]
                        expected = entry['arrays'][name]
                        if len(payload) != TOTAL*(60*4 if name in ('tx','qx') else 8):
                            raise ValueError('Invalid canonical array byte count: '+name)
                        if header['fortran_order'] or list(header['shape']) != expected['shape'] or header['descr'] != expected['dtype']:
                            raise ValueError('Array format/shape/dtype mismatch: '+name)
                        if hashlib.sha256(payload).hexdigest() != expected['sha256']:
                            raise ValueError('Array byte hash mismatch: '+name)
                        arrays[name] = payload
                if hashlib.sha256(arrays['tx']+arrays['ty']).hexdigest() != entry['train_sha256'] or hashlib.sha256(arrays['qx']).hexdigest() != entry['query_sha256']:
                    ae.append('Combined train/query hash mismatch')
            except Exception as exc:
                ae.append(str(exc))
        archives.append({'bundle':key,'local_archive':local,'errors':ae})
        errors.extend(key+': '+e for e in ae)
    return manifest, errors, archives


def probabilities(member):
    """Optional confirmation schema: FP32 little-endian zlib/base64, or Nx10 lists."""
    raw = member.get('probabilities_f32_zlib_base64')
    if raw is None and member.get('probabilities_codec') == 'zlib-base64-f32-le':
        if member.get('probabilities_shape') != [TOTAL, 10]:
            raise ValueError('Wrong declared probability shape')
        raw = member.get('probabilities')
    if raw is not None:
        data = zlib.decompress(base64.b64decode(raw, validate=True))
        if len(data) != TOTAL * 10 * 4:
            raise ValueError('Wrong FP32 probability byte count')
        if member.get('probabilities_sha256') and hashlib.sha256(data).hexdigest() != member['probabilities_sha256']:
            raise ValueError('Probability byte hash mismatch')
        values = array.array('f')
        values.frombytes(data)
        import sys
        if sys.byteorder != 'little':
            values.byteswap()
        return values
    raw = member.get('probabilities')
    if raw is None:
        return None
    if len(raw) != TOTAL or any(len(r) != 10 for r in raw):
        raise ValueError('Wrong probability shape')
    return array.array('f', (v for r in raw for v in r))


def probability_argmax(values):
    return [max(range(10), key=lambda c: values[10*i+c]) for i in range(TOTAL)]


def audit_row(row, configured, references, expected_bundle=None, manifest_sha=None, scorer_sha=None, require_probabilities=False, require_algorithm_clock=False):
    errors, warnings = [], []
    seed = row.get('draw_seed')
    truth = row.get('test_labels')
    if not labels_ok(truth):
        return {'seed': seed, 'errors': ['Invalid query-label vector'], 'prefixes': []}
    label_hash = hashlib.sha256(bytes(truth)).hexdigest()
    bundle_checks = None
    if expected_bundle is not None:
        label_bytes = struct.pack(('<' if not expected_bundle['arrays']['qy']['dtype'].startswith('>') else '>')+'10000q', *truth)
        bundle_checks = {'bundle_key':row.get('data_bundle_key') == expected_bundle['key'],
                         'archive_hash':row.get('data_bundle_npz_sha256') == expected_bundle['npz_sha256'],
                         'input_manifest_hash':row.get('input_manifest_sha256') == manifest_sha,
                         'source_release_hash':row.get('source_release_sha256') == scorer_sha,
                         'all_array_hashes':row.get('array_sha256') == {k:v['sha256'] for k,v in expected_bundle['arrays'].items()},
                         'train_bytes':row.get('train_sha256') == expected_bundle['train_sha256'],
                         'query_bytes':row.get('query_sha256') == expected_bundle['query_sha256'],
                         'saved_truth_bytes':hashlib.sha256(label_bytes).hexdigest() == expected_bundle['arrays']['qy']['sha256']}
        errors.extend('Frozen bundle '+k+' mismatch' for k,v in bundle_checks.items() if not v)
    for key in ('train_sha256', 'query_sha256'):
        if not hexhash(row.get(key)):
            errors.append('Invalid ' + key)
    ref = references.get(seed)
    pairing = None
    if ref:
        pairing = {key: row.get(key) == ref.get(key) for key in ('train_sha256', 'query_sha256')}
        pairing['query_labels_identical'] = truth == ref['test_labels']
        if not pairing['query_labels_identical']:
            errors.append('Query labels differ from paired seed reference')
        if not all(pairing[k] for k in ('train_sha256', 'query_sha256')):
            warnings.append('Released coordinates are not byte-identical to paired seed reference')
    members = row.get('members', [])
    expected_indices = list(range(1, len(members)+1))
    if [m.get('member') for m in members] != expected_indices:
        errors.append('Member indices not consecutive')
    if len(members) > configured['members']:
        errors.append('More members than configured')
    saved_prefixes = row.get('prefixes', [])
    prefix_ids = [p.get('members') for p in saved_prefixes]
    if len(set(prefix_ids)) != len(prefix_ids) or any(k not in configured['prefixes'] for k in prefix_ids):
        errors.append('Duplicate or unconfigured prefixes')
    member_audits, prefix_audits = [], []
    aggregate = array.array('f', [0.0]) * (TOTAL*10)
    probability_prefixes, have_all_probabilities = {}, True
    for member in members:
        me = []
        prediction = member.get('prediction')
        if not labels_ok(prediction):
            me.append('Invalid member prediction labels')
        rebuilt = sum(a == b for a, b in zip(prediction or [], truth))
        if type(member.get('correct')) is not int or member.get('correct') != rebuilt or member.get('total') != TOTAL:
            me.append('Member correct/total mismatch')
        if member.get('model_seed') != 11 + 997*(member.get('member', 0)-1):
            me.append('Unexpected member seed')
        if not finite_time(member.get('full_call_seconds')):
            me.append('Invalid member time')
        if 'algorithm_seconds' in member and (not finite_time(member['algorithm_seconds']) or not finite_time(member.get('full_call_seconds')) or member['algorithm_seconds'] + 1e-8 < member['full_call_seconds']):
            me.append('Invalid member algorithm time')
        if require_algorithm_clock:
            post = member.get('probability_accumulation_argmax_seconds')
            if not finite_time(member.get('algorithm_seconds')) or not finite_time(member.get('full_call_seconds')) or not finite_time(post) or not math.isclose(member['algorithm_seconds'],member['full_call_seconds']+post,rel_tol=1e-12,abs_tol=1e-8):
                me.append('Required fit+accumulation/argmax member clock missing or inconsistent')
        if member.get('training_total') != TOTAL or type(member.get('training_errors')) is not int or not 0 <= member['training_errors'] <= TOTAL:
            me.append('Invalid reported training-error count')
        status = 'not_saved'
        if require_probabilities and (member.get('probabilities_codec') != 'zlib-base64-f32-le' or member.get('probabilities_shape') != [TOTAL,10] or not hexhash(member.get('probabilities_sha256'))):
            me.append('Required probability codec/shape/byte-hash metadata missing or invalid')
        try:
            values = probabilities(member)
            if values is None:
                have_all_probabilities = False
                if require_probabilities:
                    me.append('Required member probabilities missing')
            else:
                status = 'reconstructed'
                if any(not math.isfinite(v) or v < 0 for v in values):
                    me.append('Invalid saved probabilities')
                elif probability_argmax(values) != prediction:
                    me.append('Probability/member-label argmax mismatch')
                if any(abs(sum(values[i:i+10])-1) > 1e-4 for i in range(0, len(values), 10)):
                    me.append('Member probabilities are not normalized')
                if have_all_probabilities:
                    # array('f') rounds after each addition, matching ordered FP32 accumulation.
                    for i, value in enumerate(values):
                        aggregate[i] += value
                    probability_prefixes[member['member']] = probability_argmax(aggregate)
        except Exception as exc:
            me.append('Probability decoding: ' + str(exc))
            have_all_probabilities = False
        errors.extend(f"member{member.get('member')}: {e}" for e in me)
        member_audits.append({'member': member.get('member'), 'rebuilt_correct': rebuilt, 'errors': me,
                              'probability_audit': status, 'training_count_scope': 'Reported count bounded only; no training predictions saved.'})
    for prefix in saved_prefixes:
        pe = []
        k = prefix.get('members')
        prediction = prefix.get('prediction')
        if not labels_ok(prediction):
            pe.append('Invalid prefix prediction labels')
        rebuilt = sum(a == b for a, b in zip(prediction or [], truth))
        if type(prefix.get('correct')) is not int or prefix.get('correct') != rebuilt or prefix.get('total') != TOTAL:
            pe.append('Prefix correct/total mismatch')
        accuracy = 100*rebuilt/TOTAL
        if type(prefix.get('accuracy_pct')) not in (int, float) or not math.isfinite(prefix['accuracy_pct']) or abs(prefix['accuracy_pct']-accuracy) > 1e-9:
            pe.append('Prefix accuracy mismatch')
        if not finite_time(prefix.get('full_call_seconds')):
            pe.append('Invalid prefix time')
        times = [m.get('full_call_seconds') for m in members[:k]] if type(k) is int and k > 0 else []
        summed = sum(times) if len(times) == k and all(finite_time(t) for t in times) else None
        if summed is None or not math.isclose(summed, prefix.get('full_call_seconds', -1), rel_tol=1e-12, abs_tol=1e-8):
            pe.append('Prefix time differs from member time sum')
        if k == 1 and members and prediction != members[0].get('prediction'):
            pe.append('Single-member prefix labels differ from member1')
        probability_status = 'not_reconstructable_from_labels_only'
        if k in probability_prefixes:
            probability_status = 'reconstructed'
            if probability_prefixes[k] != prediction:
                pe.append('FP32 probability ensemble argmax mismatch')
        elif require_probabilities:
            pe.append('Required probability averaging cannot be reconstructed')
        budget_seconds = prefix.get('cumulative_algorithm_seconds', prefix.get('full_call_seconds'))
        if not finite_time(budget_seconds) or (finite_time(prefix.get('full_call_seconds')) and budget_seconds + 1e-8 < prefix['full_call_seconds']):
            pe.append('Invalid cumulative algorithm/budget time')
        if 'cumulative_algorithm_seconds' in prefix:
            algorithm_times = [m.get('algorithm_seconds') for m in members[:k]]
            if len(algorithm_times) != k or not all(finite_time(t) for t in algorithm_times) or not math.isclose(sum(algorithm_times), budget_seconds, rel_tol=1e-12, abs_tol=1e-7):
                pe.append('Prefix algorithm clock differs from member algorithm time sum')
        elif require_algorithm_clock:
            pe.append('Required cumulative algorithm clock missing')
        prefix_audits.append({'members': k, 'rebuilt_correct': rebuilt, 'total': TOTAL, 'accuracy_pct': accuracy,
                              'full_call_seconds': prefix.get('full_call_seconds'), 'budget_seconds':budget_seconds,
                              'budget_clock': 'fit_inference_plus_aggregation_argmax' if 'cumulative_algorithm_seconds' in prefix else 'legacy_member_fit_inference_sum',
                              'within_budget': finite_time(budget_seconds) and budget_seconds <= 300,
                              'errors': pe, 'probability_averaging_audit': probability_status})
    return {'seed': seed, 'query_labels_sha256_u8': label_hash, 'train_sha256': row.get('train_sha256'),
            'query_sha256': row.get('query_sha256'), 'pairing': pairing, 'errors': errors, 'warnings': warnings,
            'frozen_bundle_checks':bundle_checks, 'budget_exhausted_reported': row.get('budget_exhausted', False), 'members': member_audits, 'prefixes': prefix_audits}


def analyze(root=ROOT, strict_pairing=False):
    specifications = [('plan.json', 'development'), ('extension-plan.json', 'development-extension')]
    plans, arms, errors = [], [], []
    for plan_file, directory in specifications:
        path = root/plan_file
        if not path.exists():
            errors.append('Missing configured plan: ' + plan_file)
            continue
        plan = load(path)
        plans.append({'path': plan_file, 'sha256': sha(path), 'plan': plan, 'raw_directory': directory})
        for arm in plan['arms']:
            arms.append((plan, arm, directory))
    if len(arms) != EXPECTED_ARMS:
        errors.append(f'Expected19 logical arms; found{len(arms)}')
    names = [a['name'] for _, a, _ in arms]
    if len(names) != len(set(names)):
        errors.append('Duplicate logical arm names')
    records = {}
    for plan, arm, directory in arms:
        path = existing(root/directory/(arm['name']+'.json'))
        if path.exists():
            try:
                records[arm['name']] = load(path)
            except Exception as exc:
                errors.append(f'Unreadable record {path}: {exc}')
    references = {}
    phase_references = {}
    arm_phases = {arm['name']: plan['phase'] for plan, arm, _ in arms}
    for name in ['baseline450'] + [n for n in names if n != 'baseline450']:
        for row in records.get(name, {}).get('rows', []):
            if labels_ok(row.get('test_labels')):
                references.setdefault(row.get('draw_seed'), row)
                phase_references.setdefault(arm_phases[name], {}).setdefault(row.get('draw_seed'), row)
    audited, candidates, manifest_cache = [], [], {}
    for plan, arm, directory in arms:
        name = arm['name']; path = existing(root/directory/(name+'.json'))
        source = root/arm.get('source_path', 'sources/'+name+'.py')
        ae, integrity = [], []
        if not source.exists() or sha(source) != arm['source_sha256']:
            integrity.append('Frozen source hash mismatch or missing')
        if source.exists() and source.stat().st_size != arm.get('source_bytes'):
            integrity.append('Frozen source byte count mismatch')
        for file,key in [('canonical/mnist.py','scorer_sha256'),('canonical/run_modal.py','runner_sha256')]:
            if not (root/file).exists() or sha(root/file) != plan.get(key):
                integrity.append('Pinned canonical hash mismatch: '+file)
        if plan['phase'] not in manifest_cache:
            manifest_cache[plan['phase']] = (None, [], []) if plan['phase'] == 'development' else audit_input_manifest(root, plan)
        manifest, manifest_errors, archive_audit = manifest_cache[plan['phase']]
        integrity.extend(manifest_errors)
        raw = records.get(name)
        finalized = raw is not None and type(raw.get('completed')) is bool
        row_audits = []
        if raw is not None:
            for key, expected in [('arm', name), ('phase', plan['phase']), ('source_sha256', arm['source_sha256']), ('scorer_sha256', plan['scorer_sha256']), ('parameters', arm)]:
                if raw.get(key) != expected:
                    integrity.append('Raw '+key+' mismatch')
            if not raw.get('completed'):
                ae.append('Execution failed or incomplete: '+str(raw.get('error', 'No error supplied')))
            seeds = [r.get('draw_seed') for r in raw.get('rows', [])]
            if seeds != plan['draw_seeds']:
                ae.append('Missing, duplicated or reordered configured development draws')
                if raw.get('completed') is True:
                    integrity.append('Completed record does not contain the configured development draws exactly once')
            if manifest and (raw.get('input_manifest_sha256') != plan.get('input_manifest_sha256') or raw.get('data_bundle_dict') != manifest['dict_name']):
                integrity.append('Raw shared input manifest/Dict identity mismatch')
            if manifest and raw.get('runner_sha256') != plan['runner_sha256']:
                integrity.append('Raw canonical runner SHA mismatch')
            if manifest and (raw.get('completed') is True or 'warmup_bundle' in raw) and raw.get('warmup_bundle') != manifest['entries'][f'fashion:{plan["warmup_seed"]}']:
                integrity.append('Raw warmup bundle identity mismatch')
            row_audits = [audit_row(row, arm, phase_references.get(plan['phase'], references),
                                   manifest.get('entries',{}).get(f'mnist:{row.get("draw_seed")}') if manifest else None,
                                   plan.get('input_manifest_sha256'),plan['scorer_sha256'],plan.get('save_probabilities',False),plan['phase']!='development') for row in raw.get('rows', [])]
            for original, audited_row in zip(raw.get('rows', []), row_audits):
                initial_ref = references.get(original.get('draw_seed'))
                audited_row['pairing_reference_phase'] = plan['phase']
                audited_row['initial_phase_coordinate_hashes_identical'] = bool(initial_ref) and all(original.get(k) == initial_ref.get(k) for k in ('train_sha256','query_sha256'))
                if initial_ref and original.get('test_labels') != initial_ref['test_labels']:
                    audited_row['errors'].append('Query labels differ across development phases')
                integrity.extend(f'draw{audited_row["seed"]}: '+e for e in audited_row['errors'])
                for prefix in audited_row['prefixes']:
                    integrity.extend(f'draw{audited_row["seed"]}/prefix{prefix["members"]}: '+e for e in prefix['errors'])
        if raw is None:
            ae.append('Awaiting initial result')
        ae.extend(integrity)
        errors.extend(name+': '+e for e in integrity)
        arm_record = {'name': name, 'phase': plan['phase'], 'parameters': arm, 'source_path': str(source.relative_to(root)),
                      'raw_path': str(path.relative_to(root)), 'raw_sha256': sha(path) if path.exists() else None,
                      'raw_file_sha256':hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None,
                      'finalized': finalized, 'completed': raw.get('completed') if raw else None, 'errors': ae, 'integrity_errors':integrity,
                      'input_archive_audit':archive_audit, 'rows': row_audits}
        audited.append(arm_record)
        for k in arm['prefixes']:
            reasons = list(ae)
            selected_rows = []
            for seed in plan['draw_seeds']:
                rows_for_seed = [r for r in row_audits if r['seed'] == seed]
                if len(rows_for_seed) != 1:
                    reasons.append(f'Missing unique draw{seed}')
                    continue
                row = rows_for_seed[0]
                reasons.extend(f'draw{seed}: {e}' for e in row['errors'])
                ps = [p for p in row['prefixes'] if p['members'] == k]
                if len(ps) != 1:
                    reasons.append(f'Prefix{k} absent on draw{seed}')
                    continue
                p = ps[0]; selected_rows.append({'draw_seed': seed, **p})
                reasons.extend(f'draw{seed}: {e}' for e in p['errors'])
                if not p['within_budget']:
                    reasons.append(f'Prefix{k} exceeds300sec or has invalid time on draw{seed}')
                if (strict_pairing or plan['phase'] != 'development') and row.get('pairing') and not all(row['pairing'].values()):
                    reasons.append(f'Byte-exact paired coordinates differ on draw{seed}')
            correct = sum(p['rebuilt_correct'] for p in selected_rows)
            total = sum(p['total'] for p in selected_rows)
            times = [p['full_call_seconds'] for p in selected_rows]
            budgets = [p['budget_seconds'] for p in selected_rows]
            eligible = not reasons and len(selected_rows) == 3
            candidates.append({'arm': name, 'members': k, 'eligible': eligible, 'reasons': sorted(set(reasons)),
                               'pooled_correct': correct, 'total': total, 'accuracy_pct': 100*correct/total if total else None,
                               'mean_full_call_seconds': statistics.mean(times) if times and all(finite_time(t) for t in times) else None,
                               'max_full_call_seconds': max(times) if times and all(finite_time(t) for t in times) else None,
                               'mean_budget_seconds':statistics.mean(budgets) if budgets and all(finite_time(t) for t in budgets) else None,
                               'max_budget_seconds':max(budgets) if budgets and all(finite_time(t) for t in budgets) else None,
                               'rows': selected_rows, 'paired_coordinate_hashes_identical': bool(row_audits) and all(r.get('pairing') and all(r['pairing'].values()) for r in row_audits)})
    final = len(arms) == EXPECTED_ARMS and all(a['finalized'] for a in audited)
    ranking = sorted((c for c in candidates if c['eligible']), key=lambda c: (-c['pooled_correct'], c['mean_full_call_seconds'], c['arm'], c['members']))
    selected = ranking[0] if final and not errors and ranking else None
    report = {'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'expected_logical_arms': EXPECTED_ARMS,
              'configured_logical_arms': len(arms), 'finalized_logical_arms': sum(a['finalized'] for a in audited), 'all_development_finalized': final,
              'selection_locked': not final or bool(errors), 'selected': selected, 'global_errors': errors, 'strict_byte_pairing_required': strict_pairing,
              'plans': [{k:v for k,v in p.items() if k != 'plan'} for p in plans], 'arms': audited, 'all_prefix_candidates': candidates,
              'provisional_ranking': ranking, 'ranking_rule': 'Only a prefix present and valid on all 3 configured draws, each cumulative budget clock <=300 seconds. Extension clocks include fit, inference, probability accumulation and argmax; initial legacy clocks include member fit/inference only. Rank pooled correct descending, then mean full_call_seconds ascending; exact ties deterministic arm/prefix order. Source, truth, archive or saved-count integrity errors lock all selection; retained execution failures do not.',
              'limitations': ['No best arm is selected until all19 configured logical arms have terminal saved outcomes; failed/incomplete arms are retained.',
                             'Initial/extension records save labels. Soft-probability averaging is independently reconstructable only when member FP32 probabilities are also saved.',
                             'mean_full_call_seconds sums timed member _predict calls and excludes aggregation. Extension budget clocks additionally include synchronized probability accumulation and member/prefix argmax; initial legacy clocks lack that overhead. Warmup, draw generation, truth comparison, label transfer and evidence serialization are excluded.',
                             'Initial released-coordinate drift is warning-only. Extension coordinates must match within the extension phase. Raw input tensors are not in the initial records, so cross-phase drift cannot be numerically bounded here. Label/source/count mismatches are fatal.',
                             'Training-error counts are range-checked only; training predictions are absent.',
                             'Exploratory selection, including the adaptive no-guard extension, needs frozen fresh-draw confirmation and cannot prove a universal upper bound.'],
              'prior_infrastructure_failures': [{'path': str(p.relative_to(root)), 'sha256': sha(p)} for p in sorted(root.glob('*failure-status.json'))]}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--strict-pairing', action='store_true')
    args = parser.parse_args()
    report = analyze(args.root, args.strict_pairing)
    (args.root/'analysis.json').write_text(json.dumps(report, indent=2)+'\n')
    table = {'provisional': not report['all_development_finalized'], 'selection_locked': report['selection_locked'],
             'selected': report['selected'], 'rows': [{k:c[k] for k in ('arm','members','eligible','pooled_correct','total','accuracy_pct','mean_full_call_seconds','max_full_call_seconds','mean_budget_seconds','max_budget_seconds','paired_coordinate_hashes_identical','reasons')} for c in sorted(report['all_prefix_candidates'], key=lambda c: (not c['eligible'], -c['pooled_correct'], c['mean_full_call_seconds'] or float('inf')))]}
    (args.root/'provisional-table.json').write_text(json.dumps(table, indent=2)+'\n')
    print(json.dumps({'finalized': report['finalized_logical_arms'], 'configured': report['configured_logical_arms'], 'selection_locked': report['selection_locked'], 'selected': report['selected'], 'global_errors': report['global_errors'], 'eligible_prefixes': len(report['provisional_ranking'])}, indent=2))


if __name__ == '__main__':
    main()
