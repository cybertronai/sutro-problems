"""Freeze the prescribed D6 development extension; CPU-only, never launches jobs."""
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def sha(data):
    return hashlib.sha256(data).hexdigest()

def put(path, data):
    path = ROOT / path
    if isinstance(data, dict):
        data = json.dumps(data, indent=2, sort_keys=True) + '\n'
    if path.exists():
        assert path.read_text() == data, f'Refuse to replace frozen artifact: {path}'
    else:
        path.write_text(data)

def replace_once(source, old, new):
    assert source.count(old) == 1, old
    return source.replace(old, new)

initial_bytes = (ROOT / 'plan.json').read_bytes()
initial = json.loads(initial_bytes)
for arm in initial['arms']:
    raw = (ROOT / 'sources' / (arm['name'] + '.py')).read_bytes()
    assert sha(raw) == arm['source_sha256'], arm['name']
    assert len(raw) == arm['source_bytes'], arm['name']

specs = [
    ('noguard7200', 'long7200', {}, []),
    ('noguard_l2_1em4_7200', 'l2_7200', {'L2_lambda_matrix_only': 0.0001}, [
        ('+0.0005*sum(w.square().sum() for w in o.enc+o.dec)', '+0.00005*sum(w.square().sum() for w in o.enc+o.dec)')]),
    ('noguard_l2_1em3_7200', 'l2_7200', {'L2_lambda_matrix_only': 0.001}, []),
    ('noguard_dropout_01_7200', 'dropout7200', {'hidden_dropout': 0.1}, []),
    ('noguard_dropout_02_7200', 'dropout7200', {'hidden_dropout': 0.2}, [
        ('F.dropout(F.relu(z+b),0.1,True)', 'F.dropout(F.relu(z+b),0.2,True)')]),
    ('noguard_dropconnect_01_7200', 'dropconnect7200', {'encoder_dropconnect': 0.1}, []),
    ('noguard_width1024_dropout_01_7200', 'dropout7200', {'hidden_dropout': 0.1, 'first_hidden_width': 1024}, []),
    ('noguard_width1536_dropout_01_7200', 'dropout7200', {'hidden_dropout': 0.1, 'first_hidden_width': 1536}, []),
    ('noguard_noise7200', 'noise7200', {'gaussian_noise': 0.4}, []),
]
remove = ' chosen=guard(s,p.argmax(1),e)\n p=torch.where((chosen!=p.argmax(1))[:,None],F.softmax(s/TEACH_TAU,1),p)\n'
base_arms = {arm['name']: arm for arm in initial['arms']}
base_ast = ast.parse((ROOT / 'sources/long7200.py').read_text())
base_defs = {node.name: ast.dump(node, include_attributes=False) for node in base_ast.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
metadata = []
for name, upstream, overrides, replacements in specs:
    upstream_path = ROOT / 'sources' / (upstream + '.py')
    original = upstream_path.read_text()
    source = replace_once(original, remove, '')
    for old, new in replacements:
        source = replace_once(source, old, new)
    width = overrides.get('first_hidden_width', 768)
    if width != 768:
        source = replace_once(source, 'M,DIMS=1,(60,768,512,256,10)', f'M,DIMS=1,(60,{width},512,256,10)')
        source = replace_once(source, 'KCFG={768:(16,16,4,4,64),512:(16,16,4,3,64),256:(16,16,2,4,64)}',
                              f'KCFG={{{width}:(16,16,4,4,64),768:(16,16,4,4,64),512:(16,16,4,3,64),256:(16,16,2,4,64)}}')
    tree = ast.parse(source)
    compile(tree, name + '.py', 'exec')
    defs = {node.name: ast.dump(node, include_attributes=False) for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    for symbol in ['teacher', 'sym_scores', 'hk_fix', 'hknn', 'tangents', 'comb_bwd_kernel', 'comb_bwd', 'CombFn', 'guard']:
        assert defs[symbol] == base_defs[symbol], (name, symbol)
    predict = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == '_predict')
    assert not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'guard' for node in ast.walk(predict))
    assert source.endswith(' p=p/p.sum(1,keepdim=True)\n return p,e\n')
    path = 'sources/' + name + '.py'
    put(path, source)
    params = dict(base_arms[upstream])
    params.update(overrides)
    params.update(name=name, members=4, prefixes=[1, 2, 4], steps=7200, LR=0.01467,
                  first_hidden_width=width, DIMS=[60, width, 512, 256, 10], guard_enabled=False,
                  source_sha256=sha(source.encode()), source_bytes=len(source.encode()),
                  source_path=path, upstream_initial_arm=upstream,
                  upstream_source_sha256=sha(original.encode()))
    metadata.append(params)

# Factorized exhaustive proof of row/channel ownership, then explicit DP and WS
# ownership. Rows and channels are Cartesian products; each being covered once
# proves each activation element is covered once without allocating a huge matrix.
configs = {256: (16,16,2,4,64), 512: (16,16,4,3,64), 768: (16,16,4,4,64),
           1024: (16,16,4,4,64), 1536: (16,16,4,4,64)}
proofs = []
for S, cfg in configs.items():
    BC, SUB, W, k, SB = cfg
    B, sm = 2500, 108
    n = S // BC
    assert S % BC == 0
    target = max(1, (sm*k - (4 if k > 1 else 0)) // n)
    rows = ((B + target - 1) // target + SB - 1) // SB * SB
    nrs = (B + rows - 1) // rows
    row1, row2, channels = Counter(), Counter(), Counter()
    for cb in range(n):
        for c in range(cb*BC, (cb+1)*BC):
            assert 0 <= c < S
            channels[c] += 1
    for rs in range(nrs):
        r0, r1 = rs*rows, min((rs+1)*rows, B)
        for chunk in range(r0, r1, SUB):
            for r in range(chunk, chunk+SUB):
                if r < r1:
                    row1[r] += 1
        for chunk in range(r0, r1, SB):
            for r in range(chunk, chunk+SB):
                if r < r1:
                    row2[r] += 1
    assert len(row1) == len(row2) == B and set(row1.values()) == set(row2.values()) == {1}
    assert len(channels) == S and set(channels.values()) == {1}
    ws = Counter(rs*20*S + p*S + c for rs in range(nrs) for p in range(20) for c in range(S))
    assert len(ws) == nrs*20*S and set(ws.values()) == {1} and min(ws) == 0 and max(ws) == nrs*20*S-1
    nprog = n*nrs
    ch = (17*S + nprog - 1) // nprog
    dp = Counter()
    for pid in range(nprog):
        lo, hi = pid*ch, min((pid+1)*ch, 17*S)
        for e0 in range(lo, hi, 256):
            for e in range(e0, e0+256):
                if e < hi:
                    assert 0 <= e < 17*S
                    dp[e] += 1
    assert len(dp) == 17*S and set(dp.values()) == {1} and min(dp) == 0 and max(dp) == 17*S-1
    # Every phase-2 WS read is within the exact preceding phase-1 workspace.
    for row in range(nrs):
        assert row*20*S + 17*S-1 < nrs*20*S
        assert row*20*S + 19*S+S-1 < nrs*20*S
    proofs.append({'S':S,'B':B,'SM_count':sm,'KCFG':list(cfg),'NCB':n,'ROWS':rows,'NRS':nrs,
                   'programs_per_phase':nprog,'DP_chunk':ch,'DP_elements':17*S,
                   'activation_elements':B*S,'workspace_elements':nrs*20*S,
                   'channel_tail':False,'activation_row_coverage_exactly_once':True,
                   'phase1_workspace_coverage_exactly_once':True,'phase2_DP_coverage_exactly_once':True,
                   'all_addresses_in_allocations':True})

snapshot = {}
for path in sorted((ROOT / 'development').glob('*.json')):
    result = json.loads(path.read_text())
    snapshot[path.name] = {'sha256':sha(path.read_bytes()),'completed':result.get('completed'),
                          'training_errors':[[member['training_errors'] for member in row['members']] for row in result.get('rows',[])],
                          'prefix_correct':[[[prefix['members'],prefix['correct']] for prefix in row['prefixes']] for row in result.get('rows',[])]}

created = datetime.now(timezone.utc).isoformat()
plan = {key: initial[key] for key in ['parent_PR','parent_head','parent_sha256','canonical_main','scorer_sha256','runner_sha256','draw_seeds','warmup_seed']}
plan.update(created_utc=created, phase='development-extension', frozen=True,
            launch_status='held; no extension GPU launch authorized or performed',
            initial_plan_path='plan.json', initial_plan_sha256=sha(initial_bytes), initial_arm_count=len(initial['arms']),
            arms=metadata, max_fit_seconds=300,
            members_are_sequential_full_fits=True, model_seed_formula='11 + 997 * zero_based_member',
            source_internal_members_M=1,
            reason='The initial guard can substitute teacher probabilities when student training errors exceed 40, so stronger regularization may be hidden by that substitution. These arms disable both final guard conditions; teacher features, soft targets and HKNN readout remain.',
            L2_convention='Loss adds (lambda/2) times the sum of squared encoder and decoder matrix entries; biases, gamma and combinator parameters are unpenalized.',
            width_convention='Only the first hidden width changes; later hidden widths remain 512 and 256.',
            scope='Exploratory extension fixed after inspecting initial development outcomes, before extension outcomes. Uses exactly the same three canonical development draws and warmup seed. No official D6 qualification, unseen validation result, or accuracy upper bound is claimed.',
            selection='Root will jointly select among all initial and extension arms/prefixes using pooled accuracy under 300 seconds per draw, tie by mean full-call time, after all initial and extension outcomes are retained. Then root will freeze the chosen algorithm and evaluate 11 fresh draws never used for hyperparameter selection. No new development draws or iterative extension replacements are authorized here.',
            preserve='The initial ten source files, plan, raw outcomes and submitted D5 PR source are unchanged.',
            coverage_proof='extension-kernel-coverage.json', omissions=[])
put('extension-kernel-coverage.json', {'created_utc':created,'status':'pass','kind':'CPU-only address and launch-metadata proof; not a numerical or compilation GPU test',
     'M':1,'B':2500,'SM_count':108,'configurations':proofs,
     'DIMS_coverage':'CombFn is called only for hidden decoder widths DIMS[1:-1]. Both wider arms add the needed KCFG key and retain 512/256 keys. Input 60 and output 10 use the existing Torch combine path.',
     'shape_allocation':'Encoder and decoder matrices, biases, combinators, noise shapes and activations derive from DIMS. Feature readout hx[2]/hq[2] remains 256 wide. KCFG affects row partition and uses the same 16x16 register tile and four warps as 768; no fixed width 768 remains in the kernel body.',
     'synchronization':'Each phase is a separate Triton launch on the current stream. Phase 1 writes every WS channel for every row segment before phase 2 reads the complete workspace. Each activation element has one owner in each phase.',
     'mask_condition':'The kernel has some unmasked channel loads/stores, so exact divisibility by BC=16 is required and verified for all proposed widths. Last row segments are masked.',
     'runtime_limitations':'Static coverage does not prove Triton compilation, numerical behavior or accuracy. These remain first-run observations to retain; do not replace a failed arm.'})
put('extension-development-context.json', {'created_utc':created,'scope':'Initial development outcomes already visible before this extension was frozen; not held-out evidence.',
     'initial_plan_sha256':sha(initial_bytes),'results_present_at_freeze':snapshot})
put('extension-plan.json', plan)
put('extension-source-audit.json', {'created_utc':created,'status':'pass','arm_count':len(metadata),
    'source_hashes':{x['name']:x['source_sha256'] for x in metadata},
    'initial_source_hashes_verified':True,'initial_plan_sha256':sha(initial_bytes),
    'all_sources_parse_and_compile_without_execution':True,
    'all_sources_disable_guard_call_and_teacher_probability_substitution':True,
    'training_errors_remain_observable':True,
    'teacher_HKNN_tangents_and_custom_kernel_AST_unchanged':True,
    'all_steps':7200,'all_LR':0.01467,'all_sequential_members':4,'all_prefixes':[1,2,4],
    'all_dev_seeds':initial['draw_seeds'],'warmup_seed':initial['warmup_seed'],
    'wider_DIMS':[[60,1024,512,256,10],[60,1536,512,256,10]],
    'larger_width_coverage':'pass; see extension-kernel-coverage.json',
    'jobs_launched':0,'shared_initial_files_modified':False,
    'deviations':[]})
print(json.dumps({'plan':'extension-plan.json','arms':len(metadata),'coverage':'pass','jobs_launched':0,'source_hashes':{x['name']:x['source_sha256'] for x in metadata}},indent=2))
