"""Read-only verification of the frozen extension; imports no learner or GPU code."""
import ast
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
sha = lambda data: hashlib.sha256(data).hexdigest()

def literal(node):
    if isinstance(node, ast.Tuple):
        return tuple(literal(item) for item in node.elts)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        return literal(node.left) / literal(node.right)
    return ast.literal_eval(node)
plan = json.loads((root / 'extension-plan.json').read_text())
initial_raw = (root / 'plan.json').read_bytes()
initial = json.loads(initial_raw)
assert sha(initial_raw) == plan['initial_plan_sha256']
assert plan['phase'] == 'development-extension' and len(plan['arms']) == 9
assert plan['draw_seeds'] == initial['draw_seeds']
assert plan['warmup_seed'] == initial['warmup_seed']
for arm in initial['arms']:
    assert sha((root / 'sources' / (arm['name']+'.py')).read_bytes()) == arm['source_sha256']
for arm in plan['arms']:
    raw = (root / arm['source_path']).read_bytes()
    assert sha(raw) == arm['source_sha256'] and len(raw) == arm['source_bytes']
    tree = ast.parse(raw.decode())
    compile(tree, arm['name']+'.py', 'exec')
    assert (arm['steps'], arm['LR'], arm['members'], arm['prefixes']) == (7200, 0.01467, 16, [1, 2, 4, 8, 16])
    source = (root / 'sources' / (arm['upstream_initial_arm']+'.py')).read_text()
    assert sha(source.encode()) == arm['upstream_source_sha256']
    original = ast.parse(source)
    before = next(n for n in original.body if isinstance(n, ast.FunctionDef) and n.name == '_predict')
    after = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_predict')
    # The two removed statements are exactly the final guard call and teacher
    # probability replacement; the preceding normalization and return remain.
    assert len(before.body) == len(after.body) + 2
    del before.body[-3:-1]
    assert ast.dump(before, include_attributes=False) == ast.dump(after, include_attributes=False)
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'guard' for n in ast.walk(after))
    assignments = {}
    for n in tree.body:
        if isinstance(n, ast.Assign):
            for target in n.targets:
                if isinstance(target, ast.Tuple):
                    try:
                        values = literal(n.value)
                    except (ValueError, TypeError):
                        continue
                    for symbol, value in zip(target.elts, values):
                        if isinstance(symbol, ast.Name):
                            assignments[symbol.id] = value
                elif isinstance(target, ast.Name) and target.id == 'KCFG':
                    assignments['KCFG'] = ast.literal_eval(n.value)
    assert assignments['M'] == 1
    assert list(assignments['DIMS']) == arm['DIMS']
    assert (assignments['STEPS'], assignments['LR']) == (7200, 0.01467)
    for size in assignments['DIMS'][1:-1]:
        assert size in assignments['KCFG']
        assert size % assignments['KCFG'][size][0] == 0
proof = json.loads((root / 'extension-kernel-coverage.json').read_text())
assert proof['status'] == 'pass'
assert {row['S'] for row in proof['configurations']} >= {256, 512, 768, 1024, 1536}
print(json.dumps({'status':'pass','initial_sources_unchanged':True,'extension_sources':9,
                  'source_hashes_exact':True,'guard_removal_only_predict_change':True,
                  'KCFG_covers_all_hidden_dimensions':True,'GPU_jobs_launched_by_verifier':0},indent=2))
