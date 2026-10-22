"""CPU-only one-shot winner/control freeze after all19 development arms finish."""
import argparse
import ast
import copy
import datetime
import hashlib
import json
from pathlib import Path

import analyze

ROOT = Path(__file__).resolve().parent


def has_regularization(parameters):
    return any(parameters.get(k, 0) for k in ('L2_lambda_matrix_only','hidden_dropout','encoder_dropconnect')) or parameters.get('gaussian_noise', .25) != .25


def without_docstrings(tree):
    tree = copy.deepcopy(tree)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
            node.body.pop(0)
    return ast.dump(tree, include_attributes=False)


def matrix_l2_term(node):
    return any(isinstance(n, ast.GeneratorExp) and isinstance(n.elt, ast.Call) and isinstance(n.elt.func, ast.Attribute) and n.elt.func.attr == 'sum' and isinstance(n.elt.func.value, ast.Call) and isinstance(n.elt.func.value.func, ast.Attribute) and n.elt.func.value.func.attr == 'square' and isinstance(n.elt.func.value.func.value, ast.Name) and n.elt.func.value.func.value.id == 'w' for n in ast.walk(node)) and any(isinstance(n, ast.Attribute) and n.attr == 'enc' for n in ast.walk(node)) and any(isinstance(n, ast.Attribute) and n.attr == 'dec' for n in ast.walk(node))


def strip_l2(node):
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        if matrix_l2_term(node.right):
            left, count = strip_l2(node.left)
            return left, count + 1
        left, count = strip_l2(node.left)
        node.left = left
        return node, count
    return node, 0


def matched_control(source, parameters):
    """Strip exactly explicit matrix L2/training dropout/dropconnect/extra Gaussian noise."""
    tree = ast.parse(source)
    edits = {'removed_dropout_calls': 0, 'removed_L2_terms': 0, 'gaussian_noise_reset': False}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Tuple) and isinstance(node.value, ast.Tuple):
            for i, target in enumerate(node.targets[0].elts):
                if isinstance(target, ast.Name) and target.id == 'NOISE' and parameters.get('gaussian_noise', .25) != .25:
                    node.value.elts[i] = ast.Constant(.25)
                    edits['gaussian_noise_reset'] = True
        if not isinstance(node, ast.ClassDef) or node.name != 'Ladder':
            continue
        for function in node.body:
            if not isinstance(function, ast.FunctionDef) or function.name != 'loss':
                continue
            class RemoveDropout(ast.NodeTransformer):
                def visit_IfExp(self, expr):
                    expr = self.generic_visit(expr)
                    if ast.dump(expr.body,include_attributes=False)==ast.dump(expr.orelse,include_attributes=False) and not any(isinstance(n,(ast.Call,ast.NamedExpr,ast.Await,ast.Yield)) for n in ast.walk(expr.test)):
                        return expr.body
                    return expr
                def visit_Call(self, call):
                    call = self.generic_visit(call)
                    if isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name) and call.func.value.id == 'F' and call.func.attr == 'dropout':
                        if len(call.args) < 2 or not isinstance(call.args[1], ast.Constant) or call.args[1].value not in (parameters.get('hidden_dropout', 0), parameters.get('encoder_dropconnect', 0)):
                            raise ValueError('Unexpected dropout intervention; manual review required')
                        edits['removed_dropout_calls'] += 1
                        return call.args[0]
                    return call
            RemoveDropout().visit(function)
            if parameters.get('L2_lambda_matrix_only', 0):
                for ret in ast.walk(function):
                    if isinstance(ret, ast.Return) and isinstance(ret.value, ast.Tuple):
                        ret.value.elts[0], count = strip_l2(ret.value.elts[0])
                        edits['removed_L2_terms'] += count
    expected_dropout = int(bool(parameters.get('hidden_dropout', 0))) + int(bool(parameters.get('encoder_dropconnect', 0)))
    if edits['removed_dropout_calls'] != expected_dropout or edits['removed_L2_terms'] != int(bool(parameters.get('L2_lambda_matrix_only', 0))):
        raise ValueError('Regularization stripping differs from frozen intervention definition')
    if edits['gaussian_noise_reset'] != (parameters.get('gaussian_noise', .25) != .25):
        raise ValueError('Extra-noise stripping failed')
    # Whole-file AST is copied; only the inspected Ladder.loss and NOISE expressions above change.
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)+'\n', edits


def freeze(root=ROOT, plan_name='confirmation-plan.json'):
    destination = root/plan_name
    if destination.exists():
        raise RuntimeError('Refusing to replace an existing frozen confirmation plan')
    report = analyze.analyze(root)
    if report['selection_locked'] or report['selected'] is None:
        raise RuntimeError('All19 development arms must have terminal audited outcomes before winner freeze')
    best = report['selected']
    selected = next(a for a in report['arms'] if a['name'] == best['arm'])
    baseline = next(a for a in report['arms'] if a['name'] == 'baseline450')
    sources = root/'confirmation-sources'
    if sources.exists():
        raise RuntimeError('Refusing to replace confirmation source snapshots')
    source_records = []
    def add(role, original, source, members):
        tree_identity = without_docstrings(ast.parse(source))
        for old in source_records:
            if old['algorithm_ast'] == tree_identity and old['parameters']['members'] == members:
                old['roles'].append(role)
                return old
        parameters = copy.deepcopy(original['parameters'])
        parameters.update(name=role, members=members, prefixes=[members])
        encoded = source.encode()
        parameters.update(source_sha256=hashlib.sha256(encoded).hexdigest(), source_bytes=len(encoded), source_path=f'confirmation-sources/{role}.py')
        record = {'roles': [role], 'parameters': parameters, 'source': source, 'algorithm_ast': tree_identity}
        source_records.append(record)
        return record
    selected_source = (root/selected['source_path']).read_text()
    add('selected', selected, selected_source, best['members'])
    add('baseline450', baseline, (root/baseline['source_path']).read_text(), 1)
    p = selected['parameters']
    control_audit = {'comparisons':[]}
    regularized_candidate = best
    regularized_arm = selected
    if not has_regularization(p):
        regularized_candidate = next((c for c in report['provisional_ranking'] if has_regularization(next(a for a in report['arms'] if a['name']==c['arm'])['parameters'])), None)
        if regularized_candidate is None:
            raise RuntimeError('No eligible regularized prefix for the prespecified secondary confirmation')
        regularized_arm = next(a for a in report['arms'] if a['name']==regularized_candidate['arm'])
        add('best_regularized',regularized_arm,(root/regularized_arm['source_path']).read_text(),regularized_candidate['members'])
    regularized_source = (root/regularized_arm['source_path']).read_text()
    control_source, edits = matched_control(regularized_source,regularized_arm['parameters'])
    control_original = copy.deepcopy(regularized_arm)
    control_original['parameters'].update(L2_lambda_matrix_only=0,hidden_dropout=0,encoder_dropconnect=0,gaussian_noise=.25)
    control_role = 'matched_unregularized_control' if has_regularization(p) else 'best_regularized_control'
    add(control_role,control_original,control_source,regularized_candidate['members'])
    control_audit['comparisons'].append({'regularized_role':'selected' if has_regularization(p) else 'best_regularized','control_role':control_role,'development_arm':regularized_candidate['arm'],'prefix':regularized_candidate['members'],'changes':edits})
    source_names = []
    for record in source_records:
        parameters = record['parameters']
        record['algorithm_ast_sha256'] = hashlib.sha256(record.pop('algorithm_ast').encode()).hexdigest()
        source_names.append(parameters['name'])
    seeds = list(range(2610222101, 2610222112))
    plan = {'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'phase': 'confirmation', 'frozen': True,
            'selected_development_arm': best['arm'], 'selected_prefix': best['members'], 'development_selection': best,
            'prespecified_secondary_regularization_selection':regularized_candidate,
            'development_plans': report['plans'], 'development_raw_sha256': {a['raw_path']:a['raw_sha256'] for a in report['arms']},
            'scorer_sha256': selected['parameters'].get('scorer_sha256') or json.loads((root/'plan.json').read_text())['scorer_sha256'],
            'runner_sha256': json.loads((root/'plan.json').read_text())['runner_sha256'],
            'study_producer_sha256': analyze.sha(root/'study.py'),
            'input_bundle_helper_sha256': analyze.sha(root/'input_bundles.py'),
            'draw_seeds': seeds, 'warmup_seed': 2610222190, 'shards': [seeds[:3],seeds[3:6],seeds[6:9],seeds[9:]],
            'max_fit_seconds': 300, 'max_concurrent_containers': 4, 'save_probabilities': True,
            'arms': [{'roles': r['roles'], **r['parameters'], 'algorithm_ast_sha256': r['algorithm_ast_sha256']} for r in source_records],
            'matched_control_audit': control_audit,
            'fresh_inputs': {'status': 'Fresh input manifest must be supplied and hash-verified before launch; no confirmation outcomes inspected during winner freeze.', 'input_manifest': None, 'input_manifest_sha256': None},
            'acceptance': {'accuracy_target_pct': 98.6, 'required_correct': 108460, 'total': 110000, 'every_draw_cumulative_algorithm_seconds_max': 300},
            'no_tuning': 'Freeze selected source, prefix, seeds, baseline and matched control before fresh labels/results. Retain every outcome; no model/regularization/width/guard/prefix reselection or replacement after confirmation.',
            'scope': 'Unsandboxed research, not official canonical qualification or a universal upper-accuracy proof. Report per-draw counts, paired differences and all failures. Future budget clocks include fit, inference, probability accumulation and argmax; warmup, draw preparation, truth comparison and evidence serialization are excluded. Initial legacy screen clocks lack aggregation overhead.'}
    sources.mkdir()
    for record in source_records:
        (root/record['parameters']['source_path']).write_text(record['source'])
    (root/'confirmation-frozen-development-analysis.json').write_text(json.dumps(report,indent=2)+'\n')
    destination.write_text(json.dumps(plan,indent=2)+'\n')
    return plan, analyze.sha(destination)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', default='confirmation-plan.json')
    args = parser.parse_args()
    plan, digest = freeze(args.root,args.output)
    print(json.dumps({'frozen_plan':str(args.root/args.output),'sha256':digest,'selected_arm':plan['selected_development_arm'],'prefix':plan['selected_prefix'],'roles':[a['roles'] for a in plan['arms']],'shard_jobs':4*len(plan['arms']),'GPU_launched':False},indent=2))
