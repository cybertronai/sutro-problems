"""Reproducible, explicit size adaptation of historical sources; never edits them."""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / 'mlp-panels-v4-20260911'
BASE = HERE.parent / 'small60-grid-20260912'


def replace(source, old, new):
    assert old in source, old
    return source.replace(old, new)


def numbers(source, mapping):
    # Replace numeric tokens only, never substrings of parameter offsets or hashes.
    import io
    import tokenize
    return tokenize.untokenize([
        token._replace(string=str(mapping.get(token.string, token.string)))
        if token.type == tokenize.NUMBER else token
        for token in tokenize.generate_tokens(io.StringIO(source).readline)
    ])


def main():
    assert not (HERE / 'protocol.json').exists(), 'Never regenerate frozen sources'
    sources = {}
    def emit(name, source):
        ast.parse(source)
        (HERE / name).write_text(source)

    for root, name in [(OLD, 'panels.py'), (OLD, 'reference.py'), (OLD, 'gpu_panels.py'),
                       (OLD, 'gpu_benchmark.py'), (OLD, 'selected.json'), (BASE, 'run.py'),
                       (BASE, 'reference.py')]:
        sources[str(root.name + '/' + name)] = hashlib.sha256((root / name).read_bytes()).hexdigest()

    panel = numbers((OLD / 'panels.py').read_text(), {'30':25, '270':225, '600':1000})
    emit('panels.py', panel)
    config = json.loads((OLD / 'selected.json').read_text())
    config['inference_batch'] = 25
    (HERE / 'selected.json').write_text(json.dumps(config, indent=2) + '\n')

    ref = numbers((OLD / 'reference.py').read_text(), {'30':25})
    ref = replace(ref, 'import accuracy_study as study', 'import baseline_reference as study\nBATCH = 25')
    ref = replace(ref, "params = study.parameters(32,101)", "assert len(data['train_images']) % BATCH == 0\n    params = study.parameters(32,101)")
    emit('reference.py', ref)
    emit('baseline_reference.py', (BASE / 'reference.py').read_text())

    run = (BASE / 'run.py').read_text()
    run = replace(run, 'target_accuracy=0.60', 'target_accuracy=0.67')
    run = replace(run, 'target_total_correct=6600', 'target_total_correct=7370')
    run = replace(run, '>=6600', '>=7370')
    run = replace(run, "('run.py','reference.py')", "('run.py','reference.py','baseline_reference.py','panels.py','selected.json','adapt_sources.py')")
    run = replace(run, "x=reference.transform(arrays['train_images']); q=reference.transform(arrays['test_images'])\n        target=(arrays['train_labels'][:,None]==np.arange(10)).astype(np.float32)\n        params=reference.parameters(32,101)\n        for _ in range(300): params=reference.epoch(x,target,params,0.2)\n        scores=reference.forward(q,params)[2]", "config=json.loads((HERE/'selected.json').read_text())\n        result=reference.cpu(arrays,config,300,check_baseline=True)\n        params=[result['params']]\n        scores=result['scores']")
    emit('run.py', run)

    gpu = (OLD / 'gpu_panels.py').read_text()
    # PANEL_SOURCE is itself Python inside a string: adapt that independently.
    tree = ast.parse(gpu)
    node = next(n for n in tree.body if isinstance(n, ast.Assign) and n.targets[0].id == 'PANEL_SOURCE')
    lines = gpu.splitlines(keepends=True)
    panel_source = numbers(ast.literal_eval(node.value), {'30':25,'270':225,'960':800,'600':1000})
    gpu = ''.join(lines[:node.lineno-1]) + 'PANEL_SOURCE = ' + repr(panel_source) + '\n' + ''.join(lines[node.end_lineno:])
    gpu = replace(gpu, "definitions.append(definition)", "definitions.append(adapt_numeric_tokens(definition))")
    gpu += '\n\ndef adapt_numeric_tokens(source):\n    from adapt_sources import numbers\n    return numbers(source, {\"600\":1000,\"5400\":9000,\"6000\":10000,\"19200\":32000,\"30\":25,\"960\":800,\"300\":250})\n'
    emit('gpu_panels.py', gpu)

    bench = numbers((OLD / 'gpu_benchmark.py').read_text(), {
        '600':1000,'30':25,'270':225,'5400':9000,'6000':10000,'960':800,'19200':32000,'24004':48004})
    bench = replace(bench, "MODES = ('baseline','energy','no_slowdown')", "MODES = ('energy',)")
    bench = replace(bench, "ORDERS = (MODES,('energy','no_slowdown','baseline'),('no_slowdown','baseline','energy'))", "ORDERS = (MODES, MODES, MODES)")
    bench = replace(bench, 'triton.cdiv(300,128)', 'triton.cdiv(250,128)')
    bench = replace(bench, 'inference_hidden_panels[(20,)]', 'inference_hidden_panels[(40,)]')
    bench = replace(bench, 'inference_output_panels[(20,)]', 'inference_output_panels[(40,)]')
    # Only measure the predeclared energy policy, not an optimization sweep.
    start = bench.index("    balanced = json.loads")
    end = bench.index("    source = g.generated_source()", start)
    bench = bench[:start] + "    assert energy['inference_batch']==25 and energy['cache_x']\n" + bench[end:]
    start = bench.index("        manifest = json.loads")
    end = bench.index('    provenance = ', start)
    bench = bench[:start] + "        draw = json.loads((HERE/'draw_manifest.json').read_text())['draws'][0]\n        for key in ('train_images','train_labels','test_images'):\n            assert hashlib.sha256(arrays[key].tobytes()).hexdigest()==draw['arrays'][key]['sha256']\n" + bench[end:]
    bench = replace(bench, "'energy_config':energy,'no_slowdown_config':balanced,", "'energy_config':energy,'dataset_seed':20261201,")
    bench = replace(bench, "'theory_costs_sha256':hashlib.sha256((HERE/'costs.json').read_bytes()).hexdigest()", "'protocol_sha256':hashlib.sha256((HERE/'protocol.json').read_bytes()).hexdigest()")
    start = bench.index("        'grids':")
    end = bench.index("        'protocol':", start)
    bench = bench[:start] + "        'grids':{'energy':{'hidden':1,'output':4,'backward':8,'update':12,'infer_hidden':40,'infer_output':40}},\n" + bench[end:]
    bench = replace(bench, 'Reset650 parameters; normalize600 train600 query inputs; targets;300epochs6000minibatches;600predictions6000scores', 'Reset650 parameters; normalize1000 train1000 query inputs; targets;300epochs12000minibatches;1000predictions10000scores')
    bench = bench.replace('real270word', 'real225word').replace('retains600hidden', 'retains1000hidden').replace('24,004', '48,004')
    emit('gpu_benchmark.py', bench)
    (HERE / 'source_adaptation.json').write_text(json.dumps(sources, indent=2) + '\n')


if __name__ == '__main__':
    main()
