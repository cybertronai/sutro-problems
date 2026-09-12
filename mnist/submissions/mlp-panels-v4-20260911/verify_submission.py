"""Read-only evidence checks plus optional local numerical replay; never use GPU."""
import argparse
import ast
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import statistics
import sys
import tempfile
import numpy as np
import panels as p
import learner
import reference
import gpu_panels

sys.path.insert(0,str(p.ROOT))

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full',action='store_true')
    parser.add_argument('--output',type=Path,default=HERE/'generated/verification.json')
    args = parser.parse_args()
    checks = {}
    manifest = json.loads((HERE/'artifact_manifest.json').read_text())
    for name,digest in manifest['files'].items():
        assert sha(HERE/name)==digest,name
    checks['artifact_hashes'] = len(manifest['files'])
    costs = json.loads((HERE/'costs.json').read_text())
    for name,variant in (('baseline','baseline'),('optimized','energy'),('no_slowdown','no_slowdown')):
        document = p.build(None if variant=='baseline' else learner.config(variant))
        assert document==json.loads((HERE/(name+'.il.json')).read_text()),name
        score = p.il.score(document)
        for key in ('energy_fj','time_ticks_0_2_ps','peak_initialized_scratch_words','instructions','charged_reads','charged_writes'):
            assert score[key]==costs[name][key],(name,key)
    checks['model_costs_exact'] = True
    original = ast.parse(gpu_panels.FROZEN.read_text())
    generated = ast.parse(gpu_panels.generated_source())
    for name in gpu_panels.BASELINE_NAMES:
        a = next(n for n in ast.walk(original) if isinstance(n,ast.FunctionDef) and n.name==name)
        b = next(n for n in ast.walk(generated) if isinstance(n,ast.FunctionDef) and n.name==name)
        assert ast.dump(a)==ast.dump(b),name
    gpu = json.loads((HERE/'evidence/gpu/results.json').read_text())
    assert hashlib.sha256(gpu_panels.generated_source().encode()).hexdigest()==gpu['provenance']['generated_sha256']
    measured = ast.parse((HERE/'evidence/gpu/runner.py').read_text())
    portable = ast.parse((HERE/'gpu_benchmark.py').read_text())
    def remote(tree):
        return next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='benchmark')
    assert ast.dump(remote(measured))==ast.dump(remote(portable))
    for name,digest in gpu['kernel_ptx_sha256'].items():
        path = HERE/'evidence/gpu'/(name+'.ptx')
        assert sha(path)==digest
        assert not re.search(rb'\b(?:fma|mad)\.[^;\n]*\.f(?:16|32|64)\b',path.read_bytes())
    for row in gpu['trials']:
        for key in ('idle_before','active','idle_after'):
            value = row[key]
            energy = (value['end']['energy_mj']-value['start']['energy_mj'])/1000
            seconds = value['end']['time_s']-value['start']['time_s']
            assert abs(energy-value['energy_j'])<1e-9
            assert abs(seconds-value['duration_s'])<1e-9
            assert abs(energy/seconds-value['average_power_w'])<1e-9
        power = (row['idle_before']['average_power_w']+row['idle_after']['average_power_w'])/2
        adjusted = (row['active']['energy_j']-power*row['active']['duration_s'])*1000/row['replays']
        assert abs(adjusted-row['adjusted_mj'])<1e-8
    for mode in gpu['summary']:
        assert gpu['graphs'][mode]=={'total':24004,'kernels':24004}
        for key in ('cuda_ms','wall_ms','adjusted_mj','gross_mj'):
            values = [r[key] for r in gpu['trials'] if r['mode']==mode]
            assert statistics.median(values)==gpu['summary'][mode][key]['median']
    checks['gpu_source_graphs_ptx_and_nvml_verified'] = True
    archived = json.loads((HERE/'evidence/gpu/canonical_outputs.json').read_text())
    gpu_arrays = {'params':np.array(archived['parameters_u32'],np.uint32).view(np.float32),
                  'scores':np.array(archived['scores_u32'],np.uint32).view(np.float32).reshape(600,10),
                  'predictions':np.array(archived['predictions'],np.int32)}
    for mode in ('baseline','energy','no_slowdown'):
        for key,array in gpu_arrays.items():
            assert learner.array_hash(array)==gpu['checks'][mode+'_eager'][key]['sha256']
    accuracy_dir = HERE/'evidence/accuracy'
    plan = json.loads((accuracy_dir/'plan.json').read_text())
    frozen = json.loads((accuracy_dir/'predictions_frozen.json').read_text())
    accuracy = json.loads((accuracy_dir/'accuracy.json').read_text())
    assert sha(accuracy_dir/'plan.json')==frozen['plan_sha256']
    assert sha(accuracy_dir/'predictions_frozen.json')==accuracy['predictions_frozen_sha256']
    assert plan['created_unix_seconds']<frozen['frozen_unix_seconds']<accuracy['scored_unix_seconds']
    for name,digest in plan['source_hashes'].items():
        assert sha(HERE/name)==digest,name
    assert len(accuracy['draws'])==11
    assert [r['seed'] for r in accuracy['draws']]==plan['dataset_seeds']
    counts = []
    for row in accuracy['draws']:
        assert sha(accuracy_dir/row['manifest_file'])==row['manifest_sha256']
        assert sha(accuracy_dir/row['prediction_file'])==row['prediction_file_sha256']
        prediction = np.load(accuracy_dir/row['prediction_file'],allow_pickle=False)
        assert prediction.shape==(600,) and prediction.dtype==np.int64
        assert learner.array_hash(prediction)==row['prediction_array_sha256']
        draw = json.loads((accuracy_dir/row['manifest_file']).read_text())['tiers']['small']
        assert len(set(draw['train_indices']))==len(set(draw['test_indices']))==600
        assert not set(draw['train_indices'])&set(draw['test_indices'])
        counts.append(row['correct'])
    assert sum(counts)==accuracy['correct']>=3960
    values = np.array(counts)/6
    assert np.isclose(values.mean(),accuracy['mean_percent'])
    assert np.isclose(values.std(ddof=1),accuracy['sample_sd_pp'])
    checks['fresh11_hashes_chronology_and_summary_verified'] = True
    for path in HERE.glob('*.md'):
        text = path.read_text()
        assert not re.search(r'[\u0400-\u04ff]',text),('non-English documentation',path.name)
        origin = p.ROOT/'mnist' if path.name=='table-row.md' else path.parent
        for link in re.findall(r'\[[^\]]+\]\(([^)]+)\)',text):
            if not link.startswith(('https://','http://','#')):
                assert (origin/link.split('#')[0]).exists(),(path.name,link)
    page = p.ROOT/'docs/submissions/mlp-panels-v4-20260911/index.html'
    class Links(HTMLParser):
        def handle_starttag(self,tag,attrs):
            if tag=='a':
                href = dict(attrs).get('href','')
                if href and not href.startswith(('https://','http://','#')):
                    assert (page.parent/href.split('#')[0]).exists(),href
    Links().feed(page.read_text())
    expected_row = (HERE/'table-row.md').read_text().strip()
    matching = [line for line in (p.ROOT/'mnist/README.md').read_text().splitlines()
                if '(submissions/mlp-panels-v4-20260911/report.md)' in line]
    assert len(matching)==1 and matching[0].replace('\u00b1','+/-')==expected_row
    checks['english_documentation_links_and_table_row_verified'] = True
    if args.full:
        data = learner.load_inputs(HERE/'generated/canonical-inputs.npz',p.ROOT/'mnist/doc/dataset_manifest.json')
        result = learner.learn(data)
        for key,array in gpu_arrays.items():
            assert result[key].astype(array.dtype).tobytes()==array.tobytes(),key
        checks['full_cpu_matches_measured_gpu_arrays'] = True
        with np.load(HERE/'generated/gpu_reference.npz',allow_pickle=False) as refs:
            for case in ('changed_queries','changed_labels'):
                for key in ('params','scores','predictions'):
                    digest = learner.array_hash(refs[f'{case}_baseline_{key}'])
                    for phase in ('before_timing','after_timing'):
                        for mode in ('baseline','energy','no_slowdown'):
                            assert digest==gpu['checks'][f'{phase}_{mode}_{case}'][key]['sha256']
        checks['portable_gpu_mutation_references_match_measured'] = True
        cases = ((30,7,1,False),(60,37,2,False),(30,3,1,True))
        checks['actual_IL_cases'] = [reference.execute(learner.config(),data,*case) for case in cases]
        with tempfile.TemporaryDirectory(dir=HERE/'generated') as temporary:
            path = Path(temporary)/'poison.npz'
            np.savez(path,**data,test_labels=np.array(['must not be unpickled']*600,dtype=object))
            allowed = learner.load_inputs(path)
            assert set(allowed)==set(learner.KEYS)
            for key in data:
                assert allowed[key].tobytes()==data[key].tobytes()
        checks['learner_ignores_poisoned_test_labels'] = True
        from mnist.code import data as dataset
        filename,md5 = dataset.SOURCES['train_labels']
        label_path = dataset.download_source(p.ROOT/'mnist/data/raw',filename,md5)
        first_manifest = json.loads((accuracy_dir/accuracy['draws'][0]['manifest_file']).read_text())
        assert sha(label_path)==first_manifest['source_sha256']['train_labels']
        raw = dataset.read_idx(label_path,60000,False)
        for row in accuracy['draws']:
            draw = json.loads((accuracy_dir/row['manifest_file']).read_text())['tiers']['small']
            prediction = np.load(accuracy_dir/row['prediction_file'],allow_pickle=False)
            assert int(np.sum(prediction==raw[draw['test_indices']]))==row['correct']
        checks['fresh11_correct_counts_independently_recomputed'] = True
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(checks,indent=2)+'\n')
    print(json.dumps(checks,indent=2))


if __name__=='__main__':
    main()
