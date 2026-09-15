"""Recheck frozen evidence and diagnose differences from the archived H32 run."""
import hashlib
import json
from pathlib import Path
import statistics
import numpy as np
import baseline_reference as baseline
import grid_score
import run

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / 'small60-grid-20260912'

# Main's Adam extension (39da2cc) changed these shared files after this
# experiment. Its unchanged SGD path independently reproduces every panel
# metric and passes the panel tests. Accept only these exact reviewed versions;
# keep the experiment's original source hashes in grid-score.json.
COMPATIBLE_SHARED_SOURCES = {
    'submissions/grid-mlp-scoring-20260912/affine.py': '8ac38bf9fccbef87ba05c4dd6a4e5d9901dceae4e71baad0ddd5794c22298d45',
    'submissions/grid-mlp-scoring-20260912/model_ir.py': 'cc039abc33a731a471c4caf9e35d9c1799ea5246501ae9bea962406a91bcf910',
    'submissions/grid-mlp-scoring-20260912/score.py': '41b47384ec525718a4546f98f5ad25a4d82dd8bdfb97a762e451476cae29eaf8',
}


def main():
    run.evaluate(HERE.parents[1] / 'data/raw', verify=True)
    draws = json.loads((HERE / 'draw_manifest.json').read_text())['draws']
    archived_manifest = json.loads((OLD / 'draw_manifest.json').read_text())
    assert json.loads((HERE / 'draw_manifest.json').read_text())['source_files'] == archived_manifest['source_files']
    archived = archived_manifest['draws']
    comparisons = []
    for new, old in zip(draws, archived, strict=True):
        assert new['dataset_seed'] == old['dataset_seed']
        assert new['train_indices'] == old['train_indices']
        assert new['test_indices'] == old['test_indices']
        comparisons.append({'seed':new['dataset_seed'], 'indices_equal':True,
            'array_hash_equal':{name:new['arrays'][name]['sha256']==old['arrays'][name]['sha256']
                                for name in new['arrays']}})
    with np.load(HERE / draws[0]['archive'], allow_pickle=False) as z:
        x = baseline.transform(z['train_images'])
        q = baseline.transform(z['test_images'])
        target = (z['train_labels'][:,None] == np.arange(10)).astype(np.float32)
    fast, ordered = baseline.parameters(32,101), baseline.parameters(32,101)
    first_difference = None
    for epoch in range(300):
        fast = baseline.epoch(x,target,fast,.2)
        ordered = baseline.epoch(x,target,ordered,.2,baseline.ordered_mm,baseline.ordered_rows)
        if first_difference is None and any(a.tobytes()!=b.tobytes() for a,b in zip(fast,ordered)):
            first_difference = epoch+1
    fast_scores = baseline.forward(q,fast)[2]
    ordered_scores = baseline.forward(q,ordered,baseline.ordered_mm)[2]
    frozen = json.loads((HERE / 'prediction_manifest.json').read_text())['draws'][0]
    assert baseline.digest(ordered_scores) == frozen['scores_sha256']
    old_pred = np.load(OLD / 'predictions/draw-00.npy', allow_pickle=False)
    numerical = {'first_epoch_fast_vs_ordered_parameters_differ':first_difference,
        'local_fast_matches_archived_predictions':bool(np.array_equal(fast_scores.argmax(1),old_pred)),
        'local_fast_vs_archived_prediction_differences':int(np.sum(fast_scores.argmax(1)!=old_pred)),
        'local_ordered_matches_panel_scores':True,
        'local_fast_vs_panel_prediction_differences':int(np.sum(fast_scores.argmax(1)!=ordered_scores.argmax(1)))}
    all_draws = []
    frozen_draws = json.loads((HERE / 'prediction_manifest.json').read_text())['draws']
    for draw, frozen_draw in zip(draws, frozen_draws, strict=True):
        with np.load(HERE / draw['archive'], allow_pickle=False) as z:
            x = baseline.transform(z['train_images'])
            q = baseline.transform(z['test_images'])
            target = (z['train_labels'][:,None] == np.arange(10)).astype(np.float32)
        params = baseline.parameters(32,101)
        fast = baseline.parameters(32,101)
        for _ in range(300):
            params = baseline.epoch(x,target,params,.2,baseline.ordered_mm,baseline.ordered_rows)
            fast = baseline.epoch(x,target,fast,.2)
        scores = baseline.forward(q,params,baseline.ordered_mm)[2]
        assert baseline.digest(np.concatenate([a.ravel() for a in params])) == frozen_draw['parameter_sha256'][0]
        assert baseline.digest(scores) == frozen_draw['scores_sha256']
        np.testing.assert_array_equal(scores.argmax(1),np.load(HERE / frozen_draw['path'],allow_pickle=False))
        archived_pred = np.load(OLD / frozen_draw['path'],allow_pickle=False)
        fast_pred = baseline.forward(q,fast)[2].argmax(1)
        all_draws.append({'seed':draw['dataset_seed'],'ordered_reference_all_bits_match_panel':True,
            'local_fast_vs_archived_prediction_differences':int(np.sum(fast_pred!=archived_pred)),
            'local_fast_vs_panel_prediction_differences':int(np.sum(fast_pred!=scores.argmax(1)))})
        print(f'Independent full ordered reference verified: {draw["dataset_seed"]}',flush=True)
    gpu = json.loads((HERE / 'gpu_measured/results.json').read_text())
    provenance = gpu['provenance']
    for key, path in [('runner_sha256','gpu_benchmark.py'),('generator_sha256','gpu_panels.py'),
                      ('reference_sha256','generated/gpu_reference.npz'),('protocol_sha256','protocol.json')]:
        assert provenance[key] == run.sha(HERE / path)
    assert provenance['generated_sha256'] == run.sha(HERE / 'gpu_measured/kernels.py')
    with np.load(HERE / 'gpu_measured/arrays.npz', allow_pickle=False) as z:
        assert baseline.digest(z['energy_params']) == frozen['parameter_sha256'][0]
        assert baseline.digest(z['energy_scores']) == frozen['scores_sha256']
        np.testing.assert_array_equal(z['energy_predictions'],np.load(HERE / frozen['path'],allow_pickle=False))
    for name, digest in gpu['kernel_ptx_sha256'].items():
        assert run.sha(HERE / 'gpu_measured' / (name+'.ptx')) == digest
    for trial in gpu['trials']:
        active = trial['active']
        idle_power = (trial['idle_before']['average_power_w']+trial['idle_after']['average_power_w'])/2
        expected = (active['energy_j']-idle_power*active['duration_s'])*1000/trial['replays']
        assert abs(expected-trial['adjusted_mj']) < 1e-9
    assert gpu['graphs']['energy'] == {'total':48004,'kernels':48004}
    assert len(gpu['trials']) == 3
    for key in ('cuda_ms','wall_ms','adjusted_mj','gross_mj'):
        assert statistics.mean(t[key] for t in gpu['trials']) == gpu['summary']['energy'][key]['mean']
    grid = json.loads((HERE / 'grid-score.json').read_text())
    compatible_sources_used = {}
    for name, digest in grid['source_sha256'].items():
        actual = run.sha(HERE.parents[1] / name)
        if actual != digest:
            assert actual == COMPATIBLE_SHARED_SOURCES.get(name), name
            compatible_sources_used[name] = {'recorded_sha256': digest, 'current_sha256': actual}
    assert run.sha(HERE / 'program.spatial.json') == grid['program_file_sha256']
    recomputed = grid_score.spatial.score(json.loads((HERE / 'program.spatial.json').read_text()))
    for key in ('energy_fj','cycles','peak_allocated_scratch_bytes','word_node_hops','program_sha256'):
        assert recomputed[key] == grid[key]
    tests = json.loads((HERE / 'grid-test-results.json').read_text())
    assert tests['successful'] and len(tests['panel_checks']) == 4
    result = {'archived_baseline_input_comparison':comparisons, 'raw_source_hashes_match_archived':True,
        'numerical_diagnostic_draw_0':numerical,'all_draw_numerical_checks':all_draws,
        'a100_full_parameter_score_prediction_bits_match_cpu':True,
        'a100_nvml_energy_and_summary_recomputed':True,'grid_costs_and_source_hashes_verified':True,
        'compatible_shared_sources_used':compatible_sources_used,
        'grid_tests':tests['tests_run'],'all_11_accuracy_draws_verified':True,
        'software':{'numpy':np.__version__}, 'audit_sha256':run.sha(Path(__file__))}
    run.write(HERE / 'verification.json',result)
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
