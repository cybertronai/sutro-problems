"""Track a fresh matching-model GPU energy audit in Weights & Biases.

The SDK lives on the local controller. GPU telemetry is buffered by the existing
measurement harness and uploaded after measurement, avoiding SDK perturbation.
"""
from pathlib import Path
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time

import modal
import wandb
import modal_same_hardware as benchmark

ROOT = Path(__file__).resolve().parent


def read(name):
    return json.loads((ROOT / name).read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert not (ROOT / 'results-sxm40').exists(), 'Preserve previous results; use a fresh directory.'
    config = {
        'submission': 'medium-pca-qda-20260915',
        'required_gpu': 'NVIDIA A100-SXM4-40GB',
        'task': 'Train on 10000 examples and predict 10000 examples',
        'timed_draw': 0,
        'validation_draws': 11,
        'original_protocol': '5 rounds x 3000 replays; 5 second paired idle windows',
        'independent_protocol': '6000/12000/6000 replays; 10 second paired idle; two 20 second idle-only controls',
        'power_sample_interval_s': 0.05,
        'telemetry': 'NVML total-energy counter and independent sampled-power integration; shared hardware sensors',
        'scope': 'Warm device-resident complete training and prediction; excludes preprocessing, transfers, allocation and capture',
        'wandb_logging': 'Local controller; GPU telemetry buffered and uploaded after timed measurement',
        'source_sha256': {p.name: digest(p) for p in (ROOT / 'source').glob('*.py')},
        'measurement_sha256': digest(ROOT / 'independent_measure.py'),
        'dispatch_sha256': digest(ROOT / 'modal_same_hardware.py'),
        'tracking_sha256': digest(Path(__file__)),
        'wandb_version': wandb.__version__,
    }
    (ROOT / 'plan-wandb.json').write_text(json.dumps(config, indent=2) + '\n')
    with wandb.init(
        entity=os.environ.get('WANDB_ENTITY', 'yaroslavvb'),
        project=os.environ.get('WANDB_PROJECT', 'sutro-mnist-tiers'),
        name='medium-pca-qda-energy-audit-a100-sxm40-' + time.strftime('%Y%m%d-%H%M%S'),
        job_type='energy-reproduction', tags=['AI-authored', 'energy-audit', 'pca-qda', 'A100-SXM4-40GB'],
        notes='Fresh GPU rerun. Energy is per full 10000-training + 10000-prediction task. Distinguishes gross board energy from idle-subtracted energy. SDK runs off-GPU.',
        config=config, dir=str(ROOT),
        settings=wandb.Settings(disable_git=True, x_disable_stats=True, disable_job_creation=True),
    ) as run:
        identity = {'url': run.url, 'id': run.id, 'entity': run.entity, 'project': run.project}
        (ROOT / 'wandb-run.json').write_text(json.dumps(identity, indent=2) + '\n')
        print('WANDB_RUN_URL=' + run.url, flush=True)
        run.summary['audit/status'] = 'running'
        run.define_metric('telemetry/elapsed_s')
        run.define_metric('telemetry/*', step_metric='telemetry/elapsed_s')
        run.define_metric('original/round')
        run.define_metric('original/*', step_metric='original/round')
        run.define_metric('independent/round')
        run.define_metric('independent/*', step_metric='independent/round')
        with modal.enable_output(), benchmark.app.run():
            records, files = benchmark.execute.remote()
        destination = ROOT / 'results-sxm40'
        destination.mkdir()
        (destination / 'execution.json').write_text(json.dumps(records, indent=2) + '\n')
        for name, data in files.items():
            (destination / name).write_bytes(data)
        assert not any(r.get('hardware_unavailable') for r in records), records
        assert all(r.get('returncode', 0) == 0 for r in records), records
        subprocess.run([sys.executable, str(ROOT / 'analyze.py'), '--results', 'results-sxm40'], check=True)
        original = read('results-sxm40/original.json')
        independent = read('results-sxm40/independent.json')
        summary = read('summary-sxm40.json')
        assert summary['all_110000_predictions_match'] and summary['raw_measurement_arithmetic_verified']
        run.config.update({'actual_hardware': summary['hardware'], 'actual_software': summary['software']})
        for i, row in enumerate(original['rounds']):
            run.log({'original/round': i, **{'original/' + k: row[k] for k in (
                'adjusted_mj', 'cuda_ms', 'wall_ms', 'idle_before_w', 'idle_after_w')}})
        for i, row in enumerate(independent['comparisons']):
            run.log({'independent/round': i, 'independent/is_sham': int(row['kind'] == 'sham'),
                     **{'independent/' + k: v for k, v in row.items() if isinstance(v, (int, float)) and 'gross' not in k}})
        samples = sorted(independent['samples'], key=lambda r: r['t'])
        start = samples[0]['t']
        intervals = independent['intervals']
        for row in samples:
            phase = next((r['name'] for r in intervals if r['start']['t'] <= row['t'] <= r['end']['t']), 'gap')
            run.log({'telemetry/elapsed_s': row['t'] - start,
                     'telemetry/active': int(phase.endswith('-active')),
                     'telemetry/sham': int(phase.endswith('-sham')),
                     **{'telemetry/' + k: v for k, v in row.items() if k != 't'}})
        validation = independent['validation']
        run.log({'validation/by_draw': wandb.Table(columns=['draw', 'matches', 'correct'],
                     data=[[r['draw'], r['matches'], r['correct']] for r in validation]),
                 'energy/power_trace': wandb.Image(str(ROOT / 'energy-audit-sxm40.png'))})
        total_matches = sum(r['matches'] for r in validation)
        total_correct = sum(r['correct'] for r in validation)
        metrics = {'audit/status': 'passed', 'validation/prediction_matches': total_matches,
                   'validation/correct': total_correct, 'validation/accuracy': total_correct / 110000,
                   'validation/raw_arithmetic_verified': True,
                   'original/idle_adjusted_mj_median': summary['original_protocol']['adjusted_mj_median'],
                   'independent/cuda_ms_median': statistics.median(summary['independent']['cuda_ms_values']),
                   'sensitivity/trimmed_idle_adjusted_mj': summary['trimmed_idle_sensitivity']['median_mj'],
                   'published/idle_adjusted_mj': read('published-gpu-results.json')['summary']['adjusted_mj_median']}
        for meter in ('counter', 'integrated_power'):
            for kind in ('idle_adjusted',):
                metrics[f'independent/{meter}_{kind}_mj_median'] = summary['independent'][meter][kind]['median_mj']
        metrics['comparison/net_energy_ratio_to_published'] = metrics['independent/integrated_power_idle_adjusted_mj_median'] / metrics['published/idle_adjusted_mj']
        run.summary.update(metrics)
        (ROOT / 'wandb-metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
        artifact = wandb.Artifact('pca-qda-energy-audit-' + run.id, type='energy-audit', metadata=config)
        for folder in ('source', 'results-sxm40'):
            for path in sorted((ROOT / folder).glob('*')):
                if path.is_file(): artifact.add_file(str(path), name=str(path.relative_to(ROOT)))
        for name in ('plan-wandb.json', 'wandb-run.json', 'wandb-metrics.json', 'summary-sxm40.json',
                     'published-gpu-results.json', 'independent_measure.py', 'modal_same_hardware.py',
                     'run_wandb.py', 'analyze.py', 'energy-audit-sxm40.png'):
            artifact.add_file(str(ROOT / name), name=name)
        run.log_artifact(artifact).wait()
        inputs = wandb.Artifact('pca-qda-frozen-inputs-' + run.id, type='dataset',
                                metadata={'draws': 11, 'payload_manifest_sha256': digest(ROOT / 'payloads/manifest.json')})
        inputs.add_dir(str(ROOT / 'payloads'))
        run.log_artifact(inputs).wait()
        print('FINAL_METRICS=' + json.dumps(metrics), flush=True)
    print('FINISHED_WANDB_RUN=' + identity['url'], flush=True)


if __name__ == '__main__':
    main()
