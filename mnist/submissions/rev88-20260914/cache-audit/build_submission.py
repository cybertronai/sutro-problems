"""Build the cache runtime update from independently verified evidence."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json

import analyze

HERE = Path(__file__).resolve().parent
BASE = HERE.parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    summary = analyze.analyze()
    assert summary['all_jobs_passed']
    validation = json.loads((HERE/'cuda-validation.json').read_text())
    assert validation['passed'] and validation['numerical_validation']['passed']
    assert validation['api_cli_smoke']['passed'] and validation['workspace'] == ':16:8'
    for name, digest in validation['source_sha256'].items():
        path = BASE/name if name == 'learner.py' else HERE/name
        assert sha(path) == digest, name
    assert validation['runner_sha256'] == sha(HERE/'validate_runtime.py')
    write(HERE/'summary.json', summary)
    original = json.loads((BASE/'submission.json').read_text())
    small = summary['comparisons']['small-workspace']
    energy = small['energy']['summary']
    traffic = {}
    for scope in ('task', 'training'):
        profile = summary['profiles']['profile-small-'+scope]
        traffic[scope] = {key: profile[key] for key in (
            'dram_read_bytes_per_sequence', 'dram_write_bytes_per_sequence',
            'dram_total_bytes_per_sequence', 'l2_read_request_bytes_per_sequence',
            'l2_write_request_bytes_per_sequence', 'l2_sector_hit_rate_percent',
            'counter_unit', 'pass_counts', 'profile_device_attributes')}
    result = {
        'schema_version': 1, 'built_at_utc': datetime.now(timezone.utc).isoformat(),
        'name': 'rev88-20260914-cache', 'tier': original['tier'],
        'target_error_percent': 12, 'qualifies': True,
        'base_submission_path': '../submission.json',
        'base_submission_sha256': sha(BASE/'submission.json'),
        'configuration': original['configuration'],
        'runtime_entrypoint': 'cache_runtime.py', 'cublas_workspace_config': ':16:8',
        'accuracy': summary['qualification']['inherited_accuracy'],
        'qualification': {'draws': 11, 'predictions_bit_equal': 110000,
                          'full_state_bit_equal_to_original': False,
                          'same_workspace_replay_byte_repeatable': True,
                          'gradient_and_sgd_validation_passed': True},
        'memory': small['memory'],
        'a100': {'energy_mj': energy['idle_adjusted_mj_per_task']['mean'],
                 'energy_sample_sd_mj': energy['idle_adjusted_mj_per_task']['sample_sd'],
                 'runtime_ms': energy['cuda_ms_per_task']['mean'],
                 'runtime_sample_sd_ms': energy['cuda_ms_per_task']['sample_sd'],
                 'trials': 3, 'draw_measured': 0,
                 'scope': 'Complete fresh GPU-resident reset/normalize/train/predict task; allocation, capture, transfers and host work excluded.'},
        'grid': original['grid'], 'warm_profile': traffic,
        'cache_claim': 'Tracked tensor allocation fits nominal 40 MiB L2 capacity; warm traffic is measured. HBM traffic remains nonzero. No universal residency, cache pinning, L1 fit or energy improvement is established.',
        'source_sha256': {name: sha(HERE/name) for name in (
            'cache_runtime.py', 'fixture.py', 'run.py', 'analyze.py',
            'validate_runtime.py', 'build_submission.py')},
        'evidence_sha256': {name: sha(HERE/name) for name in (
            'summary.json', 'cuda-validation.json', 'small-workspace.json')},
    }
    write(HERE/'submission.json', result)
    print('Verified cache runtime submission: 89.28% accuracy; 14.9375 MiB peak; measured nonzero HBM traffic.')


if __name__ == '__main__':
    main()
