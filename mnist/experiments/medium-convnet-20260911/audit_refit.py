"""Independently check frozen refit artifacts without reading query labels."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def check(selection_path, output):
    frozen = json.loads(selection_path.read_text())
    for name,digest in frozen['source_sha256'].items():
        assert file_sha(Path(__file__).resolve().parent/name) == digest
    assert file_sha(selection_path.parent/'protocol.json') == frozen['protocol_sha256']
    assert file_sha(selection_path.parent/'validation_selection.json') == frozen['validation_selection_sha256']
    summary = json.loads((output/'refit_summary.json').read_text())
    plan = json.loads((output/'refit_plan.json').read_text())
    provenance = summary['provenance']
    assert plan['selection'] == frozen and plan['provenance'] == provenance
    assert provenance['selection_sha256'] == file_sha(selection_path)
    assert provenance['source_sha256'] == frozen['source_sha256']
    assert provenance['input_sha256'] == frozen['input_sha256']
    assert provenance['protocol_sha256'] == frozen['protocol_sha256']
    assert provenance['validation_selection_sha256'] == frozen['validation_selection_sha256']
    assert provenance['allowed_input_arrays'] == ['test_images','train_images','train_labels']
    assert provenance['checkpoint_inputs'] == []
    assert summary['selected_name'] == frozen['selected_name']
    assert frozen['selected_name'] == ('ensemble' if frozen['selected_inference']=='ensemble' else 'seed101')
    assert summary['seeds'] == frozen['seeds'] == [101,102,103]
    assert summary['epochs'] == frozen['epochs']
    assert summary['config'] == frozen['config']
    assert summary['test_labels_opened'] is False
    config = frozen['config']; c = config['width']; d = config['depth']; h = config['head_width']
    spatial = 4 if config['pooling'] == 'max' else 9
    parameters = 9*c+(d-1)*9*c*c + 2*c*d + c*spatial*spatial*h+h+h*10+10
    seeds, logits = [], []
    for seed in frozen['seeds']:
        result = json.loads((output/f'result-seed{seed}.json').read_text())
        assert result['seed'] == seed and result['config'] == config
        assert result['epochs'] == frozen['epochs'] and result['schedule_epochs'] == 100
        assert result['provenance'] == provenance
        assert result['parameter_count'] == parameters
        assert result['training_examples'] == result['query_examples'] == 6000
        assert result['normalization']['source'] == 'all 6000 training images only'
        assert np.isfinite([result['normalization']['mean'],result['normalization']['std']]).all()
        assert result['normalization']['std'] > 0
        assert [x['epoch'] for x in result['history']] == list(range(1,frozen['epochs']+1))
        for row in result['history']:
            assert row['train']['total'] == 6000 and 0 <= row['train']['correct'] <= 6000
            assert np.isfinite(row['train']['loss']) and row['train']['loss'] >= 0
            expected = config['learning_rate']*(.02+.98*(1+np.cos(np.pi*(row['epoch']-1)/100))/2)
            assert np.isclose(row['learning_rate'],expected,rtol=1e-12,atol=1e-15)
        for name,path in result['files'].items():
            assert file_sha(output/path) == result['file_sha256'][name]
        assert result['file_sha256']['checkpoint'] == result['checkpoint_sha256']
        values = np.load(output/result['files']['logits'],allow_pickle=False)
        predictions = np.load(output/result['files']['predictions'],allow_pickle=False)
        assert values.dtype == np.float32 and values.shape == (6000,10) and np.isfinite(values).all()
        assert predictions.dtype == np.int64 and predictions.shape == (6000,)
        assert array_sha(values) == result['logits_sha256']
        assert array_sha(predictions) == result['predictions_sha256']
        assert np.array_equal(predictions,values.argmax(1))
        assert result['determinism'] == {'algorithms':True,'cudnn_benchmark':False,
            'cudnn_deterministic':True,'tf32_matmul':False,'tf32_cudnn':False}
        assert datetime.fromisoformat(result['completed_at_utc']) > datetime.fromisoformat(frozen['frozen_at_utc'])
        seeds.append(seed); logits.append(values)
    ensemble = np.mean(np.stack(logits),axis=0,dtype=np.float64)
    stored = np.load(output/'logits-ensemble.npy',allow_pickle=False)
    predictions = np.load(output/'predictions-ensemble.npy',allow_pickle=False)
    assert stored.dtype == np.float64 and np.array_equal(stored,ensemble)
    assert predictions.dtype == np.int64 and np.array_equal(predictions,ensemble.argmax(1))
    assert array_sha(stored) == summary['ensemble_logits_sha256']
    assert array_sha(predictions) == summary['ensemble_predictions_sha256']
    for name,digest in summary['ensemble_file_sha256'].items(): assert file_sha(output/name) == digest
    report = {'checked_at_utc':datetime.now(timezone.utc).isoformat(),'passed':True,
        'seeds':seeds,'parameter_count_per_model':parameters,'epochs':frozen['epochs'],
        'selection_sha256':file_sha(selection_path),'refit_summary_sha256':file_sha(output/'refit_summary.json'),
        'checks':['frozen provenance and input/source hashes','fresh-model plan with no checkpoint inputs',
          'fixed training history and cosine learning rates','finite logits and prediction argmax',
          'checkpoint byte hashes (checkpoint tensors were not unpickled)',
          'exact float64 ensemble mean and argmax','deterministic backend flags'],
        'test_labels_opened':False,'audit_source_sha256':file_sha(Path(__file__))}
    (output/'artifact_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args(); check(args.selection,args.output)
