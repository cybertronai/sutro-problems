"""Verify retained validation artifacts before replication or final selection."""
from pathlib import Path
import argparse
from datetime import datetime, timezone
import hashlib
import json
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=REPO / 'mnist/data/medium-train-only.npz')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--study', type=Path, default=HERE)
    parser.add_argument('--replications-complete', action='store_true')
    args = parser.parse_args()
    study = args.study
    protocol = json.loads((study / 'protocol.json').read_text())
    split = json.loads((study / 'validation_split.json').read_text())
    assert sha(study / 'validation_split.json') == protocol['split_file_sha256']
    for file, expected in protocol['source_sha256'].items():
        assert sha(HERE / file) == expected
    with np.load(args.data, allow_pickle=False) as z:
        assert set(z.files) <= {'train_images', 'train_labels', 'train_indices'}
        x, y = z['train_images'], z['train_labels']
    for key, array in [('train_images', x), ('train_labels', y)]:
        assert hashlib.sha256(array.tobytes()).hexdigest() == protocol['input_sha256'][key]
    fit, val = np.array(split['fit_positions']), np.array(split['validation_positions'])
    assert len(fit) == 4800 and len(val) == 1200
    assert np.array_equal(np.sort(np.r_[fit,val]), np.arange(6000))
    expected_provenance = {'protocol_sha256': sha(study / 'protocol.json'),
                          'source_sha256': protocol['source_sha256'], 'input_sha256': protocol['input_sha256'],
                          'split_file_sha256': protocol['split_file_sha256'],
                          'allowed_input_arrays': ['train_images', 'train_labels']}
    configs = {c['id']:c for c in protocol['candidates']}
    search_paths = sorted((study / 'results').glob('search-*.json'))
    assert len(search_paths) == len(configs)
    search = [json.loads(p.read_text()) for p in search_paths]
    rank = lambda r: (-r['best']['validation']['correct'],r['best']['validation']['loss'],r['parameter_count'],r['config']['id'])
    top3 = [r['config']['id'] for r in sorted(search, key=rank)[:3]]
    paths = sorted((study / 'results').glob('*.json'))
    expected = {(c, 'search', 11) for c in configs}
    if args.replications_complete:
        expected |= {(c,'replicate',seed) for c in top3 for seed in (22,33)}
    actual, records = set(), []
    for path in paths:
        r = json.loads(path.read_text())
        c = r['config']; key = (c['id'],r['phase'],r['seed'])
        assert key not in actual
        actual.add(key)
        assert r['provenance'] == expected_provenance and c == configs[c['id']]
        assert r['id'] == f"{r['phase']}-{c['id']}-s{r['seed']}"
        assert path.name == r['id']+'.json'
        assert r['container_image'] == protocol['container_image']
        logits_path = study / r['validation_logits_file']
        logits = np.load(logits_path, allow_pickle=False)
        assert logits.shape == (1200,10) and logits.dtype == np.float32 and np.isfinite(logits).all()
        assert hashlib.sha256(logits.tobytes()).hexdigest() == r['best_validation_logits_sha256']
        predictions = logits.argmax(axis=1)
        assert np.array_equal(predictions, r['best_validation_predictions'])
        assert int(np.count_nonzero(predictions == y[val])) == r['best']['validation']['correct']
        shifted = logits.astype(np.float64) - logits.max(axis=1,keepdims=True).astype(np.float64)
        loss = float(np.mean(np.log(np.exp(shifted).sum(axis=1))-shifted[np.arange(1200),y[val]]))
        assert abs(loss-r['best']['validation']['loss']) < 2e-6
        assert sha(study / r['checkpoint_file']) == r['best_checkpoint_sha256']
        assert [h['epoch'] for h in r['history']] == list(range(1,c['epochs']+1))
        best = min(r['history'], key=lambda h:(-h['validation']['correct'],h['validation']['loss'],h['epoch']))
        assert best == r['best'] and r['last'] == r['history'][-1]
        w,h,d = c['width'],c['head_width'],c['depth']
        spatial = 4 if c['pooling'] == 'max' else 9
        params = 9*w+9*w*w*(d-1)+2*w*d+(spatial*spatial*w+1)*h+(h+1)*10
        assert params == r['parameter_count']
        records.append({'id':r['id'], 'result_sha256':sha(path), 'logits_file_sha256':sha(logits_path),
                        'correct':r['best']['validation']['correct'], 'checkpoint_sha256':r['best_checkpoint_sha256']})
    assert actual == expected, (actual,expected)
    record = {'audited_at_utc':datetime.now(timezone.utc).isoformat(), 'passed':True,
              'test_labels_opened':False, 'replications_complete':args.replications_complete,
              'protocol_sha256':sha(study / 'protocol.json'), 'top3_ids':top3, 'runs':records,
              'checks':['canonical training arrays and disjoint split','source and provenance hashes','exact config and seeds',
                        'logit/checkpoint hashes','recomputed validation predictions and cross-entropy',
                        'history-based checkpoint selection','independent parameter counts']}
    args.output.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({k:v for k,v in record.items() if k != 'runs'},indent=2))


if __name__ == '__main__':
    main()
