"""Prepare isolated inputs, freeze all predictions, then evaluate eleven draws.

This program never imports the learner. Query labels are extracted only by the
evaluate command, after all eleven prediction archives have been globally hashed.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from mnist.code import data as generator

DATASET_SEEDS = list(range(20261101, 20261112))
INPUT_NAMES = {'train_images', 'train_labels', 'test_images'}
ATTEMPT_TARGET = 96
STRICTEST_TARGET = 98
ERROR_TARGETS = [10,8,6,4,2]
REQUIRED_CORRECT = 105600
RAW_LABEL_HASH = '3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def read(path):
    return json.loads(path.read_text())


def spec(array):
    return dict(shape=list(array.shape), dtype=str(array.dtype), sha256=generator.array_hash(array))


def safe_path(relative):
    path = Path(relative)
    require(not path.is_absolute() and '..' not in path.parts, 'Expected a submission-relative path')
    return HERE / path


def prepare(args):
    require(not (HERE/'protocol.json').exists(), 'Protocol already frozen')
    config = read(args.config)
    require(config['target_percent'] == ATTEMPT_TARGET, 'This attempt targets 4 percent error / 96 percent accuracy')
    require(config['ensemble_arithmetic'] == 'ordered_fp32_sum', 'Declare FP32 ensemble order')
    require(len(config['member_seeds']) >= 1 and len(set(config['member_seeds'])) == len(config['member_seeds']),
            'Expected distinct fixed member seeds')
    source_names = sorted(set(args.sources + ['evaluation.py']))
    source_hashes = {name: sha(safe_path(name)) for name in source_names}
    raw_sources = {}
    for name in ('train_images', 'train_labels'):
        filename, md5 = generator.SOURCES[name]
        path = args.raw/filename
        require(generator.file_hash(path, 'md5') == md5, 'Wrong canonical raw MNIST file')
        raw_sources[name] = dict(filename=filename, sha256=sha(path), md5=md5)
    protocol = dict(schema='sutro-mnist-eleven-draw/2', frozen_at_utc=now(), tier='medium',
        profile='medium-error-targets-v1', target_error_percent=4, error_targets=ERROR_TARGETS, target_percent=ATTEMPT_TARGET, strictest_target_percent=STRICTEST_TARGET, config=config,
        dataset_seeds=DATASET_SEEDS, n_train=10000, n_test=10000, source_pool=60000,
        source_sha256=source_hashes, generator_sha256=sha(Path(generator.__file__)),
        raw_sources=raw_sources, allowed_input_arrays=sorted(INPUT_NAMES),
        preparation_software=dict(python=platform.python_version(), numpy=np.__version__),
        sampling='PCG64(SeedSequence(seed).spawn(2)[0]); permutation[:10000] train, [10000:20000] query',
        overlap='Disjoint within each draw; independent draws may overlap',
        resize='Canonical float32 uint8/255, separable exact box-area 28-to-9, clip to [0,1]',
        reset='Fresh learned parameters and optimizer state for every member of every draw',
        prediction_freeze='All eleven archives frozen before deriving or scoring any query label',
        accuracy_rule='Unrounded mean across 11; sample SD in percentage points, ddof=1; inclusive 105600/110000 for this 96 percent attempt',
        selection='Configuration and stopping fixed before these draws; no per-draw tuning or early stopping',
        development_disclosure='Training-only architecture search used a previous canonical training subset. '
            'Previous attempts informed the architecture family. This attempt uses a new fixed set of eleven draws.',
        test_labels_created=False)
    write_new(HERE/'protocol.json', protocol)
    images = generator.read_idx(args.raw/raw_sources['train_images']['filename'], 60000, True)
    labels = generator.read_idx(args.raw/raw_sources['train_labels']['filename'], 60000, False)
    entries = []
    for index, seed in enumerate(DATASET_SEEDS):
        order, _ = generator.source_permutations(seed=seed)
        fit, query = order[:10000], order[10000:20000]
        require(len(np.unique(np.r_[fit, query])) == 20000, 'Within-draw overlap')
        arrays = {'train_labels': labels[fit].copy()}
        for name, positions in [('train_images', fit), ('test_images', query)]:
            pixels = images[positions].astype(np.float32)/np.float32(255)
            resized = generator.area_resize(pixels, 9)
            np.clip(resized, 0, 1, out=resized)
            arrays[name] = np.ascontiguousarray(resized[:,None])
        archive = HERE/'data'/f'draw-{index:02d}.npz'
        archive.parent.mkdir(exist_ok=True)
        require(not archive.exists(), 'Refusing to overwrite a draw')
        np.savez_compressed(archive, **arrays)
        relative = f'draws/draw-{index:02d}.json'
        metadata = dict(draw_index=index, dataset_seed=seed, protocol_sha256=sha(HERE/'protocol.json'),
            archive=str(archive.relative_to(HERE)), archive_sha256=sha(archive),
            arrays={name:spec(array) for name,array in arrays.items()},
            train_indices=fit.tolist(), test_indices=query.tolist(), test_labels_created=False)
        write_new(safe_path(relative), metadata)
        entries.append(dict(draw_index=index, dataset_seed=seed, path=relative, sha256=sha(safe_path(relative))))
        print(f'Prepared draw {index:02d}, seed {seed}; no query labels extracted', flush=True)
    write_new(HERE/'draw_manifest.json', dict(protocol_sha256=sha(HERE/'protocol.json'),
        draws=entries, created_at_utc=now(), test_labels_created=False))


def verify_protocol():
    protocol = read(HERE/'protocol.json')
    master = read(HERE/'draw_manifest.json')
    require(protocol['dataset_seeds'] == DATASET_SEEDS and protocol['target_percent'] == ATTEMPT_TARGET,
            'Changed evaluation draw set or target')
    require(protocol['n_train'] == protocol['n_test'] == 10000 and protocol['source_pool'] == 60000,
            'Wrong dataset sizes')
    require(protocol['config']['ensemble_arithmetic'] == 'ordered_fp32_sum', 'Changed ensemble rule')
    for name,digest in protocol['source_sha256'].items():
        require(sha(safe_path(name)) == digest, 'Source changed after freeze: '+name)
    require(sha(Path(generator.__file__)) == protocol['generator_sha256'], 'Changed data generator')
    require(master['protocol_sha256'] == sha(HERE/'protocol.json'), 'Changed protocol')
    require([row['draw_index'] for row in master['draws']] == list(range(11)), 'Missing or reordered draws')
    draws = []
    for row in master['draws']:
        require(sha(safe_path(row['path'])) == row['sha256'], 'Changed draw metadata')
        draw = read(safe_path(row['path']))
        index = row['draw_index']
        require(draw['dataset_seed'] == row['dataset_seed'] == DATASET_SEEDS[index], 'Wrong dataset seed')
        require(draw['protocol_sha256'] == sha(HERE/'protocol.json'), 'Wrong draw provenance')
        require(set(draw['arrays']) == INPUT_NAMES, 'Unexpected learner arrays')
        for name,entry in draw['arrays'].items():
            shape = [10000] if name == 'train_labels' else [10000,1,9,9]
            dtype = 'int64' if name == 'train_labels' else 'float32'
            require(entry['shape'] == shape and entry['dtype'] == dtype, 'Wrong input shape or dtype')
        order,_ = generator.source_permutations(seed=DATASET_SEEDS[index])
        require(draw['train_indices'] == order[:10000].tolist() and draw['test_indices'] == order[10000:20000].tolist(),
                'Wrong source rows')
        draws.append(draw)
    return protocol,master,draws


def validate_outputs(protocol, draws):
    """Only prediction/provenance files; no training or raw label access."""
    rows, verified = [], []
    seeds = protocol['config']['member_seeds']
    names = {'predictions'} | {f'logits_seed{seed}' for seed in seeds}
    for draw in draws:
        index = draw['draw_index']
        relative = f'predictions/draw-{index:02d}.npz'
        result_relative = f'results/draw-{index:02d}.json'
        result = read(safe_path(result_relative))
        require(result['draw_index'] == index and result['dataset_seed'] == draw['dataset_seed'], 'Wrong result draw')
        require(result['protocol_sha256'] == sha(HERE/'protocol.json'), 'Wrong result protocol')
        require(result['source_sha256'] == protocol['source_sha256'], 'Wrong executed source versions')
        require(result['config'] == protocol['config'], 'Wrong executed learner configuration')
        require(result['input_sha256'] == {k:v['sha256'] for k,v in draw['arrays'].items()}, 'Wrong learner inputs')
        require(result['member_seeds'] == seeds and result['test_labels_opened'] is False, 'Wrong evaluation scope')
        require(result['fresh_state_per_member'] is True, 'Learned state was reused')
        members=result['members']
        require([m['seed'] for m in members] == seeds, 'Wrong member order or count')
        for member in members:
            expected=dict(protocol['config'])
            expected['epochs']=expected.get('member_epochs',{}).get(str(member['seed']),expected['epochs'])
            require(member['config'] == expected and member['epochs'] == expected['epochs'], 'Wrong member configuration or stopping epoch')
            require(member['minibatches_per_epoch'] == (10000+expected['batch_size']-1)//expected['batch_size'], 'Wrong training minibatch count')
            require(member['fresh_state'] is True, 'Member learned state was reused')
        require(result['prediction_archive_sha256'] == sha(safe_path(relative)), 'Changed output archive')
        with np.load(safe_path(relative),allow_pickle=False) as archive:
            require(set(archive.files) == names, 'Unexpected prediction archive contents')
            pred = archive['predictions']
            require(pred.shape == (10000,) and pred.dtype == np.dtype('int64') and
                np.all((pred >= 0) & (pred <= 9)), 'Invalid predictions')
            accumulated = np.zeros((10000,10), dtype=np.float32)
            for seed in seeds:
                logits = archive[f'logits_seed{seed}']
                require(logits.shape == (10000,10) and logits.dtype == np.dtype('float32') and
                        np.isfinite(logits).all(), 'Invalid member logits')
                member=next(m for m in members if m['seed']==seed)
                require(generator.array_hash(logits) == member['logits_sha256'], 'Member logit hash mismatch')
                accumulated = np.add(accumulated, logits, dtype=np.float32)
            require(np.array_equal(pred, accumulated.argmax(axis=1)), 'Predictions disagree with ordered FP32 ensemble')
        rows.append(dict(draw_index=index,dataset_seed=draw['dataset_seed'],path=relative,
            sha256=sha(safe_path(relative)),result_path=result_relative,result_sha256=sha(safe_path(result_relative)),
            predictions_sha256=generator.array_hash(pred)))
        verified.append(pred.copy())
    return rows,verified


def freeze(args):
    protocol,master,draws = verify_protocol()
    rows,_ = validate_outputs(protocol,draws)
    write_new(HERE/'prediction_manifest.json', dict(frozen_at_utc=now(),
        protocol_sha256=sha(HERE/'protocol.json'), draw_manifest_sha256=sha(HERE/'draw_manifest.json'),
        predictions=rows, total_draws=11,total_predictions=110000,test_labels_opened=False))
    print('All 110,000 predictions frozen; query labels have not been opened')


def summarize(counts):
    require(len(counts) == 11 and all(type(c) is int and 0 <= c <= 10000 for c in counts), 'Invalid accuracy counts')
    mean = Fraction(100*sum(counts),110000)
    variance = sum((Fraction(c,100)-mean)**2 for c in counts)/10
    return dict(total_correct=sum(counts), total_predictions=110000,
        mean_accuracy_percent=float(mean),mean_error_rate_percent=float(100-mean),
        sample_standard_deviation_pp=math.sqrt(float(variance)),
        error_targets={str(error):dict(required_correct=110000*(100-error)//100,
            meets_target=sum(counts)>=110000*(100-error)//100) for error in ERROR_TARGETS},
        sample_variance_exact=dict(numerator=variance.numerator,denominator=variance.denominator),
        ddof=1,target_percent=ATTEMPT_TARGET,required_correct=REQUIRED_CORRECT,
        meets_target=sum(counts)>=REQUIRED_CORRECT,margin_correct=sum(counts)-REQUIRED_CORRECT,
        strictest_target_percent=STRICTEST_TARGET,meets_strictest_target=sum(counts)>=107800)


def evaluate(args):
    protocol,master,draws = verify_protocol()
    rows,predictions = validate_outputs(protocol,draws)
    frozen = read(HERE/'prediction_manifest.json')
    require(frozen['protocol_sha256'] == sha(HERE/'protocol.json') and
        frozen['draw_manifest_sha256'] == sha(HERE/'draw_manifest.json') and
        frozen['predictions'] == rows and frozen['test_labels_opened'] is False,
        'Global prediction freeze failed verification')
    raw = args.raw/generator.SOURCES['train_labels'][0]
    require(sha(raw) == RAW_LABEL_HASH, 'Wrong official label file')
    # This is the first command that extracts query-label arrays.
    labels = generator.read_idx(raw,60000,False)
    counts,records = [],[]
    for draw,pred in zip(draws,predictions):
        truth = labels[np.asarray(draw['test_indices'])]
        correct = int(np.count_nonzero(pred == truth))
        counts.append(correct)
        records.append(dict(draw_index=draw['draw_index'],dataset_seed=draw['dataset_seed'],
            correct=correct,total=10000,accuracy_percent=correct/100,
            test_labels_sha256=generator.array_hash(truth)))
    output = dict(**summarize(counts),draws=records,evaluated_at_utc=now(),
        prediction_manifest_sha256=sha(HERE/'prediction_manifest.json'),
        protocol_sha256=sha(HERE/'protocol.json'),raw_labels_sha256=RAW_LABEL_HASH,
        evaluator_sha256=sha(Path(__file__)),predictions_globally_frozen_before_query_labels=True)
    write_new(HERE/'accuracy.json',output)
    print(json.dumps(output,indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='phase',required=True)
    p = sub.add_parser('prepare'); p.add_argument('--raw',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True); p.add_argument('--sources',nargs='+',required=True)
    sub.add_parser('freeze')
    p = sub.add_parser('evaluate'); p.add_argument('--raw',type=Path,required=True)
    args = parser.parse_args()
    {'prepare':prepare,'freeze':freeze,'evaluate':evaluate}[args.phase](args)


if __name__ == '__main__':
    main()
