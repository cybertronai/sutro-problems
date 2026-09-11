"""Freeze all eleven prediction records before separately evaluating accuracy.

The freeze phase opens no raw labels. The evaluate phase verifies the complete
global freeze again before deriving query labels from the official training IDX.
This tool imports no learner code and does not train or select any model.
"""
import argparse
from datetime import datetime, timezone
from fractions import Fraction
import gzip
import hashlib
import json
import math
from pathlib import Path
import platform
import struct

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DATASET_SEEDS = list(range(20261001, 20261012))
MEMBER_SEEDS = [101, 102, 103]
RAW_LABEL_SHA256 = '3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c'


def now():
    return datetime.now(timezone.utc).isoformat()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array_hash(value):
    value = np.ascontiguousarray(value.astype(value.dtype.newbyteorder('<'), copy=False))
    return hashlib.sha256(value.tobytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def write_new(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as output:
        output.write(json.dumps(record, indent=2, allow_nan=False) + '\n')


def artifact_path(study, relative):
    require(isinstance(relative, str), 'Artifact path must be a relative string')
    path = Path(relative)
    require(not path.is_absolute() and '..' not in path.parts, 'Artifact path must stay within the study')
    return study / path


def predictions(value):
    require(value.shape == (6000,) and value.dtype == np.dtype('int64'), 'Expected 6000 int64 predictions')
    require(bool(np.all((value >= 0) & (value <= 9))), 'Predictions must be digits 0 through 9')
    return value


def logits(value, dtype):
    require(value.shape == (6000, 10) and value.dtype == np.dtype(dtype), 'Unexpected logit shape or dtype')
    require(bool(np.isfinite(value).all()), 'Nonfinite logits')
    return value


def validate_global_freeze(study):
    """Read predictions and provenance only. Never open a raw label source."""
    protocol = read_json(study / 'protocol.json')
    master = read_json(study / 'draw_manifest.json')
    frozen = read_json(study / 'prediction_manifest.json')
    protocol_hash = sha(study / 'protocol.json')
    master_hash = sha(study / 'draw_manifest.json')
    require(protocol['tier'] == 'medium' and protocol['dataset_profile'] == 'competition-v2', 'Wrong dataset profile')
    require(protocol['dataset_seeds'] == DATASET_SEEDS, 'Expected the eleven predeclared dataset seeds')
    require(protocol['training_examples'] == protocol['test_examples'] == 6000 and
            protocol['source_pool_size'] == 60000, 'Incorrect dataset sizes')
    require(protocol['target_percent'] == '98', 'Expected the frozen 98% target')
    require(protocol['member_seeds'] == MEMBER_SEEDS and protocol['selected_inference'] == 'ensemble',
            'Changed learner membership or inference rule')
    require(protocol['epochs'] == 71 and protocol['schedule_epochs'] == 100, 'Changed stopping policy')
    expected_config = {'id': 'cnn-09', 'architecture': 'cnn', 'width': 64, 'depth': 3,
        'activation': 'gelu', 'pooling': 'none', 'dropout': 0.2, 'learning_rate': 0.001,
        'weight_decay': 0.001, 'augmentation': 'mild_affine', 'batch_size': 128,
        'epochs': 100, 'head_width': 256, 'image_size': 9}
    require(protocol['config'] == expected_config, 'Changed frozen configuration')
    require(set(protocol['source_sha256']) == {'learner.py', 'prepare_draws.py', 'modal_accuracy.py'},
            'Unexpected source manifest')
    for name, digest in protocol['source_sha256'].items():
        require(sha(HERE / name) == digest, 'Changed source: ' + name)
    require(sha(REPO / 'mnist/code/data.py') == protocol['generator_sha256'], 'Changed dataset generator')
    require(master['protocol_sha256'] == frozen['protocol_sha256'] == protocol_hash, 'Changed protocol provenance')
    require(frozen['draw_manifest_sha256'] == master_hash, 'Changed draw manifest')
    require(frozen['selected_inference'] == 'ensemble' and frozen['member_seeds'] == MEMBER_SEEDS,
            'Changed selected predictor')
    require(frozen['total_draws'] == 11 and frozen['total_predictions'] == 66000 and
            frozen['test_labels_opened'] is False, 'Incomplete global prediction freeze')
    require(len(master['draws']) == len(frozen['predictions']) == 11, 'Expected exactly eleven draws')
    files = {name: sha(study / name) for name in ('protocol.json', 'draw_manifest.json', 'prediction_manifest.json')}
    common = {'protocol_sha256': protocol_hash, 'source_sha256': protocol['source_sha256'],
        'generator_sha256': protocol['generator_sha256'], 'draw_manifest_sha256': master_hash,
        'allowed_input_arrays': ['train_images', 'train_labels', 'test_images'],
        'checkpoint_inputs': [], 'state_policy': 'Fresh model, optimizer, scheduler and RNG for every member/draw'}
    metadata_by_index = {entry['draw_index']: entry for entry in master['draws']}
    prediction_by_index = {entry['draw_index']: entry for entry in frozen['predictions']}
    require(set(metadata_by_index) == set(prediction_by_index) == set(range(11)), 'Missing or duplicate draw indices')
    verified = []
    for index in range(11):
        metadata = metadata_by_index[index]
        entry = prediction_by_index[index]
        seed = DATASET_SEEDS[index]
        require(metadata['dataset_seed'] == entry['dataset_seed'] == seed, 'Incorrect dataset seed')
        draw_path = artifact_path(study, metadata['path'])
        require(sha(draw_path) == metadata['sha256'], 'Changed draw metadata')
        files[metadata['path']] = metadata['sha256']
        draw = read_json(draw_path)
        require(draw['draw_index'] == index and draw['dataset_seed'] == seed and
                draw['protocol_sha256'] == protocol_hash and draw['generator_sha256'] == protocol['generator_sha256'],
                'Incorrect draw provenance')
        require(draw['profile'] == 'competition-v2' and draw['source_pool_size'] == 60000 and
                draw['train_test_disjoint'] is True and draw['test_labels_created'] is False, 'Incorrect draw scope')
        fit = np.asarray(draw['train_indices'], dtype=np.int64)
        query = np.asarray(draw['test_indices'], dtype=np.int64)
        require(fit.shape == query.shape == (6000,), 'Incorrect index count')
        child = np.random.SeedSequence(seed).spawn(2)[0]
        permutation = np.random.Generator(np.random.PCG64(child)).permutation(60000)
        require(np.array_equal(fit, permutation[:6000]) and np.array_equal(query, permutation[6000:12000]),
                'Source indices do not match the declared draw')
        require(len(np.unique(np.concatenate((fit, query)))) == 12000, 'Within-draw overlap')
        require(array_hash(fit) == draw['train_indices_spec']['sha256'] and
                array_hash(query) == draw['test_indices_spec']['sha256'], 'Changed source-index hashes')
        require(set(draw['arrays']) == {'train_images', 'train_labels', 'test_images'}, 'Unexpected learner input')
        for name in draw['arrays']:
            spec = draw['arrays'][name]
            expected_shape = [6000] if name == 'train_labels' else [6000, 1, 9, 9]
            expected_dtype = 'int64' if name == 'train_labels' else 'float32'
            require(spec['shape'] == expected_shape and spec['dtype'] == expected_dtype, 'Wrong input shape or dtype')
        for path_key, hash_key in (('path', 'sha256'), ('logits_path', 'logits_sha256'), ('result_path', 'result_sha256')):
            path = artifact_path(study, entry[path_key])
            require(sha(path) == entry[hash_key], 'Changed frozen artifact: ' + entry[path_key])
            files[entry[path_key]] = entry[hash_key]
        pred = predictions(np.load(artifact_path(study, entry['path']), allow_pickle=False))
        require(array_hash(pred) == entry['array_sha256'], 'Changed prediction array hash')
        result = read_json(artifact_path(study, entry['result_path']))
        require(result['draw_index'] == index and result['dataset_seed'] == seed and
                result['protocol_sha256'] == protocol_hash and result['manifest_sha256'] == metadata['sha256'],
                'Incorrect result provenance')
        require(result['predictions_file'] == entry['path'] and result['predictions_file_sha256'] == entry['sha256'] and
                result['predictions_sha256'] == entry['array_sha256'] and
                result['logits_file'] == entry['logits_path'] and result['logits_file_sha256'] == entry['logits_sha256'] and
                result['test_labels_opened'] is False, 'Result does not match the global freeze')
        require(len(result['members']) == 3 and [m['seed'] for m in result['members']] == MEMBER_SEEDS,
                'Incorrect member set or ordering')
        expected_provenance = {**common, 'draw_index': index, 'dataset_seed': seed,
            'manifest_sha256': metadata['sha256'],
            'input_sha256': {name: spec['sha256'] for name, spec in draw['arrays'].items()}}
        member_predictions, member_logits = {}, []
        with np.load(artifact_path(study, entry['logits_path']), allow_pickle=False) as archive:
            require(set(archive.files) == {'seed101', 'seed102', 'seed103', 'ensemble',
                    'predictions_seed101', 'predictions_seed102', 'predictions_seed103'}, 'Unexpected logit archive contents')
            for member in result['members']:
                member_seed = member['seed']
                require(member['provenance'] == expected_provenance and member['config'] == expected_config,
                        'Incorrect member provenance or configuration')
                require(member['epochs'] == 71 and member['schedule_epochs'] == 100 and
                        member['training_examples'] == member['query_examples'] == 6000 and
                        member['parameter_count'] == 1404618, 'Incorrect training extent or model size')
                require([h['epoch'] for h in member['history']] == list(range(1, 72)), 'Incomplete training history')
                require(member['container_image'] == protocol['container_image'], 'Incorrect training image')
                values = logits(archive['seed' + str(member_seed)], 'float32')
                member_pred = predictions(archive['predictions_seed' + str(member_seed)])
                require(array_hash(values) == member['logits_sha256'] and
                        array_hash(member_pred) == member['predictions_sha256'] and
                        np.array_equal(member_pred, values.argmax(axis=1)), 'Incorrect member logits or predictions')
                if index == 0:
                    checkpoint = artifact_path(study, member['checkpoint_file'])
                    require(sha(checkpoint) == member['checkpoint_sha256'], 'Changed retained checkpoint')
                    files[member['checkpoint_file']] = member['checkpoint_sha256']
                member_logits.append(values)
                member_predictions[str(member_seed)] = member_pred
            ensemble = logits(archive['ensemble'], 'float64')
            recomputed = np.mean(np.stack(member_logits), axis=0, dtype=np.float64)
            require(np.array_equal(ensemble, recomputed), 'Ensemble does not equal the frozen arithmetic mean')
            require(array_hash(ensemble) == result['ensemble_logits_sha256'] and
                    np.array_equal(pred, ensemble.argmax(axis=1)), 'Incorrect ensemble prediction')
        verified.append({'draw_index': index, 'dataset_seed': seed, 'query_indices': query,
                         'predictions': pred, 'member_predictions': member_predictions})
    return protocol, frozen, files, verified


def summarize_counts(counts):
    require(len(counts) == 11 and all(type(c) is int and 0 <= c <= 6000 for c in counts),
            'Expected exactly eleven integer counts from 0 through 6000')
    total_correct = sum(counts)
    mean_percent = Fraction(100 * total_correct, 66000)
    variance_pp = sum((Fraction(c, 60) - mean_percent) ** 2 for c in counts) / 10
    return {'total_correct': total_correct, 'total_predictions': 66000,
        'mean_accuracy': total_correct / 66000, 'mean_accuracy_percent': float(mean_percent),
        'sample_standard_deviation_pp': math.sqrt(float(variance_pp)), 'ddof': 1,
        'sample_variance_pp_squared_exact': {'numerator': variance_pp.numerator, 'denominator': variance_pp.denominator},
        'accuracy_target_percent': '98', 'required_total_correct': 64680,
        'meets_accuracy_target': total_correct >= 64680, 'margin_correct': total_correct - 64680}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['freeze', 'evaluate'])
    parser.add_argument('--study', type=Path, default=HERE)
    parser.add_argument('--raw', type=Path, help='Directory containing official training IDX gzip files; evaluate only')
    parser.add_argument('--freeze-manifest', type=Path)
    parser.add_argument('--output', type=Path, help='Accuracy JSON; evaluate only, default STUDY/accuracy.json')
    args = parser.parse_args()
    study = args.study
    freeze_path = args.freeze_manifest or study / 'evaluation_freeze.json'
    output_path = args.output or study / 'accuracy.json'
    evaluator_hash = sha(Path(__file__))
    if args.phase == 'freeze':
        require(not freeze_path.exists(), 'Evaluation freeze already exists')
        protocol, frozen, files, _ = validate_global_freeze(study)
        record = {'schema_version': 1, 'frozen_at_utc': now(), 'evaluator_sha256': evaluator_hash,
            'protocol_sha256': files['protocol.json'], 'prediction_manifest_sha256': files['prediction_manifest.json'],
            'evidence_sha256': files, 'dataset_seeds': DATASET_SEEDS, 'member_seeds': MEMBER_SEEDS,
            'selected_inference': 'ensemble', 'total_draws': 11, 'total_predictions': 66000,
            'prediction_frozen_at_utc': frozen['frozen_at_utc'], 'raw_labels_opened': False}
        write_new(freeze_path, record)
        print(json.dumps({k: v for k, v in record.items() if k != 'evidence_sha256'}, indent=2))
        return
    require(args.raw is not None, '--raw is required for evaluate')
    require(not output_path.exists(), 'Accuracy output already exists; do not overwrite historical results')
    evaluation_freeze = read_json(freeze_path)
    require(evaluation_freeze['evaluator_sha256'] == evaluator_hash, 'Evaluator changed after freeze')
    protocol, frozen, files, verified = validate_global_freeze(study)
    require(evaluation_freeze['evidence_sha256'] == files, 'Evidence changed after evaluation freeze')
    require(evaluation_freeze['dataset_seeds'] == DATASET_SEEDS and evaluation_freeze['member_seeds'] == MEMBER_SEEDS and
            evaluation_freeze['selected_inference'] == 'ensemble' and evaluation_freeze['total_draws'] == 11 and
            evaluation_freeze['total_predictions'] == 66000 and evaluation_freeze['raw_labels_opened'] is False,
            'Incorrect evaluation freeze')
    # Every draw, member, prediction, and evidence hash has passed before this boundary.
    labels_opened_at = now()
    source = protocol['raw_sources']['train_labels']
    require(source['filename'] == 'train-labels-idx1-ubyte.gz' and source['sha256'] == RAW_LABEL_SHA256,
            'Wrong official label source')
    compressed = (args.raw / source['filename']).read_bytes()
    require(hashlib.sha256(compressed).hexdigest() == RAW_LABEL_SHA256, 'Raw label source hash mismatch')
    payload = gzip.decompress(compressed)
    require(len(payload) == 60008 and struct.unpack('>II', payload[:8]) == (2049, 60000), 'Invalid label IDX header')
    labels = np.frombuffer(payload, dtype=np.uint8, offset=8)
    require(bool(np.all(labels <= 9)), 'Invalid raw digit label')
    rows = []
    for draw in verified:
        truth = labels[draw['query_indices']]
        correct = int(np.count_nonzero(draw['predictions'] == truth))
        rows.append({'draw_index': draw['draw_index'], 'dataset_seed': draw['dataset_seed'],
            'correct': correct, 'total': 6000, 'accuracy': correct / 6000,
            'accuracy_percent': float(Fraction(correct, 60)),
            'diagnostic_member_correct': {seed: int(np.count_nonzero(pred == truth))
                                          for seed, pred in draw['member_predictions'].items()}})
    record = {'schema_version': 1, 'tier': 'medium', 'selected_inference': 'ensemble',
        'member_seeds': MEMBER_SEEDS, 'dataset_seeds': DATASET_SEEDS, 'total_draws': 11,
        **summarize_counts([row['correct'] for row in rows]), 'draws': rows,
        'protocol_sha256': files['protocol.json'], 'draw_manifest_sha256': files['draw_manifest.json'],
        'prediction_manifest_sha256': files['prediction_manifest.json'], 'evaluation_freeze_sha256': sha(freeze_path),
        'evaluator_sha256': evaluator_hash, 'raw_label_source_sha256': RAW_LABEL_SHA256,
        'prediction_frozen_at_utc': frozen['frozen_at_utc'], 'evaluation_frozen_at_utc': evaluation_freeze['frozen_at_utc'],
        'raw_labels_opened_at_utc': labels_opened_at, 'evaluated_at_utc': now(),
        'software': {'python': platform.python_version(), 'numpy': np.__version__},
        'scope': 'Accuracy of the fixed ensemble over eleven dataset draws; no model-cost or hardware certification'}
    write_new(output_path, record)
    print(json.dumps({k: v for k, v in record.items() if k not in ('draws', 'software')}, indent=2))


if __name__ == '__main__':
    main()
