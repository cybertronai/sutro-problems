"""Trusted-controller array archives; never expose archive metadata or truth to a learner."""
import hashlib
import io
import json
import numpy as np

DICT_NAME = 'sutro-d6-accuracy-20261022-inputs-v1'
MAX_BUNDLE_BYTES = 5 * 1024 * 1024
NAMES = ('tx', 'ty', 'qx', 'qy')

def sha(data):
    return hashlib.sha256(data).hexdigest()

def bundle_key(dataset, seed, scorer_sha256):
    return f'v1:{scorer_sha256}:{dataset}:{int(seed)}'

def encode_arrays(dataset, seed, arrays, scorer_sha256, release_version):
    arrays = {name: np.ascontiguousarray(value) for name, value in zip(NAMES, arrays)}
    assert set(arrays) == set(NAMES)
    for name in NAMES:
        arr = arrays[name]
        assert arr.shape == ((10000, 60) if name in ('tx', 'qx') else (10000,))
        assert arr.dtype == (np.dtype('float32') if name in ('tx', 'qx') else np.dtype('int64'))
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    archive = buffer.getvalue()
    assert len(archive) < MAX_BUNDLE_BYTES, 'Use a Volume if a bundle exceeds the small-Dict policy'
    meta = {'format':'npz-arrays-v1','key':bundle_key(dataset, seed, scorer_sha256),
            'dataset':dataset,'seed':int(seed),'scorer_sha256':scorer_sha256,
            'source_release_sha256':scorer_sha256,'release_version':release_version,
            'npz_sha256':sha(archive),'npz_bytes':len(archive),
            'arrays':{name:{'shape':list(arr.shape),'dtype':arr.dtype.str,
                            'sha256':sha(arr.tobytes())} for name,arr in arrays.items()},
            'train_sha256':sha(arrays['tx'].tobytes()+arrays['ty'].tobytes()),
            'query_sha256':sha(arrays['qx'].tobytes()),'numpy':str(np.__version__)}
    return {'metadata':meta,'npz':archive}

def decode_arrays(payload, expected=None):
    meta, archive = payload['metadata'], payload['npz']
    assert len(archive) < MAX_BUNDLE_BYTES
    assert len(archive) == meta['npz_bytes'] and sha(archive) == meta['npz_sha256']
    if expected is not None:
        assert meta == expected, 'Frozen bundle metadata does not match the manifest'
    with np.load(io.BytesIO(archive), allow_pickle=False) as loaded:
        assert set(loaded.files) == set(NAMES)
        arrays = tuple(np.array(loaded[name], copy=True, order='C') for name in NAMES)
    for name,arr in zip(NAMES,arrays):
        entry = meta['arrays'][name]
        assert list(arr.shape) == entry['shape'] and arr.dtype.str == entry['dtype']
        assert sha(arr.tobytes()) == entry['sha256']
    assert sha(arrays[0].tobytes()+arrays[1].tobytes()) == meta['train_sha256']
    assert sha(arrays[2].tobytes()) == meta['query_sha256']
    return arrays

def required_entries(plan):
    return [('fashion',int(plan['warmup_seed']))] + [('mnist',int(seed)) for seed in plan['draw_seeds']]

def validate_manifest(manifest, plan):
    assert manifest['format'] == 'sutro-d6-input-manifest-v1'
    assert manifest['scorer_sha256'] == plan['scorer_sha256']
    for dataset,seed in required_entries(plan):
        entry = manifest['entries'][f'{dataset}:{seed}']
        assert entry['key'] == bundle_key(dataset,seed,plan['scorer_sha256'])
        assert (entry['dataset'],entry['seed'],entry['scorer_sha256']) == (dataset,seed,plan['scorer_sha256'])
    return manifest
