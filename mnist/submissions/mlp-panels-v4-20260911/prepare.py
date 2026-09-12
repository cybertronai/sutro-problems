"""Recreate canonical allowed inputs and CPU references without test labels."""
import hashlib
import json
from pathlib import Path
import numpy as np
import panels as p
import learner

HERE = Path(__file__).resolve().parent


def main():
    directory = HERE/'generated'
    directory.mkdir(exist_ok=True)
    tape_path = p.ROOT/'mnist/submissions/1nn-v4-20260911/input-tape.u32le'
    tape = np.fromfile(tape_path,dtype='<u4')
    assert tape.shape==(11400,)
    data = {'train_images':tape[:5400].view('<f4').reshape(600,1,3,3),
            'train_labels':tape[5400:6000].astype('<i8'),
            'test_images':tape[6000:].view('<f4').reshape(600,1,3,3)}
    manifest_path = p.ROOT/'mnist/doc/dataset_manifest.json'
    manifest = json.loads(manifest_path.read_text())['tiers']['small']['arrays']
    for key,array in data.items():
        assert learner.array_hash(array)==manifest[key]['sha256_c_order_little_endian']
    np.savez(directory/'canonical-inputs.npz',**data)
    arrays = dict(data)
    arrays['mutated_queries'] = np.random.default_rng(20260911).uniform(0,1,(600,9)).astype(np.float32)
    for case in ('canonical','changed_queries','changed_labels'):
        inputs = dict(data)
        if case!='canonical':
            inputs['test_images'] = arrays['mutated_queries'].reshape(600,1,3,3)
        if case=='changed_labels':
            inputs['train_labels'] = (data['train_labels']+1)%10
        reference = learner.learn(inputs,'baseline')
        for key in ('params','scores','predictions'):
            value = reference[key]
            if key=='predictions':
                value = value.astype(np.int32)
            arrays[f'{case}_baseline_{key}'] = value
    np.savez(directory/'gpu_reference.npz',**arrays)
    provenance = {'source':str(tape_path.relative_to(p.ROOT)),
        'source_sha256':hashlib.sha256(tape_path.read_bytes()).hexdigest(),
        'array_hashes':{key:learner.array_hash(value) for key,value in data.items()},
        'reference_array_hashes':{key:learner.array_hash(value) for key,value in arrays.items() if '_baseline_' in key},
        'test_labels_accessed':False}
    (directory/'input_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps(provenance,indent=2))


if __name__=='__main__':
    main()
