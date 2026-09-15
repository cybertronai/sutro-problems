"""Build label-isolated CPU bit oracles for the revised first draw and mutations."""
import json
from pathlib import Path
import numpy as np
import reference
import run

HERE = Path(__file__).resolve().parent


def main():
    run.check_protocol()
    draw = json.loads((HERE / 'draw_manifest.json').read_text())['draws'][0]
    path = HERE / draw['archive']
    assert run.sha(path) == draw['archive_sha256']
    with np.load(path, allow_pickle=False) as z:
        assert set(z.files) == {'train_images', 'train_labels', 'test_images'}
        data = {name:z[name] for name in z.files}
    config = json.loads((HERE / 'selected.json').read_text())
    mutated_queries = np.random.default_rng(7845).uniform(0, 1, (1000, 9)).astype(np.float32)
    packed = {**data, 'mutated_queries':mutated_queries}
    for case in ('canonical', 'changed_queries', 'changed_labels'):
        inputs = {name:array.copy() for name,array in data.items()}
        if case != 'canonical':
            inputs['test_images'] = mutated_queries
        if case == 'changed_labels':
            inputs['train_labels'] = (inputs['train_labels'].astype(np.int32)+1)%10
        result = reference.cpu(inputs, config, 300, check_baseline=True)
        for name in ('params', 'scores', 'predictions'):
            packed[f'{case}_baseline_{name}'] = result[name].astype(np.int32) if name == 'predictions' else result[name]
        print(f'CPU bit oracle verified: {case}', flush=True)
    destination = HERE / 'generated'
    destination.mkdir(exist_ok=True)
    assert not (destination / 'gpu_reference.npz').exists()
    np.savez_compressed(destination / 'gpu_reference.npz', **packed)


if __name__ == '__main__':
    main()
