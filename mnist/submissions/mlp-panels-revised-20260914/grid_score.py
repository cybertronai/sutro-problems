"""Price the panel program with the validated serialized spatial v4 scorer."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
GRID = HERE.parent / 'grid-mlp-scoring-20260912'


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Explicit paths avoid accidentally importing the historical single-core score.py
# or a third-party package named affine.
affine = load_module('affine', GRID / 'affine.py')
load_module('model_ir', GRID / 'model_ir.py')
spatial = load_module('score', GRID / 'score.py')


def spatial_document(document):
    """Change only the machine header; retain regions, literals and loop order."""
    if document.get('arithmetic') != 'fp32-rne; cmp=lt; select=raw-word':
        raise ValueError('Unsupported panel arithmetic')
    result = copy.deepcopy(document)
    header = affine.make_program([], [])
    for key in ('format', 'model_spec_commit', 'arithmetic', 'placement'):
        result[key] = header[key]
    return result


def source_hashes():
    """Hash loaded project dependencies as well as both owned validation scripts."""
    root = HERE.parents[1]
    paths = {HERE / 'grid_score.py', HERE / 'test_grid.py', HERE / 'reference.py',
             HERE / 'selected.json', GRID / 'test_score.py'}
    for module in tuple(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if filename:
            path = Path(filename).resolve()
            if path.is_relative_to(root) and path.suffix == '.py':
                paths.add(path)
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(paths)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--n-train', type=int, default=1000)
    parser.add_argument('--n-test', type=int, default=1000)
    args = parser.parse_args()
    panels = load_module('panels', HERE / 'panels.py')
    config = json.loads((HERE / 'selected.json').read_text())
    original = panels.build(config, args.epochs, args.n_train, args.n_test)
    document = spatial_document(original)
    result = spatial.score(document)
    result['adapter'] = 'Header-only conversion; panel primitive bodies and region order unchanged'
    result['source_sha256'] = source_hashes()
    result['source_hash_root'] = str(HERE.parents[1])
    result['software'] = {'python': sys.version, 'numpy': np.__version__, 'platform': platform.platform()}
    program_path = HERE / 'program.spatial.json'
    program_path.write_text(json.dumps(document, indent=2) + '\n')
    result['program_file_sha256'] = hashlib.sha256(program_path.read_bytes()).hexdigest()
    (HERE / 'grid-score.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in (
        'energy_fj', 'energy_mj', 'cycles', 'time_ms', 'peak_allocated_scratch_bytes',
        'total_executed_instructions', 'time_to_score_seconds')}, indent=2))


if __name__ == '__main__':
    main()
