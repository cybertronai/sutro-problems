#!/usr/bin/env python3
"""Run the shared exact spatial scorer without retaining operand histograms.

This changes the scorer's memory use, not its address counts or schedule. All
histograms still come from the shared validated affine Program implementation.
The resulting score contains every original field plus reproduction metadata.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time

import numpy as np


def histogram_counts_nocache(program):
    """Accumulate exact address counts, discarding each histogram after use."""
    counts = {name: np.zeros(program.words + 1, dtype=np.int64)
              for name in ('reads', 'writes', 'input_destinations', 'output_sources')}
    computed = 0

    def charge(name, operand, scope):
        nonlocal computed
        first, histogram = program.histogram(operand, scope)
        computed += 1
        counts[name][first:first + len(histogram)] += histogram

    for node, scope, multiplicity in program.leaves:
        if not multiplicity:
            continue
        if node['op'] == 'recv':
            charge('input_destinations', node['dst'], scope)
        elif node['op'] == 'send':
            charge('output_sources', node['src'][0], scope)
        else:
            for source in node.get('src', []):
                charge('reads', source, scope)
            charge('writes', node['dst'], scope)
    return counts, computed


def load_shared(scorer_path):
    """Load score.py and its local dependencies from the declared directory."""
    scorer_path = Path(scorer_path).resolve()
    if scorer_path.is_dir():
        scorer_path /= 'score.py'
    if scorer_path.name != 'score.py' or not scorer_path.is_file():
        raise ValueError('--shared-scorer must name the shared score.py or its directory')
    sys.path.insert(0, str(scorer_path.parent))
    for name in ('score', 'affine', 'model_ir'):
        loaded = sys.modules.get(name)
        if loaded is not None and Path(loaded.__file__).resolve().parent != scorer_path.parent:
            raise ValueError(f'{name} was already imported from a different directory')
    module = importlib.import_module('score')
    if Path(module.__file__).resolve() != scorer_path:
        raise ValueError('Loaded shared scorer path differs from requested path')
    return module


def score_uncached(shared, document):
    """Call the unchanged shared scorer with only its histogram cache disabled."""
    original = shared.histogram_counts
    shared.histogram_counts = histogram_counts_nocache
    try:
        return shared.score(document)
    finally:
        shared.histogram_counts = original


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def compare_scores(cached, uncached):
    """Only cache-computation counts and measured host durations may differ."""
    ignored = {'histograms_computed', 'time_to_score_seconds'}
    before = {key: value for key, value in cached.items() if key not in ignored}
    after = {key: value for key, value in uncached.items() if key not in ignored}
    if before != after:
        changed = [key for key in before.keys() | after.keys()
                   if before.get(key) != after.get(key)]
        raise AssertionError(f'Cached and uncached score differ: {sorted(changed)}')
    return {'all_invariant_score_fields_equal': True,
            'invariant_score_field_count': len(before),
            'excluded_fields': sorted(ignored),
            'cached_histograms_computed': cached['histograms_computed'],
            'uncached_histograms_computed': uncached['histograms_computed']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('program', type=Path, help='Generated affine program JSON')
    parser.add_argument('--shared-scorer', type=Path, required=True,
                        help='Shared score.py path or directory')
    parser.add_argument('--output', type=Path, help='Full JSON output (stdout if omitted)')
    parser.add_argument('--compare-cache', action='store_true',
                        help='Also check the cached scorer; use only on reduced programs')
    args = parser.parse_args()
    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    program_path = args.program.resolve()
    document = json.loads(program_path.read_text())
    shared = load_shared(args.shared_scorer)
    source_paths = [Path(__file__).resolve(), Path(shared.__file__).resolve(),
                    Path(sys.modules['affine'].__file__).resolve(),
                    Path(sys.modules['model_ir'].__file__).resolve()]
    source_hashes = {path.name: sha256(path) for path in source_paths}
    print('Validating and scoring with uncached exact histograms...', file=sys.stderr, flush=True)
    result = score_uncached(shared, document)
    comparison = None
    if args.compare_cache:
        print('Checking the shared cached scorer...', file=sys.stderr, flush=True)
        comparison = compare_scores(shared.score(document), result)
    max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform != 'darwin':
        max_rss *= 1024
    result['reproduction'] = {
        'wrapper': 'Unchanged shared scorer with histogram_counts cache disabled',
        'histograms_computed_semantics': 'Operand histogram calls, including repeated identical operands',
        'program_file_sha256': sha256(program_path),
        'source_sha256': source_hashes,
        'started_at_utc': started_utc,
        'completed_at_utc': datetime.now(timezone.utc).isoformat(),
        'host': {'hostname': platform.node(), 'platform': platform.platform(),
                 'machine': platform.machine(), 'logical_cpu_count': os.cpu_count(),
                 'python': platform.python_version(), 'numpy': np.__version__},
        'peak_process_rss_bytes': max_rss,
        'peak_process_rss_scope': 'Whole wrapper process including optional cached comparison',
        'wrapper_elapsed_seconds': time.perf_counter() - started,
        'wrapper_elapsed_scope': 'JSON input, imports, exact scorer and optional cached comparison; excludes output serialization and file write',
        'cached_comparison': comparison,
    }
    serialized = json.dumps(result, indent=2, sort_keys=True) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized)
        print(f'Wrote complete score to {args.output}', file=sys.stderr)
    else:
        sys.stdout.write(serialized)


if __name__ == '__main__':
    main()
