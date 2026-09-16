"""Validate spatial costs and reduced Python execution; optionally execute all full draws in C.

python validate_reduced.py 300 120 100 --compiled-executable generated/compiled/reduced \
    --full-executable generated/compiled/full --output results/grid_verification.json
"""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import spatial_program as sp
import reference as ref
import run
from run import ds, require, read_json
from score import score, placement
from affine import Program

PARAMETERS = {'m': 'm', 'Wn': 'W', 'mu': 'mu', 'pk': 'packed', 'kap': 'kappa'}


def program_hash(document):
    return hashlib.sha256(json.dumps(document, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def sources():
    paths = {name: HERE / name for name in ('validate_reduced.py', 'spatial_program.py', 'reference.py', 'run.py')}
    shared = HERE.parent / 'grid-mlp-scoring-20260912'
    paths.update({f'../grid-mlp-scoring-20260912/{name}': shared / name for name in ('affine.py', 'score.py')})
    paths['mnist/code/data.py'] = Path(ds.__file__)
    return {name: ds.file_hash(path) for name, path in paths.items()}


def compiled_provenance(document, executable):
    generated = sp.compiled_source(document).encode()
    source = executable.with_suffix('.c')
    require(source.read_bytes() == generated, f'C source differs: {source}')
    result = {'executable_sha256': ds.file_hash(executable),
              'c_source_sha256': hashlib.sha256(generated).hexdigest(),
              'program_sha256': program_hash(document)}
    info = executable.parent / 'compiler.json'
    if info.exists():
        result['compiler'] = read_json(info)
    return result


def verify_grid():
    document = sp.build_submitted()
    saved = read_json(HERE / 'grid/grid-score.json')
    fresh = score(document)
    for key, value in fresh.items():
        if key != 'time_to_score_seconds':
            require(saved[key] == value, f'Spatial scorer field differs: {key}')
    blob = (json.dumps(document, separators=(',', ':')) + '\n').encode()
    require(hashlib.sha256(blob).hexdigest() == saved['program_file_sha256'], 'Spatial JSON hash differs')
    compressed = HERE / 'grid/program.spatial.json.gz'
    if compressed.exists():
        require(gzip.decompress(compressed.read_bytes()) == blob, 'Compressed program differs')
    shared = HERE.parent / 'grid-mlp-scoring-20260912'
    for name, expected in saved['source_sha256'].items():
        path = shared / name if name in ('affine.py', 'score.py') else HERE / name
        require(ds.file_hash(path) == expected, f'Scored source hash differs: {name}')
    program = Program(document)
    owners, _, _, links, _ = placement(program.words)
    n_input, n_output = fresh['input_tape_words'], fresh['output_tape_words']
    components = fresh['components']
    reads = components['reads']['scratch_accesses'] + n_input + 2 * n_output
    writes = components['writes']['scratch_accesses'] + 2 * n_input + n_output
    regions = [{'name': name, 'words': words, 'first_owner': owners[base].tolist(),
                'last_owner': owners[base + words - 1].tolist(),
                'words_in_compute_tile': int((links[base:base + words] == 0).sum())}
               for name, (base, words) in program.regions.items()]
    return document, {
        'all_deterministic_score_fields_match': True,
        'grid_score_sha256': ds.file_hash(HERE / 'grid/grid-score.json'),
        'program_file_sha256': saved['program_file_sha256'],
        **{key: fresh[key] for key in ('program_sha256', 'energy_fj', 'word_node_hops', 'energy_mj',
            'cycles', 'time_ms', 'total_executed_instructions', 'peak_allocated_scratch_bytes',
            'program_scratch_words', 'stage_scratch_words', 'memory_tiles', 'max_tile_scratch_words',
            'instruction_issuing_processors', 'max_simultaneous_instructions', 'placement_sha256')},
        'scratch_reads_including_staging_and_tape': reads,
        'scratch_writes_including_staging_and_tape': writes,
        'scratch_accesses_including_staging_and_tape': reads + writes,
        'input_tape_words': n_input, 'output_tape_words': n_output,
        'regions': regions, 'components': components, 'fresh_scoring_seconds': fresh['time_to_score_seconds'],
    }


def verify_reduced(n_train, n_test, n_basis, executable):
    start = time.perf_counter()
    require(0 < n_basis <= n_train <= 10000 and 0 < n_test <= 50000, 'Invalid reduced dimensions')
    pixels, labels = run._arrays()
    seed = 20261121
    order = np.random.Generator(np.random.PCG64(seed)).permutation(60000)
    train, test = order[:n_train], order[10000:10000 + n_test]
    x = run.resize_recorded(pixels[train].astype(np.float32) / np.float32(255)).reshape(n_train, 81)
    q = run.resize_recorded(pixels[test].astype(np.float32) / np.float32(255)).reshape(n_test, 81)
    old_basis_count = ref.NP
    try:
        ref.NP = n_basis
        params, scores, pred = ref.train_predict(x, labels[train], q)
    finally:
        ref.NP = old_basis_count
    document = sp.build_program(n_train, n_test, n_basis=n_basis, **sp.SUBMITTED)
    tape = sp.tape_words(x, labels[train], q)
    begin = time.perf_counter()
    out, memory = sp.execute(document, tape, return_memory=True)
    python_seconds = time.perf_counter() - begin
    out = np.asarray(out, dtype=np.int64)
    require(out.shape == pred.shape and np.array_equal(out, pred), 'Reduced spatial/reference labels differ')
    checks = {}
    for region, name in PARAMETERS.items():
        expected = np.asarray(params[name], dtype=np.float32).reshape(-1)
        require(np.isfinite(expected).all(), f'Reduced reference has nonfinite {name}')
        require(memory[region].shape == expected.shape and
                np.array_equal(memory[region].view(np.uint32), expected.view(np.uint32)),
                f'Reduced spatial/reference parameter bits differ: {name}')
        checks[name] = {'bitwise_equal': True, 'sha256': ds.array_hash(expected), 'words': expected.size}
    result = {'passed': True, 'dataset_seed': seed, 'n_train': n_train, 'n_test': n_test, 'n_basis': n_basis,
              'program_sha256': program_hash(document), 'parameter_checks': checks,
              'labels_agree': int((out == pred).sum()), 'total': n_test,
              'correct': int((out == labels[test]).sum()), 'prediction_sha256': ds.array_hash(out),
              'reference_scores_sha256': ds.array_hash(scores),
              'input_tape_sha256': ds.array_hash(np.asarray(tape, dtype=np.uint32)),
              'python_execution_seconds': python_seconds}
    if executable is not None:
        provenance = compiled_provenance(document, executable)
        begin = time.perf_counter()
        compiled_out, compiled_memory = sp.execute_compiled(document, tape, executable)
        require(np.array_equal(compiled_out, out), 'Reduced C/Python labels differ')
        for name in memory:
            require(np.array_equal(memory[name].view(np.uint32), compiled_memory[name].view(np.uint32)),
                    f'Reduced C/Python memory bits differ: {name}')
        result['compiled'] = {**provenance, 'labels_equal_python': True,
                              'all_memory_regions_bitwise_equal_python': True,
                              'execution_seconds': time.perf_counter() - begin}
    result['verification_seconds'] = time.perf_counter() - start
    print(f'Reduced {n_train}/{n_test}/{n_basis}: parameters and {n_test} labels match; '
          f'Python execution {python_seconds:.2f}s', flush=True)
    return result


def verify_full(document, executable, payload_dir):
    provenance = compiled_provenance(document, executable)
    manifest = read_json(payload_dir / 'manifest.json')
    run.validate_records(manifest, 'payload manifest')
    frozen, predictions = run.frozen_predictions(run.EVIDENCE)
    require(manifest['prediction_manifest_sha256'] == ds.file_hash(run.EVIDENCE / 'prediction_manifest.json'),
            'Payloads reference a different frozen manifest')
    draws = []
    for record, frozen_record, pred in zip(manifest['draws'], frozen['draws'], predictions, strict=True):
        begin = time.perf_counter()
        path = payload_dir / record['path']
        require(path.resolve().is_relative_to(payload_dir.resolve()), 'Payload escapes directory')
        require(ds.file_hash(path) == record['sha256'], 'Payload file hash differs')
        with np.load(path, allow_pickle=False) as archive:
            values = {name: archive[name] for name in archive.files}
        for name, expected in record['arrays'].items():
            value = values[name]
            require(list(value.shape) == expected['shape'] and str(value.dtype) == expected['dtype'] and
                    ds.array_hash(value) == expected['sha256'], f'Payload array differs: {name}')
        train, test, x, q, train_labels, _, labels = run._load(record['dataset_seed'])
        for name, expected in (('x', x), ('q', q), ('labels', train_labels), ('test_labels', labels[test]),
                               ('frozen_predictions', pred)):
            require(np.array_equal(values[name], expected), f'Payload differs from source/freeze: {name}')
        tape = sp.tape_words(values['x'], values['labels'], values['q'])
        out, memory = sp.execute_compiled(document, tape, executable)
        out = np.asarray(out, dtype=np.int64)
        require(np.array_equal(out, pred), f'Draw {record["draw"]}: full spatial labels differ')
        hashes = {}
        for region, name in PARAMETERS.items():
            digest = ds.array_hash(memory[region])
            require(digest == frozen_record[name + '_sha256'], f'Draw {record["draw"]}: spatial {name} differs')
            hashes[name] = digest
        require(ds.array_hash(out) == frozen_record['array_sha256'], 'Full prediction hash differs')
        draw = {'draw': record['draw'], 'dataset_seed': record['dataset_seed'],
                'correct': int((out == labels[test]).sum()), 'total': len(pred),
                'all_parameter_hashes_match': True, 'parameter_sha256': hashes,
                'labels_agree_with_frozen': len(pred), 'prediction_sha256': ds.array_hash(out),
                'payload_sha256': record['sha256'],
                'input_tape_sha256': ds.array_hash(np.asarray(tape, dtype=np.uint32)),
                'verification_seconds': time.perf_counter() - begin}
        draws.append(draw)
        print(f'Full spatial draw {draw["draw"]:02d}: parameters and {len(pred)} labels match '
              f'({draw["verification_seconds"]:.2f}s)', flush=True)
    return {**provenance, 'passed': True, 'draws': draws,
            'payload_manifest_sha256': ds.file_hash(payload_dir / 'manifest.json'),
            'all_eleven_frozen_draws_checked': True, 'all_parameter_hashes_match': True,
            'labels_agree_with_frozen': sum(draw['labels_agree_with_frozen'] for draw in draws),
            'correct': sum(draw['correct'] for draw in draws), 'total': sum(draw['total'] for draw in draws)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('n_train', type=int)
    parser.add_argument('n_test', type=int)
    parser.add_argument('n_basis', type=int)
    parser.add_argument('--compiled-executable', type=Path)
    parser.add_argument('--full-executable', type=Path)
    parser.add_argument('--payload-dir', type=Path, default=HERE / 'generated/payloads')
    parser.add_argument('--output', type=Path, default=HERE / 'results/grid_verification.json')
    args = parser.parse_args()
    require(args.full_executable is None or args.compiled_executable is not None,
            'Full C execution requires reduced C/Python cross-check')
    require(not args.output.resolve().is_relative_to((HERE / 'evidence').resolve()), 'Do not overwrite imported evidence')
    begin = time.perf_counter(); initial_sources = sources(); raw = run.verify_raw()
    document, grid = verify_grid()
    reduced = verify_reduced(args.n_train, args.n_test, args.n_basis, args.compiled_executable)
    full = verify_full(document, args.full_executable, args.payload_dir) if args.full_executable else None
    require(sources() == initial_sources, 'Verification sources changed during execution')
    result = {'passed': True, 'verified_at_utc': datetime.now(timezone.utc).isoformat(),
              'source_sha256': initial_sources, 'canonical_raw_mnist': raw,
              'software': {'python': sys.version, 'numpy': np.__version__, 'platform': platform.platform()},
              'full_grid': grid, 'reduced': reduced, 'full_execution': full,
              'scope': 'Static costs reproduce the submitted grid score. Reduced Python execution compares parameters '
                       'and labels with the reference; optional C execution cross-checks every reduced memory word '
                       'before comparing all five parameter arrays and labels on eleven full draws. Numeric C '
                       'execution does not simulate network contention or independently measure grid costs.',
              'verification_seconds': time.perf_counter() - begin}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(f'Wrote {args.output}', flush=True)


if __name__ == '__main__':
    main()
