"""Build physically separate search/refit inputs without opening test labels."""
from pathlib import Path
import argparse
from datetime import datetime, timezone
import hashlib
import json
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def digest(a):
    a = np.ascontiguousarray(a.astype(a.dtype.newbyteorder('<'), copy=False))
    return hashlib.sha256(a.tobytes(order='C')).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=REPO / 'mnist/data/medium.npz')
    parser.add_argument('--phase', choices=['search', 'refit'], default='search')
    parser.add_argument('--selection', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--record', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((REPO / 'mnist/doc/dataset_manifest.json').read_text())['tiers']['medium']['arrays']
    keys = ['train_images', 'train_labels', 'train_indices']
    selection_sha = None
    if args.phase == 'refit':
        if args.selection is None:
            parser.error('--selection is required before making refit inputs')
        selection = json.loads(args.selection.read_text())
        if not isinstance(selection, dict) or not selection:
            raise ValueError('Expected a nonempty frozen selection object')
        selection_sha = hashlib.sha256(args.selection.read_bytes()).hexdigest()
        keys = ['train_images', 'train_labels', 'test_images']
    with np.load(args.data, allow_pickle=False) as archive:
        allowed = {key: archive[key] for key in keys}
    evidence = {}
    for key, a in allowed.items():
        entry = manifest[key]
        sha = digest(a)
        if (list(a.shape) != entry['shape'] or str(a.dtype) != entry['dtype']
                or sha != entry['sha256_c_order_little_endian']):
            raise ValueError(f'Noncanonical {key}')
        evidence[key] = {'shape': list(a.shape), 'dtype': str(a.dtype), 'sha256': sha}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **allowed)
    record = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'phase': args.phase,
              'dataset_profile': 'competition-v2', 'tier': 'medium', 'arrays': evidence,
              'test_labels_opened': False, 'frozen_selection_sha256': selection_sha,
              'archive_sha256': hashlib.sha256(args.output.read_bytes()).hexdigest()}
    args.record.parent.mkdir(parents=True, exist_ok=True)
    args.record.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
