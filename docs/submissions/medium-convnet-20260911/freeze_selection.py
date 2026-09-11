"""Freeze validation-selected architecture, epochs, seeds and prediction rule."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, default=HERE)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Refusing to overwrite a frozen selection')
    study = args.study
    protocol = json.loads((study / 'protocol.json').read_text())
    audit = json.loads((study / 'validation_audit.json').read_text())
    assert audit['passed'] and audit['replications_complete']
    assert audit['protocol_sha256'] == sha(study / 'protocol.json')
    for row in audit['runs']:
        assert sha(study / 'results' / (row['id']+'.json')) == row['result_sha256']
    validated = json.loads((study / 'validation_selection.json').read_text())
    assert validated['provenance']['protocol_sha256'] == sha(study / 'protocol.json')
    chosen = validated['preferred_candidate']
    for name, digest in protocol['source_sha256'].items():
        assert sha(HERE / name) == digest
    input_hashes = dict(protocol['input_sha256'])
    canonical = json.loads((HERE.parents[1]/'doc/dataset_manifest.json').read_text())
    input_hashes['test_images'] = canonical['tiers']['medium']['arrays']['test_images']['sha256_c_order_little_endian']
    record = {'schema_version':1, 'frozen_at_utc':datetime.now(timezone.utc).isoformat(),
              'config':chosen['config'], 'epochs':chosen['refit_epochs'], 'schedule_epochs':100,
              'seeds':[101,102,103], 'primary_single_seed':101,
              'selected_inference':chosen['selected_inference'],
              'selected_name':'ensemble' if chosen['selected_inference']=='ensemble' else 'seed101',
              'ensemble_rule':protocol['ensemble_rule'], 'validation_evidence':chosen,
              'source_sha256':{name:sha(HERE/name) for name in ('runner.py','modal_train.py','refit.py')},
              'input_sha256':input_hashes, 'protocol_sha256':sha(study/'protocol.json'),
              'validation_selection_sha256':sha(study/'validation_selection.json'),
              'validation_audit_sha256':sha(study/'validation_audit.json'),
              'test_arrays_accessed':False,
              'phase':'Frozen before refitting and before any ConvNet test-label evaluation'}
    args.output.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))


if __name__ == '__main__':
    main()
