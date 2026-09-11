"""Reclassify retained results under revised rules without changing frozen evidence."""
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
def read(name): return json.loads((HERE/name).read_text())
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    rules = read('current-targets.json')
    accuracy = read('accuracy.json')
    protocol = read('protocol.json')
    counts = [row['correct'] for row in accuracy['draws']]
    total = sum(row['total'] for row in accuracy['draws'])
    correct = sum(counts)
    assert len(counts) == rules['draws'] == 11
    assert all(row['total'] == rules['test_examples_per_draw'] for row in accuracy['draws'])
    assert protocol['n_train'] == rules['train_examples_per_draw']
    assert total == accuracy['total_predictions'] and correct == accuracy['total_correct']
    rows = []
    for error in rules['error_targets_percent']:
        percent = 100-Fraction(error)
        required = (total*percent.numerator + 100*percent.denominator-1)//(100*percent.denominator)
        rows.append(dict(error_target_percent=error, accuracy_target_percent=str(percent),
            required_correct=required, meets_target=correct>=required, margin_correct=correct-required))
    qualifying = [Fraction(row['error_target_percent']) for row in rows if row['meets_target']]
    result = dict(schema='sutro-mnist-revised-target-assessment/1',
        assessed_at_utc=datetime.now(timezone.utc).isoformat(),
        scope='Eligibility reassessed after evaluation following a correction to the target levels; not a new predeclared experiment. Frozen algorithm, predictions, model scores and GPU measurements are unchanged.',
        original_predeclared_error_percent=protocol['target_error_percent'],
        current_levels=rows, strictest_qualifying_error_percent=str(min(qualifying)) if qualifying else None,
        total_correct=correct,total_predictions=total,
        retained_evidence_sha256={name:sha(HERE/name) for name in
            ('accuracy.json','protocol.json','config.json','prediction_manifest.json',
             'evaluation.py','audit.json','model-score.json','benchmark/results.json')},
        current_rules_sha256=sha(HERE/'current-targets.json'),assessor_sha256=sha(Path(__file__)))
    (HERE/'target-assessment.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'strictest_qualifying_error_percent':result['strictest_qualifying_error_percent'],
        'correct':correct,'total':total,'new_training_performed':False}))

if __name__ == '__main__': main()
