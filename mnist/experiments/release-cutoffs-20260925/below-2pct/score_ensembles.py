"""Score every pre-registered candidate of protocol.draft.json on the eleven final seeds.

    /tmp/penv/bin/python score_ensembles.py      # after score.py --freeze-final and --stage final

Members: ladder-is06-long learner seed 11 (the frozen final predictions of
../../release-ladder-20260924, 12,000 steps), and this study's ladder-long-s12,
ladder-long-s13 (12,000 steps) and ladder-xlong-s11 (24,000 steps). Every prediction
file's SHA-256 is checked against its own study's final_freeze.json before any label is
read. Ensembles: equal-weight mean of softmax(logits), argmax with ties to the lowest
class (release-ladder-20260924/score.py build_ensemble's rule). Writes results/below2.json.
"""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

import study  # this sub-study's copy: same draw protocol and query rows

HERE = Path(__file__).resolve().parent
EARLIER = HERE.parents[1] / 'release-ladder-20260924'
T10 = 2.2281388519649385
SOURCES = {
    's11': (EARLIER, 'final-ladder-is06-long-s{seed}-n10000'),
    's12': (HERE, 'final-ladder-long-s12-s{seed}-n10000'),
    's13': (HERE, 'final-ladder-long-s13-s{seed}-n10000'),
    'xlong': (HERE, 'final-ladder-xlong-s11-s{seed}-n10000'),
}
CANDIDATES = {
    'single s11 (the recipe, 12,000 steps)': ['s11'],
    'single s12': ['s12'],
    'single s13': ['s13'],
    'xlong: 24,000 steps, seed 11': ['xlong'],
    'ens2: seeds 11+12': ['s11', 's12'],
    'ens3: seeds 11+12+13': ['s11', 's12', 's13'],
    'ens4: ens3 + xlong': ['s11', 's12', 's13', 'xlong'],
}
PRIMARY = ['ens3: seeds 11+12+13', 'xlong: 24,000 steps, seed 11']


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def softmax(logits):
    z = logits.astype(np.float64)
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def main():
    freezes = {}
    for root in (EARLIER, HERE):
        freeze = json.loads((root / 'predictions' / 'final_freeze.json').read_text())
        freezes[root] = freeze['jobs']
    probs = {}
    for member, (root, pattern) in SOURCES.items():
        for seed in study.FINAL_SEEDS:
            job_id = pattern.format(seed=seed)
            entry = freezes[root].get(job_id)
            if entry is None:
                raise SystemExit(f'{job_id} is not in {root.name} final_freeze.json')
            path = root / entry['predictions_path']
            if sha256(path) != entry['predictions_sha256']:
                raise SystemExit(f'{path}: prediction file differs from its freeze')
            probs[(member, seed)] = softmax(np.load(path)['logits'])
    labels = {seed: study.query_labels(seed) for seed in study.FINAL_SEEDS}  # read only after every check
    out = {}
    for name, members in CANDIDATES.items():
        errors, wrong, total = [], 0, 0
        for seed in study.FINAL_SEEDS:
            mixture = sum(probs[(m, seed)] for m in members) / len(members)
            predicted = mixture.argmax(axis=1)  # numpy argmax returns the lowest index on ties
            w = int((predicted != labels[seed]).sum())
            wrong, total = wrong + w, total + len(predicted)
            errors.append(100.0 * w / len(predicted))
        mean = sum(errors) / len(errors)
        sd = math.sqrt(sum((e - mean) ** 2 for e in errors) / (len(errors) - 1))
        half = T10 * sd / math.sqrt(len(errors))
        out[name] = {'members': members, 'pooled_error_pct': 100.0 * wrong / total, 'per_seed_pct': errors,
                     'ci95_pct': [mean - half, mean + half], 'sd_pp': sd, 'wrong': wrong, 'total': total,
                     'at_or_below_2pct': 100.0 * wrong / total <= 2.0, 'primary': name in PRIMARY}
    (HERE / 'results' / 'below2.json').write_text(json.dumps(out, indent=2) + '\n')
    print('| Candidate | Pooled error | 95% CI | SD (pp) | At or below 2.00% |')
    print('| --- | ---: | --- | ---: | --- |')
    for name, r in out.items():
        print(f"| {name}{' (primary)' if r['primary'] else ''} | {r['pooled_error_pct']:.3f}% | "
              f"{r['ci95_pct'][0]:.3f}-{r['ci95_pct'][1]:.3f}% | {r['sd_pp']:.3f} | "
              f"{'yes' if r['at_or_below_2pct'] else 'no'} |")


if __name__ == '__main__':
    main()
