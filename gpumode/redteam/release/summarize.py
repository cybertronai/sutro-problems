"""Merge the five attack result files into results.json.

Run after blind_attack.py, informed_attack.py, artifact_size.py,
holdout_check.py and exact_source.py:

  /tmp/pmnist-env/bin/python .../redteam/release/summarize.py
"""
from __future__ import annotations

import json

import harness

FILES = {
    'blind': 'results-blind.json',
    'informed': 'results-informed.json',
    'artifact_size': 'results-artifact-size.json',
    'holdout': 'results-holdout.json',
    'exact_source': 'results-exact-source.json',
}


def main():
    parts, missing = {}, []
    for name, fn in FILES.items():
        path = harness.HERE / fn
        if path.exists():
            parts[name] = json.loads(path.read_text())
        else:
            missing.append(fn)

    out = {
        'harness': {
            'eval_version': getattr(harness.harness_eval, 'HARNESS_VERSION', 'unknown'),
            'release_dims': harness.RELEASE_DIMS,
            'case_seed': harness.CASE_SEED,
            'max_source_bytes': 20480, 'max_literal_bytes': 4096,
            'holdout_min_bp': 3000,
        },
        'missing_result_files': missing,
        'attacks': parts,
    }

    s = {}
    if 'blind' in parts:
        b = parts['blind']
        s['blind'] = {
            'best_edge_precision_at_1.5': max(
                m['lattice']['edge_precision_at_1.5'] for m in b['methods'].values()),
            'chance_edge_precision_at_1.5': max(
                m['lattice']['chance_precision_at_1.5'] for m in b['methods'].values()),
            'best_cnn_on_recovered_lattice': b['best_cnn_on_recovered_lattice'],
            'cnn_on_random_layout': b['learners']['cnn_on_random_layout']['test_acc'],
            'honest_mlp': b['learners']['mlp_on_released_features']['test_acc'],
            'gain_points': b['cnn_gain_over_mlp_points'],
            'verdict': 'no gain over the honest MLP: the lattice is not recovered well enough '
                       'to be worth convolving over',
        }
    if 'informed' in parts:
        i = parts['informed']
        s['informed'] = {
            'label_permutation_classes_correct': i['label_permutation']['classes_correct'],
            'pixel_rel_rms': i['attacker_pick_rel_rms'],
            'offline_cnn_on_recovered': i['attack_accuracy'],
            'offline_cnn_on_true_pixels': i['offline_cnn_on_true_test_pixels'],
            'honest_mlp': i['honest_mlp_on_released_features'],
            'gain_points': i['gain_over_honest_mlp_points'],
            'verdict': 'works; stopped only by the submission size cap',
        }
    if 'artifact_size' in parts:
        a = parts['artifact_size']
        s['artifact_size'] = {
            'class_statistics_bytes_k60': a['class_statistics']['int8']['bytes_in_submission'],
            'study_cnn_int8_bytes': a['offline_models']['study_CNN_32_64_64']['bytes_in_submission'],
            'payload_over_cap_factor': a['verdict']['over_cap_factor'],
            'fits': a['verdict']['fits'],
        }
        if 'cap_compliant_route' in a:
            c = a['cap_compliant_route']
            s['artifact_size'].update({
                'cap_compliant_artifact_bytes': c['artifact_bytes_in_submission'],
                'cap_compliant_adapter_acc': c['adapter_random_init_test_acc'],
                'cap_compliant_gain_points': c['gain_over_honest_mlp_points'],
            })
        if 'minimal_payload_attack' in a:
            m = a['minimal_payload_attack']
            s['artifact_size'].update({
                'minimal_payload_bytes': m['payload_bytes'],
                'minimal_payload_bytes_left_for_code': m['bytes_left_for_code'],
                'minimal_payload_verbatim_code_bytes': m['verbatim_attack_code_bytes'],
                'minimal_payload_fits_with_verbatim_code': m['fits_with_verbatim_code'],
                'minimal_payload_test_acc': m['tiny_cnn_on_recovered_test_acc'],
            })
    if 'holdout' in parts:
        h = parts['holdout']
        s['holdout'] = {
            'floor_percent': h['holdout_floor_percent'],
            'smuggler_accuracy': h['smuggler_accuracy'],
            'smuggler_passes': h['smuggler_passes_holdout'],
            'honest_mlp_accuracy': h['honest_mlp_on_holdout_release'],
            'honest_mlp_passes': h['honest_mlp_passes_holdout'],
            'label_permutation_classes_correct': h['label_permutation']['classes_correct'],
        }
    if 'exact_source' in parts:
        e = parts['exact_source']
        s['exact_source'] = {
            'label_accuracy_public_fit': e['fits']['public_40k']['label_accuracy_percent'],
            'row_hits_public_fit': e['fits']['public_40k']['exact_row_hit_percent'],
            'label_accuracy_oracle_fit': e['fits']['secret_train_10k_ORACLE']['label_accuracy_percent'],
            'median_rank_of_true_row_public_fit': e['fits']['public_40k']['median_rank_of_true_row'],
            'cheapest_table_bytes': e['lookup_table']['cheapest_working_table_bytes'],
            'entropy_floor_base64_bytes': e['lookup_table']['entropy_floor_base64_bytes'],
        }

    # the one number that crosses two files: can the smuggler tell a hold-out draw
    # from a ranked one by the moment-matching residual alone?
    if 'informed' in parts and 'holdout' in parts:
        mi = min(v['final_loss'] for k, v in parts['informed']['inits'].items()
                 if not k.startswith('__'))
        mh = min(v['final_loss'] for v in parts['holdout']['inits'].values())
        s['holdout_detectability'] = {
            'best_moment_loss_on_mnist_draw': mi,
            'best_moment_loss_on_fashion_holdout': mh,
            'ratio': mh / mi,
            'note': 'an adaptive smuggler can read this residual and fall back to an honest '
                    'learner on hold-out calls, so the hold-out alone does not stop the '
                    'informed attack; the size cap is what does',
        }

    out['summary'] = s
    harness.jdump(harness.HERE / 'results.json', out)
    print(json.dumps(s, indent=1))
    if missing:
        print('MISSING:', missing)


if __name__ == '__main__':
    main()
