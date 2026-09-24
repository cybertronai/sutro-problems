"""Collect every measured number into research/whitening-attack-results.{json,md}.

Each block records the script that produced it.  Values are pasted from those
scripts' stdout (archived in this directory) except where a results file exists.
"""
import json, sys, time
from pathlib import Path
import numpy as np

RESEARCH = Path(__file__).resolve().parent.parent
SCRATCH = Path('/tmp/whiten-attack')


def load(name, default=None):
    path = SCRATCH / name
    if path.exists():
        return json.loads(path.read_text())
    return default


stage1_n1000 = load('stage1_N1000.json', [])
stage1_n10000 = load('stage1_N10000.json', [])
second = load('secondorder.json', {})
dense = load('dense_N1000.json', {})
seedvar = load('seedvar.json', {})
batches = {'n1000_60epochs_1member': {}, 'n10000_15epochs_1member': {}}
for name in ('batchA1', 'batchA2', 'batchB1'):
    batches['n1000_60epochs_1member'].update(load(name + '.json', {}) or {})
for name in ('batchB_10k_a', 'batchB_10k_b'):
    batches['n10000_15epochs_1member'].update(load(name + '.json', {}) or {})
batches['facet_attack_end_to_end'] = {
    'n1000_all_63_channels': load('facet2_N1000.json', {}),
    'n10000_all_55_channels': load('facet2_N10000.json', {}),
    'n1000_topk_by_atom_score': load('facet3_N1000.json', {})}
batches['note'] = ('single draw, single cnn-09 member, reduced epochs; member-seed '
                   'spread measured separately in cnn_member_seed_spread')


def attack_rows(reports):
    rows = []
    for r in reports:
        for mode, entry in r['layouts'].items():
            q = entry['quality']
            rows.append({
                'n': r['n'], 'unlabeled_rows': r['unlabeled_rows'], 'fun': r['fun'],
                'whiten_mode': r['whiten_mode'], 'affinity': mode,
                'ica_converged': r['ica']['converged'], 'ica_iterations': r['ica']['n_iter'],
                'max_abs_pixel_correlation_mean':
                    r['localisation']['max_abs_pixel_correlation_mean'],
                'max_abs_pixel_correlation_median':
                    r['localisation']['max_abs_pixel_correlation_median'],
                'synthesis_top1_energy': r['localisation']['synthesis']['mean_top1_energy'],
                'synthesis_3x3_energy': r['localisation']['synthesis']['mean_window3x3_energy'],
                'distinct_peak_pixels': r['localisation']['distinct_peak_pixels'],
                'affinity_neighbour_mean': entry['neighbour_affinity_ratio']['mean_neighbour'],
                'affinity_far_mean': entry['neighbour_affinity_ratio']['mean_far'],
                'affinity_top4_true_neighbour_fraction':
                    entry['neighbour_affinity_ratio']['top4_true_neighbour_fraction'],
                'adjacency_agreement': q['adjacency_agreement'],
                'mean_manhattan_after_dihedral': q['mean_manhattan_after_dihedral'],
                'exact_cell_fraction': q['exact_cell_fraction'],
                'chance_adjacency_agreement': r['chance']['adjacency_agreement_mean'],
                'chance_adjacency_agreement_p95': r['chance']['adjacency_agreement_p95'],
                'chance_mean_manhattan': r['chance']['mean_manhattan_mean'],
                'oracle_adjacency_agreement':
                    r['oracle_layout_quality']['adjacency_agreement'],
            })
    return rows


results = {
    'title': 'Does whitening + a random rotation remove the ability to recover the 9x9 topology?',
    'generated_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    'author_note': ('Independent second opinion produced alongside a concurrent agent that '
                    'wrote whiten.py and its own attack_ica.py; this file reports only '
                    'numbers measured by research/attack_ica_topographic.py and the scripts '
                    'in research/whitening-attack-scripts/.'),
    'protocol': {
        'dataset_seed': 2026092301,
        'seed_kind': 'dev seed; no FINAL seed (2026092001..11) was touched',
        'query_labels': ('read only through raw/pool_labels.npy indexed by '
                         'study.query_indices(2026092301), for scoring CPU pilots'),
        'compute': 'CPU only (Intel Mac, /tmp/pmnist-env python 3.11). No Modal, no GPU.',
        'variant': ('whiten.default_transform(): z = A (x - mean) with A = P Q W, '
                    'W = ZCA whitening with variance floor epsilon=1e-3, Q a Haar rotation '
                    '(qr_pcg64, seed 20260923), P the study 81-feature permutation'),
        'transform_condition_number': 20.64,
        'transform_is_invertible': True,
        'pilot_status': ('every downstream error below is a SINGLE draw and (except where a '
                         'member-seed spread is given) a SINGLE cnn-09 member with a reduced '
                         'epoch budget; they rank methods, they are not protocol numbers'),
    },
    'second_order_attack_on_the_whitened_variant': second,
    'ica_attack': {
        'n1000': attack_rows(stage1_n1000),
        'n10000': attack_rows(stage1_n10000),
    },
    'ceiling_of_any_white_basis': {
        'source': 'ceiling.py, ceiling2.py',
        'note': ('every ICA output is white, and raw MNIST pixels are strongly correlated, '
                 'so no ICA solution can BE the pixels.  The closest white basis to the '
                 'pixel basis is ZCA itself; these are its numbers.'),
        'zca_eps1e-3_corr_with_own_pixel_mean': 0.896,
        'zca_eps1e-3_corr_with_own_pixel_median': 0.892,
        'zca_eps1e-3_argmax_is_own_pixel': '81/81',
        'zca_eps0_rank75_corr_mean': 0.856,
        'zca_row_top1_energy': 0.899,
        'zca_row_3x3_energy': 0.960,
        'fastica_source_vs_raw_pixel_matched_abs_r_N10000': 0.683,
        'fastica_source_vs_zca_pixel_matched_abs_r_N10000': 0.722,
        'fastica_source_vs_raw_pixel_matched_abs_r_N1000': 0.680,
    },
    'why_it_fails': {
        'source': 'objective.py',
        'note': ('FastICA is not losing to the optimiser or to sample size: at N=10000 its '
                 'solution scores HIGHER on the ICA contrast than the pixel-aligned basis '
                 'does, so the pixel basis is not the optimum being searched for.  The '
                 'pixel-aligned basis was mapped into the sample-white coordinates and '
                 'Lowdin-orthogonalised to make it feasible (|B-Q|_max = 0.364).'),
        'logcosh_contrast_fastica': 3.33,
        'logcosh_contrast_pixel_aligned_zca': 2.733,
        'logcosh_contrast_untouched_released': 0.2756,
        'cube_contrast_fastica': 4.454e7,
        'cube_contrast_pixel_aligned_zca': 3.589e7,
        'cube_contrast_untouched_released': 1.129e4,
        'mean_excess_kurtosis_fastica_N10000': 918.4,
        'mean_excess_kurtosis_zca_pixels_N10000': 894.4,
        'mean_excess_kurtosis_raw_active_pixels_N10000': 195.5,
    },
    'nonnegativity_attack': {
        'source': 'atom.py, facet.py, facet2.py, nonneg.py',
        'idea': ('MNIST pixels are >= 0 and 59% of all pixel values are EXACTLY 0, so in '
                 'pixel space the cloud lies on the faces of a translated simplicial cone '
                 'with 81 facets.  A linear map preserves that geometry, so the facets -- '
                 'the pixel functionals -- are identifiable WITHOUT any independence '
                 'assumption.  This is a strictly stronger prior than the one FastICA uses.'),
        'smoothed_atom_mass_true_pixel_projections_mean': 0.764,
        'smoothed_atom_mass_random_direction_mean': 0.059,
        'kept_subspace': ('the search must be run in the subspace where the released '
                          'covariance eigenvalue exceeds 0.3 (56 of 81 directions); the '
                          'epsilon=1e-3 floor leaves ~25 near-degenerate directions whose '
                          'quantisation atoms otherwise hijack the objective '
                          '(with the floor at 0.05 the attack degrades from 13 to 7 pixels)'),
        'single_facet_probe_24_restarts': {
            'median_abs_r_with_a_raw_pixel': 0.901,
            'top6_abs_r': [1.000, 0.999, 0.991, 0.974, 0.999, 0.937],
            'seconds': 14},
        'failed_variant_nonneg_rotation': {
            'note': ('a 400-step orthogonal-rotation search with a concave penalty above a '
                     'low quantile (Plumbley-style non-negative ICA) barely beat the null: '
                     'best-match |r| 0.31 vs 0.29 for the untouched released features and '
                     '0.695 for FastICA.  Reported as a weak implementation, not as evidence '
                     'that the family fails.')},
        'deflation_eigfloor_0.3': {'n10000': load('facet2_N10000.json', {}),
                                   'n1000': load('facet2_N1000.json', {})},
        'deflation_eigfloor_0.05_worse': load('facet_N10000.json', {}),
        'exact_rank75_whitening_variant': {
            'source': 'rank75.py',
            'note': ('the eps=0 design point: drop the null directions, whiten exactly on '
                     'the 75-dim range, rotate.  Released covariance eigenvalues 0.599-1.322 '
                     '(sample).  The same 24-restart facet probe does WORSE there: median '
                     '|r| 0.666, max 0.917, none above 0.99, versus median 0.901 and several '
                     'above 0.99 on the eps=1e-3 variant.  One configuration of one quick '
                     'attack; not a recommendation on its own.'),
            'median_abs_r': 0.666, 'max_abs_r': 0.917, 'count_above_0.99': 0,
            'random_direction_atom_score': 0.066},
        'end_to_end': {'n1000': load('facet2_N1000.json', {}),
                       'n10000': load('facet2_N10000.json', {})},
    },
    'downstream_cnn09_cpu_pilots': batches,
    'cnn_member_seed_spread': seedvar,
    'dense_baselines': dense,
}

(RESEARCH / 'whitening-attack-results.json').write_text(json.dumps(results, indent=1) + '\n')
print('wrote', RESEARCH / 'whitening-attack-results.json')
