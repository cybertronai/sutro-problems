"""Kernel and transductive learners for the gpumode 1.2.0 release ladder.

Contract (study-wide, see ``runner.call_fit_predict``)::

    fit_predict(train_z, train_y, query_z, config, seed, device, deadline_unix)
        -> {'logits': float32 (Q,10), 'labels': uint8 (Q,), 'metrics': dict}

``train_z`` / ``query_z`` are the released features ``z = Q W (x - mu)`` of one
draw: ``W`` is exact PCA whitening onto the top 60 principal directions of that
draw's *training* rows, ``mu`` their mean and ``Q`` a secret per-draw Haar
rotation.  Nothing here may assume a pixel lattice, a feature order or a scale
beyond what the two released arrays show; every family below is invariant to an
orthogonal transform of the feature space (Euclidean distances and inner
products are all that is used), which is exactly the invariance the secret ``Q``
demands.

``config['family']`` selects one of

  'krr'        kernel ridge regression on centred one-hot targets with
               'arccos' (Cho & Saul, NIPS 2009, order-1, depth d), 'rbf' or
               'laplace' kernels.  Bandwidth and ridge are chosen by k-fold CV
               on the TRAINING rows only, with one eigendecomposition per
               (bandwidth, fold) so the whole ridge grid is nearly free.
  'rfm'        Recursive Feature Machine (Radhakrishnan, Beaglehole, Pandit &
               Belkin, Science 383:1461, 2024).  Laplace kernel with a learned
               Mahalanobis metric ``K_M(x,z) = exp(-sqrt((x-z)' M (x-z))/L)``;
               ``M`` is the average gradient outer product (AGOP) of the current
               predictor, iterated.  Conventions follow the reference code
               (github.com/aradha/recursive_feature_machines, both ``rfm.py`` on
               ``main`` and ``rfm/recursive_feature_machine.py`` on the
               ``pip_install`` branch): the AGOP is accumulated with
               ``einsum('ncd,ncD->dD', G, G)``, scale-normalised, optionally
               reduced to its diagonal, and optionally raised to the power
               ``agop_power`` (the reference default ``0.5`` for the generic
               kernels; plain ``LaplaceRFM`` uses ``M`` itself, i.e. power 1,
               which is the default here).  One deliberate deviation: the
               reference normalises by the largest ENTRY of ``M``, which is
               basis dependent and would therefore make the fit depend on the
               release's secret rotation, so the default here is the invariant
               ``'max_eigenvalue'`` (``'max_entry'`` reproduces the reference).
  'labelprop'  transductive label spreading (Zhou, Bousquet, Lal, Weston &
               Schoelkopf, NIPS 2004): ``F <- alpha S F + (1-alpha) Y`` on the
               symmetrically normalised adjacency ``S = D^-1/2 W D^-1/2`` of a
               symmetric k-NN graph over train+query rows, in the Euclidean
               metric or in the RFM metric learned on the labelled rows.
  'selftrain'  wrapper: fit a 'krr' or 'rfm' base, pseudo-label the confident
               query rows, refit on train + pseudo-labelled rows.
  'ensemble'   average the standardised scores of several member configs, each
               with its own preprocessing; the members share one deadline.

Use of the query rows
---------------------
The harness hands the learner the released query FEATURES, so using them
without labels is legal; every family records the fact in
``metrics['uses_query_images_unlabeled']`` (and in the spelling the sibling
module uses, ``metrics['uses_query_features_unlabeled']``).  Those two flags are
the audit: ``metrics['rewhiten']`` is the TOP-LEVEL setting only, so a wrapper
whose base or members re-whiten reports ``'none'`` there -- read
``metrics['rewhiten_anywhere']`` and ``metrics['preprocess_effective']`` instead.  'krr' and 'rfm'
are inductive and set it False unless ``rewhiten`` is on; 'labelprop' and
'selftrain' always set it True.  Query LABELS never enter this module: the
signature has no slot for them and ``study.query_labels`` refuses to run in any
process that has imported this file.

``rewhiten``
------------
The release whitens on the draw's N training rows, so the training covariance
is *exactly* the identity while the population covariance is not: at N=500 the
60 eigenvalues are estimated from 500 rows (aspect ratio 0.12), so the release
amplifies the badly estimated directions.  ``rewhiten='query'`` estimates the
covariance ``C`` of the released query rows and applies the symmetric (ZCA)
map ``C^-1/2``, which undoes that amplification without knowing ``Q`` or ``W``
(it commutes with the secret rotation).  It is transductive and flagged.

Recommended configurations (selected on the two dev seeds, see
research/kernel-pilots.md; the module defaults are deliberately left neutral)
------------------------------------------------------------------------
  {'family': 'krr', 'kernel': 'arccos', 'depth': 3}            -- the cheap baseline
  {'family': 'krr', 'kernel': 'arccos', 'depth': 3,
   'metric_learn': 'within_class', 'metric_shrinkage': 0.05}   -- + supervised sphering
  {'family': 'selftrain', 'quantile': 0.3, 'rounds': 2,
   'base': {...the config above...}}                           -- + pseudo-labels
  {'family': 'labelprop', 'metric_learn': 'within_class',
   'metric_shrinkage': 0.05, 'rewhiten': 'query'}              -- best at N=500
  {'family': 'ensemble', 'members': [the two above]}            -- best overall

Timing and the deadline
-----------------------
Every family takes ``deadline_unix`` seriously: the CV sweep is costed from its
own first cell and truncated when the projection would overrun, and if even the
final fit does not fit in the remaining time the learner falls back to the
nearest-class-centroid predictor, which is O(nd), and sets
``metrics['emergency_fallback'] = True``.  The final-fit check does not rely on
the sweep: ``_probe_fit_cost`` measures one small kernel build, one Cholesky and
one ``256 x n`` inference block on the real features, so a single-cell config
(no sweep, hence no measured cell) and the ``n_query x n`` inference kernel --
which the cell can never price, and which reaches 25 s at N=10000 against a 5 s
``inference_margin_seconds`` -- are both costed before anything large is built.  ``metrics`` is JSON-safe (plain
Python scalars only), as ``runner.write_json`` uses ``allow_nan=False``.
"""
from __future__ import annotations

import math
import platform
import time

import numpy as np
import torch

NUM_CLASSES = 10
_TINY = 1e-30

# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------
COMMON_DEFAULTS = {
    'family': None,
    'module': 'kernels',
    'dtype': 'float64',
    # 'none' | 'query' | 'pooled': ZCA re-whitening with the query (or pooled
    # train+query) covariance of the release.  Transductive when not 'none'.
    'rewhiten': 'none',
    'rewhiten_shrinkage': 0.0,       # C <- (1-s) C + s I before the inverse root
    # Supervised sphering with the TRAINING labels only (no query rows): the
    # release whitens the total covariance, which inflates the low-variance
    # principal directions; 'within_class' divides instead by the pooled
    # within-class covariance, the classical pre-LDA metric.
    'metric_learn': 'none',          # 'none' | 'within_class'
    'metric_shrinkage': 0.2,         # S <- (1-s) S_w + s (tr S_w / d) I
    # Transductive prior matching: add one bias per class so the query rows'
    # predicted class mass matches the training prior (Sinkhorn in log space).
    'prior_match': 'none',           # 'none' | 'sinkhorn'
    'prior_match_temperature': 'auto',   # 'auto' (median top-2 margin) or a float
    'prior_match_iters': 50,
    'inference_margin_seconds': 5.0,
    'notes': '',
}

KRR_DEFAULTS = {
    'kernel': 'laplace',             # 'laplace' | 'rbf' | 'arccos'
    'depth': 1,                      # arc-cosine composition depth
    'arccos_bias': 0.0,              # constant feature appended before 'arccos'
    'bandwidth': None,               # fixed bandwidth; None -> sweep the grid
    'bandwidth_grid': None,          # absolute grid; None -> bandwidth_scales
    'bandwidth_scales': [0.25, 0.5, 1.0, 2.0],   # x median training distance
    'ridge': None,                   # fixed ridge; None -> sweep ridge_grid
    'ridge_grid': [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3],
    'ridge_mode': 'trace',           # 'trace' | 'per_sample' | 'absolute'
    'target_mode': 'centered',       # 'centered' (one-hot - 1/10) | 'onehot'
    'cv_folds': 5,
    'cv_max_rows': 2000,             # cap on the rows used for CV (cost control)
    'median_sample': 2000,           # rows used for the median-distance scale
}

RFM_DEFAULTS = {
    'bandwidth': None,
    'bandwidth_grid': [5.0, 10.0, 20.0],         # reference default is L = 10
    'bandwidth_scales': None,
    'bandwidth_mode': 'constant',    # 'constant' (reference) | 'adaptive_median'
    'ridge': None,
    'ridge_grid': [1e-7, 1e-6, 1e-5],
    'ridge_mode': 'trace',
    'iters': 5,                      # T: AGOP updates (reference default 5)
    'agop_power': 1.0,               # 1.0 = LaplaceRFM; 0.5 = the 'sqrt' option
    'agop_power_grid': None,
    'diag': False,                   # diagonal M (reference 'diag' flag)
    'diag_grid': None,
    'centering': False,              # centre the gradients before the outer product
    # Scale normalisation of the AGOP.  The reference divides by the largest
    # ENTRY of M, which depends on the basis and therefore on the release's
    # secret Haar rotation; 'max_eigenvalue' is the rotation-invariant version
    # of the same rule and is the default here (see the module docstring).
    'agop_normalisation': 'max_eigenvalue',   # 'max_eigenvalue'|'max_entry'|'trace'
    'agop_chunk': 2048,              # rows of the Jacobian held at once
    'target_mode': 'centered',
    'holdout_fraction': 0.2,         # one 80/20 split per draw for selection
    'holdout_max_rows': 6000,        # cap on the rows used for selection
    'select_iteration': True,        # pick T by validation accuracy per iteration
    'metric_floor': 1e-12,           # eigenvalue clamp for M^power
}

LABELPROP_DEFAULTS = {
    'metric': 'euclid',              # 'euclid' | 'rfm'
    'rfm_config': None,              # inner RFM that learns M when metric='rfm'
    'k': None,
    'k_grid': [5, 10, 20],
    'alpha': None,
    'alpha_grid': [0.5, 0.9, 0.99],
    'weight': 'binary',              # 'binary' | 'heat'
    'heat_scale': 1.0,               # x mean k-NN distance
    'spread_iters': 60,
    'cv_folds': 5,
    'cv_graph': 'full',              # 'full' (train+query nodes) | 'labeled_only'
    'knn_chunk': 1024,
    'fallback': 'nearest_labeled',   # for query nodes the spread never reaches
}

SELFTRAIN_DEFAULTS = {
    'base': None,                    # a 'krr' or 'rfm' config (required)
    'rounds': 1,                     # 1 or 2 pseudo-labelling rounds
    'quantile': 0.5,                 # keep query rows above this margin quantile
    'pseudo_weight': 1.0,            # sample weight of a pseudo-labelled row
    'reuse_hyperparameters': True,   # do not re-run CV on the augmented set
}

ENSEMBLE_DEFAULTS = {
    'members': None,                 # list of member configs (required, >= 1)
    'weights': None,                 # per-member weights; None -> equal
    'standardise': 'global_std',     # 'global_std' | 'none'
    'drop_failed_members': True,     # a member that runs out of time is skipped
}

FAMILY_DEFAULTS = {'krr': KRR_DEFAULTS, 'rfm': RFM_DEFAULTS,
                   'labelprop': LABELPROP_DEFAULTS, 'selftrain': SELFTRAIN_DEFAULTS,
                   'ensemble': ENSEMBLE_DEFAULTS}

KERNELS = ('laplace', 'rbf', 'arccos')
RIDGE_MODES = ('trace', 'per_sample', 'absolute')
TARGET_MODES = ('centered', 'onehot')
REWHITEN_MODES = ('none', 'query', 'pooled')
DTYPES = {'float64': torch.float64, 'float32': torch.float32}


def resolve_config(config):
    """Merge a candidate config with its family defaults; reject unknown keys."""
    if not isinstance(config, dict):
        raise TypeError('config must be a dict')
    family = config.get('family')
    if family not in FAMILY_DEFAULTS:
        raise ValueError(f"config['family'] must be one of {sorted(FAMILY_DEFAULTS)}")
    merged = {**COMMON_DEFAULTS, **FAMILY_DEFAULTS[family], **config}
    unknown = set(config) - set(COMMON_DEFAULTS) - set(FAMILY_DEFAULTS[family])
    if unknown:
        raise ValueError(f'Unknown config keys for {family}: {sorted(unknown)}')
    merged['family'] = family
    if merged['module'] != 'kernels':
        raise ValueError("config['module'] must be 'kernels' for this module")
    if merged['dtype'] not in DTYPES:
        raise ValueError(f"dtype must be one of {sorted(DTYPES)}")
    if merged['rewhiten'] not in REWHITEN_MODES:
        raise ValueError(f'rewhiten must be one of {list(REWHITEN_MODES)}')
    if not 0.0 <= float(merged['rewhiten_shrinkage']) <= 1.0:
        raise ValueError('rewhiten_shrinkage must be in [0, 1]')
    if merged['metric_learn'] not in ('none', 'within_class'):
        raise ValueError("metric_learn must be 'none' or 'within_class'")
    if not 0.0 <= float(merged['metric_shrinkage']) <= 1.0:
        raise ValueError('metric_shrinkage must be in [0, 1]')
    if merged['prior_match'] not in ('none', 'sinkhorn'):
        raise ValueError("prior_match must be 'none' or 'sinkhorn'")
    if merged['prior_match_temperature'] != 'auto':
        if float(merged['prior_match_temperature']) <= 0.0:
            raise ValueError("prior_match_temperature must be 'auto' or positive")
    if int(merged['prior_match_iters']) < 1:
        raise ValueError('prior_match_iters must be >= 1')
    if family in ('krr', 'rfm'):
        if merged['ridge_mode'] not in RIDGE_MODES:
            raise ValueError(f'ridge_mode must be one of {list(RIDGE_MODES)}')
        if merged['target_mode'] not in TARGET_MODES:
            raise ValueError(f'target_mode must be one of {list(TARGET_MODES)}')
        if not _grid(merged, 'ridge'):
            raise ValueError('ridge grid is empty')
    if family == 'krr':
        if merged['kernel'] not in KERNELS:
            raise ValueError(f'kernel must be one of {list(KERNELS)}')
        if int(merged['depth']) < 1:
            raise ValueError('depth must be >= 1')
        if int(merged['cv_folds']) < 2:
            raise ValueError('cv_folds must be >= 2')
    if family == 'rfm':
        if int(merged['iters']) < 0:
            raise ValueError('iters must be >= 0')
        if merged['agop_normalisation'] not in ('max_eigenvalue', 'max_entry', 'trace'):
            raise ValueError("agop_normalisation must be 'max_eigenvalue', 'max_entry' "
                             "or 'trace'")
        if merged['bandwidth_mode'] not in ('constant', 'adaptive_median'):
            raise ValueError("bandwidth_mode must be 'constant' or 'adaptive_median'")
        if not 0.0 < float(merged['holdout_fraction']) < 1.0:
            raise ValueError('holdout_fraction must be in (0, 1)')
        for key in ('agop_power', 'diag', 'bandwidth'):
            if not _grid(merged, key):
                raise ValueError(f'{key} grid is empty')
    if family == 'labelprop':
        if merged['metric'] not in ('euclid', 'rfm'):
            raise ValueError("metric must be 'euclid' or 'rfm'")
        if merged['weight'] not in ('binary', 'heat'):
            raise ValueError("weight must be 'binary' or 'heat'")
        if merged['cv_graph'] not in ('full', 'labeled_only'):
            raise ValueError("cv_graph must be 'full' or 'labeled_only'")
        if merged['fallback'] not in ('nearest_labeled', 'prior'):
            raise ValueError("fallback must be 'nearest_labeled' or 'prior'")
        for key in ('k', 'alpha'):
            if not _grid(merged, key):
                raise ValueError(f'{key} grid is empty')
        for value in _grid(merged, 'alpha'):
            if not 0.0 < float(value) < 1.0:
                raise ValueError('alpha must be in (0, 1)')
        for value in _grid(merged, 'k'):
            if int(value) < 1:
                raise ValueError('k must be >= 1')
        if merged['metric'] == 'rfm':
            inner = dict(merged['rfm_config'] or {})
            inner.setdefault('family', 'rfm')
            if inner['family'] != 'rfm':
                raise ValueError("labelprop rfm_config must have family 'rfm'")
            inner.setdefault('dtype', merged['dtype'])
            inner = resolve_config(inner)
            for key in ('rewhiten', 'metric_learn', 'prior_match'):
                if inner[key] != 'none':
                    raise ValueError(
                        f"labelprop rfm_config may not set {key!r}: the inner RFM sees "
                        "the features labelprop was already given; put the "
                        'preprocessing on the labelprop config itself')
            merged['rfm_config'] = inner
    if family == 'ensemble':
        members = merged['members']
        if not members:
            raise ValueError("ensemble needs a non-empty config['members']")
        resolved_members = [resolve_config(member) for member in members]
        for member in resolved_members:
            if member['family'] == 'ensemble':
                raise ValueError('ensembles may not nest')
            if member['prior_match'] != 'none':
                raise ValueError('prior_match belongs on the ensemble, not on a member')
        merged['members'] = resolved_members
        if merged['weights'] is not None:
            if len(merged['weights']) != len(resolved_members):
                raise ValueError('weights must have one entry per member')
            if min(float(w) for w in merged['weights']) < 0.0:
                raise ValueError('weights must be non-negative')
            if sum(float(w) for w in merged['weights']) <= 0.0:
                raise ValueError('weights must not all be zero')
        if merged['standardise'] not in ('global_std', 'none'):
            raise ValueError("standardise must be 'global_std' or 'none'")
    if family == 'selftrain':
        if merged['base'] is None:
            raise ValueError("selftrain needs config['base']")
        base = resolve_config(merged['base'])
        if base['family'] not in ('krr', 'rfm'):
            raise ValueError("selftrain base family must be 'krr' or 'rfm'")
        if base['prior_match'] != 'none':
            raise ValueError("prior_match belongs on the wrapper, not on its base")
        merged['base'] = base
        if int(merged['rounds']) not in (1, 2):
            raise ValueError('rounds must be 1 or 2')
        weight = float(merged['pseudo_weight'])
        if weight < 1.0 or weight != int(weight):
            raise ValueError('pseudo_weight must be a positive integer (rows are repeated)')
        if not 0.0 <= float(merged['quantile']) < 1.0:
            raise ValueError('quantile must be in [0, 1)')
    return merged


def _grid(config, key):
    """``[config[key]]`` when the scalar is set, else ``config[key + '_grid']``.

    ``bandwidth`` additionally falls back to ``bandwidth_scales`` (median-relative),
    which is resolved to absolute values once the training rows are known.
    """
    value = config.get(key)
    if value is not None:
        return [value]
    grid = config.get(f'{key}_grid')
    if grid:
        return list(grid)
    if key == 'bandwidth' and config.get('bandwidth_scales'):
        return list(config['bandwidth_scales'])
    return []


# --------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------
def first_max_labels(logits):
    """argmax with ties resolved to the lowest class (numpy returns the first max)."""
    return np.argmax(np.asarray(logits, dtype=np.float32), axis=1).astype(np.uint8)


def _jsonable(value):
    """Plain-Python copy of a metrics tree (runner.write_json forbids NaN/numpy)."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if value is None or isinstance(value, str):
        return value
    return str(value)


class _OutOfTime(Exception):
    """Raised when the remaining budget cannot cover the next mandatory stage."""


def _time_call(call, device, target_seconds=0.005, max_repeats=8):
    """Seconds per call, repeated until the measurement clears the timer noise.

    A 256-row probe of a cheap kernel runs in well under a millisecond, and a
    noisy measurement extrapolated by ``(n/m)^3`` is worse than no measurement.
    """
    started = time.time()
    repeats = 0
    while True:
        result = call()
        del result
        repeats += 1
        _sync(device)
        elapsed = time.time() - started
        if elapsed >= float(target_seconds) or repeats >= int(max_repeats):
            return max(elapsed / repeats, 1e-9)


def _probe_fit_cost(build, train, n_query, probe_rows=512, probe_query=256,
                    probe_entries=2_000_000):
    """Measure the final fit's cost directly, on blocks ~100x smaller than the fit.

    ``build(a, b, same)`` returns the kernel block the real fit will build.  Two
    measurements are taken, both on the real features and the real kernel:

      * ``m x m`` build + Cholesky with ``m = min(n, probe_rows)`` -- the
        quadratic and cubic training terms, extrapolated as ``(n/m)^2``
        and ``(n/m)^3``;
      * a ``q x n`` build -- inference, at the REAL ``n``, so the per-entry cost
        is the one the full ``n_query x n`` block will pay and the extrapolation
        is linear in the number of rows.  ``q`` never exceeds ``n`` (the rows are
        taken from ``train``: a query row costs the same as a training row).

    This is what makes a single-cell config (no CV sweep, hence no measured cell)
    and a query-heavy draw (``n_query >> n``: 10,000 query rows against N=500)
    costable at all; both used to fall through to the 0.02 s floor.
    """
    n = int(train.shape[0])
    m = int(min(n, int(probe_rows)))
    sub = train[:m]
    build_seconds = _time_call(lambda: build(sub, sub, True), train.device)
    k = build(sub, sub, True)
    zeros = torch.zeros((m, NUM_CLASSES), dtype=k.dtype, device=k.device)
    lam = max(float(torch.diagonal(k).sum()), 1.0) * 1e-6
    solve_seconds = _time_call(lambda: _solve_ridge(k, zeros, lam), train.device)
    del k, zeros
    rows = max(int(probe_query), int(math.ceil(float(probe_entries) / max(n, 1))))
    q = int(min(n, max(int(n_query), 1), rows))
    query_seconds = _time_call(lambda: build(train[:q], train, False), train.device)
    return {'probe_rows': m, 'probe_query_rows': q,
            'probe_build_seconds': build_seconds, 'probe_solve_seconds': solve_seconds,
            'probe_query_seconds': query_seconds,
            'probe_total_seconds': build_seconds + solve_seconds + query_seconds}


_PROBE_INFERENCE_MARGIN = 2.0
"""Safety factor on the extrapolated inference cost.

The probe's ``q x n`` block is small enough to stay in cache while the real
``n_query x n`` block is not, and for the elementwise-heavy kernels (arc-cosine
does three arccos/sin/cos passes over the whole block) the per-entry cost of the
full build measures up to ~2x the probe's on this CPU.  Under-costing inference
is exactly the failure this estimate exists to prevent, so the term is doubled.
"""


def _probe_terms(probe, n, n_query):
    """``(one kernel build + Cholesky at n rows, inference over n_query rows)``."""
    ratio = float(n) / max(float(probe['probe_rows']), 1.0)
    training = (probe['probe_build_seconds'] * ratio ** 2
                + probe['probe_solve_seconds'] * ratio ** 3)
    inference = (_PROBE_INFERENCE_MARGIN * probe['probe_query_seconds']
                 * float(n_query) / max(float(probe['probe_query_rows']), 1.0))
    return training, inference


def _fit_cost_estimate(n, cv_rows, folds, cell_seconds, n_query, probe=None, floor=0.02):
    """Seconds the final fit AND its inference will need.

    Training is the larger of two extrapolations: the CV cell (``folds``
    eigendecompositions of ``m = (folds-1)/folds * cv_rows`` rows, about ``9 m^3``
    flops each, against one Cholesky of ``n`` rows at ``n^3/3``) and the probe's
    own build + Cholesky.  Inference is the probe's ``probe_query x n`` block
    scaled to ``n_query`` rows -- the cell can never supply it, which is why the
    old cell-only estimate costed the ``n_query x n`` kernel at zero.
    """
    training = 0.0
    if cell_seconds and cv_rows:
        m = float(cv_rows) * (int(folds) - 1) / int(folds)
        cubic = (float(n) ** 3 / 3.0) / (int(folds) * 9.0 * max(m, 1.0) ** 3)
        training = cubic * float(cell_seconds)
    inference = 0.0
    if probe:
        probe_training, inference = _probe_terms(probe, n, n_query)
        training = max(training, probe_training)
    if training <= 0.0 and inference <= 0.0:
        return float(floor)
    return max(float(floor), 1.5 * (training + inference) + 0.1)


class Clock:
    """Wall-clock budget: ``remaining()`` already subtracts the inference margin."""

    def __init__(self, deadline_unix, margin_seconds):
        self.deadline = float(deadline_unix)
        self.margin = float(margin_seconds)
        self.started = time.time()

    def remaining(self):
        return self.deadline - self.margin - time.time()

    def allows(self, cost_seconds):
        return self.remaining() > float(cost_seconds)

    def elapsed(self):
        return time.time() - self.started


def _sync(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize()


def _device_name(device):
    if torch.device(device).type == 'cuda':                    # pragma: no cover
        try:
            return torch.cuda.get_device_name(0)
        except Exception:
            return 'cuda'
    return f'cpu ({platform.machine()})'


def _tensor(array, device, dtype):
    return torch.as_tensor(np.ascontiguousarray(array), dtype=dtype,
                           device=torch.device(device))


def _targets(y, n_classes, target_mode, dtype):
    """One-hot (optionally centred) regression targets, (n, C)."""
    target = torch.zeros((y.shape[0], n_classes), dtype=dtype, device=y.device)
    target[torch.arange(y.shape[0], device=y.device), y.long()] = 1.0
    if target_mode == 'centered':
        target -= 1.0 / n_classes
    return target


def _sqdist(a, b, same=False):
    """Squared Euclidean distances, clamped at 0 (exact zero on the diagonal)."""
    aa = (a * a).sum(1, keepdim=True)
    bb = aa.T if same else (b * b).sum(1, keepdim=True).T
    out = torch.addmm(aa + bb, a, b.T, beta=1.0, alpha=-2.0)
    out.clamp_(min=0.0)
    if same:
        out = 0.5 * (out + out.T)
        out.fill_diagonal_(0.0)
    return out


def _dist(a, b, same=False):
    return _sqdist(a, b, same=same).sqrt_()


def _median_distance(x, sample, seed):
    """Median pairwise Euclidean distance of a seeded subsample of ``x``."""
    n = x.shape[0]
    if n > sample:
        index = np.random.default_rng([int(seed), 0x9E37]).choice(
            n, int(sample), replace=False)
        x = x[torch.as_tensor(np.sort(index), device=x.device)]
    d = _dist(x, x, same=True)
    m = x.shape[0]
    if m < 2:
        return 1.0
    off = d[~torch.eye(m, dtype=torch.bool, device=d.device)]
    return float(off.median())


def _arccos_kernel(a, b, depth, bias, same=False):
    """Cho & Saul order-1 arc-cosine kernel, composed ``depth`` times.

    ``k_{l+1}(x,z) = (1/pi) sqrt(k_l(x,x) k_l(z,z)) (sin t + (pi - t) cos t)`` with
    ``t = arccos(k_l(x,z)/sqrt(k_l(x,x)k_l(z,z)))``; the self term is invariant,
    ``k_l(x,x) = ||x||^2`` at every depth, which is why ``norm`` is built once.
    """
    if bias:
        pad_a = torch.full((a.shape[0], 1), float(bias), dtype=a.dtype, device=a.device)
        a = torch.cat([a, pad_a], dim=1)
        if same:
            b = a
        else:
            pad_b = torch.full((b.shape[0], 1), float(bias), dtype=b.dtype, device=b.device)
            b = torch.cat([b, pad_b], dim=1)
    na = (a * a).sum(1, keepdim=True)
    nb = na.T if same else (b * b).sum(1, keepdim=True).T
    norm = (na * nb).clamp_(min=_TINY).sqrt_()
    k = a @ b.T
    for _ in range(int(depth)):
        cosine = (k / norm).clamp_(-1.0, 1.0)
        theta = torch.arccos(cosine)
        k = norm * (torch.sin(theta) + (math.pi - theta) * torch.cos(theta)) / math.pi
    if same:
        k = 0.5 * (k + k.T)
    return k


def _kernel_matrix(a, b, kernel, bandwidth, depth=1, bias=0.0, same=False):
    """One of the three stationary/homogeneous kernels this module offers."""
    if kernel == 'laplace':
        return torch.exp(_dist(a, b, same=same).mul_(-1.0 / float(bandwidth)))
    if kernel == 'rbf':
        return torch.exp(_sqdist(a, b, same=same).mul_(-1.0 / (2.0 * float(bandwidth) ** 2)))
    if kernel == 'arccos':
        return _arccos_kernel(a, b, depth, bias, same=same)
    raise ValueError(f'unknown kernel {kernel!r}')


def _ridge_absolute(ridge, n, kernel_trace, mode):
    """Grid value -> the number actually added to the diagonal."""
    if mode == 'absolute':
        return float(ridge)
    if mode == 'per_sample':
        return float(ridge) * float(n)
    return float(ridge) * float(kernel_trace)          # 'trace'


def _solve_ridge(k, y, lam, inplace=False):
    """``(K + lam I)^-1 Y`` by Cholesky, with jitter escalation if indefinite.

    The ridge goes on the diagonal in place, of ``k`` itself when the caller
    passes ``inplace=True`` (it must not use ``k`` afterwards).  The old form
    materialised ``torch.eye(n)`` and then ``k + (lam + jitter) * eye``, so the
    largest solve in the study -- n = 17,000 float64 rows, 2.31 GB a matrix, from
    a two-round ``selftrain`` at N=10000 -- held four such matrices where two
    suffice: 4.6 GB of avoidable peak.  ``cholesky_ex`` returns an info code
    instead of raising, which also avoids re-adding the whole matrix per attempt.
    """
    n = k.shape[0]
    a = k if inplace else k.clone()
    scale = float(torch.diagonal(a).mean()) if n else 1.0
    added = 0.0
    for attempt in range(6):
        jitter = 0.0 if attempt == 0 else max(lam, scale) * (10.0 ** (attempt - 5))
        a.diagonal().add_(lam + jitter - added)
        added = lam + jitter
        factor, info = torch.linalg.cholesky_ex(a)
        if int(info) == 0:
            return torch.cholesky_solve(y, factor)
        del factor                                             # pragma: no cover
    return torch.linalg.lstsq(a, y).solution                   # pragma: no cover


def _fold_indices(n, folds, seed):
    """Deterministic k-fold split of ``range(n)``; never depends on the query rows."""
    order = np.random.default_rng([int(seed), 0x0F01D]).permutation(n)
    return [np.sort(order[i::folds]) for i in range(int(folds))]


def _cv_rows(n, cap, seed):
    """Row subset used for cross-validation (a cost cap, not a statistical choice)."""
    if cap is None or n <= int(cap):
        return np.arange(n)
    index = np.random.default_rng([int(seed), 0xC0FFEE]).choice(n, int(cap), replace=False)
    return np.sort(index)


def _inverse_sqrt(matrix, floor=1e-12):
    """Symmetric ``C^-1/2`` with the eigenvalues clamped at ``floor``."""
    values, vectors = torch.linalg.eigh(matrix)
    values = values.clamp_(min=float(floor))
    return (vectors * values.rsqrt()) @ vectors.T


def _matrix_power(matrix, power, floor=0.0, diagonal=False):
    """``M^power`` for a PSD matrix (reference ``rfm.utils.matrix_power``).

    ``floor`` is the eigenvalue clamp ``config['metric_floor']`` advertises; it
    used to be declared and then ignored in favour of a hard 0.0, so the key was
    a no-op (two runs with ``metric_floor`` 1e-12 and 1e9 were bit-identical).
    """
    floor = float(floor)
    if diagonal:
        return matrix.clamp(min=floor) ** float(power)
    if float(power) == 1.0 and floor <= 0.0:
        return matrix
    values, vectors = torch.linalg.eigh(matrix)
    values = values.clamp_(min=floor) ** float(power)
    out = (vectors * values) @ vectors.T
    return 0.5 * (out + out.T)


def _metric_factor(metric, diagonal, floor=0.0):
    """``R`` with ``R' R = M``: Mahalanobis distances become Euclidean in ``x @ R``."""
    floor = float(floor)
    if diagonal:
        return metric.clamp(min=floor).sqrt()
    values, vectors = torch.linalg.eigh(metric)
    values = values.clamp_(min=floor).sqrt()
    out = (vectors * values) @ vectors.T
    return 0.5 * (out + out.T)


def _apply_metric(x, factor, diagonal):
    return x * factor if diagonal else x @ factor


def within_class_transform(train, train_y, shrinkage, n_classes=NUM_CLASSES):
    """``S_w^-1/2`` from the training rows: the pooled within-class sphering map.

    Uses training labels only -- it is inductive, not transductive.  It is also
    equivariant under an orthogonal change of basis (``S_w -> Q' S_w Q`` gives
    ``T -> Q' T Q``), so the release's secret Haar rotation cancels.
    """
    centred = train.clone()
    for label in range(n_classes):
        mask = train_y == label
        if int(mask.sum()) > 0:
            centred[mask] -= train[mask].mean(0, keepdim=True)
    present = int(torch.unique(train_y).numel())
    dof = max(1, train.shape[0] - present)
    within = (centred.T @ centred) / dof
    scale = float(torch.diagonal(within).mean())
    eye = torch.eye(within.shape[0], dtype=within.dtype, device=within.device)
    shrunk = (1.0 - float(shrinkage)) * within + float(shrinkage) * scale * eye
    return _inverse_sqrt(shrunk) * math.sqrt(scale)


def _preprocess(train, query, train_y, config):
    """Optional query ZCA then within-class sphering; returns (train, query, note).

    The order matters: ``rewhiten`` repairs the release's own whitening with the
    population covariance the query rows reveal, and the supervised
    ``metric_learn`` step is applied to the repaired features (the reverse order
    would let the query ZCA undo the supervised metric).
    """
    note = {}
    mode = config['rewhiten']
    if mode == 'none':
        note['rewhiten'] = 'none'
    else:
        train, query, note = _rewhiten(train, query, config)
    if config['metric_learn'] == 'within_class':
        transform = within_class_transform(train, train_y, config['metric_shrinkage'])
        train, query = train @ transform, query @ transform
        note.update({'metric_learn': 'within_class',
                     'metric_shrinkage': float(config['metric_shrinkage'])})
    return train, query, note


def _rewhiten(train, query, config):
    """ZCA with the query (or pooled) covariance of the release."""
    note = {}
    mode = config['rewhiten']
    source = query if mode == 'query' else torch.cat([train, query], dim=0)
    mean = source.mean(0, keepdim=True)
    centred = source - mean
    cov = (centred.T @ centred) / max(1, centred.shape[0] - 1)
    shrink = float(config['rewhiten_shrinkage'])
    if shrink > 0.0:
        eye = torch.eye(cov.shape[0], dtype=cov.dtype, device=cov.device)
        cov = (1.0 - shrink) * cov + shrink * float(torch.diagonal(cov).mean()) * eye
    transform = _inverse_sqrt(cov)
    note.update({'rewhiten': mode, 'rewhiten_shrinkage': shrink,
                 'rewhiten_rows': int(source.shape[0]),
                 'rewhiten_cond': float(torch.linalg.cond(cov))})
    return (train - mean) @ transform, (query - mean) @ transform, note


def sinkhorn_prior_match(scores, prior, temperature, iterations=50):
    """Add one bias per class so the query mass matches ``prior`` (log-space Sinkhorn).

    Row marginals are fixed by the softmax, column marginals are pushed onto
    ``prior`` by the dual update ``b <- b + log(prior) - log(column mean)``; the
    returned array is ``scores / T + b``, whose per-row ordering differs from
    ``scores`` only through the shared class bias.  Transductive: it reads all
    query scores at once, but no query label.
    """
    prior = prior / prior.sum()
    logits = scores / float(temperature)
    bias = torch.zeros(scores.shape[1], dtype=scores.dtype, device=scores.device)
    log_prior = torch.log(prior.clamp(min=_TINY))
    for _ in range(int(iterations)):
        probability = torch.softmax(logits + bias, dim=1)
        bias = bias + log_prior - torch.log(probability.mean(0).clamp(min=_TINY))
        bias = bias - bias.mean()
    return logits + bias, bias


def _auto_temperature(scores):
    """Median top-1 minus top-2 gap: a scale for the score matrix, not a fit."""
    top = torch.topk(scores, 2, dim=1).values
    gap = float((top[:, 0] - top[:, 1]).median())
    return gap if gap > 1e-12 else max(float(scores.abs().max()), 1e-12)


def _centroid_scores(train, train_y, query, n_classes=NUM_CLASSES):
    """Nearest-class-centroid scores; the O(nd) emergency predictor."""
    means = torch.zeros((n_classes, train.shape[1]), dtype=train.dtype, device=train.device)
    counts = torch.zeros(n_classes, dtype=train.dtype, device=train.device)
    ones = torch.ones(train.shape[0], dtype=train.dtype, device=train.device)
    means.index_add_(0, train_y.long(), train)
    counts.index_add_(0, train_y.long(), ones)
    empty = counts == 0
    means[~empty] /= counts[~empty].unsqueeze(1)
    scores = -_sqdist(query, means)
    scores[:, empty] = -1e30
    return scores


# --------------------------------------------------------------------------
# family 'krr': kernel ridge regression with k-fold CV on the training rows
# --------------------------------------------------------------------------
def _krr_bandwidths(config, train, seed):
    """Absolute bandwidth grid + how it was obtained (arc-cosine has none)."""
    if config['kernel'] == 'arccos':
        return [1.0], {'bandwidth_source': 'unused (arc-cosine is homogeneous)',
                       'median_distance': None}
    median = _median_distance(train, int(config['median_sample']), seed)
    if config['bandwidth'] is not None or config['bandwidth_grid']:
        return [float(v) for v in _grid(config, 'bandwidth')], {
            'bandwidth_source': 'absolute', 'median_distance': median}
    scales = [float(s) for s in config['bandwidth_scales']]
    return [s * median for s in scales], {
        'bandwidth_source': 'median_scaled', 'median_distance': median,
        'bandwidth_scales': scales}


def _krr_cv_fold_scores(k_full, targets, labels, folds, ridge_grid, ridge_mode):
    """Correct counts per ridge for one kernel, via one eigendecomposition per fold.

    ``K_tt = U diag(w) U'`` gives ``alpha(lam) = U diag(1/(w+lam)) U' Y`` for every
    ``lam`` at the cost of two small matrix products, so the ridge grid is free
    once the fold's eigendecomposition is paid for.
    """
    n = k_full.shape[0]
    correct = np.zeros(len(ridge_grid), dtype=np.int64)
    for validation in folds:
        mask = torch.ones(n, dtype=torch.bool, device=k_full.device)
        val = torch.as_tensor(validation, device=k_full.device, dtype=torch.long)
        mask[val] = False
        train_index = torch.nonzero(mask, as_tuple=False).squeeze(1)
        k_tt = k_full.index_select(0, train_index).index_select(1, train_index)
        k_vt = k_full.index_select(0, val).index_select(1, train_index)
        values, vectors = torch.linalg.eigh(k_tt)
        values = values.clamp_(min=0.0)
        projected = vectors.T @ targets.index_select(0, train_index)      # (m, C)
        crossed = k_vt @ vectors                                          # (v, m)
        trace = float(torch.diagonal(k_tt).sum())
        truth = labels.index_select(0, val)
        for j, ridge in enumerate(ridge_grid):
            lam = _ridge_absolute(ridge, k_tt.shape[0], trace, ridge_mode)
            prediction = crossed @ (projected / (values + lam).unsqueeze(1))
            correct[j] += int((prediction.argmax(1) == truth).sum())
    return correct


def _fit_krr(train, train_y, query, config, seed, clock, report):
    """Sweep (bandwidth, ridge) by k-fold CV on the training rows, then refit."""
    n, device, dtype = train.shape[0], train.device, train.dtype
    targets = _targets(train_y, NUM_CLASSES, config['target_mode'], dtype)
    bandwidths, scale_note = _krr_bandwidths(config, train, seed)
    ridge_grid = [float(r) for r in _grid(config, 'ridge')]
    kernel, depth, bias = config['kernel'], int(config['depth']), float(config['arccos_bias'])

    cv_index = _cv_rows(n, config['cv_max_rows'], seed)
    cv_train = train.index_select(
        0, torch.as_tensor(cv_index, device=device, dtype=torch.long))
    cv_labels = train_y.index_select(0, torch.as_tensor(cv_index, device=device,
                                                        dtype=torch.long))
    cv_targets = _targets(cv_labels, NUM_CLASSES, config['target_mode'], dtype)
    folds = _fold_indices(len(cv_index), int(config['cv_folds']), seed)

    table, best, cell_seconds = [], None, None
    cv_started = time.time()
    sweep = len(bandwidths) * len(ridge_grid) > 1
    if sweep and not clock.allows(0.0):
        report['deadline_truncated_cv'] = True
        sweep = False
    if sweep:
        for bandwidth in bandwidths:
            if cell_seconds is not None and not clock.allows(2.0 * cell_seconds):
                report['deadline_truncated_cv'] = True
                break
            started = time.time()
            k_full = _kernel_matrix(cv_train, cv_train, kernel, bandwidth, depth, bias,
                                    same=True)
            correct = _krr_cv_fold_scores(k_full, cv_targets, cv_labels, folds,
                                          ridge_grid, config['ridge_mode'])
            del k_full
            _sync(device)
            cell_seconds = time.time() - started
            for j, ridge in enumerate(ridge_grid):
                accuracy = float(correct[j]) / float(len(cv_index))
                row = {'bandwidth': float(bandwidth), 'ridge': float(ridge),
                       'cv_accuracy': accuracy}
                table.append(row)
                key = (accuracy, math.log10(max(ridge, 1e-300)))
                if best is None or key > best[0]:
                    best = (key, row)
    chosen = best[1] if best is not None else {
        'bandwidth': float(bandwidths[0]), 'ridge': float(ridge_grid[0]),
        'cv_accuracy': None}

    cv_seconds = time.time() - cv_started
    if not clock.allows(0.0):
        raise _OutOfTime(f'krr has no budget left for the final fit '
                         f'({clock.remaining():.2f}s remain)')

    def build(a, b, same):
        return _kernel_matrix(a, b, kernel, chosen['bandwidth'], depth, bias, same=same)

    # Measured, not guessed: a single-cell config never reaches the sweep above,
    # so ``cell_seconds`` can be None, and no cell of any size costs the
    # ``n_query x n`` inference kernel (finding: 25 s of inference against a 5 s
    # margin at N=10000).
    probe = _probe_fit_cost(build, train, query.shape[0])
    estimate = _fit_cost_estimate(n, len(cv_index), int(config['cv_folds']), cell_seconds,
                                  query.shape[0], probe=probe)
    if not clock.allows(estimate):
        raise _OutOfTime(f'krr needs about {estimate:.2f}s for the final fit and '
                         f'inference, {clock.remaining():.2f}s remain')
    report['final_fit_estimate_seconds'] = estimate
    report['fit_cost_probe'] = probe

    training_started = time.time()
    k_train = build(train, train, True)
    lam = _ridge_absolute(chosen['ridge'], n, float(torch.diagonal(k_train).sum()),
                          config['ridge_mode'])
    alpha = _solve_ridge(k_train, targets, lam, inplace=True)
    del k_train
    _sync(device)
    training_seconds = time.time() - training_started

    inference_started = time.time()
    scores = build(query, train, False) @ alpha
    _sync(device)
    inference_seconds = time.time() - inference_started

    report.update({
        'kernel': kernel, 'depth': depth, 'arccos_bias': bias,
        'chosen': {'bandwidth': float(chosen['bandwidth']), 'ridge': float(chosen['ridge']),
                   'ridge_absolute': float(lam), 'cv_accuracy': chosen['cv_accuracy'],
                   'kernel': kernel, 'depth': depth},
        'cv_table': table, 'cv_rows': int(len(cv_index)), 'cv_folds': int(config['cv_folds']),
        'cv_seconds_per_bandwidth': cell_seconds, 'cv_seconds': cv_seconds,
        'training_seconds': training_seconds, 'inference_seconds': inference_seconds,
        **scale_note})
    return scores, alpha


# --------------------------------------------------------------------------
# family 'rfm': Recursive Feature Machine (Radhakrishnan et al., Science 2024)
# --------------------------------------------------------------------------
def laplace_agop(samples, centers, weights, bandwidth, metric, diagonal=False,
                 centering=False, chunk=2048, floor=0.0):
    """AGOP of the Laplace-kernel predictor, in the reference's arithmetic.

    ``f_c(x) = sum_j weights[j,c] exp(-d_M(x, centers[j]) / L)`` has Jacobian

        ``J(x)[c, :] = (1/L) sum_j weights[j,c] (K/d_M)[x,j] M (centers[j] - x)``

    (the ``j`` with ``d_M = 0`` are dropped, exactly as the reference drops the
    ``inf`` after dividing by the distance).  The return value is
    ``sum_x J(x)' J(x)`` -- the parent normalises it, as in ``fit_M``.
    """
    n, d = samples.shape
    p, c = weights.shape
    factor = _metric_factor(metric, diagonal, floor=floor)
    centers_r = _apply_metric(centers, factor, diagonal)
    metric_centers = _apply_metric(centers_r, factor, diagonal)          # centers @ M
    coefficients = (weights.view(p, c, 1) * metric_centers.view(p, 1, d)).reshape(p, c * d)
    out = torch.zeros(d if diagonal else (d, d), dtype=samples.dtype, device=samples.device)
    for start in range(0, n, int(chunk)):
        stop = min(start + int(chunk), n)
        block = samples[start:stop]
        block_r = _apply_metric(block, factor, diagonal)
        distance = _dist(block_r, centers_r)
        kernel = torch.exp(distance * (-1.0 / float(bandwidth)))
        kernel = torch.where(distance < 1e-10, torch.zeros_like(kernel),
                             kernel / distance.clamp(min=1e-10))
        samples_term = (kernel @ weights).unsqueeze(2)                   # (b, c, 1)
        centers_term = (kernel @ coefficients).view(stop - start, c, d)
        metric_block = _apply_metric(block_r, factor, diagonal)          # block @ M
        gradient = (centers_term - samples_term * metric_block.unsqueeze(1)) / float(bandwidth)
        if centering:
            gradient = gradient - gradient.mean(0)
        if diagonal:
            out += torch.einsum('ncd,ncd->d', gradient, gradient)
        else:
            out += torch.einsum('ncd,ncD->dD', gradient, gradient)
        del kernel, distance, gradient, centers_term, samples_term
    if not diagonal:
        out = 0.5 * (out + out.T)
    return out


def _normalise_agop(agop, mode='max_eigenvalue', diagonal=False):
    """Scale the AGOP to O(1).

    ``'max_entry'`` is the reference ``fit_M`` rule, ``M / (M.max() + 1e-30)``.
    It is basis dependent, so under the release's secret Haar rotation it makes
    the effective bandwidth a function of the rotation; ``'max_eigenvalue'``
    (the default) and ``'trace'`` (mean eigenvalue one) are invariant and agree
    with the reference rule whenever ``M`` is diagonal in the working basis.
    """
    if diagonal:
        # the entries ARE the eigenvalues, so 'max_entry' and 'max_eigenvalue'
        # coincide -- but 'trace' does not, and used to be silently ignored here
        if mode == 'trace':
            return agop * (agop.numel() / (agop.sum() + _TINY))
        return agop / (agop.max() + _TINY)
    if mode == 'max_entry':
        return agop / (agop.max() + _TINY)
    if mode == 'trace':
        return agop * (agop.shape[0] / (torch.diagonal(agop).sum() + _TINY))
    return agop / (torch.linalg.eigvalsh(agop).max() + _TINY)


def _rfm_kernel(a, b, bandwidth, factor, diagonal, same=False):
    ar = _apply_metric(a, factor, diagonal)
    br = ar if same else _apply_metric(b, factor, diagonal)
    return torch.exp(_dist(ar, br, same=same).mul_(-1.0 / float(bandwidth)))


def _rfm_run(train, targets, bandwidth, ridge, ridge_mode, iters, agop_power, diagonal,
             config, evaluate=None, clock=None):
    """One RFM run; ``evaluate(factor, alpha, bandwidth)`` is called after every iteration.

    Returns ``(metric, factor, alpha, bandwidth, history)``.  ``history[t]`` is
    whatever ``evaluate`` returned after ``t`` AGOP updates (``t = 0`` is the
    plain Laplace kernel), so the iteration count can be selected the way the
    reference's ``update_best_params`` selects it.
    """
    n, d = train.shape
    metric = (torch.ones(d, dtype=train.dtype, device=train.device) if diagonal
              else torch.eye(d, dtype=train.dtype, device=train.device))
    effective = metric                      # the matrix the kernel actually uses
    floor = float(config['metric_floor'])
    factor = _metric_factor(effective, diagonal, floor=floor)
    history, alpha, used = [], None, float(bandwidth)
    for step in range(int(iters) + 1):
        if step:
            # The AGOP is taken of the predictor that was just solved, i.e. with
            # the metric that kernel used (``effective``), not the raw AGOP.
            agop = laplace_agop(train, train, alpha, used, effective, diagonal=diagonal,
                                centering=bool(config['centering']),
                                chunk=int(config['agop_chunk']), floor=floor)
            metric = _normalise_agop(agop, config['agop_normalisation'], diagonal)
            effective = _matrix_power(metric, float(agop_power), floor=floor,
                                      diagonal=diagonal)
            factor = _metric_factor(effective, diagonal, floor=floor)
            if config['bandwidth_mode'] == 'adaptive_median':
                scale = _median_distance(_apply_metric(train, factor, diagonal),
                                         int(config.get('median_sample', 2000) or 2000), 0)
                used = float(bandwidth) * scale / max(_median_distance(train, 2000, 0), _TINY)
        kernel = _rfm_kernel(train, train, used, factor, diagonal, same=True)
        lam = _ridge_absolute(ridge, n, float(torch.diagonal(kernel).sum()), ridge_mode)
        alpha = _solve_ridge(kernel, targets, lam, inplace=True)
        del kernel
        if evaluate is not None:
            history.append(evaluate(factor, alpha, used, step))
        if clock is not None and not clock.allows(0.0):
            break
    return metric, factor, alpha, used, history


def _rfm_predict(query, train, alpha, bandwidth, factor, diagonal):
    return _rfm_kernel(query, train, bandwidth, factor, diagonal) @ alpha


def _fit_rfm(train, train_y, query, config, seed, clock, report):
    """Select (L, ridge, power, diag, T) on one held-out split, then refit on all rows."""
    n, device, dtype = train.shape[0], train.device, train.dtype
    targets = _targets(train_y, NUM_CLASSES, config['target_mode'], dtype)
    bandwidths, scale_note = _krr_bandwidths({**config, 'kernel': 'laplace',
                                              'median_sample': 2000}, train, seed)
    ridge_grid = [float(r) for r in _grid(config, 'ridge')]
    powers = [float(p) for p in _grid(config, 'agop_power')]
    diagonals = [bool(v) for v in _grid(config, 'diag')]
    iters = int(config['iters'])
    combinations = [(b, r, p, g) for b in bandwidths for r in ridge_grid
                    for p in powers for g in diagonals]
    sweep = len(combinations) > 1 or (bool(config['select_iteration']) and iters > 0)

    table, chosen, holdout_note = [], None, {}
    cv_started = time.time()
    if sweep and clock.allows(0.0):
        rows = _cv_rows(n, config['holdout_max_rows'], seed)
        order = np.random.default_rng([int(seed), 0xA607]).permutation(len(rows))
        cut = max(1, int(round(len(rows) * (1.0 - float(config['holdout_fraction'])))))
        cut = min(cut, len(rows) - 1)
        fit_rows = torch.as_tensor(np.sort(rows[order[:cut]]), device=device, dtype=torch.long)
        val_rows = torch.as_tensor(np.sort(rows[order[cut:]]), device=device, dtype=torch.long)
        fit_x, fit_t = train.index_select(0, fit_rows), targets.index_select(0, fit_rows)
        val_x = train.index_select(0, val_rows)
        val_y = train_y.index_select(0, val_rows)
        holdout_note = {'holdout_fit_rows': int(fit_rows.numel()),
                        'holdout_val_rows': int(val_rows.numel())}
        run_seconds = None
        for bandwidth, ridge, power, diagonal in combinations:
            budget = (run_seconds or 0.0) * 1.2
            if run_seconds is not None and not clock.allows(budget):
                report['deadline_truncated_cv'] = True
                break
            started = time.time()

            def evaluate(factor, alpha, used, step, _b=bandwidth, _r=ridge, _p=power,
                         _g=diagonal):
                prediction = _rfm_predict(val_x, fit_x, alpha, used, factor, _g)
                accuracy = float((prediction.argmax(1) == val_y).sum()) / val_y.numel()
                table.append({'bandwidth': float(_b), 'ridge': float(_r),
                              'agop_power': float(_p), 'diag': bool(_g), 'iters': int(step),
                              'bandwidth_used': float(used), 'holdout_accuracy': accuracy})
                return accuracy

            _rfm_run(fit_x, fit_t, bandwidth, ridge, config['ridge_mode'], iters, power,
                     diagonal, config, evaluate=evaluate, clock=clock)
            _sync(device)
            run_seconds = max(run_seconds or 0.0, time.time() - started)
        if table:
            best = max(range(len(table)), key=lambda i: (table[i]['holdout_accuracy'],
                                                         -table[i]['iters']))
            chosen = dict(table[best])
        holdout_note['holdout_seconds_per_combination'] = run_seconds
    holdout_note['cv_seconds'] = time.time() - cv_started
    if chosen is None:
        chosen = {'bandwidth': float(bandwidths[0]), 'ridge': float(ridge_grid[0]),
                  'agop_power': float(powers[0]), 'diag': bool(diagonals[0]),
                  'iters': iters, 'holdout_accuracy': None}

    per_iteration = holdout_note.get('holdout_seconds_per_combination')
    if not clock.allows(0.0):
        raise _OutOfTime(f'rfm has no budget left for the final fit '
                         f'({clock.remaining():.2f}s remain)')
    estimate = 0.0
    if per_iteration:
        scale = (float(n) / max(1.0, float(holdout_note['holdout_fit_rows']))) ** 3
        estimate = 1.5 * per_iteration * scale * (chosen['iters'] + 1) / (iters + 1) + 0.1

    def probe_build(a, b, same):
        return _kernel_matrix(a, b, 'laplace', chosen['bandwidth'], same=same)

    # One measured build+Cholesky; the run does ``iters + 1`` of them plus one
    # AGOP pass each (charged as one more build+solve), and then inference.
    probe = _probe_fit_cost(probe_build, train, query.shape[0])
    per_round, probe_inference = _probe_terms(probe, n, query.shape[0])
    estimate = max(estimate, 1.5 * (2.0 * (int(chosen['iters']) + 1) * per_round
                                    + probe_inference) + 0.1)
    if not clock.allows(estimate):
        raise _OutOfTime(f'rfm needs about {estimate:.2f}s for the final fit and '
                         f'inference, {clock.remaining():.2f}s remain')

    training_started = time.time()
    metric, factor, alpha, used, _ = _rfm_run(
        train, targets, chosen['bandwidth'], chosen['ridge'], config['ridge_mode'],
        int(chosen['iters']), chosen['agop_power'], bool(chosen['diag']), config, clock=clock)
    _sync(device)
    training_seconds = time.time() - training_started

    inference_started = time.time()
    scores = _rfm_predict(query, train, alpha, used, factor, bool(chosen['diag']))
    _sync(device)
    inference_seconds = time.time() - inference_started

    spectrum = (metric if bool(chosen['diag'])
                else torch.linalg.eigvalsh(metric)).clamp(min=0.0)
    spectrum = torch.sort(spectrum, descending=True).values
    report.update({
        'chosen': {k: chosen[k] for k in ('bandwidth', 'ridge', 'agop_power', 'diag',
                                          'iters', 'holdout_accuracy')},
        'bandwidth_used': float(used),
        'cv_table': table, 'final_fit_estimate_seconds': float(estimate),
        'fit_cost_probe': probe,
        'metric_effective_rank': float((spectrum.sum() ** 2)
                                      / (spectrum ** 2).sum().clamp(min=_TINY)),
        'metric_top_eigenvalues': [float(v) for v in spectrum[:10]],
        'training_seconds': training_seconds, 'inference_seconds': inference_seconds,
        **scale_note, **holdout_note})
    return scores, (metric, factor, alpha, used, bool(chosen['diag']))


# --------------------------------------------------------------------------
# family 'labelprop': label spreading (Zhou et al., NIPS 2004)
# --------------------------------------------------------------------------
def knn_neighbours(points, k, chunk=1024):
    """``(indices, distances)`` of the ``k`` nearest neighbours (self excluded).

    Both are ``(n, k)`` and sorted by increasing distance, so a graph for any
    ``k' <= k`` is a column slice: the grid over ``k`` costs one distance pass.
    """
    n = points.shape[0]
    k = int(min(int(k), max(1, n - 1)))
    indices, distances = [], []
    for start in range(0, n, int(chunk)):
        stop = min(start + int(chunk), n)
        block = _sqdist(points[start:stop], points)
        block[torch.arange(stop - start, device=points.device),
              torch.arange(start, stop, device=points.device)] = float('inf')
        nearest = torch.topk(block, k, dim=1, largest=False, sorted=True)
        indices.append(nearest.indices)
        distances.append(nearest.values.clamp(min=0.0).sqrt())
        del block, nearest
    return torch.cat(indices), torch.cat(distances)


def knn_graph(indices, distances, k, weight='binary', heat_scale=1.0):
    """Symmetric k-NN graph ``0.5 (A + A')`` as a sparse COO tensor.

    ``A[i, j]`` is set for the ``k`` nearest neighbours ``j != i`` of ``i``; the
    same convention as ``sklearn.semi_supervised.LabelSpreading(kernel='knn')``,
    whose ``_build_graph`` also symmetrises with ``0.5 (W + W')``.
    """
    n = indices.shape[0]
    k = int(min(int(k), indices.shape[1]))
    columns = indices[:, :k].reshape(-1)
    near = distances[:, :k]
    mean_distance = float(near.mean()) if near.numel() else 0.0
    rows = torch.arange(n, device=indices.device).repeat_interleave(k)
    if weight == 'heat':
        sigma = max(float(heat_scale) * mean_distance, 1e-12)
        entries = torch.exp(-(near.reshape(-1) ** 2) / (2.0 * sigma ** 2))
    else:
        entries = torch.ones(n * k, dtype=distances.dtype, device=indices.device)
    index = torch.stack([torch.cat([rows, columns]), torch.cat([columns, rows])])
    graph = torch.sparse_coo_tensor(index, torch.cat([entries, entries]) * 0.5,
                                    (n, n)).coalesce()
    return graph, mean_distance


def normalised_adjacency(graph):
    """``S = D^-1/2 W D^-1/2`` for a coalesced sparse symmetric ``W``."""
    n = graph.shape[0]
    degree = torch.sparse.sum(graph, dim=1).to_dense()
    inverse = torch.where(degree > 0, degree.clamp(min=_TINY).rsqrt(),
                          torch.zeros_like(degree))
    index, values = graph.indices(), graph.values()
    scaled = values * inverse[index[0]] * inverse[index[1]]
    return torch.sparse_coo_tensor(index, scaled, (n, n)).coalesce(), degree


def spread(adjacency, initial, alpha, iterations):
    """``F <- alpha S F + (1 - alpha) Y``; the Zhou et al. iteration."""
    f = initial.clone()
    for _ in range(int(iterations)):
        f = float(alpha) * torch.sparse.mm(adjacency, f) + (1.0 - float(alpha)) * initial
    return f


def _fit_labelprop(train, train_y, query, config, seed, clock, report):
    """Spread labels over a train+query graph; (k, alpha) by CV on the labelled rows."""
    n, q, device, dtype = train.shape[0], query.shape[0], train.device, train.dtype
    metric_note = {'metric': config['metric']}
    points_train, points_query = train, query
    if config['metric'] == 'rfm':
        inner = dict(config['rfm_config'])
        targets = _targets(train_y, NUM_CLASSES, inner['target_mode'], dtype)
        inner_bandwidth = float(_grid(inner, 'bandwidth')[0])

        def probe_build(a, b, same):
            return _kernel_matrix(a, b, 'laplace', inner_bandwidth, same=same)

        # The inner run is ``iters + 1`` Choleskys of n rows plus an AGOP pass
        # each; it used to start with no estimate at all.
        probe = _probe_fit_cost(probe_build, train, 1)
        per_round, _ = _probe_terms(probe, n, 1)
        rfm_estimate = 1.5 * 2.0 * (int(inner['iters']) + 1) * per_round + 0.1
        if not clock.allows(rfm_estimate):
            raise _OutOfTime(f"labelprop's inner rfm needs about {rfm_estimate:.2f}s, "
                             f'{clock.remaining():.2f}s remain')
        metric_note.update({'rfm_estimate_seconds': rfm_estimate,
                            'rfm_fit_cost_probe': probe})
        started = time.time()
        _, factor, _, _, _ = _rfm_run(
            train, targets, _grid(inner, 'bandwidth')[0], _grid(inner, 'ridge')[0],
            inner['ridge_mode'], int(inner['iters']), _grid(inner, 'agop_power')[0],
            bool(_grid(inner, 'diag')[0]), inner, clock=clock)
        diagonal = bool(_grid(inner, 'diag')[0])
        points_train = _apply_metric(train, factor, diagonal)
        points_query = _apply_metric(query, factor, diagonal)
        metric_note.update({'rfm_seconds': time.time() - started,
                            'rfm_iters': int(inner['iters']),
                            'rfm_bandwidth': float(_grid(inner, 'bandwidth')[0]),
                            'rfm_diag': diagonal})

    points = torch.cat([points_train, points_query], dim=0)
    labels_onehot = torch.zeros((n + q, NUM_CLASSES), dtype=dtype, device=device)
    labels_onehot[torch.arange(n, device=device), train_y.long()] = 1.0

    k_grid = sorted({int(v) for v in _grid(config, 'k')})
    alpha_grid = [float(v) for v in _grid(config, 'alpha')]
    folds = _fold_indices(n, int(config['cv_folds']), seed)
    cv_nodes = n + q if config['cv_graph'] == 'full' else n
    cv_points = points if config['cv_graph'] == 'full' else points_train

    cv_started = time.time()
    table, chosen, graph_seconds = [], None, {}
    if len(k_grid) * len(alpha_grid) > 1 and clock.allows(0.0):
        cv_indices, cv_distances = knn_neighbours(cv_points, max(k_grid),
                                                  int(config['knn_chunk']))
        for k in k_grid:
            started = time.time()
            graph, _ = knn_graph(cv_indices, cv_distances, k, config['weight'],
                                 float(config['heat_scale']))
            adjacency, _ = normalised_adjacency(graph)
            del graph
            graph_seconds[str(k)] = time.time() - started
            correct = {alpha: 0 for alpha in alpha_grid}
            for validation in folds:
                masked = labels_onehot[:cv_nodes].clone()
                index = torch.as_tensor(validation, device=device, dtype=torch.long)
                masked[index] = 0.0
                for alpha in alpha_grid:
                    scores = spread(adjacency, masked, alpha, int(config['spread_iters']))
                    correct[alpha] += int((scores.index_select(0, index).argmax(1)
                                           == train_y.index_select(0, index)).sum())
            for alpha in alpha_grid:
                row = {'k': int(k), 'alpha': float(alpha),
                       'cv_accuracy': float(correct[alpha]) / float(n)}
                table.append(row)
                if chosen is None or row['cv_accuracy'] > chosen['cv_accuracy']:
                    chosen = row
            del adjacency
            if not clock.allows(0.0):
                report['deadline_truncated_cv'] = True
                break
    if chosen is None:
        chosen = {'k': int(k_grid[0]), 'alpha': float(alpha_grid[0]), 'cv_accuracy': None}

    cv_seconds = time.time() - cv_started
    training_started = time.time()
    all_indices, all_distances = knn_neighbours(points, chosen['k'], int(config['knn_chunk']))
    graph, mean_distance = knn_graph(all_indices, all_distances, chosen['k'],
                                     config['weight'], float(config['heat_scale']))
    adjacency, degree = normalised_adjacency(graph)
    del graph, all_indices, all_distances
    _sync(device)
    training_seconds = time.time() - training_started

    inference_started = time.time()
    scores = spread(adjacency, labels_onehot, chosen['alpha'],
                    int(config['spread_iters']))[n:]
    unreached = torch.nonzero(scores.abs().sum(1) <= 1e-12, as_tuple=False).squeeze(1)
    if unreached.numel():
        if config['fallback'] == 'nearest_labeled':
            nearest = _sqdist(points_query.index_select(0, unreached), points_train).argmin(1)
            scores[unreached] = labels_onehot[:n].index_select(0, nearest) * 1e-6
        else:
            prior = torch.bincount(train_y.long(), minlength=NUM_CLASSES).to(dtype)
            scores[unreached] = prior * 1e-9 / max(1.0, float(prior.sum()))
    _sync(device)
    inference_seconds = time.time() - inference_started

    report.update({
        'chosen': {'k': int(chosen['k']), 'alpha': float(chosen['alpha']),
                   'cv_accuracy': chosen['cv_accuracy'], 'weight': config['weight'],
                   'spread_iters': int(config['spread_iters'])},
        'cv_table': table, 'cv_graph': config['cv_graph'], 'cv_folds': int(config['cv_folds']),
        'cv_seconds': cv_seconds,
        'graph_seconds': graph_seconds, 'mean_knn_distance': float(mean_distance),
        'graph_nodes': int(n + q), 'isolated_nodes': int((degree <= 0).sum()),
        'unreached_query_rows': int(unreached.numel()),
        'training_seconds': training_seconds, 'inference_seconds': inference_seconds,
        **metric_note})
    return scores


# --------------------------------------------------------------------------
# family 'selftrain': pseudo-label the confident query rows and refit
# --------------------------------------------------------------------------
def _margin(scores):
    top = torch.topk(scores, 2, dim=1).values
    return top[:, 0] - top[:, 1]


def _fit_selftrain(train, train_y, query, config, seed, clock, report):
    """Fit the base learner, then refit on train + confident pseudo-labelled query rows.

    The base runs its OWN ``rewhiten`` / ``metric_learn`` preprocessing (on top of
    whatever the wrapper's own settings already did), so a sphered base is really
    sphered; the pseudo-labelled rows then live in the base's feature space.
    """
    base = dict(config['base'])
    base_report = {}
    train, query, base_preprocess = _preprocess(train, query, train_y, base)
    report['base_preprocess'] = base_preprocess
    fitter = _fit_krr if base['family'] == 'krr' else _fit_rfm
    scores, _ = fitter(train, train_y, query, base, seed, clock, base_report)
    rounds_log = [{'round': 0, 'pseudo_rows': 0,
                   'training_seconds': base_report.get('training_seconds'),
                   'inference_seconds': base_report.get('inference_seconds')}]

    # Rounds 1..R reuse the hyperparameters the base selected, so no CV is repeated.
    if config['reuse_hyperparameters']:
        chosen = base_report.get('chosen', {})
        if base['family'] == 'krr':
            base = {**base, 'bandwidth': float(chosen['bandwidth']),
                    'ridge': float(chosen['ridge']), 'bandwidth_grid': None,
                    'bandwidth_scales': None, 'ridge_grid': None}
        else:
            base = {**base, 'bandwidth': float(chosen['bandwidth']),
                    'ridge': float(chosen['ridge']), 'agop_power': float(chosen['agop_power']),
                    'diag': bool(chosen['diag']), 'iters': int(chosen['iters']),
                    'bandwidth_grid': None, 'bandwidth_scales': None, 'ridge_grid': None,
                    'agop_power_grid': None, 'diag_grid': None, 'select_iteration': False}
    quantile = float(config['quantile'])
    weight = float(config['pseudo_weight'])
    for round_index in range(1, int(config['rounds']) + 1):
        margin = _margin(scores)
        threshold = (torch.quantile(margin.to(torch.float64), quantile)
                     if quantile > 0 else None)
        keep = torch.nonzero(margin >= threshold if threshold is not None
                             else torch.ones_like(margin, dtype=torch.bool),
                             as_tuple=False).squeeze(1)
        if keep.numel() == 0:
            break
        pseudo_y = scores.argmax(1).index_select(0, keep).to(train_y.dtype)
        pseudo_x = query.index_select(0, keep)
        if int(weight) > 1:                      # integer weights = repeated rows
            pseudo_x = pseudo_x.repeat(int(weight), 1)
            pseudo_y = pseudo_y.repeat(int(weight))
        augmented_x = torch.cat([train, pseudo_x], dim=0)
        augmented_y = torch.cat([train_y, pseudo_y], dim=0)
        round_report = {}
        estimate = base_report.get('training_seconds', 0.0) or 0.0
        scale = (float(augmented_x.shape[0]) / max(1.0, float(train.shape[0]))) ** 3
        # The refit pays its own inference again, which the round budget used to
        # ignore; it is the same query block, so the base's measurement carries.
        inference = ((base_report.get('inference_seconds', 0.0) or 0.0)
                     * float(augmented_x.shape[0]) / max(1.0, float(train.shape[0])))
        if not clock.allows(1.2 * (estimate * scale + inference) + 0.1):
            report['deadline_truncated_rounds'] = round_index
            break
        scores, _ = fitter(augmented_x, augmented_y, query, base, seed + round_index,
                           clock, round_report)
        rounds_log.append({'round': round_index, 'pseudo_rows': int(keep.numel()),
                           'pseudo_weight': int(weight),
                           'margin_threshold': (float(threshold)
                                                if threshold is not None else None),
                           'training_seconds': round_report.get('training_seconds'),
                           'inference_seconds': round_report.get('inference_seconds'),
                           'chosen': round_report.get('chosen')})
    report.update({
        'chosen': base_report.get('chosen'),
        'cv_table': base_report.get('cv_table', []),
        'base_family': config['base']['family'], 'rounds_log': rounds_log,
        'quantile': quantile, 'reuse_hyperparameters': bool(config['reuse_hyperparameters']),
        'training_seconds': float(sum(r['training_seconds'] or 0.0 for r in rounds_log)),
        # every round re-scores the full query block, so the wrapper's inference
        # cost is the SUM over rounds, not the last one (which understated the
        # margin a reviewer would size from it by ~2.6x at N=10000)
        'inference_seconds': float(sum(r['inference_seconds'] or 0.0 for r in rounds_log)),
        'inference_seconds_last_round': float(rounds_log[-1]['inference_seconds'] or 0.0),
        'cv_seconds': float(base_report.get('cv_seconds') or 0.0),
        'base_report': {k: v for k, v in base_report.items() if k != 'cv_table'}})
    return scores


# --------------------------------------------------------------------------
# family 'ensemble': average the standardised scores of several members
# --------------------------------------------------------------------------
def _standardise_scores(scores, mode):
    """Row-centre, then divide by the matrix's own scale, so members are comparable."""
    centred = scores - scores.mean(1, keepdim=True)
    if mode == 'none':
        return centred
    scale = centred.std()
    return centred / scale.clamp(min=_TINY)


def _fit_ensemble(train, train_y, query, config, seed, clock, report):
    """Fit each member on its own preprocessing and average the standardised scores.

    Members are fitted in order and share one deadline; a member that runs out of
    time is dropped (``drop_failed_members``) rather than sinking the ensemble.
    Each member re-runs ``_preprocess`` with its own ``rewhiten`` / ``metric_learn``
    settings, on top of whatever the ensemble's own settings already did.

    Every member is fitted with the ensemble's own ``seed`` -- not ``seed + index``.
    The seed drives ``_cv_rows`` and ``_fold_indices``, so an index-dependent seed
    would make each member's hyperparameter choice depend on where it sits in the
    list: the ensemble would not be invariant to the member order and would not
    reproduce the offline average of the same members run standalone.
    """
    members = config['members']
    weights = ([1.0] * len(members) if config['weights'] is None
               else [float(w) for w in config['weights']])
    total = torch.zeros((query.shape[0], NUM_CLASSES), dtype=train.dtype,
                        device=train.device)
    used, log = 0.0, []
    training_seconds = inference_seconds = cv_seconds = 0.0
    for index, member in enumerate(members):
        member_report = {}
        started = time.time()
        try:
            member_train, member_query, note = _preprocess(train, query, train_y, member)
            scores = _dispatch(member['family'], member_train, train_y, member_query,
                               member, seed, clock, member_report)
        except _OutOfTime as exc:
            if not config['drop_failed_members']:
                raise
            log.append({'member': index, 'family': member['family'], 'skipped': str(exc)})
            continue
        total += weights[index] * _standardise_scores(scores, config['standardise'])
        used += weights[index]
        training_seconds += float(member_report.get('training_seconds') or 0.0)
        inference_seconds += float(member_report.get('inference_seconds') or 0.0)
        cv_seconds += float(member_report.get('cv_seconds') or 0.0)
        log.append({'member': index, 'family': member['family'],
                    'weight': weights[index], 'seconds': time.time() - started,
                    'chosen': member_report.get('chosen'),
                    'uses_query': _uses_query(member),
                    'preprocess': note})
    if used <= 0.0:
        raise _OutOfTime('no ensemble member finished inside the deadline')
    report.update({'members_log': log, 'members_used': int(sum(1 for r in log
                                                               if 'skipped' not in r)),
                   'members_total': len(members), 'weight_used': used,
                   'standardise': config['standardise'],
                   'chosen': [r.get('chosen') for r in log], 'cv_table': [],
                   'training_seconds': training_seconds,
                   'inference_seconds': inference_seconds,
                   'cv_seconds': cv_seconds})
    return total / used


# --------------------------------------------------------------------------
# public entry point
# --------------------------------------------------------------------------
TRANSDUCTIVE_FAMILIES = ('labelprop', 'selftrain')


def _dispatch(family, train, train_y, query, config, seed, clock, report):
    if family == 'krr':
        return _fit_krr(train, train_y, query, config, seed, clock, report)[0]
    if family == 'rfm':
        return _fit_rfm(train, train_y, query, config, seed, clock, report)[0]
    if family == 'labelprop':
        return _fit_labelprop(train, train_y, query, config, seed, clock, report)
    if family == 'selftrain':
        return _fit_selftrain(train, train_y, query, config, seed, clock, report)
    if family == 'ensemble':
        return _fit_ensemble(train, train_y, query, config, seed, clock, report)
    raise ValueError(f'unknown family {family!r}')                    # pragma: no cover


def _uses_query(config):
    """True when the candidate reads the released query features (never labels).

    The cross-cutting keys are checked FIRST: ``prior_match`` may only live on an
    ensemble (``resolve_config`` rejects it on a member), so a wrapper-level test
    that dispatched on the family before looking at them would file the sanctioned
    prior-matched ensemble as inductive.
    """
    if config['rewhiten'] != 'none' or config['prior_match'] != 'none':
        return True
    if config['family'] == 'ensemble':
        return any(_uses_query(member) for member in config['members'])
    if config['family'] == 'selftrain':
        return True                        # in TRANSDUCTIVE_FAMILIES; base is moot
    return config['family'] in TRANSDUCTIVE_FAMILIES


def _rewhitens_anywhere(config):
    """True when the candidate or anything nested inside it re-whitens on the query."""
    if config['rewhiten'] != 'none':
        return True
    if config['family'] == 'ensemble':
        return any(_rewhitens_anywhere(member) for member in config['members'])
    if config['family'] == 'selftrain':
        return _rewhitens_anywhere(config['base'])
    return False


def fit_predict(train_z, train_y, query_z, config, seed, device, deadline_unix):
    """Train one candidate on a release draw and predict the query rows.

    Never sees query labels.  ``train_z`` / ``query_z`` are the released features
    ``z = Q W (x - mu)``; any release dimension is accepted as long as the two
    halves agree.  The returned ``labels`` are the first-max argmax of ``logits``
    (which for the kernel families are regression scores, not calibrated logits).
    """
    started = time.time()
    if isinstance(device, (int, float)) and isinstance(deadline_unix, str):
        # The sibling pmnist learners take (..., deadline_unix, device); accept
        # that order too rather than mis-parsing a timestamp as a device.
        device, deadline_unix = deadline_unix, device
        swapped = True
    else:
        swapped = False
    resolved = resolve_config(config)
    train_z = np.ascontiguousarray(train_z, dtype=np.float32)
    query_z = np.ascontiguousarray(query_z, dtype=np.float32)
    train_y = np.ascontiguousarray(train_y).astype(np.uint8, copy=False)
    if train_z.ndim != 2 or query_z.ndim != 2:
        raise ValueError('train_z and query_z must be 2-D (N, D) arrays')
    if train_z.shape[1] != query_z.shape[1]:
        raise ValueError('train_z and query_z must share the release dimension')
    if train_y.shape != (train_z.shape[0],):
        raise ValueError('train_y must be (N,)')
    if int(train_y.max(initial=0)) >= NUM_CLASSES:
        raise ValueError('train_y must hold class indices in 0..9')
    if not (np.isfinite(train_z).all() and np.isfinite(query_z).all()):
        raise ValueError('release features must be finite')
    torch.manual_seed(int(seed))
    np.random.seed(int(seed) % (2 ** 31))

    dtype = DTYPES[resolved['dtype']]
    clock = Clock(deadline_unix, float(resolved['inference_margin_seconds']))
    train = _tensor(train_z, device, dtype)
    query = _tensor(query_z, device, dtype)
    labels_tensor = _tensor(train_y.astype(np.int64), device, torch.int64)
    report = {'argument_order_swapped': swapped}
    preprocess_started = time.time()
    train, query, preprocess_note = _preprocess(train, query, labels_tensor, resolved)
    report.update(preprocess_note)
    report['preprocess_seconds'] = time.time() - preprocess_started

    try:
        if not clock.allows(0.0):
            raise _OutOfTime(f'no budget left at entry ({clock.remaining():.2f}s)')
        scores = _dispatch(resolved['family'], train, labels_tensor, query, resolved,
                           int(seed), clock, report)
        report['emergency_fallback'] = False
    except _OutOfTime as exc:
        fallback_started = time.time()
        scores = _centroid_scores(train, labels_tensor, query)
        report.update({'emergency_fallback': True, 'emergency_reason': str(exc),
                       'training_seconds': 0.0,
                       'inference_seconds': time.time() - fallback_started})
        report.setdefault('chosen', None)
        report.setdefault('cv_table', [])

    report['preprocess_effective'] = {
        'top': preprocess_note,
        'base': report.get('base_preprocess'),
        'members': [entry.get('preprocess') for entry in report.get('members_log', [])]}
    report['rewhiten_anywhere'] = bool(_rewhitens_anywhere(resolved))

    if resolved['prior_match'] == 'sinkhorn':
        matched_started = time.time()
        prior = torch.bincount(labels_tensor, minlength=NUM_CLASSES).to(scores.dtype) + 1.0
        temperature = (_auto_temperature(scores)
                       if resolved['prior_match_temperature'] == 'auto'
                       else float(resolved['prior_match_temperature']))
        before = torch.bincount(scores.argmax(1), minlength=NUM_CLASSES)
        scores, bias = sinkhorn_prior_match(scores, prior, temperature,
                                            int(resolved['prior_match_iters']))
        report.update({
            'prior_match': 'sinkhorn', 'prior_match_temperature': float(temperature),
            'prior_match_bias': [float(v) for v in bias],
            'prior_match_seconds': time.time() - matched_started,
            'query_class_counts_before': [int(v) for v in before],
            'query_class_counts_after': [int(v) for v in
                                         torch.bincount(scores.argmax(1),
                                                        minlength=NUM_CLASSES)]})

    logits = np.ascontiguousarray(scores.detach().to(torch.float32).cpu().numpy(),
                                  dtype=np.float32)
    if logits.shape != (query_z.shape[0], NUM_CLASSES) or not np.isfinite(logits).all():
        raise RuntimeError('Learner produced malformed logits')
    uses_query = _uses_query(resolved)
    metrics = {
        'training_seconds': 0.0, 'inference_seconds': 0.0, **report,
        'uses_query_images_unlabeled': bool(uses_query),
        'uses_query_features_unlabeled': bool(uses_query),
        'fit_wall_seconds': time.time() - started,
        'family': resolved['family'], 'config': resolved, 'seed': int(seed),
        'device': str(device), 'device_name': _device_name(device),
        'dtype': resolved['dtype'], 'deadline_unix': float(deadline_unix),
        'deadline_slack_seconds': float(deadline_unix) - time.time(),
        'train_count': int(train_z.shape[0]), 'query_count': int(query_z.shape[0]),
        'release_dim': int(train_z.shape[1]), 'query_labels_supplied': False,
        'torch': str(torch.__version__), 'numpy': str(np.__version__),
    }
    return {'logits': logits, 'labels': first_max_labels(logits),
            'metrics': _jsonable(metrics)}
