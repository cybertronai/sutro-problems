"""Neural learners for the *released* MNIST-medium task (harness 1.2.0).

What the learner sees
---------------------
A draw hands over ``train_z`` (N, 60) float32, ``train_y`` (N,) and
``query_z`` (10000, 60) float32, all produced by the same secret map
``z = Q W (x - mu)`` (``gpumode/eval.py``: ``release_map`` / ``apply_release``).
``W`` is *exact* PCA whitening onto the top 60 principal directions of that
draw's N training rows and ``Q`` is a per-draw Haar rotation.  Two consequences
drive every default below:

1. The released training rows are exactly zero-mean and unit-variance *per
   released coordinate* (whitening, by construction), and the coordinates are
   uncorrelated on the training rows.  Query rows come from a disjoint
   universe, so their per-coordinate variance is close to but not exactly 1.
   There is therefore nothing left for a normalisation layer to do on the
   training rows, which is why this module adds ``normalization='none'`` and
   makes it the default.  ``'standardize'`` is kept (it is a no-op up to the
   unbiased/biased variance convention and can still help when N is small
   enough that the whitening fit is noisy).  The pixel study's ``'4x-0.5'``
   map is *not* offered: it is calibrated for [0,1] pixels and would put the
   release at mean -0.5 with std 4.
2. Spatial structure is gone by design: no convolution, no augmentation, no
   topology recovery (``pmnist_learners.topo_cnn`` is deliberately not
   re-exported).  What is left is dense nets plus regularisation, which is what
   the three families here provide.

Families
--------
  'mlp'     batched-K dense ensemble (dropout / Gaussian input noise / mixup /
            label smoothing / EMA / cosine schedule), adapted from
            ``pmnist_learners.BatchedMLP`` and ``pmnist_learners._fit_mlp``.
  'vat'     the same MLP plus virtual adversarial training (Miyato et al.,
            2018) with ONE shared dropout draw per VAT computation
            (``pmnist_learners.vat_loss``, imported unchanged), an ``eps``
            expressed in *release* units, a linear ramp-up of the VAT weight
            over the first ``rampup_fraction`` of the steps, and Gaussian input
            noise off by default.  The pixel study's VAT collapsed when the
            input noise and eps were mis-scaled relative to each other; the
            ramp-up additionally keeps the adversarial term from dominating
            before the classifier has any structure to smooth.
  'ladder'  the vendored fully supervised AMLP[2,2] Ladder
            (``ladder_model.LadderAMLP``, Pezeshki et al., ICML 2016) on 60-d
            inputs, with a new ``input_scale``.

Why 'ladder' needs ``input_scale``
----------------------------------
Two published Ladder hyperparameters are scale sensitive:

  * ``noise_std=0.3`` is additive Gaussian noise on the inputs, so the
    signal-to-noise ratio at the input layer is std(x)/noise_std;
  * ``input_reconstruction_weight=2000`` multiplies an input-space MSE, so its
    contribution scales with var(x).

On the 9x9 box-averaged pool the marginal pixel std is 0.2416 (measured over
all 60000x81 values; ``PIXEL_STD_9X9``), i.e. a published input SNR of 0.81,
not the 1.0 the 28x28 constant 0.3081 would give.  Unit-variance release
features raise that SNR to 3.3 and inflate the reconstruction penalty ~17x
relative to the cross entropy.  ``input_scale`` multiplies the release features
by a constant before they reach the Ladder.

The default is chosen by MEASUREMENT, not by restoring the pixel SNR -- the
pixel-matched value is the worst one tried.  Real dev draws, N=1057, hidden
[500,250,250], 60 epochs, decay_start 40, batch 100, unlabeled 'train+query',
10000 query rows (binomial SE ~0.3 pp), query error:

    input_scale  0.2416   0.30    0.45    0.60    0.80    1.00
    seed ..491   14.91%  12.41%     --   10.71%    --    12.71%
    seed ..492      --   12.64%  10.70%  10.38%  10.50%  11.39%

The curve is smooth and unimodal with the optimum near 0.6 (effective input
noise-to-signal ratio 0.5, i.e. input SNR 2), which is why ``input_scale=0.6``
is the default; the pixel-matched 0.24-0.30 costs ~2 points.  Re-sweep it if
the schedule or the hidden widths change.  The map is a scalar, applied
identically to train and query rows, and is recorded in the metrics.

``rewhiten``: the kernel study's preprocessing, for neural nets
---------------------------------------------------------------
``kernels.py`` found that what helps most on this release is not the model but
the metric.  The same two maps are offered here, for every family, through one
key ``rewhiten`` in {'none' (default), 'wc', 'rw', 'wc+rw'}:

  'wc'  within-class sphering ``S_w^-1/2`` fitted on the TRAINING rows and
        labels only (``kernels.within_class_transform``, imported, not copied),
        with ``wc_shrinkage`` (default 0.05, the value the pilots call ``wc``).
        Inductive: the same linear map is applied to train and query.
  'rw'  ZCA re-whitening with the released QUERY covariance
        (``kernels._rewhiten``, ``rw_source`` 'query' or 'pooled',
        ``rw_shrinkage`` default 0.0).  Transductive -- it reads query features
        (never labels) -- and flagged in the metrics.
  'wc+rw'  both, in kernels.py's order: re-whiten first, sphere the result
        (the reverse order would let the query ZCA undo the supervised metric).

The maps are computed in float64 and the features handed to the net are cast
back to float32.  For the ladder the map is applied BEFORE ``input_scale``.

``selftrain``: pseudo-label the confident query rows and refit
--------------------------------------------------------------
``selftrain`` = {'rounds': R (0 = off, the default), 'quantile': q, 'soft':
False, 'weight': 1.0, 'refit': 'scratch'|'continue'} works for all three
families.  Each round predicts the query rows with the current model, keeps the
confident ones, and fits again on train + pseudo-labelled rows.

  * confidence is the ENSEMBLE-MEAN top-class probability (the K members are
    averaged first, exactly as for the reported logits);
  * selection is per predicted class: within each class the rows at or above
    the ``1 - q`` quantile of that class's confidences are kept, so a fraction
    ~q of every predicted class survives and the pseudo-label pool stays
    class-balanced.  NOTE the sign convention differs from
    ``kernels.SELFTRAIN_DEFAULTS['quantile']``, which is the cut quantile of a
    GLOBAL top-2 margin (there, q=0.3 keeps 70% overall); here q is the
    fraction KEPT per class.  Same idea, different knob;
  * ``weight`` is the loss weight of a pseudo row (mlp/vat: a per-row weight in
    the cross entropy; ladder: integer weights only, realised by repeating the
    rows, as ``kernels._fit_selftrain`` does);
  * ``soft`` replaces the hard pseudo-label by the predicted distribution
    (mlp/vat only; the vendored ladder's loss takes hard labels);
  * ``refit='scratch'`` re-initialises (fresh seed), ``'continue'`` warm-starts
    the round from the previous round's weights (the optimizer state is not
    carried: Adam restarts with the round's own cosine schedule);
  * the round seed is derived as ``seed + 104729 * round``;
  * the deadline is SPLIT: fit ``k`` of the ``R+1`` fits may use at most its
    share of the remaining time, and before each round the cost of the next fit
    is projected from the first fit (scaled by the planned step count and
    de-biased if that fit was itself truncated) and checked against THAT
    ROUND'S share.  A round that would overrun its share is skipped, not
    truncated, and the previous round's prediction is returned; so is a round
    that raises.  ``metrics['selftrain_rounds_completed']`` /
    ``metrics['rounds_completed']`` and ``metrics['selftrain_rounds']``
    (pseudo counts, per-class counts, seconds, seed per round) record what
    actually ran, and ``metrics['truncated']`` is True if ANY round was
    truncated or any configured round did not run at all -- the configured
    recipe did not run, whatever the last fit did.  Costs
    (``training_seconds``, ``inference_seconds``, ``epochs_completed``) are
    summed over rounds; the last round's values keep a ``*_last_round`` name.

Self-training reads the query FEATURES, so it sets both
``metrics['uses_query_features_unlabeled']`` and the sibling module's spelling
``metrics['uses_query_images_unlabeled']`` to True.  Query LABELS are never
touched: the pseudo-labels come from the model's own predictions.

``target_steps``: one config that is sane at every N
----------------------------------------------------
A fixed ``epochs`` means 20x fewer gradient steps at N=500 than at N=10000.
With ``target_steps > 0`` the schedule is sized in STEPS instead:
``epochs = clamp(ceil(target_steps / steps_per_epoch), min_epochs, max_epochs)``
with ``steps_per_epoch = ceil(N / batch_size)``.  ``epochs`` is ignored then and
the resolved value is recorded in ``metrics['epochs_planned']`` and
``metrics['epochs_resolved']``.  For the ladder ``decay_start_epoch`` is rescaled
by the same factor, so the published 2/3-of-training decay point survives.

Not implemented: 'tabm'
-----------------------
TabM-style BatchEnsemble shares a base weight matrix and learns per-member rank-1
scalings.  The ``'mlp'`` family already trains K *fully* independent members in
one batched ``baddbmm``, which is the same wall-clock trick with strictly more
capacity per member; at the widths used here (<= 2048) the parameter saving buys
nothing on an A100, so it is skipped rather than approximated badly.

Interface
---------
``resolve_config(config)`` and
``fit_predict(train_z, train_y, query_z, config, seed, device, deadline_unix)``
-> ``{'logits': float32 (Q,10), 'labels': uint8 (Q,), 'metrics': dict}``.
Note the argument order (``device`` before ``deadline_unix``) matches the
kernel module of this study, not ``pmnist_learners.fit_predict``.

Query labels never enter this module.  Transductive use of the released query
*features* is legal (the harness hands them over) and is recorded in
``metrics['uses_query_features_unlabeled']``.
"""
from __future__ import annotations

import copy
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import pmnist_learners as _pm                                    # noqa: E402
# kernels.py is import-safe: the module body only defines constants, default
# dicts and functions (no data load, no device grab, no global torch flags), so
# importing it here costs nothing and lets the two preprocessing maps be SHARED
# rather than re-derived.  study.query_labels already refuses to run in a
# process that imported either module, so nothing about the audit changes.
import kernels as _kernels                                       # noqa: E402

# Imported unchanged from the pmnist study (kept byte-identical there).
TimeGuard = _pm.TimeGuard
first_max_labels = _pm.first_max_labels
vat_loss = _pm.vat_loss
ladder_lr_factor = _pm.ladder_lr_factor
_sync = _pm._sync
_device_name = _pm._device_name
_autocast = _pm._autocast
_apply_precision_flags = _pm._apply_precision_flags
_log_mean_prob = _pm._log_mean_prob
_soft_cross_entropy = _pm._soft_cross_entropy
_one_hot = _pm._one_hot
_make_generator = _pm._make_generator
_lr_factor = _pm._lr_factor

# Imported unchanged from the kernel module of THIS study (see the docstring):
# one implementation, one set of numbers, no second copy to keep in step.
within_class_transform = _kernels.within_class_transform
_kernel_rewhiten = _kernels._rewhiten

NUM_CLASSES = 10
RELEASE_DIM = 60          # harness 1.2.0 release_dims; the data still decides
# Measured marginal std of the 9x9 area-resized [0,1] pool (60000x81 values:
# mean 0.1307, std 0.2416, RMS 0.2747).  The familiar 0.3081 is the 28x28 MNIST
# constant and does not survive box averaging.  Reference value for the SNR
# bookkeeping only -- it is NOT the ladder's input_scale (see LADDER_DEFAULTS).
PIXEL_STD_9X9 = 0.2416

# --------------------------------------------------------------------------
# configuration defaults
# --------------------------------------------------------------------------
COMMON_DEFAULTS = {
    'family': None,
    # TF32 is family-dependent (see MLP_DEFAULTS / LADDER_DEFAULTS); the
    # conservative value is the common one, families override it.
    'tf32': False,
    'autocast_bf16': False,
    'inference_margin_seconds': 20.0,
    # Release-space preprocessing shared with kernels.py; see the docstring.
    # 'none' (default, current behaviour) | 'wc' | 'rw' | 'wc+rw'.
    'rewhiten': 'none',
    # Shrinkage of the pooled within-class covariance, S <- (1-s) S_w + s
    # (tr S_w / d) I.  0.05 is what the pilots call 'wc'
    # (research/kernel_pilots.py CANDIDATE_REF), not kernels.py's neutral 0.2
    # module default, because 0.05 is the value every recommended kernel
    # configuration uses.
    'wc_shrinkage': 0.05,
    'rw_source': 'query',            # 'query' | 'pooled' (kernels' rewhiten modes)
    'rw_shrinkage': 0.0,             # C <- (1-s) C + s (tr C / d) I before C^-1/2
    'notes': '',
}

# rounds=0 is off, so every existing config keeps its behaviour bit for bit.
SELFTRAIN_DEFAULTS = {
    'rounds': 0,
    # Fraction KEPT per predicted class (threshold = the 1-q confidence
    # quantile within the class).  See the docstring: kernels.py's key of the
    # same name is the complementary global-margin cut.
    'quantile': 0.5,
    'soft': False,                   # hard pseudo-label (mlp/vat may use soft)
    'weight': 1.0,                   # loss weight of a pseudo-labelled row
    'refit': 'scratch',              # 'scratch' | 'continue' (warm start)
}

MLP_DEFAULTS = {
    # TF32 on: these are dense fp32 matmuls, which an A100 runs at 19.5 TFLOP/s
    # without it and 156 TFLOP/s with it.  At the sizes the 1200 s cap is meant
    # to buy (widths [4096,4096] x 8 members x N=10000 is ~8.3 TFLOP/epoch,
    # i.e. 0.43 s/epoch fp32 vs 0.054 s/epoch TF32) the difference decides
    # whether a config is affordable; at widths [1024,1024] launch overhead
    # hides it.  Not yet measured on a real A100 -- the first GPU job records
    # metrics['tf32_matmul'] so the assumption can be replaced by a number.
    'tf32': True,
    'widths': [1024, 1024],
    'activation': 'relu',
    'dropout': 0.0,
    'input_noise_std': 0.0,
    'mixup_alpha': 0.0,
    'label_smoothing': 0.0,
    'lr': 0.001,
    'weight_decay': 0.0,
    'epochs': 100,
    # > 0 overrides 'epochs' with ceil(target_steps / steps_per_epoch), clamped
    # to [min_epochs, max_epochs]; 0 keeps the fixed-epoch schedule.
    'target_steps': 0,
    'min_epochs': 1,
    'max_epochs': 100000,
    'selftrain': {},
    'batch_size': 128,
    'warmup_fraction': 0.0,
    'schedule': 'cosine',
    'ema_decay': 0.0,
    'members': 1,
    'eval_batch_size': 2000,
    # 'none': the release is already zero-mean / unit-variance on the training
    # rows, so the identity is the honest default.  'standardize': per-feature
    # (x - mean) / (std + std_floor) with TRAINING-row statistics only.
    'normalization': 'none',
    'std_floor': 1e-5,
}

# eps lives in RELEASE units: the perturbation is an L2 ball of radius eps in
# the 60-d release space, where a training row has E||z||^2 = 60, i.e. RMS norm
# sqrt(60) = 7.75, and the median nearest-neighbour distance is ~6.0.  Neither
# yardstick picks the default: the row-norm rule (eps 2.0 = 26% of the row norm,
# matching the pixel study's 2.5/8.52) is too large and the neighbour distance
# would argue for larger still.  Measured instead, on real dev draws with
# widths [512,512], dropout 0.2, 50 epochs, batch 128, lr 1e-3, rampup 0.2,
# 10000 query rows (binomial SE ~0.3 pp at 9%, ~0.2 pp at 4.5%):
#     eps                  0.5    1.0    2.0    4.0
#     ..491, N=1057       9.73%  8.70%  9.23%  11.02%
#     ..492, N=4729       4.87%  4.54%  4.93%   5.87%
# eps=1.0 wins at both sizes, so it is the default; sweep {0.5, 1.0, 2.0} again
# if the width or the schedule changes materially.
VAT_DEFAULTS = {
    'eps': 1.0,
    'xi': 1e-6,
    'weight': 1.0,
    'power_iterations': 1,
    'unlabeled': 'train+query',
    'batch_size': None,          # None -> same as the labeled batch size
    # Linear ramp of the VAT weight over the first fraction of the total steps;
    # 0.0 disables the ramp (the pixel study's behaviour).
    'rampup_fraction': 0.2,
}

VAT_MLP_DEFAULTS = {**MLP_DEFAULTS, 'input_noise_std': 0.0, 'vat': {}}

LADDER_DEFAULTS = {
    # Left off for the ladder until measured: its normalisation runs with
    # bn_eps=1e-10, so a 10-bit mantissa in the matmuls is not obviously safe.
    'tf32': False,
    'hidden_dims': [1000, 500, 250, 250, 250],
    'noise_std': 0.3,
    # Scalar map applied to the release before the Ladder sees it (1.0 disables
    # it).  0.6 is the measured optimum of a six-point sweep on both dev seeds,
    # ~2 points better than the pixel-SNR-matched 0.24-0.30; see the module
    # docstring for the table.
    'input_scale': 0.6,
    'input_reconstruction_weight': 2000.0,
    # full_batches (release-cutoffs-20260925/full-batches): pad every epoch's
    # shuffle with rows from a second shuffle so that every minibatch has exactly
    # batch_size labelled rows when n > batch_size. Off by default, which keeps the
    # frozen recipe bit-identical; when n is a multiple of batch_size or at most
    # batch_size it changes nothing, not even the random stream.
    'full_batches': False,
    'epochs': 150,
    # Same rule as the MLP; 'decay_start_epoch' is rescaled by the same factor
    # so the published decay point stays at the same fraction of training.
    'target_steps': 0,
    'min_epochs': 1,
    'max_epochs': 100000,
    # These defaults do NOT fit a self-training share at N=10000: 150 epochs at
    # batch 100 is 15000 steps per member, ~690 s on an A100 against a 400 s
    # share at rounds=2 (600 s at rounds=1).  Pair ladder + selftrain with an
    # explicit 'target_steps' (~6000 at rounds=1); see _selftrain's docstring.
    'selftrain': {},
    'decay_start_epoch': 100,
    'lr': 0.002,
    'batch_size': 100,
    'unlabeled': 'train+query',
    'members': 1,
    'eval_batch_size': 2000,
}

FAMILY_DEFAULTS = {'mlp': MLP_DEFAULTS, 'vat': VAT_MLP_DEFAULTS,
                   'ladder': LADDER_DEFAULTS}
NORMALIZATIONS = ('none', 'standardize')
UNLABELED_POOLS = ('train', 'train+query')
REWHITEN_MODES = ('none', 'wc', 'rw', 'wc+rw')
RW_SOURCES = ('query', 'pooled')
REFIT_MODES = ('scratch', 'continue')
# Prime, and far above the 1000*member / 7919*layer offsets BatchedMLP uses, so
# no round can accidentally reproduce another round's member initialisation.
SELFTRAIN_SEED_STRIDE = 104729


def resolve_config(config):
    """Merge a candidate config with its family defaults; reject unknown keys."""
    family = config.get('family')
    if family not in FAMILY_DEFAULTS:
        raise ValueError(f"config['family'] must be one of {sorted(FAMILY_DEFAULTS)}")
    # Deep copies: the defaults hold mutable lists/dicts ('widths',
    # 'hidden_dims', 'vat'), and a Modal container runs many jobs in one
    # process, so handing out the module-level objects by reference would let
    # one job's in-place edit rewrite every later job's defaults.
    merged = copy.deepcopy({**COMMON_DEFAULTS, **FAMILY_DEFAULTS[family]})
    merged.update(copy.deepcopy(config))
    unknown = set(config) - set(COMMON_DEFAULTS) - set(FAMILY_DEFAULTS[family])
    if unknown:
        raise ValueError(f'Unknown config keys for {family}: {sorted(unknown)}')
    merged['family'] = family
    if merged['rewhiten'] not in REWHITEN_MODES:
        raise ValueError(f'rewhiten must be one of {list(REWHITEN_MODES)}')
    if merged['rw_source'] not in RW_SOURCES:
        raise ValueError(f'rw_source must be one of {list(RW_SOURCES)}')
    for key in ('wc_shrinkage', 'rw_shrinkage'):
        if not 0.0 <= float(merged[key]) <= 1.0:
            raise ValueError(f'{key} must be in [0, 1]')
    if int(merged['target_steps']) < 0:
        raise ValueError('target_steps must be >= 0')
    if int(merged['min_epochs']) < 1:
        raise ValueError('min_epochs must be >= 1')
    if int(merged['max_epochs']) < int(merged['min_epochs']):
        raise ValueError('max_epochs must be >= min_epochs')
    selftrain = dict(merged['selftrain'] or {})
    selftrain_unknown = set(selftrain) - set(SELFTRAIN_DEFAULTS)
    if selftrain_unknown:
        raise ValueError(f'Unknown selftrain keys: {sorted(selftrain_unknown)}')
    selftrain = {**SELFTRAIN_DEFAULTS, **selftrain}
    if int(selftrain['rounds']) < 0:
        raise ValueError("selftrain['rounds'] must be >= 0")
    if not 0.0 < float(selftrain['quantile']) <= 1.0:
        raise ValueError("selftrain['quantile'] must be in (0, 1]")
    if float(selftrain['weight']) <= 0.0:
        raise ValueError("selftrain['weight'] must be > 0")
    if selftrain['refit'] not in REFIT_MODES:
        raise ValueError(f"selftrain['refit'] must be one of {list(REFIT_MODES)}")
    if family == 'ladder' and int(selftrain['rounds']) > 0:
        # The vendored LadderAMLP.loss takes hard labels and no per-row weight,
        # so the two knobs it cannot honour are rejected instead of ignored.
        if bool(selftrain['soft']):
            raise ValueError("selftrain['soft'] is unsupported for the ladder "
                             "(LadderAMLP.loss takes hard labels)")
        weight = float(selftrain['weight'])
        if abs(weight - round(weight)) > 1e-9:
            raise ValueError("selftrain['weight'] must be a positive integer for "
                             "the ladder (weights are realised by repeating rows)")
    merged['selftrain'] = selftrain
    if family in ('mlp', 'vat'):
        if merged['normalization'] not in NORMALIZATIONS:
            raise ValueError(f"normalization must be one of {list(NORMALIZATIONS)}")
        if int(merged['members']) < 1:
            raise ValueError('members must be >= 1')
    if family == 'vat':
        vat = dict(merged['vat'] or {})
        vat_unknown = set(vat) - set(VAT_DEFAULTS)
        if vat_unknown:
            raise ValueError(f'Unknown vat keys: {sorted(vat_unknown)}')
        vat = {**VAT_DEFAULTS, **vat}
        if vat['unlabeled'] not in UNLABELED_POOLS:
            raise ValueError(f"vat['unlabeled'] must be one of {list(UNLABELED_POOLS)}")
        if not 0.0 <= float(vat['rampup_fraction']) <= 1.0:
            raise ValueError("vat['rampup_fraction'] must be in [0, 1]")
        if float(vat['eps']) < 0.0:
            raise ValueError("vat['eps'] must be >= 0")
        merged['vat'] = vat
    if family == 'ladder':
        if merged['unlabeled'] not in UNLABELED_POOLS:
            raise ValueError(f"unlabeled must be one of {list(UNLABELED_POOLS)}")
        if float(merged['input_scale']) <= 0.0:
            raise ValueError('input_scale must be positive')
        if int(merged['members']) < 1:
            raise ValueError('members must be >= 1')
    return merged


# --------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------
def apply_normalization(x, q, normalization, std_floor=1e-5):
    """Return ``(x, q, note)``; ``'none'`` returns the very same tensors."""
    if normalization == 'none':
        return x, q, ('none (release features as given: exactly zero-mean and '
                      'unit-variance per coordinate on the training rows)')
    if normalization == 'standardize':
        std_floor = float(std_floor)
        if std_floor <= 0:
            raise ValueError('std_floor must be positive')
        mu = x.mean(dim=0, keepdim=True)
        sd = x.std(dim=0, unbiased=True, keepdim=True) + std_floor
        return (x - mu) / sd, (q - mu) / sd, (
            'per-feature (x-mean)/(std+%g) with training-row statistics' % std_floor)
    raise ValueError(f'Unknown normalization {normalization!r}')


def apply_rewhiten(train_z, train_y, query_z, config):
    """Return ``(train_z, query_z, note)`` after the kernel study's maps.

    Operates on float32 numpy arrays and returns float32 numpy arrays; the maps
    themselves are formed in float64 (``torch.linalg.eigh`` on a 60x60 matrix,
    microseconds).  ``'none'`` returns the very same objects.

    The two maps come from ``kernels.py`` unchanged (``within_class_transform``,
    ``_rewhiten``), so a neural ``rewhiten='wc'`` and a kernel
    ``metric_learn='within_class'`` see the identical features.
    """
    mode = str(config['rewhiten'])
    if mode == 'none':
        return train_z, query_z, {'rewhiten': 'none'}
    if mode not in REWHITEN_MODES:
        raise ValueError(f'rewhiten must be one of {list(REWHITEN_MODES)}')
    train = torch.as_tensor(np.ascontiguousarray(train_z), dtype=torch.float64)
    query = torch.as_tensor(np.ascontiguousarray(query_z), dtype=torch.float64)
    labels = torch.as_tensor(np.ascontiguousarray(train_y), dtype=torch.long)
    note = {'rewhiten': mode}
    # kernels._preprocess's order: the query ZCA first, the supervised metric on
    # top of it (the reverse would let the ZCA undo the supervised sphering).
    if 'rw' in mode.split('+'):
        train, query, rw_note = _kernel_rewhiten(
            train, query, {'rewhiten': str(config['rw_source']),
                           'rewhiten_shrinkage': float(config['rw_shrinkage'])})
        note.update({'rw_source': rw_note['rewhiten'],
                     'rw_shrinkage': rw_note['rewhiten_shrinkage'],
                     'rw_rows': rw_note['rewhiten_rows'],
                     'rw_cond': rw_note['rewhiten_cond']})
    if 'wc' in mode.split('+'):
        transform = within_class_transform(train, labels, float(config['wc_shrinkage']))
        train, query = train @ transform, query @ transform
        note.update({'wc_shrinkage': float(config['wc_shrinkage']),
                     'wc_rows': int(train.shape[0])})
    note['uses_query_features'] = 'rw' in mode.split('+')
    return (np.ascontiguousarray(train.numpy(), dtype=np.float32),
            np.ascontiguousarray(query.numpy(), dtype=np.float32), note)


def ladder_steps_per_epoch(n, batch_size):
    """Minibatches the LADDER actually runs in one epoch over ``n`` rows.

    ``_fit_ladder_once`` skips any batch with fewer than 2 rows (``LadderAMLP``
    needs a pair for its batch statistics), so a ragged final batch of exactly
    one row -- ``n % batch_size == 1``, e.g. n=15001 at batch 100, which is
    reachable once self-training appends an uncontrolled pseudo count -- never
    executes.  Counting it would make ``target_steps`` / ``planned_steps`` claim
    one step per epoch that the fit never takes, and would make the same
    ``target_steps`` mean different work for the ladder than for the mlp.
    """
    n, batch_size = max(1, int(n)), max(1, int(batch_size))
    batches = math.ceil(n / batch_size)
    if n % batch_size == 1 and batches > 1:
        batches -= 1                      # the lone trailing row is dropped
    return max(1, batches)


def resolve_epochs(config, steps_per_epoch):
    """``(epochs, note)``: the fixed ``epochs``, or the ``target_steps`` schedule.

    ``target_steps <= 0`` (the default) returns ``int(config['epochs'])`` and a
    ``None`` note, i.e. exactly the previous behaviour.
    """
    target = int(config.get('target_steps', 0) or 0)
    fixed = int(config['epochs'])
    if target <= 0:
        return fixed, None
    steps_per_epoch = max(1, int(steps_per_epoch))
    low = int(config.get('min_epochs', 1))
    high = int(config.get('max_epochs', 100000))
    epochs = max(low, min(high, int(math.ceil(target / steps_per_epoch))))
    return epochs, {'target_steps': target, 'steps_per_epoch': steps_per_epoch,
                    'epochs_from_target': epochs, 'epochs_config': fixed,
                    'min_epochs': low, 'max_epochs': high,
                    'planned_steps': epochs * steps_per_epoch}


def select_pseudo_rows(probabilities, quantile):
    """Class-balanced confident rows for self-training.

    ``probabilities`` is the (Q, 10) ENSEMBLE-MEAN probability matrix.  Each
    query row is assigned its arg-max class (ties to the lowest class, as
    ``first_max_labels``) and, within each predicted class, the rows whose
    top-class probability is at or above that class's ``1 - quantile``
    confidence quantile are kept.  A class present at all keeps at least one
    row, so the pseudo set follows the predicted class distribution instead of
    collapsing onto the two or three easiest digits.  Rows tied AT the
    threshold are all kept, so ``quantile`` is an upper bound on the fraction
    kept, not an exact count.

    Returns ``(index, labels, confidence, per_class_counts)`` with ``index``
    sorted ascending (query-row order).
    """
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.argmax(probabilities, axis=1).astype(np.int64)      # ties -> lowest
    confidence = probabilities[np.arange(probabilities.shape[0]), labels]
    quantile = float(quantile)
    keep = np.zeros(probabilities.shape[0], dtype=bool)
    counts = [0] * NUM_CLASSES
    for label in range(NUM_CLASSES):
        rows = np.nonzero(labels == label)[0]
        if rows.size == 0:
            continue
        if quantile >= 1.0:
            keep[rows] = True
        else:
            threshold = np.quantile(confidence[rows], 1.0 - quantile)
            # The quantile of a non-empty array always lies in [min, max], so
            # '>=' keeps at least one row per present class -- no degenerate
            # fallback is needed.  It also keeps EVERY row tied at the
            # threshold, so the kept fraction is an upper bound on ``quantile``
            # (all-equal confidences keep the whole class).
            keep[rows[confidence[rows] >= threshold]] = True
        counts[label] = int(keep[rows].sum())
    index = np.nonzero(keep)[0]
    return index, labels[index].astype(np.int64), confidence[index], counts


def ladder_input_scale_note(input_scale, noise_std, train_z=None):
    """Human-readable record of what ``input_scale`` did to the Ladder SNR."""
    scale, noise = float(input_scale), float(noise_std)
    note = (f'release features multiplied by input_scale={scale:g} before the '
            f'Ladder; with noise_std={noise:g} the input SNR is '
            f'{scale / noise if noise else float("inf"):.3g} '
            f'(pixels: {PIXEL_STD_9X9:g}/{noise:g}) and the input-MSE penalty '
            f'scales as input_scale^2={scale ** 2:g}')
    if train_z is not None:
        rms = float(np.sqrt(np.mean(np.asarray(train_z, dtype=np.float64) ** 2)))
        note += f'; measured train RMS {rms:.4g} -> scaled RMS {scale * rms:.4g}'
    return note


def effective_noise_ratio(train_z, input_scale, noise_std):
    """noise_std / std(input_scale * z): the Ladder's input noise-to-signal."""
    rms = float(np.sqrt(np.mean(np.asarray(train_z, dtype=np.float64) ** 2)))
    return float(noise_std) / max(float(input_scale) * rms, 1e-12)


def _weighted_cross_entropy(logits, targets, weights):
    """Weighted mean soft-target cross entropy.

    ``weights`` broadcasts against the leading dims of ``logits`` (K, b).  With
    all weights equal this equals ``_soft_cross_entropy`` up to floating-point
    summation order, which is why the unweighted path is kept separate rather
    than expressed through this one.
    """
    row_loss = -(targets * F.log_softmax(logits, dim=-1)).sum(-1)
    total = weights.sum()
    return (row_loss * weights).sum() / total.clamp(min=1e-12)


def vat_weight_at_step(step, total_steps, weight, rampup_fraction):
    """Linear ramp of the VAT weight over the first ``rampup_fraction`` steps."""
    rampup = int(round(float(rampup_fraction) * max(1, int(total_steps))))
    if rampup <= 0:
        return float(weight)
    return float(weight) * min(1.0, (int(step) + 1) / rampup)


# --------------------------------------------------------------------------
# family 'mlp' / 'vat': one batched model holding K independent members
# --------------------------------------------------------------------------
class BatchedMLP:
    """K independent MLPs as (K, in, out) stacks; bmm/baddbmm forward.

    Copied from ``pmnist_learners.BatchedMLP`` (kept unmodified there) with one
    change: the input width is a constructor argument instead of the module
    constant ``INPUT_DIM = 81``, because a release row has 60 coordinates.  The
    pixel study's ``init_input_permutation`` diagnostic is dropped: the release
    is a dense rotation, so there is no feature permutation to check against.

    Member ``k`` is initialised from ``seed + 1000*k`` alone, so running the
    same recipe with ``members=1`` and ``seed = seed + 1000*k`` reproduces it.
    """

    def __init__(self, input_dim, widths, members, seed, device,
                 activation='relu', dropout=0.0):
        self.members = int(members)
        self.device = torch.device(device)
        self.dropout = float(dropout)
        self.activation = activation
        self.input_dim = int(input_dim)
        dims = [self.input_dim, *[int(w) for w in widths], NUM_CLASSES]
        self.dims = dims
        self.weights, self.biases = [], []
        for layer, (n_in, n_out) in enumerate(zip(dims[:-1], dims[1:])):
            bound = 1.0 / math.sqrt(n_in)
            stack_w, stack_b = [], []
            for k in range(self.members):
                generator = torch.Generator(device='cpu')
                generator.manual_seed(int(seed) + 1000 * k + 7919 * layer)
                stack_w.append(torch.empty(n_in, n_out).uniform_(-bound, bound,
                                                                 generator=generator))
                stack_b.append(torch.empty(n_out).uniform_(-bound, bound,
                                                           generator=generator))
            self.weights.append(torch.stack(stack_w).to(self.device).requires_grad_(True))
            self.biases.append(torch.stack(stack_b)[:, None, :].to(self.device)
                               .requires_grad_(True))

    def parameters(self):
        return [*self.weights, *self.biases]

    def parameter_count(self):
        return int(sum(p.numel() for p in self.parameters()))

    def _activate(self, h):
        if self.activation == 'relu':
            return F.relu(h)
        if self.activation == 'gelu':
            return F.gelu(h, approximate='none')
        raise ValueError(f'Unsupported activation {self.activation!r}')

    def forward(self, h, training, generator=None):
        """h: (K, B, input_dim) -> (K, B, 10)."""
        last = len(self.weights) - 1
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            h = torch.baddbmm(bias, h, weight)
            if index < last:
                h = self._activate(h)
                if training and self.dropout > 0.0:
                    keep = 1.0 - self.dropout
                    mask = (torch.rand(h.shape, device=h.device, generator=generator)
                            < keep).to(h.dtype)
                    h = h * mask / keep
        return h

    def state(self):
        return [p.detach().clone() for p in self.parameters()]

    def load_state(self, state):
        with torch.no_grad():
            for p, value in zip(self.parameters(), state):
                p.copy_(value)


def _fit_mlp_once(train_z, train_y, query_z, config, seed, deadline_unix, device,
                  sample_weight=None, pseudo_targets=None, init_state=None,
                  return_state=False):
    """Adapted from ``pmnist_learners._fit_mlp``.

    Changes: 60-d release inputs, ``normalization='none'``, VAT moved behind
    the separate ``'vat'`` family with a weight ramp-up, the ``target_steps``
    schedule, and the three hooks the self-training driver needs:

      ``sample_weight``   (N,) per-row loss weight, or None for the unweighted
                          mean -- None takes the ORIGINAL code path, so a fit
                          without self-training is unchanged bit for bit;
      ``pseudo_targets``  (P, 10) soft targets replacing the one-hot targets of
                          the last P rows (the appended pseudo-labelled rows);
      ``init_state``      warm start (``selftrain['refit'] == 'continue'``).

    Returns ``(logits, metrics)``, or ``(logits, metrics, state)`` when
    ``return_state`` is set (the parameters that produced the prediction, i.e.
    the EMA-corrected ones when an EMA is in use).
    """
    started = time.perf_counter()
    members = int(config['members'])
    device_t = torch.device(device)
    guard = TimeGuard(deadline_unix, config['inference_margin_seconds'])

    x = torch.as_tensor(np.ascontiguousarray(train_z), dtype=torch.float32,
                        device=device_t)
    y = torch.as_tensor(np.ascontiguousarray(train_y), dtype=torch.long,
                        device=device_t)
    q = torch.as_tensor(np.ascontiguousarray(query_z), dtype=torch.float32,
                        device=device_t)
    x, q, normalization_note = apply_normalization(
        x, q, str(config['normalization']), config['std_floor'])
    n, input_dim = int(x.shape[0]), int(x.shape[1])

    model = BatchedMLP(input_dim, config['widths'], members, seed, device_t,
                       activation=config['activation'], dropout=config['dropout'])
    if init_state is not None:
        model.load_state(init_state)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config['lr']),
                                  weight_decay=float(config['weight_decay']),
                                  betas=(0.9, 0.999), eps=1e-8)
    torch_gen = _make_generator(device_t, int(seed) + 999983)
    member_rngs = [np.random.Generator(np.random.PCG64(int(seed) + 1000 * k))
                   for k in range(members)]

    vat = config.get('vat') if config['family'] == 'vat' else None
    uses_query = bool(vat and vat['unlabeled'] == 'train+query')
    if vat is not None:
        unlabeled_pool = torch.cat([x, q], dim=0) if uses_query else x
        vat_batch = int(vat['batch_size'] or config['batch_size'])
        vat_weight = float(vat['weight'])
        vat_rampup = float(vat['rampup_fraction'])
    else:
        unlabeled_pool, vat_batch, vat_weight, vat_rampup = None, 0, 0.0, 0.0

    batch_size = int(config['batch_size'])
    steps_per_epoch = max(1, math.ceil(n / batch_size))
    epochs_planned, epochs_note = resolve_epochs(config, steps_per_epoch)
    total_steps = steps_per_epoch * epochs_planned
    ema_decay = float(config['ema_decay'])
    ema = model.state() if ema_decay > 0.0 else None
    ema_init = model.state() if ema_decay > 0.0 else None

    smoothing = float(config['label_smoothing'])
    mixup_alpha = float(config['mixup_alpha'])
    noise_std = float(config['input_noise_std'])

    # Targets are precomputed once and indexed per batch.  _one_hot is exact
    # (one-hot then an elementwise smoothing multiply), so ``targets_all[index]``
    # is bit-identical to the previous ``_one_hot(y[index], smoothing)``.
    targets_all = _one_hot(y, smoothing)
    if pseudo_targets is not None:
        soft = torch.as_tensor(np.ascontiguousarray(pseudo_targets),
                               dtype=torch.float32, device=device_t)
        if soft.shape[0] > n or soft.shape[1] != NUM_CLASSES:
            raise ValueError('pseudo_targets must be (P<=N, 10)')
        targets_all[n - soft.shape[0]:] = soft
    weights_all = None
    if sample_weight is not None:
        weights_all = torch.as_tensor(np.ascontiguousarray(sample_weight),
                                      dtype=torch.float32, device=device_t)
        if weights_all.shape != (n,):
            raise ValueError('sample_weight must be (N,)')

    history, epochs_completed, step, truncated = [], 0, 0, False
    last_vat_weight = 0.0
    _sync(device_t)
    train_started = time.perf_counter()
    for epoch in range(epochs_planned):
        if guard.should_stop():
            truncated = True
            break
        guard.start_epoch()
        orders = torch.stack([
            torch.as_tensor(rng.permutation(n), dtype=torch.long, device=device_t)
            for rng in member_rngs])                                   # (K, n)
        epoch_loss = torch.zeros((), device=device_t)
        for start in range(0, n, batch_size):
            index = orders[:, start:start + batch_size]                # (K, b)
            xb = x[index]                                              # (K, b, D)
            targets = targets_all[index]                               # (K, b, 10)
            wb = None if weights_all is None else weights_all[index]   # (K, b)
            if mixup_alpha > 0.0:
                mixed_x, mixed_t = [], []
                mixed_w = [] if wb is not None else None
                for k, rng in enumerate(member_rngs):
                    lam = float(rng.beta(mixup_alpha, mixup_alpha))
                    shuffle = torch.as_tensor(rng.permutation(xb.shape[1]),
                                              dtype=torch.long, device=device_t)
                    mixed_x.append(lam * xb[k] + (1.0 - lam) * xb[k][shuffle])
                    mixed_t.append(lam * targets[k] + (1.0 - lam) * targets[k][shuffle])
                    if mixed_w is not None:
                        mixed_w.append(lam * wb[k] + (1.0 - lam) * wb[k][shuffle])
                xb = torch.stack(mixed_x)
                targets = torch.stack(mixed_t)
                if mixed_w is not None:
                    wb = torch.stack(mixed_w)
            if noise_std > 0.0:
                xb = xb + noise_std * torch.randn(xb.shape, device=device_t,
                                                  generator=torch_gen)
            factor = _lr_factor(step, total_steps, float(config['warmup_fraction']),
                                config['schedule'])
            for group in optimizer.param_groups:
                group['lr'] = float(config['lr']) * factor
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device_t, config['autocast_bf16']):
                logits = model.forward(xb, training=True, generator=torch_gen)
                # wb is None unless a pseudo-labelled round asked for weights,
                # so the ordinary fit keeps the ORIGINAL expression untouched.
                loss = (_soft_cross_entropy(logits.float(), targets) * members
                        if wb is None else
                        _weighted_cross_entropy(logits.float(), targets, wb) * members)
                if vat is not None and vat_weight > 0.0:
                    weight_now = vat_weight_at_step(step, total_steps, vat_weight,
                                                    vat_rampup)
                    last_vat_weight = weight_now
                    if weight_now > 0.0:
                        pool = unlabeled_pool.shape[0]
                        u_index = torch.stack([
                            torch.as_tensor(rng.integers(0, pool, size=vat_batch),
                                            dtype=torch.long, device=device_t)
                            for rng in member_rngs])
                        loss = loss + weight_now * members * vat_loss(
                            model, unlabeled_pool[u_index], vat, torch_gen)
            loss.backward()
            optimizer.step()
            if ema is not None:
                with torch.no_grad():
                    for shadow, p in zip(ema, model.parameters()):
                        shadow.mul_(ema_decay).add_(p.detach(), alpha=1.0 - ema_decay)
            epoch_loss = epoch_loss + loss.detach()
            step += 1
        _sync(device_t)
        guard.end_epoch()
        epochs_completed += 1
        value = float(epoch_loss) / (steps_per_epoch * members)
        if not np.isfinite(value):
            raise FloatingPointError('Nonfinite training objective')
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == epochs_planned:
            history.append({'epoch': epoch + 1, 'mean_member_loss': value,
                            'learning_rate': optimizer.param_groups[0]['lr'],
                            'vat_weight': last_vat_weight,
                            'elapsed_training_seconds': time.perf_counter() - train_started})
    if epochs_completed < epochs_planned:
        truncated = True
    if epochs_completed == 0:
        # A random init's softmax is confident and arbitrary (~90% error with a
        # large logit spread).  Returning it as a normal result would let a
        # container that started seconds before the deadline -- or a config
        # sized for an A100 that landed on a T4 -- drag a reported threshold
        # down silently, so fail the job instead: runner.py records the failure
        # and writes no result file.
        raise TimeoutError('deadline expired before any epoch completed')
    _sync(device_t)
    training_seconds = time.perf_counter() - train_started

    # Bias-correct the EMA shadow against the init it started from (Adam-style),
    # exactly as in the pixel study.
    ema_effective_decay = None if ema is None else 1.0 - ema_decay ** step
    live = None
    if ema is not None and ema_effective_decay > 1e-3:
        bias = ema_decay ** step
        corrected = [(shadow - bias * init) / (1.0 - bias)
                     for shadow, init in zip(ema, ema_init)]
        live = model.state()
        model.load_state(corrected)
    else:
        ema = None

    _sync(device_t)
    inference_started = time.perf_counter()
    probabilities = np.zeros((q.shape[0], NUM_CLASSES), dtype=np.float64)
    eval_batch = int(config['eval_batch_size'])
    with torch.no_grad():
        for start in range(0, q.shape[0], eval_batch):
            chunk = q[start:start + eval_batch].unsqueeze(0).expand(members, -1, -1)
            with _autocast(device_t, config['autocast_bf16']):
                out = model.forward(chunk.contiguous(), training=False)
            probabilities[start:start + eval_batch] = (
                F.softmax(out.float(), dim=-1).sum(0).double().cpu().numpy())
    _sync(device_t)
    inference_seconds = time.perf_counter() - inference_started
    # The state that PRODUCED this prediction (EMA-corrected when an EMA is in
    # use), captured before the live weights are restored.
    state = model.state() if return_state else None
    if ema is not None:
        model.load_state(live)

    logits = _log_mean_prob(probabilities, members)
    metrics = {
        'family': config['family'], 'epochs_planned': epochs_planned,
        'epochs_completed': epochs_completed, 'truncated': bool(truncated),
        'training_seconds': training_seconds, 'inference_seconds': inference_seconds,
        'fit_wall_seconds': time.perf_counter() - started,
        'epoch_completion_fraction': epochs_completed / max(1, epochs_planned),
        'tf32_matmul': bool(torch.backends.cuda.matmul.allow_tf32),
        'autocast_bf16': bool(config['autocast_bf16']),
        'parameters': model.parameter_count(), 'members': members,
        'input_dim': input_dim,
        'uses_query_features_unlabeled': uses_query,
        'device_name': _device_name(device_t),
        'logit_kind': 'log_mean_prob', 'steps_completed': step,
        'steps_per_epoch': steps_per_epoch, 'mean_epoch_seconds': guard.mean_epoch,
        'history': history, 'ema_used': ema is not None,
        'ema_bias_correction': ema_effective_decay,
        'normalization': normalization_note,
        'vat': None if vat is None else dict(vat),
        'vat_rampup_steps': (0 if vat is None else
                             int(round(vat_rampup * max(1, total_steps)))),
        'vat_weight_final': (None if vat is None else last_vat_weight),
        'epochs_resolved': epochs_planned, 'target_steps_note': epochs_note,
        'rows_fitted': n,            # train rows + any pseudo-labelled rows
    }
    if return_state:
        return logits, metrics, state
    return logits, metrics


# --------------------------------------------------------------------------
# family 'ladder'
# --------------------------------------------------------------------------
def _fit_ladder_once(train_z, train_y, query_z, config, seed, deadline_unix, device,
                     sample_weight=None, pseudo_targets=None, init_state=None,
                     return_state=False):
    """Adapted from ``pmnist_learners._fit_ladder``.

    Changes: 60-d release inputs instead of 81 raw pixels, the new
    ``input_scale`` factor (module docstring), 'train+query' transduction by
    default, the ``target_steps`` schedule, and the warm-start / state-return
    hooks the self-training driver uses.  ``sample_weight`` and
    ``pseudo_targets`` are NOT supported here (``resolve_config`` rejects the
    configurations that would need them; the driver repeats rows instead).
    """
    if sample_weight is not None or pseudo_targets is not None:
        raise ValueError('the ladder takes hard labels and unweighted rows; '
                         'integer selftrain weights are realised by repeating rows')
    from ladder_model import LadderAMLP, LadderConfig

    started = time.perf_counter()
    assert not config['autocast_bf16'], (
        'bf16 autocast is unsupported for the ladder: its BN uses eps=1e-10')
    device_t = torch.device(device)
    guard = TimeGuard(deadline_unix, config['inference_margin_seconds'])
    members = int(config['members'])
    input_scale = float(config['input_scale'])

    x = torch.as_tensor(np.ascontiguousarray(train_z), dtype=torch.float32,
                        device=device_t) * input_scale
    y = torch.as_tensor(np.ascontiguousarray(train_y), dtype=torch.long, device=device_t)
    q = torch.as_tensor(np.ascontiguousarray(query_z), dtype=torch.float32,
                        device=device_t) * input_scale
    n, input_dim = int(x.shape[0]), int(x.shape[1])
    uses_query = config['unlabeled'] == 'train+query'
    pool = torch.cat([x, q], dim=0) if uses_query else x
    pool_n = pool.shape[0]

    batch_size = int(config['batch_size'])
    steps_per_epoch = ladder_steps_per_epoch(n, batch_size)
    epochs_planned, epochs_note = resolve_epochs(config, steps_per_epoch)
    decay_start = int(config['decay_start_epoch'])
    if epochs_note is not None:
        # Keep the decay point at the same FRACTION of training as the config
        # asked for, so the published 100/150 schedule survives a resize.
        fraction = decay_start / max(1, int(config['epochs']))
        decay_start = min(epochs_planned, max(1, int(round(epochs_planned * fraction))))
        epochs_note['decay_start_epoch'] = decay_start
    ladder_config = LadderConfig(
        hidden_dims=tuple(int(h) for h in config['hidden_dims']),
        noise_std=float(config['noise_std']),
        input_reconstruction_weight=float(config['input_reconstruction_weight']),
        batch_size=batch_size, learning_rate=float(config['lr']),
        epochs=epochs_planned, decay_start_epoch=decay_start)

    probabilities = np.zeros((q.shape[0], NUM_CLASSES), dtype=np.float64)
    total_epochs, members_completed, parameters = 0, 0, 0
    training_seconds, inference_seconds = 0.0, 0.0
    member_epochs, history, states = [], [], []
    for k in range(members):
        torch.manual_seed(int(seed) + 1000 * k)
        if device_t.type == 'cuda':
            torch.cuda.manual_seed_all(int(seed) + 1000 * k)
        model = LadderAMLP(input_dim=input_dim, num_classes=NUM_CLASSES,
                           config=ladder_config).to(device_t)
        if init_state is not None and k < len(init_state):
            model.load_state_dict(init_state[k])          # selftrain warm start
        parameters += int(sum(p.numel() for p in model.parameters()))
        optimizer = torch.optim.Adam(model.parameters(), lr=float(config['lr']),
                                     betas=(0.9, 0.999), eps=1e-8)
        rng = np.random.Generator(np.random.PCG64(int(seed) + 1000 * k))
        model.train()
        _sync(device_t)
        member_started = time.perf_counter()
        completed = 0
        for epoch in range(epochs_planned):
            if guard.should_stop():
                break
            guard.start_epoch()
            learning_rate = float(config['lr']) * ladder_lr_factor(
                epoch, epochs_planned, decay_start)
            for group in optimizer.param_groups:
                group['lr'] = learning_rate
            shuffle = rng.permutation(n)
            if config.get('full_batches') and n > batch_size and n % batch_size:
                shuffle = np.concatenate([shuffle, rng.permutation(n)[:batch_size - n % batch_size]])
            order = torch.as_tensor(shuffle, dtype=torch.long, device=device_t)
            recon = torch.as_tensor(rng.permutation(pool_n), dtype=torch.long,
                                    device=device_t)
            loss_value = float('nan')
            for position, start in enumerate(range(0, order.numel(), batch_size)):
                index = order[start:start + batch_size]
                if index.numel() < 2:
                    continue          # LadderAMLP requires >= 2 rows per batch
                offset = (position * batch_size) % pool_n
                u_index = recon[offset:offset + index.numel()]
                if u_index.numel() < 2:
                    u_index = recon[:index.numel()]
                optimizer.zero_grad(set_to_none=True)
                loss = model.loss(x[index], y[index], x_unlabeled=pool[u_index])
                loss.backward()
                optimizer.step()
                loss_value = float(loss.detach())
            _sync(device_t)
            guard.end_epoch()
            completed += 1
            if not np.isfinite(loss_value):
                raise FloatingPointError('Nonfinite ladder objective')
            if epoch == 0 or (epoch + 1) % 25 == 0 or epoch + 1 == epochs_planned:
                history.append({'member': k, 'epoch': epoch + 1,
                                'last_minibatch_loss': loss_value,
                                'learning_rate': learning_rate,
                                'elapsed_training_seconds':
                                    time.perf_counter() - member_started})
        _sync(device_t)
        training_seconds += time.perf_counter() - member_started
        member_epochs.append(completed)
        total_epochs += completed
        if completed == 0:
            # Out of time before this member trained at all: averaging an
            # untrained net's confident-but-arbitrary softmax would corrupt the
            # ensemble, so drop it and keep the members that did train.  If it
            # was the FIRST member there is nothing to keep and the fit fails
            # below rather than reporting a random init's predictions.
            del model, optimizer
            if device_t.type == 'cuda':
                torch.cuda.empty_cache()
            break

        _sync(device_t)
        inference_started = time.perf_counter()
        model.calibrate_bn(x, batch_size=min(batch_size, max(2, n)))
        model.eval()
        eval_batch = int(config['eval_batch_size'])
        with torch.inference_mode():
            for start in range(0, q.shape[0], eval_batch):
                out = model(q[start:start + eval_batch])
                probabilities[start:start + eval_batch] += (
                    F.softmax(out, dim=-1).double().cpu().numpy())
        _sync(device_t)
        inference_seconds += time.perf_counter() - inference_started
        members_completed += 1
        if return_state:
            states.append({name: tensor.detach().clone()
                           for name, tensor in model.state_dict().items()})
        del model, optimizer
        if device_t.type == 'cuda':
            torch.cuda.empty_cache()
        if completed < epochs_planned:
            break                      # out of time: keep the members we have

    if members_completed == 0:
        raise TimeoutError('deadline expired before any epoch completed')
    logits = _log_mean_prob(probabilities, members_completed)
    truncated = (members_completed < members
                 or any(e < epochs_planned for e in member_epochs))
    metrics = {
        'family': 'ladder', 'epochs_planned': epochs_planned * members,
        'epochs_completed': total_epochs, 'truncated': bool(truncated),
        'training_seconds': training_seconds, 'inference_seconds': inference_seconds,
        'fit_wall_seconds': time.perf_counter() - started, 'parameters': parameters,
        'epoch_completion_fraction': total_epochs / max(1, epochs_planned * members),
        'tf32_matmul': bool(torch.backends.cuda.matmul.allow_tf32),
        'autocast_bf16': bool(config['autocast_bf16']),
        'members': members, 'members_completed': members_completed,
        'member_epochs_completed': member_epochs,
        'input_dim': input_dim,
        'uses_query_features_unlabeled': uses_query,
        'device_name': _device_name(device_t), 'logit_kind': 'log_mean_prob',
        'normalization': ladder_input_scale_note(input_scale, config['noise_std'],
                                                 train_z),
        'input_scale': input_scale,
        'effective_input_noise_ratio': effective_noise_ratio(
            train_z, input_scale, config['noise_std']),
        'mean_epoch_seconds': guard.mean_epoch, 'history': history,
        'epochs_resolved': epochs_planned, 'target_steps_note': epochs_note,
        'decay_start_epoch_resolved': decay_start,
        'steps_per_epoch': steps_per_epoch,
        'rows_fitted': n,            # train rows + any pseudo-labelled rows
    }
    if return_state:
        return logits, metrics, states
    return logits, metrics


# --------------------------------------------------------------------------
# self-training: pseudo-label the confident query rows and refit
# --------------------------------------------------------------------------
def _step_ratio(config, base_rows, new_rows):
    """Planned gradient steps at ``new_rows`` over those at ``base_rows``."""
    batch = max(1, int(config['batch_size']))
    if config.get('family') == 'ladder':          # a lone trailing row is skipped
        before = ladder_steps_per_epoch(base_rows, batch)
        after = ladder_steps_per_epoch(new_rows, batch)
    else:
        before = max(1, math.ceil(max(1, int(base_rows)) / batch))
        after = max(1, math.ceil(max(1, int(new_rows)) / batch))
    steps_before = resolve_epochs(config, before)[0] * before
    steps_after = resolve_epochs(config, after)[0] * after
    return float(steps_after) / max(1.0, float(steps_before))


def _selftrain(fit_once, train_z, train_y, query_z, config, seed, deadline_unix,
               device):
    """Run ``fit_once`` once, then ``selftrain['rounds']`` pseudo-labelled refits.

    ``rounds == 0`` calls ``fit_once`` exactly as the module did before this
    wrapper existed -- same arguments, same deadline, no extra RNG draws -- so
    every pre-existing configuration is reproduced bit for bit.

    The deadline is split into ``rounds + 1`` equal shares; before each round
    the next fit's cost is projected from the FIRST fit (de-biased by the
    fraction of its epochs that fit actually reached) and scaled by the planned
    gradient steps (``_step_ratio``), then compared against THAT ROUND'S OWN
    share.  A round that would not fit its share is skipped rather than started
    and truncated -- the previous round's prediction is a finished model, a
    half-trained refit is not.  A round that raises (TimeoutError, or the
    ``FloatingPointError`` a diverging refit raises) gets the same treatment:
    the previous round's prediction is returned and
    ``metrics['selftrain_round_failure']`` records why.

    Note the consequence for a tight budget: with ``rounds=R`` the base fit may
    use only 1/(R+1) of the time, so a config sized to fill the whole budget is
    truncated inside its share.  Size the epochs (or ``target_steps``) for the
    SHARE.  Quantitatively, against the planner's A100 cost model (mlp 0.0040
    s/step, wide/large-K 0.0060, vat 0.0075, ladder 0.0450, plus ~15 s fixed
    per fit) and the 1200 s per-fit cap, the budget per fit is
    ``(1200 - 15*(R+1)) / (R+1)`` seconds, i.e. a steps ceiling of

        mlp   R=1: ~72k steps/fit   R=2: ~48k     R=3: ~36k
        vat   R=1: ~39k             R=2: ~26k     R=3: ~19k
        ladder R=1: ~6.4k           R=2: ~4.3k    R=3: ~3.2k

    and the refit is ~1.5x the base at N=10000 (15k augmented rows vs 10k), so
    divide the later rounds by that.  The ladder at its defaults
    (``epochs=150``, ``batch_size=100``) already wants 15000 steps = ~690 s for
    ONE member at N=10000, which does NOT fit any share: ladder + selftrain at
    N=10000 needs an explicit ``target_steps`` (~6000 at R=1) or it truncates
    every fit.
    """
    settings = config['selftrain']
    rounds = int(settings['rounds'])
    family = config['family']
    supports_weights = family in ('mlp', 'vat')
    if rounds <= 0:
        logits, metrics = fit_once(train_z, train_y, query_z, config, seed,
                                   deadline_unix, device)
        metrics = dict(metrics)
        metrics.update({'selftrain': dict(settings), 'rounds_completed': 0,
                        'selftrain_rounds_completed': 0, 'selftrain_rounds': [],
                        'pseudo_rows_per_round': []})
        return logits, metrics

    quantile = float(settings['quantile'])
    weight = float(settings['weight'])
    soft = bool(settings['soft'])
    warm = settings['refit'] == 'continue'
    margin = float(config['inference_margin_seconds'])
    start_unix = time.time()
    share = max(0.0, float(deadline_unix) - start_unix) / (rounds + 1)

    def round_deadline(index):
        """Round ``index`` may use its own share; the last one owns the rest."""
        if index >= rounds:
            return float(deadline_unix)
        return min(float(deadline_unix), start_unix + (index + 1) * share)

    def run(rows_z, rows_y, round_seed, round_dl, **extras):
        """``fit_once`` returning ``(logits, metrics, state)``.

        The state is only materialised for a warm start; a 'scratch' refit
        would pay a full parameter copy for nothing.
        """
        if warm:
            return fit_once(rows_z, rows_y, query_z, config, round_seed, round_dl,
                            device, return_state=True, **extras)
        out = fit_once(rows_z, rows_y, query_z, config, round_seed, round_dl,
                       device, **extras)
        return out[0], out[1], None

    wrapper_started = time.perf_counter()
    fit_started = time.perf_counter()
    logits, metrics, state = run(train_z, train_y, seed, round_deadline(0))
    base_seconds = time.perf_counter() - fit_started
    base_metrics = dict(metrics)
    rounds_log = [{'round': 0, 'seed': int(seed), 'pseudo_rows': 0,
                   'rows_fitted': int(train_z.shape[0]),
                   'fit_seconds': base_seconds,
                   'training_seconds': metrics.get('training_seconds'),
                   'inference_seconds': metrics.get('inference_seconds'),
                   'epochs_planned': metrics.get('epochs_planned'),
                   'epochs_completed': metrics.get('epochs_completed'),
                   'truncated': bool(metrics.get('truncated'))}]
    skipped_from = None
    round_failure = None
    base_rows = max(1, int(train_z.shape[0]))

    for index in range(1, rounds + 1):
        probabilities = np.exp(np.asarray(logits, dtype=np.float64))  # log mean prob
        keep, pseudo_y, confidence, per_class = select_pseudo_rows(probabilities,
                                                                   quantile)
        if keep.size == 0:
            skipped_from = index
            break
        repeat = 1
        if not supports_weights and weight != 1.0:
            repeat = int(round(weight))           # ladder: repeated rows
        pseudo_x = np.repeat(query_z[keep], repeat, axis=0) if repeat > 1 \
            else query_z[keep]
        pseudo_labels = np.repeat(pseudo_y, repeat) if repeat > 1 else pseudo_y
        augmented_z = np.ascontiguousarray(np.concatenate([train_z, pseudo_x], axis=0),
                                           dtype=np.float32)
        augmented_y = np.ascontiguousarray(
            np.concatenate([train_y.astype(np.int64), pseudo_labels]), dtype=np.uint8)
        extras = {}
        if supports_weights and weight != 1.0:
            weights = np.ones(augmented_z.shape[0], dtype=np.float32)
            weights[train_z.shape[0]:] = weight
            extras['sample_weight'] = weights
        if soft:
            if not supports_weights:
                raise ValueError("selftrain['soft'] is unsupported for the ladder")
            extras['pseudo_targets'] = probabilities[keep].astype(np.float32)
        if warm:
            extras['init_state'] = state

        # Project this round from the first fit, in STEPS rather than rows: with
        # a fixed 'epochs' the two agree (more rows, more steps per epoch), but
        # under 'target_steps' the round costs the same as the base fit and a
        # row-count projection would refuse rounds that comfortably fit.
        # If the base fit was itself truncated it burned its share WITHOUT
        # finishing, so its wall seconds understate a full fit by exactly the
        # fraction of epochs it reached: de-bias before scaling.
        completion = float(base_metrics.get('epoch_completion_fraction') or 1.0)
        projected = (base_seconds / max(0.05, completion)) * _step_ratio(
            config, base_rows, augmented_z.shape[0])
        # ...and compare against THIS round's share, not the whole budget: a
        # round measured against the global deadline is started whenever any
        # time is left and then truncated inside its own (much smaller) share,
        # which is precisely what this wrapper promises not to do.
        if time.time() + 1.15 * projected + margin > round_deadline(index):
            skipped_from = index
            break

        round_seed = int(seed) + SELFTRAIN_SEED_STRIDE * index
        fit_started = time.perf_counter()
        try:
            logits, metrics, state = run(augmented_z, augmented_y, round_seed,
                                         round_deadline(index), **extras)
        except Exception as error:            # noqa: BLE001 - deliberate
            # Nothing usable came out of this round, so keep the previous
            # round's FINISHED prediction instead of failing the whole job.
            # TimeoutError is the expected case (the fit ran out of its share);
            # FloatingPointError('Nonfinite training objective') is the other
            # realistic one, since a refit trains ~10x the rows with a fresh
            # Adam restart and possibly a pseudo-label weight > 1.  On a paid
            # GPU the base fit's reservation is already spent either way.
            skipped_from = index
            round_failure = {'round': index, 'seed': round_seed,
                             'error': repr(error),
                             'fit_seconds': time.perf_counter() - fit_started}
            break
        rounds_log.append({
            'round': index, 'seed': round_seed,
            'pseudo_rows': int(keep.size), 'pseudo_rows_weighted': int(pseudo_x.shape[0]),
            'pseudo_per_class': [int(c) for c in per_class],
            'predicted_per_class': [int(c) for c in np.bincount(
                np.argmax(probabilities, axis=1), minlength=NUM_CLASSES)],
            'pseudo_weight': weight, 'pseudo_soft': soft,
            'min_confidence': float(confidence.min()),
            'mean_confidence': float(confidence.mean()),
            'rows_fitted': int(augmented_z.shape[0]),
            'projected_seconds': float(projected),
            'fit_seconds': time.perf_counter() - fit_started,
            'training_seconds': metrics.get('training_seconds'),
            'inference_seconds': metrics.get('inference_seconds'),
            'epochs_planned': metrics.get('epochs_planned'),
            'epochs_completed': metrics.get('epochs_completed'),
            'truncated': bool(metrics.get('truncated'))})

    metrics = dict(metrics)
    completed = len(rounds_log) - 1
    epochs_all = int(sum(r['epochs_completed'] or 0 for r in rounds_log))
    # A stub / a family that does not report 'epochs_planned' keeps the last
    # round's value rather than a misleading zero.
    planned_all = (int(sum(r['epochs_planned'] for r in rounds_log))
                   if all(r.get('epochs_planned') is not None for r in rounds_log)
                   else metrics.get('epochs_planned'))
    metrics.update({
        'selftrain': dict(settings),
        'rounds_completed': completed,
        'selftrain_rounds_completed': completed,
        'selftrain_rounds_planned': rounds,
        'selftrain_rounds': rounds_log,
        'pseudo_rows_per_round': [int(r['pseudo_rows']) for r in rounds_log],
        'selftrain_skipped_from_round': skipped_from,
        'selftrain_round_failure': round_failure,
        'selftrain_round_share_seconds': share,
        # 'truncated' is the ONLY truncation signal score.py (METRIC_KEYS) and
        # study.py (required_metrics) carry, and the frozen eligibility rule
        # reads it as "this fit finished the recipe it was configured with".
        # It must therefore cover the WHOLE call: an earlier round that was
        # truncated, or a round that never ran at all, both mean the configured
        # recipe did not run, even when the LAST fit completed cleanly.  The
        # two causes stay separable via 'truncated_any_round' /
        # 'selftrain_rounds_incomplete' (and the per-round log).
        'truncated_any_round': bool(any(r['truncated'] for r in rounds_log)),
        'selftrain_rounds_incomplete': bool(skipped_from is not None),
        'truncated': bool(any(r['truncated'] for r in rounds_log)
                          or skipped_from is not None),
        'truncated_last_round': bool(metrics.get('truncated')),
        # Costs are reported for the WHOLE learner call (kernels.py does the
        # same): every round trains and re-scores the full query block, so the
        # last round's seconds understate the job by a factor of rounds+1.  The
        # per-round split is in 'selftrain_rounds'.  Training, inference and
        # epochs are each summed from the rounds' OWN reports so the three
        # describe the same scope; the wrapper's wall clock (which also covers
        # tensor setup and pseudo-label selection) is 'fit_wall_seconds'.
        'training_seconds_last_round': metrics.get('training_seconds'),
        'inference_seconds_last_round': metrics.get('inference_seconds'),
        'epochs_planned_last_round': metrics.get('epochs_planned'),
        'epochs_completed_last_round': metrics.get('epochs_completed'),
        'epochs_completed_all_rounds': epochs_all,
        'epochs_completed': epochs_all,
        'epochs_planned': planned_all,
        'epoch_completion_fraction': (epochs_all / max(1, int(planned_all))
                                      if planned_all else
                                      metrics.get('epoch_completion_fraction')),
        'training_seconds': float(sum(r['training_seconds'] or 0.0
                                      for r in rounds_log)),
        'inference_seconds': float(sum(r['inference_seconds'] or 0.0
                                       for r in rounds_log)),
        'fit_seconds_all_rounds': float(sum(r['fit_seconds'] for r in rounds_log)),
        'fit_wall_seconds': time.perf_counter() - wrapper_started,
        # Pseudo-labels are the model's OWN predictions on the released query
        # features; no query label is read anywhere (there is no argument for
        # one).  Both spellings are set: kernels.py uses 'images', this module
        # used 'features'.
        'uses_query_features_unlabeled': True,
        'uses_query_images_unlabeled': True})
    return logits, metrics


def _fit_mlp(train_z, train_y, query_z, config, seed, deadline_unix, device):
    return _selftrain(_fit_mlp_once, train_z, train_y, query_z, config, seed,
                      deadline_unix, device)


def _fit_ladder(train_z, train_y, query_z, config, seed, deadline_unix, device):
    return _selftrain(_fit_ladder_once, train_z, train_y, query_z, config, seed,
                      deadline_unix, device)


# --------------------------------------------------------------------------
# public entry point
# --------------------------------------------------------------------------
_FAMILIES = {'mlp': _fit_mlp, 'vat': _fit_mlp, 'ladder': _fit_ladder}


def fit_predict(train_z, train_y, query_z, config, seed, device, deadline_unix):
    """Train one candidate on a release draw and predict the query rows.

    Never sees query labels.  ``train_z`` / ``query_z`` are the released
    features ``z = Q W (x - mu)``; any number of feature columns is accepted
    (the harness releases 60) as long as both halves agree.
    """
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
    if not (np.isfinite(train_z).all() and np.isfinite(query_z).all()):
        raise ValueError('release features must be finite')
    torch.manual_seed(int(seed))
    np.random.seed(int(seed) % (2 ** 31))
    _apply_precision_flags(resolved)
    # One preprocessing step for all three families, before anything the family
    # does (the ladder's input_scale multiplies the REWHITENED features).
    rewhiten_seconds = time.perf_counter()
    train_z, query_z, rewhiten_note = apply_rewhiten(train_z, train_y, query_z,
                                                     resolved)
    rewhiten_seconds = time.perf_counter() - rewhiten_seconds
    logits, metrics = _FAMILIES[resolved['family']](
        train_z, train_y, query_z, resolved, int(seed), float(deadline_unix), device)
    logits = np.ascontiguousarray(logits, dtype=np.float32)
    if logits.shape != (query_z.shape[0], NUM_CLASSES) or not np.isfinite(logits).all():
        raise RuntimeError('Learner produced malformed logits')
    labels = first_max_labels(logits)
    # Transduction is the union of the three ways this module can touch the
    # query FEATURES: an unlabeled pool (vat/ladder), 'rw' re-whitening, and
    # self-training's pseudo-labels.  Reported under both spellings.
    uses_query = bool(metrics.get('uses_query_features_unlabeled')
                      or metrics.get('uses_query_images_unlabeled')
                      or rewhiten_note.get('uses_query_features'))
    metrics = {'normalization': 'unrecorded', **metrics,
               'uses_query_features_unlabeled': uses_query,
               'uses_query_images_unlabeled': uses_query,
               'rewhiten': rewhiten_note,
               'rewhiten_seconds': float(rewhiten_seconds),
               'config': resolved, 'seed': int(seed),
               'deadline_unix': float(deadline_unix),
               'train_count': int(train_z.shape[0]),
               'query_count': int(query_z.shape[0]),
               'release_dim': int(train_z.shape[1]),
               'query_labels_supplied': False}
    return {'logits': logits, 'labels': labels, 'metrics': metrics}
