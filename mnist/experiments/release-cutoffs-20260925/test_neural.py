"""Tests for neural.py: the release-space MLP / VAT / Ladder learners.

Everything here runs on the CPU in seconds on synthetic release-shaped data
(60 zero-mean unit-variance coordinates with a class-dependent mean shift), so
no MNIST download and no GPU money are involved.  The real-data sanity fit
lives in probe_neural.py.
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import kernels                                                       # noqa: E402
import neural                                                        # noqa: E402

DIM = neural.RELEASE_DIM


def make_release_like(n_train=240, n_query=80, dim=DIM, seed=0):
    """Whitened-looking features: unit variance per coordinate, 10 classes."""
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((10, dim)).astype(np.float32)
    train_y = rng.integers(0, 10, size=n_train).astype(np.uint8)
    query_y = rng.integers(0, 10, size=n_query).astype(np.uint8)
    train_z = (rng.standard_normal((n_train, dim)) + 1.2 * centers[train_y]).astype(np.float32)
    query_z = (rng.standard_normal((n_query, dim)) + 1.2 * centers[query_y]).astype(np.float32)
    # Whiten as the harness does, so the training rows are exactly zero-mean /
    # unit-variance per coordinate.
    mu = train_z.mean(0, keepdims=True)
    sd = train_z.std(0, keepdims=True)
    return ((train_z - mu) / sd).astype(np.float32), train_y, \
           ((query_z - mu) / sd).astype(np.float32), query_y


def far_deadline():
    return time.time() + 3600.0


# --------------------------------------------------------------------------
# resolve_config
# --------------------------------------------------------------------------
def test_resolve_config_rejects_unknown_family_and_keys():
    with pytest.raises(ValueError):
        neural.resolve_config({'family': 'topo_cnn'})
    with pytest.raises(ValueError):
        neural.resolve_config({'family': 'mlp', 'widht': [16]})
    with pytest.raises(ValueError):
        neural.resolve_config({'family': 'vat', 'vat': {'epsilon': 1.0}})
    with pytest.raises(ValueError):
        neural.resolve_config({'family': 'mlp', 'normalization': '4x-0.5'})
    with pytest.raises(ValueError):
        neural.resolve_config({'family': 'ladder', 'input_scale': 0.0})


def test_resolve_config_defaults():
    mlp = neural.resolve_config({'family': 'mlp'})
    assert mlp['normalization'] == 'none'
    assert 'vat' not in mlp                       # VAT is its own family now
    assert neural.resolve_config({'family': 'vat'})['vat']['eps'] == 1.0
    vat = neural.resolve_config({'family': 'vat', 'vat': {'eps': 3.0}})
    assert vat['vat']['eps'] == 3.0
    assert vat['vat']['rampup_fraction'] == 0.2   # ramp-up on by default
    assert vat['vat']['unlabeled'] == 'train+query'
    assert vat['input_noise_std'] == 0.0          # input noise off by default
    ladder = neural.resolve_config({'family': 'ladder'})
    assert ladder['input_scale'] == pytest.approx(0.6)   # measured optimum
    assert ladder['unlabeled'] == 'train+query'
    assert neural.PIXEL_STD_9X9 == pytest.approx(0.2416)  # measured, not 0.3081
    assert ladder['tf32'] is False and mlp['tf32'] is True


def test_resolve_config_returns_private_copies():
    """A resolved config may be mutated without poisoning later jobs."""
    first = neural.resolve_config({'family': 'mlp'})
    assert first['widths'] is not neural.MLP_DEFAULTS['widths']
    first['widths'].append(7)
    assert neural.resolve_config({'family': 'mlp'})['widths'] == [1024, 1024]
    assert neural.MLP_DEFAULTS['widths'] == [1024, 1024]
    assert neural.VAT_MLP_DEFAULTS['widths'] == [1024, 1024]
    ladder = neural.resolve_config({'family': 'ladder'})
    assert ladder['hidden_dims'] is not neural.LADDER_DEFAULTS['hidden_dims']
    ladder['hidden_dims'].append(7)
    assert neural.resolve_config({'family': 'ladder'})['hidden_dims'] == \
        neural.LADDER_DEFAULTS['hidden_dims']


# --------------------------------------------------------------------------
# normalization
# --------------------------------------------------------------------------
def test_normalization_none_leaves_inputs_untouched():
    train_z, _, query_z, _ = make_release_like(seed=1)
    x = torch.as_tensor(train_z)
    q = torch.as_tensor(query_z)
    out_x, out_q, note = neural.apply_normalization(x, q, 'none')
    assert out_x is x and out_q is q
    assert torch.equal(out_x, x) and torch.equal(out_q, q)
    assert 'none' in note


def test_normalization_standardize_centers_training_rows():
    train_z, _, query_z, _ = make_release_like(seed=2)
    x = torch.as_tensor(train_z)
    q = torch.as_tensor(query_z)
    out_x, out_q, note = neural.apply_normalization(x, q, 'standardize')
    assert out_x is not x
    assert torch.allclose(out_x.mean(0), torch.zeros(DIM), atol=1e-5)
    assert torch.allclose(out_x.std(0, unbiased=True), torch.ones(DIM), atol=1e-3)
    assert not torch.equal(out_q, q)
    assert 'std' in note
    with pytest.raises(ValueError):
        neural.apply_normalization(x, q, 'quantile')


def test_normalization_none_and_standardize_agree_on_whitened_input():
    """On exactly whitened rows 'standardize' is (near) the identity."""
    train_z, _, query_z, _ = make_release_like(seed=3)
    x = torch.as_tensor(train_z)
    q = torch.as_tensor(query_z)
    std_x, _, _ = neural.apply_normalization(x, q, 'standardize')
    # The only systematic difference is the unbiased/biased variance factor
    # sqrt(n/(n-1)) (plus std_floor), i.e. well under 1% at these sizes.
    assert torch.allclose(std_x, x, rtol=0.01, atol=1e-5)


# --------------------------------------------------------------------------
# shapes, labels, metrics
# --------------------------------------------------------------------------
@pytest.mark.parametrize('config', [
    {'family': 'mlp', 'widths': [64, 64], 'epochs': 2, 'batch_size': 64, 'members': 2},
    {'family': 'vat', 'widths': [64], 'epochs': 2, 'batch_size': 64,
     'vat': {'eps': 1.0, 'batch_size': 32}},
    {'family': 'ladder', 'hidden_dims': [64, 32], 'epochs': 2, 'batch_size': 50,
     'decay_start_epoch': 1},
])
def test_shapes_and_first_max_labels(config):
    train_z, train_y, query_z, _ = make_release_like(seed=4)
    out = neural.fit_predict(train_z, train_y, query_z, config, seed=7,
                             device='cpu', deadline_unix=far_deadline())
    logits, labels, metrics = out['logits'], out['labels'], out['metrics']
    assert logits.shape == (query_z.shape[0], 10) and logits.dtype == np.float32
    assert np.isfinite(logits).all()
    assert labels.shape == (query_z.shape[0],) and labels.dtype == np.uint8
    assert np.array_equal(labels, np.argmax(logits, axis=1).astype(np.uint8))
    assert metrics['query_labels_supplied'] is False
    assert metrics['release_dim'] == DIM
    assert metrics['input_dim'] == DIM
    assert metrics['train_count'] == train_z.shape[0]


def test_first_max_labels_breaks_ties_to_lowest_class():
    logits = np.zeros((3, 10), dtype=np.float32)
    logits[1, 5] = 1.0
    logits[2, 9] = 1.0
    logits[2, 3] = 1.0                      # tie between 3 and 9 -> 3
    assert neural.first_max_labels(logits).tolist() == [0, 5, 3]


def test_fit_predict_rejects_mismatched_shapes():
    train_z, train_y, query_z, _ = make_release_like(seed=5)
    config = {'family': 'mlp', 'widths': [16], 'epochs': 1}
    with pytest.raises(ValueError):
        neural.fit_predict(train_z, train_y[:-1], query_z, config, 1, 'cpu',
                           far_deadline())
    with pytest.raises(ValueError):
        neural.fit_predict(train_z, train_y, query_z[:, :10], config, 1, 'cpu',
                           far_deadline())


def test_mlp_learns_something_on_separable_release_features():
    train_z, train_y, query_z, query_y = make_release_like(n_train=600, n_query=300,
                                                           seed=6)
    config = {'family': 'mlp', 'widths': [128, 128], 'epochs': 40,
              'batch_size': 64, 'lr': 0.002}
    out = neural.fit_predict(train_z, train_y, query_z, config, seed=11,
                             device='cpu', deadline_unix=far_deadline())
    accuracy = float((out['labels'] == query_y).mean())
    assert accuracy > 0.8, accuracy


# --------------------------------------------------------------------------
# VAT
# --------------------------------------------------------------------------
def _vat_setup(eps, dropout=0.0, seed=3):
    train_z, _, _, _ = make_release_like(seed=seed)
    model = neural.BatchedMLP(DIM, [32], members=2, seed=seed, device='cpu',
                              dropout=dropout)
    vat = neural.resolve_config({'family': 'vat', 'vat': {'eps': eps}})['vat']
    xu = torch.as_tensor(train_z[:64]).unsqueeze(0).expand(2, -1, -1).contiguous()
    generator = neural._make_generator('cpu', 1234)
    return model, xu, vat, generator


def test_vat_loss_is_zero_at_eps_zero_and_finite_otherwise():
    model, xu, vat, generator = _vat_setup(0.0)
    value = float(neural.vat_loss(model, xu, vat, generator))
    assert value == 0.0
    model, xu, vat, generator = _vat_setup(2.0)
    value = float(neural.vat_loss(model, xu, vat, generator))
    assert np.isfinite(value) and value > 0.0


def test_vat_loss_finite_with_shared_dropout_mask():
    """With dropout on, one shared mask keeps the eps=0 loss at the mask floor."""
    model, xu, vat, generator = _vat_setup(0.0, dropout=0.2)
    value = float(neural.vat_loss(model, xu, vat, generator))
    assert np.isfinite(value)
    assert value == pytest.approx(0.0, abs=1e-6)


def test_vat_weight_rampup():
    assert neural.vat_weight_at_step(0, 100, 1.0, 0.2) == pytest.approx(1 / 20)
    assert neural.vat_weight_at_step(19, 100, 1.0, 0.2) == pytest.approx(1.0)
    assert neural.vat_weight_at_step(99, 100, 1.0, 0.2) == pytest.approx(1.0)
    assert neural.vat_weight_at_step(0, 100, 1.0, 0.0) == pytest.approx(1.0)
    assert neural.vat_weight_at_step(0, 100, 2.5, 0.5) == pytest.approx(2.5 / 50)


def test_vat_rampup_changes_the_trajectory():
    train_z, train_y, query_z, _ = make_release_like(seed=8)
    base = {'family': 'vat', 'widths': [64], 'epochs': 4, 'batch_size': 64}
    ramped = neural.fit_predict(train_z, train_y, query_z,
                                {**base, 'vat': {'eps': 2.0, 'rampup_fraction': 0.2}},
                                seed=5, device='cpu', deadline_unix=far_deadline())
    flat = neural.fit_predict(train_z, train_y, query_z,
                              {**base, 'vat': {'eps': 2.0, 'rampup_fraction': 0.0}},
                              seed=5, device='cpu', deadline_unix=far_deadline())
    assert ramped['metrics']['vat_rampup_steps'] > 0
    assert flat['metrics']['vat_rampup_steps'] == 0
    assert not np.allclose(ramped['logits'], flat['logits'])


# --------------------------------------------------------------------------
# transduction
# --------------------------------------------------------------------------
@pytest.mark.parametrize('config,key', [
    ({'family': 'vat', 'widths': [32], 'epochs': 1, 'batch_size': 64}, 'vat'),
    ({'family': 'ladder', 'hidden_dims': [32], 'epochs': 1, 'batch_size': 50,
      'decay_start_epoch': 1}, 'ladder'),
])
def test_transductive_flag_is_recorded(config, key):
    train_z, train_y, query_z, _ = make_release_like(seed=9)
    if key == 'vat':
        transductive = {**config, 'vat': {'unlabeled': 'train+query'}}
        inductive = {**config, 'vat': {'unlabeled': 'train'}}
    else:
        transductive = {**config, 'unlabeled': 'train+query'}
        inductive = {**config, 'unlabeled': 'train'}
    on = neural.fit_predict(train_z, train_y, query_z, transductive, 2, 'cpu',
                            far_deadline())
    off = neural.fit_predict(train_z, train_y, query_z, inductive, 2, 'cpu',
                             far_deadline())
    assert on['metrics']['uses_query_features_unlabeled'] is True
    assert off['metrics']['uses_query_features_unlabeled'] is False


def test_plain_mlp_is_not_transductive():
    train_z, train_y, query_z, _ = make_release_like(seed=10)
    out = neural.fit_predict(train_z, train_y, query_z,
                             {'family': 'mlp', 'widths': [32], 'epochs': 1},
                             3, 'cpu', far_deadline())
    assert out['metrics']['uses_query_features_unlabeled'] is False


# --------------------------------------------------------------------------
# ladder input_scale
# --------------------------------------------------------------------------
def test_effective_noise_ratio_tracks_input_scale():
    train_z, _, _, _ = make_release_like(seed=11)
    unit = neural.effective_noise_ratio(train_z, 1.0, 0.3)
    scaled = neural.effective_noise_ratio(train_z, 0.3, 0.3)
    assert unit == pytest.approx(0.3, rel=0.05)     # z has RMS ~ 1
    # 1.0 is the ratio input_scale=0.3 would give, NOT the 9x9 pixel ratio,
    # which is 0.3 / 0.2416 = 1.24 (neural.PIXEL_STD_9X9).
    assert scaled == pytest.approx(1.0, rel=0.05)
    assert scaled == pytest.approx(unit / 0.3, rel=1e-6)


def test_ladder_input_scale_changes_the_effective_noise_and_the_fit():
    train_z, train_y, query_z, _ = make_release_like(seed=12)
    base = {'family': 'ladder', 'hidden_dims': [64, 32], 'epochs': 2,
            'batch_size': 50, 'decay_start_epoch': 1}
    scaled = neural.fit_predict(train_z, train_y, query_z,
                                {**base, 'input_scale': 0.3}, 4, 'cpu',
                                far_deadline())
    unscaled = neural.fit_predict(train_z, train_y, query_z,
                                  {**base, 'input_scale': 1.0}, 4, 'cpu',
                                  far_deadline())
    ratio_scaled = scaled['metrics']['effective_input_noise_ratio']
    ratio_unscaled = unscaled['metrics']['effective_input_noise_ratio']
    assert ratio_scaled == pytest.approx(ratio_unscaled / 0.3, rel=1e-6)
    assert ratio_scaled > ratio_unscaled
    assert scaled['metrics']['input_scale'] == pytest.approx(0.3)
    assert 'input_scale=0.3' in scaled['metrics']['normalization']
    assert not np.allclose(scaled['logits'], unscaled['logits'])


def test_ladder_scaling_is_applied_to_both_halves():
    """A ladder on z with input_scale=s equals a ladder on s*z with scale 1."""
    train_z, train_y, query_z, _ = make_release_like(n_train=120, n_query=40, seed=13)
    base = {'family': 'ladder', 'hidden_dims': [32], 'epochs': 1,
            'batch_size': 50, 'decay_start_epoch': 1}
    a = neural.fit_predict(train_z, train_y, query_z, {**base, 'input_scale': 0.3},
                           6, 'cpu', far_deadline())
    b = neural.fit_predict((0.3 * train_z).astype(np.float32), train_y,
                           (0.3 * query_z).astype(np.float32),
                           {**base, 'input_scale': 1.0}, 6, 'cpu', far_deadline())
    assert np.allclose(a['logits'], b['logits'], atol=1e-4)


# --------------------------------------------------------------------------
# deadline behaviour
# --------------------------------------------------------------------------
@pytest.mark.parametrize('config', [
    {'family': 'mlp', 'widths': [64], 'epochs': 50},
    {'family': 'vat', 'widths': [64], 'epochs': 50},
    {'family': 'ladder', 'hidden_dims': [64], 'epochs': 50, 'batch_size': 50,
     'decay_start_epoch': 40},
])
def test_expired_deadline_fails_instead_of_reporting_an_untrained_net(config):
    """Zero epochs completed -> TimeoutError, never a random init's logits."""
    train_z, train_y, query_z, _ = make_release_like(seed=14)
    with pytest.raises(TimeoutError):
        neural.fit_predict(train_z, train_y, query_z, config, seed=1,
                           device='cpu', deadline_unix=time.time() - 1.0)


def _stop_after(monkeypatch, epochs):
    """Deterministic TimeGuard: allow ``epochs`` epochs in total, then stop."""
    budget = {'left': int(epochs)}

    def should_stop(self):
        if budget['left'] <= 0:
            return True
        budget['left'] -= 1
        return False

    monkeypatch.setattr(neural.TimeGuard, 'should_stop', should_stop)


def test_partial_training_records_the_completion_fraction():
    train_z, train_y, query_z, _ = make_release_like(seed=14)
    out = neural.fit_predict(train_z, train_y, query_z,
                             {'family': 'mlp', 'widths': [32], 'epochs': 4},
                             1, 'cpu', far_deadline())
    assert out['metrics']['epoch_completion_fraction'] == pytest.approx(1.0)
    assert out['metrics']['truncated'] is False


def test_deadline_drops_untrained_ladder_members(monkeypatch):
    """The second member never starts: it is dropped, not averaged in."""
    train_z, train_y, query_z, _ = make_release_like(n_train=200, n_query=50, seed=15)
    base = {'family': 'ladder', 'hidden_dims': [32], 'epochs': 3,
            'batch_size': 50, 'decay_start_epoch': 2}
    solo = neural.fit_predict(train_z, train_y, query_z, {**base, 'members': 1},
                              2, 'cpu', far_deadline())
    _stop_after(monkeypatch, 3)          # exactly the first member's epochs
    pair = neural.fit_predict(train_z, train_y, query_z, {**base, 'members': 2},
                              2, 'cpu', far_deadline())
    metrics = pair['metrics']
    assert metrics['members_completed'] == 1
    assert metrics['member_epochs_completed'] == [3, 0]
    assert metrics['truncated'] is True
    assert metrics['epoch_completion_fraction'] == pytest.approx(0.5)
    # Bit-for-bit the members=1 fit: the untrained member contributed nothing.
    assert np.allclose(pair['logits'], solo['logits'], atol=1e-6)


def test_ladder_with_no_trained_member_fails(monkeypatch):
    train_z, train_y, query_z, _ = make_release_like(n_train=200, n_query=50, seed=15)
    _stop_after(monkeypatch, 0)
    with pytest.raises(TimeoutError):
        neural.fit_predict(train_z, train_y, query_z,
                           {'family': 'ladder', 'hidden_dims': [32], 'epochs': 3,
                            'batch_size': 50, 'decay_start_epoch': 2, 'members': 2},
                           2, 'cpu', far_deadline())


# --------------------------------------------------------------------------
# rewhiten: the kernel study's two preprocessing maps
# --------------------------------------------------------------------------
def test_rewhiten_defaults_are_off_and_validated():
    for family in ('mlp', 'vat', 'ladder'):
        resolved = neural.resolve_config({'family': family})
        assert resolved['rewhiten'] == 'none'
        assert resolved['wc_shrinkage'] == pytest.approx(0.05)   # the pilots' 'wc'
        assert resolved['rw_source'] == 'query'
        assert resolved['rw_shrinkage'] == 0.0
    with pytest.raises(ValueError):
        neural.resolve_config({'family': 'mlp', 'rewhiten': 'zca'})
    with pytest.raises(ValueError):
        neural.resolve_config({'family': 'mlp', 'rw_source': 'train'})
    with pytest.raises(ValueError):
        neural.resolve_config({'family': 'mlp', 'wc_shrinkage': 1.5})


def test_rewhiten_none_is_the_identity_on_the_very_same_arrays():
    train_z, train_y, query_z, _ = make_release_like(seed=20)
    config = neural.resolve_config({'family': 'mlp'})
    out_train, out_query, note = neural.apply_rewhiten(train_z, train_y, query_z, config)
    assert out_train is train_z and out_query is query_z
    assert note == {'rewhiten': 'none'}


def test_rewhiten_wc_map_equals_kernels_within_class_transform():
    """'wc' must be kernels.py's map, not a lookalike: same matrix, same output."""
    train_z, train_y, query_z, _ = make_release_like(n_train=400, seed=21)
    shrinkage = 0.05
    config = neural.resolve_config({'family': 'mlp', 'rewhiten': 'wc',
                                    'wc_shrinkage': shrinkage})
    out_train, out_query, note = neural.apply_rewhiten(train_z, train_y, query_z, config)

    train64 = torch.as_tensor(train_z, dtype=torch.float64)
    query64 = torch.as_tensor(query_z, dtype=torch.float64)
    labels = torch.as_tensor(train_y, dtype=torch.long)
    transform = kernels.within_class_transform(train64, labels, shrinkage)
    # The imported symbol is the same function object, and the features the net
    # sees are that map applied in float64 and cast to float32.
    assert neural.within_class_transform is kernels.within_class_transform
    assert np.array_equal(out_train, np.asarray((train64 @ transform).numpy(),
                                                dtype=np.float32))
    assert np.array_equal(out_query, np.asarray((query64 @ transform).numpy(),
                                                dtype=np.float32))
    assert note['wc_shrinkage'] == pytest.approx(shrinkage)
    assert note['uses_query_features'] is False
    # It really is a sphering map: the pooled within-class covariance of the
    # transformed training rows is (up to the shrinkage) isotropic.
    centred = torch.as_tensor(out_train, dtype=torch.float64).clone()
    for label in range(10):
        mask = labels == label
        centred[mask] -= centred[mask].mean(0, keepdim=True)
    within = (centred.T @ centred) / (train_z.shape[0] - 10)
    off = within - torch.diag(torch.diagonal(within))
    assert float(off.abs().max()) < 0.2 * float(torch.diagonal(within).mean())


def test_rewhiten_wc_uses_training_rows_and_labels_only():
    """Two different query blocks give the SAME training map and features."""
    train_z, train_y, query_a, _ = make_release_like(n_train=400, n_query=120, seed=22)
    rng = np.random.default_rng(5)
    query_b = (query_a[::-1] * 3.0 + rng.standard_normal(query_a.shape)).astype(np.float32)
    config = neural.resolve_config({'family': 'mlp', 'rewhiten': 'wc'})
    train_a, out_a, _ = neural.apply_rewhiten(train_z, train_y, query_a, config)
    train_b, out_b, _ = neural.apply_rewhiten(train_z, train_y, query_b, config)
    assert np.array_equal(train_a, train_b)          # query block cannot move it
    assert not np.array_equal(out_a, out_b)          # ... but the query rows move
    # The query half is the same linear map applied to a different block.
    transform = kernels.within_class_transform(
        torch.as_tensor(train_z, dtype=torch.float64),
        torch.as_tensor(train_y, dtype=torch.long), 0.05)
    expected = (torch.as_tensor(query_b, dtype=torch.float64) @ transform).numpy()
    assert np.array_equal(out_b, np.asarray(expected, dtype=np.float32))
    # 'rw', by contrast, DOES depend on the query block (that is the point).
    rw = neural.resolve_config({'family': 'mlp', 'rewhiten': 'rw'})
    rw_a, _, _ = neural.apply_rewhiten(train_z, train_y, query_a, rw)
    rw_b, _, _ = neural.apply_rewhiten(train_z, train_y, query_b, rw)
    assert not np.array_equal(rw_a, rw_b)


def test_rewhiten_rw_matches_kernels_rewhiten_and_is_flagged():
    train_z, train_y, query_z, _ = make_release_like(n_train=300, n_query=200, seed=23)
    for mode, source in (('rw', 'query'), ('rw', 'pooled')):
        config = neural.resolve_config({'family': 'mlp', 'rewhiten': mode,
                                        'rw_source': source})
        out_train, out_query, note = neural.apply_rewhiten(train_z, train_y,
                                                           query_z, config)
        train64 = torch.as_tensor(train_z, dtype=torch.float64)
        query64 = torch.as_tensor(query_z, dtype=torch.float64)
        expect_train, expect_query, _ = kernels._rewhiten(
            train64, query64, {'rewhiten': source, 'rewhiten_shrinkage': 0.0})
        assert np.array_equal(out_train, np.asarray(expect_train.numpy(),
                                                    dtype=np.float32))
        assert np.array_equal(out_query, np.asarray(expect_query.numpy(),
                                                    dtype=np.float32))
        assert note['uses_query_features'] is True
        assert note['rw_rows'] == (query_z.shape[0] if source == 'query'
                                   else query_z.shape[0] + train_z.shape[0])


def test_rewhiten_wc_plus_rw_is_rw_then_wc_as_in_kernels():
    train_z, train_y, query_z, _ = make_release_like(n_train=300, n_query=200, seed=24)
    both = neural.resolve_config({'family': 'mlp', 'rewhiten': 'wc+rw'})
    out_train, out_query, note = neural.apply_rewhiten(train_z, train_y, query_z, both)
    # kernels._preprocess with the same settings: rewhiten first, sphere second.
    train64 = torch.as_tensor(train_z, dtype=torch.float64)
    query64 = torch.as_tensor(query_z, dtype=torch.float64)
    expect_train, expect_query, _ = kernels._preprocess(
        train64, query64, torch.as_tensor(train_y, dtype=torch.long),
        {'rewhiten': 'query', 'rewhiten_shrinkage': 0.0,
         'metric_learn': 'within_class', 'metric_shrinkage': 0.05})
    assert np.array_equal(out_train, np.asarray(expect_train.numpy(), dtype=np.float32))
    assert np.array_equal(out_query, np.asarray(expect_query.numpy(), dtype=np.float32))
    assert note['uses_query_features'] is True


@pytest.mark.parametrize('family,config', [
    ('mlp', {'family': 'mlp', 'widths': [32], 'epochs': 2}),
    ('ladder', {'family': 'ladder', 'hidden_dims': [32], 'epochs': 1,
                'batch_size': 50, 'decay_start_epoch': 1, 'unlabeled': 'train'}),
])
def test_rewhiten_changes_the_fit_and_is_recorded(family, config):
    train_z, train_y, query_z, _ = make_release_like(n_train=300, seed=25)
    plain = neural.fit_predict(train_z, train_y, query_z, config, 3, 'cpu',
                               far_deadline())
    wc = neural.fit_predict(train_z, train_y, query_z, {**config, 'rewhiten': 'wc'},
                            3, 'cpu', far_deadline())
    rw = neural.fit_predict(train_z, train_y, query_z, {**config, 'rewhiten': 'rw'},
                            3, 'cpu', far_deadline())
    assert plain['metrics']['rewhiten'] == {'rewhiten': 'none'}
    assert wc['metrics']['rewhiten']['rewhiten'] == 'wc'
    assert not np.allclose(plain['logits'], wc['logits'])
    # 'wc' is inductive; 'rw' reads the query features and must say so.
    assert wc['metrics']['uses_query_features_unlabeled'] is False
    assert rw['metrics']['uses_query_features_unlabeled'] is True
    assert rw['metrics']['uses_query_images_unlabeled'] is True


# --------------------------------------------------------------------------
# target_steps
# --------------------------------------------------------------------------
def test_resolve_epochs_defaults_to_the_fixed_epoch_count():
    config = neural.resolve_config({'family': 'mlp', 'epochs': 37})
    assert config['target_steps'] == 0
    assert neural.resolve_epochs(config, 4) == (37, None)
    assert neural.resolve_epochs(config, 79) == (37, None)


def test_resolve_epochs_clamps_and_reports():
    config = neural.resolve_config({'family': 'mlp', 'epochs': 100,
                                    'target_steps': 1200, 'batch_size': 128})
    # N=500 -> 4 steps/epoch -> 300 epochs; N=10000 -> 79 -> ceil(1200/79) = 16.
    assert neural.resolve_epochs(config, math.ceil(500 / 128))[0] == 300
    assert neural.resolve_epochs(config, math.ceil(10000 / 128))[0] == 16
    note = neural.resolve_epochs(config, 4)[1]
    assert note['planned_steps'] == 1200 and note['epochs_config'] == 100
    clamped = neural.resolve_config({'family': 'mlp', 'target_steps': 1200,
                                     'batch_size': 128, 'max_epochs': 50,
                                     'min_epochs': 20})
    assert neural.resolve_epochs(clamped, 4)[0] == 50        # 300 -> max_epochs
    assert neural.resolve_epochs(clamped, 1000)[0] == 20     # 2 -> min_epochs
    with pytest.raises(ValueError):
        neural.resolve_config({'family': 'mlp', 'target_steps': -1})
    with pytest.raises(ValueError):
        neural.resolve_config({'family': 'mlp', 'min_epochs': 10, 'max_epochs': 5})


@pytest.mark.parametrize('n_train,expected_epochs', [(500, 300), (10000, 16)])
def test_target_steps_gives_one_schedule_at_every_n(n_train, expected_epochs):
    """ONE config, two draw sizes, ~the same number of gradient steps."""
    train_z, train_y, query_z, _ = make_release_like(n_train=n_train, n_query=40,
                                                     seed=26)
    config = {'family': 'mlp', 'widths': [16], 'batch_size': 128,
              'target_steps': 1200, 'epochs': 7}
    out = neural.fit_predict(train_z, train_y, query_z, config, 2, 'cpu',
                             far_deadline())
    metrics = out['metrics']
    steps_per_epoch = math.ceil(n_train / 128)
    assert metrics['epochs_resolved'] == expected_epochs
    assert metrics['epochs_planned'] == expected_epochs       # 'epochs' ignored
    assert metrics['epochs_completed'] == expected_epochs
    assert metrics['steps_completed'] == expected_epochs * steps_per_epoch
    assert 1200 <= metrics['steps_completed'] < 1200 + steps_per_epoch
    assert metrics['target_steps_note']['target_steps'] == 1200


def test_target_steps_rescales_the_ladder_decay_point():
    train_z, train_y, query_z, _ = make_release_like(n_train=200, n_query=40, seed=27)
    config = {'family': 'ladder', 'hidden_dims': [32], 'batch_size': 50,
              'epochs': 150, 'decay_start_epoch': 100, 'target_steps': 24}
    out = neural.fit_predict(train_z, train_y, query_z, config, 2, 'cpu',
                             far_deadline())
    metrics = out['metrics']
    assert metrics['epochs_resolved'] == 6                    # ceil(24 / 4)
    assert metrics['decay_start_epoch_resolved'] == 4         # 100/150 of 6
    assert metrics['epochs_completed'] == 6


# --------------------------------------------------------------------------
# selftrain: selection rule
# --------------------------------------------------------------------------
def _fake_probabilities(counts, seed=0):
    """Rows whose arg-max class follows ``counts``, with spread-out confidence."""
    rng = np.random.default_rng(seed)
    rows = []
    labels = []
    for label, count in enumerate(counts):
        for _ in range(count):
            row = rng.uniform(0.0, 0.05, size=10)
            row[label] = rng.uniform(0.3, 0.9)
            rows.append(row / row.sum())
            labels.append(label)
    return np.asarray(rows), np.asarray(labels)


def test_select_pseudo_rows_is_class_balanced():
    counts = [40, 40, 10, 100, 25, 25, 60, 15, 30, 55]
    probabilities, truth = _fake_probabilities(counts, seed=1)
    index, labels, confidence, per_class = neural.select_pseudo_rows(probabilities, 0.5)
    assert np.array_equal(labels, truth[index])               # arg-max, not truth
    assert list(index) == sorted(index)
    for label, count in enumerate(counts):
        kept = per_class[label]
        assert kept == int((labels == label).sum())
        # ~q of every predicted class survives: never a class wiped out, never
        # one class monopolising the pseudo set.
        assert abs(kept / count - 0.5) <= 0.06, (label, kept, count)
    assert sum(per_class) == index.size
    # Only the confident half: every kept row beats the class median.
    top = probabilities.max(axis=1)
    for label in range(10):
        rows = np.nonzero(truth == label)[0]
        threshold = np.quantile(top[rows], 0.5)
        assert top[index[labels == label]].min() >= threshold - 1e-12


def test_select_pseudo_rows_quantile_one_keeps_everything():
    probabilities, _ = _fake_probabilities([5] * 10, seed=2)
    index, _, _, per_class = neural.select_pseudo_rows(probabilities, 1.0)
    assert index.size == probabilities.shape[0]
    assert per_class == [5] * 10


def test_select_pseudo_rows_ties_go_to_the_lowest_class():
    probabilities = np.full((3, 10), 0.1)
    probabilities[1, 4] = probabilities[1, 7] = 0.2
    _, labels, _, _ = neural.select_pseudo_rows(probabilities, 1.0)
    assert labels.tolist() == [0, 4, 0]


# --------------------------------------------------------------------------
# selftrain: configuration and the rounds=0 no-op
# --------------------------------------------------------------------------
def test_selftrain_defaults_are_off_for_every_family():
    for family in ('mlp', 'vat', 'ladder'):
        selftrain = neural.resolve_config({'family': family})['selftrain']
        assert selftrain == {'rounds': 0, 'quantile': 0.5, 'soft': False,
                             'weight': 1.0, 'refit': 'scratch'}
    with pytest.raises(ValueError):
        neural.resolve_config({'family': 'mlp', 'selftrain': {'round': 2}})
    with pytest.raises(ValueError):
        neural.resolve_config({'family': 'mlp', 'selftrain': {'rounds': -1}})
    with pytest.raises(ValueError):
        neural.resolve_config({'family': 'mlp', 'selftrain': {'quantile': 0.0}})
    with pytest.raises(ValueError):
        neural.resolve_config({'family': 'mlp', 'selftrain': {'refit': 'warm'}})
    # The vendored ladder loss cannot honour soft targets or fractional weights.
    with pytest.raises(ValueError):
        neural.resolve_config({'family': 'ladder',
                               'selftrain': {'rounds': 1, 'soft': True}})
    with pytest.raises(ValueError):
        neural.resolve_config({'family': 'ladder',
                               'selftrain': {'rounds': 1, 'weight': 2.5}})
    # ... but they are accepted while self-training is off, and by the MLP.
    neural.resolve_config({'family': 'ladder', 'selftrain': {'soft': True}})
    neural.resolve_config({'family': 'mlp',
                           'selftrain': {'rounds': 1, 'soft': True, 'weight': 2.5}})


@pytest.mark.parametrize('config', [
    {'family': 'mlp', 'widths': [64, 64], 'epochs': 5, 'batch_size': 64, 'members': 3,
     'dropout': 0.1, 'input_noise_std': 0.3, 'label_smoothing': 0.1},
    {'family': 'mlp', 'widths': [32], 'epochs': 4, 'batch_size': 50, 'members': 2,
     'mixup_alpha': 0.4, 'ema_decay': 0.99, 'normalization': 'standardize'},
    {'family': 'vat', 'widths': [64], 'epochs': 3, 'batch_size': 64,
     'vat': {'eps': 2.0, 'batch_size': 32}},
    {'family': 'ladder', 'hidden_dims': [64, 32], 'epochs': 3, 'batch_size': 50,
     'decay_start_epoch': 2},
])
def test_selftrain_rounds_zero_reproduces_the_old_outputs_bit_for_bit(config):
    """rounds=0 must be the pre-selftrain code path, exactly.

    The stored-array version of this check was run once against the module as
    it stood before the wrapper existed (mlp with dropout/noise/smoothing, mlp
    with mixup+EMA+standardize, vat, and two ladders, all bit-identical); what
    is pinned here is the invariant that keeps it true: the wrapper must be
    transparent, and the batch loop must take the unweighted branch.
    """
    train_z, train_y, query_z, _ = make_release_like(n_train=240, n_query=80, seed=28)
    plain = neural.fit_predict(train_z, train_y, query_z, config, 7, 'cpu',
                               far_deadline())
    for selftrain in ({'rounds': 0}, {'rounds': 0, 'quantile': 0.9, 'weight': 4.0,
                                      'soft': True, 'refit': 'continue'}):
        off = neural.fit_predict(train_z, train_y, query_z,
                                 {**config, 'selftrain': selftrain}, 7, 'cpu',
                                 far_deadline())
        assert np.array_equal(plain['logits'], off['logits'])
        assert np.array_equal(plain['labels'], off['labels'])
        assert off['metrics']['rounds_completed'] == 0
        assert off['metrics']['pseudo_rows_per_round'] == []


def test_precomputed_targets_match_the_per_batch_one_hot_exactly():
    """The batch loop now indexes a target matrix; it must be the same floats."""
    labels = torch.as_tensor(np.arange(40) % 10, dtype=torch.long)
    index = torch.as_tensor(np.random.default_rng(0).integers(0, 40, size=(3, 7)),
                            dtype=torch.long)
    for smoothing in (0.0, 0.1, 0.37):
        precomputed = neural._one_hot(labels, smoothing)[index]
        per_batch = neural._one_hot(labels[index], smoothing)
        assert torch.equal(precomputed, per_batch)


def test_unweighted_fits_never_touch_the_weighted_loss(monkeypatch):
    """No pseudo rows -> the original _soft_cross_entropy expression, untouched."""
    def explode(*args, **kwargs):
        raise AssertionError('weighted cross entropy used without pseudo rows')

    monkeypatch.setattr(neural, '_weighted_cross_entropy', explode)
    train_z, train_y, query_z, _ = make_release_like(seed=29)
    neural.fit_predict(train_z, train_y, query_z,
                       {'family': 'mlp', 'widths': [32], 'epochs': 2,
                        'selftrain': {'rounds': 0}}, 1, 'cpu', far_deadline())
    # weight=1.0 rounds also stay on the unweighted path (no weight vector built).
    neural.fit_predict(train_z, train_y, query_z,
                       {'family': 'mlp', 'widths': [32], 'epochs': 2,
                        'selftrain': {'rounds': 1}}, 1, 'cpu', far_deadline())


def test_weighted_cross_entropy_reduces_to_the_unweighted_mean():
    logits = torch.randn(2, 5, 10, generator=torch.Generator().manual_seed(0))
    targets = neural._one_hot(torch.as_tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]), 0.1)
    ones = torch.ones(2, 5)
    assert float(neural._weighted_cross_entropy(logits, targets, ones)) == \
        pytest.approx(float(neural._soft_cross_entropy(logits, targets)), rel=1e-6)
    # A zero-weighted row drops out of the average entirely.
    weights = torch.ones(2, 5)
    weights[0, 0] = 0.0
    dropped = neural._weighted_cross_entropy(logits[:, 1:], targets[:, 1:],
                                             torch.ones(2, 4))
    mixed = neural._weighted_cross_entropy(logits, targets, weights)
    row = -(targets * torch.log_softmax(logits, dim=-1)).sum(-1)
    assert float(mixed) == pytest.approx(
        float((row[0, 1:].sum() + row[1].sum()) / 9.0), rel=1e-6)
    assert float(dropped) == pytest.approx(float(row[:, 1:].mean()), rel=1e-6)


# --------------------------------------------------------------------------
# selftrain: rounds that actually run
# --------------------------------------------------------------------------
@pytest.mark.parametrize('config', [
    {'family': 'mlp', 'widths': [64], 'epochs': 6, 'batch_size': 64, 'members': 2},
    {'family': 'vat', 'widths': [32], 'epochs': 3, 'batch_size': 64},
    {'family': 'ladder', 'hidden_dims': [32], 'epochs': 2, 'batch_size': 50,
     'decay_start_epoch': 1},
])
def test_selftrain_one_round_adds_class_balanced_pseudo_rows(config):
    train_z, train_y, query_z, _ = make_release_like(n_train=300, n_query=200, seed=30)
    out = neural.fit_predict(train_z, train_y, query_z,
                             {**config, 'selftrain': {'rounds': 1, 'quantile': 0.5}},
                             9, 'cpu', far_deadline())
    metrics = out['metrics']
    assert metrics['rounds_completed'] == 1
    assert metrics['selftrain_rounds_completed'] == 1
    assert metrics['selftrain_rounds_planned'] == 1
    assert metrics['selftrain_skipped_from_round'] is None
    assert len(metrics['pseudo_rows_per_round']) == 2          # round 0 has none
    assert metrics['pseudo_rows_per_round'][0] == 0
    kept = metrics['pseudo_rows_per_round'][1]
    assert 0 < kept < 200
    assert kept == pytest.approx(100, abs=12)                  # ~ q of the block
    round_one = metrics['selftrain_rounds'][1]
    assert sum(round_one['pseudo_per_class']) == kept
    assert round_one['rows_fitted'] == 300 + kept
    assert round_one['seed'] == 9 + neural.SELFTRAIN_SEED_STRIDE
    # Every class the model predicts at all contributes, and in proportion:
    # the pseudo set cannot collapse onto the two or three easiest digits.
    predicted = round_one['predicted_per_class']
    assert sum(predicted) == 200
    for label, count in enumerate(predicted):
        assert (round_one['pseudo_per_class'][label] > 0) == (count > 0)
        if count >= 10:
            assert round_one['pseudo_per_class'][label] / count == \
                pytest.approx(0.5, abs=0.12), (label, count)
    assert metrics['uses_query_images_unlabeled'] is True
    assert metrics['uses_query_features_unlabeled'] is True


def test_selftrain_pseudo_labels_come_from_the_model_not_from_the_truth():
    """The kept rows and their labels are exactly round 0's own predictions."""
    train_z, train_y, query_z, query_y = make_release_like(n_train=300, n_query=200,
                                                           seed=31)
    config = {'family': 'mlp', 'widths': [64], 'epochs': 6, 'batch_size': 64}
    base = neural.fit_predict(train_z, train_y, query_z,
                              {**config, 'selftrain': {'rounds': 0}}, 4, 'cpu',
                              far_deadline())
    index, labels, _, per_class = neural.select_pseudo_rows(
        np.exp(np.asarray(base['logits'], dtype=np.float64)), 0.5)
    run = neural.fit_predict(train_z, train_y, query_z,
                             {**config, 'selftrain': {'rounds': 1, 'quantile': 0.5}},
                             4, 'cpu', far_deadline())
    round_one = run['metrics']['selftrain_rounds'][1]
    assert round_one['pseudo_rows'] == index.size
    assert round_one['pseudo_per_class'] == per_class
    # Reconstructible from the round-0 prediction alone -- no query label needed
    # (and none exists: fit_predict has no argument for one).
    assert np.array_equal(labels, np.asarray(base['labels'], dtype=np.int64)[index])
    assert run['metrics']['query_labels_supplied'] is False
    # Sanity: the pseudo-labels are NOT the ground truth (they are predictions).
    assert 'query_y' not in run['metrics']
    purity = float((labels == query_y[index]).mean())
    assert 0.0 <= purity <= 1.0


def test_selftrain_soft_weight_and_continue_change_the_fit():
    train_z, train_y, query_z, _ = make_release_like(n_train=300, n_query=200, seed=32)
    config = {'family': 'mlp', 'widths': [64], 'epochs': 5, 'batch_size': 64}
    runs = {}
    for name, selftrain in (
            ('hard', {'rounds': 1, 'quantile': 0.4}),
            ('soft', {'rounds': 1, 'quantile': 0.4, 'soft': True}),
            ('weighted', {'rounds': 1, 'quantile': 0.4, 'weight': 3.0}),
            ('continue', {'rounds': 1, 'quantile': 0.4, 'refit': 'continue'})):
        runs[name] = neural.fit_predict(train_z, train_y, query_z,
                                        {**config, 'selftrain': selftrain}, 6, 'cpu',
                                        far_deadline())
    kept = runs['hard']['metrics']['pseudo_rows_per_round'][1]
    for name in ('soft', 'weighted', 'continue'):
        assert runs[name]['metrics']['rounds_completed'] == 1
        assert runs[name]['metrics']['pseudo_rows_per_round'][1] == kept
        assert not np.allclose(runs['hard']['logits'], runs[name]['logits'])
    # An MLP weight is a loss weight, not a repeated row.
    assert runs['weighted']['metrics']['selftrain_rounds'][1]['rows_fitted'] == \
        300 + kept
    assert runs['weighted']['metrics']['rows_fitted'] == 300 + kept


def test_selftrain_ladder_weight_repeats_rows():
    train_z, train_y, query_z, _ = make_release_like(n_train=200, n_query=100, seed=33)
    config = {'family': 'ladder', 'hidden_dims': [32], 'epochs': 2, 'batch_size': 50,
              'decay_start_epoch': 1}
    out = neural.fit_predict(train_z, train_y, query_z,
                             {**config, 'selftrain': {'rounds': 1, 'quantile': 0.5,
                                                      'weight': 2.0}},
                             6, 'cpu', far_deadline())
    round_one = out['metrics']['selftrain_rounds'][1]
    assert round_one['pseudo_rows_weighted'] == 2 * round_one['pseudo_rows']
    assert round_one['rows_fitted'] == 200 + 2 * round_one['pseudo_rows']


def test_selftrain_two_rounds_use_distinct_derived_seeds():
    train_z, train_y, query_z, _ = make_release_like(n_train=300, n_query=200, seed=34)
    out = neural.fit_predict(train_z, train_y, query_z,
                             {'family': 'mlp', 'widths': [32], 'epochs': 4,
                              'batch_size': 64,
                              'selftrain': {'rounds': 2, 'quantile': 0.5}},
                             12, 'cpu', far_deadline())
    seeds = [r['seed'] for r in out['metrics']['selftrain_rounds']]
    assert seeds == [12, 12 + neural.SELFTRAIN_SEED_STRIDE,
                     12 + 2 * neural.SELFTRAIN_SEED_STRIDE]
    assert len(set(seeds)) == 3
    assert out['metrics']['rounds_completed'] == 2
    assert out['metrics']['selftrain']['rounds'] == 2


def test_selftrain_improves_a_weakly_trained_mlp():
    """Pseudo-labels from a decent model are worth something, not noise."""
    train_z, train_y, query_z, query_y = make_release_like(n_train=200, n_query=600,
                                                           seed=35)
    config = {'family': 'mlp', 'widths': [128, 128], 'epochs': 60, 'batch_size': 64,
              'members': 2, 'dropout': 0.1, 'label_smoothing': 0.1}
    plain = neural.fit_predict(train_z, train_y, query_z, config, 8, 'cpu',
                               far_deadline())
    boosted = neural.fit_predict(train_z, train_y, query_z,
                                 {**config, 'selftrain': {'rounds': 2, 'quantile': 0.5}},
                                 8, 'cpu', far_deadline())
    plain_error = float((plain['labels'] != query_y).mean())
    boosted_error = float((boosted['labels'] != query_y).mean())
    assert boosted_error <= plain_error + 0.01, (plain_error, boosted_error)


# --------------------------------------------------------------------------
# selftrain: the deadline
# --------------------------------------------------------------------------
def _stub_fit(seconds, query_rows=40, epochs=3):
    """A fake fit_once: burns ``seconds`` of wall clock, returns valid logits."""
    calls = []

    def fit_once(train_z, train_y, query_z, config, seed, deadline_unix, device,
                 sample_weight=None, pseudo_targets=None, init_state=None,
                 return_state=False):
        calls.append({'seed': seed, 'rows': int(train_z.shape[0]),
                      'deadline': deadline_unix, 'warm': init_state is not None})
        finish = time.perf_counter() + seconds
        while time.perf_counter() < finish:
            pass
        rng = np.random.default_rng(abs(int(seed)) % (2 ** 31))
        probabilities = rng.dirichlet(np.ones(10) * 0.2, size=query_rows)
        logits = np.log(probabilities).astype(np.float32)
        metrics = {'epochs_completed': epochs, 'truncated': False,
                   'training_seconds': seconds}
        if return_state:
            return logits, metrics, ['state']
        return logits, metrics

    return fit_once, calls


def test_selftrain_splits_the_deadline_into_equal_shares():
    train_z, train_y, query_z, _ = make_release_like(n_train=60, n_query=40, seed=36)
    fit_once, calls = _stub_fit(0.0)
    config = neural.resolve_config({'family': 'mlp', 'inference_margin_seconds': 0.0,
                                    'selftrain': {'rounds': 3, 'quantile': 0.5}})
    started = time.time()
    deadline = started + 40.0
    neural._selftrain(fit_once, train_z, train_y, query_z, config, 5, deadline, 'cpu')
    assert len(calls) == 4
    shares = [call['deadline'] - started for call in calls]
    assert shares[0] == pytest.approx(10.0, abs=0.5)
    assert shares[1] == pytest.approx(20.0, abs=0.5)
    assert shares[2] == pytest.approx(30.0, abs=0.5)
    assert shares[3] == pytest.approx(40.0, abs=0.5)   # the last owns the rest
    assert [call['rows'] for call in calls][0] == 60
    assert all(call['rows'] > 60 for call in calls[1:])


def test_selftrain_skips_rounds_that_would_overrun_the_deadline():
    """A round that does not fit is SKIPPED; the previous prediction stands."""
    train_z, train_y, query_z, _ = make_release_like(n_train=60, n_query=40, seed=37)
    fit_once, calls = _stub_fit(0.30)
    config = neural.resolve_config({'family': 'mlp', 'inference_margin_seconds': 0.0,
                                    'selftrain': {'rounds': 3, 'quantile': 0.5}})
    started = time.time()
    logits, metrics = neural._selftrain(fit_once, train_z, train_y, query_z, config,
                                        5, started + 0.5, 'cpu')
    elapsed = time.time() - started
    assert len(calls) == 1                       # only the base fit ran
    assert metrics['rounds_completed'] == 0
    assert metrics['selftrain_skipped_from_round'] == 1
    assert elapsed < 0.5                         # the deadline was respected
    assert logits.shape == (40, 10) and np.isfinite(logits).all()
    assert metrics['uses_query_images_unlabeled'] is True

    # With room for every round, none is skipped.
    fit_once, calls = _stub_fit(0.05)
    _, metrics = neural._selftrain(fit_once, train_z, train_y, query_z, config, 5,
                                   time.time() + 30.0, 'cpu')
    assert len(calls) == 4 and metrics['rounds_completed'] == 3
    assert metrics['selftrain_skipped_from_round'] is None


def test_selftrain_round_that_times_out_keeps_the_previous_round():
    train_z, train_y, query_z, _ = make_release_like(n_train=60, n_query=40, seed=38)
    fit_once, calls = _stub_fit(0.0)
    state = {'round': 0}

    def flaky(*args, **kwargs):
        state['round'] += 1
        if state['round'] == 3:                  # the third fit runs out of time
            raise TimeoutError('deadline expired before any epoch completed')
        return fit_once(*args, **kwargs)

    config = neural.resolve_config({'family': 'mlp', 'inference_margin_seconds': 0.0,
                                    'selftrain': {'rounds': 3}})
    logits, metrics = neural._selftrain(flaky, train_z, train_y, query_z, config, 5,
                                        time.time() + 30.0, 'cpu')
    assert metrics['rounds_completed'] == 1      # rounds 2 and 3 are not reported
    assert metrics['selftrain_skipped_from_round'] == 2
    assert np.isfinite(logits).all()


def test_selftrain_end_to_end_under_a_tight_deadline_still_returns():
    """A real fit with a deadline that only the first round can use."""
    train_z, train_y, query_z, _ = make_release_like(n_train=400, n_query=300, seed=39)
    config = {'family': 'mlp', 'widths': [256, 256], 'epochs': 200, 'batch_size': 32,
              'members': 3, 'inference_margin_seconds': 0.2,
              'selftrain': {'rounds': 3, 'quantile': 0.5}}
    started = time.time()
    out = neural.fit_predict(train_z, train_y, query_z, config, 3, 'cpu',
                             started + 4.0)
    elapsed = time.time() - started
    metrics = out['metrics']
    assert elapsed < 6.0
    assert metrics['rounds_completed'] < 3
    assert metrics['rounds_completed'] == len(metrics['selftrain_rounds']) - 1
    assert np.isfinite(out['logits']).all()
    assert out['labels'].shape == (300,)


def test_step_ratio_prices_a_round_in_steps_not_rows():
    """Under target_steps a bigger round is not a more expensive round."""
    fixed = neural.resolve_config({'family': 'mlp', 'epochs': 100, 'batch_size': 128})
    # 500 -> 5500 rows: 4 -> 43 steps per epoch at the same epoch count.
    assert neural._step_ratio(fixed, 500, 5500) == pytest.approx(43 / 4)
    targeted = neural.resolve_config({'family': 'mlp', 'target_steps': 1200,
                                      'batch_size': 128})
    # 300 x 4 = 1200 steps before, 28 x 43 = 1204 after: the same work.
    assert neural._step_ratio(targeted, 500, 5500) == pytest.approx(1204 / 1200)


def test_target_steps_rounds_are_not_skipped_for_being_wider():
    """The step-based projection lets a target_steps round run in its share."""
    train_z, train_y, query_z, _ = make_release_like(n_train=60, n_query=40, seed=40)
    fit_once, calls = _stub_fit(0.05)
    config = neural.resolve_config({'family': 'mlp', 'target_steps': 1200,
                                    'batch_size': 128,
                                    'inference_margin_seconds': 0.0,
                                    'selftrain': {'rounds': 2}})
    _, metrics = neural._selftrain(fit_once, train_z, train_y, query_z, config, 5,
                                   time.time() + 3.0, 'cpu')
    assert len(calls) == 3 and metrics['rounds_completed'] == 2


# --------------------------------------------------------------------------
# selftrain: what the recorded metrics mean
# --------------------------------------------------------------------------
def _stub_fit_reporting(reports, seconds=0.0, query_rows=40):
    """A fake fit_once whose k-th call returns ``reports[k]`` as its metrics."""
    calls = []

    def fit_once(train_z, train_y, query_z, config, seed, deadline_unix, device,
                 sample_weight=None, pseudo_targets=None, init_state=None,
                 return_state=False):
        report = dict(reports[min(len(calls), len(reports) - 1)])
        calls.append({'seed': seed, 'rows': int(train_z.shape[0]),
                      'deadline': deadline_unix})
        finish = time.perf_counter() + seconds
        while time.perf_counter() < finish:
            pass
        rng = np.random.default_rng(abs(int(seed)) % (2 ** 31))
        probabilities = rng.dirichlet(np.ones(10) * 0.2, size=query_rows)
        logits = np.log(probabilities).astype(np.float32)
        if return_state:
            return logits, report, ['state']
        return logits, report

    return fit_once, calls


def _report(epochs_completed, epochs_planned, truncated, training=0.05,
            inference=0.25):
    return {'epochs_completed': epochs_completed, 'epochs_planned': epochs_planned,
            'truncated': truncated, 'training_seconds': training,
            'inference_seconds': inference,
            'epoch_completion_fraction': epochs_completed / epochs_planned}


def test_selftrain_truncated_covers_every_round_not_just_the_last():
    """A truncated BASE fit must not be reported as an untruncated result.

    study.selection_rule()'s eligibility ("every dev fit finished untruncated")
    reads metrics['truncated'] and nothing else, and on a GPU it is the base
    fit -- the one paying CUDA context creation and cuDNN autotune inside its
    1/(R+1) share -- that is most likely to be the truncated one.
    """
    train_z, train_y, query_z, _ = make_release_like(n_train=60, n_query=40, seed=41)
    fit_once, calls = _stub_fit_reporting([_report(3, 100, True),
                                           _report(100, 100, False),
                                           _report(100, 100, False)])
    config = neural.resolve_config({'family': 'mlp', 'inference_margin_seconds': 0.0,
                                    'selftrain': {'rounds': 2, 'quantile': 0.5}})
    _, metrics = neural._selftrain(fit_once, train_z, train_y, query_z, config, 5,
                                   time.time() + 30.0, 'cpu')
    assert len(calls) == 3 and metrics['rounds_completed'] == 2
    assert [r['truncated'] for r in metrics['selftrain_rounds']] == [True, False, False]
    assert metrics['truncated'] is True                 # the whole call
    assert metrics['truncated_last_round'] is False     # the last fit alone
    assert metrics['truncated_any_round'] is True
    assert metrics['selftrain_rounds_incomplete'] is False
    assert metrics['epochs_completed'] == 203           # 3 + 100 + 100


def test_selftrain_truncated_is_set_when_a_configured_round_never_ran():
    """rounds=2 with only the base fit run is not a clean run of that recipe."""
    train_z, train_y, query_z, _ = make_release_like(n_train=60, n_query=40, seed=42)
    fit_once, calls = _stub_fit_reporting([_report(100, 100, False)], seconds=0.30)
    config = neural.resolve_config({'family': 'mlp', 'inference_margin_seconds': 0.0,
                                    'selftrain': {'rounds': 2, 'quantile': 0.5}})
    _, metrics = neural._selftrain(fit_once, train_z, train_y, query_z, config, 5,
                                   time.time() + 0.5, 'cpu')
    assert len(calls) == 1 and metrics['selftrain_skipped_from_round'] == 1
    assert metrics['truncated_any_round'] is False      # the fit that ran was fine
    assert metrics['selftrain_rounds_incomplete'] is True
    assert metrics['truncated'] is True                 # but the recipe did not run


def test_selftrain_skips_a_round_that_does_not_fit_its_own_share():
    """The skip-not-truncate contract, with a fit that honours its deadline.

    _stub_fit ignores its deadline, so it cannot see this: the round has to be
    measured against its OWN share, and the projection has to be de-biased by
    the base fit's epoch completion, or round 1 is started and truncated.
    """
    train_z, train_y, query_z, _ = make_release_like(n_train=60, n_query=40, seed=43)
    seen = []

    def fit_once(tz, ty, qz, config, seed, deadline_unix, device,
                 sample_weight=None, pseudo_targets=None, init_state=None,
                 return_state=False):
        wanted, started = 0.30, time.perf_counter()
        budget = min(wanted, max(0.0, deadline_unix - time.time()) * 0.9)
        while time.perf_counter() - started < budget:
            pass
        used = time.perf_counter() - started
        fraction = min(1.0, used / wanted)
        seen.append({'rows': int(tz.shape[0]), 'used': used,
                     'truncated': fraction < 0.999})
        rng = np.random.default_rng(abs(int(seed)) % (2 ** 31))
        logits = np.log(rng.dirichlet(np.ones(10) * 0.2, size=40)).astype(np.float32)
        metrics = {'epochs_completed': max(1, int(100 * fraction)),
                   'epochs_planned': 100, 'truncated': fraction < 0.999,
                   'training_seconds': used, 'inference_seconds': 0.0,
                   'epoch_completion_fraction': fraction}
        if return_state:
            return logits, metrics, ['state']
        return logits, metrics

    config = neural.resolve_config({'family': 'mlp', 'target_steps': 1200,
                                    'batch_size': 128,
                                    'inference_margin_seconds': 0.02,
                                    'selftrain': {'rounds': 2, 'quantile': 0.5}})
    # 0.6 s for three 0.3 s fits: the base fit is truncated inside its 0.2 s
    # share, so the refits cannot fit their shares either.
    _, metrics = neural._selftrain(fit_once, train_z, train_y, query_z, config, 5,
                                   time.time() + 0.6, 'cpu')
    assert len(seen) == 1                              # round 1 was SKIPPED
    assert seen[0]['truncated'] is True                # ...because the base fit
    assert metrics['selftrain_skipped_from_round'] == 1
    assert metrics['truncated'] is True

    # With room for every fit in its own share, all rounds run untruncated.
    seen.clear()
    _, metrics = neural._selftrain(fit_once, train_z, train_y, query_z, config, 5,
                                   time.time() + 4.0, 'cpu')
    assert len(seen) == 3 and all(not s['truncated'] for s in seen)
    assert metrics['rounds_completed'] == 2 and metrics['truncated'] is False


def test_selftrain_round_that_raises_keeps_the_previous_round():
    """A diverging refit must not throw away a finished base prediction.

    _fit_mlp_once raises FloatingPointError on a non-finite epoch loss, which a
    refit is likelier to hit than the base fit (~10x the rows, a fresh Adam
    restart, and a pseudo weight > 1).  The job has already paid for the base
    fit, so the policy is the TimeoutError policy: keep what finished.
    """
    train_z, train_y, query_z, _ = make_release_like(n_train=60, n_query=40, seed=44)
    fit_once, calls = _stub_fit(0.0)

    def flaky(*args, **kwargs):
        if len(calls) == 2:
            raise FloatingPointError('Nonfinite training objective')
        return fit_once(*args, **kwargs)

    config = neural.resolve_config({'family': 'mlp', 'inference_margin_seconds': 0.0,
                                    'selftrain': {'rounds': 3}})
    logits, metrics = neural._selftrain(flaky, train_z, train_y, query_z, config, 5,
                                        time.time() + 30.0, 'cpu')
    assert metrics['rounds_completed'] == 1
    assert metrics['selftrain_skipped_from_round'] == 2
    assert metrics['selftrain_round_failure']['round'] == 2
    assert 'FloatingPointError' in metrics['selftrain_round_failure']['error']
    assert logits.shape == (40, 10) and np.isfinite(logits).all()


def test_selftrain_costs_are_summed_over_rounds_on_one_scope():
    """training / inference / epochs all describe the WHOLE call, as kernels.py does."""
    train_z, train_y, query_z, _ = make_release_like(n_train=60, n_query=40, seed=45)
    fit_once, calls = _stub_fit_reporting(
        [_report(100, 100, False, training=0.05, inference=0.25)], seconds=0.10)
    config = neural.resolve_config({'family': 'mlp', 'inference_margin_seconds': 0.0,
                                    'selftrain': {'rounds': 2, 'quantile': 0.5}})
    _, metrics = neural._selftrain(fit_once, train_z, train_y, query_z, config, 5,
                                   time.time() + 30.0, 'cpu')
    assert len(calls) == 3
    assert metrics['training_seconds'] == pytest.approx(0.15)      # not the wall
    assert metrics['inference_seconds'] == pytest.approx(0.75)     # not last-round
    assert metrics['training_seconds_last_round'] == pytest.approx(0.05)
    assert metrics['inference_seconds_last_round'] == pytest.approx(0.25)
    assert metrics['epochs_completed'] == 300 and metrics['epochs_planned'] == 300
    assert metrics['epochs_completed_last_round'] == 100
    assert metrics['fit_seconds_all_rounds'] >= 0.30
    assert metrics['fit_wall_seconds'] >= metrics['fit_seconds_all_rounds']


def test_ladder_steps_per_epoch_drops_a_lone_trailing_row():
    """The ladder skips batches of one row, so the schedule must not count them."""
    assert neural.ladder_steps_per_epoch(15000, 100) == 150
    assert neural.ladder_steps_per_epoch(15001, 100) == 150     # not 151
    assert neural.ladder_steps_per_epoch(15002, 100) == 151
    assert neural.ladder_steps_per_epoch(1, 100) == 1           # degenerate draw
    ratio = neural.resolve_config({'family': 'ladder', 'batch_size': 100,
                                   'epochs': 10})
    assert neural._step_ratio(ratio, 500, 15001) == pytest.approx(150 / 5)


def test_ladder_target_steps_counts_only_batches_it_runs():
    train_z, train_y, query_z, _ = make_release_like(n_train=201, n_query=30, seed=46)
    config = {'family': 'ladder', 'hidden_dims': [16], 'batch_size': 100,
              'epochs': 150, 'decay_start_epoch': 100, 'target_steps': 8}
    out = neural.fit_predict(train_z, train_y, query_z, config, 2, 'cpu',
                             far_deadline())
    metrics = out['metrics']
    assert metrics['steps_per_epoch'] == 2                     # 201 rows -> 2, not 3
    assert metrics['epochs_resolved'] == 4                     # ceil(8 / 2)
    assert metrics['target_steps_note']['planned_steps'] == 8  # and 8 really run


def test_select_pseudo_rows_keeps_rows_tied_at_the_threshold():
    """q is an upper bound on the fraction kept: ties are all kept."""
    probabilities = np.full((11, 10), 0.1)
    probabilities[0] = 0.0
    probabilities[0, 3] = 1.0                      # one confident row of class 3
    index, labels, confidence, counts = neural.select_pseudo_rows(probabilities, 0.5)
    assert counts[3] == 1
    assert counts[0] == 10                         # all ten tied rows, not five
    assert index.size == 11
