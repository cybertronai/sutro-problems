"""Tests for kernels.py -- CPU only, small synthetic problems plus one real draw.

Run with ``/tmp/penv/bin/python -m pytest test_kernels.py -q`` from the study root.

The real-draw test scores on a held-out slice of the draw's TRAINING rows, whose
labels the learner is given anyway: no test here ever asks for a query label, and
this file is not one of ``runner.SOURCE_FILES``, so it never reaches a container.
"""
from __future__ import annotations

import json
import math
import time

import numpy as np
import pytest
import torch

import kernels

DEADLINE = 600.0


def deadline():
    return time.time() + DEADLINE


def blobs(n, seed, dim=12, classes=10, spread=2.0, centers=None):
    """Ten Gaussian blobs -- an easy, rotation-symmetric stand-in for a release."""
    rng = np.random.default_rng(seed)
    if centers is None:
        centers = np.random.default_rng(1234).standard_normal((classes, dim)) * spread
    y = rng.integers(0, classes, n)
    x = centers[y] + rng.standard_normal((n, dim))
    return x.astype(np.float32), y.astype(np.uint8), centers


ALL_CONFIGS = [
    {'family': 'krr', 'kernel': 'laplace'},
    {'family': 'krr', 'kernel': 'rbf'},
    {'family': 'krr', 'kernel': 'arccos', 'depth': 3},
    {'family': 'rfm', 'iters': 2, 'bandwidth_grid': [3.0, 6.0]},
    {'family': 'labelprop', 'k_grid': [5, 10], 'alpha_grid': [0.5, 0.9]},
    {'family': 'labelprop', 'metric': 'rfm', 'k_grid': [5],
     'rfm_config': {'iters': 1, 'bandwidth': 5.0, 'ridge': 1e-6}},
    {'family': 'selftrain', 'base': {'family': 'krr', 'kernel': 'laplace'}},
    {'family': 'selftrain', 'rounds': 2, 'quantile': 0.7,
     'base': {'family': 'rfm', 'iters': 1, 'bandwidth': 4.0, 'ridge': 1e-6}},
]


# ------------------------------------------------------------------ config ---
def test_resolve_config_rejects_unknown_keys():
    for family in ('krr', 'rfm', 'labelprop'):
        with pytest.raises(ValueError, match='Unknown config keys'):
            kernels.resolve_config({'family': family, 'bogus': 1})
    with pytest.raises(ValueError, match='Unknown config keys'):
        kernels.resolve_config({'family': 'selftrain', 'base': {'family': 'krr'},
                                'learning_rate': 0.1})


def test_resolve_config_rejects_unknown_nested_keys():
    with pytest.raises(ValueError, match='Unknown config keys'):
        kernels.resolve_config({'family': 'selftrain',
                                'base': {'family': 'krr', 'nope': 2}})
    with pytest.raises(ValueError, match='Unknown config keys'):
        kernels.resolve_config({'family': 'labelprop', 'metric': 'rfm',
                                'rfm_config': {'nope': 2}})


def test_resolve_config_validates_values():
    with pytest.raises(ValueError, match='family'):
        kernels.resolve_config({'family': 'svm'})
    with pytest.raises(ValueError, match='kernel must be'):
        kernels.resolve_config({'family': 'krr', 'kernel': 'matern'})
    with pytest.raises(ValueError, match='alpha must be'):
        kernels.resolve_config({'family': 'labelprop', 'alpha': 1.0})
    with pytest.raises(ValueError, match='pseudo_weight'):
        kernels.resolve_config({'family': 'selftrain', 'base': {'family': 'krr'},
                                'pseudo_weight': 0.5})
    with pytest.raises(ValueError, match='rewhiten'):
        kernels.resolve_config({'family': 'krr', 'rewhiten': 'train'})
    with pytest.raises(ValueError, match='selftrain base family'):
        kernels.resolve_config({'family': 'selftrain',
                                'base': {'family': 'labelprop'}})


def test_resolve_config_is_idempotent():
    for config in ALL_CONFIGS:
        once = kernels.resolve_config(config)
        assert kernels.resolve_config(once) == once


def test_grid_helper():
    config = kernels.resolve_config({'family': 'rfm', 'agop_power': None,
                                     'agop_power_grid': [1.0, 0.5]})
    assert kernels._grid(config, 'agop_power') == [1.0, 0.5]
    assert kernels._grid(config, 'diag') == [False]           # a set scalar wins


# ------------------------------------------------------------------ outputs ---
@pytest.mark.parametrize('config', ALL_CONFIGS)
def test_shapes_labels_and_metrics(config):
    x, y, centers = blobs(220, 11)
    q, _, _ = blobs(150, 12, centers=centers)
    out = kernels.fit_predict(x, y, q, config, 5, 'cpu', deadline())
    assert out['logits'].shape == (150, kernels.NUM_CLASSES)
    assert out['logits'].dtype == np.float32 and np.isfinite(out['logits']).all()
    assert out['labels'].shape == (150,) and out['labels'].dtype == np.uint8
    assert np.array_equal(out['labels'], np.argmax(out['logits'], axis=1).astype(np.uint8))
    metrics = out['metrics']
    for key in ('training_seconds', 'inference_seconds', 'fit_wall_seconds',
                'uses_query_images_unlabeled', 'chosen', 'cv_table', 'config', 'family'):
        assert key in metrics, key
    assert metrics['query_labels_supplied'] is False
    json.dumps(metrics, allow_nan=False)                      # runner.write_json contract


def test_first_max_tie_break():
    logits = np.array([[1.0, 1.0, 0.0], [0.0, 2.0, 2.0], [3.0, 3.0, 3.0]], dtype=np.float32)
    assert kernels.first_max_labels(logits).tolist() == [0, 1, 0]


def test_labels_are_first_max_on_a_tie():
    """Identical training rows, one per class: every score ties, class 0 must win."""
    x = np.zeros((10, 6), dtype=np.float32)
    y = np.arange(10, dtype=np.uint8)
    q = np.zeros((7, 6), dtype=np.float32)
    out = kernels.fit_predict(x, y, q, {'family': 'krr', 'kernel': 'laplace',
                                        'bandwidth': 1.0, 'ridge': 1e-3},
                              1, 'cpu', deadline())
    rows = out['logits']
    assert float(np.abs(rows - rows[:, :1]).max()) < 1e-6      # every class ties
    assert out['labels'].tolist() == [0] * 7


# ------------------------------------------------------- query independence ---
@pytest.mark.parametrize('config', [
    {'family': 'krr', 'kernel': 'laplace'},
    {'family': 'krr', 'kernel': 'arccos', 'depth': 2},
    {'family': 'rfm', 'iters': 1, 'bandwidth_grid': [3.0, 6.0]},
    {'family': 'selftrain', 'base': {'family': 'krr', 'kernel': 'laplace'}},
    {'family': 'labelprop', 'cv_graph': 'labeled_only', 'k_grid': [5, 10]},
])
def test_selection_never_sees_the_query_rows(config):
    """Two different query sets, same training rows -> the same hyperparameters."""
    x, y, centers = blobs(200, 21)
    q1, _, _ = blobs(120, 22, centers=centers)
    q2 = np.random.default_rng(99).standard_normal(q1.shape).astype(np.float32) * 5.0
    first = kernels.fit_predict(x, y, q1, config, 5, 'cpu', deadline())['metrics']
    second = kernels.fit_predict(x, y, q2, config, 5, 'cpu', deadline())['metrics']
    assert first['chosen'] == second['chosen']
    assert first['cv_table'] == second['cv_table']


def test_transductive_labelprop_is_query_order_equivariant():
    """The full-graph CV does use query FEATURES; it must not use their order."""
    x, y, centers = blobs(200, 31)
    q, _, _ = blobs(150, 32, centers=centers)
    permutation = np.random.default_rng(7).permutation(len(q))
    config = {'family': 'labelprop', 'k_grid': [5, 10], 'alpha_grid': [0.5, 0.9]}
    plain = kernels.fit_predict(x, y, q, config, 5, 'cpu', deadline())
    shuffled = kernels.fit_predict(x, y, q[permutation], config, 5, 'cpu', deadline())
    assert plain['metrics']['chosen'] == shuffled['metrics']['chosen']
    assert np.array_equal(plain['labels'][permutation], shuffled['labels'])


def test_transductive_flags():
    x, y, centers = blobs(150, 41)
    q, _, _ = blobs(80, 42, centers=centers)
    def flag(config):
        out = kernels.fit_predict(x, y, q, config, 5, 'cpu', deadline())
        metrics = out['metrics']
        assert (metrics['uses_query_images_unlabeled']
                == metrics['uses_query_features_unlabeled'])
        return metrics['uses_query_images_unlabeled']
    assert flag({'family': 'krr'}) is False
    assert flag({'family': 'rfm', 'iters': 1, 'bandwidth': 4.0, 'ridge': 1e-6}) is False
    assert flag({'family': 'labelprop', 'k_grid': [5]}) is True
    assert flag({'family': 'selftrain', 'base': {'family': 'krr'}}) is True
    assert flag({'family': 'krr', 'rewhiten': 'query'}) is True
    # the ensemble is the only place resolve_config allows prior_match to live,
    # and the wrapper's own transductive keys used to be skipped entirely
    assert flag({'family': 'ensemble', 'prior_match': 'sinkhorn',
                 'members': [{'family': 'krr'}]}) is True
    assert flag({'family': 'ensemble', 'rewhiten': 'query',
                 'members': [{'family': 'krr'}]}) is True
    assert flag({'family': 'ensemble', 'members': [{'family': 'krr'}]}) is False


def test_rewhiten_is_recorded_through_the_wrappers():
    """`metrics['rewhiten']` is the top level only; the audit key is the tree."""
    x, y, centers = blobs(200, 43)
    q, _, _ = blobs(120, 44, centers=centers)
    base = {'family': 'krr', 'kernel': 'arccos', 'depth': 2, 'rewhiten': 'query'}
    for config in ({'family': 'selftrain', 'base': base},
                   {'family': 'ensemble', 'members': [base]}):
        metrics = kernels.fit_predict(x, y, q, config, 5, 'cpu', deadline())['metrics']
        assert metrics['rewhiten'] == 'none'              # the wrapper itself does not
        assert metrics['rewhiten_anywhere'] is True       # but something inside does
        assert metrics['cv_seconds'] is not None
        tree = metrics['preprocess_effective']
        nested = [tree['base']] if tree['base'] else tree['members']
        assert any(entry['rewhiten'] == 'query' for entry in nested), tree


# ---------------------------------------------------------------- invariance ---
def test_predictions_are_invariant_to_an_orthogonal_release_rotation():
    """The release applies a secret Haar Q; every family must ignore it."""
    x, y, centers = blobs(160, 51, dim=8)
    q, _, _ = blobs(90, 52, dim=8, centers=centers)
    rotation = np.linalg.qr(np.random.default_rng(5).standard_normal((8, 8)))[0]
    for config in [{'family': 'krr', 'kernel': 'laplace'},
                   {'family': 'krr', 'kernel': 'arccos', 'depth': 2},
                   {'family': 'rfm', 'iters': 1, 'bandwidth': 4.0, 'ridge': 1e-6},
                   {'family': 'labelprop', 'k_grid': [5]},
                   {'family': 'krr', 'rewhiten': 'query'},
                   {'family': 'krr', 'prior_match': 'sinkhorn'}]:
        plain = kernels.fit_predict(x, y, q, config, 5, 'cpu', deadline())
        turned = kernels.fit_predict((x @ rotation).astype(np.float32), y,
                                     (q @ rotation).astype(np.float32), config, 5, 'cpu',
                                     deadline())
        assert np.allclose(plain['logits'], turned['logits'], atol=2e-4), config
        assert np.array_equal(plain['labels'], turned['labels']), config


# ------------------------------------------------------------- kernel algebra ---
def test_arccos_closed_form_matches_the_definition():
    torch.manual_seed(0)
    a = torch.randn(9, 4, dtype=torch.float64)
    b = torch.randn(7, 4, dtype=torch.float64)
    got = kernels._arccos_kernel(a, b, depth=1, bias=0.0)
    expected = torch.zeros_like(got)
    for i in range(a.shape[0]):
        for j in range(b.shape[0]):
            na, nb = a[i].norm(), b[j].norm()
            theta = torch.arccos((a[i] @ b[j]) / (na * nb))
            expected[i, j] = (na * nb / math.pi) * (torch.sin(theta)
                                                    + (math.pi - theta) * torch.cos(theta))
    assert torch.allclose(got, expected, atol=1e-12)


def test_arccos_diagonal_is_the_squared_norm_at_every_depth():
    torch.manual_seed(1)
    a = torch.randn(6, 5, dtype=torch.float64)
    for depth in (1, 2, 3):
        k = kernels._arccos_kernel(a, a, depth=depth, bias=0.0, same=True)
        assert torch.allclose(torch.diagonal(k), (a * a).sum(1), atol=1e-12)


def test_arccos_matches_a_random_relu_feature_expansion():
    """k1(x,z) = E_w[2 relu(w.x) relu(w.z)] for w ~ N(0, I) (Cho & Saul 2009)."""
    torch.manual_seed(2)
    a = torch.randn(4, 3, dtype=torch.float64)
    w = torch.randn(400000, 3, dtype=torch.float64)
    features = torch.relu(a @ w.T) * math.sqrt(2.0 / w.shape[0])
    monte_carlo = features @ features.T
    exact = kernels._arccos_kernel(a, a, depth=1, bias=0.0, same=True)
    assert torch.allclose(monte_carlo, exact, rtol=0.02, atol=0.02)


def test_krr_eigen_cv_matches_a_direct_solve():
    """The one-eigendecomposition-per-fold path must equal (K + lam I)^-1 Y."""
    torch.manual_seed(3)
    x = torch.randn(40, 4, dtype=torch.float64)
    y = torch.randint(0, 10, (40,))
    k = kernels._kernel_matrix(x, x, 'laplace', 2.0, same=True)
    targets = kernels._targets(y, 10, 'centered', torch.float64)
    folds = kernels._fold_indices(40, 4, 17)
    ridge_grid = [1e-6, 1e-3]
    fast = kernels._krr_cv_fold_scores(k, targets, y, folds, ridge_grid, 'trace')
    slow = np.zeros(len(ridge_grid), dtype=np.int64)
    for validation in folds:
        mask = np.ones(40, dtype=bool)
        mask[validation] = False
        index = torch.as_tensor(np.nonzero(mask)[0])
        val = torch.as_tensor(validation)
        k_tt = k[index][:, index]
        for j, ridge in enumerate(ridge_grid):
            lam = ridge * float(torch.diagonal(k_tt).sum())
            alpha = torch.linalg.solve(
                k_tt + lam * torch.eye(len(index), dtype=torch.float64), targets[index])
            prediction = k[val][:, index] @ alpha
            slow[j] += int((prediction.argmax(1) == y[val]).sum())
    assert fast.tolist() == slow.tolist()


# ------------------------------------------------------------------- the AGOP ---
def _reference_jacobian(samples, centers, weights, bandwidth, metric):
    def predictor(point):
        difference = point.unsqueeze(0) - centers
        distance = torch.sqrt(torch.einsum('pd,dD,pD->p', difference, metric, difference))
        return torch.exp(-distance / bandwidth) @ weights
    return torch.stack([torch.autograd.functional.jacobian(predictor, samples[i])
                        for i in range(samples.shape[0])])


def test_agop_matches_autograd_jacobians():
    torch.manual_seed(4)
    samples = torch.randn(5, 3, dtype=torch.float64)
    centers = torch.randn(7, 3, dtype=torch.float64)
    weights = torch.randn(7, 2, dtype=torch.float64)
    root = torch.randn(3, 3, dtype=torch.float64)
    metric = root @ root.T + 0.5 * torch.eye(3, dtype=torch.float64)
    jacobians = _reference_jacobian(samples, centers, weights, 1.3, metric)
    expected = torch.einsum('ncd,ncD->dD', jacobians, jacobians)
    for chunk in (2, 5, 64):
        got = kernels.laplace_agop(samples, centers, weights, 1.3, metric, chunk=chunk)
        assert torch.allclose(got, expected, atol=1e-10), chunk


def test_agop_diagonal_matches_autograd():
    torch.manual_seed(5)
    samples = torch.randn(5, 3, dtype=torch.float64)
    centers = torch.randn(6, 3, dtype=torch.float64)
    weights = torch.randn(6, 2, dtype=torch.float64)
    diagonal = torch.rand(3, dtype=torch.float64) + 0.5
    jacobians = _reference_jacobian(samples, centers, weights, 0.8, torch.diag(diagonal))
    expected = torch.einsum('ncd,ncd->d', jacobians, jacobians)
    got = kernels.laplace_agop(samples, centers, weights, 0.8, diagonal,
                               diagonal=True, chunk=4)
    assert torch.allclose(got, expected, atol=1e-10)


def test_agop_drops_the_zero_distance_term_like_the_reference():
    """With samples == centers the j = i term has d_M = 0 and must be dropped.

    The Laplace kernel has a kink at zero, so that term has no gradient; the
    reference divides by the distance and zeroes the resulting infinity.  The
    explicit sum below is the same predictor Jacobian with ``j == i`` omitted.
    """
    torch.manual_seed(6)
    x = torch.randn(6, 3, dtype=torch.float64)
    weights = torch.randn(6, 2, dtype=torch.float64)
    root = torch.randn(3, 3, dtype=torch.float64)
    metric = root @ root.T + torch.eye(3, dtype=torch.float64)
    bandwidth = 1.1
    expected = torch.zeros(3, 3, dtype=torch.float64)
    for i in range(x.shape[0]):
        jacobian = torch.zeros(2, 3, dtype=torch.float64)
        for j in range(x.shape[0]):
            if i == j:
                continue
            difference = x[i] - x[j]
            distance = torch.sqrt(difference @ metric @ difference)
            scale = math.exp(-float(distance) / bandwidth) / float(distance) / bandwidth
            jacobian -= scale * torch.outer(weights[j], metric @ difference)
        expected += jacobian.T @ jacobian
    got = kernels.laplace_agop(x, x, weights, bandwidth, metric, chunk=4)
    assert torch.isfinite(got).all()
    assert torch.allclose(got, expected, atol=1e-10)


def test_agop_normalisations():
    """'max_entry' is the reference rule; the default is its invariant twin."""
    torch.manual_seed(7)
    root = torch.randn(4, 4, dtype=torch.float64)
    agop = root @ root.T
    reference = kernels._normalise_agop(agop, 'max_entry')
    assert abs(float(reference.max()) - 1.0) < 1e-12
    assert torch.allclose(reference * float(agop.max()), agop)
    spectral = kernels._normalise_agop(agop, 'max_eigenvalue')
    assert abs(float(torch.linalg.eigvalsh(spectral).max()) - 1.0) < 1e-12
    traced = kernels._normalise_agop(agop, 'trace')
    assert abs(float(torch.diagonal(traced).sum()) - 4.0) < 1e-10
    # a diagonal M used to ignore `mode` and always run the 'max_entry' rule
    diagonal = torch.diagonal(agop).clone()
    assert abs(float(kernels._normalise_agop(diagonal, 'trace', True).sum()) - 4.0) < 1e-10
    assert abs(float(kernels._normalise_agop(diagonal, 'max_eigenvalue', True).max())
               - 1.0) < 1e-12

    rotation = torch.linalg.qr(torch.randn(4, 4, dtype=torch.float64))[0]
    turned = kernels._normalise_agop(rotation.T @ agop @ rotation, 'max_eigenvalue')
    assert torch.allclose(turned, rotation.T @ spectral @ rotation, atol=1e-12)


def test_metric_floor_clamps_the_learned_metric():
    """`metric_floor` was declared, threaded, documented -- and ignored."""
    x, y, centers = blobs(220, 211, dim=12)
    q, _, _ = blobs(120, 212, dim=12, centers=centers)
    config = {'family': 'rfm', 'iters': 2, 'bandwidth': 4.0, 'ridge': 1e-6}
    tiny = kernels.fit_predict(x, y, q, {**config, 'metric_floor': 1e-12}, 5, 'cpu',
                               deadline())
    huge = kernels.fit_predict(x, y, q, {**config, 'metric_floor': 1e9}, 5, 'cpu',
                               deadline())
    assert not np.array_equal(tiny['logits'], huge['logits'])
    matrix = torch.diag(torch.tensor([4.0, 1e-30, -1.0], dtype=torch.float64))
    clamped = kernels._matrix_power(matrix, 0.5, floor=1e-6)
    assert torch.allclose(torch.diagonal(clamped),
                          torch.tensor([2.0, 1e-3, 1e-3], dtype=torch.float64),
                          atol=1e-12)


def test_rfm_learns_a_metric_that_ignores_noise_dimensions():
    """Two informative dimensions out of ten: the AGOP must concentrate there."""
    rng = np.random.default_rng(8)
    n = 400
    signal = rng.standard_normal((n, 2))
    label = ((signal[:, 0] > 0).astype(np.int64) + 2 * (signal[:, 1] > 0).astype(np.int64))
    x = np.concatenate([signal, rng.standard_normal((n, 8)) * 0.5], axis=1).astype(np.float32)
    train = torch.as_tensor(x, dtype=torch.float64)
    targets = kernels._targets(torch.as_tensor(label), 10, 'centered', torch.float64)
    config = kernels.resolve_config({'family': 'rfm'})
    metric, _, _, _, _ = kernels._rfm_run(train, targets, 3.0, 1e-6, 'trace', 3, 1.0,
                                          False, config)
    weight = torch.diagonal(metric)
    assert float(weight[:2].min()) > 3.0 * float(weight[2:].max())


# ------------------------------------------------------------- label spreading ---
def test_knn_graph_is_symmetric_and_slices_by_k():
    torch.manual_seed(9)
    points = torch.randn(40, 3, dtype=torch.float64)
    indices, distances = kernels.knn_neighbours(points, 8, chunk=7)
    graph, _ = kernels.knn_graph(indices, distances, 5)
    dense = graph.to_dense()
    assert torch.allclose(dense, dense.T)
    assert float(torch.diagonal(dense).abs().max()) == 0.0
    small_indices, small_distances = kernels.knn_neighbours(points, 5, chunk=40)
    direct, _ = kernels.knn_graph(small_indices, small_distances, 5)
    assert torch.allclose(dense, direct.to_dense())


def test_spread_converges_to_the_closed_form():
    """F = (1 - alpha) (I - alpha S)^-1 Y (Zhou et al. 2004, eq. 4)."""
    torch.manual_seed(10)
    points = torch.randn(30, 3, dtype=torch.float64)
    indices, distances = kernels.knn_neighbours(points, 6)
    graph, _ = kernels.knn_graph(indices, distances, 4)
    adjacency, _ = kernels.normalised_adjacency(graph)
    initial = torch.zeros(30, 10, dtype=torch.float64)
    initial[torch.arange(10), torch.arange(10)] = 1.0
    alpha = 0.9
    iterated = kernels.spread(adjacency, initial, alpha, 3000)
    dense = adjacency.to_dense()
    closed = (1 - alpha) * torch.linalg.solve(
        torch.eye(30, dtype=torch.float64) - alpha * dense, initial)
    assert torch.allclose(iterated, closed, atol=1e-8)


def test_labelprop_reports_graph_diagnostics():
    x, y, centers = blobs(150, 61)
    q, _, _ = blobs(100, 62, centers=centers)
    metrics = kernels.fit_predict(x, y, q, {'family': 'labelprop', 'k_grid': [5]},
                                  5, 'cpu', deadline())['metrics']
    assert metrics['graph_nodes'] == 250
    assert metrics['unreached_query_rows'] == 0
    assert metrics['isolated_nodes'] == 0


def test_selftrain_adds_pseudo_rows_and_logs_them():
    x, y, centers = blobs(150, 71)
    q, _, _ = blobs(200, 72, centers=centers)
    metrics = kernels.fit_predict(x, y, q, {'family': 'selftrain', 'quantile': 0.6,
                                            'base': {'family': 'krr'}},
                                  5, 'cpu', deadline())['metrics']
    rounds = metrics['rounds_log']
    assert len(rounds) == 2 and rounds[0]['pseudo_rows'] == 0
    assert 70 <= rounds[1]['pseudo_rows'] <= 90            # about 40% of 200 query rows


# ------------------------------------------------------------- prior matching ---
def test_sinkhorn_matches_the_class_prior():
    torch.manual_seed(11)
    scores = torch.randn(500, 10, dtype=torch.float64)
    scores[:, 3] += 2.0                                    # a badly biased predictor
    prior = torch.full((10,), 0.1, dtype=torch.float64)
    matched, bias = kernels.sinkhorn_prior_match(scores, prior, 0.5, 200)
    fraction = torch.bincount(matched.argmax(1), minlength=10).to(torch.float64) / 500
    assert float((fraction - prior).abs().sum()) < 0.10
    raw = torch.bincount(scores.argmax(1), minlength=10).to(torch.float64) / 500
    assert float((raw - prior).abs().sum()) > 0.5
    assert abs(float(bias.mean())) < 1e-9                  # the bias is centred


def test_prior_match_none_is_the_identity():
    x, y, centers = blobs(200, 111)
    q, _, _ = blobs(150, 112, centers=centers)
    config = {'family': 'krr', 'kernel': 'arccos', 'depth': 2}
    plain = kernels.fit_predict(x, y, q, config, 5, 'cpu', deadline())
    assert 'prior_match_bias' not in plain['metrics']
    matched = kernels.fit_predict(x, y, q, {**config, 'prior_match': 'sinkhorn'},
                                  5, 'cpu', deadline())
    assert matched['metrics']['uses_query_images_unlabeled'] is True
    assert len(matched['metrics']['prior_match_bias']) == 10
    counts = np.array(matched['metrics']['query_class_counts_after'])
    before = np.array(matched['metrics']['query_class_counts_before'])
    prior = np.bincount(y, minlength=10) + 1.0
    prior = prior / prior.sum()
    after_gap = np.abs(counts / counts.sum() - prior).sum()
    before_gap = np.abs(before / before.sum() - prior).sum()
    assert after_gap <= before_gap + 1e-9


# -------------------------------------------------------------------- deadline ---
def test_expired_deadline_falls_back_to_the_centroid_predictor():
    x, y, centers = blobs(400, 81)
    q, _, _ = blobs(300, 82, centers=centers)
    for config in ALL_CONFIGS:
        started = time.time()
        out = kernels.fit_predict(x, y, q, config, 5, 'cpu', time.time() - 1.0)
        assert time.time() - started < 5.0
        assert out['metrics']['emergency_fallback'] is True
        assert out['logits'].shape == (300, 10) and np.isfinite(out['logits']).all()
        centroid = kernels.fit_predict(x, y, q, config, 5, 'cpu', time.time() - 1.0)
        assert np.array_equal(out['labels'], centroid['labels'])


def test_tight_deadline_truncates_the_sweep_but_still_answers():
    x, y, centers = blobs(1200, 91, dim=20)
    q, _, _ = blobs(800, 92, dim=20, centers=centers)
    config = {'family': 'krr', 'kernel': 'laplace',
              'bandwidth_scales': [0.125, 0.25, 0.5, 1.0, 2.0, 4.0], 'cv_max_rows': 1200}
    out = kernels.fit_predict(x, y, q, config, 5, 'cpu', time.time() + 1.2)
    metrics = out['metrics']
    assert out['labels'].shape == (800,)
    assert metrics['deadline_slack_seconds'] > -1.0
    assert metrics.get('deadline_truncated_cv') or metrics['emergency_fallback']


def test_single_cell_config_is_still_costed_before_the_fit():
    """No CV sweep means no measured cell; the fit used to launch regardless."""
    rng = np.random.default_rng(12)
    x = rng.standard_normal((2600, 60)).astype(np.float32)
    y = rng.integers(0, 10, 2600).astype(np.uint8)
    q = rng.standard_normal((3000, 60)).astype(np.float32)
    config = {'family': 'krr', 'kernel': 'arccos', 'depth': 3, 'ridge': 1e-6}
    full = kernels.fit_predict(x, y, q, config, 5, 'cpu', deadline())
    cost = full['metrics']['fit_wall_seconds']
    assert full['metrics']['final_fit_estimate_seconds'] > 0.5 * cost, cost
    started = time.time()
    tight = kernels.fit_predict(x, y, q, config, 5, 'cpu', time.time() + 0.25 * cost)
    elapsed = time.time() - started
    assert tight['metrics']['emergency_fallback'] is True
    assert elapsed < 0.5 * cost, (elapsed, cost)


def test_the_estimate_prices_the_query_kernel():
    """N small, Q large: the inference block is bigger than the training one."""
    rng = np.random.default_rng(13)
    x = rng.standard_normal((400, 60)).astype(np.float32)
    y = rng.integers(0, 10, 400).astype(np.uint8)
    config = {'family': 'krr', 'kernel': 'arccos', 'depth': 3, 'ridge': 1e-6}
    small = kernels.fit_predict(x, y, rng.standard_normal((400, 60)).astype(np.float32),
                                config, 5, 'cpu', deadline())['metrics']
    large = kernels.fit_predict(x, y, rng.standard_normal((20000, 60)).astype(np.float32),
                                config, 5, 'cpu', deadline())['metrics']
    growth = (large['final_fit_estimate_seconds']
              - small['final_fit_estimate_seconds'])
    assert growth > 0.5 * large['inference_seconds'], (growth, large)
    assert large['final_fit_estimate_seconds'] > large['inference_seconds']


def test_selftrain_reports_every_round_of_inference():
    x, y, centers = blobs(400, 201)
    q, _, _ = blobs(300, 202, centers=centers)
    config = {'family': 'selftrain', 'rounds': 2, 'quantile': 0.3,
              'base': {'family': 'krr', 'kernel': 'arccos', 'depth': 2}}
    metrics = kernels.fit_predict(x, y, q, config, 5, 'cpu', deadline())['metrics']
    rounds = [r['inference_seconds'] or 0.0 for r in metrics['rounds_log']]
    assert len(rounds) == 3
    assert metrics['inference_seconds'] == pytest.approx(sum(rounds))
    assert metrics['inference_seconds'] > metrics['inference_seconds_last_round']


def test_solve_ridge_is_exact_and_adds_the_ridge_in_place():
    torch.manual_seed(21)
    root = torch.randn(40, 40, dtype=torch.float64)
    k = root @ root.T
    y = torch.randn(40, 3, dtype=torch.float64)
    expected = torch.linalg.solve(k + 1e-3 * torch.eye(40, dtype=torch.float64), y)
    keep = k.clone()
    got = kernels._solve_ridge(k, y, 1e-3)
    assert torch.allclose(got, expected, atol=1e-9)
    assert torch.equal(k, keep)                       # the default does not touch k
    inplace = kernels._solve_ridge(k, y, 1e-3, inplace=True)
    assert torch.allclose(inplace, expected, atol=1e-9)
    assert torch.allclose(torch.diagonal(k), torch.diagonal(keep) + 1e-3, atol=1e-12)


def test_fit_accepts_the_pmnist_argument_order():
    x, y, centers = blobs(120, 101)
    q, _, _ = blobs(60, 102, centers=centers)
    config = {'family': 'krr', 'kernel': 'laplace'}
    straight = kernels.fit_predict(x, y, q, config, 5, 'cpu', deadline())
    swapped = kernels.fit_predict(x, y, q, config, 5, deadline(), 'cpu')
    assert swapped['metrics']['argument_order_swapped'] is True
    assert np.array_equal(straight['labels'], swapped['labels'])


# ------------------------------------------------------------------ real draw ---
def test_real_release_draw_beats_chance():
    """One real harness draw; scored on held-out TRAINING rows, never on the query."""
    study = pytest.importorskip('study')
    arrays = study.job_arrays(study.DEV_SEEDS[0], 700)
    train_z, train_y, query_z = arrays['train_z'], arrays['train_y'], arrays['query_z']
    assert train_z.shape == (700, 60) and query_z.shape == (10000, 60)
    holdout = slice(500, 700)
    out = kernels.fit_predict(train_z[:500], train_y[:500], train_z[holdout],
                              {'family': 'krr', 'kernel': 'laplace'}, 5, 'cpu', deadline())
    error = float((out['labels'] != train_y[holdout]).mean())
    assert error < 0.30, error


def test_real_release_training_rows_are_white():
    """Sanity check on the protocol: the release whitens on the training rows."""
    study = pytest.importorskip('study')
    arrays = study.job_arrays(study.DEV_SEEDS[0], 600)
    z = arrays['train_z'].astype(np.float64)
    covariance = np.cov(z, rowvar=False)
    assert np.allclose(covariance, np.eye(60), atol=1e-4)
    assert abs(float(np.linalg.norm(z, axis=1).mean()) - math.sqrt(60)) < 0.5


# ------------------------------------------------------------- preprocessing ---
def test_within_class_transform_spheres_the_within_class_covariance():
    torch.manual_seed(12)
    means = torch.randn(10, 5, dtype=torch.float64) * 3.0
    labels = torch.randint(0, 10, (600,))
    root = torch.randn(5, 5, dtype=torch.float64)
    noise = torch.randn(600, 5, dtype=torch.float64) @ root
    x = means[labels] + noise
    transform = kernels.within_class_transform(x, labels, 0.0)
    sphered = x @ transform
    centred = sphered.clone()
    for label in range(10):
        mask = labels == label
        centred[mask] -= sphered[mask].mean(0, keepdim=True)
    within = (centred.T @ centred) / (600 - 10)
    scaled = within / float(torch.diagonal(within).mean())
    assert torch.allclose(scaled, torch.eye(5, dtype=torch.float64), atol=1e-8)


def test_within_class_transform_is_rotation_equivariant():
    torch.manual_seed(13)
    labels = torch.randint(0, 4, (200,))
    x = torch.randn(200, 4, dtype=torch.float64) + labels.unsqueeze(1).to(torch.float64)
    rotation = torch.linalg.qr(torch.randn(4, 4, dtype=torch.float64))[0]
    plain = x @ kernels.within_class_transform(x, labels, 0.2)
    turned = (x @ rotation) @ kernels.within_class_transform(x @ rotation, labels, 0.2)
    assert torch.allclose(turned, plain @ rotation, atol=1e-8)


def test_rewhiten_query_makes_the_query_covariance_identity():
    rng = np.random.default_rng(14)
    scale = np.diag(np.linspace(0.2, 3.0, 6))
    x = (rng.standard_normal((300, 6)) @ scale).astype(np.float32)
    q = (rng.standard_normal((900, 6)) @ scale).astype(np.float32)
    config = kernels.resolve_config({'family': 'krr', 'rewhiten': 'query'})
    train, query, note = kernels._preprocess(
        torch.as_tensor(x, dtype=torch.float64), torch.as_tensor(q, dtype=torch.float64),
        torch.zeros(300, dtype=torch.long), config)
    covariance = torch.cov(query.T)
    assert torch.allclose(covariance, torch.eye(6, dtype=torch.float64), atol=1e-8)
    assert note['rewhiten'] == 'query' and note['rewhiten_rows'] == 900


def test_float32_dtype_runs_and_agrees_roughly_with_float64():
    x, y, centers = blobs(200, 121)
    q, _, _ = blobs(120, 122, centers=centers)
    config = {'family': 'krr', 'kernel': 'arccos', 'depth': 2, 'ridge': 1e-4,
              'bandwidth': 1.0}
    wide = kernels.fit_predict(x, y, q, config, 5, 'cpu', deadline())
    narrow = kernels.fit_predict(x, y, q, {**config, 'dtype': 'float32'}, 5, 'cpu',
                                 deadline())
    assert narrow['metrics']['dtype'] == 'float32'
    assert float((wide['labels'] != narrow['labels']).mean()) < 0.02


# ------------------------------------------------------------------ ensemble ---
ENSEMBLE_CONFIG = {'family': 'ensemble', 'members': [
    {'family': 'krr', 'kernel': 'arccos', 'depth': 3},
    {'family': 'krr', 'kernel': 'laplace', 'rewhiten': 'query'},
    {'family': 'labelprop', 'k_grid': [5, 10]},
]}


def test_ensemble_runs_all_members_and_marks_transduction():
    x, y, centers = blobs(220, 131)
    q, _, _ = blobs(150, 132, centers=centers)
    out = kernels.fit_predict(x, y, q, ENSEMBLE_CONFIG, 5, 'cpu', deadline())
    metrics = out['metrics']
    assert metrics['members_used'] == 3 and metrics['members_total'] == 3
    assert metrics['uses_query_images_unlabeled'] is True
    assert out['logits'].shape == (150, 10)
    json.dumps(metrics, allow_nan=False)


def test_ensemble_of_one_matches_the_member_up_to_standardisation():
    x, y, centers = blobs(200, 141)
    q, _, _ = blobs(120, 142, centers=centers)
    member = {'family': 'krr', 'kernel': 'arccos', 'depth': 2}
    alone = kernels.fit_predict(x, y, q, member, 5, 'cpu', deadline())
    wrapped = kernels.fit_predict(x, y, q, {'family': 'ensemble', 'members': [member]},
                                  5, 'cpu', deadline())
    assert np.array_equal(alone['labels'], wrapped['labels'])
    assert wrapped['metrics']['uses_query_images_unlabeled'] is False


def test_ensemble_is_invariant_to_the_member_order():
    """A member used to be fitted with ``seed + index``: order changed the fit."""
    x, y, centers = blobs(500, 181, dim=30, spread=0.55)
    q, _, _ = blobs(300, 182, dim=30, spread=0.55, centers=centers)
    members = [{'family': 'krr', 'kernel': 'arccos', 'depth': 3},
               {'family': 'krr', 'kernel': 'arccos', 'depth': 2}]
    forward = kernels.fit_predict(x, y, q, {'family': 'ensemble', 'members': members},
                                  5, 'cpu', deadline())
    reverse = kernels.fit_predict(x, y, q,
                                  {'family': 'ensemble', 'members': members[::-1]},
                                  5, 'cpu', deadline())
    assert np.array_equal(forward['labels'], reverse['labels'])
    assert forward['metrics']['chosen'] == reverse['metrics']['chosen'][::-1]


def test_every_ensemble_member_is_fitted_like_the_same_config_alone():
    """The in-module average must reproduce the offline average of the members."""
    x, y, centers = blobs(500, 191, dim=30, spread=0.55)
    q, _, _ = blobs(300, 192, dim=30, spread=0.55, centers=centers)
    members = [{'family': 'krr', 'kernel': 'arccos', 'depth': 3},
               {'family': 'krr', 'kernel': 'laplace'},
               {'family': 'krr', 'kernel': 'arccos', 'depth': 2}]
    alone = [kernels.fit_predict(x, y, q, member, 5, 'cpu', deadline())
             for member in members]
    wrapped = kernels.fit_predict(x, y, q, {'family': 'ensemble', 'members': members},
                                  5, 'cpu', deadline())
    for index, member in enumerate(alone):
        assert wrapped['metrics']['chosen'][index] == member['metrics']['chosen'], index
    stacked = []
    for member in alone:                      # the offline recipe: z-score, average
        logits = member['logits'].astype(np.float64)
        logits = logits - logits.mean(axis=1, keepdims=True)
        stacked.append(logits / max(float(logits.std()), 1e-12))
    offline = np.mean(stacked, axis=0).argmax(axis=1).astype(np.uint8)
    assert np.array_equal(offline, wrapped['labels'])


def test_ensemble_weights_select_a_member():
    x, y, centers = blobs(200, 151)
    q, _, _ = blobs(120, 152, centers=centers)
    members = [{'family': 'krr', 'kernel': 'arccos', 'depth': 2},
               {'family': 'krr', 'kernel': 'laplace'}]
    first = kernels.fit_predict(x, y, q, members[0], 5, 'cpu', deadline())
    weighted = kernels.fit_predict(x, y, q, {'family': 'ensemble', 'members': members,
                                             'weights': [1.0, 0.0]}, 5, 'cpu', deadline())
    assert np.array_equal(first['labels'], weighted['labels'])


def test_ensemble_config_validation():
    with pytest.raises(ValueError, match='members'):
        kernels.resolve_config({'family': 'ensemble', 'members': []})
    with pytest.raises(ValueError, match='nest'):
        kernels.resolve_config({'family': 'ensemble', 'members': [
            {'family': 'ensemble', 'members': [{'family': 'krr'}]}]})
    with pytest.raises(ValueError, match='weights'):
        kernels.resolve_config({'family': 'ensemble', 'members': [{'family': 'krr'}],
                                'weights': [1.0, 1.0]})
    with pytest.raises(ValueError, match='Unknown config keys'):
        kernels.resolve_config({'family': 'ensemble', 'members': [{'family': 'krr'}],
                                'bogus': 1})


def test_selftrain_applies_its_base_preprocessing():
    """A sphered base must really be sphered (the wrapper used to swallow it)."""
    x, y, centers = blobs(300, 161, dim=12)
    q, _, _ = blobs(200, 162, dim=12, centers=centers)
    plain = {'family': 'selftrain', 'base': {'family': 'krr', 'kernel': 'arccos',
                                             'depth': 3}}
    sphered = {'family': 'selftrain', 'base': {'family': 'krr', 'kernel': 'arccos',
                                               'depth': 3,
                                               'metric_learn': 'within_class'}}
    first = kernels.fit_predict(x, y, q, plain, 5, 'cpu', deadline())
    second = kernels.fit_predict(x, y, q, sphered, 5, 'cpu', deadline())
    assert second['metrics']['base_preprocess']['metric_learn'] == 'within_class'
    assert first['metrics']['base_preprocess']['rewhiten'] == 'none'
    assert not np.array_equal(first['logits'], second['logits'])


def test_nested_configs_reject_top_level_only_keys():
    with pytest.raises(ValueError, match='prior_match belongs on the wrapper'):
        kernels.resolve_config({'family': 'selftrain',
                                'base': {'family': 'krr', 'prior_match': 'sinkhorn'}})
    with pytest.raises(ValueError, match='prior_match belongs on the ensemble'):
        kernels.resolve_config({'family': 'ensemble', 'members': [
            {'family': 'krr', 'prior_match': 'sinkhorn'}]})
    with pytest.raises(ValueError, match='rfm_config may not set'):
        kernels.resolve_config({'family': 'labelprop', 'metric': 'rfm',
                                'rfm_config': {'rewhiten': 'query'}})


def test_predictions_are_equivariant_to_a_relabelling():
    """The harness relabels the classes per draw; accuracy must not notice.

    ``study.py`` therefore drops the permutation, and this is the check that
    licenses dropping it: permuting the training labels permutes the score
    columns and nothing else.
    """
    x, y, centers = blobs(220, 171)
    q, _, _ = blobs(140, 172, centers=centers)
    permutation = np.random.default_rng(3).permutation(10)
    inverse = np.argsort(permutation)
    for config in [{'family': 'krr', 'kernel': 'arccos', 'depth': 3},
                   {'family': 'krr', 'kernel': 'laplace',
                    'metric_learn': 'within_class'},
                   {'family': 'labelprop', 'k_grid': [5, 10]},
                   {'family': 'selftrain', 'base': {'family': 'krr'}},
                   ENSEMBLE_CONFIG]:
        plain = kernels.fit_predict(x, y, q, config, 5, 'cpu', deadline())
        relabelled = kernels.fit_predict(x, permutation[y].astype(np.uint8), q, config,
                                         5, 'cpu', deadline())
        assert np.array_equal(inverse[relabelled['labels']], plain['labels']), config
        assert np.allclose(relabelled['logits'][:, permutation], plain['logits'],
                           atol=1e-6), config
