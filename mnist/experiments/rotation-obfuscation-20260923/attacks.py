"""Attacks on the released features.  Each function receives only what an
entrant sees (released train features + labels, released test features, and for
the informed attacker a DISJOINT public pixel-space pool).  The secret map is
passed separately to *scoring* helpers only."""
from __future__ import annotations

from pathlib import Path
import sys
import time

import numpy as np
import torch

import common

PMNIST = Path(__file__).resolve().parent / 'pmnist'
if not (PMNIST / 'topology.py').exists():
    PMNIST = Path('/Users/yaroslavvb/git/sutro-problems/mnist/experiments/pmnist-medium-cutoffs-20260923')
if str(PMNIST) not in sys.path:
    sys.path.insert(0, str(PMNIST))
import topology  # noqa: E402  (lattice recovery from a similarity matrix; pmnist study)

GRID, NFEAT = common.GRID, common.NFEAT
RC = np.array([(i // GRID, i % GRID) for i in range(NFEAT)], dtype=np.float64)


# =============================================================== blind attacks
def fastica(z_all, fun='logcosh', seed=0, max_iter=2000, tol=1e-5):
    """Returns sources S (n, d) and unmixing B with S = (z - mean) @ B.T, signs fixed
    so every source is right-skewed (pixels are)."""
    from sklearn.decomposition import FastICA
    m = FastICA(n_components=None, whiten='unit-variance', fun=fun, max_iter=max_iter,
                tol=tol, random_state=seed)
    s = m.fit_transform(z_all.astype(np.float64))
    b = np.asarray(m.components_, np.float64)
    skew = ((s - s.mean(0)) ** 3).mean(0)
    sgn = np.where(skew < 0, -1.0, 1.0)
    return s * sgn[None, :], b * sgn[:, None], m.mean_, int(m.n_iter_)


def sparse_nonneg_unmix(z_all, seed=0, steps=3000, lr=0.02, l1=1.0, neg=5.0):
    """Blind maximum-likelihood style unmixing with a Laplace (sparsity) prior and a
    penalty on negative values: min -log|det B| + l1*mean|s| + neg*mean(relu(-s)^2),
    s = B (z - mean) after the attacker's own whitening.  Returns (S, B_total, mean)."""
    torch.manual_seed(seed)
    z = np.asarray(z_all, np.float64)
    mu = z.mean(0)
    cov = np.cov(z - mu, rowvar=False)
    lam, u = np.linalg.eigh(cov)
    wz = (u / np.sqrt(np.clip(lam, 0, None) + 1e-6)[None, :]) @ u.T     # attacker's own ZCA
    v = torch.as_tensor((z - mu) @ wz.T, dtype=torch.float32)
    d = v.shape[1]
    b = torch.nn.Parameter(torch.eye(d) + 0.01 * torch.randn(d, d))
    opt = torch.optim.Adam([b], lr=lr)
    n = len(v)
    for it in range(steps):
        idx = torch.randint(0, n, (4096,))
        s = v[idx] @ b.T
        loss = -torch.logdet(b @ b.T) / 2 + l1 * s.abs().mean() * d + neg * torch.relu(-s).pow(2).mean() * d
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    b = b.detach().numpy().astype(np.float64)
    s = (v.numpy().astype(np.float64)) @ b.T
    skew = ((s - s.mean(0)) ** 3).mean(0)
    sgn = np.where(skew < 0, -1.0, 1.0)
    b_total = (b * sgn[:, None]) @ wz
    return s * sgn[None, :], b_total, mu


def source_similarity(s, kind='abs-pcorr'):
    """Neighbourhood signal between recovered sources.  ICA sources are white, so
    plain correlation carries nothing ('raw-corr' is the control); co-activation
    of magnitudes (|s|) or rectified values (max(s,0)) is what survives."""
    s = np.asarray(s, np.float64)
    if kind.startswith('abs'):
        f = np.abs(s)
    elif kind.startswith('pos'):
        f = np.maximum(s, 0.0)
    elif kind.startswith('raw'):
        f = s
    else:
        raise ValueError(kind)
    if kind.endswith('pcorr'):
        sim = topology.partial_correlation(f, ridge=topology.RIDGE)
    else:
        sim = np.corrcoef(f, rowvar=False)
        np.fill_diagonal(sim, 0.0)
    return np.nan_to_num(np.clip(sim, 0.0, None))


def recover_lattice(sim, max_seconds=120.0):
    """Arrange d <= 81 coordinates on the 9x9 lattice from a similarity matrix,
    reusing the pmnist study's embedding + QAP search (topology.py).  Returns
    layout (81,), the QAP objective of the winner, and the objective the search
    would give the identity arrangement (for reference)."""
    sim = np.asarray(sim, np.float64)
    d = sim.shape[0]
    full = np.zeros((NFEAT, NFEAT))
    full[:d, :d] = sim
    np.fill_diagonal(full, 0.0)
    starts = []
    for k in (4, 5, 6):
        for prune in (True, False):
            g = topology._mutual_knn(sim, k)
            if prune:
                g = topology._prune_shortcuts(g)
            hops = topology._hop_distances(g)
            if not np.isfinite(hops).all() or hops.max() <= 0:
                continue
            classical, _ = topology._classical_mds(hops)
            if not np.isfinite(classical).all():
                continue
            embeddings = [classical]
            try:
                embeddings.append(topology._smacof(hops, classical, 1.0 / np.maximum(hops, 1e-9)))
            except Exception:
                pass
            for emb in embeddings:
                if not np.isfinite(emb).all():
                    continue
                cells, _ = topology._align_to_lattice(emb)
                start = np.empty(NFEAT, np.int64)
                start[:d] = cells
                start[d:] = np.setdiff1d(np.arange(NFEAT), cells)
                starts.append(start)
    if not starts:
        starts = [np.random.default_rng(0).permutation(NFEAT)]
    layout, best, _ = topology._search(full, starts, time.time(), max_seconds)
    return np.asarray(layout, np.int64), float(best)


def lattice_from_sources(s, feed='abs-pcorr', max_seconds=120.0):
    t = time.perf_counter()
    layout, best = recover_lattice(source_similarity(s, feed), max_seconds)
    return layout, time.perf_counter() - t, {'qap_objective': best}


def arrange(s, layout):
    """Place source j at lattice cell layout[j]; returns (n, 9, 9)."""
    n, d = s.shape
    img = np.zeros((n, NFEAT), np.float32)
    img[:, layout[:d]] = s[:, :d]
    return img.reshape(n, GRID, GRID)


# ======================================================== informed attacks
def _sqrt_pair(cov, ridge):
    lam, u = np.linalg.eigh(0.5 * (cov + cov.T))
    lam = np.clip(lam, 0, None) + ridge
    return (u * np.sqrt(lam)[None, :]) @ u.T, (u / np.sqrt(lam)[None, :]) @ u.T, lam, u


def _class_stats(v, y):
    means, covs, skews = [], [], []
    for c in range(10):
        vc = v[y == c]
        means.append(vc.mean(0))
        covs.append(np.cov(vc, rowvar=False))
    return np.stack(means), np.stack(covs)


def _skew_along(v, dirs):
    p = v @ dirs
    p = p - p.mean(0)
    return (p ** 3).mean(0) / np.maximum((p ** 2).mean(0) ** 1.5, 1e-12)


def _procrustes(pairs_v, pairs_u, weights=None):
    """R (d x 81, orthonormal rows) minimising sum_k w_k ||R u_k - v_k||^2."""
    w = np.ones(len(pairs_v)) if weights is None else np.asarray(weights)
    cross = (pairs_v * w[:, None]).T @ pairs_u          # d x 81
    uu, _, vt = np.linalg.svd(cross, full_matrices=False)
    return uu @ vt


def informed_attack(z_train, y_train, z_test, x_pub, y_pub, init='both', refine_steps=600,
                    top_eig=6, ridge=1e-4, seed=0, log=None, lr_refine=2e-3, n_rlc=40, lbfgs_steps=200, k_att=60,
                    oracle_A=None, top_rlc=12, hops=3, nonneg_weight=5000.0):
    """Distribution-matching recovery of pixels from an informed attacker.

    Model: released z = G (x - mu) + b for an unknown linear G.  Whiten both sides
    with the attacker's own statistics: u = Sig_pub^-1/2 (x - mu_pub) on the public
    pool, v = S_rel^-1/2 (z - m_rel) on the released rows.  Then v ~ R u with R on the
    Stiefel manifold.  R is initialised from (a) rank-ordered eigenvector matching of
    the two covariances with skewness-fixed signs (unlabelled; only works when the
    release is not exactly white), (b) Procrustes on class means and class-covariance
    eigenvectors (labelled), then refined by gradient descent on the class-conditional
    second-moment mismatch.  Pixel estimate: x_hat = mu_pub + Sig_pub^1/2 R^T v.
    Returns dict of per-init results, each with the linear map H (81 x d) and offset."""
    x_pub = np.asarray(x_pub, np.float64)
    mu_pub = x_pub.mean(0)
    sig_pub = np.cov(x_pub - mu_pub, rowvar=False)
    z_all = np.concatenate([z_train, z_test]).astype(np.float64)
    m_rel = z_all.mean(0)
    s_rel = np.cov(z_all - m_rel, rowvar=False)
    d_rel = z_all.shape[1]
    # Both sides are PCA-whitened to their top-k principal subspaces (k = k_att).
    # Truncation is what makes the attack robust to an organiser's variance floor:
    # the floor inflates ~20 near-null noise directions of an 81-d release to a
    # sizeable variance, and re-whitening them would hand the fit 20 unit-variance
    # noise coordinates.  The top-k subspaces of both sides correspond because
    # whitening with a floor is monotone in the eigenvalue.
    k = min(k_att or d_rel, d_rel, NFEAT)
    lam_p, e_p = np.linalg.eigh(0.5 * (sig_pub + sig_pub.T)); lam_p = np.clip(lam_p, 0, None)
    lam_r, e_r = np.linalg.eigh(0.5 * (s_rel + s_rel.T)); lam_r = np.clip(lam_r, 0, None)
    e_p, lam_p = e_p[:, ::-1][:, :k], lam_p[::-1][:k]
    e_r, lam_r = e_r[:, ::-1][:, :k], lam_r[::-1][:k]
    isqrt_pub = (e_p / np.sqrt(lam_p + ridge)[None, :]).T       # k x 81
    sqrt_pub = e_p * np.sqrt(lam_p + ridge)[None, :]            # 81 x k
    isqrt_rel = (e_r / np.sqrt(lam_r + ridge)[None, :]).T       # k x d_rel
    d = k
    # rank-ordered eigenvectors in the whitened coordinates are the unit vectors
    v_rel = np.eye(k)[:, ::-1]                                   # ascending order like eigh
    u_pub = np.eye(k)[:, ::-1]

    u = (x_pub - mu_pub) @ isqrt_pub.T                 # public whitened, n_pub x k
    v_tr = (z_train - m_rel) @ isqrt_rel.T             # released whitened, n x k
    v_all = (z_all - m_rel) @ isqrt_rel.T

    inits = {}
    # (a) unlabelled: match eigenvectors by rank (largest eigenvalues first)
    if init in ('eig', 'both', 'all'):
        # released eigenvector i (in v-coords: isqrt_rel @ v_rel[:, i] direction = v_rel[:, i]) <-> public eigenvector
        ev_rel = v_rel[:, ::-1][:, :d]                 # k x k, descending
        ev_pub = u_pub[:, ::-1][:, :d]                 # k x k, descending
        sk_rel = _skew_along(v_all, ev_rel)
        sk_pub = _skew_along(u, ev_pub)
        sgn = np.where(np.sign(sk_rel) == np.sign(sk_pub), 1.0, -1.0)
        inits['eig'] = (ev_rel * sgn[None, :]) @ ev_pub.T          # d x 81
    # (b) labelled: class means + class covariance eigenvectors
    if init in ('cls', 'both', 'all'):
        m_u, c_u = _class_stats(u, y_pub)
        m_v, c_v = _class_stats(v_tr, y_train)
        pairs_v, pairs_u, w = [], [], []
        for c in range(10):
            pairs_v.append(m_v[c]); pairs_u.append(m_u[c]); w.append(4.0)
            lv, ev = np.linalg.eigh(c_v[c]); lu, eu = np.linalg.eigh(c_u[c])
            ev, eu = ev[:, ::-1][:, :top_eig], eu[:, ::-1][:, :top_eig]
            sv = _skew_along(v_tr[y_train == c], ev); su = _skew_along(u[y_pub == c], eu)
            sgn = np.where(np.sign(sv) == np.sign(su), 1.0, -1.0)
            for j in range(top_eig):
                pairs_v.append(ev[:, j] * sgn[j]); pairs_u.append(eu[:, j]); w.append(1.0)
        inits['cls'] = _procrustes(np.stack(pairs_v), np.stack(pairs_u), w)
    if init in ('random', 'all'):
        inits['random'] = common.haar(d, seed + 99)
    # (c) labelled, closed form: a random linear combination of class covariances is
    # NOT white, and C_v(alpha) = R C_u(alpha) R^T, so its eigenvectors match by rank
    # (generically distinct eigenvalues); signs from class-mean projections.  The
    # best-separated eigenvectors of n_rlc random combinations are pooled into one
    # Procrustes problem, which averages out the class-covariance estimation noise.
    if init in ('rlc', 'all'):
        m_u, c_u = _class_stats(u, y_pub)
        m_v, c_v = _class_stats(v_tr, y_train)
        rng = np.random.default_rng(seed)
        pairs_v, pairs_u = [], []
        for trial in range(n_rlc):
            alpha = rng.standard_normal(10)
            cu_a = np.tensordot(alpha, c_u, 1); cv_a = np.tensordot(alpha, c_v, 1)
            lu, eu = np.linalg.eigh(cu_a); lv, ev = np.linalg.eigh(cv_a)
            # the largest-magnitude eigenvalues are the best separated: keep top_rlc from each end
            ou, ov = np.argsort(-np.abs(lu)), np.argsort(-np.abs(lv))
            eu, ev = eu[:, ou[:top_rlc]], ev[:, ov[:top_rlc]]
            pu, pv = m_u @ eu, m_v @ ev                       # 10 x t class-mean projections fix the signs
            sgn = np.where((pu * pv).sum(0) >= 0, 1.0, -1.0)
            pairs_v.append(ev * sgn[None, :]); pairs_u.append(eu)
        pairs_v = np.concatenate(pairs_v, 1).T; pairs_u = np.concatenate(pairs_u, 1).T
        inits['rlc'] = _procrustes(pairs_v, pairs_u)

    # refinement on class-conditional moments (labelled) -- shared for every init
    m_u, c_u = _class_stats(u, y_pub)
    m_v, c_v = _class_stats(v_tr, y_train)
    cu_t = torch.as_tensor(c_u, dtype=torch.float64); cv_t = torch.as_tensor(c_v, dtype=torch.float64)
    mu_t = torch.as_tensor(m_u, dtype=torch.float64); mv_t = torch.as_tensor(m_v, dtype=torch.float64)

    # non-negativity prior: reconstructed pixels mu_pub + sqrt_pub R^T v must not be negative
    v_sub = torch.as_tensor(v_all[np.random.default_rng(seed).permutation(len(v_all))[:4000]], dtype=torch.float64)
    sqrt_pub_t = torch.as_tensor(sqrt_pub, dtype=torch.float64)
    mu_pub_t = torch.as_tensor(mu_pub, dtype=torch.float64)

    def moment_loss(r, with_prior=True):
        pred_c = r @ cu_t @ r.T                         # 10 x d x d
        pred_m = mu_t @ r.T                             # 10 x d
        loss = ((pred_c - cv_t) ** 2).sum() + 4.0 * ((pred_m - mv_t) ** 2).sum()
        if with_prior and nonneg_weight:
            x_hat = mu_pub_t + (v_sub @ r) @ sqrt_pub_t.T          # n x 81
            loss = loss + nonneg_weight * torch.relu(-x_hat).pow(2).mean() * NFEAT
        return loss

    out = {}
    if oracle_A is not None:      # DIAGNOSTIC ONLY: moment loss at the true map, for the write-up
        r_true = isqrt_rel @ np.asarray(oracle_A, np.float64) @ sqrt_pub          # k x k
        uu, _, vt = np.linalg.svd(r_true); r_true = uu @ vt
        out['__oracle__'] = {'init_loss': float(moment_loss(torch.as_tensor(r_true), with_prior=False)), 'final_loss': float(moment_loss(torch.as_tensor(r_true), with_prior=False)),
                             'init': {'H': sqrt_pub @ r_true.T @ isqrt_rel, 'offset': mu_pub - (sqrt_pub @ r_true.T @ isqrt_rel) @ m_rel},
                             'refined': {'H': sqrt_pub @ r_true.T @ isqrt_rel, 'offset': mu_pub - (sqrt_pub @ r_true.T @ isqrt_rel) @ m_rel}}
        if log:
            log(f'    informed ORACLE (true map, diagnostic): loss {out["__oracle__"]["init_loss"]:.3g}')
    for name, r0 in inits.items():
        rec = {'init_loss': float(moment_loss(torch.as_tensor(r0), with_prior=False))}
        r0_t = torch.as_tensor(r0, dtype=torch.float64)
        free = torch.nn.Parameter(torch.zeros(d, d, dtype=torch.float64))
        opt = torch.optim.Adam([free], lr=lr_refine)

        def current():
            skew = free - free.T
            return r0_t @ torch.linalg.matrix_exp(skew)

        for it in range(refine_steps):
            loss = moment_loss(current())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if lbfgs_steps:
            lb = torch.optim.LBFGS([free], lr=1.0, max_iter=lbfgs_steps, history_size=20,
                                   line_search_fn='strong_wolfe', tolerance_grad=1e-9, tolerance_change=1e-12)

            def closure():
                lb.zero_grad(set_to_none=True)
                l = moment_loss(current())
                l.backward()
                return l
            lb.step(closure)
        with torch.no_grad():
            r_final = current().numpy()
        best_loss = float(moment_loss(torch.as_tensor(r_final)))
        # basin hopping: perturb the solution by a small random rotation and re-polish, keep the best
        gen = torch.Generator().manual_seed(seed + 7)
        for hop in range(hops):
            with torch.no_grad():
                pert = torch.randn(d, d, generator=gen, dtype=torch.float64) * (0.15 / np.sqrt(d))
                free_backup = free.detach().clone()
                free.add_(pert)
            lb = torch.optim.LBFGS([free], lr=1.0, max_iter=lbfgs_steps, history_size=20,
                                   line_search_fn='strong_wolfe', tolerance_grad=1e-9, tolerance_change=1e-12)
            lb.step(closure)
            with torch.no_grad():
                cand = current().numpy()
                cand_loss = float(moment_loss(torch.as_tensor(cand)))
                if cand_loss < best_loss:
                    best_loss, r_final = cand_loss, cand
                else:
                    free.copy_(free_backup)
        rec['final_loss'] = float(moment_loss(torch.as_tensor(r_final), with_prior=False))
        rec['final_objective'] = best_loss
        for tag, rr in (('init', r0), ('refined', r_final)):
            h = sqrt_pub @ rr.T @ isqrt_rel             # 81 x d_rel ; x_hat = mu_pub + h (z - m_rel)
            rec[tag] = {'H': h, 'offset': mu_pub - h @ m_rel}
        out[name] = rec
        if log:
            log(f'    informed init={name}: loss {rec["init_loss"]:.3g} -> {rec["final_loss"]:.3g}')
    return out


def apply_linear(rec, z):
    return (np.asarray(z, np.float64) @ rec['H'].T + rec['offset'][None, :]).astype(np.float32)


# ============================================== exact-source (known hole) attack
def norm_match_labels(z_test, x_all, y_all, w_att, mu_att):
    """Attacker holds every source image: match each released test row to the pool
    image with the nearest rotation-invariant norm ||W_att (x - mu_att)||."""
    cand = np.linalg.norm((x_all.astype(np.float64) - mu_att) @ w_att.T, axis=1)
    order = np.argsort(cand)
    r = np.linalg.norm(z_test.astype(np.float64), axis=1)
    pos = np.searchsorted(cand[order], r)
    pos = np.clip(pos, 1, len(cand) - 1)
    left, right = order[pos - 1], order[pos]
    pick = np.where(np.abs(cand[left] - r) < np.abs(cand[right] - r), left, right)
    return y_all[pick], pick


# ================================================================= scoring
def source_centroids(composite):
    """composite[i, p] = weight of recovered coordinate i on true pixel p.
    Returns energy-weighted centroid (row, col) per coordinate and the energy
    fraction within radius 1.5 of it (localisation)."""
    e = np.asarray(composite, np.float64) ** 2
    e = e / np.maximum(e.sum(1, keepdims=True), 1e-300)
    cen = e @ RC
    d = np.sqrt(((RC[None, :, :] - cen[:, None, :]) ** 2).sum(-1))
    local = (e * (d <= 1.5)).sum(1)
    return cen, local


def layout_quality(layout, composite):
    """How good is a recovered lattice arrangement of the recovered coordinates?
    An edge of the recovered lattice joins coordinates i, j; score it by the
    distance between their true centroids.  Returns precision at <=1.0 and <=1.5
    plus the mean edge distance; chance = the same for a random layout."""
    cen, local = source_centroids(composite)
    d = composite.shape[0]
    lay = np.asarray(layout)[:d]
    rc = RC[lay]
    dd = ((rc[:, None, :] - rc[None, :, :]) ** 2).sum(-1)
    adj = np.triu(dd == 1, 1)
    i, j = np.where(adj)
    true_d = np.sqrt(((cen[i] - cen[j]) ** 2).sum(1))
    rng = np.random.default_rng(0)
    chance = []
    for _ in range(20):
        p = rng.permutation(d)
        rcp = RC[np.arange(NFEAT)[:d]][np.argsort(p)] if d == NFEAT else RC[rng.permutation(NFEAT)[:d]]
        ddp = ((rcp[:, None, :] - rcp[None, :, :]) ** 2).sum(-1)
        ii, jj = np.where(np.triu(ddp == 1, 1))
        chance.append(np.sqrt(((cen[ii] - cen[jj]) ** 2).sum(1)))
    chance = np.concatenate(chance)
    return {'n_edges': int(len(true_d)),
            'edge_precision_at_1.0': float((true_d <= 1.01).mean()),
            'edge_precision_at_1.5': float((true_d <= 1.5).mean()),
            'edge_mean_true_distance': float(true_d.mean()),
            'chance_precision_at_1.0': float((chance <= 1.01).mean()),
            'chance_precision_at_1.5': float((chance <= 1.5).mean()),
            'chance_mean_true_distance': float(chance.mean()),
            'mean_localisation_r1.5': float(local.mean())}


# ====================================== blind: zero-atom (facet) polish
def atom_polish(z_all, b_init, steps=600, lr=0.01, h_rel=0.03, h_start=0.6, q=0.002, batch=8192, seed=0):
    """Refine unmixing directions so each projection has maximal point mass at its
    minimum.  Pixels are exactly zero in a large fraction of images, so in every
    invertible linear image of the pixels the pixel axes are facets of the data
    cone; a direction that is 'almost' a pixel sees a smeared minimum, the true
    axis sees a point atom.  Objective per row m: mean_i sigmoid((h - (m.z_i - q_low)) / (h/4))
    with q_low the q-quantile of the projection (robust minimum).  The window h is
    annealed from h_start*std (a smooth 'mass near the minimum' contrast with
    gradient everywhere) down to h_rel*std (the atom itself).  Rows are kept at unit
    norm and are optimised independently.  Returns (S, B_total, mean)."""
    torch.manual_seed(seed)
    z = np.asarray(z_all, np.float64)
    mu = z.mean(0)
    zc = torch.as_tensor(z - mu, dtype=torch.float32)
    b0 = np.asarray(b_init, np.float64)
    b0 = b0 / np.linalg.norm(b0, axis=1, keepdims=True)
    m = torch.nn.Parameter(torch.as_tensor(b0, dtype=torch.float32))
    opt = torch.optim.Adam([m], lr=lr)
    n = len(zc)
    for it in range(steps):
        frac = it / max(steps - 1, 1)
        h_now = h_start * (h_rel / h_start) ** frac
        idx = torch.randint(0, n, (min(batch, n),))
        mm = m / m.norm(dim=1, keepdim=True)
        p = zc[idx] @ mm.T                                   # batch x d
        with torch.no_grad():
            low = torch.quantile(p, q, dim=0)
            h = h_now * p.std(dim=0) + 1e-6
        mass = torch.sigmoid((h - (p - low)) / (h / 4.0)).mean(0)   # d
        loss = -mass.sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        mm = (m / m.norm(dim=1, keepdim=True)).numpy().astype(np.float64)
    s = (z - mu) @ mm.T
    skew = ((s - s.mean(0)) ** 3).mean(0)
    sgn = np.where(skew < 0, -1.0, 1.0)
    return s * sgn[None, :], mm * sgn[:, None], mu


def atom_mass(s, q=0.002, h_rel=0.03):
    """Attacker-side certificate: fraction of samples within h of the robust minimum, per coordinate."""
    s = np.asarray(s, np.float64)
    low = np.quantile(s, q, axis=0)
    h = h_rel * s.std(0) + 1e-12
    return ((s - low[None, :]) <= h[None, :]).mean(0)


def unmixing_from_sources(z_all, s):
    """Recover the linear map B with s = (z - mean) B^T by least squares (attacker-side)."""
    z = np.asarray(z_all, np.float64)
    mu = z.mean(0)
    bt, *_ = np.linalg.lstsq(z - mu, np.asarray(s, np.float64), rcond=None)
    return bt.T, mu


def procrustes_from_matched(z_rows, x_matched):
    """Exact-source continuation: with rows matched to public images, fit the full
    linear map z -> x by least squares and return (H, c) with x_hat = z H^T + c."""
    z = np.asarray(z_rows, np.float64)
    x = np.asarray(x_matched, np.float64)
    zb = np.concatenate([z, np.ones((len(z), 1))], 1)
    coef, *_ = np.linalg.lstsq(zb, x, rcond=None)
    return coef[:-1].T, coef[-1]


def tail_leak_report(z_all, a_secret, m=20):
    """Scoring-side: how pixel-like are the bottom-m and top-m eigenvectors of the
    released covariance?  (eps-floored whitening leaves the low-variance tail
    identifiable; exact whitening does not.)"""
    z = np.asarray(z_all, np.float64)
    lam, v = np.linalg.eigh(np.cov(z - z.mean(0), rowvar=False))
    filt = (v.T @ np.asarray(a_secret, np.float64))         # row i: pixel-space filter of eigenvector i
    _, loc = source_centroids(filt)
    e = filt ** 2; peak = e.max(1) / np.maximum(e.sum(1), 1e-300)
    return {'bottom_m_mean_localisation': float(loc[:m].mean()), 'bottom_m_n_peak_above_0.9': int((peak[:m] > 0.9).sum()),
            'top_m_mean_localisation': float(loc[-m:].mean()), 'top_m_n_peak_above_0.9': int((peak[-m:] > 0.9).sum()),
            'eigenvalues_bottom_m': lam[:m].tolist(), 'eigenvalues_top_m': lam[-m:].tolist()}
