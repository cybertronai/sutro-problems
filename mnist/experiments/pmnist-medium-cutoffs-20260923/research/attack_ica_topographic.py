"""Topographic-ICA attack on the whitened + rotated variant (second opinion).

NOTE ON FILE LOCATION
---------------------
This module was written as ``ROOT/attack_ica.py``; while it was running, a second
agent published a different ``ROOT/attack_ica.py`` (a signed-permutation /
``recover_layout`` reuse variant of the same idea).  Rather than overwrite it, this
copy lives under ``research/``.  The two agree on the headline conclusion and were
produced independently, which is worth something.

What it does
------------
``topology.recover_layout`` reconstructs the 9x9 lattice from the *second-order*
statistics of the permuted pixels, so the permutation alone is not an obfuscation.
``whiten.py`` removes that cue: after ZCA whitening plus a random rotation the
released covariance is (nearly) the identity whatever the rotation was.  Whitening
is an *invertible linear map*, so no information is destroyed -- the question is
only whether an attacker can find the rotation from higher-order statistics.

    1. FastICA on the released features (label-free: train rows + query images).
    2. Topographic affinity between components (correlation of |s|^p energies),
       fed to the same lattice-QAP machinery ``topology.py`` uses, producing a
       candidate 9x9 layout *of the components*.
    3. Ground-truth evaluation with the known transform (evaluation only): how
       localised is each component in pixel space, and does the recovered layout
       agree with the true grid?
    4. Downstream: train the frozen cnn-09 recipe on the sources laid out by the
       recovered layout, against a random-layout control and an oracle layout.

No Modal, no GPU, no final seed.  Query labels enter only via
``study.pool_labels`` indexed by ``study.query_indices`` for a *dev* seed, and only
to score a pilot.

Conventions (verified numerically by ``verify_conventions`` and ``synthetic_check``)
-----------------------------------------------------------------------------------
``whiten.py`` releases ``X = (U - mean) @ A.T`` with ``U`` the unpermuted raw pixels.
sklearn's FastICA returns ``components_`` (``W``) and ``mixing_`` (``Mx``) with
``S = (X - ica_mean) @ W.T`` and ``X - ica_mean ~ S @ Mx.T``.  Therefore

    analysis filter  (which pixels a component reads)  f_i = (W @ A)[i]
    synthesis filter (what a component looks like)     g_i = (A_inv @ Mx)[:, i]

``A_inv @ Mx`` equals ``A_inv @ W[i]`` when ``W`` is orthogonal, which is the formula
in the task sketch; they differ when the released data is not exactly white
(``epsilon > 0``), so both are reported.  A third, algebra-free measure --
``pixel_correlation`` -- is the one the conclusions lean on.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import study      # noqa: E402
import topology   # noqa: E402
import whiten     # noqa: E402

GRID = topology.GRID
NFEAT = topology.NFEAT
DEV_SEED = 2026092301
ROW_COL = topology.ROW_COL.astype(np.int64)
ADJACENCY = topology.ADJACENCY


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def unlabeled_pool(seed: int, n: int, transform: dict) -> dict:
    """The arrays an attacker may see: train rows + query images (no labels used)."""
    arrays = whiten.job_arrays(int(seed), int(n), transform)
    pool = np.concatenate([arrays["train_x"], arrays["query_x"]], axis=0)
    return {"train_x": arrays["train_x"], "train_y": arrays["train_y"],
            "query_x": arrays["query_x"],
            "pool": np.ascontiguousarray(pool, dtype=np.float64)}


def verify_conventions(transform: dict, seed: int = DEV_SEED, n: int = 200) -> dict:
    """Check ``X = (U - mean) @ A.T`` and ``A A_inv = I`` on real arrays."""
    images = np.asarray(study.pool_images(), dtype=np.float64)
    index = study.train_indices(seed, n)
    arrays = whiten.job_arrays(seed, n, transform)
    expected = (images[index] - transform["mean"]) @ transform["A"].T
    residual = float(np.abs(arrays["train_x"].astype(np.float64) - expected).max())
    identity = float(np.abs(transform["A"] @ transform["A_inv"] - np.eye(NFEAT)).max())
    return {"max_abs_forward_residual": residual, "max_abs_A_Ainv_minus_I": identity}


def synthetic_check(seed: int = 0, rows: int = 20000, dim: int = 16) -> dict:
    """Sanity check of the filter algebra where the true sources ARE known.

    Independent Laplace sources are mixed by a known matrix, whitened+rotated the
    same way the protocol does it, and FastICA is asked to invert the whole chain.
    If the algebra above is right, the recovered synthesis filters must be the
    (signed, permuted) columns of the true mixing matrix.
    """
    rng = np.random.default_rng(seed)
    sources = rng.laplace(size=(rows, dim))
    mixing = rng.standard_normal((dim, dim))
    raw = sources @ mixing.T + 3.0
    transform = whiten.fit_transform(raw, epsilon=1e-3, rotation_seed=7,
                                     apply_permutation=False)
    released = whiten.apply(raw, transform).astype(np.float64)
    fit = ica_decompose(released, transform, fun="logcosh",
                        whiten_mode="unit-variance", random_state=0,
                        max_iter=2000, tol=1e-6, n_components=dim)
    truth = mixing / np.linalg.norm(mixing, axis=0, keepdims=True)
    got = fit["synthesis"] / np.maximum(
        np.linalg.norm(fit["synthesis"], axis=1, keepdims=True), 1e-12)
    overlap = np.abs(got @ truth)
    best = overlap.max(axis=1)
    return {"n_sources": dim, "mean_best_abs_cosine": float(best.mean()),
            "min_best_abs_cosine": float(best.min()),
            "distinct_matches": int(np.unique(overlap.argmax(axis=1)).size)}


# --------------------------------------------------------------------------- #
# ICA
# --------------------------------------------------------------------------- #
def ica_decompose(X, transform: dict, fun: str = "logcosh",
                  whiten_mode="unit-variance", random_state: int = 0,
                  max_iter: int = 2000, tol: float = 1e-5,
                  n_components: int = NFEAT) -> dict:
    """FastICA on released features; returns sources and pixel-space filters.

    ``whiten_mode=False`` takes the released data to be already white (the task
    sketch); ``'unit-variance'`` lets sklearn re-whiten first, which is what an
    attacker would really do because ``epsilon>0`` leaves the released covariance
    slightly anisotropic (29 of 81 covariance eigenvalues below 0.5 at eps=1e-3).
    """
    from sklearn.decomposition import FastICA
    X = np.ascontiguousarray(X, dtype=np.float64)
    started = time.time()
    kwargs = dict(fun=fun, max_iter=int(max_iter), tol=float(tol),
                  random_state=int(random_state), whiten=whiten_mode)
    if whiten_mode is not False:
        kwargs["n_components"] = int(n_components)
        kwargs["whiten_solver"] = "eigh"
    ica = FastICA(**kwargs)
    sources = np.ascontiguousarray(ica.fit_transform(X), dtype=np.float64)
    unmixing = np.asarray(ica.components_, dtype=np.float64)     # (k, 81)
    mixing = np.asarray(ica.mixing_, dtype=np.float64)           # (81, k)
    offset = np.asarray(getattr(ica, "mean_", np.zeros(X.shape[1])), dtype=np.float64)
    residual = float(np.abs(sources - (X - offset) @ unmixing.T).max())
    gram = unmixing @ unmixing.T
    return {
        "sources": sources, "unmixing": unmixing, "mixing": mixing,
        "analysis": unmixing @ transform["A"],                    # (k,81) pixel space
        "synthesis": (transform["A_inv"] @ mixing).T,             # (k,81) pixel space
        "n_iter": int(getattr(ica, "n_iter_", -1)),
        "converged": bool(getattr(ica, "n_iter_", max_iter) < max_iter),
        "seconds": time.time() - started,
        "max_abs_transform_residual": residual,
        "unmixing_offdiag_gram_max": float(np.abs(gram - np.diag(np.diag(gram))).max()),
        "fun": fun, "whiten_mode": str(whiten_mode), "random_state": int(random_state),
    }


# --------------------------------------------------------------------------- #
# localisation of a filter bank in pixel space
# --------------------------------------------------------------------------- #
def _window_mask():
    delta = np.abs(ROW_COL[:, None, :] - ROW_COL[None, :, :]).max(-1)
    return delta <= 1


WINDOW = _window_mask()


def pixel_correlation(sources, seed: int, n: int) -> np.ndarray:
    """(k,81) Pearson correlation between every component and every RAW pixel.

    EVALUATION ONLY.  The assumption-free version of "which pixel does this
    component track": no filter algebra, and automatically immune to the trap
    that a large filter coefficient on an always-black pixel contributes nothing.
    """
    sources = np.asarray(sources, dtype=np.float64)
    images = np.asarray(study.pool_images(), dtype=np.float64)
    index = np.concatenate([study.train_indices(seed, n), study.query_indices(seed)])
    if index.shape[0] != sources.shape[0]:
        raise ValueError("sources and (train+query) rows disagree")
    pixels = images[index]
    s = sources - sources.mean(0, keepdims=True)
    p = pixels - pixels.mean(0, keepdims=True)
    denominator = np.outer(np.linalg.norm(s, axis=0), np.linalg.norm(p, axis=0))
    return np.nan_to_num((s.T @ p) / np.maximum(denominator, 1e-300), nan=0.0)


def variance_weighted(filters) -> np.ndarray:
    """Analysis filters rescaled to standardised-pixel units (f_p * std(x_p))."""
    scale = np.asarray(study.pool_images(), dtype=np.float64).std(0)
    return np.asarray(filters, dtype=np.float64) * scale[None, :]


def localisation(filters) -> dict:
    """Per-component energy concentration of a (k,81) pixel-space filter bank."""
    f = np.asarray(filters, dtype=np.float64)
    energy = f ** 2
    energy = energy / np.maximum(energy.sum(1, keepdims=True), 1e-300)
    peak = energy.argmax(1)
    top1 = energy[np.arange(f.shape[0]), peak]
    win = np.array([energy[i, WINDOW[peak[i]]].sum() for i in range(f.shape[0])])
    order = -np.sort(-energy, axis=1)
    participation = 1.0 / np.maximum((energy ** 2).sum(1), 1e-300)
    return {
        "peak_pixel": peak.astype(np.int64),
        "top1_energy": top1, "window3x3_energy": win,
        "top3_energy": order[:, :3].sum(1), "participation_ratio": participation,
        "mean_top1_energy": float(top1.mean()),
        "median_top1_energy": float(np.median(top1)),
        "mean_window3x3_energy": float(win.mean()),
        "median_window3x3_energy": float(np.median(win)),
        "mean_participation_ratio": float(participation.mean()),
        "n_components_top1_above_0.5": int((top1 > 0.5).sum()),
        "n_components_window_above_0.8": int((win > 0.8).sum()),
        "distinct_peak_pixels": int(np.unique(peak).size),
        "uniform_top1_reference": 1.0 / NFEAT,
        "uniform_window_reference": float(WINDOW.sum(1).mean() / NFEAT),
    }


# --------------------------------------------------------------------------- #
# topographic affinity + lattice QAP
# --------------------------------------------------------------------------- #
def energy_affinity(sources, mode: str = "abs2") -> np.ndarray:
    """Correlation of component energies: the standard topographic-ICA statistic."""
    s = np.asarray(sources, dtype=np.float64)
    s = s / np.maximum(s.std(0, keepdims=True), 1e-12)
    if mode == "abs":
        values = np.abs(s)
    elif mode == "abs2":
        values = s ** 2
    elif mode == "sqrt":
        values = np.sqrt(np.abs(s))
    elif mode == "signed":
        values = s
    elif mode == "partial_abs":
        return np.clip(topology.partial_correlation(np.abs(s)), 0.0, None)
    elif mode == "partial_abs2":
        return np.clip(topology.partial_correlation(s ** 2), 0.0, None)
    elif mode == "partial_sqrt":
        return np.clip(topology.partial_correlation(np.sqrt(np.abs(s))), 0.0, None)
    else:
        raise ValueError(f"unknown affinity mode {mode!r}")
    corr = np.nan_to_num(np.corrcoef(values, rowvar=False), nan=0.0)
    np.fill_diagonal(corr, 0.0)
    return np.clip(corr, 0.0, None)


def layout_from_similarity(similarity, max_seconds: float = 90.0,
                           neighbours=(4, 5, 6, 8), random_starts: int = 8):
    """topology.py's start construction + QAP search on an arbitrary affinity.

    More permissive than ``topology.recover_layout``: an ICA energy affinity is far
    noisier than a pixel partial-correlation matrix, so the mutual-kNN graph is
    often disconnected (SMACOF then returns NaN).  Bad embeddings are dropped and
    extra random starts are added; the QAP objective still decides.
    """
    started = time.time()
    similarity = np.asarray(similarity, dtype=np.float64)
    starts, provenance = [], []
    for k in neighbours:
        for pruned in (True, False):
            graph = topology._mutual_knn(similarity, k)
            if pruned:
                graph = topology._prune_shortcuts(graph)
            hops = topology._hop_distances(graph)
            classical, stress = topology._classical_mds(hops)
            embeddings = [("mds", classical)]
            majorised = topology._smacof(hops, classical, 1.0 / np.maximum(hops, 1e-9))
            if np.isfinite(majorised).all():
                embeddings.append(("smacof", majorised))
            for name, embedding in embeddings:
                if not np.isfinite(embedding).all():
                    continue
                cells, cost = topology._align_to_lattice(embedding)
                starts.append(np.asarray(cells, dtype=np.int64))
                provenance.append({"neighbours": int(k), "embedding": name,
                                   "pruned": bool(pruned),
                                   "assignment_cost": float(cost),
                                   "embedding_stress": float(stress),
                                   "n_graph_edges": int(graph.sum() // 2),
                                   "graph_components":
                                       int(topology._component_count(graph))})
    rng = np.random.default_rng(topology.SEARCH_SEED)
    for _ in range(int(random_starts)):
        starts.append(rng.permutation(NFEAT).astype(np.int64))
        provenance.append({"neighbours": None, "embedding": "random"})
    layout, value, report = topology._search(similarity, starts, started, max_seconds)
    report = dict(report)
    report["winning_start"] = provenance[report["winning_start_index"]]
    report["qap_objective"] = float(value)
    report["qap_objective_fraction"] = float(value / max(similarity.sum(), 1e-12))
    report["seconds"] = float(time.time() - started)
    return np.asarray(layout, dtype=np.int64), report


# --------------------------------------------------------------------------- #
# ground-truth layout quality (EVALUATION ONLY)
# --------------------------------------------------------------------------- #
def layout_quality(layout, true_cell) -> dict:
    """Compare a recovered layout with the true pixel cell of every component."""
    layout = np.asarray(layout, dtype=np.int64)
    true_cell = np.asarray(true_cell, dtype=np.int64)
    placed = ADJACENCY[np.ix_(layout, layout)] > 0.5
    truth = ADJACENCY[np.ix_(true_cell, true_cell)] > 0.5
    upper = np.triu_indices(NFEAT, 1)
    placed_pairs, true_pairs = placed[upper], truth[upper]
    agree = float((placed_pairs & true_pairs).sum() / max(placed_pairs.sum(), 1))
    recall = float((placed_pairs & true_pairs).sum() / max(true_pairs.sum(), 1))
    best = None
    for index, relabel in enumerate(topology.DIHEDRAL):
        distance = np.abs(ROW_COL[relabel[layout]] - ROW_COL[true_cell]).sum(1)
        mean = float(distance.mean())
        if best is None or mean < best[0]:
            best = (mean, index, float(np.median(distance)),
                    float((distance == 0).mean()), float((distance <= 1).mean()))
    distances = np.abs(ROW_COL[:, None, :] - ROW_COL[None, :, :]).sum(-1)
    placed_distance = distances[np.ix_(true_cell, true_cell)][upper][placed_pairs]
    return {
        "adjacent_pairs_placed": int(placed_pairs.sum()),
        "adjacency_agreement": agree,
        "true_adjacent_pairs": int(true_pairs.sum()),
        "true_adjacency_recall": recall,
        "mean_true_distance_of_placed_pairs": float(placed_distance.mean()),
        "median_true_distance_of_placed_pairs": float(np.median(placed_distance)),
        "placed_pairs_within_2_cells": float((placed_distance <= 2).mean()),
        "mean_manhattan_after_dihedral": best[0],
        "median_manhattan_after_dihedral": best[2],
        "exact_cell_fraction": best[3],
        "within_1_cell_fraction": best[4],
        "best_dihedral_index": int(best[1]),
    }


def chance_layout_quality(true_cell, trials: int = 200, seed: int = 0) -> dict:
    """The same statistics for uniformly random layouts."""
    rng = np.random.default_rng(seed)
    agreement, manhattan = [], []
    for _ in range(trials):
        report = layout_quality(rng.permutation(NFEAT), true_cell)
        agreement.append(report["adjacency_agreement"])
        manhattan.append(report["mean_manhattan_after_dihedral"])
    return {"trials": int(trials),
            "adjacency_agreement_mean": float(np.mean(agreement)),
            "adjacency_agreement_p95": float(np.quantile(agreement, 0.95)),
            "mean_manhattan_mean": float(np.mean(manhattan)),
            "mean_manhattan_p05": float(np.quantile(manhattan, 0.05))}


def oracle_layout(true_cell) -> np.ndarray:
    """Best bijective layout consistent with the per-component true cells.

    EVALUATION ONLY.  ``true_cell`` may repeat (two components can peak on the same
    pixel), so it is not a permutation; a min-cost assignment on Manhattan distance
    to the desired cell gives the closest bijection.
    """
    true_cell = np.asarray(true_cell, dtype=np.int64)
    cost = np.abs(ROW_COL[true_cell][:, None, :] - ROW_COL[None, :, :]).sum(-1).astype(float)
    rows, cols = topology._assign(cost)
    layout = np.empty(NFEAT, dtype=np.int64)
    layout[rows] = cols
    return layout


# --------------------------------------------------------------------------- #
# downstream: the frozen cnn-09 recipe on CPU (mirrors learners._fit_topo_cnn)
# --------------------------------------------------------------------------- #
def train_cnn(train_images, train_y, query_images, epochs: int = 25,
              member_seed: int = 101, batch_size: int = 128) -> dict:
    """One cnn-09 member on (N,1,9,9) arrays; same arithmetic as _fit_topo_cnn."""
    import torch
    import torch.nn.functional as F
    import learners
    import spatial_learner

    started = time.time()
    config = dict(spatial_learner.CONFIG)
    torch.manual_seed(int(member_seed))
    np.random.seed(int(member_seed) % (2 ** 31))
    generator = torch.Generator(device=torch.device("cpu"))
    generator.manual_seed(int(member_seed))
    model = spatial_learner.build_model(config)
    raw = torch.as_tensor(np.ascontiguousarray(train_images, dtype=np.float32))
    truth = torch.as_tensor(np.ascontiguousarray(train_y), dtype=torch.long)
    query = torch.as_tensor(np.ascontiguousarray(query_images, dtype=np.float32))
    n = raw.shape[0]
    mean = raw.mean().item()
    std = max(raw.std(unbiased=False).item(), 1e-6)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]),
                                  weight_decay=float(config["weight_decay"]),
                                  betas=(.9, .999), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs,
                                                           eta_min=.00002)
    history = []
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(n, generator=generator)
        loss_sum = 0.0
        for positions in order.split(batch_size):
            optimizer.zero_grad(set_to_none=True)
            batch = (learners._mild_affine(raw[positions], generator) - mean) / std
            loss = F.cross_entropy(model(batch), truth[positions])
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(positions)
        scheduler.step()
        history.append({"epoch": epoch + 1, "augmented_training_loss": loss_sum / n})
    model.eval()
    logits = []
    with torch.inference_mode():
        for start in range(0, query.shape[0], 512):
            logits.append(model((query[start:start + 512] - mean) / std).numpy())
    logits = np.concatenate(logits).astype(np.float32)
    return {"logits": logits, "predictions": logits.argmax(1).astype(np.int64),
            "epochs": int(epochs), "member_seed": int(member_seed),
            "seconds": time.time() - started,
            "final_train_loss": history[-1]["augmented_training_loss"],
            "history": history[::max(1, epochs // 5)]}


def sources_to_images(sources, layout, mode: str = "signed") -> np.ndarray:
    """Standardise sources and lay them out on the 9x9 grid given by ``layout``."""
    s = np.asarray(sources, dtype=np.float64)
    if mode == "abs":
        s = np.abs(s)
    elif mode != "signed":
        raise ValueError(f"unknown source mode {mode!r}")
    s = (s - s.mean(0, keepdims=True)) / np.maximum(s.std(0, keepdims=True), 1e-12)
    return topology.unpermute(np.ascontiguousarray(s, dtype=np.float32), layout)


def dev_query_labels(seed: int) -> np.ndarray:
    """Dev-seed query labels for CPU pilots only (never a final seed)."""
    if int(seed) in set(study.FINAL_SEEDS):
        raise ValueError("refusing to read labels for a FINAL seed")
    return np.load(study.POOL_LABELS_PATH)[study.query_indices(int(seed))]


def error_rate(predictions, labels) -> float:
    return float((np.asarray(predictions) != np.asarray(labels)).mean())


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #
def _scalar_only(stats: dict) -> dict:
    return {k: v for k, v in stats.items() if not isinstance(v, np.ndarray)}


def _affinity_ratio(similarity, true_cell) -> dict:
    """Mean affinity on true-neighbour vs far pairs (a QAP feasibility check)."""
    cells = ROW_COL[np.asarray(true_cell, dtype=np.int64)]
    distance = np.abs(cells[:, None, :] - cells[None, :, :]).sum(-1)
    off = ~np.eye(NFEAT, dtype=bool)
    neighbour = (distance == 1) & off
    similarity = np.asarray(similarity, dtype=np.float64)
    top4 = np.argsort(-similarity, axis=1)[:, :4]
    return {"mean_neighbour": float(similarity[neighbour].mean()),
            "mean_far": float(similarity[distance >= 3].mean()),
            "mean_offdiagonal": float(similarity[off].mean()),
            "top4_true_neighbour_fraction":
                float(neighbour[np.arange(NFEAT)[:, None], top4].mean())}


def run_attack(seed: int, n: int, transform: dict, fun: str = "logcosh",
               whiten_mode="unit-variance", n_components: int = NFEAT,
               affinity_modes=("sqrt", "partial_sqrt", "abs2"), random_state: int = 0,
               max_iter: int = 1500, qap_seconds: float = 60.0, cache_dir=None) -> dict:
    """ICA -> topographic affinity -> lattice QAP -> ground-truth evaluation."""
    started = time.time()
    pool = unlabeled_pool(seed, n, transform)["pool"]
    fit = ica_decompose(pool, transform, fun=fun, whiten_mode=whiten_mode,
                        random_state=random_state, max_iter=max_iter,
                        n_components=n_components)
    sources = fit["sources"]
    correlation = pixel_correlation(sources, seed, n)
    true_cell = np.abs(correlation).argmax(1)
    report = {
        "seed": int(seed), "n": int(n), "unlabeled_rows": int(pool.shape[0]),
        "fun": fun, "whiten_mode": str(whiten_mode), "n_components": int(n_components),
        "random_state": int(random_state),
        "ica": {k: fit[k] for k in ("n_iter", "converged", "seconds",
                                    "max_abs_transform_residual",
                                    "unmixing_offdiag_gram_max")},
        "localisation": {
            "synthesis": _scalar_only(localisation(fit["synthesis"])),
            "analysis_variance_weighted":
                _scalar_only(localisation(variance_weighted(fit["analysis"]))),
            "pixel_correlation": _scalar_only(localisation(correlation)),
            "max_abs_pixel_correlation_mean": float(np.abs(correlation).max(1).mean()),
            "max_abs_pixel_correlation_median":
                float(np.median(np.abs(correlation).max(1))),
            "distinct_peak_pixels": int(np.unique(true_cell).size),
        },
        "layouts": {},
        "chance": chance_layout_quality(true_cell),
    }
    best = None
    for mode in affinity_modes:
        similarity = energy_affinity(sources, mode)
        layout, search = layout_from_similarity(similarity, max_seconds=qap_seconds)
        quality = layout_quality(layout, true_cell)
        report["layouts"][mode] = {
            "search": {k: v for k, v in search.items() if k != "winning_start"},
            "winning_start": search["winning_start"], "quality": quality,
            "layout": layout.tolist(),
            "neighbour_affinity_ratio": _affinity_ratio(similarity, true_cell)}
        if best is None or quality["adjacency_agreement"] > best[1]:
            best = (mode, quality["adjacency_agreement"])
    report["best_affinity_mode"] = best[0]
    report["oracle_layout_quality"] = layout_quality(oracle_layout(true_cell), true_cell)
    report["seconds"] = time.time() - started
    if cache_dir is not None:
        cache = Path(cache_dir)
        cache.mkdir(parents=True, exist_ok=True)
        stem = f"{fun}_{whiten_mode}_{n_components}_N{n}_s{seed}"
        np.save(cache / f"sources_{stem}.npy", sources.astype(np.float32))
        np.save(cache / f"correlation_{stem}.npy", correlation)
        np.save(cache / f"synthesis_{stem}.npy", fit["synthesis"])
        for mode, entry in report["layouts"].items():
            np.save(cache / f"layout_{stem}_{mode}.npy",
                    np.asarray(entry["layout"], dtype=np.int64))
        report["cache_stem"] = stem
    return report


if __name__ == "__main__":  # pragma: no cover - thin CLI
    transform = whiten.default_transform()
    print(json.dumps(verify_conventions(transform), indent=2))
    print(json.dumps(synthetic_check(), indent=2))
