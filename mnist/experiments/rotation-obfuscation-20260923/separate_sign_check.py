#!/usr/bin/env python
"""Is the separate-PCA collapse a sign-convention artefact?  Refit the PCA-60 whitener on the
test rows with (a) the study's max-entry sign rule, (b) a sign rule anchored on the all-ones
vector (u . 1 > 0), and measure basis mismatch + logistic-regression accuracy."""
import json
import numpy as np
import common, models

x, y = common.pool()
out = {}
for seed in (1, 2, 3):
    tr, te, pub = common.draw(seed)
    q = common.haar(60, seed * 7919 + 17)
    for rule in ('max-entry', 'ones-anchor'):
        maps = []
        for rows in (x[tr], x[te]):
            xr = np.asarray(rows, np.float64); mu = xr.mean(0)
            lam, u = np.linalg.eigh(np.cov(xr - mu, rowvar=False)); lam = np.clip(lam, 0, None)
            if rule == 'max-entry':
                sgn = np.sign(u[np.argmax(np.abs(u), axis=0), np.arange(81)])
            else:
                sgn = np.sign(u.sum(0))
            sgn[sgn == 0] = 1; u = u * sgn[None, :]
            idx = np.arange(21, 81); lam, u = lam[idx], u[:, idx]
            maps.append((q @ (u / np.sqrt(lam)[None, :]).T, mu))
        (a_tr, mu_tr), (a_te, mu_te) = maps
        z_tr = ((x[tr] - mu_tr) @ a_tr.T).astype(np.float32); z_te = ((x[te] - mu_te) @ a_te.T).astype(np.float32)
        mism = float(np.linalg.norm(a_te @ np.linalg.pinv(a_tr) - np.eye(60)) / np.sqrt(60))
        flips = int((np.sign(np.diag(a_te @ np.linalg.pinv(a_tr))) < 0).sum())
        p, _ = models.logreg_fit_predict(z_tr, y[tr], z_te)
        out[f'{rule}-s{seed}'] = {'basis_mismatch_rms': mism, 'sign_flips_of_60': flips, 'logreg_acc': float((p == y[te]).mean() * 100)}
        print(seed, rule, out[f'{rule}-s{seed}'], flush=True)
json.dump(out, open(common.HERE / 'results' / 'separate-sign-convention.json', 'w'), indent=1)
