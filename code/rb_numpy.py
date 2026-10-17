"""NumPy reimplementation of the Regression Block.

Source: Coulombe, "Ordinary Least Squares as an Attention Mechanism",
arXiv:2504.09663v3, Appendix C (architecture) and Appendix D (protocol).

Reference implementation consulted (MIT license, adapted with attribution):
https://github.com/philgoucou/ols-attention
experiments/audit/audit_probes.py :: train_rb (the Table-1 pipeline).
Reimplemented here from scratch in NumPy (no torch); torch defaults for
initialization are reproduced.

Q-sparse variants replace the full O(P^2) pair set {i<=j} with a fixed
random subset of q = 4P (primary) or q = 2P (variant) off-diagonal pairs,
drawn once per (dataset, split) from RandomState(seed + 999). Everything
else (tokenizer, PCA->200, ridge warm start, optimizer, seeds, splits) is
identical to the baseline.
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

D = 64          # d_model
H = 128         # d_ff
CMAX = 200      # PCA components
LR = 1e-3
WD = 1e-4
BATCH = 256
EPOCHS = 200
PATIENCE = 10
VAL_FRAC = 0.15
DROPOUT = 0.1
LN_EPS = 1e-5   # torch LayerNorm default


def _u(rng, fan_in, shape):
    lim = 1.0 / np.sqrt(fan_in)
    return rng.uniform(-lim, lim, size=shape).astype(np.float32)


def init_tokenizer(rng, P):
    # nn.Linear(1, D): weight (D,1) U(-1,1); stored as (D,)
    We = rng.uniform(-1.0, 1.0, size=(D,)).astype(np.float32)
    be = rng.uniform(-1.0, 1.0, size=(D,)).astype(np.float32)
    pos = (rng.randn(P, D) * 0.02).astype(np.float32)
    return {"We": We, "be": be, "pos": pos}


def init_rest(rng, C):
    p = {
        "W": _u(rng, C, (C, D)), "bW": _u(rng, C, (D,)),
        "g1": np.ones(D, np.float32), "bn1": np.zeros(D, np.float32),
        "W1": _u(rng, D, (D, H)), "b1": _u(rng, D, (H,)),
        "W2": _u(rng, H, (H, D)), "b2": _u(rng, H, (D,)),
        "g2": np.ones(D, np.float32), "bn2": np.zeros(D, np.float32),
        "Wh": _u(rng, D, (D, 1)), "bh": _u(rng, D, (1,)),
    }
    return p


def tokenize(X, tok):
    # X: (B, P) -> (B, P, D)
    return X[..., None] * tok["We"] + tok["be"] + tok["pos"]


def polymap(e, pairs):
    """e: (B, P, D). Returns (per, pv).

    per: (B, P, 2D) = [e, e^2 - 1] (token-specific).
    pv:  (B, len(pairs)*D), the elementwise products e_i*e_j for each pair
    (i, j) — identical across tokens (matches the reference, where the pair
    vector is expanded to every token).
    Full width is 2D + len(pairs)*D.
    """
    B, P, Dd = e.shape
    per = np.concatenate([e, e * e - 1.0], axis=-1)
    cols = [e[:, i, :] * e[:, j, :] for (i, j) in pairs]
    pv = np.concatenate(cols, axis=-1)  # (B, len(pairs)*D)
    return per, pv


def polymap_backward(de_accum, e, pairs, dper, dpv):
    """Accumulate dL/de given dL/dper (B,P,2D) and dL/dpv (B,npair*D)."""
    B, P, Dd = e.shape
    de = dper[..., :Dd] + 2.0 * e * dper[..., Dd:]
    npair = len(pairs)
    gpair = dpv.reshape(B, npair, Dd)
    for k, (i, j) in enumerate(pairs):
        g = gpair[:, k, :]
        de[:, i, :] += g * e[:, j, :]
        de[:, j, :] += g * e[:, i, :]
    de_accum += de
    return de_accum


def layernorm_fwd(x, g, b):
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    xh = (x - mu) / np.sqrt(var + LN_EPS)
    return g * xh + b, (x, xh, mu, var, g)


def layernorm_bwd(dy, cache):
    x, xh, mu, var, g = cache
    Dd = x.shape[-1]
    std = np.sqrt(var + LN_EPS)
    dg = (dy * xh).sum(axis=(0, 1))
    db = dy.sum(axis=(0, 1))
    dxh = dy * g
    dvar = (dxh * (x - mu) * -0.5 / (var + LN_EPS) ** 1.5).sum(axis=-1, keepdims=True)
    dmu = ((dxh * -1.0 / std).sum(axis=-1, keepdims=True)
           + dvar * (-2.0 * (x - mu)).mean(axis=-1, keepdims=True))
    dx = dxh / std + dvar * 2.0 * (x - mu) / Dd + dmu / Dd
    return dx, dg, db


def pca_transform(per, pv, pca):
    """Frozen PCA. per: (B,P,2D), pv: (B,npair*D), pca comp: (C,F).

    g = (per-mean1)@comp1.T + ((pv-mean2)@comp2.T) broadcast over tokens.
    """
    D2 = 2 * D
    g_per = (per - pca["mean"][:D2]) @ pca["comp"][:, :D2].T      # (B,P,C)
    g_pv = (pv - pca["mean"][D2:]) @ pca["comp"][:, D2:].T        # (B,C)
    return g_per + g_pv[:, None, :]


def pca_backward(dg, pca):
    """dg: (B,P,C) -> (dper (B,P,2D), dpv (B,npair*D))."""
    D2 = 2 * D
    dper = dg @ pca["comp"][:, :D2]                               # (B,P,2D)
    dpv = dg.sum(axis=1) @ pca["comp"][:, D2:]                    # (B,npair*D)
    return dper, dpv


def forward(X, params, pca, pairs, rng_drop=None, train=True):
    """Returns (out, cache). out: (B,)."""
    tok = {k: params[k] for k in ("We", "be", "pos")}
    e = tokenize(X, tok)                                   # (B,P,D)
    per, pv = polymap(e, pairs)
    g = pca_transform(per, pv, pca)                        # (B,P,C)
    m = g @ params["W"] + params["bW"]                     # (B,P,D)
    s1 = e + m
    z1, c1 = layernorm_fwd(s1, params["g1"], params["bn1"])
    a = z1 @ params["W1"] + params["b1"]
    r = np.maximum(a, 0)
    if train and DROPOUT > 0:
        mask1 = (rng_drop.random(r.shape) >= DROPOUT).astype(np.float32) / (1.0 - DROPOUT)
        r = r * mask1
    else:
        mask1 = None
    h = r @ params["W2"] + params["b2"]
    if train and DROPOUT > 0:
        mask2 = (rng_drop.random(h.shape) >= DROPOUT).astype(np.float32) / (1.0 - DROPOUT)
        h = h * mask2
    else:
        mask2 = None
    s2 = z1 + h
    z2, c2 = layernorm_fwd(s2, params["g2"], params["bn2"])
    pooled = z2.mean(axis=1)                               # (B,D)
    out = (pooled @ params["Wh"] + params["bh"]).ravel()   # (B,)
    cache = dict(e=e, per=per, pv=pv, g=g, m=m, s1=s1, z1=z1, c1=c1, a=a,
                 r=r, mask1=mask1, h=h, mask2=mask2, s2=s2, z2=z2, c2=c2,
                 pooled=pooled, X=X)
    return out, cache


def backward(dout, params, pca, pairs, cache):
    """dout: dL/dout, shape (B,). Returns grads dict with same keys as params."""
    B = dout.shape[0]
    P = cache["e"].shape[1]
    gr = {}
    # head
    dpooled = (dout[:, None] @ params["Wh"].T)              # (B,D)
    gr["Wh"] = cache["pooled"].T @ dout[:, None]           # (D,1)
    gr["bh"] = dout.sum(axis=0, keepdims=True)              # (1,)
    dz2 = np.broadcast_to(dpooled[:, None, :],
                          (B, P, D)) / P
    ds2, dg2, dbn2 = layernorm_bwd(dz2, cache["c2"])
    gr["g2"] = dg2
    gr["bn2"] = dbn2
    dz1 = ds2.copy()
    dh = ds2
    if cache["mask2"] is not None:
        dh = dh * cache["mask2"]
    gr["W2"] = np.einsum("bph,bpd->hd", cache["r"], dh)
    gr["b2"] = dh.sum(axis=(0, 1))
    dr = np.einsum("bpd,hd->bph", dh, params["W2"])
    if cache["mask1"] is not None:
        dr = dr * cache["mask1"]
    da = dr * (cache["a"] > 0)
    dz1 = dz1 + np.einsum("bph,dh->bpd", da, params["W1"])
    gr["W1"] = np.einsum("bpd,bph->dh", cache["z1"], da)
    gr["b1"] = da.sum(axis=(0, 1))
    ds1, dg1, dbn1 = layernorm_bwd(dz1, cache["c1"])
    gr["g1"] = dg1
    gr["bn1"] = dbn1
    # mixer: s1 = e + m
    de = ds1.copy()                                        # dL/de via residual
    dm = ds1
    dg = np.einsum("bpd,cd->bpc", dm, params["W"])          # (B,P,C)
    gr["W"] = np.einsum("bpc,bpd->cd", cache["g"], dm)
    gr["bW"] = dm.sum(axis=(0, 1))
    dper, dpv = pca_backward(dg, pca)
    de = polymap_backward(de, cache["e"], pairs, dper, dpv)
    # tokenizer: e = X*We + be + pos
    X = cache["X"]
    gr["We"] = (de * X[..., None]).sum(axis=(0, 1))
    gr["be"] = de.sum(axis=(0, 1))
    gr["pos"] = de.sum(axis=0)
    return gr


def clip_grads(grads, max_norm=1.0):
    total = np.sqrt(sum(float((g ** 2).sum()) for g in grads.values()))
    if total > max_norm:
        s = max_norm / total
        for k in grads:
            grads[k] = grads[k] * s
    return grads


class Adam:
    def __init__(self, keys, b1=0.9, b2=0.999, eps=1e-8):
        self.m = {k: None for k in keys}
        self.v = {k: None for k in keys}
        self.b1, self.b2, self.eps = b1, b2, eps
        self.t = 0

    def step(self, params, grads, lr, wd):
        self.t += 1
        for k in params:
            g = grads[k].astype(np.float64) + wd * params[k].astype(np.float64)
            if self.m[k] is None:
                self.m[k] = np.zeros_like(g)
                self.v[k] = np.zeros_like(g)
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            params[k] = (params[k].astype(np.float64)
                         - lr * mh / (np.sqrt(vh) + self.eps)).astype(np.float32)


def mse(y_pred, y):
    return float(np.mean((y_pred - y) ** 2))


def r2_score(y_true, y_pred):
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def train_rb(Xtr, ytr, Xte, seed, mode="full"):
    """Train one Regression Block (baseline or Q-sparse). Returns dict of results."""
    import time
    t0 = time.time()
    P = Xtr.shape[1]
    N = Xtr.shape[0]

    # ---- pair set ----
    if mode == "full":
        pairs = [(i, j) for i in range(P) for j in range(i, P)]
        q = None
    else:
        q = 4 * P if mode == "q4p" else 2 * P
        prng = np.random.RandomState(seed + 999)
        allpairs = [(i, j) for i in range(P) for j in range(i + 1, P)]
        q = min(q, len(allpairs))
        sel = prng.choice(len(allpairs), q, replace=False)
        pairs = [allpairs[i] for i in sorted(sel)]
    F = 2 * D + len(pairs) * D

    # ---- PCA on tmp tokenizer (seeded identically to model init) ----
    rng_init = np.random.RandomState(seed)
    tok_tmp = init_tokenizer(rng_init, P)
    Xp = Xtr if N <= 5000 else Xtr[np.random.RandomState(seed).choice(N, 5000, replace=False)]
    e_tmp = tokenize(Xp.astype(np.float32), tok_tmp)
    per_tmp, pv_tmp = polymap(e_tmp, pairs)
    Pt = per_tmp.shape[1]
    phi = np.concatenate([per_tmp.reshape(-1, 2 * D),
                          np.repeat(pv_tmp, Pt, axis=0)], axis=-1)
    n_comp = min(CMAX, F, phi.shape[0])
    pca_skl = PCA(n_components=n_comp, random_state=seed)
    pca_skl.fit(phi)
    pca = {"mean": pca_skl.mean_.astype(np.float32),
           "comp": pca_skl.components_.astype(np.float32)}

    # ---- fresh identical init ----
    rng_init2 = np.random.RandomState(seed)
    params = init_tokenizer(rng_init2, P)
    params.update(init_rest(rng_init2, n_comp))

    # ---- ridge warm start on pooled PCA features ----
    e0 = tokenize(Xtr.astype(np.float32), {k: params[k] for k in ("We", "be", "pos")})
    per0, pv0 = polymap(e0, pairs)
    g0 = pca_transform(per0, pv0, pca)
    pooled0 = g0.mean(axis=1)
    reg = Ridge(alpha=1.0)
    reg.fit(pooled0, ytr)
    params["W"][:, 0] = reg.coef_.astype(np.float32)
    params["bW"][:] = 0.0
    params["Wh"][:] = 0.0
    params["Wh"][0, 0] = 1.0
    params["bh"][:] = float(reg.intercept_)

    n_params = sum(v.size for v in params.values())

    # ---- train/val split ----
    rng_data = np.random.RandomState(seed)
    rng_drop = np.random.RandomState(seed + 2)
    n_val = max(1, int(N * VAL_FRAC))
    idx = rng_data.permutation(N)
    iv, it = idx[:n_val], idx[n_val:]
    Xv, yv = Xtr[it], ytr[it]
    Xvv, yvv = Xtr[iv], ytr[iv]
    Xv = Xv.astype(np.float32)
    Xvv = Xvv.astype(np.float32)
    Xte = Xte.astype(np.float32)

    opt = Adam(list(params.keys()))
    best = float("inf")
    best_params = None
    pat = 0
    epochs_run = 0
    for ep in range(1, EPOCHS + 1):
        perm = rng_data.permutation(len(Xv))
        for s in range(0, len(Xv), BATCH):
            bi = perm[s: s + BATCH]
            out, cache = forward(Xv[bi], params, pca, pairs, rng_drop, train=True)
            d = (2.0 / len(bi)) * (out - yv[bi])
            grads = backward(d, params, pca, pairs, cache)
            grads = clip_grads(grads, 1.0)
            opt.step(params, grads, LR, WD)
        out_v, _ = forward(Xvv, params, pca, pairs, train=False)
        vl = mse(out_v, yvv)
        epochs_run = ep
        if vl < best:
            best = vl
            best_params = {k: v.copy() for k, v in params.items()}
            pat = 0
        else:
            pat += 1
            if pat >= PATIENCE:
                break
    if best_params is not None:
        params = best_params

    # ---- predict ----
    preds = []
    for s in range(0, len(Xte), 4096):
        out, _ = forward(Xte[s: s + 4096], params, pca, pairs, train=False)
        preds.append(out)
    y_pred = np.concatenate(preds)

    return {
        "epochs": epochs_run,
        "wall_s": time.time() - t0,
        "n_params": n_params,
        "feat_width": F,
        "n_pairs": len(pairs),
        "n_comp": n_comp,
        "pairs": pairs,
        "y_pred": y_pred,
    }
