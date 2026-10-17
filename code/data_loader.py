"""Dataset loading for the reproduction package (public data only).

Yacht (OpenML id 42370) and Energy (OpenML id 41478); the same
fetch logic as the full study. Falls back to a local cache so the
verification grid can re-run offline once fetched.
"""
import os

import numpy as np
import pandas as pd  # noqa: F401 - keeps parity with the study loader
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 42
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "datasets_cache.npz")

REGISTRY = {
    "Yacht": (42370, (300, 320), 6),
    "Energy": (41478, (700, 800), 8),
}


def _fetch(name):
    data_id, (n_min, n_max), expected_p = REGISTRY[name]
    data = fetch_openml(data_id=data_id, as_frame=True, parser="auto")
    X = data.data.select_dtypes(include=[np.number]).values
    y_raw = data.target
    if hasattr(y_raw, "columns"):
        y_raw = y_raw.iloc[:, 0]
    y = np.array(y_raw).astype(float)
    mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X, y = X[mask], y[mask]
    assert n_min <= X.shape[0] <= n_max, f"{name}: unexpected N"
    assert X.shape[1] == expected_p, f"{name}: unexpected P"
    return X, y


def load_yacht_energy():
    """Return {name: (X, y)}; uses a local cache after the first fetch."""
    if os.path.exists(CACHE):
        z = np.load(CACHE)
        return {k: (z[f"{k}_X"], z[f"{k}_y"]) for k in REGISTRY}
    out = {name: _fetch(name) for name in REGISTRY}
    np.savez(CACHE,
             **{f"{k}_X": v[0] for k, v in out.items()},
             **{f"{k}_y": v[1] for k, v in out.items()})
    return out


def prep_split(X_full, y_full, seed):
    """80/20 split; features and targets standardized on the train split."""
    Xtr, Xte, ytr, yte = train_test_split(
        X_full, y_full, test_size=0.2, random_state=seed)
    sc = StandardScaler()
    Xtr = sc.fit_transform(Xtr).astype(np.float32)
    Xte = sc.transform(Xte).astype(np.float32)
    scy = StandardScaler()
    ytr = scy.fit_transform(ytr.reshape(-1, 1)).ravel().astype(np.float32)
    yte = scy.transform(yte.reshape(-1, 1)).ravel().astype(np.float32)
    return Xtr, ytr, Xte, yte
