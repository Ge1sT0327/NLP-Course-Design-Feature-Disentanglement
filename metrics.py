"""Disentanglement metrics and factor probes.

- MIG (Mutual Information Gap): measures the gap in mutual information between the
  most important and second most important latent dimensions; it is a widely used
  disentanglement metric proposed in the beta-TCVAE paper (Chen et al. 2018).
- Factor Probe: trains a linear regression / linear classifier to predict known
  factors (rotation, stroke thickness, digit class) from the latent code z. The
  higher the R^2 / accuracy, and the more concentrated the importance is on a few
  dimensions, the better the factor is disentangled.
"""
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression


def entropy(counts):
    """Entropy of a discrete distribution (in nats)."""
    counts = np.asarray(counts, dtype=np.float64)
    p = counts / counts.sum()
    p = p[p > 0]
    return -float((p * np.log(p)).sum())


def discrete_mi(z, factor, n_bins=20):
    """Empirical mutual information between a continuous z and a discrete factor (in nats).

    z is discretized into equal-width bins and the joint distribution is estimated
    with a 2D histogram.
    """
    z = np.asarray(z, dtype=np.float64)
    factor = np.asarray(factor, dtype=np.int64)
    lo, hi = z.min(), z.max()
    if hi - lo < 1e-9:
        return 0.0
    edges = np.linspace(lo, hi, n_bins + 1)
    z_bin = np.clip(np.digitize(z, edges[1:-1]), 0, n_bins - 1)
    n_factor = int(factor.max()) + 1
    joint = np.zeros((n_bins, n_factor), dtype=np.float64)
    np.add.at(joint, (z_bin, factor), 1)
    joint /= joint.sum()
    p_z = joint.sum(1)
    p_f = joint.sum(0)
    mi = 0.0
    for i in range(n_bins):
        for j in range(n_factor):
            v = joint[i, j]
            if v > 0:
                mi += v * np.log(v / (p_z[i] * p_f[j] + 1e-12))
    return float(mi)


def mig(z, factor, n_bins=20):
    """MIG = (max MI - second max MI) / H(factor).

    Returns (mig, per-dimension MI array, dimension indices sorted by MI descending).
    """
    D = z.shape[1]
    mis = np.array([discrete_mi(z[:, j], factor, n_bins) for j in range(D)])
    order = np.argsort(mis)[::-1]
    gap = mis[order[0]] - mis[order[1]]
    h = entropy(np.bincount(factor))
    return (gap / h if h > 0 else 0.0), mis, order


def _standardize(z):
    z = z - z.mean(0, keepdims=True)
    return z / (z.std(0, keepdims=True) + 1e-8)


def linear_probe_regression(z, y):
    """Linear regression probe. Returns (R^2, per-dimension importance |coef|)."""
    zs = _standardize(z)
    reg = LinearRegression().fit(zs, y)
    return float(reg.score(zs, y)), np.abs(reg.coef_)


def linear_probe_classification(z, y):
    """Linear classification probe. Returns (accuracy, per-dimension importance)."""
    zs = _standardize(z)
    clf = LogisticRegression(max_iter=2000).fit(zs, y)
    return float(clf.score(zs, y)), np.abs(clf.coef_).mean(0)
