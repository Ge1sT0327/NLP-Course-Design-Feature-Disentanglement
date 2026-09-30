"""解耦评估指标与因子探针。

- MIG (Mutual Information Gap)：衡量“最重要的维度”与“次重要维度”之间的
  互信息差距，是 β-TCVAE 论文 (Chen et al. 2018) 提出的常用解耦指标。
- 因子探针 (Factor Probe)：训练线性回归 / 线性分类器，从隐变量 z 预测已知
  因子（旋转角度、笔画粗细、数字类别）。R² / 准确率越高、且重要性集中在
  少数维度上，说明解耦越成功。
"""
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression


def entropy(counts):
    """离散分布的熵（nat）。"""
    counts = np.asarray(counts, dtype=np.float64)
    p = counts / counts.sum()
    p = p[p > 0]
    return -float((p * np.log(p)).sum())


def discrete_mi(z, factor, n_bins=20):
    """连续隐变量 z 与离散因子 factor 的经验互信息（nat）。

    对 z 做等宽分箱后，用 2D 直方图估计联合分布。
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
    """MIG = (最高 MI - 次高 MI) / H(factor)。

    返回 (mig, 各维度 MI 数组, 按 MI 降序的维度下标)。
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
    """线性回归探针。返回 (R², 各维度重要性 |coef|)。"""
    zs = _standardize(z)
    reg = LinearRegression().fit(zs, y)
    return float(reg.score(zs, y)), np.abs(reg.coef_)


def linear_probe_classification(z, y):
    """线性分类探针。返回 (准确率, 各维度重要性)。"""
    zs = _standardize(z)
    clf = LogisticRegression(max_iter=2000).fit(zs, y)
    return float(clf.score(zs, y)), np.abs(clf.coef_).mean(0)
