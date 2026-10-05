"""Interface metrics shared by every scorer arm, so arms differ only in the model."""
import numpy as np

def _d0(n):
    n = np.maximum(n, 27)
    return 1.24 * np.cbrt(n - 15) - 1.8

def ipsae_directional(pae, a, b, cutoff=10.0):
    """Dunbrack ipSAE a->b: per-residue-of-a mean of ptm(PAE) over b residues
    with PAE < cutoff, d0 from that residue's own count; best residue wins."""
    sub = pae[np.ix_(a, b)]
    valid = sub < cutoff
    n0 = valid.sum(1)
    d0 = _d0(n0)[:, None]
    ptm = 1.0 / (1.0 + (sub / d0) ** 2)
    per = np.where(n0 > 0, (ptm * valid).sum(1) / np.maximum(n0, 1), 0.0)
    return float(per.max())

def ipsae(pae, n_target, n_binder, cutoff=10.0):
    """Chain order is target then binder. Returns (min, max) over both directions."""
    pae = np.asarray(pae, float)
    t = np.arange(n_target); b = np.arange(n_target, n_target + n_binder)
    x, y = ipsae_directional(pae, b, t, cutoff), ipsae_directional(pae, t, b, cutoff)
    return min(x, y), max(x, y)

def auroc(labels, scores):
    from scipy.stats import rankdata
    labels = np.asarray(labels); s = np.asarray(scores, float)
    ok = ~np.isnan(s); labels, s = labels[ok], s[ok]
    p, n = (labels == 1).sum(), (labels == 0).sum()
    if p == 0 or n == 0: return float("nan")
    r = rankdata(s)
    return float((r[labels == 1].sum() - p * (p + 1) / 2) / (p * n))
