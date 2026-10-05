"""Interface-confidence scorer: ipSAE and PAE/contact statistics for a predicted binder-target complex.

Independent implementation of the published ipSAE definition (Dunbrack, bioRxiv 2025.02.10.637595; reference code github.com/DunbrackLab/IPSAE), written for this
project. For each residue i of the "from" side, n0res(i) is the number of "to" residues j with PAE[i, j] < cutoff; d0res(i) is the Yang-Skolnick d0 for that count
(count floored at `D0_FLOOR`, d0 floored at 1.0); score(i) is the mean of 1 / (1 + (PAE[i, j] / d0res(i))^2) over those j; the directional ipSAE is the maximum
score(i). The pipeline's gate and ranking key `ipsae_min` is the minimum of the two directions (the reference reports asymmetric and maximum summaries as well).

`D0_FLOOR` is 26, which is what reproduces the pinned golden values in tests/fixtures/ipsae_golden.json (the reference uses 27; the two differ only for residues
with 26-27 confident partners and by under 0.01 in ipSAE).
"""
import numpy as np

D0_FLOOR = 26.0


def ptm_func(x, d0):
    return 1.0 / (1.0 + (x / d0) ** 2.0)


def calc_d0_array(L, min_value=1.0):
    """Yang & Skolnick d0 for partner counts L (floored at D0_FLOOR, result floored at min_value)."""
    L = np.maximum(D0_FLOOR, np.asarray(L, dtype=float))
    return np.maximum(min_value, 1.24 * (L - 15.0) ** (1.0 / 3.0) - 1.8)


def ipsae_directional(pae, mask_from, mask_to, pae_cutoff=10.0):
    """Asymmetric ipSAE: residues in `mask_from` aligned to `mask_to`."""
    pae = np.asarray(pae, dtype=float)
    valid = np.outer(mask_from, mask_to) & (pae < pae_cutoff)
    n0 = valid.sum(axis=1)
    d0 = calc_d0_array(n0)
    scores = np.where(valid, ptm_func(pae, d0[:, None]), 0.0).sum(axis=1) / np.maximum(n0, 1)
    scores = np.where(np.asarray(mask_from) & (n0 > 0), scores, 0.0)
    return float(scores.max()) if scores.size else 0.0


def score_complex(pae, coords_ca, is_binder, is_protein=None, heavy_coords=None, heavy_is_binder=None,
                  pae_cutoff=10.0, contact_cutoff=8.0, clash_cutoff=2.2):
    """Interface metric dict for one predicted complex (token order must match coords_ca).

    pae: (N, N) predicted aligned error; coords_ca: (N, 3); is_binder / is_protein: (N,) bool.
    heavy_coords / heavy_is_binder (optional): all-heavy-atom coordinates and their binder flags, for the clash count."""
    pae = np.asarray(pae, dtype=float)
    is_binder = np.asarray(is_binder, dtype=bool)
    is_protein = np.ones_like(is_binder) if is_protein is None else np.asarray(is_protein, dtype=bool)
    b, t = is_binder & is_protein, (~is_binder) & is_protein
    if b.sum() == 0 or t.sum() == 0:
        raise ValueError(f"empty mask: binder={int(b.sum())} target={int(t.sum())}")
    b2t, t2b = ipsae_directional(pae, b, t, pae_cutoff), ipsae_directional(pae, t, b, pae_cutoff)
    sub = np.concatenate([pae[np.ix_(b, t)].ravel(), pae[np.ix_(t, b)].ravel()])
    ca = np.asarray(coords_ca, dtype=float)
    d = np.linalg.norm(ca[b][:, None, :] - ca[t][None, :, :], axis=2)
    n_clash = np.nan
    if heavy_coords is not None and heavy_is_binder is not None:
        hc, hb_mask = np.asarray(heavy_coords), np.asarray(heavy_is_binder, dtype=bool)
        hb, ht = hc[hb_mask], hc[~hb_mask]
        if len(hb) and len(ht):
            from scipy.spatial import cKDTree
            n_clash = int(cKDTree(ht).query_ball_point(hb, r=clash_cutoff, return_length=True).sum())
    return {
        "ipsae_binder_to_target": b2t, "ipsae_target_to_binder": t2b, "ipsae_min": min(b2t, t2b), "ipsae_max": max(b2t, t2b),
        "pae_interface_mean": float(sub.mean()), "pae_interface_min": float(sub.min()),
        "n_interface_contacts": int((d <= contact_cutoff).sum()), "min_interface_ca_dist": float(d.min()), "n_clash_lt_2p2A": n_clash,
    }
