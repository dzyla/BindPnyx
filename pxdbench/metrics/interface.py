"""Interface confidence metrics for binder designs.

Pure numpy: no Boltz, no PXDesign, no GPU, no file I/O. Two metrics share a
functional form and must never be confused with each other:

  i_pDAE  Ca-contact mask (<= 8 A), d0 partner-count floor 19, max over both
          chain directions. Port of BindCraft2 bindcraft/filters.py:117-168.
          RECORDED ONLY - not a selection key (see the spec, section 3).

  ipSAE   PAE mask (< 10 A), d0 floor 26, MIN over directions (ipsae_min).
          Delegated to metrics/_common_scorer.py, an independent implementation of
          the published ipSAE definition (checked against pinned golden values in
          tests/fixtures/ipsae_golden.json). Never reimplemented elsewhere: two ipSAE
          conventions have already caused a campaign-wide metric defect. An operator may
          still pin another implementation with PXD_COMMON_SCORER_DIR.

Both share ptm(x, d0) = 1 / (1 + (x/d0)^2) with the Yang-Skolnick d0.
"""
import hashlib
import importlib.util
import os

import numpy as np

#: An operator may still pin the original implementation by directory. Unlike
#: the previous default, an explicit override that cannot be loaded RAISES: a
#: silent fall-back to the bundled copy would make the recorded scorer identity
#: a lie.
COMMON_SCORER_ENV = "PXD_COMMON_SCORER_DIR"

IPDAE_D0_FLOOR = 19.0
IPDAE_CUTOFF = 8.0
IPSAE_PAE_CUTOFF = 10.0

#: The directional reduction this project uses. NOT the published convention,
#: which is `max`; ours is `min` because that is what produced the calibration.
#: A `bz_ipsae` is therefore never comparable to an externally reported ipSAE.
IPSAE_DIRECTIONAL_REDUCTION = "min"


def _load_scorer_from_dir(directory):
    """Import `common_scorer` from an explicit directory, by file path.

    Loaded by path rather than by name so that an unrelated `common_scorer`
    already in `sys.modules` cannot be returned instead.
    """
    path = os.path.join(directory, "common_scorer.py")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{COMMON_SCORER_ENV}={directory!r} does not contain "
            f"common_scorer.py. An explicit override is honoured exactly or "
            f"not at all: falling back to the bundled copy would record a "
            f"scorer identity that is not the one in use."
        )
    spec = importlib.util.spec_from_file_location(
        "pxdbench_external_common_scorer", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _common_scorer(override_dir=None):
    """The scorer in use: the bundled copy, or an explicitly pinned original.

    The bundled copy is the default, so a clean machine can score. An earlier
    revision imported the implementation from a directory outside the repository,
    and a machine without that directory could not score at all.
    """
    directory = (
        override_dir
        if override_dir is not None
        else (os.environ.get(COMMON_SCORER_ENV) or "").strip()
    )
    if directory:
        return _load_scorer_from_dir(directory)
    from pxdbench.metrics import _common_scorer as bundled

    return bundled


def scorer_provenance():
    """What is actually loaded, for the evaluation-context digest.

    A scorer hash change invalidates context identity even when golden values
    agree: agreement on a handful of cases is not identity.
    """
    scorer = _common_scorer()
    path = getattr(scorer, "__file__", None)
    digest = ""
    if path and os.path.isfile(path):
        with open(path, "rb") as handle:
            digest = hashlib.sha256(handle.read()).hexdigest()
    bundled = bool(path and path.endswith("_common_scorer.py"))
    return {
        "directional_reduction": IPSAE_DIRECTIONAL_REDUCTION,
        "pae_cutoff": IPSAE_PAE_CUTOFF,
        "ipdae_cutoff": IPDAE_CUTOFF,
        "ipdae_d0_floor": IPDAE_D0_FLOOR,
        "scorer_path": path or "",
        "scorer_sha256": digest,
        "bundled": bundled,
    }


def ptm_transform(pae, d0):
    """TM-score-like transform of a predicted alignment error."""
    return 1.0 / (1.0 + (np.asarray(pae, dtype=float) / d0) ** 2.0)


def d0_from_count(n, floor):
    """Yang-Skolnick d0 from a partner count, floored below at `floor`."""
    n = np.maximum(np.asarray(n, dtype=float), float(floor))
    return np.maximum(1.24 * (n - 15.0) ** (1.0 / 3.0) - 1.8, 1.0)


def anchored_scores(pae, contact, d0_floor=IPDAE_D0_FLOOR):
    """Per-anchor mean of the TM transform over that anchor's contact partners.

    `pae` and `contact` are (n_anchors, n_partners). Anchors with no partners
    score 0.0.
    """
    pae = np.asarray(pae, dtype=float)
    contact = np.asarray(contact, dtype=bool)
    counts = contact.sum(-1).astype(float)
    d0 = d0_from_count(counts, d0_floor)[:, None]
    scored = np.where(contact, ptm_transform(pae, d0), 0.0)
    return scored.sum(-1) / np.maximum(counts, 1.0)


def _split(is_binder):
    is_binder = np.asarray(is_binder, dtype=bool)
    return is_binder, ~is_binder


def _ca_distances(ca, b, t):
    ca = np.asarray(ca, dtype=float)
    return np.linalg.norm(ca[b][:, None, :] - ca[t][None, :, :], axis=2)


def i_pdae(pae, ca, is_binder, cutoff=IPDAE_CUTOFF, d0_floor=IPDAE_D0_FLOOR):
    """i_pDAE and its per-residue track. Returns (None, all-NaN) if no interface.

    The reverse direction uses the TRANSPOSE of the same binder-by-target
    submatrix, matching bindcraft/filters.py:126. Independently slicing
    pae[target, binder] is a different number for an asymmetric PAE.
    """
    pae = np.asarray(pae, dtype=float)
    b, t = _split(is_binder)
    n = b.size
    nan_track = np.full(n, np.nan)
    if b.sum() == 0 or t.sum() == 0:
        return None, nan_track

    contact = _ca_distances(ca, b, t) <= cutoff
    if not contact.any():
        return None, nan_track

    sub = pae[np.ix_(b, t)]
    fwd = anchored_scores(sub, contact, d0_floor)
    rev = anchored_scores(sub.T, contact.T, d0_floor)

    track = nan_track.copy()
    track[b] = np.where(contact.sum(-1) > 0, fwd, np.nan)
    track[t] = np.where(contact.sum(0) > 0, rev, np.nan)
    return float(max(fwd.max(), rev.max())), track


def ipsae(pae, is_binder, pae_cutoff=IPSAE_PAE_CUTOFF):
    """ipsae_min, the calibrated gate metric. Delegates to common_scorer."""
    cs = _common_scorer()
    pae = np.asarray(pae, dtype=float)
    b, t = _split(is_binder)
    if b.sum() == 0 or t.sum() == 0:
        return None
    b2t = cs.ipsae_directional(pae, b, t, pae_cutoff)
    t2b = cs.ipsae_directional(pae, t, b, pae_cutoff)
    return float(min(b2t, t2b))


def pae_interface_min(pae, is_binder):
    """Minimum PAE over binder-target pairs. LOWER IS BETTER."""
    pae = np.asarray(pae, dtype=float)
    b, t = _split(is_binder)
    if b.sum() == 0 or t.sum() == 0:
        return None
    both = np.concatenate([pae[np.ix_(b, t)].ravel(), pae[np.ix_(t, b)].ravel()])
    return float(both.min())


def interface_plddt(plddt_0_100, ca, is_binder, cutoff=IPDAE_CUTOFF):
    """Mean pLDDT over binder residues within `cutoff` A (Ca) of the target.

    THIS PROJECT'S convention, recorded and never gated: the published
    `>= 80` clause has no traceable implementation (see the spec, section 3).
    `plddt_0_100` must already be on the 0-100 scale; callers scale Boltz's
    0-1 output before calling.
    """
    plddt = np.asarray(plddt_0_100, dtype=float)
    b, t = _split(is_binder)
    if b.sum() == 0 or t.sum() == 0:
        return None
    mask = (_ca_distances(ca, b, t) <= cutoff).any(-1)
    if not mask.any():
        return None
    return float(plddt[b][mask].mean())
