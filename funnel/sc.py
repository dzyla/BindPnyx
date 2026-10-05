"""Shape complementarity (Lawrence & Colman 1993) for predicted complexes: pure numpy/scipy, CPU, ~0.2-0.5 s per complex.

Algorithm (same steps as the original `sc` program / Rosetta's ShapeComplementarityFilter, with a contact-surface approximation of the molecular surface):
 1. For every atom near the interface, place `density` dots/A^2 on its van der Waals sphere (outward unit normal). A dot is on the probe-accessible
    surface of its own molecule if the probe centre (dot + probe * normal) is not inside any other atom of the molecule enlarged by the probe.
 2. A dot is BURIED if it is accessible in the isolated molecule but its probe centre is blocked by atoms of the partner.
 3. Remove the peripheral band: buried dots within `band` A of a dot that stays accessible in the complex.
 4. For each remaining dot of A take the nearest remaining dot of B: S = (nA . -nB) * exp(-w * d^2).  SC_A = median(S); likewise SC_B; SC = mean of the two.
Approximations: only the contact (atom-sphere) part of the molecular surface is used (no re-entrant surface); heavy atoms only with Bondi-like radii.
Range -1..1 (typical interfaces 0.6-0.8, designed binders 0.5-0.7). Scaled x100 it is comparable to Rosetta/Adaptyv-style values (mean ~54 for Nipah G designs)."""
import numpy as np
from scipy.spatial import cKDTree

# united-atom (Chothia-like) radii: structures carry no hydrogens, so heavy atoms are enlarged to cover them. Bondi radii (C 1.70, N 1.55, O 1.52) gave ~0.07 lower SC on reference complexes.
RADII = {"C": 1.90, "N": 1.65, "O": 1.40, "S": 1.85, "SE": 1.90, "P": 1.80}

def _sphere(n):
    i = np.arange(n) + 0.5; phi = np.arccos(1 - 2 * i / n); th = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)

_SPH = {}
def _unit_dots(r, density):
    n = max(12, int(round(density * 4 * np.pi * r * r)))
    if (n) not in _SPH: _SPH[n] = _sphere(n)
    return _SPH[n]

def _blocked(q, tree, radii, probe, skip_self=None):
    """True where probe centre q is inside any atom (radius + probe) of `tree`. skip_self: per-point atom index to ignore."""
    k = min(16, tree.n)
    d, idx = tree.query(q, k=k)
    d = d.reshape(len(q), -1); idx = idx.reshape(len(q), -1)
    inside = d < (radii[idx] + probe - 1e-6)
    if skip_self is not None: inside &= idx != skip_self[:, None]
    return inside.any(1)

def _surface(xyz, rad, other_xyz, other_rad, probe, density, reach):
    """Contact dots (pos, normal, buried, acc_in_complex) for the atoms of one molecule within `reach` A of the partner."""
    treeA, treeB = cKDTree(xyz), cKDTree(other_xyz)
    near = np.where(treeB.query(xyz, k=1)[0] < reach)[0]
    pos, nrm, own = [], [], []
    for i in near:
        u = _unit_dots(rad[i], density); pos.append(xyz[i] + rad[i] * u); nrm.append(u); own.append(np.full(len(u), i))
    pos = np.concatenate(pos); nrm = np.concatenate(nrm); own = np.concatenate(own)
    q = pos + probe * nrm
    acc_alone = ~_blocked(q, treeA, rad, probe, skip_self=own)
    blocked_by_partner = _blocked(q, treeB, other_rad, probe)
    buried = acc_alone & blocked_by_partner; acc_complex = acc_alone & ~blocked_by_partner
    return pos, nrm, buried, acc_complex

def shape_complementarity(xyzA, radA, xyzB, radB, density=15.0, probe=1.4, band=1.5, w=0.5, reach=None, details=False):
    """SC between two atom sets (arrays Nx3 and N radii). Returns dict(sc, sc_a, sc_b, area_a, area_b, n_a, n_b);
    details=True adds per-dot cosine (cos_a/cos_b) and nearest-dot distance (d_a/d_b) so w and the statistic can be varied offline."""
    reach = reach or (2 * probe + 2 * 1.9 + 1.0)
    pa, na, ba, ca = _surface(xyzA, radA, xyzB, radB, probe, density, reach)
    pb, nb, bb, cb = _surface(xyzB, radB, xyzA, radA, probe, density, reach)
    def trim(p, b, c):
        keep = b.copy()
        if c.any() and b.any() and band > 0:
            keep[b] = cKDTree(p[c]).query(p[b], k=1)[0] > band
        return keep
    ka, kb = trim(pa, ba, ca), trim(pb, bb, cb)
    if ka.sum() < 10 or kb.sum() < 10: return dict(sc=float("nan"), sc_a=float("nan"), sc_b=float("nan"), area_a=0.0, area_b=0.0, n_a=int(ka.sum()), n_b=int(kb.sum()))
    def one(p1, n1, p2, n2):
        d, j = cKDTree(p2).query(p1, k=1); c = (n1 * -n2[j]).sum(1); return c, d
    ca_, da_ = one(pa[ka], na[ka], pb[kb], nb[kb]); cb_, db_ = one(pb[kb], nb[kb], pa[ka], na[ka])
    sa = float(np.median(ca_ * np.exp(-w * da_ ** 2))); sb = float(np.median(cb_ * np.exp(-w * db_ ** 2)))
    out = dict(sc=0.5 * (sa + sb), sc_a=sa, sc_b=sb, area_a=ka.sum() / density, area_b=kb.sum() / density, n_a=int(ka.sum()), n_b=int(kb.sum()))
    if details: out.update(cos_a=ca_, d_a=da_, cos_b=cb_, d_b=db_)
    return out

def _atoms(chain):
    xyz, rad = [], []
    for res in chain:
        if res.id[0] != " ": continue                    # no waters/hetero
        for a in res:
            el = (a.element or a.get_id()[0]).strip().upper()
            if el == "H" or el == "D": continue
            xyz.append(a.coord); rad.append(RADII.get(el, 1.7))
    return np.array(xyz, float), np.array(rad, float)

def complex_sc(path, chains_a=("A",), chains_b=("B",), **kw):
    """SC of a complex file (.cif/.pdb): chains_a vs chains_b."""
    from Bio.PDB import MMCIFParser, PDBParser
    s = (MMCIFParser(QUIET=True) if str(path).endswith(".cif") else PDBParser(QUIET=True)).get_structure("x", str(path))[0]
    ga = [_atoms(s[c]) for c in chains_a]; gb = [_atoms(s[c]) for c in chains_b]
    xa = np.concatenate([g[0] for g in ga]); ra = np.concatenate([g[1] for g in ga]); xb = np.concatenate([g[0] for g in gb]); rb = np.concatenate([g[1] for g in gb])
    return shape_complementarity(xa, ra, xb, rb, **kw)
