"""fastPISA interface metrics (https://github.com/dzyla/fastPISA) for predicted complexes: CPU only, ~0.4 s each.
Installed outside the project envs: .pxd/ext/fastPISA (code) + .pxd/ext/py (freesasa, gemmi); this module adds them to sys.path."""
import sys
from pathlib import Path
_REPO = Path(__file__).resolve().parent.parent
for p in (_REPO / ".pxd" / "ext" / "fastPISA", _REPO / ".pxd" / "ext" / "py"):
    if p.exists() and str(p) not in sys.path: sys.path.insert(0, str(p))

import logging
logging.getLogger("fastpisa").setLevel(logging.WARNING)      # fastPISA logs one INFO line per structure

KEYS = ["interface_area", "solvation_energy", "solvation_energy_apolar", "solvation_energy_polar", "stabilization_energy", "p_value", "css",
        "n_interface_residues", "n_hydrogen_bonds", "n_salt_bridges"]
APOLAR = set("ALA VAL LEU ILE MET PHE TRP TYR PRO".split()); AROMATIC = set("PHE TRP TYR".split())

def pisa_metrics(cif, a="A", b="B", hotspot_idx=None):
    """Interface metrics between chain a (target) and chain b (binder) of a predicted complex.
    hotspot_idx: 1-based residue indices on chain a. Adds `hotspot_bsa` (mean buried area on them, A^2; 0 = not buried),
    `hotspot_buried_frac` (fraction with BSA >= 10 A^2), and binder-interface composition (`bsa_apolar_frac`, `n_aromatic_iface`)."""
    import fastpisa
    out = {k: float("nan") for k in KEYS}
    try:
        res = fastpisa.analyze(str(cif)); itf = res.interface_between(a, b)
        df = res.to_dataframe(); m = df.interface_id == itf.interface_id     # the pair's OWN row: with >2 chains (or glycans) row 0 is another interface
        if not m.any():                                      # never fall back to another pair's numbers
            raise ValueError(f"interface {itf.interface_id} ({a}/{b}) absent from the {len(df)}-row summary")
        row = df[m].iloc[0]
        for k in KEYS: out[k] = float(row.get(k, float("nan")))
        s1, s2 = itf.residues(side=1), itf.residues(side=2)
        if hotspot_idx:
            bsa = {int(r["seq"]): float(r["bsa"]) for r in s1}
            vals = [bsa.get(i, 0.0) for i in hotspot_idx]
            out["hotspot_bsa"] = sum(vals) / len(vals); out["hotspot_buried_frac"] = sum(v >= 10 for v in vals) / len(vals)
        tot = sum(float(r["bsa"]) for r in s2) or float("nan")
        out["bsa_apolar_frac"] = sum(float(r["bsa"]) for r in s2 if r["name"] in APOLAR) / tot
        out["n_aromatic_iface"] = sum(1 for r in s2 if r["name"] in AROMATIC)
        out["iface_area_per_binder_res"] = out["interface_area"] / max(1, len(s2) if False else out["n_interface_residues"])
    except Exception as e:                                   # one malformed prediction must not stop a batch
        out["error"] = str(e)[:120]
    return out

def batch(items, workers=4):
    """items: list of (key, cif, hotspot_idx or None) -> list of dicts (key + metrics). CPU multiprocessing."""
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(_one, items, chunksize=4))
    return res

def sc_metrics(cif):
    """Shape complementarity (funnel/sc.py): sc (both sides), sc_binder, sc_target, sc_area (A^2 of trimmed interface), all x1 (0-1 scale)."""
    try:
        import sc
        r = sc.complex_sc(str(cif), ("A",), ("B",)); return dict(sc=r["sc"], sc_target=r["sc_a"], sc_binder=r["sc_b"], sc_area=r["area_a"] + r["area_b"])
    except Exception as e:
        return dict(sc=float("nan"), sc_target=float("nan"), sc_binder=float("nan"), sc_area=float("nan"), sc_error=str(e)[:100])

def _one(it):
    key, cif, hs = it; d = pisa_metrics(cif, hotspot_idx=hs); d.update(sc_metrics(cif)); d["id"] = key; return d
