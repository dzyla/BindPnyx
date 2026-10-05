"""Validate a docking-derived 'sticky' epitope map (funnel/dock_epitope.py) two ways, residue-level AUROC on exposed residues:
  (a) known functional site: target hotspots in funnel/targets/<t>.json  (pdl1, mdm2, fimh, fima)
  (b) where the 206 wet-lab-labelled designs' predicted binders land (bench/): residue touched by >=25% of that target's binders
Compared with the surface-feature score of funnel/hotspots.py.   python funnel/epitope_validate.py out/epitope [targets...]"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent)); import hotspots as hs
from hotspot_analysis import auroc
REPO = Path(__file__).resolve().parent.parent
root = Path(sys.argv[1]); targets = sys.argv[2:] or [p.name for p in sorted(root.iterdir()) if (p / "epitope.csv").exists()]
rows = []
for t in targets:
    df = pd.read_csv(root / t / "epitope.csv"); pdb = root / t / "receptor.pdb"; res = hs.read_chain(pdb, "A"); rel = hs.sasa_per_residue(res); ok = rel > 0.15
    names = [f"{r['aa']}{r['num']}{r['ins']}" for r in res]; assert names == df.residue.tolist()
    surf, _, _ = hs.score_residues(res); surf = np.where(np.isfinite(surf), surf, -9)
    labs = {}
    cfgf = REPO / f"funnel/targets/{t}.json"
    if cfgf.exists():
        known = {h for h in json.load(open(cfgf))["hotspots"]}; labs["known_site"] = np.array([n in known for n in names])
    ef = root / t / "binder_epitope.npy"
    if ef.exists(): labs["binder_epitope"] = np.load(ef) >= 0.25
    for lab, y in labs.items():
        if y[ok].sum() < 2: continue
        r = dict(target=t, label=lab, n_pos=int(y[ok].sum()), auroc_dock_mean=auroc(y[ok], df["mean"].values[ok]), auroc_surface=auroc(y[ok], surf[ok]),
                 auroc_combined=auroc(y[ok], (pd.Series(df["mean"].values[ok]).rank().values + pd.Series(surf[ok]).rank().values)))
        for c in df.columns[1:-1]: r[f"auroc_{c}"] = auroc(y[ok], df[c].values[ok])
        rows.append(r)
R = pd.DataFrame(rows); print(R.round(2).to_string(index=False)); R.to_csv(REPO / "funnel/results/epitope_validation.csv", index=False)
