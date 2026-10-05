"""Prepare benchmark receptors: chain A of one Boltz complex per labelled target + the fraction of that target's wet-lab BINDERS whose predicted binder touches each residue."""
import glob, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent)); from hotspot_analysis import epitope
REPO = Path(__file__).resolve().parent.parent; M = pd.read_csv(REPO / "bench/inputs/manifest.csv")
for tg, g in M.groupby("target"):
    name = "mdm2_b" if tg == "mdm2" else tg; H = []; src = None
    for r in g[g.label == 1].itertuples():
        f = glob.glob(str(REPO / f"bench/out/boltz/{tg}/pred/**/{r.id}_model_0.cif"), recursive=True)
        if f: src = src or f[0]; H.append(epitope(f[0], 0))
    d = REPO / "out/epitope_bench" / name; d.mkdir(parents=True, exist_ok=True)
    np.save(d / "binder_epitope.npy", np.array(H).mean(0)); (d / "source.txt").write_text(src); print(name, len(H), "binders", len(H[0]), "residues")
