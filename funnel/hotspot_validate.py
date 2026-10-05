"""Does the surface screen recover KNOWN functional sites? For each funnel target: run hotspots.propose on the raw PDB (no site given) and compare the proposed patches with the
site we defined from the literature / ligand (funnel/targets/*.json 'hotspots'). Reports rank of the first patch that overlaps >=2 known hotspots (or within 6 A of them).
  python funnel/hotspot_validate.py [--weights default|fitted]"""
import json, sys, argparse
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent)); import hotspots as hs
REPO = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser(); ap.add_argument("--weights", default="default"); a = ap.parse_args()
W = hs.DEFAULT_W if a.weights == "default" else hs.load_weights()
for t in ["pdl1", "mdm2", "fimh", "fima"]:
    cfg = json.load(open(REPO / f"funnel/targets/{t}.json")); src = cfg["source"]; pdb = next((REPO / f"data/targets/{t}/raw").glob("*.pdb"))
    rng = tuple(int(x) for x in src["range"].split("-")) if src.get("range") else None
    patches, res, s = hs.propose(pdb, src["chain"], rng, None, n_hot=5, top=8, weights=W)
    known = {h for h in cfg["hotspots"]}; kn = {int("".join(c for c in h if c.isdigit())) for h in known}
    pos = {r["num"]: r["ca"] for r in res}; kc = np.array([pos[n] for n in kn if n in pos])
    row = []
    for p in patches:
        nums = [int("".join(c for c in h if c.isdigit())) for h in p["hotspots"]]
        ov = len(set(nums) & kn); dmin = np.mean([np.linalg.norm(kc - pos[n], axis=1).min() for n in nums])
        row.append((p["rank"], ov, round(dmin, 1)))
    hit = next((r for r, ov, dm in row if ov >= 2 or dm <= 6.0), None)
    print(f"{t:5s} known {sorted(known)}  first matching patch: rank {hit}  (rank, overlap, mean dist to known site A) {row[:5]}  top: {patches[0]['hotspots']}")
