"""Blind rigid-body docking of small probe proteins onto a target (LightDock, no restraints): where do probes stick? The target residues most often contacted by the best-scoring
poses form a 'consensus sticky epitope' - a data-driven alternative or complement to funnel/hotspots.py (surface features).

    python funnel/dock_epitope.py --pdb X.pdb --chain A [--range a-b] --out out/epitope/X [--probes ubiquitin,protein_g_b1,sh3] [--top 300]

Resumable (docking is skipped for probes whose ranking exists). CPU only. Output: epitope.csv (per residue: contact frequency per probe + mean), patches.json (ranked hotspot sets).
Probes are the public scaffolds in funnel/scaffolds.json; LightDock is called as a subprocess (GPL-3.0, never imported). See docs/HOTSPOTS.md for what this does and does not show."""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import dock_redesign as dr, hotspots as hs
REPO = Path(__file__).resolve().parent.parent

def write_receptor(pdb, chain, rng, dest):
    """Standard residues of one chain -> chain A PDB (heavy atoms, source numbering kept; LightDock needs unique residue ids)."""
    res = hs.read_chain(pdb, chain, rng); lines = []; k = 0
    for r in res:
        for at in r["res"]:
            if at.element == "H": continue
            k += 1; x, y, z = at.coord; el = (at.element or at.get_id()[0]).strip()
            lines.append(f"ATOM  {k:5d} {at.get_id():<4s} {r['res'].get_resname():>3s} A{r['num']:4d}{r['ins'] or ' '}   {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {el:>2s}")
    Path(dest).write_text("\n".join(lines) + "\nEND\n"); return res

def contacts(pose_pdb, rec_xyz, rec_res_of_atom, cutoff=5.0):
    lig = np.array([[float(l[30:38]), float(l[38:46]), float(l[46:54])] for l in open(pose_pdb) if l.startswith("ATOM")])
    d = np.linalg.norm(rec_xyz[:, None] - lig[None], axis=-1).min(1); return set(rec_res_of_atom[d < cutoff])

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pdb", required=True); ap.add_argument("--chain", default="A"); ap.add_argument("--range"); ap.add_argument("--out", required=True)
    ap.add_argument("--probes", default="ubiquitin,protein_g_b1,sh3"); ap.add_argument("--steps", type=int, default=60); ap.add_argument("--glowworms", type=int, default=30)
    ap.add_argument("--swarms", type=int, default=0, help="0 = LightDock decides from the surface"); ap.add_argument("--cores", type=int, default=min(30, os.cpu_count() or 4))
    ap.add_argument("--top", type=int, default=300, help="best-scoring poses per probe that vote"); ap.add_argument("--n-hot", type=int, default=5); ap.add_argument("--patches", type=int, default=5)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); rng = tuple(int(x) for x in a.range.split("-")) if a.range else None
    rec = out / "receptor.pdb"; res = write_receptor(a.pdb, a.chain, rng, rec)
    lib = json.load(open(Path(__file__).with_name("scaffolds.json"))); freq = {}
    rx, ro = [], []
    for i, r in enumerate(res):
        for at in r["atoms"]: rx.append(at); ro.append(i)
    rx, ro = np.array(rx), np.array(ro)
    for name in a.probes.split(","):
        w = out / "dock" / name; w.mkdir(parents=True, exist_ok=True); t0 = time.time()
        sc, _ = dr.prep_scaffold(name, lib[name], w)
        poses = dr.dock_one(rec, sc, w, a.steps, a.glowworms, a.cores, a.swarms)
        v = np.zeros(len(res)); n = 0
        for sw, g, cl, score, pdb in poses[:a.top]:
            if not Path(pdb).exists(): continue
            for i in contacts(pdb, rx, ro): v[i] += 1
            n += 1
        freq[name] = v / max(1, n); dr.log(f"{name}: {len(poses)} poses, {n} voting, {time.time()-t0:.0f}s")
    import pandas as pd
    df = pd.DataFrame(freq); df.insert(0, "residue", [f"{r['aa']}{r['num']}{r['ins']}" for r in res]); df["mean"] = df[list(freq)].mean(1); df.to_csv(out / "epitope.csv", index=False)
    s = df["mean"].values.copy(); rel, d = hs.sasa_per_residue(res), np.linalg.norm(np.array([r["cb"] for r in res])[:, None] - np.array([r["cb"] for r in res])[None], axis=-1)
    s[rel < 0.15] = -np.inf
    patches = hs.pick_patches(res, (s - np.nanmean(s[np.isfinite(s)])) / (np.nanstd(s[np.isfinite(s)]) + 1e-9), rel, d, a.n_hot, a.patches)
    json.dump(patches, open(out / "patches.json", "w"), indent=1)
    for p in patches: print(f"  #{p['rank']} hotspots {p['hotspots']}  mean contact freq {np.mean([df['mean'][[r['aa']+str(r['num'])+r['ins'] for r in res].index(h)] for h in p['hotspots']]):.3f}")

if __name__ == "__main__": main()
