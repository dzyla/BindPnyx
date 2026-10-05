"""Where do wet-lab-labelled binders land on their targets, and does a surface-feature score find those places?
Uses the Boltz-2 complexes of the 206 ProteinBase designs (bench/out/boltz/<target>/pred/.../<id>_model_0.cif; chain A = target, B = binder).
  1. epitope map per target: for each target residue, the fraction of designs whose predicted binder touches it (heavy atoms < 5 A), split by wet-lab label
  2. does the epitope of binders differ from that of non-binders?
  3. fit residue-score weights on 3 targets, test on the 4th (leave-one-target-out); report AUROC for 'residue is in the binders' epitope'
  python funnel/hotspot_analysis.py"""
import glob, json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent)); import hotspots as hs
from sklearn.linear_model import LogisticRegression
REPO = Path(__file__).resolve().parent.parent; M = pd.read_csv(REPO / "bench/inputs/manifest.csv")

def auroc(y, s):
    from scipy.stats import rankdata
    y = np.asarray(y, int); s = np.asarray(s, float); ok = np.isfinite(s); y, s = y[ok], s[ok]
    p, n = y.sum(), (1 - y).sum()
    return float((rankdata(s)[y == 1].sum() - p * (p + 1) / 2) / (p * n)) if p and n else float("nan")

def epitope(cif, n_target):
    from Bio.PDB import MMCIFParser, NeighborSearch
    m = MMCIFParser(QUIET=True).get_structure("x", str(cif))[0]; A = [r for r in m["A"] if r.id[0] == " "]; B = [a for r in m["B"] for a in r]
    ns = NeighborSearch(list(B)); hit = np.zeros(len(A), bool)
    for i, r in enumerate(A):
        hit[i] = any(ns.search(a.coord, 5.0) for a in r)
    return hit

def main():
    out = {}; tables = {}
    for tg, g in M.groupby("target"):
        seq = g.target_seq.iloc[0]; rows = []
        for r in g.itertuples():
            f = glob.glob(str(REPO / f"bench/out/boltz/{tg}/pred/**/{r.id}_model_0.cif"), recursive=True)
            if not f: continue
            h = epitope(f[0], len(seq))
            if len(h) == len(seq): rows.append((r.label, h))
        if not rows: continue
        H = np.array([h for _, h in rows]); y = np.array([l for l, _ in rows])
        tables[tg] = dict(seq=seq, H=H, y=y, cif=f[0])
    print("designs with structures:", {t: (len(v['y']), int(v['y'].sum())) for t, v in tables.items()})
    feats = {}
    for tg, v in tables.items():
        res = hs.read_chain(v["cif"], "A"); assert "".join(r["aa"] for r in res) == v["seq"][:len(res)], tg
        msa = REPO / f"bench/msa/out/boltz_results_{tg}/msa/{tg}_0.csv"
        cons = hs.conservation(msa, v["seq"]) if msa.exists() else None
        Z, rel, d = hs.residue_features(res, cons); feats[tg] = (res, Z, rel, d, cons is not None)
    # 1-2. epitope maps
    print("\n== epitope concentration (fraction of designs touching the most-contacted 5-residue patch) and binder vs non-binder overlap ==")
    for tg, v in tables.items():
        H, y = v["H"], v["y"]; fb, fn = H[y == 1].mean(0), H[y == 0].mean(0)
        top = np.argsort(-H.mean(0))[:5]
        cb = np.corrcoef(fb, fn)[0, 1]
        print(f"{tg:6s} n={len(y)} (binders {int(y.sum())})  most-contacted residues {[feats[tg][0][i]['aa']+str(feats[tg][0][i]['num']) for i in top]}  "
              f"binder/non-binder epitope-map correlation {cb:.2f}  mean epitope size binders {H[y==1].sum(1).mean():.1f} vs non {H[y==0].sum(1).mean():.1f}")
        # does hitting the dominant patch predict being a binder?
        dom = H[:, top].sum(1) >= 3; 
        if dom.sum() and (~dom).sum(): print(f"       hits >=3 of the dominant 5: {dom.sum()} designs, binder rate {y[dom].mean():.0%} vs {y[~dom].mean():.0%} for the others")
    # 3. residue-level score: label = residue touched by >=25% of the BINDERS of that target
    print("\n== residue score vs 'in the binders' epitope' (label: touched by >= 25% of that target's binders) ==")
    LOTO = []
    for held in tables:
        Xtr, ytr = [], []
        for tg in tables:
            if tg == held: continue
            res, Z, rel, d, _ = feats[tg]; fb = tables[tg]["H"][tables[tg]["y"] == 1].mean(0)
            keep = rel > 0.15; Xtr.append(Z[keep]); ytr.append((fb[keep] >= 0.25).astype(int))
        Xtr, ytr = np.vstack(Xtr), np.concatenate(ytr)
        clf = LogisticRegression(C=0.5, class_weight="balanced", max_iter=500).fit(Xtr, ytr)
        res, Z, rel, d, hascons = feats[held]; fb = tables[held]["H"][tables[held]["y"] == 1].mean(0); keep = rel > 0.15; yl = (fb[keep] >= 0.25).astype(int)
        w_fit = dict(zip(hs.FEATURES, clf.coef_[0])); w_def = hs.DEFAULT_W
        s_fit = Z @ np.array([w_fit[k] for k in hs.FEATURES]); s_def = Z @ np.array([w_def[k] for k in hs.FEATURES])
        Kf = lambda s: 0.5 * s + 0.5 * ((np.exp(-(d / 4.0) ** 2) * (rel[None] > .15)) @ s) / ((np.exp(-(d / 4.0) ** 2) * (rel[None] > .15)).sum(1) + 1e-9)
        a = dict(held_out=held, n_res=int(keep.sum()), n_pos=int(yl.sum()), auroc_exposure_only=auroc(yl, Z[keep, 0]), auroc_default=auroc(yl, Kf(s_def)[keep]), auroc_fitted=auroc(yl, Kf(s_fit)[keep]))
        LOTO.append(a); print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in a.items()})
    # final weights: fit on all four
    X, Y = [], []
    for tg in tables:
        res, Z, rel, d, _ = feats[tg]; fb = tables[tg]["H"][tables[tg]["y"] == 1].mean(0); keep = rel > 0.15; X.append(Z[keep]); Y.append((fb[keep] >= 0.25).astype(int))
    clf = LogisticRegression(C=0.5, class_weight="balanced", max_iter=500).fit(np.vstack(X), np.concatenate(Y))
    w = {k: float(c) for k, c in zip(hs.FEATURES, clf.coef_[0])}; print("\nweights fitted on all four targets:", {k: round(v, 2) for k, v in w.items()})
    (Path(__file__).parent / "results").mkdir(exist_ok=True)
    json.dump(w, open(Path(__file__).parent / "results/hotspot_weights.json", "w"), indent=1); pd.DataFrame(LOTO).to_csv(Path(__file__).parent / "results/hotspot_loto.csv", index=False)
    # what do binder epitopes look like compared with the rest of the exposed surface (raw values)?
    print("\n== feature means: binder-epitope residues vs other exposed residues (z-scored within target) ==")
    rows = []
    for tg, v in tables.items():
        res, Z, rel, d, _ = feats[tg]; fb = v["H"][v["y"] == 1].mean(0); ep = (fb >= 0.25) & (rel > 0.15); oth = (~ep) & (rel > 0.15)
        rows.append({k: Z[ep, i].mean() - Z[oth, i].mean() for i, k in enumerate(hs.FEATURES)} | {"target": tg})
    R = pd.DataFrame(rows).set_index("target"); print(R.round(2).to_string()); R.to_csv(Path(__file__).parent / "results/hotspot_feature_shift.csv")

if __name__ == "__main__": main()
