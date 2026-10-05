"""How well does the fast screen (Protenix-0.5-mini) predict what Boltz-2 / Protenix-v2 will accept? Uses out/calib/<target>/calib.csv (random samples of the screened designs,
folded by both expensive models), so enrichment is measured over the WHOLE range of fast scores. CPU only.
  python funnel/analyze_calibration.py [targets...]    -> printed tables + funnel/results/calibration_summary.csv"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent)); import common, sc as scmod
from scipy.stats import spearmanr
from concurrent.futures import ProcessPoolExecutor
REPO = common.REPO

def auroc(y, s):
    y = np.asarray(y).astype(int); s = np.asarray(s, float); ok = ~np.isnan(s); y, s = y[ok], s[ok]
    p, n = (y == 1).sum(), (y == 0).sum()
    if p == 0 or n == 0: return float("nan")
    from scipy.stats import rankdata
    r = rankdata(s); return float((r[y == 1].sum() - p * (p + 1) / 2) / (p * n))

def fast_struct_feats(args):
    cif, nt, hot = args
    try:
        r = scmod.complex_sc(cif, ("A",), ("B",)); hf, hn = common.hotspot_contacts(cif, nt, hot)
        return dict(f_sc=r["sc"], f_sc_binder=r["sc_b"], f_area=r["area_a"] + r["area_b"], f_hot=hf)
    except Exception: return dict(f_sc=np.nan, f_sc_binder=np.nan, f_area=np.nan, f_hot=np.nan)

def pose_rmsd(a, b, nt):
    from Bio.PDB import MMCIFParser
    def ca(p):
        s = MMCIFParser(QUIET=True).get_structure("x", str(p))[0]; ids = sorted(c.id for c in s); return [np.array([x["CA"].coord for x in s[c] if "CA" in x]) for c in ids[:2]]
    try:
        ta, ba = ca(a); tb, bb = ca(b)
        if len(ta) != len(tb) or len(ba) != len(bb): return np.nan
        R, t = common._kabsch(ta, tb); return float(np.sqrt((((bb @ R.T + t) - ba) ** 2).sum(1).mean()))
    except Exception: return np.nan

def main():
    targets = sys.argv[1:] or ["pdl1", "mdm2", "fima"]; rows = []; allr = []
    for tg in targets:
        f = REPO / "out/calib" / tg / "calib.csv"
        if not f.exists(): print("missing", f); continue
        t = common.load_target(tg); d = pd.read_csv(f); d["target"] = tg; nt = len(t["seq"])
        with ProcessPoolExecutor(8) as ex: feats = list(ex.map(fast_struct_feats, [(r.fast_cif, nt, t["hotspot_idx"]) if isinstance(r.fast_cif, str) else (None, nt, t["hotspot_idx"]) for r in d.itertuples()], chunksize=4))
        d = pd.concat([d, pd.DataFrame(feats)], axis=1)
        d["pose_rmsd"] = [pose_rmsd(r.fast_cif, r.b_cif, nt) if isinstance(r.fast_cif, str) and isinstance(r.b_cif, str) else np.nan for r in d.itertuples()]
        allr.append(d); y = d.consensus_pass.astype(int); base = y.mean()
        row = dict(target=tg, n=len(d), boltz_gate=round(d.b_gate.mean(), 3), v2_pass=round(d.v2_pass.mean(), 3), consensus_pass=round(base, 3),
                   rho_fast_vs_boltz=round(spearmanr(d.fast_ipsae, d.b_ipsae, nan_policy="omit")[0], 2), rho_fast_vs_v2=round(spearmanr(d.fast_ipsae, d.v2_ipsae, nan_policy="omit")[0], 2),
                   auroc_pass=round(auroc(y, d.fast_ipsae), 2), auroc_pass_iptm=round(auroc(y, d.fast_iptm), 2), auroc_pass_with_struct=np.nan)
        for q in (0.1, 0.2, 0.3, 0.5):
            top = d.sort_values("fast_ipsae", ascending=False).head(int(len(d) * q)); row[f"prec@top{int(q*100)}%"] = round(top.consensus_pass.mean(), 2); row[f"recall@top{int(q*100)}%"] = round(top.consensus_pass.sum() / max(1, y.sum()), 2)
        rows.append(row)
    S = pd.DataFrame(rows); print(S.to_string(index=False)); Path(REPO / "funnel/results").mkdir(exist_ok=True); S.to_csv(REPO / "funnel/results/calibration_summary.csv", index=False)
    A = pd.concat(allr); A.to_csv(REPO / "out/calib/all_features.csv", index=False)
    print("\npose agreement (fast vs Boltz-2, binder RMSD after target alignment, A): median %.1f; fraction < 5 A: %.2f" % (A.pose_rmsd.median(), (A.pose_rmsd < 5).mean()))
    for tg, g in A.groupby("target"):
        hi = g[g.fast_ipsae >= g.fast_ipsae.quantile(.8)]; print(f"  {tg}: pose RMSD median {g.pose_rmsd.median():.1f} overall, {hi.pose_rmsd.median():.1f} in the top fifth by fast ipSAE")
    # does geometry computed on the FAST structure add information beyond the fast ipSAE?
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    A = A.dropna(subset=["fast_ipsae", "f_sc", "f_hot"]).copy(); y = A.consensus_pass.astype(int).values
    for c in ["fast_ipsae", "fast_iptm", "f_sc", "f_sc_binder", "f_area", "f_hot"]: A[c + "_z"] = A.groupby("target")[c].transform(lambda s: (s - s.mean()) / (s.std() + 1e-9))
    print("\nCV AUROC for predicting consensus_pass (pooled over targets, features z-scored per target):")
    for name, cols in [("fast ipSAE", ["fast_ipsae_z"]), ("fast ipSAE + ipTM", ["fast_ipsae_z", "fast_iptm_z"]), ("fast ipSAE + SC", ["fast_ipsae_z", "f_sc_z"]), ("fast ipSAE + hotspot contact", ["fast_ipsae_z", "f_hot_z"]),
                       ("fast ipSAE + ipTM + SC + area + hotspot", ["fast_ipsae_z", "fast_iptm_z", "f_sc_z", "f_area_z", "f_hot_z"])]:
        if y.sum() < 8: print("  too few passes"); break
        a = np.mean([auroc(y, cross_val_predict(LogisticRegression(max_iter=2000), A[cols].values, y, cv=StratifiedKFold(5, shuffle=True, random_state=s), method="predict_proba")[:, 1]) for s in range(10)])
        print(f"  {name:42s} {a:.3f}")

if __name__ == "__main__": main()
