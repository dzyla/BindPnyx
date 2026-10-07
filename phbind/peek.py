"""Look at S2 results NOW, including designs finished inside the batch that is still running (Boltz writes each prediction as it completes).
Partial-batch rows are marked partial=True (their batch carriers may be missing). Prints per-set/length pass rates and the top designs;
writes out/phbind/candidates_screen.csv (1-seed screening statistic, NOT a validated shortlist)."""
import sys, glob
from pathlib import Path
import pandas as pd
REPO = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(REPO))
from phbind import boltz_trimer as B, s2_prescreen as S
meta = pd.read_csv(REPO / "out/phbind/designs_all.csv"); seq = meta.set_index("id").seq
done = pd.concat([pd.read_csv(f) for f in sorted(S.OUT.glob("batch_*.csv"))], ignore_index=True); done["partial"] = False
parts = []
for od in sorted(S.OUT.glob("b[0-9][0-9][0-9]")):
    if (S.OUT / f"batch_{od.name[1:]}.csv").exists(): continue
    root = od / "seed101/boltz_results_yaml/predictions"
    ids = [p.name for p in root.glob("*") if B._done(root, p.name) and not p.name.startswith("carrier")] if root.exists() else []
    if ids:
        r = B.score_seed(pd.DataFrame({"id": ids, "seq": [seq[i] for i in ids]}), od, 101); r = r.merge(meta[["id", "bb", "set", "L", "run"]], on="id"); r["partial"] = True; parts.append(r)
a = pd.concat([done] + parts, ignore_index=True); a = a[~a.id.str.startswith("carrier")]
a["pass_screen"] = a.ipsae_min >= S.SCREEN_CUT; a["pass45"] = a.ipsae_min >= 0.45; a["cut60"] = a.ipsae_min >= 0.60
print(f"{len(a)} designs scored ({int(a.partial.sum())} from the running batch)  |  >={S.SCREEN_CUT} (screen cut): {a.pass_screen.sum()} ({100*a.pass_screen.mean():.1f}%)  >=0.45: {a.pass45.sum()} ({100*a.pass45.mean():.1f}%)  >=0.60: {a.cut60.sum()} ({100*a.cut60.mean():.1f}%)")
print(a.groupby("set")[["pass_screen", "pass45", "cut60"]].agg(["sum", "count"]).rename(columns={"sum": "n_pass"}).to_string()); print(a.groupby("L")[["pass_screen", "pass45", "cut60"]].mean().round(3).T.to_string())
out = a[a.pass_screen].sort_values("ipsae_min", ascending=False).copy(); out["sequence"] = [seq[i] for i in out.id]
cols = ["id", "sequence", "binder_len", "set", "L", "bb", "ipsae_min", "ipsae_max", "b2t_pair_iptm", "foot_recall", "p1x_recall", "groove", "n_prot_engaged", "partial", "cif"]
def sse(cif):
    import biotite.structure.io as bio, biotite.structure as bs, numpy as np
    a = bio.load_structure(cif, model=1); a = a[(a.chain_id == "D") & (a.atom_name == "CA")]; x = bs.annotate_sse(a)
    return round(float((x == "a").mean()), 2), round(float((x == "b").mean()), 2)
ss = [sse(c) for c in out.cif]; out["helix_frac"] = [x[0] for x in ss]; out["sheet_frac"] = [x[1] for x in ss]; cols += ["helix_frac", "sheet_frac"]
out[cols].to_csv(REPO / "out/phbind/candidates_screen.csv", index=False); print(f"\nwrote out/phbind/candidates_screen.csv ({len(out)} rows)\n")
print(out[["id", "L", "ipsae_min", "ipsae_max", "foot_recall", "p1x_recall", "groove", "helix_frac", "sheet_frac"]].head(12).round(3).to_string(index=False))
