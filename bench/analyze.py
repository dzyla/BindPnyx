"""Join every arm's scores with the ProteinBase labels; AUROC (bootstrap CI) per target + pooled."""
import glob, numpy as np, pandas as pd
from metrics import auroc

m = pd.read_csv("inputs/manifest.csv")
d = m[["target", "id", "label", "method"]].copy()
for f in ["out/boltz/scores_all.csv"] + sorted(glob.glob("out/ptx*/scores_all.csv")):
    s = pd.read_csv(f).drop_duplicates(["target", "id"]); d = d.merge(s, on=["target", "id"], how="left")
d.to_csv("out/joined.csv", index=False)
cols = [c for c in d.columns if c.endswith(("_iptm", "_ipsae_min", "_ipsae_max", "_rank")) and not c.startswith("target")]
cols = [c for c in cols if d[c].notna().any()]

def ci(y, s, n=500, rng=np.random.default_rng(0)):
    y, s = np.asarray(y), np.asarray(s, float); ok = ~np.isnan(s); y, s = y[ok], s[ok]
    v = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y)); a = auroc(y[i], s[i])
        if not np.isnan(a): v.append(a)
    return np.percentile(v, [5, 95]) if v else (np.nan, np.nan)

rows = []
for t, g in list(d.groupby("target")) + [("POOLED", d)]:
    for c in cols:
        a = auroc(g.label, g[c]); lo, hi = ci(g.label, g[c]) if g[c].notna().sum() > 10 else (np.nan, np.nan)
        rows.append(dict(target=t, score=c, n=int(g[c].notna().sum()), auroc=round(a, 3), lo90=round(lo, 3), hi90=round(hi, 3)))
r = pd.DataFrame(rows)
print(r.pivot(index="score", columns="target", values="auroc").to_string())
r.to_csv("out/auroc.csv", index=False)

# within-method AUROC (controls for method<->label confound); methods with both classes, >=8 designs
print("\nwithin-method AUROC (method, target, n):")
for (t, mth), g in d.groupby(["target", "method"]):
    if len(g) >= 8 and g.label.nunique() == 2:
        print(f"  {t:7s} {mth:12s} n={len(g):2d} pos={int(g.label.sum()):2d} " + "  ".join(f"{c}={auroc(g.label, g[c]):.2f}" for c in cols if c.endswith(("_ipsae_min", "_iptm"))))
sec = {c: d[c].mean() for c in d.columns if c.endswith("_sec")}
print("\nmean s/design:", {k: round(v, 1) for k, v in sec.items()})
