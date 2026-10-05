"""Check 1: does Protenix-0.5-mini 'fast' rank generated designs like Boltz-2 does?"""
import sys, numpy as np, pandas as pd
from scipy.stats import spearmanr
from metrics import auroc
def run(tag, tgt):
    b = pd.read_csv(f"out/pxd_{tag}_scored.csv"); f = pd.read_csv(f"out/fast_pxd_{tag}/scores_all.csv")
    d = b.merge(f, on="id"); d["bgate"] = (d.b_ipsae_min >= .5) & (d.paemin <= 2); d["bpass"] = d.b_ipsae_min >= .5
    print(f"\n=== {tgt}: {len(d)} designs, Boltz-2 gate pass {d.bgate.sum()} ({d.bgate.mean():.1%}), ipSAE>=.5 {d.bpass.sum()}")
    out = []
    for m in ["ptx05_fast_iptm", "ptx05_fast_ipsae_min", "ptx05_fast_ipsae_max", "ptx05_fast_rank"]:
        dd = d.dropna(subset=[m]); r = spearmanr(dd[m], dd.b_ipsae_min)[0]
        row = dict(metric=m.replace("ptx05_fast_", ""), spearman_vs_boltz_ipsae=round(r, 2), auroc_gate=round(auroc(dd.bgate.astype(int), dd[m]), 2))
        for q in (.05, .10, .20):
            sel = dd.sort_values(m, ascending=False).head(int(len(dd) * q)); row[f"gate recall@top{int(q*100)}%"] = round(sel.bgate.sum() / max(dd.bgate.sum(), 1), 2); row[f"precision@top{int(q*100)}%"] = round(sel.bgate.mean(), 2)
        out.append(row)
    print(pd.DataFrame(out).to_string(index=False)); print("base rate (gate pass):", round(d.bgate.mean(), 3))
    # backbone-level: does fast pick the backbones that hold the Boltz-2 winners?
    d["bb"] = d.id.str.extract(r"sample_(\d+)_")[0]
    bf = d.groupby("bb").ptx05_fast_ipsae_min.max(); bb_ = d.groupby("bb").bgate.max()
    top = bf.sort_values(ascending=False).head(20).index; print(f"top-20 backbones by fast score contain {int(bb_[top].sum())} of {int(bb_.sum())} gate-passing backbones (random expectation {20*bb_.mean():.1f})")
    return d
for tag, tgt in [x.split(":") for x in sys.argv[1:]]: run(tag, tgt)
