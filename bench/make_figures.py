"""Figures and numbers for docs/FINAL_REPORT.md. Reads bench/out + bench/results, writes
bench/results/figures/*.png and bench/results/stats.json."""
import json, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from metrics import auroc

OUT = Path("results/figures"); OUT.mkdir(parents=True, exist_ok=True)
C = dict(boltz="#0072B2", v2="#009E73", fast="#CC79A7", v1="#8a8a8a", pxd="#D55E00", bind="#0072B2", non="#6b6b6b")
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e6e6e6", "grid.linewidth": .6, "axes.axisbelow": True, "figure.dpi": 130, "savefig.bbox": "tight"})
S = {}
tg = ["pd-l1", "il7r", "mdm2", "egfr"]; names = {"pd-l1": "PD-L1", "il7r": "IL7R", "mdm2": "MDM2", "egfr": "EGFR", "POOLED": "Pooled"}

# ---------- data
j = pd.read_csv("out/joined.csv")
def paemin(t, i, nt):
    pae = np.load(f"out/boltz/{t}/pred/boltz_results_yaml/predictions/{i}/pae_{i}_model_0.npz")["pae"]
    return min(pae[nt:, :nt].min(), pae[:nt, nt:].min())
nt = {t: len(pd.read_csv("inputs/manifest.csv").query("target==@t").target_seq.iloc[0]) for t in tg}
j["b_paemin"] = [paemin(r.target, r.id, nt[r.target]) for r in j.itertuples()]
au = pd.read_csv("results/auroc_by_target.csv")
x_pd = pd.read_csv("out/pxd_pdl1_scored.csv"); x_md = pd.read_csv("out/pxd_mdm2_scored.csv")

# ---------- fig1: AUROC by scorer and target (ipTM and ipSAE), CI bars
arms = [("b", "Boltz-2", C["boltz"], "o"), ("ptx_v2", "Protenix v2", C["v2"], "s"), ("ptx05_fast", "Protenix 0.5-mini fast", C["fast"], "^"), ("ptx_v1", "Protenix v1.0", C["v1"], "D")]
fig, axs = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
for ax, met, ttl in zip(axs, ["iptm", "ipsae_min"], ["ipTM", "ipSAE (min of directions)"]):
    cols = ["POOLED"] + tg
    for k, (a, lab, col, mk) in enumerate(arms):
        sc = f"{a}_{met}"; d = au[au.score == sc].set_index("target")
        xs = np.arange(len(cols)) + (k - 1.5) * 0.17
        ax.errorbar(xs, [d.loc[c, "auroc"] for c in cols],
                    yerr=[[d.loc[c, "auroc"] - d.loc[c, "lo90"] for c in cols], [d.loc[c, "hi90"] - d.loc[c, "auroc"] for c in cols]],
                    fmt=mk, color=col, ms=5, lw=1, capsize=2, label=lab)
    ax.axhline(.5, color="#444", lw=.8, ls="--"); ax.text(4.45, .505, "chance", ha="right", va="bottom", fontsize=8, color="#444")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels([names[c] for c in cols]); ax.set_title(ttl, loc="left", fontsize=10)
    ax.set_ylim(.3, 1.03)
axs[0].set_ylabel("AUROC vs wet-lab binding label"); axs[0].legend(frameon=False, fontsize=8, loc="lower left", ncol=2)
fig.suptitle("Fig 1. How well each scorer separates experimental binders from non-binders (90% bootstrap CI)", x=0.01, ha="left", fontsize=10.5)
fig.savefig(OUT / "fig1_auroc.png"); plt.close(fig)

# ---------- gate pass rates in ProteinBase (Boltz-2)
j["gate"] = (j.b_ipsae_min >= .5) & (j.b_paemin <= 2)
def wilson(k, n, z=1.645):
    if n == 0: return (np.nan, np.nan)
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)); return ((c - h) / d, (c + h) / d)
rows = []
for t in tg + ["ALL"]:
    g = j if t == "ALL" else j[j.target == t]
    for lab in (1, 0):
        s = g[g.label == lab]; k = int(s.gate.sum()); lo, hi = wilson(k, len(s)); rows.append(dict(target=t, label=lab, k=k, n=len(s), rate=k / len(s), lo=lo, hi=hi))
gt = pd.DataFrame(rows); S["gate_table"] = gt.round(3).to_dict("records")
fig, ax = plt.subplots(figsize=(7.5, 3.6)); cols = ["ALL"] + tg
for off, lab, col, nm in [(-.18, 1, C["bind"], "confirmed binders"), (.18, 0, C["non"], "confirmed non-binders")]:
    d = gt[gt.label == lab].set_index("target").loc[cols]
    ax.bar(np.arange(5) + off, d.rate * 100, .34, color=col, label=nm)
    ax.errorbar(np.arange(5) + off, d.rate * 100, yerr=[(d.rate - d.lo) * 100, (d.hi - d.rate) * 100], fmt="none", ecolor="#222", lw=.8, capsize=2)
    for xx, (hi_, k, n) in enumerate(zip(d.hi, d.k, d.n)): ax.text(xx + off, hi_ * 100 + 2.5, f"{k}/{n}", ha="center", fontsize=7)
ax.set_xticks(range(5)); ax.set_xticklabels(["All"] + [names[t] for t in tg]); ax.set_ylabel("% passing the repo gate"); ax.set_ylim(0, 118)
ax.legend(frameon=False, fontsize=8, loc="upper center", ncol=2, bbox_to_anchor=(.5, -.1))
ax.set_title("Fig 2. The provisional gate (ipSAE ≥ 0.5, PAE_min ≤ 2 Å) on ProteinBase designs (90% Wilson CI)", loc="left", fontsize=10)
fig.savefig(OUT / "fig2_gate.png"); plt.close(fig)

# ---------- fig3: ECDF of ipSAE, PXDesign vs ProteinBase (two panels)
fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.7), sharey=True)
for ax, t, xd in zip(axs, ["pd-l1", "mdm2"], [x_pd, x_md]):
    for v, col, lab in [(j[(j.target == t) & (j.label == 1)].b_ipsae_min, C["bind"], "ProteinBase binders"),
                        (j[(j.target == t) & (j.label == 0)].b_ipsae_min, C["non"], "ProteinBase non-binders"),
                        (xd.b_ipsae_min, C["pxd"], "PXDesign (n=400)")]:
        v = np.sort(v.values); ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", color=col, lw=1.8, label=lab)
    ax.axvline(.5, color="#444", lw=.8, ls="--"); ax.set_xlabel("Boltz-2 ipSAE"); ax.set_title(names[t], loc="left", fontsize=10); ax.set_xlim(0, 1)
axs[0].set_ylabel("cumulative fraction of designs"); h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, frameon=False, fontsize=8, loc="lower center", ncol=3, bbox_to_anchor=(.5, -.08))
fig.suptitle("Fig 3. Generated designs vs wet-lab-tested designs on the same scorer (curves further right = better)", x=0.01, ha="left", fontsize=10.5)
fig.savefig(OUT / "fig3_ecdf.png"); plt.close(fig)

# ---------- fig4: best-of-N
fig, ax = plt.subplots(figsize=(6.4, 3.6)); rng = np.random.default_rng(0); Ns = [1, 2, 4, 8, 16, 32, 64, 100, 200, 400]
for xd, col, lab, mk in [(x_pd, C["pxd"], "PD-L1", "o"), (x_md, C["boltz"], "MDM2", "s")]:
    ys = [np.mean([rng.choice(xd.b_ipsae_min.values, n).max() for _ in range(600)]) for n in Ns]
    ax.plot(Ns, ys, marker=mk, color=col, lw=1.6, ms=4); ax.text(Ns[-1] * 1.05, ys[-1], lab, color=col, va="center", fontsize=9)
    S[f"bestofN_{lab}"] = dict(zip(map(str, Ns), np.round(ys, 3)))
ax.axvline(8, color="#444", lw=.8, ls="--"); ax.text(8.5, .02, "repo default (8)", fontsize=8, color="#444", rotation=90, va="bottom")
ax.axvline(100, color="#444", lw=.8, ls=":"); ax.text(106, .02, "upstream preview (100)", fontsize=8, color="#444", rotation=90, va="bottom")
ax.set_xscale("log"); ax.set_xlim(1, 700); ax.set_ylim(0, 1); ax.set_xlabel("designs generated (N, log scale)"); ax.set_ylabel("expected best ipSAE in the pool")
ax.set_title("Fig 4. Why N matters: expected best score vs number of designs", loc="left", fontsize=10)
fig.savefig(OUT / "fig4_bestofN.png"); plt.close(fig)

# ---------- fig5: MPNN settings
mp = pd.read_csv("out/boltz_mpnn_variants/scores.csv").merge(pd.read_csv("inputs/manifest_mpnn_variants.csv")[["id", "method", "binder_seq"]], on="id")
b0 = x_pd.merge(pd.read_csv("inputs/manifest_pxd_pdl1.csv")[["id", "binder_seq"]], on="id"); b0["method"] = "soluble_T0.1"
a = pd.concat([b0, mp]); lab = {"soluble_T0.1": "soluble, T=0.1\n(this fork)", "original_T0.1": "original, T=0.1", "original_greedy": "original, greedy\n(upstream default)"}
rows = []
for m, g in a.groupby("method"):
    t = "".join(g.binder_seq); k = (g.b_ipsae_min >= .5).sum()
    lo, hi = wilson(k, len(g)); rows.append(dict(cond=m, n=len(g), frac=k / len(g), lo=lo, hi=hi, median=g.b_ipsae_min.median(), KE=(t.count("K") + t.count("E")) / len(t)))
mt = pd.DataFrame(rows).set_index("cond").loc[list(lab)]; S["mpnn"] = mt.round(3).reset_index().to_dict("records")
fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.4))
axs[0].bar(range(3), mt.frac * 100, .55, color=C["pxd"]); axs[0].errorbar(range(3), mt.frac * 100, yerr=[(mt.frac - mt.lo) * 100, (mt.hi - mt.frac) * 100], fmt="none", ecolor="#222", capsize=3, lw=.8)
for i, (f, h, n) in enumerate(zip(mt.frac, mt.hi, mt.n)): axs[0].text(i, h * 100 + .8, f"{f*100:.1f}%  (n={n})", ha="center", fontsize=8)
axs[0].set_ylim(0, 20); axs[0].set_ylabel("% of designs with ipSAE ≥ 0.5"); axs[0].set_title("Score tail", loc="left", fontsize=10)
axs[1].bar(range(3), mt.KE * 100, .55, color="#6b6b6b"); axs[1].axhline(28, color=C["bind"], lw=1.2); axs[1].text(2.45, 29, "real PD-L1 binders 28%", ha="right", fontsize=8, color=C["bind"])
for i, v in enumerate(mt.KE): axs[1].text(i, v * 100 + 1, f"{v*100:.0f}%", ha="center", fontsize=8)
axs[1].set_ylim(0, 55); axs[1].set_ylabel("Lys + Glu share of residues (%)"); axs[1].set_title("Sequence composition", loc="left", fontsize=10)
for ax in axs: ax.set_xticks(range(3)); ax.set_xticklabels([lab[c] for c in mt.index], fontsize=8)
fig.suptitle("Fig 5. ProteinMPNN settings on the same 100 PD-L1 backbones: no effect", x=0.01, ha="left", fontsize=10.5)
fig.savefig(OUT / "fig5_mpnn.png"); plt.close(fig)

# ---------- fig6: cost vs accuracy
tm = j.groupby("target")[[c for c in j.columns if c.endswith("_sec")]].mean()
fig, ax = plt.subplots(figsize=(6.4, 3.8)); pooled = au[(au.target == "POOLED") & au.score.str.endswith("_iptm")].set_index("score")
cs = tm.loc[["pd-l1", "il7r", "mdm2"]].mean()   # s/design on the three smaller targets
for a_, lab_, col, mk in arms:
    ax.errorbar(cs[f"{a_ if a_ != 'b' else 'b'}_sec"], pooled.loc[f"{a_}_iptm", "auroc"], yerr=[[pooled.loc[f"{a_}_iptm", "auroc"] - pooled.loc[f"{a_}_iptm", "lo90"]], [pooled.loc[f"{a_}_iptm", "hi90"] - pooled.loc[f"{a_}_iptm", "auroc"]]],
                fmt=mk, color=col, ms=7, capsize=2, lw=1); dy = {"b": .045, "ptx_v2": -.05}.get(a_, .012); dx = {"ptx_v2": .78}.get(a_, 1.07); ax.text(cs[f"{a_}_sec"] * dx, pooled.loc[f"{a_}_iptm", "auroc"] + dy, lab_, fontsize=8, color=col, ha="right" if a_ == "ptx_v2" else "left")
ax.set_xscale("log"); ax.set_xlabel("seconds per design (inference only, PD-L1/IL7R/MDM2 average, log scale)"); ax.set_ylabel("pooled AUROC (ipTM)"); ax.set_ylim(.48, .82); ax.set_xlim(.5, 14)
ax.set_title("Fig 6. Accuracy vs cost per scorer", loc="left", fontsize=10); fig.savefig(OUT / "fig6_cost.png"); plt.close(fig)
S["seconds_per_design"] = tm.round(2).to_dict()

# ---------- fig7: backbone-level supply of strong designs
fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.4), sharey=True)
for ax, xd, t in zip(axs, [x_pd, x_md], ["PD-L1", "MDM2"]):
    xd = xd.copy(); xd["bb"] = xd.id.str.extract(r"sample_(\d+)_")[0]
    best = xd.groupby("bb").b_ipsae_min.max().sort_values(ascending=False).values
    ax.bar(range(len(best)), best, width=.85, color=[C["pxd"] if v >= .5 else "#c9c9c9" for v in best]); ax.axhline(.5, color="#444", lw=.8, ls="--")
    ax.set_xlabel("backbones, sorted by their best of 4 sequences"); ax.set_title(f"{t}: {int((best>=.5).sum())} of {len(best)} backbones reach ipSAE ≥ 0.5", loc="left", fontsize=9.5)
axs[0].set_ylabel("best ipSAE per backbone")
fig.suptitle("Fig 7. Only a minority of backbones carry good designs (orange = ≥ 0.5)", x=0.01, ha="left", fontsize=10.5)
fig.savefig(OUT / "fig7_backbones.png"); plt.close(fig)
S["backbones_ge_0.5"] = {t: int((xd.assign(bb=xd.id.str.extract(r'sample_(\d+)_')[0]).groupby('bb').b_ipsae_min.max() >= .5).sum()) for t, xd in [("pd-l1", x_pd), ("mdm2", x_md)]}
json.dump(S, open("results/stats.json", "w"), indent=1, default=float); print(json.dumps(S["backbones_ge_0.5"]), "\nok")
