"""Aggregate judged arms across targets: consensus passes, passes per GPU-hour, hotspot contact, diversity. Writes
results/ tables and a figure.   python funnel/compare.py [out_dir=out] [results_dir=funnel/results]"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent)); import common
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "out"); RES = Path(sys.argv[2] if len(sys.argv) > 2 else "funnel/results"); RES.mkdir(parents=True, exist_ok=True)
TARGETS = ["mdm2", "pdl1", "fima"]; ARMS = ["default", "scaled", "funnel_nocycle", "funnel"]
LAB = {"default": "pipeline default\n(8×4)", "scaled": "pipeline scaled\n(100×4)", "funnel_nocycle": "funnel\n(no cycling)", "funnel": "funnel\n(full)"}
COL = {"default": "#b9b9b9", "scaled": "#6b6b6b", "funnel_nocycle": "#56B4E9", "funnel": "#D55E00"}

def gpu_seconds(t, arm):
    if arm in ("default", "scaled"):
        f = OUT / "baseline" / f"{t}_{arm}" / "timer.json"; return json.load(open(f))["seconds"] if f.exists() else np.nan
    T = json.load(open(OUT / "funnel" / t / "timers.json")); base = sum(T.get(k, 0) for k in ("1_generate", "2_mpnn", "3_fast_screen"))
    if arm == "funnel_nocycle": return base + T.get("5_boltz_nocycle", 0) + T.get("5_v2_nocycle", 0)
    return base + T.get("4_cycling", 0) + T.get("5_boltz_cycled", 0) + T.get("5_v2_cycled", 0)

rows = []
for t in TARGETS:
    jf = OUT / "judge" / t / "judged.csv"
    if not jf.exists(): continue
    d = pd.read_csv(jf)
    for arm in ARMS:
        g = d[d.arms.fillna("").str.split(";").apply(lambda l: arm in l)]
        if not len(g): continue
        sec = gpu_seconds(t, arm); p = g[g.consensus_pass]
        rows.append(dict(target=t, arm=arm, judged=len(g), consensus_pass=len(p), boltz_gate=int(g.b_gate.sum()), v2_pass=int(g.v2_pass.sum()),
                         median_boltz_ipsae=g.b_ipsae.median(), median_v2_ipsae=g.v2_ipsae.median(), best_consensus=g.consensus.max(),
                         hotspot_ge60=int((g.hotspot_frac >= .6).sum()), pass_and_hotspot=int((g.consensus_pass & (g.hotspot_frac >= .6)).sum()),
                         pass_clusters=common.cluster_count(p.seq.tolist()) if len(p) else 0, gpu_hours=sec / 3600, pass_per_gpu_hour=len(p) / (sec / 3600) if sec == sec else np.nan))
R = pd.DataFrame(rows); R.round(3).to_csv(RES / "arm_comparison.csv", index=False); print(R.round(2).to_string(index=False))

fig, axs = plt.subplots(1, 3, figsize=(12.5, 3.8)); w = .2
for ax, col, ttl in zip(axs, ["consensus_pass", "pass_and_hotspot", "pass_per_gpu_hour"],
                        ["Designs passing both judges (of ≤20)", "…and contacting ≥60% of the requested hotspots", "Passing designs per GPU-hour"]):
    for k, arm in enumerate(ARMS):
        v = [R[(R.target == t) & (R.arm == arm)][col].iloc[0] if len(R[(R.target == t) & (R.arm == arm)]) else np.nan for t in TARGETS]
        ax.bar(np.arange(3) + (k - 1.5) * w, v, w * .92, color=COL[arm], label=LAB[arm].replace("\n", " "))
        for x, y in zip(np.arange(3) + (k - 1.5) * w, v):
            if y == y: ax.text(x, y, f"{y:.0f}" if col != "pass_per_gpu_hour" else f"{y:.1f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(range(3)); ax.set_xticklabels(["MDM2", "PD-L1", "FimA"]); ax.set_title(ttl, loc="left", fontsize=9.5)
    ax.grid(axis="y", color="#e6e6e6"); ax.set_axisbelow(True); [ax.spines[s].set_visible(False) for s in ("top", "right")]
axs[0].legend(frameon=False, fontsize=7.5, loc="upper left")
fig.suptitle("Strategy comparison, judged identically (Boltz-2 fresh seeds + Protenix v2)", x=0.01, y=1.04, ha="left", fontsize=10.5)
fig.savefig(RES / "fig_strategy_comparison.png", dpi=130, bbox_inches="tight")
