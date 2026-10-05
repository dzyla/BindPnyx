import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e6e6", "axes.axisbelow": True, "figure.dpi": 130, "savefig.bbox": "tight"})
T = pd.read_csv("results/followup/cycling_trajectory_mdm2.csv")
fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.6), sharey=True)
for ax, col, ttl in zip(axs, ["score", "b_ipsae_min", "ptx_v2_ipsae_min"], ["In-loop judge: Protenix 0.5-mini fast", "Independent: Boltz-2", "Independent: Protenix v2"]):
    w = T.pivot(index="bb", columns="rd", values=col)
    for b, r in w.iterrows():
        d = r[3] - r[0]; ax.plot([0, 1, 2, 3], r.values, color="#D55E00" if d > .03 else ("#6b6b6b" if d < -.03 else "#c9c9c9"), lw=1.2, alpha=.9)
    ax.plot([0, 1, 2, 3], w.mean().values, color="#0072B2", lw=2.5, marker="o", ms=4); ax.axhline(.5, color="#444", ls="--", lw=.8)
    ax.set_xticks(range(4)); ax.set_xticklabels(["start", "round 1", "round 2", "round 3"]); ax.set_title(f"{ttl}\nmean {w[0].mean():.2f} → {w[3].mean():.2f}", loc="left", fontsize=9.5)
axs[0].set_ylabel("best-sequence ipSAE per backbone")
fig.suptitle("Fig 8. Refold-redesign cycling on 20 MDM2 backbones (orange = improved > 0.03, grey = worse, blue = mean)", x=0.01, y=1.1, ha="left", fontsize=10.5)
fig.savefig("results/figures/fig8_cycling.png"); print("ok")
