"""Draws docs/figures/workflow.png (the end-to-end workflow). python docs/make_workflow_figure.py"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

COL = {"gpu": "#D6E8F7", "cpu": "#DDF0DD", "human": "#FBE5CC", "data": "#EEEEEE"}; EDGE = {"gpu": "#0072B2", "cpu": "#2E8B57", "human": "#D55E00", "data": "#777777"}
fig, ax = plt.subplots(figsize=(15, 9.2)); ax.set_xlim(0, 15); ax.set_ylim(0, 9.2); ax.axis("off")

def box(x, y, w, h, title, body="", kind="gpu", fs=9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=COL[kind], ec=EDGE[kind], lw=1.6))
    ax.text(x + w / 2, y + h - 0.2, title, ha="center", va="top", fontsize=fs + 0.5, fontweight="bold", color="#222")
    if body: ax.text(x + w / 2, y + h - 0.55, body, ha="center", va="top", fontsize=fs - 1, color="#333", linespacing=1.35)

def arrow(x1, y1, x2, y2, label="", style="-|>", color="#444", ls="-", rad=0.0, lx=0, ly=0.12, fs=8):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=14, lw=1.5, color=color, linestyle=ls, connectionstyle=f"arc3,rad={rad}"))
    if label: ax.text((x1 + x2) / 2 + lx, (y1 + y2) / 2 + ly, label, ha="center", va="bottom", fontsize=fs, color=color, style="italic")

ax.text(0.15, 8.95, "Binder design workflow: many cheap backbones, one cheap screen, expensive models only on survivors", fontsize=13, fontweight="bold", va="top")
ax.text(0.15, 8.55, "blue = GPU stage   green = CPU stage   orange = human / lab   grey = data.   Times: one RTX 5090, 70-85 residue binder (MDM2 / PD-L1).", fontsize=9, color="#555", va="top")

# --- row A: setup
box(0.2, 6.55, 2.7, 1.45, "1  Target", "PDB structure + WT sequence\n+ MSA (query = construct)\nchoose ONE face from structure", "human")
box(3.4, 6.55, 3.0, 1.45, "2  Prepare shard", "scripts/prepare_target.py\nshard + provenance + MSA check", "cpu")
box(6.9, 6.55, 3.3, 1.45, "3  Target definition", 'funnel/targets/<name>.json\nhotspots as "Y56" (PDB number + letter)\nidentity-checked -> shard indices', "cpu")
box(10.7, 6.55, 4.1, 1.45, "Guards (fail loudly)", "wrong residue letter / MSA mismatch\nhotspot beyond shard (legacy guard)\nONE GPU job at a time (check nvidia-smi)", "data", fs=9)
for x1, x2 in [(2.9, 3.4), (6.4, 6.9)]: arrow(x1, 7.27, x2, 7.27)
arrow(10.2, 7.27, 10.7, 7.27, style="-", ls="--", color="#999")

# --- row B: funnel
y = 4.15; h = 1.75
box(0.2, y, 2.35, h, "4  Generate", "PXDesign diffusion\n500 backbones\nhotspot-conditioned\n~2 s each (20 min)", "gpu")
box(2.85, y, 2.35, h, "5  Design", "ProteinMPNN\n4 sequences / backbone\ntarget fixed\noptional interface bias", "gpu")
box(5.5, y, 2.5, h, "6  Fast screen", "Protenix 0.5-mini (2 cycles,\n5 steps), all 2000 designs\n0.4-0.7 s each (15-25 min)\nbest per backbone, top 100", "gpu")
box(8.3, y, 2.55, h, "7  Cycle  (optional)", "MPNN on the PREDICTED\ncomplex, 8 children, fast\nrefold, keep best (elitist)\n3 rounds, +0.4-0.6 h", "gpu")
box(11.15, y, 3.65, h, "8  Consensus (top 60)", "Boltz-2, 3 seeds  +  Protenix-v2\nconsensus = min(ipSAE)\ngate: Boltz ipSAE>=0.5, PAE<=2 A,\nv2 ipSAE>=0.5   (~0.5 h)", "gpu")
for x1, x2 in [(2.55, 2.85), (5.2, 5.5), (8.0, 8.3), (10.85, 11.15)]: arrow(x1, y + h / 2, x2, y + h / 2)
ax.text(9.575, y - 0.17, "optional: skip when the top-20 already passes (rule below)", fontsize=7.8, color="#777", ha="center", style="italic")
arrow(3.7, 6.55, 1.4, y + h, label="shard + hotspots", color="#0072B2", lx=-0.2)

# --- row C: triage / outputs
y2 = 1.95; h2 = 1.45
box(0.2, y2, 3.4, h2, "9  Triage  (CPU, 1-2 s each)", "de-duplicate (<60% identity)\nfastPISA, shape complementarity,\nhotspot burial, composition flags", "cpu")
box(3.9, y2, 3.3, h2, "10  Shortlist", "final_*.csv + predicted complexes\nflags: thin interface, few H-bonds,\nno aromatic contact, low SC", "data")
box(7.5, y2, 3.3, h2, "11  Independent judge", "fresh seeds: Boltz-2 x3 + v2\nsame protocol for every arm\n(only for method comparisons)", "gpu")
box(11.1, y2, 3.7, h2, "Adaptive rule", "pass < 90% of top-20  OR  v2 median < 0.7\n  ->  re-run with --rounds 3 (resumes)\nelse keep the no-cycling shortlist", "data")
ax.plot([12.9, 12.9, 1.9], [y, 3.72, 3.72], color="#2E8B57", lw=1.5); arrow(1.9, 3.72, 1.9, y2 + h2, style="-|>", color="#2E8B57")
ax.text(7.4, 3.78, "survivors + predicted complexes (every stage output is on disk and resumable)", fontsize=8, color="#2E8B57", ha="center", style="italic")
arrow(3.6, y2 + h2 / 2, 3.9, y2 + h2 / 2); arrow(7.2, y2 + h2 / 2, 7.5, y2 + h2 / 2, style="-", ls="--", color="#999"); arrow(10.8, y2 + h2 / 2, 11.1, y2 + h2 / 2, style="-", ls="--", color="#999")

# --- row D: lab
y3 = 0.15; h3 = 1.3
box(0.2, y3, 4.2, h3, "12  Order a DIVERSE panel", "one per sequence cluster, spread over tiers,\n+ a few flagged, + positive and negative controls", "human")
box(4.8, y3, 3.6, h3, "13  Wet lab", "expression + binding\n(target-specific results)", "human")
box(8.8, y3, 6.0, h3, "14  Calibrate  (>=20 results first)", "binder rate per tier / metric band with intervals -> update bands and gates;\ndecide: change generator, sequence design (aromatics), or target site", "cpu")
for x1, x2 in [(4.4, 4.8), (8.4, 8.8)]: arrow(x1, y3 + h3 / 2, x2, y3 + h3 / 2)
arrow(5.5, y2, 2.3, y3 + h3, label="", color="#D55E00")
arrow(11.8, y3 + h3, 11.8, y2 , style="-|>", color="#D55E00", ls="--"); ax.text(12.0, 1.62, "feedback", fontsize=8, color="#D55E00", style="italic")
fig.savefig("docs/figures/workflow.png", dpi=140, bbox_inches="tight"); print("ok")
