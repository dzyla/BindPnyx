"""Build <run>/final_design/ : the review package for one funnel run.

  python funnel/final_design.py --out out/funnel/<name> --target <name> [--top 20] [--variant auto|cycled|nocycle]

Contents of final_design/
  README.md              review report: settings, stage counts and GPU time, shortlist table, flags, composition, diversity, caveats, next step
  designs.csv            the shortlist, ranked, with every score, flag and tier
  evaluated_all.csv      every design that reached the expensive models (not only the shortlist)
  sequences.fasta        shortlist sequences
  models/                predicted complexes: rankNN_<id>_boltz2.{cif,pdb} (+ _protenixv2.cif); chain A target, chain B binder
  plots/*.png            funnel yield, score scatter, per-design heatmap, hotspot burial, composition vs wet-lab binders, diversity, seed stability, cycling
  view_pymol.pml / view_chimerax.cxc   open the top models, hotspots shown as sticks
  stats.json             machine-readable statistics
Everything here is computational. Nothing in this folder has been tested in a wet lab."""
import argparse, json, shutil, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent)); import common
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

REPO = common.REPO
BLUE, VERM, GREEN, GREY, SKY = "#0072B2", "#D55E00", "#009E73", "#8a8a8a", "#56B4E9"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e6e6", "axes.axisbelow": True, "figure.dpi": 130, "savefig.bbox": "tight"})

O2_LABEL = {"protenix-v2": "Protenix-v2", "af3": "AlphaFold3"}

def o2_label(ev):
    """Display name of the second oracle (columns are called v2_* whichever model produced them; run folders written before --judge existed have no o2_name)."""
    n = ev["o2_name"].iloc[0] if "o2_name" in ev and len(ev) else "protenix-v2"
    return O2_LABEL.get(n, n)

def n_boltz_seeds(ev): return int(ev["b_nseeds"].max()) if "b_nseeds" in ev and len(ev) else 3

def _p(x):
    x = Path(str(x)); return x if x.is_absolute() else REPO / x

def comp(seq):
    n = len(seq); f = lambda a: sum(seq.count(c) for c in a) / n
    return dict(KE=f("KE"), aromatic=f("FWY"), hydrophobic=f("AILMFVW"), net_charge=seq.count("K") + seq.count("R") - seq.count("D") - seq.count("E"))

def tier(r):
    """A: both models >=0.7, interface PAE <=1 A, >=80% of hotspots contacted, no quality flag.  B: passes both judges (any flag, or lower scores).  C: everything else.
    Historical reference only: on 206 wet-lab-labelled designs 'both models >= 0.7' had a 75% binder rate (a balanced set, so absolute rates are NOT transferable)."""
    flags = isinstance(r.get("pisa_flags"), str) and r["pisa_flags"] != ""
    if r["consensus_pass"] and r["b_ipsae"] >= .7 and r["v2_ipsae"] >= .7 and r["b_paemin"] <= 1.0 and r["hotspot_frac"] >= .8 and not flags: return "A"
    return "B" if r["consensus_pass"] else "C"

def build(out, target_name, top=20, variant="auto"):
    out = _p(out); t = common.load_target(target_name); fd = out / "final_design"; shutil.rmtree(fd, ignore_errors=True)
    for sub in ("models", "plots"): (fd / sub).mkdir(parents=True)
    if variant == "auto": variant = "cycled" if (out / "final_cycled.csv").exists() else "nocycle"
    ev = pd.read_csv(out / f"consensus_{variant}.csv"); sl = pd.read_csv(out / f"final_{variant}.csv").head(top).copy()
    timers = json.load(open(out / "timers.json")) if (out / "timers.json").exists() else {}
    screen = pd.read_csv(out / "screen.csv") if (out / "screen.csv").exists() else pd.DataFrame()
    gen = json.load(open(out / "gen/converted.json")) if (out / "gen/converted.json").exists() else {}
    args = json.load(open(out / "run_args.json")) if (out / "run_args.json").exists() else {}
    for d in (ev, sl):
        d["tier"] = [tier(r) for r in d.to_dict("records")]
        c = pd.DataFrame([comp(s) for s in d.seq]);
        for k in c: d[k] = c[k].values
    sl.insert(0, "rank", range(1, len(sl) + 1))
    if len(screen):
        f = screen.set_index("id")["fast_ipsae"].to_dict(); sl["fast_ipsae_at_screen"] = sl.id.map(f)
    # ---- models
    for r in sl.itertuples():
        for key, suf in (("b_cif", "boltz2"), ("v2_cif", "protenixv2")):
            src = getattr(r, key, None)
            if isinstance(src, str) and _p(src).exists():
                dst = fd / "models" / f"rank{r.rank:02d}_{r.id}_{suf}.cif"; shutil.copy2(_p(src), dst)
                if suf == "boltz2":
                    try:
                        from Bio.PDB import MMCIFParser, PDBIO
                        io = PDBIO(); io.set_structure(MMCIFParser(QUIET=True).get_structure("x", str(dst))); io.save(str(dst.with_suffix(".pdb")))
                    except Exception: pass
    # ---- per-hotspot burial (fastPISA per-residue BSA) for the shortlist
    hs = []
    try:
        import pisa, fastpisa
        for r in sl.itertuples():
            m = fd / "models" / f"rank{r.rank:02d}_{r.id}_boltz2.cif"; res = fastpisa.analyze(str(m)); rs = res.interface_between("A", "B").residues(side=1)
            b = {int(x["seq"]): float(x["bsa"]) for x in rs}; hs.append([b.get(i, 0.0) for i in t["hotspot_idx"]])
    except Exception as e:
        print("hotspot burial skipped:", repr(e)[:100]); hs = []
    sl.to_csv(fd / "designs.csv", index=False); ev.to_csv(fd / "evaluated_all.csv", index=False)
    (fd / "sequences.fasta").write_text("".join(f">rank{r.rank:02d}_{r.id} tier={r.tier} consensus={r.consensus:.3f}\n{r.seq}\n" for r in sl.itertuples()))
    # ---- plots
    _plots(fd, t, sl, ev, screen, gen, timers, hs, out, variant)
    # ---- viewer scripts
    cols = ["tomato", "orange", "yellow", "green", "cyan", "blue", "purple", "magenta"]
    hsel = "+".join(map(str, t["hotspot_idx"])); P = ["bg_color white", "set cartoon_fancy_helices, 1"]
    cx = ["set bgColor white"]
    for r in sl.head(10).itertuples():
        n = f"rank{r.rank:02d}_{r.id}_boltz2"; P += [f"load models/{n}.pdb, {n}", f"color gray70, {n} and chain A", f"color {cols[(r.rank - 1) % len(cols)]}, {n} and chain B", f"show sticks, {n} and chain A and resi {hsel}", f"color red, {n} and chain A and resi {hsel}"]
        cx += [f"open models/{n}.cif", f"color #{r.rank} gray target", f"color #{r.rank} {cols[(r.rank - 1) % len(cols)]} & /B", f"show #{r.rank}/A:{hsel.replace('+', ',')} atoms", f"color #{r.rank}/A:{hsel.replace('+', ',')} red"]
    P += ["align " + " or ".join(f"rank{r.rank:02d}_{r.id}_boltz2 and chain A" for r in sl.head(10).itertuples()).replace(" or ", ", ", 1) if len(sl) > 1 else "zoom", "zoom"]
    (fd / "view_pymol.pml").write_text("\n".join(P) + "\n"); (fd / "view_chimerax.cxc").write_text("\n".join(cx + ["matchmaker #2-10 to #1 & /A", "view"]) + "\n")
    # ---- stats
    S = _stats(t, sl, ev, screen, gen, timers, args, variant, hs); json.dump(S, open(fd / "stats.json", "w"), indent=1, default=float)
    (fd / "README.md").write_text(_report(t, sl, ev, S, variant, hs)); print("final_design written:", fd); return fd

def _stats(t, sl, ev, screen, gen, timers, args, variant, hs):
    S = dict(target=t["name"], variant=variant, hotspots=t["hotspots"], hotspot_idx=t["hotspot_idx"], binder_length=t["binder_length"], target_length=len(t["seq"]))
    S["settings"] = args; S["timers_s"] = {k: round(v) for k, v in timers.items()}
    keep = ("1_generate", "2_mpnn", "3_fast_screen") + (("4_cycling",) if variant == "cycled" else ()) + (f"5_boltz_{variant}", f"5_v2_{variant}", f"5_af3_{variant}")
    S["gpu_hours_this_variant"] = round(sum(timers.get(k, 0) for k in keep) / 3600, 2)
    S["counts"] = dict(backbones=gen.get("n"), designs_screened=int(len(screen)), screened_ok=int(screen.fast_ok.sum()) if "fast_ok" in screen else None,
                       evaluated_expensive=int(len(ev)), consensus_pass=int(ev.consensus_pass.sum()), boltz_gate=int(ev.b_gate.sum()), v2_pass=int(ev.v2_pass.sum()),
                       shortlist=int(len(sl)), shortlist_consensus_pass=int(sl.consensus_pass.sum()))
    S["shortlist"] = {k: (float(sl[k].median()), float(sl[k].min()), float(sl[k].max())) for k in ("b_ipsae", "v2_ipsae", "b_paemin", "hotspot_frac", "pisa_sc", "pisa_interface_area", "KE", "aromatic", "hydrophobic") if k in sl}
    S["tiers"] = sl.tier.value_counts().to_dict(); S["tiers_evaluated"] = ev.tier.value_counts().to_dict()
    S["flags"] = sl.pisa_flags.fillna("").str.split(";").explode().replace("", np.nan).dropna().value_counts().to_dict() if "pisa_flags" in sl else {}
    S["clusters_60pct"] = common.cluster_count(sl.seq.tolist(), 0.6); S["pairwise_identity_mean"] = float(np.mean([common.identity(a, b) for i, a in enumerate(sl.seq) for b in sl.seq[i + 1:]])) if len(sl) > 1 else None
    if len(screen) and "fast_ipsae" in screen:
        e = ev.merge(screen[["id", "fast_ipsae"]], on="id", how="left", suffixes=("", "_s")); fcol = "fast_ipsae_s" if "fast_ipsae_s" in e else "fast_ipsae"
        from scipy.stats import spearmanr
        ok = e.dropna(subset=[fcol, "consensus"]); S["screen_vs_consensus_spearman_within_top60"] = float(spearmanr(ok[fcol], ok.consensus)[0]) if len(ok) > 5 else None
    S["hotspot_bsa_A2_median_per_residue"] = dict(zip(t["hotspots"], np.median(np.array(hs), axis=0).round(1).tolist())) if len(hs) else None
    # adaptive rule from the docs
    top20 = sl.head(20); S["adaptive_rule"] = dict(top20_pass_fraction=float(top20.consensus_pass.mean()), v2_median=float(top20.v2_ipsae.median()),
                                                   recommend_cycling=bool(top20.consensus_pass.mean() < 0.9 or top20.v2_ipsae.median() < 0.7) and variant == "nocycle")
    return S

def _plots(fd, t, sl, ev, screen, gen, timers, hs, out, variant):
    P = fd / "plots"
    # 1 funnel yield
    n = [("backbones generated", gen.get("n")), ("sequences designed (MPNN)", len(screen) or None), ("fast-screened", int(screen.fast_ok.sum()) if "fast_ok" in screen else None),
         (f"given Boltz-2 + {o2_label(ev)}", len(ev)), ("pass both models", int(ev.consensus_pass.sum())), ("shortlist (de-duplicated)", len(sl))]
    n = [(a, b) for a, b in n if b]; fig, ax = plt.subplots(figsize=(7.4, 3.3)); y = np.arange(len(n))[::-1]
    ax.barh(y, [b for _, b in n], color=[GREY, GREY, SKY, BLUE, GREEN, VERM][:len(n)], height=.62); ax.set_xscale("log"); ax.set_yticks(y); ax.set_yticklabels([a for a, _ in n])
    for yy, (_, b) in zip(y, n): ax.text(b * 1.08, yy, f"{b:,}", va="center", fontsize=9)
    ax.set_xlim(1, max(b for _, b in n) * 6); ax.set_xlabel("designs (log scale)"); ax.set_title(f"Funnel yield: {t['name']}", loc="left"); fig.savefig(P / "01_funnel_yield.png"); plt.close(fig)
    # 2 score scatter
    fig, ax = plt.subplots(figsize=(5.4, 4.9)); sset = set(sl.id)
    ok = ev.dropna(subset=["b_ipsae", "v2_ipsae"]); a = ok[~ok.id.isin(sset)]; b = ok[ok.id.isin(sset)]
    ax.scatter(a.b_ipsae, a.v2_ipsae, s=26, c=GREY, alpha=.7, label=f"evaluated, not in shortlist ({len(a)})")
    for tr, m, c in (("A", "o", GREEN), ("B", "s", BLUE), ("C", "^", VERM)):
        g = b[b.tier == tr]
        if len(g): ax.scatter(g.b_ipsae, g.v2_ipsae, s=62, marker=m, c=c, edgecolor="k", linewidth=.6, label=f"shortlist tier {tr} ({len(g)})")
    ax.axvline(.5, c="#444", lw=.8, ls="--"); ax.axhline(.5, c="#444", lw=.8, ls="--"); ax.set_xlabel("Boltz-2 ipSAE (mean of seeds)"); ax.set_ylabel(f"{o2_label(ev)} ipSAE"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("Two independent predictors (dashed = 0.5 gate)", loc="left"); ax.legend(frameon=False, fontsize=7.5, loc="lower right"); fig.savefig(P / "02_scores_scatter.png"); plt.close(fig)
    # 3 heatmap of the shortlist
    cols = [("b_ipsae", "Boltz ipSAE", 1), ("v2_ipsae", "v2 ipSAE", 1), ("b_iptm", "Boltz ipTM", 1), ("b_paemin", "interface PAE (A)", -1), ("hotspot_frac", "hotspots contacted", 1), ("pisa_sc", "shape compl.", 1),
            ("pisa_interface_area", "interface area (A2)", 1), ("pisa_n_hydrogen_bonds", "H-bonds", 1), ("pisa_n_aromatic_iface", "aromatic contacts", 1), ("pisa_bsa_apolar_frac", "apolar fraction", 1), ("KE", "Lys+Glu", -1)]
    cols = [c for c in cols if c[0] in sl]; M = sl[[c[0] for c in cols]].astype(float).values
    Z = (M - np.nanmean(M, 0)) / (np.nanstd(M, 0) + 1e-9) * np.array([c[2] for c in cols])
    fig, ax = plt.subplots(figsize=(1.0 + 0.82 * len(cols), 0.9 + 0.33 * len(sl))); im = ax.imshow(Z, cmap="RdBu", vmin=-2.2, vmax=2.2, aspect="auto"); ax.grid(False)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels([c[1] for c in cols], rotation=40, ha="right", fontsize=8); ax.set_yticks(range(len(sl))); ax.set_yticklabels([f"{r.rank:>2d} {r.id} [{r.tier}]" for r in sl.itertuples()], fontsize=7.5)
    for i in range(M.shape[0]):
        for j, c in enumerate(cols):
            v = M[i, j]; ax.text(j, i, f"{v:.0f}" if c[0] in ("pisa_interface_area", "pisa_n_hydrogen_bonds", "pisa_n_aromatic_iface") else f"{v:.2f}", ha="center", va="center", fontsize=6.5)
    ax.set_title("Shortlist metrics. Colour = rank WITHIN this shortlist (blue = better), not absolute quality; read the numbers", loc="left", fontsize=8.5); fig.savefig(P / "03_design_heatmap.png"); plt.close(fig)
    # 4 hotspot burial
    if len(hs):
        H = np.array(hs); fig, ax = plt.subplots(figsize=(1.2 + .9 * H.shape[1], 1.0 + .3 * H.shape[0])); im = ax.imshow(H, cmap="Greens", aspect="auto"); ax.grid(False)
        ax.set_xticks(range(H.shape[1])); ax.set_xticklabels(t["hotspots"]); ax.set_yticks(range(len(sl))); ax.set_yticklabels([f"{r.rank:>2d} {r.id}" for r in sl.itertuples()], fontsize=7.5)
        for i in range(H.shape[0]):
            for j in range(H.shape[1]): ax.text(j, i, f"{H[i, j]:.0f}", ha="center", va="center", fontsize=7)
        ax.set_title("Buried surface area on each requested hotspot (A2, fastPISA; 0 = not buried)", loc="left", fontsize=9); fig.colorbar(im, ax=ax, shrink=.7); fig.savefig(P / "04_hotspot_burial.png"); plt.close(fig)
    # 5 composition vs wet-lab binders
    ref = _reference(); fig, axs = plt.subplots(1, 3, figsize=(9, 3.1))
    for ax, (k, lab) in zip(axs, (("KE", "Lys + Glu fraction"), ("aromatic", "aromatic (F,W,Y) fraction"), ("hydrophobic", "hydrophobic (AILMFVW) fraction"))):
        ax.hist(ev[k], bins=14, color=GREY, alpha=.6, label="evaluated"); ax.hist(sl[k], bins=14, color=VERM, alpha=.8, label="shortlist")
        if ref.get(k) is not None: ax.axvline(ref[k], c=BLUE, lw=2); ax.text(ref[k], ax.get_ylim()[1] * .96, " wet-lab binders\n (median)", color=BLUE, fontsize=7.5, va="top")
        ax.set_xlabel(lab)
    axs[0].legend(frameon=False, fontsize=7.5); fig.suptitle("Sequence composition vs ProteinBase wet-lab binders", x=.01, ha="left", fontsize=10); fig.savefig(P / "05_composition.png"); plt.close(fig)
    # 6 diversity
    if len(sl) > 1:
        n_ = len(sl); I = np.array([[common.identity(a, b) for b in sl.seq] for a in sl.seq]); fig, ax = plt.subplots(figsize=(4.8, 4.4)); im = ax.imshow(I, cmap="viridis", vmin=0, vmax=1); ax.grid(False)
        ax.set_xticks(range(n_)); ax.set_xticklabels(sl["rank"], fontsize=7); ax.set_yticks(range(n_)); ax.set_yticklabels(sl["rank"], fontsize=7); ax.set_title(f"Pairwise sequence identity (mean {I[np.triu_indices(n_, 1)].mean():.2f})", loc="left", fontsize=9); fig.colorbar(im, ax=ax, shrink=.8); fig.savefig(P / "06_diversity.png"); plt.close(fig)
    # 7 seed stability
    fig, ax = plt.subplots(figsize=(7.2, 3.3)); x = np.arange(len(sl))
    ax.errorbar(x, sl.b_ipsae, yerr=sl.get("b_ipsae_sd", 0), fmt="o", color=BLUE, capsize=2, label="Boltz-2 mean +- SD over seeds"); ax.plot(x, sl.v2_ipsae, "s", color=GREEN, label=f"{o2_label(ev)} (1 seed)")
    ax.axhline(.5, c="#444", lw=.8, ls="--"); ax.set_xticks(x); ax.set_xticklabels(sl["rank"]); ax.set_xlabel("shortlist rank"); ax.set_ylabel("ipSAE"); ax.set_ylim(0, 1); ax.legend(frameon=False, fontsize=8, ncol=2, loc="lower left"); ax.set_title("Score stability across seeds", loc="left"); fig.savefig(P / "07_seed_stability.png"); plt.close(fig)
    # 8 cycling
    cp, sp = out / "cycled_parents.csv", out / "start_parents.csv"
    if variant == "cycled" and cp.exists() and sp.exists():
        a = pd.read_csv(sp).set_index("bb").fast_ipsae; b = pd.read_csv(cp).set_index("bb").fast_ipsae; j = pd.concat([a, b], axis=1, keys=["start", "final"]).dropna()
        fig, ax = plt.subplots(figsize=(4.8, 4.6)); ax.scatter(j.start, j.final, s=22, c=BLUE, alpha=.8); ax.plot([0, 1], [0, 1], c="#444", lw=.8); ax.set_xlabel("start (best of 4 MPNN sequences)"); ax.set_ylabel("after cycling (fast ipSAE)")
        ax.set_title(f"Cycling: {len(j)} backbones, mean {j.start.mean():.2f} -> {j.final.mean():.2f}\n(in-loop judge: expect over-optimism; see independent scores)", loc="left", fontsize=8.5); fig.savefig(P / "08_cycling.png"); plt.close(fig)

def _reference():
    """Median composition of wet-lab-confirmed binders in the ProteinBase benchmark set (bench/inputs/manifest.csv, label == 1; 4 targets); {} if absent."""
    try:
        x = pd.read_csv(REPO / "bench/inputs/manifest.csv"); b = x[x.label == 1].binder_seq.dropna(); c = pd.DataFrame([comp(s) for s in b]); return {k: float(c[k].median()) for k in ("KE", "aromatic", "hydrophobic")}
    except Exception:
        return {}

def _report(t, sl, ev, S, variant, hs):
    c = S["counts"]; ref = _reference(); sh = S["shortlist"]; rule = S["adaptive_rule"]
    L = [f"# Final design review: {t['name']}", "",
         f"*Generated {time.strftime('%Y-%m-%d %H:%M')} by `funnel/final_design.py`. **All numbers are computational predictions; nothing here is wet-lab validated.***", "",
         "## 1. Summary", "",
         f"- **Target:** {t['description']}", f"- **Requested site:** {', '.join(t['hotspots'])} (shard indices {t['hotspot_idx']}); binder length {t['binder_length']}; target construct {S['target_length']} aa",
         f"- **Strategy:** funnel, variant **{variant}** ({'with' if variant == 'cycled' else 'without'} refold-redesign cycling); GPU time for this variant **{S['gpu_hours_this_variant']} h**",
         f"- **Result:** {c['consensus_pass']} of {c['evaluated_expensive']} designs given the expensive models pass both judges; shortlist of {c['shortlist']} de-duplicated designs, of which {c['shortlist_consensus_pass']} pass; tiers {S['tiers']}",
         f"- **Pass rule:** Boltz-2 mean ipSAE >= 0.5 and interface PAE <= 2 A, **and** {o2_label(ev)} ipSAE >= 0.5", ""]
    L += ["## 2. Stage counts and time", "", "| stage | count | seconds |", "|---|---|---|"]
    tm = S["timers_s"]; rows = [("backbones generated", c["backbones"], tm.get("1_generate")), ("MPNN sequences", c["designs_screened"], tm.get("2_mpnn")), ("fast screen (Protenix 0.5-mini)", c["screened_ok"], tm.get("3_fast_screen")),
                                ("cycling", "" , tm.get("4_cycling")), (f"Boltz-2 ({n_boltz_seeds(ev)} seed{'s' if n_boltz_seeds(ev) != 1 else ''})", c["evaluated_expensive"], tm.get(f"5_boltz_{variant}")), (o2_label(ev), c["evaluated_expensive"], tm.get(f"5_v2_{variant}", tm.get(f"5_af3_{variant}")))]
    L += [f"| {a} | {b if b is not None else ''} | {s if s is not None else ''} |" for a, b, s in rows]
    L += ["", "![funnel](plots/01_funnel_yield.png)", ""]
    L += ["## 3. Shortlist", "", "| rank | id | tier | Boltz ipSAE | v2 ipSAE | PAE min (A) | hotspots | SC | area (A2) | H-bonds | arom. contacts | flags | sequence |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sl.itertuples():
        g = lambda k, f="{:.2f}": (f.format(getattr(r, k)) if hasattr(r, k) and getattr(r, k) == getattr(r, k) else "")
        L.append(f"| {r.rank} | `{r.id}` | {r.tier} | {g('b_ipsae')} | {g('v2_ipsae')} | {g('b_paemin')} | {g('hotspot_frac')} | {g('pisa_sc')} | {g('pisa_interface_area', '{:.0f}')} | {g('pisa_n_hydrogen_bonds', '{:.0f}')} | {g('pisa_n_aromatic_iface', '{:.0f}')} | {getattr(r, 'pisa_flags', '') if isinstance(getattr(r, 'pisa_flags', ''), str) else ''} | `{r.seq}` |")
    L += ["", "Tiers: **A** = both models >= 0.7, interface PAE <= 1 A, >= 80% of hotspots contacted, no quality flag. **B** = passes both judges. **C** = other. Tiers rank *model agreement*, not binding.", "",
          "![heatmap](plots/03_design_heatmap.png)", "", "![scatter](plots/02_scores_scatter.png)", ""]
    if len(hs): L += ["![hotspots](plots/04_hotspot_burial.png)", ""]
    L += ["## 4. Statistics for review", "", "| quantity | median | min | max |", "|---|---|---|---|"]
    names = dict(b_ipsae="Boltz-2 ipSAE", v2_ipsae=f"{o2_label(ev)} ipSAE", b_paemin="interface PAE (A)", hotspot_frac="hotspot fraction contacted", pisa_sc="shape complementarity", pisa_interface_area="interface area (A2)", KE="Lys+Glu fraction", aromatic="aromatic fraction", hydrophobic="hydrophobic fraction")
    L += [f"| {names[k]} | {v[0]:.3g} | {v[1]:.3g} | {v[2]:.3g} |" for k, v in sh.items()]
    L += ["", f"- **Diversity:** {S['clusters_60pct']} sequence clusters (60% identity) among {len(sl)} designs; mean pairwise identity {S['pairwise_identity_mean']:.2f}" if S["pairwise_identity_mean"] is not None else "",
          f"- **Quality flags** (fastPISA / shape complementarity; advisory only): {S['flags'] if S['flags'] else 'none'}",
          f"- **Composition vs wet-lab binders (medians of the 206-design benchmark set):** Lys+Glu {sh['KE'][0]:.2f} vs {ref.get('KE', float('nan')):.2f}; aromatics {sh['aromatic'][0]:.3f} vs {ref.get('aromatic', float('nan')):.3f}; hydrophobic {sh['hydrophobic'][0]:.2f} vs {ref.get('hydrophobic', float('nan')):.2f}",
          f"- **Screen quality:** Spearman(fast-screen score, consensus score) within the top-60 that reached the expensive models = {S.get('screen_vs_consensus_spearman_within_top60')} (restricted range, expect low)", "",
          "![composition](plots/05_composition.png)", "", "![diversity](plots/06_diversity.png)", "", "![stability](plots/07_seed_stability.png)", ""]
    if variant == "cycled": L += ["![cycling](plots/08_cycling.png)", ""]
    L += ["## 5. Recommendation from the adaptive rule", "",
          f"Top-20 pass fraction **{rule['top20_pass_fraction']:.2f}**, {o2_label(ev)} median **{rule['v2_median']:.2f}**. " + ("**Re-run with `--rounds 3`** (stages 1-3 are reused) - the no-cycling shortlist is below the 90% / 0.7 bar." if rule["recommend_cycling"] else "No further cycling is indicated by the rule."), "",
          "## 6. How to review", "",
          "1. Open `view_pymol.pml` (or `view_chimerax.cxc`): target grey, binders coloured by rank, requested hotspots as red sticks. Check the binder sits on the intended face and does not clash.",
          "2. Read the flags: *no aromatic contact*, *polar interface*, *few H-bonds*, *thin interface*, *low shape complementarity* are triage hints, validated only weakly (see docs/RECOMMENDATIONS.md section 7).",
          "3. Compare compositions: Lys+Glu-rich and aromatic-poor sequences are typical of MPNN designs on helical backbones; consider the interface-bias option (docs/RECOMMENDATIONS.md section 6/10).",
          "4. Choose a *diverse* panel (one per cluster, spread over tiers, plus controls) instead of the top few. Record the run folder, tiers and settings when ordering.", "",
          "## 7. What this does and does not show", "",
          "- Both judges are structure predictors used in selection as well; on 206 wet-lab-labelled designs they separated binders from non-binders only moderately (pooled AUROC 0.67-0.75) and agreed with each other at Spearman ~0.5.",
          "- The two models place the binder within 5 A of each other in only ~40% of designs; one predicted pose is not a binding mode.",
          "- High predicted confidence is not affinity, specificity, expression or developability.", "",
          "## 8. Files", "", "`designs.csv` (shortlist) | `evaluated_all.csv` | `sequences.fasta` | `models/` | `plots/` | `view_pymol.pml` | `view_chimerax.cxc` | `stats.json`"]
    return "\n".join(x for x in L if x is not None) + "\n"

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--target", required=True); ap.add_argument("--top", type=int, default=20); ap.add_argument("--variant", default="auto")
    a = ap.parse_args(); build(a.out, a.target, a.top, a.variant)
