"""pH-release scoring on predicted poses: per-pose PROPKA linkage release, the paired variant-minus-parent delta, the worst-pose check and the parent-baseline triage.

SIGN: every `rel*` column is POSITIVE when binding is weaker at the acid pH than at pH 7.4 (acid RELEASE, what a hold-at-7.4 / release-at-6.0 objective wants).
This is a protonation model on PREDICTED poses, not a measured pH switch, and one run of it carries about +/-0.26 kcal/mol (measured twice by two agents; see
phbind/validation_ph.json). It needs the organisers' `tnf_ph_score.py` (layer K, PROPKA 3): set $TNF_PH_SCORE_DIR (default $TNF_BUNDLE/code) and run in an
environment that has `propka`, `biopython`, `numpy`, `scipy` (a separate venv is fine; do not install into a shared env).

Pure functions (no propka needed, unit-tested): `summarise`, `baseline_ok`. Heavy function (`score_poses`) imports the scorer lazily.
CLI:  python phbind/ph_score.py <manifest.csv> <pose_dir> <out_prefix>     manifest columns: id,parent,role('parent'|'variant'),seed,file
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np
import pandas as pd

PH_RUN_TO_RUN_SD = 0.26          # kcal/mol, sd of the shift in a pH mean between two independent Boltz runs of the same sequences (one agent, 5 variants)
BASELINE_FLOOR = -1.2            # parents below this (pH 6.0 vs 7.4) cannot be rescued by a 1-2 His edit worth ~+1.2: do not spend a panel on them
SUPPORTED = dict(min_pose=5, delta=0.5, z=2.0)    # same bar as the ranker's 'ph_supported'
WORST_POSE_MIN = 0.0             # proposed (not yet a shared rule): every absolute pose release must be > 0


def baseline_ok(rel60_mean: float, floor: float = BASELINE_FLOOR) -> bool:
    """Is a parent's single- or multi-pose baseline high enough for a histidine edit to reach release? A TRIAGE test (one pose has sd ~0.5-0.9), not evidence."""
    return bool(rel60_mean >= floor)


def summarise(per_pose: pd.DataFrame, rel: str = "rel60") -> dict:
    """per_pose: id, parent, role, seed, <rel>. Variants are compared with their parent at the SAME seed (paired).
    Returns {'parents': DataFrame, 'variants': DataFrame}. A variant row has abs_mean/abs_worst (absolute release), d_mean/d_se/z/d_worst (paired delta),
    and the flags supported (delta, pose count and z) and all_positive (every absolute pose > WORST_POSE_MIN AND every paired delta > 0)."""
    need = {"id", "parent", "role", "seed", rel}
    if not need <= set(per_pose.columns): raise ValueError(f"per-pose table lacks {sorted(need - set(per_pose.columns))}")
    d = per_pose.copy()
    if d[rel].isna().any(): raise ValueError("NaN release values: a pose failed to score; never summarise around a missing pose")
    par = d[d.role == "parent"]
    if par.empty: raise ValueError("no parent poses")
    if par.duplicated(["parent", "seed"]).any(): raise ValueError("duplicate parent pose for one (parent, seed)")
    pr = par.set_index(["parent", "seed"])[rel]
    parents = par.groupby("parent")[rel].agg(n_pose="size", rel_mean="mean", rel_sd="std", rel_worst="min")
    parents["baseline_ok"] = parents.rel_mean.map(baseline_ok)
    v = d[d.role == "variant"].copy()
    rows = []
    for vid, g in v.groupby("id"):
        p = g.parent.iloc[0]; missing = [s for s in g.seed if (p, s) not in pr.index]
        if missing: raise ValueError(f"{vid}: no parent pose at seeds {missing}")
        delta = g[rel].values - np.array([pr[(p, s)] for s in g.seed]); n = len(g)
        se = float(delta.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan"); dm = float(delta.mean())
        z = dm / se if se and se > 0 else float("nan")
        rows.append(dict(id=vid, parent=p, n_pose=n, abs_mean=float(g[rel].mean()), abs_worst=float(g[rel].min()), d_mean=dm, d_se=se, z=z, d_worst=float(delta.min()),
                         supported=bool(n >= SUPPORTED["min_pose"] and dm >= SUPPORTED["delta"] and z >= SUPPORTED["z"]),
                         all_positive=bool(g[rel].min() > WORST_POSE_MIN and delta.min() > 0)))
    return {"parents": parents, "variants": pd.DataFrame(rows).set_index("id") if rows else pd.DataFrame()}


def _scorer():
    d = os.environ.get("TNF_PH_SCORE_DIR") or (str(Path(os.environ["TNF_BUNDLE"]) / "code") if "TNF_BUNDLE" in os.environ else "")
    if not d or not (Path(d) / "tnf_ph_score.py").exists():
        raise FileNotFoundError("set $TNF_PH_SCORE_DIR (or $TNF_BUNDLE) to the folder holding tnf_ph_score.py")
    sys.path.insert(0, d)
    import tnf_ph_score
    return tnf_ph_score


def _score_one(args):
    """Module-level so a process pool can pickle it (a function defined inside score_poses cannot be). Imports the scorer in the worker."""
    r, pose_dir, workdir, binder_chain, target_chains = args
    ps = _scorer()
    row = ps.score_one(os.path.join(pose_dir, r["file"]), binder_chain, list(target_chains), None, os.path.join(workdir, r["file"].replace(".pdb", "")), True)
    return dict(id=r["id"], parent=r["parent"], role=r["role"], seed=r["seed"], k_status=row.get("k_status"), n_iface=row.get("k_n_iface_ionisable"),
                rel60=row.get("k_release_60v74"), rel55=row.get("k_release_55v74"))


def score_poses(manifest: pd.DataFrame, pose_dir, workdir="_phwork", workers: int = 8, binder_chain="D", target_chains=("A", "B", "C")) -> pd.DataFrame:
    """Run the K layer on every pose (PDB, chains A,B,C target + D binder). Raises if any pose fails; ~2 s per pose. NOT for a shared login node (use a CPU job).
    workers=1 runs inline (no process pool)."""
    jobs = [(r, str(pose_dir), str(workdir), binder_chain, tuple(target_chains)) for r in manifest.to_dict("records")]
    if workers <= 1:
        out = pd.DataFrame([_score_one(j) for j in jobs])
    else:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(workers) as ex: out = pd.DataFrame(list(ex.map(_score_one, jobs)))
    bad = out[out.k_status != "ok"]
    if len(bad): raise RuntimeError(f"{len(bad)} poses failed to score (first: {bad.iloc[0].to_dict()})")
    return out


if __name__ == "__main__":
    man, pdir, pre = sys.argv[1:4]
    pp = score_poses(pd.read_csv(man), pdir, workdir=pre + "_work"); pp.to_csv(pre + "_per_pose.csv", index=False)
    r = summarise(pp); r["parents"].to_csv(pre + "_parents.csv"); r["variants"].to_csv(pre + "_variants.csv")
    print(r["parents"].round(2).to_string()); print(r["variants"].sort_values("d_mean", ascending=False).round(2).to_string() if len(r["variants"]) else "(no variants)")
