"""Validate a scorer ON THIS TARGET before it is allowed to screen or gate. Positive controls + a reference set, one batch, one decision rule written down.

Inputs (data, never committed): a table with columns id, seq, group where group is
  real_binder     experimentally solved binders of the target (the one thing that is not another predictor's opinion)
  reference_pass  designs a trusted instrument calls passing    reference_fail  designs it calls failing
The scorer is run once on all rows (one batch). Metrics at gate 0.5 on `<prefix>_ipsae_min`:
  real_recall, reference_recall, reference_fp_rate  (+ Spearman vs the reference instrument when `ref_score` is given).
Decision (a heuristic written down so it can be argued with, not a statistic):
  SCREEN-capable : reference_recall >= 0.8                                     (a pre-filter must not lose what the reference keeps)
  GATE-capable   : (reference_recall > 0 or real_recall > 0) and reference_fp_rate <= 1/3     (some signal, and selective)
  A scorer with real_recall == 0 AND reference_recall == 0 is unusable on this target (e.g. OpenFold3 here: it uses the MSA but gives exactly 0.0 to every real binder).
Caveat printed with every record: reference labels from one instrument favour models like it; n is small; real_recall is 'weak' below 0.5 for EVERY model tried here.
Usage:  PYTHONPATH=<repo> python phbind/validate.py <scorer> <set.csv> <out_dir> [--write]
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent))

def evaluate(df: pd.DataFrame, col: str, gate: float = 0.5) -> dict:
    g = {k: df[df.group == k] for k in ("real_binder", "reference_pass", "reference_fail")}
    for k, v in g.items():
        if not len(v): raise ValueError(f"validation set has no '{k}' rows")
    rr, fr = float((g["real_binder"][col] >= gate).mean()), float((g["reference_pass"][col] >= gate).mean())
    fp = float((g["reference_fail"][col] >= gate).mean())
    m = dict(n_real=len(g["real_binder"]), real_recall=rr, n_ref_pass=len(g["reference_pass"]), reference_recall=fr, n_ref_fail=len(g["reference_fail"]), reference_fp_rate=fp)
    if "ref_score" in df and df.ref_score.notna().all() and len(df) > 5:
        from scipy.stats import spearmanr
        m["spearman_vs_reference"] = float(spearmanr(df[col], df.ref_score)[0])
    return m

def decide(m: dict) -> dict:
    unusable = m["real_recall"] == 0 and m["reference_recall"] == 0
    screen = (not unusable) and m["reference_recall"] >= 0.8
    gate = (not unusable) and (m["reference_recall"] > 0 or m["real_recall"] > 0) and m["reference_fp_rate"] <= 1 / 3
    why = "no signal: exactly zero on every real binder and every reference pass" if unusable else (
        "" if (screen or gate) else "neither sensitive enough to screen nor selective enough to gate")
    return dict(roles=dict(screen=bool(screen), gate=bool(gate)), why=why, weak=bool(m["real_recall"] < 0.5))

def main(argv):
    from phbind import scorers
    name, csv, out = argv[0], argv[1], argv[2]
    sc = scorers.SCORERS[name]; df = pd.read_csv(csv); assert {"id", "seq", "group"} <= set(df.columns)
    res = scorers.check_contract(sc.run(df[["id", "seq"]], Path(out)), sc).merge(df, on="id")
    m = evaluate(res, f"{sc.prefix}_ipsae_min"); rec = dict(status="validated", measured=m, **decide(m),
        caveat="reference labels favour models like the instrument that made them; n is small; real_recall < 0.5 means a miss is uninformative")
    print(json.dumps(rec, indent=1))
    if "--write" in argv:
        p = HERE / "validation.json"; allr = json.load(open(p)); allr[name] = {**allr.get(name, {}), **rec}; json.dump(allr, open(p, "w"), indent=1); print("wrote", p)
    return rec

if __name__ == "__main__":
    main(sys.argv[1:])
