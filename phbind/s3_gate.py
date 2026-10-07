"""S3: THE GATE on the intact trimer = TWO MODELS, ONE SEED EACH: Boltz-2 x1 + AlphaFold3 x1 (mean ipSAE over AF3's diffusion samples), consensus across models.

  PASS   : both models' grouped ipSAE (min direction, the direction the 0.5 gate was calibrated on) >= 0.5      -> tier B
  STRONG : both >= 0.65                                                                                       -> tier A
  ORDER  : mean of the two models' ipSAE (funnel/oracles.consensus_score); min is for the gate, mean ranks better for the same pair (docs/JUDGE_REGIME.md).
A model missing for a design gives NaN, never a one-model score. The pair is fixed. Seeds of one model are NOT used as the gate: they are highly correlated
(Spearman 0.77-0.93), while a second architecture adds independent information.

Measured on this target (20 reference designs whose labels are themselves Boltz-derived, so they favour Boltz-like models; 5 real PDB binders as controls; docs/HOMO_OLIGOMER_TARGETS.md):
  second model   agreement with Boltz-2 (Spearman)   recall of the 5 reference gate-passers / false positives (of 15), both >= 0.5
  AF3 (1 sample)        0.16                          1/5, 1/15        (5 samples: see the doc)
  Protenix-v2           0.22                          3/5, 0/15        (non-default alternative: --second ptx)
  OpenFold3            -0.16                          0/5, 1/15        (UNUSABLE here: exactly 0.0 interface confidence on all five real binders; it does use the MSA)
No model recovers most of the five experimentally known TNF binders. These are agreement checks between predictors, not evidence of binding.

Usage: s3_gate.py survivors.csv OUT_DIR [prescreen_dir|-] [af3|ptx|of3].  Input CSV: id,seq. The Boltz-2 leg is REUSED from the prescreen batches when the id was
scored there (seed 101, a batch with reference carriers); otherwise it is run. Needs $PXD_AF3_PYTHON, $PXD_AF3_DIR, $PXD_AF3_MODELS for AF3.
Output: <out>/gate.csv.  GPU only for the folding stages; `consensus` is pure and tested."""
import sys
from pathlib import Path
import pandas as pd
HERE = Path(__file__).resolve().parent; sys.path[:0] = [str(HERE.parent), str(HERE.parent / "funnel")]
import oracles  # noqa: E402  (funnel/oracles.py: consensus_score, shared column conventions)

GATE, STRONG = 0.5, 0.65

def consensus(boltz: pd.DataFrame, second: pd.DataFrame, prefix="af3", gate=GATE, strong=STRONG) -> pd.DataFrame:
    """boltz: id, ipsae_min (+ optional ipsae_max, p1x_recall, foot_recall, ...). second: af3_trimer.run (prefix 'af3', the gate), protenix_trimer.run ('ptx') or of3_trimer.run ('of3') output."""
    t = boltz.merge(second[[c for c in second.columns if c == "id" or c.startswith(prefix + "_")]], on="id", how="left")
    t["second_model"] = {"af3": "af3", "ptx": "protenix-v2", "of3": "of3"}[prefix]
    t["b_ipsae"], t["v2_ipsae"] = t["ipsae_min"], t[f"{prefix}_ipsae_min"]                       # shared column names so oracles.consensus_score applies unchanged
    t["consensus_pass"] = (t["b_ipsae"] >= gate) & (t["v2_ipsae"] >= gate)
    t["tier"] = ["A" if (b >= strong and o >= strong) else ("B" if p else "-") for b, o, p in zip(t.b_ipsae, t.v2_ipsae, t.consensus_pass)]
    t["consensus_min"] = oracles.consensus_score(t, ("b_ipsae", "v2_ipsae"), "min")
    t["consensus"] = oracles.consensus_score(t, ("b_ipsae", "v2_ipsae"), "mean")             # what the shortlist is ordered by
    t["model_gap"] = (t.b_ipsae - t.v2_ipsae).abs()                                          # large gap = the two models disagree: look before trusting either
    return t.sort_values(["consensus_pass", "consensus"], ascending=[False, False], na_position="last").reset_index(drop=True)

def boltz_leg(df, out, prescreen_dir=None, seed=101):
    """Boltz-2 seed-101 rows for df.id: from the prescreen batches where available, a fresh batch for the rest."""
    from phbind import boltz_trimer as B
    have = pd.DataFrame()
    if prescreen_dir and Path(prescreen_dir).exists():
        fs = sorted(Path(prescreen_dir).glob("batch_*.csv"))
        if fs: have = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True); have = have[have.id.isin(df.id)].drop_duplicates("id")
    need = df[~df.id.isin(have.id)] if len(have) else df
    if len(need):
        B.run_seed(need, Path(out) / "boltz", seed); fresh = B.score_seed(need, Path(out) / "boltz", seed); have = pd.concat([have, fresh], ignore_index=True)
    return have

def main(csv, out, prescreen_dir=None, second="af3", seed=1, n_samples=5, force=False):
    from phbind import scorers
    scorers.require("boltz", "gate", force=force); scorers.require(second, "gate", force=force)      # refuses a scorer whose validation record says it cannot gate
    df = pd.read_csv(csv)[["id", "seq"]]; out = Path(out); out.mkdir(parents=True, exist_ok=True)
    b = boltz_leg(df, out, prescreen_dir)
    if second == "af3":
        from phbind import af3_trimer as A; o = A.run(df, out / "af3", seed, n_samples=n_samples)
    elif second == "ptx":
        from phbind import protenix_trimer as P; o = P.run(df, out / "ptx", 101)
    else:
        from phbind import of3_trimer as O; o = O.run(df, out / "of3", 101)
    t = consensus(b, o, second); t.to_csv(out / "gate.csv", index=False)
    print(f"{len(t)} designs; both models >= {GATE}: {int(t.consensus_pass.sum())} (tier A, both >= {STRONG}: {int((t.tier == 'A').sum())})")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] != "-" else None, sys.argv[4] if len(sys.argv) > 4 else "af3")
