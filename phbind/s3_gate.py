"""S3: the gate on the intact trimer. Boltz-2 (N seeds, one batch per seed) + OpenFold3 (1 seed) on survivors of the prescreen, then:
  PASS/FAIL  : seed-unanimous Boltz-2 (min-direction ipSAE) AND mean >= 0.65 AND worst >= 0.50 AND OpenFold3 ipSAE >= o2_gate
  ORDER      : mean of the two oracles' ipSAE (funnel/oracles.consensus_score). docs/JUDGE_REGIME.md: on 1,320 labelled designs min is right for a gate and
               poor for ranking; mean ranks better for the same pair (within-target AUROC +0.025 [+0.011, +0.041]). The pair is fixed, never re-chosen per target.
Input CSV: id,seq. Output: <out>/gate.csv. Needs a GPU only for the two folding stages; `combine` is pure and tested."""
import sys
from pathlib import Path
import pandas as pd
HERE = Path(__file__).resolve().parent; sys.path[:0] = [str(HERE.parent), str(HERE.parent / "funnel")]
import oracles  # noqa: E402  (funnel/oracles.py: consensus_score, shared column conventions)

def combine(boltz_summary: pd.DataFrame, of3: pd.DataFrame, o2_gate=0.5) -> pd.DataFrame:
    """boltz_summary: boltz_trimer.summarise output. of3: of3_trimer.run output (of3_ipsae_min/max). Rows missing either oracle get NaN ranks, never a one-oracle score."""
    t = boltz_summary.merge(of3[["id", "of3_ipsae_min", "of3_ipsae_max"]], on="id", how="left")
    t["b_ipsae"], t["v2_ipsae"] = t["ipsae_mean"], t["of3_ipsae_min"]                      # shared column names so oracles.consensus_score applies unchanged
    t["o2_pass"] = t["v2_ipsae"] >= o2_gate
    t["consensus_pass"] = t["gate_pass"] & t["o2_pass"]
    t["consensus_min"] = oracles.consensus_score(t, ("b_ipsae", "v2_ipsae"), "min")
    t["consensus"] = oracles.consensus_score(t, ("b_ipsae", "v2_ipsae"), "mean")           # what the shortlist is ordered by
    return t.sort_values(["consensus_pass", "consensus"], ascending=[False, False], na_position="last").reset_index(drop=True)

def main(csv, out, seeds=(101, 202, 303, 404, 505), of3_seed=101):
    from phbind import boltz_trimer as B, of3_trimer as O
    df = pd.read_csv(csv)[["id", "seq"]]; out = Path(out); out.mkdir(parents=True, exist_ok=True)
    s = B.summarise(B.run(df, out / "boltz", list(seeds)))
    o = O.run(df, out / "of3", of3_seed)
    t = combine(s, o); t.to_csv(out / "gate.csv", index=False)
    print(f"{len(t)} designs; Boltz seed-unanimous gate {int(t.gate_pass.sum())}; + OpenFold3 {int(t.consensus_pass.sum())}")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
