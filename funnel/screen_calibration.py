"""How much does the fast screen (Protenix-0.5-mini) enrich for what Boltz-2 / Protenix-v2 accept? UNBIASED: a random sample of the screened designs is
folded with both expensive models, so enrichment is measured on the whole range of fast scores (not only its top slice).
usage: screen_calibration.py TARGET [N=150]   -> out/calib/<target>/calib.csv"""
import sys, json
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent)); import common, run_funnel
tgt, N = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 150
t = common.load_target(tgt); REPO = common.REPO; out = REPO / "out" / "calib" / tgt; out.mkdir(parents=True, exist_ok=True)
s = pd.read_csv(REPO / "out/funnel" / tgt / "screen.csv").dropna(subset=["fast_ipsae"]).sample(N, random_state=7).reset_index(drop=True)
T = run_funnel.Timers(out / "timers.json")
d = run_funnel.consensus(t, s[["id", "seq"]], out, (1,), T, "calib")        # Boltz-2 seed 1 + Protenix-v2 seed 1 + hotspot contact (+ PISA/SC columns)
d = d.merge(s.drop(columns=["seq"]), on="id", how="left"); d.to_csv(out / "calib.csv", index=False)
print(tgt, len(d), "designs; Boltz gate", int(d.b_gate.sum()), "v2 pass", int(d.v2_pass.sum()), "both", int(d.consensus_pass.sum()))
