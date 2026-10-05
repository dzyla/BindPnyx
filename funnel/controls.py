"""Negative controls for a shortlist: is a high score specific to the target and the sequence, or does the predictor give it away for free?

  python funnel/controls.py --target fimh --designs out/funnel/fimh/final_cycled.csv --out out/controls/fimh --decoys pdl1,mdm2,fima --shuffles 2

For every design: Boltz-2 ipSAE (seed 1) against each DECOY target (a real, specific binder should not score high there) and for composition-preserving SHUFFLES of its
sequence against the true target (should collapse; the repository's earlier oracle test saw AUC 1.0 for designs vs shuffles on one target). Reports the specificity margin
= on-target ipSAE - max(decoy, shuffle). Resumable (folds already on disk are reused)."""
import argparse, json, random, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent)); import common, run_funnel

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--target", required=True); ap.add_argument("--designs", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--decoys", default="pdl1,mdm2,fima"); ap.add_argument("--shuffles", type=int, default=2); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--top", type=int, default=20)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    t = common.load_target(a.target); d = pd.read_csv(a.designs).head(a.top)[["id", "seq", "b_ipsae", "v2_ipsae"]].copy(); rng = random.Random(a.seed)
    res = d[["id", "seq", "b_ipsae", "v2_ipsae"]].rename(columns={"b_ipsae": "on_target_ipsae", "v2_ipsae": "on_target_v2"})
    for dec in [x for x in a.decoys.split(",") if x and x != a.target]:
        td = common.load_target(dec); f = common.boltz_fold(d[["id", "seq"]], td, out / f"decoy_{dec}", seeds=(1,)).set_index("id")
        res[f"decoy_{dec}"] = res.id.map(f.b_ipsae); print(f"decoy {dec}: median ipSAE {res[f'decoy_{dec}'].median():.3f}", flush=True)
    sh = []
    for r in d.itertuples():
        for k in range(a.shuffles):
            s = list(r.seq); rng.shuffle(s); sh.append(dict(id=f"{r.id}_sh{k}", base=r.id, seq="".join(s)))
    S = pd.DataFrame(sh); f = common.boltz_fold(S[["id", "seq"]], t, out / "shuffles", seeds=(1,)).set_index("id"); S["ipsae"] = S.id.map(f.b_ipsae)
    res["shuffle_max"] = res.id.map(S.groupby("base").ipsae.max()); res["shuffle_mean"] = res.id.map(S.groupby("base").ipsae.mean())
    dc = [c for c in res if c.startswith("decoy_")]; res["decoy_max"] = res[dc].max(axis=1) if dc else np.nan
    res["margin"] = res.on_target_ipsae - res[["decoy_max", "shuffle_max"]].max(axis=1)
    run_funnel.save_csv(res, out / "controls.csv")
    summ = dict(n=len(res), on_target_median=float(res.on_target_ipsae.median()), shuffle_max_median=float(res.shuffle_max.median()), decoy_max_median=float(res.decoy_max.median()) if dc else None,
                margin_median=float(res.margin.median()), margin_ge_0_3=int((res.margin >= 0.3).sum()), margin_ge_0_5=int((res.margin >= 0.5).sum()), decoys=a.decoys, shuffles_per_design=a.shuffles)
    json.dump(summ, open(out / "controls_summary.json", "w"), indent=1); print(json.dumps(summ, indent=1))

if __name__ == "__main__": main()
