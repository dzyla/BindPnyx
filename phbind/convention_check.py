"""Which ipSAE direction was your gate calibrated on?  Re-score a reference set whose gate numbers you trust, in ONE batch at several seeds, and compare
the min- and max-direction grouped ipSAE with those reference numbers.

Why this exists: a published gate (mean >= 0.65, worst >= 0.50, seed-unanimous) was written as 'use max', but its reference table reproduces with
the MIN direction (Spearman 0.93, same pass count) and not with max (bias +0.20, passes 14/20 instead of 5/20). Thresholds are only valid in the
convention they were fitted in. This prints the evidence; it decides nothing.

Input CSV ($PHBIND_REFERENCE): columns id,seq,ref_mean[,ref_worst]. A shuffled-sequence negative control is added automatically.
  PYTHONPATH=$(pwd) python phbind/convention_check.py 101,202,303 out/phbind/convention
"""
import os, random, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from phbind import boltz_trimer as B

def main(seeds, out):
    ref = pd.read_csv(os.environ["PHBIND_REFERENCE"]); assert {"id", "seq", "ref_mean"} <= set(ref.columns) and len(ref) >= 10
    s = list(ref.seq.iloc[0][:62]); random.Random(7).shuffle(s)
    df = pd.concat([ref[["id", "seq"]], pd.DataFrame({"id": ["negctl62"], "seq": ["".join(s)]})], ignore_index=True)
    res = B.run(df, Path(out), seeds); r = res[res.id != "negctl62"]; g = r.groupby("id")
    t = pd.DataFrame({k: g[f"ipsae_{k}"].mean() for k in ("min", "max")}).join(ref.set_index("id").ref_mean)
    from scipy.stats import spearmanr
    for k in ("min", "max"):
        d = t[k] - t.ref_mean; print(f"{k}: Spearman vs reference {spearmanr(t[k], t.ref_mean)[0]:.3f}  median|d| {d.abs().median():.3f}  bias {d.mean():+.3f}")
    n = res[res.id == "negctl62"]; print("negative control (shuffled 62-mer) per seed, min/max/global iptm:", n[["ipsae_min", "ipsae_max", "iptm_global"]].round(3).values.tolist())

if __name__ == "__main__":
    main([int(x) for x in sys.argv[1].split(",")], sys.argv[2])
