"""Independent judge applied identically to every arm's shortlist.

  judge.py --target pdl1 --out out/judge/pdl1 --arm default=out/baseline/pdl1_default/design_outputs/*/summary.csv \
           --arm scaled=... --arm funnel_nocycle=out/funnel/pdl1/final_nocycle.csv --arm funnel=out/funnel/pdl1/final_cycled.csv

Designs from all arms are pooled (identical sequences folded once), folded with Boltz-2 on FRESH seeds (101,102,103)
and a second oracle (seed 101; --second-oracle protenix-v2 (default), af3 or of3), and scored with the same gate: Boltz-2 mean ipSAE >= 0.5
and interface PAE <= 2 A AND second-oracle ipSAE >= 0.5 ("consensus pass"), plus hotspot contact in the Boltz-2 complex. Whatever the model,
its columns are called v2_* (downstream files depend on that); `o2_name` in arm_summary.csv records which one ran."""
import argparse, glob, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent)); import common, oracles, run_funnel

def read_arm(path):
    f = sorted(glob.glob(path))[0]; d = pd.read_csv(f)
    if "sequence" in d.columns and "seq" not in d.columns: d = d.rename(columns={"sequence": "seq"})   # pipeline summary.csv (ranked)
    return d[["seq"]].drop_duplicates().reset_index(drop=True)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--target", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--arm", action="append", required=True, help="name=csv (glob ok); first --top rows are judged")
    ap.add_argument("--top", type=int, default=20); ap.add_argument("--gpus", default=None)
    ap.add_argument("--second-oracle", default="protenix-v2", choices=oracles.ORACLES, help="the model that gates alongside Boltz-2 (use the one the run was selected with)")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); t = common.load_target(a.target); T = run_funnel.Timers(out / "timers.json")
    members, pool = {}, {}
    for spec in a.arm:
        name, path = spec.split("=", 1); seqs = read_arm(path).seq.head(a.top).tolist(); members[name] = seqs
        for s in seqs: pool.setdefault(s, []).append(name)
    df = pd.DataFrame(dict(id=[f"j{i:03d}" for i in range(len(pool))], seq=list(pool), arms=[";".join(v) for v in pool.values()]))
    print(f"judging {len(df)} unique designs from {len(members)} arms: " + ", ".join(f"{k}={len(v)}" for k, v in members.items()), flush=True)
    d = run_funnel.consensus(t, df, out, (101, 102, 103), T, "judge", common.gpu_list(a.gpus), oracle2=a.second_oracle); d.to_csv(out / "judged.csv", index=False)
    rows = []
    for name, seqs in members.items():
        g = d[d.seq.isin(seqs)]
        row = dict(arm=name, n=len(g), o2_name=a.second_oracle, consensus_pass=int(g.consensus_pass.sum()), boltz_gate=int(g.b_gate.sum()), v2_pass=int(g.v2_pass.sum()),
                   median_boltz_ipsae=round(g.b_ipsae.median(), 3), median_v2_ipsae=round(g.v2_ipsae.median(), 3),
                   hotspot_contact_ge_60pct=int((g.hotspot_frac >= 0.6).sum()), clusters=common.cluster_count(g.seq.tolist()))
        if "site_near" in g: row.update(site_occupied=int((g.site_near >= 1).sum()), site_clash=int((g.site_clash >= 1).sum()), median_site_min_dist=round(g.site_min_dist.median(), 2))
        if "pisa_flags" in g: row.update(flagged=int((g.pisa_flags.fillna("") != "").sum()), median_sc=round(g.pisa_sc.median(), 3), median_aromatic_contacts=float(g.pisa_n_aromatic_iface.median()))
        rows.append(row)
    r = pd.DataFrame(rows); r.to_csv(out / "arm_summary.csv", index=False); print(r.to_string(index=False))
if __name__ == "__main__": main()
