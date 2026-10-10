#!/usr/bin/env python3
"""Select a panel from judged designs by a rule you write down BEFORE looking at results.

  python scripts/campaign/select_panel.py --judged out/judge/X/judged.csv [--judged ...] --top 10 --out panel.csv
        [--min-hotspot 0.6] [--identity 0.6] [--rank-flags thin_interface,few_hbonds,low_shape_compl] [--exclude-arm controls]

The rule (edit it only before the first result is seen, and keep a copy of it next to the panel):
  1. eligible    consensus_pass (the gate: Boltz-2 mean ipSAE >= 0.5, interface PAE <= 2 A, second oracle >= 0.5) AND hotspot_frac >= --min-hotspot
  2. tier A      both gate oracles >= 0.7, interface PAE <= 1.0 A, hotspot_frac >= 0.8, and none of --rank-flags raised; otherwise tier B.
                 Flags NOT listed in --rank-flags are reported but never ranked on. Check, on the target's NATIVE interface, which flags it
                 trips itself: a natural high-affinity interface can fail the generic ones, and ranking on them would penalise the biology.
  3. diversity   at most one design per --identity cluster (position-wise identity); the best-ranked member represents the cluster
  4. rank        tier, then consensus = min(Boltz-2 ipSAE, second-oracle ipSAE); take the top N.
Designs from every --judged table are pooled by sequence; use ONE judge batch if the ranking is to mean anything (scores from different batches are
not comparable; SD ~0.02). Controls are never members: pass --exclude-arm for the arm that holds them. Columns that other oracles add (for example a
third model family) are carried through untouched as report columns, never gates.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "funnel"))
import common  # noqa: E402

DEFAULT_RANK_FLAGS = ("thin_interface", "few_hbonds", "low_shape_compl")


def tier(row, rank_flags=DEFAULT_RANK_FLAGS):
    flags = str(row["pisa_flags"]).split(";") if isinstance(row.get("pisa_flags"), str) else []
    raised = any(f in rank_flags for f in flags)
    ok = (row["b_ipsae"] >= .7 and row["v2_ipsae"] >= .7 and row["b_paemin"] <= 1.0 and row["hotspot_frac"] >= .8 and not raised)
    return "A" if ok else "B"


def select(df, top=10, min_hotspot=0.6, identity=0.6, rank_flags=DEFAULT_RANK_FLAGS):
    """Return (panel, eligible) DataFrames. `eligible` has tier, cluster and keep columns for audit."""
    e = df[df.consensus_pass.astype(bool) & (df.hotspot_frac >= min_hotspot)].copy()
    e["tier"] = [tier(r, rank_flags) for r in e.to_dict("records")]
    e = e.sort_values(["tier", "consensus"], ascending=[True, False]).reset_index(drop=True)
    reps, cluster, keep = [], [], []
    for s in e.seq:
        k = next((i for i, r in enumerate(reps) if common.identity(s, r) >= identity), None)
        if k is None:
            reps.append(s); k = len(reps) - 1; keep.append(True)
        else:
            keep.append(False)
        cluster.append(k)
    e["cluster"], e["keep"] = cluster, keep
    panel = e[e.keep].head(top).copy()
    panel.insert(0, "rank", range(1, len(panel) + 1))
    return panel, e


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--judged", action="append", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--top", type=int, default=10); ap.add_argument("--min-hotspot", type=float, default=0.6)
    ap.add_argument("--identity", type=float, default=0.6); ap.add_argument("--rank-flags", default=",".join(DEFAULT_RANK_FLAGS))
    ap.add_argument("--exclude-arm", action="append", default=["controls"])
    a = ap.parse_args(argv)
    d = pd.concat([pd.read_csv(p) for p in a.judged], ignore_index=True).drop_duplicates("seq")
    if "arms" in d:
        d = d[~d.arms.fillna("").map(lambda s: any(x in a.exclude_arm for x in s.split(";")))]
    panel, elig = select(d, a.top, a.min_hotspot, a.identity, tuple(x for x in a.rank_flags.split(",") if x))
    panel.to_csv(a.out, index=False); elig.to_csv(str(a.out).replace(".csv", "") + "_eligible.csv", index=False)
    print(f"judged {len(d)}; eligible {len(elig)} (tier A {int((elig.tier == 'A').sum())}, B {int((elig.tier == 'B').sum())}); "
          f"clusters {elig.cluster.nunique()}; panel {len(panel)} -> {a.out}")
    cols = [c for c in ("rank", "id", "arms", "tier", "b_ipsae", "v2_ipsae", "consensus", "hotspot_frac", "pisa_flags") if c in panel]
    print(panel[cols].round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
