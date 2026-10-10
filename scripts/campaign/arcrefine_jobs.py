#!/usr/bin/env python3
"""Build ArcRefine job folders (one config.json per design) from a judged table.

  python scripts/campaign/arcrefine_jobs.py --judged out/judge/X/judged.csv --target NAME --jobs-root DIR --n 12

Picks the top --n designs that pass the gate and are on-target (by consensus) and writes DIR/<id>/config.json {name, parent_sequence, targets:[{id,sequence}]}
plus DIR/parents.csv. No template_pdb on purpose (a residue mismatch aborts the run). Run the jobs with scripts/campaign/arcrefine_worker.sh.
Read this before spending GPU time: on the two targets tried, ArcRefine's own confidence did not say whether a refinement worked, and on the second
target refinement of already-passing designs cost passes (docs/RECOMMENDATIONS.md section 20). Treat it as a lottery ticket per design: judge the refined
sequence AND its parent in one batch, on a second oracle family, and keep the parent unless the refined one wins.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "funnel"))


def pick(df, n, min_hotspot=0.6):
    return df[df.consensus_pass.astype(bool) & (df.hotspot_frac >= min_hotspot)].sort_values("consensus", ascending=False).drop_duplicates("seq").head(n)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--judged", required=True); ap.add_argument("--target", required=True); ap.add_argument("--jobs-root", required=True)
    ap.add_argument("--n", type=int, default=12); ap.add_argument("--min-hotspot", type=float, default=0.6)
    a = ap.parse_args(argv)
    import common
    seq = common.load_target(a.target)["seq"]
    p = pick(pd.read_csv(a.judged), a.n, a.min_hotspot)
    root = Path(a.jobs_root); root.mkdir(parents=True, exist_ok=True)
    for r in p.itertuples():
        d = root / f"{a.target}_{r.id}"; d.mkdir(exist_ok=True)
        (d / "config.json").write_text(json.dumps({"name": d.name, "parent_sequence": r.seq, "targets": [{"id": "A", "sequence": seq}]}, indent=1))
    p[["id", "consensus", "hotspot_frac", "seq"]].to_csv(root / "parents.csv", index=False)
    print(f"built {len(p)} jobs in {root} (target {a.target}, {len(seq)} aa)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
