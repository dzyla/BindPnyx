#!/usr/bin/env python3
"""Build AlphaFold 3 fold-input JSONs (one per design, target chain A + binder chain B) to run on a machine that has the AF3 weights.

  python scripts/campaign/make_af3_inputs.py --target NAME --designs designs.csv --out DIR --remote-msa /path/on/the/cluster/target_msa.a3m [--seed 1]

designs.csv has columns id,seq. Writes DIR/json/<name>.json, DIR/<target>_target_msa.a3m (sanitised: no NUL/non-ASCII bytes, no ragged rows - a single NUL byte
in an alignment once cost a whole campaign) and DIR/name_map.csv. Upload the .a3m to --remote-msa, the json folder, and run AF3 with
`--norun_data_pipeline --num_diffusion_samples=5 --input_dir=...`. The binder gets a single-sequence MSA; the target MSA is injected, so no database search runs.
Score the outputs back with funnel/oracles.py:parse_af3_result (mean ipSAE over the samples, minimum direction). AF3 weights are licensed: never commit them.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "funnel"))
import common  # noqa: E402
import oracles  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--target", required=True); ap.add_argument("--designs", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--remote-msa", required=True, help="where the sanitised alignment will live ON THE CLUSTER (written into every JSON)")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args(argv)
    t = common.load_target(a.target)
    out = Path(a.out); (out / "json").mkdir(parents=True, exist_ok=True)
    msa = out / f"{a.target}_target_msa.a3m"
    stats = oracles.sanitize_a3m(t["msa"], msa, t["seq"])
    print(f"alignment: {stats} -> {msa}")
    d = pd.read_csv(a.designs)
    assert {"id", "seq"} <= set(d.columns), "designs.csv needs columns id,seq"
    names = {r.id: oracles.af3_name(r.id) for r in d.itertuples()}
    assert len(set(names.values())) == len(names), "AF3 job names collide (it lower-cases and strips ids)"
    for r in d.itertuples():
        (out / "json" / f"{names[r.id]}.json").write_text(json.dumps(oracles.af3_input(names[r.id], t["seq"], a.remote_msa, r.seq, a.seed), indent=1))
    (out / "name_map.csv").write_text("id,af3_name,binder_len\n" + "".join(f"{r.id},{names[r.id]},{len(r.seq)}\n" for r in d.itertuples()))
    print(f"wrote {len(d)} inputs to {out / 'json'}; target {a.target} {len(t['seq'])} aa; binder lengths {sorted({len(s) for s in d.seq})}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
