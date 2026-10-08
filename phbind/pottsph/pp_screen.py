"""Boltz-2 human 1-seed screen of one shard of designs (id,seq) + a failing and a passing carrier, in ONE batch. Runs on any of our machines.
env: PHBIND_REPO (repo checkout whose phbind/target was built), PHBIND_CARRIERS (carriers.csv), PXD_BOLTZ_BIN, optional PHBIND_PYLIB.
usage: pp_screen.py <shard.csv> <out_dir> [seeds comma list]"""
import os, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, os.environ["PHBIND_REPO"])
if os.environ.get("PHBIND_PYLIB"): sys.path.insert(0, os.environ["PHBIND_PYLIB"])
from phbind import boltz_trimer as B
shard, out = sys.argv[1], Path(sys.argv[2]); seeds = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "101").split(",")]
df = pd.read_csv(shard)[["id", "seq"]]; n = len(df)
car = pd.read_csv(os.environ["PHBIND_CARRIERS"]); df = pd.concat([df, car[car.id.isin(["carrier_fail_A91", "carrier_valid_D68"])][["id", "seq"]]], ignore_index=True)
assert df.id.is_unique, "duplicate ids in shard"
res = B.run(df, out, seeds); s = B.summarise(res); s.to_csv(out / "summary.csv", index=False)
assert len(s) == len(df), f"scored {len(s)} of {len(df)}"
print("SHARD DONE", shard, "designs", n, "scored", len(s), "pass>=0.5:", int(((s.ipsae_mean >= 0.5) & ~s.id.str.startswith("carrier")).sum()), flush=True)
