"""Pipeline/diffusion input JSON for a funnel target (hotspots already in the shard's numbering)."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); import common

def manifest(target, length=None, absolute=True):
    t = common.load_target(target) if isinstance(target, str) else target
    R = common.REPO
    return [{"name": f"{t['name']}_L{length or t['binder_length']}",
             "condition": {"structure_file": t["shard_abs"], "filter": {"chain_id": ["A"], "crop": {}},
                           "msa": {"A": {"precomputed_msa_dir": str((R / t["msa_dir"]).resolve()), "pairing_db": "uniref100"}}},
             "hotspot": {"A": t["hotspot_idx"]},
             "generation": [{"type": "protein", "length": length or t["binder_length"], "count": 1}]}]

if __name__ == "__main__":
    json.dump(manifest(sys.argv[1]), open(sys.argv[2], "w"), indent=1); print("wrote", sys.argv[2])
