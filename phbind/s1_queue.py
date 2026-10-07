"""Resumable S1 queue: for every (hotspot set, length) run -> PXDesign backbones -> SolubleMPNN -> out/phbind/gen/<run>/designs.csv.
A run is finished only when its designs.csv exists with the expected row count (counted artifact, never an exit code or a directory).
One GPU job at a time. Re-run the same command to resume. Then concatenates all finished runs into out/phbind/designs_all.csv."""
import json, os, subprocess, sys, time
from pathlib import Path
import pandas as pd
REPO = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "phbind")); sys.path.insert(0, str(REPO / "funnel"))
import s1_generate as S, common
from phbind import config as _cfg
_C = _cfg.load()["generate"]                       # PHBIND_CONFIG=... overrides; defaults are the settings the campaign ran with
S.HOTSPOT_SETS.update(_C["hotspot_sets"])
SETS, LENGTHS = _C["sets"], _C["lengths"]
N_PER, SEQS = int(os.environ.get("S1_N", _C["backbones_per_run"])), _C["seqs_per_backbone"]
GEN = REPO / "out/phbind/gen"

def run_id(s, L): return f"{s}_L{L}"
def seed_of(s, L): return 100000 + 1000 * SETS.index(s) + L      # unique per (set, length)

def one(s, L):
    rid = run_id(s, L); d = GEN / rid; csv = d / "designs.csv"
    if csv.exists() and len(pd.read_csv(csv)) == N_PER * SEQS: return
    if not (d / "done.json").exists(): S.generate(s, L, N_PER, seed_of(s, L), d)
    meta = json.load(open(d / "done.json")); t0 = time.time()
    p = subprocess.run([common.PXD_PY(), str(REPO / "phbind/mpnn_dimer.py"), str(d / "pdbs"), str(SEQS), str(d / "mpnn.csv"), ",".join(meta["cond"]), meta["bind"][0]],
                       env={**common.gpu_env(), "PYTHONPATH": str(REPO), "XLA_PYTHON_CLIENT_PREALLOCATE": "false"}, capture_output=True, text=True)
    (d / "mpnn.log").write_text(p.stdout[-5000:] + p.stderr[-5000:])
    m = pd.read_csv(d / "mpnn.csv")          # raises if MPNN wrote nothing
    m["bb"] = m.name; m["id"] = [f"{rid}_{b.rsplit('_', 1)[1]}_{k}" for b, k in zip(m.name, m.seq_idx)]
    m = m.rename(columns={"sequence": "seq"}); m["set"], m["L"], m["run"] = s, L, rid
    m[["id", "bb", "seq", "set", "L", "run"]].to_csv(csv, index=False); print(f"{rid}: mpnn {len(m)} seqs {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    for s in SETS:
        for L in LENGTHS: one(s, L)
    mine = [pd.read_csv(GEN / run_id(s, L) / "designs.csv") for s in SETS for L in LENGTHS]; assert sum(map(len, mine)) == len(SETS) * len(LENGTHS) * N_PER * SEQS
    parts = [pd.read_csv(f) for f in sorted(GEN.glob("*/designs.csv"))]                  # designs_all.csv = EVERY finished run on disk, so a new campaign never drops an earlier one
    a = pd.concat(parts, ignore_index=True); assert a.id.is_unique
    a.to_csv(REPO / "out/phbind/designs_all.csv", index=False); print("designs_all.csv", len(a), "sequences,", a.bb.nunique(), "backbone ids,", a.seq.nunique(), "distinct sequences")
