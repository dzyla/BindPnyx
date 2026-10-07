"""Where is the campaign? Counts of REAL artifacts (never exit codes, never 'directory exists'), what is using the GPU, and what an agent should do next.

  PYTHONPATH=<repo> python phbind/status.py [--json]
"""
from __future__ import annotations
import glob, json, subprocess, sys
from pathlib import Path
import pandas as pd
REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "out" / "phbind"

def gpu_jobs() -> list[str]:
    ps = subprocess.run(["ps", "-eo", "pid,etime,args"], capture_output=True, text=True).stdout.splitlines()
    keys = ("boltz predict", "protenix pred", "run_alphafold", "run_openfold", "s2_prescreen", "s1_queue", "s3_gate", "pxdesign.runner.inference")
    return [l.strip()[:110] for l in ps if any(k in l for k in keys) and "status.py" not in l and "grep" not in l]

def report(out: Path = OUT, cut: float = 0.35) -> dict:
    r: dict = {"gpu_jobs": gpu_jobs()}
    bb = glob.glob(str(out / "gen" / "*" / "pdbs" / "*.pdb")); r["backbones_generated"] = len(bb)
    da = out / "designs_all.csv"; r["designs_total"] = len(pd.read_csv(da)) if da.exists() else 0
    bs = sorted((out / "s2").glob("batch_*.csv")) if (out / "s2").exists() else []
    if bs:
        a = pd.concat([pd.read_csv(f) for f in bs], ignore_index=True); car = a[a.id.str.startswith("carrier_")]; a = a[~a.id.str.startswith("carrier_")]
        r.update(prescreen_batches=len(bs), designs_screened=len(a), survivors_at_cut=int((a.ipsae_min >= cut).sum()), boltz_ge_0_5=int((a.ipsae_min >= 0.5).sum()),
                 last_carriers={k: round(float(v), 3) for k, v in car[car.batch == car.batch.max()].set_index("id").ipsae_min.items()})
    else: r.update(prescreen_batches=0, designs_screened=0)
    gates = sorted(out.glob("gate*/run*/gate.csv")) + sorted(out.glob("gate*/gate.csv"))
    r["gate_tables"] = {str(g.relative_to(out)): dict(designs=len(pd.read_csv(g)), passed=int(pd.read_csv(g).consensus_pass.sum())) for g in gates}
    nxt = []
    if not r["backbones_generated"]: nxt.append("generate backbones: phbind/s1_queue.py (needs phbind/s0_target.py first)")
    elif r["designs_screened"] < r["designs_total"] and not any("s2_prescreen" in j for j in r["gpu_jobs"]): nxt.append("resume the prescreen: phbind/run_after.sh 1 (one GPU job at a time)")
    if r.get("boltz_ge_0_5", 0) and not r["gate_tables"]: nxt.append("run the gate on designs with Boltz >= 0.5: phbind/s3_gate.py")
    r["next"] = nxt or ["nothing blocked; check the experiment log (phbind/experiments.py) before changing anything"]
    return r

if __name__ == "__main__":
    rep = report()
    print(json.dumps(rep, indent=1) if "--json" in sys.argv else "\n".join(f"{k}: {v}" for k, v in rep.items()))
