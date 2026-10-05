"""Score ProteinBase designs with Boltz-2 (target MSA reused, binder single-sequence)."""
import json, subprocess, sys, time, numpy as np, pandas as pd
from pathlib import Path
from metrics import ipsae

BOLTZ = Path("../.pxd/envs/boltz/bin/boltz").resolve()

def run(manifest="inputs/manifest.csv", out="out/boltz", targets=None, recycles=3, steps=200):
    m = pd.read_csv(manifest); out = Path(out); rows = []
    for t, g in m.groupby("target"):
        if targets and t not in targets: continue
        yd = out / t / "yaml"; yd.mkdir(parents=True, exist_ok=True)
        msa = Path(f"msa/{t}.a3m").resolve()
        for _, r in g.iterrows():
            (yd / f"{r['id']}.yaml").write_text(
                f"version: 1\nsequences:\n  - protein:\n      id: A\n      sequence: {r.target_seq}\n      msa: {msa}\n"
                f"  - protein:\n      id: B\n      sequence: {r.binder_seq}\n      msa: empty\n")
        t0 = time.time()
        p = subprocess.run([str(BOLTZ), "predict", str(yd), "--out_dir", str(out / t / "pred"),
            "--recycling_steps", str(recycles), "--sampling_steps", str(steps), "--diffusion_samples", "1",
            "--write_full_pae", "--accelerator", "gpu", "--override", "--no_kernels"], capture_output=True, text=True)
        dt = time.time() - t0
        print(t, "rc", p.returncode, f"{dt:.0f}s", flush=True)
        if p.returncode: print(p.stderr[-2000:])
        for _, r in g.iterrows():
            d = out / t / "pred" / f"boltz_results_yaml" / "predictions" / r["id"]
            cj = d / f"confidence_{r['id']}_model_0.json"; pn = d / f"pae_{r['id']}_model_0.npz"
            if not cj.exists(): rows.append(dict(target=t, id=r["id"])); continue
            c = json.load(open(cj)); pae = np.load(pn)["pae"]
            lo, hi = ipsae(pae, len(r.target_seq), len(r.binder_seq))
            rows.append(dict(target=t, id=r["id"], b_iptm=c["iptm"], b_ptm=c["ptm"], b_ciplddt=c.get("complex_iplddt"),
                             b_ipde=c.get("complex_ipde"), b_ipsae_min=lo, b_ipsae_max=hi, b_sec=dt / len(g)))
    pd.DataFrame(rows).to_csv(out / "scores.csv", index=False)

if __name__ == "__main__":
    run(targets=sys.argv[1].split(",") if len(sys.argv) > 1 else None)
