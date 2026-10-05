"""Score ProteinBase designs with a Protenix model (arms differ only by checkpoint/speed settings).
usage: score_protenix.py ARM [target,target]"""
import json, os, re, subprocess, sys, time, numpy as np, pandas as pd
from pathlib import Path
from metrics import ipsae

PTX = Path("../.pxd/envs/pxd/bin/protenix").resolve()
ARMS = {  # arm -> (model_name, pairformer cycles, diffusion steps)
    "ptx_v2": ("protenix-v2", 4, 200),
    "ptx_v1": ("protenix_base_default_v1.0.0", 4, 200),
    "ptx05_fast": ("protenix_mini_default_v0.5.0", 2, 5),
}

def a3m(path, seq):
    path.write_text(f">query\n{seq}\n"); return str(path.resolve())

def run(arm, targets=None, manifest="inputs/manifest.csv", out=None):
    model, cyc, steps = ARMS[arm]
    m = pd.read_csv(manifest); out = Path(out) if out else Path("out") / arm; rows = []
    for t, g in m.groupby("target"):
        if targets and t not in targets: continue
        d = out / t; (d / "msa").mkdir(parents=True, exist_ok=True)
        tq = a3m(d / "msa" / "target_query.a3m", g.target_seq.iloc[0])
        tmsa = str(Path(f"msa/{t}.a3m").resolve())
        jobs = []
        for _, r in g.iterrows():
            bq = a3m(d / "msa" / f"{r['id']}.a3m", r.binder_seq)
            jobs.append({"name": r["id"], "covalent_bonds": [], "sequences": [
                {"proteinChain": {"sequence": r.target_seq, "count": 1, "unpairedMsaPath": tmsa, "pairedMsaPath": tq}},
                {"proteinChain": {"sequence": r.binder_seq, "count": 1, "unpairedMsaPath": bq, "pairedMsaPath": bq}}]})
        json.dump(jobs, open(d / "in.json", "w"))
        t0 = time.time()
        p = subprocess.run([str(PTX), "pred", "-i", str(d / "in.json"), "-o", str(d / "pred"), "-s", "101", "-n", model,
                            "-e", "1", "-c", str(cyc), "-p", str(steps), "--use_msa", "true", "--need_atom_confidence", "true"],
                           capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(Path("..").resolve())})
        wall = time.time() - t0
        (d / "log.txt").write_text(p.stdout + p.stderr)
        fwd = [float(x) / len(g) for x in re.findall(r"Seed \d+ completed in ([\d.]+)s", p.stdout + p.stderr)]  # per design
        print(arm, t, "rc", p.returncode, f"wall {wall:.0f}s infer/design {np.mean(fwd) if fwd else float('nan'):.1f}s", flush=True)
        for _, r in g.iterrows():
            pr = d / "pred" / r["id"] / "seed_101" / "predictions"
            sj, fj = pr / f"{r['id']}_summary_confidence_sample_0.json", pr / f"{r['id']}_full_data_sample_0.json"
            if not (sj.exists() and fj.exists()): rows.append(dict(target=t, id=r["id"])); continue
            s = json.load(open(sj)); pae = np.array(json.load(open(fj))["token_pair_pae"])
            lo, hi = ipsae(pae, len(r.target_seq), len(r.binder_seq))
            rows.append({"target": t, "id": r["id"], f"{arm}_iptm": s["iptm"], f"{arm}_rank": s["ranking_score"],
                         f"{arm}_ptm": s["ptm"], f"{arm}_ipsae_min": lo, f"{arm}_ipsae_max": hi,
                         f"{arm}_sec": np.mean(fwd) if fwd else np.nan})
    pd.DataFrame(rows).to_csv(out / ("scores_%s.csv" % ("_".join(targets) if targets else "all")), index=False)

if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2].split(",") if len(sys.argv) > 2 else None)
