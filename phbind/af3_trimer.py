"""AlphaFold3 as the SECOND model on the intact TNF trimer (the pair is Boltz-2 x1 + AF3 x1; see s3_gate.py).
Three copies as one protein entry (id A,B,C) with the sanitised `_nulfix` target MSA injected, binder (id D) with a query-only MSA and no templates,
`--norun_data_pipeline`. One process per design (the JAX cache makes later compiles cheap). ipSAE per diffusion sample, grouped (A+B+C vs D); reported as
the MEAN over samples (the convention the public benchmark finding used) and the best-ranked sample. Raises when a design produced no usable output.
Config (no paths hard-coded): $PXD_AF3_PYTHON, $PXD_AF3_DIR (has run_alphafold.py), $PXD_AF3_MODELS."""
from __future__ import annotations
import json, os, subprocess, sys, time
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "funnel"))
import trimer as T
from boltz_trimer import msa_human, epitope_metrics
from oracles import sanitize_a3m, af3_name, af3_config, af3_env

def af3_input(name, binder_seq, msa_path, seed):
    return {"name": name, "modelSeeds": [int(seed)], "dialect": "alphafold3", "version": 1, "sequences": [
        {"protein": {"id": ["A", "B", "C"], "sequence": T.TNF_HUMAN, "unpairedMsaPath": str(Path(msa_path).resolve()), "pairedMsa": "", "templates": []}},
        {"protein": {"id": "D", "sequence": binder_seq, "unpairedMsa": f">query\n{binder_seq}\n", "pairedMsa": "", "templates": []}}]}

def parse(design_dir: Path, name: str, binder_seq: str):
    lo, hi, ip, rk, best = [], [], [], [], None
    for cj in sorted(design_dir.glob("seed-*_sample-*/*_confidences.json")):
        if cj.name.endswith("summary_confidences.json"): continue
        sj = cj.with_name(cj.name.replace("_confidences.json", "_summary_confidences.json")); J = json.load(open(cj)); S = json.load(open(sj))
        tc = np.array(J["token_chain_ids"]); pae = np.asarray(J["pae"], float)
        ti, bi = np.where(tc != "D")[0], np.where(tc == "D")[0]
        assert len(ti) == 471 and len(bi) == len(binder_seq) and pae.shape == (len(tc),) * 2, f"{name}: AF3 tokens {len(ti)}+{len(bi)} (expected 471+{len(binder_seq)})"
        a, b = T.ipsae_grouped(pae, ti, bi); lo.append(a); hi.append(b); ip.append(float(S["iptm"])); rk.append(float(S["ranking_score"]))
        if best is None or rk[-1] > best[0]: best = (rk[-1], cj.parent)
    cif = design_dir / f"{name}_model.cif"
    if not lo or not cif.exists(): return None
    return dict(af3_ipsae_min=float(np.mean(lo)), af3_ipsae_max=float(np.mean(hi)), af3_ipsae_min_best=float(lo[int(np.argmax(rk))]), af3_iptm=float(np.mean(ip)), af3_n_samples=len(lo),
                af3_cif=str(cif), **{"af3_" + k: v for k, v in epitope_metrics(str(cif)).items()})

def run(df, out, seed=1, n_samples=5, gpu=None, chunk=200):
    """df: id, seq. ONE AF3 process per chunk of designs (--input_dir): the weights and compile cache are loaded once, not per design. RESUMABLE on real outputs
    (a design with parseable output is not re-run). Returns the table; raises if any design has no usable output."""
    cfg = af3_config(); out = Path(out); (out / "logs").mkdir(parents=True, exist_ok=True); cache = out / "jax_cache"; cache.mkdir(exist_ok=True)
    assert b"\x00" not in Path(msa_human()).read_bytes()
    msa = out / "target_msa.a3m"
    if not msa.exists(): json.dump(sanitize_a3m(msa_human(), msa, T.TNF_HUMAN), open(out / "target_msa.stats.json", "w"))
    names = {r.id: af3_name(r.id) for r in df.itertuples()}
    assert len(set(names.values())) == len(names), "AF3 job names collide"
    todo = [r for r in df.itertuples() if parse(out / "pred" / names[r.id], names[r.id], r.seq) is None]
    for c0 in range(0, len(todo), chunk):
        ind = out / f"in_{c0}"; ind.mkdir(exist_ok=True)
        for f in ind.glob("*.json"): f.unlink()
        for r in todo[c0:c0 + chunk]: json.dump(af3_input(names[r.id], r.seq, msa, seed), open(ind / f"{names[r.id]}.json", "w"))
        cmd = [str(cfg["PXD_AF3_PYTHON"]), str(Path(cfg["PXD_AF3_DIR"]) / "run_alphafold.py"), f"--input_dir={ind}", f"--output_dir={out / 'pred'}", f"--model_dir={cfg['PXD_AF3_MODELS']}",
               "--norun_data_pipeline", "--gpu_device=0", f"--num_diffusion_samples={n_samples}", f"--jax_compilation_cache_dir={cache}"]
        ts = time.time(); p = subprocess.run(cmd, capture_output=True, text=True, env=af3_env(gpu))
        (out / "logs" / f"chunk_{c0}.log").write_text(p.stdout[-20000:] + p.stderr[-20000:]); print(f"af3 chunk {c0}: {len(todo[c0:c0 + chunk])} designs rc={p.returncode} {time.time()-ts:.0f}s", flush=True)
    rows = []
    for r in df.itertuples():
        res = parse(out / "pred" / names[r.id], names[r.id], r.seq)
        if res is None: raise RuntimeError(f"AlphaFold3 produced no usable output for {r.id}; see {out / 'logs'}")
        rows.append(dict(id=r.id, **res))
    return pd.DataFrame(rows)
