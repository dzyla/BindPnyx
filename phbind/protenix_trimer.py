"""Protenix-v2 as a candidate SECOND model on the intact TNF trimer (3 copies as one entity with count=3, the same `_nulfix` MSA on every copy,
binder with a query-only MSA, no templates). Scored with the same grouped ipSAE after the same group_indices assertions as the Boltz-2 arm."""
from __future__ import annotations
import json, subprocess, sys, time
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "funnel"))
import trimer as T, common
from boltz_trimer import msa_human, epitope_metrics

ARMS = {"v2": ("protenix-v2", "4", "200"), "fast": ("protenix_mini_default_v0.5.0", "2", "5")}     # fast = the funnel's Protenix-0.5-mini screen (docs/REPORT.md)

def run(df, out, seed=101, chunk=400, arm="v2"):
    out = Path(out); (out / "msa").mkdir(parents=True, exist_ok=True); tm = msa_human()
    assert b"\x00" not in Path(tm).read_bytes()
    tq = out / "msa" / "target_query.a3m"; tq.write_text(f">query\n{T.TNF_HUMAN}\n")
    paths = lambda i: (lambda pr: (pr / f"{i}_summary_confidence_sample_0.json", pr / f"{i}_full_data_sample_0.json", pr / f"{i}_sample_0.cif"))(out / "pred" / i / f"seed_{seed}" / "predictions")
    todo = df[[not all(p.exists() for p in paths(r.id)) for r in df.itertuples()]]
    for c0 in range(0, len(todo), chunk):
        jobs = []
        for r in todo.iloc[c0:c0 + chunk].itertuples():
            bq = out / "msa" / f"{r.id}.a3m"; bq.write_text(f">query\n{r.seq}\n")
            jobs.append({"name": r.id, "covalent_bonds": [], "sequences": [
                {"proteinChain": {"sequence": T.TNF_HUMAN, "count": 3, "unpairedMsaPath": str(tm), "pairedMsaPath": str(tq.resolve())}},
                {"proteinChain": {"sequence": r.seq, "count": 1, "unpairedMsaPath": str(bq.resolve()), "pairedMsaPath": str(bq.resolve())}}]})
        jf = out / f"in_{c0}.json"; json.dump(jobs, open(jf, "w")); t0 = time.time()
        p = subprocess.run([common.PXD_BIN("protenix"), "pred", "-i", str(jf), "-o", str(out / "pred"), "-s", str(seed), "-n", ARMS[arm][0], "-e", "1", "-c", ARMS[arm][1], "-p", ARMS[arm][2],
                            "--use_msa", "true", "--need_atom_confidence", "true"], capture_output=True, text=True, env={**common.gpu_env(), "PYTHONPATH": str(HERE.parent)})
        (out / f"log_{c0}.txt").write_text(p.stdout[-20000:] + p.stderr[-20000:]); print(f"protenix chunk {c0}: {len(jobs)} designs rc={p.returncode} {time.time()-t0:.0f}s", flush=True)
    rows = []
    for r in df.itertuples():
        sj, fj, cf = paths(r.id)
        if not all(x.exists() for x in (sj, fj, cf)): raise RuntimeError(f"Protenix produced no output for {r.id}")
        pae = np.array(json.load(open(fj))["token_pair_pae"], float); s = json.load(open(sj))
        ti, bi, info = T.group_indices(str(cf), species="human", binder_seq=r.seq, pae_n=pae.shape[0]); assert info["n_target_residues"] == 471
        lo, hi = T.ipsae_grouped(pae, ti, bi)
        rows.append(dict(id=r.id, ptx_ipsae_min=lo, ptx_ipsae_max=hi, ptx_iptm=s.get("iptm"), ptx_cif=str(cf), **{"ptx_" + k: v for k, v in epitope_metrics(str(cf)).items()}))
    return pd.DataFrame(rows)
