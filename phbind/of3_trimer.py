"""OpenFold3 (p2-155k) as the SECOND oracle on the intact TNF trimer (handoff S3: a fixed pair, never re-selected).
Same inputs as the Boltz arm: 3 target copies with the `_nulfix` MSA, binder with a query-only MSA, no templates. Chain order A,B,C,D.
Scored with the same grouped ipSAE (min direction gates, max reported) after the same group_indices assertions."""
from __future__ import annotations
import json, subprocess, sys, time
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "funnel"))
import trimer as T, common
from boltz_trimer import msa_human, epitope_metrics

def run(df, out, seed=101, chunk=400):
    """df: id, seq. RESUMABLE on real outputs. Raises if any design lacks output. Returns the per-design table."""
    out = Path(out); (out / "msa").mkdir(parents=True, exist_ok=True)
    tmsa = out / "msa/target/colabfold_main.a3m"; tmsa.parent.mkdir(exist_ok=True)
    assert b"\x00" not in Path(msa_human()).read_bytes()
    assert common.a3m_rectangular(msa_human(), tmsa) >= 2
    base = lambda i: out / "pred" / i / f"seed_{seed}"
    files = lambda i: (base(i) / f"{i}_seed_{seed}_sample_1_confidences.json", base(i) / f"{i}_seed_{seed}_sample_1_confidences_aggregated.json", base(i) / f"{i}_seed_{seed}_sample_1_model.cif")
    todo = df[[not all(p.exists() for p in files(r.id)) for r in df.itertuples()]]
    (out / "runner.yml").write_text(f"experiment_settings:\n  seeds:\n    - {seed}\n")
    for c0 in range(0, len(todo), chunk):
        qs = {}
        for r in todo.iloc[c0:c0 + chunk].itertuples():
            bm = out / "msa" / r.id / "colabfold_main.a3m"; bm.parent.mkdir(exist_ok=True); bm.write_text(f">query\n{r.seq}\n")
            qs[r.id] = {"chains": [{"molecule_type": "protein", "chain_ids": ["A", "B", "C"], "sequence": T.TNF_HUMAN, "main_msa_file_paths": str(tmsa.resolve())},
                                   {"molecule_type": "protein", "chain_ids": ["D"], "sequence": r.seq, "main_msa_file_paths": str(bm.resolve())}]}
        qf = out / f"query_{c0}.json"; json.dump({"queries": qs}, open(qf, "w")); t0 = time.time()
        p = subprocess.run([common.OF3(), "predict", "--query-json", str(qf), "--use-msa-server=False", "--output-dir", str(out / "pred"), "--num-diffusion-samples", "1",
                            "--runner-yaml", str(out / "runner.yml"), "--inference-ckpt-path", common.OF3_CKPT()], capture_output=True, text=True, env=common.gpu_env())
        (out / f"log_{c0}.txt").write_text(p.stdout[-20000:] + p.stderr[-20000:]); print(f"of3 chunk {c0}: {len(qs)} designs rc={p.returncode} {time.time()-t0:.0f}s", flush=True)
    rows = []
    for r in df.itertuples():
        cj, aj, cf = files(r.id)
        if not all(x.exists() for x in (cj, aj, cf)): raise RuntimeError(f"OpenFold3 produced no output for {r.id}")
        pae = np.array(json.load(open(cj))["pae"], float); a = json.load(open(aj))
        ti, bi, info = T.group_indices(str(cf), species="human", binder_seq=r.seq, pae_n=pae.shape[0])
        assert info["n_target_residues"] == 471 and info["chain_order"] == ["A", "B", "C", "D"], info
        lo, hi = T.ipsae_grouped(pae, ti, bi)
        rows.append(dict(id=r.id, of3_ipsae_min=lo, of3_ipsae_max=hi, of3_iptm=a.get("iptm"), of3_cif=str(cf), **{"of3_" + k: v for k, v in epitope_metrics(str(cf)).items()}))
    return pd.DataFrame(rows)
