"""Refold-redesign cycling with Protenix-0.5-mini 'fast' as the in-loop judge.
For each starting backbone: keep its best sequence (parent); each round redesign N_NEW sequences on the parent's
PREDICTED complex, fast-predict them, keep the best of {parent, children} (elitist).
usage (pxd env, PYTHONPATH=repo): cycle.py TAG TARGET_SEQ_TARGET METRIC K ROUNDS N_NEW"""
import subprocess, sys, os, shutil
import pandas as pd
from pathlib import Path
from Bio.PDB import MMCIFParser, PDBIO
import score_protenix as sp

tag, tgt, metric, K, R, NNEW = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
W = Path(f"out/cycle_{tag}"); W.mkdir(parents=True, exist_ok=True)
PY = str(Path("../.pxd/envs/pxd/bin/python").resolve()); REPO = str(Path("..").resolve())
tseq = pd.read_csv("inputs/manifest.csv").query("target==@tgt").target_seq.iloc[0]
col = f"ptx05_fast_{metric}"

def cif_to_pdb(cif, pdb):
    s = MMCIFParser(QUIET=True).get_structure("x", cif); io = PDBIO(); io.set_structure(s); io.save(str(pdb))

def fast(ids_seqs, rd):
    mf = W / f"manifest_r{rd}.csv"
    pd.DataFrame(dict(target=tgt, id=[i for i, _ in ids_seqs], label=-1, method=f"r{rd}", binder_seq=[s for _, s in ids_seqs], target_seq=tseq)).to_csv(mf, index=False)
    sp.run("ptx05_fast", manifest=str(mf), out=str(W / f"fast_r{rd}"))
    return pd.read_csv(W / f"fast_r{rd}" / "scores_all.csv")

# round 0: starting designs = the K backbones with highest best-of-4 fast score (existing scores)
base = pd.read_csv(f"out/fast_pxd_{tag}/scores_all.csv").merge(pd.read_csv(f"inputs/manifest_pxd_{tag}.csv")[["id", "binder_seq"]], on="id")
base["bb"] = base.id.str.extract(r"sample_(\d+)_")[0]
best0 = base.sort_values(col, ascending=False).groupby("bb").head(1).sort_values(col, ascending=False).head(K).copy()
best0["pred_dir"] = [f"out/fast_pxd_{tag}/{tgt}/pred/{i}/seed_101/predictions/{i}_sample_0.cif" for i in best0.id]
parents = {r.bb: dict(id=r.id, seq=r.binder_seq, score=getattr(r, col), cif=r.pred_dir, rd=0) for r in best0.itertuples()}
log = [dict(bb=b, rd=0, id=p["id"], seq=p["seq"], score=p["score"], kept=True) for b, p in parents.items()]
for rd in range(1, R + 1):
    pd_dir = W / f"pdb_r{rd}"; shutil.rmtree(pd_dir, ignore_errors=True); pd_dir.mkdir(parents=True)
    for b, p in parents.items(): cif_to_pdb(p["cif"], pd_dir / f"bb{b}.pdb")
    env = {**os.environ, "PYTHONPATH": REPO, "XLA_PYTHON_CLIENT_PREALLOCATE": "false"}
    subprocess.run([PY, "mpnn_round.py", str(pd_dir), str(NNEW), str(W / f"mpnn_r{rd}.csv")], check=True, env=env, capture_output=True)
    m = pd.read_csv(W / f"mpnn_r{rd}.csv"); m["bb"] = m.name.str.replace("bb", ""); m["id"] = [f"c{rd}_{b}_{i}" for b, i in zip(m.bb, m.seq_idx)]
    sc = fast(list(zip(m.id, m.sequence)), rd).merge(m[["id", "bb", "sequence"]], on="id")
    for b, g in sc.groupby("bb"):
        g = g.dropna(subset=[col]); top = g.sort_values(col, ascending=False).iloc[0] if len(g) else None
        for r in g.itertuples(): log.append(dict(bb=b, rd=rd, id=r.id, seq=r.sequence, score=getattr(r, col), kept=False))
        if top is not None and top[col] > parents[b]["score"]:
            parents[b] = dict(id=top["id"], seq=top["sequence"], score=top[col], rd=rd, cif=str(W / f"fast_r{rd}" / tgt / "pred" / top["id"] / "seed_101" / "predictions" / f"{top['id']}_sample_0.cif"))
    print(f"round {rd}: mean parent fast {metric} = {sum(p['score'] for p in parents.values())/len(parents):.3f}", flush=True)
    pd.DataFrame(log).to_csv(W / "cycle_log.csv", index=False)
pd.DataFrame([dict(bb=b, start_id=best0[best0.bb == b].id.iloc[0], start_seq=best0[best0.bb == b].binder_seq.iloc[0], start_score=best0[best0.bb == b][col].iloc[0],
                   final_id=p["id"], final_seq=p["seq"], final_score=p["score"], final_round=p["rd"]) for b, p in parents.items()]).to_csv(W / "summary.csv", index=False)
