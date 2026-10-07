"""Build the PXDesign dimer shard from the registered target (idempotent): organiser B+C dimer PDB -> shard with provenance, residue 143 reconciled by an explicit edit,
MSA cropped to the observed residues, per-chain MSA dirs. Needs phbind/target/ (run s0_target.py with $TNF_BUNDLE first) and the pxd env (PXD_PYTHON or .pxd/envs/pxd).
Writes out/phbind/shard/{tnf_dimerBC.pkl.gz, provenance.json, msa/{A,B}/0}.   PYTHONPATH=<repo> python phbind/s0b_shard.py
"""
import json, os, shutil, subprocess, sys
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "phbind")); sys.path.insert(0, str(REPO / "funnel"))
import trimer as T, msa_crop, common

def build(out: Path | None = None) -> Path:
    out = Path(out or REPO / "out/phbind/shard"); tgt = REPO / "phbind/target"; man = tgt / "target_manifest.json"
    if (out / "tnf_dimerBC.pkl.gz").exists(): return out
    if not man.exists(): raise SystemExit("run phbind/s0_target.py first (it needs $TNF_BUNDLE)")
    m = json.load(open(man)); out.mkdir(parents=True, exist_ok=True); (out / "msa").mkdir(exist_ok=True)
    (out / "wt.fasta").write_text(f">tnf_human_157\n{T.TNF_HUMAN}\n")
    crop = out / "msa" / "tnf_6_157.a3m"; n = msa_crop.crop_a3m(m["files"]["msa_human"]["path"], crop, 6, 152 + 5)         # local 6-157: the organiser PDB lacks the 5 disordered N-terminal residues
    assert b"\x00" not in crop.read_bytes() and n > 100
    q = crop.read_text().splitlines()[1]
    for c in "AB":
        d = out / "msa" / c / "0"; d.mkdir(parents=True, exist_ok=True); shutil.copy(crop, d / "non_pairing.a3m"); (d / "pairing.a3m").write_text(f">query\n{q}\n")
    cmd = [common.PXD_PY(), str(REPO / "scripts/prepare_target.py"), "--source", m["files"]["human_dimerBC"]["path"], "--wt-fasta", str(out / "wt.fasta"), "--wt-numbering", "mature", "--wt-offset", "0",
           "--chain", "B", "--chain", "C", "--msa", f"B={crop}", "--msa", f"C={crop}", "--edit", "B:143:LEU>ASP:truncate_to_cb", "--edit", "C:143:LEU>ASP:truncate_to_cb", "--name", "tnf_dimerBC", "--out-dir", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(REPO)}); (out / "prepare.log").write_text(r.stdout[-6000:] + r.stderr[-6000:])
    if not (out / "tnf_dimerBC.pkl.gz").exists(): raise SystemExit(f"prepare_target produced no shard; see {out / 'prepare.log'}")      # gate on the artifact, not the exit code
    return out

if __name__ == "__main__":
    print("shard ready:", build())
