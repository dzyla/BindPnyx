"""Build a design target from a PUBLIC PDB entry: no structures are shipped with the repository.

  python funnel/fetch_target.py funnel/targets/fimh.json            # downloads from RCSB, builds the shard + MSA + site probe under data/targets/<name>/
  python funnel/fetch_target.py funnel/targets/fimh.json --force

Target definition (the "source" block is what this script reads):
  {"name": "fimh", "source": {"pdb_id": "3MCY", "chain": "A", "range": "1-158", "ligand": "ZH1", "probe_radius": 5.0},
   "hotspots": ["Y48", "D54"], "binder_length": 65, "description": "..."}
Steps: RCSB download -> resolved sequence of the chain/range (WT, sequential alignment) -> MSA from the ColabFold MMseqs2 server via Boltz (internet needed; CPU is
enough) -> scripts/prepare_target.py (shard + provenance, residue identities checked) -> optional ligand site probe (reference CA + probe atoms) for occlusion tests.
Outputs (git-ignored): data/targets/<name>/{raw,wt.fasta,msa/A/0,prepared,site}/"""
import argparse, csv, json, os, subprocess, sys, urllib.request
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent)); import common
REPO = common.REPO
THREE = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}

def parse_chain(pdb_text, chain, lo, hi):
    """Resolved standard residues of one chain within [lo, hi]: list of (resnum, letter, CA xyz). First altloc only."""
    seen, out = set(), []
    for l in pdb_text.splitlines():
        if not l.startswith("ATOM") or l[21] != chain or l[16] not in " A": continue
        rn = int(l[22:26]); ic = l[26]
        if ic != " " or rn < lo or rn > hi or l[17:20] not in THREE: continue
        if l[12:16].strip() == "CA" and rn not in seen:
            seen.add(rn); out.append((rn, THREE[l[17:20]], np.array([float(l[30:38]), float(l[38:46]), float(l[46:54])])))
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("target_json"); ap.add_argument("--force", action="store_true"); a = ap.parse_args()
    cfg = json.load(open(a.target_json)); name = cfg["name"]; src = cfg["source"]; base = REPO / "data" / "targets" / name; base.mkdir(parents=True, exist_ok=True)
    raw = base / "raw" / f"{src['pdb_id']}.pdb"; raw.parent.mkdir(exist_ok=True)
    if not raw.exists() or a.force:
        print("downloading", src["pdb_id"]); urllib.request.urlretrieve(f"https://files.rcsb.org/download/{src['pdb_id']}.pdb", raw)
    txt = raw.read_text(); lo, hi = (int(x) for x in src.get("range", "1-9999").split("-")); ch = src["chain"]
    res = parse_chain(txt, ch, lo, hi); seq = "".join(r[1] for r in res); print(f"{name}: chain {ch} {res[0][0]}-{res[-1][0]}, {len(res)} resolved residues")
    (base / "wt.fasta").write_text(f">{name}_{src['pdb_id']}_{ch}\n{seq}\n")
    msa = base / "msa" / "A" / "0"
    if not (msa / "non_pairing.a3m").exists() or a.force:
        msa.mkdir(parents=True, exist_ok=True); tmp = base / "msa_tmp"; tmp.mkdir(exist_ok=True)
        (tmp / "q.yaml").write_text(f"version: 1\nsequences:\n  - protein:\n      id: A\n      sequence: {seq}\n")
        print("fetching MSA from the ColabFold server (via Boltz, CPU)")
        p = subprocess.run([common.BOLTZ(), "predict", str(tmp / "q.yaml"), "--out_dir", str(tmp / "out"), "--use_msa_server", "--recycling_steps", "1", "--sampling_steps", "1", "--override", "--accelerator", "cpu"], capture_output=True, text=True)
        csvf = next((tmp / "out").glob("boltz_results_q/msa/q_0.csv"), None)
        if csvf is None: raise RuntimeError("MSA server step failed:\n" + (p.stderr or p.stdout)[-1500:])
        rows = [r["sequence"] for r in csv.DictReader(open(csvf))]
        assert rows[0].replace("-", "") == seq, "MSA query differs from the structure sequence"
        (msa / "non_pairing.a3m").write_text("".join(f">{i}\n{s}\n" for i, s in enumerate(rows))); (msa / "pairing.a3m").write_text(f">query\n{seq}\n"); print(len(rows), "MSA sequences")
    prep = base / "prepared"
    if not (prep / f"{name}.pkl.gz").exists() or a.force:
        cmd = [common.PXD_PY(), str(REPO / "scripts/prepare_target.py"), "--source", str(raw), "--wt-fasta", str(base / "wt.fasta"), "--wt-align", "sequential", "--chain", f"{ch}:{lo}-{hi}={'A'}",
               "--msa", f"A={msa}", "--min-depth", "1", "--allow-gaps", "--name", name, "--out-dir", str(prep)]
        p = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(REPO)})
        if p.returncode: raise RuntimeError("prepare_target failed:\n" + (p.stdout + p.stderr)[-2500:])
        print("shard built:", prep / f"{name}.pkl.gz")
    # optional ligand site probe for occlusion tests
    if src.get("ligand"):
        site = base / "site"; site.mkdir(exist_ok=True); lig = []
        for l in txt.splitlines():
            if l.startswith("HETATM") and l[17:20].strip() == src["ligand"] and l[21] == ch:
                lig.append((l[12:16].strip(), np.array([float(l[30:38]), float(l[38:46]), float(l[46:54])])))
        if not lig: raise RuntimeError(f"ligand {src['ligand']} not found on chain {ch}")
        L = np.array([x for _, x in lig]); ca = np.array([r[2] for r in res]); allp = np.array([[float(l[30:38]), float(l[38:46]), float(l[46:54])] for l in txt.splitlines()
                                                                                                if l.startswith("ATOM") and l[21] == ch and np.linalg.norm(np.array([float(l[30:38]), float(l[38:46]), float(l[46:54])]) - L.mean(0)) < 9])
        pocket_centre = allp.mean(0); d = np.linalg.norm(L - pocket_centre, axis=1); probe = L[d <= src.get("probe_radius", 5.0)]
        np.save(site / "ref_ca.npy", ca); np.save(site / "probe.npy", probe); np.save(site / "ligand_all.npy", L)
        json.dump(dict(ligand=src["ligand"], n_ligand_atoms=len(L), n_probe_atoms=len(probe), probe_radius=src.get("probe_radius", 5.0), pocket_centre=pocket_centre.tolist()), open(site / "site.json", "w"), indent=1)
        print(f"site probe: {len(probe)} of {len(L)} {src['ligand']} atoms within {src.get('probe_radius', 5.0)} A of the pocket centre")
    t = common.load_target(a.target_json); print("target ready:", t["name"], len(t["seq"]), "aa; hotspots", t["hotspots"], "->", t["hotspot_idx"])

if __name__ == "__main__": main()
