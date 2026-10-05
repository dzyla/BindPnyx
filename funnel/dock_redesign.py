"""Dock known scaffolds onto a target, redesign ONLY the new interface with ProteinMPNN, then predict and judge with the funnel.

  python funnel/dock_redesign.py --target fimh --out out/dock/fimh --scaffolds all
  python funnel/dock_redesign.py --target fimh --out out/dock/fimh --scaffolds ubiquitin,fn3_monobody --my-scaffold mybinder.pdb:A

Pipeline: scaffold PDB (public, see funnel/scaffolds.json, or your own) -> LightDock rigid-body docking (GSO swarms, `fastdfire` score) restrained to the target hotspots
-> diverse top poses -> ProteinMPNN on the BINDER INTERFACE ONLY (target and the rest of the scaffold fixed; funnel/mpnn_run.py scope=interface) -> design table ->
run_funnel.py --designs-csv (fast screen, interface-only cycling, Boltz-2 + Protenix-v2 consensus, final_design/ review package).
The un-redesigned scaffold sequence is included as a control. Every stage is resumable (docking per scaffold, MPNN per pose set, then the funnel's own resume).

LightDock is GPL-3.0 and is NOT bundled: it lives in its own environment (.pxd/envs/lightdock; see docs/RECOMMENDATIONS.md 'Install LightDock') and is called as a subprocess.
Another docker can be plugged in by replacing dock_one(): it must write ranked complex poses (target chain A, scaffold chain B)."""
import argparse, json, os, re, shutil, subprocess, sys, time, urllib.request
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent)); import common, run_funnel
REPO = common.REPO
THREE = {"A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS", "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE", "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO", "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL"}
ONE = {v: k for k, v in THREE.items()}

def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)

def ld_bin():
    for c in (os.environ.get("PXD_LIGHTDOCK_BIN"), str(REPO / ".pxd/envs/lightdock/bin")):
        if c and (Path(c) / "lightdock3.py").exists(): return Path(c)
    raise FileNotFoundError("LightDock not found: python -m venv .pxd/envs/lightdock && .pxd/envs/lightdock/bin/pip install 'numpy<2' 'cython<3.1' setuptools wheel && "
                            "CFLAGS='-Wno-error=incompatible-pointer-types' .pxd/envs/lightdock/bin/pip install --no-build-isolation lightdock   (or set $PXD_LIGHTDOCK_BIN)")

def ld(script, *args, cwd):
    b = ld_bin(); p = subprocess.run([str(b / "python"), str(b / script), *map(str, args)], cwd=cwd, capture_output=True, text=True)
    if p.returncode: raise RuntimeError(f"{script} failed:\n" + (p.stdout + p.stderr)[-1500:])
    return p.stdout + p.stderr

def prep_scaffold(name, spec, work):
    """Public PDB entry -> single-chain, first-model, standard-residue heavy-atom PDB with sequential numbering and chain B. Returns (path, sequence)."""
    raw = REPO / "data/scaffolds/raw" / f"{spec['pdb_id']}.pdb"; raw.parent.mkdir(parents=True, exist_ok=True)
    if not raw.exists(): urllib.request.urlretrieve(f"https://files.rcsb.org/download/{spec['pdb_id']}.pdb", raw)
    out, seq, seen, n, model = [], [], {}, 0, 0
    for l in raw.read_text().splitlines():
        if l.startswith("MODEL"): model += 1
        if model > 1: break
        if not l.startswith("ATOM") or l[21] != spec.get("chain", "A") or l[16] not in " A" or l[17:20] not in ONE or l[76:78].strip() == "H": continue
        key = (l[22:26], l[26])
        if key not in seen: n += 1; seen[key] = n; seq.append(ONE[l[17:20]])
        out.append(f"{l[:21]}B{seen[key]:4d} {l[27:]}")
    p = Path(work) / "scaffold.pdb"; p.write_text("\n".join(out) + "\nEND\n"); return p, "".join(seq)

def prep_receptor(t, work):
    """Prepared target PDB (source numbering kept, chain A) and the LightDock restraint lines for the hotspots."""
    rec = REPO / "data/targets" / t["name"] / "prepared" / f"{t['name']}.pdb"
    if not rec.exists(): rec = REPO / t["shard"].replace(".pkl.gz", ".pdb")
    lines = [l for l in rec.read_text().splitlines() if l.startswith("ATOM")]; Path(work, "rec.pdb").write_text("\n".join(lines) + "\nEND\n")
    Path(work, "restraints.list").write_text("".join(f"R A.{THREE[h[0]]}.{h[1:]}\n" for h in t["hotspots"])); return Path(work, "rec.pdb")

RANK_RE = re.compile(r"^\s*(\d+)\s+(\d+)\s+\(.*?\)\s+\d+\s+\d+\s+\S+\s+\d+\s+\S+\s+\S+\s+(\S+)\s+(\d+)\s+(\S+)\s*$")
def dock_one(rec, scaf, work, steps, glowworms, cores, swarms=0):
    """LightDock with hotspot restraints. Returns [(swarm, glowworm, clashes, score, pose_pdb)] best first. Resumable: skips if the ranking exists."""
    work = Path(work); rk = work / "rank_by_scoring.list"
    if not rk.exists():
        shutil.copy2(rec, work / "rec.pdb"); shutil.copy2(scaf, work / "lig.pdb")
        ld("lightdock3_setup.py", "rec.pdb", "lig.pdb", *(["-s", swarms] if swarms else []), "-g", glowworms, "--noxt", "--noh", *(["-r", "restraints.list"] if (work / "restraints.list").exists() else []), cwd=work)
        ld("lightdock3.py", "setup.json", steps, "-c", cores, "-s", "fastdfire", cwd=work)
        sw = sorted(work.glob("swarm_*"), key=lambda p: int(p.name.split("_")[1]))
        for s in sw: ld("lgd_generate_conformations.py", "rec.pdb", "lig.pdb", f"{s.name}/gso_{steps}.out", glowworms, cwd=work)
        out = ld("lgd_rank.py", len(sw), steps, cwd=work)
    poses = []
    for l in rk.read_text().splitlines():
        m = RANK_RE.match(l)
        if m: poses.append((int(m[1]), int(m[2]), int(m[4]), float(m[5]), work / f"swarm_{m[1]}" / m[3]))
    return sorted(poses, key=lambda x: -x[3])

def centroid(pdb):
    xyz = [[float(l[30:38]), float(l[38:46]), float(l[46:54])] for l in open(pdb) if l.startswith("ATOM") and l[12:16].strip() == "CA"]; return np.mean(xyz, 0)

def pick_poses(poses, k, min_sep=6.0, max_clashes=None):
    """Top-k by score, at most one pose per ligand-centroid neighbourhood (min_sep A) so the set covers different placements."""
    keep, cs = [], []
    for sw, g, cl, sc, pdb in poses:
        if max_clashes is not None and cl > max_clashes: continue
        if not pdb.exists(): continue
        c = centroid(pdb)
        if all(np.linalg.norm(c - x) >= min_sep for x in cs): keep.append((sw, g, cl, sc, pdb)); cs.append(c)
        if len(keep) >= k: break
    return keep

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", required=True); ap.add_argument("--out", required=True); ap.add_argument("--scaffolds", default="all", help="comma list of names in funnel/scaffolds.json, or 'all'")
    ap.add_argument("--my-scaffold", action="append", default=[], help="PATH.pdb:CHAIN (repeatable)")
    ap.add_argument("--steps", type=int, default=100); ap.add_argument("--glowworms", type=int, default=50); ap.add_argument("--cores", type=int, default=min(16, os.cpu_count() or 4))
    ap.add_argument("--poses", type=int, default=8, help="poses kept per scaffold"); ap.add_argument("--designs-per-pose", type=int, default=8); ap.add_argument("--iface-cutoff", type=float, default=10.0)
    ap.add_argument("--max-clashes", type=int, default=None); ap.add_argument("--temperature", default="0.2")
    ap.add_argument("--rounds", type=int, default=2); ap.add_argument("--no-funnel", action="store_true", help="stop after writing designs.csv")
    ap.add_argument("--funnel-args", default="", help="extra arguments for run_funnel.py")
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); t = common.load_target(a.target)
    lib = json.load(open(Path(__file__).with_name("scaffolds.json"))); names = list(lib) if a.scaffolds == "all" else a.scaffolds.split(",")
    scaf = {n: lib[n] for n in names}
    for s in a.my_scaffold:                                           # your own scaffold: copy into the raw cache under its stem
        p, ch = s.split(":"); stem = Path(p).stem; (REPO / "data/scaffolds/raw").mkdir(parents=True, exist_ok=True); shutil.copy2(p, REPO / "data/scaffolds/raw" / f"{stem}.pdb"); scaf[stem] = dict(pdb_id=stem, chain=ch)
    rec_dir = out / "receptor"; rec_dir.mkdir(exist_ok=True); rec = prep_receptor(t, rec_dir); log(f"target {t['name']}: restraints {[h for h in t['hotspots']]}; {len(scaf)} scaffolds")
    rows, seqs_native = [], {}; posedir = out / "poses"; posedir.mkdir(exist_ok=True)
    for name, spec in scaf.items():
        w = out / "dock" / name; w.mkdir(parents=True, exist_ok=True)
        sp, seq = prep_scaffold(name, spec, w)
        seqs_native[name] = seq; shutil.copy2(rec_dir / "restraints.list", w / "restraints.list"); t0 = time.time()
        poses = dock_one(rec_dir / "rec.pdb", sp, w, a.steps, a.glowworms, a.cores); keep = pick_poses(poses, a.poses, max_clashes=a.max_clashes)
        log(f"{name}: {len(seq)} aa, {len(poses)} poses docked in {time.time() - t0:.0f}s; kept {len(keep)} (best score {keep[0][3] if keep else float('nan'):.2f})")
        for k, (sw, g, cl, sc, pdb) in enumerate(keep):
            recpdb = [l for l in (w / "lightdock_rec.pdb").read_text().splitlines() if l.startswith("ATOM")]; lig = [l for l in Path(pdb).read_text().splitlines() if l.startswith("ATOM")]
            (posedir / f"{name}_p{k}.pdb").write_text("\n".join(recpdb) + "\nTER\n" + "\n".join(lig) + "\nEND\n"); rows.append(dict(scaffold=name, pose=k, swarm=sw, glowworm=g, clashes=cl, dock_score=sc, native_seq=seq))
    P = pd.DataFrame(rows); P.to_csv(out / "poses.csv", index=False); log(f"{len(P)} docked poses written to {posedir}")
    # ---- interface-only redesign (resumable: skip if the csv exists)
    mcsv = out / "mpnn_interface.csv"
    if not mcsv.exists():
        run_funnel.run_mpnn(posedir, a.designs_per_pose, mcsv, "soluble", a.temperature, "none", f"interface:{a.iface_cutoff}")
    m = pd.read_csv(mcsv); key = {f"{r.scaffold}_p{r.pose}": i for i, r in enumerate(P.itertuples())}
    D = [dict(id=f"d{i:05d}", bb=key[r.name], seq=r.sequence, scaffold=P.iloc[key[r.name]].scaffold, kind="redesign", dock_score=P.iloc[key[r.name]].dock_score) for i, r in enumerate(m.itertuples())]
    for n_, s_ in seqs_native.items():                                      # controls: the scaffold sequence, un-redesigned, in its best pose
        j = P.index[P.scaffold == n_]
        if len(j): D.append(dict(id=f"n_{n_}", bb=int(j[0]), seq=s_, scaffold=n_, kind="native_control", dock_score=P.loc[j[0], "dock_score"]))
    D = pd.DataFrame(D).drop_duplicates("seq"); D.to_csv(out / "designs.csv", index=False); log(f"{len(D)} designs ({int((D.kind == 'redesign').sum())} redesigned + controls) -> {out / 'designs.csv'}")
    if a.no_funnel: return
    cmd = [common.PXD_PY(), str(Path(__file__).with_name("run_funnel.py")), "--target", a.target, "--out", str(out / "funnel"), "--designs-csv", str(out / "designs.csv"),
           "--rounds", str(a.rounds), "--cycle-scope", f"interface:{a.iface_cutoff}", "--cycle-k", "40", "--final-m", "60", "--top", "20", *a.funnel_args.split()]
    log("running funnel:", " ".join(cmd[1:])); rc = subprocess.run(cmd, env={**os.environ, "PYTHONPATH": str(REPO)}).returncode
    if rc: raise SystemExit(f"funnel failed (rc={rc}); re-run the same command to resume")

if __name__ == "__main__": main()
