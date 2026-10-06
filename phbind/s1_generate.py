"""S1: PXDesign backbones against the B+C dimer, then ProteinMPNN (SolubleMPNN, no cysteine). One run = (arm, binder length).

Hotspots are given as organiser chain + author number ("B87"); they are converted to the shard's own numbering through the
provenance residue map with an AMINO-ACID IDENTITY CHECK (the stock pipeline silently reads shard indices). Counted-artifact
guards: backbones written == requested, all backbones distinct, no CYS in any designed sequence.

  python phbind/s1_generate.py --arm foot12 --length 75 --n 8 --seed 11 --out out/phbind/gen/smoke
"""
import argparse, hashlib, json, os, shutil, subprocess, sys, time
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "funnel")); sys.path.insert(0, str(REPO / "phbind"))
import trimer as T
import common

SHARD = REPO / "out/phbind/shard"
HOTSPOT_SETS = {
    "foot12": ["B87", "B88", "B90", "C21", "C33", "C65", "C67", "C113", "C115", "C144", "C145", "C146"],   # confirmed-binder footprint (handoff)
    "decl8": ["B75", "B86", "B87", "C32", "C33", "C115", "C145", "C146"],      # staged declared set; best sibling arm used it (digest s16.4: re-aimed not better, p=1.0)
    "core6": ["B87", "B90", "C33", "C65", "C115", "C145"],                      # compact two-protomer subset (repo advice: 3-6 hotspots)
}

def shard_hotspots(ids):
    prov = json.load(open(SHARD / "provenance.json")); to_shard = {c["prepared_chain"]: c["shard_chain"] for c in prov["shard"]["chains"]}
    out = {}
    for h in ids:
        ch, num = h[0], int(h[1:])
        e = next(x for x in prov["residue_map"][ch] if x["source_resnum"] == num)
        assert e["residue"] == T.TNF_HUMAN[num - 1], f"{h}: shard residue {e['residue']} != construct {T.TNF_HUMAN[num-1]}"
        out.setdefault(to_shard[ch], []).append(e["shard_resnum"])
    return out

def manifest(name, L, hs):
    return [{"name": name, "condition": {"structure_file": str(SHARD / "tnf_dimerBC.pkl.gz"),
             "filter": {"chain_id": ["A", "B"], "crop": {}},
             "msa": {c: {"precomputed_msa_dir": str(SHARD / "msa" / c / "0"), "pairing_db": "uniref100"} for c in "AB"}},
             "hotspot": shard_hotspots(hs), "generation": [{"type": "protein", "length": L, "count": 1}]}]

def generate(arm, L, n, seed, out, steps=400):
    out = Path(out); (out / "run").mkdir(parents=True, exist_ok=True)
    name = f"tnf_{arm}_L{L}_s{seed}"; mf = out / "input.json"; json.dump(manifest(name, L, HOTSPOT_SETS[arm]), open(mf, "w"), indent=1)
    t0 = time.time()
    p = subprocess.run([common.PXD_PY(), "-u", "-m", "pxdesign.runner.inference", "--input_json_path", str(mf), "--dump_dir", str(out / "run"),
                        "--load_checkpoint_dir", common.CKPT(), "--N_sample", str(n), "--N_step", str(steps), "--seeds", str(seed)],
                       env={**common.gpu_env(), "PYTHONPATH": str(REPO)}, capture_output=True, text=True)
    (out / "log.txt").write_text(p.stdout[-30000:] + p.stderr[-30000:])
    pdirs = [d for d in (out / "run").glob("*/seed_*/predictions") if d.is_dir()]
    if not pdirs: raise RuntimeError(f"diffusion produced no predictions (rc={p.returncode}); see {out/'log.txt'}")
    from pxdbench.utils import convert_cifs_to_pdbs
    cdir, names, cond, bind = convert_cifs_to_pdbs(str(pdirs[0]))
    assert len(names) == n, f"requested {n} backbones, got {len(names)}"
    pdb_dir = out / "pdbs"; pdb_dir.mkdir(exist_ok=True)
    for nm in names: shutil.copy2(Path(cdir) / f"{nm}.pdb", pdb_dir / f"{arm}_L{L}_s{seed}_{nm.rsplit('_sample_',1)[1]}.pdb")
    hashes = {hashlib.md5(open(f, "rb").read()).hexdigest() for f in pdb_dir.glob("*.pdb")}
    assert len(hashes) == n, "duplicate backbones (seed not effective)"
    json.dump(dict(arm=arm, L=L, n=n, seed=seed, steps=steps, cond=cond, bind=bind, seconds=time.time() - t0, hotspots=HOTSPOT_SETS[arm]), open(out / "done.json", "w"))
    print(f"generated {n} backbones arm={arm} L={L} in {time.time()-t0:.0f}s cond={cond} bind={bind}", flush=True)
    return pdb_dir, cond, bind

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--arm", required=True); ap.add_argument("--length", type=int, required=True)
    ap.add_argument("--n", type=int, required=True); ap.add_argument("--seed", type=int, required=True); ap.add_argument("--out", required=True); ap.add_argument("--steps", type=int, default=400)
    a = ap.parse_args(); generate(a.arm, a.length, a.n, a.seed, a.out, a.steps)
