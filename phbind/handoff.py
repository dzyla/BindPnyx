"""Package structures for ANOTHER session/tool to screen (e.g. a structural-novelty search): one binder-chain PDB per design inside a tar.gz, plus an index.
Novelty is a property of the backbone, so screen backbones before spending oracle GPU on their sequences; the predicted-binder-chain pack is the faithful query for designs that already
have a model. This module only builds the packs; it runs no search and needs no database.

  PYTHONPATH=<repo> python phbind/handoff.py backbones --min-length 120 --out DIR      binder chains (poly-Gly, from the generator) of every backbone >= min length
  PYTHONPATH=<repo> python phbind/handoff.py predicted --min-boltz 0.5 --out DIR       Boltz-predicted binder chains of designs that cleared the prescreen
Results come back as a CSV with columns `bbid` (or `id`) and `n_strict_hits` (PDB hits with TM >= 0.80 at >= 70% query coverage); s2_prescreen reads it (config prescreen.novelty_file).
"""
from __future__ import annotations
import glob, io, sys, tarfile
from pathlib import Path
import pandas as pd
REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "out" / "phbind"

def chain_pdb(pdb_text: str, chain: str) -> str:
    out = [l for l in pdb_text.splitlines(keepends=True) if l.startswith("ATOM") and l[21] == chain]
    if not out: raise ValueError(f"no ATOM records for chain {chain}")
    return "".join(out) + "END\n"

def write_tar(files: dict, tar_path: Path, root: str) -> int:
    tar_path = Path(tar_path); tar_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "w:gz") as t:
        for name, text in sorted(files.items()):
            data = text.encode(); ti = tarfile.TarInfo(f"{root}/{name}.pdb"); ti.size = len(data); t.addfile(ti, io.BytesIO(data))
    return len(files)

def backbone_files(min_length: int = 0, gen: Path | None = None, runs: set | None = None) -> tuple[dict, pd.DataFrame]:
    """{bbid: binder-chain PDB} for every generated backbone with length >= min_length (chain C of the dimer-target complex)."""
    gen = Path(gen or OUT / "gen"); files, rows = {}, []
    for d in sorted(gen.glob("*_L*")):
        if not (d / "pdbs").exists() or not (d / "done.json").exists(): continue
        name, L = d.name.rsplit("_L", 1); L = int(L)
        if L < min_length or (runs is not None and d.name not in runs): continue
        for p in sorted((d / "pdbs").glob("*.pdb")):
            k = int(p.stem.rsplit("_", 1)[1]); bb = f"{name}_L{L}_{k}"; files[bb] = chain_pdb(p.read_text(), "C"); rows.append(dict(bbid=bb, set=name, L=L, k=k))
    return files, pd.DataFrame(rows)

def predicted_files(min_boltz: float = 0.5, s2: Path | None = None, min_length: int = 0) -> dict:
    from Bio.PDB import MMCIFParser, PDBIO, Select
    class D(Select):
        def accept_chain(self, c): return c.id == "D"
    a = pd.concat([pd.read_csv(f) for f in sorted(Path(s2 or OUT / "s2").glob("batch_*.csv"))], ignore_index=True)
    a = a[~a.id.str.startswith("carrier_") & (a.ipsae_min >= min_boltz)].drop_duplicates("id")
    a = a[a.id.str.extract(r"_L(\d+)_")[0].astype(int) >= min_length]
    out = {}
    for r in a.itertuples():
        io_ = PDBIO(); io_.set_structure(MMCIFParser(QUIET=True).get_structure("x", str(r.cif))); b = io.StringIO(); io_.save(b, D()); out[r.id] = b.getvalue()
    return out

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("what", choices=["backbones", "predicted"]); ap.add_argument("--min-length", type=int, default=0)
    ap.add_argument("--min-boltz", type=float, default=0.5); ap.add_argument("--out", required=True); a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.what == "backbones":
        f, idx = backbone_files(a.min_length); n = write_tar(f, out / f"backbones_L{a.min_length}plus.tar.gz", f"backbones_L{a.min_length}plus"); idx.to_csv(out / f"backbones_L{a.min_length}plus_index.csv", index=False)
    else:
        f = predicted_files(a.min_boltz, min_length=a.min_length); n = write_tar(f, out / f"predicted_boltz{a.min_boltz}.tar.gz", "predicted_binder"); pd.DataFrame({"id": sorted(f)}).to_csv(out / f"predicted_boltz{a.min_boltz}_index.csv", index=False)
    print(n, "structures packed ->", out)
