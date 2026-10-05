"""Where does the high Lys+Glu come from? Redesign (a) backbones of wet-lab-tested designs from other tools and (b) our diffusion backbones with
the funnel's MPNN settings, and compare composition with the native sequences, split by burial class. CPU only.
usage: ke_cause.py OUT_CSV"""
import os, sys, glob, numpy as np, pandas as pd
os.environ.setdefault("JAX_PLATFORMS", "cpu")
from pathlib import Path
from Bio.PDB import MMCIFParser, PDBIO
from colabdesign.mpnn import mk_mpnn_model

REPO = Path(__file__).resolve().parent.parent; OUT = sys.argv[1]
man = pd.read_csv(REPO / "bench/inputs/manifest.csv"); meth = man.assign(m=man.method.fillna("unknown").str.replace(r"-[A-Za-z0-9_-]{8,}$", "", regex=True)).set_index("id")
tmp = Path("/tmp/claude-1000/ke_pdb"); tmp.mkdir(exist_ok=True)

def classes(ca, nt):
    """binder residue classes from CA geometry: interface (CA within 10 A of target), core (>=14 CA neighbours within 10 A, not interface), surface."""
    nb = ca[nt:]; d_t = np.linalg.norm(nb[:, None] - ca[None, :nt], axis=-1).min(1)
    dens = (np.linalg.norm(nb[:, None] - nb[None], axis=-1) < 10).sum(1) - 1
    cls = np.where(d_t < 10, "interface", np.where(dens >= 14, "core", "surface")); return cls

def comp(seqs):
    t = "".join(seqs); n = max(1, len(t)); return dict(KE=sum(t.count(c) for c in "KE") / n, aromatic=sum(t.count(c) for c in "FWY") / n, n_res=n)

rows = []
def design(model, pdb, nseq, T=0.1):
    model.prep_inputs(pdb_filename=str(pdb), chain="A,B", fix_pos="A", rm_aa="C"); nt = int(model._lengths[0])
    cls = classes(model._inputs["X"][:, 1], nt); return cls, [s.split("/")[-1] for s in model.sample(num=nseq, temperature=T)["seq"]]

model = mk_mpnn_model(backbone_noise=0.0, model_name="v_48_020", weights="soluble")
# (a) backbones of wet-lab-tested designs from other tools
for f in sorted(glob.glob(str(REPO / "bench/out/boltz/*/pred/boltz_results_yaml/predictions/*/*_model_0.cif"))):
    i = Path(f).parent.name
    if i not in meth.index: continue
    s = MMCIFParser(QUIET=True).get_structure("x", f); pdb = tmp / f"{i}.pdb"; io = PDBIO(); io.set_structure(s); io.save(str(pdb))
    try:
        cls, seqs = design(model, pdb, 2)
    except Exception: continue
    nat = meth.loc[i, "binder_seq"]
    if len(nat) != len(cls): continue
    for lab, ss in (("native", [nat]), ("mpnn_redesign", seqs)):
        for c in ("interface", "core", "surface", "all"):
            idx = np.arange(len(cls)) if c == "all" else np.where(cls == c)[0]
            if len(idx): rows.append(dict(source="other tools (" + meth.loc[i, "m"] + ")", method=meth.loc[i, "m"], kind=lab, cls=c, **comp(["".join(x[j] for j in idx) for x in ss])))
# (b) our diffusion backbones, several temperatures
D = glob.glob(str(REPO / "out/funnel/pdl1/gen/run/*/seed_*/predictions/converted_pdbs"))[0]
for T in (0.0001, 0.1, 0.3):
    for p in sorted(Path(D).glob("*.pdb"))[:40]:
        cls, seqs = design(model, p, 2, T)
        for c in ("interface", "core", "surface", "all"):
            idx = np.arange(len(cls)) if c == "all" else np.where(cls == c)[0]
            if len(idx): rows.append(dict(source="PXDesign diffusion backbones", method="pxdesign", kind=f"mpnn_redesign T={T}", cls=c, **comp(["".join(x[j] for j in idx) for x in seqs])))
R = pd.DataFrame(rows); R.to_csv(OUT, index=False)
def agg(g): return pd.Series(dict(KE=(g.KE * g.n_res).sum() / g.n_res.sum(), aromatic=(g.aromatic * g.n_res).sum() / g.n_res.sum(), residues=int(g.n_res.sum())))
print(R.groupby(["source", "kind", "cls"]).apply(agg).round(3).unstack("cls")[["KE", "aromatic"]].to_string())
o = R[R.source.str.startswith("other")]
print("\nper-method, ALL positions: native vs MPNN-redesign on the same backbones (Lys+Glu / aromatic)")
print(o[o.cls == "all"].groupby(["method", "kind"]).apply(agg).round(3).unstack("kind")[["KE", "aromatic"]].to_string())
