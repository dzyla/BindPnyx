"""Does ProteinMPNN weight choice or a position-specific logit bias change the composition of designed binders?
CPU only (JAX_PLATFORMS=cpu). usage: mpnn_bias_test.py PDB_DIR N_BACKBONES OUT_CSV   (chains: target A fixed, binder B)"""
import os, sys, numpy as np, pandas as pd
os.environ.setdefault("JAX_PLATFORMS", "cpu")
from pathlib import Path
from colabdesign.mpnn import mk_mpnn_model
from colabdesign.mpnn.model import aa_order

D, NB, OUT = sys.argv[1], int(sys.argv[2]), sys.argv[3]
names = sorted(Path(D).glob("*.pdb"))[:NB]
HYD = {"F": 1.0, "W": 1.0, "Y": 1.0, "L": .5, "I": .5, "M": .5, "V": .5}; POLAR = {k: -.5 for k in "KEDNQ"}
def bias_for(cond, iface):
    b = np.zeros_like(iface, dtype=float)[:, None].repeat(20, 1)
    if cond.endswith("iface_bias"):
        for a, v in {**HYD, **POLAR}.items(): b[iface, aa_order[a]] += v
    elif cond.endswith("global_bias"):
        for a, v in {"K": -.5, "E": -.5, "F": .3, "W": .3, "Y": .3, "L": .3, "I": .3, "M": .3}.items(): b[:, aa_order[a]] += v
    return b
CONDS = {"soluble (fork default)": ("soluble", "plain"), "original weights": ("original", "plain"),
         "original + global bias": ("original", "global_bias"), "original + interface bias": ("original", "iface_bias"), "soluble + interface bias": ("soluble", "iface_bias")}
rows = []
for cond, (w, kind) in CONDS.items():
    m = mk_mpnn_model(backbone_noise=0.0, model_name="v_48_020", weights=w)
    for p in names:
        m.prep_inputs(pdb_filename=str(p), chain="A,B", fix_pos="A", rm_aa="C")
        X = m._inputs["X"]; ca = X[:, 1]; nt = int(m._lengths[0]); nb = int(m._lengths[1])
        d = np.linalg.norm(ca[nt:, None] - ca[None, :nt], axis=-1).min(1); iface = np.zeros(nt + nb, bool); iface[nt:] = d < 10.0
        m._inputs["bias"] = m._inputs["bias"] + bias_for(kind, iface)
        out = m.sample(num=4, temperature=0.1)
        for s in out["seq"]:
            s = s.split("/")[-1]; ip = iface[nt:]
            rows.append(dict(cond=cond, backbone=p.stem, seq=s, n_iface=int(ip.sum()), iface_seq="".join(c for c, f in zip(s, ip) if f)))
    print(cond, "done", flush=True)
pd.DataFrame(rows).to_csv(OUT, index=False)
R = pd.DataFrame(rows)
def comp(ss):
    t = "".join(ss); n = len(t); f = lambda a: sum(t.count(x) for x in a) / n
    return dict(KE=f("KE"), aromatic=f("FWY"), hydrophobic=f("AILMFVW"), polar_DNQ=f("DNQ"))
tab = []
for c, g in R.groupby("cond", sort=False):
    a, b = comp(g.seq), comp(g.iface_seq); tab.append(dict(cond=c, n=len(g), **{f"all_{k}": v for k, v in a.items()}, **{f"iface_{k}": v for k, v in b.items()}))
print(pd.DataFrame(tab).set_index("cond").round(3).T.to_string())
