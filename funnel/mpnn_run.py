"""ProteinMPNN on complexes: target chain fixed (A), binder chain (B) redesigned. Run in the pxd env with PYTHONPATH=<repo>.

usage: mpnn_run.py PDB_DIR N_SEQS OUT_CSV [weights=soluble] [temperature=0.1] [bias=none|iface|iface:SCALE] [scope=full|interface[:CUTOFF]]

bias=iface adds a position-specific logit bias on binder residues whose CA lies within 10 A of a target CA:
  F,Y +1.0, W +0.5, L/I/V +0.5, M +0.25, K/E/D/N/Q -0.5 (all multiplied by SCALE, default 1).
  Measured on PD-L1 backbones this raises interface aromatics from ~3.6% to ~7% (real PD-L1 binders: 6.6%) while the rest stays polar.
scope=interface redesigns ONLY binder residues with a CA within CUTOFF (default 10) A of a target CA; every other binder residue keeps its identity
  (use for dock-then-redesign of an existing scaffold, or for interface-only cycling)."""
import os, sys, numpy as np, pandas as pd
from pathlib import Path

d, n, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
w = sys.argv[4] if len(sys.argv) > 4 else "soluble"; T = sys.argv[5] if len(sys.argv) > 5 else "0.1"
bias = sys.argv[6] if len(sys.argv) > 6 else "none"; scope = sys.argv[7] if len(sys.argv) > 7 else "full"
scale = float(bias.split(":")[1]) if ":" in bias else 1.0; bias = bias.split(":")[0]
cutoff = float(scope.split(":")[1]) if ":" in scope else 10.0; scope = scope.split(":")[0]
names = sorted(p.stem for p in Path(d).glob("*.pdb"))

def chain_resnums(pdb, chain):
    """Residue numbers (with insertion code appended if any) of the residues carrying a CA atom, in file order."""
    seen, out_ = set(), []
    for l in open(pdb):
        if l.startswith("ATOM") and l[12:16].strip() == "CA" and l[21] == chain:
            k = l[22:26].strip() + l[26].strip()
            if k not in seen: seen.add(k); out_.append(k)
    return out_

if bias == "none" and scope == "full":                       # unchanged behaviour: the vendored design_binder
    from ml_collections import ConfigDict
    from pxdbench.tools.protmpnn.main_mpnn import design_binder
    res = design_binder(d, names, n, ["B"], ["A"], ConfigDict({"weights": w, "rm_aa": "C", "temperature": T, "fix_interface": False}), if_print=False)
else:
    from colabdesign.mpnn import mk_mpnn_model
    from colabdesign.mpnn.model import aa_order
    BIAS = {k: v * scale for k, v in {**{"F": 1.0, "Y": 1.0, "W": 0.5, "L": .5, "I": .5, "M": .25, "V": .5}, **{k: -.5 for k in "KEDNQ"}}.items()}   # W and M lower: expression / oxidation risk
    model = mk_mpnn_model(backbone_noise=0.0, model_name="v_48_020", weights=w); res = []
    for nm in names:
        pdb = str(Path(d) / f"{nm}.pdb")
        model.prep_inputs(pdb_filename=pdb, chain="A,B", fix_pos="A", rm_aa="C")             # first pass: geometry
        ca = model._inputs["X"][:, 1]; nt = int(model._lengths[0]); nb = int(model._lengths[1])
        near = np.linalg.norm(ca[nt:, None] - ca[None, :nt], axis=-1).min(1) < (10.0 if scope == "full" else cutoff)
        if scope == "interface":
            rn = chain_resnums(pdb, "B")
            if len(rn) == nb:
                fixed = ",".join(f"B{r}" for r, nr in zip(rn, near) if not nr)
                model.prep_inputs(pdb_filename=pdb, chain="A,B", fix_pos="A" + ("," + fixed if fixed else ""), rm_aa="C")
            else:
                print(f"WARNING {nm}: residue numbering mismatch ({len(rn)} vs {nb}); designing the full binder", file=sys.stderr)
        b = np.zeros((nt + nb, 20))
        if bias == "iface":
            for a, v in BIAS.items(): b[nt:][near, aa_order[a]] += v
        model._inputs["bias"] = model._inputs["bias"] + b
        for i, s in enumerate(model.sample(num=n, temperature=float(T))["seq"]): res.append({"name": nm, "seq_idx": i, "sequence": s.split("/")[-1]})
pd.DataFrame(res).to_csv(out, index=False)
