"""SolubleMPNN on dimer-target + 1 binder complexes. Target chains fixed, cysteine banned (rm_aa=C), T=0.1.
usage: mpnn_dimer.py PDB_DIR N_SEQS OUT_CSV COND_CHAINS BINDER_CHAIN   (run in the pxd env with PYTHONPATH=<repo>)"""
import sys, pandas as pd
from pathlib import Path
from ml_collections import ConfigDict
from pxdbench.tools.protmpnn.main_mpnn import design_binder
d, n, out, cond, bind = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4].split(","), sys.argv[5]
W = "soluble"          # SolubleMPNN only. Original ProteinMPNN weights are refused (user decision 2026-10-06: it gave poor binders).
from colabdesign.mpnn import model as _m
assert "weights_soluble" in __import__("inspect").getsource(_m), "colabdesign no longer exposes the soluble weights"
names = sorted(p.stem for p in Path(d).glob("*.pdb"))
res = design_binder(d, names, n, [bind], cond, ConfigDict({"weights": W, "rm_aa": "C", "temperature": "0.1", "fix_interface": False}), if_print=False)
df = pd.DataFrame(res); assert len(df) == n * len(names), (len(df), n, len(names))
assert not df.sequence.str.contains("C").any(), "cysteine in a designed sequence"
df["mpnn_model"] = "SolubleMPNN_v_48_020"
df.to_csv(out, index=False); print("mpnn", len(df), "sequences for", len(names), "backbones")
