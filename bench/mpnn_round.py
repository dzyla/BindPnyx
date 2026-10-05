"""One redesign step: ProteinMPNN on predicted complexes (chain A target fixed, chain B binder redesigned).
usage (pxd env, PYTHONPATH=repo): mpnn_round.py PDB_DIR N_SEQS OUT_CSV [weights] [temperature]"""
import sys, pandas as pd
from pathlib import Path
from ml_collections import ConfigDict
from pxdbench.tools.protmpnn.main_mpnn import design_binder
d, n, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
w = sys.argv[4] if len(sys.argv) > 4 else "soluble"; T = sys.argv[5] if len(sys.argv) > 5 else "0.1"
names = sorted(p.stem for p in Path(d).glob("*.pdb"))
res = design_binder(d, names, n, ["B"], ["A"], ConfigDict({"weights": w, "rm_aa": "C", "temperature": T, "fix_interface": False}), if_print=False)
pd.DataFrame(res).to_csv(out, index=False)
