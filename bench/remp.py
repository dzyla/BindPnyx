"""Re-run ProteinMPNN on an existing set of PXDesign backbones under different settings,
so sequence-stage choices can be compared on identical backbones.
usage (pxd env, PYTHONPATH=repo): remp.py CONVERTED_PDB_DIR OUT_CSV"""
import sys, pandas as pd
from pathlib import Path
from ml_collections import ConfigDict
from pxdbench.tools.protmpnn.main_mpnn import design_binder

CONDITIONS = {  # name -> (weights, temperature, seqs/backbone)
    "soluble_T0.1": ("soluble", "0.1", 4),     # what this fork's run_campaign.sh uses
    "original_T0.1": ("original", "0.1", 4),   # same temperature, vanilla weights
    "original_greedy": ("original", "0.0001", 1),  # upstream PXDesign default
}

def main(pdb_dir, out):
    names = sorted(p.stem for p in Path(pdb_dir).glob("*.pdb"))
    rows = []
    for cond, (w, T, n) in CONDITIONS.items():
        res = design_binder(pdb_dir, names, n, ["B"], ["A"],
                            ConfigDict({"weights": w, "rm_aa": "C", "temperature": T, "fix_interface": False}), if_print=False)
        rows += [dict(cond=cond, **r) for r in res]
        print(cond, len(res), flush=True)
    pd.DataFrame(rows).to_csv(out, index=False)

if __name__ == "__main__":
    main(*sys.argv[1:3])
