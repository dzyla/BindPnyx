"""Add fastPISA columns (pisa_*, pisa_flags) to finished run tables in place. Idempotent; CPU only.
usage: add_pisa_to_runs.py TARGET csv [csv ...]   (each csv needs id and b_cif columns)"""
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent)); import common, run_funnel
t = common.load_target(sys.argv[1])
for f in sys.argv[2:]:
    d = pd.read_csv(f); d = d.drop(columns=[c for c in d.columns if c.startswith("pisa_")])
    d = run_funnel.add_pisa(d, t); d.to_csv(f, index=False)
    print(f, len(d), "rows;", int((d.pisa_flags != "").sum()), "flagged;", d.pisa_flags.str.split(";").explode().value_counts().to_dict())
