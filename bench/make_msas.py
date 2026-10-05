"""One ColabFold-server MSA per benchmark target (via boltz --use_msa_server) -> msa/<target>.a3m.
Rows from boltz's csv are already aligned to the query, so the a3m has no insertions."""
import csv, subprocess, sys
import pandas as pd
from pathlib import Path
BOLTZ = Path("../.pxd/envs/boltz/bin/boltz").resolve()
m = pd.read_csv("inputs/manifest.csv"); Path("msa").mkdir(exist_ok=True)
for t, g in m.groupby("target"):
    y = Path(f"msa/{t}.yaml"); y.write_text(f"version: 1\nsequences:\n  - protein:\n      id: A\n      sequence: {g.target_seq.iloc[0]}\n")
    subprocess.run([str(BOLTZ), "predict", str(y), "--out_dir", "msa/out", "--use_msa_server", "--recycling_steps", "1",
                    "--sampling_steps", "5", "--override", "--accelerator", "gpu"], check=True, capture_output=True)
    rows = list(csv.DictReader(open(f"msa/out/boltz_results_{t}/msa/{t}_0.csv")))
    Path(f"msa/{t}.a3m").write_text("".join(f">{i}\n{r['sequence']}\n" for i, r in enumerate(rows)))
    print(t, len(rows), "sequences")
