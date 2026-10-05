"""Production check of the fused LayerNorm kernel: identical 300-design Protenix-fast job run (1) fused, (2) fused again, (3) with the kernel disabled.
(1) vs (2) = run-to-run noise of the pipeline; (1) vs (3) = effect of the kernel. Moves the .so aside for run (3) and ALWAYS restores it."""
import sys, json, shutil, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); import common, protenix
from scipy.stats import spearmanr
LN = Path(protenix.__file__).parent / "model" / "layer_norm"; so = next(LN.glob("fast_layer_norm_cuda_v2*.so")); off = so.with_suffix(".so.disabled")
t = common.load_target("pdl1"); s = pd.read_csv(common.REPO / "out/funnel/pdl1/screen.csv").dropna(subset=["fast_ipsae"]).sample(300, random_state=0)[["id", "seq"]]
def run(tag):
    r = common.protenix_fold(s, t, common.REPO / "out/prod_test" / tag, arm="fast"); return r.set_index("id"), r.attrs["seconds"]
try:
    a, ta = run("fused_a"); b, tb = run("fused_b")
    so.rename(off)
    try: c, tc = run("torch_ln")
    finally: off.rename(so)
finally:
    if off.exists() and not so.exists(): off.rename(so)
print(f"wall: fused {ta:.0f}s / {tb:.0f}s, torch layer_norm {tc:.0f}s  (300 designs; includes process start-up)")
def cmp(x, y, n):
    d = pd.concat([x.fast_ipsae, y.fast_ipsae], axis=1, keys=["x", "y"]).dropna(); e = (d.x - d.y).abs()
    print(f"  {n:34s} n={len(d)}  max|diff| {e.max():.4f}  mean|diff| {e.mean():.4f}  Spearman {spearmanr(d.x, d.y)[0]:.4f}  flips at 0.5: {int(((d.x >= .5) != (d.y >= .5)).sum())}")
cmp(a, b, "fused vs fused (same job twice)"); cmp(a, c, "fused vs torch layer_norm"); cmp(b, c, "fused(2nd) vs torch layer_norm")
