"""S2: 1-seed grouped-ipSAE prescreen of every designed sequence against the INTACT trimer (Boltz-2, seed 101).

Batches of BATCH designs, each carrying the same reference designs (2 known to pass, 1 known to fail; see carriers()): scores are
only comparable inside a batch, and the carriers' ORDER (not just offset) is checked. First sequence of every backbone is screened
before any second sequence, so a partial run is still a uniform sample of backbones. A batch is finished when its CSV has every row.
A 1-seed number is a SCREENING statistic: never quote it as a design's score.
"""
import os, sys
from pathlib import Path
import pandas as pd
REPO = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(REPO))
from phbind import boltz_trimer as B
from phbind import config as _cfg
_P = _cfg.load()["prescreen"]
BATCH, SEED = _P["batch"], _P["seed"]
OUT = REPO / "out/phbind/s2"
CARRIERS_CSV = Path(os.environ.get("PHBIND_CARRIERS", REPO / "out/phbind/carriers.csv"))


def carriers():
    """Reference designs carried into EVERY batch (columns id,seq; ids must start with 'carrier_'). Use 2 designs known to pass and 1 known to fail
    on your own measurements; they are how batch effects are detected. The file is data and is never committed."""
    if not CARRIERS_CSV.exists():
        raise FileNotFoundError(f"{CARRIERS_CSV}: provide >=3 reference designs (id,seq) via $PHBIND_CARRIERS")
    c = pd.read_csv(CARRIERS_CSV)[["id", "seq"]]
    assert len(c) >= 3 and c.id.str.startswith("carrier_").all(), "carrier ids must start with 'carrier_'"
    return c


def main(src=REPO / "out/phbind/designs_all.csv"):
    """Order: first sequence of every backbone before any second sequence; WITHIN that, a seeded shuffle so every batch is a stratified
    random sample of (hotspot set, length) and any early look is representative (the first ordering sorted by run name and batch 0 held only core6).
    Batches already on disk are skipped; an interrupted batch is re-derived identically and resumed from its real predictions."""
    import hashlib
    d = pd.read_csv(src); d["k"] = d.id.str.rsplit("_", n=1).str[1].astype(int)
    assert d.id.is_unique and not d.id.str.startswith('carrier_').any()
    OUT.mkdir(parents=True, exist_ok=True); car = carriers()
    done = {i for f in OUT.glob("batch_*.csv") for i in pd.read_csv(f).id}
    d = d[~d.id.isin(done)].copy(); d["h"] = d.id.map(lambda x: hashlib.md5(("s2" + x).encode()).hexdigest())
    d = d.sort_values(["k", "h"], kind="stable").reset_index(drop=True)
    n0 = len(list(OUT.glob("batch_*.csv")))
    for b in range((len(d) + BATCH - 1) // BATCH):
        part = d.iloc[b * BATCH:(b + 1) * BATCH]; n = n0 + b; f = OUT / f"batch_{n:03d}.csv"
        df = pd.concat([part[["id", "seq"]], car], ignore_index=True); od = OUT / f"b{n:03d}"
        B.run_seed(df, od, SEED); r = B.score_seed(df, od, SEED)
        r = r.merge(d[["id", "bb", "set", "L", "run", "k"]], on="id", how="left"); r["batch"] = n; r.to_csv(f, index=False)
        c = r[r.id.isin(car.id)].set_index("id").ipsae_min
        print(f"batch {n}: {len(part)} designs; carriers (min-ipSAE): " + ", ".join(f"{k}={v:.3f}" for k, v in c.items()), flush=True)
        import shutil; shutil.rmtree(od / "todo", ignore_errors=True)
    print("S2 done")

if __name__ == "__main__": main()


SCREEN_CUT = _P["cut"]   # 1-seed grouped ipSAE(min) cut taking designs to the 5-seed gate. Lowered 0.45 -> 0.35 after SHARED_BRIEFING 9.6: with 7 sibling positives the recall CI is [0.59, 1.0],
                    # and every PXDesign backbone is a new lineage (the cut is an economy only inside a lineage already understood). SHARED_BRIEFING 1.2: 7/7 sibling seed-unanimous designs >= 0.69 at one seed,
                    # cut 0.45 kept 7/7 and dropped 46%; reproduced here on the 20 known designs (5/5 gate-passers >= 0.63 at each of 3 seeds, n=5, wide interval).
def shortlist(cut=SCREEN_CUT):
    a = pd.concat([pd.read_csv(f) for f in sorted(OUT.glob("batch_*.csv"))], ignore_index=True)
    a = a[~a.id.str.startswith('carrier_')]
    s = a[a.ipsae_min >= cut].sort_values("ipsae_min", ascending=False)
    return a, s
