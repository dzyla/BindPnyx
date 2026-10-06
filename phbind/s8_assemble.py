"""S8: assemble and validate the submission CSV (handoff section 1 contract C1-C7 + section 4 S8 rules). Every check raises.

Input table columns: name, sequence, block ('ph' | 'xreact' | 'affinity'), lineage (backbone-family id), rank (lower = better inside block).
Order: ph block first, then xreact, then affinity; inside each block lineages are interleaved round-robin (so an early lineage
failure cannot consume the whole screened set). Track 1 screens the first 20 rows in THIS order with no re-ranking.
"""
from __future__ import annotations
import re
import pandas as pd
AA = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+$")
BLOCKS = ["ph", "xreact", "affinity"]
MAX_LINEAGE_SHARE = 0.40
TRACK_SLOTS = {1: 40, 2: 20, 3: 20}

def interleave(df):
    out = []
    for b in BLOCKS:
        q = {k: g.sort_values("rank").to_dict("records") for k, g in df[df.block == b].groupby("lineage")}
        while any(q.values()):
            for k in sorted(q, key=lambda k: q[k][0]["rank"] if q[k] else 1e9):    # best remaining head first within each round
                if q[k]: out.append(q[k].pop(0))
    return pd.DataFrame(out)

def check_submission(sub: pd.DataFrame, track=1, lineage=None):
    assert list(sub.columns) == ["name", "sequence", "molecule_class"], f"C1 header: {list(sub.columns)}"
    assert len(sub) == TRACK_SLOTS[track], f"C4/C5: {len(sub)} rows, track {track} needs {TRACK_SLOTS[track]}"
    assert sub.name.is_unique, "duplicate names"
    for s in sub.sequence:
        assert 10 <= len(s) <= 250, f"C2 length {len(s)}"
        assert AA.match(s), "C3 non-canonical or lowercase residue"
        assert "C" not in s, "C7 cysteine"
        assert len(s) >= 60, "C6 floor: binders must be >= 60 aa"
    assert sub.sequence.is_unique, "S8: duplicate sequences under different names"
    if lineage is not None:
        share = pd.Series(list(lineage)).value_counts(normalize=True).max()
        assert share <= MAX_LINEAGE_SHARE + 1e-9, f"S8: one lineage holds {share:.0%} of the panel (cap {MAX_LINEAGE_SHARE:.0%})"
    return True

def assemble(cand: pd.DataFrame, track=1, mol="protein"):
    c = cand.drop_duplicates("sequence").copy()                       # 1. dedupe by sequence
    n = TRACK_SLOTS[track]; cap = int(MAX_LINEAGE_SHARE * n)
    c = c.sort_values(["block", "rank"], key=lambda s: s.map({b: i for i, b in enumerate(BLOCKS)}) if s.name == "block" else s)
    pick, count = [], {}
    for r in c.to_dict("records"):                                    # 2. lineage cap by backbone family
        if count.get(r["lineage"], 0) >= cap: continue
        pick.append(r); count[r["lineage"]] = count.get(r["lineage"], 0) + 1
    p = interleave(pd.DataFrame(pick)).head(n)                        # 4. order
    assert len(p) == n, f"only {len(p)} eligible designs for {n} slots after dedupe and lineage cap"
    sub = pd.DataFrame({"name": p["name"], "sequence": p["sequence"], "molecule_class": mol})
    check_submission(sub, track, p["lineage"]); return sub, p

if __name__ == "__main__":      # acceptance 11: format check on a dummy panel, and negative cases
    import random
    rng = random.Random(1); aas = "ADEFGHIKLMNPQRSTVWY"
    d = pd.DataFrame([dict(name=f"d{i}", sequence="".join(rng.choice(aas) for _ in range(70 + i % 40)), block=["ph", "affinity", "xreact"][i % 3], lineage=f"L{i % 6}", rank=i) for i in range(120)])
    sub, p = assemble(d); print("dummy panel OK:", len(sub), "rows; first 6 lineages:", list(p.lineage[:6]), "blocks:", p.block.value_counts().to_dict())
    bad = [("cysteine", lambda s: s.assign(sequence=s.sequence.str.replace("A", "C", n=1))), ("short", lambda s: s.assign(sequence=s.sequence.str[:30])),
           ("lower", lambda s: s.assign(sequence=s.sequence.str.lower())), ("header", lambda s: s.rename(columns={"name": "id"})), ("dup", lambda s: s.assign(sequence=s.sequence.iloc[0]))]
    for k, f in bad:
        try: check_submission(f(sub.copy())); raise SystemExit(f"FAIL: {k} was accepted")
        except AssertionError: print("rejects", k)
    one = d.assign(lineage="L0")
    try: assemble(one); raise SystemExit("FAIL: single-lineage panel accepted")
    except AssertionError as e: print("rejects single lineage:", str(e)[:60])
