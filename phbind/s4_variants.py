"""S4 (mechanise): enumerate binder-side histidine placements on a VALIDATED backbone, from its own co-folded trimer structure.
Recipe from STAGE2_DIGEST s17.3 (sibling campaign, unvalidated by wet lab):
  interfacial = binder heavy atom within 5.0 A of a target heavy atom;  mutable = not G/P/C
  burial = 1 - relSASA(complex) >= 0.50
  AP1 veto: no His within 3.5 A of a target carboxylate O (Glu23/127/135/146, Asp143) -- that is the Challenge-1 direction, inverted
  M3: His-His pair, CB-CB <= 8.0 A, |i-j| >= 3, both burial 0.50-0.90      (primary; needs no target partner; species independent)
  M1: single His, burial 0.50-0.80, target Arg/Lys side-chain N within 6.0 A, cation at an epitope position {6,31,32,65,82,90,112,138}
Panel: 1-His (M1) , 2-His (M3 pairs), 3-His (a pair + an M1). Variants differ from the parent ONLY at the declared positions.
Protected positions are the His sites; S4 later re-folds children against the parent's in-batch score (anchored)."""
from __future__ import annotations
import itertools, sys
from pathlib import Path
import numpy as np
from Bio.PDB import MMCIFParser, ShrakeRupley
sys.path.insert(0, str(Path(__file__).resolve().parent)); import trimer as T
MAXASA = {"ALA": 129, "ARG": 274, "ASN": 195, "ASP": 193, "CYS": 167, "GLN": 225, "GLU": 223, "GLY": 104, "HIS": 224, "ILE": 197, "LEU": 201, "LYS": 236,
          "MET": 224, "PHE": 240, "PRO": 159, "SER": 155, "THR": 172, "TRP": 285, "TYR": 263, "VAL": 174}
CATION_SITES = {6, 31, 32, 65, 82, 90, 112, 138}; CARBOXYLATE = {23: "GLU", 127: "GLU", 135: "GLU", 146: "GLU", 143: "ASP"}
M3_CB_MAX = 8.0   # SHARED_BRIEFING s3.4: spec His-His distance is 8.0 A CB-CB; 6.5 A gave 16 pairs where 8.0 A gives 31
CARB_O = {"GLU": ("OE1", "OE2"), "ASP": ("OD1", "OD2")}

def enumerate_sites(cif, binder_seq):
    s = MMCIFParser(QUIET=True).get_structure("x", str(cif))[0]
    tab = T.chain_table(str(cif)); bid = [c["chain_id"] for c in tab if c["seq"] != T.TNF_HUMAN]; assert len(bid) == 1 and len(binder_seq) == tab[-1]["n_res"]
    bid = bid[0]; B = {r.id[1]: r for r in s[bid]}; assert "".join(T.AA3to1[r.get_resname()] for r in B.values()) == binder_seq
    ShrakeRupley(probe_radius=1.4, n_points=100).compute(s, level="R")
    tchains = [c for c in s if c.id != bid]; tat = [a for c in tchains for a in c.get_atoms() if a.element != "H"]; tx = np.array([a.coord for a in tat])
    car = np.array([a.coord for c in tchains for r in c if r.id[1] in CARBOXYLATE and r.get_resname() == CARBOXYLATE[r.id[1]] for a in r if a.get_id() in CARB_O[r.get_resname()]])
    cat = [(c.id, r.id[1], a.coord) for c in tchains for r in c if r.id[1] in CATION_SITES for a in r if a.get_id() in ("NH1", "NH2", "NE", "NZ")]
    out = {}
    for i, r in B.items():
        aa = T.AA3to1[r.get_resname()]; xyz = np.array([a.coord for a in r if a.element != "H"])
        d = np.linalg.norm(xyz[:, None] - tx[None], axis=-1).min()
        burial = 1 - r.sasa / MAXASA[r.get_resname()]
        cb = (r["CB"] if "CB" in r else r["CA"]).coord
        near_carb = float(np.linalg.norm(xyz[:, None] - car[None], axis=-1).min()) if len(car) else 99.0
        sc = np.array([a.coord for a in r if a.element != 'H' and a.get_id() not in ('N', 'C', 'O')] or [cb])
        near_cat = min((float(np.linalg.norm(sc - c[2], axis=-1).min()) for c in cat), default=99.0)   # closest side-chain atom to a cation N
        out[i] = dict(pos=i, aa=aa, interfacial=bool(d <= 5.0), mutable=aa not in "GPC", burial=float(burial), cb=cb, ap1=bool(near_carb <= 3.5), cation_dist=float(near_cat))
    return out

def candidates(sites):
    ok = {i: s for i, s in sites.items() if s["interfacial"] and s["mutable"] and s["burial"] >= 0.5 and not s["ap1"]}
    m3 = [(i, j) for i, j in itertools.combinations(sorted(ok), 2) if j - i >= 3 and ok[i]["burial"] <= 0.9 and ok[j]["burial"] <= 0.9
          and np.linalg.norm(ok[i]["cb"] - ok[j]["cb"]) <= M3_CB_MAX]
    m1 = [i for i, s in ok.items() if s["burial"] <= 0.8 and s["cation_dist"] <= 6.0]
    return ok, m3, m1

def variants(seq, m3, m1, max_pairs=6, max_single=3, max_triple=2):
    """Parent differs only at the declared positions; every variant is a distinct sequence of unchanged length with no Cys."""
    def mut(pos):
        l = list(seq)
        for p in pos: l[p - 1] = "H"
        return "".join(l)
    v = {}
    for p in m1[:max_single]: v[f"H{p}"] = (p,)
    for a, b in m3[:max_pairs]: v[f"H{a}H{b}"] = (a, b)
    # 3 His broke the fold 4 times in 6 in the sibling campaign (retained 2/6); 1 His kept 6/6, 2 His 15/20: keep triples to a token few
    for (a, b), p in list(itertools.product(m3[:3], m1[:2]))[:max_triple]:
        if p not in (a, b): v[f"H{a}H{b}H{p}"] = tuple(sorted((a, b, p)))
    out = {k: (mut(p), p) for k, p in v.items()}
    assert all(len(s) == len(seq) and "C" not in s and sum(x != y for x, y in zip(s, seq)) <= len(p) for s, p in out.values())
    return out

if __name__ == "__main__":
    import glob, pandas as pd
    cif = sys.argv[1]; seq = sys.argv[2]
    sites = enumerate_sites(cif, seq); ok, m3, m1 = candidates(sites)
    print(f"{len(sites)} binder residues; interfacial+mutable+buried+AP1-clean: {len(ok)}; M3 pairs {len(m3)}: {m3[:12]}; M1 singles {len(m1)}: {m1}")


CARBOXYL = {"GLU": ("OE1", "OE2"), "ASP": ("OD1", "OD2")}
def ap1_census(cif, binder_seq, his_positions, cutoff=3.5):
    """AP1 (His donating to a carboxylate = the Challenge-1 direction, inverted) re-checked on the REFOLDED variant, against
    carboxylates of the target AND of the binder: refolding moves a His next to a different carboxylate than the parent's geometry
    predicted (SHARED_BRIEFING s3.3: 5 variants passed a parent-geometry veto and failed this). Returns {pos: [(chain, resnum, resname)]}."""
    s = MMCIFParser(QUIET=True).get_structure("x", str(cif))[0]; tab = T.chain_table(str(cif))
    bid = [c["chain_id"] for c in tab if c["seq"] != T.TNF_HUMAN][0]; B = {r.id[1]: r for r in s[bid]}
    O = [(c.id, r.id[1], r.get_resname(), a.coord) for c in s for r in c if r.get_resname() in CARBOXYL for a in r if a.get_id() in CARBOXYL[r.get_resname()]]
    out = {}
    for p in his_positions:
        r = B[p]; N = [a.coord for a in r if a.get_id() in ("ND1", "NE2")]
        hits = sorted({(c, n, rn) for c, n, rn, xyz in O if not (c == bid and n == p) and any(np.linalg.norm(xyz - q) <= cutoff for q in N)})
        if hits: out[p] = hits
    return out
