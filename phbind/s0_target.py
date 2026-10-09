"""S0: build and verify the TNF-alpha target contract (the project's target contract).

Every guard raises; nothing warns. Output: phbind/target/target_manifest.json plus copies of the
organiser PDBs, dimers and NUL-fixed MSAs. Nothing downstream may read a target file directly.

Run:  TNF_BUNDLE=<bundle dir> PYTHONPATH=$(pwd) .pxd/envs/pxd/bin/python phbind/s0_target.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser, ShrakeRupley

import os
if "TNF_BUNDLE" not in os.environ:
    raise SystemExit("set $TNF_BUNDLE to the campaign bundle (targets/, msa/, tables/, code/trimer.py)")
BUNDLE = Path(os.environ["TNF_BUNDLE"])
OUT = Path(__file__).resolve().parent / "target"
HOTSPOTS = ["B87", "B88", "B90", "C21", "C33", "C65", "C67", "C113", "C115", "C144", "C145", "C146"]
P1X_FACE = {"B": [75, 77, 79, 82, 84, 86, 87, 88, 90, 91, 92, 127, 135, 137],
            "C": [17, 18, 21, 23, 29, 32, 33, 34, 35, 65, 66, 67, 113, 115, 143, 144, 145, 146, 147]}
AA3 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G",
       "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
       "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}
sys.path.insert(0, str(Path(__file__).resolve().parent))
import trimer as T  # noqa: E402  (trimer-aware primitives: sequences, numbering maps)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def residues(path):
    s = PDBParser(QUIET=True).get_structure("x", path)
    return {c.id: {r.id[1]: r for r in c if r.id[0] == " "} for c in s[0]}, s


def rsa_per_residue(path, chains):
    """Shrake-Rupley, probe 1.4 A, 200 points/atom; SASA per (chain, resnum)."""
    s = PDBParser(QUIET=True).get_structure("x", path)
    for ch in list(s[0]):
        if ch.id not in chains:
            s[0].detach_child(ch.id)
    ShrakeRupley(probe_radius=1.4, n_points=200).compute(s[0], level="R")
    return {(c.id, r.id[1]): r.sasa for c in s[0] for r in c if r.id[0] == " "}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    src = {"human": BUNDLE / "targets/tnf_target_human.pdb", "mouse": BUNDLE / "targets/tnf_target_mouse.pdb",
           "human_dimerBC": BUNDLE / "targets/tnf_target_human_dimerBC.pdb",
           "mouse_dimerAC": BUNDLE / "targets/tnf_target_mouse_dimerAC.pdb",
           "msa_human": BUNDLE / "msa/msa_tnf_human_nulfix.a3m", "msa_mouse": BUNDLE / "msa/msa_tnf_mouse_nulfix.a3m"}
    dst = {k: OUT / v.name for k, v in src.items()}
    for k in src:
        shutil.copy(src[k], dst[k])
    manifest = {"files": {k: {"path": str(dst[k]), "sha256": sha(dst[k])} for k in dst}}

    # ---- sequences / lengths (TRAP: 471 vs 456)
    assert len(T.TNF_HUMAN) == 157 and len(T.TNF_MOUSE) == 156
    assert 3 * len(T.TNF_HUMAN) == 471 and 3 * len(T.TNF_MOUSE) == 468
    # ---- TRAP 3: NUL bytes, and the nulfix checksum recorded by the bundle
    for k in ("msa_human", "msa_mouse"):
        b = dst[k].read_bytes()
        assert b"\x00" not in b, f"NUL byte in {dst[k]}"
        lines = b.decode().splitlines()
        assert lines[1] == (T.TNF_HUMAN if k == "msa_human" else T.TNF_MOUSE), f"{k} query != construct sequence"
    rec = dict(l.split()[::-1] for l in (BUNDLE / "msa/NULFIX.sha256").read_text().splitlines())
    assert rec["msa/msa_tnf_human_nulfix.a3m"] == manifest["files"]["msa_human"]["sha256"]
    assert rec["msa/msa_tnf_mouse_nulfix.a3m"] == manifest["files"]["msa_mouse"]["sha256"]

    # ---- structure vs construct sequence; TRAP 1 (143 ASP in construct, LEU in structure), exactly one
    ch, _ = residues(dst["human"])
    assert sorted(ch) == ["A", "B", "C"]
    mism = {}
    for cid, res in ch.items():
        assert len(res) == 152, (cid, len(res))
        for n, r in res.items():
            if AA3[r.get_resname()] != T.TNF_HUMAN[n - 1]:
                mism.setdefault(n, set()).add((T.TNF_HUMAN[n - 1], AA3[r.get_resname()]))
    assert mism == {143: {("D", "L")}}, mism      # one position, all three chains
    # ---- TRAP 2: mouse numbering
    chm, _ = residues(dst["mouse"])
    for cid, res in chm.items():
        assert 73 not in res and len(res) == 148, cid
        assert T.author_to_local("mouse", 74) == 73 and T.local_to_author("mouse", 73) == 74
        for n, r in res.items():
            assert AA3[r.get_resname()] == T.TNF_MOUSE[T.author_to_local("mouse", n) - 1], (cid, n)

    # ---- hotspots: identity through the numbering, veto, P1x membership
    surf = {int(r["local_resnum"]): r for r in csv.DictReader(open(BUNDLE / "tables/tnf_surface_map.csv"))}
    veto = {n for n, r in surf.items() if r["hard_veto_trimer_buried"] == "True"}
    cls = {}
    for r in surf.values():
        cls[r["surface_class"]] = cls.get(r["surface_class"], 0) + 1
    assert cls.get("trimer_buried") == 45 and cls.get("rim") == 36 and cls.get("designable_outer") == 75, cls
    p1x = {n for n, r in surf.items() if r["epi_P1x_CROSS_REACTIVE_SUBPATCH"] == "True"}
    assert len(p1x) == 33, len(p1x)
    assert p1x == set(P1X_FACE["B"]) | set(P1X_FACE["C"]) and len(set(P1X_FACE["B"]) | set(P1X_FACE["C"])) == 33
    hs = []
    for h in HOTSPOTS:
        c, n = h[0], int(h[1:])
        assert n not in veto, f"hotspot {h} is trimer-buried"
        assert n in p1x and n in P1X_FACE[c], f"hotspot {h} not in its P1x face"
        assert n != 143
        hs.append({"id": h, "chain": c, "num": n, "aa": T.TNF_HUMAN[n - 1], "structure_aa": AA3[ch[c][n].get_resname()]})
        assert hs[-1]["aa"] == hs[-1]["structure_aa"]
    manifest["hotspots"] = hs

    # ---- dimer is epitope-neutral (recomputed, not trusted from the table)
    tri = rsa_per_residue(dst["human"], "ABC")
    dim = rsa_per_residue(dst["human_dimerBC"], "BC")
    dB = {k: abs(tri[k] - dim[k]) for c, ns in P1X_FACE.items() for n in ns if (k := (c, n)) in tri and k in dim}
    assert len(dB) >= 30
    assert max(dB.values()) < 1e-4, max(dB.items(), key=lambda kv: kv[1])
    gained = {k: dim[k] - tri[k] for k in dim if dim[k] - tri[k] >= 10.0}

    # ---- coldspots: table value cross-checked against my own SASA
    cold = [(r["chain"], int(r["resnum"])) for r in csv.DictReader(open(BUNDLE / "tables/tnf_dimer_exposed_face.csv"))
            if r["coldspot"] == "True"]
    assert len(cold) == 49, len(cold)
    mine = {k for k in gained}
    assert len(mine) == 52, len(mine)      # 52 artificially exposed residues
    assert set(cold) <= mine, sorted(set(cold) - mine)
    assert not set(cold) & {(h["chain"], h["num"]) for h in hs}
    assert not any(n in p1x and c in P1X_FACE and n in P1X_FACE[c] for c, n in cold)

    # ---- dimer file content
    dch, _ = residues(dst["human_dimerBC"])
    assert sorted(dch) == ["B", "C"] and sum(len(v) for v in dch.values()) == 304
    mch, _ = residues(dst["mouse_dimerAC"])
    assert sorted(mch) == ["A", "C"]

    manifest.update({
        "oracle_seq_human": T.TNF_HUMAN * 3, "oracle_seq_mouse": T.TNF_MOUSE * 3,
        "oracle_seq_human_len": 471, "oracle_seq_mouse_len": 468,
        "struct_seq_mismatches": {"143": ["ASP(construct)", "LEU(structure)"]},
        "mouse_author_to_local_rule": "author<=72 -> author; author>=74 -> author-1; 73 vacant",
        "p1x": sorted(p1x), "p1x_faces": P1X_FACE, "trimer_buried": sorted(veto),
        "surface_class_counts": cls, "coldspots": [{"chain": c, "num": n} for c, n in sorted(cold)],
        "n_target_residues_dimer": 304, "n_coldspots": 49, "n_artificially_exposed": len(mine),
        "max_dRSA_P1x_trimer_vs_dimer_A2": max(dB.values()),
        "chain_equivalent_human_to_mouse": T.CHAIN_EQUIVALENT["human_to_mouse"],
        "his_on_target": {n: T.TNF_HUMAN[n - 1] for n in range(1, 158) if T.TNF_HUMAN[n - 1] == "H"},
        "organiser_url": "https://proteinbase.com/structures/0f97cffa931a141af7107a1edb26cb46/human.pdb",
        "organiser_check": "ATOM/HETATM records identical to live organiser file (diff, 2026-10-06); bundle only adds REMARK lines",
    })
    (OUT / "target_manifest.json").write_text(json.dumps(manifest, indent=1))
    print("S0 OK: 12 hotspots, 49 coldspots, 304 dimer residues, 471-res oracle sequence, "
          f"max dRSA(P1x)={max(dB.values()):.2e}, 143 mismatch exactly one position")


if __name__ == "__main__":
    main()
