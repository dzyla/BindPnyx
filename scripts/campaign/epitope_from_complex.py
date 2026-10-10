#!/usr/bin/env python3
"""Contact-derived epitope on one chain of a complex, from the structure itself (not from convention).

  python scripts/campaign/epitope_from_complex.py COMPLEX.pdb TARGET_CHAIN PARTNER [PARTNER ...] [--cutoff 4.5] [--top 5]
         [--compare OTHER.pdb OTHER_TARGET_CHAIN OTHER_PARTNER [OTHER_PARTNER ...]]

Prints, for the target chain, every residue within --cutoff A (heavy atoms) of the partner chain(s), ordered by atom-contact count; the
geometry of that face (maximum C-alpha span, radius of gyration - a binder of 60-85 residues covers roughly 20-28 A, so a face much larger
than that needs more than one site); and a suggested hotspot list in the `"E35"` form `funnel/targets/*.json` expects (PDB number + letter).
With --compare it adds the overlap with a second, independent partner (for example a blocking antibody) and says which suggested hotspots
both partners touch. Hotspots remain a human decision: look at the table and the geometry before using the suggestion.
Pick the partner chains that are the biological contact: a chain that only touches the target through the crystal lattice has few contacts.
"""
import argparse
import sys
from collections import defaultdict

import numpy as np

THREE = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
         "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}


def read_chains(path):
    """{chain: {"atoms": [(resnum, letter, atom_name, xyz)], }} - standard residues, first altloc, heavy atoms, no insertion codes."""
    out = defaultdict(list)
    skipped_ins = 0
    for line in open(path):
        if not line.startswith("ATOM") or line[16] not in " A" or line[17:20] not in THREE:
            continue
        if line[26] != " ":
            skipped_ins += 1
            continue
        name = line[12:16].strip()
        if name.startswith("H") or line[76:78].strip() == "H":
            continue
        out[line[21]].append((int(line[22:26]), THREE[line[17:20]], name, (float(line[30:38]), float(line[38:46]), float(line[46:54]))))
    if skipped_ins:
        print(f"note: {skipped_ins} atoms with insertion codes were skipped", file=sys.stderr)
    return out


def contacts(target_atoms, partner_atoms, cutoff=4.5):
    """{resnum: (letter, n_atom_contacts, min_distance)} for target residues with any atom within `cutoff` of the partner."""
    if not partner_atoms:
        return {}
    P = np.array([a[3] for a in partner_atoms], float)
    res = {}
    for rn, letter, _, xyz in target_atoms:
        d = np.linalg.norm(P - np.asarray(xyz), axis=1)
        n = int((d < cutoff).sum())
        if n:
            prev = res.get(rn, (letter, 0, 1e9))
            res[rn] = (letter, prev[1] + n, min(prev[2], float(d.min())))
    return res


def geometry(target_atoms, resnums):
    """Maximum pairwise C-alpha distance and radius of gyration (A) of the listed residues."""
    ca = {rn: np.asarray(x) for rn, _, name, x in target_atoms if name == "CA"}
    pts = np.array([ca[r] for r in resnums if r in ca])
    if len(pts) < 2:
        return 0.0, 0.0
    span = max(float(np.linalg.norm(a - b)) for i, a in enumerate(pts) for b in pts[i + 1:])
    return span, float(np.sqrt(((pts - pts.mean(axis=0)) ** 2).sum(axis=1).mean()))


def analyse(path, target, partners, cutoff=4.5):
    chains = read_chains(path)
    if target not in chains:
        raise SystemExit(f"{path}: target chain {target!r} not found (chains: {sorted(chains)})")
    missing = [p for p in partners if p not in chains]
    if missing:
        raise SystemExit(f"{path}: partner chain(s) {missing} not found (chains: {sorted(chains)})")
    partner_atoms = [a for p in partners for a in chains[p]]
    return chains[target], contacts(chains[target], partner_atoms, cutoff)


def suggest(res, top=5):
    """Hotspot strings 'E35' for the `top` residues with the most contacts, in sequence order."""
    best = sorted(res, key=lambda r: -res[r][1])[:top]
    return [f"{res[r][0]}{r}" for r in sorted(best)]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("pdb"); ap.add_argument("target"); ap.add_argument("partners", nargs="+")
    ap.add_argument("--cutoff", type=float, default=4.5); ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--compare", nargs="+", metavar="PDB CHAIN PARTNER", help="second structure: PDB TARGET_CHAIN PARTNER [PARTNER ...]")
    a = ap.parse_args(argv)
    atoms, res = analyse(a.pdb, a.target, a.partners, a.cutoff)
    total = sum(v[1] for v in res.values())
    print(f"{a.pdb}: chain {a.target} vs {','.join(a.partners)}: {len(res)} residues, {total} atom contacts (<= {a.cutoff} A)")
    for r in sorted(res, key=lambda r: -res[r][1]):
        let, n, d = res[r]
        print(f"  {let}{r:<5d} contacts={n:<4d} min_dist={d:.2f}")
    span, rg = geometry(atoms, list(res))
    print(f"face geometry: max C-alpha span {span:.1f} A, radius of gyration {rg:.1f} A")
    hs = suggest(res, a.top)
    print(f'suggested hotspots (verify by eye): "hotspots": {hs}'.replace("'", '"'))
    if a.compare:
        if len(a.compare) < 3:
            raise SystemExit("--compare needs PDB TARGET_CHAIN PARTNER [PARTNER ...]")
        _, res2 = analyse(a.compare[0], a.compare[1], a.compare[2:], a.cutoff)
        s1, s2 = set(res), set(res2)
        both = sorted(s1 & s2)
        print(f"\n{a.compare[0]}: {len(res2)} residues; shared with the first: {len(both)}; Jaccard {len(both) / max(1, len(s1 | s2)):.2f}")
        print("suggested hotspots touched by BOTH partners:", [h for h in hs if int(h[1:]) in s2] or "none")
    return 0


if __name__ == "__main__":
    sys.exit(main())
