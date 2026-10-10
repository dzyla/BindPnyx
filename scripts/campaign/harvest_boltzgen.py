#!/usr/bin/env python3
"""Extract designed binder sequences from a BoltzGen workbench into an `id,seq` CSV (the form `run_funnel.py --designs-csv` reads).

  python scripts/campaign/harvest_boltzgen.py WORKBENCH ID_PREFIX OUT.csv --target-len 115 [--min-len 20 --max-len 300]

Reads WORKBENCH/intermediate_designs_inverse_folded/design_spec_*.cif and keeps, per design, the chain whose length is NOT the target's.
Needs gemmi (present in the BoltzGen environment). It prints how many designs it kept and skipped and exits non-zero if it kept none: a
short or empty CSV must never be silent (an earlier version capped binder length at 130 aa and quietly dropped every 140-aa design).
Use only the `design` and `inverse_folding` steps of BoltzGen and score the sequences with the common evaluator; BoltzGen's own scores are
not comparable across generators.
"""
import argparse
import csv
import glob
import os
import sys

AA3 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L",
       "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}


def binder_sequence(chains, target_len, min_len, max_len):
    """`chains` = {name: sequence}. The binder is the chain whose length is not the target's and lies in [min_len, max_len]; None if ambiguous/absent."""
    cand = [s for s in chains.values() if len(s) != target_len and "X" not in s and min_len <= len(s) <= max_len]
    return cand[0] if len(cand) == 1 else None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("workbench"); ap.add_argument("prefix"); ap.add_argument("out")
    ap.add_argument("--target-len", type=int, required=True)
    ap.add_argument("--min-len", type=int, default=20); ap.add_argument("--max-len", type=int, default=300)
    a = ap.parse_args(argv)
    try:
        import gemmi
    except ImportError:
        raise SystemExit("gemmi is required (it is in the BoltzGen environment)")
    files = sorted(glob.glob(os.path.join(a.workbench, "intermediate_designs_inverse_folded", "design_spec_*.cif")))
    rows, skipped = [], 0
    for f in files:
        try:
            st = gemmi.read_structure(f)
        except Exception:
            skipped += 1
            continue
        chains = {ch.name: "".join(AA3.get(r.name, "X") for r in ch) for ch in st[0]}
        s = binder_sequence(chains, a.target_len, a.min_len, a.max_len)
        if s is None:
            skipped += 1
            continue
        rows.append({"id": f"{a.prefix}_{os.path.basename(f).split('_')[-1].split('.')[0]}", "seq": s})
    uniq = len({r["seq"] for r in rows})
    print(f"{a.prefix}: {len(files)} files, kept {len(rows)}, unique {uniq}, lengths {sorted({len(r['seq']) for r in rows})}, skipped {skipped}")
    if not rows:
        print("nothing kept: check --target-len and the length bounds before concluding the run failed", file=sys.stderr)
        return 1
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, ["id", "seq"]); w.writeheader(); w.writerows(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
