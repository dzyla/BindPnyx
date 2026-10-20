#!/usr/bin/env python
"""BLOCK 5 - band-holding rate per case, grouped by the role each case was BUILT for.

Why group by build-time role rather than by score. A batch usually exists to answer a question
about a category - "do the rows we never measured hold the band?", "do the permutations stay at
zero?" - and a table sorted by rate buries the category. The roles come from the provenance.csv
that block 1 wrote, so the grouping is the one declared before any number existed and cannot be
drawn around the answer after the fact.

Read the control group first. Positives at or near full rate, negatives at or near zero. If they
do not read that way, stop: the batch is not evidence and no other group in the table means
anything.

--prior takes a CSV of name,rate recorded in an EARLIER batch and prints it beside the new rate.
It is printed, never subtracted: two batches are two draws under two bands, so the comparison to
make is "same neighbourhood or not". Run block 4 on the two batches before you believe even that.

Usage:
  group_report.py --run <shim_dir> [--batch <batch_dir>] [--prior prior.csv] [--csv out.csv]
"""
import argparse
import csv
import importlib.util
import json
import os
import sys

import band_keys

CAMPAIGN = os.environ["CAMPAIGN"]  # campaign workspace root; no default, so a
# misconfigured clone fails here rather than silently scoring against another tree
CONTROL_FIRST = ("control",)


def load_in_band():
    """Reuse the scorer's own band test - a second copy of it would be a second definition."""
    spec = importlib.util.spec_from_file_location(
        "rbr", os.path.join(CAMPAIGN, "scripts", "rank_band_rate.py"))
    mod = importlib.util.module_from_spec(spec)
    argv, sys.argv = sys.argv, [sys.argv[0]]
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.argv = argv
    return mod.in_band


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="scored shim directory (block 3 output)")
    ap.add_argument("--batch", default=None, help="batch dir holding provenance.csv")
    ap.add_argument("--prior", default=None, help="CSV name,rate from an earlier batch")
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()

    batch = a.batch or a.run.replace("_shim", "")
    in_band = load_in_band()
    terms = json.load(open(os.path.join(a.run, "joint_terms.json")))
    bands = json.load(open(os.path.join(a.run, "joint_bands.json")))["bands"]

    prov = {}
    pp = os.path.join(batch, "provenance.csv")
    if os.path.exists(pp):
        prov = {r["name"]: r for r in csv.DictReader(open(pp))}
    else:
        print("note: no provenance.csv under %s - every case will read as 'unknown'" % batch)

    prior = {}
    if a.prior:
        for r in csv.DictReader(open(a.prior)):
            k = next((c for c in r if c.lower() in ("name", "design", "case")), None)
            v = next((c for c in r if "rate" in c.lower() or "hits" in c.lower()), None)
            if k and v:
                prior[r[k]] = r[v]

    rows = []
    for case, entry in terms.items():
        ps = entry.get("per_sample") or []
        hits = sum(1 for s in ps if in_band(s, bands))
        p = prov.get(case, {})
        rows.append(dict(case=case, role=p.get("role", "unknown"), parent=p.get("parent", ""),
                         length=p.get("length", ""), n=len(ps), hits=hits,
                         rate=round(hits / float(len(ps)), 3) if ps else "",
                         prior=prior.get(case, "")))

    print("band: anchor [%s, %s]  core >= %s  depth %s  (from this batch's %s positive controls)"
          % (band_keys.get("anchor_lo", bands), band_keys.get("anchor_hi", bands),
             band_keys.get("core_min", bands),
             bands["depth_band_Z"], bands.get("n_agonist_controls", "?")))
    if prior:
        print("`prior` is a rate from another batch: printed, never subtracted - run block 4 "
              "on the two batches first.")
    print()

    roles = sorted({r["role"] for r in rows},
                   key=lambda r: (r not in CONTROL_FIRST, r))
    for role in roles:
        sel = sorted((r for r in rows if r["role"] == role), key=lambda r: (-r["hits"], r["case"]))
        print("=== %s (%d)" % (role, len(sel)))
        print("%-28s %6s %9s %9s %s" % ("case", "len", "rate", "prior", "parent"))
        for r in sel:
            print("%-28s %6s %9s %9s %s" % (r["case"], r["length"], "%d/%d" % (r["hits"], r["n"]),
                                            r["prior"] or "-", r["parent"]))
        print()

    if a.csv:
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(sorted(rows, key=lambda r: (r["role"], -r["hits"], r["case"])))
        print("wrote %s" % a.csv)


if __name__ == "__main__":
    main()
