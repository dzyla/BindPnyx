#!/usr/bin/env python
"""BLOCK 4 - compare two scored batches' BANDS before comparing anything about their designs.

The failure this exists to prevent, measured in this campaign. The same 17 cases, the same
controls, the same precomputed MSAs, one oracle, two seeds. One batch derived an anchor window of
1.96-3.41 A and the other 2.03-5.32 A, because one positive control's worst pose landed
differently. Every rate rose with the wider window, two composition-matched permutations of one
design went from 0/25 and 1/25 to 20/25 and 24/25, and that design's margin nearly halved - with
the design unchanged. Read as a result about designs, that is a conclusion about a window.

So: when two batches disagree, diff their bands first. If the bands differ materially, their rates
are not two draws of one quantity and the correct report is the band change, not a design ranking.

The permutation diagnostic (--case). For one case present in both batches, this prints the
per-pose distribution of the anchor term in each, with the fraction inside each batch's own
window. A case whose distribution barely moved while its in-window fraction jumped is proof that
the window moved and the case did not. The campaign's clean example: a permutation at 3.52-4.63 A
in one batch and 3.54-4.91 A in the other, holding the anchor term 0/25 in the first and 25/25 in
the second, because the ceiling moved 1.9 A.

Usage: band_diff.py <shim_a> <shim_b> [--case NAME] [--term anchor]
"""
import argparse
import json
import os
import statistics
import sys

import band_keys

# term names are resolved per target by band_keys, so this block works on a band file that
# names its anchor term after a different residue
TERMS = ("anchor_lo", "anchor_hi", "core_min")


POSITIVE = ("ctl_pos", "ctl_ago")


def pooled_positive(terms, field=None):
    """Every pose of every positive control, pooled - the rows the band edge is taken from."""
    out = []
    for c in terms:
        if not c.startswith(POSITIVE):
            continue
        for p in terms[c].get("per_sample", []):
            f = field or band_keys.key("anchor_per_pose", p)
            if f in p:
                out.append(p[f])
    return out


def quantile(values, f):
    v = sorted(values)
    if not v:
        return float("nan")
    i = f * (len(v) - 1)
    lo = int(i)
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (i - lo)


def load(shim):
    p = os.path.join(shim, "joint_bands.json")
    if not os.path.exists(p):
        sys.exit("%s: no joint_bands.json - run block 3 first" % shim)
    return json.load(open(p))["bands"], json.load(open(os.path.join(shim, "joint_terms.json")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--case", default=None,
                    help="a case present in both, for the per-pose anchor diagnostic")
    a = ap.parse_args()

    ba, ta = load(a.a)
    bb, tb = load(a.b)
    na, nb = os.path.basename(a.a.rstrip("/")), os.path.basename(a.b.rstrip("/"))

    alo_a, ahi_a, cmin_a, dz_a = band_keys.band_terms(ba)
    alo_b, ahi_b, cmin_b, dz_b = band_keys.band_terms(bb)
    wa = ahi_a - alo_a
    wb = ahi_b - alo_b
    da = dz_a[1] - dz_a[0]
    db = dz_b[1] - dz_b[0]

    print("%-22s %-26s %-26s %s" % ("term", na, nb, "change"))
    print("%-22s %-26s %-26s %s" % (
        "anchor window", "%.2f - %.2f A" % (alo_a, ahi_a),
        "%.2f - %.2f A" % (alo_b, ahi_b),
        "ceiling %+.2f A" % (ahi_b - ahi_a)))
    print("%-22s %-26s %-26s %s" % (
        "anchor width", "%.2f A" % wa, "%.2f A" % wb, "%+.2f A" % (wb - wa)))
    print("%-22s %-26s %-26s %s" % (
        "core minimum", cmin_a, cmin_b, "%+d" % (cmin_b - cmin_a)))
    print("%-22s %-26s %-26s %s" % (
        "depth window", "%.2f to %.2f" % tuple(dz_a),
        "%.2f to %.2f" % tuple(bb["depth_band_Z"]), "width %+.2f" % (db - da)))
    print("%-22s %-26s %-26s" % ("positive controls", ba.get("n_agonist_controls", "?"),
                                 bb.get("n_agonist_controls", "?")))

    if ba.get("n_agonist_controls") != bb.get("n_agonist_controls"):
        print("\nThe two batches did not derive their bands from the same NUMBER of positive "
              "controls. That alone moves the window; nothing below is a design comparison.")

    rel = abs(wb - wa) / max(wa, 1e-9)
    print("\nanchor width changed by %.0f%%." % (100 * rel), end=" ")
    if rel >= 0.25:
        print("That is a different instrument: report the band change, and do NOT read\n"
              "a rate difference between these two batches as a statement about the designs.")
    else:
        print("The windows are close; rates remain two draws of one quantity,\n"
              "which is sampling noise and not a baseline to subtract.")

    # Is the ceiling resting on ONE control pose? The band edge in use is an extremum over the
    # pooled positive-control poses, so a single outlier sets it. Measured in this campaign: a
    # Protenix ceiling moved 1.91 A between two seeds of one oracle, while the 95th percentile of
    # the same poses moved 0.20 A, and the extremum came from a different control each time.
    pa, pb = pooled_positive(ta), pooled_positive(tb)
    if pa and pb:
        print("\n%-22s %-26s %-26s %s" % ("ceiling defined as", na, nb, "change"))
        for label, f in (("extremum (in use)", 1.00), ("95th percentile", 0.95),
                         ("90th percentile", 0.90)):
            va, vb = quantile(pa, f), quantile(pb, f)
            print("%-22s %-26s %-26s %+.2f A" % (label, "%.2f A" % va, "%.2f A" % vb, vb - va))
        for name, pool in ((na, pa), (nb, pb)):
            top = sorted(pool)[-1]
            rest = sorted(v for v in pool if v < top)
            gap = top - (rest[-1] if rest else top)
            flag = "  <- edge rests on one pose" if gap > 0.15 else ""
            print("%-26s highest control pose %.2f, next %.2f (gap %.2f)%s"
                  % (name, top, rest[-1] if rest else top, gap, flag))
        print("A quantile edge that barely moves while the extremum does means the window is\n"
              "hostage to a single draw. Note the trade: under an extremum edge the positive\n"
              "controls pass BY CONSTRUCTION, so only the negative controls certify the batch.")

    both = sorted(set(ta) & set(tb))
    print("\n%d case(s) in both batches." % len(both))
    if a.case:
        if a.case not in both:
            sys.exit("%s is not in both batches" % a.case)
        print("\nper-pose anchor distribution for %s" % a.case)
        print("%-26s %7s %7s %7s %9s" % ("batch", "min", "median", "max", "in window"))
        for name, terms, band in ((na, ta, ba), (nb, tb, bb)):
            ps = terms[a.case]["per_sample"]
            fld = band_keys.key("anchor_per_pose", ps[0])
            r = sorted(p[fld] for p in ps)
            blo, bhi = band_keys.get("anchor_lo", band), band_keys.get("anchor_hi", band)
            inw = sum(1 for v in r if blo <= v <= bhi)
            print("%-26s %7.2f %7.2f %7.2f %6d/%-3d"
                  % (name, r[0], statistics.median(r), r[-1], inw, len(r)))
        print("If the three distances barely moved and the in-window count did, the window "
              "moved and\nthe case did not.")


if __name__ == "__main__":
    main()
