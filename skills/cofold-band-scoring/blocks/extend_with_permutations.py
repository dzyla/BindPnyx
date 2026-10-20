#!/usr/bin/env python
"""BLOCK 7 - give a design in an already-scored batch its composition-matched permutation triple.

The task this exists for. A batch finishes, one design holds the band well, and it has no
permutation baseline - so what it has is a RATE and the designs it will be compared against have a
MARGIN. A rate and a margin are not the same claim: the rate says "this sequence holds the
mechanism band in k of n poses", the margin says "...and its own residues in a different order do
not". Only the second rules out a composition artefact, and in this campaign the difference was
decisive - permutations of a 21/25 design hold the band in 1 of 75 poses.

Why this EXTENDS the batch instead of making a new one. The band is derived from the batch's own
positive controls and from nothing else. Add cases without re-folding the controls and the band is
bit-identical, so every pre-existing rate survives exactly and the new permutations are scored
against the same instrument. Fold the design again in a fresh batch instead and you have paid for
a new band, under which the old numbers are not comparable. Verified both ways here: an 18-case
extension of a 43-case batch reproduced the band to the digit on every term.

Why the design itself is NOT re-folded. It is already in the batch at these seeds, and these
oracles are deterministic given seed and input - a re-fold reproduces the rate to the structure and
buys nothing. Block 2 skips it on the artefact count.

What is checked before anything is written (each of these has been a real failure):
  - a queue running on this batch           bash reads cases.txt by byte offset; rewriting it
                                            under a running queue corrupts the read position
  - the design has no artefacts yet          then its rate does not exist and there is nothing to
                                            give a baseline to; fold the batch first
  - the permutation is not composition-matched   then it is not a baseline, it is a different
                                            sequence, and the margin means nothing
  - the permutation already exists elsewhere in the corpus   then it is not an independent draw;
                                            pass a different --seed-base
  - the case name already exists in the batch   silently overwriting a scored case

Usage:
  extend_with_permutations.py <batch_dir> --designs NAME [NAME ...] --seed-base N [--n 3]
  extend_with_permutations.py <batch_dir> --min-hits 13 --seed-base N        (reads <batch>_shim)
  ... --dry-run                                                             (prints, writes nothing)

Then: blocks/preflight.sh, blocks/run_queue.sh, blocks/score_batch.sh, blocks/group_report.py.
"""
import argparse
import csv
import glob
import json
import os
import random
import subprocess
import sys

import band_keys


def die(msg):
    sys.exit("[refused] %s" % msg)


def queue_is_running(batch):
    """Any shell executing a queue over THIS batch. Checked by path, not by script name."""
    pats = ("run_queue.sh", "run_af3_queue.sh")
    try:
        out = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout
    except OSError:
        return False
    tag = os.path.basename(batch.rstrip("/"))
    return any(tag in line and any(p in line for p in pats) for line in out.splitlines())


def band_rates(shim):
    """Rate per case under the shim's own band, plus how many permutations each case already has."""
    T = json.load(open(os.path.join(shim, "joint_terms.json")))
    B = json.load(open(os.path.join(shim, "joint_bands.json")))["bands"]
    lo, hi, cm, dz = band_keys.band_terms(B)

    def rate(c):
        ps = T[c]["per_sample"]
        fld = band_keys.key("anchor_per_pose", ps[0]) if ps else "anchor"
        return sum(1 for s in ps if lo <= s[fld] <= hi and s["core"] >= cm
                   and dz[0] <= s["depth_heavy_Z"] <= dz[1]), len(ps)
    return {c: (rate(c), sum(1 for k in (1, 2, 3) if "scr%d_%s" % (k, c) in T)) for c in T}


def corpus_sequences(runs_root, exclude_batch):
    """Every design sequence this campaign has already folded, by sequence -> where it was seen."""
    seen = {}
    for p in glob.glob(os.path.join(runs_root, "*", "designs.csv")):
        if os.path.dirname(p) == exclude_batch:
            continue
        try:
            for row in csv.reader(open(p)):
                if len(row) >= 2 and row[1].isalpha():
                    seen.setdefault(row[1], os.path.basename(os.path.dirname(p)))
        except (OSError, csv.Error):
            continue
    return seen


def binder_chain(entry):
    """The one protein chain in a json entry, whichever dialect wrote it."""
    for key in ("protein", "proteinChain"):
        if key in entry:
            return entry[key]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("batch")
    ap.add_argument("--designs", nargs="*", default=[])
    ap.add_argument("--min-hits", type=int, default=None,
                    help="instead of --designs: every scored row at or above this rate that has "
                         "no permutation in the batch")
    ap.add_argument("--seed-base", type=int, required=True,
                    help="base for the shuffle seeds; keep it disjoint from every base this "
                         "campaign has used, and the corpus check below will catch a collision")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--shim", default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    B = os.path.abspath(a.batch.rstrip("/"))
    shim = a.shim or (B + "_shim")
    runs_root = os.path.dirname(B)
    for f in ("cases.txt", "designs.csv", "json"):
        if not os.path.exists(os.path.join(B, f)):
            die("%s is not a batch directory: no %s" % (B, f))
    if queue_is_running(B):
        die("a queue is running over this batch - rewriting cases.txt under it corrupts the "
            "read position; wait for it to finish")

    cases = [l.strip() for l in open(os.path.join(B, "cases.txt")) if l.strip()]
    seqs = {}
    for row in csv.reader(open(os.path.join(B, "designs.csv"))):
        if len(row) >= 2:
            seqs[row[0]] = row[1]

    targets = list(a.designs)
    if a.min_hits is not None:
        if not os.path.exists(os.path.join(shim, "joint_terms.json")):
            die("--min-hits needs a scored shim; %s has none" % shim)
        R = band_rates(shim)
        for c, ((h, _n), nperm) in sorted(R.items(), key=lambda kv: -kv[1][0][0]):
            if c.startswith(("ctl_", "scr")) or nperm >= a.n or h < a.min_hits:
                continue
            targets.append(c)
    if not targets:
        die("no designs named and none selected - nothing to do")
    targets = list(dict.fromkeys(targets))

    # Each target must already BE measured in this batch. A baseline for an unmeasured design is
    # a baseline for nothing.
    for d in targets:
        if d not in seqs:
            die("%s is not in %s/designs.csv" % (d, B))
        n = len(glob.glob(os.path.join(B, "out", d, "**", "*_model.cif"), recursive=True)) \
            or len(glob.glob(os.path.join(B, "out", d, "**", "*.cif"), recursive=True))
        if n == 0:
            die("%s has no structures in this batch - it has no rate yet, so there is nothing to "
                "give a baseline to; fold the batch first" % d)
        existing = [k for k in range(1, a.n + 1) if "scr%d_%s" % (k, d) in cases]
        if existing:
            die("%s already has permutation(s) %s in this batch" % (d, existing))

    print("batch   %s" % B)
    print("designs %s" % ", ".join(targets))
    print("drawing %d permutation(s) each, seed base %d\n" % (a.n, a.seed_base))

    corpus = corpus_sequences(runs_root, B)
    new_rows, new_names = [], []
    for i, d in enumerate(targets):
        s = seqs[d]
        drawn = []
        for k in range(a.n):
            L = list(s)
            random.Random(a.seed_base + i * 10 + k).shuffle(L)
            sc = "".join(L)
            # a permutation that is not composition-matched is not a baseline
            if sorted(sc) != sorted(s):
                die("internal: shuffle changed the composition of %s" % d)
            if sc == s:
                die("permutation %d of %s came back identical to the parent - pass a different "
                    "--seed-base" % (k + 1, d))
            if sc in drawn or sc in seqs.values():
                die("permutation %d of %s repeats a sequence already in this batch - pass a "
                    "different --seed-base" % (k + 1, d))
            if sc in corpus:
                die("permutation %d of %s was already folded in %s, so it is not an independent "
                    "draw - pass a different --seed-base" % (k + 1, d, corpus[sc]))
            drawn.append(sc)
            tag = "scr%d_%s" % (k + 1, d)
            if tag in cases:
                die("%s is already a case in this batch" % tag)
            new_names.append(tag)
            new_rows.append(dict(name=tag, role="permutation", parent=d, length=str(len(sc))))
            seqs[tag] = sc
        print("  %-28s %d aa -> %s" % (d, len(s), " ".join("scr%d" % (k + 1)
                                                           for k in range(a.n))))

    print("\n%d new case(s)" % len(new_names))
    if a.dry_run:
        print("--dry-run: nothing written")
        return

    # The json template is the PARENT's own json, never a sibling permutation's. A sibling belongs
    # to a different parent, so the binder chain cannot be identified in it - the first draft of
    # this block templated on `sib[0]` and refused with "0 candidate chains of 78 aa", because the
    # sibling's binder was 67. Templating on the parent also means every other field - seeds, the
    # receptor sequence, the receptor's precomputed alignment - is carried over by construction
    # rather than by assumption, and the binder is identified by EXACT sequence identity with the
    # parent design rather than by a length that two chains could share.
    for tag in new_names:
        parent = next(r["parent"] for r in new_rows if r["name"] == tag)
        tpl_name = parent
        tpl = json.load(open(os.path.join(B, "json", "%s.json" % tpl_name)))
        s = seqs[tag]

        entries = tpl["sequences"] if isinstance(tpl, dict) else tpl[0]["sequences"]
        pseq = seqs[parent]
        hits = [e for e in entries if binder_chain(e) is not None
                and binder_chain(e)["sequence"] == pseq]
        if len(hits) != 1:
            die("cannot identify the binder chain in %s.json: %d chain(s) carry the design's own "
                "sequence - extend this block rather than guessing" % (tpl_name, len(hits)))
        ch = binder_chain(hits[0])
        ch["sequence"] = s
        for f in ("unpairedMsa", "pairedMsa"):
            if f in ch:
                ch[f] = ">query\n%s\n" % s
        blocked = [f for f in ("unpairedMsaPath", "pairedMsaPath", "precomputedMsaDir")
                   if f in ch]
        if blocked:
            die("the binder chain in %s.json takes its alignment from a file (%s). A permutation "
                "must not inherit its parent's alignment, and this block will not fabricate an "
                "MSA file; write a single-sequence a3m for it and pass that batch a template "
                "permutation instead" % (tpl_name, ", ".join(blocked)))
        if isinstance(tpl, dict):
            tpl["name"] = tag
        else:
            tpl[0]["name"] = tag
        json.dump(tpl, open(os.path.join(B, "json", "%s.json" % tag), "w"), indent=1)

    # Append, never rewrite: every existing case keeps its position in cases.txt and its structures.
    with open(os.path.join(B, "cases.txt"), "a") as fh:
        fh.write("\n".join(new_names) + "\n")
    with open(os.path.join(B, "designs.csv"), "a", newline="") as fh:
        csv.writer(fh).writerows((n, seqs[n]) for n in new_names)
    pv = os.path.join(B, "provenance.csv")
    if os.path.exists(pv):
        # Adopt the batch's OWN spelling for a permutation row rather than imposing one. x40
        # writes role=scramble, substitutions=permutation; x41 writes role=permutation with no
        # substitutions column. Reports group by these values, so a row that spells its role
        # differently from its siblings is silently reported as a separate group - which is what
        # the first run of this block did here. Any existing scr* row is the template; only name,
        # parent and length are overridden.
        rows = list(csv.DictReader(open(pv)))
        cols = list(rows[0].keys()) if rows else ["name", "role", "parent", "length"]
        tpl = next((r for r in rows if (r.get("name") or "").startswith("scr")), None)
        with open(pv, "a", newline="") as fh:
            w = csv.writer(fh)
            for r in new_rows:
                base = dict(tpl) if tpl else {}
                base.update(name=r["name"], parent=r["parent"], length=r["length"])
                if tpl is None:
                    base["role"] = "permutation"
                w.writerow([base.get(c, "") for c in cols])
    mf = os.path.join(B, "manifest.json")
    if os.path.exists(mf):
        man = json.load(open(mf))
        man.setdefault("entries", []).extend(
            dict(name=r["name"], sequence=seqs[r["name"]], length=int(r["length"]),
                 role="design", provenance_role="permutation", parent=r["parent"])
            for r in new_rows)
        man.setdefault("extensions", []).append(
            dict(what="permutation triples", designs=targets, seed_base=a.seed_base,
                 n_per_design=a.n,
                 note="controls not re-folded, so the band is unchanged and every "
                      "pre-existing rate stands"))
        json.dump(man, open(mf, "w"), indent=1)

    print("appended; cases.txt now holds %d" % (len(cases) + len(new_names)))
    print("the named designs are NOT re-folded - block 2 skips them on the artefact count")


if __name__ == "__main__":
    main()
