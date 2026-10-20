#!/usr/bin/env python3
"""BLOCK 1 - turn sequences, controls and seeds into a complete, self-describing batch.

What this prevents. A batch is "complete" when everything the SCORER will later read already
exists: the inputs, the case list, the per-case provenance, and the manifest. This campaign once
finished 500 structures and then died on the opening line of scoring with a missing manifest.json
that the builder had never written - all the compute paid for, none of it readable. So the builder
writes the manifest, and block 0 checks it.

Three checks run here because each has cost a batch somewhere:
  * a case name containing "." is rejected - some runners name the output record after the file
    stem, which strips only the last suffix, so `design.A.B` silently becomes `design.A`;
  * every permutation named `scrN_<parent>` must be composition-matched to a `<parent>` that is
    also in the batch, because a permutation that is not composition-matched is not a baseline;
  * `--assert-prefix` reports any design not starting with a fixed motif. Three documents in this
    campaign asserted all 20 panel rows shared a fixed N-terminal arm. Nineteen did. Reported,
    not fatal: a fact about the panel should not block a batch.

Control policy. Controls are passed in a SEPARATE file from designs, and the batch is refused
without at least three positive and one negative control, because block 3 cannot say whether a
batch is evidence otherwise. Name positives `ctl_pos_*` (or `ctl_ago_*`) and negatives `ctl_neg_*`
(or `ctl_antag_*`). The band is derived from the positives, so a batch with two instead of three
derives a DIFFERENT band - measured in this campaign: the same anchor ceiling came out 3.01 A with
two positive controls and 3.38-3.42 A with three.

Case order. Controls are written first, then any group given by `--group`, in the order given.
The queue is resumable per case, so an interruption leaves a prefix that still has its control
block and is still evidence.

Usage:
  build_cofold_batch.py --out runs/x99 --designs d.csv --controls c.csv \
      --receptor receptor.json --receptor-msa rec.a3m --seeds 11,12,13,14,15 \
      [--group leader:d6,d7] [--group permutation:scr1_d6,scr2_d6] \
      [--assert-prefix WVELTFWNA] [--dialect alphafold3]

  --designs / --controls   two-column CSV: name,sequence
  --receptor               JSON holding the receptor chain, or a bare .txt/.fasta sequence
  --receptor-msa           precomputed a3m for the receptor (binder is folded single-sequence)
  --seeds                  comma-separated; these ARE the replicate axis, so record them
"""
import argparse
import csv
import json
import os
import sys

AA = set("ACDEFGHIKLMNPQRSTVWY")
POS = ("ctl_pos", "ctl_ago")
NEG = ("ctl_neg", "ctl_antag")


def read_pairs(path):
    rows = list(csv.DictReader(open(path)))
    if not rows:
        sys.exit("%s: empty" % path)
    col = next((c for c in rows[0] if "seq" in c.lower()), None)
    name = next((c for c in rows[0] if c.lower() in ("name", "design", "design_id", "id")), None)
    if not col or not name:
        sys.exit("%s: need a name column and a sequence column, found %s"
                 % (path, list(rows[0])))
    out = {}
    for r in rows:
        n, s = (r[name] or "").strip(), (r[col] or "").strip().upper()
        if not n or not s:
            continue
        if n in out and out[n] != s:
            sys.exit("%s: %s appears twice with different sequences" % (path, n))
        out[n] = s
    return out


def receptor_sequence(path):
    if path.endswith(".json"):
        d = json.load(open(path))
        seqs = d["sequences"] if isinstance(d, dict) else d
        for c in seqs:
            for key in ("protein", "proteinChain"):
                if key in c and c[key].get("sequence"):
                    return c[key]["sequence"].strip().upper()
        sys.exit("%s: no protein chain found" % path)
    txt = open(path).read()
    return "".join(l.strip() for l in txt.splitlines() if not l.startswith(">")).upper()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--designs", required=True)
    ap.add_argument("--controls", required=True)
    ap.add_argument("--receptor", required=True)
    ap.add_argument("--receptor-msa", required=True)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--group", action="append", default=[],
                    help="role:name,name,... - written in the order given, after the controls")
    ap.add_argument("--assert-prefix", default=None,
                    help="motif every design is believed to start with; mismatches are REPORTED")
    ap.add_argument("--dialect", default="alphafold3")
    ap.add_argument("--batch-name", default=None)
    a = ap.parse_args()

    out = os.path.abspath(a.out)
    batch = a.batch_name or os.path.basename(out)
    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    if not seeds:
        sys.exit("--seeds is the replicate axis and cannot be empty")

    designs, controls = read_pairs(a.designs), read_pairs(a.controls)
    npos = sum(1 for n in controls if n.startswith(POS))
    nneg = sum(1 for n in controls if n.startswith(NEG))
    if npos < 3 or nneg < 1:
        sys.exit("controls: need >= 3 positive (ctl_pos_*/ctl_ago_*) and >= 1 negative "
                 "(ctl_neg_*/ctl_antag_*); got %d and %d. Block 3 cannot certify a batch "
                 "without them." % (npos, nneg))
    clash = set(designs) & set(controls)
    if clash:
        sys.exit("names appear in both files: %s" % sorted(clash))

    allseq = dict(designs)
    allseq.update(controls)
    for n, s in allseq.items():
        if "." in n:
            sys.exit("case name %r contains a dot; some runners truncate at the last dot" % n)
        bad = set(s) - AA
        if bad:
            sys.exit("%s: non-standard residues %s" % (n, sorted(bad)))

    # permutations must be composition-matched to a parent that is in this batch
    perms = []
    for n in designs:
        if not n.startswith("scr"):
            continue
        parent = n.split("_", 1)[1] if "_" in n else ""
        if parent not in allseq:
            sys.exit("%s looks like a permutation but its parent %r is not in the batch"
                     % (n, parent))
        if sorted(allseq[n]) != sorted(allseq[parent]):
            sys.exit("%s is NOT composition-matched to %s - it is not a baseline" % (n, parent))
        perms.append((n, parent))

    role = {n: "control" for n in controls}
    ordered = sorted(controls)
    for g in a.group:
        if ":" not in g:
            sys.exit("--group takes role:name,name,...")
        r, names = g.split(":", 1)
        for n in [x.strip() for x in names.split(",") if x.strip()]:
            if n not in allseq:
                sys.exit("--group %s names %s, which is not in the inputs" % (r, n))
            if n in role:
                sys.exit("%s is in two groups" % n)
            role[n] = r
            ordered.append(n)
    for n in sorted(designs):
        if n not in role:
            role[n] = "permutation" if n.startswith("scr") else "design"
            ordered.append(n)

    if a.assert_prefix:
        # permutations are excluded: a composition-matched permutation is scrambled by
        # construction, so flagging it would bury the one row that is genuinely unexpected
        subject = [n for n in designs if not n.startswith("scr")]
        off = [n for n in subject if not allseq[n].startswith(a.assert_prefix)]
        print("prefix check: %d of %d non-permutation designs start with %r"
              % (len(subject) - len(off), len(subject), a.assert_prefix))
        for n in sorted(off):
            pre = allseq[n][:len(a.assert_prefix)]
            diff = sum(1 for x, y in zip(a.assert_prefix, pre) if x != y)
            print("  %-26s differs at %d position(s): %s" % (n, diff, pre))

    rec = receptor_sequence(a.receptor)
    msa = os.path.abspath(a.receptor_msa)
    if not os.path.exists(msa):
        sys.exit("receptor MSA not found: %s" % msa)

    os.makedirs(os.path.join(out, "json"), exist_ok=True)
    parent_of = dict(perms)
    for n in ordered:
        s = allseq[n]
        json.dump({"name": n, "modelSeeds": seeds,
                   "sequences": [
                       {"protein": {"id": "R", "sequence": rec,
                                    "unpairedMsaPath": msa, "templates": []}},
                       {"protein": {"id": "B", "sequence": s,
                                    "unpairedMsa": ">query\n%s\n" % s, "templates": []}}],
                   "dialect": a.dialect, "version": 1},
                  open(os.path.join(out, "json", n + ".json"), "w"), indent=1)

    open(os.path.join(out, "cases.txt"), "w").write("\n".join(ordered) + "\n")
    with open(os.path.join(out, "designs.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["name", "sequence"])
        w.writerows((n, allseq[n]) for n in ordered)
    with open(os.path.join(out, "provenance.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["name", "role", "parent", "length"])
        w.writeheader()
        w.writerows(dict(name=n, role=role[n], parent=parent_of.get(n, ""),
                         length=len(allseq[n])) for n in ordered)
    with open(os.path.join(out, "manifest.json"), "w") as fh:
        json.dump(dict(batch=batch, seeds=seeds, n_samples_per_seed=5, entries=[
            dict(name=n, sequence=allseq[n], length=len(allseq[n]),
                 lane="control" if role[n] == "control" else batch,
                 role="control" if role[n] == "control" else "design",
                 provenance_role=role[n], parent=parent_of.get(n, "")) for n in ordered]),
            fh, indent=1)

    lens = sorted({len(allseq[n]) for n in ordered})
    print("%s: %d cases (%d control, %d permutation, %d other), seeds %s"
          % (batch, len(ordered), len(controls), len(perms),
             len(ordered) - len(controls) - len(perms), seeds))
    print("controls first in cases.txt; %d distinct binder lengths %s" % (len(lens), lens))
    print("wrote json/, cases.txt, designs.csv, provenance.csv, manifest.json under %s" % out)


if __name__ == "__main__":
    main()
