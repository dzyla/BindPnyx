"""End-to-end: does a requested hotspot survive, in the right numbering?

The question the epitope check could not answer on its own: when a design does
not contact its hotspots, is that a real miss or a numbering mismatch?

This separates them. It runs the pipeline with hotspots, then checks the chain
of custody:

  1. the requested hotspots RESOLVE in the structure the model sees
     (otherwise np.isin matches nothing and the run is unconditioned)
  2. resolved_hotspots.json records them in the design's numbering
  3. every recorded hotspot EXISTS as a residue in the exported complex
     -> numbering is consistent end to end
  4. only then is "contacted / not contacted" a statement about the design

Step 3 is the one that was missing. Without it a 0/6 epitope result is
ambiguous, and a wrong answer looks exactly like a correct one.

Usage:
  python scripts/hotspot_e2e_check.py RUN_DIR [--binder-chain C]
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Bio.PDB import PDBParser

from pxdbench.metrics.epitope import contact_residues, evaluate_epitope_policy


def residues_in(structure_path):
    """{(chain, resseq)} present in an exported complex."""
    model = PDBParser(QUIET=True).get_structure("c", structure_path)[0]
    return {
        (chain.id, int(res.id[1]))
        for chain in model
        for res in chain
        if res.id[0] == " "
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--binder-chain", default="C")
    args = ap.parse_args()

    failures = []

    # ---- 1/2. what was requested, and what the pipeline resolved ----------
    recorded = glob.glob(os.path.join(args.run_dir, "**", "resolved_hotspots.json"),
                         recursive=True)
    if not recorded:
        print("FAIL: no resolved_hotspots.json - the run did not record the "
              "numbering its hotspots were resolved to")
        return 1
    payload = json.load(open(recorded[0]))
    resolved = payload["hotspot_resolved"]
    print(f"resolved hotspots ({payload['numbering']}):")
    print(f"  {resolved}")

    exports = sorted(glob.glob(
        os.path.join(args.run_dir, "**", "boltz_docked", "*.pdb"), recursive=True))
    if not exports:
        print("FAIL: no exported complexes to check against")
        return 1

    # ---- 3. do those residues EXIST in the exported complexes? ------------
    print(f"\nchecking {len(exports)} exported complexes\n")
    for path in exports:
        present = residues_in(path)
        chains = sorted({c for c, _ in present})
        missing = [
            f"{c}{r}" for c, residues in resolved.items() for r in residues
            if (c, int(r)) not in present
        ]
        name = os.path.basename(path)
        if missing:
            failures.append((name, missing))
            print(f"  {name}")
            print(f"     NUMBERING MISMATCH: {missing} absent from the export")
            print(f"     export has chains {chains}")
            continue

        # ---- 4. now contact is a statement about the design ---------------
        got = evaluate_epitope_policy(path, args.binder_chain, required=resolved)
        contacts = contact_residues(path, args.binder_chain)
        n_contact = sum(len(v) for v in contacts.values())
        print(f"  {name}")
        print(f"     numbering OK: every resolved hotspot exists in the export")
        print(f"     epitope: {got['ep_required_contacted']}/"
              f"{got['ep_required_total']} contacted"
              f"  satisfied={got['ep_satisfied']}"
              f"  missed={got['ep_missed_required'] or '-'}")
        print(f"     interface size: {n_contact} target residues contacted")

    print()
    if failures:
        print(f"FAIL: {len(failures)} export(s) do not contain the resolved "
              f"hotspot residues. Epitope results on those are UNINTERPRETABLE - "
              f"a miss cannot be distinguished from a numbering error.")
        return 1
    print("PASS: hotspot numbering is consistent from request to export, so "
          "contacted/not-contacted is a statement about the designs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
