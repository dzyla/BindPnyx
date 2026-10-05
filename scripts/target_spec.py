"""Load a target spec in either schema this repo uses, and validate its MSAs.

There are two target-JSON shapes in the tree and they are not interchangeable:

  rescore_panel      {name, seq_file, msa_dir, chain?, min_depth?}   ONE chain
  screened_campaign  {id, sequence, msa_dir, label_asym_id?}         MANY chains

A two-chain target such as obj2 cannot be expressed in the first, which is why
the oracle tests could not be pointed at the campaign's own target. This accepts
either and returns the entity list the backend wants, having validated every
resolved chain/MSA pair first - so a mismatched alignment fails here rather than
after the GPU work.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _is_campaign_schema(entry):
    return "sequence" in entry and "id" in entry


def load_orig_seqs(targets_json):
    """-> (label, orig_seqs, [validation records]). Raises MsaMismatch."""
    from pxdbench.tools.boltz import msa_check as mc
    from pxdbench.tools.boltz.targets import target_chains_from_orig_seqs

    spec = json.load(open(targets_json))
    if not isinstance(spec, list) or not spec:
        raise SystemExit(f"{targets_json}: expected a non-empty JSON list")

    if _is_campaign_schema(spec[0]):
        # Many chains, sequences inline. One target made of N chains.
        orig_seqs = [{
            "proteinChain": {
                "sequence": e["sequence"],
                "count": 1,
                "label_asym_id": [e.get("label_asym_id", e["id"] + "0")],
                "use_msa": True,
                "msa": {"precomputed_msa_dir": e["msa_dir"]},
            }
        } for e in spec]
        label = os.path.splitext(os.path.basename(targets_json))[0]
        min_depth = mc.DEFAULT_MIN_DEPTH
    else:
        # One chain, sequence in a file. A list here means several SEPARATE
        # targets, which would be several batches - refuse it.
        if len(spec) != 1:
            raise SystemExit(
                f"{targets_json}: {len(spec)} single-chain targets. These are "
                f"separate batches; the oracle tests need exactly one target."
            )
        e = spec[0]
        raw = open(e["seq_file"]).read()
        if ">" in raw:
            raise SystemExit(
                f"target {e['name']!r}: {e['seq_file']!r} contains '>', so it "
                f"is FASTA, not the raw sequence this reads."
            )
        seq = raw.strip().replace("\n", "")
        orig_seqs = [{
            "proteinChain": {
                "sequence": seq, "count": 1,
                "label_asym_id": [e.get("chain", "A0")],
                "use_msa": True,
                "msa": {"precomputed_msa_dir": e["msa_dir"]},
            }
        }]
        label = e["name"]
        min_depth = e.get("min_depth", mc.DEFAULT_MIN_DEPTH)

    # Validate the RESOLVED pairs, before any GPU work.
    records = mc.validate_target_chains(
        target_chains_from_orig_seqs(orig_seqs), min_depth=min_depth
    )
    total = sum(len(r["sequence_len"] * "x") for r in records)
    print(f"=== target {label!r}: {len(records)} chain(s), {total} aa total")
    for r in records:
        print(f"      {r['label']}: {r['sequence_len']} aa, "
              f"{r['depth_unique']} unique / {r['depth_headers']} records "
              f"({r['match']})")
    return label, orig_seqs, records
