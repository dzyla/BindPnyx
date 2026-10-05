#!/usr/bin/env python3
"""Validate a campaign's target chain/MSA pairs BEFORE diffusion.

`BoltzBackend.prepare_json` already refuses a chain whose cached MSA would be
discarded - but it runs after diffusion and ProteinMPNN. The EGFR campaign of
2026-10-02 therefore spent its GPU budget and then died on one residue
(`targets/egfr_ecd/PROVENANCE.md`). This answers the same question in a second,
from the same code:

    orig_seqs -> target_chains_from_orig_seqs -> validate_target_chains

Only the first arrow is reconstructed here, and it is reconstructed from the
shard, exactly as `pxdesign/data/infer_data_pipeline.py` does it:

  * the chains are `condition.filter.chain_id`, IN THAT ORDER
    (`make_cond_sequences`);
  * each chain's sequence is the shard's own entity sequence,
    `bioassembly_dict["sequences"][label_entity_id]`, NOT a sequence read off
    the coordinates (`get_and_map_sequence_from_atom_array`);
  * each chain's MSA is `condition.msa[<chain>]`, keyed by the chain as the
    SHARD names it - which is not what the source structure called it;
  * `label_asym_id` is positional, `chr(ord('A') + index) + "0"`
    (`pxdesign/runner/inference.py:167-171`), so chain identity downstream is
    the ORDER of this list and nothing else.

What it cannot see: a `tools.boltz.target_msa` override or a non-default
`a3m_name` set in the pipeline config. Pass --a3m-name if the campaign does.
"""
import argparse
import gzip
import json
import os
import pickle
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))

from boltz_light import msa_check, targets as boltz_targets  # noqa: E402

ENTITY_KEY = {
    "protein": "proteinChain",
    "dna": "dnaSequence",
    "rna": "rnaSequence",
}


class PreflightError(Exception):
    """The campaign would fail. Stop before the GPU."""


class CannotPreflight(PreflightError):
    """Not a bad pair - a pair this check cannot see yet. Warn, do not stop.

    `convert_to_bioassembly_dict` builds the shard at run time from a .pdb or
    .cif, so its entity sequences do not exist when this runs. Failing here
    would reject a configuration the pipeline supports.
    """


def orig_seqs_from_shard(sample, a3m_name=boltz_targets.A3M_DEFAULT):
    """Rebuild the entity list the pipeline will hand to Boltz.

    Returns (orig_seqs, notes). Raises PreflightError on anything the pipeline
    itself would raise on (`check_input_validity`).
    """
    cond = sample.get("condition") or {}
    shard_path = cond.get("structure_file")
    if not shard_path:
        raise PreflightError("condition has no structure_file")
    if not shard_path.endswith(".pkl.gz"):
        raise CannotPreflight(
            f"structure_file {shard_path!r} is not a .pkl.gz shard. The "
            f"pipeline builds the shard at run time, so its sequences do not "
            f"exist yet and this pair cannot be checked here. Build it first "
            f"with scripts/prepare_target.py, which validates the pair as it "
            f"writes."
        )
    if not os.path.isfile(shard_path):
        raise PreflightError(f"structure_file {shard_path!r}: no such file")

    with gzip.open(shard_path, "rb") as fh:
        bio = pickle.load(fh)
    atom_array = bio["atom_array"]
    entity_sequences = bio.get("sequences", {})

    entity_of, type_of = {}, {}
    for chain, entity, mol in zip(
        atom_array.chain_id.tolist(),
        atom_array.label_entity_id.tolist(),
        atom_array.mol_type.tolist(),
    ):
        entity_of.setdefault(chain, entity)
        type_of.setdefault(chain, mol)

    notes = []
    filt = cond.get("filter", {}) or {}
    chain_ids = filt.get("chain_id")
    if chain_ids:
        chain_ids = list(chain_ids)
    else:
        chain_ids = sorted(entity_of)
        notes.append(
            "condition.filter.chain_id is absent, so the pipeline takes the "
            "chains from `list(set(atom_array.chain_id))` - a SET, whose order "
            "is not reproducible across processes. Checked in sorted order "
            f"{chain_ids}; set filter.chain_id explicitly to make the "
            "chain-to-MSA pairing deterministic."
        )

    for chain in chain_ids:
        if chain not in entity_of:
            raise PreflightError(
                f"chain {chain!r} is not in the shard (it has "
                f"{sorted(entity_of)}). The pipeline raises the same way, "
                f"after loading the model."
            )
    for key in ("msa", "hotspot"):
        where = cond if key == "msa" else sample
        for chain in (where.get(key) or {}):
            if chain not in chain_ids:
                raise PreflightError(
                    f"{key} is specified for chain {chain!r}, which is not a "
                    f"selected chain ({chain_ids})."
                )

    orig_seqs = []
    for idx, chain in enumerate(chain_ids):
        mol = type_of[chain]
        key = ENTITY_KEY.get(mol)
        if key is None:                       # ligand / ion chain
            notes.append(f"chain {chain}: mol_type {mol!r}, no sequence to check")
            continue
        sequence = entity_sequences.get(entity_of[chain])
        if not sequence:
            raise PreflightError(
                f"chain {chain}: the shard has no sequence for entity "
                f"{entity_of[chain]!r}. The shard is incomplete."
            )
        entity = {
            "sequence": sequence,
            "count": 1,
            "label_asym_id": [f"{chr(ord('A') + idx)}0"],
            "json_chain_id": chain,
        }
        msa = (cond.get("msa") or {}).get(chain)
        if msa is not None:
            entity["msa"] = msa
        orig_seqs.append({key: entity})
    return orig_seqs, notes


def check_sample(sample, min_depth, a3m_name):
    """-> (records, notes). Raises PreflightError / MsaMismatch."""
    orig_seqs, notes = orig_seqs_from_shard(sample, a3m_name)
    chains = boltz_targets.target_chains_from_orig_seqs(
        orig_seqs, a3m_name=a3m_name
    )
    return msa_check.validate_target_chains(chains, min_depth=min_depth), notes


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                epilog=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("campaign_json")
    p.add_argument("--min-depth", type=int, default=msa_check.DEFAULT_MIN_DEPTH)
    p.add_argument("--a3m-name", default=boltz_targets.A3M_DEFAULT)
    args = p.parse_args(argv)

    try:
        with open(args.campaign_json) as fh:
            spec = json.load(fh)
    except (OSError, ValueError) as exc:
        print(f"  FAIL: {args.campaign_json}: {exc}", file=sys.stderr)
        return 1
    samples = spec if isinstance(spec, list) else [spec]

    failed = False
    for sample in samples:
        name = sample.get("name", "<unnamed>")
        if "condition" not in sample:
            print(f"  {name}: no condition block, nothing to check")
            continue
        try:
            records, notes = check_sample(sample, args.min_depth, args.a3m_name)
        except msa_check.MsaMismatch as exc:
            failed = True
            print(f"  FAIL {name}: {exc}", file=sys.stderr)
            print("        This is the check prepare_json makes AFTER diffusion "
                  "and MPNN.\n"
                  "        Fix the target before spending the GPU: see "
                  "scripts/prepare_target.py", file=sys.stderr)
            continue
        except CannotPreflight as exc:
            print(f"  SKIP {name}: {exc}", file=sys.stderr)
            print("        The pair is still checked by prepare_json - after "
                  "diffusion and MPNN.", file=sys.stderr)
            continue
        except PreflightError as exc:
            failed = True
            print(f"  FAIL {name}: {exc}", file=sys.stderr)
            continue
        for note in notes:
            print(f"  note ({name}): {note}")
        for r in records:
            print(f"  {name} {r['label']}: {r['sequence_len']} aa, "
                  f"{r['depth_unique']} unique / {r['depth_headers']} records "
                  f"({r['match']})")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
