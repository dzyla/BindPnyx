#!/usr/bin/env python3
"""Build a target shard from a source structure, with its provenance.

Why this exists
---------------
`pxdesign/utils/infer.py:convert_to_bioassembly_dict` already writes `.pkl.gz`
shards. What was never recorded is *how a particular crop was produced*: which
structure, which chains, which residue ranges, in which numbering, and what was
edited. A campaign aborted on 2026-10-02 because a shard disagreed with its
cached MSA at one residue, and nothing in the shard said where the residue had
come from - the shard renumbers every chain to 1..N and renames chains to
A, B, ..., so the source numbering is not recoverable from it. See
`targets/egfr_ecd/PROVENANCE.md`.

What it guarantees
------------------
* The emitted sequence is compared, residue by residue, against an
  authoritative WT sequence. **Every** disagreement must be declared as an
  `--edit`, or nothing is written.
* An edit never merely relabels a residue. Relabelling leaves the donor's
  side-chain atoms behind - a LYS wearing an ASN label - so the only mutating
  operation offered is `truncate_to_cb`, which deletes the side chain past CB
  and leaves the residue *modelled* rather than misrepresented. The dropped
  atoms are listed in the record.
* The chain/MSA pairs are validated with the same code the Boltz path uses
  (`pxdbench.tools.boltz.msa_check`), before the shard is written.
* The shard is read back and its sequences compared with what was intended. A
  disagreement deletes the shard.

Example (the EGFR obj2 crop, rebuilt from the 1NQL crystal against UniProt WT):

    python scripts/prepare_target.py \\
        --source targets/egfr_ecd/1nql_full.pdb \\
        --wt-fasta targets/egfr_ecd/P00533_uniprot_wt.fasta \\
        --wt-numbering mature \\
        --chain A:227-306=B --chain A:512-614=D \\
        --msa B=<cache>/obj2_chainB/0 --msa D=<cache>/obj2_chainD/0 \\
        --edit 'D:516:LYS>ASN:truncate_to_cb' \\
        --name obj2_target_wt --out-dir targets/egfr_ecd/prepared/obj2_wt

Exit codes: 0 written, 2 refused (provenance still written, status=refused).
"""
import argparse
import datetime
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, REPO)          # the vendored pxdesign, per CLAUDE.md 2

from boltz_light import msa_check  # noqa: E402

#: Standard residues plus the selenomethionine that crystallographers leave in
#: place of MET. Anything else cannot be compared against a WT sequence, so it
#: is a refusal rather than an 'X'.
THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M",
}
ONE_TO_THREE = {v: k for k, v in THREE_TO_ONE.items() if k != "MSE"}

#: Atoms every amino acid shares. `truncate_to_cb` keeps these and nothing else
#: (OXT too, where the crystal has a real C-terminus). GLY has no CB.
BACKBONE = ("N", "CA", "C", "O", "OXT")
CB = "CB"

#: UniProt P00533 numbering: the signal peptide is residues 1-24, so a
#: structure numbered from the mature N-terminus is 24 behind the precursor.
#: Stated rather than inferred, because getting it wrong shifts every mapping.
SIGNAL_PEPTIDE_LEN = 24


class Refused(Exception):
    """Something would have produced a shard that is not what it claims."""


# --------------------------------------------------------------------------
# source structure
# --------------------------------------------------------------------------

class Residue:
    __slots__ = ("chain", "resseq", "icode", "resname", "atoms", "hetero")

    def __init__(self, chain, resseq, icode, resname, hetero):
        self.chain = chain
        self.resseq = resseq
        self.icode = icode
        self.resname = resname
        self.hetero = hetero
        self.atoms = []          # list of raw PDB lines

    @property
    def key(self):
        return (self.resseq, self.icode)

    @property
    def one(self):
        return THREE_TO_ONE[self.resname]

    def atom_names(self):
        return [line[12:16].strip() for line in self.atoms]


def parse_pdb(path):
    """-> (chains: {chain_id: [Residue, ...]}, notes: [str]).

    Only ATOM records and the amino-acid HETATMs (MSE) become residues;
    everything else - waters, NAG, ligands - is counted in `notes` and dropped,
    because a target shard built for binder design conditions on the protein.
    Alternate locations other than ' ' and 'A' are dropped and counted: keeping
    both copies would double atoms in the shard.
    """
    chains, order = {}, {}
    dropped_het, dropped_alt = {}, 0
    with open(path, errors="ignore") as fh:
        for line in fh:
            rec = line[:6]
            if rec not in ("ATOM  ", "HETATM"):
                continue
            resname = line[17:20].strip()
            if rec == "HETATM" and resname not in THREE_TO_ONE:
                dropped_het[resname] = dropped_het.get(resname, 0) + 1
                continue
            altloc = line[16]
            if altloc not in (" ", "A"):
                dropped_alt += 1
                continue
            chain = line[21]
            try:
                resseq = int(line[22:26])
            except ValueError:
                continue
            icode = line[26]
            key = (resseq, icode)
            bucket = chains.setdefault(chain, {})
            residue = bucket.get(key)
            if residue is None:
                residue = Residue(chain, resseq, icode, resname, rec == "HETATM")
                bucket[key] = residue
                order.setdefault(chain, []).append(key)
            elif residue.resname != resname:
                raise Refused(
                    f"{path}: chain {chain} residue {resseq}{icode.strip()} is "
                    f"both {residue.resname} and {resname}. A microheterogeneous "
                    f"site cannot be mapped onto one WT residue; pick one with "
                    f"an external edit and re-run."
                )
            residue.atoms.append(line.rstrip("\n"))
    notes = []
    if dropped_het:
        notes.append(
            "dropped non-polymer HETATM groups: "
            + ", ".join(f"{k}x{v}" for k, v in sorted(dropped_het.items()))
        )
    if dropped_alt:
        notes.append(f"dropped {dropped_alt} atoms in alternate locations other than A")
    return {c: [chains[c][k] for k in order[c]] for c in order}, notes


# --------------------------------------------------------------------------
# command-line specs
# --------------------------------------------------------------------------

def parse_chain_spec(spec):
    """'A:227-306=B' -> ('A', [(227,306)], 'B'). Ranges and rename optional."""
    rename = None
    if "=" in spec:
        spec, rename = spec.split("=", 1)
        rename = rename.strip()
        if len(rename) != 1:
            raise Refused(
                f"--chain {spec}={rename}: an output chain id is one character "
                f"(the PDB format has one column for it)."
            )
    if ":" in spec:
        src, ranges = spec.split(":", 1)
    else:
        src, ranges = spec, ""
    src = src.strip()
    if len(src) != 1:
        raise Refused(f"--chain {spec!r}: source chain id must be one character")
    from pxdesign.utils.residue_numbering import parse_ranges
    parsed = parse_ranges(ranges) if ranges.strip() else None
    return src, parsed, (rename or src)


def parse_edit_spec(spec):
    """'D:516:LYS>ASN:truncate_to_cb' -> dict. One- or three-letter codes."""
    parts = spec.split(":")
    if len(parts) != 4:
        raise Refused(
            f"--edit {spec!r}: expected CHAIN:RESNUM:FROM>TO:OP, e.g. "
            f"--edit 'D:516:LYS>ASN:truncate_to_cb'. Quote it: an unquoted "
            f"'>' is a shell redirection, which is how this argument arrived "
            f"truncated."
        )
    chain, resnum, change, op = (p.strip() for p in parts)
    if ">" not in change:
        raise Refused(f"--edit {spec!r}: the change must read FROM>TO")
    frm, to = (c.strip().upper() for c in change.split(">", 1))

    def three(code):
        if len(code) == 1:
            if code not in ONE_TO_THREE:
                raise Refused(f"--edit {spec!r}: {code!r} is not an amino acid")
            return ONE_TO_THREE[code]
        if code not in THREE_TO_ONE:
            raise Refused(f"--edit {spec!r}: {code!r} is not an amino acid")
        return code

    if frm == to and op == "truncate_to_cb":
        raise Refused(
            f"--edit {spec!r}: FROM and TO are the same residue, so this "
            f"reconciles nothing with the WT - it just deletes a real side "
            f"chain (a CYS>CYS truncation silently breaks a disulfide). State "
            f"a substitution, or drop_residue if removal is what you mean."
        )
    if op not in ("truncate_to_cb", "drop_residue"):
        raise Refused(
            f"--edit {spec!r}: unknown operation {op!r}. Supported: "
            f"truncate_to_cb (delete the side chain past CB and relabel - the "
            f"residue becomes modelled, not misrepresented), drop_residue "
            f"(remove it, leaving a chain break). There is deliberately no "
            f"relabel-only operation: it would leave the donor's side-chain "
            f"atoms under the new name."
        )
    try:
        resnum = int(resnum)
    except ValueError:
        raise Refused(f"--edit {spec!r}: {resnum!r} is not a residue number")
    return {
        "chain": chain, "resnum": resnum, "icode": " ",
        "from": three(frm), "to": three(to), "op": op,
    }


def parse_msa_spec(spec):
    if "=" not in spec:
        raise Refused(f"--msa {spec!r}: expected CHAIN=DIR_OR_A3M")
    chain, path = spec.split("=", 1)
    return chain.strip(), path.strip()


# --------------------------------------------------------------------------
# WT alignment
# --------------------------------------------------------------------------

def read_fasta_one(path):
    header, chunks = None, []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    raise Refused(
                        f"{path}: more than one FASTA record. The authoritative "
                        f"WT sequence must be unambiguous."
                    )
                header = line[1:]
                continue
            chunks.append(line)
    if not chunks:
        raise Refused(f"{path}: no sequence")
    return header or os.path.basename(path), "".join(chunks)


def wt_offsets(wt_numbering):
    """WT index = structure residue number + offset, 1-based into the record."""
    if wt_numbering == "precursor":
        return 0
    if wt_numbering == "mature":
        return SIGNAL_PEPTIDE_LEN
    raise Refused(f"unknown --wt-numbering {wt_numbering!r}")


def detect_offset(residues, wt, declared=None, numbering="mature"):
    """Return (offset, identity, [candidates]) for WT_pos = resseq + offset.

    `declared` short-circuits detection; detection exists so a human does not
    have to know a construct's numbering convention, but an ambiguous or poor
    fit is a refusal, never a quiet pick.
    """
    if declared is not None:
        return declared, _identity(residues, wt, declared), [declared]
    base = wt_offsets(numbering)
    best = []
    for offset in range(base - 60, base + 61):
        ident = _identity(residues, wt, offset)
        best.append((ident, offset))
    best.sort(reverse=True)
    top_ident, top_offset = best[0]
    ties = [o for i, o in best if i == top_ident]
    if top_ident < 0.9:
        raise Refused(
            f"could not place the structure on the WT sequence: the best "
            f"offset ({top_offset}, WT position = residue number + {top_offset}) "
            f"matches only {top_ident:.1%} of residues. Either this is not the "
            f"same protein, or the numbering is not an integer shift. State it "
            f"with --wt-offset and the mismatches will be reported individually."
        )
    if len(ties) > 1:
        raise Refused(
            f"the structure places equally well ({top_ident:.1%}) at offsets "
            f"{sorted(ties)}. Choose one with --wt-offset."
        )
    return top_offset, top_ident, ties


def _identity(residues, wt, offset):
    hit = total = 0
    for res in residues:
        idx = res.resseq + offset - 1
        if 0 <= idx < len(wt):
            total += 1
            if wt[idx] == res.one:
                hit += 1
    return (hit / total) if total else 0.0


# --------------------------------------------------------------------------
# edits
# --------------------------------------------------------------------------

def apply_truncate_to_cb(residue, target_three):
    """Relabel AND rebuild: keep only atoms every residue shares.

    Returns the list of atom names removed. Refuses when the result would be a
    residue whose CB cannot exist - building an atom that is not in the source
    is modelling, and this tool does not model.
    """
    keep_cb = target_three != "GLY"
    if keep_cb and residue.resname == "GLY":
        raise Refused(
            f"chain {residue.chain} {residue.resname}{residue.resseq}->"
            f"{target_three}: the source residue is GLY, so there is no CB to "
            f"keep and building one would be modelling. Supply a WT-matching "
            f"structure for this position, or model it with a tool that does "
            f"side-chain rebuilding and pass the result as --source."
        )
    kept, removed = [], []
    for line in residue.atoms:
        name = line[12:16].strip()
        if name in BACKBONE or (keep_cb and name == CB):
            kept.append(line[:17] + f"{target_three:>3}" + line[20:])
        else:
            removed.append(name)
    present = {line[12:16].strip() for line in kept}
    missing = [a for a in ("N", "CA", "C", "O") if a not in present]
    if missing:
        raise Refused(
            f"chain {residue.chain} {residue.resname}{residue.resseq}: the "
            f"source residue is missing backbone atoms {missing}; truncating it "
            f"would leave a fragment, not a residue."
        )
    if keep_cb and CB not in present:
        raise Refused(
            f"chain {residue.chain} {residue.resname}{residue.resseq}->"
            f"{target_three}: the source residue has no CB (unresolved side "
            f"chain), so the result would have no side chain at all. Drop it "
            f"with drop_residue, or supply a WT-matching structure."
        )
    residue.atoms = kept
    residue.resname = target_three
    return removed


# --------------------------------------------------------------------------
# the run
# --------------------------------------------------------------------------

def git_commit():
    try:
        return subprocess.run(
            ["git", "-C", REPO, "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def new_record():
    return {
        "tool": "scripts/prepare_target.py",
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "repo_commit": git_commit(),
        "argv": sys.argv[1:],
        "status": "refused",
        "numbering_note": (
            "wt_position is 1-based into the --wt-fasta record; "
            "source_resnum/prepared_resnum are the source structure's own "
            "residue numbers, which the prepared PDB keeps; shard_resnum is "
            "the .pkl.gz's own numbering, which restarts at 1 per chain - the "
            "shard also renames chains A, B, ..., so neither the source chain "
            "id nor the source numbering survives into it"
        ),
    }


def build(args, record):
    if not os.path.isfile(args.source):
        raise Refused(f"--source {args.source!r}: no such file")
    if not args.source.endswith(".pdb"):
        raise Refused(
            f"--source {args.source!r}: only .pdb is read here. A CIF carries "
            f"auth and label numbering that this tool would have to choose "
            f"between silently; convert it first and pass the .pdb."
        )
    record["source"] = {
        "path": os.path.abspath(args.source),
        "sha256": msa_check.file_sha256(args.source),
        "format": "pdb",
    }

    wt_header, wt_seq = read_fasta_one(args.wt_fasta)
    record["wt"] = {
        "fasta": os.path.abspath(args.wt_fasta),
        "fasta_sha256": msa_check.file_sha256(args.wt_fasta),
        "header": wt_header,
        "length": len(wt_seq),
        "sequence_sha256": msa_check.sequence_sha256(wt_seq),
        "numbering": args.wt_numbering,
    }

    chains, source_notes = parse_pdb(args.source)
    record["source"]["chains_present"] = {c: len(r) for c, r in chains.items()}
    record["source"]["notes"] = source_notes

    specs = [parse_chain_spec(s) for s in args.chain]
    if not specs:
        raise Refused("at least one --chain is required")
    out_ids = [s[2] for s in specs]
    if len(set(out_ids)) != len(out_ids):
        raise Refused(f"two --chain selections share an output id: {out_ids}")

    edits = [parse_edit_spec(s) for s in args.edit]
    for e in edits:
        if e["chain"] not in out_ids:
            raise Refused(
                f"--edit on chain {e['chain']!r}, which is not an output chain "
                f"({out_ids}). Edits name the OUTPUT chain id and the SOURCE "
                f"residue number, which the output keeps."
            )
    msas = dict(parse_msa_spec(s) for s in args.msa)
    for cid in msas:
        if cid not in out_ids:
            raise Refused(f"--msa on chain {cid!r}, not an output chain ({out_ids})")

    record["selection"] = {"numbering": args.numbering, "chains": []}
    record["edits"] = []
    record["residue_map"] = {}
    record["result"] = {"chains": []}

    prepared = []          # (out_id, [Residue, ...], wt_offset, {resseq: wt_pos})
    for src_id, ranges, out_id in specs:
        if src_id not in chains:
            raise Refused(
                f"--chain {src_id}: not in {args.source} (present: "
                f"{sorted(chains)})"
            )
        residues = chains[src_id]
        if ranges:
            wanted, gaps = residues_in_ranges(
                residues, ranges, src_id, args.allow_gaps
            )
        else:
            wanted, gaps = list(residues), []
        bad = [r for r in wanted if r.resname not in THREE_TO_ONE]
        if bad:
            raise Refused(
                f"chain {src_id}: non-standard residues in the crop "
                f"({[r.resname + str(r.resseq) for r in bad][:8]}). They cannot "
                f"be compared against the WT sequence."
            )
        ins = [r for r in wanted if r.icode.strip()]
        if args.wt_align == "sequential":
            # The i-th kept residue is the i-th WT position. Only valid when
            # the WT record IS this crop's resolved sequence, so the lengths
            # must agree exactly - otherwise the mapping is a guess.
            if len(wanted) != len(wt_seq):
                raise Refused(
                    f"chain {src_id}: --wt-align sequential needs the WT "
                    f"record to BE the resolved sequence, but the crop has "
                    f"{len(wanted)} residues and the record has {len(wt_seq)}. "
                    f"Use --wt-align offset with a continuously numbered WT, "
                    f"or crop to the residues the record covers."
                )
            offset = None
            wtpos = {r.resseq: i + 1 for i, r in enumerate(wanted)}
            hit = sum(1 for i, r in enumerate(wanted) if wt_seq[i] == r.one)
            identity = hit / len(wanted) if wanted else 0.0
            if identity < 0.9:
                raise Refused(
                    f"chain {src_id}: mapped in order onto the WT record, only "
                    f"{identity:.1%} of residues agree. The record is probably "
                    f"not this chain's sequence."
                )
        else:
            offset, identity, _ = detect_offset(
                wanted, wt_seq, args.wt_offset, args.wt_numbering
            )
            wtpos = {r.resseq: r.resseq + offset for r in wanted}
        record["selection"]["chains"].append({
            "source_chain": src_id,
            "output_chain": out_id,
            "ranges": format_spec(ranges) if ranges else "whole chain",
            "n_residues": len(wanted),
            "numbering": args.numbering,
            "wt_align": args.wt_align,
            "wt_offset": offset,
            "wt_offset_source": (
                "not applicable (sequential)" if args.wt_align == "sequential"
                else "declared" if args.wt_offset is not None else "detected"),
            "identity_to_wt_before_edits": round(identity, 6),
            "insertion_codes": [f"{r.resseq}{r.icode.strip()}" for r in ins],
            "unresolved_in_range": gaps,
        })
        if ins:
            raise Refused(
                f"chain {src_id}: residues with insertion codes "
                f"{[f'{r.resseq}{r.icode.strip()}' for r in ins][:8]} are in the "
                f"crop. Their WT position is not residue_number + offset, so the "
                f"map this tool writes would be wrong."
            )
        prepared.append((out_id, wanted, offset, wtpos))

    # ---- reconcile against WT, edit by declared edit -----------------------
    used_edits = set()
    for out_id, residues, offset, wtpos in prepared:
        kept = []
        for res in residues:
            idx = wtpos[res.resseq] - 1
            if not (0 <= idx < len(wt_seq)):
                raise Refused(
                    f"chain {out_id} residue {res.resseq}: WT position "
                    f"{idx + 1} is outside the {len(wt_seq)}-residue WT record. "
                    f"The offset ({offset}) or the crop is wrong."
                )
            wt_one = wt_seq[idx]
            edit = next(
                (e for e in edits
                 if e["chain"] == out_id and e["resnum"] == res.resseq), None
            )
            if edit is not None:
                used_edits.add(id(edit))
                if edit["from"] != res.resname:
                    raise Refused(
                        f"--edit {out_id}:{res.resseq}: declares FROM="
                        f"{edit['from']} but the structure has {res.resname}."
                    )
                if THREE_TO_ONE[edit["to"]] != wt_one:
                    raise Refused(
                        f"--edit {out_id}:{res.resseq}: declares TO="
                        f"{edit['to']} ({THREE_TO_ONE[edit['to']]}) but WT "
                        f"position {idx + 1} is {wt_one}. An edit may only move "
                        f"a residue TOWARDS the declared WT."
                    )
                entry = {
                    "output_chain": out_id,
                    "source_resnum": res.resseq,
                    "wt_position": idx + 1,
                    "from": edit["from"],
                    "to": edit["to"],
                    "op": edit["op"],
                }
                if edit["op"] == "drop_residue":
                    entry["atoms_removed"] = res.atom_names()
                    record["edits"].append(entry)
                    continue
                entry["atoms_removed"] = apply_truncate_to_cb(res, edit["to"])
                entry["atoms_kept"] = res.atom_names()
                entry["note"] = (
                    "side chain past CB deleted; the residue is modelled as "
                    f"{edit['to']} with an unresolved side chain, NOT relabelled"
                )
                record["edits"].append(entry)
            elif res.one != wt_one:
                raise Refused(
                    f"chain {out_id} residue {res.resseq} is {res.resname} "
                    f"({res.one}) but the declared WT has {wt_one} at position "
                    f"{idx + 1}. Nothing is written. Either supply a "
                    f"WT-matching source structure, or declare the edit that "
                    f"reconciles them:\n"
                    f"    --edit '{out_id}:{res.resseq}:{res.resname}>"
                    f"{ONE_TO_THREE[wt_one]}:truncate_to_cb'   "
                    f"(quote it - the shell eats the '>')"
                )
            kept.append(res)
        residues[:] = kept

    unused = [e for e in edits if id(e) not in used_edits]
    if unused:
        raise Refused(
            "declared edits that match no residue in the crop: "
            + ", ".join(f"{e['chain']}:{e['resnum']}" for e in unused)
            + ". An edit that does nothing means the record describes a "
              "different structure from the one being written."
        )

    # ---- resulting sequences ----------------------------------------------
    for out_id, residues, offset, wtpos in prepared:
        seq = "".join(r.one for r in residues)
        mism = [i for i, r in enumerate(residues)
                if wt_seq[wtpos[r.resseq] - 1] != r.one]
        if mism:                                    # belt and braces
            raise Refused(
                f"chain {out_id}: {len(mism)} residue(s) still disagree with WT "
                f"after the declared edits. Refusing to write."
            )
        record["result"]["chains"].append({
            "output_chain": out_id,
            "length": len(seq),
            "sequence": seq,
            "sequence_sha256": msa_check.sequence_sha256(seq),
        })
        # Three numberings, because the shard keeps none of the first two:
        # it renumbers every chain from 1 and renames chains A, B, ...
        record["residue_map"][out_id] = [
            {
                "source": f"{r.chain}{r.resseq}",
                "source_resnum": r.resseq,
                "prepared_resnum": r.resseq,      # the written PDB keeps it
                "shard_resnum": i + 1,            # verified against the shard
                "wt_position": wtpos.get(r.resseq),
                "residue": r.one,
                "edited": any(
                    e["output_chain"] == out_id and e["source_resnum"] == r.resseq
                    for e in record["edits"]
                ),
            }
            for i, r in enumerate(residues)
        ]

    # ---- MSA pairing, before anything is written ---------------------------
    record["msa"] = []
    seq_of = {c["output_chain"]: c["sequence"] for c in record["result"]["chains"]}
    for out_id in out_ids:
        path = msas.get(out_id)
        if path is None:
            record["msa"].append({
                "output_chain": out_id,
                "status": "not supplied",
                "note": "no --msa for this chain; the pair was NOT validated",
            })
            continue
        a3m = path if path.endswith(".a3m") else os.path.join(path, "non_pairing.a3m")
        try:
            vr = msa_check.validate_a3m(
                seq_of[out_id], a3m, label=f"output chain {out_id}",
                min_depth=args.min_depth,
            )
        except msa_check.MsaMismatch as exc:
            record["msa"].append({
                "output_chain": out_id, "msa_dir": path, "a3m": a3m,
                "status": "INVALID", "error": str(exc),
            })
            raise Refused(
                f"the prepared chain {out_id} does not match the MSA it is "
                f"meant to pair with:\n    {exc}"
            )
        vr.update({"output_chain": out_id, "msa_dir": path, "status": "valid"})
        record["msa"].append(vr)

    return prepared


def residues_in_ranges(residues, ranges, src_id, allow_gaps):
    """-> (residues, missing). Refuses on a gap unless it was asked for.

    A disordered loop is normal in a crystal and a legitimate thing to crop
    around - but so is a range written in the wrong numbering, and the two look
    identical here. --allow-gaps is how the caller says which one this is; the
    gap is recorded either way.
    """
    by_num = {}
    for r in residues:
        by_num.setdefault(r.resseq, []).append(r)
    out, missing = [], []
    for lo, hi in ranges:
        for num in range(lo, hi + 1):
            if num in by_num:
                out.extend(by_num[num])
            else:
                missing.append(num)
    if missing and not allow_gaps:
        from pxdesign.utils.residue_numbering import format_ranges
        raise Refused(
            f"--chain {src_id}: residues {format_ranges(missing)} in the "
            f"requested range are not in the structure. Either the range is "
            f"written in the wrong numbering, or they are unresolved in the "
            f"crystal and the crop really does have a chain break. Pass "
            f"--allow-gaps to say it is the second; the gap is recorded in "
            f"the provenance."
        )
    return out, missing


def format_spec(ranges):
    return ",".join(f"{a}-{b}" if a != b else str(a) for a, b in ranges)


def write_pdb(path, prepared):
    serial = 1
    with open(path, "w") as fh:
        for out_id, residues, _offset, _wtpos in prepared:
            for res in residues:
                for line in res.atoms:
                    line = line.ljust(80)
                    fh.write(
                        f"ATOM  {serial:>5}" + line[11:16] + " " + line[17:21]
                        + out_id + line[22:80] + "\n"
                    )
                    serial += 1
            fh.write(f"TER   {serial:>5}      {residues[-1].resname:>3} {out_id}"
                     f"{residues[-1].resseq:>4}\n")
            serial += 1
        fh.write("END\n")


def build_shard(pdb_path, out_dir, chain_ids):
    """Call the project's own shard writer. Imported late: it costs ~80 s."""
    from pxdesign.utils.infer import convert_to_bioassembly_dict
    input_dict = {
        "condition": {
            "structure_file": pdb_path,
            "filter": {"chain_id": list(chain_ids), "crop": {}},
        }
    }
    convert_to_bioassembly_dict(input_dict, out_dir=out_dir)
    return input_dict["condition"]["structure_file"]


def verify_shard(shard_path, record):
    """Read the shard back and compare it with what was intended."""
    import gzip
    import pickle
    with gzip.open(shard_path, "rb") as fh:
        d = pickle.load(fh)
    aa = d["atom_array"]
    seqs = d["sequences"]
    entity_of = {}
    for chain, entity in zip(aa.chain_id.tolist(), aa.label_entity_id.tolist()):
        entity_of.setdefault(chain, entity)
    shard_chains = sorted(entity_of)
    intended = [c["sequence"] for c in record["result"]["chains"]]
    if len(shard_chains) != len(intended):
        raise Refused(
            f"the shard has {len(shard_chains)} chains, {len(intended)} were "
            f"prepared. Refusing to keep it."
        )
    readback = []
    for shard_chain, want, meta in zip(shard_chains, intended,
                                       record["result"]["chains"]):
        got = seqs[entity_of[shard_chain]]
        if got != want:
            raise Refused(
                f"shard chain {shard_chain}: the sequence read back from the "
                f"shard differs from the prepared chain "
                f"{meta['output_chain']}.\n  shard: {got}\n  wanted: {want}"
            )
        mask = aa.chain_id == shard_chain
        res_ids = sorted({int(x) for x in aa.res_id[mask].tolist()})
        if res_ids != list(range(1, len(want) + 1)):
            raise Refused(
                f"shard chain {shard_chain}: residue ids are "
                f"{res_ids[0]}..{res_ids[-1]} ({len(res_ids)} of them), not "
                f"1..{len(want)}. The `shard_resnum` column of the residue map "
                f"would be wrong, so the shard is not kept."
            )
        readback.append({
            "shard_chain": shard_chain,
            "prepared_chain": meta["output_chain"],
            "label_entity_id": entity_of[shard_chain],
            "res_id_range": [res_ids[0], res_ids[-1]],
            "sequence_sha256": msa_check.sequence_sha256(got),
        })
        meta["shard_chain"] = shard_chain
        meta["shard_res_id_first"] = res_ids[0]
    return readback


def render_markdown(record):
    L = ["# Target provenance: " + record.get("name", "(unnamed)"), ""]
    L.append(f"Written by `{record['tool']}` on {record['generated']}"
             + (f" at repo commit `{record['repo_commit'][:10]}`"
                if record.get("repo_commit") else "") + ".")
    L.append("")
    L.append(f"**Status: {record['status'].upper()}**")
    if record.get("refusal"):
        L += ["", "```", record["refusal"], "```"]
    if "source" in record:
        L += ["", "## Source", "",
              f"- structure: `{record['source']['path']}`",
              f"- sha256: `{record['source']['sha256']}`"]
        for note in record["source"].get("notes", []):
            L.append(f"- {note}")
    if "wt" not in record:
        return "\n".join(L) + "\n"
    L += ["", "## Authoritative WT", "",
          f"- fasta: `{record['wt']['fasta']}`",
          f"- header: {record['wt']['header']}",
          f"- file sha256: `{record['wt']['fasta_sha256']}`",
          f"- sequence sha256: `{record['wt']['sequence_sha256']}` "
          f"({record['wt']['length']} aa)",
          f"- numbering: {record['wt']['numbering']} "
          f"(WT position = residue number + offset)"]
    L += ["", "## Chain selection and crop", "",
          "| source | ranges | output | n | numbering | WT offset | identity before edits |",
          "|---|---|---|---|---|---|---|"]
    for c in record.get("selection", {}).get("chains", []):
        L.append(f"| {c['source_chain']} | {c['ranges']} | {c['output_chain']} | "
                 f"{c['n_residues']} | {c['numbering']} | {c['wt_offset']} "
                 f"({c['wt_offset_source']}) | "
                 f"{c['identity_to_wt_before_edits']:.4f} |")
        if c.get("unresolved_in_range"):
            L.append(f"| | unresolved in range: "
                     f"{c['unresolved_in_range']} | | | | | |")
    L += ["", "## Structural edits", ""]
    if not record.get("edits"):
        L.append(
            "None applied before the refusal above."
            if record["status"] == "refused"
            else "None. The crop matches the declared WT residue for residue."
        )
    for e in record.get("edits", []):
        L.append(f"- **{e['op']}** chain {e['output_chain']} residue "
                 f"{e['source_resnum']} (WT position {e['wt_position']}): "
                 f"{e['from']} -> {e['to']}; atoms removed: "
                 f"{', '.join(e['atoms_removed']) or 'none'}"
                 + (f"; kept: {', '.join(e.get('atoms_kept', []))}"
                    if e.get("atoms_kept") else ""))
        if e.get("note"):
            L.append(f"  - {e['note']}")
    L += ["", "## Result", ""]
    for c in record.get("result", {}).get("chains", []):
        L.append(f"- chain {c['output_chain']}"
                 + (f" (shard chain {c['shard_chain']})" if c.get("shard_chain") else "")
                 + f": {c['length']} aa, sha256 `{c['sequence_sha256']}`")
    L += ["", "## MSA pairing", ""]
    for m in record.get("msa", []):
        if m.get("status") == "valid":
            L.append(f"- chain {m['output_chain']}: `{m['msa_dir']}` - VALID, "
                     f"{m['depth_unique']} unique / {m['depth_headers']} records, "
                     f"{m['match']}")
        elif m.get("status") == "INVALID":
            L.append(f"- chain {m['output_chain']}: `{m['msa_dir']}` - **INVALID**: "
                     f"{m['error']}")
        else:
            L.append(f"- chain {m['output_chain']}: {m['note']}")
    L += ["", "## Residue map", "",
          "Full source <-> output <-> WT map in `provenance.json` "
          "(`residue_map`). The shard renumbers every chain from 1, so the "
          "source numbering is NOT recoverable from the shard alone - this map "
          "is the only record of it.", ""]
    return "\n".join(L) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--source", required=True, help="source structure (.pdb)")
    p.add_argument("--wt-fasta", required=True,
                   help="authoritative WT sequence, one FASTA record")
    p.add_argument("--wt-numbering", default="mature",
                   choices=("mature", "precursor"),
                   help="what the source residue numbers count from [mature]")
    p.add_argument("--wt-offset", type=int, default=None,
                   help="WT position = residue number + OFFSET; detected when "
                        "omitted, and an ambiguous or poor fit is refused")
    p.add_argument("--wt-align", default="offset",
                   choices=("offset", "sequential"),
                   help="how the structure maps onto --wt-fasta. 'offset' (the "
                        "default) assumes WT position = residue number + a "
                        "constant, which breaks whenever the chain has "
                        "unresolved stretches and the WT record is the "
                        "RESOLVED sequence: the shift changes at every gap. "
                        "'sequential' maps the i-th kept residue to the i-th "
                        "WT position, which is correct when --wt-fasta is the "
                        "structure's own resolved sequence")
    p.add_argument("--numbering", default="auth", choices=("auth",),
                   help="numbering scheme of --chain ranges [auth]")
    p.add_argument("--chain", action="append", default=[], metavar="SRC[:RANGES][=OUT]")
    p.add_argument("--msa", action="append", default=[], metavar="CHAIN=DIR_OR_A3M")
    p.add_argument("--edit", action="append", default=[],
                   metavar="CHAIN:RESNUM:FROM>TO:OP")
    p.add_argument("--min-depth", type=int, default=msa_check.DEFAULT_MIN_DEPTH)
    p.add_argument("--name", default=None, help="output basename")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--allow-gaps", action="store_true",
                   help="accept residues missing from the requested range "
                        "(a disordered loop); they are recorded")
    p.add_argument("--no-shard", action="store_true",
                   help="write the PDB and the record, skip the .pkl.gz "
                        "(which costs the protenix import)")
    args = p.parse_args(argv)

    name = args.name or os.path.splitext(os.path.basename(args.source))[0] + "_prepared"
    os.makedirs(args.out_dir, exist_ok=True)
    json_path = os.path.join(args.out_dir, "provenance.json")
    md_path = os.path.join(args.out_dir, "PROVENANCE.md")

    record = new_record()
    record["name"] = name
    try:
        prepared = build(args, record)
        pdb_path = os.path.join(args.out_dir, f"{name}.pdb")
        write_pdb(pdb_path, prepared)
        record["prepared_pdb"] = {
            "path": os.path.abspath(pdb_path),
            "sha256": msa_check.file_sha256(pdb_path),
        }
        if args.no_shard:
            record["shard"] = {"status": "not built (--no-shard)"}
        else:
            shard = build_shard(pdb_path, args.out_dir, [o for o, _, _, _ in prepared])
            # convert_to_bioassembly_dict writes an intermediate .cif beside
            # the shard. Nothing reads it afterwards - the residue map is built
            # from the source .pdb and the shard - and leaving it invites it
            # being mistaken for the prepared structure.
            intermediate = os.path.join(args.out_dir, f"{name}.cif")
            if os.path.isfile(intermediate):
                os.remove(intermediate)
            try:
                record["shard"] = {
                    "path": os.path.abspath(shard),
                    "sha256": msa_check.file_sha256(shard),
                    "chains": verify_shard(shard, record),
                    "status": "verified against the prepared sequences",
                }
            except Refused:
                os.remove(shard)
                raise
        record["status"] = "written"
    except Refused as exc:
        record["status"] = "refused"
        record["refusal"] = str(exc)
        _dump(record, json_path, md_path)
        print(f"REFUSED: {exc}", file=sys.stderr)
        print(f"(record written to {json_path})", file=sys.stderr)
        return 2

    _dump(record, json_path, md_path)
    print(f"wrote {record['prepared_pdb']['path']}")
    if record["shard"].get("path"):
        print(f"wrote {record['shard']['path']}")
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    for c in record["result"]["chains"]:
        print(f"  chain {c['output_chain']}: {c['length']} aa "
              f"sha256={c['sequence_sha256'][:16]}")
    return 0


def _dump(record, json_path, md_path):
    os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
    with open(json_path, "w") as fh:
        json.dump(record, fh, indent=1, sort_keys=False)
        fh.write("\n")
    with open(md_path, "w") as fh:
        fh.write(render_markdown(record))


if __name__ == "__main__":
    sys.exit(main())
