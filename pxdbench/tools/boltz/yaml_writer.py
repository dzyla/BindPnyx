"""Boltz-2 input YAML generation.

One YAML per design. The calibrated configuration is: target chains carry their
cached a3m, the binder is single-sequence (`msa: empty`), template-free. Changing
any of that invalidates the gate thresholds.
"""
import json
import os


#: Boltz's own default for a pocket constraint, in angstroms.
DEFAULT_POCKET_MAX_DISTANCE = 6.0


def validate_pocket_contacts(pocket_contacts, target_chains):
    """Reject contacts that would silently constrain the wrong thing.

    Separate from write_design_yaml so a caller can check before spending GPU
    time. Two ways to get this wrong, both silent in Boltz:

    - the chain id is the YAML's, and target_chains_from_orig_seqs renames the
      campaign's (A0 -> A), so the wrong spelling is easy to pass;
    - the residue number is a 1-BASED INDEX INTO THE YAML SEQUENCE, not an
      author numbering. Verified against boltz 2.2.1: contacts
      [[B,3],[B,7],[D,5]] come back as (binder=2, [(0,2),(0,6),(1,4)]) -
      chain index and zero-based residue index. A number past the end of the
      sequence resolves somewhere unintended rather than erroring, so the
      bound is checked here.
    """
    if not pocket_contacts:
        raise ValueError(
            "pocket_contacts is empty: that would constrain nothing. Pass "
            "None to write an unconstrained YAML."
        )
    lengths = {c["id"]: len(c["seq"]) for c in target_chains}
    for chain_id, res_id in pocket_contacts:
        if chain_id not in lengths:
            raise ValueError(
                f"pocket contact {chain_id}{res_id}: {chain_id!r} is not a "
                f"target chain {sorted(lengths)}. A contact on an absent chain "
                f"(or on the binder) constrains nothing silently."
            )
        if not 1 <= int(res_id) <= lengths[chain_id]:
            raise ValueError(
                f"pocket contact {chain_id}{res_id} is outside chain "
                f"{chain_id!r}, which is {lengths[chain_id]} residues long. "
                f"The number is a 1-based position in the YAML sequence, not "
                f"an author numbering; out of range resolves silently to the "
                f"wrong residue."
            )
    return list(pocket_contacts)


def write_design_yaml(
    path,
    target_chains,
    binder_id,
    binder_seq,
    pocket_contacts=None,
    pocket_max_distance=DEFAULT_POCKET_MAX_DISTANCE,
):
    """Write one Boltz complex YAML. Returns `path`.

    target_chains: [{"id": "B", "seq": "...", "msa": "/path/chainB.a3m"}, ...]
    The binder is written LAST, so it is the last chain (see the chain-order
    constraint: the binder must be chain A or the last chain, because
    tasks/base.py hardcodes index 0 for "A" and ptx.py resolves None to the
    last chain).

    pocket_contacts: optional [(chain_id, res_id), ...] to constrain the binder
    against. Boltz is otherwise template-free and knows nothing about the
    requested epitope, so it re-docks the binder wherever it likes - measured
    across 8 designs, 7-42 A from where the diffusion placed it. Passing
    contacts emits a `pocket` constraint so the SCORED pose is the requested
    interface.

    This is OPT-IN and defaults to None, which emits byte-identical YAML to the
    unconstrained writer. Supplying contacts changes the Boltz input, so
    `bz_*` scores from such a run are NOT comparable to the gate thresholds in
    backend.GATE, which were calibrated unconstrained.
    """
    if not binder_seq:
        raise ValueError("binder sequence is empty")
    target_ids = [c["id"] for c in target_chains]
    if binder_id in target_ids:
        raise ValueError(
            f"binder id {binder_id!r} collides with a target chain id {target_ids}"
        )
    if pocket_contacts is not None:
        validate_pocket_contacts(pocket_contacts, target_chains)

    lines = ["version: 1", "sequences:"]
    for chain in target_chains:
        lines.append(
            f"  - protein: {{id: {chain['id']}, sequence: {chain['seq']}, "
            f"msa: {chain['msa']}}}"
        )
    lines.append(
        f"  - protein: {{id: {binder_id}, sequence: {binder_seq}, msa: empty}}"
    )

    if pocket_contacts:
        contacts = ", ".join(f"[{c}, {int(r)}]" for c, r in pocket_contacts)
        lines += [
            "constraints:",
            "  - pocket:",
            f"      binder: {binder_id}",
            f"      contacts: [{contacts}]",
            f"      max_distance: {float(pocket_max_distance)}",
        ]

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\n")
    return path


def write_manifest(path, entries):
    """Write the design manifest that `predict` reads back. Returns `path`."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as handle:
        json.dump({"designs": list(entries)}, handle, indent=2)
    return path
