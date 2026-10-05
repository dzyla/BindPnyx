"""Does the targeting path work on targets other than the one it was built on?

Everything in this package was developed against one EGFR shard. That is
exactly the circumstance in which a pipeline quietly acquires target-specific
assumptions - a constant numbering offset, a chain-naming habit, a two-chain
shape.

This drives the real resolution path against a real structure:

    read structure -> renumber to a shard -> pair -> assert identities
      -> resolve roles -> write the record -> check stage consistency

Renumbering each source chain to 1..L is what the real shard-building step
does, so the pairing exercised here is the real one. No predictor and no GPU.

Lives in the package rather than in `scripts/` so the regression test can
import it without making `scripts` a package - any module at the repository
root shadows an installed package of the same name (CLAUDE.md section 2).
"""
import json
import os
import tempfile

from pxdbench.targets.consistency import (
    check_export_residues,
    check_feature_mask,
    check_map_freshness,
    check_pocket_contacts,
)
from pxdbench.targets.epitope_policy import resolve_policy
from pxdbench.targets.record import SelectionMode, write_resolution_record
from pxdbench.targets.residue_map import (
    THREE_TO_ONE,
    TargetResidueMap,
    UnsupportedMapError,
    _read_structure_residues,
)


#: Structurally diverse real targets. Each is a different protein, a different
#: chain count, or a different author numbering.
# Supply them with $PXD_SMOKE_TARGETS (os.pathsep-separated PDB paths); none ship with the repository.
DEFAULT_TARGETS = [p for p in os.environ.get("PXD_SMOKE_TARGETS", "").split(os.pathsep) if p]


def shard_from_source(source):
    """Renumber each chain to 1..L, which is what shard-building does."""
    shard, sequences = {}, {}
    for chain, entries in source.items():
        renumbered, letters = [], []
        index = 0
        for _c, _r, resname, ins in entries:
            if resname not in THREE_TO_ONE:
                # The real builder drops non-polymer entities; a nonstandard
                # polymer residue must still reach the map, which rejects it.
                renumbered.append((chain, index + 1, resname, ins))
                index += 1
                continue
            index += 1
            renumbered.append((chain, index, resname, ins))
            letters.append(THREE_TO_ONE[resname])
        shard[chain] = renumbered
        sequences[chain] = "".join(letters)
    return shard, sequences


def smoke_one(path):
    """Returns (status, detail). Exercises the whole preparation path."""
    name = os.path.basename(path)
    if not os.path.exists(path):
        return "absent", f"{name}: not on this machine"

    try:
        source = _read_structure_residues(path)
    except Exception as exc:  # noqa: BLE001
        return "fail", f"{name}: could not read structure: {exc}"

    chains = {c: len(v) for c, v in sorted(source.items())}
    shard, sequences = shard_from_source(source)

    try:
        mapping = TargetResidueMap.from_residue_lists(
            source=source,
            shard=shard,
            chain_filter=sorted(source),
            sequences=sequences,
        )
    except UnsupportedMapError as exc:
        # A rejection is a RESULT, not a crash: v1 is bounded on purpose, and
        # the whole point is that it says which structure it cannot take.
        return "rejected", f"{name}: chains={chains}\n      {exc}"
    except Exception as exc:  # noqa: BLE001
        return "fail", f"{name}: unexpected error: {type(exc).__name__}: {exc}"

    # Pick real residues to target: a histidine if there is one, else the
    # first three residues of the longest chain.
    longest = max(source, key=lambda c: len(source[c]))
    rows = [r for r in mapping.rows if r.work_chain == longest]
    his = [r for r in rows if r.resname == "HIS"]
    required = his[:1] or rows[:1]
    advisory = [r for r in rows if r not in required][:2]
    forbidden = [r for r in rows if r not in required + advisory][:1]

    def spec(row):
        return f"{THREE_TO_ONE[row.resname]}{row.auth_res_id}"

    policy_input = {
        "required": {required[0].auth_chain: [spec(required[0])]},
        "advisory": {a.auth_chain: [spec(a)] for a in advisory},
        "forbidden": {forbidden[0].auth_chain: [spec(forbidden[0])]}
        if forbidden
        else {},
    }
    try:
        policy = resolve_policy(policy_input, mapping, numbering="auth")
    except Exception as exc:  # noqa: BLE001
        return "fail", f"{name}: policy resolution failed: {exc}"

    with tempfile.TemporaryDirectory() as out_dir:
        record_path = write_resolution_record(
            out_dir=out_dir, task_name=name, policy=policy,
            residue_map=mapping, selection_mode=SelectionMode.POLICY_V1,
            source_digest="smoke", shard_digest="smoke",
            sequence_digests={c: "smoke" for c in sequences},
            original_requests=policy_input,
        )
        record = json.loads(open(record_path).read())

        failures = []
        failures += check_map_freshness(record, mapping)
        # The mask the featurizer WOULD produce for this policy.
        mask = {
            chain: [False] * len(shard[chain]) for chain in sorted(shard)
        }
        for row in policy.required + policy.advisory:
            mask[row.work_chain][row.work_res_id - 1] = True
        failures += check_feature_mask(record, mask)
        present = {
            (r.work_chain, r.work_res_id): r.resname for r in mapping.rows
        }
        failures += check_export_residues(record, present)
        failures += check_pocket_contacts(
            record, [(r.work_chain, r.seq_index) for r in policy.required]
        )

    if failures:
        return "fail", f"{name}: chains={chains}\n      " + "\n      ".join(failures)

    req = required[0]
    return "ok", (
        f"{name}: chains={chains}, {len(mapping.rows)} residues mapped\n"
        f"      required {req.auth_chain}{req.auth_res_id} ({req.resname}) "
        f"-> work {req.work_chain}{req.work_res_id}, seq_index {req.seq_index}"
    )


