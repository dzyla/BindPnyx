"""Stage consistency: one residue is the same residue at every stage.

```
config auth D535
  -> record:    auth D535 = work B24 = seq_index 24 = HIS
  -> diffusion: hotspot feature set at token index 23
  -> Boltz:     pocket contact (chain B, seq 24)
  -> export:    residue B24 exists and is HIS
  -> metric:    B24 evaluated
```

Review A2 moved these checks ahead of authorized scoring. A number is not
interpretable if the feature mask, residue map or exported pose was wrong
during that run, and checking afterwards cannot rescue the compute. The
post-prediction checks are therefore meant to run as each result arrives, not
only in a final reporting script.

**Existence is not conditioning** (R9). A diffusion structure containing a HIS
at the expected residue proves the residue is there - not that its hotspot
feature was set. `check_feature_mask` runs against the actual
parser-produced mask, which is the only artifact that can distinguish an
unconditioned run from a conditioned one.

Legacy runs report `NOT_EVALUATED`, which is deliberately not success. A
campaign that *declares* schema 2 and has no record is a FAILURE: exiting zero
there would let provenance deletion pass as legacy inspection.

Standard library only.
"""
import os
from enum import Enum

from pxdbench.targets.record import RECORD_NAME


class ConsistencyStatus(Enum):
    """Three outcomes, and only one of them is success."""

    OK = "ok"
    FAILED = "failed"
    NOT_EVALUATED = "not_evaluated"

    @property
    def is_success(self):
        """`NOT_EVALUATED` is not a pass.

        A property rather than truthiness, because `if status:` would read
        not-evaluated as success - which is exactly how an unverified legacy
        run comes to be displayed as a verified one.
        """
        return self is ConsistencyStatus.OK


def _roles(record):
    return record["policy"]["roles"]


def _keys(rows):
    return [(r["work_chain"], r["work_res_id"]) for r in rows]


def check_feature_mask(record, mask):
    """The parser-produced hotspot mask matches required + advisory exactly.

    `mask` is {work_chain: [bool per token]}, 0-based, as the featurizer
    produced it. Conditioning on a forbidden residue is the inverse of the
    request, and an all-zero mask is an unconditioned run that looks identical
    downstream, so both fail.
    """
    failures = []
    expected = set(_keys(_roles(record)["required"]) + _keys(_roles(record)["advisory"]))
    forbidden = set(_keys(_roles(record)["forbidden"]))

    chains = {chain for chain, _res in expected | forbidden}
    for chain in sorted(chains):
        if chain not in mask:
            failures.append(
                f"chain {chain!r} is absent from the feature mask; the record "
                f"resolves residues on it"
            )
            continue
        tokens = list(mask[chain])
        on_chain = [
            r for r in record["policy"]["roles"]["required"]
            + record["policy"]["roles"]["advisory"]
            + record["policy"]["roles"]["forbidden"]
            if r["work_chain"] == chain
        ]
        longest = max((r["work_res_id"] for r in on_chain), default=0)
        if len(tokens) < longest:
            failures.append(
                f"chain {chain!r} feature mask has length {len(tokens)}, but "
                f"the record resolves residue {chain}{longest}"
            )
            continue
        for chain_id, res_id in sorted(expected):
            if chain_id != chain:
                continue
            if not tokens[res_id - 1]:
                failures.append(
                    f"{chain_id}{res_id} is required or advisory but its "
                    f"hotspot feature is NOT set at token index {res_id - 1}. "
                    f"The residue existing is not the same as it being "
                    f"conditioned on."
                )
        for chain_id, res_id in sorted(forbidden):
            if chain_id != chain:
                continue
            if tokens[res_id - 1]:
                failures.append(
                    f"{chain_id}{res_id} is FORBIDDEN but its hotspot feature "
                    f"is set at token index {res_id - 1}; the model has no "
                    f"coldspot channel, so this conditions toward the residue "
                    f"the policy forbids"
                )
    if expected and not failures:
        if not any(any(mask.get(c, [])) for c, _ in expected):
            failures.append(
                "the feature mask is empty for every resolved chain: this run "
                "is unconditioned, which is indistinguishable downstream from "
                "a conditioned one"
            )
    return failures


def check_export_residues(record, present):
    """Every resolved residue exists in the export with the recorded identity.

    `present` is {(chain, res_id): resname}. Catches a cropped-out residue and
    a swapped export - right numbering, wrong structure. A forbidden residue
    must exist too: if it is absent it cannot produce a contact violation, so
    its absence is unevaluable rather than successful avoidance.
    """
    failures = []
    for role in ("required", "advisory", "forbidden"):
        for row in _roles(record)[role]:
            key = (row["work_chain"], row["work_res_id"])
            label = f"{row['work_chain']}{row['work_res_id']}"
            if key not in present:
                failures.append(
                    f"{label} ({role}) is in the record but absent from the "
                    f"export"
                    + (
                        "; an absent forbidden residue cannot produce a "
                        "contact violation, so this is unevaluable rather "
                        "than avoided"
                        if role == "forbidden"
                        else ""
                    )
                )
                continue
            if present[key] != row["resname"]:
                failures.append(
                    f"{label} ({role}) is {row['resname']} in the record but "
                    f"{present[key]} in the export - the numbering matches "
                    f"and the structure does not"
                )
    return failures


def check_map_freshness(record, residue_map):
    """The map in hand is the map the record was built from."""
    if residue_map.map_digest != record["map_digest"]:
        return [
            f"map digest mismatch: the record was built from "
            f"{record['map_digest']!r} but the map in hand is "
            f"{residue_map.map_digest!r}. A stale map re-read against changed "
            f"structures silently remaps every residue."
        ]
    return []


def check_pocket_contacts(record, contacts):
    """Pocket contacts are exactly the required residues, by `seq_index`.

    Only required residues are expected when pocket conditioning is enabled
    (R9): advisory residues are conditioned but not constrained, and forbidden
    residues must not be constrained toward at all.
    """
    failures = []
    required = {
        (r["work_chain"], r["seq_index"]): r for r in _roles(record)["required"]
    }
    advisory = {(r["work_chain"], r["seq_index"]) for r in _roles(record)["advisory"]}
    forbidden = {(r["work_chain"], r["seq_index"]) for r in _roles(record)["forbidden"]}
    supplied = {(str(c), int(i)) for c, i in contacts}

    for key in sorted(required.keys() - supplied):
        failures.append(
            f"{key[0]}{key[1]} is required but has no pocket contact"
        )
    for key in sorted(supplied & advisory):
        failures.append(
            f"{key[0]}{key[1]} is advisory but was constrained as a pocket "
            f"contact; advisory residues are conditioned, not constrained"
        )
    for key in sorted(supplied & forbidden):
        failures.append(
            f"{key[0]}{key[1]} is forbidden but was constrained as a pocket "
            f"contact"
        )
    for key in sorted(supplied - required.keys() - advisory - forbidden):
        failures.append(
            f"{key[0]}{key[1]} is a pocket contact but is not in the record; "
            f"a contact index that is not a resolved seq_index points at an "
            f"unintended residue"
        )
    return failures


def evaluate_record_presence(run_dir, declared_schema):
    """Whether this run's provenance is where it should be.

    `declared_schema=None` is a legacy run: NOT_EVALUATED, which is not
    success. `declared_schema=2` with a missing or schema-1 record is a
    FAILURE, because exiting zero there would let provenance deletion pass as
    legacy inspection.
    """
    path = os.path.join(run_dir, RECORD_NAME)
    if declared_schema is None:
        return (
            ConsistencyStatus.NOT_EVALUATED,
            "this run declares no schema; legacy inspection cannot verify "
            "stage consistency and must not be reported as a pass",
        )
    if declared_schema != 2:
        return (
            ConsistencyStatus.FAILED,
            f"unsupported declared schema {declared_schema!r}; expected 2",
        )
    if not os.path.exists(path):
        return (
            ConsistencyStatus.FAILED,
            f"this run declares schema 2 but {RECORD_NAME} is missing from "
            f"{run_dir!r}. Treating that as legacy would let provenance "
            f"deletion pass as an un-evaluated run.",
        )
    try:
        from pxdbench.targets.record import read_resolution_record

        read_resolution_record(path)
    except (ValueError, OSError) as exc:
        return ConsistencyStatus.FAILED, f"schema 2 record unreadable: {exc}"
    return ConsistencyStatus.OK, "schema 2 record present and readable"
