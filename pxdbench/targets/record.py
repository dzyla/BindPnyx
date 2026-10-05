"""The schema-2 resolution record: one artifact every stage reads.

Translating a target config into a campaign input was done by hand and recorded
nowhere, which is why a wrong translation was indistinguishable from a design
that missed. This record replaces that step. Every consumer reads it rather
than re-deriving:

| consumer | field |
| --- | --- |
| diffusion featurizer | `work_chain`, `work_res_id` |
| Boltz pocket constraint | YAML chain id, `seq_index` |
| epitope / pose-targeting metrics | `work_chain`, `work_res_id` |
| reports and summaries | `auth_chain`, `auth_res_id`, `resname` |

So a report can say `His535` instead of `B24`.

`hotspot_resolved` is preserved verbatim, carrying **required + advisory only**,
because `scripts/pose_targeting_report.py` and `scripts/hotspot_e2e_check.py`
read that key and treat it as the conditioning set. A forbidden residue there
would be conditioned on - the inverse of the request.

Writes are atomic and **required** (targeting-contracts section 3): a run whose
provenance is missing is not interpretable, so a write failure stops new-mode
execution rather than degrading to a warning. `out_dir=None` validates and
prepares without publishing.

Standard library only.
"""
import json
import os
from enum import Enum
from pathlib import Path

RECORD_NAME = "resolved_hotspots.json"
MAP_NAME = "target_residue_map.json"


class SelectionMode(str, Enum):
    """Resolved from configuration before scoring, never inferred.

    Not inferred from `ep_satisfied`, a CSV column, or a missing file: a
    missing policy artifact must never cause a downgrade to legacy.
    """

    LEGACY = "legacy"
    POLICY_V1 = "policy_v1"


class RecordWriteError(RuntimeError):
    """Provenance could not be persisted, so the run must not continue."""


def _atomic_write(path, payload):
    """Write JSON via a temporary file and rename, leaving no partial file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp.replace(path)
    return str(path)


def build_record(
    task_name,
    policy,
    residue_map,
    selection_mode,
    source_digest,
    shard_digest,
    sequence_digests,
    original_requests,
):
    """The schema-2 payload. Pure, so it can be validated without writing."""
    return {
        "schema": 2,
        "name": task_name,
        "selection_mode": SelectionMode(selection_mode).value,
        "declared_numbering": policy.numbering,
        # What was asked, verbatim, so a reader can see the request and not
        # only what it resolved to.
        "original_requests": original_requests,
        "policy": policy.to_dict(),
        "chain_table": dict(residue_map.chain_table),
        "map_digest": residue_map.map_digest,
        "policy_digest": policy.policy_digest,
        "source_digest": source_digest,
        "shard_digest": shard_digest,
        "sequence_digests": dict(sequence_digests or {}),
        # Preserved for the existing readers. Required + advisory only.
        "hotspot_resolved": policy.hotspot_resolved(),
        # Schema-1 key, kept because scripts/hotspot_e2e_check.py:63 reads it
        # directly. Dropping it was a real break, caught by pinning what those
        # readers actually require rather than assuming compatibility.
        "numbering": (
            "design (chain_id, res_id) - the same numbering as the exported "
            "complexes"
        ),
        "numbering_note": (
            "work = shard res_id; auth = the supplied source structure's "
            "numbering; seq_index = 1-based position in the sequence emitted "
            "to the predictor"
        ),
    }


def write_resolution_record(
    out_dir,
    task_name,
    policy,
    residue_map,
    selection_mode,
    source_digest,
    shard_digest,
    sequence_digests,
    original_requests,
    per_task_subdir=False,
):
    """Persist the record and its map. Returns the record path, or None.

    `out_dir=None` validates and builds the payload without publishing, which
    is what the preparation boundary uses when no task output directory was
    supplied.
    """
    record = build_record(
        task_name=task_name,
        policy=policy,
        residue_map=residue_map,
        selection_mode=selection_mode,
        source_digest=source_digest,
        shard_digest=shard_digest,
        sequence_digests=sequence_digests,
        original_requests=original_requests,
    )
    if out_dir is None:
        return None

    target_dir = os.path.join(out_dir, task_name) if per_task_subdir else out_dir
    try:
        residue_map.to_json(os.path.join(target_dir, MAP_NAME))
        return _atomic_write(os.path.join(target_dir, RECORD_NAME), record)
    except OSError as exc:
        raise RecordWriteError(
            f"could not persist provenance for task {task_name!r} in "
            f"{target_dir!r}: {exc}. This is fatal in "
            f"{SelectionMode(selection_mode).value} mode - a run whose record "
            f"is missing cannot be interpreted, and continuing would produce "
            f"numbers nobody can check."
        ) from exc


def read_resolution_record(path):
    """Load a record, rejecting anything that is not schema 2."""
    payload = json.loads(Path(path).read_text())
    if payload.get("schema") != 2:
        raise ValueError(
            f"{path}: record schema {payload.get('schema')!r}, expected 2. A "
            f"legacy record is reported as not-evaluated, never silently "
            f"upgraded."
        )
    return payload
