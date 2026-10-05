"""One residue must be the same residue at every stage.

```
config auth D535
  -> record:    auth D535 = work B24 = seq_index 24 = HIS
  -> diffusion: hotspot feature set at token index 23
  -> Boltz:     pocket contact (chain B, seq 24)
  -> export:    residue B24 exists and is HIS
  -> metric:    B24 evaluated
```

Any disagreement fails the run. Review A2 moved this ahead of authorized
scoring, because a number is not interpretable if the feature mask, residue map
or exported pose was wrong during that run - checking afterwards cannot rescue
the compute.

R9: existence is not conditioning. A diffusion structure containing a HIS at
the expected residue proves the residue is there, not that its hotspot feature
was set, so the check runs against the actual parser-produced mask.

Legacy runs report `not_evaluated` and must never display as a verified pass.
A campaign that DECLARES schema 2 and has no record is a failure, not legacy.
"""
import json

import pytest

from pxdbench.targets.consistency import (
    ConsistencyStatus,
    check_export_residues,
    check_feature_mask,
    check_map_freshness,
    check_pocket_contacts,
    evaluate_record_presence,
)
from pxdbench.targets.epitope_policy import resolve_policy
from pxdbench.targets.record import SelectionMode, build_record

POLICY = {
    "required": {"D": ["H512"]},
    "advisory": {"D": ["D514"]},
    "forbidden": {"D": ["A513"]},
}


@pytest.fixture
def record(hadq_map):
    policy = resolve_policy(POLICY, hadq_map, numbering="auth")
    return build_record(
        task_name="t", policy=policy, residue_map=hadq_map,
        selection_mode=SelectionMode.POLICY_V1, source_digest="s",
        shard_digest="h", sequence_digests={"B": "q"},
        original_requests=POLICY,
    )


# --- the feature mask: conditioning, not existence -------------------------

def test_a_correct_mask_passes(record):
    """Required + advisory are work B1 and B3 -> token indices 0 and 2."""
    mask = {"B": [True, False, True, False]}
    assert check_feature_mask(record, mask) == []


def test_a_mask_missing_a_required_token_fails(record):
    """The failure a residue-identity check cannot see: the residue is there,
    its hotspot feature is not set."""
    mask = {"B": [False, False, True, False]}
    failures = check_feature_mask(record, mask)
    assert failures and "B1" in failures[0]


def test_a_mask_setting_a_forbidden_token_fails(record):
    """B2 is forbidden. Conditioning on it is the inverse of the request."""
    mask = {"B": [True, True, True, False]}
    failures = check_feature_mask(record, mask)
    assert failures and "B2" in failures[0]


def test_a_mask_too_short_for_a_resolved_residue_fails(record):
    """The record resolves up to work B3, so a length-3 mask is adequate; a
    length-2 one cannot address B3 at all. The record does not store chain
    lengths, so coverage of the RESOLVED residues is what can be checked."""
    failures = check_feature_mask(record, {"B": [True, False]})
    assert failures and "length" in failures[0].lower()


def test_an_all_zero_mask_fails_rather_than_passing_quietly(record):
    """An unconditioned run looks exactly like a conditioned one downstream."""
    failures = check_feature_mask(record, {"B": [False] * 4})
    assert failures


def test_a_missing_chain_in_the_mask_fails(record):
    failures = check_feature_mask(record, {})
    assert failures and "B" in failures[0]


# --- the export ------------------------------------------------------------

def test_an_export_containing_every_resolved_residue_passes(record):
    present = {("B", 1): "HIS", ("B", 2): "ALA", ("B", 3): "ASP", ("B", 4): "GLN"}
    assert check_export_residues(record, present) == []


def test_a_cropped_out_required_residue_fails(record):
    present = {("B", 2): "ALA", ("B", 3): "ASP", ("B", 4): "GLN"}
    failures = check_export_residues(record, present)
    assert failures and "B1" in failures[0]


def test_a_swapped_export_fails_on_identity(record):
    """Right numbering, wrong structure: B1 is a lysine here, not a histidine."""
    present = {("B", 1): "LYS", ("B", 2): "ALA", ("B", 3): "ASP", ("B", 4): "GLN"}
    failures = check_export_residues(record, present)
    assert failures and "HIS" in failures[0] and "LYS" in failures[0]


def test_a_forbidden_residue_must_also_exist_in_the_export(record):
    """An absent forbidden residue cannot produce a contact violation, so its
    absence is unevaluable rather than a successful avoidance."""
    present = {("B", 1): "HIS", ("B", 3): "ASP", ("B", 4): "GLN"}
    failures = check_export_residues(record, present)
    assert failures and "B2" in failures[0]


# --- a stale map -----------------------------------------------------------

def test_a_matching_map_digest_passes(record, hadq_map):
    assert check_map_freshness(record, hadq_map) == []


def test_a_changed_map_fails(record, residues):
    """Re-reading a stale map against changed structures must fail."""
    from pxdbench.targets.residue_map import TargetResidueMap

    other = TargetResidueMap.from_residue_lists(
        source={"D": residues("D", 600, "HADQ")},
        shard={"B": residues("B", 1, "HADQ")},
        chain_filter=["B"], sequences={"B": "HADQ"},
    )
    failures = check_map_freshness(record, other)
    assert failures and "digest" in failures[0].lower()


# --- pocket contacts -------------------------------------------------------

def test_pocket_contacts_use_seq_index_and_required_only(record):
    """R9: only required residues are expected when pocket conditioning is on;
    advisory and forbidden have different obligations."""
    assert check_pocket_contacts(record, [("B", 1)]) == []


def test_a_pocket_contact_on_an_advisory_residue_fails(record):
    failures = check_pocket_contacts(record, [("B", 1), ("B", 3)])
    assert failures and "B3" in failures[0]


def test_a_pocket_contact_at_the_wrong_index_fails(record):
    """An off-by-one here points the constraint at a different residue."""
    failures = check_pocket_contacts(record, [("B", 2)])
    assert failures


def test_a_missing_required_pocket_contact_fails(record):
    failures = check_pocket_contacts(record, [])
    assert failures and "B1" in failures[0]


# --- record presence: the legacy / schema-2 distinction --------------------

def test_a_legacy_run_is_not_evaluated_not_a_pass(tmp_path):
    """Legacy inspection may say 'not evaluated'; it must never be displayed
    as a verified pass."""
    status, detail = evaluate_record_presence(
        str(tmp_path), declared_schema=None
    )
    assert status is ConsistencyStatus.NOT_EVALUATED
    assert "legacy" in detail.lower() or "no schema-2" in detail.lower()


def test_a_schema_2_campaign_with_no_record_fails(tmp_path):
    """Exiting zero here would let provenance deletion pass as legacy."""
    status, detail = evaluate_record_presence(str(tmp_path), declared_schema=2)
    assert status is ConsistencyStatus.FAILED
    assert "schema 2" in detail.lower()


def test_a_schema_2_campaign_with_a_record_is_evaluated(tmp_path, record):
    (tmp_path / "resolved_hotspots.json").write_text(json.dumps(record))
    status, _detail = evaluate_record_presence(str(tmp_path), declared_schema=2)
    assert status is ConsistencyStatus.OK


def test_a_schema_1_record_in_a_schema_2_campaign_fails(tmp_path):
    (tmp_path / "resolved_hotspots.json").write_text(
        json.dumps({"name": "t", "hotspot_resolved": {"B": [1]},
                    "numbering": "design"})
    )
    status, detail = evaluate_record_presence(str(tmp_path), declared_schema=2)
    assert status is ConsistencyStatus.FAILED
    assert "schema" in detail.lower()


def test_not_evaluated_is_not_truthy_as_success():
    """Guard against `if status:` reading not-evaluated as a pass."""
    assert ConsistencyStatus.NOT_EVALUATED is not ConsistencyStatus.OK
    assert not ConsistencyStatus.NOT_EVALUATED.is_success
    assert ConsistencyStatus.OK.is_success
    assert not ConsistencyStatus.FAILED.is_success
