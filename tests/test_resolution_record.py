"""The schema-2 resolution record: one artifact every stage reads.

Config-to-campaign-input translation was done by hand and recorded nowhere,
which is why a wrong translation was indistinguishable from a design that
missed. This record is what every consumer reads instead of re-deriving:
diffusion takes work numbering, Boltz takes `seq_index`, the metrics take work
numbering, and reports take author numbering so output says `His535` rather
than `B24`.

Targeting-contracts section 3: writes are atomic and REQUIRED - a write failure
stops new-mode execution, because a run whose provenance is missing is not
interpretable. Schema 2 carries selection mode, declared numbering, the
original requests, all resolved roles, the chain table, source/shard/sequence
hashes and the compatibility hotspot key.

`hotspot_resolved` is preserved verbatim so `scripts/pose_targeting_report.py`
and `scripts/hotspot_e2e_check.py` keep working unchanged.
"""
import json
import os

import pytest

from pxdbench.targets.epitope_policy import resolve_policy
from pxdbench.targets.record import (
    RecordWriteError,
    SelectionMode,
    write_resolution_record,
)

POLICY = {
    "required": {"D": ["H512"]},
    "advisory": {"D": ["D514"]},
    "forbidden": {"D": ["A513"]},
}


@pytest.fixture
def written(tmp_path, hadq_map):
    policy = resolve_policy(POLICY, hadq_map, numbering="auth")
    path = write_resolution_record(
        out_dir=str(tmp_path),
        task_name="t1",
        policy=policy,
        residue_map=hadq_map,
        selection_mode=SelectionMode.POLICY_V1,
        source_digest="src-abc",
        shard_digest="shard-def",
        sequence_digests={"B": "seq-ghi"},
        original_requests=POLICY,
    )
    return path, json.loads(open(path).read())


def test_the_record_declares_schema_2_and_its_mode(written):
    _path, record = written
    assert record["schema"] == 2
    assert record["selection_mode"] == "policy_v1"


def test_the_record_carries_every_role_in_both_numberings(written):
    _path, record = written
    required = record["policy"]["roles"]["required"][0]
    assert (required["auth_chain"], required["auth_res_id"]) == ("D", 512)
    assert (required["work_chain"], required["work_res_id"]) == ("B", 1)
    assert required["seq_index"] == 1
    assert required["resname"] == "HIS"
    assert [r["work_res_id"] for r in record["policy"]["roles"]["forbidden"]] == [2]


def test_the_compatibility_key_is_preserved_for_existing_readers(written):
    """pose_targeting_report.py and hotspot_e2e_check.py read this key."""
    _path, record = written
    assert record["hotspot_resolved"] == {"B": [1, 3]}


def test_the_compatibility_key_never_contains_a_forbidden_residue(written):
    """A513 is work B2. The legacy readers treat this key as conditioning."""
    _path, record = written
    assert 2 not in record["hotspot_resolved"]["B"]


def test_the_record_carries_the_chain_table_not_a_digit_strip(written):
    _path, record = written
    assert record["chain_table"] == {"B": "D"}


def test_the_record_carries_the_original_requests_verbatim(written):
    """So a reader can see what was asked, not only what it resolved to."""
    _path, record = written
    assert record["original_requests"] == POLICY
    assert record["declared_numbering"] == "auth"


def test_the_record_carries_input_digests(written):
    _path, record = written
    assert record["source_digest"] == "src-abc"
    assert record["shard_digest"] == "shard-def"
    assert record["sequence_digests"] == {"B": "seq-ghi"}
    assert record["map_digest"]
    assert record["policy_digest"]


def test_the_map_is_written_alongside_the_record(written):
    path, record = written
    map_path = os.path.join(os.path.dirname(path), "target_residue_map.json")
    assert os.path.exists(map_path)
    assert json.loads(open(map_path).read())["map_digest"] == record["map_digest"]


def test_the_record_is_named_so_existing_globs_find_it(written):
    path, _record = written
    assert os.path.basename(path) == "resolved_hotspots.json"


# --- the cases that must still serialise ------------------------------------

def test_a_forbidden_only_policy_is_recorded(tmp_path, hadq_map):
    """Its conditioning set is empty; the policy is not."""
    policy = resolve_policy({"forbidden": {"D": [513]}}, hadq_map, numbering="auth")
    path = write_resolution_record(
        out_dir=str(tmp_path), task_name="t", policy=policy,
        residue_map=hadq_map, selection_mode=SelectionMode.POLICY_V1,
        source_digest="s", shard_digest="h", sequence_digests={"B": "q"},
        original_requests={"forbidden": {"D": [513]}},
    )
    record = json.loads(open(path).read())
    assert record["hotspot_resolved"] == {}
    assert [r["work_res_id"] for r in record["policy"]["roles"]["forbidden"]] == [2]
    assert record["policy"]["has_enforced_requirements"] is True


def test_an_advisory_only_policy_records_no_enforced_requirements(tmp_path, hadq_map):
    policy = resolve_policy({"advisory": {"D": [512]}}, hadq_map, numbering="auth")
    path = write_resolution_record(
        out_dir=str(tmp_path), task_name="t", policy=policy,
        residue_map=hadq_map, selection_mode=SelectionMode.POLICY_V1,
        source_digest="s", shard_digest="h", sequence_digests={"B": "q"},
        original_requests={"advisory": {"D": [512]}},
    )
    record = json.loads(open(path).read())
    assert record["policy"]["has_enforced_requirements"] is False
    assert record["hotspot_resolved"] == {"B": [1]}


# --- atomicity and failure --------------------------------------------------

def test_the_write_is_atomic_leaving_no_partial_file(tmp_path, hadq_map):
    policy = resolve_policy(POLICY, hadq_map, numbering="auth")
    write_resolution_record(
        out_dir=str(tmp_path), task_name="t", policy=policy,
        residue_map=hadq_map, selection_mode=SelectionMode.POLICY_V1,
        source_digest="s", shard_digest="h", sequence_digests={"B": "q"},
        original_requests=POLICY,
    )
    leftovers = [p for p in os.listdir(tmp_path) if p.endswith(".tmp")]
    assert leftovers == []


def test_a_write_failure_is_fatal_in_policy_mode(tmp_path, hadq_map, monkeypatch):
    """Section 3: writes are required. A run with no provenance is not
    interpretable, so this must not degrade to a warning."""
    policy = resolve_policy(POLICY, hadq_map, numbering="auth")

    def boom(*_a, **_k):
        raise OSError("disk full")

    monkeypatch.setattr("pxdbench.targets.record._atomic_write", boom)
    with pytest.raises(RecordWriteError, match="disk full|provenance"):
        write_resolution_record(
            out_dir=str(tmp_path), task_name="t", policy=policy,
            residue_map=hadq_map, selection_mode=SelectionMode.POLICY_V1,
            source_digest="s", shard_digest="h", sequence_digests={"B": "q"},
            original_requests=POLICY,
        )


def test_out_dir_none_validates_without_publishing(hadq_map):
    """Section 3: out_dir=None prepares without writing records."""
    policy = resolve_policy(POLICY, hadq_map, numbering="auth")
    assert write_resolution_record(
        out_dir=None, task_name="t", policy=policy, residue_map=hadq_map,
        selection_mode=SelectionMode.POLICY_V1, source_digest="s",
        shard_digest="h", sequence_digests={"B": "q"},
        original_requests=POLICY,
    ) is None


def test_two_tasks_do_not_overwrite_each_others_record(tmp_path, hadq_map):
    """Section 3: each task gets a unique task directory."""
    policy = resolve_policy(POLICY, hadq_map, numbering="auth")
    paths = [
        write_resolution_record(
            out_dir=str(tmp_path), task_name=name, policy=policy,
            residue_map=hadq_map, selection_mode=SelectionMode.POLICY_V1,
            source_digest="s", shard_digest="h", sequence_digests={"B": "q"},
            original_requests=POLICY, per_task_subdir=True,
        )
        for name in ("task_a", "task_b")
    ]
    assert paths[0] != paths[1]
    assert all(os.path.exists(p) for p in paths)


# --- the existing readers must keep working ---------------------------------

def test_the_record_satisfies_every_key_the_existing_readers_require(written):
    """Pinned from the scripts, not assumed.

    scripts/pose_targeting_report.py:89 reads ["hotspot_resolved"].
    scripts/hotspot_e2e_check.py:62-63 reads ["hotspot_resolved"] AND
    ["numbering"] - dropping the latter was a real break, found by checking
    rather than by trusting the claim.
    """
    _path, record = written
    for key in ("hotspot_resolved", "numbering"):
        assert key in record, f"{key} is required by an existing reader"


def test_the_existing_e2e_reader_can_parse_the_record(written):
    """Drive the real reader's own expression against a schema-2 record."""
    path, _record = written
    payload = json.loads(open(path).read())
    resolved = payload["hotspot_resolved"]          # hotspot_e2e_check.py:62
    label = payload["numbering"]                     # hotspot_e2e_check.py:63
    assert isinstance(resolved, dict) and isinstance(label, str)
    missing = [
        f"{c}{r}" for c, residues in resolved.items() for r in residues
    ]
    assert missing == ["B1", "B3"]
