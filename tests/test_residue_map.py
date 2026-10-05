"""Source-to-working residue provenance, with a bounded v1 contract.

`ResidueMaps` (pxdesign/utils/residue_numbering.py) already gives a bijective
`res_id <-> auth_res_id` map with uniqueness checks in both directions. It
cannot help a pre-built shard for one reason: the shard's `auth_res_id` EQUALS
its `res_id`, so the map it builds is the identity and carries no provenance.
Measured on obj2: `auth_seq_id` is the string form of `res_id` for all 183
residues.

This module adds the two things it lacks - real author numbering taken from the
supplied source structure, and residue identity, without which no identity
assertion is possible. It reuses those helpers rather than reimplementing them.

V1 is deliberately bounded (targeting-contracts section 2). Everything in the
rejection table fails BEFORE model initialisation, naming what it found. No
offset guessing, no first-match chain selection, no sequence shortening.

These tests initialise no model and need no private data, except the two marked
obj2 pairing cases.
"""
import os

import pytest

from pxdbench.targets.residue_map import (
    AmbiguousChainError,
    ResidueRow,
    TargetResidueMap,
    UnsupportedMapError,
)

SOURCE = "/data/private/private_target/obj2_target.pdb"
SHARD = "/data/private/private_target/shard_L55_c01/obj2_target.pkl.gz"

needs_obj2 = pytest.mark.skipif(
    not (os.path.exists(SOURCE) and os.path.exists(SHARD)),
    reason="obj2 source and shard not both on this machine",
)


# --- portable fixtures ------------------------------------------------------
# A "source" and "shard" are duck-typed: a residue list is all the pairing
# needs, so the contract is testable without biotite or a real structure.

from conftest import make_residues as residues  # noqa: E402


def test_a_clean_pairing_produces_rows_with_both_numberings():
    """The supplied source may be a FRAGMENT in biological numbering: obj2's
    begins at author residue 512. Pairing compares the whole selected chain in
    the supplied source, never a guessed full-length protein."""
    mapping = TargetResidueMap.from_residue_lists(
        source={"D": residues("D", 512, "HADQ")},
        shard={"B": residues("B", 1, "HADQ")},
        chain_filter=["B"],
        sequences={"B": "HADQ"},
    )
    assert mapping.chain_table == {"B": "D"}
    assert mapping.to_work("D", 512) == ResidueRow("D", 512, "B", 1, 1, "HIS")
    assert mapping.to_auth("B", 4).auth_res_id == 515


def test_seq_index_is_stored_explicitly_even_when_it_equals_work_res_id():
    """Section 2 requires it stored, because Boltz pocket contacts are 1-based
    indices into the emitted sequence and the equality is a v1 property, not a
    law."""
    mapping = TargetResidueMap.from_residue_lists(
        source={"D": residues("D", 512, "HADQ")},
        shard={"B": residues("B", 1, "HADQ")},
        chain_filter=["B"],
        sequences={"B": "HADQ"},
    )
    assert [r.seq_index for r in mapping.rows] == [1, 2, 3, 4]
    assert all(r.seq_index == r.work_res_id for r in mapping.rows)


# --- the rejection table ----------------------------------------------------

def test_a_nonblank_insertion_code_is_rejected_naming_the_residue():
    with pytest.raises(UnsupportedMapError, match="insertion code"):
        TargetResidueMap.from_residue_lists(
            source={"D": residues("D", 512, "HAD", ins_code="A")},
            shard={"B": residues("B", 1, "HAD")},
            chain_filter=["B"],
            sequences={"B": "HAD"},
        )


def test_two_matching_source_chains_are_ambiguous_not_first_match():
    """An indistinguishable homomer copy: v1 does not choose the first."""
    with pytest.raises(AmbiguousChainError, match="D.*E|E.*D"):
        TargetResidueMap.from_residue_lists(
            source={
                "D": residues("D", 512, "HADQ"),
                "E": residues("E", 900, "HADQ"),
            },
            shard={"B": residues("B", 1, "HADQ")},
            chain_filter=["B"],
            sequences={"B": "HADQ"},
        )


def test_an_explicit_chain_table_resolves_legitimate_ambiguity():
    """Section 2: an explicit mapping may support additional cases."""
    mapping = TargetResidueMap.from_residue_lists(
        source={
            "D": residues("D", 512, "HADQ"),
            "E": residues("E", 900, "HADQ"),
        },
        shard={"B": residues("B", 1, "HADQ")},
        chain_filter=["B"],
        sequences={"B": "HADQ"},
        chain_table={"B": "E"},
    )
    assert mapping.to_auth("B", 1).auth_res_id == 900


def test_no_matching_source_chain_is_a_pairing_error():
    with pytest.raises(UnsupportedMapError, match="no source chain"):
        TargetResidueMap.from_residue_lists(
            source={"D": residues("D", 512, "KKKK")},
            shard={"B": residues("B", 1, "HADQ")},
            chain_filter=["B"],
            sequences={"B": "HADQ"},
        )


def test_unequal_lengths_name_both_sides():
    with pytest.raises(UnsupportedMapError, match="4 residues.*3 residues|3 residues.*4 residues"):
        TargetResidueMap.from_residue_lists(
            source={"D": residues("D", 512, "HAD")},
            shard={"B": residues("B", 1, "HADQ")},
            chain_filter=["B"],
            sequences={"B": "HADQ"},
            chain_table={"B": "D"},
        )


def test_an_identity_mismatch_names_both_identities_and_positions():
    with pytest.raises(UnsupportedMapError, match="HIS.*LYS|LYS.*HIS"):
        TargetResidueMap.from_residue_lists(
            source={"D": residues("D", 512, "KADQ")},
            shard={"B": residues("B", 1, "HADQ")},
            chain_filter=["B"],
            sequences={"B": "HADQ"},
            chain_table={"B": "D"},
        )


@pytest.mark.parametrize("bad_ids", [[2, 3, 4, 5], [1, 2, 4, 5]])
def test_work_ids_must_be_contiguous_from_one(bad_ids):
    """Shifted or gapped work IDs: no offset guessing."""
    shard = [("B", i, n, "") for i, (_c, _r, n, _i) in
             zip(bad_ids, residues("B", 1, "HADQ"))]
    with pytest.raises(UnsupportedMapError, match="contiguous"):
        TargetResidueMap.from_residue_lists(
            source={"D": residues("D", 512, "HADQ")},
            shard={"B": shard},
            chain_filter=["B"],
            sequences={"B": "HADQ"},
        )


def test_a_duplicate_residue_key_is_rejected():
    shard = residues("B", 1, "HAD") + [("B", 3, "GLN", "")]
    with pytest.raises(UnsupportedMapError, match="duplicate"):
        TargetResidueMap.from_residue_lists(
            source={"D": residues("D", 512, "HADQ")},
            shard={"B": shard},
            chain_filter=["B"],
            sequences={"B": "HADQ"},
        )


def test_a_nonstandard_residue_is_not_silently_converted():
    shard = residues("B", 1, "HAD") + [("B", 4, "MSE", "")]
    with pytest.raises(UnsupportedMapError, match="MSE"):
        TargetResidueMap.from_residue_lists(
            source={"D": residues("D", 512, "HAD") + [("D", 515, "MSE", "")]},
            shard={"B": shard},
            chain_filter=["B"],
            sequences={"B": "HADM"},
        )


def test_the_map_must_agree_with_the_sequence_emitted_to_the_predictor():
    """Section 2: exact agreement, because the pocket contact is an index into
    that sequence."""
    with pytest.raises(UnsupportedMapError, match="sequence"):
        TargetResidueMap.from_residue_lists(
            source={"D": residues("D", 512, "HADQ")},
            shard={"B": residues("B", 1, "HADQ")},
            chain_filter=["B"],
            sequences={"B": "HADK"},        # last position disagrees
        )


def test_a_shorter_emitted_sequence_is_not_tolerated():
    with pytest.raises(UnsupportedMapError, match="sequence"):
        TargetResidueMap.from_residue_lists(
            source={"D": residues("D", 512, "HADQ")},
            shard={"B": residues("B", 1, "HADQ")},
            chain_filter=["B"],
            sequences={"B": "HAD"},
        )


def test_duplicate_output_chain_ids_are_rejected():
    with pytest.raises(UnsupportedMapError, match="duplicate|chain table"):
        TargetResidueMap.from_residue_lists(
            source={"D": residues("D", 512, "HADQ"),
                    "E": residues("E", 900, "KKKK")},
            shard={"B": residues("B", 1, "HADQ"),
                   "C": residues("C", 1, "KKKK")},
            chain_filter=["B", "C"],
            sequences={"B": "HADQ", "C": "KKKK"},
            chain_table={"B": "D", "C": "D"},   # both claim source D
        )


# --- lookups ---------------------------------------------------------------

def test_an_absent_requested_residue_raises_rather_than_returning_none():
    mapping = TargetResidueMap.from_residue_lists(
        source={"D": residues("D", 512, "HADQ")},
        shard={"B": residues("B", 1, "HADQ")},
        chain_filter=["B"],
        sequences={"B": "HADQ"},
    )
    with pytest.raises(KeyError, match="not found"):
        mapping.to_work("D", 9999)
    with pytest.raises(KeyError, match="not found"):
        mapping.to_auth("B", 9999)


def test_json_round_trip_preserves_rows_and_chain_table(tmp_path):
    mapping = TargetResidueMap.from_residue_lists(
        source={"D": residues("D", 512, "HADQ")},
        shard={"B": residues("B", 1, "HADQ")},
        chain_filter=["B"],
        sequences={"B": "HADQ"},
    )
    path = mapping.to_json(str(tmp_path / "map.json"))
    back = TargetResidueMap.from_json(path)
    assert back.rows == mapping.rows
    assert back.chain_table == mapping.chain_table
    assert back.map_digest == mapping.map_digest


def test_the_map_digest_changes_with_the_pairing():
    a = TargetResidueMap.from_residue_lists(
        source={"D": residues("D", 512, "HADQ")},
        shard={"B": residues("B", 1, "HADQ")},
        chain_filter=["B"], sequences={"B": "HADQ"})
    b = TargetResidueMap.from_residue_lists(
        source={"D": residues("D", 600, "HADQ")},
        shard={"B": residues("B", 1, "HADQ")},
        chain_filter=["B"], sequences={"B": "HADQ"})
    assert a.map_digest != b.map_digest


# --- the real obj2 pairing, separately marked ------------------------------

@needs_obj2
def test_obj2_pairing_recovers_the_measured_provenance():
    """Measured 2026-10-01: source B 227-306 -> work A 1-80, source D 512-614
    -> work B 1-103, zero identity mismatches."""
    mapping = TargetResidueMap.from_structures(SOURCE, SHARD, ["A", "B"])
    assert mapping.chain_table == {"A": "B", "B": "D"}
    assert mapping.to_work("D", 535).work_res_id == 24
    assert mapping.to_work("B", 251).work_res_id == 25
    assert mapping.to_auth("B", 24).auth_res_id == 535
    assert len(mapping.rows) == 183


@needs_obj2
def test_his535_is_a_histidine():
    """The claim the whole campaign rested on, checkable at last."""
    mapping = TargetResidueMap.from_structures(SOURCE, SHARD, ["A", "B"])
    assert mapping.to_work("D", 535).resname == "HIS"
