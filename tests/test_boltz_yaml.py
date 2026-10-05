import json
import os

import pytest

from pxdbench.tools.boltz.yaml_writer import write_design_yaml, write_manifest

TARGETS = [
    {"id": "B", "seq": "MKTAYIAK", "msa": "/msa/chainB.a3m"},
    {"id": "D", "seq": "QRSTVWYC", "msa": "/msa/chainD.a3m"},
]


def test_yaml_per_chain_msa(tmp_path):
    """Each target chain carries its OWN a3m; the binder is single-sequence."""
    path = write_design_yaml(str(tmp_path / "c.yaml"), TARGETS, "C", "GGSGGS")
    text = open(path).read()
    assert "msa: /msa/chainB.a3m" in text
    assert "msa: /msa/chainD.a3m" in text
    assert "msa: empty" in text
    assert text.count("msa:") == 3


def test_binder_is_last_chain(tmp_path):
    """Chain order is load-bearing: binder must be chain A or the LAST chain."""
    path = write_design_yaml(str(tmp_path / "c.yaml"), TARGETS, "C", "GGSGGS")
    lines = [ln for ln in open(path).read().splitlines() if "protein:" in ln]
    assert len(lines) == 3
    assert "id: C" in lines[-1]
    assert "GGSGGS" in lines[-1]


def test_version_header(tmp_path):
    path = write_design_yaml(str(tmp_path / "c.yaml"), TARGETS, "C", "GG")
    assert open(path).read().startswith("version: 1\nsequences:\n")


def test_rejects_binder_id_colliding_with_target(tmp_path):
    with pytest.raises(ValueError, match="collides"):
        write_design_yaml(str(tmp_path / "c.yaml"), TARGETS, "B", "GG")


def test_rejects_empty_binder_sequence(tmp_path):
    with pytest.raises(ValueError, match="empty"):
        write_design_yaml(str(tmp_path / "c.yaml"), TARGETS, "C", "")


def test_manifest_round_trips(tmp_path):
    entries = [{"sample_name": "bb_seq0", "yaml": "/w/a.yaml", "binder_id": "C"}]
    path = write_manifest(str(tmp_path / "m.json"), entries)
    assert json.load(open(path))["designs"] == entries


def test_yaml_parses_as_boltz_expects(tmp_path):
    """The emitted text must actually be valid YAML with the right shape.

    Asserting on substrings alone would pass for text Boltz cannot parse.
    """
    yaml = pytest.importorskip("yaml")
    path = write_design_yaml(str(tmp_path / "c.yaml"), TARGETS, "C", "GGSGGS")
    doc = yaml.safe_load(open(path))
    assert doc["version"] == 1
    chains = [entry["protein"] for entry in doc["sequences"]]
    assert [c["id"] for c in chains] == ["B", "D", "C"]
    assert chains[0]["msa"] == "/msa/chainB.a3m"
    assert chains[-1]["msa"] == "empty"
    assert chains[-1]["sequence"] == "GGSGGS"


# --- pocket constraints -------------------------------------------------
# The scored pose is Boltz's free re-prediction: measured on 8 designs, the
# binder lands 7-42 A from where the diffusion put it, so the hotspot-
# conditioned interface is not the one that gets scored. A pocket constraint
# ties the scored pose to the requested epitope. It is OPT-IN because it
# changes the Boltz input, which invalidates the gate thresholds.

# in range: the TARGETS chains are 8 residues each
HOTSPOTS = [("B", 2), ("B", 5), ("D", 7)]


def test_no_constraints_block_by_default(tmp_path):
    """Default output must be unchanged: the gate thresholds depend on it."""
    path = write_design_yaml(str(tmp_path / "c.yaml"), TARGETS, "C", "GGSGGS")
    assert "constraints" not in open(path).read()


def test_pocket_constraint_has_the_shape_boltz_parses(tmp_path):
    yaml = pytest.importorskip("yaml")
    path = write_design_yaml(
        str(tmp_path / "c.yaml"), TARGETS, "C", "GGSGGS",
        pocket_contacts=HOTSPOTS,
    )
    doc = yaml.safe_load(open(path))
    pocket = doc["constraints"][0]["pocket"]
    assert pocket["binder"] == "C"
    assert pocket["contacts"] == [["B", 2], ["B", 5], ["D", 7]]
    assert pocket["max_distance"] == 6.0


def test_pocket_max_distance_is_overridable(tmp_path):
    yaml = pytest.importorskip("yaml")
    path = write_design_yaml(
        str(tmp_path / "c.yaml"), TARGETS, "C", "GG",
        pocket_contacts=[("B", 2)], pocket_max_distance=8.5,
    )
    assert yaml.safe_load(open(path))["constraints"][0]["pocket"][
        "max_distance"] == 8.5


def test_pocket_contact_on_unknown_chain_is_rejected(tmp_path):
    """A contact on a chain not in the complex silently constrains nothing."""
    with pytest.raises(ValueError, match="not a target chain"):
        write_design_yaml(
            str(tmp_path / "c.yaml"), TARGETS, "C", "GG",
            pocket_contacts=[("Z", 2)],
        )


def test_pocket_contact_on_the_binder_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="not a target chain"):
        write_design_yaml(
            str(tmp_path / "c.yaml"), TARGETS, "C", "GG",
            pocket_contacts=[("C", 3)],
        )


def test_empty_pocket_contacts_is_rejected(tmp_path):
    """An empty list means "constrain nothing" - almost certainly a bug."""
    with pytest.raises(ValueError, match="empty"):
        write_design_yaml(
            str(tmp_path / "c.yaml"), TARGETS, "C", "GG", pocket_contacts=[],
        )


def test_pocket_contact_beyond_the_target_sequence_is_rejected(tmp_path):
    """Boltz reads the contact number as a 1-based index into the YAML sequence.

    Verified against boltz 2.2.1's own parser: contacts [[B,3],[B,7],[D,5]] on
    chains B,D,C come back as (binder=2, [(0,2),(0,6),(1,4)]) - chain index
    plus ZERO-based residue index. So a number past the end of the sequence
    does not error in Boltz, it just resolves somewhere unintended.
    """
    with pytest.raises(ValueError, match="outside chain"):
        write_design_yaml(
            str(tmp_path / "c.yaml"), TARGETS, "C", "GG",
            pocket_contacts=[("B", 99)],   # chain B is 8 residues long
        )


def test_pocket_contact_below_one_is_rejected(tmp_path):
    """The numbering is 1-based; 0 would silently shift every contact."""
    with pytest.raises(ValueError, match="outside chain"):
        write_design_yaml(
            str(tmp_path / "c.yaml"), TARGETS, "C", "GG",
            pocket_contacts=[("B", 0)],
        )


def test_pocket_contact_at_the_sequence_boundaries_is_accepted(tmp_path):
    yaml = pytest.importorskip("yaml")
    path = write_design_yaml(
        str(tmp_path / "c.yaml"), TARGETS, "C", "GG",
        pocket_contacts=[("B", 1), ("B", 8)],   # first and last of 8
    )
    assert yaml.safe_load(open(path))["constraints"][0]["pocket"][
        "contacts"] == [["B", 1], ["B", 8]]
