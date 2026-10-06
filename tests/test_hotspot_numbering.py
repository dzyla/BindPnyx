"""Hotspot numbering: the contract used to depend on the file extension.

convert_to_bioassembly_dict has three branches and they behaved differently:

  .pkl.gz  returned immediately - no remap, no validation at all
  .cif     built no chain mapping and never called rewrite_input_dict_inplace,
           so auth numbering was passed through unconverted
  .pdb     built residue maps and remapped strictly

So the same target in two formats meant two different things, and in two of the
three cases a mistyped or mismapped residue was silently ignored - json_parser
turns hotspots into np.isin(res_id, hotspot), which simply matches nothing.
"""
import json

import numpy as np
import pytest
pytest.importorskip("torch", reason="needs the full pxd environment (torch)")

from pxdesign.utils.infer import (
    record_resolved_hotspots,
    validate_hotspots,
)


class FakeAtomArray:
    """Just the two annotations the validator reads."""

    def __init__(self, pairs):
        self.chain_id = np.array([c for c, _ in pairs])
        self.res_id = np.array([r for _, r in pairs])


@pytest.fixture
def structure():
    # chain A residues 1..30, chain B residues 1..40
    pairs = [("A", i) for i in range(1, 31)] + [("B", i) for i in range(1, 41)]
    return FakeAtomArray(pairs)


def test_valid_hotspots_pass_through_unchanged(structure):
    req = {"A": [25, 27], "B": [20, 35]}
    assert validate_hotspots(req, structure) == req


def test_a_missing_residue_is_an_error_not_a_silent_miss(structure):
    """np.isin would match nothing and the design would run unconditioned."""
    with pytest.raises(KeyError, match="hotspot residues not found"):
        validate_hotspots({"A": [25, 999]}, structure)


def test_the_error_names_the_offending_residues_and_the_chains(structure):
    with pytest.raises(KeyError) as excinfo:
        validate_hotspots({"A": [999], "C": [1]}, structure)
    msg = str(excinfo.value)
    assert "A999" in msg and "C1" in msg
    assert "'A', 'B'" in msg or "A" in msg and "B" in msg


def test_a_wrong_chain_is_caught(structure):
    """A residue number that exists on another chain is still wrong."""
    with pytest.raises(KeyError, match="C"):
        validate_hotspots({"C": [10]}, structure)


def test_off_by_one_numbering_is_caught(structure):
    """The failure mode this exists for: numbering shifted by a constant."""
    with pytest.raises(KeyError):
        validate_hotspots({"B": [41]}, structure)   # 1..40 only


def test_non_strict_warns_instead_of_raising(structure, caplog):
    import logging

    with caplog.at_level(logging.WARNING):
        out = validate_hotspots({"A": [999]}, structure, strict=False)
    assert out == {"A": [999]}
    assert "not found" in caplog.text


def test_empty_hotspots_are_a_no_op(structure):
    assert validate_hotspots({}, structure) == {}
    assert validate_hotspots(None, structure) is None


def test_resolved_hotspots_are_recorded_for_downstream(tmp_path):
    """The epitope check must not have to guess which numbering a residue is in."""
    d = {"name": "task1", "hotspot": {"A": [25, 27], "B": [50]}}
    record_resolved_hotspots(d, str(tmp_path))
    payload = json.load(open(tmp_path / "resolved_hotspots.json"))
    assert payload["hotspot_resolved"] == {"A": [25, 27], "B": [50]}
    assert payload["name"] == "task1"
    assert "design" in payload["numbering"]


def test_nothing_recorded_without_hotspots(tmp_path):
    record_resolved_hotspots({"name": "t"}, str(tmp_path))
    assert not (tmp_path / "resolved_hotspots.json").exists()


def test_recording_tolerates_a_missing_out_dir():
    record_resolved_hotspots({"name": "t", "hotspot": {"A": [1]}}, None)


def test_pkl_gz_branch_now_validates(tmp_path, monkeypatch):
    """The branch our campaigns actually use. It used to return immediately."""
    import pxdesign.utils.infer as infer

    pairs = [("A", i) for i in range(1, 31)]
    monkeypatch.setattr(
        infer, "load_gzip_pickle", lambda path: {"atom_array": FakeAtomArray(pairs)}
    )
    good = {
        "name": "t",
        "hotspot": {"A": [5, 10]},
        "condition": {"structure_file": str(tmp_path / "x.pkl.gz")},
    }
    assert infer.convert_to_bioassembly_dict(good, str(tmp_path)).endswith(".pkl.gz")
    assert (tmp_path / "resolved_hotspots.json").exists()

    bad = {
        "name": "t",
        "hotspot": {"A": [5, 999]},
        "condition": {"structure_file": str(tmp_path / "x.pkl.gz")},
    }
    with pytest.raises(KeyError, match="hotspot residues not found"):
        infer.convert_to_bioassembly_dict(bad, str(tmp_path))
