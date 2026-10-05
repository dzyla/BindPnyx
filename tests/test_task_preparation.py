"""One preparation boundary, so a campaign input reaches the resolver.

Everything below this was built and tested and nothing fed it: a campaign still
ran in legacy mode because no entry point resolved an `epitope` block. This is
that wiring.

Targeting-contracts section 3:

- one idempotent boundary, `convert_and_resolve(task, out_dir)`, extending the
  existing conversion rather than competing with it;
- it returns a JSON-serialisable task - no dataclass or atom array crosses the
  configuration or subprocess boundary;
- prepare the structure FIRST with the user's residue requests held separately,
  then resolve those original requests exactly once against the resulting map.
  Injecting resolved work hotspots into the PDB conversion path - which remaps
  author hotspots - would convert them twice;
- mode is resolved from configuration before scoring and written to the
  manifest. Never inferred from `ep_satisfied`, a column, or a missing file.

The structure conversion is injected, so these run without a model.
"""
import json

import pytest

from pxdbench.targets.preparation import (
    PreparationError,
    convert_and_resolve,
    resolve_selection_mode,
)
from pxdbench.targets.record import SelectionMode

SHARD = "/fake/shard.pkl.gz"
SOURCE = "/fake/source.pdb"

EPITOPE = {"required": {"D": ["H512"]}, "forbidden": {"D": ["A513"]}}


def _task(**over):
    task = {
        "name": "t1",
        "condition": {
            "structure_file": SHARD,
            "filter": {"chain_id": ["B"], "crop": {}},
        },
    }
    task.update(over)
    return task


@pytest.fixture
def convert(residues):
    """Stands in for convert_to_bioassembly_dict + map construction."""
    def _convert(task, out_dir):
        from pxdbench.targets.residue_map import TargetResidueMap

        return TargetResidueMap.from_residue_lists(
            source={"D": residues("D", 512, "HADQ")},
            shard={"B": residues("B", 1, "HADQ")},
            chain_filter=["B"],
            sequences={"B": "HADQ"},
        ), {"source": "src-1", "shard": "shard-1", "sequences": {"B": "seq-1"}}
    return _convert


# --- mode resolution, from configuration -----------------------------------

def test_a_legacy_hotspot_input_resolves_to_legacy():
    assert resolve_selection_mode(
        _task(hotspot={"B": [1, 3]})
    ) is SelectionMode.LEGACY


def test_an_input_with_no_epitope_information_resolves_to_legacy():
    assert resolve_selection_mode(_task()) is SelectionMode.LEGACY


def test_an_epitope_block_selects_policy_v1():
    assert resolve_selection_mode(
        _task(epitope=EPITOPE, source_structure=SOURCE, numbering="auth")
    ) is SelectionMode.POLICY_V1


def test_an_EMPTY_epitope_block_still_selects_policy_v1():
    """Section 1: including an explicitly empty block."""
    assert resolve_selection_mode(
        _task(epitope={}, source_structure=SOURCE, numbering="auth")
    ) is SelectionMode.POLICY_V1


def test_requesting_legacy_with_an_epitope_block_is_rejected():
    with pytest.raises(PreparationError, match="legacy"):
        resolve_selection_mode(
            _task(epitope=EPITOPE, selection_mode="legacy",
                  source_structure=SOURCE, numbering="auth")
        )


def test_legacy_hotspot_and_an_epitope_block_together_are_rejected():
    """Section 1: rejected rather than merged."""
    with pytest.raises(PreparationError, match="both"):
        resolve_selection_mode(
            _task(hotspot={"B": [1]}, epitope=EPITOPE,
                  source_structure=SOURCE, numbering="auth")
        )


def test_mode_is_never_inferred_from_a_result_value():
    """A stray ep_satisfied must not promote a legacy task."""
    assert resolve_selection_mode(
        _task(hotspot={"B": [1]}, ep_satisfied=True)
    ) is SelectionMode.LEGACY


# --- policy_v1 preflight ---------------------------------------------------

def test_policy_v1_requires_an_explicit_source_structure(convert):
    """Author numbering is meaningless without the structure it refers to."""
    with pytest.raises(PreparationError, match="source_structure"):
        convert_and_resolve(
            _task(epitope=EPITOPE, numbering="auth"), None, convert=convert
        )


def test_policy_v1_requires_an_explicit_numbering(convert):
    """Section 2: both are explicit. The same number means different residues
    depending on the input file's extension, so it is declared."""
    with pytest.raises(PreparationError, match="numbering"):
        convert_and_resolve(
            _task(epitope=EPITOPE, source_structure=SOURCE), None,
            convert=convert,
        )


def test_an_unknown_numbering_is_rejected(convert):
    with pytest.raises(PreparationError, match="numbering"):
        convert_and_resolve(
            _task(epitope=EPITOPE, source_structure=SOURCE,
                  numbering="biological"),
            None, convert=convert,
        )


# --- resolution ------------------------------------------------------------

def test_the_resolved_task_is_json_serialisable(convert, tmp_path):
    """No dataclass or atom array crosses the configuration boundary."""
    out = convert_and_resolve(
        _task(epitope=EPITOPE, source_structure=SOURCE, numbering="auth"),
        str(tmp_path), convert=convert,
    )
    json.dumps(out)      # must not raise


def test_the_derived_hotspot_field_is_required_plus_advisory(convert, tmp_path):
    """Section 1: that field must EQUAL required + advisory and is verified,
    never treated as a second user request."""
    out = convert_and_resolve(
        _task(epitope=EPITOPE, source_structure=SOURCE, numbering="auth"),
        str(tmp_path), convert=convert,
    )
    assert out["hotspot"] == {"B": [1]}, "forbidden B2 must not appear"


def test_the_record_and_map_are_written(convert, tmp_path):
    convert_and_resolve(
        _task(epitope=EPITOPE, source_structure=SOURCE, numbering="auth"),
        str(tmp_path), convert=convert,
    )
    assert (tmp_path / "resolved_hotspots.json").exists()
    assert (tmp_path / "target_residue_map.json").exists()


def test_the_task_carries_its_mode_and_digests(convert, tmp_path):
    out = convert_and_resolve(
        _task(epitope=EPITOPE, source_structure=SOURCE, numbering="auth"),
        str(tmp_path), convert=convert,
    )
    assert out["selection_mode"] == "policy_v1"
    assert out["policy_digest"] and out["map_digest"]


def test_an_identity_assertion_failure_stops_preparation(convert, tmp_path):
    """Before any GPU work: H512 is a histidine, so asserting R512 fails."""
    from pxdbench.targets.residue_spec import IdentityAssertionError

    with pytest.raises(IdentityAssertionError):
        convert_and_resolve(
            _task(epitope={"required": {"D": ["R512"]}},
                  source_structure=SOURCE, numbering="auth"),
            str(tmp_path), convert=convert,
        )


# --- idempotence -----------------------------------------------------------

def test_preparing_twice_does_not_resolve_twice(convert, tmp_path):
    """Section 3: a digest check on re-entry prevents double conversion."""
    first = convert_and_resolve(
        _task(epitope=EPITOPE, source_structure=SOURCE, numbering="auth"),
        str(tmp_path), convert=convert,
    )
    second = convert_and_resolve(first, str(tmp_path), convert=convert)
    assert second["hotspot"] == first["hotspot"]
    assert second["policy_digest"] == first["policy_digest"]


def test_a_prepared_task_is_not_reinterpreted_as_a_new_request(convert, tmp_path):
    """The prepared task carries `hotspot` AND `epitope`; that combination is
    rejected from a RAW user task but must not be rejected on re-entry."""
    first = convert_and_resolve(
        _task(epitope=EPITOPE, source_structure=SOURCE, numbering="auth"),
        str(tmp_path), convert=convert,
    )
    assert "hotspot" in first and "epitope" in first
    convert_and_resolve(first, str(tmp_path), convert=convert)  # must not raise


# --- legacy is untouched ---------------------------------------------------

def test_a_legacy_task_passes_through_without_a_map(tmp_path):
    """No source structure, no map, no record - exactly as before."""
    def _refuse(_task, _out):
        raise AssertionError("a legacy task must not build a residue map")

    out = convert_and_resolve(
        _task(hotspot={"B": [1, 3]}), str(tmp_path), convert=_refuse
    )
    assert out["selection_mode"] == "legacy"
    assert out["hotspot"] == {"B": [1, 3]}
    assert not (tmp_path / "resolved_hotspots.json").exists()


def test_out_dir_none_prepares_without_publishing(convert):
    out = convert_and_resolve(
        _task(epitope=EPITOPE, source_structure=SOURCE, numbering="auth"),
        None, convert=convert,
    )
    assert out["selection_mode"] == "policy_v1"
    assert out["hotspot"] == {"B": [1]}
