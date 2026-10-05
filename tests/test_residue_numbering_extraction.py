"""The numbering helpers moved, and nothing else changed.

Targeting-contracts section 2 requires the map and role contracts to be
testable without initialising a model, and requires the existing helpers to be
EXTRACTED rather than reimplemented: "Do not duplicate their algorithms to
avoid an import."

Measured before the move: importing `pxdesign.utils.infer` took 1.3 s and
pulled in torch, protenix, biotite and ml_collections.

These tests pin both halves of that contract - the new module is light, and it
is the same single implementation the old import path still serves.
"""
import importlib
import subprocess
import sys

import pytest

PURE_NAMES = (
    "ResidueMaps",
    "build_chain_mapping",
    "build_residue_maps",
    "convert_crop_auth_to_new",
    "convert_hotspot_auth_to_new",
    "format_ranges",
    "parse_ranges",
    "record_resolved_hotspots",
    "rewrite_input_dict_inplace",
    "validate_hotspots",
)


def test_the_new_module_imports_no_predictor_dependencies():
    """A subprocess, because this session has already imported torch."""
    code = (
        "import sys;"
        "import pxdesign.utils.residue_numbering as m;"
        "heavy=sorted({k.split('.')[0] for k in sys.modules} & "
        "{'torch','protenix','biotite','ml_collections','deepspeed','numpy'});"
        "print(','.join(heavy))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "", f"pulled in {out.stdout.strip()}"


def test_every_public_helper_is_importable_from_the_light_module():
    module = importlib.import_module("pxdesign.utils.residue_numbering")
    for name in PURE_NAMES:
        assert hasattr(module, name), name


def test_the_old_import_path_serves_the_same_objects():
    """One implementation, not two. Identity, not equality."""
    light = importlib.import_module("pxdesign.utils.residue_numbering")
    infer = importlib.import_module("pxdesign.utils.infer")
    for name in PURE_NAMES:
        assert getattr(infer, name) is getattr(light, name), name


def test_the_private_structure_index_is_also_re_exported():
    """`_structure_residue_index` is used by validate_hotspots' callers."""
    light = importlib.import_module("pxdesign.utils.residue_numbering")
    infer = importlib.import_module("pxdesign.utils.infer")
    assert infer._structure_residue_index is light._structure_residue_index


# --- behaviour is unchanged ------------------------------------------------
# Spot checks on the moved algorithms, so the move cannot have altered them.

def test_build_chain_mapping_still_detects_an_inconsistent_mapping():
    from pxdesign.utils.residue_numbering import build_chain_mapping

    assert build_chain_mapping(["B", "D"], ["A", "B"]) == {"B": "A", "D": "B"}
    with pytest.raises(ValueError, match="Inconsistent mapping"):
        build_chain_mapping(["B", "B"], ["A", "C"])


def test_build_chain_mapping_respects_keep_chains():
    from pxdesign.utils.residue_numbering import build_chain_mapping

    assert build_chain_mapping(
        ["B", "D"], ["A", "B"], keep_chains=["D"]
    ) == {"D": "B"}


def test_parse_and_format_ranges_round_trip():
    from pxdesign.utils.residue_numbering import format_ranges, parse_ranges

    assert parse_ranges("1-3,7") == [(1, 3), (7, 7)]
    assert format_ranges([1, 2, 3, 7]) == "1-3,7"


class _FakeAtomArray:
    """Duck-typed stand-in: build_residue_maps only reads four attributes."""

    def __init__(self, rows):
        self.chain_id = [r[0] for r in rows]
        self.res_id = [r[1] for r in rows]
        self.auth_asym_id = [r[2] for r in rows]
        self.auth_res_id = [r[3] for r in rows]


def test_build_residue_maps_is_bijective_and_still_checks_it():
    from pxdesign.utils.residue_numbering import build_residue_maps

    maps = build_residue_maps(
        _FakeAtomArray([("A", 1, "B", 227), ("A", 2, "B", 228)])
    )
    assert maps.resid2auth[("A", 1)] == ("B", 227)
    assert maps.auth2resid[("B", 228)] == ("A", 2)

    with pytest.raises(ValueError, match="Non-unique mapping"):
        build_residue_maps(
            _FakeAtomArray([("A", 1, "B", 227), ("A", 1, "B", 228)])
        )


def test_convert_hotspot_auth_to_new_uses_the_residue_map():
    from pxdesign.utils.residue_numbering import (
        build_residue_maps,
        convert_hotspot_auth_to_new,
    )

    maps = build_residue_maps(
        _FakeAtomArray([("A", 24, "D", 535), ("A", 42, "D", 553)])
    )
    assert convert_hotspot_auth_to_new({"D": [535]}, maps) == {"A": [24]}
