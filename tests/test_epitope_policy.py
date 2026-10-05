"""Hotspots are a SOFT design condition and nothing verified them.

A hotspot biases generation through a per-token model feature, but no stage
checked that the predicted complex actually contacts it, so a design could score
well at a different site and still be selected. Coldspots had no support at all.
"""
import numpy as np
import pytest

from pxdbench.metrics.epitope import (
    DEFAULT_CONTACT_CUTOFF,
    EPITOPE_KEYS,
    contact_residues,
    evaluate_epitope_policy,
)


def _pdb(path, residues):
    """residues: [(chain, resseq, resname, (x,y,z)), ...] - one CA each."""
    lines, serial = [], 1
    for chain, resseq, resname, (x, y, z) in residues:
        lines.append(
            f"ATOM  {serial:5d}  CA  {resname:>3s} {chain}{resseq:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C"
        )
        serial += 1
    lines.append("END")
    open(path, "w").write("\n".join(lines) + "\n")
    return str(path)


@pytest.fixture
def complex_pdb(tmp_path):
    """Binder C sits next to target A10 and B20; A99 and B99 are far away."""
    return _pdb(
        tmp_path / "c.pdb",
        [
            ("A", 10, "GLU", (0.0, 0.0, 0.0)),
            ("A", 99, "LYS", (100.0, 0.0, 0.0)),
            ("B", 20, "ASP", (0.0, 5.0, 0.0)),
            ("B", 99, "ARG", (100.0, 5.0, 0.0)),
            ("C", 1, "LEU", (2.0, 0.0, 0.0)),      # 2 A from A10
            ("C", 2, "VAL", (2.0, 5.0, 0.0)),      # 2 A from B20
        ],
    )


def test_contacts_are_found_on_every_target_chain(complex_pdb):
    got = contact_residues(complex_pdb, "C")
    assert got == {"A": [10], "B": [20]}


def test_distant_residues_are_not_contacts(complex_pdb):
    got = contact_residues(complex_pdb, "C", cutoff=4.0)
    assert 99 not in got["A"] and 99 not in got["B"]


def test_cutoff_is_respected(complex_pdb):
    assert contact_residues(complex_pdb, "C", cutoff=1.0) == {"A": [], "B": []}
    wide = contact_residues(complex_pdb, "C", cutoff=200.0)
    assert wide["A"] == [10, 99]


def test_unknown_binder_chain_raises(complex_pdb):
    with pytest.raises(ValueError, match="binder chain"):
        contact_residues(complex_pdb, "Z")


def test_policy_satisfied_when_required_hit_and_forbidden_clear(complex_pdb):
    got = evaluate_epitope_policy(
        complex_pdb, "C", required={"A": [10], "B": [20]}, forbidden={"A": [99]}
    )
    assert got["ep_satisfied"] is True
    assert got["ep_required_total"] == 2
    assert got["ep_required_contacted"] == 2
    assert got["ep_required_frac"] == 1.0
    assert got["ep_forbidden_contacted"] == 0
    assert got["ep_missed_required"] == ""
    assert got["ep_status"] == "ok"


def test_policy_reports_the_residues_that_were_missed(complex_pdb):
    """The design bound the wrong site: this is the case nothing used to catch."""
    got = evaluate_epitope_policy(complex_pdb, "C", required={"A": [10, 99]})
    assert got["ep_satisfied"] is False
    assert got["ep_required_contacted"] == 1
    assert got["ep_required_frac"] == 0.5
    assert got["ep_missed_required"] == "A99"


def test_policy_reports_forbidden_violations(complex_pdb):
    got = evaluate_epitope_policy(
        complex_pdb, "C", required={"A": [10]}, forbidden={"B": [20]}
    )
    assert got["ep_satisfied"] is False
    assert got["ep_forbidden_contacted"] == 1
    assert got["ep_violated_forbidden"] == "B20"
    # the required residue was still hit; both facts are reported
    assert got["ep_required_contacted"] == 1


def test_forbidden_only_policy_is_supported(complex_pdb):
    got = evaluate_epitope_policy(complex_pdb, "C", forbidden={"A": [99]})
    assert got["ep_satisfied"] is True
    assert got["ep_required_total"] == 0


def test_empty_policy_is_unverified_not_passed(complex_pdb):
    """Reporting a pass for a policy nobody set is the failure this prevents."""
    got = evaluate_epitope_policy(complex_pdb, "C")
    assert got["ep_satisfied"] is None
    assert got["ep_status"] == "no_policy"
    assert set(got) == set(EPITOPE_KEYS)


def test_unevaluable_structure_is_reported_not_passed(tmp_path):
    got = evaluate_epitope_policy(
        str(tmp_path / "missing.pdb"), "C", required={"A": [10]}
    )
    assert got["ep_status"] == "unevaluable"
    assert got["ep_satisfied"] is None


def test_binder_chain_absent_is_unevaluable_not_passed(complex_pdb):
    got = evaluate_epitope_policy(complex_pdb, "Z", required={"A": [10]})
    assert got["ep_status"] == "unevaluable"
    assert got["ep_satisfied"] is None


def test_every_result_has_the_full_key_set(complex_pdb):
    for kwargs in (
        {},
        {"required": {"A": [10]}},
        {"forbidden": {"A": [99]}},
        {"required": {"A": [10]}, "forbidden": {"B": [20]}},
    ):
        got = evaluate_epitope_policy(complex_pdb, "C", **kwargs)
        assert set(got) == set(EPITOPE_KEYS), kwargs


def test_default_cutoff_matches_the_hotspot_helper():
    """Both must mean the same thing by 'contact'."""
    import inspect

    from pxdbench.tools import biopython_utils

    sig = inspect.signature(biopython_utils.hotspot_residues)
    assert sig.parameters["atom_distance_cutoff"].default == DEFAULT_CONTACT_CUTOFF


# --------------------------------------------------------------------------- #
# Patch awareness: "contact all of them" is not always a satisfiable request
# --------------------------------------------------------------------------- #


@pytest.fixture
def two_patch_pdb(tmp_path):
    """Two requested sites 40 A apart, with the binder sitting on one of them.

    Mirrors the measured EGFR case: six hotspots in two groups 28-42 A apart,
    while a 55-mer's contact footprint is ~20-25 A across.
    """
    return _pdb(
        tmp_path / "tp.pdb",
        [
            ("A", 25, "GLU", (0.0, 0.0, 0.0)),      # patch 1
            ("A", 27, "LYS", (6.0, 0.0, 0.0)),      # patch 1, 6 A away
            ("B", 50, "ASP", (40.0, 0.0, 0.0)),     # patch 2, 40 A from patch 1
            ("B", 57, "ARG", (45.0, 0.0, 0.0)),     # patch 2
            ("C", 1, "LEU", (2.0, 2.0, 0.0)),       # binder on patch 1 only
            ("C", 2, "VAL", (5.0, 2.0, 0.0)),
        ],
    )


def test_patches_are_split_by_distance(two_patch_pdb):
    from pxdbench.metrics.epitope import spatial_patches

    patches = spatial_patches(
        two_patch_pdb, {"A": [25, 27], "B": [50, 57]}, separation=15.0
    )
    assert len(patches) == 2, patches
    sizes = sorted(sum(len(v) for v in p.values()) for p in patches)
    assert sizes == [2, 2]


def test_a_single_site_stays_one_patch(two_patch_pdb):
    from pxdbench.metrics.epitope import spatial_patches

    assert len(spatial_patches(two_patch_pdb, {"A": [25, 27]}, separation=15.0)) == 1


def test_best_patch_frac_separates_spec_error_from_design_failure(two_patch_pdb):
    """all-of-6 reads as total failure; per-patch shows one site fully engaged."""
    got = evaluate_epitope_policy(
        two_patch_pdb, "C",
        required={"A": [25, 27], "B": [50, 57]},
        patch_separation=15.0,
    )
    assert got["ep_satisfied"] is False          # cannot span 40 A
    assert got["ep_required_frac"] == 0.5        # 2 of 4
    assert got["ep_best_patch_frac"] == 1.0      # but patch 1 is FULLY engaged
    assert got["ep_patches_engaged"] == 1


def test_patch_columns_absent_without_a_separation(two_patch_pdb):
    got = evaluate_epitope_policy(
        two_patch_pdb, "C", required={"A": [25, 27], "B": [50, 57]}
    )
    assert got["ep_best_patch_frac"] is None
    assert set(got) == set(EPITOPE_KEYS)
