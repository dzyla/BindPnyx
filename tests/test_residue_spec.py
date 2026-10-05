"""Residue requests that say what they mean, and fail when they are wrong.

`validate_hotspots` only catches residues that are ABSENT. A residue that
exists but is the wrong one passes silently, which is exactly why "did we
target His535?" was unanswerable: the number resolved, so nothing complained.

A spec of `"H535"` is an assertion - *and the residue there is a histidine*. A
mismatch fails preflight, before any GPU work.

Targeting-contracts section 2 also requires finite, exact-integer validation
rather than `int(float(x))`, which silently turns `"22.9"` into position 22.
"""
import pytest

from pxdbench.targets.residue_spec import (
    IdentityAssertionError,
    ResidueSpec,
    parse_residue_spec,
    resolve_specs,
)



# --- parsing ----------------------------------------------------------------

@pytest.mark.parametrize("value,expected", [
    (535, ResidueSpec(None, 535, None)),
    ("535", ResidueSpec(None, 535, None)),
    ("H535", ResidueSpec(None, 535, "HIS")),
    ("D:H535", ResidueSpec("D", 535, "HIS")),
    ("D:535", ResidueSpec("D", 535, None)),
    ("  D:H535  ", ResidueSpec("D", 535, "HIS")),
])
def test_parse_forms(value, expected):
    assert parse_residue_spec(value) == expected


def test_default_chain_is_applied_when_the_spec_omits_one():
    assert parse_residue_spec("H535", default_chain="D") == \
        ResidueSpec("D", 535, "HIS")


def test_an_explicit_chain_overrides_the_default():
    assert parse_residue_spec("E:H535", default_chain="D").chain == "E"


def test_an_unknown_one_letter_code_is_rejected():
    """A typo must not silently become "no assertion"."""
    with pytest.raises(ValueError, match="not a one-letter"):
        parse_residue_spec("Z535")


@pytest.mark.parametrize("bad", ["22.9", "22.0", "1e3", " ", "", "H", "HIS",
                                 "D:", "--5", "0x10"])
def test_non_integer_positions_are_rejected_not_truncated(bad):
    """int(float("22.9")) is 22. Exact-integer validation instead."""
    with pytest.raises(ValueError):
        parse_residue_spec(bad)


def test_a_float_that_happens_to_be_integral_is_still_rejected():
    """A `protect` column of mixed blanks and integers types as float in
    pandas, so "22.0" arrives where 22 was meant. Reject rather than guess."""
    with pytest.raises(ValueError, match="exact integer"):
        parse_residue_spec(22.0)


def test_a_nonpositive_position_is_rejected():
    for bad in (0, -5):
        with pytest.raises(ValueError, match="positive"):
            parse_residue_spec(bad)


# --- resolution against a map ----------------------------------------------

def test_resolve_translates_auth_numbering_to_work(hadq_map):
    rows = resolve_specs({"D": ["H512", "D514"]}, hadq_map, numbering="auth")
    assert [(r.work_chain, r.work_res_id) for r in rows] == [("B", 1), ("B", 3)]


def test_resolve_accepts_work_numbering_too(hadq_map):
    rows = resolve_specs({"B": ["H1"]}, hadq_map, numbering="work")
    assert rows[0].auth_res_id == 512


def test_an_identity_assertion_failure_names_both_identities(hadq_map):
    """H512 is a histidine; asserting an arginine there must fail loudly."""
    with pytest.raises(IdentityAssertionError, match="HIS"):
        resolve_specs({"D": ["R512"]}, hadq_map, numbering="auth")


def test_an_identity_assertion_failure_names_the_request(hadq_map):
    with pytest.raises(IdentityAssertionError, match="R512"):
        resolve_specs({"D": ["R512"]}, hadq_map, numbering="auth")


def test_a_spec_without_an_assertion_resolves_without_checking_identity(hadq_map):
    rows = resolve_specs({"D": [512]}, hadq_map, numbering="auth")
    assert rows[0].resname == "HIS"


def test_an_absent_residue_is_a_resolution_error(hadq_map):
    with pytest.raises(KeyError, match="not found"):
        resolve_specs({"D": ["H999"]}, hadq_map, numbering="auth")


def test_results_are_unique_and_sorted(hadq_map):
    """Section 2: resolve roles to unique sorted residue keys."""
    rows = resolve_specs({"D": [514, 512, 514]}, hadq_map, numbering="auth")
    assert [(r.work_chain, r.work_res_id) for r in rows] == [("B", 1), ("B", 3)]


def test_an_unknown_numbering_is_rejected(hadq_map):
    with pytest.raises(ValueError, match="numbering"):
        resolve_specs({"D": [512]}, hadq_map, numbering="biological")


def test_an_empty_request_resolves_to_nothing(hadq_map):
    assert resolve_specs({}, hadq_map, numbering="auth") == []
