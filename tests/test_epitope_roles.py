"""Three epitope roles, and what each one is allowed to do.

| role | conditions the model | enforced at selection |
| --- | --- | --- |
| `required`  | yes | must be engaged |
| `advisory`  | yes | recorded, never enforced |
| `forbidden` | **no** | must not be engaged |

`required + advisory` becomes the diffusion `hotspot` feature, preserving
today's behaviour where every listed hotspot is conditioned on. Only `required`
and `forbidden` are enforced.

Coldspots are a post-prediction filter only: the model has no coldspot input
feature, and adding one would change the model and need its own training and
validation. Holding a *target* residue's identity fixed does nothing to prevent
the binder contacting it.

Targeting-contracts section 2: unique sorted keys; required/forbidden and
advisory/forbidden overlap rejected; required wins over advisory;
`hotspot_resolved` is required + advisory only; a forbidden-only policy must
still serialise even though its conditioning set is empty.
"""
import pytest

from pxdbench.targets.epitope_policy import (
    RoleConflictError,
    resolve_policy,
)

POLICY = {
    "required": {"D": ["H512"]},
    "advisory": {"D": ["D514", "Q515"]},
    "forbidden": {"D": ["A513"]},
}


def test_required_and_advisory_condition_the_model(hadq_map):
    policy = resolve_policy(POLICY, hadq_map, numbering="auth")
    assert sorted(policy.conditioning_hotspots()["B"]) == [1, 3, 4]


def test_forbidden_is_never_conditioned_on(hadq_map):
    """A513 is work B2. It must not appear in the conditioning set."""
    policy = resolve_policy(POLICY, hadq_map, numbering="auth")
    assert 2 not in policy.conditioning_hotspots()["B"]


def test_enforced_returns_required_and_forbidden_only(hadq_map):
    """The exact argument shape evaluate_epitope_policy already accepts."""
    required, forbidden = resolve_policy(
        POLICY, hadq_map, numbering="auth"
    ).enforced()
    assert required == {"B": [1]}
    assert forbidden == {"B": [2]}


def test_the_compatibility_key_excludes_forbidden_residues(hadq_map):
    """hotspot_resolved feeds the legacy readers, which treat it as
    conditioning. A forbidden residue there would be conditioned on."""
    policy = resolve_policy(POLICY, hadq_map, numbering="auth")
    assert policy.hotspot_resolved() == {"B": [1, 3, 4]}


def test_required_wins_when_a_residue_is_also_advisory(hadq_map):
    policy = resolve_policy(
        {"required": {"D": ["H512"]}, "advisory": {"D": ["H512", "D514"]}},
        hadq_map, numbering="auth",
    )
    assert [r.work_res_id for r in policy.required] == [1]
    assert [r.work_res_id for r in policy.advisory] == [3]


@pytest.mark.parametrize("conflicting", [
    {"required": {"D": [512]}, "forbidden": {"D": [512]}},
    {"advisory": {"D": [512]}, "forbidden": {"D": [512]}},
])
def test_overlap_with_forbidden_is_rejected(hadq_map, conflicting):
    with pytest.raises(RoleConflictError, match="D512|B1"):
        resolve_policy(conflicting, hadq_map, numbering="auth")


# --- enforcement state ------------------------------------------------------

def test_an_absent_policy_is_not_enforced(hadq_map):
    policy = resolve_policy({}, hadq_map, numbering="auth")
    assert policy.is_empty()
    assert policy.has_enforced_requirements() is False


def test_an_advisory_only_policy_conditions_but_does_not_enforce(hadq_map):
    """Section 1: advisory-only conditions generation but adds no mandatory
    contact clause."""
    policy = resolve_policy(
        {"advisory": {"D": [512]}}, hadq_map, numbering="auth"
    )
    assert policy.has_enforced_requirements() is False
    assert policy.conditioning_hotspots() == {"B": [1]}
    assert policy.enforced() == ({}, {})


def test_a_forbidden_only_policy_is_enforced_with_an_empty_hotspot_set(hadq_map):
    """It must serialise even though nothing is conditioned on."""
    policy = resolve_policy(
        {"forbidden": {"D": [513]}}, hadq_map, numbering="auth"
    )
    assert policy.is_empty() is False
    assert policy.has_enforced_requirements() is True
    assert policy.conditioning_hotspots() == {}
    assert policy.enforced() == ({}, {"B": [2]})


def test_a_required_only_policy_is_enforced(hadq_map):
    policy = resolve_policy(
        {"required": {"D": [512]}}, hadq_map, numbering="auth"
    )
    assert policy.has_enforced_requirements() is True


# --- legacy compatibility ---------------------------------------------------

def test_the_legacy_hotspot_key_is_advisory(hadq_map):
    """Today's `hotspot` means conditioned, not enforced. Migrating it to
    `advisory` makes that intent explicit without changing behaviour."""
    policy = resolve_policy(
        {"hotspot": {"B": [1, 3]}}, hadq_map, numbering="work"
    )
    assert policy.required == [] and policy.forbidden == []
    assert sorted(policy.conditioning_hotspots()["B"]) == [1, 3]
    assert policy.has_enforced_requirements() is False


def test_legacy_hotspot_and_a_new_epitope_block_are_rejected(hadq_map):
    """Section 1: rejected rather than merged."""
    with pytest.raises(RoleConflictError, match="legacy"):
        resolve_policy(
            {"hotspot": {"B": [1]}, "required": {"D": [512]}},
            hadq_map, numbering="work",
        )


def test_an_unknown_role_is_rejected(hadq_map):
    with pytest.raises(ValueError, match="unknown role"):
        resolve_policy({"preferred": {"D": [512]}}, hadq_map, numbering="auth")


# --- serialisation ----------------------------------------------------------

def test_the_policy_serialises_all_roles_with_both_numberings(hadq_map):
    payload = resolve_policy(POLICY, hadq_map, numbering="auth").to_dict()
    assert payload["numbering"] == "auth"
    required = payload["roles"]["required"][0]
    assert required["auth_chain"] == "D" and required["auth_res_id"] == 512
    assert required["work_chain"] == "B" and required["work_res_id"] == 1
    assert required["seq_index"] == 1 and required["resname"] == "HIS"
    assert [r["work_res_id"] for r in payload["roles"]["forbidden"]] == [2]


def test_the_policy_digest_changes_with_the_roles(hadq_map):
    a = resolve_policy({"required": {"D": [512]}}, hadq_map, numbering="auth")
    b = resolve_policy({"advisory": {"D": [512]}}, hadq_map, numbering="auth")
    assert a.policy_digest != b.policy_digest, (
        "the same residue in a different role is a different policy"
    )
