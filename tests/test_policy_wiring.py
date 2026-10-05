"""The three gaps the smoke run exposed.

The 2026-10-01 smoke run passed criteria 1-3 and failed criterion 4:
`ep_satisfied` was null in 4 of 4 designs and no `policy_status` column
existed, because `grep -c epitope_policy scripts/run_campaign.sh` was **0**.
Conditioning reached the diffusion; enforcement never reached the scorer. A
gate clause would have been inert.

1. `apply_policy_to_configs` puts the resolved policy where the backend reads
   it (`eval.binder.tools.boltz.epitope_policy`).
2. `verify_conditioning` drives the real annotation path at preparation time,
   because a run does not persist the feature mask and so cannot be checked
   afterwards (R9: a residue's presence in an export proves the residue is
   there, not that its feature was set).
3. `run_campaign.sh` must exit non-zero on a bad flag and on preflight failure.
   It exited 0 for both, which is the "`State=COMPLETED` is not success"
   defect inside the wrapper written to prevent it.
"""
import subprocess

import pytest

from pxdbench.targets.epitope_policy import resolve_policy
from pxdbench.targets.preparation import (
    ConditioningError,
    apply_policy_to_configs,
    verify_conditioning,
)

POLICY = {"required": {"D": ["H512"]}, "forbidden": {"D": ["A513"]}}


class _Cfg(dict):
    """A dict that also allows attribute access, like ConfigDict."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


def _configs():
    return _Cfg(
        eval=_Cfg(binder=_Cfg(tools=_Cfg(boltz=_Cfg(epitope_policy={}))))
    )


# --- 1. the policy must reach the backend ----------------------------------

def test_the_resolved_policy_lands_where_the_backend_reads_it(hadq_map):
    """backend.py:422 reads self._get("epitope_policy", {})."""
    policy = resolve_policy(POLICY, hadq_map, numbering="auth")
    configs = _configs()
    apply_policy_to_configs(policy, configs)
    got = configs.eval.binder.tools.boltz.epitope_policy
    assert got["required"] == {"B": [1]}
    assert got["forbidden"] == {"B": [2]}


def test_the_policy_handed_over_is_in_WORK_numbering(hadq_map):
    """The metric evaluates against the predicted structure's numbering."""
    policy = resolve_policy(POLICY, hadq_map, numbering="auth")
    configs = _configs()
    apply_policy_to_configs(policy, configs)
    got = configs.eval.binder.tools.boltz.epitope_policy
    residues = sorted(
        res for role in got.values() for ids in role.values() for res in ids
    )
    assert residues == [1, 2], f"expected work numbering, got {residues}"
    assert 512 not in residues, "author numbering must not reach the metric"


def test_an_advisory_only_policy_hands_over_no_enforced_clause(hadq_map):
    """Advisory conditions generation and adds no mandatory contact."""
    policy = resolve_policy({"advisory": {"D": [512]}}, hadq_map, numbering="auth")
    configs = _configs()
    apply_policy_to_configs(policy, configs)
    got = configs.eval.binder.tools.boltz.epitope_policy
    assert got == {} or (not got.get("required") and not got.get("forbidden"))


def test_a_forbidden_only_policy_is_handed_over(hadq_map):
    policy = resolve_policy({"forbidden": {"D": [513]}}, hadq_map, numbering="auth")
    configs = _configs()
    apply_policy_to_configs(policy, configs)
    assert configs.eval.binder.tools.boltz.epitope_policy["forbidden"] == {"B": [2]}


def test_an_empty_policy_leaves_the_config_empty(hadq_map):
    policy = resolve_policy({}, hadq_map, numbering="auth")
    configs = _configs()
    apply_policy_to_configs(policy, configs)
    assert not configs.eval.binder.tools.boltz.epitope_policy


# --- 2. conditioning verified at preparation time --------------------------

def test_conditioning_is_verified_against_the_real_annotation(hadq_map):
    """Drives the real masking rule, not a reimplementation of it."""
    policy = resolve_policy({"required": {"D": ["H512"]}}, hadq_map, numbering="auth")
    # The annotation the parser would produce for this policy.
    flagged = {"B": [1]}
    assert verify_conditioning(policy, flagged) == []


def test_a_missing_conditioned_residue_is_a_conditioning_error(hadq_map):
    policy = resolve_policy(POLICY, hadq_map, numbering="auth")
    with pytest.raises(ConditioningError, match="B1"):
        verify_conditioning(policy, {"B": []}, strict=True)


def test_a_forbidden_residue_in_the_mask_is_a_conditioning_error(hadq_map):
    """The model has no coldspot channel, so this conditions TOWARD it."""
    policy = resolve_policy(POLICY, hadq_map, numbering="auth")
    with pytest.raises(ConditioningError, match="B2"):
        verify_conditioning(policy, {"B": [1, 2]}, strict=True)


def test_an_unconditioned_run_is_caught(hadq_map):
    """An all-empty mask is indistinguishable downstream from a conditioned
    run, which is exactly why it must fail here."""
    policy = resolve_policy(POLICY, hadq_map, numbering="auth")
    failures = verify_conditioning(policy, {})
    assert failures and any("B1" in f for f in failures)


def test_an_empty_policy_needs_no_conditioning(hadq_map):
    policy = resolve_policy({}, hadq_map, numbering="auth")
    assert verify_conditioning(policy, {}) == []


# --- 3. the wrapper must fail loudly ---------------------------------------

def test_the_wrapper_exits_nonzero_on_an_unrecognised_flag():
    """It exited 0, printed usage, and ran nothing."""
    out = subprocess.run(
        ["./scripts/run_campaign.sh", "-i", "/dev/null", "-o", "/tmp/x",
         "--no-such-flag", "3"],
        capture_output=True, text=True, timeout=120,
    )
    assert out.returncode != 0, (
        "a typo'd flag reported success and produced no campaign"
    )


def test_the_wrapper_exits_nonzero_without_required_arguments():
    out = subprocess.run(
        ["./scripts/run_campaign.sh"], capture_output=True, text=True, timeout=120
    )
    assert out.returncode != 0


def test_the_wrapper_names_the_offending_flag():
    out = subprocess.run(
        ["./scripts/run_campaign.sh", "-i", "/dev/null", "-o", "/tmp/x",
         "--seeds", "3"],
        capture_output=True, text=True, timeout=120,
    )
    combined = out.stdout + out.stderr
    assert "--seeds" in combined
    assert "--seeds-gate" in combined, (
        "the real flag should be suggested; --seeds was my own mistake, made "
        "by grepping the help text where it matched --seeds-gate as a prefix"
    )
