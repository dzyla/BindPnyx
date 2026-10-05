"""Per-seed policy results that survive aggregation.

This is the R1 blocker. Two places threw the epitope result away:

- `backend._mean_scores` rebuilds each row as `{"bz_status": ...}` plus
  `MEAN_KEYS` - seven numeric `bz_*` fields and no `ep_*`;
- `final_batch.py:76` propagates columns by a `"bz_"` prefix whitelist.

So `ep_satisfied` never reached selection, and a gate clause on it read a
column that did not exist. Targeting-contracts section 6 requires an explicit
result schema rather than a prefix whitelist, and a tri-state conjunction for
the policy - never a float mean, and never a list handed to the filter engine,
which reduces list-valued metrics in the FAVOURABLE direction and would let one
good pose carry a design.

Section 4: every requested scoring seed must be present, evaluable, and satisfy
the policy. Missing results do not silently reduce the denominator.
"""
import math

import pytest

from pxdbench.targets.eligibility import PolicyStatus
from pxdbench.tools.boltz.result_schema import (
    POLICY_FIELDS,
    aggregate_policy,
    propagated_columns,
)


def _seed(seed, *, satisfied=True, status="ok", ipsae=0.8, ep_status="ok"):
    return {
        "seed": seed,
        "bz_status": status,
        "bz_ipsae": ipsae,
        "ep_satisfied": satisfied,
        "ep_status": ep_status,
        "ep_missed_required": "",
        "ep_violated_forbidden": "",
        "ep_contacted_residues": "B24",
    }


# --- the tri-state conjunction ---------------------------------------------

def test_all_seeds_satisfying_gives_a_pass():
    got = aggregate_policy(
        [_seed(1), _seed(2), _seed(3)], requested_seeds=(1, 2, 3),
        has_enforced_requirements=True,
    )
    assert got["policy_status"] == PolicyStatus.PASS.value
    assert got["ep_satisfied"] is True
    assert got["policy_pass_fraction"] == 1.0


def test_one_failing_seed_fails_the_design():
    """A conjunction, not a vote: section 4 requires EVERY requested seed to
    satisfy the policy."""
    got = aggregate_policy(
        [_seed(1), _seed(2, satisfied=False), _seed(3)],
        requested_seeds=(1, 2, 3), has_enforced_requirements=True,
    )
    assert got["policy_status"] == PolicyStatus.FAIL.value
    assert got["ep_satisfied"] is False
    assert got["policy_pass_fraction"] == pytest.approx(2 / 3)


def test_a_majority_of_passing_seeds_is_still_a_failure():
    """Guards against the favourable-direction reduction the filter engine
    would apply to a list."""
    got = aggregate_policy(
        [_seed(1), _seed(2), _seed(3, satisfied=False)],
        requested_seeds=(1, 2, 3), has_enforced_requirements=True,
    )
    assert got["ep_satisfied"] is False


def test_ep_satisfied_is_never_a_float_or_a_list():
    got = aggregate_policy(
        [_seed(1), _seed(2, satisfied=False)], requested_seeds=(1, 2),
        has_enforced_requirements=True,
    )
    assert isinstance(got["ep_satisfied"], bool)


# --- incomplete rosters ----------------------------------------------------

def test_a_missing_seed_is_unevaluable_not_a_reduced_denominator():
    got = aggregate_policy(
        [_seed(1), _seed(2)], requested_seeds=(1, 2, 3),
        has_enforced_requirements=True,
    )
    assert got["policy_status"] == PolicyStatus.UNEVALUABLE.value
    assert got["ep_satisfied"] is None
    assert got["requested_seed_count"] == 3
    assert got["valid_seed_count"] == 2
    assert got["policy_pass_fraction"] is None, (
        "a fraction over an incomplete roster would read as evidence"
    )


def test_a_duplicate_seed_is_unevaluable():
    got = aggregate_policy(
        [_seed(1), _seed(1), _seed(2)], requested_seeds=(1, 2),
        has_enforced_requirements=True,
    )
    assert got["policy_status"] == PolicyStatus.UNEVALUABLE.value


def test_a_failed_predictor_seed_is_unevaluable():
    got = aggregate_policy(
        [_seed(1), _seed(2, status="failed")], requested_seeds=(1, 2),
        has_enforced_requirements=True,
    )
    assert got["policy_status"] == PolicyStatus.UNEVALUABLE.value


def test_an_unevaluable_metric_result_is_unevaluable_not_a_pass():
    """ep_status='unevaluable' returns ep_satisfied=None from the metric; that
    must not read as 'no policy'."""
    got = aggregate_policy(
        [_seed(1), _seed(2, satisfied=None, ep_status="unevaluable")],
        requested_seeds=(1, 2), has_enforced_requirements=True,
    )
    assert got["policy_status"] == PolicyStatus.UNEVALUABLE.value


def test_no_policy_from_the_metric_with_enforced_requirements_is_unevaluable():
    """The R1 state itself: the columns were dropped."""
    got = aggregate_policy(
        [_seed(1, satisfied=None, ep_status="no_policy")],
        requested_seeds=(1,), has_enforced_requirements=True,
    )
    assert got["policy_status"] == PolicyStatus.UNEVALUABLE.value


def test_unevaluable_outranks_a_known_failure():
    got = aggregate_policy(
        [_seed(1, satisfied=False), _seed(2, status="timeout")],
        requested_seeds=(1, 2), has_enforced_requirements=True,
    )
    assert got["policy_status"] == PolicyStatus.UNEVALUABLE.value


# --- no enforced policy ----------------------------------------------------

def test_without_enforced_requirements_the_status_is_not_applicable():
    got = aggregate_policy(
        [_seed(1, satisfied=None, ep_status="no_policy")],
        requested_seeds=(1,), has_enforced_requirements=False,
    )
    assert got["policy_status"] == PolicyStatus.NOT_APPLICABLE.value
    assert got["ep_satisfied"] is None
    assert got["policy_pass_fraction"] is None


def test_an_advisory_only_run_is_not_applicable_even_with_contacts():
    got = aggregate_policy(
        [_seed(1)], requested_seeds=(1,), has_enforced_requirements=False
    )
    assert got["policy_status"] == PolicyStatus.NOT_APPLICABLE.value


# --- per-seed references ---------------------------------------------------

def test_per_seed_results_are_retained_not_only_summarised():
    got = aggregate_policy(
        [_seed(1), _seed(2, satisfied=False)], requested_seeds=(1, 2),
        has_enforced_requirements=True,
    )
    per_seed = got["per_seed_policy"]
    assert [r["seed"] for r in per_seed] == [1, 2]
    assert [r["ep_satisfied"] for r in per_seed] == [True, False]


def test_contact_sets_are_kept_per_seed(_=None):
    got = aggregate_policy(
        [_seed(1)], requested_seeds=(1,), has_enforced_requirements=True
    )
    assert got["per_seed_policy"][0]["ep_contacted_residues"] == "B24"


# --- the explicit propagation schema ---------------------------------------

def test_the_propagated_columns_are_explicit_not_a_bz_prefix_whitelist():
    """final_batch.py:76 used `k.startswith("bz_")`, which silently dropped
    every ep_* and policy field."""
    columns = propagated_columns({"bz_ipsae", "ep_satisfied", "policy_status",
                                  "unrelated_column"})
    assert "bz_ipsae" in columns
    assert "ep_satisfied" in columns
    assert "policy_status" in columns
    assert "unrelated_column" not in columns


def test_every_policy_field_is_in_the_propagation_schema():
    for field in POLICY_FIELDS:
        assert field in propagated_columns(set(POLICY_FIELDS)), field


def test_a_bz_column_absent_from_the_results_is_not_invented():
    assert "bz_nonexistent" not in propagated_columns({"bz_ipsae"})
