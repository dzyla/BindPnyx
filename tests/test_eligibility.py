"""A failed policy must not reach export through padding.

Review A1, confirmed by CPU probe and by reading `pre_filter_boltz`:

```python
need = min_total_return - len(successes)
out = pd.concat([successes, ordered[ordered["bucket"].eq(9)].head(need)])
```

Bucket 9 is the gate FAILURES. Below `min_total_return` (default 20) they pad
into the selection. The label `pass_boltz=False` is correct, but the row is
still selected, and `_export_structure` copies any resolvable prediction. So a
design with `ep_satisfied=False` ships, and adding a gate clause alone changes
nothing about what is exported.

Targeting-contracts section 5 keeps ONE ranking algorithm: `pre_filter_boltz`
gains keyword-only mode/context, the legacy default is preserved exactly, and
in `policy_v1` mandatory eligibility is filtered BEFORE the existing quality
buckets. Padding may then use quality-failing but mandatory-ELIGIBLE rows only.

The four-state truth table is section 4: `not_applicable`, `unevaluable`,
`fail`, provisional `pass`. Only the first and last permit eligibility.
"""
import numpy as np
import pandas as pd
import pytest
pytest.importorskip("protenix", reason="needs the full pxd environment (protenix)")

from pxdbench.targets.eligibility import (
    PolicyStatus,
    SelectionContext,
    policy_status_for,
    resolve_eligibility,
)
from pxdesign.runner.helpers import pre_filter_boltz


def _frame(rows):
    return pd.DataFrame(rows)


def _row(name, ipsae, gate_success, **extra):
    base = {
        "name": name,
        "bz_ipsae": ipsae,
        "bz_final_batch_id": "b1",
        "bz_gate_egfr_provisional_v1_success": gate_success,
    }
    base.update(extra)
    return base


# --- the truth table -------------------------------------------------------

@pytest.mark.parametrize("enforced,seeds_complete,satisfied,expected", [
    (False, True, None, PolicyStatus.NOT_APPLICABLE),
    (True, False, None, PolicyStatus.UNEVALUABLE),
    (True, True, False, PolicyStatus.FAIL),
    (True, True, True, PolicyStatus.PASS),
])
def test_the_four_states(enforced, seeds_complete, satisfied, expected):
    assert policy_status_for(
        has_enforced_requirements=enforced,
        all_expected_seeds_evaluable=seeds_complete,
        every_requirement_satisfied=satisfied,
    ) is expected


@pytest.mark.parametrize("status,permits", [
    (PolicyStatus.NOT_APPLICABLE, True),
    (PolicyStatus.PASS, True),
    (PolicyStatus.UNEVALUABLE, False),
    (PolicyStatus.FAIL, False),
])
def test_only_not_applicable_and_pass_permit_eligibility(status, permits):
    assert status.permits_eligibility is permits


def test_no_policy_from_the_metric_with_enforced_requirements_is_unevaluable():
    """Section 4: that combination is a transport error, never not-applicable.

    It is the R1 state - the ep_* columns were dropped before selection.
    """
    assert policy_status_for(
        has_enforced_requirements=True,
        all_expected_seeds_evaluable=True,
        every_requirement_satisfied=None,
    ) is PolicyStatus.UNEVALUABLE


def test_unevaluable_takes_precedence_over_a_known_failure():
    """Section 4: for combined epitope and mechanism policy, an unevaluable
    component outranks a failing one."""
    assert policy_status_for(
        has_enforced_requirements=True,
        all_expected_seeds_evaluable=False,
        every_requirement_satisfied=False,
    ) is PolicyStatus.UNEVALUABLE


# --- eligibility is strict -------------------------------------------------

def test_eligibility_requires_a_permitting_policy_status():
    df = _frame([
        _row("ok", 0.9, 1, policy_status="pass"),
        _row("bad", 0.9, 1, policy_status="fail"),
        _row("unk", 0.9, 1, policy_status="unevaluable"),
    ])
    out = resolve_eligibility(df, SelectionContext(mode="policy_v1"))
    assert out.set_index("name")["eligible"].to_dict() == {
        "ok": True, "bad": False, "unk": False
    }


def test_a_missing_eligibility_column_is_never_filled_with_true():
    """Section 5: do not silently fill missing eligibility with true."""
    df = _frame([_row("x", 0.9, 1)])
    out = resolve_eligibility(df, SelectionContext(mode="policy_v1"))
    assert out["eligible"].tolist() == [False]


def test_a_nonfinite_active_metric_is_not_eligible():
    """Section 4: new-mode eligibility also requires finite active metrics."""
    df = _frame([
        _row("nan", float("nan"), 1, policy_status="pass"),
        _row("fine", 0.9, 1, policy_status="pass"),
    ])
    out = resolve_eligibility(
        df, SelectionContext(mode="policy_v1", active_metrics=("bz_ipsae",))
    )
    assert out.set_index("name")["eligible"].to_dict() == {
        "nan": False, "fine": True
    }


# --- A1: the padding bypass ------------------------------------------------

def test_a_policy_failure_cannot_pad_the_candidate_table():
    """Set min_total_return above the eligible count: the table must stay
    UNDERFILLED rather than borrow a policy failure."""
    df = _frame([
        _row("ok", 0.90, 1, policy_status="pass", eligible=True),
        _row("bad", 0.95, 0, policy_status="fail", eligible=False),
    ])
    out = pre_filter_boltz(
        df, min_total_return=5, per_backbone_cap=1,
        mode="policy_v1", context=SelectionContext(mode="policy_v1"),
    )
    assert out["name"].tolist() == ["ok"], "a policy failure was padded in"


def test_an_unevaluable_policy_cannot_pad_either():
    df = _frame([
        _row("ok", 0.90, 1, policy_status="pass", eligible=True),
        _row("unk", 0.99, 0, policy_status="unevaluable", eligible=False),
    ])
    out = pre_filter_boltz(
        df, min_total_return=5, per_backbone_cap=1,
        mode="policy_v1", context=SelectionContext(mode="policy_v1"),
    )
    assert out["name"].tolist() == ["ok"]


def test_a_quality_failure_may_still_pad_within_the_eligible_subset():
    """Section 5: padding can use quality-failing but mandatory-eligible rows."""
    df = _frame([
        _row("pass_both", 0.90, 1, policy_status="pass", eligible=True),
        _row("quality_fail", 0.10, 0, policy_status="pass", eligible=True),
    ])
    out = pre_filter_boltz(
        df, min_total_return=2, per_backbone_cap=1,
        mode="policy_v1", context=SelectionContext(mode="policy_v1"),
    )
    assert sorted(out["name"].tolist()) == ["pass_both", "quality_fail"]


def test_zero_eligible_designs_yields_an_empty_candidate_table():
    """Section 5: possibly header-only, and never padded to look non-empty."""
    df = _frame([
        _row("bad", 0.95, 0, policy_status="fail", eligible=False),
    ])
    out = pre_filter_boltz(
        df, min_total_return=5, per_backbone_cap=1,
        mode="policy_v1", context=SelectionContext(mode="policy_v1"),
    )
    assert len(out) == 0
    assert "name" in out.columns, "an empty table still carries its columns"


def test_preset_none_cannot_bypass_the_policy():
    """Section 4: `none` removes optional thresholds, not policy evaluation."""
    df = _frame([
        _row("bad", 0.95, 1, policy_status="fail", eligible=False),
    ])
    out = pre_filter_boltz(
        df, min_total_return=5, per_backbone_cap=1,
        mode="policy_v1",
        context=SelectionContext(mode="policy_v1", preset="none"),
    )
    assert len(out) == 0


# --- legacy is untouched ---------------------------------------------------

def test_a_caller_with_no_mode_argument_takes_the_legacy_path_exactly():
    """Including the padding, which legacy campaigns depend on."""
    df = _frame([
        _row("ok", 0.90, 1),
        _row("fail", 0.95, 0),
    ])
    out = pre_filter_boltz(df, min_total_return=2, per_backbone_cap=1)
    assert sorted(out["name"].tolist()) == ["fail", "ok"], (
        "legacy padding must be preserved"
    )


def test_a_policy_task_sent_down_the_legacy_path_is_rejected():
    """Section 5: the resolver rejects that rather than silently degrading."""
    df = _frame([_row("x", 0.9, 1, policy_status="pass", eligible=True)])
    with pytest.raises(ValueError, match="policy_v1|mode"):
        pre_filter_boltz(
            df, min_total_return=1, per_backbone_cap=1,
            context=SelectionContext(mode="policy_v1"),
        )


def test_the_rejected_population_is_recoverable_for_the_audit_table():
    """Section 5: the orchestration layer writes rejected rows from the
    original population, so selection must not be the only record."""
    df = _frame([
        _row("ok", 0.90, 1, policy_status="pass", eligible=True),
        _row("bad", 0.95, 0, policy_status="fail", eligible=False),
    ])
    out = pre_filter_boltz(
        df, min_total_return=5, per_backbone_cap=1,
        mode="policy_v1", context=SelectionContext(mode="policy_v1"),
    )
    rejected = set(df["name"]) - set(out["name"])
    assert rejected == {"bad"}
