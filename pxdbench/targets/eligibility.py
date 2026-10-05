"""Mandatory eligibility, separated from quality ranking.

The defect this closes (review A1). `pre_filter_boltz` selects gate successes,
then pads up to `min_total_return` from bucket 9 - the gate FAILURES:

```python
need = min_total_return - len(successes)
out = pd.concat([successes, ordered[ordered["bucket"].eq(9)].head(need)])
```

The label `pass_boltz=False` is correct, but the row is still selected, and
`_export_structure` copies any resolvable prediction. So a design with
`ep_satisfied=False` ships, and a gate clause alone changes nothing about what
is exported. A confirming CPU probe returned exactly that row at
`min_total_return=1`.

The fix keeps **one ranking algorithm** (targeting-contracts section 5):
`pre_filter_boltz` gains keyword-only mode/context, legacy stays byte-exact,
and in `policy_v1` mandatory eligibility is filtered *before* the existing
quality buckets. Padding may then draw on quality-failing but
mandatory-eligible rows only.

Four states, and only two of them permit eligibility (section 4). Policy
presence comes from the manifest, never from a result value: the metric returns
`ep_satisfied=None` for both "no policy" and "unevaluable", so a value cannot
tell them apart.

Standard library plus pandas/numpy, which the selection path already uses.
"""
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

LEGACY_MODE = "legacy"
POLICY_V1 = "policy_v1"


class PolicyTransportError(RuntimeError):
    """A policy was declared but did not survive the journey to selection.

    This is deliberately NOT a quiet degradation. The whole failure mode this
    module exists to prevent is a declared requirement that silently evaporates
    between the manifest and the ranking, leaving `not_applicable` on every row
    and every design eligible. That reads exactly like "no policy was asked
    for", which is why it went unnoticed for so long.
    """


class PolicyStatus(Enum):
    """The four-state truth table of section 4."""

    NOT_APPLICABLE = "not_applicable"
    UNEVALUABLE = "unevaluable"
    FAIL = "fail"
    PASS = "pass"

    @property
    def permits_eligibility(self):
        """Only an absent policy or a satisfied one permits a candidate.

        `UNEVALUABLE` deliberately does not: missing, unparseable or
        incomplete results are the state R1 produces, and treating them as
        permissive is how an unverified design becomes a shipped one.
        """
        return self in (PolicyStatus.NOT_APPLICABLE, PolicyStatus.PASS)


@dataclass(frozen=True)
class SelectionContext:
    """What mode this frame was scored under, and what to check."""

    mode: str = LEGACY_MODE
    #: Metrics that must be finite for a candidate to be eligible in new mode.
    active_metrics: tuple = ()
    #: Did the MANIFEST declare enforced requirements? Taken from the config,
    #: never inferred from a result value - `ep_satisfied=None` means both "no
    #: policy" and "unevaluable", so a value cannot tell them apart.
    enforced: bool = False
    preset: str | None = None
    target_id: str | None = None
    stage_id: str | None = None
    extras: dict = field(default_factory=dict)

    @property
    def is_policy(self):
        return self.mode == POLICY_V1


def policy_status_for(
    has_enforced_requirements,
    all_expected_seeds_evaluable,
    every_requirement_satisfied,
):
    """Resolve the four-state status from manifest facts, not result values."""
    if not has_enforced_requirements:
        return PolicyStatus.NOT_APPLICABLE
    if not all_expected_seeds_evaluable:
        # Precedence: an unevaluable component outranks a known failure, so a
        # partially-missing evaluation is never reported as a clean fail.
        return PolicyStatus.UNEVALUABLE
    if every_requirement_satisfied is None:
        # `no_policy` from the metric while the manifest declares enforced
        # requirements is a transport error - the R1 state - not an absence.
        return PolicyStatus.UNEVALUABLE
    return PolicyStatus.PASS if every_requirement_satisfied else PolicyStatus.FAIL


def assert_policy_config_consistent(policy, has_enforced_requirements):
    """Refuse a configuration that disagrees with itself.

    Required residues with `has_enforced_requirements=False` yields
    `not_applicable` on every row, which permits eligibility - so the policy is
    declared, reported as absent, and enforced on nothing. The reverse
    (enforcement declared with nothing to enforce) marks every row
    `unevaluable` and rejects the entire batch. Both are configuration errors
    that present as plausible output, so neither may pass.
    """
    policy = policy or {}
    declared = bool(policy.get("required") or policy.get("forbidden"))
    if declared and not has_enforced_requirements:
        raise PolicyTransportError(
            "epitope_policy declares required/forbidden residues but "
            "has_enforced_requirements is False. Every row would be scored "
            "not_applicable, which PERMITS eligibility, so the policy would "
            "be enforced on nothing while appearing to be configured. "
            f"policy keys: {sorted(policy)}"
        )
    if has_enforced_requirements and not declared:
        raise PolicyTransportError(
            "has_enforced_requirements is True but epitope_policy declares "
            "neither required nor forbidden residues. Every row would be "
            "scored unevaluable and the whole batch rejected."
        )


def _status_of(value):
    if isinstance(value, PolicyStatus):
        return value
    if value is None:
        return PolicyStatus.UNEVALUABLE
    try:
        return PolicyStatus(str(value))
    except ValueError:
        return PolicyStatus.UNEVALUABLE


def status_permits(value):
    """Does this policy_status permit a candidate? Public form of _status_of."""
    return _status_of(value).permits_eligibility


def resolve_eligibility(df, context):
    """Add a strict boolean `eligible` column. Never fills missing with True."""
    df = df.copy()
    if not context.is_policy:
        df["eligible"] = True
        return df

    if "policy_status" in df.columns:
        if context.enforced:
            # The manifest declared requirements, so no row may claim the
            # policy did not apply to it. A row that does was scored under a
            # different configuration, or the verdict never reached it.
            stale = [
                str(v) for v in df["policy_status"]
                if _status_of(v) is PolicyStatus.NOT_APPLICABLE
            ]
            if stale:
                raise PolicyTransportError(
                    f"{len(stale)} of {len(df)} rows carry "
                    f"policy_status='not_applicable' while the manifest "
                    f"declares enforced requirements. not_applicable PERMITS "
                    f"eligibility, so these rows would be selected without "
                    f"ever being checked against the policy."
                )
        permits = df["policy_status"].map(
            lambda v: _status_of(v).permits_eligibility
        )
    else:
        # Absent policy_status in new mode is a transport failure, not an
        # absent policy. Section 5: do not silently fill with true.
        permits = np.zeros(len(df), dtype=bool)

    eligible = np.asarray(permits, dtype=bool)

    for metric in context.active_metrics:
        if metric not in df.columns:
            eligible = np.zeros(len(df), dtype=bool)
            break
        finite = df[metric].map(
            lambda v: v is not None and np.isfinite(_as_float(v))
        )
        eligible = eligible & np.asarray(finite, dtype=bool)

    df["eligible"] = eligible
    return df


def _as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")
