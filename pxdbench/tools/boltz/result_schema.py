"""The explicit result schema: what survives seed aggregation and rescoring.

This closes R1. Two places threw the epitope result away:

- `backend._mean_scores` rebuilt each row as `{"bz_status": ...}` plus
  `MEAN_KEYS` - seven numeric `bz_*` fields and no `ep_*`;
- `final_batch.py` propagated columns by a `"bz_"` prefix whitelist.

So `ep_satisfied` never reached selection, and a gate clause on it read a
column that did not exist. Targeting-contracts section 6 therefore requires an
**explicit schema** rather than a prefix test.

The aggregation rule is a **tri-state conjunction**, never a float mean and
never a list. Section 4 requires every requested seed to be present, evaluable
and satisfying; and the filter engine reduces list-valued metrics in the
*favourable* direction, so handing it per-seed booleans would let one good pose
carry a design.

Missing results do not silently reduce the denominator: an incomplete roster is
`unevaluable`, and `policy_pass_fraction` is null there rather than a fraction
that would read as evidence.

Standard library only.
"""
from pxdbench.targets.eligibility import PolicyStatus

#: Predictor statuses that count as a usable evaluation.
OK_STATUS = "ok"

#: Typed policy fields carried through aggregation, rescoring and export.
#: Explicit, because a prefix test is what dropped them.
POLICY_FIELDS = (
    "policy_status",
    "ep_satisfied",
    "policy_pass_fraction",
    "requested_seed_count",
    "valid_seed_count",
    "per_seed_policy",
)

#: Per-seed epitope detail retained alongside the summary. Section 6: an empty
#: collection is distinct from an unevaluated null.
PER_SEED_EPITOPE_FIELDS = (
    "ep_satisfied",
    "ep_status",
    "ep_missed_required",
    "ep_violated_forbidden",
    "ep_contacted_residues",
)


def aggregate_policy(seed_rows, requested_seeds, has_enforced_requirements):
    """Collapse per-seed policy results into one typed verdict.

    `requested_seeds` is the stage's declared roster. A rank stage asking for
    one seed is complete at one seed; it is not final-confirmed at the final
    stage's greater depth.
    """
    requested = tuple(requested_seeds)
    seen = {}
    duplicated = False
    for row in seed_rows:
        seed = row.get("seed")
        if seed in seen:
            duplicated = True
        seen[seed] = row

    per_seed = [
        {
            "seed": seed,
            **{
                field: seen[seed].get(field)
                for field in PER_SEED_EPITOPE_FIELDS
            },
        }
        for seed in sorted(seen)
    ]

    usable = [
        row
        for seed, row in sorted(seen.items())
        if row.get("bz_status") == OK_STATUS
    ]
    complete = (
        not duplicated
        and set(seen) == set(requested)
        and len(usable) == len(requested)
    )

    summary = {
        "requested_seed_count": len(requested),
        "valid_seed_count": len(usable),
        "per_seed_policy": per_seed,
    }

    if not has_enforced_requirements:
        # Nothing was asked, so nothing was verified. Reporting a pass here is
        # the failure the epitope metric already refuses to make.
        summary.update(
            policy_status=PolicyStatus.NOT_APPLICABLE.value,
            ep_satisfied=None,
            policy_pass_fraction=None,
        )
        return summary

    satisfied = [row.get("ep_satisfied") for row in usable]
    if not complete or any(value is None for value in satisfied):
        # Precedence: unevaluable outranks a known failure, so a partially
        # missing evaluation is never reported as a clean fail.
        summary.update(
            policy_status=PolicyStatus.UNEVALUABLE.value,
            ep_satisfied=None,
            policy_pass_fraction=None,
        )
        return summary

    passed = sum(1 for value in satisfied if value)
    every = passed == len(satisfied)
    summary.update(
        policy_status=(
            PolicyStatus.PASS.value if every else PolicyStatus.FAIL.value
        ),
        ep_satisfied=bool(every),
        policy_pass_fraction=passed / len(satisfied),
    )
    return summary


def propagated_columns(available):
    """Which columns the common final batch carries forward.

    An explicit union of the numeric `bz_*` results actually present, the
    epitope detail and the typed policy fields - not `k.startswith("bz_")`,
    which is what silently dropped every policy column.
    """
    from pxdbench.metrics.epitope import EPITOPE_KEYS

    schema = set(POLICY_FIELDS) | set(EPITOPE_KEYS) | {"ep_status"}
    return sorted(
        {name for name in available if name.startswith("bz_")}
        | (schema & set(available))
    )
