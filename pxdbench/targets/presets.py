"""Adjustable selection gates, with a shipped basic set.

Named presets in `gate_presets/*.json`, resolved into one effective gate. The
shape is borrowed from BindCraft's `settings_filters/*.json`, with three limits
from review A6:

1. **This pipeline's own metric catalog.** BindCraft's inventory and numbers are
   not imported because its schema happens to be convenient. (Its "218 metrics"
   is a counting convention anyway - counting nested threshold leaves gives
   332.)
2. **Exact operators are retained.** `higher` is derived direction metadata, not
   the comparison: `<` and `<=` are different predicates at the boundary, and
   `tests/test_filters.py` depends on that distinction.
3. **Mandatory validity sits outside preset deactivation.** `none` disables the
   optional quality clauses; it can never make an invalid candidate, a failed
   enforced policy, or an execution failure eligible.

**Every metric in the catalog appears in every preset**, with a `null`
threshold meaning *deliberately inactive*. An absent key is an error. That is
what makes a typo distinguishable from a disable - which matters because
`BaseTask.compute_success_rate` turns a clause whose column is missing into
`None`/unknown for the whole gate rather than reporting the mistake.

**Only `default` and `none` ship.** A6 forbids inventing calibration to make
presets differ, and the only thresholds here with provenance are the legacy
gate's three - themselves provisional. Adding a preset later is a data change,
not a code change.
"""
import json
from dataclasses import replace
from pathlib import Path

from pxdbench.targets.registry import GateSpec, _digest

PRESET_DIR = Path(__file__).with_name("gate_presets")

#: The metrics a preset may constrain. Drawn from this pipeline's own outputs:
#: backend.MEAN_KEYS, plus the seed count, plus the epitope enforcement flag.
METRIC_CATALOG = (
    "bz_n_seeds",
    "bz_ipsae",
    "bz_ipdae",
    "bz_pae_interface_min",
    "bz_interface_plddt",
    "bz_iptm",
    "bz_ptm",
    "bz_complex_plddt",
    "ep_satisfied",
)


def available_presets() -> list[str]:
    return sorted(p.stem for p in PRESET_DIR.glob("*.json"))


class PresetGate(GateSpec):
    """A GateSpec that remembers the catalog it came from and what was overridden."""

    def all_metrics(self) -> tuple[str, ...]:
        return METRIC_CATALOG


def _load_payload(name: str) -> dict:
    path = PRESET_DIR / f"{name}.json"
    if not path.exists():
        raise KeyError(
            f"no gate preset {name!r}; available: {available_presets()}. "
            f"Only `default` and `none` ship: there is no measured basis for "
            f"other tiers, and inventing thresholds to make presets differ "
            f"would imply calibration that does not exist."
        )
    payload = json.load(open(path))
    missing = [m for m in METRIC_CATALOG if m not in payload["clauses"]]
    if missing:
        raise ValueError(
            f"preset {name!r} omits {missing}. Every catalog metric must be "
            f"listed, with a null threshold for a deliberately inactive "
            f"clause, so that an omission cannot pass as a disable."
        )
    unknown = [m for m in payload["clauses"] if m not in METRIC_CATALOG]
    if unknown:
        raise ValueError(
            f"preset {name!r} names {unknown}, which are not in the metric "
            f"catalog {list(METRIC_CATALOG)}"
        )
    return payload


def load_preset(name: str) -> PresetGate:
    """One named preset, as a gate with no target context yet."""
    payload = _load_payload(name)
    thresholds = {
        metric: (clause["operator"], clause["threshold"])
        for metric, clause in payload["clauses"].items()
        if clause["threshold"] is not None
    }
    gate = PresetGate(
        gate_id=f"preset:{name}",
        target_id="",
        target_label=f"preset {name}",
        thresholds=thresholds,
        provisional=payload.get("provisional", True),
        engagement_required=False,
        calibrated_on=payload.get("calibrated_on", "uncalibrated"),
    )
    object.__setattr__(gate, "mandatory_validity",
                       payload.get("mandatory_validity", True))
    object.__setattr__(gate, "overridden", frozenset())
    object.__setattr__(gate, "preset_name", name)
    return gate


def resolve_gate(
    preset,
    target_id: str | None = None,
    overrides: dict | None = None,
    available_columns: set | None = None,
) -> PresetGate:
    """Resolve one named preset into the effective gate for a campaign.

    Order: preset, then per-target overrides, then campaign overrides. An
    override value of `None` deactivates that clause explicitly.

    `available_columns`, when given, is checked against the active clauses: a
    clause whose column will never be computed makes the whole gate `None` for
    every row rather than reporting the mistake, so it is rejected here.
    """
    if not isinstance(preset, str):
        raise ValueError(
            f"resolve_gate takes one preset name, got {preset!r}. Presets are "
            f"named and resolved one at a time; that is a scope decision, not "
            f"a technical limit - an override with an explicit null can "
            f"deactivate a clause if that is what you need."
        )
    gate = load_preset(preset)
    overridden = set()

    if overrides:
        unknown = [m for m in overrides if m not in METRIC_CATALOG]
        if unknown:
            raise ValueError(
                f"override names {unknown}, which is not in the metric catalog "
                f"{list(METRIC_CATALOG)}. A typo must not silently become a "
                f"weaker gate."
            )
        thresholds = dict(gate.thresholds)
        for metric, clause in overrides.items():
            if clause is None:
                thresholds.pop(metric, None)
            else:
                thresholds[metric] = tuple(clause)
            overridden.add(metric)
        gate = replace(gate, thresholds=thresholds)

    if available_columns is not None:
        absent = sorted(set(gate.thresholds) - set(available_columns))
        if absent:
            raise ValueError(
                f"gate clauses {absent} reference columns that will not be "
                f"computed for this run. compute_success_rate turns a missing "
                f"active metric into an UNKNOWN gate result for every row, so "
                f"this would silently make the gate undecidable rather than "
                f"failing here."
            )

    if target_id:
        gate = replace(gate, target_id=target_id)
    object.__setattr__(gate, "mandatory_validity", True)
    object.__setattr__(gate, "overridden", frozenset(overridden))
    object.__setattr__(gate, "preset_name", preset)
    return gate


def effective_digest(gate: GateSpec) -> str:
    """Identity of a resolved gate: what it decides AND what it applies to.

    A preset NAME does not reproduce a run, because the preset file can change.
    This digest goes to the run artifact alongside the resolved clauses.
    """
    return _digest(
        {
            "rule": gate.decision_rule_digest,
            "context": gate.evaluation_context_digest,
        }
    )


# Attached as a method so callers can write gate.effective_digest().
GateSpec.effective_digest = effective_digest  # type: ignore[attr-defined]
