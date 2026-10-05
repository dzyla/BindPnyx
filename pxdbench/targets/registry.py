"""Per-target calibration as data, with an identity that means something.

Before this, the application was built around one target at the calibration
layer: `GATE` was a single module-level dict in `tools/boltz/backend.py` and the
gate id a string literal in four places, including inside the early-stop key
built by concatenating it. Adding a second target meant reusing one protein's
calibrated thresholds on a different protein, or editing source.

**Identity is three things, not one.** A digest of thresholds alone is not an
identity: identical numbers describe different DECISIONS when the epitope
policy, seed aggregation, comparator inclusivity or distance cutoffs differ, and
an unconstrained prediction and a pocket-conditioned one can share a threshold
dict while having completely different applicability. So:

  target_label              human-readable, display only, no decision weight
  decision_rule_digest      schema, active clauses AND exact operators,
                            enforced policy, missing-data behaviour, seed
                            aggregation - what the rule DECIDES
  evaluation_context_digest target and input hashes, conditioning mode, scorer
                            version and settings - what the rule was APPLIED to

An interpretable evaluation needs both digests. Comparing a number to a
threshold whose context digest differs is the mistake this structure exists to
make visible.

Noise floors default to **unmeasured**. A borrowed floor is worse than no
floor: it licenses a false conclusion.

Pure stdlib. No numpy, no torch, no predictor imports, so it is importable
anywhere and testable without private data.
"""
import hashlib
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

#: The registry is DATA. Adding a target is a data change, not a code change.
REGISTRY_PATH = Path(__file__).with_name("registry.json")

#: Operators that may appear in a threshold. Exact operators are retained
#: rather than reduced to a `higher` boolean, because `<` and `<=` are
#: different predicates at the boundary and the existing filter tests rely on
#: that distinction.
SUPPORTED_OPS = ("<", "<=", ">", ">=")

#: `>` and `>=` mean higher-is-better; `<` and `<=` mean lower-is-better.
_HIGHER_IS_BETTER = {">", ">="}


def _digest(payload: Any) -> str:
    """A stable digest of a JSON-serialisable payload."""
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class NoiseFloors:
    """Measured reproducibility bounds for one (target, host).

    All default to None, meaning NOT MEASURED. These are properties of a setup,
    never of a method, and are never inherited from another project's report.
    """

    within_run: float | None = None
    across_run: float | None = None
    same_seed: float | None = None
    measured_on: str | None = None

    @property
    def measured(self) -> bool:
        return any(
            v is not None for v in (self.within_run, self.across_run, self.same_seed)
        )


@dataclass(frozen=True)
class TargetSpec:
    """One target, and how its residue numbering is to be read."""

    target_id: str
    target_label: str
    chain_filter: tuple[str, ...]
    #: "auth" or "work". Declared, never inferred from a file extension: the
    #: two branches of convert_to_bioassembly_dict give the same number two
    #: different meanings depending on the input format.
    numbering: str = "work"
    shard: str | None = None
    source_structure: str | None = None
    epitope: dict | None = None


@dataclass(frozen=True)
class GateSpec:
    """A selection gate, identified by what it decides and what it was applied to."""

    gate_id: str
    target_id: str
    target_label: str
    thresholds: dict[str, tuple[str, float]]
    provisional: bool = True
    engagement_required: bool = False
    seed_aggregation: str = "mean"
    missing_data: str = "unknown"
    conditioning_mode: str = "unconstrained"
    calibrated_on: str = "uncalibrated"
    context: dict = field(default_factory=dict)

    def __post_init__(self):
        for metric, (op, _value) in self.thresholds.items():
            if op not in SUPPORTED_OPS:
                raise ValueError(
                    f"gate {self.gate_id!r} clause {metric!r}: unsupported "
                    f"operator {op!r}; expected one of {list(SUPPORTED_OPS)}"
                )

    # -- identity ---------------------------------------------------------- #
    @property
    def decision_rule_digest(self) -> str:
        """What this gate DECIDES. Changes when any decision-affecting field does.

        Deliberately excludes `target_label` and `calibrated_on`: a display
        string and a provenance note do not change an outcome.
        """
        return _digest(
            {
                "schema": 1,
                "clauses": sorted(
                    (m, op, v) for m, (op, v) in self.thresholds.items()
                ),
                "engagement_required": self.engagement_required,
                "seed_aggregation": self.seed_aggregation,
                "missing_data": self.missing_data,
                "provisional": self.provisional,
            }
        )

    @property
    def evaluation_context_digest(self) -> str:
        """What this gate was APPLIED to. A pocket-conditioned run is not
        interpretable against an unconstrained calibration even when the
        thresholds are identical."""
        return _digest(
            {
                "schema": 1,
                "target_id": self.target_id,
                "conditioning_mode": self.conditioning_mode,
                "context": self.context,
            }
        )

    def higher_is_better(self, metric: str) -> bool:
        """Direction, derived from the operator rather than stored separately."""
        return self.thresholds[metric][0] in _HIGHER_IS_BETTER

    # -- derivation -------------------------------------------------------- #
    def with_overrides(self, overrides: dict[str, tuple[str, float]]) -> "GateSpec":
        merged = dict(self.thresholds)
        merged.update(overrides)
        return replace(self, thresholds=merged)

    def with_label(self, target_label: str) -> "GateSpec":
        return replace(self, target_label=target_label)

    def with_engagement_required(self, required: bool) -> "GateSpec":
        return replace(self, engagement_required=required)

    def with_context(self, **context) -> "GateSpec":
        conditioning = context.pop("conditioning_mode", self.conditioning_mode)
        merged = dict(self.context)
        merged.update(context)
        return replace(self, conditioning_mode=conditioning, context=merged)


def early_stop_key(gate_id: str) -> str:
    """The summary key counting this gate's successes.

    Derived, never a literal: `pipeline.py` previously built this by
    concatenating a target name into a module constant.
    """
    return f"{gate_id}_success.count"


class Registry:
    """Targets, gates and measured noise floors, loaded from `registry.json`."""

    def __init__(self, payload: dict):
        self._payload = payload

    @classmethod
    def load(cls, path: str | Path) -> "Registry":
        with open(path) as handle:
            return cls(json.load(handle))

    @classmethod
    def load_default(cls) -> "Registry":
        return cls.load(REGISTRY_PATH)

    # -- lookups ----------------------------------------------------------- #
    def target(self, target_id: str) -> TargetSpec:
        entry = self._payload.get("targets", {}).get(target_id)
        if entry is None:
            raise KeyError(
                f"no target {target_id!r} in the registry "
                f"(known: {sorted(self._payload.get('targets', {}))})"
            )
        return TargetSpec(
            target_id=target_id,
            target_label=entry.get("target_label", target_id),
            chain_filter=tuple(entry.get("chain_filter") or ()),
            numbering=entry.get("numbering", "work"),
            shard=entry.get("shard"),
            source_structure=entry.get("source_structure"),
            epitope=entry.get("epitope"),
        )

    def gate(self, gate_id: str) -> GateSpec:
        entry = self._payload.get("gates", {}).get(gate_id)
        if entry is None:
            raise KeyError(
                f"no gate {gate_id!r} in the registry "
                f"(known: {sorted(self._payload.get('gates', {}))})"
            )
        return GateSpec(
            gate_id=gate_id,
            target_id=entry["target_id"],
            target_label=entry.get("target_label", gate_id),
            thresholds={
                metric: (op, value)
                for metric, (op, value) in (
                    (m, tuple(ov)) for m, ov in entry["thresholds"].items()
                )
            },
            provisional=entry.get("provisional", True),
            engagement_required=entry.get("engagement_required", False),
            seed_aggregation=entry.get("seed_aggregation", "mean"),
            missing_data=entry.get("missing_data", "unknown"),
            conditioning_mode=entry.get("conditioning_mode", "unconstrained"),
            calibrated_on=entry.get("calibrated_on", "uncalibrated"),
        )

    def default_gate_id(self) -> str:
        """The gate a campaign uses when none is named.

        Read from data so that no Python module needs to contain a gate id
        literal - which is how one target came to be named in four files.
        """
        gate_id = self._payload.get("default_gate")
        if not gate_id:
            raise KeyError("the registry declares no default_gate")
        return gate_id

    def default_target_id(self) -> str:
        target_id = self._payload.get("default_target")
        if not target_id:
            raise KeyError("the registry declares no default_target")
        return target_id

    def gates_for(self, target_id: str) -> list[GateSpec]:
        found = [
            self.gate(gate_id)
            for gate_id, entry in self._payload.get("gates", {}).items()
            if entry.get("target_id") == target_id
        ]
        if not found:
            raise KeyError(
                f"no gate is calibrated for target {target_id!r}. A gate from "
                f"another target must NOT be reused: its thresholds were "
                f"calibrated on different data."
            )
        return found

    def floors(self, target_id: str, host: str) -> NoiseFloors:
        """Measured floors for this (target, host), or all-None if unmeasured."""
        entry = (
            self._payload.get("noise_floors", {})
            .get(target_id, {})
            .get(host)
        )
        if not entry:
            return NoiseFloors()
        return NoiseFloors(
            within_run=entry.get("within_run"),
            across_run=entry.get("across_run"),
            same_seed=entry.get("same_seed"),
            measured_on=entry.get("measured_on", host),
        )
