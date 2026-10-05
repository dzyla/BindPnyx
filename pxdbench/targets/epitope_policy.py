"""Three epitope roles, and what each one is permitted to do.

| role | conditions the model | enforced at selection |
| --- | --- | --- |
| `required`  | yes | must be engaged |
| `advisory`  | yes | recorded, never enforced |
| `forbidden` | **no** | must not be engaged |

`required + advisory` becomes the diffusion `hotspot` feature, which preserves
today's behaviour: every hotspot a campaign lists is conditioned on, and none of
them is enforced. Only `required` and `forbidden` are enforced at selection.

**Coldspots are a filter, not conditioning.** The model has no coldspot input
feature, and adding one would change the model and need its own training and
validation. A pocket constraint on the required residues discourages the
forbidden region indirectly; it does not condition against it. Note also that
holding a *target* residue's identity fixed - which is implied by fixing target
sequences - does nothing to stop the binder contacting it.

`enforced()` returns the exact argument shape
`metrics.epitope.evaluate_epitope_policy(required=..., forbidden=...)` already
accepts, so no metric code changes.

Standard library only.
"""
from dataclasses import dataclass, field

from pxdbench.targets.registry import _digest
from pxdbench.targets.residue_spec import resolve_specs

#: The roles a policy may declare. `hotspot` is the legacy key, read as
#: advisory because that is what it has always meant.
ROLES = ("required", "advisory", "forbidden")
LEGACY_KEY = "hotspot"


class RoleConflictError(ValueError):
    """Roles that cannot be held simultaneously."""


def _grouped(rows):
    """{work_chain: [work_res_id, ...]} sorted, for the model and the metric."""
    out = {}
    for row in rows:
        out.setdefault(row.work_chain, []).append(row.work_res_id)
    return {chain: sorted(ids) for chain, ids in out.items()}


@dataclass(frozen=True)
class ResolvedPolicy:
    numbering: str
    required: list = field(default_factory=list)
    advisory: list = field(default_factory=list)
    forbidden: list = field(default_factory=list)
    from_legacy_key: bool = False

    # -- what the model sees ----------------------------------------------- #
    def conditioning_hotspots(self):
        """Work numbering, required + advisory. Forbidden is excluded."""
        return _grouped(list(self.required) + list(self.advisory))

    def hotspot_resolved(self):
        """The schema-2 compatibility key.

        Required + advisory only. The legacy readers treat this as the
        conditioning set, so a forbidden residue here would be conditioned on -
        the inverse of what was asked.
        """
        return self.conditioning_hotspots()

    # -- what selection enforces ------------------------------------------- #
    def enforced(self):
        """(required, forbidden) in work numbering, for the epitope metric."""
        return _grouped(self.required), _grouped(self.forbidden)

    def has_enforced_requirements(self):
        """Whether any mandatory contact clause exists.

        Advisory-only conditions generation and enforces nothing, so it is
        False here even though the policy is not empty.
        """
        return bool(self.required or self.forbidden)

    def is_empty(self):
        return not (self.required or self.advisory or self.forbidden)

    # -- identity and persistence ------------------------------------------ #
    def to_dict(self):
        def rows(items):
            return [
                {
                    "auth_chain": r.auth_chain,
                    "auth_res_id": r.auth_res_id,
                    "work_chain": r.work_chain,
                    "work_res_id": r.work_res_id,
                    "seq_index": r.seq_index,
                    "resname": r.resname,
                }
                for r in items
            ]

        return {
            "schema": 2,
            "numbering": self.numbering,
            "from_legacy_key": self.from_legacy_key,
            "roles": {
                "required": rows(self.required),
                "advisory": rows(self.advisory),
                "forbidden": rows(self.forbidden),
            },
            "hotspot_resolved": self.hotspot_resolved(),
            "has_enforced_requirements": self.has_enforced_requirements(),
        }

    @property
    def policy_digest(self):
        return _digest(self.to_dict())


def resolve_policy(requests, residue_map, numbering):
    """Resolve a raw epitope block into roles, in work numbering.

    The legacy `hotspot` key resolves to `advisory`. It may not be combined
    with a new role block: those are rejected rather than merged, so a
    migration makes the advisory intent explicit instead of guessing it.
    """
    requests = dict(requests or {})
    unknown = [k for k in requests if k not in ROLES and k != LEGACY_KEY]
    if unknown:
        raise ValueError(
            f"unknown role(s) {unknown}; expected any of {list(ROLES)} "
            f"or the legacy {LEGACY_KEY!r} key"
        )

    from_legacy = LEGACY_KEY in requests
    if from_legacy and any(role in requests for role in ROLES):
        raise RoleConflictError(
            f"the legacy {LEGACY_KEY!r} key and a new role block "
            f"({[r for r in ROLES if r in requests]}) are both present. They "
            f"are rejected rather than merged: the legacy key means "
            f"'conditioned, not enforced', and silently folding it into one of "
            f"the new roles would change what is enforced."
        )

    if from_legacy:
        resolved = {
            "advisory": resolve_specs(requests[LEGACY_KEY], residue_map, numbering),
            "required": [],
            "forbidden": [],
        }
    else:
        resolved = {
            role: resolve_specs(requests.get(role, {}), residue_map, numbering)
            for role in ROLES
        }

    required_keys = {(r.work_chain, r.work_res_id) for r in resolved["required"]}
    forbidden_keys = {(r.work_chain, r.work_res_id) for r in resolved["forbidden"]}

    # Required wins over advisory; the same residue in both is a redundant
    # request rather than a conflict.
    resolved["advisory"] = [
        r
        for r in resolved["advisory"]
        if (r.work_chain, r.work_res_id) not in required_keys
    ]
    advisory_keys = {(r.work_chain, r.work_res_id) for r in resolved["advisory"]}

    for label, keys in (("required", required_keys), ("advisory", advisory_keys)):
        clash = sorted(keys & forbidden_keys)
        if clash:
            offenders = [
                f"{r.auth_chain}{r.auth_res_id} (work {r.work_chain}{r.work_res_id})"
                for r in resolved[label]
                if (r.work_chain, r.work_res_id) in set(clash)
            ]
            raise RoleConflictError(
                f"residue(s) {offenders} are both {label} and forbidden. A "
                f"residue cannot be required to contact and required not to."
            )

    return ResolvedPolicy(
        numbering=numbering,
        required=resolved["required"],
        advisory=resolved["advisory"],
        forbidden=resolved["forbidden"],
        from_legacy_key=from_legacy,
    )
