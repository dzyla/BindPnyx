"""One preparation boundary, so a campaign input reaches the resolver.

Everything underneath this was built and tested and nothing fed it: a campaign
ran in legacy mode because no entry point resolved an `epitope` block. This is
the wiring.

Targeting-contracts section 3, in the order it matters:

1. **One idempotent boundary.** `convert_and_resolve(task, out_dir)` extends
   the existing conversion rather than competing with it, and returns a
   JSON-serialisable task - no dataclass and no atom array crosses the
   configuration or subprocess boundary.
2. **Structure first, requests held separately.** The structure is prepared
   while the user's residue requests are kept aside, then those *original*
   requests are resolved exactly once against the resulting map. Injecting
   resolved work hotspots back into the PDB conversion path - which itself
   remaps author hotspots - would convert them twice.
3. **Mode comes from configuration**, before scoring, and is written into the
   record. Never inferred from `ep_satisfied`, from a CSV column, or from a
   missing file: a missing artifact must not cause a downgrade to legacy.

The structure conversion is injected (`convert=`), which is the same seam the
project already uses for `_fold_batch`, so the contracts are testable without
initialising a model.
"""
from pxdbench.targets.epitope_policy import resolve_policy
from pxdbench.targets.record import SelectionMode, write_resolution_record

#: Set on a prepared task so re-entry is recognised rather than re-resolved.
PREPARED_MARKER = "_prepared_schema"


class PreparationError(ValueError):
    """The task cannot be prepared as declared."""


class ConditioningError(ValueError):
    """The feature mask does not match the policy it was built from."""


def apply_policy_to_configs(policy, configs):
    """Hand the resolved policy to the scorer, in WORK numbering.

    The gap the 2026-10-01 smoke run exposed: `grep -c epitope_policy
    scripts/run_campaign.sh` was **0**, so `backend.py` read
    `self._get("epitope_policy", {})` and got nothing. `parse._epitope` then ran
    with `policy=None`, `ep_satisfied` was null for every design, and a gate
    clause on it would have been inert. Conditioning reached the diffusion;
    enforcement never reached the scorer.

    Advisory residues are deliberately NOT handed over: they condition
    generation and add no mandatory contact clause, so passing them would
    invent requirements the campaign did not ask for.
    """
    required, forbidden = policy.enforced()
    handed = {}
    if required:
        handed["required"] = required
    if forbidden:
        handed["forbidden"] = forbidden
    configs.eval.binder.tools.boltz.epitope_policy = handed
    return handed


def apply_policy_dict_to_configs(handed, configs):
    """Set an already-serialised enforced policy on the scorer's config.

    The serialisable twin of `apply_policy_to_configs`, for the campaign path
    where the policy crosses a JSON boundary before reaching the backend.
    """
    configs.eval.binder.tools.boltz.epitope_policy = dict(handed or {})
    return configs.eval.binder.tools.boltz.epitope_policy


def verify_conditioning(policy, flagged, strict=False):
    """Check the parser-produced hotspot annotation against the policy.

    This has to happen at PREPARATION time. A run does not persist the feature
    mask, so afterwards there is nothing to check: a residue's presence in an
    exported structure proves the residue is there, not that its hotspot
    feature was set (review R9). An all-empty mask is indistinguishable
    downstream from a correctly conditioned run, which is precisely why it must
    fail here.

    `flagged` is {work_chain: [work_res_id, ...]} as the annotation produced
    it. Returns a list of failures, or raises when `strict`.
    """
    failures = []
    expected = policy.conditioning_hotspots()
    forbidden = {
        (r.work_chain, r.work_res_id) for r in policy.forbidden
    }
    got = {
        (chain, int(res))
        for chain, residues in (flagged or {}).items()
        for res in residues
    }

    for chain, residues in expected.items():
        for res in residues:
            if (chain, res) not in got:
                failures.append(
                    f"{chain}{res} is required or advisory but its hotspot "
                    f"feature is NOT set. The residue existing in the "
                    f"structure is not the same as it being conditioned on."
                )
    for chain, res in sorted(forbidden & got):
        failures.append(
            f"{chain}{res} is FORBIDDEN but its hotspot feature IS set; the "
            f"model has no coldspot channel, so this conditions toward the "
            f"residue the policy forbids"
        )

    if failures and strict:
        raise ConditioningError(
            f"{len(failures)} conditioning problem(s): " + "; ".join(failures)
        )
    return failures


def resolve_selection_mode(task):
    """Which mode this task runs in, from its configuration alone."""
    has_epitope = "epitope" in task or "mechanism" in task
    has_legacy = bool(task.get("hotspot"))
    requested = task.get("selection_mode")
    already_prepared = task.get(PREPARED_MARKER) == 2

    if has_epitope and has_legacy and not already_prepared:
        raise PreparationError(
            "this task declares both the legacy `hotspot` key and an "
            "`epitope` block. They are rejected rather than merged: the legacy "
            "key means 'conditioned, not enforced', and folding it into a role "
            "silently would change what is enforced. Migrate the hotspots to "
            "`epitope.advisory` to say that explicitly."
        )
    if requested == SelectionMode.LEGACY.value and has_epitope:
        raise PreparationError(
            "this task requests selection_mode='legacy' and also declares an "
            "epitope or mechanism block. A policy block cannot be evaluated in "
            "legacy mode; remove one."
        )
    if has_epitope:
        return SelectionMode.POLICY_V1
    if requested == SelectionMode.POLICY_V1.value:
        return SelectionMode.POLICY_V1
    return SelectionMode.LEGACY


def _validate_policy_inputs(task):
    if not task.get("source_structure"):
        raise PreparationError(
            "policy_v1 requires an explicit `source_structure`. Author "
            "numbering is meaningless without the structure it refers to, and "
            "discovering it from the shard's parent directory is the kind of "
            "convention that breaks on the next target."
        )
    numbering = task.get("numbering")
    if numbering not in ("auth", "work"):
        raise PreparationError(
            f"policy_v1 requires an explicit `numbering` of 'auth' or 'work', "
            f"got {numbering!r}. It is declared rather than inferred: the two "
            f"branches of convert_to_bioassembly_dict give the same number "
            f"different meanings depending on the input file's extension."
        )


def _default_convert(task, out_dir):
    """Build the residue map for a real task. Imported lazily: heavy."""
    from pxdbench.targets.residue_map import TargetResidueMap, _load_shard

    condition = task["condition"]
    shard_path = condition["structure_file"]
    chain_filter = list(condition.get("filter", {}).get("chain_id") or [])
    residue_map = TargetResidueMap.from_structures(
        task["source_structure"], shard_path, chain_filter
    )
    payload = _load_shard(shard_path)
    import hashlib

    def _digest_file(path):
        with open(path, "rb") as handle:
            return hashlib.sha256(handle.read()).hexdigest()[:16]

    return residue_map, {
        "source": _digest_file(task["source_structure"]),
        "shard": _digest_file(shard_path),
        "sequences": {
            chain: hashlib.sha256(seq.encode()).hexdigest()[:16]
            for chain, seq in (payload.get("sequences") or {}).items()
        },
    }


def _validate_prebuilt_hotspots(task, out_dir):
    """Raise KeyError naming hotspot residues that do not exist in a prebuilt shard.

    Existence only: an in-range but wrong index cannot be detected without the
    source residue map (use policy_v1, or funnel/common.py:load_target, which
    asserts residue identity)."""
    shard = str((task.get("condition") or {}).get("structure_file") or "")
    if not task.get("hotspot") or not shard.endswith(".pkl.gz"):
        return
    from pxdesign.utils.infer import load_gzip_pickle, validate_hotspots

    try:
        prebuilt = load_gzip_pickle(shard)
    except OSError:
        return  # unreadable here (e.g. a path resolved later); nothing to check against
    aa = prebuilt["atom_array"] if isinstance(prebuilt, dict) else prebuilt
    validate_hotspots(task["hotspot"], aa)  # raises KeyError naming the offenders; records nothing


def convert_and_resolve(task, out_dir, convert=None):
    """Prepare one task. Returns a JSON-serialisable task dict.

    Idempotent: a task already prepared under schema 2 is validated and its
    resolution reused rather than resolved again.
    """
    task = dict(task)
    mode = resolve_selection_mode(task)

    if mode is SelectionMode.LEGACY:
        # No source structure, no map, no policy: the behaviour every existing
        # campaign JSON depends on. One check is added: hotspots on a prebuilt
        # shard are in the SHARD's own numbering (restarts at 1), and nothing
        # validated them here, so a PDB-numbered list ran silently on the wrong
        # residues (observed: PD-L1 "Y56,E58,..." landed on ASP/LYS/...; MDM2
        # 93 and 96 on an 85-residue shard were ignored).
        _validate_prebuilt_hotspots(task, out_dir)
        task["selection_mode"] = mode.value
        return task

    _validate_policy_inputs(task)
    convert = convert or _default_convert

    # The structure is prepared with the user's ORIGINAL requests held aside,
    # then those are resolved once. Resolving a prepared task's derived
    # `hotspot` field instead would convert the numbering twice.
    original_requests = task.get("epitope") or {}
    residue_map, digests = convert(task, out_dir)
    policy = resolve_policy(
        original_requests, residue_map, numbering=task["numbering"]
    )

    record_path = write_resolution_record(
        out_dir=out_dir,
        task_name=task.get("name", "task"),
        policy=policy,
        residue_map=residue_map,
        selection_mode=mode,
        source_digest=digests.get("source"),
        shard_digest=digests.get("shard"),
        sequence_digests=digests.get("sequences"),
        original_requests=original_requests,
    )

    task["selection_mode"] = mode.value
    # Derived, and verified against the policy rather than trusted: required +
    # advisory only, so a forbidden residue is never conditioned on.
    task["hotspot"] = policy.hotspot_resolved()
    task["policy_digest"] = policy.policy_digest
    task["map_digest"] = residue_map.map_digest
    task["has_enforced_requirements"] = policy.has_enforced_requirements()
    # Carried on the task so it survives the JSON boundary into the backend
    # config. Enforced roles only: advisory conditions generation and must not
    # become a mandatory contact clause.
    required, forbidden = policy.enforced()
    handed = {}
    if required:
        handed["required"] = required
    if forbidden:
        handed["forbidden"] = forbidden
    task["resolved_policy_for_scoring"] = handed
    task[PREPARED_MARKER] = 2
    if record_path:
        task["resolved_hotspots_path"] = record_path
    return task
