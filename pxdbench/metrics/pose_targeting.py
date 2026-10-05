"""Did the hotspot conditioning survive into the pose that gets scored?

`epitope.py` answers "did the design bind where we asked" for ONE structure.
This module answers the prior question: which structure should you ask about?

Hotspot conditioning is a per-token model feature (model/embedders.py:121) and
it only ever influences the DIFFUSION pose. What the pipeline exports and
scores is the BOLTZ pose - a free re-prediction, template-free, with no pocket
constraint (tools/boltz/yaml_writer.py), and with the target assembly
re-predicted as well. Measured on 8 designs across 3 runs: the binder lands
7-42 A from where the diffusion placed it, so hotspots the diffusion engaged at
5-15 A are 20-49 A away in the scored pose. Reading epitope engagement off the
Boltz pose alone therefore conflates two independent failures - diffusion
targeting, and diffusion-to-Boltz pose agreement - and reports the second as
though it were the first.

Everything here is CA-CA. That is not a simplification, it is the only honest
choice: the diffusion binder is the `xpb` placeholder and carries N, CA, C and
O only, so a heavy-atom contact count is lower in the designed pose for reasons
that have nothing to do with where the binder sits.

Pure numpy plus Biopython. No Boltz, no GPU, no project state.
"""
import os

import numpy as np

# One parser choice for both modules: which parser reads a .cif vs a .pdb is
# exactly the kind of detail that drifts if it is written down twice.
from pxdbench.metrics.epitope import _parse as _parse_model

#: CA-CA distance below which a hotspot counts as engaged. Wider than
#: epitope.DEFAULT_CONTACT_CUTOFF (4.0 A) on purpose - that one is heavy-atom,
#: and a CA sits roughly a side chain away from the contact it makes.
DEFAULT_CA_CUTOFF = 10.0

#: How much the joint superposition must exceed the worst per-chain one before
#: the target's chains count as rearranged relative to each other.
REARRANGEMENT_MARGIN = 2.0

POSE_TARGETING_KEYS = (
    "pt_required_total",
    "pt_designed_engaged",
    "pt_scored_engaged",
    "pt_designed_hotspots",
    "pt_scored_hotspots",
    "pt_retained",
    "pt_lost",
    "pt_gained",
    "pt_agree",
    "pt_designed_min_ca",
    "pt_scored_min_ca",
)


def _ca_coords(model):
    """{(chain_id, resseq): CA coord} for the standard residues."""
    return {
        (chain.id, residue.id[1]): residue["CA"].coord
        for chain in model
        for residue in chain
        if residue.id[0] == " " and "CA" in residue
    }


def _resolve_chain(present, chain, where):
    """Which spelling of `chain` this structure uses.

    Diffusion CIFs name chains A0/B0/C0 while resolved_hotspots.json records
    A/B. Resolving that is a one-line heuristic, which is why it raises instead
    of guessing when both spellings are there: measuring the wrong chain looks
    exactly like a design that missed.
    """
    exact = chain in present
    suffixed = f"{chain}0" in present
    if exact and suffixed:
        raise ValueError(
            f"chain {chain!r} is ambiguous in {where}: both {chain!r} and "
            f"{chain + '0'!r} are present, so the requested chain cannot be "
            f"resolved without guessing"
        )
    if exact:
        return chain
    if suffixed:
        return f"{chain}0"
    raise ValueError(
        f"chain {chain!r} is not present in {where} "
        f"(present: {sorted(present)})"
    )


def map_required_to_structure(structure_path, required):
    """Rewrite {chain: [res]} into the chain spelling this structure uses."""
    model = _parse_model(structure_path)
    present = {chain.id for chain in model}
    where = os.path.basename(str(structure_path))
    return {
        _resolve_chain(present, chain, where): [int(r) for r in residues]
        for chain, residues in required.items()
    }


def min_ca_distances(structure_path, binder_chain, required):
    """Closest binder CA to each requested hotspot, as {(chain, res): A}.

    Keys come back in the spelling `required` used, not the structure's, so
    results from two poses are directly comparable. Raises if a requested
    residue is absent - a numbering mismatch must not read as "not engaged".
    """
    model = _parse_model(structure_path)
    ca = _ca_coords(model)
    where = os.path.basename(str(structure_path))
    present = {chain for chain, _ in ca}
    # Symmetry: the hotspot chains below go through _resolve_chain, so the
    # binder must too. Diffusion CIFs name it C0 while the policy says C, and
    # comparing literally raised "binder chain 'C' is not present" - which
    # reads exactly like a run that produced no binder, on every designed pose.
    binder_chain = _resolve_chain(present, binder_chain, where)
    binder = np.array([c for (chain, _), c in ca.items() if chain == binder_chain])

    out = {}
    for chain, residues in required.items():
        resolved = _resolve_chain(present, chain, where)
        for res in residues:
            if (resolved, int(res)) not in ca:
                raise ValueError(
                    f"requested hotspot {chain}{res} is not present in {where} "
                    f"(resolved to chain {resolved!r}); a numbering mismatch "
                    f"cannot be distinguished from a design that missed"
                )
            out[(chain, int(res))] = float(
                np.linalg.norm(binder - ca[(resolved, int(res))], axis=1).min()
            )
    return out


def engaged_hotspots(
    structure_path, binder_chain, required, cutoff=DEFAULT_CA_CUTOFF
):
    """Which requested hotspots this pose engages, at a CA-CA cutoff."""
    distances = min_ca_distances(structure_path, binder_chain, required)
    engaged = sorted(k for k, v in distances.items() if v <= cutoff)
    missed = sorted(k for k, v in distances.items() if v > cutoff)
    return {
        "distances": distances,
        "engaged": engaged,
        "missed": missed,
        "n_engaged": len(engaged),
        "n_required": len(distances),
        "min_distance": min(distances.values()) if distances else None,
        "cutoff": cutoff,
    }


def compare_poses(
    designed_path,
    designed_binder,
    scored_path,
    scored_binder,
    required,
    cutoff=DEFAULT_CA_CUTOFF,
):
    """Hotspot engagement in the designed pose vs the scored pose.

    `pt_lost` is the interesting column: hotspots the diffusion engaged that
    the scored pose does not. A run where that list is routinely non-empty has
    a pose-agreement problem, not a targeting problem, and tuning the hotspot
    conditioning will not move it.

    `pt_agree` compares the two engaged SETS, so two poses that both miss
    everything do agree - they agree that nothing was engaged.
    """
    designed = engaged_hotspots(designed_path, designed_binder, required, cutoff)
    scored = engaged_hotspots(scored_path, scored_binder, required, cutoff)
    d_set, s_set = set(designed["engaged"]), set(scored["engaged"])
    return {
        "pt_required_total": designed["n_required"],
        "pt_designed_engaged": designed["n_engaged"],
        "pt_scored_engaged": scored["n_engaged"],
        "pt_designed_hotspots": designed["engaged"],
        "pt_scored_hotspots": scored["engaged"],
        "pt_retained": sorted(d_set & s_set),
        "pt_lost": sorted(d_set - s_set),
        "pt_gained": sorted(s_set - d_set),
        "pt_agree": d_set == s_set,
        "pt_designed_min_ca": designed["min_distance"],
        "pt_scored_min_ca": scored["min_distance"],
        "pt_cutoff": cutoff,
    }


def _kabsch_rmsd(mobile, reference):
    """CA RMSD after optimal superposition of `mobile` onto `reference`."""
    mobile = mobile - mobile.mean(0)
    reference = reference - reference.mean(0)
    u, _, vt = np.linalg.svd(mobile.T @ reference)
    correction = np.diag([1.0, 1.0, float(np.sign(np.linalg.det(u @ vt)))])
    rotation = u @ correction @ vt
    return float(np.sqrt((((mobile @ rotation) - reference) ** 2).sum(1).mean()))


def target_assembly_rmsd(designed_path, scored_path, target_chains):
    """Does the target hold still between the two poses?

    Boltz re-predicts the target as well as the binder. Measured on a real run:
    each target chain superposes well on its own (3.45 A and 1.77 A) while the
    two together do not (15.82 A) - the chains are arranged differently. A
    hotspot patch spanning more than one target chain is then not the same
    surface in the two poses, and an engagement comparison across them is
    measuring two different epitopes.
    """
    designed = _ca_coords(_parse_model(designed_path))
    scored = _ca_coords(_parse_model(scored_path))
    d_present = {chain for chain, _ in designed}
    s_present = {chain for chain, _ in scored}
    d_where = os.path.basename(str(designed_path))
    s_where = os.path.basename(str(scored_path))

    per_chain, joint_d, joint_s = {}, [], []
    for chain in target_chains:
        d_chain = _resolve_chain(d_present, chain, d_where)
        s_chain = _resolve_chain(s_present, chain, s_where)
        common = sorted(
            res
            for (c, res) in designed
            if c == d_chain and (s_chain, res) in scored
        )
        if len(common) < 3:
            continue
        d_xyz = np.array([designed[(d_chain, r)] for r in common])
        s_xyz = np.array([scored[(s_chain, r)] for r in common])
        per_chain[chain] = _kabsch_rmsd(s_xyz, d_xyz)
        joint_d.append(d_xyz)
        joint_s.append(s_xyz)

    if not per_chain:
        return {
            "per_chain": {},
            "joint": None,
            "rearranged": False,
            "n_chains": 0,
        }
    joint = _kabsch_rmsd(np.vstack(joint_s), np.vstack(joint_d))
    return {
        "per_chain": per_chain,
        "joint": joint,
        "margin": joint - max(per_chain.values()),
        "rearranged": (
            len(per_chain) > 1
            and joint > max(per_chain.values()) + REARRANGEMENT_MARGIN
        ),
        "n_chains": len(per_chain),
    }
