"""Did the design actually bind where it was asked to?

Hotspots enter PXDesign as a SOFT condition: a per-token model feature that
biases generation (model/embedders.py:121). Nothing downstream ever checks that
the predicted complex contacts them, so a design can score well at a different
site and be selected on its interface metrics alone.

This module closes that loop after prediction. It is a transparent filter and an
audit trail, deliberately not a learned conditioning channel - adding a coldspot
input feature would change the model and need its own training and validation.

Pure numpy plus Biopython: no Boltz, no GPU, no project state.
"""
import os

import numpy as np
from Bio.PDB import MMCIFParser, PDBParser

#: heavy-atom distance defining a contact, matching hotspot_residues'
#: atom_distance_cutoff so the two agree
DEFAULT_CONTACT_CUTOFF = 4.0

EPITOPE_KEYS = (
    "ep_required_total",
    "ep_required_contacted",
    "ep_required_frac",
    "ep_forbidden_total",
    "ep_forbidden_contacted",
    "ep_satisfied",
    "ep_contacted_residues",
    "ep_missed_required",
    "ep_violated_forbidden",
    "ep_best_patch_frac",
    "ep_patches_engaged",
    "ep_status",
)


def _parse(structure_path):
    parser = (
        MMCIFParser(QUIET=True)
        if str(structure_path).endswith(".cif")
        else PDBParser(QUIET=True)
    )
    return parser.get_structure("complex", structure_path)[0]


def contact_residues(
    structure_path, binder_chain, target_chains=None, cutoff=DEFAULT_CONTACT_CUTOFF
):
    """Target residues contacted by the binder, as {chain_id: sorted resseq}.

    Heavy-atom distance, hydrogens ignored, so the cutoff means the same thing
    as it does in tools/biopython_utils.py::hotspot_residues.
    """
    model = _parse(structure_path)
    if binder_chain not in model:
        raise ValueError(
            f"binder chain {binder_chain!r} not in {os.path.basename(str(structure_path))}"
        )
    if target_chains is None:
        target_chains = [c.id for c in model if c.id != binder_chain]
    missing = [c for c in target_chains if c not in model]
    if missing:
        raise ValueError(f"target chains {missing} not in the structure")

    binder = np.array(
        [
            a.coord
            for a in model[binder_chain].get_atoms()
            if a.element != "H"
        ]
    )
    if binder.size == 0:
        return {c: [] for c in target_chains}

    out = {}
    for chain_id in target_chains:
        hits = set()
        for residue in model[chain_id]:
            if residue.id[0] != " ":          # skip waters/hetero
                continue
            coords = np.array([a.coord for a in residue if a.element != "H"])
            if coords.size == 0:
                continue
            d = np.linalg.norm(
                coords[:, None, :] - binder[None, :, :], axis=2
            )
            if (d <= cutoff).any():
                hits.add(int(residue.id[1]))
        out[chain_id] = sorted(hits)
    return out


def _flatten(policy):
    """{chain: [resseq]} -> set of (chain, resseq); tolerates None/empty."""
    if not policy:
        return set()
    return {
        (str(chain), int(res))
        for chain, residues in policy.items()
        for res in (residues or [])
    }


def spatial_patches(structure_path, required, separation=15.0):
    """Split a required set into patches no binder could span in one interface.

    A requested epitope is often several residues on one face, but a set spread
    over two faces cannot be contacted at once: measured on the EGFR target, the
    six requested hotspots formed two groups 28-42 A apart while a 55-mer's
    contact footprint is ~20-25 A across. Demanding all of them is then
    unsatisfiable by construction, and reporting 0/6 reads as a design failure
    when it is a specification error.

    Single-linkage on Ca distance; returns a list of {chain: [res]} dicts.
    """
    pairs = sorted(_flatten(required))
    if len(pairs) < 2:
        return [required] if required else []
    model = _parse(structure_path)
    coords = {}
    for chain, res in pairs:
        try:
            residue = model[chain][(" ", int(res), " ")]
        except KeyError:
            continue
        atom = residue["CA"] if "CA" in residue else next(iter(residue), None)
        if atom is not None:
            coords[(chain, res)] = np.asarray(atom.coord, dtype=float)
    keys = [k for k in pairs if k in coords]
    if len(keys) < 2:
        return [required]

    groups = [[k] for k in keys]
    merged = True
    while merged:
        merged = False
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                if any(
                    np.linalg.norm(coords[a] - coords[b]) <= separation
                    for a in groups[i]
                    for b in groups[j]
                ):
                    groups[i] = groups[i] + groups[j]
                    groups.pop(j)
                    merged = True
                    break
            if merged:
                break

    out = []
    for group in groups:
        patch = {}
        for chain, res in sorted(group):
            patch.setdefault(chain, []).append(res)
        out.append(patch)
    return out


def evaluate_epitope_policy(
    structure_path,
    binder_chain,
    required=None,
    forbidden=None,
    target_chains=None,
    cutoff=DEFAULT_CONTACT_CUTOFF,
    patch_separation=None,
):
    """Score one complex against a required/forbidden residue policy.

    `required` and `forbidden` are {chain_id: [residue numbers]} in the DESIGN's
    numbering, i.e. the numbering of the structure being checked. Mapping from
    the source structure's numbering is the caller's job and should be recorded
    in the campaign manifest.

    Returns the EPITOPE_KEYS dict. An empty policy yields ep_satisfied=None, not
    True: nothing was asked for, so nothing was verified, and silently reporting
    a pass would be the failure this module exists to prevent.
    """
    req, forb = _flatten(required), _flatten(forbidden)
    blank = {key: None for key in EPITOPE_KEYS}

    if not req and not forb:
        blank["ep_status"] = "no_policy"
        return blank

    try:
        contacts = contact_residues(
            structure_path, binder_chain, target_chains=target_chains, cutoff=cutoff
        )
    except (ValueError, FileNotFoundError) as exc:
        blank["ep_status"] = "unevaluable"
        blank["ep_contacted_residues"] = ""
        print(f"[WARN] epitope policy not evaluated: {exc}")
        return blank

    contacted = {(chain, res) for chain, hits in contacts.items() for res in hits}
    missed = sorted(req - contacted)
    violated = sorted(forb & contacted)

    def _fmt(pairs):
        return ";".join(f"{c}{r}" for c, r in sorted(pairs))

    n_req_hit = len(req) - len(missed)

    # Per-patch view. ep_satisfied demands EVERY required residue, which is
    # unsatisfiable when the request spans faces a binder cannot bridge, so the
    # best single patch is reported alongside it.
    best_patch_frac, patches_engaged = None, None
    if req and patch_separation:
        patches = spatial_patches(structure_path, required, patch_separation)
        if patches:
            fracs = []
            for patch in patches:
                want = _flatten(patch)
                if want:
                    fracs.append(len(want & contacted) / len(want))
            if fracs:
                best_patch_frac = max(fracs)
                patches_engaged = sum(1 for f in fracs if f > 0)

    return {
        "ep_required_total": len(req),
        "ep_required_contacted": n_req_hit,
        "ep_required_frac": (n_req_hit / len(req)) if req else None,
        "ep_forbidden_total": len(forb),
        "ep_forbidden_contacted": len(violated),
        # satisfied means EVERY required residue contacted and NO forbidden one
        "ep_satisfied": bool(not missed and not violated),
        "ep_contacted_residues": _fmt(contacted),
        "ep_missed_required": _fmt(missed),
        "ep_violated_forbidden": _fmt(violated),
        "ep_best_patch_frac": best_patch_frac,
        "ep_patches_engaged": patches_engaged,
        "ep_status": "ok",
    }
