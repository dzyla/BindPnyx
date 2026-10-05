"""Hotspot engagement in the DESIGNED pose vs the SCORED pose.

Why this module exists. Hotspot conditioning only ever influences the diffusion
pose. The epitope numbers were being read off the Boltz pose, which is a free
re-prediction: template-free, no pocket constraint, and the target assembly
re-predicted too. Measured on 8 designs across 3 runs, the binder lands 7-42 A
from where the diffusion put it. So "the design did not engage its hotspots"
was conflating two different failures.

Comparing the two poses needs a backbone-only metric. The diffusion binder is
the `xpb` placeholder and carries N, CA, C, O and nothing else, so a heavy-atom
contact count is systematically lower in the designed pose for reasons that
have nothing to do with where the binder sits. CA-CA distance is the honest
primitive: it means the same thing in both poses.
"""
import pytest

from pxdbench.metrics.pose_targeting import (
    DEFAULT_CA_CUTOFF,
    compare_poses,
    engaged_hotspots,
    map_required_to_structure,
    min_ca_distances,
    target_assembly_rmsd,
)


def write_pdb(path, atoms):
    """atoms: [(chain, resseq, resname, atomname, (x, y, z)), ...]"""
    with open(path, "w") as handle:
        for i, (chain, resseq, resname, name, (x, y, z)) in enumerate(atoms, 1):
            # columns matter: name 13-16, altLoc 17, resName 18-20,
            # chainID 22, resSeq 23-26, coords from 31
            handle.write(
                f"ATOM  {i:5d} {name:<4s} {resname:>3s} {chain:1s}{resseq:4d}"
                f"    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00"
                f"          {name[0]:>2s}\n"
            )
        handle.write("END\n")
    return str(path)


CIF_HEADER = """data_test
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.auth_seq_id
_atom_site.auth_asym_id
_atom_site.pdbx_PDB_model_num
"""


def write_cif(path, atoms):
    """Multi-character chain ids need mmCIF; PDB has one column for them.

    The diffusion poses are CIFs with chains named A0/B0/C0, so the chain
    resolution this exercises is only reachable through this format.
    """
    with open(path, "w") as handle:
        handle.write(CIF_HEADER)
        for i, (chain, resseq, resname, name, (x, y, z)) in enumerate(atoms, 1):
            handle.write(
                f"ATOM {i} {name[0]} {name} . {resname} {chain} {resseq} ? "
                f"{x:.3f} {y:.3f} {z:.3f} 1.00 0.00 {resseq} {chain} 1\n"
            )
    return str(path)


def backbone(chain, resseq, resname, origin):
    """The four atoms the xpb placeholder actually has."""
    x, y, z = origin
    return [
        (chain, resseq, resname, "N", (x - 1.0, y, z)),
        (chain, resseq, resname, "CA", (x, y, z)),
        (chain, resseq, resname, "C", (x + 1.0, y, z)),
        (chain, resseq, resname, "O", (x + 1.5, y, z)),
    ]


@pytest.fixture
def two_chain(tmp_path):
    """Target chain B at x=0, binder chain C at x=12 - so 12 A CA-CA."""
    atoms = []
    atoms += backbone("B", 24, "HIS", (0.0, 0.0, 0.0))
    atoms += backbone("B", 42, "ASP", (0.0, 20.0, 0.0))
    atoms += backbone("C", 1, "xpb", (12.0, 0.0, 0.0))
    return write_pdb(tmp_path / "pose.pdb", atoms)


REQUIRED = {"B": [24, 42]}


def test_min_ca_distance_is_to_the_closest_binder_ca(two_chain):
    got = min_ca_distances(two_chain, "C", REQUIRED)
    assert got[("B", 24)] == pytest.approx(12.0, abs=1e-3)
    assert got[("B", 42)] == pytest.approx((12.0**2 + 20.0**2) ** 0.5, abs=1e-3)


def test_ca_distance_is_unchanged_by_side_chains(tmp_path, two_chain):
    """The bias this module exists to remove.

    Same CA positions, but the binder now carries a side chain reaching toward
    the target. A heavy-atom contact count would change; the CA answer must not,
    because the binder has not moved.
    """
    atoms = []
    atoms += backbone("B", 24, "HIS", (0.0, 0.0, 0.0))
    atoms += backbone("B", 42, "ASP", (0.0, 20.0, 0.0))
    atoms += backbone("C", 1, "ARG", (12.0, 0.0, 0.0))
    atoms += [("C", 1, "ARG", "CB", (8.0, 0.0, 0.0))]  # reaches the target
    with_sidechain = write_pdb(tmp_path / "sc.pdb", atoms)

    assert min_ca_distances(with_sidechain, "C", REQUIRED) == pytest.approx(
        min_ca_distances(two_chain, "C", REQUIRED)
    )


def test_requested_residue_absent_from_the_structure_raises(two_chain):
    """A numbering mismatch must not read as "not engaged"."""
    with pytest.raises(ValueError, match="not present"):
        min_ca_distances(two_chain, "C", {"B": [24, 999]})


def test_requested_chain_absent_from_the_structure_raises(two_chain):
    with pytest.raises(ValueError, match="not present"):
        min_ca_distances(two_chain, "C", {"Z": [1]})


def test_engaged_hotspots_splits_on_the_cutoff(two_chain):
    got = engaged_hotspots(two_chain, "C", REQUIRED, cutoff=15.0)
    assert got["engaged"] == [("B", 24)]
    assert got["missed"] == [("B", 42)]
    assert got["n_engaged"] == 1
    assert got["n_required"] == 2


def test_default_cutoff_is_a_ca_cutoff_not_a_heavy_atom_one():
    """4 A is the heavy-atom contact cutoff; CA-CA needs a wider one."""
    assert DEFAULT_CA_CUTOFF > 4.0


def test_compare_poses_reports_hotspots_lost_between_the_poses(tmp_path, two_chain):
    """The designed pose engages B24; the scored pose has moved 40 A away."""
    atoms = []
    atoms += backbone("B", 24, "HIS", (0.0, 0.0, 0.0))
    atoms += backbone("B", 42, "ASP", (0.0, 20.0, 0.0))
    atoms += backbone("C", 1, "ARG", (52.0, 0.0, 0.0))
    scored = write_pdb(tmp_path / "scored.pdb", atoms)

    got = compare_poses(two_chain, "C", scored, "C", REQUIRED, cutoff=15.0)
    assert got["pt_designed_engaged"] == 1
    assert got["pt_scored_engaged"] == 0
    assert got["pt_lost"] == [("B", 24)]
    assert got["pt_gained"] == []
    assert got["pt_retained"] == []
    assert got["pt_agree"] is False


def test_compare_poses_agrees_when_both_engage_the_same_hotspot(two_chain):
    got = compare_poses(two_chain, "C", two_chain, "C", REQUIRED, cutoff=15.0)
    assert got["pt_retained"] == [("B", 24)]
    assert got["pt_agree"] is True


def test_target_assembly_rearrangement_is_reported(tmp_path):
    """Each target chain superposes, but their arrangement differs.

    Measured on a real run: chain A 3.45 A, chain B 1.77 A, A+B jointly
    15.82 A. A composite epitope spanning both chains is then not the same
    surface in the two poses, and that must be visible rather than inferred.
    """
    def assembly(b_y):
        return (
            backbone("A", 1, "ALA", (0.0, 0.0, 0.0))
            + backbone("A", 2, "ALA", (4.0, 0.0, 0.0))
            + backbone("A", 3, "ALA", (8.0, 0.0, 0.0))
            + backbone("B", 1, "ALA", (0.0, b_y, 0.0))
            + backbone("B", 2, "ALA", (4.0, b_y, 0.0))
            + backbone("B", 3, "ALA", (8.0, b_y, 0.0))
        )

    designed = write_pdb(tmp_path / "d.pdb", assembly(20.0))
    # chain B rigidly displaced by 15 A; each chain is internally identical
    scored = write_pdb(tmp_path / "s.pdb", assembly(35.0))
    got = target_assembly_rmsd(designed, scored, ["A", "B"])
    assert got["per_chain"]["A"] == pytest.approx(0.0, abs=1e-6)
    assert got["per_chain"]["B"] == pytest.approx(0.0, abs=1e-6)
    assert got["joint"] > 5.0
    assert got["rearranged"] is True


def test_single_chain_target_is_never_flagged_as_rearranged(tmp_path):
    """With one target chain there is no relative arrangement to change."""
    designed = write_pdb(
        tmp_path / "d1.pdb",
        backbone("B", 1, "ALA", (0.0, 0.0, 0.0))
        + backbone("B", 2, "ALA", (4.0, 0.0, 0.0))
        + backbone("B", 3, "ALA", (8.0, 0.0, 0.0)),
    )
    got = target_assembly_rmsd(designed, designed, ["B"])
    assert got["rearranged"] is False


def test_map_required_to_structure_resolves_the_pxdesign_chain_suffix(tmp_path):
    """Diffusion CIFs name chains A0/B0/C0; resolved_hotspots.json says A/B."""
    pose = write_cif(
        tmp_path / "suffixed.cif",
        backbone("B0", 24, "HIS", (0.0, 0.0, 0.0)),
    )
    assert map_required_to_structure(pose, {"B": [24]}) == {"B0": [24]}


def test_map_required_to_structure_prefers_an_exact_chain_match(two_chain):
    assert map_required_to_structure(two_chain, REQUIRED) == {"B": [24, 42]}


def test_map_required_to_structure_raises_when_both_spellings_exist(tmp_path):
    """Guessing between B and B0 would silently measure the wrong chain."""
    pose = write_cif(
        tmp_path / "ambig.cif",
        backbone("B", 24, "HIS", (0.0, 0.0, 0.0))
        + backbone("B0", 24, "HIS", (0.0, 50.0, 0.0)),
    )
    with pytest.raises(ValueError, match="ambiguous"):
        map_required_to_structure(pose, {"B": [24]})
