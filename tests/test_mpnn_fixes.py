"""Three MPNN-path defects from the handoff's frontier list, now fixed.

All three were documented but unfixed, and two of them were described wrongly in
the handoff, which the code disproves.
"""
import os

import pytest

from pxdbench.globals import MPNN_CKPT_PATH
from pxdbench.tools.biopython_utils import get_interface_residue_id, hotspot_residues
from pxdbench.tools.protmpnn.main_mpnn import _binder_residue_names


def _write_pdb(path, chains):
    """chains: {chain_id: [(resseq, resname, (x,y,z)), ...]}"""
    lines, serial = [], 1
    for chain_id, residues in chains.items():
        for resseq, resname, (x, y, z) in residues:
            lines.append(
                f"ATOM  {serial:5d}  CA  {resname:>3s} {chain_id}{resseq:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C"
            )
            serial += 1
    lines.append("END")
    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\n")
    return path


# --------------------------------------------------------------------------- #
# 1. the "soluable" assert typo
# --------------------------------------------------------------------------- #


def test_checkpoint_keys_match_the_assert_list():
    """The handoff says the misspelling silently used vanilla weights. It does
    not: MPNN_CKPT_PATH has no "soluable" key, so the misspelling KeyError'd in
    __init__ while the correct spelling tripped the assert. Both spellings
    failed; the path was unreachable, not quietly wrong.
    """
    assert "soluble" in MPNN_CKPT_PATH
    assert "soluable" not in MPNN_CKPT_PATH


def test_assert_list_now_accepts_the_real_spelling():
    import inspect

    from pxdbench.tools.protmpnn.vanilla_mpnn_predictor import VanillaMPNNPredictor

    src = inspect.getsource(VanillaMPNNPredictor.run_mpnn)
    # check the assert STATEMENT, not the whole source: the fix's comment
    # mentions the old misspelling on purpose
    assert_line = [
        ln for ln in src.splitlines() if ln.strip().startswith("assert model_type")
    ]
    assert len(assert_line) == 1, assert_line
    assert '"soluble"' in assert_line[0]
    assert '"soluable"' not in assert_line[0]
    # every accepted value must be a real checkpoint key
    for key in ("ca", "bb", "soluble"):
        assert key in MPNN_CKPT_PATH, key
    # and the branch that adds --use_soluble_model uses the same spelling
    assert 'model_type == "soluble"' in inspect.getsource(VanillaMPNNPredictor)


# --------------------------------------------------------------------------- #
# 2. hotspot_residues hardcoded chain A as the target
# --------------------------------------------------------------------------- #


def test_hotspot_counts_every_target_chain(tmp_path):
    """With a multi-chain target only chain A's contacts used to count."""
    pdb = _write_pdb(
        str(tmp_path / "c.pdb"),
        {
            "A": [(1, "ALA", (0.0, 0.0, 0.0))],
            "B": [(1, "LEU", (0.0, 0.0, 20.0))],      # far from A, near C res 2
            "C": [(1, "VAL", (2.0, 0.0, 0.0)), (2, "VAL", (2.0, 0.0, 20.0))],
        },
    )
    got = hotspot_residues(pdb, binder_chain="C", atom_distance_cutoff=4.0)
    # residue 1 contacts chain A, residue 2 contacts chain B; both must appear
    assert set(got) == {1, 2}, got


def test_hotspot_defaults_to_all_non_binder_chains(tmp_path):
    pdb = _write_pdb(
        str(tmp_path / "c.pdb"),
        {
            "A": [(1, "ALA", (0.0, 0.0, 0.0))],
            "B": [(1, "LEU", (0.0, 0.0, 20.0))],
            "C": [(1, "VAL", (2.0, 0.0, 20.0))],
        },
    )
    # only chain B is near the binder; the old hardcoded "A" would miss it
    assert set(hotspot_residues(pdb, binder_chain="C")) == {1}


def test_hotspot_rejects_an_unknown_target_chain(tmp_path):
    pdb = _write_pdb(
        str(tmp_path / "c.pdb"),
        {"A": [(1, "ALA", (0.0, 0.0, 0.0))], "C": [(1, "VAL", (2.0, 0.0, 0.0))]},
    )
    with pytest.raises(ValueError, match="not in"):
        hotspot_residues(pdb, binder_chain="C", target_chains=["Z"])


def test_interface_ids_are_chain_qualified(tmp_path):
    ids = get_interface_residue_id({3: "V", 7: "L"}, binder_chain="C")
    assert sorted(ids.split(",")) == ["C3", "C7"]


# --------------------------------------------------------------------------- #
# 3. fix_interface dropped the condition chains and would pin GLY
# --------------------------------------------------------------------------- #


def test_binder_residue_names_detects_the_placeholder(tmp_path):
    placeholder = _write_pdb(
        str(tmp_path / "ph.pdb"),
        {
            "A": [(1, "ALA", (0.0, 0.0, 0.0))],
            "C": [(1, "GLY", (2.0, 0.0, 0.0)), (2, "GLY", (3.0, 0.0, 0.0))],
        },
    )
    assert _binder_residue_names(placeholder, "C") == {"GLY"}

    designed = _write_pdb(
        str(tmp_path / "d.pdb"),
        {
            "A": [(1, "ALA", (0.0, 0.0, 0.0))],
            "C": [(1, "LEU", (2.0, 0.0, 0.0)), (2, "GLY", (3.0, 0.0, 0.0))],
        },
    )
    assert _binder_residue_names(designed, "C") == {"LEU", "GLY"}
    assert _binder_residue_names(designed, "Z") == set()


def test_fix_interface_keeps_the_condition_chains_in_fix_pos():
    """It used to REPLACE fix_pos with the interface ids, leaving the target
    designable so MPNN could rewrite the target's own sequence."""
    import inspect

    from pxdbench.tools.protmpnn import main_mpnn

    src = inspect.getsource(main_mpnn.design_binder)
    assert 'fix_pos = ",".join([fix_pos, interface_ids])' in src
    # and fix_pos is seeded with the condition chains before the branch
    assert src.index('fix_pos = ",".join(cond_chains)') < src.index(
        "if mpnn_cfg.fix_interface:"
    )


def test_fix_interface_refuses_the_glycine_placeholder():
    """Pinning interface positions on an all-GLY placeholder freezes glycine at
    every contact, which is the opposite of holding a designed contact."""
    import inspect

    from pxdbench.tools.protmpnn import main_mpnn

    src = inspect.getsource(main_mpnn.design_binder)
    assert "entirely GLY" in src
    assert "raise ValueError" in src
