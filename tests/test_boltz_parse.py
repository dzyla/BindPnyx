import glob
import os

import numpy as np
import pytest

from pxdbench.tools.boltz.parse import read_prediction, score_prediction

REAL = sorted(
    glob.glob(
        "/data/private/external_scorer/bc2_rescore/out_*/work/*/seed1/"
        "boltz_results_c/predictions/c"
    )
)


@pytest.fixture(scope="module")
def real_pred():
    if not REAL:
        pytest.skip("no on-disk Boltz prediction available")
    for d in REAL:
        if os.path.exists(os.path.join(d, "pae_c_model_0.npz")) and os.path.exists(
            os.path.join(d, "c_model_0.pdb")
        ):
            return d
    pytest.skip("no complete Boltz prediction found")


@pytest.fixture(scope="module")
def binder_chain(real_pred):
    """The binder is the LAST chain by this project's convention."""
    from Bio.PDB import PDBParser

    model = PDBParser(QUIET=True).get_structure("c", os.path.join(
        real_pred, "c_model_0.pdb"))[0]
    return [c.id for c in model][-1]


def test_read_prediction_shapes_agree(real_pred, binder_chain):
    """PAE dimension must equal the residue count, or the mapping is wrong."""
    out = read_prediction(real_pred, "c", binder_chain=binder_chain)
    n = len(out["ca"])
    assert out["pae"].shape == (n, n)
    assert out["is_binder"].shape == (n,)
    assert out["plddt_0_100"].shape == (n,)
    assert out["is_binder"].any() and not out["is_binder"].all()


def test_plddt_is_scaled_to_0_100(real_pred, binder_chain):
    out = read_prediction(real_pred, "c", binder_chain=binder_chain)
    assert np.nanmax(out["plddt_0_100"]) > 1.5
    assert np.nanmax(out["plddt_0_100"]) <= 100.0


def test_score_prediction_emits_bz_columns(real_pred, binder_chain):
    scored = score_prediction(real_pred, "c", binder_chain=binder_chain)
    assert scored["bz_status"] == "ok"
    for key in (
        "bz_ipdae",
        "bz_ipsae",
        "bz_pae_interface_min",
        "bz_interface_plddt",
    ):
        assert scored[key] is not None, key
    assert 0.0 <= scored["bz_ipsae"] <= 1.0
    assert 0.0 <= scored["bz_ipdae"] <= 1.0
    assert scored["bz_pae_interface_min"] >= 0.0
    assert 0.0 <= scored["bz_interface_plddt"] <= 100.0


def test_missing_pae_is_a_status_not_an_exception(tmp_path):
    """A failed prediction yields None metrics and a reason, never NaN."""
    scored = score_prediction(str(tmp_path), "c", binder_chain="B")
    assert scored["bz_status"] in ("missing_pae", "missing_pdb")
    assert scored["bz_ipsae"] is None
    assert scored["bz_ipdae"] is None
    assert not any(
        isinstance(v, float) and np.isnan(v) for v in scored.values()
    ), "failures must be None, never NaN"


def test_unknown_binder_chain_is_a_status(real_pred):
    scored = score_prediction(real_pred, "c", binder_chain="Z")
    assert scored["bz_status"] == "binder_chain_absent"
    assert scored["bz_ipsae"] is None


def test_prediction_dir_single_file_layout():
    """Input c.yaml -> boltz_results_c (boltz/main.py:1134 uses the input stem)."""
    from pxdbench.tools.boltz.parse import prediction_dir

    got = prediction_dir("/out/seed_1", "bb1_seq0")
    assert got == "/out/seed_1/boltz_results_bb1_seq0/predictions/bb1_seq0"


def test_prediction_dir_batched_layout():
    """Input is a DIRECTORY named 'input', so the results dir is named after it,
    NOT after each design. Caught only by a real batched run: in the
    single-file case the design name and the file stem coincide.
    """
    from pxdbench.tools.boltz.parse import BATCH_INPUT_STEM, prediction_dir

    assert BATCH_INPUT_STEM == "input"
    got = prediction_dir("/out/seed_1", "bb1_seq0", input_stem=BATCH_INPUT_STEM)
    assert got == "/out/seed_1/boltz_results_input/predictions/bb1_seq0"
    other = prediction_dir("/out/seed_1", "bb1_seq1", input_stem=BATCH_INPUT_STEM)
    assert os.path.dirname(got) == os.path.dirname(other), (
        "batched designs share one results dir"
    )


def test_batched_layout_matches_a_real_boltz_run(tmp_path):
    """Builds the tree boltz actually produced for a 2-design directory input
    and asserts the resolved path lands on the file.
    """
    from pxdbench.tools.boltz.parse import BATCH_INPUT_STEM, prediction_dir

    seed_root = tmp_path / "boltz_pred" / "rank" / "seed_1"
    for name in ("smoke_bb1_seq0", "smoke_bb1_seq1"):
        d = seed_root / f"boltz_results_{BATCH_INPUT_STEM}" / "predictions" / name
        d.mkdir(parents=True)
        (d / f"{name}_model_0.pdb").write_text("ATOM\n")
    for name in ("smoke_bb1_seq0", "smoke_bb1_seq1"):
        resolved = prediction_dir(
            str(seed_root), name, input_stem=BATCH_INPUT_STEM
        )
        assert os.path.isfile(os.path.join(resolved, f"{name}_model_0.pdb"))


def test_scores_agree_with_common_scorer_on_the_real_prediction(
    real_pred, binder_chain
):
    """The parser's ipSAE must equal common_scorer's on the same structure.

    This is the end-to-end check that the PAE-to-residue mapping is right: a
    transposed or misaligned matrix would still produce a plausible number.
    """
    import sys

    from pxdbench.metrics.interface import _common_scorer

    common_scorer = _common_scorer()

    data = read_prediction(real_pred, "c", binder_chain=binder_chain)
    ref = common_scorer.score_complex(data["pae"], data["ca"], data["is_binder"])
    scored = score_prediction(real_pred, "c", binder_chain=binder_chain)
    assert scored["bz_ipsae"] == pytest.approx(ref["ipsae_min"], abs=1e-9)
    assert scored["bz_pae_interface_min"] == pytest.approx(
        ref["pae_interface_min"], abs=1e-9
    )
