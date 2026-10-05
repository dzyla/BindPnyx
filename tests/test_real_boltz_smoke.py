"""One real Boltz fold, end to end. Opt-in, because it needs a GPU.

CLAUDE.md s7 open question 4: "No test runs real Boltz. The suite stubs the
fold seam; correctness rests on the integration scripts." Today showed the
price - `run_campaign.sh` discarded every argument it was given for weeks while
600 stubbed tests stayed green, and the bug was only visible once something
actually tried to fold.

This drives the real seam: prepare_json validates a real cached MSA, writes
real YAMLs, and _fold_batch invokes the real boltz binary, which writes real
artifacts that parse.py then scores. Nothing here is stubbed.

Run it deliberately:

    PXD_RUN_REAL_BOLTZ=1 PYTHONPATH=$(pwd) \
        .pxd/envs/pxd/bin/python -m pytest tests/test_real_boltz_smoke.py -q

It is skipped otherwise, so the default suite stays fast and CPU-only. A skip
is not a pass: if this never runs, open question 4 is still open.
"""
import json
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))

pytestmark = pytest.mark.skipif(
    not os.environ.get("PXD_RUN_REAL_BOLTZ"),
    reason="needs a GPU and a real boltz; set PXD_RUN_REAL_BOLTZ=1",
)

# The smallest real target in the repo with a verified MSA: 183 aa over two
# chains, ~5.3 s per prediction.
TARGET = os.path.join(REPO, "panel", "target_obj2.json")


@pytest.fixture(scope="module")
def folded(tmp_path_factory):
    """One design, one seed, through the real backend. Returns the scored row."""
    from target_spec import load_orig_seqs
    from design_roster import load_design_roster
    from pxdbench.tools.boltz.backend import BoltzBackend

    if not os.path.isfile(TARGET):
        pytest.skip(f"{TARGET} not present")
    _, orig_seqs, _ = load_orig_seqs(TARGET)
    pool = load_design_roster(os.path.join(REPO, "panel", "pool54.csv"))
    design = pool.iloc[0]

    out = tmp_path_factory.mktemp("smoke")
    backend = BoltzBackend(cfg={
        "boltz_bin": os.environ.get("PXD_BOLTZ_BIN"),
        "seeds_rank": 1, "seeds_gate": 1, "workers": 1,
        "keep_structures": "all",
    }, device="cuda")
    data = [{"name": design.uid, "seq_idx": 0, "sequence": design.sequence}]
    manifest = backend.prepare_json("", data, dump_dir=str(out),
                                    orig_seqs=orig_seqs)
    scored = backend.score_flat(backend._entries, str(out), 1,
                                batch_tag="smoke")
    return {"out": out, "manifest": manifest, "scored": scored,
            "uid": design.uid}


def test_the_fold_produced_a_scoreable_prediction(folded):
    """The end-to-end assertion: a real design, really folded, really scored."""
    row = folded["scored"][f"{folded['uid']}_seq0"]
    assert row["bz_status"] == "ok", (
        f"status {row['bz_status']!r} - the fold did not produce a scoreable "
        f"prediction"
    )
    assert row["bz_ipsae"] is not None
    assert 0.0 <= row["bz_ipsae"] <= 1.0, row["bz_ipsae"]


def test_the_structure_path_points_at_a_real_file(folded):
    """`State=COMPLETED` is not success: assert the artifact exists."""
    row = folded["scored"][f"{folded['uid']}_seq0"]
    assert row["bz_struct_path"], "no structure path recorded"
    assert os.path.isfile(row["bz_struct_path"])
    assert os.path.getsize(row["bz_struct_path"]) > 0


def test_the_msa_was_validated_and_recorded(folded):
    """prepare_json writes provenance before folding; without it a discarded
    MSA leaves no trace."""
    rec_path = os.path.join(str(folded["out"]), "msa_validation.json")
    assert os.path.isfile(rec_path)
    recs = json.load(open(rec_path))
    assert len(recs) == 2, "obj2 is two chains"
    for r in recs:
        assert r["match"] == "exact match"
        assert r["a3m_sha256"] and r["sequence_sha256"]


def test_boltz_did_not_silently_discard_the_msa(folded):
    """The failure this whole branch exists to catch. If boltz had swapped in
    a dummy, every row would be msa_discarded instead of ok."""
    row = folded["scored"][f"{folded['uid']}_seq0"]
    assert row["bz_status"] != "msa_discarded"


def test_the_manifest_round_trips(folded):
    entries = json.load(open(folded["manifest"]))["designs"]
    assert len(entries) == 1
    assert entries[0]["sample_name"] == f"{folded['uid']}_seq0"
    assert os.path.isfile(entries[0]["yaml"])
