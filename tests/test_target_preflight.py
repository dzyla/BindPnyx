"""scripts/preflight_target.py - the target/MSA check that run_campaign.sh runs.

`BoltzBackend.prepare_json` already refuses a chain whose cached MSA would be
discarded, but only after diffusion and MPNN have run. These tests pin that the
same refusal is reachable from the campaign JSON alone, and that the entity
list it is reached through is the one `infer_data_pipeline` really builds -
which is why they run against the committed EGFR shard rather than a dict
shaped to match the code.
"""
import json
import os
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))

import preflight_target as pf  # noqa: E402

PREPARED = os.path.join(REPO, "targets", "egfr_ecd", "prepared", "obj2_wt")
SHARD = os.path.join(PREPARED, "obj2_target_wt.pkl.gz")
#: The campaign that aborted on 2026-10-02, with the K516 shard still in it.
HISTORICAL = os.path.join(REPO, "manifests", "smoke_egfr_his535.json")

needs_shard = pytest.mark.skipif(
    not os.path.isfile(SHARD),
    reason="targets/egfr_ecd/prepared/obj2_wt not built; run scripts/prepare_target.py",
)


def shard_sequences():
    seqs, _ = pf.orig_seqs_from_shard(_sample())
    return [list(e.values())[0]["sequence"] for e in seqs]


def _sample(msa=None, chain_ids=("A", "B"), shard=None, **extra):
    cond = {
        "structure_file": shard or SHARD,
        "filter": {"chain_id": list(chain_ids), "crop": {}},
    }
    if msa:
        cond["msa"] = msa
    sample = {"name": "t", "condition": cond}
    sample.update(extra)
    return sample


def a3m_dir(tmp_path, seq, name, depth=140):
    from pxdbench.tools.boltz.msa_check import a3m_depths

    d = tmp_path / name / "0"
    d.mkdir(parents=True, exist_ok=True)
    out = [f">query\n{seq}\n"]
    for i in range(depth - 1):
        pos = i % len(seq)
        sub = "ACDEFGHIKLMNPQRSTVWY"[(i // len(seq) + 1) % 20]
        if sub == seq[pos]:
            sub = "W" if sub != "W" else "Y"
        out.append(f">hom{i}\n{seq[:pos]}{sub}{seq[pos + 1:]}\n")
    (d / "non_pairing.a3m").write_text("".join(out))
    _, unique = a3m_depths(str(d / "non_pairing.a3m"))
    assert unique >= depth - 2, unique
    return str(d)


def campaign(tmp_path, sample):
    p = tmp_path / "campaign.json"
    p.write_text(json.dumps([sample]))
    return str(p)


# ------------------------------------------------- the entity-list contract

@needs_shard
def test_chains_are_taken_in_filter_order_and_labelled_positionally():
    """`inference.py` names chains by POSITION, so the order of this list is
    the only thing tying a chain to its MSA downstream."""
    seqs, _ = pf.orig_seqs_from_shard(_sample(chain_ids=("B", "A")))
    ids = [list(e.values())[0]["label_asym_id"][0] for e in seqs]
    assert ids == ["A0", "B0"]
    # reversing the filter really reverses the sequences, not just the labels
    forward = [list(e.values())[0]["sequence"]
               for e in pf.orig_seqs_from_shard(_sample())[0]]
    reverse = [list(e.values())[0]["sequence"] for e in seqs]
    assert reverse == forward[::-1]


@needs_shard
def test_the_sequence_comes_from_the_shard_entities():
    import gzip
    import pickle

    with gzip.open(SHARD, "rb") as fh:
        payload = pickle.load(fh)
    assert sorted(shard_sequences()) == sorted(payload["sequences"].values())


@needs_shard
def test_a_chain_that_is_not_in_the_shard_is_refused():
    with pytest.raises(pf.PreflightError, match="not in the shard"):
        pf.orig_seqs_from_shard(_sample(chain_ids=("A", "Z")))


@needs_shard
def test_an_msa_for_an_unselected_chain_is_refused():
    with pytest.raises(pf.PreflightError, match="not a selected chain"):
        pf.orig_seqs_from_shard(
            _sample(msa={"Q": {"precomputed_msa_dir": "/nowhere"}})
        )


@needs_shard
def test_a_hotspot_on_an_unselected_chain_is_refused():
    with pytest.raises(pf.PreflightError, match="not a selected chain"):
        pf.orig_seqs_from_shard(_sample(hotspot={"Q": ["H1"]}))


@needs_shard
def test_an_absent_chain_filter_is_flagged_as_non_reproducible():
    sample = _sample()
    sample["condition"]["filter"].pop("chain_id")
    _, notes = pf.orig_seqs_from_shard(sample)
    assert any("not reproducible" in n for n in notes)


# ----------------------------------------------------------- the real check

@needs_shard
def test_a_matching_pair_passes(tmp_path):
    a, b = shard_sequences()
    sample = _sample(msa={
        "A": {"precomputed_msa_dir": a3m_dir(tmp_path, a, "ma")},
        "B": {"precomputed_msa_dir": a3m_dir(tmp_path, b, "mb")},
    })
    assert pf.main([campaign(tmp_path, sample)]) == 0


@needs_shard
def test_one_substituted_residue_fails_and_names_it(tmp_path, capsys):
    """Exactly the 2026-10-02 failure, reduced: one residue, whole MSA lost."""
    a, b = shard_sequences()
    wrong = b[:4] + ("K" if b[4] != "K" else "N") + b[5:]
    sample = _sample(msa={
        "A": {"precomputed_msa_dir": a3m_dir(tmp_path, a, "ma")},
        "B": {"precomputed_msa_dir": a3m_dir(tmp_path, wrong, "mb")},
    })
    assert pf.main([campaign(tmp_path, sample)]) == 1
    err = capsys.readouterr().err
    assert "target chain B" in err
    assert "pos 4 sequence=N msa=K" in err
    assert "discard all" in err


@needs_shard
def test_a_shallow_msa_fails(tmp_path):
    a, b = shard_sequences()
    sample = _sample(msa={
        "A": {"precomputed_msa_dir": a3m_dir(tmp_path, a, "ma", depth=9)},
        "B": {"precomputed_msa_dir": a3m_dir(tmp_path, b, "mb")},
    })
    assert pf.main([campaign(tmp_path, sample)]) == 1


@needs_shard
def test_a_chain_with_no_msa_at_all_fails(tmp_path):
    """`target_chains_from_orig_seqs` refuses an unpaired chain, and so must
    this: the calibrated configuration never folds a target unconditioned."""
    a, _ = shard_sequences()
    sample = _sample(msa={"A": {"precomputed_msa_dir": a3m_dir(tmp_path, a, "ma")}})
    with pytest.raises(ValueError, match="no MSA on the entity"):
        pf.check_sample(sample, 100, "non_pairing.a3m")


def test_a_structure_that_is_not_a_shard_is_a_skip_not_a_failure(tmp_path, capsys):
    """The pipeline builds a shard from a .pdb at run time, so there is nothing
    to check yet - and rejecting it would reject a supported configuration."""
    sample = _sample(shard=os.path.join(PREPARED, "obj2_target_wt.pdb"))
    assert pf.main([campaign(tmp_path, sample)]) == 0
    assert "SKIP" in capsys.readouterr().err


def test_a_missing_shard_is_a_failure(tmp_path):
    assert pf.main([campaign(tmp_path, _sample(shard="/nowhere/x.pkl.gz"))]) == 1


@needs_shard
def test_the_prepared_egfr_manifest_passes():
    """The deliverable: manifests/egfr_obj2_wt.json is runnable."""
    manifest = os.path.join(REPO, "manifests", "egfr_obj2_wt.json")
    msa = json.load(open(manifest))[0]["condition"]["msa"]
    for entry in msa.values():
        if not os.path.isdir(entry["precomputed_msa_dir"]):
            pytest.skip(f"cached MSA {entry['precomputed_msa_dir']} not on this machine")
    assert pf.main([manifest]) == 0


@pytest.mark.skipif(
    not os.path.isfile(HISTORICAL), reason="historical manifest removed"
)
def test_the_historical_campaign_is_caught_before_the_gpu(capsys):
    """manifests/smoke_egfr_his535.json is the run that died after diffusion."""
    sample = json.load(open(HISTORICAL))[0]
    if not os.path.isfile(sample["condition"]["structure_file"]):
        pytest.skip("the 2026-10-02 shard is not on this machine")
    assert pf.main([HISTORICAL]) == 1
    assert "sequence=K msa=N" in capsys.readouterr().err


# ----------------------------------------------------------------- wiring

def test_run_campaign_sh_runs_the_preflight():
    text = open(os.path.join(REPO, "scripts", "run_campaign.sh")).read()
    assert "scripts/preflight_target.py" in text
    # before the pipeline, and before the ~80 s CUDA/protenix probe: a bad
    # target must not wait for a healthy GPU to be reported
    assert text.index("preflight_target.py") < text.index("pxdesign.runner.pipeline")
    assert text.index("preflight_target.py") < text.index("torch.cuda.is_available")
    # and a non-zero exit really stops the launch
    call = text.index('"$REPO/scripts/preflight_target.py" "$INPUT"')
    assert "exit 1" in text[call:call + 700]


def test_the_preflight_costs_no_torch_import(tmp_path):
    """68 s measured for `import pxdbench.tools.boltz.msa_check`. A guard that
    expensive gets switched off, so this one must not pay it."""
    out = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.path.insert(0, %r); import preflight_target; "
         "print('torch' in sys.modules)"
         % os.path.join(REPO, "scripts")],
        capture_output=True, text=True, timeout=300, cwd=REPO,
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False"
