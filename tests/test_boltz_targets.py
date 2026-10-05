import json
import os

import pytest

from pxdbench.tools.boltz.targets import (
    chain_id_of,
    msa_path_of,
    target_chains_from_orig_seqs,
)

REAL_INPUT = (
    "/data/private/private_target/shard_L55_c01/global_run_0/obj2_L55/"
    "seed_330153/predictions/ptx_pred/protenix_inputs.json"
)


def _entity(chain, seq, msa_dir=None, use_msa=True):
    inner = {"sequence": seq, "count": 1, "label_asym_id": [chain]}
    if msa_dir is not None:
        inner["use_msa"] = use_msa
        inner["msa"] = {"precomputed_msa_dir": msa_dir}
    return {"proteinChain": inner}


def test_chain_id_strips_the_positional_index():
    """inference.py writes 'A0'; downstream normalizes to 'A'."""
    assert chain_id_of(_entity("A0", "MK")) == "A"
    assert chain_id_of(_entity("B0", "MK")) == "B"
    assert chain_id_of(_entity("A", "MK")) == "A"


def test_msa_path_appends_the_a3m_filename():
    entity = _entity("A0", "MK", msa_dir="/cache/obj2_chainB/0")
    assert msa_path_of(entity) == "/cache/obj2_chainB/0/non_pairing.a3m"


def test_msa_path_is_none_without_an_msa():
    assert msa_path_of(_entity("C0", "GG")) is None


def test_conversion_from_a_list_not_a_dict():
    """orig_seqs is a LIST of entity dicts (inference.py:165-171)."""
    orig_seqs = [
        _entity("A0", "MKTAYIAK", msa_dir="/cache/obj2_chainB/0"),
        _entity("B0", "QRSTVWYC", msa_dir="/cache/obj2_chainD/0"),
    ]
    chains = target_chains_from_orig_seqs(orig_seqs)
    assert [c["id"] for c in chains] == ["A", "B"]
    assert chains[0]["seq"] == "MKTAYIAK"
    assert chains[0]["msa"] == "/cache/obj2_chainB/0/non_pairing.a3m"
    assert chains[1]["msa"] == "/cache/obj2_chainD/0/non_pairing.a3m"


def test_msa_is_paired_to_the_design_chain_not_the_source_chain():
    """The trap: design chains are A and B; the MSA dirs are NAMED after the
    source structure's chains B and D. Pairing by directory name would hand
    chain A the wrong MSA."""
    orig_seqs = [
        _entity("A0", "SEQ_A", msa_dir="/cache/obj2_chainB/0"),
        _entity("B0", "SEQ_B", msa_dir="/cache/obj2_chainD/0"),
    ]
    chains = {c["id"]: c["msa"] for c in target_chains_from_orig_seqs(orig_seqs)}
    assert "obj2_chainB" in chains["A"]
    assert "obj2_chainD" in chains["B"]


def test_explicit_override_wins():
    orig_seqs = [_entity("A0", "SEQ_A", msa_dir="/cache/obj2_chainB/0")]
    chains = target_chains_from_orig_seqs(
        orig_seqs, target_msa_override={"A": "/explicit/a.a3m"}
    )
    assert chains[0]["msa"] == "/explicit/a.a3m"


def test_chain_without_msa_raises():
    """The calibration requires the cached target MSA; silence is not an option."""
    with pytest.raises(ValueError, match="no MSA"):
        target_chains_from_orig_seqs([_entity("A0", "SEQ_A")])


def test_dict_input_is_rejected_with_a_clear_message():
    with pytest.raises(TypeError, match="list of entity dicts"):
        target_chains_from_orig_seqs({"A": "SEQ_A"})


def test_empty_input_raises():
    with pytest.raises(ValueError, match="empty"):
        target_chains_from_orig_seqs([])


def test_against_the_real_pipeline_input():
    """Ground truth from a completed run."""
    if not os.path.exists(REAL_INPUT):
        pytest.skip("real pipeline input not available")
    data = json.load(open(REAL_INPUT))
    record = data[0] if isinstance(data, list) else data
    entities = record["sequences"]
    # the binder is the entity without an MSA; orig_seqs has it popped already
    targets = [e for e in entities if e[list(e)[0]].get("use_msa")]
    chains = target_chains_from_orig_seqs(targets)
    assert [c["id"] for c in chains] == ["A", "B"]
    for chain in chains:
        assert chain["msa"].endswith("non_pairing.a3m")
        assert os.path.exists(chain["msa"]), chain["msa"]
        assert len(chain["seq"]) > 20
    # and the trap holds on real data, not just the synthetic fixture
    by_id = {c["id"]: c["msa"] for c in chains}
    assert "obj2_chainB" in by_id["A"]
    assert "obj2_chainD" in by_id["B"]
