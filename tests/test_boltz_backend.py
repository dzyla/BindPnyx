import inspect
import json
import os
import stat

import numpy as np
import pytest

from pxdbench.tools.boltz.backend import BoltzBackend


def _orig_seqs(tmp_path):
    """Real shape: a LIST of entity dicts with per-chain MSA directories."""
    chains = []
    for chain, seq, src in (
        ("A0", "MKTAYIAKQR", "chainB"),
        ("B0", "QRSTVWYCDE", "chainD"),
    ):
        msa_dir = tmp_path / f"obj2_{src}" / "0"
        msa_dir.mkdir(parents=True, exist_ok=True)
        # A real MSA: the query IS the chain's sequence, and the depth is one
        # boltz would not treat as a dummy. ">q\nMK\n" described an alignment
        # boltz discards, which is the bug this fixture must not encode.
        records = [f">query\n{seq}\n"]
        for i in range(119):
            records.append(f">hom{i}\n{seq[:-1]}{'ACDEFGHIKLMNPQRSTVWY'[i % 20]}\n")
        (msa_dir / "non_pairing.a3m").write_text("".join(records))
        chains.append(
            {
                "proteinChain": {
                    "sequence": seq,
                    "count": 1,
                    "label_asym_id": [chain],
                    "use_msa": True,
                    "msa": {"precomputed_msa_dir": str(msa_dir)},
                }
            }
        )
    return chains


def _cfg(tmp_path, **over):
    cfg = {
        "boltz_bin": "/bin/true",
        "seeds_rank": 1,
        "seeds_gate": 3,
        "ranking_key": "bz_ipsae",
        "compare_ranking_keys": False,
        "keep_structures": "winners",
        # The fixture's MSA is small by design; identity is what these tests
        # exercise. Production manifests keep the 100 default.
        "min_msa_depth": 5,
    }
    cfg.update(over)
    return cfg


def _data_list():
    return [
        {"name": "bb1", "seq_idx": 0, "sequence": "AAAA"},
        {"name": "bb1", "seq_idx": 1, "sequence": "CCCC"},
        {"name": "bb2", "seq_idx": 0, "sequence": "DDDD"},
        {"name": "bb2", "seq_idx": 1, "sequence": "EEEE"},
    ]


def _fake_boltz(tmp_path):
    """A boltz that exits 0 and writes nothing. Enough to exercise staging."""
    p = tmp_path / "fake_boltz"
    p.write_text("#!/bin/sh\nexit 0\n")
    p.chmod(p.stat().st_mode | stat.S_IEXEC)
    return p


class BatchSpy:
    """Replaces the single Boltz-launching seam.

    Records ONE call per (seed, tag, set-of-designs), which is what lets the
    tests assert that folding is batched and that passes are isolated.
    """

    def __init__(self, scores):
        self.scores = scores
        self.batches = []

    def fold(self, entries, seed, dump_dir, tag="fold"):
        names = [e["sample_name"] for e in entries]
        self.batches.append(
            {"seed": seed, "names": tuple(names), "tag": tag, "dump": dump_dir}
        )
        return {n: dict(self.scores[n], bz_status="ok") for n in names}


SCORES = {
    # bb1: seq1 wins on ipsae, seq0 wins on ipdae  -> the keys DISAGREE
    "bb1_seq0": {"bz_ipsae": 0.40, "bz_ipdae": 0.90, "bz_pae_interface_min": 1.0,
                 "bz_interface_plddt": 90.0, "bz_iptm": 0.7, "bz_ptm": 0.8,
                 "bz_complex_plddt": 0.9},
    "bb1_seq1": {"bz_ipsae": 0.70, "bz_ipdae": 0.50, "bz_pae_interface_min": 1.2,
                 "bz_interface_plddt": 88.0, "bz_iptm": 0.7, "bz_ptm": 0.8,
                 "bz_complex_plddt": 0.9},
    # bb2: both keys agree on seq0
    "bb2_seq0": {"bz_ipsae": 0.80, "bz_ipdae": 0.80, "bz_pae_interface_min": 0.9,
                 "bz_interface_plddt": 95.0, "bz_iptm": 0.7, "bz_ptm": 0.8,
                 "bz_complex_plddt": 0.9},
    "bb2_seq1": {"bz_ipsae": 0.30, "bz_ipdae": 0.20, "bz_pae_interface_min": 5.0,
                 "bz_interface_plddt": 60.0, "bz_iptm": 0.7, "bz_ptm": 0.8,
                 "bz_complex_plddt": 0.9},
}


def _backend(tmp_path, **over):
    backend = BoltzBackend(cfg=_cfg(tmp_path, **over), device="cpu")
    backend._spy = BatchSpy(SCORES)
    backend._fold_batch = backend._spy.fold
    return backend


def _run(backend, tmp_path, data_list, sub="dump", **predict_kw):
    dump = str(tmp_path / sub)
    manifest = backend.prepare_json(
        str(tmp_path), data_list, dump_dir=dump, orig_seqs=_orig_seqs(tmp_path)
    )
    out = backend.predict(
        input_json_path=manifest, data_list=data_list, dump_dir=dump, **predict_kw
    )
    return out, dump


def _confirmed(data_list):
    return {
        f"{i['name']}_seq{i['seq_idx']}" for i in data_list if i["bz_n_seeds"] == 3
    }


def test_predict_mutates_data_list(tmp_path):
    """The in-place mutation IS the contract; the return value is discarded."""
    backend = _backend(tmp_path)
    data_list = _data_list()
    _run(backend, tmp_path, data_list)
    for item in data_list:
        assert item["bz_status"] == "ok"
        assert item["bz_ipsae"] is not None
        assert item["bz_n_seeds"] in (1, 3)
        assert "bz_batch_id" in item


def test_predict_returns_name_keyed_dict(tmp_path):
    backend = _backend(tmp_path)
    data_list = _data_list()
    out, _dump = _run(backend, tmp_path, data_list)
    assert isinstance(out, dict)
    assert set(out) == {"bb1_seq0", "bb1_seq1", "bb2_seq0", "bb2_seq1"}


def test_folding_is_batched_one_call_per_seed(tmp_path):
    """The whole point of Task 5. Per-design folding reintroduces the model
    load that batching removes."""
    backend = _backend(tmp_path)
    data_list = _data_list()
    _run(backend, tmp_path, data_list)
    batches = backend._spy.batches
    # 1 rank seed + 3 gate seeds = 4 invocations total, regardless of 4 designs
    assert len(batches) == 4, batches
    rank_batches = [b for b in batches if len(b["names"]) == 4]
    assert len(rank_batches) == 1
    assert rank_batches[0]["seed"] == 1
    gate_batches = [b for b in batches if len(b["names"]) < 4]
    assert sorted(b["seed"] for b in gate_batches) == [1, 2, 3]
    assert all(len(b["names"]) == 2 for b in gate_batches), "2 winners, 2 backbones"


def test_rank_and_gate_passes_use_distinct_staging(tmp_path):
    """Both passes use seed 1. Sharing a directory would make the gate pass
    refold every rank-only design, since boltz folds EVERY yaml in the dir."""
    backend = _backend(tmp_path)
    data_list = _data_list()
    _run(backend, tmp_path, data_list)
    tags = {b["tag"] for b in backend._spy.batches}
    assert len(tags) == 2, f"rank and gate must not share a tag: {tags}"
    seed1 = [b for b in backend._spy.batches if b["seed"] == 1]
    assert len({b["tag"] for b in seed1}) == 2
    assert {len(b["names"]) for b in seed1} == {4, 2}


def test_real_fold_stages_only_its_own_designs(tmp_path):
    """Exercises the REAL _fold_batch with a fake boltz. A stubbed fold cannot
    catch a staging-directory leak, which is how this bug survived review."""
    probe = tmp_path / "fake_boltz"
    probe.write_text("#!/bin/sh\nexit 0\n")
    probe.chmod(probe.stat().st_mode | stat.S_IEXEC)

    backend = BoltzBackend(cfg=_cfg(tmp_path, boltz_bin=str(probe)), device="cpu")
    data_list = _data_list()
    dump = str(tmp_path / "realfold")
    manifest = backend.prepare_json(
        str(tmp_path), data_list, dump_dir=dump, orig_seqs=_orig_seqs(tmp_path)
    )
    entries = json.load(open(manifest))["designs"]
    backend.batch_id = "probe"
    # stage the full set under the rank tag, then a subset under the gate tag
    backend._fold_batch(entries, 1, dump, tag="rank")
    backend._fold_batch(entries[:2], 1, dump, tag="gate")
    gate_dir = os.path.join(dump, "boltz_pred", "gate", "seed_1", "input")
    staged = sorted(s for s in os.listdir(gate_dir) if s.endswith(".yaml"))
    assert staged == ["bb1_seq0.yaml", "bb1_seq1.yaml"], staged
    rank_dir = os.path.join(dump, "boltz_pred", "rank", "seed_1", "input")
    assert len([s for s in os.listdir(rank_dir) if s.endswith(".yaml")]) == 4


def test_real_fold_out_dir_is_not_the_yaml_dir(tmp_path):
    """check_inputs raises on a subdirectory inside the input dir."""
    probe = tmp_path / "fake_boltz"
    probe.write_text("#!/bin/sh\nexit 0\n")
    probe.chmod(probe.stat().st_mode | stat.S_IEXEC)
    backend = BoltzBackend(cfg=_cfg(tmp_path, boltz_bin=str(probe)), device="cpu")
    data_list = _data_list()
    dump = str(tmp_path / "layout")
    manifest = backend.prepare_json(
        str(tmp_path), data_list, dump_dir=dump, orig_seqs=_orig_seqs(tmp_path)
    )
    entries = json.load(open(manifest))["designs"]
    backend.batch_id = "probe"
    backend._fold_batch(entries, 1, dump, tag="rank")
    yaml_dir = os.path.join(dump, "boltz_pred", "rank", "seed_1", "input")
    # nothing but .yaml files may live in the input directory
    assert all(f.endswith(".yaml") for f in os.listdir(yaml_dir))
    assert not any(
        os.path.isdir(os.path.join(yaml_dir, f)) for f in os.listdir(yaml_dir)
    )


def test_batch_count_does_not_grow_with_design_count(tmp_path):
    """Doubling the designs must not double the Boltz invocations."""
    backend = _backend(tmp_path)
    data_list = _data_list()
    _run(backend, tmp_path, data_list)
    few = len(backend._spy.batches)

    wide = BoltzBackend(cfg=_cfg(tmp_path), device="cpu")
    scores = dict(SCORES)
    extra = []
    for i in range(8):
        name = f"bb3_seq{i}"
        scores[name] = dict(SCORES["bb2_seq1"])
        extra.append({"name": "bb3", "seq_idx": i, "sequence": "FFFF"})
    wide._spy = BatchSpy(scores)
    wide._fold_batch = wide._spy.fold
    _run(wide, tmp_path, _data_list() + extra, sub="wide")
    assert len(wide._spy.batches) == few, "invocation count must be seed-bound"


def test_tier1_winner_uses_configured_ranking_key(tmp_path):
    """Pins the selection step against the declared policy."""
    by_ipsae = _backend(tmp_path, ranking_key="bz_ipsae")
    data_list = _data_list()
    _run(by_ipsae, tmp_path, data_list, sub="d1")
    assert _confirmed(data_list) == {"bb1_seq1", "bb2_seq0"}

    by_ipdae = _backend(tmp_path, ranking_key="bz_ipdae")
    data_list2 = _data_list()
    _run(by_ipdae, tmp_path, data_list2, sub="d2")
    assert _confirmed(data_list2) == {"bb1_seq0", "bb2_seq0"}


def test_compare_mode_confirms_union(tmp_path):
    """Both arms must be confirmed at equal seed depth, or the comparison
    measures seed depth instead of metric quality."""
    backend = _backend(tmp_path, compare_ranking_keys=True)
    data_list = _data_list()
    _run(backend, tmp_path, data_list)
    assert _confirmed(data_list) == {"bb1_seq0", "bb1_seq1", "bb2_seq0"}


def test_score_flat_scores_everything_at_one_depth(tmp_path):
    """Task 11's seam: no winner selection, every entry at `seeds`."""
    backend = _backend(tmp_path)
    entries = [
        {"sample_name": n, "yaml": f"/w/{n}.yaml", "binder_id": "C"}
        for n in ("bb1_seq0", "bb1_seq1", "bb2_seq0", "bb2_seq1")
    ]
    out = backend.score_flat(entries, str(tmp_path / "flat"), seeds=3)
    assert set(out) == {e["sample_name"] for e in entries}
    assert all(v["bz_n_seeds"] == 3 for v in out.values())
    assert len(backend._spy.batches) == 3, "one invocation per seed"
    assert all(len(b["names"]) == 4 for b in backend._spy.batches)
    # score_flat must stamp its OWN batch id, or the final rank refuses the rows
    ids = {v["bz_final_batch_id"] for v in out.values()}
    assert len(ids) == 1 and None not in ids


def test_rank_and_gate_get_distinct_final_batch_ids(tmp_path):
    """They are different co-fold batches; one id would let triage sort a
    1-seed score against a 3-seed one."""
    backend = _backend(tmp_path)
    data_list = _data_list()
    _run(backend, tmp_path, data_list)
    by_seeds = {}
    for item in data_list:
        by_seeds.setdefault(item["bz_n_seeds"], set()).add(item["bz_final_batch_id"])
    assert len(by_seeds[1]) == 1 and len(by_seeds[3]) == 1
    assert by_seeds[1] != by_seeds[3], "rank and gate must differ"


def test_prepare_json_accepts_the_real_orig_seqs_list(tmp_path):
    """A chain-keyed dict must be rejected loudly, not silently mishandled."""
    backend = _backend(tmp_path)
    data_list = _data_list()
    with pytest.raises(TypeError, match="list of entity dicts"):
        backend.prepare_json(
            str(tmp_path), data_list, dump_dir=str(tmp_path / "bad"),
            orig_seqs={"B": "MKTAYIAK", "D": "QRSTVWYC"},
        )


def test_yaml_target_chains_come_from_orig_seqs(tmp_path):
    backend = _backend(tmp_path)
    data_list = _data_list()
    _out, dump = _run(backend, tmp_path, data_list)
    yaml_text = open(os.path.join(dump, "yamls", "bb1_seq0.yaml")).read()
    assert "id: A" in yaml_text and "id: B" in yaml_text
    assert "obj2_chainB/0/non_pairing.a3m" in yaml_text
    assert "obj2_chainD/0/non_pairing.a3m" in yaml_text
    assert "msa: empty" in yaml_text


def test_ignores_protenix_diffusion_parameters(tmp_path):
    """N_sample/N_step/step_scale_eta/gamma0 have no Boltz equivalent.
    Accepting and ignoring them is a decision, not an oversight."""
    backend = _backend(tmp_path)
    data_list = _data_list()
    _run(
        backend, tmp_path, data_list,
        N_sample=7, N_step=99, step_scale_eta=3.3, gamma0=0.5, N_cycle=11,
    )
    seeds = sorted({b["seed"] for b in backend._spy.batches})
    assert seeds == [1, 2, 3], "seed depth comes from cfg, not N_sample"


def test_emits_no_ptx_alias_columns(tmp_path):
    backend = _backend(tmp_path)
    data_list = _data_list()
    _run(backend, tmp_path, data_list)
    for item in data_list:
        assert not [k for k in item if k.startswith("ptx")]


def test_failed_seed_makes_the_design_unusable(tmp_path):
    """A design is only as good as its worst seed; None, never NaN."""
    backend = _backend(tmp_path)
    original = backend._spy.fold

    def flaky(entries, seed, dump_dir, tag="fold"):
        out = original(entries, seed, dump_dir, tag=tag)
        if seed == 2 and "bb2_seq0" in out:
            out["bb2_seq0"] = {"bz_status": "boltz_failed"}
        return out

    backend._fold_batch = flaky
    data_list = _data_list()
    _run(backend, tmp_path, data_list)
    row = [i for i in data_list if i["name"] == "bb2" and i["seq_idx"] == 0][0]
    assert row["bz_status"] == "boltz_failed"
    assert row["bz_ipsae"] is None
    assert not any(
        isinstance(v, float) and np.isnan(v) for v in row.values()
    ), "failures must be None, never NaN"


def test_calibration_record_is_written(tmp_path):
    backend = _backend(tmp_path)
    data_list = _data_list()
    _out, dump = _run(backend, tmp_path, data_list)
    record = json.load(open(os.path.join(dump, "bz_calibration.json")))
    assert record["gate"]["bz_ipsae"] == [">=", 0.5]
    assert record["seeds_gate"] == 3
    assert record["aggregation"] == "mean"
    assert record["ranking_key"] == "bz_ipsae"
    assert record["gate_is_provisional"] is True
    assert "batch_id" in record
    assert record["boltz_version_expected"] == "2.2.1"
    # the fake boltz has no interpreter beside it: not measured, and says so
    assert record["boltz_version_measured"] is None
    # MSA provenance comes from the entities, keyed by DESIGN chain
    assert set(record["target_msa"]) == {"A", "B"}
    assert "obj2_chainB" in record["target_msa"]["A"]["path"]
    assert "obj2_chainD" in record["target_msa"]["B"]["path"]
    # headers vs what boltz actually loads after dedup (see msa_validation.json)
    assert record["target_msa"]["A"]["depth_headers"] == 120
    assert record["target_msa"]["A"]["depth_unique"] == 20
    assert "n_seqs" not in record["target_msa"]["A"]


# --------------------------------------------------------------------------- #
# Concurrency: parallel SEEDS, never sharded designs
#
# A design's Boltz score depends on which other designs share its invocation:
# measured, the same sequence scored bz_ipsae 0.082758 alone and 0.519555 in a
# batch of four, while reordering the other three changed nothing. So batch
# composition moves the number and designs must never be split across
# processes. Parallelism is by seed, where each invocation still folds the full
# entry list.
# --------------------------------------------------------------------------- #


class SeedSpy:
    """Records one call per _fold_batch, i.e. per seed."""

    def __init__(self, scores):
        self.scores = scores
        self.calls = []

    def fold(self, entries, seed, dump_dir, tag="fold"):
        names = tuple(e["sample_name"] for e in entries)
        self.calls.append({"seed": seed, "tag": tag, "names": names})
        return {n: dict(self.scores[n], bz_status="ok") for n in names}


def _seed_backend(tmp_path, **over):
    backend = BoltzBackend(cfg=_cfg(tmp_path, **over), device="cpu")
    backend._spy = SeedSpy(SCORES)
    backend._fold_batch = backend._spy.fold
    return backend


def _entries():
    return [
        {"sample_name": n, "yaml": f"/w/{n}.yaml", "binder_id": "C"}
        for n in ("bb1_seq0", "bb1_seq1", "bb2_seq0", "bb2_seq1")
    ]


def test_every_seed_folds_the_full_entry_list(tmp_path):
    """The invariant that keeps scores unchanged: no invocation ever sees a
    subset of the batch."""
    backend = _seed_backend(tmp_path, workers=3)
    backend.score_flat(_entries(), str(tmp_path / "f"), seeds=3)
    assert len(backend._spy.calls) == 3
    for call in backend._spy.calls:
        assert len(call["names"]) == 4, call
    assert sorted(c["seed"] for c in backend._spy.calls) == [1, 2, 3]


def test_workers_one_still_folds_each_seed_once(tmp_path):
    backend = _seed_backend(tmp_path, workers=1)
    backend.score_flat(_entries(), str(tmp_path / "f"), seeds=3)
    assert len(backend._spy.calls) == 3
    assert all(len(c["names"]) == 4 for c in backend._spy.calls)


def test_workers_are_capped_by_the_seed_count(tmp_path):
    """workers > seeds cannot help: there is nothing else to run in parallel."""
    backend = _seed_backend(tmp_path, workers=8)
    out = backend.score_flat(_entries(), str(tmp_path / "f"), seeds=2)
    assert len(backend._spy.calls) == 2
    assert all(v["bz_n_seeds"] == 2 for v in out.values())


def test_seed_results_are_matched_to_their_own_seed(tmp_path):
    """Futures complete out of order; the per-seed rows must not be shuffled."""
    backend = BoltzBackend(cfg=_cfg(tmp_path, workers=3), device="cpu")
    seen = []

    def fold(entries, seed, dump_dir, tag="fold"):
        seen.append(seed)
        # seed-dependent value so a mix-up would show up in the mean
        return {
            e["sample_name"]: dict(
                SCORES[e["sample_name"]], bz_ipsae=0.1 * seed, bz_status="ok"
            )
            for e in entries
        }

    backend._fold_batch = fold
    out = backend.score_flat(_entries(), str(tmp_path / "f"), seeds=3)
    assert sorted(seen) == [1, 2, 3]
    # mean of 0.1, 0.2, 0.3
    for row in out.values():
        assert row["bz_ipsae"] == pytest.approx(0.2, abs=1e-9)
        assert row["bz_ipsae_sd"] == pytest.approx(0.1, abs=1e-9)


def test_a_raising_seed_marks_only_that_seed(tmp_path):
    backend = BoltzBackend(cfg=_cfg(tmp_path, workers=3), device="cpu")

    def fold(entries, seed, dump_dir, tag="fold"):
        if seed == 2:
            raise RuntimeError("boom")
        return {
            e["sample_name"]: dict(SCORES[e["sample_name"]], bz_status="ok")
            for e in entries
        }

    backend._fold_batch = fold
    out = backend.score_flat(_entries(), str(tmp_path / "f"), seeds=3)
    # one bad seed makes the design unusable, by design: None, never a partial mean
    for row in out.values():
        assert row["bz_status"] == "seed_error"
        assert row["bz_ipsae"] is None


def test_fold_batch_is_never_handed_a_subset(tmp_path):
    """Guards the regression directly: _fold_batch must receive every entry."""
    backend = BoltzBackend(cfg=_cfg(tmp_path, workers=4), device="cpu")
    sizes = []

    def fold(entries, seed, dump_dir, tag="fold"):
        sizes.append(len(entries))
        return {
            e["sample_name"]: dict(SCORES[e["sample_name"]], bz_status="ok")
            for e in entries
        }

    backend._fold_batch = fold
    backend.score_flat(_entries(), str(tmp_path / "f"), seeds=3)
    assert set(sizes) == {4}, sizes


# --------------------------------------------------------------------------- #
# Epitope policy and stable design ids (from the app review)
# --------------------------------------------------------------------------- #


def test_design_uid_is_stable_and_content_addressed():
    from pxdbench.tools.boltz.backend import design_uid

    a = {"name": "bb1", "seq_idx": 0, "sequence": "AAAA"}
    assert design_uid(a) == design_uid(dict(a))          # stable
    assert design_uid(a) != design_uid({**a, "sequence": "CCCC"})
    assert design_uid(a) != design_uid({**a, "seq_idx": 1})
    assert design_uid(a) != design_uid({**a, "name": "bb2"})
    assert len(design_uid(a)) == 16


def test_predict_emits_a_design_uid(tmp_path):
    backend = _backend(tmp_path)
    data_list = _data_list()
    _run(backend, tmp_path, data_list)
    uids = {i["design_uid"] for i in data_list}
    assert len(uids) == len(data_list), "uids must be unique per design"


def test_epitope_policy_reaches_the_calibration_record(tmp_path):
    policy = {"required": {"A": [25, 27]}, "cutoff": 4.0}
    backend = _backend(tmp_path, epitope_policy=policy)
    data_list = _data_list()
    _out, dump = _run(backend, tmp_path, data_list)
    record = json.load(open(os.path.join(dump, "bz_calibration.json")))
    assert record["epitope_policy"] == policy, (
        "the policy a campaign ran under must be recorded with its results"
    )


# --- pocket constraints ------------------------------------------------- #
# Boltz is template-free and knows nothing about the requested epitope, so it
# re-docks the binder wherever it likes. Measured across 7 designs in 3 runs:
# the diffusion engaged 3.0-3.5 of 6 hotspots, the scored pose 1.0-2.5, and the
# two poses agreed on the engaged set in 1 of 7. A pocket constraint ties the
# scored pose to the requested patch - at the cost of the gate thresholds,
# which were calibrated unconstrained.

# NOTE the chain ids: target_chains_from_orig_seqs renames orig_seqs' A0/B0 to
# the YAML's A/B, and the binder is C. Pocket contacts are in the YAML's
# spelling, which is NOT the campaign JSON's.
# in range: _orig_seqs gives each target chain 10 residues
POCKET = [["B", 2], ["B", 5]]


def test_pocket_contacts_reach_every_design_yaml(tmp_path):
    backend = _backend(tmp_path, pocket_contacts=POCKET)
    data_list = _data_list()
    _out, dump = _run(backend, tmp_path, data_list)
    yamls = [
        os.path.join(dump, "yamls", f)
        for f in os.listdir(os.path.join(dump, "yamls"))
    ]
    assert len(yamls) == len(data_list)
    for path in yamls:
        text = open(path).read()
        assert "pocket:" in text, f"{path} is unconstrained"
        assert "contacts: [[B, 2], [B, 5]]" in text


def test_no_pocket_contacts_by_default(tmp_path):
    """The gate thresholds depend on the unconstrained YAML."""
    backend = _backend(tmp_path)
    data_list = _data_list()
    _out, dump = _run(backend, tmp_path, data_list)
    for name in os.listdir(os.path.join(dump, "yamls")):
        # "constraints", not "pocket": the MSA paths in the YAML carry the
        # tmp_path, whose name contains the test's own name.
        assert "constraints" not in open(
            os.path.join(dump, "yamls", name)).read()


def test_calibration_records_that_pocket_constraints_void_the_gate(tmp_path):
    """A constrained run's bz_* scores are not the gate's scores.

    Without this in the provenance, a constrained run's numbers look exactly
    like an unconstrained run's and will be compared to thresholds that do not
    apply to them.
    """
    backend = _backend(tmp_path, pocket_contacts=POCKET)
    _out, dump = _run(backend, tmp_path, _data_list())
    record = json.load(open(os.path.join(dump, "bz_calibration.json")))
    assert record["pocket_contacts"] == POCKET
    assert record["gate_applies"] is False


def test_calibration_says_the_gate_applies_when_unconstrained(tmp_path):
    backend = _backend(tmp_path)
    _out, dump = _run(backend, tmp_path, _data_list())
    record = json.load(open(os.path.join(dump, "bz_calibration.json")))
    assert record["gate_applies"] is True
    assert record["pocket_contacts"] == []


def test_pocket_contact_on_an_unknown_chain_fails_before_any_folding(tmp_path):
    """Fail at prepare_json, not after the GPU work.

    The chain ids are the YAML's (A/B), not orig_seqs' (A0/B0). That rename is
    a live trap, so passing the wrong spelling must cost nothing but an error.
    """
    backend = _backend(tmp_path, pocket_contacts=[["B0", 2]])
    with pytest.raises(ValueError, match="not a target chain"):
        backend.prepare_json(
            str(tmp_path), _data_list(), dump_dir=str(tmp_path / "early"),
            orig_seqs=_orig_seqs(tmp_path),
        )
    assert backend._spy.batches == [], "nothing may be folded"


def test_a_mismatched_msa_is_rejected_before_any_yaml_is_written(tmp_path):
    """The whole point: fail before the GPU, and before touching the disk."""
    from pxdbench.tools.boltz.msa_check import MsaMismatch

    msa_dir = tmp_path / "msa" / "0"
    msa_dir.mkdir(parents=True)
    # A real-looking MSA for MKTAYIAKQR: the query plus distinct homologues.
    records = [">query\nMKTAYIAKQR\n"]
    for letter in "ACDEFGHIKLMNPQRSTVWY":
        records.append(f">hom{letter}\nMKTAYIAKQ{letter}\n")
    msa_dir.joinpath("non_pairing.a3m").write_text("".join(records))
    orig = [{
        "proteinChain": {
            "sequence": "MKTAYIAKNR",          # one substitution from the query
            "count": 1,
            "label_asym_id": ["A0"],
            "use_msa": True,
            "msa": {"precomputed_msa_dir": str(msa_dir)},
        }
    }]
    backend = BoltzBackend(cfg=_cfg(tmp_path), device="cpu")
    dump = tmp_path / "dump"
    with pytest.raises(MsaMismatch, match="unconditioned"):
        backend.prepare_json(str(tmp_path), _data_list(), dump_dir=str(dump),
                             orig_seqs=orig)
    assert not (dump / "yamls").exists(), "YAMLs were written despite a bad MSA"
    assert not (dump / "boltz_manifest.json").exists()


def test_a_valid_msa_records_provenance_beside_the_manifest(tmp_path):
    dump = tmp_path / "dump"
    backend = _backend(tmp_path)
    backend.prepare_json(
        str(tmp_path), _data_list(), dump_dir=str(dump),
        orig_seqs=_orig_seqs(tmp_path),
    )
    rec = json.load(open(dump / "msa_validation.json"))
    # the resolver renames A0/B0 to A/B, and these are the RESOLVED chains
    assert [r["label"] for r in rec] == ["target chain A", "target chain B"]
    assert all(r["a3m_sha256"] and r["sequence_sha256"] for r in rec)


def test_min_msa_depth_is_configurable(tmp_path):
    """A caller may accept a shallow cache; none may accept a mismatched one."""
    from pxdbench.tools.boltz.msa_check import MsaMismatch

    msa_dir = tmp_path / "shallow" / "0"
    msa_dir.mkdir(parents=True)
    msa_dir.joinpath("non_pairing.a3m").write_text(">q\nMKTAYIAKQR\n")
    orig = [{
        "proteinChain": {
            "sequence": "MKTAYIAKQR", "count": 1, "label_asym_id": ["A0"],
            "use_msa": True, "msa": {"precomputed_msa_dir": str(msa_dir)},
        }
    }]
    ok = BoltzBackend(cfg=_cfg(tmp_path, min_msa_depth=1), device="cpu")
    ok.prepare_json(str(tmp_path), _data_list(),
                    dump_dir=str(tmp_path / "a"), orig_seqs=orig)

    strict = BoltzBackend(cfg=_cfg(tmp_path, min_msa_depth=50), device="cpu")
    with pytest.raises(MsaMismatch, match="below 50"):
        strict.prepare_json(str(tmp_path), _data_list(),
                            dump_dir=str(tmp_path / "b"), orig_seqs=orig)


def test_a_target_msa_override_is_validated_at_the_backend(tmp_path):
    """A valid entity MSA plus a mismatched override must not reach the GPU.
    Validating orig_seqs would have passed this."""
    from pxdbench.tools.boltz.msa_check import MsaMismatch

    bad = tmp_path / "wrong" / "0"
    bad.mkdir(parents=True)
    bad.joinpath("non_pairing.a3m").write_text(">q\nQRSTVWYCDE\n")
    backend = BoltzBackend(
        cfg=_cfg(tmp_path, min_msa_depth=1,
                 target_msa={"A": str(bad / "non_pairing.a3m")}),
        device="cpu",
    )
    with pytest.raises(MsaMismatch):
        backend.prepare_json(
            str(tmp_path), _data_list(), dump_dir=str(tmp_path / "ov"),
            orig_seqs=_orig_seqs(tmp_path),
        )


def test_validation_cannot_be_switched_off(tmp_path):
    """There is no cfg key that disables the identity check. If one is added,
    this test is the place the decision gets argued, not a config file."""
    from pxdbench.tools.boltz import msa_check

    src = inspect.getsource(BoltzBackend.prepare_json)
    assert "validate_target_chains" in src
    assert msa_check.DEFAULT_MIN_DEPTH == 100


def test_a_changed_yaml_refreshes_its_staged_copy(tmp_path):
    """Reusing an output dir with a changed sequence must not fold the old one.

    _fold_batch copied only when the staged file was absent, so a rerun with a
    new sequence silently predicted the previous one.
    """
    backend = BoltzBackend(cfg=_cfg(tmp_path, boltz_bin=str(_fake_boltz(tmp_path))),
                           device="cpu")
    dump = str(tmp_path / "d")
    src = tmp_path / "src.yaml"
    src.write_text("version: 1\nsequences: [FIRST]\n")
    entries = [{"sample_name": "d1", "yaml": str(src), "binder_id": "C"}]

    backend._fold_batch(entries, 1, dump, tag="gate")
    staged = os.path.join(dump, "boltz_pred", "gate", "seed_1", "input", "d1.yaml")
    assert "FIRST" in open(staged).read()

    src.write_text("version: 1\nsequences: [SECOND]\n")
    backend._fold_batch(entries, 1, dump, tag="gate")
    assert "SECOND" in open(staged).read(), "stale staged YAML was reused"


def test_the_batch_digest_covers_the_inputs_that_change_a_score(tmp_path):
    backend = BoltzBackend(cfg=_cfg(tmp_path), device="cpu")
    a = tmp_path / "a.yaml"
    a.write_text("sequences: [A]\n")
    e = [{"sample_name": "d1", "yaml": str(a), "binder_id": "C"}]

    base = backend._batch_digest(e, 1, "gate")
    assert base == backend._batch_digest(e, 1, "gate"), "digest is not stable"

    a.write_text("sequences: [B]\n")
    assert backend._batch_digest(e, 1, "gate") != base, "yaml content ignored"

    assert backend._batch_digest(e, 2, "gate") != base, "seed ignored"

    wider = e + [{"sample_name": "d2", "yaml": str(a), "binder_id": "C"}]
    assert backend._batch_digest(wider, 1, "gate") != base, "composition ignored"

    other = BoltzBackend(cfg=_cfg(tmp_path, recycling_steps=9), device="cpu")
    assert other._batch_digest(e, 1, "gate") != base, "boltz params ignored"


def test_a_changed_input_also_clears_boltzs_processed_input_cache(tmp_path):
    """Pins the mechanism, not just the predictions: boltz skips any record
    already in <out>/boltz_results_*/processed/records (boltz/main.py:724-742),
    so refreshing the staged YAML alone would re-predict the OLD input. The
    clearing of boltz_results_* is what makes the refresh effective."""
    backend = BoltzBackend(cfg=_cfg(tmp_path, boltz_bin=str(_fake_boltz(tmp_path))),
                           device="cpu")
    dump = str(tmp_path / "d")
    src = tmp_path / "s.yaml"
    src.write_text("sequences: [FIRST]\n")
    entries = [{"sample_name": "d1", "yaml": str(src), "binder_id": "C"}]

    backend._fold_batch(entries, 1, dump, tag="gate")
    seed_root = os.path.join(dump, "boltz_pred", "gate", "seed_1")
    stale = os.path.join(seed_root, "boltz_results_input", "predictions", "d1")
    os.makedirs(stale, exist_ok=True)
    marker = os.path.join(stale, "d1_model_0.pdb")
    open(marker, "w").write("ATOM  stale\n")

    cache = os.path.join(seed_root, "boltz_results_input", "processed",
                         "records", "d1.json")
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    open(cache, "w").write("{}")

    src.write_text("sequences: [SECOND]\n")
    backend._fold_batch(entries, 1, dump, tag="gate")
    assert not os.path.exists(marker), "a stale prediction survived an input change"
    assert not os.path.exists(cache), (
        "boltz's processed-input cache survived; it would skip re-converting "
        "the changed YAML and predict the previous input"
    )


def test_an_edited_msa_changes_the_digest(tmp_path):
    """The YAML names the MSA by PATH, so editing a cached a3m in place leaves
    every YAML byte unchanged while changing what boltz conditions on."""
    a3m = tmp_path / "m" / "non_pairing.a3m"
    a3m.parent.mkdir(parents=True)
    a3m.write_text(">q\nMKTA\n")
    backend = BoltzBackend(cfg=_cfg(tmp_path), device="cpu")
    backend._target_chains = [{"id": "A", "seq": "MKTA", "msa": str(a3m)}]
    y = tmp_path / "a.yaml"
    y.write_text("sequences: [A]\n")
    e = [{"sample_name": "d1", "yaml": str(y), "binder_id": "C"}]

    before = backend._batch_digest(e, 1, "gate")
    a3m.write_text(">q\nMKTA\n>h1\nMKTV\n")
    assert backend._batch_digest(e, 1, "gate") != before, (
        "an edited MSA left the digest unchanged"
    )


def _digest_for_bin(tmp_path, binary):
    y = tmp_path / "a.yaml"
    y.write_text("sequences: [A]\n")
    e = [{"sample_name": "d1", "yaml": str(y), "binder_id": "C"}]
    return BoltzBackend(cfg=_cfg(tmp_path, boltz_bin=str(binary)),
                        device="cpu")._batch_digest(e, 1, "gate")


def test_a_different_predictor_path_changes_the_digest(tmp_path):
    """Same content and size, different path: only the path differs."""
    b1 = _fake_boltz(tmp_path)
    b2 = tmp_path / "other_boltz"
    b2.write_bytes(b1.read_bytes())
    b2.chmod(b1.stat().st_mode)
    os.utime(b2, (int(b1.stat().st_mtime),) * 2)
    assert _digest_for_bin(tmp_path, b1) != _digest_for_bin(tmp_path, b2)


def test_a_touched_predictor_changes_the_digest(tmp_path):
    """Same path, same size, same content; only the mtime moves."""
    b = _fake_boltz(tmp_path)
    before = _digest_for_bin(tmp_path, b)
    st = b.stat()
    os.utime(b, (st.st_atime, st.st_mtime + 100))
    assert _digest_for_bin(tmp_path, b) != before


def test_predictions_are_cleared_on_every_run(tmp_path):
    """Clearing is unconditional, even when nothing changed. boltz caches its
    INPUTS under boltz_results_*/processed (boltz/main.py:724-742) regardless
    of --override, so the tree is removed on every run; a 'keep it when the
    digest matches' optimisation would restore stale-input scoring."""
    backend = BoltzBackend(cfg=_cfg(tmp_path, boltz_bin=str(_fake_boltz(tmp_path))),
                           device="cpu")
    dump = str(tmp_path / "d")
    src = tmp_path / "s.yaml"
    src.write_text("sequences: [ONLY]\n")
    entries = [{"sample_name": "d1", "yaml": str(src), "binder_id": "C"}]

    backend._fold_batch(entries, 1, dump, tag="gate")
    seed_root = os.path.join(dump, "boltz_pred", "gate", "seed_1")
    stale = os.path.join(seed_root, "boltz_results_input", "predictions", "d1")
    os.makedirs(stale, exist_ok=True)
    marker = os.path.join(stale, "d1_model_0.pdb")
    open(marker, "w").write("ATOM  from a previous attempt\n")

    backend._fold_batch(entries, 1, dump, tag="gate")
    assert not os.path.exists(marker)


def test_predictions_with_no_config_record_are_treated_as_stale(tmp_path):
    """Absence of evidence that they match is not evidence that they do."""
    backend = BoltzBackend(cfg=_cfg(tmp_path, boltz_bin=str(_fake_boltz(tmp_path))),
                           device="cpu")
    dump = str(tmp_path / "d")
    src = tmp_path / "s.yaml"
    src.write_text("sequences: [ONLY]\n")
    entries = [{"sample_name": "d1", "yaml": str(src), "binder_id": "C"}]
    seed_root = os.path.join(dump, "boltz_pred", "gate", "seed_1")
    orphan = os.path.join(seed_root, "boltz_results_input", "predictions", "d1")
    os.makedirs(orphan, exist_ok=True)
    marker = os.path.join(orphan, "d1_model_0.pdb")
    open(marker, "w").write("ATOM  orphaned\n")

    backend._fold_batch(entries, 1, dump, tag="gate")
    assert not os.path.exists(marker)


@pytest.mark.parametrize("junk", ["{not json", "[]", "null", "3", '"s"'])
def test_a_corrupt_config_record_does_not_crash_the_fold(tmp_path, junk):
    backend = BoltzBackend(cfg=_cfg(tmp_path, boltz_bin=str(_fake_boltz(tmp_path))),
                           device="cpu")
    dump = str(tmp_path / "d")
    src = tmp_path / "s.yaml"
    src.write_text("sequences: [ONLY]\n")
    entries = [{"sample_name": "d1", "yaml": str(src), "binder_id": "C"}]
    seed_root = os.path.join(dump, "boltz_pred", "gate", "seed_1")
    os.makedirs(seed_root, exist_ok=True)
    open(os.path.join(seed_root, ".batch_config.json"), "w").write(junk)

    backend._fold_batch(entries, 1, dump, tag="gate")
    rec = json.load(open(os.path.join(seed_root, ".batch_config.json")))
    assert rec["digest"] == backend._batch_digest(entries, 1, "gate")


def test_the_digest_is_recorded_for_the_audit(tmp_path):
    backend = BoltzBackend(cfg=_cfg(tmp_path, boltz_bin=str(_fake_boltz(tmp_path))),
                           device="cpu")
    dump = str(tmp_path / "d")
    src = tmp_path / "s.yaml"
    src.write_text("sequences: [ONLY]\n")
    entries = [{"sample_name": "d1", "yaml": str(src), "binder_id": "C"}]
    backend._fold_batch(entries, 1, dump, tag="gate")

    rec = json.load(open(os.path.join(
        dump, "boltz_pred", "gate", "seed_1", ".batch_config.json")))
    assert rec["digest"] == backend._batch_digest(entries, 1, "gate")
    assert rec["entries"] == ["d1"]
    assert rec["params"]["seed"] == 1


def _fold_setup(tmp_path):
    backend = BoltzBackend(cfg=_cfg(tmp_path, boltz_bin=str(_fake_boltz(tmp_path))),
                           device="cpu")
    dump = str(tmp_path / "d")
    src = tmp_path / "s.yaml"
    src.write_text("sequences: [ONE]\n")
    entries = [{"sample_name": "d1", "yaml": str(src), "binder_id": "C"}]
    return backend, dump, src, entries


def test_changed_inputs_warn_naming_both_digests(tmp_path, capsys):
    backend, dump, src, entries = _fold_setup(tmp_path)
    backend._fold_batch(entries, 1, dump, tag="gate")
    old = backend._batch_digest(entries, 1, "gate")
    src.write_text("sequences: [TWO]\n")
    new = backend._batch_digest(entries, 1, "gate")
    capsys.readouterr()
    backend._fold_batch(entries, 1, dump, tag="gate")
    out = capsys.readouterr().out
    assert "configuration changed" in out
    assert old[:12] in out and new[:12] in out


def test_predictions_without_a_record_warn_unverifiable(tmp_path, capsys):
    backend, dump, _, entries = _fold_setup(tmp_path)
    seed_root = os.path.join(dump, "boltz_pred", "gate", "seed_1")
    os.makedirs(os.path.join(seed_root, "boltz_results_input", "predictions"))
    backend._fold_batch(entries, 1, dump, tag="gate")
    out = capsys.readouterr().out
    assert "no verifiable configuration record" in out
    assert "configuration changed" not in out


def test_unchanged_inputs_emit_neither_warning(tmp_path, capsys):
    backend, dump, _, entries = _fold_setup(tmp_path)
    backend._fold_batch(entries, 1, dump, tag="gate")
    capsys.readouterr()
    backend._fold_batch(entries, 1, dump, tag="gate")
    out = capsys.readouterr().out
    assert "configuration changed" not in out
    assert "no verifiable configuration record" not in out


def test_a_discarded_msa_fails_the_seed_rather_than_scoring_it(tmp_path, capsys):
    """boltz prints this to stdout and exits 0. Scores from that seed describe
    an unconditioned prediction, so they must not be reported as scores."""
    from pxdbench.tools.boltz.msa_check import DUMMY_MSA_WARNING

    stub = tmp_path / "b"
    stub.write_text(f"#!/bin/sh\necho \"{DUMMY_MSA_WARNING} 1 ...\"\nexit 0\n")
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)

    backend = BoltzBackend(cfg=_cfg(tmp_path, boltz_bin=str(stub)), device="cpu")
    src = tmp_path / "s.yaml"
    src.write_text("sequences: [A]\n")
    entries = [{"sample_name": "d1", "yaml": str(src), "binder_id": "C"}]
    out_rows = backend._fold_batch(entries, 1, str(tmp_path / "d"), tag="gate")

    assert out_rows["d1"]["bz_status"] == "msa_discarded"
    printed = capsys.readouterr().out
    assert "DISCARDED" in printed
    # M10: the operator's next step, and what the count is a count of
    assert "msa_validation.json" in printed
    assert "design x chain" in printed
    assert "unconditioned" in printed


def test_the_warning_is_caught_when_buried_in_progress_output(tmp_path, capsys):
    """End to end: the detection must not depend on the warning landing in a
    tail window."""
    from pxdbench.tools.boltz.msa_check import DUMMY_MSA_WARNING

    stub = tmp_path / "b"
    stub.write_text(
        f"#!/bin/sh\necho \"{DUMMY_MSA_WARNING} 1\"\n"
        "i=0; while [ $i -lt 400 ]; do echo '..................'; "
        "i=$((i+1)); done\nexit 0\n"
    )
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)

    backend = BoltzBackend(cfg=_cfg(tmp_path, boltz_bin=str(stub)), device="cpu")
    src = tmp_path / "s.yaml"
    src.write_text("sequences: [A]\n")
    entries = [{"sample_name": "d1", "yaml": str(src), "binder_id": "C"}]
    out_rows = backend._fold_batch(entries, 1, str(tmp_path / "d"), tag="gate")

    assert out_rows["d1"]["bz_status"] == "msa_discarded"


def test_the_zero_structure_warning_quotes_both_streams(tmp_path, capsys):
    """stderr is deliberately LONGER than the printed window. With a few bytes
    of stderr this passed even when stdout was displaced, because everything
    fit; the defect only exists when stderr overflows the tail."""
    noise_bytes = 4096
    stub = tmp_path / "b"
    stub.write_text(
        "#!/bin/sh\necho 'Number of failed examples: 1'\n"
        f"i=0; while [ $i -lt {noise_bytes // 64} ]; do "
        "echo 'torch noise torch noise torch noise torch noise torch noise.' >&2; "
        "i=$((i+1)); done\nexit 0\n"
    )
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    from pxdbench.tools.boltz import runner as br
    probe = br.run_seed(str(stub), str(tmp_path), str(tmp_path / "o"), 1)
    assert len(probe["stderr_raw_tail"]) > 1200, (
        "the stub's stderr must exceed the printed window or this proves nothing"
    )

    backend = BoltzBackend(cfg=_cfg(tmp_path, boltz_bin=str(stub)), device="cpu")
    src = tmp_path / "s.yaml"
    src.write_text("sequences: [A]\n")
    entries = [{"sample_name": "d1", "yaml": str(src), "binder_id": "C"}]
    backend._fold_batch(entries, 1, str(tmp_path / "d"), tag="gate")

    out = capsys.readouterr().out
    assert "Number of failed examples: 1" in out, "the stdout diagnostic was lost"
    assert "--- stderr (tail) ---" in out and "torch noise" in out


def _real_prediction_files(tmp_path):
    """A PDB and PAE that parse.py really scores (status `ok`): a 3-residue
    target (chain A) against a 3-residue binder (chain C), all within contact.
    Garbage bytes would parse to `parse_error`, which would make the
    completeness checks below pass or fail for the wrong reason."""
    import numpy as np

    pdb = tmp_path / "real.pdb"
    lines = []
    for i, chain in enumerate("AAACCC"):
        resi = i % 3 + 1
        lines.append(
            f"ATOM  {i + 1:5d}  CA  GLY {chain}{resi:4d}    "
            f"{3.8 * i:8.3f}{0:8.3f}{0:8.3f}  1.00  0.00           C"
        )
    pdb.write_text("\n".join(lines) + "\nEND\n")
    pae = tmp_path / "real_pae.npz"
    np.savez(pae, pae=np.full((6, 6), 2.0))
    plddt = tmp_path / "real_plddt.npz"
    np.savez(plddt, plddt=np.full(6, 90.0))
    return pdb, pae, plddt


def _artifact_boltz(tmp_path, pdb_names, npz_names, with_plddt=True):
    """A boltz that exits 0 and writes the named predictions.

    `pdb_names` get a real PDB; `npz_names` get a real PAE (and pLDDT unless
    `with_plddt` is False)."""
    pdb, pae, plddt = _real_prediction_files(tmp_path)
    lines = ["#!/bin/sh", 'out="$4"']
    lines.append('d="$out/boltz_results_input/predictions"')
    for n in set(pdb_names) | set(npz_names):
        lines.append(f'mkdir -p "$d/{n}"')
    for n in pdb_names:
        lines.append(f'cp "{pdb}" "$d/{n}/{n}_model_0.pdb"')
    for n in npz_names:
        lines.append(f'cp "{pae}" "$d/{n}/pae_{n}_model_0.npz"')
        if with_plddt:
            lines.append(f'cp "{plddt}" "$d/{n}/plddt_{n}_model_0.npz"')
    lines.append("exit 0")
    p = tmp_path / "artifact_boltz"
    p.write_text("\n".join(lines) + "\n")
    p.chmod(p.stat().st_mode | stat.S_IEXEC)
    return p


def _fold_artifacts(tmp_path, pdbs, npzs, names=("d1", "d2"), with_plddt=True):
    backend = BoltzBackend(
        cfg=_cfg(tmp_path, boltz_bin=str(
            _artifact_boltz(tmp_path, pdbs, npzs, with_plddt))),
        device="cpu",
    )
    entries = []
    for n in names:
        src = tmp_path / f"{n}.yaml"
        src.write_text("sequences: [A]\n")
        entries.append({"sample_name": n, "yaml": str(src), "binder_id": "C"})
    return backend._fold_batch(entries, 1, str(tmp_path / "d"), tag="gate")


def test_a_pdb_without_its_npz_files_is_called_out(tmp_path, capsys):
    _fold_artifacts(tmp_path, ["d1", "d2"], ["d1"])
    out = capsys.readouterr().out
    assert "d2 has status `missing_pae`" in out
    assert "wrote NO structures" not in out
    assert "only 1/2" in out
    # M9: a clause is emitted only when its count is non-zero
    assert "0 wrote" not in out and "wrote no structure" not in out


def test_a_prediction_missing_only_plddt_is_scored_and_not_a_failure(
    tmp_path, capsys
):
    """parse.read_prediction treats plddt as optional, so the completeness
    message must agree with the `ok` row beside it."""
    rows = _fold_artifacts(tmp_path, ["d1", "d2"], ["d1", "d2"],
                           with_plddt=False)
    out = capsys.readouterr().out
    assert rows["d1"]["bz_status"] == "ok", "fixture must produce a scored row"
    assert "[WARN]" not in out


def test_all_pdbs_but_no_artifacts_does_not_claim_no_structures(tmp_path, capsys):
    _fold_artifacts(tmp_path, ["d1", "d2"], [])
    out = capsys.readouterr().out
    assert "wrote 2 structure(s) but none could be scored" in out
    assert "wrote NO structures" not in out
    assert "GPU contention" not in out
    assert "MSA" not in out
    assert out.count("[WARN]") == 1


def test_complete_artifacts_emit_no_warning(tmp_path, capsys):
    _fold_artifacts(tmp_path, ["d1", "d2"], ["d1", "d2"])
    out = capsys.readouterr().out
    assert "[WARN]" not in out


def test_no_pdbs_at_all_still_reports_no_structures(tmp_path, capsys):
    _fold_artifacts(tmp_path, [], [])
    out = capsys.readouterr().out
    assert "wrote NO structures" in out
    assert "GPU contention" in out
    assert "could be scored" not in out


def test_the_measured_boltz_version_is_read_from_the_interpreter_beside_it(tmp_path):
    from pxdbench.tools.boltz.backend import measured_boltz_version

    py = tmp_path / "python"
    py.write_text("#!/bin/sh\necho 9.9.9\n")
    py.chmod(py.stat().st_mode | stat.S_IEXEC)
    assert measured_boltz_version(str(tmp_path / "boltz")) == "9.9.9"
    assert measured_boltz_version(str(tmp_path / "nowhere" / "boltz")) is None


def test_an_empty_data_list_still_returns_a_manifest(tmp_path):
    """The msa_validation.json write used to precede any makedirs, so an empty
    roster into a fresh directory died with FileNotFoundError."""
    backend = _backend(tmp_path)
    dump = str(tmp_path / "fresh" / "dump")
    manifest = backend.prepare_json(
        str(tmp_path), [], dump_dir=dump, orig_seqs=_orig_seqs(tmp_path)
    )
    assert manifest and os.path.isfile(os.path.join(dump, "msa_validation.json"))
