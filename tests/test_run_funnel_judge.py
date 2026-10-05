"""run_funnel.py judge selection, run forking and re-ranking, with the predictors replaced by fakes (CPU only)."""
import json, os, subprocess, sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "funnel")); sys.path.insert(0, str(REPO))
import common, oracles, run_funnel  # noqa: E402


# ----------------------------------------------------------------------------- config guard
def test_a_run_folder_from_before_judges_existed_counts_as_legacy(tmp_path):
    (tmp_path / "run_config.json").write_text(json.dumps(dict(target="pdl1", chunk=100)))      # no judge keys, as written by every earlier version
    legacy = dict(target="pdl1", chunk=100, judge="legacy", second_oracle="protenix-v2", boltz_seeds="1,2,3")
    run_funnel.check_config(tmp_path, legacy, False)                                           # resuming it with the default judge is fine
    (tmp_path / "run_config.json").write_text(json.dumps(dict(target="pdl1", chunk=100)))
    with pytest.raises(SystemExit, match="judge"):                                             # but a different judge must not mix into its consensus table
        run_funnel.check_config(tmp_path, {**legacy, "judge": "new", "second_oracle": "af3", "boltz_seeds": "1"}, False)


def test_judge_defaults():
    assert run_funnel.JUDGES["legacy"] == dict(second_oracle="protenix-v2", boltz_seeds="1,2,3", rank_rule="min")
    assert run_funnel.JUDGES["new"] == dict(second_oracle="af3", boltz_seeds="1", rank_rule="mean")


def test_cli_exposes_the_new_options():
    p = subprocess.run([sys.executable, str(REPO / "funnel" / "run_funnel.py"), "--help"], capture_output=True, text=True)
    assert p.returncode == 0 and all(f in p.stdout for f in ("--judge", "--second-oracle", "--boltz-seeds", "--rank-rule", "--o2-gate", "--fork-from"))


# ----------------------------------------------------------------------------- fork_run
def _make_finished_run(src):
    (src / "gen" / "chunk_0000").mkdir(parents=True); (src / "gen" / "chunk_0000" / "done.json").write_text("{}")
    (src / "screen_parts").mkdir(); (src / "screen_parts" / "part_0.csv").write_text("id\nd0\n")
    for f in ("screen.csv", "start_parents.csv", "consensus_nocycle.csv", "final_nocycle.csv"): (src / f).write_text("id\nd0\n")
    for sub in ("boltz/seed1/boltz_results_yaml", "v2/pred"): (src / "consensus_nocycle" / sub).mkdir(parents=True)
    (src / "consensus_nocycle" / "boltz" / "seed1" / "boltz_results_yaml" / "pred.npz").write_text("p")
    (src / "consensus_nocycle" / "v2" / "pred" / "x.json").write_text("v")
    (src / "final_design").mkdir(); (src / "final_design" / "README.md").write_text("old")
    (src / "timers.json").write_text(json.dumps({"1_generate": 10, "3_fast_screen": 5, "5_boltz_nocycle": 99, "5_v2_nocycle": 77}))
    (src / "run_state.json").write_text(json.dumps({"stages": {"generate": {"status": "done"}, "consensus_nocycle": {"status": "done"}, "complete": {"status": "done"}}}))
    (src / "run_config.json").write_text(json.dumps(dict(target="pdl1", chunk=100, steps=400)))


def test_fork_run_links_judge_independent_work_and_leaves_judging_to_the_new_run(tmp_path):
    src, dst = tmp_path / "a", tmp_path / "b"; src.mkdir(); _make_finished_run(src)
    run_funnel.fork_run(src, dst, dict(judge="new", second_oracle="af3", boltz_seeds="1"))
    same = lambda p: os.stat(src / p).st_ino == os.stat(dst / p).st_ino
    assert same("screen.csv") and same("gen/chunk_0000/done.json") and same("screen_parts/part_0.csv")      # hard links: no copy, no recompute
    assert same("consensus_nocycle/boltz/seed1/boltz_results_yaml/pred.npz")                               # Boltz-2 predictions are shared
    assert not (dst / "consensus_nocycle.csv").exists() and not (dst / "final_nocycle.csv").exists() and not (dst / "final_design").exists()
    assert not (dst / "consensus_nocycle" / "v2").exists()                                                 # the other judge's predictions are not carried over
    assert json.load(open(dst / "timers.json")) == {"1_generate": 10, "3_fast_screen": 5}
    assert list(json.load(open(dst / "run_state.json"))["stages"]) == ["generate"]
    cfg = json.load(open(dst / "run_config.json")); assert cfg["judge"] == "new" and cfg["second_oracle"] == "af3" and cfg["steps"] == 400
    run_funnel.check_config(dst, {**cfg}, False)                                                           # the fork resumes under its own judge
    assert (src / "consensus_nocycle.csv").exists()                                                        # source untouched


def test_fork_run_refuses_a_non_run_source_and_a_used_destination(tmp_path):
    (tmp_path / "not_a_run").mkdir()
    with pytest.raises(SystemExit): run_funnel.fork_run(tmp_path / "not_a_run", tmp_path / "x")
    src, dst = tmp_path / "a", tmp_path / "b"; src.mkdir(); _make_finished_run(src); dst.mkdir(); (dst / "keep.txt").write_text("mine")
    with pytest.raises(SystemExit): run_funnel.fork_run(src, dst)
    assert (dst / "keep.txt").read_text() == "mine"


# ----------------------------------------------------------------------------- consensus() dispatch
@pytest.fixture
def fakes(monkeypatch, tmp_path):
    calls = dict(boltz=[], protenix=[], af3=[])
    def boltz(df, t, out, seeds=(1,), recycles=3, steps=200, keep_structures=True, gpus=None):
        calls["boltz"].append(dict(seeds=tuple(seeds), gpus=gpus))
        return pd.DataFrame(dict(id=df.id, b_ok=True, b_ipsae=[0.9, 0.7, 0.55], b_ipsae_sd=0.0, b_paemin=[1.0, 1.5, 1.9], b_iptm=0.8, b_nseeds=len(seeds), b_cif=[f"{i}.cif" for i in df.id]))
    def v2(ipsae):
        return lambda df: pd.DataFrame(dict(id=df.id, v2_ok=~np.isnan(ipsae), v2_ipsae=ipsae, v2_ipsae_max=ipsae, v2_iptm=0.7, v2_rank=0.7, v2_paemin=1.0, v2_cif="c.cif"))
    state = dict(v2=v2(np.array([0.6, 0.8, np.nan])))
    def protenix(df, t, out, arm="fast", seed=101, chunk=1500, tag=""):
        calls["protenix"].append(dict(arm=arm, seed=seed)); return state["v2"](df)
    def af3(df, t, out, seed=1, gpus=None, n_samples=5, hold=None):
        calls["af3"].append(dict(seed=seed, gpus=gpus, hold=hold)); return state["v2"](df)
    monkeypatch.setattr(common, "boltz_fold", boltz); monkeypatch.setattr(common, "protenix_fold", protenix); monkeypatch.setattr(oracles, "af3_fold", af3)
    monkeypatch.setattr(common, "hotspot_contacts", lambda cif, nt, hs: (0.5, 1)); monkeypatch.setattr(run_funnel, "add_pisa", lambda d, t: d)
    t = dict(seq="A" * 10, hotspot_idx=[1, 2]); cands = pd.DataFrame(dict(id=["a", "b", "c"], seq=["MKKAAAAAAA", "MQQDDDDDDD", "MHHEEEEEEE"]))   # distinct: shortlist() de-duplicates similar sequences
    return calls, t, cands, run_funnel.Timers(tmp_path / "timers.json")


def test_legacy_judge_is_unchanged(fakes, tmp_path):
    calls, t, cands, T = fakes
    d = run_funnel.consensus(t, cands, tmp_path / "c", (1, 2, 3), T, "x")
    assert calls["boltz"][0]["seeds"] == (1, 2, 3) and calls["protenix"] == [dict(arm="v2", seed=1)] and calls["af3"] == []
    assert (d.o2_name == "protenix-v2").all() and "5_v2_x" in T and "5_af3_x" not in T
    assert d.consensus.iloc[:2].tolist() == [0.6, 0.7] and d.consensus_min.iloc[:2].tolist() == [0.6, 0.7]      # min rule
    assert d.consensus_pass.tolist() == [True, True, False]                                                    # c: Boltz gate passes? 0.55 and pae 1.9 do, but v2 missing -> False


def test_new_judge_uses_one_boltz_seed_af3_and_the_mean_rule(fakes, tmp_path):
    calls, t, cands, T = fakes
    d = run_funnel.consensus(t, cands, tmp_path / "c", (1,), T, "x", gpus=["1", "0", "2"], oracle2="af3", rank_rule="mean")
    assert calls["boltz"][0]["seeds"] == (1,) and calls["protenix"] == [] and calls["af3"][0]["seed"] == 1
    assert calls["af3"][0]["gpus"] == ["1", "0", "2"] and set(calls["af3"][0]["hold"]) == {"1"}                 # AF3 uses every GPU; the one Boltz-2 started on joins when it is done
    assert (d.o2_name == "af3").all() and "5_af3_x" in T and "5_af3_gpu_s_x" in T and (d.rank_rule == "mean").all()
    assert d.consensus.iloc[:2].tolist() == pytest.approx([0.75, 0.75]) and d.consensus_min.iloc[:2].tolist() == [0.6, 0.7]


def test_a_design_missing_the_second_oracle_has_no_consensus_and_cannot_pass(fakes, tmp_path):
    calls, t, cands, T = fakes
    for rule in ("min", "mean"):
        d = run_funnel.consensus(t, cands, tmp_path / f"c{rule}", (1,), T, rule, oracle2="af3", rank_rule=rule).set_index("id")
        assert np.isnan(d.loc["c", "consensus"]) and not d.loc["c", "consensus_pass"]
        assert not np.isnan(d.loc["a", "consensus"])
        assert set(run_funnel.shortlist(d.reset_index(), 5).id) == {"a", "b"}                  # the shortlist skips designs without a consensus


def test_the_second_oracle_gate_is_configurable(fakes, tmp_path):
    calls, t, cands, T = fakes
    d = run_funnel.consensus(t, cands, tmp_path / "c", (1,), T, "g", oracle2="af3", o2_gate=0.7)
    assert d.v2_pass.tolist() == [False, True, False]


def test_unknown_second_oracle_is_rejected(fakes, tmp_path):
    calls, t, cands, T = fakes
    with pytest.raises(ValueError, match="unknown second oracle"): run_funnel.consensus(t, cands, tmp_path / "c", (1,), T, "x", oracle2="chai")


# ----------------------------------------------------------------------------- cheap re-ranking
def test_changing_the_rank_rule_reorders_finished_predictions_without_refolding(tmp_path, monkeypatch):
    d = pd.DataFrame(dict(id=list("abc"), seq=["MKKAAA", "MQQDDD", "MHHEEE"], b_ipsae=[0.95, 0.70, 0.74], v2_ipsae=[0.55, 0.72, 0.74], fast_ipsae=[0.9, 0.8, 0.7]))
    d["consensus"] = oracles.consensus_score(d, ("b_ipsae", "v2_ipsae"), "min")                  # an old (legacy) consensus table: no rank_rule column
    d.to_csv(tmp_path / "consensus_x.csv", index=False); run_funnel.shortlist(d, 3).to_csv(tmp_path / "final_x.csv", index=False)
    assert pd.read_csv(tmp_path / "final_x.csv").id.tolist() == ["c", "b", "a"]                  # min: c=0.74, b=0.70, a=0.55
    monkeypatch.setattr(run_funnel, "consensus", lambda *a, **k: pytest.fail("predictions must be reused"))
    T = run_funnel.Timers(tmp_path / "t.json"); st = run_funnel.RunState(tmp_path / "s.json")
    src = d[["id", "seq", "fast_ipsae"]]
    same = dict(seeds=(1, 2, 3), oracle2="protenix-v2", o2_gate=0.5, rank_rule="min")
    run_funnel.finalize_variant(dict(), tmp_path, "x", src, 3, 3, T, st, False, None, same)       # same rule: nothing changes
    assert pd.read_csv(tmp_path / "final_x.csv").id.tolist() == ["c", "b", "a"]
    run_funnel.finalize_variant(dict(), tmp_path, "x", src, 3, 3, T, st, False, None, {**same, "rank_rule": "mean"})
    out = pd.read_csv(tmp_path / "final_x.csv"); assert out.id.tolist() == ["a", "c", "b"] and (out.rank_rule == "mean").all()   # mean: a=0.75, c=0.74, b=0.71 (min put a last)
