"""CPU-only tests for the agent-facing layer of phbind/: config, scorer registry + validation records, validate.py decision rule, experiment log, status report."""
import json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO), str(REPO / "phbind"), str(REPO / "funnel")]
from phbind import config, scorers, validate, experiments, status  # noqa: E402


def test_defaults_are_the_campaign_settings_and_unknown_keys_are_rejected(tmp_path):
    c = config.load()
    assert c["prescreen"]["cut"] == 0.35 and c["gate"]["legs"] == ["boltz", "af3"] and c["generate"]["mpnn"] == "soluble"
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"prescreen": {"cutt": 0.4}}))                         # a typo must not silently run the default
    with pytest.raises(KeyError, match="unknown config key"):
        config.load(p)
    p.write_text(json.dumps({"prescreen": {"cut": 0.45}, "generate": {"lengths": [70, 90]}}))
    c = config.load(p)
    assert c["prescreen"]["cut"] == 0.45 and c["generate"]["lengths"] == [70, 90] and c["prescreen"]["batch"] == 72     # partial override keeps the rest


@pytest.mark.parametrize("over,msg", [({"generate": {"mpnn": "original"}}, "mpnn"), ({"generate": {"lengths": [40]}}, "60..250"),
                                      ({"gate": {"legs": ["boltz", "boltz"]}}, "TWO different models"), ({"generate": {"sets": ["nope"]}}, "no entry"),
                                      ({"generate": {"hotspot_sets": {"bad": ["87"]}, "sets": ["bad"]}}, "organiser chain")])
def test_config_refuses_the_known_bad_settings(tmp_path, over, msg):
    p = tmp_path / "c.json"; p.write_text(json.dumps(over))
    with pytest.raises((ValueError, KeyError), match=msg):
        config.load(p)


def test_registry_refuses_a_scorer_for_a_role_its_record_does_not_support():
    assert scorers.require("boltz", "screen") and scorers.require("af3", "gate") and scorers.require("ptx", "gate")
    with pytest.raises(scorers.ScorerRefused, match="no signal"):
        scorers.require("of3", "gate")                                              # measured: exactly 0.0 on every real binder
    with pytest.raises(scorers.ScorerRefused):
        scorers.require("af3", "screen")                                            # too insensitive to pre-filter
    with pytest.raises(scorers.ScorerRefused, match="no usable signal"):
        scorers.require("ptx_fast", "screen")                                       # tested on the trimer: exactly 0 for 89% of designs
    assert scorers.require("ptx_fast", "screen", force=True).prefix == "ptxf"
    import tempfile
    f = Path(tempfile.mkdtemp()) / "v.json"; f.write_text(json.dumps({"ptx": {"status": "untested"}}))
    with pytest.raises(scorers.ScorerRefused, match="no validation record"):
        scorers.require("ptx", "screen", path=f)                                    # an untested scorer is refused until validated
    assert scorers.require("ptx", "screen", path=f, allow_untested=True)
    assert scorers.require("of3", "gate", force=True)
    with pytest.raises(KeyError):
        scorers.require("alphafold9", "gate")
    with pytest.raises(ValueError):
        scorers.require("boltz", "rank")


def test_every_registered_scorer_has_a_record_and_the_contract_rejects_missing_scores():
    rec = scorers.records()
    assert set(scorers.SCORERS) == set(rec)
    sc = scorers.SCORERS["af3"]
    ok = pd.DataFrame({"id": ["a"], "af3_ipsae_min": [0.4]})
    assert scorers.check_contract(ok, sc) is ok
    with pytest.raises(ValueError, match="NaN"):
        scorers.check_contract(pd.DataFrame({"id": ["a"], "af3_ipsae_min": [np.nan]}), sc)
    with pytest.raises(ValueError, match="lacks"):
        scorers.check_contract(pd.DataFrame({"id": ["a"]}), sc)


def _vset(real, rp, rf):
    rows = [("real_binder", s) for s in real] + [("reference_pass", s) for s in rp] + [("reference_fail", s) for s in rf]
    return pd.DataFrame({"group": [g for g, _ in rows], "x_ipsae_min": [s for _, s in rows]})


def test_validate_decision_rule_on_synthetic_scorers():
    good = validate.evaluate(_vset([0.7, 0.6, 0.2], [0.8, 0.9, 0.6, 0.7, 0.85], [0.1] * 10), "x_ipsae_min")
    d = validate.decide(good); assert d["roles"] == {"screen": True, "gate": True} and not d["weak"]
    silent = validate.decide(validate.evaluate(_vset([0.0] * 4, [0.0] * 5, [0.0] * 10), "x_ipsae_min"))            # OpenFold3-like: zero on everything real
    assert silent["roles"] == {"screen": False, "gate": False} and "no signal" in silent["why"]
    permissive = validate.decide(validate.evaluate(_vset([0.9], [0.9] * 5, [0.9] * 6 + [0.1] * 4), "x_ipsae_min"))   # passes everything: not selective
    assert permissive["roles"]["gate"] is False and permissive["roles"]["screen"] is True
    weak = validate.decide(validate.evaluate(_vset([0.0] * 4, [0.6, 0.0, 0.0, 0.0, 0.0], [0.1] * 15), "x_ipsae_min"))   # AF3-like: some signal, selective, misses real binders
    assert weak["roles"]["gate"] is True and weak["weak"] is True
    with pytest.raises(ValueError, match="no 'real_binder'"):
        validate.evaluate(_vset([], [0.9], [0.1]), "x_ipsae_min")


def test_experiment_log_demands_a_measurement_and_a_decision(tmp_path):
    p = tmp_path / "e.jsonl"
    experiments.log("cut 0.35", "a lower cut keeps real passers", {"prescreen.cut": 0.35}, {"gate_passers_kept": "5/5"}, "keep", path=p)
    with pytest.raises(ValueError, match="measured result"):
        experiments.log("x", "h", {}, {}, "keep", path=p)
    with pytest.raises(ValueError, match="keep | revert"):
        experiments.log("x", "h", {}, {"n": 1}, "great", path=p)
    r = experiments.read(p); assert len(r) == 1 and r[0]["decision"] == "keep" and r[0]["config_sha"]


def test_status_counts_real_artifacts_not_directories(tmp_path):
    out = tmp_path / "phbind"; (out / "s2").mkdir(parents=True); (out / "gen" / "r1" / "pdbs").mkdir(parents=True)
    (out / "gen" / "r1" / "pdbs" / "a.pdb").write_text("x"); (out / "gen" / "r1" / "pdbs").joinpath("b.pdb").write_text("x"); (out / "gen" / "empty" / "pdbs").mkdir(parents=True)
    pd.DataFrame({"id": [f"d{i}" for i in range(6)], "seq": ["A"] * 6}).to_csv(out / "designs_all.csv", index=False)
    pd.DataFrame({"id": ["d0", "d1", "d2", "carrier_x"], "ipsae_min": [0.9, 0.4, 0.1, 0.7], "batch": 0}).to_csv(out / "s2" / "batch_000.csv", index=False)
    r = status.report(out)
    assert r["backbones_generated"] == 2 and r["designs_total"] == 6 and r["designs_screened"] == 3          # carriers are not designs; an empty pdbs/ dir counts for nothing
    assert r["survivors_at_cut"] == 2 and r["boltz_ge_0_5"] == 1 and r["last_carriers"] == {"carrier_x": 0.7}
    assert any("gate" in n for n in r["next"]) or any("prescreen" in n for n in r["next"])


def test_prescreen_order_novelty_first_failed_never_and_tiers_before_shuffle():
    from phbind import s2_prescreen as s2
    d = pd.DataFrame({"id": [f"set_L120_{b}_{k}" for b in range(6) for k in (0, 1)]}); d["k"] = d.id.str.rsplit("_", n=1).str[1].astype(int); d["seq"] = "A" * 5
    nov = pd.DataFrame({"bbid": ["set_L120_1", "set_L120_2", "set_L120_3"], "n_strict_hits": [0, 5, 0]})
    o = s2.eligible(d, set(), novelty=nov)
    assert not o.id.str.startswith("set_L120_2_").any()                           # a backbone with strict hits is never screened
    first = list(o.id[:4]); assert all(x.startswith(("set_L120_1_", "set_L120_3_")) for x in first) and all(x.endswith("_0") for x in first[:2])   # novelty-passed first, first-sequence tier before the second
    assert set(s2.eligible(d, set(), novelty=None).id) == set(d.id)                # no novelty table: nothing is dropped
    assert not set(s2.eligible(d, {"set_L120_0_0"}).id) & {"set_L120_0_0"}         # done designs are skipped
    assert len(s2.eligible(d, set(), min_length=130)) == 0 and len(s2.eligible(d, set(), max_seq_index=0)) == 6


def test_handoff_extracts_only_the_binder_chain(tmp_path):
    from phbind import handoff
    pdb = "".join(f"ATOM  {i:5d}  CA  GLY {c}{i:4d}    {float(i):8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00           C\n" for i, c in ((1, "A"), (2, "B"), (3, "C"), (4, "C")))
    out = handoff.chain_pdb(pdb, "C"); assert out.count("ATOM") == 2 and " A " not in out and out.endswith("END\n")
    with pytest.raises(ValueError): handoff.chain_pdb(pdb, "Z")
    n = handoff.write_tar({"bb1": out}, tmp_path / "x.tar.gz", "root"); import tarfile; assert n == 1 and tarfile.open(tmp_path / "x.tar.gz").getnames() == ["root/bb1.pdb"]


def _wave_tree(tmp_path, novelty=None, gated=False):
    from phbind import wave
    c = config.validate(config._merge(config.DEFAULTS, {"generate": {"hotspot_sets": {"decl8z": ["B75", "B86"], "core6z": ["B87", "B90"]}, "sets": ["decl8z", "core6z"], "lengths": [120, 132],
                                                         "backbones_per_run": 2, "seqs_per_backbone": 1, "seed_base": 300000}, "prescreen": {"min_length": 120, "max_seq_index": 0},
                                                         "wave": {"campaign": "zl1", "export_dir": str(tmp_path / "export"), "af3_cap": 2}}))
    out = tmp_path / "out/phbind"; (out / "s2").mkdir(parents=True); ids = [f"{s}_L{L}_{k}_0" for s in ("decl8z", "core6z") for L in (120, 132) for k in (0, 1)]
    for r in ("decl8z_L120", "decl8z_L132", "core6z_L120", "core6z_L132"):
        d = out / "gen" / r; d.mkdir(parents=True); pd.DataFrame({"id": [i for i in ids if i.startswith(r + "_")]}).to_csv(d / "designs.csv", index=False)
    pd.DataFrame({"id": ids, "seq": ["ACDEFGHIKL"] * len(ids)}).to_csv(out / "designs_all.csv", index=False)
    pd.DataFrame({"id": ids[:6], "ipsae_min": [0.9, 0.8, 0.7, 0.6, 0.55, 0.2], "batch": 0, "cif": "x"}).to_csv(out / "s2" / "batch_000.csv", index=False)
    if novelty is not None:
        (tmp_path / "export").mkdir(exist_ok=True); novelty.to_csv(tmp_path / "export/novelty_backbones_zl1.csv", index=False)
    return c, wave


def test_wave_plan_reads_real_artifacts_and_survivors_respect_novelty_and_the_cap(tmp_path):
    c, wave = _wave_tree(tmp_path)
    st = {s: d for s, d, _ in wave.plan(c, tmp_path)}
    assert st["generate"] is True and st["shard"] is False and st["pack"] is False and st["prescreen"] is False       # 2 designs per run complete; shard absent; 2 designs still unscreened
    sv = wave.survivors(c, tmp_path); assert list(sv.id)[:2] == ["decl8z_L120_0_0", "decl8z_L120_1_0"] and len(sv) == 2     # best first, capped at af3_cap=2
    nov = pd.DataFrame({"bbid": ["decl8z_L120_0", "decl8z_L120_1"], "n_strict_hits": [3, 0]})
    old = pd.read_csv(tmp_path / "out/phbind/s2/batch_000.csv"); extra = pd.DataFrame({"id": ["decl8z_L62_0_0", "decl8z_L120_0_5"], "ipsae_min": [0.99, 0.99], "batch": 0, "cif": "x"})
    pd.concat([old, extra]).to_csv(tmp_path / "out/phbind/s2/batch_000.csv", index=False)
    assert not {"decl8z_L62_0_0"} & set(wave.survivors(c, tmp_path).id)                      # a design from a length this wave did not configure (an earlier campaign) is never a survivor
    c, wave = _wave_tree(tmp_path / "b", novelty=nov); sv = wave.survivors(c, tmp_path / "b"); assert "decl8z_L120_0_0" not in set(sv.id) and "decl8z_L120_1_0" in set(sv.id)   # a novelty-failed backbone never reaches the AF3 leg


def test_seed_base_is_a_validated_config_key_and_separates_campaigns():
    a = config.load(); assert a["generate"]["seed_base"] == 100000
    import json as _j, tempfile
    f = Path(tempfile.mkdtemp()) / "c.json"; f.write_text(_j.dumps({"generate": {"seed_base": 300000}})); b = config.load(f)
    # PXDesign seed = seed_base + 1000*index(set) + length: two campaigns with different bases never share a seed for the same (set index, length)
    seeds = lambda base: {base + 1000 * i + L for i in range(3) for L in (120, 132, 144)}
    assert not seeds(a["generate"]["seed_base"]) & seeds(b["generate"]["seed_base"])


def test_a_wave_refuses_a_checkout_that_another_live_wave_owns(tmp_path):
    import json, os, socket
    from phbind import wave
    c = config.validate(config._merge(config.DEFAULTS, {"wave": {"campaign": "zl2"}}))
    f = tmp_path / "out/phbind/.wave_owner.json"; f.parent.mkdir(parents=True)
    f.write_text(json.dumps({"campaign": "zl1", "pid": os.getppid(), "host": socket.gethostname()}))      # a live process of ANOTHER campaign
    with pytest.raises(SystemExit, match="own clone"):
        wave.claim_checkout(c, tmp_path)
    f.write_text(json.dumps({"campaign": "zl1", "pid": 2 ** 22 + 12345, "host": socket.gethostname()}))     # a dead owner: take over
    wave.claim_checkout(c, tmp_path); assert json.loads(f.read_text())["campaign"] == "zl2"
    f.write_text(json.dumps({"campaign": "zl2", "pid": os.getppid(), "host": socket.gethostname()})); wave.claim_checkout(c, tmp_path)   # the same campaign may resume


def _poses(parent, variants, seeds=(101, 202, 303, 404, 505)):
    rows = [dict(id="P", parent="P", role="parent", seed=s, rel60=parent[i]) for i, s in enumerate(seeds)]
    for name, vals in variants.items(): rows += [dict(id=name, parent="P", role="variant", seed=s, rel60=vals[i]) for i, s in enumerate(seeds)]
    return pd.DataFrame(rows)


def test_ph_summary_is_paired_by_seed_and_the_worst_pose_check_catches_a_sign_reversal():
    from phbind import ph_score as P
    par = [-0.2, -0.3, -0.1, -0.2, -0.2]
    r = P.summarise(_poses(par, {"clean": [0.9, 1.0, 0.8, 1.1, 0.9], "mean_only": [1.6, 1.5, 1.4, 1.8, -0.3]}))
    c, m = r["variants"].loc["clean"], r["variants"].loc["mean_only"]
    assert c.supported and c.all_positive and abs(c.d_mean - 1.1) < 0.05
    assert m.d_mean > 1.0 and m.supported and not m.all_positive        # a good MEAN hides a pose that binds harder at acid pH: that is what the check is for
    assert abs(r["parents"].loc["P", "rel_mean"] + 0.2) < 1e-9 and r["parents"].loc["P", "baseline_ok"]


def test_ph_summary_refuses_unpaired_or_missing_data_and_baseline_triage_has_a_floor():
    from phbind import ph_score as P
    df = _poses([0.0] * 5, {"v": [1.0] * 5})
    with pytest.raises(ValueError, match="no parent pose at seeds"):
        P.summarise(df[~((df.role == "parent") & (df.seed == 505))])      # a variant pose with no parent at the same seed cannot be paired
    with pytest.raises(ValueError, match="NaN"):
        P.summarise(df.assign(rel60=df.rel60.where(df.seed != 101)))
    assert P.baseline_ok(-1.0) and not P.baseline_ok(-2.7)               # parents far below zero are not worth a histidine panel
    ok5 = P.summarise(_poses([0.0] * 5, {"v": [0.9, 1.0, 0.8, 1.1, 0.9]}))["variants"].loc["v"]
    few = P.summarise(_poses([0.0] * 5, {"v": [0.9, 1.0, 0.8, 1.1, 0.9]}).query("seed <= 303"))["variants"].loc["v"]
    assert ok5.supported and few.z > 5 and few.d_mean >= 0.5 and not few.supported      # same effect on 3 poses is large and significant but still not 'supported' (needs 5)


def test_ph_validation_record_states_what_it_does_not_show():
    import json
    from pathlib import Path
    rec = json.load(open(Path(__file__).resolve().parent.parent / "phbind" / "validation_ph.json"))["propka_k"]
    assert "NOT a measured pH switch" in rec["status"] and rec["limits"] and "run_to_run_sd_kcal_per_mol" in rec["measured"]


def test_ph_score_poses_worker_is_picklable_and_failures_raise(tmp_path, monkeypatch):
    import pickle, types
    from phbind import ph_score as P
    pickle.dumps(P._score_one)                  # a process pool needs this; the first version defined the worker inside score_poses and could not run in parallel
    fake = types.SimpleNamespace(score_one=lambda path, bc, tc, potts, wd, do: dict(k_status="ok" if "bad" not in path else "propka_not_installed", k_n_iface_ionisable=3, k_release_60v74=0.5, k_release_55v74=0.9))
    monkeypatch.setattr(P, "_scorer", lambda: fake)
    man = pd.DataFrame([dict(id="v", parent="p", role="variant", seed=101, file="v__seed101.pdb")])
    out = P.score_poses(man, tmp_path, workers=1)
    assert float(out.rel60.iloc[0]) == 0.5 and out.k_status.iloc[0] == "ok"
    with pytest.raises(RuntimeError, match="failed to score"):
        P.score_poses(pd.DataFrame([dict(id="v", parent="p", role="variant", seed=101, file="bad.pdb")]), tmp_path, workers=1)
