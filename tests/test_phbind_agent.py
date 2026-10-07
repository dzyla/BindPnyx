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
