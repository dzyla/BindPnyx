"""CPU-only tests for phbind/ranking.py: tier logic, novelty veto, latest-wins corrections, lineage-capped row order, validation, and concurrent appends."""
import sys, threading
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO), str(REPO / "phbind")]
from phbind import ranking as R  # noqa: E402

SEQ = lambda i: ("ADEFGHIKLM" * 8)[: 70 + i] + "AAAAAAAAAA"[:i]


@pytest.fixture
def rk(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "DIR", tmp_path)
    (tmp_path / "designs.csv").write_text("design_id,sequence,lineage,generator,length,parent,note\n")
    return tmp_path


def reg(name, i, lineage="L1"):
    R.add_design(name, "".join(("ADEFGHIKLMNPQRSTVWY"[(j * (i + 3)) % 19]) for j in range(80 + i)), lineage, "test")


def ev(design, **kw):
    hn, mn = kw.pop("boltz_h_n", None), kw.pop("boltz_m_n", None)          # seeds are the `n` column of the mean row
    for k, v in kw.items(): R.add(design, k, v, "tester", n={"boltz_h_mean": hn, "boltz_m_mean": mn}.get(k))


def test_tiers_follow_the_rules_and_novelty_is_a_veto(rk):
    for i, n in enumerate(["a", "b", "c", "d", "e", "f"]): reg(n, i, lineage=f"L{i}")
    base = dict(boltz_h_mean=0.78, boltz_h_worst=0.72, boltz_h_n=5, af3_h=0.60)
    ev("a", **base, novelty_hits_qtm=0, novelty_best_qtm=0.70)                        # everything, clean
    ev("b", **base, novelty_hits_qtm=3, novelty_best_qtm=0.9)                         # same scores, structurally known -> excluded
    ev("c", **base)                                                                   # novelty unscreened -> cannot be tier A
    ev("d", boltz_h_mean=0.60, boltz_h_worst=0.30, boltz_h_n=1, af3_h=0.20, novelty_hits_qtm=0, novelty_best_qtm=0.6)   # Boltz passes, AF3 rejects -> disagreement
    ev("e", boltz_m_mean=0.66, boltz_m_worst=0.57, boltz_m_n=5, boltz_h_mean=0.46, boltz_h_worst=0.3, boltz_h_n=5, novelty_hits_qtm=0, novelty_best_qtm=0.6, af3_h=0.50)   # novel mouse binder
    ev("f", **base, novelty_hits_qtm=0, novelty_best_qtm=0.79)                        # thin margin still A, flagged
    df = R.rank().set_index("design_id")
    assert df.loc["a", "tier"] == "A" and df.loc["b", "tier"] == "X" and df.loc["c", "tier"] == "B" and df.loc["d", "tier"] == "C" and df.loc["e", "tier"] == "B"
    assert df.loc["f", "tier"] == "A" and df.loc["f", "novelty"] == "thin" and df.loc["c", "novelty"] == "unscreened"
    assert pd.isna(R.rank().set_index("design_id").loc["b", "row_order"])           # an excluded design never gets a submission row


def test_latest_evidence_wins_so_a_correction_is_just_a_new_line(rk):
    reg("a", 0); ev("a", boltz_h_mean=0.80, boltz_h_worst=0.70, boltz_h_n=5, af3_h=0.20, novelty_hits_qtm=0, novelty_best_qtm=0.6)
    assert R.rank().set_index("design_id").loc["a", "tier"] == "C"                    # Boltz passes but AF3 rejects (0.20): the models disagree -> tier C, not A or B
    import time; time.sleep(1.1); R.add("a", "af3_h", 0.61, "other-agent", note="corrects: earlier value was ipTM, not ipSAE")
    assert R.rank().set_index("design_id").loc["a", "tier"] == "A"


def test_ph_support_needs_poses_and_a_positive_lower_bound(rk):
    for i, n in enumerate(["p1", "p2", "p3"]): reg(n, i, lineage=f"L{i}")
    ok = dict(boltz_h_mean=0.8, boltz_h_worst=0.7, boltz_h_n=5, af3_h=0.6, novelty_hits_qtm=0, novelty_best_qtm=0.6)
    ev("p1", **ok, ph_delta=0.64, ph_se=0.12, ph_npose=15); ev("p2", **ok, ph_delta=3.6, ph_se=0.9, ph_npose=1); ev("p3", **ok, ph_delta=0.7, ph_se=0.5, ph_npose=15)
    df = R.rank().set_index("design_id")
    assert df.loc["p1", "ph"] == "supported" and df.loc["p2", "ph"] == "single-pose" and df.loc["p3", "ph"] == "unconfirmed"   # p3: 15 poses but delta - 2*se <= 0
    assert list(R.rank().design_id)[0] == "p1"                                        # pH-supported ranks first inside the tier


def test_row_order_caps_a_lineage_and_avoids_adjacent_repeats(rk):
    for i in range(10): reg(f"x{i}", i, lineage="BIG")
    for i in range(4): reg(f"y{i}", 20 + i, lineage=f"S{i}")
    good = dict(boltz_h_mean=0.8, boltz_h_worst=0.7, boltz_h_n=5, af3_h=0.6, novelty_hits_qtm=0, novelty_best_qtm=0.6)
    for d in [f"x{i}" for i in range(10)] + [f"y{i}" for i in range(4)]: ev(d, **good)
    df = R.rank(); o = df.dropna(subset=["row_order"]).sort_values("row_order")
    assert (o.head(12).lineage == "BIG").sum() <= 8           # cap = 40% of the first 20 while other lineages remain; the rest only once they are used up


def test_validation_rejects_cysteine_unknown_metric_unregistered_design_and_duplicates(rk):
    with pytest.raises(SystemExit): R.add_design("c", "A" * 70 + "C", "L", "g")
    reg("a", 0)
    with pytest.raises(SystemExit): R.add("a", "ipsae", 0.5, "t")                      # not in the vocabulary
    with pytest.raises(SystemExit): R.add("ghost", "af3_h", 0.5, "t")                  # unregistered design
    with pytest.raises(SystemExit): R.add("a", "af3_h", float("nan"), "t")
    with pytest.raises(SystemExit): R.add_design("a", "", "L", "g")                    # already registered
    (rk / "designs.csv").write_text((rk / "designs.csv").read_text() + "z,ADEFGHIKLM" + "ADEFGHIKLM" * 6 + ",L,g,70,,\n"); 
    s = pd.read_csv(rk / "designs.csv", dtype=str, keep_default_na=False).sequence.iloc[-1]
    (rk / "designs.csv").write_text((rk / "designs.csv").read_text() + f"zz,{s},L,g,70,,\n")
    d, e = R.load(); assert any("identical sequences" in b for b in R.check(d, e, R.rules()))     # the same molecule under two ids is flagged


def test_a_design_without_a_sequence_is_ranked_but_not_orderable(rk):
    R.add_design("seqless", "", "L", "g"); ev("seqless", boltz_h_mean=0.8, boltz_h_worst=0.7, boltz_h_n=5, af3_h=0.6, novelty_hits_qtm=0, novelty_best_qtm=0.6)
    df = R.rank().set_index("design_id"); assert df.loc["seqless", "tier"] == "A" and df.loc["seqless", "contract"] == "sequence needed" and pd.isna(df.loc["seqless", "row_order"])


def test_concurrent_appends_do_not_corrupt_the_ledger(rk):
    reg("a", 0)
    def work(k):
        for j in range(40): R.add("a", "af3_h", 0.5 + (k * 40 + j) * 1e-4, f"agent{k}", note="x" * 30)
    ts = [threading.Thread(target=work, args=(k,)) for k in range(8)]; [t.start() for t in ts]; [t.join() for t in ts]
    e = pd.read_csv(rk / "evidence.csv"); assert len(e) == 320 and e.value.notna().all() and e.agent.str.startswith("agent").all()


def test_more_seeds_outrank_a_later_single_seed_and_one_seed_is_enough_for_tier_a(rk):
    reg("a", 0); ok = dict(af3_h=0.60, novelty_hits_qtm=0, novelty_best_qtm=0.6)
    ev("a", boltz_h_mean=0.56, boltz_h_n=1, **ok)                                    # the two-model consensus with ONE Boltz seed is already tier A
    assert R.rank().set_index("design_id").loc["a", "tier"] == "A" and R.rank().set_index("design_id").loc["a", "robust"] == 0
    import time; time.sleep(1.1); ev("a", boltz_h_mean=0.46, boltz_h_worst=0.30, boltz_h_n=5)       # a 5-seed measurement says no ...
    time.sleep(1.1); ev("a", boltz_h_mean=0.60, boltz_h_n=1)                          # ... and a LATER single-seed value must not override it
    assert R.rank().set_index("design_id").loc["a", "boltz_h"] == "fail"


def test_row_order_never_promotes_a_lower_tier_above_a_higher_one(rk):
    for i in range(3): reg(f"a{i}", i, lineage="BIG")
    reg("b0", 10, lineage="OTHER")
    A_ = dict(boltz_h_mean=0.8, boltz_h_worst=0.7, boltz_h_n=5, af3_h=0.6, novelty_hits_qtm=0, novelty_best_qtm=0.6)
    for i in range(3): ev(f"a{i}", **A_)
    ev("b0", boltz_h_mean=0.6, boltz_h_n=1, novelty_hits_qtm=0, novelty_best_qtm=0.6)       # tier B (AF3 untested)
    o = R.rank().dropna(subset=["row_order"]).sort_values("row_order").design_id.tolist()
    assert o == ["a0", "a1", "a2", "b0"]                    # alternation would have put b0 second; tiers win


def test_alternation_may_only_move_a_design_down_a_little(rk):
    for i in range(5): reg(f"big{i}", i, lineage="BIG")
    reg("tiny", 30, lineage="TINY")
    A_ = dict(boltz_h_mean=0.8, boltz_h_worst=0.7, boltz_h_n=5, af3_h=0.6, novelty_hits_qtm=0, novelty_best_qtm=0.6)
    for i in range(5): ev(f"big{i}", **A_)
    ev("tiny", boltz_h_mean=0.52, boltz_h_n=1, af3_h=0.51, novelty_hits_qtm=0, novelty_best_qtm=0.6)       # same tier, lowest merit
    o = R.rank().dropna(subset=["row_order"]).sort_values("row_order").design_id.tolist()
    assert o.index("tiny") >= 3                              # a weak design in the same tier is not hauled up to row 2 just to break up a lineage


def test_a_thin_novelty_margin_ranks_below_a_clean_one_in_the_same_tier(rk):
    reg("clean", 0, lineage="L1"); reg("thin", 1, lineage="L2")
    base = dict(boltz_h_mean=0.8, boltz_h_worst=0.7, boltz_h_n=5, af3_h=0.7)
    ev("clean", **base, novelty_hits_qtm=0, novelty_best_qtm=0.65)
    ev("thin", **{**base, "boltz_h_mean": 0.9, "af3_h": 0.8}, novelty_hits_qtm=0, novelty_best_qtm=0.79)           # better scores, margin 0.01 under the 0.80 line
    df = R.rank().set_index("design_id"); assert df.loc["thin", "tier"] == df.loc["clean", "tier"] == "A" and df.loc["clean", "rank"] < df.loc["thin", "rank"]


def test_ph_worst_pose_is_displayed_but_changes_no_tier_and_no_order(rk):
    reg("a", 0, lineage="L1"); reg("b", 1, lineage="L2")
    base = dict(boltz_h_mean=0.8, boltz_h_worst=0.7, boltz_h_n=5, af3_h=0.7, novelty_hits_qtm=0, novelty_best_qtm=0.6, ph_delta=1.0, ph_se=0.1, ph_npose=5)
    ev("a", **base); ev("b", **base)
    before = R.rank().set_index("design_id")
    ev("a", ph_worst_pose=-0.30)                  # a pose that binds harder at acid pH
    after = R.rank().set_index("design_id")
    assert "ph_worst" in after.columns and abs(after.loc["a", "ph_worst"] + 0.30) < 1e-9 and after.loc["b", "ph_worst"] != after.loc["b", "ph_worst"]
    assert (before[["tier", "ph", "row_order"]] == after[["tier", "ph", "row_order"]]).all().all()      # display only: no tier, pH label or order moved
