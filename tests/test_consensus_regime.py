"""Pins the benchmark behind the funnel's consensus rule, and the rule implementation.

The fixture is the per-design mean-over-seeds ipSAE of three co-folding models (Boltz-2, OpenFold3, Protenix-v2) on the 1,320 wet-lab-labelled designs of the
Anthropic binder-design release (CC BY 4.0), in both directions of ipSAE. The numbers below were computed with an independent implementation; if one of these
tests moves, either the rule code changed or the fixture did - decide which before touching the pins. They document the evidence behind funnel/oracles.py; they
are an in-silico benchmark against binding labels, not a promise about any new target.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "funnel")); sys.path.insert(0, str(REPO))
import oracles  # noqa: E402

FIX = Path(__file__).parent / "fixtures" / "release_scores_b2_of3_ptx.csv"


@pytest.fixture(scope="module")
def rel():
    return pd.read_csv(FIX, comment="#")


def _auc(y, s):
    y = np.asarray(y, bool); s = np.asarray(s, float); pos, neg = s[y], s[~y]
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))


def within_target_auroc(df, s):
    """Sample-weighted mean of per-target AUROCs (targets with both classes and >=10 designs)."""
    num = den = 0.0
    for _, d in df.assign(_s=s).groupby("target"):
        if d.y.nunique() < 2 or len(d) < 10: continue
        num += _auc(d.y, d._s) * len(d); den += len(d)
    return num / den


def precision_at_10(df, s):
    """Mean over targets (>=10 designs) of the hit rate among the 10 top-scored designs."""
    return float(np.mean([d.nlargest(10, "_s").y.mean() for _, d in df.assign(_s=s).groupby("target") if len(d) >= 10]))


def test_fixture_is_the_release_slice(rel):
    assert rel.shape == (1320, 8) and rel.target.nunique() == 15 and int(rel.y.sum()) == 354 and not rel.isna().any().any()


# (metric, scorer) -> (precision@10, within-target AUROC)
PINS = {
    ("min", "b2"): (0.3867, 0.6891), ("min", "min(b2,of3)"): (0.5133, 0.7190), ("min", "mean(b2,of3)"): (0.5267, 0.7441),
    ("min", "min(b2,ptx)"): (0.3667, 0.6832), ("min", "mean(b2,ptx)"): (0.4200, 0.7240),
    ("max", "b2"): (0.4600, 0.7034), ("max", "min(b2,of3)"): (0.5600, 0.7234), ("max", "mean(b2,of3)"): (0.5733, 0.7519),
    ("max", "min(b2,ptx)"): (0.4667, 0.6985), ("max", "mean(b2,ptx)"): (0.4600, 0.7263),
}


def _score(rel, metric, scorer):
    d = rel.rename(columns={f"b2_ipsae_{metric}": "b", f"of3_ipsae_{metric}": "o", f"ptx_ipsae_{metric}": "p"})
    if scorer == "b2": return d.b
    second = "o" if "of3" in scorer else "p"; rule = scorer.split("(")[0]
    return oracles.consensus_score(d, ("b", second), rule)


@pytest.mark.parametrize("key", sorted(PINS))
def test_pinned_benchmark_numbers(rel, key):
    s = _score(rel, *key); p10, auc = PINS[key]
    assert precision_at_10(rel, s) == pytest.approx(p10, abs=2e-3)
    assert within_target_auroc(rel, s) == pytest.approx(auc, abs=5e-4)


def test_the_historical_min_rule_does_not_beat_boltz_alone_with_protenix_v2(rel):
    """The funnel's published judge (Boltz-2 x Protenix-v2, min rule, funnel's own ipsae_min) is no better than Boltz-2 alone on these labels:
    paired bootstrap over targets gave p@10 -0.020 [-0.06, +0.02] and AUROC -0.006 [-0.038, +0.026]. The pin: it is not better."""
    assert precision_at_10(rel, _score(rel, "min", "min(b2,ptx)")) <= precision_at_10(rel, _score(rel, "min", "b2")) + 1e-9


@pytest.mark.parametrize("metric,second", [("min", "of3"), ("max", "of3"), ("min", "ptx"), ("max", "ptx")])
def test_mean_ranks_better_than_min_for_every_second_oracle(rel, metric, second):
    """Ranking by the mean of the two ipSAE values beats ranking by the worst one (within-target AUROC; bootstrap CI for Boltz-2 + OpenFold3: +0.025
    [+0.011, +0.041] on ipsae_min). Keep `min` for PASS/FAIL gates; use `mean` to order candidates."""
    assert within_target_auroc(rel, _score(rel, metric, f"mean(b2,{second})")) > within_target_auroc(rel, _score(rel, metric, f"min(b2,{second})"))


def test_consensus_score_rules_and_nan_handling():
    d = pd.DataFrame({"a": [0.8, 0.6, np.nan, 0.4], "b": [0.4, 0.6, 0.9, np.nan]})
    assert oracles.consensus_score(d, ("a", "b"), "min").tolist()[:2] == [0.4, 0.6]
    assert oracles.consensus_score(d, ("a", "b"), "mean").tolist()[:2] == pytest.approx([0.6, 0.6])
    for rule in ("min", "mean", "zmean"):                       # a design that lacks one judge has NO consensus (it used to inherit the other judge's score)
        s = oracles.consensus_score(d, ("a", "b"), rule)
        assert s.isna().tolist() == [False, False, True, True], rule
    with pytest.raises(ValueError): oracles.consensus_score(d, ("a", "b"), "median")
