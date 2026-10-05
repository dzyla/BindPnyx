"""The epitope policy reaches selection, and selection obeys it.

CLAUDE.md s7.1 recorded three severed links between a declared policy and what
ships. Verified live on 2026-10-02: all 55 rows of a real rescore came back
`policy_status='not_applicable'`, which PERMITS eligibility - so the policy was
configured, reported absent, and enforced on nothing.

The acceptance criteria these tests encode:

  * a high-scoring OFF-SITE design loses its backbone's confirmation slot to a
    lower-scoring eligible ON-SITE sibling;
  * no failed or unevaluable design reaches export, INCLUDING through padding,
    which is the path that previously shipped them.
"""
import numpy as np
import pandas as pd
import pytest

from pxdbench.targets.eligibility import (
    POLICY_V1,
    PolicyTransportError,
    SelectionContext,
    assert_policy_config_consistent,
    resolve_eligibility,
)
from pxdbench.tools.boltz.backend import BoltzBackend
from pxdesign.runner.helpers import BOLTZ_GATE_COL, pre_filter_boltz

POLICY = {"required": {"A": [10, 11, 12]}, "cutoff": 5.0}


def _backend(policy):
    return BoltzBackend(cfg={"epitope_policy": policy}, device="cpu")


# --- the manifest fact -------------------------------------------------------

def test_a_declared_policy_is_seen_as_enforced():
    assert _backend(POLICY)._has_enforced_requirements() is True


def test_an_absent_policy_is_not_enforced():
    assert _backend({})._has_enforced_requirements() is False
    assert _backend(None)._has_enforced_requirements() is False


def test_enforcement_is_not_inferred_from_a_result_value():
    """ep_satisfied=None means both 'no policy' and 'unevaluable', so the
    verdict must come from config (metrics/epitope.py:180)."""
    b = _backend(POLICY)
    assert b._policy_eligible_names({"d_seq0": {"ep_satisfied": None}}) == set()


# --- safeguard 1: contradictory configuration is refused ---------------------

def test_required_residues_without_enforcement_is_refused():
    with pytest.raises(PolicyTransportError, match="not_applicable"):
        assert_policy_config_consistent(POLICY, False)


def test_enforcement_without_required_residues_is_refused():
    with pytest.raises(PolicyTransportError, match="unevaluable"):
        assert_policy_config_consistent({}, True)


def test_a_consistent_configuration_passes():
    assert_policy_config_consistent(POLICY, True) is None
    assert_policy_config_consistent({}, False) is None


def test_not_applicable_under_declared_enforcement_is_a_transport_error():
    """The exact live defect: a declared policy, and every row claiming it did
    not apply. not_applicable permits eligibility, so these would all ship."""
    df = pd.DataFrame({"policy_status": ["not_applicable"] * 3})
    ctx = SelectionContext(mode=POLICY_V1, enforced=True)
    with pytest.raises(PolicyTransportError, match="3 of 3"):
        resolve_eligibility(df, ctx)


def test_not_applicable_is_fine_when_nothing_was_declared():
    df = pd.DataFrame({"policy_status": ["not_applicable"] * 3})
    out = resolve_eligibility(df, SelectionContext(mode=POLICY_V1))
    assert out["eligible"].all()


# --- safeguard 2: eligibility before the confirmation budget -----------------

def _pair():
    """Two designs on ONE backbone: the off-site one scores higher."""
    data_list = [
        {"name": "bb1", "seq_idx": 0},   # off-site, high score
        {"name": "bb1", "seq_idx": 1},   # on-site, lower score
    ]
    scored = {
        "bb1_seq0": {"bz_ipsae": 0.90, "policy_status": "fail"},
        "bb1_seq1": {"bz_ipsae": 0.40, "policy_status": "pass"},
    }
    return data_list, scored


def test_the_off_site_design_wins_when_the_policy_is_not_enforced():
    """Guards the legacy path: without a policy nothing changes."""
    data_list, scored = _pair()
    got = BoltzBackend._argmax_per_backbone(data_list, scored, "bz_ipsae")
    assert got == {"bb1_seq0"}


def test_a_high_scoring_off_site_design_loses_to_an_eligible_sibling():
    """THE ACCEPTANCE TEST. 0.90 off-site must not take the slot from 0.40
    on-site: the confirmation budget is per backbone and cannot be given back."""
    data_list, scored = _pair()
    eligible = _backend(POLICY)._policy_eligible_names(scored)
    assert eligible == {"bb1_seq1"}
    got = BoltzBackend._argmax_per_backbone(
        data_list, scored, "bz_ipsae", eligible
    )
    assert got == {"bb1_seq1"}, "the off-site sibling consumed the slot"


def test_an_unevaluable_design_does_not_consume_the_slot():
    data_list = [{"name": "bb1", "seq_idx": 0}]
    scored = {"bb1_seq0": {"bz_ipsae": 0.99, "policy_status": "unevaluable"}}
    eligible = _backend(POLICY)._policy_eligible_names(scored)
    assert BoltzBackend._argmax_per_backbone(
        data_list, scored, "bz_ipsae", eligible) == set()


# --- the export path, including padding --------------------------------------

def _frame():
    """Four designs, all failing the quality gate, so selection MUST pad."""
    return pd.DataFrame({
        "name": ["bb1", "bb2", "bb3", "bb4"],
        "bz_ipsae": [0.90, 0.80, 0.70, 0.60],
        "policy_status": ["fail", "unevaluable", "pass", "fail"],
        BOLTZ_GATE_COL: [False, False, False, False],
        "bz_final_batch_id": ["b1"] * 4,
    })


def test_padding_previously_shipped_policy_failures():
    """Legacy behaviour, kept byte-exact: this is the defect, not a regression."""
    out = pre_filter_boltz(_frame(), min_total_return=4, per_backbone_cap=1)
    assert set(out["policy_status"]) == {"fail", "unevaluable", "pass"}


def test_no_failed_or_unevaluable_design_reaches_export_through_padding():
    """THE ACCEPTANCE TEST. Padding is the path that shipped them before."""
    ctx = SelectionContext(mode=POLICY_V1, enforced=True)
    out = pre_filter_boltz(
        _frame(), min_total_return=4, per_backbone_cap=1,
        mode=POLICY_V1, context=ctx,
    )
    assert list(out["policy_status"]) == ["pass"]
    assert set(out["name"]) == {"bb3"}, "an ineligible design was padded in"


def test_padding_may_still_draw_on_eligible_quality_failures():
    """Eligibility is mandatory; the quality gate is not. An on-site design
    that misses the gate is still a legitimate candidate to pad with."""
    df = _frame()
    df["policy_status"] = ["pass"] * 4
    out = pre_filter_boltz(
        df, min_total_return=3, per_backbone_cap=1,
        mode=POLICY_V1, context=SelectionContext(mode=POLICY_V1, enforced=True),
    )
    assert len(out) == 3
    assert not out["pass_boltz"].any()


# --- selection respects measurement uncertainty ------------------------------

from pxdbench.tools.boltz.backend import (  # noqa: E402
    GATE_TAG,
    IPSAE_TIE_TOLERANCE,
    RANK_TAG,
)


def _siblings(score_a, score_b, eng_a, eng_b):
    data_list = [{"name": "bb1", "seq_idx": 0}, {"name": "bb1", "seq_idx": 1}]
    scored = {
        "bb1_seq0": {"bz_ipsae": score_a, "ep_best_patch_frac": eng_a},
        "bb1_seq1": {"bz_ipsae": score_b, "ep_best_patch_frac": eng_b},
    }
    return data_list, scored


def test_within_the_noise_band_engagement_breaks_the_tie():
    """0.90 vs 0.88 is a 0.02 difference against a 0.03 noise floor - the
    oracle cannot tell them apart, so the better-engaging design wins."""
    data_list, scored = _siblings(0.90, 0.88, 0.10, 0.80)
    got = BoltzBackend._argmax_per_backbone(data_list, scored, "bz_ipsae")
    assert got == {"bb1_seq1"}


def test_outside_the_noise_band_the_score_still_decides():
    """0.90 vs 0.70 is a real difference; engagement must NOT override it."""
    data_list, scored = _siblings(0.90, 0.70, 0.10, 0.80)
    got = BoltzBackend._argmax_per_backbone(data_list, scored, "bz_ipsae")
    assert got == {"bb1_seq0"}


def test_an_unmeasured_engagement_does_not_win_a_tie():
    data_list, scored = _siblings(0.90, 0.89, 0.50, None)
    assert BoltzBackend._argmax_per_backbone(
        data_list, scored, "bz_ipsae") == {"bb1_seq0"}


def test_the_tie_tolerance_is_the_measured_one():
    assert IPSAE_TIE_TOLERANCE == 0.03


def test_a_zero_tolerance_restores_the_old_behaviour():
    data_list, scored = _siblings(0.90, 0.88, 0.10, 0.80)
    got = BoltzBackend._argmax_per_backbone(
        data_list, scored, "bz_ipsae", tie_tolerance=0.0)
    assert got == {"bb1_seq0"}


# --- confirmation seeds ------------------------------------------------------

def test_the_default_seed_roster_is_unchanged():
    b = BoltzBackend(cfg={}, device="cpu")
    assert b._seed_list(1, RANK_TAG) == [1]
    assert b._seed_list(3, GATE_TAG) == [1, 2, 3]


def test_confirmation_can_withhold_the_screening_seeds():
    """Seed 1 screens AND confirms by default, so a winner partly confirms
    itself. This makes the gate pass an independent test."""
    b = BoltzBackend(
        cfg={"seeds_rank": 1, "disjoint_confirmation_seeds": True},
        device="cpu",
    )
    rank = b._seed_list(1, RANK_TAG)
    gate = b._seed_list(3, GATE_TAG)
    assert rank == [1]
    assert gate == [2, 3, 4]
    assert not set(rank) & set(gate), "confirmation reused a screening seed"


def test_an_explicit_roster_overrides_everything():
    b = BoltzBackend(cfg={"seeds_gate_list": [7, 11]}, device="cpu")
    assert b._seed_list(3, GATE_TAG) == [7, 11]


# --- the provenance record must survive a real policy -------------------------

def test_a_config_wrapped_policy_serialises():
    """The first campaign ever to carry a real policy died HERE - after the GPU
    work, writing provenance - because the config system hands back a
    ConfigDict and json.dump refuses it."""
    import json

    from pxdbench.tools.boltz.backend import _jsonable

    try:
        import ml_collections as mc
    except ImportError:            # pragma: no cover
        pytest.skip("ml_collections not installed")
    cfg = mc.ConfigDict({"required": {"B": [24]}, "cutoff": 5.0})
    out = json.loads(json.dumps(_jsonable({"epitope_policy": cfg})))
    assert out["epitope_policy"]["required"]["B"] == [24]
    assert out["epitope_policy"]["cutoff"] == 5.0


def test_jsonable_leaves_plain_data_alone():
    from pxdbench.tools.boltz.backend import _jsonable

    value = {"a": [1, 2.5, "x", None, True], "b": {"c": 3}}
    assert _jsonable(value) == value


def test_jsonable_coerces_numpy_scalars():
    from pxdbench.tools.boltz.backend import _jsonable

    assert _jsonable({"x": np.float64(1.5)}) == {"x": 1.5}
    assert isinstance(_jsonable({"n": np.int64(3)})["n"], int)
