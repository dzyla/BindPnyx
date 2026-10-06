import numpy as np
import pandas as pd
import pytest
pytest.importorskip("protenix", reason="needs the full pxd environment (protenix)")

from pxdesign.runner.helpers import infer_mode_from_df, pre_filter_boltz


def _df(rows):
    return pd.DataFrame(rows)


def _row(name, seq_idx, ipsae, gate=1, batch="b1", ipdae=None):
    return {
        "name": name,
        "seq_idx": seq_idx,
        "task_name": "t",
        "sequence": "AAAA",
        "bz_ipsae": ipsae,
        "bz_ipdae": ipsae if ipdae is None else ipdae,
        "bz_gate_egfr_provisional_v1_success": gate,
        "bz_final_batch_id": batch,
    }


def test_mode_is_boltz_when_bz_columns_present():
    assert infer_mode_from_df(_df([_row("bb1", 0, 0.9)])) == "boltz"


def test_boltz_check_precedes_ptx_resolution():
    """A stray ptx column must not reroute selection to the mixed rule."""
    row = _row("bb1", 0, 0.9)
    row.update(
        {"ptx_success": 1, "ptx_basic_success": 1, "ptx_iptm": 0.5,
         "af2_easy_success": 1, "af2_opt_success": 1, "unscaled_i_pAE": 5.0}
    )
    assert infer_mode_from_df(_df([row])) == "boltz"


def test_mode_is_unchanged_without_bz_columns():
    """Regression: existing preview/extended behaviour must not shift."""
    extended = _df([{"ptx_success": 1, "ptx_basic_success": 1, "ptx_iptm": 0.5}])
    assert infer_mode_from_df(extended) == "extended"
    preview = _df([{"af2_easy_success": 1, "unscaled_i_pAE": 5.0}])
    assert infer_mode_from_df(preview) == "preview"


def test_orders_by_ranking_key_descending():
    out = pre_filter_boltz(
        _df([_row("bb1", 0, 0.6), _row("bb2", 0, 0.9), _row("bb3", 0, 0.7)]),
        min_total_return=3,
    )
    assert list(out["bz_ipsae"]) == [0.9, 0.7, 0.6]
    assert list(out["rank"]) == [1, 2, 3]


def test_ranking_key_is_configurable():
    rows = [_row("bb1", 0, 0.4, ipdae=0.95), _row("bb2", 0, 0.9, ipdae=0.10)]
    by_ipsae = pre_filter_boltz(_df(rows), min_total_return=2, ranking_key="bz_ipsae")
    by_ipdae = pre_filter_boltz(_df(rows), min_total_return=2, ranking_key="bz_ipdae")
    assert by_ipsae.iloc[0]["name"] == "bb2"
    assert by_ipdae.iloc[0]["name"] == "bb1"


def test_gate_passers_outrank_failures():
    out = pre_filter_boltz(
        _df([_row("bb1", 0, 0.95, gate=0), _row("bb2", 0, 0.30, gate=1)]),
        min_total_return=2,
    )
    assert out.iloc[0]["name"] == "bb2"
    assert list(out["bucket"]) == [1, 9]


def test_per_backbone_cap_limits_successes():
    rows = [_row("bb1", i, 0.9 - i * 0.01) for i in range(8)]
    out = pre_filter_boltz(_df(rows), min_total_return=1, per_backbone_cap=1)
    assert len(out) == 1


def test_per_backbone_cap_applies_to_padding():
    """A sparse gate is exactly when padding dominates the shortlist."""
    rows = [_row("bb1", i, 0.9 - i * 0.01, gate=0) for i in range(8)]
    rows += [_row("bb2", 0, 0.10, gate=0)]
    out = pre_filter_boltz(_df(rows), min_total_return=5, per_backbone_cap=1)
    assert out["name"].value_counts().max() == 1
    assert set(out["name"]) == {"bb1", "bb2"}


def test_ranking_refuses_mixed_final_batch():
    rows = [_row("bb1", 0, 0.9, batch="b1"), _row("bb2", 0, 0.8, batch="b2")]
    with pytest.raises(ValueError, match="bz_final_batch_id"):
        pre_filter_boltz(_df(rows), min_total_return=2)


def test_triage_allocates_within_each_batch():
    """Triage must not SORT across batches; it allocates inside each and unions,
    so no design is excluded by a cross-batch comparison."""
    rows = [_row("bb1", 0, 0.9, batch="b1"), _row("bb2", 0, 0.8, batch="b2")]
    out = pre_filter_boltz(
        _df(rows), min_total_return=2, max_success_return=2, within_batch=True
    )
    assert set(out["bz_final_batch_id"]) == {"b1", "b2"}
    assert len(out) == 2


def test_triage_separates_rank_and_gate_passes():
    """Rank-pass rows carry 1 seed and gate-pass rows 3, under different
    per-pass batch ids. Triage must allocate inside each rather than sorting a
    1-seed score against a 3-seed one."""
    rows = [
        dict(_row("bb1", 0, 0.42, batch="b1_rank"), bz_n_seeds=1),
        dict(_row("bb1", 1, 0.55, batch="b1_gate"), bz_n_seeds=3),
        dict(_row("bb2", 0, 0.38, batch="b1_rank"), bz_n_seeds=1),
        dict(_row("bb2", 1, 0.61, batch="b1_gate"), bz_n_seeds=3),
    ]
    out = pre_filter_boltz(
        _df(rows), min_total_return=2, max_success_return=2,
        per_backbone_cap=2, within_batch=True,
    )
    assert set(out["bz_final_batch_id"]) == {"b1_rank", "b1_gate"}
    assert set(out["bz_n_seeds"]) == {1, 3}


def test_triage_does_not_drop_a_whole_weak_batch():
    """A batch whose scores are uniformly lower must still get places: the
    difference may be a batch artefact, which is the entire hazard."""
    strong = [_row("s%d" % i, 0, 0.90 - i * 0.01, batch="b1") for i in range(6)]
    weak = [_row("w%d" % i, 0, 0.40 - i * 0.01, batch="b2") for i in range(6)]
    out = pre_filter_boltz(
        _df(strong + weak),
        min_total_return=4,
        max_success_return=4,
        within_batch=True,
    )
    assert set(out["bz_final_batch_id"]) == {"b1", "b2"}
    assert out["bz_final_batch_id"].value_counts()["b2"] >= 1


def test_missing_ranking_key_raises():
    df = _df([_row("bb1", 0, 0.9)]).drop(columns=["bz_ipsae"])
    with pytest.raises(ValueError, match="bz_ipsae"):
        pre_filter_boltz(df, min_total_return=1)


def test_none_scores_sort_last_and_never_rank_first():
    out = pre_filter_boltz(
        _df([_row("bb1", 0, None, gate=0), _row("bb2", 0, 0.2, gate=0)]),
        min_total_return=2,
    )
    assert out.iloc[0]["name"] == "bb2"


def test_is_idempotent_on_its_own_output():
    """The boltz dispatch calls it TWICE: bounded triage, then the final rank
    after the common batch. The second call receives rank/bucket/pass_boltz
    from the first and must not choke on them."""
    rows = [_row("bb1", 0, 0.9), _row("bb2", 0, 0.8)]
    once = pre_filter_boltz(_df(rows), min_total_return=2, max_success_return=2)
    twice = pre_filter_boltz(once, min_total_return=2, max_success_return=2)
    assert list(twice["rank"]) == [1, 2]
    assert list(twice["name"]) == list(once["name"])
    assert (twice["rank"].value_counts() == 1).all()


def test_is_idempotent_after_a_triage_union():
    """The real failure path: triage across batches returns a ranked union, and
    the final call then re-ranks it."""
    rows = [_row("bb1", 0, 0.9, batch="b1"), _row("bb2", 0, 0.8, batch="b2")]
    triage = pre_filter_boltz(
        _df(rows), min_total_return=2, max_success_return=2, within_batch=True
    )
    assert "rank" in triage.columns
    # simulate the common batch collapsing them to one id
    triage = triage.assign(bz_final_batch_id="final-1")
    final = pre_filter_boltz(triage, min_total_return=2, max_success_return=2)
    assert list(final["rank"]) == [1, 2]


def test_absent_gate_column_means_nothing_passes():
    df = _df([_row("bb1", 0, 0.9)]).drop(
        columns=["bz_gate_egfr_provisional_v1_success"]
    )
    out = pre_filter_boltz(df, min_total_return=1)
    assert list(out["bucket"]) == [9]
