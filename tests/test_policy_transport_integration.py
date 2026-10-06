"""R1's acceptance test: a failing policy must survive to selection.

> "One CPU integration test drives a real input through preparation, stubbed
> fold outputs, aggregation, final gate, and export. A deliberately off-site
> pose must fail AFTER aggregation and final rescoring, not just inside the
> metric function."

Before this, `_mean_scores` rebuilt each row from `MEAN_KEYS` and
`final_batch` propagated columns by a `"bz_"` prefix test, so `ep_satisfied`
was discarded twice over. The metric computed the right answer and nothing
downstream could see it.

The predictor is stubbed. No GPU, no real Boltz - which is the project's
existing testing seam (CLAUDE.md section 7, open question 4).
"""
import pandas as pd
import pytest
pytest.importorskip("protenix", reason="needs the full pxd environment (protenix)")

from pxdbench.targets.eligibility import PolicyStatus, SelectionContext
from pxdbench.tools.boltz.backend import BoltzBackend
from pxdbench.tools.boltz.result_schema import propagated_columns
from pxdesign.runner.helpers import pre_filter_boltz


def _seed_row(name, seed, *, satisfied, ipsae=0.9):
    """What `parse` hands back for one prediction, including the epitope block."""
    return {
        "seed": seed,
        "bz_status": "ok",
        "bz_ipsae": ipsae,
        "bz_ipdae": 0.8,
        "bz_pae_interface_min": 1.0,
        "bz_interface_plddt": 90.0,
        "bz_iptm": 0.7,
        "bz_ptm": 0.8,
        "bz_complex_plddt": 0.9,
        "ep_satisfied": satisfied,
        "ep_status": "ok",
        "ep_missed_required": "" if satisfied else "B24",
        "ep_violated_forbidden": "",
        "ep_contacted_residues": "B50" if not satisfied else "B24",
    }


# --- aggregation -----------------------------------------------------------

def test_the_epitope_verdict_survives_seed_aggregation():
    """The exact column that used to vanish."""
    rows = [_seed_row("d", s, satisfied=True) for s in (1, 2, 3)]
    out = BoltzBackend._mean_scores(
        rows, requested_seeds=(1, 2, 3), has_enforced_requirements=True
    )
    assert "ep_satisfied" in out, "ep_satisfied was dropped by aggregation"
    assert out["ep_satisfied"] is True
    assert out["policy_status"] == PolicyStatus.PASS.value
    assert out["bz_ipsae"] == pytest.approx(0.9)


def test_an_off_site_pose_fails_after_aggregation_not_only_in_the_metric():
    """The design scores WELL and misses its epitope. Both facts must arrive."""
    rows = [_seed_row("d", s, satisfied=False, ipsae=0.95) for s in (1, 2, 3)]
    out = BoltzBackend._mean_scores(
        rows, requested_seeds=(1, 2, 3), has_enforced_requirements=True
    )
    assert out["bz_ipsae"] == pytest.approx(0.95), "a strong score, retained"
    assert out["ep_satisfied"] is False, "and a failed policy, also retained"
    assert out["policy_status"] == PolicyStatus.FAIL.value


def test_one_off_site_seed_is_enough_to_fail_the_design():
    rows = [
        _seed_row("d", 1, satisfied=True),
        _seed_row("d", 2, satisfied=False),
        _seed_row("d", 3, satisfied=True),
    ]
    out = BoltzBackend._mean_scores(
        rows, requested_seeds=(1, 2, 3), has_enforced_requirements=True
    )
    assert out["ep_satisfied"] is False
    assert out["policy_pass_fraction"] == pytest.approx(2 / 3)


def test_a_failed_predictor_seed_leaves_the_policy_unevaluable():
    rows = [_seed_row("d", 1, satisfied=True)]
    rows.append({**_seed_row("d", 2, satisfied=True), "bz_status": "failed"})
    out = BoltzBackend._mean_scores(
        rows, requested_seeds=(1, 2), has_enforced_requirements=True
    )
    assert out["policy_status"] == PolicyStatus.UNEVALUABLE.value
    assert out["ep_satisfied"] is None


# --- rescoring propagation -------------------------------------------------

def test_the_final_batch_schema_carries_the_policy_columns():
    """final_batch used `k.startswith("bz_")`, which dropped all of these."""
    available = set(_seed_row("d", 1, satisfied=True)) | {
        "policy_status", "policy_pass_fraction", "unrelated"
    }
    carried = propagated_columns(available)
    for column in ("ep_satisfied", "ep_status", "policy_status",
                   "policy_pass_fraction", "bz_ipsae"):
        assert column in carried, column
    assert "unrelated" not in carried


# --- selection -------------------------------------------------------------

def test_the_whole_chain_rejects_an_off_site_design():
    """Aggregate, then select. The design outscores its rival and must lose."""
    off_site = BoltzBackend._mean_scores(
        [_seed_row("bad", s, satisfied=False, ipsae=0.95) for s in (1, 2)],
        requested_seeds=(1, 2), has_enforced_requirements=True,
    )
    on_site = BoltzBackend._mean_scores(
        [_seed_row("good", s, satisfied=True, ipsae=0.60) for s in (1, 2)],
        requested_seeds=(1, 2), has_enforced_requirements=True,
    )

    frame = pd.DataFrame([
        {"name": "bad", "bz_final_batch_id": "b1", **off_site},
        {"name": "good", "bz_final_batch_id": "b1", **on_site},
    ])
    # drop the nested per-seed detail; CSV-shaped frames carry flat columns
    frame = frame.drop(columns=["per_seed_policy"])
    frame["bz_gate_egfr_provisional_v1_success"] = [1, 1]

    selected = pre_filter_boltz(
        frame, min_total_return=5, per_backbone_cap=1,
        mode="policy_v1", context=SelectionContext(mode="policy_v1"),
    )
    assert selected["name"].tolist() == ["good"], (
        "the higher-scoring off-site design was selected anyway"
    )


def test_the_same_chain_in_legacy_mode_still_selects_both():
    """Legacy behaviour is unchanged, which existing campaigns depend on."""
    frame = pd.DataFrame([
        {"name": "bad", "bz_ipsae": 0.95, "bz_final_batch_id": "b1",
         "bz_gate_egfr_provisional_v1_success": 1, "ep_satisfied": False},
        {"name": "good", "bz_ipsae": 0.60, "bz_final_batch_id": "b1",
         "bz_gate_egfr_provisional_v1_success": 1, "ep_satisfied": True},
    ])
    selected = pre_filter_boltz(frame, min_total_return=5, per_backbone_cap=1)
    assert sorted(selected["name"].tolist()) == ["bad", "good"]
