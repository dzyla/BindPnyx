"""Adjustable selection gates with a shipped basic set.

Shape borrowed from BindCraft's `settings_filters/*.json`, with three limits
from review A6:

  1. this pipeline's OWN metric catalog - not BindCraft's inventory or its
     numbers, because its schema happens to be convenient;
  2. exact operators retained - `higher` is derived direction metadata, not the
     comparison, because `<` and `<=` are different predicates at the boundary
     and the existing filter tests rely on that;
  3. mandatory validity sits OUTSIDE preset deactivation - `none` may disable
     optional quality filters but can never make an invalid candidate eligible.

Only `default` and `none` ship. A6 forbids inventing calibration to make
presets differ, and the only thresholds with any provenance here are the legacy
gate's three - themselves provisional, inherited from a three-term gate whose
dropped clause strictly loosened it. Adding a preset later is a data change.
"""
import pytest

from pxdbench.targets.presets import (
    METRIC_CATALOG,
    load_preset,
    resolve_gate,
)


def test_only_default_and_none_ship():
    """No invented tiers. A6: do not fabricate calibration to differ."""
    from pxdbench.targets.presets import available_presets
    assert sorted(available_presets()) == ["default", "none"]


def test_every_preset_carries_the_complete_metric_set():
    """A disabled clause must be visible as `threshold: null`, so that a typo
    and a deliberate disable are distinguishable."""
    sets = [set(load_preset(n).all_metrics()) for n in ("default", "none")]
    assert sets[0] == sets[1]
    assert sets[0] == set(METRIC_CATALOG)


def test_null_threshold_means_deliberately_inactive():
    none = load_preset("none")
    assert none.all_metrics(), "metrics are still listed"
    assert none.thresholds == {}, "but no clause is active"


def test_default_reproduces_the_legacy_gate_thresholds():
    """The only numbers here with provenance."""
    assert load_preset("default").thresholds == {
        "bz_n_seeds": (">=", 3),
        "bz_pae_interface_min": ("<=", 2.0),
        "bz_ipsae": (">=", 0.5),
    }


def test_exact_operators_survive_migration():
    """A6: `higher` is direction, not strictness."""
    for metric, (op, _thr) in load_preset("default").thresholds.items():
        assert op in (">", ">=", "<", "<="), (metric, op)


def test_direction_is_derived_from_the_operator():
    default = load_preset("default")
    assert default.higher_is_better("bz_ipsae") is True
    assert default.higher_is_better("bz_pae_interface_min") is False


def test_mandatory_validity_is_outside_preset_deactivation():
    """A6: `none` disables optional quality filters, never required validity."""
    assert load_preset("none").mandatory_validity is True
    assert load_preset("default").mandatory_validity is True


def test_shipped_presets_declare_themselves_provisional():
    """No labelled set is on this machine; a preset must not imply otherwise."""
    for name in ("default", "none"):
        preset = load_preset(name)
        assert preset.provisional is True
        assert preset.calibrated_on


def test_unknown_preset_names_the_available_ones():
    with pytest.raises(KeyError, match="default"):
        load_preset("strict")


# --- resolution -------------------------------------------------------------

def test_resolve_gate_refuses_a_list_of_preset_names():
    """A product decision, not an impossibility: A6 notes an override with an
    explicit null could express a deactivation."""
    with pytest.raises(ValueError, match="one preset"):
        resolve_gate(preset=["default", "none"])


def test_campaign_overrides_win_and_are_recorded():
    gate = resolve_gate(preset="default", overrides={"bz_ipsae": (">=", 0.72)})
    assert gate.thresholds["bz_ipsae"] == (">=", 0.72)
    assert "bz_ipsae" in gate.overridden


def test_an_override_with_an_explicit_null_deactivates_a_clause():
    gate = resolve_gate(preset="default", overrides={"bz_ipsae": None})
    assert "bz_ipsae" not in gate.thresholds
    assert "bz_ipsae" in gate.overridden


def test_a_metric_outside_the_catalog_is_rejected():
    """A typo must not silently become a weaker gate."""
    with pytest.raises(ValueError, match="not in the metric catalog"):
        resolve_gate(preset="default", overrides={"bz_nonexistent": (">=", 1.0)})


def test_a_clause_whose_column_will_not_be_computed_is_rejected():
    """compute_success_rate makes a missing active metric UNKNOWN, so an
    unsatisfiable clause silently makes every row unknown rather than erroring."""
    with pytest.raises(ValueError, match="will not be computed"):
        resolve_gate(
            preset="default",
            available_columns={"bz_ipsae", "bz_n_seeds"},  # no bz_pae_interface_min
        )


def test_resolution_succeeds_when_every_active_column_is_available():
    gate = resolve_gate(
        preset="default",
        available_columns={"bz_ipsae", "bz_n_seeds", "bz_pae_interface_min"},
    )
    assert len(gate.thresholds) == 3


def test_none_resolves_against_any_column_set():
    """No active clauses, so nothing to be missing."""
    assert resolve_gate(preset="none", available_columns=set()).thresholds == {}


# --- the artifact identity --------------------------------------------------

def test_effective_digest_changes_with_an_override():
    """A preset NAME does not reproduce a run: the preset file can change."""
    base = resolve_gate(preset="default")
    tightened = resolve_gate(preset="default", overrides={"bz_ipsae": (">=", 0.9)})
    assert base.effective_digest() != tightened.effective_digest()


def test_effective_digest_is_stable_for_the_same_resolution():
    first = resolve_gate(preset="default").effective_digest()
    second = resolve_gate(preset="default").effective_digest()
    assert first == second
