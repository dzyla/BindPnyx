"""Per-target calibration as data, not module-level constants.

The application was built around one target at the calibration layer: `GATE` was
a single module-level dict and the gate id a string literal in four places,
including inside the early-stop key. Adding a second target meant reusing one
protein's calibrated thresholds on a different protein, or editing source.

Identity is deliberately three things (review A3). A digest of thresholds alone
is not an identity: identical numbers describe different DECISIONS when the
epitope policy, seed aggregation, comparator inclusivity or distance cutoffs
differ, and an unconstrained and a pocket-conditioned prediction can share a
threshold dict while having different applicability.

No private files and no predictor imports: this must run anywhere.
"""
import pytest

from pxdbench.targets.registry import (
    NoiseFloors,
    Registry,
    early_stop_key,
)

LEGACY = "bz_gate_egfr_provisional_v1"


# --- the live gate must not move -------------------------------------------
# Renaming or re-deriving it would orphan every past run.

def test_the_existing_gate_is_reproduced_exactly():
    gate = Registry.load_default().gate(LEGACY)
    assert gate.thresholds == {
        "bz_n_seeds": (">=", 3),
        "bz_pae_interface_min": ("<=", 2.0),
        "bz_ipsae": (">=", 0.5),
    }
    assert gate.provisional is True
    assert gate.engagement_required is False


def test_backend_gate_constants_come_from_the_registry():
    from pxdbench.tools.boltz import backend
    gate = Registry.load_default().gate(backend.GATE_NAME)
    assert backend.GATE == gate.thresholds


def test_no_module_level_gate_literal_remains():
    """A literal is how one target got baked into four files."""
    import inspect

    from pxdbench.tools.boltz import backend
    from pxdesign.runner import helpers, pipeline

    for module in (backend, helpers, pipeline):
        source = inspect.getsource(module)
        assert LEGACY not in source, module.__name__


# --- identity: three things, not one ---------------------------------------

def test_identity_is_two_digests_plus_a_human_label():
    gate = Registry.load_default().gate(LEGACY)
    assert gate.target_label
    assert gate.decision_rule_digest
    assert gate.evaluation_context_digest
    assert "egfr" not in gate.decision_rule_digest


def test_changing_a_decision_field_changes_the_rule_digest():
    """Comparator inclusivity is decision-affecting: `>` and `>=` differ."""
    base = Registry.load_default().gate(LEGACY)
    strict = base.with_overrides({"bz_ipsae": (">", 0.5)})
    assert strict.decision_rule_digest != base.decision_rule_digest


def test_changing_an_enforced_policy_changes_the_rule_digest():
    """Identical thresholds, different decision."""
    base = Registry.load_default().gate(LEGACY)
    enforced = base.with_engagement_required(True)
    assert enforced.thresholds == base.thresholds
    assert enforced.decision_rule_digest != base.decision_rule_digest


def test_changing_only_a_display_label_changes_no_digest():
    base = Registry.load_default().gate(LEGACY)
    relabelled = base.with_label("EGFR obj2 (provisional)")
    assert relabelled.decision_rule_digest == base.decision_rule_digest
    assert relabelled.evaluation_context_digest == base.evaluation_context_digest


def test_conditioning_mode_changes_the_context_digest_not_the_rule():
    """A pocket-conditioned run is not interpretable against an unconstrained
    calibration even when the thresholds are identical."""
    base = Registry.load_default().gate(LEGACY)
    conditioned = base.with_context(conditioning_mode="pocket")
    assert conditioned.thresholds == base.thresholds
    assert conditioned.decision_rule_digest == base.decision_rule_digest
    assert conditioned.evaluation_context_digest != base.evaluation_context_digest


# --- derived keys, never literals ------------------------------------------

def test_early_stop_key_is_derived():
    assert early_stop_key(LEGACY) == f"{LEGACY}_success.count"


def test_provisional_early_stop_consults_metadata_not_a_name():
    """`PROVISIONAL_GATE_KEYS` matched fixed success-key STRINGS, so a
    registry-identified gate would have bypassed the guard entirely."""
    from pxdesign.runner.pipeline import may_early_stop

    gate = Registry.load_default().gate(LEGACY)
    assert gate.provisional is True
    assert may_early_stop(gate, allow_provisional=False) is False
    assert may_early_stop(gate, allow_provisional=True) is True


def test_a_new_provisional_gate_also_cannot_early_stop():
    """The failure A3 predicted: a gate the static list never heard of."""
    from pxdesign.runner.pipeline import may_early_stop

    gate = Registry.load_default().gate(LEGACY).with_label("something else")
    assert may_early_stop(gate, allow_provisional=False) is False


# --- a second target cannot inherit the first one's calibration ------------

def test_an_unknown_target_has_no_gate():
    with pytest.raises(KeyError, match="no gate"):
        Registry.load_default().gates_for("some_other_target")


def test_an_unknown_gate_id_raises():
    with pytest.raises(KeyError, match="no gate"):
        Registry.load_default().gate("bz_gate_does_not_exist")


# --- noise floors are measured, never inherited ----------------------------

def test_noise_floors_default_to_unmeasured():
    """A borrowed floor is worse than no floor: it licenses a false
    conclusion. Defaults are None, meaning NOT MEASURED."""
    floors = Registry.load_default().floors("egfr_obj2", host="any-host")
    assert floors.within_run is None
    assert floors.across_run is None
    assert floors.same_seed is None


def test_noise_floors_report_whether_they_were_measured():
    assert NoiseFloors().measured is False
    assert NoiseFloors(within_run=0.035, measured_on="host-x").measured is True


# --- the registry is data ---------------------------------------------------

def test_targets_are_loaded_from_data_not_code():
    """Adding a target must be a data change."""
    import pxdbench.targets.registry as reg_module

    assert reg_module.REGISTRY_PATH.exists()
    assert reg_module.REGISTRY_PATH.suffix == ".json"


def test_target_spec_carries_its_own_numbering_declaration():
    """C-1: numbering is declared, not inferred from a file extension."""
    target = Registry.load_default().target("egfr_obj2")
    assert target.numbering in ("auth", "work")
    assert target.chain_filter
