import json
import os

import pytest
pytest.importorskip("torch", reason="needs the full pxd environment (torch)")

from pxdesign.runner.pipeline import accumulate_successes, early_stop_success_key


def _summary(tmp_path, name, payload):
    path = os.path.join(str(tmp_path), f"{name}.json")
    with open(path, "w") as handle:
        json.dump(payload, handle)
    return {"name": name, "summary_save_path": path}


def test_key_follows_the_mode():
    assert early_stop_success_key(use_boltz_filter=True) == (
        "bz_gate_egfr_provisional_v1_success.count"
    )
    assert early_stop_success_key(use_boltz_filter=False) == "af2_easy_success.count"


def test_accumulates_counts(tmp_path):
    results = [_summary(tmp_path, "t", {"af2_easy_success.count": 3})]
    out = accumulate_successes(results, {}, "af2_easy_success.count")
    assert out == {"t": 3}
    out = accumulate_successes(results, out, "af2_easy_success.count")
    assert out == {"t": 6}


def test_accumulates_per_task(tmp_path):
    results = [
        _summary(tmp_path, "a", {"af2_easy_success.count": 2}),
        _summary(tmp_path, "b", {"af2_easy_success.count": 5}),
    ]
    assert accumulate_successes(results, {}, "af2_easy_success.count") == {
        "a": 2,
        "b": 5,
    }


def test_missing_key_raises(tmp_path):
    """A selection policy that silently counts nothing is the bug being fixed."""
    results = [_summary(tmp_path, "t", {"something_else": 1})]
    with pytest.raises(KeyError, match="bz_gate_egfr_provisional_v1_success.count"):
        accumulate_successes(results, {}, "bz_gate_egfr_provisional_v1_success.count")


def test_error_names_the_available_keys(tmp_path):
    results = [_summary(tmp_path, "t", {"af2_easy_success.count": 1})]
    with pytest.raises(KeyError) as excinfo:
        accumulate_successes(results, {}, "bz_gate_egfr_provisional_v1_success.count")
    assert "af2_easy_success.count" in str(excinfo.value)


def test_does_not_mutate_the_input_mapping(tmp_path):
    results = [_summary(tmp_path, "t", {"af2_easy_success.count": 1})]
    before = {"t": 10}
    after = accumulate_successes(results, before, "af2_easy_success.count")
    assert before == {"t": 10}
    assert after == {"t": 11}


def test_old_get_default_would_have_hidden_this(tmp_path):
    """Documents the fixed bug: with AF2 disabled the key never exists, and
    summary.get(key, 0) kept cumulative_success at 0 forever, so early_stop
    silently never fired."""
    results = [_summary(tmp_path, "t", {"bz_gate_egfr_provisional_v1_success.count": 4})]
    # the new key resolves and counts
    assert accumulate_successes(
        results, {}, "bz_gate_egfr_provisional_v1_success.count"
    ) == {"t": 4}
    # the old hardcoded key is absent and must now be loud, not silent
    with pytest.raises(KeyError):
        accumulate_successes(results, {}, "af2_easy_success.count")


# --------------------------------------------------------------------------- #
# A provisional gate must not silently decide how much compute a campaign gets
# --------------------------------------------------------------------------- #


def test_provisional_gate_cannot_early_stop_by_default():
    """--early_stop defaults True and --min_early_stop_successes to 1, so one
    pass of the provisional Boltz gate would otherwise end the campaign."""
    from pxdesign.runner.pipeline import (
        BOLTZ_EARLY_STOP_KEY,
        resolve_early_stop,
    )

    enabled, reason = resolve_early_stop(
        early_stop=True, success_key=BOLTZ_EARLY_STOP_KEY, allow_provisional=False
    )
    assert enabled is False
    assert "provisional" in reason
    assert "--allow_provisional_early_stop" in reason


def test_provisional_early_stop_can_be_opted_into():
    from pxdesign.runner.pipeline import BOLTZ_EARLY_STOP_KEY, resolve_early_stop

    enabled, _ = resolve_early_stop(
        early_stop=True, success_key=BOLTZ_EARLY_STOP_KEY, allow_provisional=True
    )
    assert enabled is True


def test_a_validated_gate_still_early_stops():
    """The guard is about gate validity, not about disabling the feature."""
    from pxdesign.runner.pipeline import AF2_EARLY_STOP_KEY, resolve_early_stop

    enabled, _ = resolve_early_stop(
        early_stop=True, success_key=AF2_EARLY_STOP_KEY, allow_provisional=False
    )
    assert enabled is True


def test_early_stop_off_stays_off():
    from pxdesign.runner.pipeline import AF2_EARLY_STOP_KEY, resolve_early_stop

    enabled, reason = resolve_early_stop(
        early_stop=False, success_key=AF2_EARLY_STOP_KEY, allow_provisional=True
    )
    assert enabled is False
    assert "not requested" in reason


def test_the_boltz_gate_is_registered_as_provisional():
    from pxdesign.runner.pipeline import (
        BOLTZ_EARLY_STOP_KEY,
        PROVISIONAL_GATE_KEYS,
        early_stop_success_key,
    )

    assert early_stop_success_key(use_boltz_filter=True) in PROVISIONAL_GATE_KEYS
    assert "provisional" in BOLTZ_EARLY_STOP_KEY
