import numpy as np
import pandas as pd
import pytest

from pxdbench.tasks.base import BaseTask


def _df(**cols):
    return pd.DataFrame({k: [v] for k, v in cols.items()})


def test_le_boundary_is_inclusive():
    """<= 2.0 and < 2.0 are different predicates at exactly 2.0."""
    df = _df(m=2.0)
    inclusive = BaseTask.compute_success_rate({"f": {"m": ("<=", 2.0)}}, df.copy())
    strict = BaseTask.compute_success_rate({"f": {"m": ("<", 2.0)}}, df.copy())
    assert inclusive["f_success"].iloc[0] == 1
    assert strict["f_success"].iloc[0] == 0


def test_ge_boundary_is_inclusive():
    df = _df(m=0.5)
    inclusive = BaseTask.compute_success_rate({"f": {"m": (">=", 0.5)}}, df.copy())
    strict = BaseTask.compute_success_rate({"f": {"m": (">", 0.5)}}, df.copy())
    assert inclusive["f_success"].iloc[0] == 1
    assert strict["f_success"].iloc[0] == 0


@pytest.mark.parametrize("bad", ["=<", "==", "!=", "=>", "~", ""])
def test_filter_rejects_unsupported_operator(bad):
    with pytest.raises(ValueError, match="unsupported operator"):
        BaseTask.compute_success_rate({"f": {"m": (bad, 1.0)}}, _df(m=1.0))


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_filter_rejects_nonfinite(bad):
    """A nonfinite metric is unknown, never a pass."""
    out = BaseTask.compute_success_rate({"f": {"m": ("<=", 2.0)}}, _df(m=bad))
    assert out["f_success"].iloc[0] is None


def test_none_is_unknown():
    out = BaseTask.compute_success_rate({"f": {"m": ("<=", 2.0)}}, _df(m=None))
    assert out["f_success"].iloc[0] is None


def test_all_clauses_must_pass():
    cfg = {"g": {"a": (">=", 0.5), "b": ("<=", 2.0)}}
    assert BaseTask.compute_success_rate(cfg, _df(a=0.6, b=1.0))["g_success"].iloc[0] == 1
    assert BaseTask.compute_success_rate(cfg, _df(a=0.6, b=9.0))["g_success"].iloc[0] == 0


def test_list_values_use_the_generous_direction():
    """A list of per-sample values passes if ANY sample passes."""
    cfg = {"f": {"m": ("<=", 2.0)}}
    out = BaseTask.compute_success_rate(cfg, pd.DataFrame({"m": [[9.0, 1.0]]}))
    assert out["f_success"].iloc[0] == 1


def test_missing_column_yields_none_and_ignore_missing_variant():
    out = BaseTask.compute_success_rate({"f": {"absent": (">=", 1.0)}}, _df(m=1.0))
    assert out["f_success"].iloc[0] is None
    assert "f_success_ignore_missing" in out.columns


def test_existing_protenix_filters_still_evaluate():
    """Regression: the real eval.py blocks must keep working for finite values.

    This fix changes behaviour only for nonfinite metrics and unsupported
    operators, both of which previously PASSED silently.
    """
    from pxdbench.pxd_configs.eval import eval_configs

    filters = eval_configs["binder"]["filters"]
    row = {
        "ptx_iptm_binder": 0.90,
        "ptx_ptm_binder": 0.92,
        "ptx_pred_design_rmsd": 1.0,
        "ptx_mini_iptm_binder": 0.90,
        "ptx_mini_ptm_binder": 0.92,
        "ptx_mini_pred_design_rmsd": 1.0,
        "pLDDT": 0.95,
        "i_pTM": 0.7,
        "i_pAE": 0.2,
        "bound_unbound_RMSD": 1.0,
        "unscaled_i_pAE": 5.0,
        "af2_binder_pred_design_rmsd": 1.0,
    }
    out = BaseTask.compute_success_rate(filters, pd.DataFrame([row]))
    assert out["ptx_success"].iloc[0] == 1
    assert out["ptx_mini_success"].iloc[0] == 1
    assert out["af2_easy_success"].iloc[0] == 1
    assert out["af2_opt_success"].iloc[0] == 1

    failing = dict(row, ptx_iptm_binder=0.10)
    out2 = BaseTask.compute_success_rate(filters, pd.DataFrame([failing]))
    assert out2["ptx_success"].iloc[0] == 0


def test_nan_used_to_pass_and_now_does_not():
    """Pins the exact behaviour change, so it is a decision not a surprise.

    Old code returned None only for `value is None`; np.nan >= t and
    np.nan <= t are both False, so a NaN row fell through to `return 1`.
    """
    cfg = {"f": {"m": (">", 0.5)}}
    out = BaseTask.compute_success_rate(cfg, _df(m=np.nan))
    assert out["f_success"].iloc[0] is not None or out["f_success"].iloc[0] is None
    assert out["f_success"].iloc[0] is None, "NaN must be unknown, not a pass"
