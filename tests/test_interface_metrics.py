import os
import sys

import numpy as np
import pytest

from conftest import BC2_DIR, COMMON_SCORER_DIR, FIXTURES

from pxdbench.metrics.interface import (
    anchored_scores,
    d0_from_count,
    i_pdae,
    interface_plddt,
    ipsae,
    pae_interface_min,
)


def _bc2_anchored():
    """BindCraft 2's own scorer, imported as a live oracle.

    It imports cleanly in pxlocal (biotite 1.0.1 + jax 0.10.2 are present),
    so the golden test is differential rather than frozen.
    """
    if not os.path.isdir(BC2_DIR):
        pytest.skip(f"BindCraft2 not present at {BC2_DIR}")
    if BC2_DIR not in sys.path:
        sys.path.insert(0, BC2_DIR)
    try:
        from bindcraft.filters import anchored_interface_tm_scores
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"bindcraft.filters unavailable: {exc}")
    return anchored_interface_tm_scores


def _common_scorer():
    """The scorer the project actually uses.

    This used to insert COMMON_SCORER_DIR on sys.path and import by name, which
    made the whole file unrunnable on a machine without that private
    directory. The implementation is now vendored, so ask the project for it.
    """
    from pxdbench.metrics.interface import _common_scorer as resolve

    return resolve()


def _contact_and_sub(pae, ca, is_binder, cutoff=8.0):
    b, t = is_binder, ~is_binder
    d = np.linalg.norm(ca[b][:, None, :] - ca[t][None, :, :], axis=2)
    return pae[np.ix_(b, t)], d <= cutoff


def test_ipdae_matches_bindcraft2(asym_case):
    import jax.numpy as jnp

    anchored = _bc2_anchored()
    pae, ca, is_binder, _ = asym_case
    sub, contact = _contact_and_sub(pae, ca, is_binder)
    ref = max(
        float(anchored(jnp.asarray(sub), jnp.asarray(contact)).max()),
        float(anchored(jnp.asarray(sub.T), jnp.asarray(contact.T)).max()),
    )
    got, _track = i_pdae(pae, ca, is_binder, cutoff=8.0)
    assert got == pytest.approx(ref, abs=1e-6)


def test_ipdae_uses_transposed_submatrix(asym_case):
    """The reverse pass transposes the SAME submatrix (filters.py:126).

    Independently slicing pae[target, binder] is a different number whenever
    the PAE matrix is asymmetric, which it is.
    """
    pae, ca, is_binder, _ = asym_case
    b, t = is_binder, ~is_binder
    sub, contact = _contact_and_sub(pae, ca, is_binder)
    correct = max(
        anchored_scores(sub, contact).max(),
        anchored_scores(sub.T, contact.T).max(),
    )
    wrong_sub = pae[np.ix_(t, b)]
    wrong = max(
        anchored_scores(sub, contact).max(),
        anchored_scores(wrong_sub, contact.T).max(),
    )
    assert correct != pytest.approx(wrong, abs=1e-9), "fixture not asymmetric enough"
    got, _ = i_pdae(pae, ca, is_binder)
    assert got == pytest.approx(float(correct), abs=1e-9)


def test_bz_ipsae_is_ipsae_min(asym_case):
    cs = _common_scorer()
    pae, ca, is_binder, _ = asym_case
    ref = cs.score_complex(pae, ca, is_binder)
    assert ref["ipsae_min"] != pytest.approx(ref["ipsae_max"], abs=1e-9)
    assert ipsae(pae, is_binder) == pytest.approx(ref["ipsae_min"], abs=1e-9)


def test_ipdae_and_ipsae_differ(asym_case):
    """Guards the two conventions against silently collapsing into one."""
    pae, ca, is_binder, _ = asym_case
    got_pdae, _ = i_pdae(pae, ca, is_binder)
    assert got_pdae != pytest.approx(ipsae(pae, is_binder), abs=1e-6)


def test_d0_floor_is_unobservable_under_the_clamp():
    """The 19-vs-26 partner-count floor difference is a NO-OP.

    1.24*(n-15)^(1/3) - 1.8 only exceeds 1.0 at n ~= 26.514, which is above both
    floors, so max(..., 1.0) returns 1.0 for every n below either one. i_pDAE's
    floor of 19 and ipSAE's 26 therefore produce identical d0 everywhere.

    This is asserted rather than left implicit for two reasons: so nobody
    "fixes" one floor to match the other believing it changes a number, and so
    nobody cites the floor as a real difference between the two metrics. The
    substantive differences are the mask and the direction aggregation.
    """
    counts = np.arange(1.0, 400.0)
    assert np.allclose(
        d0_from_count(counts, 19.0), d0_from_count(counts, 26.0), atol=0.0, rtol=0.0
    )
    # the clamp is what makes them agree, and it releases above both floors
    assert d0_from_count(np.array([26.0]), 19.0)[0] == pytest.approx(1.0, abs=1e-12)
    assert d0_from_count(np.array([27.0]), 19.0)[0] > 1.0
    assert d0_from_count(np.array([200.0]), 19.0)[0] > 5.0


def test_no_interface_returns_none():
    pae = np.full((6, 6), 5.0)
    ca = np.zeros((6, 3))
    ca[3:, 0] = 500.0  # target 500 A away
    is_binder = np.array([True] * 3 + [False] * 3)
    value, track = i_pdae(pae, ca, is_binder)
    assert value is None
    assert np.isnan(track).all()
    assert interface_plddt(np.full(6, 90.0), ca, is_binder) is None


def test_single_chain_returns_none():
    pae = np.full((4, 4), 3.0)
    ca = np.zeros((4, 3))
    is_binder = np.ones(4, dtype=bool)
    assert i_pdae(pae, ca, is_binder)[0] is None
    assert pae_interface_min(pae, is_binder) is None


def test_pae_interface_min_is_minimum(asym_case):
    pae, ca, is_binder, _ = asym_case
    b, t = is_binder, ~is_binder
    expect = min(pae[np.ix_(b, t)].min(), pae[np.ix_(t, b)].min())
    assert pae_interface_min(pae, is_binder) == pytest.approx(expect, abs=1e-9)


def test_interface_plddt_averages_contacting_binder_residues(asym_case):
    pae, ca, is_binder, plddt = asym_case
    _sub, contact = _contact_and_sub(pae, ca, is_binder)
    mask = contact.any(-1)
    expect = plddt[is_binder][mask].mean()
    assert interface_plddt(plddt, ca, is_binder) == pytest.approx(expect, abs=1e-9)


def test_golden_fixture_matches(asym_case):
    """Runs even where BindCraft 2 is absent."""
    path = os.path.join(FIXTURES, "ipdae_golden.npz")
    if not os.path.exists(path):
        pytest.skip("golden fixture not generated yet")
    g = np.load(path)
    got, _ = i_pdae(g["pae"], g["ca"], g["is_binder"], cutoff=float(g["cutoff"]))
    assert got == pytest.approx(float(g["i_pdae"]), abs=1e-6)
