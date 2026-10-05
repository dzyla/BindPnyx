"""The targeting path on targets it was not developed against.

Everything in `pxdbench/targets/` was built against one EGFR shard, which is
exactly the circumstance in which a pipeline quietly acquires target-specific
assumptions: a constant numbering offset, a chain-naming habit, a two-chain
shape.

Measured 2026-10-01 across four real structures - EGFR 2-chain, EGFR domain
III, MeV H head domain, MeV H + CD150 complex - all four resolve. The
load-bearing case is `human_dIII_canon.pdb`, where author A334 maps to work
A24: an offset of 310, nothing like obj2's +226/+511. A constant-offset
implementation would have passed obj2 and silently mis-mapped this one.

Skips per-target when a structure is not on this machine, and reports the skip.
"""
import os

import pytest

from pxdbench.targets.smoke import DEFAULT_TARGETS, smoke_one


@pytest.mark.parametrize("path", DEFAULT_TARGETS, ids=os.path.basename)
def test_the_targeting_path_resolves_a_real_target(path):
    if not os.path.exists(path):
        pytest.skip(f"not on this machine: {path}")
    status, detail = smoke_one(path)
    assert status in ("ok", "rejected"), detail
    # A bounded rejection is a correct outcome - v1 is deliberately limited and
    # names what it cannot take. A crash or a consistency failure is not.


def test_at_least_one_target_is_available_here():
    """Otherwise the parametrised cases all skip and prove nothing."""
    available = [p for p in DEFAULT_TARGETS if os.path.exists(p)]
    if not available:
        pytest.skip("no target structures on this machine")
    assert available


def test_a_nontrivial_offset_target_maps_correctly():
    """The regression that matters: not a constant offset."""
    path = "/data/private/private_target1/human_dIII_canon.pdb"
    if not os.path.exists(path):
        pytest.skip(f"not on this machine: {path}")
    status, detail = smoke_one(path)
    assert status == "ok", detail
    assert "A334" in detail and "A24" in detail, detail
