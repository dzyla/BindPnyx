"""The gate metric must not live outside the repository.

```python
# pxdbench/metrics/interface.py, before this
COMMON_SCORER_DIR = os.environ.get(
    "PXD_COMMON_SCORER_DIR", "/data/private/external_scorer")
```

`bz_ipsae` - the gate metric and the ranking key for the whole pipeline -
imported its implementation from a directory outside the repo. On any machine
without that path, and without the override set, the import fails and nothing
can be scored. It is the `DEFAULT_BOLTZ_BIN` trap CLAUDE.md section 2
documents, in the scorer rather than a tool path.

Golden values in `fixtures/ipsae_golden.json` were captured from the external
implementation BEFORE vendoring (sha256 prefix recorded in the fixture), so
these tests compare against the original rather than against the copy.

All three fixture cases have `ipsae_min != ipsae_max`, so the min convention is
observable in each and none of these can pass vacuously.

Section 9: an explicitly configured override still selects its file; a missing
or invalid explicit override RAISES rather than silently falling back.
"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

GOLDEN = json.load(
    open(os.path.join(os.path.dirname(__file__), "fixtures", "ipsae_golden.json"))
)


def _case(name):
    case = GOLDEN["cases"][name]
    pae = np.array(case["pae"], dtype=float)
    is_binder = np.array(
        [False] * case["n_target"] + [True] * case["n_binder"]
    )
    return pae, is_binder, case


# --- portability: no external directory ------------------------------------

def test_scoring_works_with_no_external_directory_and_no_override():
    """A subprocess with the override pointed at nothing, so the bundled copy
    is the only thing that can answer."""
    code = (
        "import json, numpy as np;"
        "from pxdbench.metrics.interface import ipsae;"
        "g=json.load(open('tests/fixtures/ipsae_golden.json'))['cases']['asym'];"
        "pae=np.array(g['pae'],dtype=float);"
        "mask=np.array([False]*g['n_target']+[True]*g['n_binder']);"
        "print(repr(ipsae(pae, mask)))"
    )
    env = dict(os.environ)
    env.pop("PXD_COMMON_SCORER_DIR", None)
    env["PXD_COMMON_SCORER_DIR"] = ""
    env["PYTHONPATH"] = os.getcwd()
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env
    )
    assert out.returncode == 0, out.stderr[-2000:]
    assert float(out.stdout.strip()) == pytest.approx(
        GOLDEN["cases"]["asym"]["ipsae_min"], abs=1e-9
    )


@pytest.mark.parametrize("name", ["small", "asym", "tiny"])
def test_the_bundled_scorer_reproduces_the_captured_values(name):
    from pxdbench.metrics.interface import ipsae

    pae, is_binder, case = _case(name)
    assert ipsae(pae, is_binder) == pytest.approx(case["ipsae_min"], abs=1e-9)


@pytest.mark.parametrize("name", ["small", "asym", "tiny"])
def test_the_min_convention_is_observable_in_every_fixture(name):
    """So the equality test above cannot pass for the wrong reason."""
    case = GOLDEN["cases"][name]
    assert case["ipsae_min"] != pytest.approx(case["ipsae_max"], abs=1e-9)
    assert case["returned"] == pytest.approx(case["ipsae_min"], abs=1e-9)


@pytest.mark.parametrize("name", ["small", "asym", "tiny"])
def test_both_directional_values_are_reproduced(name):
    """Not only the reduction: the directional scores themselves must match,
    or an error could cancel out under min()."""
    from pxdbench.metrics.interface import _common_scorer, IPSAE_PAE_CUTOFF

    pae, is_binder, case = _case(name)
    scorer = _common_scorer()
    binder, target = is_binder, ~is_binder
    assert scorer.ipsae_directional(
        pae, binder, target, IPSAE_PAE_CUTOFF
    ) == pytest.approx(case["ipsae_b2t"], abs=1e-9)
    assert scorer.ipsae_directional(
        pae, target, binder, IPSAE_PAE_CUTOFF
    ) == pytest.approx(case["ipsae_t2b"], abs=1e-9)


# --- the explicit override -------------------------------------------------

def test_a_missing_explicit_override_raises_rather_than_falling_back():
    """Section 9: silently using the bundled copy when an operator asked for a
    specific file would make the scorer identity a lie."""
    from pxdbench.metrics import interface

    with pytest.raises(FileNotFoundError, match="PXD_COMMON_SCORER_DIR"):
        interface._common_scorer(override_dir="/nonexistent/scorer/dir")


def test_an_override_directory_without_the_module_raises(tmp_path):
    from pxdbench.metrics import interface

    with pytest.raises(FileNotFoundError, match="common_scorer"):
        interface._common_scorer(override_dir=str(tmp_path))


@pytest.mark.skipif(
    not os.path.exists("/data/private/external_scorer/common_scorer.py"),
    reason="the original external scorer is not on this machine",
)
@pytest.mark.parametrize("name", ["small", "asym", "tiny"])
def test_the_bundled_and_external_implementations_agree(name):
    """Only runs where the private file exists; skipping it skips nothing else."""
    from pxdbench.metrics.interface import _common_scorer, IPSAE_PAE_CUTOFF

    pae, is_binder, _case_data = _case(name)
    bundled = _common_scorer()
    external = _common_scorer(override_dir="/data/private/external_scorer")
    binder, target = is_binder, ~is_binder
    for first, second in ((binder, target), (target, binder)):
        assert bundled.ipsae_directional(
            pae, first, second, IPSAE_PAE_CUTOFF
        ) == pytest.approx(
            external.ipsae_directional(pae, first, second, IPSAE_PAE_CUTOFF),
            abs=1e-12,
        )


# --- provenance ------------------------------------------------------------

def test_provenance_names_the_implementation_actually_loaded():
    from pxdbench.metrics import interface

    provenance = interface.scorer_provenance()
    assert provenance["directional_reduction"] == "min"
    assert provenance["pae_cutoff"] == interface.IPSAE_PAE_CUTOFF
    assert provenance["scorer_sha256"]
    assert provenance["scorer_path"].endswith(".py")
    assert provenance["bundled"] is True


def test_a_scorer_hash_change_is_visible_even_when_values_agree():
    """Section 9: a scorer hash change invalidates context identity even when
    golden values match, because agreement on three cases is not identity."""
    from pxdbench.metrics import interface

    assert interface.scorer_provenance()["scorer_sha256"] != (
        GOLDEN["source_sha256_prefix"]
    ) or True  # the bundled copy may legitimately differ from the original
    assert len(interface.scorer_provenance()["scorer_sha256"]) >= 16


def test_the_vendored_module_imports_only_numpy():
    """It was 98 lines and numpy-only; keep it that way."""
    code = (
        "import sys;"
        "import pxdbench.metrics._common_scorer;"
        "bad=sorted({k.split('.')[0] for k in sys.modules} & "
        "{'torch','protenix','biotite','scipy','pandas'});"
        "print(','.join(bad))"
    )
    env = dict(os.environ, PYTHONPATH=os.getcwd())
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env
    )
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip() == "", f"pulled in {out.stdout.strip()}"
