"""Shared fixtures. Tests run in the pxlocal env with the repo on sys.path."""
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

BC2_DIR = os.environ.get("PXD_BC2_DIR", "/data/private/BindCraft2")
COMMON_SCORER_DIR = os.environ.get(
    "PXD_COMMON_SCORER_DIR", "/data/private/external_scorer"
)
FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")


@pytest.fixture
def asym_case():
    """An ASYMMETRIC PAE matrix with real binder/target contacts.

    Asymmetry is load-bearing: with a symmetric PAE, transposing the
    binder-by-target submatrix and independently slicing the target-by-binder
    submatrix give the same answer, so the transpose test passes vacuously.
    """
    rng = np.random.default_rng(0)
    n_b, n_t = 12, 30
    n = n_b + n_t
    pae = rng.uniform(0.5, 25.0, size=(n, n))
    np.fill_diagonal(pae, 0.0)
    assert not np.allclose(pae, pae.T), "fixture must be asymmetric"
    ca = rng.uniform(-25.0, 25.0, size=(n, 3))
    # place each binder residue within contact range of one target residue
    ca[:n_b] = ca[n_b : n_b + n_b] + rng.uniform(-2.5, 2.5, size=(n_b, 3))
    is_binder = np.zeros(n, dtype=bool)
    is_binder[:n_b] = True
    plddt = rng.uniform(40.0, 98.0, size=n)
    return pae, ca, is_binder, plddt


# --- residue-map fixtures ---------------------------------------------------
# Duck-typed residue lists, so the bounded map contract is testable without
# biotite, Biopython or a real structure. Shared by the map, spec and role
# tests; `tests` is not a package, so this is the one place they can live.

_ONE_TO_THREE = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}


def make_residues(chain, start, seq, ins_code=""):
    """[(chain, res_id, resname, ins_code)] from a one-letter sequence."""
    return [
        (chain, start + i, _ONE_TO_THREE[aa], ins_code)
        for i, aa in enumerate(seq)
    ]


@pytest.fixture
def residues():
    return make_residues


@pytest.fixture
def hadq_map():
    """work B 1..4 <- source D 512..515, emitted sequence HADQ."""
    from pxdbench.targets.residue_map import TargetResidueMap

    return TargetResidueMap.from_residue_lists(
        source={"D": make_residues("D", 512, "HADQ")},
        shard={"B": make_residues("B", 1, "HADQ")},
        chain_filter=["B"],
        sequences={"B": "HADQ"},
    )


# A clean CPU environment has none of the heavy stack. A test that needs one of these packages is reported as SKIPPED with the package named,
# not as a failure: the failure is "not installed", which says nothing about the code. Any other error, and a ModuleNotFoundError for a package
# not listed here, still fails. In the full `pxd` env these packages are installed and this hook never fires.
HEAVY_PACKAGES = {"torch", "protenix", "jax", "biotite", "matplotlib", "colabdesign", "deepspeed", "openfold3"}


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    exc = call.excinfo.value if call.excinfo is not None else None
    missing = getattr(exc, "name", None) if isinstance(exc, ModuleNotFoundError) else None
    if rep.failed and missing and missing.split(".")[0] in HEAVY_PACKAGES:
        rep.outcome = "skipped"
        rep.longrepr = (str(item.path), item.location[1] or 0, f"Skipped: needs the full environment ('{missing}' is not installed)")
