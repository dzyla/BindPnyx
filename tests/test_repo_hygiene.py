"""Two defects that made a fresh clone unusable, kept from coming back."""
import os, shutil, subprocess, sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
VENDORED = REPO / "colabdesign" / "af" / "alphafold" / "data"


def test_vendored_alphafold_data_package_is_present():
    """colabdesign/af/prep.py imports colabdesign.af.alphafold.data. An unanchored `data/` rule in .gitignore once kept this package out of the repo, so
    every fresh clone failed at import (12 tests, 9 modules uncollectable)."""
    need = ["__init__.py", "parsers.py", "pipeline.py", "pipeline_multimer.py", "prep_inputs.py", "mmcif_parsing.py", "tools/__init__.py", "tools/utils.py"]
    assert [n for n in need if not (VENDORED / n).is_file()] == []


@pytest.mark.skipif(shutil.which("git") is None or not (REPO / ".gitignore").exists(), reason="no git or no .gitignore")
def test_gitignore_does_not_swallow_source_packages(tmp_path):
    """The rules are checked in a scratch repository that holds only a copy of .gitignore. Asking git about paths inside THIS checkout breaks as soon as `data/`
    or `out/` is a symlink into a workspace folder (`git check-ignore` exits 128, 'beyond a symbolic link'), which is the normal state of a working clone."""
    shutil.copy(REPO / ".gitignore", tmp_path / ".gitignore")
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    for f in ("colabdesign/af/alphafold/data/parsers.py", "colabdesign/af/alphafold/data/tools/utils.py"):
        r = subprocess.run(["git", "check-ignore", "-q", f], cwd=tmp_path)
        assert r.returncode == 1, f"{f} is git-ignored: an unanchored pattern in .gitignore matches a python package again"
    r = subprocess.run(["git", "check-ignore", "-q", "data/targets/pdl1/x"], cwd=tmp_path)
    assert r.returncode == 0                                      # the top-level data/ directory (fetched targets) is still ignored


BLOCK = "import sys\nfor m in ('torch', 'protenix', 'jax'): sys.modules[m] = None\n"


def _run(code):
    return subprocess.run([sys.executable, "-c", BLOCK + code], cwd=REPO, capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(REPO)})


def test_pxdbench_tools_imports_without_the_gpu_stack():
    """19 of ~40 test modules import pxdbench.tools; it used to import ProtenixFilter (torch + protenix) eagerly, so none of them could even be collected
    on a CPU-only machine."""
    r = _run("import pxdbench.tools\nfrom pxdbench.tools.registry import get_backend\nassert callable(get_backend('public'))\nprint('ok')")
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr[-800:]


def test_protenix_filter_is_still_importable_from_pxdbench_tools_when_the_stack_exists():
    """Back-compat: `from pxdbench.tools import ProtenixFilter` resolves lazily; without torch it is an ImportError at access time, not an AttributeError."""
    r = _run("import pxdbench.tools as t\ntry:\n    t.ProtenixFilter\nexcept ImportError:\n    print('lazy-import-error')\nexcept AttributeError:\n    print('attribute-error')\n")
    assert r.returncode == 0 and "lazy-import-error" in r.stdout, r.stdout + r.stderr[-500:]
    r = _run("import pxdbench.tools as t\ntry:\n    t.nonexistent\nexcept AttributeError:\n    print('ok')\n")
    assert "ok" in r.stdout
