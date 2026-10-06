"""`import pxdbench.tools` must not pull torch/protenix: nearly every pxdbench import goes through it, and CPU-only code (and a clean CI env) never folds."""
import subprocess, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def test_importing_pxdbench_tools_does_not_import_torch_or_protenix():
    code = "import sys, pxdbench.tools; bad = [m for m in ('torch', 'protenix') if m in sys.modules]; sys.exit('imported: ' + ','.join(bad) if bad else 0)"
    r = subprocess.run([sys.executable, "-c", code], cwd=REPO, env={"PYTHONPATH": str(REPO), "PATH": "/usr/bin"}, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-500:]


def test_the_public_backend_is_still_registered_and_callable():
    from pxdbench.tools import registry
    assert callable(registry.get_backend("public"))
