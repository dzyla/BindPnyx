"""Import the Boltz MSA validators without paying for torch and protenix.

`import pxdbench.tools.boltz.msa_check` costs 68 s on this machine, measured:
`pxdbench/tools/__init__.py` imports the Protenix filter, which imports torch
and protenix, which JIT-builds (and fails to build) the fused LayerNorm. A
pre-flight whose job is to fail in seconds cannot pay that.

`msa_check.py` and `targets.py` depend on nothing but the standard library, so
they are loaded straight from their files here. This is NOT a copy: the same
source file is executed, and when the package has already been imported (inside
the pipeline, or in a test session that imported it) the already-imported module
is returned instead. `tests/test_prepare_target.py` pins that the file loaded
here is the file the package exposes, so the two cannot drift apart.
"""
import importlib.util
import os
import sys

_BOLTZ_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "pxdbench", "tools", "boltz",
)


def load(name):
    """`msa_check` or `targets`, as the package would expose it."""
    qualified = f"pxdbench.tools.boltz.{name}"
    already = sys.modules.get(qualified)
    if already is not None:
        return already
    path = os.path.join(_BOLTZ_DIR, f"{name}.py")
    if not os.path.isfile(path):
        raise ImportError(
            f"{qualified}: expected the module at {path}. The light loader "
            f"resolves it by path to avoid importing pxdbench.tools, which "
            f"costs 68 s of torch/protenix import."
        )
    spec = importlib.util.spec_from_file_location(f"_boltz_light_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


msa_check = load("msa_check")
targets = load("targets")
