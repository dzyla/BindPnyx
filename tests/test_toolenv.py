"""The app must not depend on anyone else's environment.

The regression this guards: `pxdbench/tools/boltz/runner.py` shipped

    DEFAULT_BOLTZ_BIN = "/home/someone/conda/envs/boltz2local/bin/boltz"

a path inside one person's unrelated conda env. On any other machine the pipeline
generated backbones, designed sequences, and then died at the first fold with
FileNotFoundError - after the GPU work, not before it.
"""
import os

import pytest

from pxdbench.toolenv import (
    PROJECT_ENV_ROOT,
    REPO_ROOT,
    ToolNotFound,
    project_env_bin,
    resolve_boltz_bin,
    resolve_diffusion_python,
)


@pytest.fixture(autouse=True)
def _no_ambient_overrides(monkeypatch):
    """Each test states its own environment; the host's must not leak in."""
    monkeypatch.delenv("PXD_BOLTZ_BIN", raising=False)
    monkeypatch.delenv("PXD_PYTHON", raising=False)


def _exe(path):
    path.write_text("#!/bin/sh\nexit 0\n")
    path.chmod(0o755)
    return str(path)


# --------------------------------------------------------------------------- #
# No borrowed paths
# --------------------------------------------------------------------------- #


def test_no_module_hardcodes_a_home_directory():
    """A path under /home/<someone> used as a VALUE in shipped code is the bug.

    Parsed rather than grepped: prose in a docstring describing the history is
    fine, and upstream ProteinMPNN's `help="... e.g. /home/my_pdbs/"` is an
    example string for a CLI, not a path anything opens. What must not exist is
    a string constant that some code would actually use as a path.
    """
    import ast

    offenders = []
    for root in ("pxdbench", "pxdesign"):
        for dirpath, _, files in os.walk(os.path.join(REPO_ROOT, root)):
            for name in sorted(files):
                if not name.endswith(".py"):
                    continue
                full = os.path.join(dirpath, name)
                try:
                    tree = ast.parse(open(full, errors="ignore").read())
                except SyntaxError:
                    continue

                docstrings = {
                    node.body[0].value
                    for node in ast.walk(tree)
                    if isinstance(
                        node, (ast.Module, ast.ClassDef, ast.FunctionDef)
                    )
                    and node.body
                    and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)
                }
                help_texts = {
                    kw.value
                    for node in ast.walk(tree)
                    if isinstance(node, ast.Call)
                    for kw in node.keywords
                    if kw.arg in ("help", "metavar") and isinstance(kw.value, ast.Constant)
                }

                for node in ast.walk(tree):
                    if not isinstance(node, ast.Constant):
                        continue
                    if not isinstance(node.value, str):
                        continue
                    if node in docstrings or node in help_texts:
                        continue
                    if node.value.startswith("/home/") or "boltz2local" in node.value:
                        offenders.append(
                            f"{full}:{node.lineno}: {node.value[:70]!r}"
                        )
    assert not offenders, "hardcoded foreign paths used as values:\n" + "\n".join(
        offenders
    )


def test_project_env_root_is_inside_the_repo():
    assert PROJECT_ENV_ROOT.startswith(REPO_ROOT)
    assert ".pxd" in PROJECT_ENV_ROOT


# --------------------------------------------------------------------------- #
# Resolution order
# --------------------------------------------------------------------------- #


def test_explicit_setting_wins(tmp_path, monkeypatch):
    chosen, other = _exe(tmp_path / "chosen"), _exe(tmp_path / "other")
    monkeypatch.setenv("PXD_BOLTZ_BIN", other)
    assert resolve_boltz_bin(explicit=chosen) == chosen


def test_env_var_is_used_when_no_explicit_setting(tmp_path, monkeypatch):
    target = _exe(tmp_path / "boltz")
    monkeypatch.setenv("PXD_BOLTZ_BIN", target)
    assert resolve_boltz_bin() == target


def test_empty_explicit_setting_falls_through(tmp_path, monkeypatch):
    """The config default is "" - it must not be treated as a path."""
    target = _exe(tmp_path / "boltz")
    monkeypatch.setenv("PXD_BOLTZ_BIN", target)
    assert resolve_boltz_bin(explicit="") == target


def test_project_env_is_preferred_over_path(tmp_path, monkeypatch):
    """The app's own environment comes before whatever is installed globally."""
    own = tmp_path / "envs" / "boltz" / "bin"
    own.mkdir(parents=True)
    own_exe = _exe(own / "boltz")
    monkeypatch.setattr("pxdbench.toolenv.PROJECT_ENV_ROOT", str(tmp_path / "envs"))
    monkeypatch.setattr("shutil.which", lambda _: _exe(tmp_path / "global_boltz"))
    assert resolve_boltz_bin() == own_exe


def test_path_is_the_last_resort(tmp_path, monkeypatch):
    found = _exe(tmp_path / "boltz_on_path")
    monkeypatch.setattr("pxdbench.toolenv.PROJECT_ENV_ROOT", str(tmp_path / "absent"))
    monkeypatch.setattr("shutil.which", lambda _: found)
    assert resolve_boltz_bin() == found


def test_a_non_executable_candidate_is_skipped_not_returned(tmp_path, monkeypatch):
    dud = tmp_path / "boltz"
    dud.write_text("not executable")
    good = _exe(tmp_path / "real")
    monkeypatch.setenv("PXD_BOLTZ_BIN", str(dud))
    monkeypatch.setattr("pxdbench.toolenv.PROJECT_ENV_ROOT", str(tmp_path / "absent"))
    monkeypatch.setattr("shutil.which", lambda _: good)
    assert resolve_boltz_bin() == good


def test_a_directory_is_not_mistaken_for_the_executable(tmp_path, monkeypatch):
    (tmp_path / "boltz").mkdir()
    monkeypatch.setenv("PXD_BOLTZ_BIN", str(tmp_path / "boltz"))
    monkeypatch.setattr("pxdbench.toolenv.PROJECT_ENV_ROOT", str(tmp_path / "absent"))
    monkeypatch.setattr("shutil.which", lambda _: None)
    with pytest.raises(ToolNotFound):
        resolve_boltz_bin()


# --------------------------------------------------------------------------- #
# Failing usefully
# --------------------------------------------------------------------------- #


def test_missing_tool_names_everywhere_it_looked_and_the_remedy(tmp_path, monkeypatch):
    monkeypatch.setattr("pxdbench.toolenv.PROJECT_ENV_ROOT", str(tmp_path / "absent"))
    monkeypatch.setattr("shutil.which", lambda _: None)
    with pytest.raises(ToolNotFound) as exc:
        resolve_boltz_bin()
    msg = str(exc.value)
    assert "PXD_BOLTZ_BIN" in msg            # the override
    assert "setup.sh" in msg                 # how to create it
    assert ".pxd/envs/boltz" in msg          # where it looked
    assert "PATH" in msg


def test_required_false_reports_absence_instead_of_raising(tmp_path, monkeypatch):
    monkeypatch.setattr("pxdbench.toolenv.PROJECT_ENV_ROOT", str(tmp_path / "absent"))
    monkeypatch.setattr("shutil.which", lambda _: None)
    assert resolve_boltz_bin(required=False) is None


def test_diffusion_python_resolves_and_fails_the_same_way(tmp_path, monkeypatch):
    monkeypatch.setattr("pxdbench.toolenv.PROJECT_ENV_ROOT", str(tmp_path / "absent"))
    with pytest.raises(ToolNotFound) as exc:
        resolve_diffusion_python()
    assert "setup.sh --pxd" in str(exc.value)
    assert "protenix 2.x" in str(exc.value)

    own = tmp_path / "envs" / "pxd" / "bin"
    own.mkdir(parents=True)
    target = _exe(own / "python")
    monkeypatch.setattr("pxdbench.toolenv.PROJECT_ENV_ROOT", str(tmp_path / "envs"))
    assert resolve_diffusion_python() == target


def test_project_env_bin_requires_executability(tmp_path, monkeypatch):
    monkeypatch.setattr("pxdbench.toolenv.PROJECT_ENV_ROOT", str(tmp_path))
    (tmp_path / "boltz" / "bin").mkdir(parents=True)
    (tmp_path / "boltz" / "bin" / "boltz").write_text("plain file")
    assert project_env_bin("boltz", "boltz") is None
