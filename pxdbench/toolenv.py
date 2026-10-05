"""Where the external tools live, without hard-coding anyone's home directory.

An earlier revision shipped `DEFAULT_BOLTZ_BIN =
"/home/someone/conda/envs/boltz2local/bin/boltz"`. That is one
person's unrelated environment, borrowed. On any other machine the pipeline ran
diffusion and MPNN to completion and then died at the first fold with
`FileNotFoundError: .../boltz2local/bin/boltz` - an hour of GPU time spent before
the missing dependency was discovered.

So: the project owns its environments. `scripts/setup.sh` builds them under
`<repo>/.pxd/envs/`, and that is the first place looked. An existing install can
still be pointed at with PXD_BOLTZ_BIN, and anything on PATH is used as a last
resort, but nothing outside the project is ever assumed to exist.

A missing tool raises ToolNotFound naming every place searched and the command
that creates it, because the useful moment to learn a dependency is absent is
before the GPU work, not after.
"""
import os
import shutil

#: repo root - two levels up from this file (pxdbench/toolenv.py)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: environments scripts/setup.sh creates; self-contained and gitignored
PROJECT_ENV_ROOT = os.path.join(REPO_ROOT, ".pxd", "envs")

#: the Boltz the parser was written against (parse.py assumes this layout)
BOLTZ_EXPECTED_VERSION = "2.2.1"


class ToolNotFound(RuntimeError):
    """An external tool could not be located. Carries the remedy."""


def project_env_bin(env_name, exe):
    """`<repo>/.pxd/envs/<env_name>/bin/<exe>` if it is executable."""
    candidate = os.path.join(PROJECT_ENV_ROOT, env_name, "bin", exe)
    return candidate if os.access(candidate, os.X_OK) else None


def _usable(path):
    return bool(path) and os.access(path, os.X_OK) and not os.path.isdir(path)


def resolve_boltz_bin(explicit=None, env_var="PXD_BOLTZ_BIN", required=True):
    """The boltz executable to run, or raise ToolNotFound explaining the fix.

    Order: explicit argument, then $PXD_BOLTZ_BIN, then the project's own env,
    then PATH. `required=False` returns None instead of raising, for callers that
    only want to report availability.
    """
    searched = []

    for label, candidate in (
        ("explicit boltz_bin setting", explicit),
        (f"${env_var}", os.environ.get(env_var)),
        ("project env (.pxd/envs/boltz)", project_env_bin("boltz", "boltz")),
        ("PATH", shutil.which("boltz")),
    ):
        if candidate:
            if _usable(candidate):
                return candidate
            searched.append(f"{label}: {candidate} (not executable)")
        else:
            searched.append(f"{label}: not set")

    if not required:
        return None

    # Say what IS on this machine, without using it. Borrowing another env
    # silently is the bug this module exists to prevent; leaving the operator to
    # guess is merely unkind.
    suggestion = ""
    nearby = _discover_boltz_installs()
    if nearby:
        suggestion = (
            "\nBoltz installs found on this machine (not used automatically):\n"
            + "".join(f"  {path}{extra}\n" for path, extra in nearby)
            + "To use one for this run:\n"
            f"  export {env_var}=<one of the above>\n"
        )

    raise ToolNotFound(
        "Boltz executable not found. Searched:\n  "
        + "\n  ".join(searched)
        + "\n\nCreate the project's own environment (touches no other env):\n"
        f"  {os.path.join(REPO_ROOT, 'scripts', 'setup.sh')} --boltz\n"
        + suggestion
    )


#: conda roots scanned ONLY to suggest, never to resolve
_SCAN_ROOTS = (
    "~/.claude-science/conda/envs",
    "~/miniconda3/envs",
    "~/anaconda3/envs",
    "~/micromamba/envs",
    "~/.conda/envs",
    "/opt/conda/envs",
)


def _discover_boltz_installs(limit=6):
    """Boltz executables present on this machine, for the error message only."""
    import glob

    found = []
    for root in _SCAN_ROOTS:
        for env in sorted(glob.glob(os.path.join(os.path.expanduser(root), "*"))):
            candidate = os.path.join(env, "bin", "boltz")
            if os.access(candidate, os.X_OK):
                found.append((candidate, ""))
                if len(found) >= limit:
                    return found
    return found


def resolve_diffusion_python(explicit=None, env_var="PXD_PYTHON", required=True):
    """The interpreter that can run diffusion (needs protenix 2.x).

    Not verified here - importability is the caller's check, since probing costs
    a subprocess. scripts/pxd_env.py does the full verification.
    """
    searched = []
    for label, candidate in (
        ("explicit setting", explicit),
        (f"${env_var}", os.environ.get(env_var)),
        ("project env (.pxd/envs/pxd)", project_env_bin("pxd", "python")),
    ):
        if candidate:
            if _usable(candidate):
                return candidate
            searched.append(f"{label}: {candidate} (not executable)")
        else:
            searched.append(f"{label}: not set")

    if not required:
        return None

    raise ToolNotFound(
        "No interpreter for the diffusion stage. Searched:\n  "
        + "\n  ".join(searched)
        + "\n\nCreate the project's own environment:\n"
        f"  {os.path.join(REPO_ROOT, 'scripts', 'setup.sh')} --pxd\n\n"
        f"Or point at an existing protenix 2.x env:\n"
        f"  export {env_var}=/path/to/env/bin/python"
    )
