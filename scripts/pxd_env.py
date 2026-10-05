"""Find what this machine can actually run, and say so.

The paths in an earlier revision of the docs were absolute and machine-specific.
They worked on the host they were written on and nowhere else: a second machine
had no `miniconda3/envs/pxdesign`, no shared checkpoint mount, and only protenix
0.5.5 - so the campaign runner failed at preflight with a stale path and no
suggestion of what to do instead.

This module discovers each component, reports what is usable, and - importantly -
says which PARTS of the pipeline can run when not all of them can. A machine that
cannot diffuse backbones can still score existing ones and run the tests.

  python scripts/pxd_env.py                 # human-readable report
  python scripts/pxd_env.py --json          # machine-readable
  python scripts/pxd_env.py --what diffusion_python   # just the path, for scripts

Overrides always win: PXD_PYTHON, PXD_BOLTZ_BIN, PXD_CHECKPOINTS.
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: Where conda environments tend to live. Order is preference, not importance.
CONDA_ROOTS = [
    os.path.expanduser("~/.claude-science/conda/envs"),
    os.path.expanduser("~/miniconda3/envs"),
    os.path.expanduser("~/anaconda3/envs"),
    os.path.expanduser("~/micromamba/envs"),
    os.path.expanduser("~/.conda/envs"),
    "/opt/conda/envs",
]

CHECKPOINT_DIRS = [
    os.path.join(REPO, "checkpoints"),
    os.path.join(REPO, "release_data", "checkpoint"),
    os.path.expanduser("~/checkpoint"),
]

#: the one checkpoint diffusion cannot start without
DIFFUSION_CKPT = "pxdesign_v0.1.0.pt"

_PROBE = """
import json, sys
out = {"python": sys.executable}
try:
    import protenix
    out["protenix"] = getattr(protenix, "__version__", "unknown")
except Exception as exc:
    out["protenix"] = None
    out["protenix_error"] = str(exc)[:120]
try:
    # the symbol pxdesign/model/embedders.py needs; absent before protenix 2.x
    from protenix.model.protenix import update_input_feature_dict  # noqa: F401
    out["diffusion_symbol"] = True
except Exception as exc:
    out["diffusion_symbol"] = False
    out["diffusion_error"] = str(exc)[:160]
try:
    import torch
    out["torch"] = torch.__version__
    out["cuda"] = bool(torch.cuda.is_available())
    if out["cuda"]:
        # a REAL matmul: torch 2.3/cu121 passes the availability flag on sm_120
        # and then dies on the first kernel launch
        a = torch.randn(256, 256, device="cuda")
        float((a @ a).sum())
        out["cuda_usable"] = True
        out["gpu"] = torch.cuda.get_device_name(0)
        out["sm"] = list(torch.cuda.get_device_capability(0))
    else:
        out["cuda_usable"] = False
except Exception as exc:
    out["torch"] = None
    out["cuda_usable"] = False
    out["torch_error"] = str(exc)[:160]
try:
    import pytest  # noqa: F401
    out["pytest"] = True
except Exception:
    out["pytest"] = False
# Can it import the PROJECT? pytest alone is not enough - the suite needs
# natsort, transformers, biopython and protenix to be importable together.
import os as _os
_repo = _os.environ.get("PXD_REPO", "")
if _repo:
    sys.path.insert(0, _repo)
try:
    import pxdbench.tasks.base  # noqa: F401
    import pxdesign.runner.helpers  # noqa: F401
    out["imports_project"] = True
except Exception as exc:
    out["imports_project"] = False
    out["project_error"] = str(exc)[:160]
print("PXD_PROBE " + json.dumps(out))
"""


def candidate_pythons():
    """Every interpreter worth probing, overrides first, de-duplicated."""
    seen, out = set(), []

    def add(path):
        if path and os.path.isfile(path) and os.access(path, os.X_OK):
            real = os.path.realpath(path)
            if real not in seen:
                seen.add(real)
                out.append(path)

    add(os.environ.get("PXD_PYTHON"))
    # The project owns its environments: toolenv.py looks in .pxd/envs FIRST,
    # and this function disagreeing with it is how a machine with a perfectly
    # good project env still paid for a full sweep of every conda root.
    add(os.path.join(REPO, ".pxd", "envs", "pxd", "bin", "python"))
    for root in CONDA_ROOTS:
        for env in sorted(glob.glob(os.path.join(root, "*"))):
            add(os.path.join(env, "bin", "python"))
    add(shutil.which("python3"))
    add(sys.executable)
    return out


def probe(python_path, timeout=120):
    """What that interpreter can do. Never raises."""
    try:
        env = dict(os.environ, PXD_REPO=REPO, PYTHONPATH=REPO)
        res = subprocess.run(
            [python_path, "-c", _PROBE],
            capture_output=True, text=True, timeout=timeout, env=env,
        )
        for line in (res.stdout or "").splitlines():
            if line.startswith("PXD_PROBE "):
                return json.loads(line[len("PXD_PROBE "):])
    except Exception:  # noqa: BLE001 - a broken env is data, not a crash
        pass
    return {"python": python_path, "protenix": None, "diffusion_symbol": False,
            "torch": None, "cuda_usable": False, "pytest": False,
            "imports_project": False}


#: the Boltz the scorer was written against; parse.py assumes this layout
BOLTZ_EXPECTED = "2.2.1"


def _boltz_version(path):
    """The boltz python's own package version. Cheaper and more reliable than
    running the CLI, which loads torch."""
    env_py = os.path.join(os.path.dirname(path), "python")
    if not os.access(env_py, os.X_OK):
        return None
    try:
        res = subprocess.run(
            [env_py, "-c",
             "import boltz,sys;"
             "from importlib.metadata import version;"
             "print(version('boltz'))"],
            capture_output=True, text=True, timeout=120,
        )
        return (res.stdout or "").strip() or None
    except Exception:  # noqa: BLE001
        return None


def find_boltz():
    """Prefer a boltz whose version matches what parse.py expects."""
    override = os.environ.get("PXD_BOLTZ_BIN")
    if override and os.access(override, os.X_OK):
        return override, _boltz_version(override)

    found = []
    for root in CONDA_ROOTS:
        for env in sorted(glob.glob(os.path.join(root, "*"))):
            candidate = os.path.join(env, "bin", "boltz")
            if os.access(candidate, os.X_OK):
                found.append(candidate)
    on_path = shutil.which("boltz")
    if on_path and on_path not in found:
        found.append(on_path)

    versioned = [(c, _boltz_version(c)) for c in found]
    for candidate, ver in versioned:
        if ver == BOLTZ_EXPECTED:
            return candidate, ver
    return versioned[0] if versioned else (None, None)


def find_checkpoints():
    """First directory holding a RESOLVABLE diffusion checkpoint.

    Resolvable matters: these are often symlinks onto a mount, and a mount that
    has gone away leaves a path that exists as a link and fails as a file.
    """
    dirs = []
    if os.environ.get("PXD_CHECKPOINTS"):
        dirs.append(os.environ["PXD_CHECKPOINTS"])
    dirs.extend(CHECKPOINT_DIRS)
    for d in dirs:
        target = os.path.join(d, DIFFUSION_CKPT)
        if os.path.isfile(target):          # follows symlinks
            return d, True
    for d in dirs:                           # exists but dangling: say so
        if os.path.lexists(os.path.join(d, DIFFUSION_CKPT)):
            return d, False
    return None, False


#: packages the repo legitimately vendors at its root
VENDORED = {"pxdesign", "pxdbench", "colabdesign"}


def shadowing_packages():
    """Repo-root directories that MASK an installed package of the same name.

    Everything runs with PYTHONPATH=<repo>, so a directory here wins over
    site-packages. A protenix source tree extracted into the repo root once
    shadowed the installed protenix 2.0.0 with an incomplete copy, and the whole
    project stopped importing: `No module named protenix.data.infer_data_pipeline`
    from a package that plainly existed. Nothing reported the cause.

    Returns [(name, repo_path, installed_path)] for unexpected shadowing only.
    """
    out = []
    for entry in sorted(os.listdir(REPO)):
        full = os.path.join(REPO, entry)
        if entry in VENDORED or entry.startswith(".") or not os.path.isdir(full):
            continue
        if not os.path.isfile(os.path.join(full, "__init__.py")):
            continue
        # is there an INSTALLED package of the same name it would mask?
        for root in CONDA_ROOTS:
            for env in glob.glob(os.path.join(root, "*")):
                for libdir in glob.glob(
                    os.path.join(env, "lib", "python3.*", "site-packages", entry)
                ):
                    if os.path.isdir(libdir):
                        out.append((entry, full, libdir))
                        break
                else:
                    continue
                break
            else:
                continue
            break
    return out


def _classify(e):
    """The two capability flags, in one place so lazy and full paths agree."""
    e["can_diffuse"] = bool(e.get("diffusion_symbol") and e.get("cuda_usable"))
    # pytest alone is useless if the env cannot import the project
    e["can_test"] = bool(e.get("pytest") and e.get("imports_project"))
    return e


def resolve_one(want):
    """One resolved value, probing LAZILY and stopping at the first match.

    `discover()` probes EVERY candidate interpreter and only then takes the
    first that qualifies. On a machine with many conda envs that is minutes of
    silence: each probe has a 120 s timeout, and any env holding protenix also
    pays ~80 s for the fused-LayerNorm build that always fails here and is
    therefore never cached. Measured: a campaign launch sat at `== preflight ==`
    for over three minutes having printed nothing, which is indistinguishable
    from a hang.

    Nothing that needs ONE value should pay for the full report, and `shadowing`
    needs no interpreter at all.
    """
    if want == "shadowing":
        return shadowing_packages()
    if want == "boltz_bin":
        return find_boltz()[0]
    if want == "checkpoints":
        return find_checkpoints()[0]

    need = {"diffusion_python": "can_diffuse", "test_python": "can_test"}.get(want)
    if need is None:
        return discover().get(want)
    for path in candidate_pythons():
        env = _classify(probe(path))
        if env[need]:
            return env["python"]
    return None


def discover():
    """Resolve every component and decide which stages can run.

    This probes every candidate interpreter on purpose: it backs the human
    report ("what can this machine actually run?"). Callers wanting a single
    value should use `resolve_one`, which stops at the first match.
    """
    envs = [_classify(probe(p)) for p in candidate_pythons()]

    diffusion = next((e for e in envs if e["can_diffuse"]), None)
    testing = next((e for e in envs if e["can_test"]), None)
    boltz, boltz_version = find_boltz()
    ckpt_dir, ckpt_ok = find_checkpoints()

    return {
        "envs": envs,
        "diffusion_python": diffusion["python"] if diffusion else None,
        "test_python": testing["python"] if testing else None,
        "boltz_bin": boltz,
        "boltz_version": boltz_version,
        "checkpoints": ckpt_dir,
        "checkpoints_usable": ckpt_ok,
        "shadowing": shadowing_packages(),
        "can": {
            "run_tests": bool(testing),
            "generate_backbones": bool(diffusion and ckpt_ok),
            "score_with_boltz": bool(boltz),
            "full_campaign": bool(diffusion and ckpt_ok and boltz),
        },
    }


def report(info):
    print("== what this machine can run ==\n")
    labels = {
        "run_tests": "run the test suite",
        "generate_backbones": "generate backbones (diffusion)",
        "score_with_boltz": "score complexes with Boltz-2",
        "full_campaign": "run a full campaign",
    }
    for key, label in labels.items():
        print(f"  [{'x' if info['can'][key] else ' '}] {label}")

    print("\n== resolved ==")
    for key in ("diffusion_python", "test_python", "boltz_bin", "checkpoints"):
        print(f"  {key:<18} {info[key] or '-- not found --'}")
    if info.get("boltz_version") and info["boltz_version"] != BOLTZ_EXPECTED:
        print(f"  {'':<18} ^ boltz {info['boltz_version']}, expected "
              f"{BOLTZ_EXPECTED}; parse.py assumes that output layout")
    if info["checkpoints"] and not info["checkpoints_usable"]:
        print(f"  {'':<18} ^ {DIFFUSION_CKPT} is a DANGLING symlink there")

    if info.get("shadowing"):
        print("\n== WARNING: repo directories masking installed packages ==")
        for name, repo_path, installed in info["shadowing"]:
            print(f"  {name}: {repo_path}")
            print(f"    masks {installed}")
        print("  Everything runs with PYTHONPATH=<repo>, so these win over the")
        print("  installed package. If that is not deliberate, move them aside.")

    print("\n== interpreters probed ==")
    for e in info["envs"]:
        name = os.path.basename(os.path.dirname(os.path.dirname(e["python"])))
        bits = []
        bits.append(f"protenix {e['protenix']}" if e.get("protenix") else "no protenix")
        if e.get("protenix") and not e.get("diffusion_symbol"):
            bits.append("too old for diffusion")
        if e.get("torch"):
            bits.append(f"torch {e['torch']}")
            bits.append("cuda OK" if e.get("cuda_usable") else "cuda unusable")
        if e.get("pytest"):
            bits.append("pytest")
        if e.get("imports_project"):
            bits.append("imports project")
        print(f"  {name:<18} {', '.join(bits)}")

    if not info["can"]["full_campaign"]:
        print("\n== what to do ==")
        if not info["diffusion_python"]:
            print("  * No interpreter can run diffusion. It needs protenix >= 2.x\n"
                  "    (pxdesign/model/embedders.py imports update_input_feature_dict,\n"
                  "     absent before 2.x) AND a working CUDA device.\n"
                  "    Point PXD_PYTHON at one, or create it:\n"
                  "      conda create -n pxdesign python=3.11 && \\\n"
                  "      <env>/bin/pip install 'protenix==2.0.0' torch\n"
                  "    Do NOT install protenix 2.x into an env you rely on:\n"
                  "    it JIT-builds a fused LayerNorm that fails on some torch\n"
                  "    versions, and 0.5.5 pins torch==2.3.1 which is broken on sm_120.")
        if not info["checkpoints_usable"]:
            print(f"  * No usable {DIFFUSION_CKPT}. Checked: "
                  f"{', '.join(CHECKPOINT_DIRS)}\n"
                  f"    Set PXD_CHECKPOINTS, or repair the symlinks - these often\n"
                  f"    point onto a mount that has gone away.")
        if not info["boltz_bin"]:
            print("  * No boltz executable. Set PXD_BOLTZ_BIN, or install Boltz 2.2.1\n"
                  "    in its own env (it is invoked as a subprocess, so it does not\n"
                  "    need to share the pipeline's env).")
        if info["can"]["run_tests"]:
            print("\n  The test suite runs regardless and covers everything except\n"
                  "  real folds:\n"
                  f"    PYTHONPATH={REPO} {info['test_python']} -m pytest tests -q")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--what", help="print one resolved value and exit")
    ap.add_argument(
        "--shadowing",
        action="store_true",
        help="print masked packages only; probes NO interpreters",
    )
    args = ap.parse_args()

    # Both of these resolve lazily. Only the full report below sweeps every
    # interpreter, because only it needs to.
    if args.shadowing:
        for name, repo_path, installed in shadowing_packages():
            print(f"{name}: {repo_path} masks {installed}")
        return 0
    if args.what:
        value = resolve_one(args.what)
        if not value:
            return 1
        print(value)
        return 0

    info = discover()
    if args.json:
        print(json.dumps(info, indent=2))
        return 0
    report(info)
    return 0 if info["can"]["full_campaign"] else 1


if __name__ == "__main__":
    sys.exit(main())
