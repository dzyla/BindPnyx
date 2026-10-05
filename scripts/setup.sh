#!/usr/bin/env bash
# Build the project's OWN environments under <repo>/.pxd/envs.
#
# Nothing here borrows another project's conda env. An earlier revision pointed
# at one person's `boltz2local` env; on every other machine the pipeline ran
# diffusion and MPNN to completion and then died at the first fold with
# FileNotFoundError. The project now owns its dependencies.
#
# Two environments, because Boltz and the pipeline pin incompatible torch
# builds. Boltz is invoked as a subprocess, so they never share an interpreter.
#
#   ./scripts/setup.sh            # both
#   ./scripts/setup.sh --boltz    # just the scorer
#   ./scripts/setup.sh --pxd      # just diffusion + MPNN
#   ./scripts/setup.sh --check    # report, change nothing
#
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENVS="$REPO/.pxd/envs"

# Versions that are MEASURED to work together on sm_120 (RTX 5090), from the
# environment this pipeline was developed and benchmarked in. Do not float them
# casually: protenix 0.5.5 pins torch==2.3.1, which reports CUDA as available on
# sm_120 and then dies on the first kernel launch.
PY_VERSION="3.11"
TORCH_SPEC="torch==2.11.0"
TORCH_INDEX="https://download.pytorch.org/whl/cu128"
PROTENIX_SPEC="protenix==2.0.0"
BOLTZ_SPEC="boltz==2.2.1"

DO_BOLTZ=false; DO_PXD=false; CHECK_ONLY=false; FORCE=false
if [[ $# -eq 0 ]]; then DO_BOLTZ=true; DO_PXD=true; fi
while [[ $# -gt 0 ]]; do
    case "$1" in
        --boltz) DO_BOLTZ=true; shift ;;
        --pxd)   DO_PXD=true; shift ;;
        --all)   DO_BOLTZ=true; DO_PXD=true; shift ;;
        --check) CHECK_ONLY=true; shift ;;
        --force) FORCE=true; shift ;;
        -h|--help) sed -n '2,15p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

say() { printf '%s\n' "$*"; }

# --- find a base interpreter to build venvs from -------------------------------
find_base_python() {
    local c
    for c in "python${PY_VERSION}" python3.12 python3.11 python3; do
        if command -v "$c" >/dev/null 2>&1; then
            local v
            v=$("$c" -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>/dev/null || echo "")
            case "$v" in
                3.10|3.11|3.12|3.13) command -v "$c"; return 0 ;;
            esac
        fi
    done
    return 1
}

report() {
    say "== project environments =="
    say "  root: $ENVS"
    local name exe
    for name in boltz pxd; do
        exe="$ENVS/$name/bin/python"
        if [[ -x "$exe" ]]; then
            say "  [x] $name  ($("$exe" -c 'import sys;print("python %d.%d"%sys.version_info[:2])'))"
            case "$name" in
                boltz) "$exe" -c "
from importlib.metadata import version
try: print('        boltz', version('boltz'))
except Exception: print('        boltz NOT installed')" ;;
                pxd) "$exe" -c "
try:
    import protenix, torch
    print('        protenix', protenix.__version__, '/ torch', torch.__version__)
except Exception as e: print('        incomplete:', str(e)[:70])" ;;
            esac
        else
            say "  [ ] $name  -- not created"
        fi
    done
    say ""
    say "The pipeline uses these first, then \$PXD_BOLTZ_BIN / \$PXD_PYTHON,"
    say "then PATH. Run without --check to create what is missing."
}

if $CHECK_ONLY; then report; exit 0; fi

BASE_PY="$(find_base_python)" || {
    say "FAIL: no python 3.10-3.13 found to build environments from."
    say "Install one (e.g. 'apt install python${PY_VERSION}-venv') and re-run."
    exit 1
}
say "base interpreter: $BASE_PY ($("$BASE_PY" -V 2>&1))"
mkdir -p "$ENVS"

make_env() {
    local name="$1"; shift
    local target="$ENVS/$name"
    if [[ -x "$target/bin/python" ]] && ! $FORCE; then
        say "  $name already exists (use --force to rebuild)"
        return 0
    fi
    $FORCE && rm -rf "$target"
    say "  creating $target"
    "$BASE_PY" -m venv "$target" || {
        say "  FAIL: could not create the venv. The python3-venv package may be missing."
        return 1
    }
    "$target/bin/python" -m pip install --quiet --upgrade pip setuptools wheel
}

# --- the scorer ----------------------------------------------------------------
if $DO_BOLTZ; then
    say ""
    say "== boltz env =="
    make_env boltz
    PIP="$ENVS/boltz/bin/pip"
    if "$ENVS/boltz/bin/python" -c "import boltz" 2>/dev/null && ! $FORCE; then
        say "  boltz already installed"
    else
        say "  installing $BOLTZ_SPEC (this pulls its own torch; several GB)"
        "$PIP" install "$BOLTZ_SPEC"
    fi
    "$ENVS/boltz/bin/python" - <<'PY'
from importlib.metadata import version
v = version("boltz")
print(f"  boltz {v} installed")
if v != "2.2.1":
    print(f"  WARNING: parse.py was written against 2.2.1; {v} may lay out "
          f"its results differently")
PY
fi

# --- diffusion + MPNN ----------------------------------------------------------
if $DO_PXD; then
    say ""
    say "== pxd env (diffusion + MPNN) =="
    make_env pxd
    PIP="$ENVS/pxd/bin/pip"
    if "$ENVS/pxd/bin/python" -c "
from protenix.model.protenix import update_input_feature_dict" 2>/dev/null && ! $FORCE; then
        say "  protenix 2.x already installed"
    else
        # torch FIRST from the CUDA index: installing protenix first lets its
        # own pin drag in a torch that cannot run on this GPU.
        say "  installing $TORCH_SPEC from $TORCH_INDEX"
        "$PIP" install --index-url "$TORCH_INDEX" "$TORCH_SPEC"
        say "  installing $PROTENIX_SPEC"
        "$PIP" install "$PROTENIX_SPEC"
        # What the pipeline itself imports, beyond protenix's own deps.
        say "  installing pipeline dependencies"
        "$PIP" install natsort biopython pandas scipy ml_collections \
                      "jax[cuda12]" dm-haiku joblib
    fi
    say "  verifying"
    "$ENVS/pxd/bin/python" - <<'PY'
import sys
try:
    import torch
    from protenix.model.protenix import update_input_feature_dict  # noqa: F401
except Exception as exc:
    sys.exit(f"  FAIL: {exc}")
print(f"  protenix 2.x OK / torch {torch.__version__}")
if torch.cuda.is_available():
    a = torch.randn(256, 256, device="cuda")   # a REAL matmul, not the flag
    float((a @ a).sum())
    print(f"  CUDA OK: {torch.cuda.get_device_name(0)} "
          f"sm{torch.cuda.get_device_capability(0)}")
else:
    print("  WARNING: no CUDA device visible; diffusion will not run")
PY
    cat <<'EOF'
  NOTE: protenix tries to JIT-build a fused LayerNorm on first import and that
  build FAILS against torch 2.11 headers with GCC >= 14 (a C++ error in
  ATen/core/List_inl.h, unrelated to the GPU). It is harmless - the code falls
  back to torch.nn.functional.layer_norm, numerically identical - but costs ~80s
  on every launch. ./scripts/setup_extras.sh builds it with a one-line header
  override (funnel/build_layernorm.py) and removes that delay.
EOF
fi

say ""
report
