#!/usr/bin/env bash
# Optional components used by the funnel, installed OUTSIDE the two core environments so nothing can disturb their numpy/torch pins.
#   ./scripts/setup_extras.sh            # install what is missing (idempotent)
#   ./scripts/setup_extras.sh --check    # report only
# Run ./scripts/setup.sh first. Needs internet (GitHub, PyPI) and, for the LayerNorm kernel, nvcc.
#   plotting/stats libs   -> pxd env            (matplotlib, scikit-learn: final_design plots, benchmark analysis)
#   fastPISA              -> .pxd/ext           (interface area, H-bonds, salt bridges, solvation energy; github.com/dzyla/fastPISA)
#   LightDock             -> .pxd/envs/lightdock (rigid-body docking; GPL-3.0, called as a subprocess, never imported)
#   fused LayerNorm       -> protenix            (removes the ~80 s failed compile at every Protenix start; builds a one-line header override, see funnel/build_layernorm.py)
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; ENVS="$REPO/.pxd/envs"; PX="$ENVS/pxd/bin"; CHECK=false; [[ "${1:-}" == "--check" ]] && CHECK=true
ok() { printf '  [x] %s\n' "$*"; }; no() { printf '  [ ] %s\n' "$*"; }
[[ -x "$PX/python" ]] || { echo "pxd env missing: run ./scripts/setup.sh first"; exit 1; }

have_plot() { "$PX/python" -c "import matplotlib, sklearn" 2>/dev/null; }
have_pisa() { [[ -d "$REPO/.pxd/ext/fastPISA/fastpisa" ]] && PYTHONPATH="$REPO/.pxd/ext/fastPISA:$REPO/.pxd/ext/py" "$PX/python" -c "import fastpisa, freesasa, gemmi" 2>/dev/null; }
have_ld()   { [[ -f "$ENVS/lightdock/bin/lightdock3.py" ]]; }
have_ln()   { ls "$("$PX/python" -c 'import protenix,os;print(os.path.dirname(protenix.__file__))' 2>/dev/null)"/model/layer_norm/fast_layer_norm_cuda_v2*.so >/dev/null 2>&1; }

echo "== extras =="
have_plot && ok "matplotlib + scikit-learn (pxd env)" || no "matplotlib + scikit-learn"
have_pisa && ok "fastPISA (.pxd/ext)" || no "fastPISA"
have_ld   && ok "LightDock (.pxd/envs/lightdock)" || no "LightDock"
have_ln   && ok "fused LayerNorm kernel installed" || no "fused LayerNorm kernel"
$CHECK && exit 0

have_plot || { echo "-- plotting libs"; "$PX/pip" install matplotlib scikit-learn; }
if ! have_pisa; then
  echo "-- fastPISA"; mkdir -p "$REPO/.pxd/ext"
  [[ -d "$REPO/.pxd/ext/fastPISA" ]] || git clone https://github.com/dzyla/fastPISA.git "$REPO/.pxd/ext/fastPISA"
  "$PX/pip" install --no-deps --target "$REPO/.pxd/ext/py" freesasa gemmi        # --no-deps: must never change the env's numpy / torch
fi
if ! have_ld; then
  echo "-- LightDock (own venv; numpy<2 and a relaxed C diagnostic are needed to compile its extensions with GCC >= 14)"
  "$PX/python" -m venv "$ENVS/lightdock" && "$ENVS/lightdock/bin/pip" install -q "numpy<2" "cython<3.1" setuptools wheel \
    && CFLAGS="-Wno-error=incompatible-pointer-types -Wno-incompatible-pointer-types" "$ENVS/lightdock/bin/pip" install -q --no-build-isolation lightdock
fi
if ! have_ln; then
  echo "-- fused LayerNorm"; if command -v nvcc >/dev/null 2>&1 || [[ -x /usr/local/cuda/bin/nvcc ]]; then PYTHONPATH="$REPO" "$PX/python" "$REPO/funnel/build_layernorm.py" --install && PYTHONPATH="$REPO" "$PX/python" "$REPO/funnel/build_layernorm.py" --verify
  else echo "   nvcc not found: skipped (harmless: Protenix falls back to torch.nn.functional.layer_norm)"; fi
fi
echo; "$0" --check
