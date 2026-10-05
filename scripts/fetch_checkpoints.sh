#!/usr/bin/env bash
# Resolve the model weights this pipeline needs.
#
# Weights are NOT distributed with this repository: they are third party, they
# are large, and they carry their own terms. This script says where they come
# from and verifies what you ended up with.
#
# Resolution contract, which the pipeline itself uses - do not hardcode a path
# anywhere (CLAUDE.md section 2 records what that cost once):
#
#   $PXD_CHECKPOINTS        if set, the directory to use
#   <repo>/checkpoints      otherwise
#
# `scripts/pxd_env.py` additionally searches <repo>/release_data/checkpoint and
# ~/checkpoint, and reports which one it resolved.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT="${PXD_CHECKPOINTS:-$REPO/checkpoints}"

# The one checkpoint diffusion cannot start without (scripts/pxd_env.py:46).
REQUIRED=(pxdesign_v0.1.0.pt)
# Used by the optional Protenix scoring stage.
OPTIONAL=(protenix-v2.pt protenix_mini_default_v0.5.0.pt protenix_mini_tmpl_v0.5.0.pt)

cat <<TEXT
Checkpoint directory: $CKPT
  (override with PXD_CHECKPOINTS)

Where these come from
---------------------
  pxdesign_v0.1.0.pt   PXDesign release weights (ByteDance). Obtain from the
                       upstream PXDesign release and accept its terms.
  protenix-v2.pt       Protenix v2 weights. Required only for the optional
                       Protenix scoring stage; the Boltz path does not use it.
  protenix_mini_*.pt   Protenix mini variants, optional.

This script does not download anything on your behalf: the upstream terms are
for you to read and accept. Place the files in the directory above, or point
PXD_CHECKPOINTS at wherever they already are - a symlink farm is fine and is
what a developer machine typically has.

TEXT

mkdir -p "$CKPT"
status=0
echo "Required:"
for f in "${REQUIRED[@]}"; do
    if [[ -e "$CKPT/$f" ]]; then
        printf '  [ok]      %-40s %s\n' "$f" "$(sha256sum "$CKPT/$f" 2>/dev/null | cut -c1-16)"
    else
        printf '  [MISSING] %s\n' "$f"
        status=1
    fi
done
echo "Optional:"
for f in "${OPTIONAL[@]}"; do
    if [[ -e "$CKPT/$f" ]]; then
        printf '  [ok]      %-40s %s\n' "$f" "$(sha256sum "$CKPT/$f" 2>/dev/null | cut -c1-16)"
    else
        printf '  [absent]  %s\n' "$f"
    fi
done

if [[ $status -ne 0 ]]; then
    echo
    echo "FAIL: a required checkpoint is missing. Diffusion cannot start." >&2
    echo "Nothing downstream will warn you more clearly than this: the failure" >&2
    echo "otherwise appears after the GPU work, at the first fold." >&2
fi
exit $status
