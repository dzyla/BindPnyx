#!/bin/bash
# BLOCK 3 - shim -> per-pose terms -> in-batch band -> control block -> permutation margins.
#
# This is the block that decides whether a batch is EVIDENCE. The band is derived from this
# batch's own positive controls and from nothing else: a window copied from an earlier run is a
# cross-batch comparison wearing a within-batch disguise, and the same anchor ceiling has come out
# 3.01 A with two positive controls and 3.42 A with three.
#
# It refuses to score an incomplete batch. A rate over 19 poses and a rate over 25 are not the
# same measurement, and a batch that is short is a batch whose queue is still running.
#
# Read the control block BEFORE any design row. Positives at or near full rate, negatives at or
# near zero. If they do not read that way the batch is dropped, not explained.
#
# Usage: score_batch.sh <batch_dir> <af3|protenix> [samples_per_case]
set -eu
B=$(readlink -f "${1:?usage: score_batch.sh <batch_dir> <af3|protenix> [n]}")
ORACLE=${2:?second argument must be af3 or protenix}
NS=${3:-25}
S="${B}_shim"
C=${CAMPAIGN:?set CAMPAIGN to your campaign workspace root}
PY=${PY:?set PY to a python with gemmi and numpy}

if [ "$ORACLE" = af3 ]; then
  n=$(find "$B/out" -name '*_model.cif' -path '*seed-*' | wc -l); SHIM=af3_shim.py
else
  n=$(find "$B/out" -name '*.cif' | wc -l); SHIM=px_shim.py
fi
want=$(( $(grep -cve '^[[:space:]]*$' "$B/cases.txt") * NS ))
echo "[score] $n structures, expected $want"
[ "$n" -ge "$want" ] || { echo "[score] batch is short - refusing to score a partial batch"; exit 1; }

mkdir -p "$S"
"$PY" -I "$C/scripts/$SHIM" "$B" "$S" --designs "$B/designs.csv" --min-cif "$NS"
[ -f "$B/manifest.json" ] && cp "$B/manifest.json" "$S/manifest.json"

echo; echo "=== per-pose terms and the in-batch band ==="
"$PY" "$C/scripts/joint_rank.py" --run "$S" --terms-only 2>&1 | tail -4
echo; echo "=== control block - read this before any design row ==="
"$PY" -I "$C/scripts/control_rates.py" "$S"
echo; echo "=== permutation margins ==="
"$PY" -I "$C/scripts/perm_margin.py" "$S" --csv "$S/perm_margin.csv" 2>&1 | tail -40
echo; echo "scored -> $S"
