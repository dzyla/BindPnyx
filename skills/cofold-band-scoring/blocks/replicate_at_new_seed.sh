#!/bin/bash
# BLOCK 6 - re-run an existing batch at a seed it has never used, then diff the two bands.
#
# The only kind of replicate these oracles permit. They are deterministic given seed and input,
# so re-running at the same seeds reproduces the band bit-identically and every rate exactly.
# That is a pipeline check, and this campaign ran one by accident and nearly reported it as a
# confirmation. This block therefore takes the new seed as a REQUIRED argument and refuses it if
# the source batch already carries it.
#
# Inputs are shared, not copied: json/ is symlinked to the source batch, so the sequences and any
# precomputed MSAs are byte-identical to the originals and cannot drift between the two runs.
# The only difference between the batches is the seed, which is what makes the comparison mean
# something.
#
# After scoring it runs block 4 automatically, because the first question about a replicate is
# whether the band moved - not whether the designs did.
#
# Usage: replicate_at_new_seed.sh <source_batch> <new_seed> <gpu> [af3|protenix]
set -eu
SRC=$(readlink -f "${1:?usage: replicate_at_new_seed.sh <source_batch> <new_seed> <gpu> [oracle]}")
SEED=${2:?a new seed is required - a replicate at an old seed is not a replicate}
GPU=${3:?gpu index required}
ORACLE=${4:-af3}
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PY:?set PY to a python with gemmi and numpy}
DST="${SRC}_seed${SEED}"

[ -r "$SRC/cases.txt" ] || { echo "[FATAL] $SRC has no cases.txt"; exit 1; }

# refuse a seed the source already used, from whichever place records it
used=""
[ -f "$SRC/manifest.json" ] && used=$("$PY" - "$SRC/manifest.json" <<'PYEOF'
import json, sys
print(",".join(str(s) for s in json.load(open(sys.argv[1])).get("seeds", [])))
PYEOF
)
if [ -z "$used" ]; then
  f=$(find "$SRC/json" -name '*.json' | head -1)
  [ -n "$f" ] && used=$("$PY" - "$f" <<'PYEOF'
import json, sys
print(",".join(str(s) for s in json.load(open(sys.argv[1])).get("modelSeeds", [])))
PYEOF
)
fi
case ",$used," in
  *",$SEED,"*) echo "[FATAL] $SRC already used seed $SEED (seeds: $used)."
               echo "        Re-running at a seed the batch already carries reproduces its"
               echo "        numbers exactly and is a determinism check, not a replicate."
               exit 1 ;;
esac
echo "[replicate] source seeds: ${used:-unknown}; new seed: $SEED"

mkdir -p "$DST"
cp "$SRC/cases.txt" "$DST/cases.txt"
[ -f "$SRC/designs.csv" ] && cp "$SRC/designs.csv" "$DST/designs.csv"
[ -f "$SRC/provenance.csv" ] && cp "$SRC/provenance.csv" "$DST/provenance.csv"

if [ "$ORACLE" = protenix ]; then
  # Protenix takes the seed on the command line, so the inputs are shared untouched.
  ln -sfn "$SRC/json" "$DST/json"
else
  # AF3 reads modelSeeds from the input, so each json is rewritten with ONLY the seed changed.
  mkdir -p "$DST/json"
  "$PY" - "$SRC/json" "$DST/json" "$SEED" <<'PYEOF'
import glob, json, os, sys
src, dst, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
n = 0
for p in sorted(glob.glob(os.path.join(src, "*.json"))):
    d = json.load(open(p))
    old = d.get("modelSeeds") or []
    d["modelSeeds"] = [seed + i for i in range(len(old) or 5)]
    json.dump(d, open(os.path.join(dst, os.path.basename(p)), "w"), indent=1)
    n += 1
print("rewrote %d inputs with seeds starting at %d (sequences untouched)" % (n, seed))
PYEOF
fi

if [ -f "$SRC/manifest.json" ]; then
  "$PY" - "$SRC/manifest.json" "$DST/manifest.json" "$SEED" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))
n = len(d.get("seeds") or []) or 5
d["seeds"] = [int(sys.argv[3]) + i for i in range(n)]
d["replicate_of"] = sys.argv[1]
json.dump(d, open(sys.argv[2], "w"), indent=1)
PYEOF
fi

bash "$HERE/preflight.sh" "$DST" || exit 1
GPU=$GPU bash "$HERE/run_queue.sh" "$DST" "$ORACLE" "$SEED"
bash "$HERE/score_batch.sh" "$DST" "$ORACLE"
echo
echo "=== block 4: did the BAND move? ask this before reading any design row ==="
"$PY" "$HERE/band_diff.py" "${SRC}_shim" "${DST}_shim"
