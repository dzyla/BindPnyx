#!/usr/bin/env bash
# Pipeline baseline arm for a funnel target, with wall-clock accounting.
#   funnel/run_baseline.sh TARGET ARM BACKBONES SEQS KEEP     e.g.  funnel/run_baseline.sh mdm2 scaled 100 4 30
set -uo pipefail
cd "$(dirname "$0")/.."
T=$1; ARM=$2; NB=$3; NS=$4; KEEP=$5; OUT=out/baseline/${T}_${ARM}
mkdir -p "$OUT"; PY=$(python3 scripts/pxd_env.py --what diffusion_python 2>/dev/null || echo .pxd/envs/pxd/bin/python)
.pxd/envs/pxd/bin/python funnel/make_manifest.py "$T" "$OUT.json" >/dev/null
S=$(date +%s)
./scripts/run_campaign.sh -i "$OUT.json" -o "$OUT/run" --backbones "$NB" --seqs "$NS" --keep "$KEEP" > "$OUT/log.txt" 2>&1
RC=$?                              # capture $? on the next line: `cmd; echo` would report echo's status
E=$(date +%s); echo "{\"seconds\": $((E-S)), \"rc\": $RC}" > "$OUT/timer.json"; echo "$T $ARM rc=$RC $((E-S))s"
