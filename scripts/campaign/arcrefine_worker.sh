#!/bin/bash
# One ArcRefine worker with STATIC job assignment (worker K of N takes jobs K, K+N, K+2N, ...).
#   ARC_BIN=/path/to/structural-carryover JOBS_ROOT=DIR  bash scripts/campaign/arcrefine_worker.sh K N [GPU]
# GPU defaults to K. Start N workers, one per GPU. Do NOT replace this with a claim-directory lock: two workers once took the same job and the loser died,
# because ArcRefine refuses to write into an existing --output directory. Resumable: a job with run/result.json is skipped; a partial run/ is removed first.
# If the jobs live on a network disk written by another machine, read results on THAT machine (a stale directory view is common).
set -u
K=${1:?worker index}; N=${2:?number of workers}; GPU=${3:-$K}
: "${ARC_BIN:?set ARC_BIN to the structural-carryover executable}"; : "${JOBS_ROOT:?set JOBS_ROOT to the folder of job directories}"
export CUDA_VISIBLE_DEVICES=$GPU XLA_PYTHON_CLIENT_PREALLOCATE=false
i=0
for J in $(ls -d "$JOBS_ROOT"/*/ | sort); do
  if [ $((i % N)) -ne "$K" ]; then i=$((i+1)); continue; fi
  i=$((i+1)); j=$(basename "$J")
  [ -f "$J/run/result.json" ] && { echo "[w$K] skip $j (done)"; continue; }
  rm -rf "$J/run"
  echo "[w$K] $(date +%T) start $j"
  ( cd "$J" && "$ARC_BIN" config.json --output run > run.log 2>&1 )
  rc=$?                                   # capture on the very next line: anything between (even date) would overwrite $?
  echo "[w$K] $(date +%T) done $j rc=$rc result=$([ -f "$J/run/result.json" ] && echo yes || echo no)"
done
echo "[w$K] $(date +%T) WORKER_DONE"
