#!/bin/bash
# BLOCK 2 - fold every case in a batch. Resumable, artefact-counted, one case per invocation.
#
# Why one case per invocation and not one call over the whole set: a crash in case 31 of 43 then
# costs case 31, not the batch, and the skip rule below makes the re-run free for everything
# already done. Why artefacts are COUNTED: these tools exit 0 having written nothing, so an exit
# code is not evidence that a structure exists.
#
# Do NOT edit cases.txt while this runs. bash reads a redirected file incrementally by byte
# offset, so rewriting it under the loop corrupts the read position - and in one incident an
# edit while four shells were executing the file turned a later line into a different command.
#
# Usage: run_queue.sh <batch_dir> <af3|protenix> [seed_for_protenix]
#        GPU=0 bash run_queue.sh runs/x99 af3
set -u
B=$(readlink -f "${1:?usage: run_queue.sh <batch_dir> <af3|protenix> [seed]}")
ORACLE=${2:?second argument must be af3 or protenix}
SEED=${3:-101}
GPU=${GPU:-0}
NS=${NS:-25}
CASES="$B/cases.txt"                       # absolutised BEFORE any cd below
AF3_HOME=${AF3_HOME:?set AF3_HOME to your AlphaFold 3 checkout}
PX=${PX:-$HOME/miniconda3/envs/protenixscore/bin/protenix}

[ -r "$CASES" ] || { echo "[FATAL] no cases.txt in $B"; exit 1; }
case "$ORACLE" in
  af3)      [ -x "$AF3_HOME/.conda/envs/af3/bin/python" ] || { echo "[FATAL] no af3 python under $AF3_HOME"; exit 1; } ;;
  protenix) [ -x "$PX" ] || { echo "[FATAL] protenix not executable: $PX"; exit 1; } ;;
  *)        echo "[FATAL] oracle must be af3 or protenix"; exit 1 ;;
esac

count() { find "$B/out/$1" -name '*_model.cif' -path '*seed-*' 2>/dev/null | wc -l; }
pxcount() { find "$B/out/$1" -name '*.cif' 2>/dev/null | wc -l; }

while read -r c; do
  [ -z "$c" ] && continue
  if [ "$ORACLE" = af3 ]; then n=$(count "$c"); else n=$(pxcount "$c"); fi
  if [ "$n" -ge "$NS" ]; then echo "[skip] $c ($n cif)"; continue; fi
  # NEVER build an rm path from a variable without checking it is non-empty: an empty $c makes
  # this `rm -rf out/`, which is how 212 completed cases were lost in this campaign.
  if [ "$ORACLE" = protenix ] && [ -n "$c" ] && [ -d "$B/out/$c" ]; then rm -rf "$B/out/$c"; fi
  echo "[run ] $(date +%H:%M:%S) $c gpu$GPU"
  if [ "$ORACLE" = af3 ]; then
    ( cd "$AF3_HOME" && ./.conda/envs/af3/bin/python run/run_af3.py \
        --json_path "$B/json/$c.json" --output_dir "$B/out" --force_output_dir 1 \
        --model_dir "$AF3_HOME/models" --gpu_device "$GPU" \
        --jax_compilation_cache_dir "$AF3_HOME/jax_cache" ) >> "$B/oracle_gpu${GPU}.log" 2>&1
    m=$(count "$c")
  else
    CUDA_VISIBLE_DEVICES=$GPU "$PX" pred -i "$B/json/$c.json" -o "$B/out/$c" -s "$SEED" -e "$NS" \
      -d bf16 --use_msa true --use_default_params true \
      --trimul_kernel torch --triatt_kernel torch >> "$B/oracle_gpu${GPU}.log" 2>&1
    m=$(pxcount "$c")
  fi
  echo "[done] $(date +%H:%M:%S) $c -> $m cif"
  [ "$m" -eq 0 ] && echo "[WARN] $c wrote nothing - check $B/oracle_gpu${GPU}.log"
done < "$CASES"
echo "[queue finished] $(date)"
