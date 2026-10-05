#!/usr/bin/env bash
# Unattended test queue (plan and rationale: docs/OVERNIGHT_PLAN.md). One GPU job at a time (or --gpus workers). RESUMABLE: re-run this script after any interruption;
# finished steps are skipped (marker files in out/overnight/), an interrupted step continues from its last finished unit (every funnel stage is resumable).
#   funnel/run_overnight.sh                  # single GPU
#   GPUS=0,1,2,3 funnel/run_overnight.sh     # 4-GPU workstation: units run in parallel
cd "$(dirname "$0")/.."; PY=.pxd/envs/pxd/bin/python; export PYTHONPATH=$PWD; mkdir -p out/overnight
GP=""; [ -n "${GPUS:-}" ] && GP="--gpus $GPUS"
step() { n=$1; shift
  if [ -f out/overnight/done_$n ]; then echo "skip   $n"; return; fi
  echo "== $n $(date +%T)"
  if "$@" > out/overnight/$n.log 2>&1; then touch out/overnight/done_$n; echo "   ok  $n $(date +%T)"; else echo "   FAILED $n (see out/overnight/$n.log); continuing with the next step"; fi; }
fork_gen() { mkdir -p "$2"; [ -d "$2/gen" ] || cp -r "$1/gen" "$2/gen"; }      # same backbones for paired comparisons

[ -n "${WAIT_FOR:-}" ] && { echo "waiting for $WAIT_FOR"; while [ ! -f "$WAIT_FOR" ]; do sleep 20; done; }

# --- A. FimH (E. coli adhesin), mannose pocket: the headline test ------------------------------------------------------------
step fimh_fetch      $PY funnel/fetch_target.py funnel/targets/fimh.json
step fimh_funnel     $PY -u funnel/run_funnel.py --target fimh --out out/funnel/fimh --n-backbones 500 --chunk 100 --rounds 3 $GP
step fimh_dock       $PY -u funnel/dock_redesign.py --target fimh --out out/dock/fimh --scaffolds all --poses 8 --designs-per-pose 8 --rounds 3 --funnel-args "$GP"
step fimh_bias       bash -c "$(declare -f fork_gen); fork_gen out/funnel/fimh out/funnel_bias0.6/fimh; $PY -u funnel/run_funnel.py --target fimh --out out/funnel_bias0.6/fimh --n-backbones 500 --chunk 100 --rounds 0 --mpnn-bias iface:0.6 $GP"
step fimh_judge      $PY -u funnel/judge.py --target fimh --out out/judge/fimh --arm funnel_nocycle=out/funnel/fimh/final_nocycle.csv --arm funnel=out/funnel/fimh/final_cycled.csv --arm dock_redesign=out/dock/fimh/funnel/final_cycled.csv --arm bias0.6=out/funnel_bias0.6/fimh/final_nocycle.csv $GP
step fimh_controls   $PY -u funnel/controls.py --target fimh --designs out/funnel/fimh/final_cycled.csv --out out/controls/fimh --decoys pdl1,mdm2,fima --shuffles 2
step fimh_ctrl_dock  $PY -u funnel/controls.py --target fimh --designs out/dock/fimh/funnel/final_cycled.csv --out out/controls/fimh_dock --decoys pdl1,mdm2,fima --shuffles 2
step fimh_rim_fetch  $PY funnel/fetch_target.py funnel/targets/fimh_rim.json
step fimh_rim        $PY -u funnel/run_funnel.py --target fimh_rim --out out/funnel/fimh_rim --n-backbones 500 --chunk 100 --rounds 0 $GP

# --- B. Does the interface-only aromatic bias help on the targets with known results? (same backbones, paired) -------------
for T in pdl1 mdm2; do
  step ${T}_bias     bash -c "$(declare -f fork_gen); fork_gen out/funnel/$T out/funnel_bias0.6/$T; $PY -u funnel/run_funnel.py --target $T --out out/funnel_bias0.6/$T --n-backbones 500 --chunk 100 --rounds 0 --mpnn-bias iface:0.6 $GP"
  step ${T}_bias_judge $PY -u funnel/judge.py --target $T --out out/judge_bias/$T --arm funnel_nocycle=out/funnel/$T/final_nocycle.csv --arm bias0.6=out/funnel_bias0.6/$T/final_nocycle.csv $GP
done
echo "== queue finished $(date +%T)"
