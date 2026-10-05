#!/usr/bin/env bash
# End-to-end evaluation, strictly one GPU job at a time. ~14 h on one RTX 5090. Continues past a failed step.
#   funnel/run_all.sh [targets...]      default: mdm2 pdl1 fima
cd "$(dirname "$0")/.."
PY=.pxd/envs/pxd/bin/python; TARGETS=${@:-mdm2 pdl1 fima}; export PYTHONPATH=$PWD
for T in $TARGETS; do
  echo "== $T: baseline default $(date +%T)"; funnel/run_baseline.sh $T default 8 4 20
  echo "== $T: funnel $(date +%T)";   $PY -u funnel/run_funnel.py --target $T --out out/funnel/$T --n-backbones 500 > out/funnel_$T.log 2>&1 || echo "  funnel FAILED for $T"
  echo "== $T: baseline scaled $(date +%T)";  funnel/run_baseline.sh $T scaled 100 4 30
  echo "== $T: judge $(date +%T)"
  $PY -u funnel/judge.py --target $T --out out/judge/$T \
     --arm "default=out/baseline/${T}_default/run/design_outputs/*/summary.csv" --arm "scaled=out/baseline/${T}_scaled/run/design_outputs/*/summary.csv" \
     --arm "funnel_nocycle=out/funnel/$T/final_nocycle.csv" --arm "funnel=out/funnel/$T/final_cycled.csv" > out/judge_$T.log 2>&1 || echo "  judge FAILED for $T"
done
$PY funnel/compare.py out funnel/results; echo "== all done $(date +%T)"
