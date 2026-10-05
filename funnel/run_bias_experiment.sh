#!/usr/bin/env bash
# Does an interface-only hydrophobic/aromatic MPNN bias improve designs? Runs the funnel (no cycling) with --mpnn-bias iface and judges it
# head-to-head against the existing default-MPNN funnel shortlist, in one batch with fresh seeds. One GPU job at a time.
#   funnel/run_bias_experiment.sh [targets...]
cd "$(dirname "$0")/.."; PY=.pxd/envs/pxd/bin/python; export PYTHONPATH=$PWD
for T in ${@:-pdl1 mdm2}; do
  echo "== $T: funnel + interface bias $(date +%T)"
  $PY -u funnel/run_funnel.py --target $T --out out/funnel_bias/$T --n-backbones 500 --rounds 0 --mpnn-bias iface > out/funnel_bias_$T.log 2>&1 || echo "  FAILED"
  echo "== $T: judge $(date +%T)"
  $PY -u funnel/judge.py --target $T --out out/judge_bias/$T \
     --arm "funnel_nocycle=out/funnel/$T/final_nocycle.csv" --arm "funnel_bias=out/funnel_bias/$T/final_nocycle.csv" > out/judge_bias_$T.log 2>&1 || echo "  judge FAILED"
  $PY funnel/add_pisa_to_runs.py $T out/judge_bias/$T/judged.csv out/funnel_bias/$T/final_nocycle.csv out/funnel_bias/$T/consensus_nocycle.csv 2>&1 | grep -v laio
done
echo "== bias experiment done $(date +%T)"
