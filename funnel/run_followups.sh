#!/usr/bin/env bash
# Queue after the main evaluation: (1) fast-screen calibration on random samples, (2) MPNN interface-bias experiment at two strengths on the SAME backbones
# as the default funnel (paired comparison). One GPU job at a time.
cd "$(dirname "$0")/.."; PY=.pxd/envs/pxd/bin/python; export PYTHONPATH=$PWD
for T in pdl1 mdm2 fima; do echo "== calibration $T $(date +%T)"; $PY -u funnel/screen_calibration.py $T 150 > out/calib_$T.log 2>&1 || echo "  FAILED"; done
for T in pdl1 mdm2; do for S in 0.6 1.0; do
  echo "== bias iface:$S $T $(date +%T)"
  mkdir -p out/funnel_bias$S/$T; [ -d out/funnel_bias$S/$T/gen ] || cp -r out/funnel/$T/gen out/funnel_bias$S/$T/gen   # SAME backbones as the default funnel: the arms differ only in MPNN
  $PY -u funnel/run_funnel.py --target $T --out out/funnel_bias$S/$T --n-backbones 500 --rounds 0 --mpnn-bias iface:$S > out/funnel_bias${S}_$T.log 2>&1 || echo "  FAILED"
done
  echo "== bias judge $T $(date +%T)"
  $PY -u funnel/judge.py --target $T --out out/judge_bias/$T --arm "funnel_nocycle=out/funnel/$T/final_nocycle.csv" --arm "bias0.6=out/funnel_bias0.6/$T/final_nocycle.csv" --arm "bias1.0=out/funnel_bias1.0/$T/final_nocycle.csv" > out/judge_bias_$T.log 2>&1 || echo "  judge FAILED"
done
echo "== followups done $(date +%T)"
