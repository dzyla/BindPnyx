#!/bin/bash
# wait for the S1 queue (PID $1) to exit, then require its artifact, then run S2. Never gate on exit status alone.
while kill -0 "$1" 2>/dev/null; do sleep 30; done
[ -s out/phbind/designs_all.csv ] || { echo "S1 produced no designs_all.csv"; exit 1; }
PYTHONPATH=$(pwd) exec .pxd/envs/pxd/bin/python -u phbind/s2_prescreen.py
