#!/usr/bin/env bash
# Blind-dock 4 probes on 4 known-site targets and on the 4 benchmark targets, then validate. CPU only; resumable (finished probes are skipped).
cd "$(dirname "$0")/.."; PY=.pxd/envs/pxd/bin/python; export PYTHONPATH=$PWD; P=ubiquitin,protein_g_b1,sh3,protein_a_b
D() { n=$1; shift; [ -f out/epitope/$n/epitope.csv ] && return; $PY -u funnel/dock_epitope.py --out out/epitope/$n --probes $P "$@" > out/epitope/$n.log 2>&1; }
mkdir -p out/epitope
D pdl1 --pdb data/targets/pdl1/raw/3BIK.pdb --chain A --range 18-229
D mdm2 --pdb data/targets/mdm2/raw/1YCR.pdb --chain A --range 25-109
D fimh --pdb data/targets/fimh/raw/3MCY.pdb --chain A --range 1-158
D fima --pdb data/targets/fima/raw/4DWH.pdb --chain A
$PY funnel/bench_epitope_inputs.py        # writes out/epitope_bench/<t>/{receptor source cif, binder_epitope.npy}
for t in egfr il7r mdm2_b pd-l1; do
  src=$(cat out/epitope_bench/$t/source.txt); D bench_$t --pdb $src --chain A
  cp out/epitope_bench/$t/binder_epitope.npy out/epitope/bench_$t/binder_epitope.npy
done
$PY funnel/epitope_validate.py out/epitope
