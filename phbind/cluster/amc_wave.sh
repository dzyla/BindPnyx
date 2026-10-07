#!/bin/bash
# usage: amc_wave.sh <tag>   (LSF job: one GPU; one checkout per tag)
set -u; TAG=$1; ROOT=$HOME/phbind_amc; D=$ROOT/$TAG; mkdir -p $D $ROOT/export; cd $D
[ -d binder-design ] || git clone -q https://github.com/dzyla/binder-design
cd binder-design
echo "HOST $(hostname) GPU $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1) JOB ${LSB_JOBID:-none}"
PXP=$HOME/miniconda3/envs/pxdesign/bin/python
# real matmul gate (never torch.cuda.is_available alone); abort fast on an unsupported card
$HOME/miniconda3/envs/boltz2/bin/python -c "import torch;a=torch.randn(512,512,device='cuda');print('matmul',float((a@a).sum())!=0)" || { echo "ABORT boltz2 env cannot use this GPU"; exit 3; }
$PXP -c "import torch;a=torch.randn(512,512,device='cuda');float((a@a).sum())" || { echo "ABORT pxdesign env cannot use this GPU"; exit 3; }
export PYTHONPATH=$(pwd) PXD_PYTHON=$PXP PXD_BOLTZ_BIN=$HOME/miniconda3/envs/boltz2/bin/boltz
export PXD_CHECKPOINTS=$HOME/software/PXDesign/release_data/checkpoint
export TNF_BUNDLE=$ROOT/bundle PHBIND_CARRIERS=$ROOT/bundle/handoff_pxdesign_phbind/carriers.csv
export PHBIND_CONFIG=$ROOT/wave_$TAG.json XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_DEVICE_ORDER=PCI_BUS_ID
$PXP -u phbind/wave.py --until prescreen
echo "EXIT $? $(date)"
