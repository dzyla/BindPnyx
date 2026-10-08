# Smoke test: one complex on one 80 GB GPU, MSA injected (--norun_data_pipeline). Fill the placeholders as in the install script. Put an AF3 input JSON
# (with unpairedMsaPath pointing at your MSA file) in AF3_ROOT/smoke_in/smoke.json. Compare the ipSAE with a run on another machine before trusting it.
#!/bin/bash
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=aa100
#SBATCH --qos=gpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100_80gb:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --job-name=af3_smoke
#SBATCH --output=AF3_ROOT/logs/smoke_%j.out
#SBATCH --error=AF3_ROOT/logs/smoke_%j.err
set -u; R=AF3_ROOT; cd $R
echo "HOST $(hostname) GPU $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1) $(date)"
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="--xla_gpu_cuda_data_dir=CONDA_DIR"
mkdir -p smoke_out jaxcache
$R/env/bin/python $R/src/run_alphafold.py --json_path=$R/smoke_in/smoke.json --output_dir=$R/smoke_out --model_dir=$R/models --norun_data_pipeline --gpu_device=0 --num_diffusion_samples=5 --jax_compilation_cache_dir=$R/jaxcache
echo "AF3 EXIT $? $(date)"; find $R/smoke_out -name "*summary_confidences.json" | head -3; du -sh $R/smoke_out $R/jaxcache
