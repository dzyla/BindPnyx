# AlphaFold 3 install for a Slurm cluster, disk-safe, no sequence databases (MSAs are supplied as files).
# Fill in the four placeholders with sed BEFORE submitting:  YOUR_ACCOUNT  AF3_ROOT  SCRATCH_DIR  CONDA_DIR
# Needs in AF3_ROOT before running: af3_src.tgz (git archive of the tracked source INCLUDING the top-level *.md and LICENSE files:
#   pyproject.toml references README.md and the CMake install step copies the terms files), req-af3.txt (pip freeze of a working env),
#   models/af3.bin.zst (your own licensed weights; keep the directory private: the terms forbid sharing them outside your organisation).
# Three things this script exists to handle (each cost one failed attempt): (1) the cluster compiler was too old for C++20, so a conda-forge GCC 13+
# is installed INTO the env; (2) zlib.h is not on the compute nodes, so zlib and bzip2 are installed into the env; (3) Slurm says COMPLETED even
# when the build failed, so the job ends with an explicit import test and a non-zero exit.
#!/bin/bash
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --job-name=af3_install3
#SBATCH --output=AF3_ROOT/logs/install3_%j.out
#SBATCH --error=AF3_ROOT/logs/install3_%j.err
set -u; export TERM=dumb
R=AF3_ROOT; SC=SCRATCH_DIR; CONDA=CONDA_DIR
avail_gb() { curc-quota 2>/dev/null | awk 'index($0, "/projects/" ENVIRON["USER"]) {v=$3; u=substr(v,length(v)); n=substr(v,1,length(v)-1); if(u=="T")n*=1024; else if(u=="M")n/=1024; print int(n)}'; }
guard() { a=$(avail_gb); echo "[guard $1] /projects free: ${a:-unknown} GB"; if [ -z "${a:-}" ] || [ "$a" -lt "$2" ]; then echo "ABORT: less than $2 GB free (or unreadable) in /projects"; exit 4; fi; }
echo "HOST $(hostname) $(date)"; guard start 20
mkdir -p $SC/pip $SC/pkgs; export PIP_CACHE_DIR=$SC/pip CONDA_PKGS_DIRS=$SC/pkgs TMPDIR=$SC
test -f $R/src/README.md || { echo "ABORT: src/README.md missing"; exit 5; }
source $CONDA/etc/profile.d/conda.sh
conda install -p $R/env -c conda-forge --override-channels "gxx_linux-64>=13" "libstdcxx-ng>=13" zlib bzip2 -y -q 2>&1 | tail -3
guard after-compilers 12
export CPATH=$R/env/include LIBRARY_PATH=$R/env/lib CMAKE_PREFIX_PATH=$R/env LD_LIBRARY_PATH=$R/env/lib:${LD_LIBRARY_PATH:-}
ls $R/env/include/zlib.h || { echo "ABORT: zlib.h not in env"; exit 7; }
export CC=$R/env/bin/x86_64-conda-linux-gnu-gcc CXX=$R/env/bin/x86_64-conda-linux-gnu-g++
echo "CXX: $($CXX --version | head -1)"
cd $R/src && $R/env/bin/python -m pip install --no-deps . > $SC/build2.log 2>&1; RC=$?; echo "pip build rc=$RC"
if [ $RC -ne 0 ]; then grep -n -i -E "error:|fatal error|not found|undefined reference" $SC/build2.log | head -30; tail -15 $SC/build2.log | cut -c1-200; fi
test -x $R/env/bin/build_data && $R/env/bin/build_data 2>&1 | tail -3
$R/env/bin/python -c "import alphafold3, jax; print('alphafold3 import ok; jax', jax.__version__)" 2>&1 | tail -2
echo "--- sizes"; du -sh $R/env $R/src 2>/dev/null
cp $SC/build2.log $R/logs/build3_full.log 2>/dev/null; rm -rf $SC/pip $SC/pkgs $SC/build2.log; guard end 10
if $R/env/bin/python -c "import alphafold3" 2>/dev/null; then echo "INSTALL2 OK $(date)"; else echo "INSTALL2 FAILED: alphafold3 not importable"; exit 6; fi
