# Running the pipeline on a Slurm cluster (Alpine-style) and installing AlphaFold 3 there

Verified on one cluster, one account, one round. Templates: `phbind/cluster/alpine_af3_install.sbatch`, `phbind/cluster/alpine_af3_smoke.sbatch` (placeholders to fill in; nothing machine-specific is committed).

## Rules that were not optional
- **No compute on a login node.** Only editing, `sbatch`, `squeue`, `scancel`, small file copies. The cluster penalises the account automatically. Python, PROPKA, parsing, tarring a large tree all go in a job (a CPU partition is free).
- **A GPU submission needs all of:** account, partition, QOS, `--nodes=1`, `--ntasks`, an explicit GPU type in `--gres`, `--time`. Partition and QOS are paired (each partition accepts only its own QOS values) and errors arrive one at a time; `sbatch -v ... | grep -i error` shows the real one.
- **Host RAM is capped per GPU** (a hard submit rejection, not a warning); stay under the cap for the GPU type.
- **GPU caps are per type and count what you already hold** across every campaign; check `squeue -u $USER -o %b` before choosing a type. Blackwell-class slices were avoided (older framework builds have no kernels for them).
- **Disk is the usual binding constraint.** Check free quota at the start of a job and again after each big step (the install script does, and aborts under its threshold). Keep pip and conda caches on purgeable scratch and delete them at the end. Never download the sequence databases (~600 GB) when MSAs are supplied.
- **`COMPLETED` is not success.** Slurm reported COMPLETED for an install that had not installed anything. End every setup job with an explicit test and a non-zero exit.

## AlphaFold 3 on the cluster (what it took)
Three failed builds, each informative: (1) the source archive I made omitted `README.md`, which `pyproject.toml` references; (2) the system compiler (GCC 8.5) cannot build the C++20 extension, so install a conda-forge `gxx_linux-64>=13` and `libstdcxx-ng` into the env; (3) `zlib.h` is absent on the compute nodes, so install `zlib` and `bzip2` into the env and set `CPATH`, `LIBRARY_PATH`, `CMAKE_PREFIX_PATH`; and the CMake install step also copies the licence/terms files, so the archive must include the repo's top-level `*.md` and `LICENSE`. Pin the Python packages with a `pip freeze` of a working environment. Final footprint: ~7.7 GB environment, 19 MB source, ~1 GB weights.
**Cross-check before using it:** the same design, input and seed gave an AF3 ipSAE-min of 0.687 on an A100 80 GB and 0.692 on a Blackwell workstation GPU (4 min 19 s on the A100 including JIT compilation, 5 samples). Run your own pair before trusting a new machine.
**Weights:** the AF3 terms allow sharing the model parameters only within your organisation and limit use to non-commercial research. They are never in this repository; copy your own licensed file into a private directory (mode 600 on the file, 700 on the directory) and verify the checksum inside a job.
