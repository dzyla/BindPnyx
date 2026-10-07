# Running waves on an LSF cluster (one GPU job per wave)

Verified on a shared LSF cluster (AMC): `wave.py` runs unchanged inside a 1-GPU `bsub` job. A wave is generate -> pack -> Boltz-2 screen; AF3 needs a 96 GB card and stays on a workstation.

## Recipe
1. Stage once: clone the repo, copy the target bundle (`targets/ msa/*nulfix* msa/NULFIX.sha256 tables/` + `carriers.csv`), run `phbind/s0_target.py` (login node is fine, it is light).
2. One config per wave (`wave_<tag>.json`: distinct `campaign`, `seed_base`, set alias) and **one checkout per wave** (`phbind/cluster/amc_wave.sh <tag>` makes `~/phbind_amc/<tag>/binder-design`; `wave.py` refuses a shared one).
3. Submit: `bsub -q <gpu-queue> -m "<supported hosts>" -gpu "num=1:mode=exclusive_process" -n 6 -W 240 -J phb_<tag> -oo logs/<tag>.out -eo logs/<tag>.err ./amc_wave.sh <tag>`
4. Watch with `bpeek <jobid>` and by counting artifacts (`confidence_*_model_0.json`); never trust a log file or `DONE`. Check `bjobs` with `awk` on the job-name field, never `| head`.
5. Compare to the carriers: they must reproduce the reference values (here 0.778 / 0.745 / 0.000 against 0.787 / 0.753 / 0.000 on another GPU model).

## Traps found the hard way
- `-R "rusage[mem=32000]"` made the job pend forever ("resource reservation not satisfied"); the unit differs per cluster. Omit it or check `lsinfo`/`lsload` first.
- **Blackwell nodes (sm_120) abort older torch builds.** The script's real-matmul gate exits 3 in ~90 s; exclude those hosts for the PXDesign env with `-m`. A newer Boltz env may still run there, so screening-only jobs can use them.
- A fresh clone has no `release_data/ccd_cache`; PXDesign then downloads 550 MB from a server that times out intermittently. Keep **one shared cache** and symlink it into each checkout (the launchers do).
- `bsub -m` with several hosts is fine for `num=1`; pinning a single host for `num=4` pends for hours.
- Timing is dominated by fixed model-load cost: batch >= 60 designs per job.
