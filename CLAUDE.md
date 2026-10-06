# Working in this repository

PXDesign diffusion → ProteinMPNN → Boltz-2 scoring. Findings and numbers: `docs/REPORT.md`.
**Operating playbook and guardrails: `AGENTS.md`. Recommended workflow: `funnel/` and `docs/RECOMMENDATIONS.md`.**
The previous, much longer version of this file is in `.archive/docs/CLAUDE.full.md`.

## Rules that are not negotiable

- **`State=COMPLETED` is not success, and neither is a non-empty `summary.csv`.** PXDesign
  exits 0 on internal failure, and a summary can be full of `bz_status=boltz_failed`. Check
  that rows have `bz_status == ok` (`run_campaign.sh` now does).
- **Run one GPU job at a time.** Concurrent Boltz/Protenix/JAX jobs on the 32 GB card gave
  `cusolverDnCreate ... INTERNAL_ERROR` and CUDA OOM; JAX preallocates ~75% of VRAM.
  Set `XLA_PYTHON_CLIENT_PREALLOCATE=false` if you must overlap CPU-side JAX work.
- **A silently discarded MSA inflates the score** (+0.16 ipSAE, 4/4 designs). `prepare_json`
  validates MSAs unconditionally; a dummied row returns `msa_discarded`, not a number.
- **Score comparison:** only rows sharing `bz_final_batch_id` are comparable. Batch effects
  are small (SD ~0.02) but real. Concurrency is by seed, never by design.
- **ipSAE differences below ~0.03 are noise**, and single-seed ipSAE rankings between two
  setups agree only moderately (Spearman 0.50 on 399 designs).
- **The gate is provisional.** `bz_gate_egfr_provisional_v1` (ipSAE ≥ 0.5, PAE_min ≤ 2, 3 seeds)
  is hard-wired to EGFR and, against ProteinBase labels, passes 78% of binders *and* 51% of
  confirmed non-binders. A gate pass is a candidate, not a binder.
- **Multi-chain targets use `phbind/`, not `funnel/`** (`docs/HOMO_OLIGOMER_TARGETS.md`): funnel/common reads the binder as a target chain and raises on >2 chains. Gate ipSAE on the direction the gate was calibrated on (min), and on counted output files.
- **Verify CUDA with a real matmul**, never `torch.cuda.is_available()` (RTX 5090 = sm_120).
- **Never `pkill -f` / `pgrep -f` a pattern that appears in your own command line**; take PIDs
  from `ps` and use `kill -0 PID` to wait.
- **Protenix checkpoint names lie.** `protenix_base_default_v0.5.0.pt` in `checkpoints/` and
  `~/checkpoint` is the **v2** file (same sha1); in a second checkpoint folder it is **v1.0.0**.
  No genuine 0.5 *base* checkpoint is on this machine; `protenix_mini_default_v0.5.0` is real 0.5.
  Pass `-n protenix-v2` explicitly and compare sha1 before trusting a name.

## Environments

`scripts/setup.sh` builds `<repo>/.pxd/envs/{pxd,boltz}` (here: symlinks to conda envs).
`pxdbench/toolenv.py` resolves tools: explicit setting, `$PXD_BOLTZ_BIN`/`$PXD_PYTHON`, project
env, `PATH`. **Never hardcode an env path.** Run with `PYTHONPATH=<repo>`; therefore any
directory at the repo root masks an installed package of the same name (`pxd_env.py --shadowing`).
Diffusion needs protenix 2.x. Boltz and the pipeline pin different torch builds, so they
never share an interpreter.

Known noise: protenix's fused-LayerNorm JIT build always fails on import (~80 s per process,
harmless, falls back to torch layer_norm, numerically identical). A stale
`protenix/model/layer_norm/lock` hangs it; delete the lock if no `nvcc` is running.
`ld: cannot find -laio` at pytest start is DeepSpeed probing; ignore it.

## Running

```bash
./scripts/run_campaign.sh -i manifests/x.json -o out/run1 --backbones 100 --seqs 4
python3 scripts/preflight_target.py manifests/x.json
python scripts/prepare_target.py --help          # shard from a PDB, with provenance
PYTHONPATH=$(pwd) .pxd/envs/pxd/bin/python -m pytest tests -q   # 691 pass, 21 skipped (no GPU / private data / opt-in real Boltz); bare env: heavy-dep tests skip
```

Use `python -u` when redirecting. Capture `$?` on the line after the command, not after `echo`.
`PXDBENCH_BACKEND` must stay unset (it is global and silently drops `ptx_*` columns).

## Pipeline

Diffusion is cheap (~2-3 s/backbone; scoring is the cost). It emits **coordinates only** (binder = GLY placeholder). ProteinMPNN assigns every
residue. Boltz-2 ranks all designs (1 seed), re-folds per-backbone winners (3 seeds), then a
common equal-depth batch gives the final ranking. MPNN weights/temperature were tested and
do **not** change the score distribution (REPORT §3.5); backbone count does.

## Things that look like bugs and are not

`all_summary.csv`/`filtered_summary.csv` are deleted on purpose; `summary.csv` is the
deliverable. `N_sample/N_step/gamma0/eta` reaching `BoltzBackend.predict` are accepted and
ignored. The fused-LayerNorm build failure is harmless.

## Open

Nobody has wet-lab tested a design from this pipeline. ProteinBase's own PXDesign entries were
0 of 6 wet-lab-labelled designs (Nipah G; 31 submitted, 36 evaluation records incl. replicates). Epitope engagement is reported, not ranked on, and anti-correlated with
ipSAE in an earlier 32-design sample. Details and the history of each: `.archive/docs/CLAUDE.full.md`.

## Working style

Verify before asserting; check data contracts against real files; a test whose fixture encodes
your assumption proves nothing (delete the guard and confirm the test fails); write down what
a measurement does not show.
