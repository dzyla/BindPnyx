# Instructions for AI agents operating this repository

You are running a computational binder-design workflow (PXDesign diffusion → ProteinMPNN → Protenix/Boltz-2 scoring) on one shared RTX 5090.
Read `docs/RECOMMENDATIONS.md` for the why. This file is the playbook and the guardrails. It overrides convenience.

## Golden rules (violating these has already cost hours or produced wrong science)

1. **One GPU job at a time.** Before launching anything GPU-heavy run `nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader` and check
   `ps -eo pid,etime,args | grep -E "[b]oltz predict|[p]rotenix pred|[r]un_funnel|[p]ipeline"`. If something runs, wait. Overlap caused `cusolver`/OOM failures.
2. **Never enter hotspot numbers as raw shard indices.** Define them as `"Y56"` (PDB number + residue letter) in `funnel/targets/<name>.json`; `common.load_target` converts and
   asserts identity. A wrong list runs silently and produces plausible-looking wrong results.
3. **A pass is not a binder.** Report "consensus pass" (Boltz-2 and Protenix-v2 agree) and say it is computational. Never write "binder", "hit" or "validated" without wet-lab data.
4. **Compare only within one judge run** (same batch, fresh seeds). Never compare `bz_*`/`b_*` scores across runs, targets or constructs.
5. **Success is verified, not assumed.** `State=COMPLETED`, exit code 0, or a non-empty `summary.csv` are not success. Check that rows have `bz_status == ok`, that counts match
   what you requested, and read the log tail.
6. **Exit codes:** write `cmd > log 2>&1` then `RC=$?` on the next line. `cmd; echo done` reports echo's status.
7. **Process control (this has bitten twice):** never `pkill -f`/`pgrep -f` with a pattern that appears in your own command line; take PIDs from `ps` and use `kill -0 PID` to wait. Use the bracket trick
   `grep "[b]oltz"` when filtering `ps`.
8. **Do not modify vendored code** (`pxdesign/`, `pxdbench/`, `colabdesign/`) except for a reviewed, tested fix with a test that fails without it (see `tests/test_legacy_hotspot_guard.py`).
9. **Ask before** installing external code or packages, downloading large data, deleting/moving user data, committing, or any outward-facing action. If a permission is denied, stop and
   explain; do not route around it. Install external tools to `.pxd/ext/` or with `--no-deps`; never let pip change numpy/torch in `.pxd/envs/*`.
10. **When wet-lab results arrive** (see `docs/RECOMMENDATIONS.md` §14): do not tune thresholds on fewer than ~20 results; report binder rate per tier with intervals; keep a record of what was ordered and why.
11. **State what a measurement does not show** (n, interval, selection bias, circularity) every time you report a number.

## Resume, extend, and long runs

- Every funnel stage is resumable. After ANY interruption (crash, Ctrl-C, reboot) re-run the exact same command; never delete `out/<run>` to "start clean" - run `run_funnel.py --out <run> --status` first.
- To grow a run, raise `--n-backbones`, `--rounds` or `--final-m` on the same `--out`. Guarded settings (target, hotspots, chunk size, MPNN settings, `--steps`, `--seqs`) cannot change on a resume; use a new `--out`.
- Long unattended work goes through `funnel/run_overnight.sh`-style queues: one step per marker file, a failed step must not block the next, logs per step. Never start a second GPU job while one runs (unless using `--gpus`).
- Several GPUs: `--gpus 0,1,2,3` (or `$FUNNEL_GPUS`); do not oversubscribe a GPU with unrelated jobs.
- Before publishing anything: `python scripts/export_public.py --dest <new dir> --run-tests` builds a clean tree and runs the scanner; it must report 0 findings. Never push this working tree or its git history.
- Another route can be added as a CSV of sequences (`--designs-csv`); see docs/EXTENDING.md. Report a new route only with the judge and controls (`funnel/judge.py`, `funnel/controls.py`).

## Multi-chain (homo-oligomer) targets

`funnel/` is single-chain; `funnel/common.py` raises on complexes with more than two chains. Use `phbind/` (guide: `docs/HOMO_OLIGOMER_TARGETS.md`): generate on a dimer shard,
score on the intact oligomer, group all target copies as one in ipSAE, **gate on the MIN direction unless your own reference set says otherwise** (`phbind/convention_check.py`),
screen with stratified batches that carry reference designs, and gate on counted output files (Boltz-2 exits 0 on a skipped input). Never use plain ProteinMPNN weights.
**Agents: start at `docs/AGENT_PLAYBOOK_PHBIND.md`; `python phbind/run.py status` shows the state, `python phbind/run.py scorers` the validated scorers; a scorer is changed only through `phbind/validate.py` and logged with `phbind/experiments.py`.**

## Environment facts

- Interpreters: `.pxd/envs/pxd/bin/python` (diffusion, MPNN, Protenix, funnel), `.pxd/envs/boltz/bin/boltz` (Boltz-2, subprocess). Always `export PYTHONPATH=<repo>`.
- Protenix model names: use **`protenix-v2`** (v2) and **`protenix_mini_default_v0.5.0`** (fast). The files named `…base_default_v0.5.0.pt` are v2 or v1.0.0 under a wrong name; check sha1.
- Harmless noise: a failed fused-LayerNorm compile (`List_inl.h … typename`) on every Protenix import (~80 s), and `ld: cannot find -laio` in pytest. A stale
  `protenix/model/layer_norm/lock` hangs the import only if no `nvcc` is running: delete it then.
- Tests (CPU): `PYTHONPATH=$(pwd) .pxd/envs/pxd/bin/python -m pytest tests -q -p no:cacheprovider` (691 pass, 21 skipped). Run before and after any code change.
- Verify CUDA with a real matmul, not `torch.cuda.is_available()`.

## Playbook: design binders for a new target

1. **Understand the request.** Target, site (from structure, not convention), binder length, how many candidates needed. Ask the user if the site is undefined.
2. **Prepare** (CPU): `scripts/prepare_target.py` → shard + `provenance.json`; crop/verify the MSA so its query equals the shard sequence; write `funnel/targets/<name>.json`.
   Gate: `common.load_target('<name>')` succeeds and prints the expected `hotspot_idx`. Add a one-line check of the three hotspot residue identities to your report.
3. **Smoke test** on tiny settings before any long run:
   `run_funnel.py --target <name> --out out/funnel_smoke/<name> --n-backbones 8 --seqs 2 --cycle-k 4 --rounds 1 --n-new 2 --final-m 4 --top 3 --force` (~10 min). All stages must complete.
4. **Run** (one GPU job): `run_funnel.py --target <name> --out out/funnel/<name> --n-backbones 500 --rounds 0` (~0.7–1.2 h). Add `--rounds 3` only for hard targets.
   Poll quietly; report progress in one line per stage, not per poll.
5. **Judge** (fresh seeds): `funnel/judge.py --target <name> --out out/judge/<name> --arm funnel=out/funnel/<name>/final_nocycle.csv [--arm other=...]`.
   Add fastPISA columns to older tables with `funnel/add_pisa_to_runs.py`.
6. **Triage**: look at `consensus_pass`, `hotspot_frac`, `pisa_flags`, interface aromatics, sequence composition (Lys+Glu, aromatics), cluster count. Flags are advisory.
7. **Report** with the template below. Offer the panel to order (diverse, with a positive control); do not order or contact anyone.

## Verification gates (stop and investigate if any fails)

| gate | check |
|---|---|
| hotspots | `load_target` identity passed; `resolved_hotspots.json`/log shows the intended indices |
| generation | requested number of backbones produced (`gen/converted.json` n); hotspot residues contacted in the shortlist (`hotspot_frac`) |
| scoring | no `ok=False`/NaN rows beyond a few percent; Boltz and Protenix row counts equal the request |
| judge | each arm judged in the same run; `n` per arm matches the shortlist size |
| sanity | consensus passes are not 100% on a hard target without explanation (check for a leak: same seeds, same model in loop and judge) |

## Failure → action

| symptom | likely cause | action |
|---|---|---|
| `cusolverDnCreate` / CUDA OOM | concurrent GPU job | wait for the other job; rerun the failed stage (stages are resumable) |
| pipeline reports success but all `boltz_failed` | OOM or bad MSA | `run_campaign.sh` now exits 1; read `log.txt` for the first error |
| `hotspot residues not found` | PDB numbers used as shard indices | use `funnel/targets` definitions, not raw lists |
| `MSA query does not equal the structure sequence` | MSA built for another construct | crop columns of the MSA to the shard sequence |
| Protenix import hangs with no `nvcc` | stale `layer_norm/lock` | delete the lock, rerun |
| Boltz `MSA does not match input sequence, creating dummy` | dummied MSA inflates scores | fix the MSA; discard those rows |
| `protenix failed` with large inputs | chunk too big | `chunk=` in `common.protenix_fold` (default 1500) |

## Reporting template

```
Target: <name> (<structure, construct, hotspots with residue letters>)   Strategy: <command + parameters>   GPU time: <h>
Result: <k> of <n> judged designs pass both models (Boltz-2 3 seeds mean ipSAE ≥ 0.5 & PAE ≤ 2 Å; Protenix-v2 ipSAE ≥ 0.5); medians/minimums for each model.
Site: <fraction of hotspots contacted>   Quality flags: <pisa_flags counts>   Composition: Lys+Glu <x>%, interface aromatics <y>   Diversity: <clusters>
Limits: computational only; predictors agree moderately (ρ≈0.5); judge uses the same two models as selection; n=<…>.
Next: <what to order / what to run / what is uncertain>
```

## File map

`funnel/` workflow (run_funnel, judge, compare, common, pisa, mpnn_run, targets/) · `bench/` ProteinBase benchmark and generation experiments ·
`scripts/` setup, run_campaign, target prep · `docs/REPORT.md` findings · `docs/RECOMMENDATIONS.md` operating advice · `.archive/` earlier work (git-ignored, nothing deleted).
