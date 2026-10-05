# Handoff (2026-10-05): state, open work, tests, next development

For the next agent (or person) picking this up on a new workstation. Read `AGENTS.md` first (operating rules), then this file. Everything below is stated with how well it is established;
**nothing has been wet-lab validated** and the owner expects lab feedback soon.

## 0. Where things are
- Public repo (pushed 2026-10-05, commit `f37d8f4`, AGPL-3.0): https://github.com/dzyla/binder-design , branch `main`. It is a **clean export** (no structures, MSAs, run outputs, private notes, original git history).
- The old workstation's working tree (`/mnt/scratch/pxdesign_local`) holds data that is **not** in the repo and will be lost unless copied: `out/` (all run outputs, FimH/PD-L1/MDM2 final_design folders, judge/controls tables, calibration CSVs),
  `data/targets/` (rebuildable with `funnel/fetch_target.py`), `.pxd/` (environments, rebuildable), `.archive/` (old docs/scripts), `checkpoints/` (weights, not redistributable).
  If the owner wants the numbers behind `docs/OVERNIGHT_RESULTS.md` reproducible without re-running, copy `out/judge`, `out/controls`, `out/funnel/*/final_design` and `out/calib`.
- **Development now happens in a clone of the GitHub repo.** Do NOT re-run `scripts/export_public.py` from a private tree to publish again (it creates an unrelated history). In the clone, run `python scripts/check_public.py` before every push.
- This handoff file was written after the push and has not been pushed; it is in the old working tree (`docs/HANDOFF.md`). The owner should commit it.

## 1. Bring-up on the new machine (do this first)
```bash
git clone https://github.com/dzyla/binder-design.git && cd binder-design
./scripts/setup.sh && ./scripts/setup_extras.sh          # envs, fastPISA, LightDock (own venv), fused LayerNorm (needs nvcc)
./scripts/fetch_checkpoints.sh                            # prints where weights come from; place pxdesign_v0.1.0.pt, protenix-v2, protenix_mini_default_v0.5.0 (see AGENTS.md: names lie, compare sha1)
PYTHONPATH=$(pwd) .pxd/envs/pxd/bin/python -m pytest tests -q   # expect ~623 pass, ~68 skip (data-dependent), 0 fail
for t in pdl1 mdm2 fima fimh; do python funnel/fetch_target.py funnel/targets/$t.json; done   # needs internet (RCSB, ColabFold MSA server)
```
Verify CUDA with a real matmul (sm_120 notes in `CLAUDE.md`), then a smoke run: `run_funnel.py --target mdm2 --out out/smoke --n-backbones 8 --chunk 4 --seqs 2 --rounds 1 --n-new 4 --final-m 4 --top 3` (~6 min on one 5090).
The 4-GPU workstation's cards may not be sm_120; if not, torch/protenix builds and the LayerNorm build may differ - re-check `setup.sh` pins (`TORCH_SPEC`).

## 2. What exists and how well it is established
| component | status | evidence |
|---|---|---|
| Funnel (`funnel/run_funnel.py`): backbones -> MPNN -> Protenix-0.5-mini screen -> optional cycling -> Boltz-2 x3 + Protenix-v2 consensus -> `final_design/` | works, resumable, tested | PD-L1/MDM2 20/20, FimA 20/20, FimH 20/20 passes (computational) |
| Resume/extend (chunks, rounds, `--final-m`), config guard, `--status` | tested by kill/resume/extend | `tests/test_funnel.py`, docs/RECOMMENDATIONS.md s16 |
| Multi-GPU (`--gpus`) | **logic tested only with `--gpus 0,0` on one card**; never run on real multiple GPUs | s4 below |
| Fast-screen calibration | measured on 3 targets x 150 random designs | docs/REPORT.md s10: AUROC 0.75-0.87, top-20% holds 38-67% of passes |
| Controls (`controls.py`) | FimH cycled: 90% margin >= 0.3; dock arm 0% | docs/OVERNIGHT_RESULTS.md |
| Dock + interface-only redesign (`dock_redesign.py`) | mechanically works; **scientifically negative** (1/20 pass, 0% control margin, ipTM 0.11, binders 21-34 A from docked pose) | docs/RECOMMENDATIONS.md s17 |
| Interface aromatic bias (`--mpnn-bias iface:0.6`) | inconclusive: FimH site-occupancy 12 vs 9 (no controls), PD-L1 neutral, MDM2 small plus | OVERNIGHT_RESULTS |
| Surface site finder (`hotspots.py`) | known site in top 3 of 8 patches on 4/4 targets; fitted weights did not generalise (LOTO AUROC 0.56-0.62) | docs/HOTSPOTS.md |
| Blind-docking consensus epitope (`dock_epitope.py`) | **experimental, validation incomplete** (see s3.1) | PD-L1 known-site AUROC 0.91 vs surface 0.80; MDM2 0.26 with only 1 of 4 probes |
| Public export + scanner | works, 0 findings, tests pass in the clean tree | `scripts/export_public.py`, `scripts/check_public.py` |

## 3. Unfinished work, in priority order

### 3.1 Finish the epitope-consensus validation (was running when stopped, ~1 CPU-h so far)
`funnel/run_epitope_validation.sh` docks 4 probes on pdl1, mdm2, fimh, fima, and on the 4 benchmark receptors (egfr, il7r, mdm2_b, pd-l1) then runs `funnel/epitope_validate.py`.
State at stop: `out/epitope/pdl1` complete (4 probes), `out/epitope/mdm2` has only ubiquitin (but an `epitope.csv`), fimh just started, rest not started.
**Trap:** the script skips a target if `epitope.csv` exists, so delete `out/epitope/mdm2/epitope.csv` before re-running (per-probe docking is itself resumable). EGFR is ~620 residues: docking time grows with swarm count (PD-L1 212 aa: ~8-9 min/probe on 30 cores), so expect hours; consider `--range` on the EGFR extracellular domain or fewer swarms.
Decision to make from the result: does the dock consensus beat or complement the surface score at (a) known sites and (b) the labelled binders' epitopes (residue-level AUROC; combined rank-sum is already computed)? Update docs/HOTSPOTS.md either way. A negative result is fine; MDM2 (deep narrow pocket) already looks poor for rigid probes.

### 3.2 Fix the overnight evaluator and complete the controls (cheap, do before trusting any "best route")
- `funnel/evaluate_overnight.py` treats **missing controls as a pass** in the "best route" rule, so it printed "bias0.6" as best. Make missing controls mean "undetermined".
- Run `controls.py` on the **no-cycling** and **bias** FimH shortlists (`out/funnel/fimh/final_nocycle.csv`, `out/funnel_bias0.6/fimh/final_nocycle.csv`) and a paired **bias + cycling** arm. Without these the bias comparison is not valid (bias arm lacks cycling).
- Write the FimH verdict into `docs/REPORT.md` (the plan's morning checklist item 4 was never done); regenerate `docs/OVERNIGHT_RESULTS.md`.

### 3.3 Real multi-GPU run (needed before any speed claim)
`GPUS=0,1,2,3 funnel/run_overnight.sh` or `run_funnel.py ... --gpus 0,1,2,3` on a fresh target. Measure: wall time vs 1 GPU (docs claim "roughly a third" - **projected only**), no OOM/cusolver errors, results identical to single-GPU for the same seeds (designs are not split inside a Boltz batch by design; check `bz_final_batch_id`-style batch equivalence for Boltz and that Protenix-v2 on the last GPU doesn't collide with the Boltz seeds on `gpus[:-1]`). With < 2 GPUs `consensus` must fall back correctly.

## 4. What still needs implementing
1. **Protocol automation:** `docs/PROTOCOL.md` stage 3 (stop if >= 18/20 pass and site satisfied; else add cycling; < 8/20 -> diagnose) is prose; implement as `funnel/auto_protocol.py` or a `run_funnel.py --adaptive` flag that reads the consensus output and reruns with `--rounds 3`.
2. **Site scouting automation:** `hotspots.py --write-targets` + `fetch_target.py` + short runs + controls to rank candidate sites. All pieces exist; the loop has **never been run end to end**.
3. **Cyclic peptides** (owner asked earlier): Boltz-2 cyclic scoring, a peptide generator, cyclic-aware MPNN/judge. Plan in docs/RECOMMENDATIONS.md s15; nothing implemented.
4. **Better judge for docked / non-hallucinated poses:** co-folding models do not recognise redesigned scaffolds (null dock result). Needs a physics/energy term (Rosetta interface dG, MPNN complex score, or a template-conditioned fold with a shuffle control). Do not "fix" the null result by forcing the pose - the shuffle control would show the inflation.
5. **Additional generators through `--designs-csv`:** hallucination (BoltzDesign1 / Mosaic / BindCraft), BoltzGen, RFdiffusion. The seam works (`dock_redesign.py` is the example); none beyond docking has been wired or compared. Compare under `funnel/judge.py` + `controls.py`.
6. **Additional scores** (earlier discussion, none implemented): ESM-C/ESM3 pseudo-likelihood as a developability/plausibility term, global interface dG. Only add to ranking if they add information on the ProteinBase labels (`bench/`).
7. **Lab-feedback calibration:** a script to ingest wet-lab results (design id, binding yes/no, KD) and recompute the gate AUROCs / thresholds against `bench/`'s AUROC tables. Gate currently: Boltz ipSAE >= 0.5, interface PAE <= 2 A, v2 ipSAE >= 0.5; against labels it separates binders only moderately (AUROC 0.67-0.75).
8. **Housekeeping:** project name (`BinderFunnel` is a working name), licence confirmation (AGPL-3.0 chosen; `pxdbench/metrics/_common_scorer.py` was re-written independently to remove a licence problem - owner should confirm), CI (GitHub Actions running pytest + `check_public.py`), pin/verify dependency versions on the new GPUs, remove the ProteinBase CSV path assumption in `bench/` (download instructions in `bench/README.md`).

## 5. What needs testing (not yet tested or only weakly tested)
- **Backbone-count curve:** is 500 optimal? run 100/250/500/1000 on 2 targets, report passes per GPU-hour. Same for `--final-m` (60 -> 40/100) and `--seqs`.
- **Cycling beyond 3 rounds, and `--cycle-k`/`--n-new` sensitivity;** the adaptive stop rule's thresholds (18/20, 8/20) are judgement, not fitted.
- **Generalisation:** every positive result is on 4 targets (PD-L1, MDM2, FimA, FimH), all pockets/grooves/faces with decent MSAs. Test flat or glycosylated epitopes, targets with shallow MSAs, larger targets, binder lengths outside 55-90, and a target the funnel fails on (to learn failure signatures; the "< 8/20" branch is untested).
- **Fresh-seed independence:** judge seeds are 101-103 but the same two models are used for selection and judging; add a third predictor (AF2-multimer, Chai-1, OpenFold3) as an external check and see whether pass rates survive.
- **Site check for non-ligand targets:** `site_occlusion` only works with a ligand/probe set; hotspot-fraction is the fallback.
- **Resume under hard kills** (SIGKILL mid-write, disk full, node reboot) - atomic writes are tested for the normal path only; also resume after changing `--chunk` (guarded, refused) and after adding `--designs-csv` mid-run.
- **Unit-test gaps:** cycling state machine (`stage_cycle` archive of `cycle_old_*`), `consensus` shard logic with >1 GPU, `controls.py` margin computation, `dock_redesign.pick_poses`, `fetch_target` network paths (mock RCSB/ColabFold), `final_design.py` plots on degenerate input (0-1 passing designs).
- **The LightDock path on another machine:** build needs `numpy<2`, `cython<3.1`, `--no-build-isolation`, and relaxed GCC >= 14 pointer diagnostics (`setup_extras.sh` encodes this; untested outside this host).
- **Statistical caveat to keep testing:** a random MPNN design passes the two-model gate 10-40% of the time on these pockets. Always quote controls, never raw pass counts alone.

## 6. Further development ideas (lower priority, owner interested)
- Aromatic scarcity / Lys-Glu bias: traced to ProteinMPNN's prior, not the scorer; interface-bias is the first attempt; a MPNN variant trained with more hydrophobics or LigandMPNN could be compared.
- AF2-free hallucination / joint sequence-structure design (Mosaic, BoltzDesign1, BindCraft2) as additional generators; compare per GPU-hour.
- Shape complementarity (`funnel/sc.py`, flag 0.53) and fastPISA flags are report-only; promote to ranking only after they add AUROC on `bench/` labels (currently they did not).
- Per-target adaptive fast-screen keep-fraction from a 150-design calibration (`screen_calibration.py`, ~1 GPU-h).
- Antibody/VHH and small-molecule targets are unsupported.

## 7. Rules that have already cost time (details in `AGENTS.md`, `CLAUDE.md`)
- Hotspots are `"Y56"` names identity-checked against the PDB; shard numbering restarts at 1 - never pass raw shard indices (it silently ran wrong-site campaigns).
- One GPU job at a time per card (OOM / cusolver). Unattended work: one marker per step, a failed step must not block later ones (`funnel/run_overnight.sh` pattern).
- Never `pkill -f`/`pgrep -f` with a pattern in your own command line; take PIDs from `ps`.
- `State=COMPLETED` / exit 0 is not success; check rows (`bz_status == ok`) and non-empty outputs. Capture `$?` on the line after the command.
- Only compare scores inside one `judge.py` call (fresh seeds, same batch); ipSAE differences below ~0.03 are noise.
- A test whose fixture encodes your assumption proves nothing (the site-finder toy fixture was wrong three times); delete the guard and confirm the test fails.
- Never push the owner's private working tree or history; the owner commits/pushes with their identity unless they explicitly say to push.
