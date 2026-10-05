# Overnight test plan (2026-10-05)

Run unattended with `funnel/run_overnight.sh` (single GPU; `GPUS=0,1,2,3 funnel/run_overnight.sh` on the 4-GPU workstation). One GPU job at a time on a single card.
**Every step is resumable**: re-running the script skips finished steps (`out/overnight/done_*`) and an interrupted step continues from its last finished unit
(`python funnel/run_funnel.py --out <run> --status` shows exactly where). Logs: `out/overnight.log` (queue) and `out/overnight/<step>.log` (per step).

## Question being tested

> Can this software produce binders that are *credibly excellent* for a target nobody tuned it on, here the **mannose-binding pocket of E. coli FimH**, and which
> design route is best: diffusion backbones + MPNN (the funnel), dock-a-known-scaffold-then-redesign-the-interface, or the funnel with an interface-aromatic bias?

FimH is a hard, informative test: a narrow, deep, polar pocket (lectin domain 1-158, PDB 3MCY with a bound mannoside), so a protein has to *insert* side chains or seal the entrance.
There is no wet-lab result; "excellent" can only mean computational evidence that is hard to fake, so every claim is paired with a control.

## Steps (in order; hours are estimates on one RTX 5090)

| # | step | what it does | GPU h | main output |
|---|---|---|---|---|
| 0 | wait | FimA fast-screen calibration (running) finishes | - | `out/calib/*/calib.csv` |
| 1 | `fimh_fetch` | build FimH from PDB 3MCY: shard, MSA (ColabFold server), ligand site probe | 0 | `data/targets/fimh/` |
| 2 | `fimh_funnel` | 500 backbones (5 resumable chunks), 4 MPNN seqs each, fast screen, 3 cycling rounds, Boltz-2 x3 + Protenix-v2 on 60, mannose-site occlusion | 1.3 | `out/funnel/fimh/final_design/` |
| 3 | `fimh_dock` | LightDock 8 public scaffolds (ubiquitin, protein G/A, FN3, DARPin, Top7, SH3, VHH) restrained to the pocket, 8 diverse poses each, **interface-only** MPNN, fast screen, interface-only cycling x3, consensus | 1.2 (+CPU docking ~1.6 h, already running) | `out/dock/fimh/funnel/final_design/` |
| 4 | `fimh_bias` | same 500 backbones as step 2, MPNN with the interface-only aromatic bias (scale 0.6), no cycling | 0.9 | `out/funnel_bias0.6/fimh/` |
| 5 | `fimh_judge` | all four arms re-folded together with **fresh seeds** (Boltz-2 x3 + Protenix-v2): the only fair comparison | 0.7 | `out/judge/fimh/arm_summary.csv` |
| 6 | `fimh_controls`, `fimh_ctrl_dock` | per shortlist design: fold against 3 unrelated **decoy** targets and 2 composition-preserving **shuffles** against FimH | 0.4 | `out/controls/*/controls_summary.json` |
| 7 | `fimh_rim_fetch`, `fimh_rim` | the same funnel with only 3 rim hotspots (Y48, N135, F142): are fewer, more accessible hotspots better? | 0.9 | `out/funnel/fimh_rim/` |
| 8 | `pdl1_bias`, `mdm2_bias` (+ judges) | does the aromatic bias help on the targets we already know (paired backbones)? | 3.1 | `out/judge_bias/*/arm_summary.csv` |

Total about 8.5 GPU hours single-GPU; roughly a third of that on 4 GPUs (chunks and screening units run in parallel; the three Boltz seeds run on separate GPUs).

## Pre-registered success criteria (decided before seeing results)

Step 2/3/4 shortlists are judged in step 5 (fresh seeds, both models). For **FimH**:

1. **Computational binder candidate (per arm):** >= 10 of the top 20 pass the consensus gate (Boltz-2 mean ipSAE >= 0.5 and interface PAE <= 2 A, Protenix-v2 ipSAE >= 0.5).
2. **Right site:** >= 5 of those occupy the mannose site, i.e. a binder heavy atom within 3.5 A of >= 1 of the 11 mannose-headgroup probe atoms (`site_near >= 1`) **and** >= 60% of the requested hotspots contacted.
3. **Not a free lunch (step 6):** for >= 70% of the shortlist the on-target ipSAE exceeds both the best decoy and the best shuffle by >= 0.3 (`margin >= 0.3`). Otherwise the pocket is "sticky" for the predictor and passes mean little.
4. **Interface quality:** no more than half of the shortlist carries a `pisa_flags` entry (advisory flags: thin interface, few H-bonds, no aromatic contact, polar interface, low shape complementarity).
5. A route is called **best** if it meets 1-4 and has the most designs satisfying 1+2 *per GPU-hour*; ties go to the cheaper route.

For the **bias arm** (step 4 and step 8): adopt interface-bias as default only if (a) interface aromatic contacts and shape complementarity rise, **and** (b) consensus passes do not fall by more than 2 of 20 (paired backbones).
For the **dock-redesign arm**: it is *not* expected to win by default. Early smoke tests (4 poses, 1 round) gave ipSAE ~ 0 for all 34 designs (ipTM 0.11; the predictors put the binder 21-34 A away from the docked position), i.e. they did not
reproduce the docked poses and did not recognise the redesigned scaffolds as binders. The full run decides whether more poses, more designs per pose and interface-only cycling change that. A null result is a valid result and must be reported as such.

## What the run does not test (so nobody over-reads it)

- No wet lab. Passing means two structure predictors agree; on 206 labelled designs such agreement had only moderate AUROC (0.67-0.75).
- The mannose-site probe is geometric (ligand atoms from one crystal structure); it does not prove competitive inhibition.
- One target, one seed per design in selection; judge seeds are fresh but the models are the same ones used in selection.
- The bias test on PD-L1/MDM2 uses only the *computational* judge; the 206-design benchmark is the only link to experiment.

## Morning checklist

1. `cat out/overnight.log` - which steps are ok / FAILED; failures keep their log in `out/overnight/<step>.log`.
2. `out/judge/fimh/arm_summary.csv` (arms side by side) and `out/funnel/fimh/final_design/README.md` (best FimH models, plots, statistics, PyMOL/ChimeraX scripts).
3. `out/controls/*/controls_summary.json` (margin_ge_0_3 over the shortlist), `out/dock/fimh/poses.csv` + dock funnel README.
4. Compare each criterion above and write the verdict into `docs/REPORT.md` (section "FimH mannose-site test").
5. To continue or extend any run: run the same command again (optionally with a larger `--n-backbones`, `--rounds` or `--final-m`).

## If something fails

| symptom | action |
|---|---|
| a step prints FAILED | read `out/overnight/<step>.log`; fix; re-run `funnel/run_overnight.sh` (finished steps are skipped) |
| GPU out-of-memory / `cusolver` errors | another GPU job is running; stop it and re-run (units that finished are kept) |
| `fimh_dock` complains about LightDock | see docs/RECOMMENDATIONS.md "Install LightDock" |
| docking is slow | it is CPU-bound (16 cores); `--cores`, `--glowworms` and `--steps` trade time for coverage |
