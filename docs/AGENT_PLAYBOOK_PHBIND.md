# Agent playbook: running and improving the trimer binder-design pipeline (`phbind/`)

For any AI agent (or person) operating this repo on a homo-oligomer target. Read it before touching a setting. It tells you what the machine can do, which knobs exist, what each
one is known to do, how to test a change so you do not fool yourself, and what has already failed. It does **not** tell you the designs bind: nothing here has been tested at the
bench. "Better" below means *measurably better by the criteria in section 5*, and every claim in the evidence column was measured on this project's data.

## 0. Start here
```bash
export PYTHONPATH=$(pwd)
python phbind/run.py status           # counted artifacts (backbones, designs scored, survivors, gate passers), GPU jobs, what to do next
python phbind/run.py scorers          # every scorer, its measured validation record, and the roles it may play
python phbind/run.py config           # the effective config (defaults + $PHBIND_CONFIG overrides), validated
python phbind/run.py require of3 gate # exits 2 with the measured reason when a scorer is not allowed for a role
```
One GPU job at a time (`status` lists what is running). Everything is resumable; re-run the same command after any interruption. Gate on counted output files, never on an exit
code: Boltz-2, BindCraft2, PXDesign and `tnf_ph_score.py` have all exited 0 while producing nothing.

## 1. The pipeline and its files
| stage | script | reads | writes | cost (RTX 5090) |
|---|---|---|---|---|
| S0 target contract | `s0_target.py` ($TNF_BUNDLE) | organiser structures, MSAs | `phbind/target/` (never committed) | seconds |
| S1 generate | `s1_queue.py` -> `s1_generate.py`, `mpnn_dimer.py` | target shard (`prepare_target.py`) | `out/phbind/gen/*`, `designs_all.csv` | 2.7 s/backbone |
| S2 prescreen | `s2_prescreen.py`, `peek.py` | `designs_all.csv` | `out/phbind/s2/batch_*.csv` | 18 s/design (1 Boltz-2 seed) |
| S3 gate | `s3_gate.py` | survivors (Boltz-2 >= 0.5) | `gate.csv` | +36 s/design (AF3) |
| S4 pH variants | `s4_variants.py` | gated designs' structures | variant sequences | CPU, then re-score |
| S8 assemble | `s8_assemble.py` | candidate table | submission CSV, raises on any contract violation | CPU |

## 2. The gate, stated once
**Two different models, one seed each, consensus across models:** Boltz-2 x1 + AlphaFold3 x1 (mean over 5 diffusion samples), grouped ipSAE (all target copies = one group), **min
direction**. Pass = both >= 0.5; tier A = both >= 0.65; order by the mean of the two. A model missing for a design -> NaN, never a one-model score. Seeds of one model are not a
gate: they correlate at Spearman 0.77-0.93, so they re-measure the same thing, while a second architecture is independent information. (`config.py` rejects a gate of two
identical legs.) A pass means two predictors agree on an interface, not that the protein binds.

## 3. Knobs, what is known about each, and how to change it
| knob | where | now | evidence | cost of testing |
|---|---|---|---|---|
| hotspot string | `generate.hotspot_sets/sets` | declared-8, footprint-12, 6-subset | first-pass 1-seed pass rate favoured declared-8; at the two-model gate 10/35 vs 7/30 vs 4/26 (no significant difference). Re-aimed vs declared is a null in the sibling campaign too | 100 backbones + screening ~ 40 min |
| binder length | `generate.lengths` | 62-112 | passers cluster 104-112 and 62-86; 110-130 aa Ig-like is a novelty trap | per length 100 backbones |
| sequence model | `generate.mpnn` | SolubleMPNN only | plain ProteinMPNN gave poor binders (user decision, release data: 4.8% vs 27.9%); config refuses it | - |
| screen cut | `prescreen.cut` | 0.35 | 0.45 kept 5/5 reference passers but 7/7 has a 95% interval of [0.59, 1.0] | re-score a reference set |
| screen scorer | `prescreen.scorer` | boltz (1 seed) | only scorer with screen validation; the cheaper `ptx_fast` was TESTED and fails on the trimer (AUROC 0.56, 89% of scores exactly 0) | see section 4 |
| gate legs | `gate.legs` | boltz + af3 | Protenix-v2 is the validated alternative (3/5 reference passers, 0/15 false positives); OpenFold3 is refused (exactly 0.0 on every real binder) | `validate.py` ~10 min |
| gate thresholds | `gate.gate/strong` | 0.5 / 0.65 | fitted on the MIN direction; the max direction is biased +0.165 and would loosen it. AF3's 0.5 is uncalibrated | needs labelled designs |
| AF3 samples | `gate.af3_samples` | 5 | 5 samples cost the same as 1 (trunk dominates) and agree within a design | - |
| histidine count (pH) | `s4_variants.variants` | <= 2 one-shot | 1 His kept binding 6/6, 2 His 15/20, 3 His 2/6 (sibling campaign); re-check the anti-pattern census on the REFOLDED child | per variant: 1 prescreen + 1 gate |

## 4. How to test a change without fooling yourself (the experiment protocol)
1. **State the hypothesis and the criterion first** ("replacing the Boltz screen with `ptx_fast` keeps >= 80% of Boltz >= 0.5 designs in its top 30%, at <= 1/4 the cost").
2. **Test on the same designs, in one batch, with carriers.** Scores are comparable only inside one batch (`--seed` is global; batch composition moved unchanged designs by up to 0.09).
   Carry 2 known passers and 1 known failure into every batch and check their order, not just their offset.
3. **Use positive controls and a reference set**, never a benchmark average from other targets: `python phbind/validate.py <scorer> set.csv out --write` (columns id, seq, group =
   real_binder | reference_pass | reference_fail). The decision rule is written in `validate.py`; argue with it there, not in your head. A scorer that gives exactly 0.0 to every real
   binder cannot gate, however well it did on a benchmark.
4. **Report what the measurement does not show** (n, reference-label bias: labels from one instrument favour models like it; a miss on a real binder is uninformative).
5. **Log it**: `python phbind/run.py log "<name>" keep|revert|inconclusive '<change json>' '<measured result json>' "<hypothesis>"`. No measured result, no log entry.
6. **Promote only after 3**: write the config change, `validation.json` is updated by `validate.py --write`, and the guard (`scorers.require`) now enforces it for the next agent.

Adding a scorer: write `_name(df, out, **kw)` in `scorers.py` returning `id` + `<prefix>_ipsae_min` (+ `_max`), register it, run step 3, add a test. Grouping must go through
`trimer.group_indices` (binder found by sequence, 471 target residues asserted); never hand-slice a PAE matrix.

## 5. What "better" means (and what is not an improvement)
Optimise **cost per gate-passing, structurally distinct backbone**, then **positive-control recall** (still 1/4 at best), then **false-positive rate on the reference set**. When wet-lab
data exist: binder rate per tier with an interval, and do not tune thresholds on fewer than ~20 results.
Not improvements: a higher single-seed ipSAE; more seeds of one model; hotspot contact or footprint recall (descriptors only: rho -0.17 to +0.11 vs an independent oracle); the
generator's own confidence; global ipTM (reads 0.86 on a negative control that grouped ipSAE reads 0); the max direction with min-fitted thresholds; counting designs instead of backbones
(2-3 sequences per backbone inflate diversity ~3x).

## 6. Never (each cost real compute here)
- gate on an exit code or on "directory exists"; run two GPU jobs at once; compare scores across batches; use plain ProteinMPNN weights; bias histidines at generation (418 trajectories,
  0 refolds); optimise human and mouse jointly at generation (0/67 accepted); inherit mouse (or AF3, or pH) results from a parent design or from footprint recall; feed an MSA with a
  stray NUL byte (Boltz-2 skips it and exits 0); trust a scorer because it ranks well on other targets; present a ranking as confidence of binding; commit target files, MSAs, run
  outputs or reference designs (`scripts/check_public.py` must report 0 findings).

## 7. Open problems an agent can take (highest value first)
1. **Mouse scoring in this driver** (species switch: 468-residue mouse trimer, mouse MSA; score per finalist and per variant; never inherited).
2. ~~Validate `ptx_fast`~~ done: it fails on the trimer (see `validation.json`); a cheaper prescreen needs a different model or a different readout.
3. **Structure clustering of passers** (TM-align on binder chains) so lineages are counted by fold, not sequence.
4. **pH variants** on the best gated designs, each re-gated on both models and mouse, anti-pattern census on the refolded child.
5. **Calibrate the gate** on labelled designs (the public release) with the pair fixed.
6. **Close the loop with wet-lab results**: record what was ordered and why; report binder rate per tier.
7. A scorer that finds real binders. Boltz-2 recovers 1 of 4 solved TNF binders here and every other model tried recovers none; anything that moves this number is worth more than any tuning above.

## 8. Where the evidence lives
`docs/HOMO_OLIGOMER_TARGETS.md` (the contract and measurements), `docs/JUDGE_REGIME.md` (benchmark findings on a second model), `docs/REPORT.md` (single-chain benchmark, fast-screen
calibration), `phbind/validation.json` (per-scorer records), `out/phbind/experiments.jsonl` (what has been tried).
