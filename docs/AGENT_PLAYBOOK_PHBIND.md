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
| structural novelty | `prescreen.novelty_file` | none (another session screens) | **Novelty is the filter that mattered: 18 of 21 two-model gate passers (62-112 aa) had a PDB structure matching at TM >= 0.8 over >= 70% of the chain, the portal's own failure rule, and every dual-species binder was among them.** Run it BEFORE any oracle (`phbind/handoff.py` packs structures for whoever has the search database; `s2_prescreen` then scores novelty-passed backbones first, unscreened next, failed never) | search time per query, no GPU |
| binder length | `generate.lengths` | 62-112 (first campaign); 120-180 (second) | longer chains must match a larger fold over >= 70% of their length: 3/13 gate passers at >= 104 aa were novel vs 0/8 at <= 96 aa (a bet, not a result). Boltz-2 pass rate is not lower at 104-112 aa (7.8% vs 6.5%) | passers cluster 104-112 and 62-86; 110-130 aa Ig-like is a novelty trap | per length 100 backbones |
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
0. **Put novelty first** (see the knob table) and learn which (hotspot set, length) cells produce backbones with zero strict PDB hits. A geometric predictor of the oracle does not exist here (gradient-boosted model on backbone features: AUROC 0.60), so novelty is the only cheap prefilter found.
1. **Mouse scoring in this driver** (species switch: 468-residue mouse trimer, mouse MSA; score per finalist and per variant; never inherited).
2. ~~Validate `ptx_fast`~~ done: it fails on the trimer (see `validation.json`); a cheaper prescreen needs a different model or a different readout.
3. **Structure clustering of passers** (TM-align on binder chains) so lineages are counted by fold, not sequence.
4. **pH variants** on the best gated designs, each re-gated on both models and mouse, anti-pattern census on the refolded child.
5. **Calibrate the gate** on labelled designs (the public release) with the pair fixed.
6. **Close the loop with wet-lab results**: record what was ordered and why; report binder rate per tier.
7. A scorer that finds real binders. Boltz-2 recovers 1 of 4 solved TNF binders here and every other model tried recovers none; anything that moves this number is worth more than any tuning above.

## 8. More throughput: waves on other cards or machines
One card screens ~150 designs/hour. A second card or machine runs a **wave** (`phbind/wave.py`, `docs/WAVES_AND_SECOND_MACHINES.md`): same pipeline, its own campaign tag, `seed_base` and hotspot-set aliases so it never regenerates another wave's backbones, results returned as small CSVs. `python phbind/wave.py --plan` shows what is done (decided from real artifacts) before any GPU time is spent.

## 9. The shared submission ranking
Several agents can maintain ONE ranking without overwriting each other: an append-only evidence ledger plus a rule-based ranker (`phbind/ranking.py`, usage in `docs/SHARED_RANKING.md`). Agents record measurements with `rank.py add` (design, metric, value, seeds, source note); tiers, merit and the suggested submission row order are recomputed by rule. Never hand-edit the output.

## 10. Where the evidence lives
`docs/HOMO_OLIGOMER_TARGETS.md` (the contract and measurements), `docs/JUDGE_REGIME.md` (benchmark findings on a second model), `docs/REPORT.md` (single-chain benchmark, fast-screen
calibration), `phbind/validation.json` (per-scorer records), `out/phbind/experiments.jsonl` (what has been tried).


## 10. The pH arm and the confirmation standard (added after the first full round)

**Order of work for a pH objective.** (1) Triage parents by their pH baseline (`phbind/ph_score.py`: `baseline_ok`; a pose scores in ~2 s, so score the existing prescreen pose of every Boltz-passing design first). Parents far below zero cannot be rescued by a 1-2 histidine edit, and in our pool most were far below. (2) Variants on the survivors only (`s4_variants.py`: M3 pairs, M1 singles, <= 2 His in the list that is submitted). (3) Refold all variants in ONE batch with the parent and the carriers, filter: binding retained, <= 2 His, no His-to-carboxylate contact in ANY refolded pose. (4) AF3 and the other species on the survivors; nothing is inherited from the parent. (5) Paired multi-pose pH (`ph_score.summarise`), 5 poses minimum, plus a second independent run for anything that will be called a switch.

**Boltz-2 is deterministic for a given (sequences, seed, batch composition and order, command line, GPU) and changes when ANY of them changes.** Measured here on the same four sequences at seed 101: the same batch run twice gave byte-identical output; adding `--num_workers 0` changed every prediction (reproducibly); putting the same sequences in a 22-design batch changed the scores by 0.02-0.09 (and a failing control by 0.18). A first version of this paragraph blamed "nondeterminism" and a later one blamed the flag alone; both came from comparisons that changed two things at once. Consequences: compare designs only inside ONE batch; the exact command is saved in `boltz.cmd` beside every run and must be quoted with the numbers; a "replicate" in a different batch is a legitimate independent draw (spread about 0.1 on the mean, more on the worst seed), an identical rerun tells you nothing; prefer designs with margin over designs at the bar.

**pH numbers carry about +/-0.26 kcal/mol per run** (`validation_ph.json`). Require every pose positive for any claim of a switch (proposed rule, not yet shared), report the worst pose next to the mean, and say that the number is a protonation model on predicted poses.

**Standard of evidence for a final-list row:** >= 5 seeds on every species the objectives name; two independent runs for the headline designs; two different models; a failing carrier and a shuffle control in the batch; novelty screened on the design's own refolded backbone (point mutants barely move it, but screen, do not inherit).
