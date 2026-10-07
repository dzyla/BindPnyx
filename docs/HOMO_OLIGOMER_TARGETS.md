# Designing against homo-oligomer targets (`phbind/`)

`funnel/` assumes **one target chain + one binder**. A target whose binding site lies between protomers (a homotrimer such as TNF-alpha) breaks that
assumption silently: the binder gets read as a target chain and every contact/ipSAE number is target-vs-target. `funnel/common.py` now **raises** on a
complex with more than two chains. `phbind/` is the multi-chain path. It is a set of separately runnable stages (each reads the previous stage's files),
not a replacement for the funnel.

```
S0 s0_target.py     verify + register the target contract (numbering, MSAs, dimer, hotspots, coldspots)   needs $TNF_BUNDLE
S1 s1_generate.py   PXDesign backbones on a DIMER shard (cheaper; epitope-neutral if the groove lies between two protomers)
   s1_queue.py      resumable (hotspot set x length) queue -> SolubleMPNN -> out/phbind/designs_all.csv
S2 s2_prescreen.py  1-seed Boltz-2 on the INTACT trimer, stratified batches with reference carriers (resumable)
   peek.py          score what has finished NOW (even inside the running batch) -> out/phbind/candidates_screen.csv
S3 s3_gate.py       THE GATE: Boltz-2 x1 + AlphaFold3 x1 (af3_trimer.py). PASS = both models' grouped ipSAE (min direction) >= 0.5, tier A = both >= 0.65,
                    ORDER by the mean of the two (funnel/oracles.py). boltz_trimer.py = the Boltz-2 driver/scorer; protenix_trimer.py, of3_trimer.py = other second models (not the gate)
S4 s4_variants.py   histidine placements (M3 pair, M1 single) from a refolded structure + a post-refold AP1 census
S8 s8_assemble.py   dedupe, lineage cap, row order, format contract (raises on any violation)
   convention_check.py  which ipSAE direction your gate was calibrated on (see below)
```

## The contract (every item below was a silent failure somewhere)

- **Generate on the dimer, score on the whole trimer.** Every co-fold carries all copies, the same MSA on each copy, binder single-sequence, no templates;
  `group_indices` asserts the target residue count and finds the binder **by sequence**, never by chain letter or order.
- **Grouped ipSAE**: all target copies are one group. Global `iptm` averages the protomers against each other (0.96) and swamps the one pair that matters;
  a negative control reads ~0.86 global and ~0 grouped. The grouped implementation matches an independent reference implementation to 1e-10 (both directions).
- **The gate is calibrated on the MIN direction.** A published gate (seed-unanimous, mean >= 0.65, worst >= 0.5) said "use max", but its reference table is
  reproduced by min (Spearman 0.93, same pass count) and not by max (bias +0.20; 14/20 vs 5/20 pass). `phbind/convention_check.py` prints that evidence for
  your own reference set. Thresholds are only valid in the convention they were fitted in. Report both; gate on the calibrated one.
- **The gate is two models, one seed each, never seeds of one model.** Seeds of a single model are highly correlated (Spearman 0.77-0.93 between seeds; best/median/worst all track the mean at 0.95-0.99), so they re-measure the same thing; a second architecture is independent information (docs/JUDGE_REGIME.md: Boltz-2 x1 + a second model beat Boltz-2 x3 at lower cost). Consensus is across the two models, not the worst of several seeds.
- **Validate a second model on YOUR target before trusting it.** Measured here (20 reference designs whose labels are themselves Boltz-derived, so they favour Boltz-like models; 5 experimentally solved binders as controls): AlphaFold3 (5-sample mean) agrees with Boltz-2 at Spearman 0.25 and recovers 1/5 reference gate-passers at 0/15 false positives with both >= 0.5; Protenix-v2 0.22 and 3/5 at 0/15; OpenFold3 -0.16 and 0/5 (it does use the MSA, bare-trimer pTM 0.91 vs 0.24 without, but gave exactly 0.0 interface confidence to all five real binders and to the reference designs, which makes it unusable here). No model recovers most of the five real binders. A benchmark average does not transfer to a hard homotrimer target; positive controls and a labelled set do.
- **Gate on min, order by mean.** Independent finding in this repo's single-chain benchmark (`docs/JUDGE_REGIME.md`): with the same two models, `min` is right for pass/fail and
  poor for ranking, `mean` ranks better; the pair is fixed, not chosen per target. `phbind/s3_gate.py` applies exactly that on the trimer, reusing `funnel/oracles.consensus_score`.
  The same document finds the ipSAE direction hardly matters for *ranking* at that sample size and keeps gates on `ipsae_min`: consistent with the calibration check above.
- **Gate on counted artifacts, not exit codes.** Boltz-2 exits 0 and writes nothing when one input fails to parse (a single stray NUL byte at the end of an
  a3m did exactly that). `run_seed` raises if any design lacks its confidence+PAE files. `s0_target.py` asserts no NUL in any registered MSA.
- **A residue conflict between structure and construct is declared, not ignored.** `prepare_target.py` refuses it; reconcile with
  `--edit 'B:143:LEU>ASP:truncate_to_cb'` and crop the MSA to the observed residues (`msa_crop.py`; the repo refuses an MSA that would be silently discarded).
- **Scores are comparable only inside one batch** (`--seed` is global). Every S2 batch carries the same reference designs (2 known passers, 1 known failure,
  `$PHBIND_CARRIERS`); check their *order*, not just offset.
- **Stratify batches.** Sorting by run name put one hotspot set in batch 0, so an "early look" described one stratum. `s2_prescreen.py` shuffles (seeded) inside
  each sequence-index tier so every batch is a representative sample.
- **One-seed screening is for economy only.** Cut at 0.35 (min direction), then 5 seeds + a second oracle on survivors. 0.45 looked safe on 7 positives, but the
  95% interval on 7/7 is [0.59, 1.0]; every new backbone is a new lineage, where a hard cut is the expensive error.
- **Never rank on generator metrics or hotspot contact**; they were uncorrelated with an independent oracle in every measurement here. Keep hotspot/coldspot
  fractions and footprint recall as **descriptors** (a design can be confident and off-site).
- **Mouse (or any second species) is scored per finalist and per variant, never inherited from the parent or inferred from footprint recall.**
- **pH variants:** at most two histidines one-shot (3 His broke the fold in most scaffolds tested); recompute the anti-pattern (His against a carboxylate) census
  on the *refolded child*, against target **and** binder carboxylates; score each variant minus its own parent, in the same batch.
- **Never use plain ProteinMPNN weights** for binder sequences; `mpnn_dimer.py` hard-codes SolubleMPNN and cysteine is banned.

## Throughput measured (one RTX 5090, 32 GB; one GPU job at a time)

| step | cost |
|---|---|
| PXDesign, 304-residue dimer target, 62-112 aa binder | ~2.7 s / backbone (+ ~40 s start-up per run) |
| SolubleMPNN, 2 sequences per backbone | ~15 s / 100 backbones |
| Boltz-2, intact trimer + 60-110 aa binder (~560 tokens), 3 recycles, 200 steps | ~17-18 s / design / seed |
| AlphaFold3 (input_dir batch, MSA injected, 5 diffusion samples = same cost as 1) | ~36 s / design (a per-design process costs ~60-75 s: it reloads the weights) |
| Protenix-v2 / OpenFold3 on the same complex | ~14 s / ~14-16 s per design |
| the same with 1 recycle and 100 steps | ~10 s / design, **rejected**: only 1.8x faster, and a validated reference design fell from 0.78 to 0.00 |

A 72-design batch takes ~22 min. Use `peek.py` for an unbiased early look at any time.

## What this does not show

Nothing here has been tested at the bench. A gate pass means two structure predictors agree on an interface, not that the protein binds; on the public
labelled benchmark the predictors separate binders from non-binders only modestly. Counts of "designs" overstate diversity (2-3 sequences per backbone); report
backbones. The 1-seed prescreen is a screening statistic and is never a design's score.
