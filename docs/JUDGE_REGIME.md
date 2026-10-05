# Judge regimes: how the expensive consensus ranks designs

The funnel's last stage folds each candidate with Boltz-2 and a second co-folding model and keeps designs on which both agree (`consensus_pass`). Two things about
*how the two scores are combined and ranked* were measured on the only public set of designs with wet-lab labels, the Anthropic binder-design release
(1,320 labelled designs, 354 binders, 15 targets; CC BY 4.0). This page records what was found, what is selectable now, and what is **not** established.

## What the labelled data says

All numbers: per-design mean over seeds of the funnel's `ipsae_min`; within-target AUROC is the sample-weighted mean of per-target AUROCs, so a hard target does not
masquerade as a good model; differences are paired bootstraps over targets (95% interval). `tests/test_consensus_regime.py` pins every number below.

| scorer | precision@10 | within-target AUROC |
|---|---|---|
| Boltz-2 alone | 0.387 | 0.689 |
| min(Boltz-2, Protenix-v2)  (the funnel as first published) | 0.367 | 0.683 |
| mean(Boltz-2, Protenix-v2) | 0.420 | 0.724 |
| min(Boltz-2, OpenFold3) | 0.513 | 0.719 |
| mean(Boltz-2, OpenFold3) | 0.527 | 0.744 |

1. **The `min` rule discards the second model's information when it is used to rank.** Boltz-2 x Protenix-v2 with `min` is not better than Boltz-2 alone (AUROC -0.006,
   interval -0.038..+0.026). The same two models with `mean` are (+0.035, +0.010..+0.063). `mean` beats `min` for the same pair of models: +0.025 (+0.011..+0.041) for
   Boltz-2 + OpenFold3. Keep `min` for the pass/fail gate; order candidates by `mean`.
2. **A second, different model helps; which one is not settled.** Under `mean`, OpenFold3 vs Protenix-v2 as the second judge differs by +0.020 (-0.015..+0.054): not distinguishable.
3. **ipSAE direction does not matter at this sample size.** `ipsae_max` vs `ipsae_min`: +0.004..+0.008 AUROC for the consensus, intervals include 0. Gates (0.5) stay on `ipsae_min`.
4. In the same data, re-choosing the model pair per target was worse than a fixed pair, and adding sequence-composition or provenance features to the consensus lowered
   leave-one-target-out precision. The regime therefore uses a fixed pair and no learned scorer.

Limits: 15 targets; precision@10 is noisy (one design moves it by 0.007 per target); selection used the same release for choosing and for reporting, so treat the effect sizes
as optimistic; the labels are binding in one assay, not function. The `af3of3` column of the release is AlphaFold3 *code* with OpenFold3 *weights*; it says nothing about
AlphaFold3's own weights.

## Selecting a regime

| option | legacy (default) | new |
|---|---|---|
| `--judge` | `legacy` | `new` |
| Boltz-2 seeds (`--boltz-seeds`) | 1,2,3 | 1 |
| second model (`--second-oracle`) | `protenix-v2` | `af3` |
| shortlist order (`--rank-rule`) | `min` | `mean` |
| second-model gate (`--o2-gate`) | 0.5 | 0.5, **not calibrated for AlphaFold3** |

`legacy` reproduces earlier runs. Mixing is refused: the judge is a guarded setting, and a run folder written before this option existed counts as `legacy`.
`--rank-rule` alone can be changed on a finished run: predictions are reused and only the shortlist is re-ordered.

Why one Boltz-2 seed plus one AlphaFold3 seed: on the release, Boltz-2 with 1/3/5 seeds scored AUROC 0.689/0.698/0.703 (precision@10 .454/.460/.460), while
Boltz-2 x1 + OpenFold3 x1 (cost 2) beat Boltz-2 x3 (cost 3) by +0.076 precision@10 (+0.022..+0.137; random seed subsets, 40 draws). That test used `ipsae_max` with a mean of
within-table z-scores, unlike the `ipsae_min` table above, and OpenFold3 weights, not AlphaFold3's.

## AlphaFold3 backend

No path is hard-coded. Set
`PXD_AF3_PYTHON` (python of an environment that runs AlphaFold3 with CUDA), `PXD_AF3_DIR` (folder with `run_alphafold.py`), `PXD_AF3_MODELS` (parameters folder; obtain the
weights from their publisher under its terms). The target MSA is the one the Boltz-2 / Protenix path already uses, sanitised once (records that are truncated, contain NUL bytes or
have the wrong length are dropped); the binder gets a single-sequence MSA; AlphaFold3 runs with `--norun_data_pipeline`. ipSAE is the mean over the diffusion samples of one seed.
Each design is its own process with `CUDA_VISIBLE_DEVICES` pinned to one card, JAX pre-allocation off and a per-card compilation cache.

With `--gpus 1,0,2,3` all cards pull designs from one queue, so a faster card takes proportionally more (measured on one RTX 6000 + three RTX 4000: 141 s vs ~280 s per design, so ~2:1:1:1)
without hand-tuned weights; the card Boltz-2 is using joins when Boltz-2 finishes.

## Comparing regimes fairly: generate once, judge twice

```
python funnel/run_funnel.py --target T --out out/T_legacy --n-backbones 500 --rounds 0                       # legacy judge
python funnel/run_funnel.py --target T --out out/T_new --fork-from out/T_legacy --judge new --rounds 0        # same candidates, new judge
```
`--fork-from` hard-links generation, design, fast-screen and cycling output and the Boltz-2 seed-1 predictions, and drops the judging timers, so each arm reports its own cost
(`timers.json`: `5_boltz_*`, `5_v2_*` / `5_af3_*`, and `5_af3_gpu_s_*` = summed per-design GPU seconds). Decide the success criteria **before** reading either result, and score both
arms with a model that neither used (agreement with an independent predictor is not binding evidence).

## What is not established

* No wet-lab data exist for designs chosen under `new`; every number above is an in-silico benchmark against previous wet-lab labels.
* AlphaFold3 (real weights) has not been calibrated on the release; the default gate of 0.5 is a placeholder until it has been checked on labelled designs.
* The mean-over-min result is an exploratory comparison on one benchmark, not a pre-registered one.
