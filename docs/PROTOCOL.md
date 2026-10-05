# Recommended protocol: best binders per GPU-hour

Derived from the measurements in `docs/REPORT.md` and `docs/OVERNIGHT_RESULTS.md` (RTX 5090, 70-85 aa binders). Computational evidence only; nothing here is wet-lab validated.
Costs below are measured; items marked *untested* are judgement, not data.

## What the data say (the basis for every choice)

| finding | consequence |
|---|---|
| Correct hotspots were the largest single factor (stock pipeline read shard numbering silently) | define hotspots as `"Y56"` names; never skip the identity check |
| 8 backbones is far too few; 100-500 backbones with a cheap screen gives 16-20/20 passes | generate wide, filter cheap |
| Fast screen (Protenix-0.5-mini, 0.4-0.7 s/design) is a 3-5x enricher: top 20% holds 38-67% of passes, top 50% holds 73-90% | keep the top ~30-50% for the expensive stage, not 5% |
| Boltz-2 x3 seeds + Protenix-v2 consensus (both must agree) | the only judge used for selection; single-model passes are cheap to get (random MPNN designs pass 10-40% on these pockets) |
| Cycling (redesign on predicted complex, refold, keep best) cost +0.4-0.6 GPU-h: FimA 16->20/20, FimH 13->20/20, no gain where already 20/20 (MDM2, PD-L1) | make cycling conditional, run it second |
| Docking known scaffolds + interface redesign: 1/20 pass, 0% control margin | do not use as a primary route |
| Interface aromatic bias: neutral on PD-L1, small plus on MDM2, FimH site-occupancy 12 vs 9 (unpaired controls) | optional; not default until controls and a paired cycling comparison exist |
| Rim-only hotspots raise scores but moved binders off the pocket (11 vs 16 site-occupied) | choose hotspots for the function you want, not for easy scores |
| Passing = predictors agree; AUROC vs wet-lab labels only 0.67-0.75 | order a diverse panel, and calibrate with lab results |

## Protocol

**Stage 0 - target (10 min, CPU).** Write `funnel/targets/<t>.json` (PDB id, chain, range, 3-6 hotspots on one face, binder length 65-85; add `ligand` to get a site-occlusion test if you want to block a pocket).
`python funnel/fetch_target.py funnel/targets/<t>.json`. Check the logged hotspot residues are the ones you meant.

**Stage 1 - smoke test (~10 min).** `run_funnel.py ... --n-backbones 8 --chunk 4 --rounds 0` in a scratch `--out`. Confirms environment, MSA, hotspot mapping.

**Stage 2 - wide pass, no cycling (~0.7-1.2 GPU-h).**
```
run_funnel.py --target T --out out/T --n-backbones 500 --chunk 100 --seqs 4 --rounds 0 --final-m 60 --top 20 [--gpus 0,1,2,3]
```
500 backbones x 4 MPNN sequences = 2,000 designs, fast-screened, top 60 get Boltz-2 x3 + v2.
Gate: count consensus passes in the top 20.

**Stage 3 - decision rule (free, reads Stage 2 output).**
- >= 18/20 pass and the site check (hotspot fraction, ligand occlusion if defined) is satisfied -> **stop**; cycling would only add cost. (MDM2, PD-L1.)
- 8-17/20 pass, or fewer than 5 occupy the intended site -> **run Stage 4** (FimA, FimH).
- < 8/20 pass -> the target or hotspots are the problem; do not spend more GPU: re-examine hotspot choice, binder length, MSA, or enlarge to 1,500+ backbones (*untested*).

**Stage 4 - cycling on the same run (~+0.4-0.6 GPU-h).** Re-run the same command with `--rounds 3`. Stages 1-3 are reused; only cycling and the consensus on its candidates run.
Stop at 3 rounds: round-by-round gains were front-loaded (rounds beyond 3 *untested*).

**Stage 5 - controls on the shortlist (~0.1-0.4 GPU-h).**
`controls.py --target T --designs out/T/final_cycled.csv --decoys <3 unrelated targets> --shuffles 2`.
Keep designs whose on-target ipSAE beats the best decoy and best shuffle by >= 0.3 (FimH cycled: 90%). Drop the rest. This is what separates a real result from a sticky pocket.

**Stage 6 - independent re-judge (optional, ~0.2-0.7 GPU-h).** `judge.py` with fresh seeds, when comparing settings or before committing to a purchase list. Not needed for routine runs.

**Stage 7 - order panel.** From `final_design/`: take the tier-1 designs that survived Stage 5, drop `pisa_flags` / shape-complementarity < 0.53 flags unless the rest of the design is unusually strong, cluster-dedupe, then order ~8-24 sequences spanning different backbones/clusters. Include a few lower-ranked designs: they are what let you calibrate the gate against lab data.
Check composition by eye (Lys/Glu content, aromatics at the interface) for expression risk.

## Cost summary (single GPU)

| target type | path | GPU-h | outcome |
|---|---|---|---|
| easy (MDM2, PD-L1) | Stages 0-3, stop | 0.7-1.2 | 20/20 |
| hard (FimA, FimH) | Stages 0-5 | 1.0-1.7 + controls | 20/20 (FimH 14 in-site, 90% control margin) |
| 4 GPUs | same, `--gpus 0,1,2,3` | roughly a third (*projected, not yet measured*) | same |

Cheapest big lever not yet measured: lower `--final-m` (60 -> 40) when the fast screen is well calibrated for the target; use `funnel/screen_calibration.py` (150 random designs, ~1 GPU-h once) to decide per target.

## Do not

- Do not run cycling by default, or more than 3 rounds without evidence it helps.
- Do not use docking-plus-redesign as the main route; do not trust any pass without Stage 5 controls.
- Do not compare scores across runs; compare only in one `judge.py` call.
- Do not run two GPU jobs at once on one card (OOM/cusolver); use `--gpus` for parallelism.
- Do not change guarded settings on a resume; use a new `--out`.

## Open items that could still improve this

1. Controls for the no-cycle and bias arms, and a paired cycling-vs-bias run, before the bias becomes default.
2. Real 4-GPU timing.
3. Backbone count vs yield curve (is 500 optimal, or would 250 do?) and `--final-m` sensitivity: only 3 targets, 500 backbones tested.
4. Lab feedback to recalibrate the gate (AUROC 0.67-0.75 is the ceiling until then).
