# Running the design pipeline successfully: recommendations

Status: written 2026-10-04 from the evidence in `docs/REPORT.md` (scorer benchmark, generation experiments) and the funnel evaluation
(`funnel/`). Items marked **[not evaluated]** were never completed (FimA, the MPNN-bias experiment); no result exists and none is claimed.
For agents (AI assistants) operating the tools, see `AGENTS.md`.

## 1. The workflow to use

```
define target ─► build + verify shard ─► funnel (500 backbones, ~1.2 GPU-h) ─► judge ─► triage with fastPISA flags ─► order a diverse panel
```

Default command, after the target is defined (§3):

```bash
.pxd/envs/pxd/bin/python funnel/run_funnel.py --target <name> --out out/funnel/<name> --n-backbones 500 --rounds 0
```

`--rounds 0` = no cycling. On the evidence so far this is the best cost/quality point (§5); add `--rounds 3` for hard targets or when you can afford
~50% more GPU time. One GPU job at a time.

## 2. The five things that matter most (in order of measured effect)

1. **Condition on the right residues.** Hotspot numbers on a prebuilt shard are *the shard's own numbering (restarting at 1)*, not PDB numbering. Wrong
   indices silently ran in earlier campaigns and cut hit rates from ~100% to ~7–26%. Use `funnel/targets/*.json`, which takes `"Y56"`-style PDB numbers and
   **asserts residue identity** through the provenance map. Never type shard indices by hand.
2. **Generate at scale, screen cheaply.** The shipped default (8 backbones) is 12× below upstream's smallest preset. Diffusion costs ~1.5–2.5 s per backbone;
   scoring is what costs. 500 backbones × 4 sequences, Protenix-0.5-mini "fast" (0.4 s/design) as the screen.
3. **Use two independent judges.** Boltz-2 *and* Protenix-v2 together beat either (AUROC 0.75 vs 0.69–0.70 on 206 wet-lab-labelled designs); they agree
   only moderately (Spearman 0.5), so each adds information. Require both (the "consensus pass").
4. **Triage; do not over-trust the score.** Predictor confidence separates binders from non-binders only modestly (AUROC ≈ 0.65–0.75 anywhere in the
   benchmark; 0.93 on MDM2, ~0.56–0.84 elsewhere). A pass is "two predictors agree", not "binds".
5. **Order a diverse panel, not the top few**, with a known binder as positive control. Expect most to fail; the wet lab is the only calibration.

## 3. Target preparation checklist

- Pick the **site from structure**, not from a convention: contact residues with the natural partner, buried area, conservation (see
  `.archive/docs/measurements/2026-10-03-fima-groove-campaign.md` for a worked example where the inherited epitope was off-site).
- Build the shard with `scripts/prepare_target.py --source X.pdb --wt-fasta WT.fasta --wt-align sequential --chain A --msa A=<msa_dir> ...`.
  The MSA directory needs `non_pairing.a3m` whose **query equals the shard sequence** (checked); crop an existing MSA by columns, do not re-query blindly.
- Write `funnel/targets/<name>.json` (`shard`, `provenance`, `source_chain`, `msa_dir`, `hotspots`, `binder_length`). `load_target` fails loudly on a wrong
  letter/number or MSA mismatch. Run `python -c "import sys; sys.path.insert(0,'funnel'); import common; print(common.load_target('<name>')['hotspot_idx'])"`.
- **Hotspots:** 3–6 residues on **one face**. A binder of 60–85 residues covers ~20–28 Å; do not request two distant faces. Prefer deep, hydrophobic pockets
  (MDM2-like) where predictors are most informative; flat faces (PD-L1) are harder and the scores less trustworthy.
- **Binder length:** 60–85 for pockets/grooves; match published binders for the target where known.
- Legacy-mode pipeline manifests now reject hotspots beyond the shard (new guard), but cannot catch an in-range wrong index: only identity assertions do.

## 4. Running it

| item | recommendation |
|---|---|
| GPU | **One job at a time.** Concurrent Boltz/Protenix/JAX jobs gave `cusolver`/OOM failures. Check `nvidia-smi` first. JAX preallocates ~75% VRAM unless `XLA_PYTHON_CLIENT_PREALLOCATE=false` (already set in `funnel/`). |
| Cost, one RTX 5090 | 500 backbones, no cycling: **0.7 h (MDM2, 70 aa) / 1.2 h (PD-L1, 85 aa)**; with 3 cycling rounds 1.1 h / 1.8 h. Pipeline baseline at 100×4: 1.2 h / 1.6 h. |
| Resume | Every stage's output is reused; `--force` redoes. Safe to stop between stages. |
| Waiting | Poll with `kill -0 PID` or file existence. **Never `pkill -f`/`pgrep -f` a pattern in your own command line.** |
| Exit codes | `cmd > log 2>&1; echo` reports echo's status. Capture `$?` on the next line. `State=COMPLETED` / non-empty `summary.csv` are not success: check `bz_status == ok` (the campaign script now does). |
| Process start-up | Each Protenix/Boltz process pays ~80–140 s start-up; batch many designs per call (the funnel does). |

## 5. Which strategy: evidence so far

Judged identically (Boltz-2, 3 fresh seeds + Protenix-v2; consensus pass = Boltz ipSAE ≥ 0.5 and interface PAE ≤ 2 Å and v2 ipSAE ≥ 0.5; top 20 per arm):

| target | pipeline default 8×4 | pipeline scaled 100×4 | funnel, no cycling | funnel, 3 cycles |
|---|---|---|---|---|
| MDM2 | 5 / 8 | 20 / 20 | 20 / 20 (0.7 GPU-h) | 20 / 20 (1.1 h) |
| PD-L1 | 2 / 8 | 16 / 20 | **20 / 20 (1.2 h)** | 20 / 20 (1.8 h) |
| FimA | **[not evaluated]** | **[not evaluated]** | **[not evaluated]** | **[not evaluated]** |

- MDM2 saturates, so it cannot rank large-N strategies. PD-L1 does: the funnel's *worst* design on Protenix-v2 is 0.72–0.74 versus 0.17 for the scaled baseline.
- Cycling adds ~0.02 to median ipSAE for ~45–50% more GPU time and equal pass counts: **optional**, not default. Always judge cycled designs with models that were not in the loop.
- Do **not** add an MPNN "designability" pre-screen (ESMFold pLDDT / MPNN score are ≈ chance for binding; the fast refold already measures it).

## 6. ProteinMPNN: can we model more hydrophobics?

Short answer: **you can change the sequences, but it buys nothing.** Interface bias does raise the aromatic content of *raw* MPNN
output, and neither that nor reverting the weights changes the judged score, the interface aromatic contacts or the pass count
(measured 2026-10-09, below). **The default stays `--mpnn-bias none`.**

| PD-L1 funnel backbones, 160 sequences | Lys+Glu (all) | aromatic (interface) | hydrophobic (interface) |
|---|---|---|---|
| soluble weights (fork default) | 27.7% | 3.6% | 46% |
| original weights | 25.1% | 4.0% | 47% |
| original + global bias | 7.4% | 4.2% | 53% (too aggressive; aggregation risk) |
| **original + interface-only bias** | 20.5% | **6.8%** | **56%** |
| soluble + interface-only bias | 24.3% | **7.0%** | 52% |

- Reverting to `original` changes hydrophobics by ~4 points and aromatics not at all. Earlier tests also showed the weights do not change the Boltz-2 score tail.
- Real PD-L1 binders (ProteinBase) have 6.6% aromatics and PISA-measured 3.5 aromatic residues at the interface; the funnel's shortlists have ~1.
- `funnel/run_funnel.py --mpnn-bias iface` adds +1.0 (F,W,Y), +0.5 (L,I,M,V), −0.5 (K,E,D,N,Q) logits **only on binder residues within 10 Å of the target**.
- **Resolved (2026-10-09): the bias does not survive selection.** `iface:0.6` vs the default, matched runs (500 backbones, `--rounds 0`,
  `final_m` 60, top 20, soluble weights), judged head-to-head in one batch (`out/judge_bias/{mdm2,pdl1}`, n = 20 per arm):

  | target | b_ipSAE | second-judge ipSAE | interface aromatic contacts | SC | interface area Å² | consensus pass |
  |---|---|---|---|---|---|---|
  | MDM2 | 0.899 → 0.898 | 0.902 → 0.901 | 3.0 → 3.5 | 0.618 → 0.624 | 1108 → 1142 | 20/20 → 20/20 |
  | PD-L1 | 0.748 → 0.755 | 0.761 → 0.769 | 1.0 → **1.0** | 0.578 → 0.579 | 930 → 914 | 20/20 → 19/20 |

  Every difference is ≤ 0.009 in ipSAE and every Mann-Whitney p ≥ 0.19; the repository's own noise floor is 0.03, so these are
  indistinguishable from zero. The *goal* of the bias — aromatic contacts at the interface — did not move on PD-L1 at all.
  **Why:** the bias raises whole-sequence aromatics in raw MPNN output (PD-L1: 2.7% → 3.9%, `out/mpnn_bias.log`), but the shortlists of
  both arms land on the same 3.5% — the default arm's rises 2.7% → 3.5% under selection while the biased arm's falls 3.9% → 3.5%.
  The fast screen and consensus converge on the same composition whatever prior MPNN was given, so a sequence-stage prior cannot be
  the lever. Flagged designs also went *up* (MDM2 2 → 4, PD-L1 4 → 7). What this does not show: n = 20 per arm, unpaired (different
  designs), two targets, one bias scale; it cannot exclude an effect smaller than ~0.03 ipSAE, and it does not test a *stronger* bias
  (which §10 already warns carries aggregation risk) or a hallucination/co-design generator (§12), which remains the open route.

## 7. Reading the outputs

- `final_*.csv` (shortlist, de-duplicated at 60% identity) and `consensus_*.csv` (all 60 evaluated). Key columns: `b_ipsae`, `b_paemin`, `v2_ipsae`, `consensus_pass`,
  `hotspot_frac`, `pisa_*`, `pisa_flags`.
- **fastPISA flags are triage, not ranking.** Flagged designs (thin interface < 630 Å², < 4 H-bonds, no aromatic contact, apolar fraction < 0.33) were less likely to be real
  binders among gate-passers (38% vs 65%, p = 0.01, small n). They add ≈ +0.01 AUROC beyond ipSAE: do not weight them into a score.
- **Check the flags against your own target's native interface before you let them triage anything.** The thresholds come from a
  mixed wet-lab set, and a natural, high-affinity interface can fail them. Measured on CD47/SIRPα (2JJS chains C/A, the real
  complex): 922 Å², 53 interface residues, 17 H-bonds, **13 salt bridges**, apolar BSA fraction **0.24**, **1** interface aromatic.
  That native interface trips both `apolar_fraction < 0.33` and the aromatic flag, so on CD47 those two flags penalise designs for
  resembling the biology. Compute the native reference first (`funnel/pisa.py:pisa_metrics(pdb, a=target_chain, b=partner_chain)`
  on a two-chain extract) and say which flags are informative for that target.
- `pisa_metrics` takes the interface's **own** row (matched on `interface_id`). It used to read row 0 of the summary, which is the
  requested pair only when the structure has exactly two chains — true for every funnel complex, so no funnel number was affected,
  but on a 4-chain input it was wrong by 6.5× (2JJS C/A: 149 Å² reported vs 975 Å² actual). Relevant to `phbind/` and to any
  reference measurement on a crystal structure; a pair with no interface now returns NaN + `error` rather than another pair's numbers.
- Shape complementarity and interface pLDDT were the strongest single extras in ProteinBase but are target-dependent / not yet implemented here.
- Never compare `bz_*` scores across runs or batches; compare only within one judge run.

## 8. Known weaknesses to design around

| weakness | mitigation |
|---|---|
| Designs are ~84% helical, 30–45% Lys+Glu, aromatic-poor | **not fixable at the sequence stage** (§6: interface bias measured, no effect — selection erases it); inspect `pisa_n_aromatic_iface`; order a diverse panel; the open route is a co-design generator (§12) |
| Two predictors place the binder within 5 Å in only ~40% of cases | do not treat one predicted pose as the binding mode; consider ensembles/third predictor |
| Gate `bz_gate_egfr_provisional_v1` is EGFR-derived | use the consensus judge; calibrate per target on labelled designs where available |
| ProteinBase negatives are pre-filtered | benchmark AUROCs are pessimistic against junk, optimistic about nothing |
| No wet-lab loop | ask for 10–20 experimental results to calibrate before scaling up |

## 9. What not to do

- Do not enter hotspots as raw numbers on a shard; do not skip `load_target`'s identity check.
- Do not run two GPU jobs; do not `pkill -f`; do not trust an exit code from `cmd; echo`.
- Do not report a consensus pass as a binder, or compare scores across runs, or quote a gain without n and an interval.
- Do not pip-install into the working envs without `--no-deps` (numpy/torch pins); external code lives in `.pxd/ext/`.
- Do not use composition/length features to predict binding: they identify the *design method* (AUROC 0.84 random CV → 0.73 held-out methods).

## 10. Why are the designs Lys/Glu-rich? (investigated 2026-10-04)

Tested, in this order:

| hypothesis | result |
|---|---|
| The scorers prefer charged helices, so selection enriches Lys+Glu | **No.** Raw MPNN sequences already carry it (MDM2 42.5%, PD-L1 30.4%); the top 10% by predictor score has *less* (MDM2 39%) or equal (PD-L1 31%) Lys+Glu. |
| The "soluble" MPNN weights | **Minor.** Original weights lower it by ~2–3 points. |
| MPNN temperature | **No effect** (T = 0.0001 / 0.1 / 0.3: 28.2 / 28.1 / 25.9%). |
| Our diffusion backbones are unusually helical | **No.** 95–96% helical, same as mosaic (95%) and RFdiffusion (94%); BoltzGen 81%, BindCraft 88%. |
| It is specific to PXDesign | **No.** ProteinBase medians: mosaic 32.5%, PXDesign submissions 31%, RFdiffusion 28%, BindCraft 26%, **BindCraft2 25.5%**, BoltzGen 14%. RFdiffusion's own native IL7R designs (81% wet-lab hit rate) are 42% Lys+Glu. |

**Cause (best supported):** ProteinMPNN's amino-acid prior on idealized helical surfaces. Half of all surface positions come out Lys/Glu (50%), whichever tool made the backbone.
**What does differ from BindCraft(2) is aromatics:** 11–12% vs 3.7–4.8% for the MPNN-only tools (RFdiffusion, PXDesign). BindCraft keeps its AF2-hallucinated sequence and runs MPNN
only on the interface; MPNN-from-scratch on a helical bundle asks for few aromatics. Our backbones also elicit fewer aromatics from MPNN (2.7%) than other tools' backbones do (~6%).
**So the lever is the sequence stage, not the generator or the scorer:** interface-only bias (built, test queued), a moderate K/E penalty on surface positions, or a hallucination/co-design
step that supplies aromatics (§12). High Lys+Glu alone is not disqualifying (wet-lab RFdiffusion binders have it); aromatic-poor interfaces are the better warning sign.

## 11. Metric bands (binder-probability tiers from wet-lab data)

Binder rate per band, 90% Wilson intervals, ProteinBase wet-lab labels:

| metric | band → binder rate | data |
|---|---|---|
| Boltz-2 ipSAE (min) | <0.3: 21% · 0.3–0.5: 41% · 0.5–0.7: 56% · 0.7–0.8: 61% · ≥0.8: 75% (n=8) | our 206 (base 50%) |
| Protenix-v2 ipSAE | <0.3: 40% · 0.3–0.7: 13–32% · 0.7–0.8: 74% · ≥0.8: 78% | our 206 (non-monotone below 0.7) |
| interface PAE min | <0.8 Å: 73% · 0.8–1.5: 51% · 1.5–3: 38% · >3: 8–27% | our 206 |
| **both models ≥ 0.7** | **75% [65–84]** vs one ≥ 0.5: 45% · neither: 29% | our 206 |
| Boltz-2 ipSAE (Adaptyv) | <0.2: 3% · 0.2–0.6: 7–8% · 0.6–0.8: 13% · ≥0.8: 39% (n=18) | Nipah G (base 10%) |
| shape complementarity (Adaptyv, 0–100) | quintiles: 6 / 4 / 5 / 11 / **22%** (>58.5) | Nipah G |
| Boltz-2 interface pLDDT | <0.7: 1% · 0.7–0.85: 6–8% · 0.85–0.9: 18% | Nipah G |

Use: tier designs by how many independent signals are in their top band (both models ≥ 0.7; interface PAE < 0.8 Å; shape complementarity in the top quintile where computed) and
report the *tier's* empirical binder rate, not a single score. Caveats: bands are target-dependent (rates differ 5× between Nipah G and PD-L1), intervals are wide, ProteinBase negatives are
pre-filtered, and the Anthropic × Adaptyv competition results were not available in the 2026-01-28 export; add them (same code, `funnel/calibrate.py` planned) and refresh the bands.

## 12. Hallucination / joint sequence-structure design without AF2

Verified by search (2026-10-04), not yet installed or tested here:
- **BoltzDesign1** (github.com/yehlincho/BoltzDesign1): gradient hallucination through Boltz-1 (Pairformer + confidence head), then LigandMPNN; reported without experimental validation at the time.
- **Mosaic** (github.com/escalante-bio/mosaic, JAX): composes Boltz-1/2, BoltzGen, AF2, OpenFold3, ESMFold2, **Protenix**, ProteinMPNN variants, ESM and stability models into one loss, so composition priors
  (aromatics, charge) can be explicit loss terms. Listed in ProteinBase with high hit rates (PD-L1 80%, IL7R 85%; n = 20 each).
- **BoltzGen** (github.com/HannesStark/boltzgen, MIT): all-atom generative co-design of sequence and structure with a binding-site specification language; ProteinBase shows 1–31% hit rates depending on target.
- ColabDesign (vendored) supports AF2 hallucination; independent of the judges, which is a plus for AF2 specifically.
Recommendation: pilot Mosaic (Boltz-2 + Protenix + an aromatic/charge prior) on one pocket target and judge with the existing consensus protocol; it is the most direct fix for §10.

## 13. Engineering: the LayerNorm compile

Cause: `torch/include/ATen/core/List_inl.h:202` uses `typename decltype(impl_->list)::difference_type`, which GCC 15 rejects ([-Wtemplate-body]); `-fpermissive` does not help; `g++-13` is not installed.
Fix (verified: builds in 26 s): override that one line via an include path ahead of torch's (torch's file untouched). `funnel/build_layernorm.py` builds it inside `.pxd/ln_build/`;
`--install` copies the `.so` where `layer_norm.py` imports it first, removing the ~80 s failed compile from every Protenix start (≈15–20% of a funnel run);
`--verify` (idle GPU) compares it against `torch.nn.functional.layer_norm`. **Not installed in the reference environment**; install it into a private environment and run `--verify` on an idle GPU first.

## 14. Preparing for lab feedback

1. **Record, at ordering:** design id, sequence, run folder, all scores (Boltz-2 per seed, v2), tier, `pisa_flags`, MPNN settings, hotspot residues.
2. **Order for learning, not only for winning:** ~10–20 designs stratified over tiers (include a few from the middle and lowest passing tiers and a few flagged), plus a known binder (positive) and a
   scrambled or off-target-looking design (negative). That is what lets the bands be recalibrated.
3. **Order for diversity:** one design per sequence cluster (<60% identity) and per backbone family; include a small set from the interface-bias arm if its test is positive.
4. **When results arrive:** compute binder rate per tier/band with intervals, per target; update §11; decide on the gate and whether to change generator or sequence design (§10, §12).

## 15. Cyclic peptides: what this setup can and cannot do today (checked 2026-10-04)

Facts verified in the installed code (not yet exercised on cyclic data):
- **Boltz-2** supports cyclic chains natively: a chain property `cyclic: true` sets `cyclic_period = len(sequence)` in the parser. This is the primary scorer for peptides.
- **This repo's AF2 evaluator** already has an `is_cyclic` option and `add_cyclic_offset` (`pxdbench/tools/af2/`): usable as an independent second judge (different model family from Boltz).
- **Protenix** shows no cyclic-chain option (only a `cyclic-pseudo-peptide` ligand type). The Protenix-0.5-mini fast screen and Protenix-v2 judge would model a linear chain, so they are not valid for head-to-tail cyclic peptides.
- **PXDesign diffusion** is not cyclic-aware and is built for 50-150 aa proteins; it is the wrong generator for 8-20 aa cyclic peptides.
- **BoltzGen** (MIT; weights are downloaded on first use) has a `peptide-anything` protocol with disulfide-bond constraints (`bond: atom1 [S, i, SG] ... atom2 [S, j, SG]`) and per-residue `binding_types` for the target. I could not confirm head-to-tail
  cyclization support from its README; check the docs before relying on it. ColabDesign-style AF2 hallucination with a cyclic offset (as in BindCraft's cyclic option) is the established alternative.

Proposed adaptation of the funnel (not built):
1. Target JSON: `binder_cyclic: true`, `binder_length: 10-18`, hotspot residues as now.
2. Generate: BoltzGen `peptide-anything` (or AF2 cyclic hallucination); no MPNN-on-Gly step. If sequences are redesigned, MPNN needs residue-index offsets so the termini are neighbours (small patch, untested).
3. Screen: Boltz-2 single seed with `cyclic: true` (small binders fold in ~3 s; there is no cheap cyclic-aware model, so the "fast screen" is Boltz-2 itself).
4. Judge: Boltz-2 (3 seeds, cyclic) + AF2 with cyclic offset. Replace ipSAE-only gating with ipTM, interface PAE, peptide pLDDT, **closure check** (N-C peptide-bond distance ~1.33 A), rigidity (spread over seeds) and fastPISA/SC.
5. Composition rules differ: peptides are synthesised (non-natural residues possible but not modelled), so Lys/Glu and aromatic rules from the protein work do not transfer; Cys only if a disulfide is intended.
6. Validation first: fold known cyclic-peptide/protein complexes with `cyclic: true` and with the chain linear; the cyclic run must close the ring and score higher; then compare Boltz-2 vs AF2-cyclic agreement. Effort: scoring changes are hours; a validated generator integration is ~1-2 days.


## 16. Resume, extend and multi-GPU (implemented 2026-10-05)

Every stage is split into units that are saved as they finish: generation chunks (`--chunk`, default 100 backbones), per-chunk design + fast screen, every cycling round (`cycle/state.json`),
every Boltz/Protenix prediction. Re-running the same command continues from the last finished unit; raising `--n-backbones`, `--rounds` or `--final-m` extends the same run.
Smoke-tested by killing a run mid-stage, resuming (only the unfinished chunk was redone), extending the backbone count (only the new chunk) and raising the rounds (continued after round 1).
A changed start set (more backbones) restarts cycling and archives the old state (`cycle_old_*`). `--gpus` runs units in parallel (tested logically with two workers on one card; a real 4-GPU run has not been done).

## 17. Dock a known scaffold, redesign the interface (experimental)

`funnel/dock_redesign.py`: LightDock (GPL-3.0, own venv, subprocess only) with restraints on the hotspots -> diverse top poses -> `mpnn_run.py scope=interface` (only binder residues within 10 A of the
target change; the rest of the scaffold keeps its identity) -> `--designs-csv` into the funnel (interface-only cycling). The un-redesigned scaffold is included as a control.
Install: `./scripts/setup_extras.sh`. Replace `dock_one()` to use another docker.

**First evidence (smoke test, FN3 monobody and protein A domain on FimH, 4 poses x 4 designs):** all 32 redesigns scored ipSAE ~ 0. Diagnosis (`ipTM` 0.11, median binder displacement 21-34 A from the docked position):
the co-folding predictors do not believe in the interface at all, so they place the binder anywhere. A natural scaffold with 14-23 swapped interface residues carries no interface signal the predictor recognises, even if the docked pose is physically reasonable.
Consequences: (1) judging a docked pose with a co-folding model measures *recognisability of the sequence as a binder*, not pose quality; (2) forcing the pose (template or contact constraints) would inflate scores for any sequence - the shuffle control would expose it;
(3) a fair judge of a docked pose needs an energy/physics term (e.g. Rosetta interface dG, MPNN score of the complex), which is not installed here. The full overnight run measures how far more poses and interface-only cycling move this.

## 18. Publishing the code without private data

`python scripts/export_public.py --dest ../bindpnyx-public --run-tests` copies an allowlist (no structures, MSAs, run outputs, per-design benchmark tables, private manifests or git history),
sanitizes machine paths, runs `scripts/check_public.py` (personal paths/handles, emails, credentials, data files, large files, symlinks) and the test suite inside the clean tree
(616 passed, 64 skipped, 0 failed at the time of writing). It does `git init` but no commit and no remote. Targets are rebuilt from public PDB ids by `funnel/fetch_target.py`.

## 19. Second judge: OpenFold3 instead of Protenix-v2 (implemented 2026-10-05; measured elsewhere, not wet-lab validated by us)

`run_funnel.py --second-judge of3` folds the survivors with OpenFold3 (checkpoint **p2-155k**, template-free, target MSA, binder as a query-only MSA: the benchmark's `of3` setup) in place of Protenix-v2. The gate is unchanged (`consensus = min(b_ipsae, <judge>_ipsae)`; Boltz-2 3 seeds, OpenFold3 1 seed). The default is still `v2`.

Why: on the `Anthropic/claude-protein-binder-design` v1.0 release (1,320 designs, 354 binders, 15 targets), holding the funnel's rule fixed and changing only the second oracle, precision@10 (paired bootstrap over targets) was:

| second judge | vs Protenix-v2 | 95% CI |
|---|---|---|
| OpenFold3 | +0.093 | [+0.020, +0.173] |
| RoseTTAFold3 | +0.053 | [-0.013, +0.120] |
| ESMFold2 (full) | +0.040 | [-0.033, +0.107] |

Only OpenFold3 clears zero; the funnel's current pair scored 0.467 (worst of the pairs tried) and Boltz-2 + OpenFold3 0.560. **These numbers come from an external analysis. They could not be re-derived here: the release's wet-lab labels are not included in this repository** (neither is the score/PAE bundle). Treat them as a prior, and recheck on your own lab results.

Rules that came with it:
- **Two oracles, not three:** 2 vs Boltz-2 alone +0.100 [+0.027, +0.187]; a third +0.000 [-0.020, +0.020].
- **Freeze the pair.** Choosing the oracle combination per target scored 0.480 vs 0.553 for a fixed pair (-0.073 [-0.153, -0.007]).
- **Do not fit a scorer** on seed instability, length, cycling rounds or provenance: every added feature lowered held-out precision@10 (plain consensus 0.587, everything 0.467).
- **Seeds:** aggregation (mean/median/min/max) is worth 0.003; the second oracle +0.100. The funnel still spends 3 Boltz-2 seeds; moving to 1 Boltz-2 + 1 OpenFold3 at equal cost is *not* implemented, because the gate's `b_ipsae >= 0.5 & b_paemin <= 2` was calibrated on 3 seeds.
- **Check the second oracle on a native binder of YOUR target before gating on it (measured on a hard single-chain target, 2026-10-10).**
  Folding the real natural partner (a known nM-pM binder) and its composition-matched shuffle in the same batch: Boltz-2 0.526 / 0.105,
  **Protenix-v2 0.742 / 0.000**, **OpenFold3 0.000 / 0.000**, AF3 0.33-0.39 / 0.000 (two GPU types, median replicate difference 0.002).
  Across 83 designs OpenFold3 passed only 39% at >= 0.5 against 96% (Boltz-2) and 87% (Protenix-v2), with rank agreement of only 0.35-0.49.
  So on this target OpenFold3 could not recognise the one binder known to be real, and the external +0.093 above did not transfer.
  With n = 1 native positive (an Ig domain, unlike the helical designs) this cannot separate "strict" from "blind"; it is enough to
  refuse to gate on it. Protenix-v2 stayed the gate partner; OpenFold3 and AF3 are reported columns. The general lesson is in
  `docs/DESIGN_PATH.md` section 2: an oracle that scores a known binder 0 cannot rank designs, whatever a benchmark says.
- **Directional ipSAE:** `*_ipsae_b2t` (binder->target) is reported for Boltz-2, Protenix and OpenFold3. Across oracles it ranked slightly better within target than the min (AUROC 0.718 vs 0.701), but the gate is calibrated on the min, so it is report-only.

**`funnel/judge.py --second-oracle {protenix-v2,af3,of3}` (added 2026-10-10)** gives the independent judge the same choice the run was
selected with; before, `judge.py` always used Protenix-v2 whatever oracle selected the designs. Columns keep the shared `v2_*` names and
`arm_summary.csv` gains `o2_name`. Judging with an oracle that was not in the selection loop also removes the circularity AGENTS.md's
"sanity" gate warns about. The gate's thresholds were calibrated on Boltz-2 + Protenix-v2, so carrying `>= 0.5` over to another model is a
transfer, not a recalibration - report both oracles rather than only the one that gates.

Setup: `.pxd/envs/openfold3` is a symlink to a conda env (python 3.12, torch cu128, `pip install -e <openfold-3 clone>[cuequivariance]`); weights in `~/.openfold3/of3-p2-155k.pt` (override with `PXD_OF3_CKPT`, binary with `PXD_OF3_BIN`). Run one GPU job at a time. OpenFold3 needs rectangular a3m files named like a database (`colabfold_main.a3m`); our target MSAs have ragged rows, which `common.a3m_rectangular` pads (a target MSA with fewer than 2 rows is refused). Measured speed: 6 designs in 52 s including model load.

## 20. Improving designs you already have: ArcRefine (measured 2026-10 on two targets, n = 15 and n = 24)

Once a pool has plateaued, the lever is refining specific candidates rather than generating more.
Two routes exist here: MPNN cycling (§5, `--rounds`), and **ArcRefine**
([github.com/ken-osumi/ArcRefine](https://github.com/ken-osumi/ArcRefine), FoldArc; built on Mosaic),
which optimises a given binder sequence while carrying Boltz-2's single and pair representations
between sequence updates, with the stated aim of preserving the fold and binding mode.
Licence: **PolyForm Noncommercial 1.0.0** — academic/public-research use is permitted, commercial use
needs the author's permission. Install it in its own environment (it bundles Mosaic modules and will
collide with another Mosaic install).

Input is a JSON with `parent_sequence` plus one entry per target chain; run `--validate-only` first,
then `--output run`, which writes `optimized.fasta`/`result.json` (`optimized_sequence`) and separate
parent/final predictions. **The CIF B-factors are placeholders: read the saved pLDDT arrays (0-1).**

Measured over 15 completed runs (parents 65-200 aa, several generators, one target; each refinement
re-judged against its own parent):

| identity to parent | n | mean change in independent Boltz-2 ipSAE | kept after judging |
|---|---|---|---|
| 0.00-0.20 | 5 | **-0.646** | 0/5 |
| 0.20-0.34 | 4 | -0.358 | 0/4 |
| 0.34-1.00 | 6 | **+0.212** | 4/6 |

Two rules follow, and both matter more than any setting:

1. **ArcRefine's own ipTM does not tell you whether the refinement worked** (r = +0.08 with the
   change in independent ipSAE, n = 14). It failed in both directions: one run's ipTM *fell*
   0.813 -> 0.534 while independent Boltz-2 ipSAE *rose* 0.76 -> 0.88 (kept), and another's ipTM rose
   0.844 -> 0.850 while ipSAE collapsed 0.77 -> 0.08 (discarded). Never gate on it.
2. **Identity to the parent is the usable predictor** (r = +0.88). ArcRefine rewrites most of the
   sequence - identity to parent ranged 0.07-0.43, so it is not a conservative tweak. Compute identity
   first and drop anything below ~0.35 before spending oracle time. When it does retain the fold the
   gains are large: best observed 0.56 -> 0.91 and 0.63 -> 0.92, with the orthologue moving the same way.

**Judge the refined sequence and its parent in the SAME batch, on an oracle family that was not in
ArcRefine's loop.** ArcRefine optimises against Boltz-2, so Boltz-2 alone is partly circular; carry
AF3 or OpenFold3 alongside. Comparing a refined score against a parent score from an earlier batch is
invalid (§4).

**Replication on a second target did NOT confirm the identity rule, and ArcRefine hurt.** Twelve
refinements of already-passing, cycled single-chain designs (parents 0.68-0.89 consensus), each
re-judged against its own parent in one fresh-seed batch (Boltz-2 x3 + a second oracle):
parents passing 12/12 -> refined passing **5/12**. In the >= 0.35-identity band (n = 10) the mean change
in consensus was **-0.357** (the same band gained +0.212 on the first target), only 2 of 10 improved by
more than 0.03 (best +0.129), and r(identity, change in consensus) was **+0.22** against +0.88 on the
first target. The damage was oracle-specific: several refined sequences kept their Boltz-2 ipSAE
(8 of 10 still >= 0.5) while the second oracle collapsed (0.860 vs 0.077 for one pair), and two did the
reverse. That is what optimising through one model should produce, and it is why the second-family
re-judge is not optional. Untested hypothesis: headroom - the first target's keepers started at
0.56-0.63, these parents started at 0.68-0.89, so there was little to gain and much to lose.
**Practical rule: treat ArcRefine as a lottery ticket per design, never as an upgrade.** Keep the parent,
add the refined sequence only where it wins on the independent oracle in the same batch.

What this does not show: n = 15 on the first target and n = 12 on the second, heterogeneous parents,
and on the first target identity is confounded with length (every keeper was a 160-192 aa parent). The
0.35 cut was a post-hoc split on 15 points and did not replicate - re-derive it per target or do not
use it. Known input error: if `template_pdb` residues differ
from the configured target sequence the run aborts; rerunning without the template succeeded.

### Binder length: what was and was not shown
On the same campaign, longer binders were **far more structurally novel** - Foldseek-vs-PDB clean rate
rose from 21% at 62-112 aa to ~98% at 180-200 aa (TM >= 0.8 over >= 70% coverage counted as a hit), which
is why generation moved to long binders there. That was a *novelty* result, driven by short helical
bundles recapitulating known folds. A separate claim that length improved *binding scores* was made
from one favourable batch and later **retracted**. So: use length to buy novelty when novelty is a
requirement; do not expect it to raise ipSAE, and do not carry the novelty finding over to a target
where nothing is filtering on novelty.
