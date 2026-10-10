# The design path: what we actually learned running this pipeline

This is the distilled operating knowledge from running binder-design campaigns with this code —
the things that changed a decision, with the measurement attached. It is deliberately
**target-agnostic**: the stages and the rules transfer, the thresholds do not.

Every number here comes from one of four sources, named each time:
- **own single-chain targets** — this repository's own runs (`docs/REPORT.md`, `funnel/results/`).
- **own multi-chain campaign** — a 3-chain homo-oligomer campaign run with `phbind/`.
- **public wet-lab release** — the public 1,320-design / 354-binder / 15-target dataset
  (`Anthropic/claude-protein-binder-design`, CC-BY) analysed in `bench/`.
- **public benchmark portal** — aggregated per-target hit rates from a public design portal.

Nothing in this repository has been wet-lab validated by us. A "pass" means independent
predictors agree on a confident interface; it is not evidence of binding.

---

## 1. The path

```
define the site from structure  ->  generate at scale  ->  cheap screen  ->  optimisation depth
  ->  novelty (if it is a requirement)  ->  two-oracle gate with carriers  ->  controls
  ->  triage on quality flags  ->  diverse panel
```

The expensive models only ever see designs that survived cheap ones. Generation is not the cost;
scoring is. On one consumer card a 500-backbone campaign with three optimisation rounds costs
~1.2 GPU-hours end to end (own single-chain targets, 70-aa binder on a 115-aa target).

## 2. Pick the oracle pair per target class — never inherit it

**Two oracles, not one and not three.** On the public wet-lab release, holding the gate fixed and
changing only the second oracle: two oracles beat one by +0.100 precision@10 (95% CI
[+0.027, +0.187]); adding a third was worth **+0.000** ([-0.020, +0.020]).

**Freeze the pair across targets.** Choosing the best oracle combination per target scored 0.480
versus 0.553 for one fixed pair (-0.073, [-0.153, -0.007]) — picking per target overfits.

**But the pair must be chosen for the target *class*, and that choice must be validated on solved
complexes before it is trusted.** The same oracle behaved completely differently in two regimes:

| oracle | single-chain targets (public wet-lab release) | 3-chain homo-oligomer (own multi-chain campaign) |
|---|---|---|
| second-oracle A | **+0.093 precision@10** over oracle B ([+0.020, +0.173]) | **exactly 0.0 on every solved reference binder** — unusable |
| second-oracle B | baseline | usable, but kept only as an alternative |
| a fast "mini" screening model | fine as a cheap pre-filter | **89% of predictions exactly 0** — no signal at all |
| an all-atom model from a third lineage | not wired into the single-chain gate | the one that worked; became half the gate |

So: **fold the target's known binders first.** If an oracle cannot recognise a solved complex for
your target, it cannot rank your designs, whatever a benchmark says. Prefer a second oracle from a
different model lineage than the first — correlated oracles add less information (two predictors
here agree only moderately, Spearman ~0.5, and place the binder within 5 Å in only ~40% of cases).

**Do not fit a scorer.** On the public wet-lab release every added feature lowered held-out
precision@10 (plain consensus 0.587 -> 0.467 with seed-stability, length, provenance and rounds
added). Seed aggregation (mean/median/min/max) is worth 0.003; the second oracle is worth +0.100.

**Report the full oracle table for the final panel and name the dissenters.** On a ten-design panel
chosen by a two-oracle gate, an independent third family agreed on nine; the tenth was scored 0.85 by
both gate oracles and 0.27 / 0.01 by the other two. Keep the pre-registered panel, flag that design,
and name the best alternate that all families agree on, with its batch-to-batch stability, so the
person ordering can decide. Do not swap silently after seeing the cross-check.

## 3. Carriers in every batch

Every scoring batch carries reference designs with known expected ranges: at least one expected
failure and two expected passers. If the failing carrier does not read ≈ 0, or a passing carrier
drifts far outside its range, **stop and find out why before reading anything else**.

This is what makes a number interpretable, and it produces the single most important asymmetry:

> **A miss is uninformative; only a high score means anything.** In the own multi-chain campaign the
> primary oracle recovered 1 of 4 solved binders and the second recovered 0 of 4 — and a scrambled
> sequence scored like the misses. So a low score cannot distinguish "bad design" from "oracle
> cannot see this target".

Corollary: never report a low score as evidence against a design without showing that the oracle
can score a known binder for that target.

## 4. Batch discipline, and what counts as a replicate

- Compare designs **only within one batch**, with the parent or reference present. Batch effects
  are small but real (SD ≈ 0.02); cross-batch comparisons of absolute scores are invalid.
- A **replicate** is a differently composed batch (different members or order) or a different
  machine. An identical rerun is byte-identical and says nothing. Output can depend on the command
  line and on batch composition.
- Report the pooled mean over batches and the per-batch means — **never the best batch**. In the
  own multi-chain campaign the first batch overstated an effect by roughly 2x, twice; a single
  batch of five is not a result.
- A repeated number from the same operator and method is replication, not independent
  confirmation. Say which one you have.

## 5. The metric

- Use **interface ipSAE** (PAE between binder and target tokens, 10 Å cutoff), in the **minimum of
  the two directions**. On a multi-chain target the maximum direction reads ~0.17-0.20 higher and
  admits designs an earlier calibration rejected. Group all target copies as one.
- **Global ipTM is not a binding metric** — it read 0.86 on a negative control.
- **Interface descriptors are not ranking signals**: buried area, contact count, H-bonds and salt
  bridges gave within-target AUC 0.507-0.549 on 1,320 assayed designs. Use them as triage flags
  only, never as score terms.
- Thresholds are target-dependent. Binder rates per band differ ~5x between targets in the public
  wet-lab release, so a band table calibrated on one target is a prior, not a rule.

## 6. Optimisation depth is the largest generator-independent lever

Iterating redesign-on-the-predicted-complex then refolding, keeping the best (this repo:
`--rounds N`), matters more than which generator produced the backbone.

- Public wet-lab release: **45.9% hit rate at 5+ optimisation rounds vs 18.9% uncycled.**
- Own single-chain easy targets: cycling adds only ~0.02 median ipSAE for ~50% more GPU time —
  which is why this repo long described it as "optional".
- Own single-chain **hard** target: three rounds took consensus passes from **12/60 to 28/60** and
  the best consensus from 0.806 to 0.876, for 14 minutes of extra GPU time.

- **How many rounds?** On a second pool from a different generator (2,400 designs, same hard target), mean
  parent fast-screen ipSAE by round was 0.589, 0.621, 0.637, 0.647, 0.656, 0.661 (steps +0.032, +0.016,
  +0.010, +0.009, +0.005). Consensus passes among the top 80 were 39 uncycled, 49 after 3 rounds and 53
  after 6. Those counts come from separate runs and separate batches, so a difference of 4 is inside
  batch and sampling noise; rounds 4-6 cost about as much GPU time as rounds 1-3. **Three rounds
  captured nearly all of the measurable gain; do not assume more is better without a same-batch test.**

**Rule: cycling is optional only when the target already saturates.** If the fast screen's median
is near zero, cycle. Always judge cycled designs with an oracle that was not inside the loop.

## 7. Sequence design

- **Never plain ProteinMPNN.** Public wet-lab release: SolubleMPNN 27.9% hit rate vs plain
  ProteinMPNN 4.8%.
- MPNN weights and temperature do **not** move the score distribution (own single-chain targets).
- An interface aromatic/hydrophobic logit bias raises the *raw* sequence composition but **does not
  survive selection**: both arms' shortlists converge to the same composition, scores move
  <= 0.009 (all p >= 0.19), and quality flags went up. See `docs/RECOMMENDATIONS.md` §6.
- High Lys+Glu is not disqualifying (wet-lab-validated binders from other tools carry it);
  aromatic-poor interfaces are the better warning sign — **but check that against your own
  target's native interface** (§10).

## 8. Novelty is often the binding constraint, not affinity

If designs must be structurally novel (a portal rule, or freedom-to-operate), **screen novelty
before spending oracle time**, because it can reject most of what passes the gate.

- Own multi-chain campaign: 18 of 21 two-model passers matched a known structure at TM >= 0.8 over
  >= 70% of the chain.
- The rate is strongly **length-dependent**: ~21% clean at 62-112 aa rising to ~98% at 180-200 aa.
  Short helical bundles recapitulate known folds. Generation moved to long binders for this reason.
- **This is a novelty result, not an affinity result.** A separate claim that length improved
  binding scores came from one favourable batch and was retracted. Do not expect length to raise
  ipSAE, and do not carry this over to a target where nothing filters on novelty.
- **Affinity-side length curve on a second, single-chain target with a compact epitope** (fast screen,
  fraction of designs >= 0.5 ipSAE; compare lengths only within one screen run): 55 aa 1.0%, 60 aa 2.8%,
  65 aa 2.5%, 70 aa 2.7%, 85 aa 0.5%, 95 aa 0.0%, 140 aa 0.0%, **180 aa 1.0%** (2 of 200). The best
  lengths were the short ones, performance collapsed from ~85 aa upward, and a small tail returned at
  180 aa. n = 200 per length above 70 aa, so the 180 aa figure is two designs - read it as "not
  strictly monotonic", not as a finding. For a compact epitope, extra length buys surface the site
  cannot use; length is a novelty lever, not an affinity lever.
- A **whole-chain** TM proxy is not the same check a domain-segmenting portal runs: a design built
  from two separately known-like domains can pass a whole-chain check and fail a per-domain one.
  Treat many strict hits on an alignment-normalised column as a likely failure.

## 9. Refining what you already have

Two routes: MPNN cycling (§6) and ArcRefine (structural carryover through a co-folding model).
Full measured rules in `docs/RECOMMENDATIONS.md` §20. The two that matter:

1. **A refiner's own confidence does not tell you whether it worked** (r = +0.08 with the change in
   independent ipSAE, n = 14). It failed in both directions.
2. **Identity to the parent predicted the outcome on the first target** (r = +0.88; below ~20% the
   binder is destroyed, mean -0.646; at >= ~35% the gains were large, +0.212 mean, best 0.56 -> 0.91)
   **but did not replicate on a second one** (n = 12 refinements of already-passing designs: r = +0.22,
   the >= 0.35 band lost 0.357 on average, parents 12/12 passing -> refined 5/12). Refinement
   moved the score on the model it optimised through and lost it on the other. Treat a refiner as a
   lottery ticket per design: keep the parent, accept the refined sequence only if it wins on the
   independent oracle in the same batch.

Always judge the refined sequence **and its parent in the same batch**, on an oracle family the
refiner did not optimise against. This is also the section where one target's rule failed on the
next: a threshold fitted on 15 points (here an identity cut) is a hypothesis until a second target
agrees.

## 10. Calibrate your triage flags against the target's own native interface

Quality flags (thin interface, few H-bonds, no aromatic contact, low apolar fraction) come from a
mixed set and **a genuine high-affinity interface can fail them**. Measured on the native complex of
this repo's current hard target: 922 Å², 53 interface residues, 17 H-bonds, 13 salt bridges, apolar
BSA fraction **0.24**, **1** interface aromatic — which trips both the apolar and the aromatic flag.
On that target those two flags would penalise designs for resembling the biology.

Compute the native reference first, state which flags are informative for that target, and decide
before you look at your designs.

## 11. Negative results — do not pay for these again

| tried | result |
|---|---|
| ESM-based stability (ΔG) as a binder discriminator | ~chance. Within-target AUC 0.502 (binder) / 0.587 (complex); adds nothing to the primary oracle (0.697 -> 0.704 combined). Useful for stability, not binding. |
| ESM-C sequence likelihood | **inverse** to binding (AUC 0.345; 0.655 inverted) |
| ProteinMPNN score / ESMFold pLDDT as a designability pre-screen | ≈ chance for binding; the fast refold already measures it |
| Composition and length features | identify the *design method*, not binding (AUROC 0.84 random CV -> 0.73 held-out methods) |
| Dock a known scaffold, redesign only the interface | all redesigns scored ipSAE ≈ 0. Co-folding models score **how recognisable a sequence is as a binder**, not pose quality; forcing the pose would inflate any sequence. A fair judge of a docked pose needs a physics/energy term. |
| Hotspot-string features to predict the oracle | null across 4 independent tests (AUROC 0.60) |
| A third oracle in the gate | +0.000 precision@10 |

## 12. Trap catalogue — each of these cost real time

| trap | what happens | guard |
|---|---|---|
| Hotspot numbers typed as shard indices | runs silently, wrong site; cut hit rates from ~100% to 7-26% | define hotspots as `"Y56"` (PDB number + letter) and **assert residue identity** through the provenance map |
| A silently dummied MSA | inflates the score (+0.16 ipSAE on 4/4 designs) | validate unconditionally; a mismatched MSA must return a status, not a number |
| One NUL byte in an alignment | aborted a whole campaign's predictions | sanitise alignments before use (drop non-ASCII/ragged records) and assert no NUL |
| Checkpoint filenames that lie | a file named `...v0.5.0` was actually a v2 checkpoint | pass the model name explicitly and compare sha1 before trusting a name |
| `cmd; echo done` | reports `echo`'s exit status, not the command's | capture `$?` on the next line |
| `State=COMPLETED` / non-empty summary | the pipeline exits 0 on internal failure | check per-row status values and that counts match the request |
| `pkill -f <pattern>` | the pattern matches your own command line and kills the shell | take PIDs from `ps`; wait with `kill -0 PID` |
| A shared lock across parallel workers | two workers claimed one job; the refiner then died on an existing output dir | prefer **static round-robin assignment** over a claim directory |
| Concurrent GPU jobs | `cusolver` / CUDA OOM; a JAX job preallocates most of the card | one GPU job per card; set the allocator to non-preallocating if you must overlap |
| A test fixture encoding your assumption | proves nothing (one fixture here used chain letters where the API uses integer ids) | delete the guard and confirm the test fails |

## 13. External anchors worth knowing

- Generator hit rates on the public benchmark portal: BindCraft 42%, Complexa 34%, BoltzGen 28%,
  RFdiffusion3 27%, PXDesign 24%, Genie3 22%. Generator choice matters **less** than optimisation
  depth (§6).
- **Label ceiling:** two contract labs agreed on the binder call for 89.0% of 1,235 doubly-tested
  designs. No in-silico method can be meaningfully "better" than that agreement.
- A generator's own reported score is near-useless for cross-pipeline ranking (own-vs-common
  evaluator correlation r ≈ 0.11-0.13). Re-score everything with one evaluator before choosing.

---

## How to report a campaign

State, every time: what the gate was, which oracles, how many seeds, the carriers and what they
read, n, the interval, and **what the measurement does not show** (selection bias, circularity
between the loop and the judge, batch composition, and that nothing is wet-lab validated).
Order a diverse panel stratified over tiers with a positive and a negative control, not the top few
by score — that is what lets the thresholds be recalibrated when results arrive.
