# Selection and evidence: choosing which designs to submit, and knowing what the numbers mean

What one campaign learned about the half of binder design that is not generation. Generation is
covered by `DESIGN_PATH.md` and `RECOMMENDATIONS.md`; this document is about everything after you
have more candidates than slots, and it exists because **every expensive error in that campaign was
an inference error, not a compute error.** No infrastructure failure produced a wrong number. Every
wrong number came from reasoning about a correct one.

Measured on one peptide-agonist campaign against a class B GPCR: three co-folding oracles,
~40 scored batches, ~20 000 predicted complexes, one 20-row panel. Numbers below carry their n.
Re-check each on your own target; the *shapes* of the errors generalise further than the constants.

Companion skill: `skills/cofold-band-scoring/` — the method below as runnable blocks.

---

## 1. Select on mechanism, not on interface quality

The campaign's single most useful decision. If the target's function has a structural requirement -
a peptide that must insert into a pocket to activate, a loop that must be displaced, a specific
contact that distinguishes agonism from mere binding - then **score that requirement directly** and
let interface quality be a filter, not a ranking key.

The reason is not elegance. It is that **every generic interface descriptor tried in that campaign
failed the same control**:

> Buried surface area, solvation energy, hydrogen-bond count, salt bridges, contact counts, epitope
> recovery and domain-profile similarity were each scored over a full control block. Every one
> correctly separated a competitive antagonist, and **every one placed a composition-matched
> permutation of the native agonist level with or above the native agonist itself.**

Buried area was worst: the inert permutation buried more surface than the real hormone in **6 of 6
batches**, and among real peptides the tightest binder buried the least.

**The test to run before trusting any new quantity:** score it on a composition-matched permutation
of a known binder. A quantity that cannot separate a scrambled non-binder from the real thing is a
description, and should be reported as one. This single check retired seven candidate ranking
metrics in an afternoon.

### What a mechanism score looks like

Pick two or three geometric terms that together express the requirement - in that campaign an anchor
salt-bridge distance, a conserved-core contact count, and an insertion depth - and require a pose to
satisfy **all of them at once**. Then the score of a design is the **fraction of its sampled poses
that satisfy the conjunction**. See `skills/cofold-band-scoring/` for the construction, and §3 below
for why the fraction and not the best pose.

---

## 2. The control block is the experiment; the positives may be decorative

Put in every batch: **three real binders of the class you want, one binder that binds but does not
do the thing** (a competitive antagonist is ideal), and **one composition-matched permutation** of a
real binder. Then apply your gates to the negatives *before* looking at any design.

**Know which half of your control block is actually a test.** If you define your acceptance window
as extrema over the positive controls' pooled poses - `max(distance)`, `min(contacts)`,
`[min(depth), max(depth)]`, which is the natural construction - then **every positive pose lies
inside every edge by construction and the positives cannot fail.** Verified exactly, on every term,
in four batches across three different oracles: positives came back at exactly 25/25 every time.

All the certifying work is done by the two negatives, which are not used to derive the window. Say
so when reporting. Offering "our positive controls scored 100%" as evidence the batch is sound is
offering a tautology, and it is an easy one to publish by accident.

**The extremum construction has a second cost.** Each edge is then hostage to a single pose out of
~75. Measured: one oracle's anchor ceiling moved **1.91 Å between two of its own seeds** while the
95th percentile of the same poses moved **0.20 Å**, with the extremum coming from a different
control each time. **Prefer a quantile edge in a new campaign.** The price is that positives then
read 23-24/25 instead of 25/25 - which is the point. And never re-derive the edge definition after
the candidates are known.

---

## 3. A rate converges; an extremum wanders

Do not rank on a design's best or worst pose. Measured: the same design re-run at the same sample
count moved its anchor floor from **3.31 Å to 3.77 Å and flipped pass to fail.** An extremum over 25
draws is a property of the draw, not of the design.

Rank on the **fraction of poses satisfying the whole requirement**, and subtract a
composition-matched baseline:

    score(design) = rate(design) − mean over its permutations of rate(permutation)

**Use three permutations per design, not one** - a single permutation's rate carries the same
sampling noise as the design's. A design whose three permutations read 0/25, 0/25, 12/25 is telling
you something a mean hides, so print the individual rates beside the margin.

Why a permutation and not an unrelated decoy: it has the same residues and the same length, so
whatever the model's prior contributes to threading that composition into the site, it contributes
to both. The difference is what remains.

In that campaign this separation was overwhelming and consistent: **design vs its own permutations
p ≤ 1e−8**, on permutations holding the band in 0-4 of 75 poses. That is the claim such a panel
rests on, and it is a claim about each design against its own composition - never about one design
against another.

---

## 4. Band width is a property of the oracle, so a higher rate is not a better design

Same target, same control block, same seeds:

| oracle | acceptance window on the anchor term | width |
|---|---|---|
| AlphaFold 3 | 2.40-2.74 Å | **0.34 Å** |
| Boltz-2 | 2.26-3.42 Å | 1.16 Å |
| Protenix | 1.96-3.41 Å | **1.45 Å** |

The same design scored **12/25 on one and 20/25 on another**. That is not "harder here, easier
there" - the wide window admits poses the narrow one rejects outright.

**Use the narrow oracle to rank and the wide ones to confirm the sign**, state which is which, and
expect the orders to disagree: in that campaign the design ranked first on one oracle was ranked
**last** on another. A cross-oracle rank disagreement is the expected result, not a bug to
reconcile.

### A saturated band ranks nothing — say so instead of reporting an order

Real case: 15 sequences on the wide oracle returned **11 at 25/25 and the rest at 23-24/25,
parents included.** The correct report is "no ranking is possible in this batch", plus whatever the
batch does establish - there, that no variant broke the mechanism, which is a real and useful
negative. Then re-measure on the narrow oracle. Ranking inside the noise of a saturated band and
calling the top row a winner is the failure to avoid.

---

## 5. What counts as a replicate, and what a replicate will actually tell you

**A re-fold at the same seeds is not a replicate.** These oracles are deterministic given seed and
input, so re-running with the seed set you already used reproduces the rate exactly and the window
bit-identically. That is a pipeline check worth having, and it is not a second draw. Measured: a
26-case batch re-folded two leaders at their original seeds returned their earlier rates to the
structure, proving only that the scorer is deterministic.

Useful consequence of the same fact: if you need a **new** design's rate to be comparable to an
existing batch's rates, re-fold that batch's controls at its original seeds inside the new batch.
The window then comes out bit-identical and the comparison is *demonstrated* rather than assumed.
Verified: a 15-case batch built this way reproduced all three window terms exactly and returned both
re-folded parents' original rates.

**A real replicate uses seeds the campaign has not used — and here is what one found.** 26 designs
with their own permutations, re-folded at fresh seeds, scored against a window re-derived from the
new batch's own controls:

| quantity | result |
|---|---|
| sign of every margin | **reproduced — all 23 designs beat their own permutations again** |
| identity of the top row | reproduced |
| **rank order** | **Spearman ρ = 0.673 on rates, 0.623 on margins** |
| individual rows | moved by up to **+8/25**; rank 2 → rank 9, rank 15 → rank 4 |
| top row vs next two | **p = 1.00 and p = 1.00** (Fisher, two-sided) |

**So: a margin from one seed set is evidence of SIGN. It is never a tiebreak between two designs.**
That campaign wrote ordered tables for two weeks before measuring this, and every one of them
overstated what was known. Its own method document had said exactly this; writing it down was not
enough to stop it.

**Report the window alongside the rates in any replicate.** When a replicate disagrees, diff the
windows *before* concluding anything about the designs. The clean demonstration, worth copying as a
diagnostic: take one permutation whose per-pose distances barely moved between two batches
(3.52-4.63 Å against 3.54-4.91 Å) and show it holding the term **0/25 in one and 25/25 in the
other**. The permutation did not improve; the ceiling moved 1.9 Å.

---

## 6. Fix the selection rule before the numbers exist, and write it to a file with a timestamp

Not "decide a rule" - **write it down, with its primary key and every tiebreaker, in a file, before
the measurements it will be applied to exist.** Then do not re-open it when a late arrival favours a
candidate you have reason to want. A metric that appears after the candidates are known does not get
to reorder them; record it as an extra column.

The reason to put it in a timestamped file rather than in your head is that the alternative is
**indistinguishable from choosing the rows and then describing them** - including to yourself.

### Three tiebreakers that are secretly assertions

1. **"Rows with no measurement sort below rows that have one."** This is not a tiebreaker, it is a
   claim that unmeasured means bad. When the missing measurements were finally taken, several came
   back high: the panel was carrying a **0/25 row at rank 12 and a 1/25 row at rank 8** while rows at
   13/25 and 10/25 sat displaced out of it entirely. **An absence of measurement is not a low
   score**, and when your tiebreaker is doing this, the cheapest decisive experiment in the campaign
   is usually to go and measure them.
2. **Ranking on a key that is uncorrelated with the key that ranked the rest.** Check this
   explicitly; in that campaign two halves of one ordered table had been ranked on keys with
   Kendall τ = −0.05 between them.
3. **Lineage-blind slot filling.** If your candidates come from a handful of parent scaffolds or
   backbones, twenty slots spent on five lineages is five experiments with four replicates each. A
   cap of ~3 rows per lineage turned one panel from ~5 lineages to **10** at no cost in evidence
   quality, which is a broader test of the same hypothesis. Name the lineage from the design id and
   cap it in the rule.

### Breadth beats ordering when nothing separates

Once §5 has told you the order is not reproducible, the panel's job is not to rank 20 winners. It is
to **spend 20 slots on 20 individually-evidenced rows covering as much independent lineage as
possible.** Eligibility criteria (each row clears the novelty screen, has a measured rate above a
floor, and has a permutation baseline) plus a lineage cap is a better rule than any ordering, and it
is robust: when that panel was re-measured at fresh seeds, **all 20 rows still cleared the floor.**

---

## 7. Novelty screens, and the threshold trap that cost this campaign four days

If the rules require designs not derived from known binders, screen **before** spending oracle time.
Shuffle-based p-values against a precursor set are cheap and effective. But:

> **A binary gate on a Monte Carlo estimate will disagree with itself, and that disagreement is
> about your sample size, not about the sequence.**

One sequence returned p = 0.045, 0.043, 0.051, 0.052, 0.051 over five shuffle seeds, then **14
reject / 6 clear over twenty seeds.** That was read as "the sequence sits on the threshold, so its
verdict is unstable", and the campaign held it back on that basis for four days while using its
"instability" to argue that other rejected rows were a different kind of case.

Wrong. A proportion near 0.05 estimated from 2000 shuffles has a standard error of
`sqrt(0.05 × 0.95 / 2000) = 0.0049`. A true value of 0.045 therefore scatters across ~0.035-0.055
and lands above the cut about a third of the time - **which is exactly the observed 14/6 split.** At
**50 000** shuffles, four seeds returned 0.0449, 0.0449, 0.0458, 0.0444: a stable reject at
0.0450 ± 0.0007.

**The tell was in the data the whole time:** the observed spread (0.0395-0.0605) matched the
predicted ±2 SE band (0.040-0.060) almost exactly. When the scatter of your measurements equals the
known noise of your instrument, you are measuring the instrument.

**The rule.** For any threshold applied to a sampled estimate: compute the standard error *at the
threshold* and compare it to your seed-to-seed scatter. If the scatter does not exceed it, your
disagreements are your own sampling and **more seeds will never help - increase draws, not seeds.**

**Then ask the mirror-image question, which is the one nobody asks.** If under-sampling can push a
true 0.045 *above* the cut, it can push a true 0.048 above it too - so some of your *cleared* rows
may be clearing by luck. Check every clear row for proximity to the cut. In that campaign, none
were: the clears' nearest-to-firing p-values started at **0.209**, about 32 SE above the cut, with
nothing between 0.045 and 0.209. The verdicts were bimodal with a wide gap, which is a far stronger
statement about the cleared rows than anything the campaign had been entitled to say before. That
statement exists only because the error was found.

**Scope honestly.** A sequence-level screen against a precursor set is not a structural-novelty
check, and de novo provenance comes from the generation record, not from any screen. And a screen
firing is not the same as the organiser's unpublished filter firing - say which one you measured.

---

## 8. Picking a representative pose, and the tautology waiting there

Interface tools take one structure, so a pose has to be chosen. Three rules, each of which caught a
real error:

- **Choose among the poses that satisfy the requirement**, and mark the case `OUT_OF_BAND` if there
  are none. A picture of a pose that never satisfied the band is informative and must not be
  presented as a picture of a working pose.
- **Take the pose closest to the median of the batch's own in-band positive-control poses.** Do not
  hardcode that target: the same three controls gave 2.77 Å on one oracle and 2.64 Å on another, so
  a constant lifted from one batch quietly becomes a cross-batch reference.
- **Re-measure the chosen quantity on the file you wrote and refuse to write if it disagrees** with
  the scored value (tolerance ~0.015 Å). This is the only check that catches a wrong pose-to-file
  mapping. See §9.1.

### Then state what those files cannot show

Because the representative pose was chosen as the in-band pose closest to the controls' median, the
selected quantity is **nearly constant across the shipped files by construction**:

    across the in-band representative FILES (n=32)     2.60 - 2.72 Å, sd 0.026
    across their underlying scored poses (n=800)       2.35 - 18.21 Å, sd 0.906   (34× wider)

Anyone who opens those files and concludes "the designs reproduce the native binder's anchor
geometry" has read a tautology. Every individual step here was documented - the selection rule, the
re-measurement check, the median target - and **the consequence was still never stated anywhere**
until an outside reviewer pointed at it. Put that sentence next to the figure, not in a methods
appendix.

### Rates for interface features too

If an interface feature is going to be claimed as a property of a design, measure it over the whole
pose ensemble. Converting one such feature from a single-pose observation to a 25-pose rate changed
the conclusion from "the designs mostly miss this contact" to "the designs make it in **0 of 25**
poses and all four real peptides make it in **25 of 25**" - a different and much stronger statement.
Interface tooling is cheap enough for this (~0.6 s per complex, so 500 poses is minutes on 8
workers).

---

## 9. The failure taxonomy, ordered by how much damage each class can do

### 9.1 Silent corruption that a control cannot catch

Scorers commonly enumerate poses as `sorted(glob("*_model_*.cif"))`, which is **lexicographic**:
`model_0, model_1, model_10, …, model_2`. Globbing the raw prediction directories instead gives a
different order. Every design's analysed pose came out **0.04-0.22 Å off** - while the control,
whose 25 poses are nearly identical, **matched perfectly and hid the error for a full analysis
cycle.**

The lesson is general and uncomfortable: **the control that validates everything else cannot
validate pose indexing, because its poses are interchangeable.** Only an end-to-end re-measurement
on the written file catches this class.

### 9.2 The error where every individual step is true

The four most expensive errors in that campaign all had this shape: the pose tautology above; a
caveat that was correctly recorded in a result note and then summarised out of the document people
actually read; a prose claim ("all panel rows share this N-terminal segment") carried forward from
an earlier panel and used to justify two decisions, which **a three-line assertion falsified on row
5**; and the threshold misdiagnosis in §7.

Local correctness does not compose into a correct conclusion. The only reliable detector found was
**an outside reader who did not build the thing.** Commission that review, give them the structures
and the scripts, and budget a day for it. In that campaign it returned five accepted findings
including the best one in the whole review - and one finding that was *favourable* to the panel and
wrong, which matters: check findings that favour you at least as hard as findings that do not. That
one would otherwise have gone into a submission as a headline.

**Corollary worth adopting as policy: the agent that proposed a design does not validate it.**

### 9.3 Jobs that exit 0 having produced nothing

Every tool in this space exits 0 on inputs that produced nothing, and several print a success
message while doing it. Collected signatures:

| what happened | what it looked like |
|---|---|
| disk quota exhausted during a weight download | `OSError: [Errno 28]`, then a **truncated checkpoint** that failed as a corrupt zip on the next run |
| CUDA build vs GPU architecture mismatch | logged **`Seed N completed in 7.61s`** and exited 0, for a job that takes tens of minutes |
| JIT-compiled extension with no compiler in the env | `ModuleNotFoundError`, then `Ninja is required`, then no `nvcc` |
| interpreter missing a dependency | all 20 parallel processes died on import and the driver printed **`all 20 seeds complete`** with zero output files |
| cluster queue contention | 14 h `PENDING`, `Elapsed 00:00:00`, exit code `0:0` |

**Count artefacts, never trust exit codes.** Count structures per case against the expected sample
count and name any case that is short. End every driver script with a count of what it produced, not
with `echo complete`. A gate that refused to submit 99 follow-on array tasks unless the first task
had produced ≥25 structures **blocked five consecutive silent failures**; without it, one of them
would have produced 100 empty case directories that looked exactly like a finished batch.

**And treat an implausibly fast success as an error.** Budget from a measured case: a ~400-residue
receptor plus a 60-110 residue binder at 5 seeds × 5 samples runs ~2.3 min per case on one
current-generation card. The first case with a new chain length pays an extra compilation cost, so a
*jump* in per-case time usually means a recompile and not a hang - check the process is alive and
the log still writing before intervening.

### 9.4 Infrastructure papercuts worth pre-empting

- **Resolve every path to absolute in queue scripts.** A runner doing `CASES=${5:-$BD/cases.txt}`
  and then `cd "$BD"` resolves a relative cases path against the wrong directory and runs nothing,
  while still printing `queue finished`.
- **Write the batch manifest at build time, in the script that writes the inputs.** One batch
  finished 500 structures then died on the opening line of scoring with
  `FileNotFoundError: manifest.json`, after all the compute was paid for.
- **Order `cases.txt` controls-first.** The queue is resumable per case, so an interruption then
  leaves a prefix that still has its control block and is still evidence. Controls last means an
  interrupted batch is worth nothing.
- **Never give a case name a dot.** Some runners name the output record after the filename stem,
  which strips only the last suffix, so `design.A.B` silently becomes `design.A`.
- **Chain batches on a count-based waiter, not a process-based one.** A process waiter fired in the
  gap between two queues and took a GPU that was still in use. And **check the waiter's glob against
  a real filename**: one polled for `*_model_*.cif` against outputs named
  `<case>_seed-11_sample-0_model.cif`, matched nothing, printed `0/2800`, and would have waited
  forever.
- **Pin GPUs with `CUDA_VISIBLE_DEVICES`, not with the tool's own device flag.** One oracle ignores
  its `--gpu_device` for allocation and preallocates on **every visible GPU**; a lane died with
  `RESOURCE_EXHAUSTED` trying to allocate 54 MiB while its own card held 117 MiB of 24 GB, because
  it was fighting another user's job on a different card.
- **On a shared machine, never run a broad process kill.** Take PIDs from `ps`, scope to your own
  working directory, and confirm the other user's process count and GPU memory afterwards.
- **Pre-compute the receptor MSA once and inject it per case.** No MSA server is contacted at fold
  time, so the batch does not queue behind a one-ticket-at-a-time alignment service and is
  reproducible across machines.
- **Assert, in the builder, every documented invariant your own prose asserts.** Three lines each,
  run on every rebuild, failing before any GPU time. Also assert that permutations are
  composition-matched to their parents, and that a sequence copied into several files is
  byte-identical in all of them - one design existed in seven files, so a single stale copy decides
  what gets folded. Once a check has fired and the fact is understood, downgrade it to a printed
  report so a fact about the project cannot block a batch.

### 9.5 Your own gates will catch you, which is the point

An evidence attachment **failed the campaign's own prompt-injection scanner with 42 findings**,
because verdict words like `PASS` and `RANK` sitting in a submitted cell read as instructions to
whatever parses the file. The fix was a non-lexicon vocabulary (`no finding`, `reject`) and running
the scanner on the evidence file and not only on the submission. A panel generator later reproduced
the same failure, and a failure-taxonomy document failed the prose scanner on a row that **quoted
one of the imperatives it was documenting**. Run the gates from where a reviewer would run them: a
validator that passed in the working tree reported **20 blocking failures** from inside the handoff
package, because the design index had not been shipped with it.

---

## 10. Reporting, and the three sentences that keep a campaign honest

A result note for a batch states, in this order: what was folded and how many structures came back;
the control block with its numbers; the window those controls produced; the design rates with their
permutation baselines; **and then what the batch does not establish.** Quote the batch directory for
every number so a reader can re-derive it.

That last section is the one that does the work. Write it as explicit headings - *Follows*, *Does
not follow* - because a limitation in a paragraph gets summarised away and a limitation under a
heading does not.

Three claims to check before submitting anything:

1. **"Design A is better than design B."** Almost certainly unsupported. Check it with a Fisher
   exact test on the two rates, then check it again at a second seed set. In that campaign the only
   significant design-to-design separation ever measured was a **negative** one: an engineered
   variant that was worse than its own parent at p = 0.0072.
2. **"Our top design separates from the field."** Check what it was compared *against*. A p-value
   against "the row the ordering happened to put second" is a statement about your ordering, not
   about your design. Same comparison against the field: p = 0.32 and p = 0.20.
3. **"These designs reproduce the native geometry."** If the figure was made by selecting poses on
   that geometry, this is §8.

### Three patterns, if you remember nothing else

1. **Exit codes are not results; artefact counts are.**
2. **The error that survives is the one where every individual step is true** - so get an outside
   reader, and never let the proposer validate the proposal.
3. **Distinguish the precision of a measurement from the position of what it measures.** That
   campaign's two biggest retractions were the same error twice: once on a shuffle p-value (§7) and
   once on a pose rate (§5). Both times a quantity's run-to-run scatter was read as information
   about the object rather than about the estimator.

---

## 11. Honest limitations of everything above

- **One campaign, one target class.** The constants are measurements, not laws. ρ = 0.673 is the
  reliability of a 25-pose rate on one target with an extremum-edge band, not a property of
  co-folding.
- **Nothing here is wet-lab validated.** Every number is a computational agreement statistic. A
  design that beats its own composition on two oracles at two seed sets has cleared a bar that
  retired seven other metrics - and has not been shown to bind anything.
- **Seed replication does not address oracle bias.** If one oracle measured your whole panel, your
  panel has one opinion, however many seeds it has.
- **A mechanism band tests the mechanism you chose to encode.** In that campaign a contact made by
  all four real control peptides in 25/25 poses - including by a competitive *antagonist*, which is
  why it could not be promoted to a gate - was absent in the designs: **21 of 29 sat further from it
  than a composition-matched non-binder**, and the one attempt to engineer it cost the leader half
  its rate. That gap was documented and shipped unresolved. It is the most likely reason those
  designs would fail in a lab, and no amount of band-rate evidence addresses it.
