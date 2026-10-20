---
name: cofold-band-scoring
description: Score peptide/miniprotein designs against a receptor with co-folding oracles (AlphaFold 3, Boltz-2, Protenix) using a mechanism band derived from each batch's own controls, and rank on the band-holding RATE minus a composition-matched permutation baseline. Use when scoring or re-scoring a design batch, when adding a second or third oracle, when a batch's numbers look too good or too uniform, when picking a representative pose for interface analysis, or when asked whether two batches' numbers can be compared. Carries the traps this campaign actually hit - a silent pose-to-file mismatch, a saturated band that ranks nothing, a missing manifest that killed scoring after 500 structures were already paid for, and a relative cases path that sent a queue to the wrong directory.
metadata:
  authored_via: "claude-code"
  measured_on: "one peptide-agonist design campaign against a class B GPCR, 2026-10, three co-folding oracles, ~40 scored batches"
  scope: "method only; candidate sequences, per-run score logs and operator paths stay in the campaign workspace"
---

# Scoring a design batch on a mechanism band

The unit of evidence is a **batch**: a set of designs co-folded with the receptor under identical
conditions, together with a fixed control block. Nothing here is comparable between batches.

## Runnable blocks

`blocks/` holds this skill's rules as commands that run. Each one checks its own preconditions and
carries, at the top of the file, the failure it exists to prevent. Nothing in them is specific to
one target: the receptor, the controls and the seeds are arguments.

| # | block | one line |
|---|---|---|
| 0 | `blocks/preflight.sh` | everything cheap to check before a GPU is committed |
| 1 | `blocks/build_cofold_batch.py` | sequences + controls + seeds -> a complete, self-describing batch |
| 2 | `blocks/run_queue.sh` | fold every case, resumable, artefacts counted not inferred |
| 3 | `blocks/score_batch.sh` | shim -> per-pose terms -> in-batch band -> control block -> margins |
| 4 | `blocks/band_diff.py` | compare two scored batches' BANDS before comparing their designs |
| 5 | `blocks/group_report.py` | rates by the role each case was built for |
| 6 | `blocks/replicate_at_new_seed.sh` | the same inputs at an unused seed, which is the only real replicate |
| 7 | `blocks/extend_with_permutations.py` | turn a RATE into a MARGIN by appending a design's permutation triple to its own scored batch |

`blocks/README.md` gives the argument shapes, says why the order is the method, and marks the one
block that is a driver around a per-target scorer you must supply (block 3) together with the
three-function interface it expects. Block 1 refuses
a batch with fewer than three positive controls, a permutation that is not composition-matched, or
a case name containing a dot; block 3 refuses to score a short batch; block 6 refuses a seed the
source batch already used.

## The rules, in the order they bite

**1. The band comes from the batch's own controls, never from a note.** Compute the acceptance
window for each mechanism term (anchor-contact distance, conserved-core contact count, insertion
depth) from the agonist controls folded *in this batch*. Co-folding output depends on batch
composition, so a window copied from an earlier run is a cross-batch comparison wearing a
within-batch disguise. Measured example: the same anchor ceiling came out 3.01 A with two agonist
controls in the batch and 3.38-3.42 A with three.

**2. If the controls do not read as expected, the batch is not evidence.** Apply the gates to the
negative controls *before* looking at any design. The block that works: three real agonists (expect
at or near 25/25), one competitive antagonist that keeps affinity and loses efficacy (expect 0/25),
one composition-matched permutation of an agonist (expect 0/25). An antagonist that passes or an
agonist that fails means the batch is dropped, not explained.

**Know which half of that block is a test.** If the band's edges are extrema over the positive
controls' pooled poses - `max(anchor)`, `min(core)`, `[min(depth), max(depth)]`, which is the usual
construction - then every positive pose lies inside every edge and **the positives read 25/25 by
construction and cannot fail**. Verified exactly, on every term, in four batches of this campaign,
which is why its positives came back at exactly 25/25 every time in three different oracles. The
certifying work is done entirely by the negative controls, which are not used to derive the band.
Say so when reporting, rather than offering the positives as evidence the batch is sound. The same
construction puts each edge at the mercy of one pose in 75: measured here, one oracle's anchor
ceiling moved 1.91 A between two of its own seeds while the 95th percentile of the same poses moved
0.20 A, with the extremum coming from a different control each time. Prefer a quantile edge in a
new campaign - the cost, that positives then read 23-24/25 instead of 25/25, is the point - and
never re-derive the edge definition after the candidates are known.

**3. Rank on the rate, not on the worst pose.** An extremum over 25 draws is not a property of a
design: the same design re-run at the same sample count moved its anchor floor 3.31 -> 3.77 A and
flipped pass to fail. The fraction of poses that satisfy **the whole band at once** converges
instead of wandering. Score `rate(design) - rate(its own composition-matched permutation, same
batch)`, because the permutation carries the same residues and the same length, so whatever the
model's prior contributes to threading that composition into the site it contributes to both.
Three permutations per design, not one - a single permutation's rate has the same sampling noise as
the design's.

**4. Band WIDTH is a property of the oracle, so a higher rate is not a better design.** Measured
anchor windows for the same target and the same control block:

| oracle | anchor window | width |
|---|---|---|
| AlphaFold 3 | 2.40-2.74 A | 0.34 A |
| Boltz-2 | 2.26-3.42 A | 1.16 A |
| Protenix | 1.96-3.41 A | 1.45 A |

The same design scored 12/25 on AF3 and 20/25 on Protenix. That is not "harder there, easier here":
the wide window admits poses the narrow one rejects outright. Use the narrow oracle to rank and the
wide ones to confirm, state which is which, and expect the orders to disagree - in this campaign the
design ranked first on AF3 was ranked last on Protenix.

**4a. The width is a property of the oracle AND the seed, so a replicate must report its band, not
only its rates.** The same 17 cases, the same controls, the same precomputed MSAs, one oracle, two
seeds: the anchor window came out 1.96-3.41 A at one seed and 2.03-5.32 A at the other, because one
agonist control's worst pose moved. Every rate rose with it and two composition-matched permutations
of one design went from 0/25 and 1/25 to 20/25 and 24/25 - that design's margin nearly halved
without the design changing. The clean demonstration, worth copying as a diagnostic: take one
permutation whose per-pose anchor distances barely moved between the two batches (3.52-4.63 A
against 3.54-4.91 A) and show it holding the anchor term 0/25 in one and 25/25 in the other. The
permutation did not improve; the ceiling moved 1.9 A. So when a replicate disagrees, diff the bands
before concluding anything about the designs, and treat a margin from one seed as evidence of SIGN,
never as a tiebreak between two designs.

**4b. A re-fold at the same seeds is not a replicate.** These oracles are deterministic given seed
and input, so re-running a design with the seed set it already used reproduces its rate exactly and
the band bit-identically. That is a pipeline check worth having and it is not a second draw; a batch
built to replicate a result must use seeds the campaign has not used. Measured: a 26-case batch
re-folded two leaders at the original seeds and returned their earlier rates to the structure, which
proved only that the scorer is deterministic. Equally, an absence of measurement is not a low score
— when a ranking's tiebreaker is "rows with no measurement sort below rows that have one", the
cheapest decisive experiment in the campaign is usually to measure them.

**5. Fix the ranking rule before reading the rows, and do not re-open it for a late oracle.** Write
the rule down (primary key, tiebreakers) and keep it when new numbers arrive, especially when the
late arrival favours a candidate you have reason to want. A metric that appears after the candidates
are known does not get to reorder them; record it as an extra column in the evidence file.

## A saturated band ranks nothing - say so instead of reporting an order

A batch where almost every row sits at the maximum is a measurement of the instrument, not of the
designs. Real case: 15 sequences on the wide oracle came back 11 at 25/25 and the rest at 23-24/25,
**parents included**. The correct report is "no ranking is possible in this batch", plus whatever
the batch does establish - there, that no variant broke the mechanism, which is a real and useful
negative. Then re-measure on the narrow oracle. The failure mode to avoid is ranking inside the
noise of a saturated band and calling the top row a winner.

## Picking a representative pose for interface analysis

Interface tools (PISA-type, contact maps, renders) take one structure, so a pose has to be chosen,
and this is where a silent error is easiest:

- Choose **among the in-band poses**, and mark the case `OUT_OF_BAND` if there are none. An
  interface picture of a pose that never satisfies the band is informative but must not be read as
  a picture of a working pose.
- Among those, take the pose closest to the **median of the batch's own in-band agonist-control
  poses**. Do not hardcode that target: the same three controls gave 2.77 A on one oracle and
  2.64 A on another, so a constant lifted from one batch quietly becomes a cross-batch reference.
- **Re-measure the chosen quantity on the file you write, and refuse to write if it disagrees with
  the scored value** (tolerance ~0.015 A). This is the only check that catches a wrong
  pose-to-file mapping, and nothing downstream is trustworthy without it.

### The pose-to-file trap, concretely

Scorers usually enumerate poses as `sorted(glob("*_model_*.cif"))`, which is **lexicographic**:
`model_0, model_1, model_10, ..., model_2, model_20`. `per_sample[i]` means the i-th file in *that*
order. Globbing the raw prediction directories instead (`seed-*_sample-*/`) gives a different order,
and every design's analysed pose came out 0.04-0.22 A off while the control - whose 25 poses are
nearly identical - matched and hid the error for a full analysis cycle. Reproduce the scorer's sort
exactly, from the same directory the scorer read.

## Rates for interface features, too

If an interface feature is going to be claimed as a property of a design, measure it over the whole
pose ensemble, not on the representative pose. Converting one such feature from a single-pose
observation to a 25-pose rate changed the conclusion from "the designs mostly miss this contact" to
"the designs make it in 0 of 25 poses and all four real peptides make it in 25 of 25", which is a
different and much stronger statement. Interface tooling is cheap enough for this (~0.6 s per
complex, so 500 poses is a few minutes on 8 workers).

## Before you spend GPU time

- **Write the batch manifest at build time**, in the same script that writes the inputs. A batch
  finished 500 structures and then died on the opening line of scoring with
  `FileNotFoundError: manifest.json`, after all the compute was already paid for. Anything the
  scorer reads should be produced by the builder.
- **Resolve every path to absolute in queue scripts.** A runner that does `CASES=${5:-$BD/cases.txt}`
  and then `cd "$BD"` resolves a relative cases path against the wrong directory and runs nothing.
  One queue script in the same campaign already had the `readlink -f` fix and the other never got
  it; copy the fix, do not re-learn it.
- **Count artefacts, do not trust exit codes.** These tools exit 0 on inputs that produced nothing.
  Count the structures per case against the expected sample count and name any case that is short.
- **Never give a case name a dot.** Some runners name the output record after the filename stem,
  which strips only the last suffix, so `design.A.B` silently becomes `design.A`.
- **Chain batches on a count-based waiter, not a process-based one.** A process waiter fired in the
  gap between two queues and took a GPU that was still in use; a waiter that polls for the expected
  artefact count does not.
- **Assert, in the builder, every documented invariant the batch relies on.** Not the ones you can
  look up — the ones the project's own prose asserts. A build script given three lines to check
  that all 20 panel rows shared a committed 34-residue N-terminal arm aborted on row 5, which
  differs at six positions: the claim had been true of an earlier panel, was carried forward in
  prose, and had been used in two documents to justify a decision. Nobody had checked it because it
  read as settled. Also assert that permutation baselines are composition-matched to their parents,
  and that a sequence copied into several files is byte-identical in all of them — here one design
  existed in seven files, so a single stale copy decides what gets folded. Each check costs three
  lines, runs on every rebuild, and fails before any GPU time is spent. Once a check has fired and
  the fact is understood, downgrade it from an assert to a printed report so a fact about the
  project cannot block a batch.
- **Order `cases.txt` controls-first.** The queue is resumable per case, so an interruption then
  leaves a prefix that still has its control block and is still evidence. Controls last means an
  interrupted batch is worth nothing.

## Running the oracles, practical notes

- **Pre-compute the receptor MSA once and inject it per case.** With the receptor alignment passed
  as a file path and the binder given a single-sequence alignment, no MSA server is contacted at
  fold time, so the batch does not queue behind a one-ticket-at-a-time alignment service and is
  reproducible across machines. Every batch in this campaign used the same receptor alignment and a
  single-sequence binder.
- **Budget from a measured case, not from a guess.** A 409-residue receptor plus a 60-110 residue
  binder, 5 seeds x 5 samples, runs about 2.3 minutes per case on one current-generation card, so a
  22-case batch is under an hour. The first case of a batch with a new chain length pays an extra
  compilation cost; a jump in the per-case time usually means a recompile, not a hang - check that
  the process is alive and the log is still writing before intervening.
- **Keep the queue resumable.** A runner that skips any case already holding the expected number of
  structures can be re-launched after an interruption without re-folding what is done.

## Descriptors are not gates, and the controls are how you find out

Every interface descriptor tried in this campaign - buried surface area, solvation energy, hydrogen
bonds, salt bridges, contact counts, epitope recovery, domain-profile similarity - failed the same
test: it separated the competitive antagonist correctly and placed the **inert permutation** level
with or above the real agonist. Buried area was the worst: the permutation buries more surface than
the hormone in 6 of 6 batches, and among real peptides the tightest binder buries the least. Before
using any new quantity to select, score the full control block with it and check the permutation
specifically. A quantity that cannot separate a non-binder from the hormone is a description; report
it as one.

## Write the recurring computations down once

Two quantities in this workflow were recomputed by hand in three separate batches before anyone
made them scripts, and a hand-recomputation is where an arithmetic slip enters a result note:

- **the permutation margin** - `rate(design) - mean over its permutations of rate(permutation)`,
  with the per-permutation rates printed alongside, because a design whose three permutations read
  0, 0, 12 is saying something a mean hides. Take the band from the batch's own file and the
  membership test from the same module the ranker uses, so there is one implementation of the band
  and not two.
- **the interface-feature rate** over the whole pose ensemble, not the representative pose.

Validate a new implementation of either against a batch whose numbers you already have by hand
before using it on a new one.

## Reporting

A result note for a batch states, in this order: what was folded and how many structures came back;
the control block with its numbers; the band the controls produced; the design rates with their
permutation baselines; and then what the batch does **not** establish. The last part is the one that
keeps a campaign honest - a batch that cannot rank, an oracle whose band is too wide, a descriptor
that describes. Quote the batch directory for every number so a reader can re-derive it.

Keep candidate sequences, per-run score logs and operator paths in the campaign directory. If a
lesson from them is worth generalising into a public repository, generalise the *lesson* and leave
the data behind - and get the operator's explicit go before anything is pushed outward.
