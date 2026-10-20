# Runnable blocks

Every one of these was a thing this campaign did more than twice, by hand, slightly differently
each time. Each block is a single runnable command with its arguments named, its preconditions
checked, and the failure it exists to prevent written at the top of the file. They are
campaign-agnostic: nothing here contains a design sequence, a receptor, or a result. The target
and the controls are arguments.

Run them in this order. Later blocks refuse to run on the output of an earlier one that did not
finish.

## What is in this repository, and the one thing that is not

Blocks **0, 1, 4, 5, 6 and 7 are self-contained** and run against any target: they read and write
the batch layout described below and nothing else.

**Block 3 (`score_batch.sh`) is a thin driver around a per-target scorer that is NOT shipped here.**
It shells out to three scripts under `$CAMPAIGN/scripts/` - a shim that extracts per-pose geometric
terms from the predicted structures, a ranker that derives the band from the batch's own positive
controls and tests membership, and a margin calculator. Those are target-specific by nature: they
name the anchor residue, the conserved core and the depth frame for one receptor, and generalising
them without a second target to test against would produce a worse abstraction than an honest
boundary. So the boundary is documented instead.

**To use block 3 on your own target, supply these three and keep the interface:**

| you provide | contract |
|---|---|
| `shim` | reads the batch's prediction outputs, writes `<batch>_shim/` with one record per case holding a `per_sample` list; enumerate poses with the **same sort the rest of your tooling uses** (see the pose-to-file trap in `SKILL.md`) |
| `band + membership` | derives each term's acceptance window from **this batch's positive controls only**, writes `joint_bands.json`, and exposes one membership test used by every consumer - one implementation of the band, never two |
| `margin` | `rate(design) - mean over its permutations of rate(permutation)`, printing the per-permutation rates beside the mean |

### Naming the band's terms for your target

`band_keys.py` is the one place that knows what a band file's terms are called, and every block
reads the band through it. The generic names are `anchor_subfloor` / `anchor_max` / `core_min` /
`depth_band`, with the per-pose anchor field called `anchor`. If your scorer names them after the
residue it measures - the natural thing to do - point the blocks at those names instead of renaming
your data:

    export BAND_KEY_ANCHOR_LO=<your key>
    export BAND_KEY_ANCHOR_HI=<your key>
    export BAND_KEY_ANCHOR_PER_POSE=<your per-pose field>

A band file whose keys match nothing raises an error naming the candidates it looked for, rather
than a bare `KeyError` on a file that is perfectly valid for a different target. `band_keys.holds()`
is the single membership test; use it rather than re-implementing the conjunction, because a pose
satisfying two of three terms is not a hit and per-term scoring is how a design with no good pose
comes to look acceptable.

`group_report.py` and `band_diff.py` read `joint_bands.json` and the shim records, so once your
scorer writes those two things in this shape, blocks 4 and 5 work unchanged. `perm_margin.py` in
this campaign was 122 lines; the shim and ranker were larger because they encode the mechanism.

The useful content of block 3 is therefore the **order** it enforces - shim, then band from
controls, then check the negative controls, then look at designs - which is the method, and which
`score_batch.sh` makes executable even where its callees differ per target.

| # | block | one line | typical cost |
|---|---|---|---|
| 0 | `preflight.sh` | everything that is cheap to check before a GPU is committed | seconds |
| 1 | `build_cofold_batch.py` | sequences + controls + seeds -> a complete, self-describing batch | seconds |
| 2 | `run_queue.sh` | fold every case, resumable, artefacts counted not inferred | hours |
| 3 | `score_batch.sh` | shim -> per-pose terms -> in-batch band -> control block -> margins | minutes |
| 4 | `band_diff.py` | compare two scored batches before comparing their designs | seconds |
| 5 | `group_report.py` | rates by the role each case was built for, next to any prior rate | seconds |
| 6 | `replicate_at_new_seed.sh` | the same inputs at an unused seed, which is the only real replicate | hours |
| 7 | `extend_with_permutations.py` | give a scored design its composition-matched baseline, band unchanged | ~3 cases |

## The order is the method, not a convenience

0 before 1 because a missing manifest or a case name with a dot costs a whole batch, and both are
free to catch. 3 before anything interpretive because a batch whose controls misread is not
evidence and must be dropped rather than explained. **4 before any cross-batch claim**: two
batches of the same oracle can derive different bands, and comparing their rates without diffing
their bands first is how this campaign nearly concluded something about designs that was a fact
about a window. 6 last because a replicate at a seed you already used reproduces arithmetic, not
a measurement.

## What each block takes

```bash
B=runs/x99_example                 # batch directory, created by block 1

bash blocks/preflight.sh $B                                   # 0
python blocks/build_cofold_batch.py --out $B \                # 1
    --designs designs.csv --controls controls.csv \
    --receptor receptor.json --receptor-msa rec.a3m --seeds 11,12,13,14,15
GPU=0 bash blocks/run_queue.sh $B af3                         # 2  (or `protenix <seed>`)
bash blocks/score_batch.sh $B af3                             # 3
python blocks/band_diff.py ${B}_shim runs/x98_earlier_shim    # 4
python blocks/group_report.py --run ${B}_shim --batch $B      # 5
bash blocks/replicate_at_new_seed.sh $B 202 2                 # 6
python blocks/extend_with_permutations.py $B \            # 7
    --designs sc_example --seed-base 4400000
```

`designs.csv` and `controls.csv` are two columns, `name,sequence`. The control file is separate
from the design file on purpose: block 1 refuses to build a batch whose controls do not include at
least three positives and one negative, because that is the block that makes block 3 able to say
whether the batch is evidence.

## The five failures these blocks exist to prevent

1. **A batch that cannot be scored after it is paid for.** 500 structures finished, then scoring
   died on a missing `manifest.json` the builder never wrote. Block 1 writes everything the scorer
   reads, and block 0 checks it is there.
2. **A rate read against another batch's window.** Block 3 derives the band from the batch's own
   positive controls and records it; block 4 turns the comparison of two bands into a deliberate
   step with its own output, rather than an unwritten habit.
3. **A per-pose join that silently pairs the wrong rows.** Pose enumeration is `sorted(glob(...))`,
   which is lexicographic: `model_0, model_1, model_10, ..., model_2`. Any second measurement to be
   joined per pose must be stored in that order, not in worker-completion order. Block 5 refuses a
   join on a pose-count mismatch.
4. **A replicate that is not one.** These oracles are deterministic given seed and input, so
   re-running at the same seeds reproduces the result exactly and proves only that the pipeline is
   deterministic. Block 6 takes the new seed as a required argument and refuses to reuse one the
   batch already carries.
5. **An invariant the documents assert and nobody checks.** Block 1 takes `--assert-prefix` and
   reports every case that does not carry it. Three documents in this campaign said all 20 rows
   shared a fixed N-terminal arm; nineteen did.

## Conventions every block follows

- **Count artefacts, never trust an exit code.** These tools exit 0 having written nothing.
- **Absolutise paths before any `cd`.** A relative cases path after a `cd` runs zero cases and
  still prints a cheerful finish line.
- **Controls first in `cases.txt`.** The queue is resumable per case, so an interrupted run then
  leaves a prefix that still has its control block and is still evidence.
- **Never `rm -rf` a path built from a possibly-empty variable.** An empty case name turns
  `rm -rf out/$c` into `rm -rf out/`; this campaign lost 212 completed cases that way.
- **Stdout is the report.** Blocks print the table they computed, and write the machine-readable
  copy next to the batch.
