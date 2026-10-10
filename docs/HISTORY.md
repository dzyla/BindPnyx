# What the first week of work taught (a digest of the project's early history)

This repository's public history starts later than the work. Before it, one author and an AI assistant spent seven days (121 commits) building the scoring layer,
the target tooling and the guards that the rest of this repository assumes. That history is retired; this page keeps what it established, in the order it was
learned, with the claims that did **not** survive re-measurement marked as such. For the working rules and their evidence, read `docs/DESIGN_PATH.md`; for the
operating guardrails, `AGENTS.md`.

## The phases

1. **A calibrated Boltz-2 scorer, specified before it was built.** Ranking key = interface ipSAE (minimum direction). The gate was named provisional because the
   labelled set it came from was not available to validate against, and early stopping was kept off ("the gate is descriptive, not a controller"). Boltz is invoked
   once per seed over a directory of inputs, never once per design. A **single final common batch** re-scores the pooled shortlist, and ranking refuses mixed batches.
2. **Making it fast without moving the numbers.** GPU utilisation on a full run was about 14%: the idle time is model load and single-threaded MSA featurisation. Running
   **seeds** concurrently (never splitting designs across processes) gave 1.33x with bit-identical scores; three workers was the plateau; the fixed per-invocation cost
   means bigger batches beat more workers. Boltz's own dataloader workers (no speedup) and its kernel path (1.06x) were rejected because each changed all 42 compared metric values
   (largest differences 3.88 and 5.26, in the units of the metric that moved). A throughput gain that moves scores is not a gain.
3. **Hotspots that silently did nothing.** Three input formats meant three behaviours; in two of them a mistyped or mismapped hotspot matched nothing and the run proceeded
   unconditioned on it. Every path now validates the requested residues against the structure the model sees, and a check asserts each resolved hotspot exists in the exported
   complex before any contact is counted. Designs formed large interfaces that missed the requested residues, and nothing in selection noticed, because **interface quality and
   epitope correctness are different axes**. Residue specs carry the residue letter and are identity-checked (`"Y56"`, never a bare index).
4. **The app owns its environments.** A campaign died after its GPU work on a hardcoded path into someone else's environment; with `PYTHONPATH=<repo>` an extracted source
   tree at the repo root masked the installed package and the error named the package, not the shadowing. Tools are now resolved from an explicit setting, then the project's own
   environments, then `PATH`, and are never borrowed silently.
5. **Input integrity.** Boltz discards a whole mismatched MSA and prints one line; one substitution cost 3,109 sequences. Every cached MSA is now validated against the sequence
   at the single chokepoint and before any GPU work, Boltz's own output is kept (it was being discarded whenever anything went to stderr), a discarded MSA returns a status instead of a
   real-looking score, a campaign that scored nothing exits non-zero, and a changed configuration cannot reuse old predictions (inputs are digested). Later, **one NUL byte** in a
   4,024-sequence alignment silently cost a 20-backbone campaign (every example skipped, exit code 0); alignment parseability is now checked, not just the query line.
6. **Is the oracle any good?** Designs against composition-preserving scrambles folded in the same batch: AUC 1.000 on one target, a no-signal floor near 0.05, within-batch rank
   agreement across compositions (mean Spearman +0.78), signal-to-noise about 1.9, so **differences below about 0.03 ipSAE are noise**. On another construct the campaign's own designs
   scored at scramble level (AUC 0.48): selection and gate tuning there had been operating on noise. Raw scores are not comparable across batches; the AUC against each run's own
   scrambles is. The noise floor differs per target (0.25 on one, 0.09 on another), so the control runs per target.
7. **Where does the intended epitope get lost?** Measured stage by stage: for one hard residue, 47 of 768 generated backbones reached it at 10 A, 6 at the 7 A screening cutoff, and the
   screening step (which ranked on the *total* number of hotspots engaged) selected it zero times of 32; the fold then lost a quarter of what survived. Needing 32 eligible backbones for that
   residue meant generating about 4,000. Enforcing the epitope (fail closed) over seven campaigns scored 78 designs, found 6 eligible at screening and **0 confirmed**: backbone
   proximity predicted nothing (0 of 30 near, 6 contacts after folding), one seed does not confirm, and high scores came with none of the required residues engaged.
8. **Choose the site from the structure.** For one target the epitope already in the repository was off-site (it required a residue that buries 7 A^2 when the natural partner binds).
   Nominating the site from the partner's actual contacts raised engagement from 7.7% to 85% and exported 13 designs where every earlier campaign exported none; five of them cleared the
   whole scramble distribution and engaged the residue, a combination not seen before.
9. **Second judge and multi-chain targets.** OpenFold3 became a selectable second judge. On a homo-oligomer the single-chain workflow silently scored target against target, so it now raises on
   more than two chains and the multi-chain path lives in `phbind/`.

## Claims the history withdrew (do not cite these)

* "The wrapper exits 0 on a usage error": it did not; the 0 was a trailing `tail` in the author's own command. The real defect was validation running after a slow environment probe.
* "The provisional gate does not drive early stopping": under shipped defaults it did.
* "A helper blamed the MSA for a mismatch": the shard was the outlier; the remedy is a matching structure, not a regenerated MSA.
* **"Batch composition moves a score by ~0.4":** the figure did not reproduce (re-measured: SD 0.02 across batch sizes 1-32; alone vs among 31 others, 0.006). Batch discipline stays because
  comparing within a batch is free, but the effect is small, and design sharding was removed for its own, sounder reason (an equivalence check failed).
* "A lost MSA lowers a score": measured the other way, **+0.16 ipSAE in 4 of 4 designs**, the dangerous direction against a fixed gate.
* "The fused-LayerNorm build has no fallback": it always fails here and falls back to the torch implementation with zero difference (the arch patch is a no-op).
* Several tests had been proving nothing (eight fixtures that passed regardless; a test that reimplemented the rule it tested). The practice since: delete the guard and confirm the test fails.

## Design principles that came out of it

* A metric that cannot say "unmeasured" will say "fine": `not_applicable` once *permitted* eligibility, so a declared policy was enforced on nothing; a dead species was ranked on the survivors.
* Fail closed, and say why: a run that ships nothing names the cause; an absent verdict under a declared policy is a transport failure, not an absence.
* Validate at the chokepoint, before the GPU. A failed check should cost a second, not a campaign.
* Record provenance as an artifact every stage reads (numbering map, policy digest, configuration digest, the target record with the commit it was built at), and refuse stale reuse.
* Measure the stage where the loss happens before choosing the investment: generation supply, screening, the fold and enforcement are four different fixes.
* Put the retraction where the claim was made, in the same change.

Detailed documents, the scripts, the raw run outputs and the full commit narrative are kept outside the repository (they name machines and local paths); ask whoever holds the private notes folder.
