# Round report: designing pH-switchable TNF-alpha binders with PXDesign + SolubleMPNN (phbind line)

*Written 2026-10-08 00:22 MDT. Competition: Adaptyv x Anthropic 2026 Challenge 2 (human TNF-alpha homotrimer; hold at pH 7.4, release at pH 6.0; objectives in order: pH-dependent binding, mouse cross-reactivity, human affinity). Deadline Mon 12 Oct 05:59 MDT; novelty queue to the user by Sat 11 Oct 18:00 MDT. Nobody has wet-lab tested a design from this pipeline: every number below is a model prediction, and none is an affinity.*

## 1. Methods (what was actually done)

**1.1 Target contract.** The target is the intact 471-residue human trimer (3 x 157; mouse 3 x 156 = 468). Generation uses the epitope-neutral B+C dimer (304 residues) as the PXDesign shard; scoring always uses the intact trimer. A construct/structure discrepancy at residue 143 (Asp in the construct, Leu in the structure) is reconciled in the shard, not ignored. Every target MSA has a NUL byte repaired first (a NUL silently makes Boltz-2 skip the input and exit 0). Guards raise instead of warn: residue counts, one 143 mismatch exactly, no NUL, hotspots not buried, dimer epitope-neutral (change in relative SASA below 1e-4).

**1.2 Generation.** PXDesign diffusion on the dimer shard with one of three hotspot strings (declared-8, a 6-residue subset, a 12-residue confirmed footprint), at lengths from 62 to 248 aa. It emits coordinates only; **sequences are assigned by SolubleMPNN, never plain ProteinMPNN** (plain ProteinMPNN did not give good binders; no Cys). Backbone count, not MPNN settings, moves the score distribution.

**1.3 Two-model gate.** Boltz-2 (1 seed, then more) and AlphaFold3 (5-sample mean, seed 1, target MSA injected, binder single-sequence, no templates) are the two judges, chosen as different models rather than seeds of one model. Score = ipSAE with the three target copies as one group, taking the **minimum** direction (the thresholds were fitted on min; max reads about +0.17 to +0.20 higher). Global ipTM is not used (0.86 on a negative control). Pass = both models >= 0.5. A failing negative-control carrier and passing carriers go in every Boltz batch. Boltz-2 is deterministic for a given (sequences, seed, batch composition and order, command line, GPU) and changes when any of them changes: the same batch run twice gave byte-identical output, a dataloader flag changed every prediction, and the same sequences inside a 22-design batch shifted by 0.02-0.09 (a failing control by 0.18). So designs are compared only within one batch, the exact command is saved with every run, and a rerun in a different batch counts as an independent draw while an identical rerun counts for nothing.

**1.4 Robustness.** A single-seed prescreen is a recall filter. Survivors get 3, then 5 seeds of Boltz-2 on **both species**; the human gate is mean >= 0.65 and worst seed >= 0.50, mouse mean >= 0.50 and worst >= 0.50. A measurement by one of the other agents showed a 3-seed worst is about 0.04 looser than a 5-seed worst and unstable by about 0.15, so 5 seeds is the standard for anything in the final list. Replicates in a differently composed batch are real independent draws; there is no run-to-run noise to average in an identical batch.

**1.5 Structural novelty.** The competition portal rejects designs resembling known structures, so each backbone is searched against the PDB with Foldseek (TM-align, 70% query coverage). A strict hit is TM >= 0.8; the query-normalised TM (qtm) is the primary column, the alignment-normalised one (alntm) is reported as the adverse case. Sequence-level novelty is a non-issue (one of 208 designs had an mmseqs hit). Novelty is screened for backbones only, before the expensive oracles; the screening itself was done by the other two agents and returned as tables.

**1.6 Controls and what they showed.** Four solved TNF binders as positive controls: Boltz-2 recovers 1 of 4 and AF3 0 of 4, so a low score from either is nearly uninformative and only a high score means anything. Residue-shuffled copies of eight passing designs score 0.00 on AF3, so the AF3 signal depends on the sequence, not on chain length. Hotspot-string choice made no measurable difference across four tests; backbone geometry did not predict the oracle (AUROC 0.60).

**1.7 pH.** The competition needs release at pH 6.0. The route that worked in principle is histidine substitution on a folded design (M3: His-His pair, CB-CB <= 8.0 A; M1: single His near a target cation; at most 2 to 3 His; re-run a His-to-carboxylate veto on the refolded child), scored by a multi-pose PROPKA paired delta (parent versus variant, same pose). A single pose has a spread of about 0.9 kcal/mol and is not evidence. Asking PXDesign for a histidine-enriched interface directly (an amino-acid bias) was tried and closed: 418 trajectories, none refolded. A protonation-aware design model (Proton-PottsMPNN) was then used as a generator with its objective sign flipped to the release direction: of 175 unique designs from 13 parents, 115 kept binding on a one-seed screen and ten finalists were confirmed on five machines; five passed human, mouse (two machines) and AF3, but over three poses none showed PROPKA release above the existing histidine-substituted designs and the redesigns of the best of them released less than it did. Its selective energy proved to be an objective, not evidence of release. pH was **not established** for any of the new long designs.

**1.8 Shared ranking.** One append-only evidence ledger (every number carries agent, seed count and note) plus a rule-based ranker that only reads it: tiers A to D and X, more seeds beat fewer then newest wins, a clean novelty margin sorts ahead of a thin one, a lineage cap, and a fixed first-20 rule. Agents add evidence; nobody edits the output.

**1.9 Compute and infrastructure.** One 32 GB workstation GPU for scoring; the other agents' 4-GPU workstation (one card per wave, one checkout per wave, enforced); an LSF cluster for 1-GPU generate-and-screen jobs. AF3 was run on the 96 GB card, the 32 GB card and a cluster A100 80 GB (one design gave 0.687 on the A100 and 0.692 on the 32 GB card, same input and seed; a second gave 0.619-0.642 over three runs on two machines); the 24 GB cards were not tried for it. Screening was sharded over a workstation, a 4-GPU server, an LSF cluster and a Slurm cluster. The software layer (config, scorer registry, wave orchestration, experiment log, ranker, agent playbook) is in the repository under `phbind/` and `docs/`.

## 2. What happened (aggregate results; no design identities)

- Short (62-112 aa) helical designs scored well but **most failed the novelty screen**: 18 of 21 two-model passers had a PDB hit. Novelty, not affinity, was the binding constraint.
- Novelty-clean fraction rises with length: 21% at 62-112 aa, 60% at 120, 88% at 144, 99-100% at 180-184, 98% at 200, 91% at 248 (a rise then a mild fall, not a ramp).
- Boltz-2 single-seed pass rate is about 4-5% per backbone with no demonstrated length effect between 120 and 248 aa (1,231 designs scored in the first pool; the rate was similar across three compute lanes).
- Of 51 survivors sent to AF3, 19 passed both models. Three-seed Boltz-2 kept 8; mouse and five seeds left **3 designs that hold on both species**. The single-seed rank barely predicted who survived (the eight survivors ranked 1, 6, 8, 11, 13, 14, 15, 16 of 19).
- Mean single-seed to 3-seed shrinkage over the 19 was only 0.024; the damage was to individual designs, not the average.

## 3. Detours, errors and corrections (kept because they changed decisions)

**Process and bookkeeping**
- Repeated `find /` calls hung the shell; replaced by restricted paths.
- A fast-screen recall figure was inflated by tie ordering (positives sorted first among tied zeros). Recomputed with random tie-breaking: the fast Protenix variant was a complete loss on the trimer and was dropped.
- A script labelled its threshold 0.45 while using 0.35 after a config change; I reported a comparison at the wrong threshold and recomputed it.
- I substituted one model for a failed one without asking; the user corrected it and the gate became Boltz-2 plus AF3.
- A guard in an auto-chain required 40 finished runs when 36 were correct, which idled the GPU for about 45 minutes. A wait loop watched the wrong PID. A sync loop copied only files that already existed locally and missed new modules.
- I reported "14 of 20 pass" from a 3-seed batch while another agent's figure was 5-seed 8 of 20; reconciled in the shared file.
- A remark from the user about a dead agent was applied to the wrong agent without checking the file; the agent was alive, and a correction was posted within the hour.
- Shared-file note numbers collided several times (two 47s, 51s, 59s, 25s, 33s); my notes were renumbered or I used the next free number.

**Claims I made and withdrew this round**
- "Length lead above about 190 aa" from one lucky batch (6 of 72) was retracted after the next batch (1 of 72); pooled it was not significant (p = 0.10).
- "Large winner's curse" in the 3-seed table was wrong; the measured average shrinkage was 0.024. What was true was that single-seed rank did not predict survival.
- A "3 of 11 will not survive a second seed" figure had no measurement behind it and was removed within minutes.
- I accepted another agent's "run-to-run nondeterminism" explanation and reinterpreted my own control readings as noise; both were wrong. A direct test showed identical batches are byte-identical, and the control readings followed batch composition. The agent who proposed it retracted it three times and reached the same conclusion independently.
- A design I called a provisional dual-species candidate did not replicate in an independent 5-seed run and was dropped. Another looked dual-species at 3 seeds and failed mouse at 5 (worst seed 0.57 to 0.24).
- I told the other agents three waves were running; two had silently stalled for about 35 minutes (a fresh clone lacked a cached chemical-components file and the download timed out). Found from idle GPUs, fixed with a shared cache.

**Infrastructure detours**
- A memory reservation on the cluster job used the wrong units and the job never started; four jobs aborted on a node whose GPUs the PXDesign environment's torch cannot drive (caught by a real matmul check in about 90 s); one config edit silently did not apply and a job was resubmitted at the wrong length until noticed.
- The local pH scorer's self-check needs a control file that is not in the shared tree, so it cannot be validated locally. I did not build a pH oracle on it; the validated multi-pose stack belongs to another agent.
- A ranker patch sat uncommitted until commit time; the mouse-scoring test was shown to fail on the old code before the fix was committed.

**Other agents' corrections (recorded in their notes)**
- AF3 pair-ipTM was reported as ipSAE and corrected; a novelty veto used the adverse TM column (max of three) and was withdrawn after a recomputation; two diagnoses and several claims were retracted by their authors; an AMC Boltz run silently produced nothing (the guard counted preprocessed files); a foldseek wrapper was run twice concurrently and wiped its own working directory; a hand-typed carrier sequence was 62 aa instead of 111 and was caught by a length assertion. Lesson shared in the file: never hand-type a sequence, id or hash; read it from the ledger.

## 4. Lessons that transfer
1. Run the novelty screen before the oracles; it was the real filter.
2. Score survivors on 5 seeds and on every species before ranking; single-seed rank is close to noise for ordering.
3. Put an in-batch failing carrier and a shuffle control in every comparison; a control that does not read zero is information.
4. Make guards count artefacts, not exit codes; tools here exit 0 on failure.
5. Give every wave its own checkout and a cached copy of any downloaded data; look at the GPU, not at whether the process is alive.
6. Say what a measurement does not show, and retract in the same place you claimed.

## 5. What this round does not show
No experimental binding, expression or pH-dependence of any design. ipSAE is an interface confidence, not an affinity. The two judges share training lineage; where they disagree the evidence is weaker than it looks, and where they agree it is weaker than two unrelated methods. The leading designs share one generator and one hotspot family, so a common failure mode would remove them together. No AF3 mouse runs. The pH arm for the new designs was still in progress at the time of writing.
