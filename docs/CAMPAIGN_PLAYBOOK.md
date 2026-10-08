# Campaign playbook: starting a new binder-design round with several agents

For the agent(s) taking the next round, whose target and objectives are not known yet. It is the part of our experience that is not about one target. Target-specific machinery is in `docs/AGENT_PLAYBOOK_PHBIND.md`, `docs/HOMO_OLIGOMER_TARGETS.md`, `docs/WAVES_AND_SECOND_MACHINES.md`, `docs/CLUSTER_LSF.md` and `docs/ROUND_LOG_PHBIND.md` (what went wrong last time, with the corrections). **Numbers below come from one target (a homotrimer, pH switch) and were measured once; treat them as priors to re-check, not constants.**

## 0. The human decides
The user owns the final list, spends the compute and sets the deadline. Agents propose, measure and retract; they do not submit, spend money, change shared rules unilaterally or publish. Anything outward-facing (submitting, pushing result data, contacting a vendor) needs the user's explicit go. A permission given for one action is not a permission for the next.

## 1. The first two hours (before any GPU runs)
1. Read the round's brief end to end and write down, in one file, the **contract**: what is the target construct (exact residue count and stoichiometry), which species/conditions must be scored, the objectives in priority order, the portal's rules for what gets rejected, the deadline, and the date by which the user needs a shortlist. Assert the residue counts in code.
2. Verify the brief against the real files (sequence vs structure, numbering offsets, MSAs). Every mismatch we found (a residue that differed between construct and structure, a NUL byte that made the oracle skip an input and exit 0, a numbering shift in one species) would have scored plausibly and wrongly.
3. Ask which **objective cannot be measured by our tools** (last time: pH). Say so on day 1 and decide who owns a validated route for it. Do not build a sign-sensitive scorer on a script whose self-check cannot run.
4. Put **controls** in before anything is ranked: a known-failing design, one or two known-passing designs, a shuffled-sequence copy of anything that passes, and any solved binders for the target. Find out which controls the oracles recover and say it in writing: if an oracle misses real binders, a low score from it is not evidence.
5. Find the **real filter** early. Last time it was structural novelty, not predicted affinity; we found out only after good-scoring designs were already in hand. Ask what the portal rejects and screen for that first, on backbones, before the expensive scoring.
6. Agree who has which GPUs, and where shared data and the ledger live (section 3).

## 2. Work division (three roles; one agent can hold two)
| role | owns | hands over |
|---|---|---|
| **Pipeline lead** | generation, sequence design, first-pass scoring, the software, waves on all compute | packs of backbones, candidate tables, pushed code with tests |
| **Independent validator** | novelty/other screens, replication of anyone's top results on its own cards, controls | tables back to the lead; a replication of every design that reaches the shortlist |
| **Property specialist** | the objective the others cannot measure (pH, selectivity, species), with its own controls | per-design evidence in the ledger with its n and its caveats |
Rules for the split: the validator **never scores its own designs** for promotion; the person who found a result is not the one who confirms it. Compute follows throughput of the bottleneck (here: screening, not generation). Anyone can run another's pipeline through a wave, but only through its own checkout, config and seed range.

## 3. Communication protocol (what kept three agents honest)
- **One append-only briefing file** plus a **shared ledger** where every number carries (design, metric, value, agent, n, note). Agents add evidence; nobody edits another's output or the ranked table.
- **Numbering:** compute the next note number at the moment of writing, from the file, and re-check after. Five note numbers appear twice in our file, and one of my notes had to be renumbered before it was posted. Timestamps come from the clock, never typed.
- **Never hand-type a sequence, an id or a hash.** Read it from the ledger or a file. A 62-residue sequence typed in place of a 111-residue one was caught only by a length assertion.
- **Claims carry their evidence level:** n of seeds, n of poses, which controls, which model. "Lead" is not "result"; say which you are posting.
- **Retract in the same place you claimed**, fast, with the number that killed the claim. Every agent on the last round retracted something; the retractions were the most useful notes.
- **Ask specific questions that can be answered with a number**, and say what you will do if there is no answer. Silence is not agreement: do not change shared rules on silence. A question that data can decide should be decided by data (we did this for the seed floor).
- **Read what the others posted before starting related work** (collisions: two agents were about to run the same pH analysis; found by reading).
- Tell the others what you started on shared hardware, where, and how to stop it, before you walk away.

## 4. Evidence standards (re-derive thresholds for a new target, keep the structure)
1. **Novelty/rule-out screen first**, on backbones, with the stricter of two plausible conventions reported next to the primary one.
2. **Two different models, not two seeds of one**, and a failing control in every batch. Check how well each recovers solved binders; use a model that misses most real binders only to promote, never to remove.
3. **Re-score survivors on every condition the objectives name** (both species, both pH states) with **>= 5 seeds** before they enter the final list. On our target: single-seed rank hardly predicted which designs survived, a 3-seed worst-seed statistic was about 0.04 looser than the 5-seed one and unstable by about 0.15, and one design flipped on a second species between 3 and 5 seeds.
4. **Independent replication** of a top result by a different agent on a different card, before it is called best.
5. **Control reads before conclusions:** if the controls do not read as expected in a batch, the batch is not evidence. Only compare scores from the same batch.
6. **Say what a measurement does not show** in the same sentence as the result. Interface confidence is not affinity; two models of one family agreeing is weaker than it sounds; one generator family failing together is a risk.
7. Retire an idea cleanly: record the number that closed it (we closed a histidine-biased generation route after 418 trajectories gave no refold).

## 5. Compute rules
- **One job per GPU** unless you have measured that two fit. Concurrent jobs gave out-of-memory and library errors.
- **Count artefacts, not exit codes**: these tools exit 0 on failure. A guard that checks the number and content of outputs is part of every stage.
- **Watch the hardware, not the process.** A job whose process is alive and whose GPU is at 0% has stalled; ours sat for 35 minutes before the idle card was noticed.
- **One checkout per concurrent run**, a shared cache for anything downloaded, distinct seed ranges and set names so two runs cannot regenerate the same thing.
- **Verify a GPU with a real matmul in each environment** before a long job; newer cards can be unsupported by older framework builds.
- **Never run a broad process kill** on a shared machine; take PIDs from `ps` and scope to your own working directory.
- On a cluster: no memory reservation unless you know its units, a restricted host list for hardware your environment supports, an explicit shared output path, and a real-matmul gate at job start (`docs/CLUSTER_LSF.md`).
- Do a one-item end-to-end smoke test on any new machine before scaling; most of our infrastructure bugs appeared there.

## 6. Do / do not
**Do:** start with the contract and controls; screen novelty early; keep every number in the ledger with its n; make guards raise; document detours as they happen; commit software with tests that fail on the old code; keep result data and anything personal out of the public repository (run the repo's public check before every push).
**Do not:** rank on single-seed scores; compare scores across batches; treat a high prescreen score as strength; invent an oracle for the objective you cannot measure; substitute a different model for one that failed without asking; change a shared rule to admit a particular design; let the same agent that proposed a design validate it; leave a wave running without telling the others; claim novelty of a method you have not checked; submit or push on assumption.

## 7. Time plan template (scale to the real deadline)
- **Day 0 (first 2 h):** contract, controls, bottleneck question, role split, ledger, shared-file conventions.
- **First third:** broad generation across lengths/hotspots on all compute; novelty and rule-out screen on a first pack; check the oracles on controls.
- **Middle third:** converge on the length/family that survives the screens; multi-model gate; start the hard objective (the one tools cannot score) in parallel, not at the end.
- **Last third:** 5-seed all-condition re-scoring and independent replication of the shortlist; stop generating new families once the shortlist is stable; write the methods and limitations while compute runs.
- **Final day:** freeze the list with the user, regenerate every table from the ledger (never from memory), reconcile the documents, push the public layer without results.

## 8. Handoff checklist (end of round, and for the agent that follows)
- [ ] Ledger complete and `check` passes; final ranking regenerated from it.
- [ ] Round report written: methods, detours, what was withdrawn, communication timeline (generated from the briefing file, not retyped).
- [ ] Sanitised methods and lessons committed to the public repository; no designs, sequences or per-design results.
- [ ] Submission methods section reviewed by every agent; each edits their own contribution.
- [ ] Compute left clean: jobs stopped, caches noted, no stray processes.
- [ ] Open problems listed with the evidence for each, so the next round starts from a measured position.

## 9. Questions the next team should settle in the first hour
Which objective is hardest to measure and who owns it? What does the portal reject, and can we screen for it before spending GPUs? Which oracle misses real binders on this target? How many seeds and conditions is "robust" here? What is the unit of lineage, and what is the cap? Which compute is free, and who is allowed to use it?
