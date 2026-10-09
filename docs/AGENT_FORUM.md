# The agent forum: how several agents (and a person) coordinate on one campaign

In the Challenge 2 campaign three AI agents on different machines worked on the same target for several days with one person making the final decisions. They did not talk to each other directly. They used a **forum**: one append-only Markdown file in a shared directory, plus a shared evidence ledger. This page describes the protocol so that it can be reused. No agent or session identifiers appear here; roles are called agents A, B and C.

## Why a file
- It needs no service. Any agent that can read and append a file can take part, including agents on a cluster login node or a workstation that is offline for hours.
- It is a record. Every claim, ask, retraction and decision is there with a time and an author, so the methods write-up and the hand-off can be generated from it, and so that nobody has to trust memory.
- It forces reading before writing. The first thing an agent does in a session is read the last notes.

## What goes where
| place | content | who writes |
|---|---|---|
| `<shared>/docs/FORUM.md` | the forum: numbered notes, append-only | every agent, only by appending |
| `<shared>/ranking/evidence.csv` | the evidence ledger: one measurement per line (time, agent, design, metric, value, seeds, note) | every agent, only by appending (`phbind/ranking.py add`) |
| `<shared>/ranking/designs.csv` | the registry of designs (id, sequence, lineage, generator, parent) | an agent adds a design once |
| `<shared>/ranking/FINAL_RANKING.*` | the ranker's output | nobody edits it; `rank` regenerates it |
| `<shared>/handoff_<agent>/` | each agent's drop folder for files others need (tables, packs, poses) | the owner of the folder |
| `<shared>/FINAL/` | the end-of-challenge package | the agent that closes the challenge |

## Rules that made it work
1. **Append only; never edit another agent's note or ledger line.** A correction is a new note that says which note it corrects. Retract in the same place you made the claim.
2. **Compute the next note number at write time.** Numbers collided three times when agents counted from a stale copy. `scripts/forum_note.py` reads the file, takes the highest number plus one, saves a backup copy, and appends.
3. **Every note starts with a headline that states the result or the ask**, then the evidence, then what the author does *not* claim.
4. **Asks name an owner and a deadline** ("agent B: score these ten poses by 15:00"). Silence is not agreement and a quiet agent may simply be offline, so plan so that nothing waits on a reply: say what you will do if there is none.
5. **Claims carry their evidence level**: how many seeds, which batch, which machine, which operator. A repeated number from the same operator and method is a replication, not independent confirmation.
6. **Shared rules change only by argument in a note**, with both other agents given a chance to object; then the rules file is edited once.
7. **The person owns the final list, spends the compute and sets the deadline.** Agents propose, measure and retract; they do not submit, publish or push result data without an explicit go.
8. **Decide with data, not opinion.** When two agents disagreed (a threshold at three versus five seeds, a pH veto), the question was turned into a measurement and the answer recorded in the next note.

## Note format
```
## NOTE <n> — <agent>, <date time zone> — **<headline: the result or the ask>**
### <n>.1 To <agent>: <what was done / what is asked / deadline>
### <n>.2 Evidence: <numbers with seeds, batch, machine>
### <n>.3 Not claimed: <what the measurement does not show>
```
Reading order for a new session: the last ten notes, the ledger's `FINAL_RANKING.md`, the person's latest instruction, then the notes addressed to you.

## Typical day (from Challenge 2)
- Morning: each agent reads the new notes, replies to asks addressed to it, and posts its plan with the machines it will use.
- During the day: results go into the ledger as they land (not at the end of a stage), and a note is posted for anything that changes the ranking or contradicts an earlier claim.
- Evening: a status note lists what finished, what is running, what is needed from the person, and what the next list looks like; the list builder is rerun from the ledger.
- Last day: freeze candidate intake about twelve hours before the person submits; regenerate every table from the ledger; list open items honestly.

## What the forum was used for (themes, anonymised)
About a hundred notes over six days. The main themes were: agreeing the gate (two different models, not seeds of one); finding that structural novelty, not predicted affinity, limited the number of usable designs; handing batches of backbones between agents for the novelty screen; measuring what changes a co-folding model's output (command line flag, batch composition); building the pH scoring and replicating it; deciding the list order for the portal's first-20 screening; and about a dozen retractions. The retractions were the most useful part: they were cheap because there was a place to make them.

## Anti-patterns we hit
- Counting note numbers from a stale copy (collisions).
- Assuming an agent was stopped because it was quiet (it was running).
- Posting a number without saying which batch or seeds it came from.
- Writing the plan in a note but not putting the evidence in the ledger, so the ranking did not change.
- Letting a note become the only record of a number that the methods text then quoted.

## Starting a forum for a new challenge
```bash
mkdir -p <shared>/docs <shared>/ranking
printf '# Forum\n\nAppend-only. Use scripts/forum_note.py to add a note.\n' > <shared>/docs/FORUM.md
python scripts/forum_note.py <shared>/docs/FORUM.md --agent A --headline "plan and machines" --body "..."
```
