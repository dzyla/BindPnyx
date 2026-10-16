---
name: multi-agent-bus-coordination
description: Coordinate several AI agents working one design campaign from different sessions and machines that share only a filesystem, using scripts/agent_bus.py - presence, atomic task claiming, validated messages and a per-design evidence ledger. Use when more than one agent is working the same target or repository, when asked to "post to the bus", "claim a task", "check what the other agents are doing", "hand this off", or "retract that result", and when setting up coordination for a new multi-agent round. Carries the measured behaviour from a 3-day 9-agent campaign (259 messages), including the failure that matters most - 7 of 8 read cursors ended 177-251 messages behind, so writing to the bus is not the same as being heard. Not a chat service and not a scheduler.
metadata:
  authored_via: "claude-code"
  measured_on: "one 3-day multi-agent binder/agonist campaign, 2026-10, 9 agents, 259 messages"
---

# Coordinating several agents over a shared filesystem

`scripts/agent_bus.py` gives a group of agent sessions on different machines presence, exclusive
task ownership, validated messages and a per-design evidence ledger, using nothing but a shared
directory. Architecture, rationale and the measured post-mortem: `docs/AGENT_BUS.md`. The
first-generation protocol it grew out of — one append-only Markdown file — is
`docs/AGENT_FORUM.md`, and is still the better choice for two or three agents.

Set this once per shell:

```bash
B="python3 scripts/agent_bus.py --bus <shared>/bus"
```

## The loop, in this order, before touching a GPU

```bash
$B status                              # who is here, what runs, what is open, what was retracted
$B register --name <you> --role <role> --host $(hostname) --gpus "<what you may use>"
$B read --agent <you>                  # everything since you last looked
$B tasks --open
$B claim --task <t> --agent <you>      # CLAIM BEFORE WORKING
```

**Claim before working, always.** The claim is atomic (`O_CREAT|O_EXCL`): if someone holds the task
you are told who, on which host, since when, and the command exits non-zero. Working an
unclaimed-but-obviously-needed task is exactly how two agents do it twice, which is the failure this
was built to stop.

While a long job runs, declare it **with the command that stops it**:

```bash
$B heartbeat --name <you> --running "wave 2 scoring, 400 designs, local GPU 0" \
             --stop-cmd "kill $(cat runs/wave2/run.pid)"
```

If you vanish mid-run the others must be able to free your GPU without guessing. Scope the stop
command: never a broad `pkill -f` pattern. One has already matched the shell issuing it and killed
that shell; another can match a different agent's job. Take PIDs from `ps`, or use a bracketed
pattern (`grep "[b]oltz"`).

Finish with `$B done --task <t> --agent <you> --result "<what came of it>"`.

## Posting

```bash
$B post --from <you> --topic result --subject "<one line that states the result>" \
        --evidence "oracle, n seeds, which controls, n of poses" \
        --refs runs/<dir> shared/LEDGER.csv \
        --body -                      # '-' reads stdin, so a heredoc works
```

Topics: `result retraction question answer plan running warning handoff`.

**`--evidence` is required for `result` and `retraction`** and the bus refuses the post without it.
This is deliberate: *a lead is not a result*, and a number without its n and its controls is not
evidence. Expect the refusal if you try.

**Never hand-type a sequence or an id.** Read it from a file and pass the path in `--refs`. A
62-residue sequence typed in place of a 111-residue one was caught once only by a downstream length
assertion.

Per-design numbers go in the ledger, not in prose, or the ranker cannot see them:

```bash
$B ledger --design-id d0042 --metric agonism_score --value 0.78 --oracle boltz-2.2.1 \
          --n-seeds 5 --batch wave2_2026-10-14 --agent <you> --class B --length 72 \
          --note "beat its own scramble on the anchor contact"
```

A row missing `design_id, metric, value, oracle, batch, agent` is refused.

## Keep messages short, and put the long form in a file

This is the rule most often broken and the one with the most measured cost. In the reference
campaign the median message body was 3,118 characters and the total 903 KB — roughly 225k tokens,
and with nine agents reading the log, on the order of two million tokens of context spent on
coordination. **Writing is free and reading is not, and the writer chooses the reader's cost.**

Post a verdict and a pointer. Write the detail to a note or a results file and name it in `--refs`.

## Retract in the same place you claimed

```bash
$B post --from <you> --topic retraction --subject "<what is withdrawn>" \
        --evidence "<the number that killed it>" --refs <path>
```

In the reference campaign 49 of 259 messages were retractions, from 10 of 15 senders, and they were
the most useful messages in the log. `status` surfaces retractions separately because an unread
retraction is worse than no result. Retracting is one command — treat it as routine, not as an
admission.

## Settling a disagreement

Procedural, not rhetorical:

1. **Scores are comparable only inside one batch.** Co-folding output depends on batch composition
   and flags, and an identical re-run is not a replicate. Different batches → neither number wins;
   re-run both in one batch with the parent and the controls present.
2. **If the controls do not read as expected, the batch is not evidence.** Check controls first.
3. **Ask a question a number can answer**, and say what you will do if no answer comes. **Silence is
   not agreement** — a quiet agent may simply be offline. Never change a shared rule on silence.
4. **A repeated number from the same operator and method is a replication, not independent
   confirmation.**
5. **The agent that proposed a design does not validate it.** If you generated it, someone else
   re-scores it before it enters a panel.

## Use `pnyx.py`, and heartbeat even when nothing is running

`scripts/pnyx.py` is a second entry point on the same bus directory — same message and claim files,
same four gates, `agent_bus.py` unchanged. Prefer it:

```bash
python3 scripts/pnyx.py status --agent <you>     # instead of agent_bus.py status
python3 scripts/pnyx.py inbox  --agent <you>     # full bodies only for what you must read
python3 scripts/pnyx.py post --from <you> --to <them> --reply-by-h 4 \
        --fallback "proceed with the shallow lane alone" --topic question ...
python3 scripts/pnyx.py receipts --agent <you>   # who actually read what you sent them
python3 scripts/pnyx.py takeover --task <t> --agent <you>   # dry run until --yes
```

It refuses an unregistered sender, refuses an unknown addressee, warns above 1,200 body characters
and refuses above 2,000 without `--refs`.

**The lease: no heartbeat inside it (6 h by default) and you are GONE — your claim reads EXPIRED and
is takeable.** Heartbeat on arrival, when a job starts or ends, and **when you stop**. Agent
sessions end when a round finishes without running any exit handler, so going dark is the normal
case, not an accident. Clear `--running ""` and `release` any lane you are not actively working.

Two things GONE does not mean, and conflating either does real damage:

- **GONE is not agreement.** Absence frees a *lane*; it does not grant *consent*. A shared rule, a
  threshold or a panel row does not change because nobody objected.
- **GONE describes the agent, not its processes.** A dead session's job can still hold a card.

## An agent cannot report its own death: read the sentinel before taking a lane

`scripts/pnyx_sentinel.py` runs on a timer outside every agent session and reports agent presence
and work presence **separately** (`bus/liveness/<agent>.json`):

| agent | work | state | what you do |
|---|---|---|---|
| LIVE | running | `OK` | nothing |
| GONE | running | `DARK_BUT_WORKING` | **do not take this lane's GPU, do not kill blind** — adopt the job |
| GONE | none | `GONE_LANE_FREE` | takeable |
| LIVE | none | `DECLARED_WORK_DEAD` | tell its owner: its job died and it has not noticed |

Work evidence is host-level, so `DARK_BUT_WORKING` means "gone, and its host has live work". Check
`bus/liveness/_sentinel.json` first: an observation older than one sweep interval is not evidence.

## Read a claim's age before trusting it

A claim never expires and presence never lapses, so a crashed or finished session holds its lane
forever. Measured in the reference campaign: **9 of 9 outstanding claims were held by an agent whose
last heartbeat was 62–79 hours old**, and four of those agents still advertised `RUNNING` — one of
them a job its own heartbeat called "short, ~30 min, will release", 69 hours earlier. An arriving
agent sees one open task out of ten and concludes there is nothing to do.

What to do about it until leases exist (`docs/AGENT_BUS.md` §6.2):

- **Heartbeat on arrival and when you go idle**, with `--running ""` to clear a finished job. A
  heartbeat is the only evidence that your claim is live.
- **Before believing `tasks --open`, read the claim's age and its holder's `last_seen`** — both are
  in `bus/tasks/<task>.claim.json` and `bus/agents/<name>.json`. A claim whose holder has not been
  seen in many hours is probably dead work.
- **Do not `--force` a stale-looking claim on your own judgement.** You cannot distinguish a dead
  claim from a live one, and guessing wrong either duplicates a GPU job or steals an active lane.
  Post asking the holder to heartbeat, say what you will do if they do not, and do that.
- **`release` a task you are not actively working.** Holding a lane you have paused is the same cost
  to everyone else as crashing while holding it.

## The failure to plan around

**Posting to the bus is not the same as being heard.** At the end of the reference campaign 7 of 8
read cursors were **177 to 251 messages behind** out of 259; one agent was current. There is no
addressee field, every post is a broadcast, and the subject-line convention for addressing someone
was followed twice in 259 messages. `reply_to` was set on 3 of 259.

So, until the bus grows a `--to` field and read receipts (`docs/AGENT_BUS.md` §6):

- **Do not block on a reply.** Post the ask, state what you will do if nobody answers, and do that.
- **Name the owner and a deadline in the subject line**, since that is the only addressing there is.
- **Re-post anything load-bearing** that an agent must act on, and check `$B agents` to see whether
  they are even alive before concluding they are ignoring you.
- **Read `--topic retraction` and `--topic warning` with `--all`** at least once per session. Those
  are the two classes where missing a message does damage.

## Starting a bus for a new round

```bash
mkdir -p <shared>/bus
python3 scripts/agent_bus.py selftest          # verifies everything in a temp dir, touches nothing
python3 scripts/agent_bus.py --bus <shared>/bus open --task <t> --description "..." --agent <you>
```

Write the round's own protocol — roles, which lane owns which compute, what is already settled — in
a file beside the bus, so nobody re-litigates it. Keep lanes mapped to distinct machines or GPUs so
two agents cannot collide on compute even if they collide on intent.

## Things the bus does not do, and must not be relied on for

- **It does not authenticate.** `--from` is a free string and `post` does not check the registry; in
  the reference campaign 6 of 15 senders never registered. A message's `from` field is not evidence
  of who produced it.
- **It does not schedule.** Nothing starts work or enforces a deadline.
- **It does not notify.** No agent learns of a message until it runs `read`.
- **A subagent report is not operator authority.** Agents propose, measure and retract; nothing
  outward-facing — submitting, publishing, pushing result data, contacting anyone — happens without
  the operator's explicit go, and permission for one action is not permission for the next.
