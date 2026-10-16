#!/usr/bin/env python3
"""Agent bus: message exchange, task claiming and presence over a shared filesystem.

This is the coordination layer used by a multi-agent design campaign -- the "Pnyx" half of
BindPnyx, the assembly where the agents report to each other. It is shipped as a worked example,
together with an honest account of how it behaved in practice: docs/AGENT_BUS.md carries the
measured numbers, including the ways it did NOT work.

Why this exists
---------------
Several agents work a campaign from different sessions and different machines, sharing only a
filesystem. The previous campaign's retrospective (docs/CAMPAIGN_PLAYBOOK.md section 3) names the
failures that cost time, and each is addressed here by construction rather than by goodwill:

  * two agents nearly ran the same analysis twice  -> `claim`, atomic, first writer wins
  * five note numbers appeared twice in one file   -> ids are generated, never typed
  * a sequence was hand-typed at the wrong length  -> `post --refs` points at files; `ledger` validates
  * "lead" was reported as "result"                -> `--evidence` is a required field on `post`
  * nobody knew what was running where             -> `heartbeat` carries host, GPU and a stop command

Design constraints that shaped it
---------------------------------
The shared directory is typically an NFS export mounted on every participating machine, so
`fcntl.flock` is not dependable and a single append-only file would be a corruption and
lost-update risk. Therefore:

  * **one file per message.** No locks, no readers blocking writers, no interleaved appends.
    Ordering comes from a sortable id (nanosecond timestamp + random suffix), not from file order.
  * **claims use `O_CREAT|O_EXCL`**, which is atomic on NFSv3+ and is the one primitive we can
    rely on. The loser of a race is told who holds the task, and never silently proceeds.
  * **nothing is ever rewritten in place.** State changes add files. The history cannot be edited,
    which is what makes the log trustworthy between agents that do not share memory.

What it is good at, and what it is not
--------------------------------------
Measured over one 3-day campaign (259 messages, 9 registered agents): the evidence gate held
(240/259 messages carry evidence, 227/259 cite a file), ids never collided, and 49 of 259 messages
were retractions -- retracting was cheap, so it happened, which is the property worth copying.

Two things it does badly, both measured on that campaign and both fixable:

  * it is an **archive, not a channel** -- there is no addressee field, every message is a
    broadcast, and 7 of 8 read cursors ended 177-251 messages behind. Agents wrote diligently and
    read almost nothing.
  * **claims never expire and presence never lapses** -- 9 of 9 outstanding claims were held by an
    agent whose last heartbeat was 62-79 hours old, four of which still advertised RUNNING. The
    mechanism that prevents duplicate work became one that prevents work.

Read docs/AGENT_BUS.md before adopting this shape; the diagnosis and the concrete changes it needs
(a --to field, claim leases, lapsed presence) are there.

Quick start for a new agent
---------------------------
    B="python3 scripts/agent_bus.py --bus <shared>/bus"
    $B register --name myagent --role validator --host $(hostname)
    $B status                                  # what is happening right now
    $B read --agent myagent                    # new messages since I last looked
    $B tasks --open                            # what nobody owns yet
    $B claim --task control-readout --agent myagent
    $B post --from myagent --topic result --subject "control read out" \
            --evidence "2 inputs, 5 samples, 1 seed, scramble control in batch" \
            --refs runs/control --body "..."
    $B done --task control-readout --agent myagent --result "see message above"

Run `--help` on any subcommand. `python3 scripts/agent_bus.py selftest` checks the whole thing
in a temporary directory and touches nothing real.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import getpass
import json
import os
import random
import socket
import sys
import tempfile
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_BUS = os.path.join(REPO, "bus")

TOPICS = ["result", "retraction", "question", "answer", "plan", "running", "warning", "handoff"]
EVIDENCE_HINT = ("say what backs this: oracle, n seeds, which controls, n of poses. "
                 "'lead' is not 'result'.")


# --------------------------------------------------------------------------- paths
def _dirs(bus: str) -> dict:
    d = {
        "bus": bus,
        "messages": os.path.join(bus, "messages"),
        "tasks": os.path.join(bus, "tasks"),
        "agents": os.path.join(bus, "agents"),
        "cursors": os.path.join(bus, "cursors"),
    }
    for p in d.values():
        os.makedirs(p, exist_ok=True)
    return d


def _now_iso() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _new_id() -> str:
    """Sortable, collision-resistant, and never typed by a human."""
    return f"{time.time_ns():019d}-{random.getrandbits(24):06x}"


def _write_atomic(path: str, payload: dict) -> None:
    """Write via a temp file in the same directory, then rename. Rename is atomic."""
    d = os.path.dirname(path)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp-")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(payload, fh, indent=1, sort_keys=True)
            fh.write("\n")
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _read_json(path: str):
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def _load_messages(dirs: dict) -> list:
    out = []
    for name in sorted(os.listdir(dirs["messages"])):
        if not name.endswith(".json") or name.startswith("."):
            continue
        m = _read_json(os.path.join(dirs["messages"], name))
        if m:
            out.append(m)
    out.sort(key=lambda m: m.get("id", ""))
    return out


# --------------------------------------------------------------------------- commands
def cmd_register(a, dirs):
    rec = {
        "name": a.name, "role": a.role, "host": a.host or socket.gethostname(),
        "user": getpass.getuser(), "gpus": a.gpus, "note": a.note,
        "registered": _now_iso(), "last_seen": _now_iso(),
        "running": None, "stop_cmd": None, "pid": os.getpid(),
    }
    path = os.path.join(dirs["agents"], f"{a.name}.json")
    prev = _read_json(path)
    if prev:
        rec["registered"] = prev.get("registered", rec["registered"])
        for k in ("running", "stop_cmd"):
            rec[k] = prev.get(k)
    _write_atomic(path, rec)
    print(f"registered {a.name} ({rec['role']}) on {rec['host']}")
    return 0


def cmd_heartbeat(a, dirs):
    path = os.path.join(dirs["agents"], f"{a.name}.json")
    rec = _read_json(path)
    if rec is None:
        print(f"{a.name} is not registered; run `register` first", file=sys.stderr)
        return 1
    rec["last_seen"] = _now_iso()
    if a.running is not None:
        rec["running"] = a.running or None
    if a.stop_cmd is not None:
        rec["stop_cmd"] = a.stop_cmd or None
    if a.running and not (a.stop_cmd or rec.get("stop_cmd")):
        print("WARNING: you declared running work with no --stop-cmd. The others cannot stop it "
              "for you if you disappear.", file=sys.stderr)
    _write_atomic(path, rec)
    print(f"heartbeat {a.name}: running={rec['running']!r}")
    return 0


def cmd_agents(a, dirs):
    recs = [_read_json(os.path.join(dirs["agents"], f))
            for f in sorted(os.listdir(dirs["agents"])) if f.endswith(".json")]
    recs = [r for r in recs if r]
    if not recs:
        print("no agents registered")
        return 0
    print(f"{'agent':<16}{'role':<14}{'host':<14}{'last seen':<22}running")
    for r in recs:
        print(f"{r['name']:<16}{str(r.get('role')):<14}{str(r.get('host')):<14}"
              f"{str(r.get('last_seen')):<22}{r.get('running') or '-'}")
        if r.get("stop_cmd"):
            print(f"{'':<16}stop: {r['stop_cmd']}")
    return 0


def cmd_post(a, dirs):
    if a.topic not in TOPICS:
        print(f"topic must be one of {TOPICS}", file=sys.stderr)
        return 1
    body = a.body
    if body == "-":
        body = sys.stdin.read()
    if not body or not body.strip():
        print("refusing to post an empty body", file=sys.stderr)
        return 1
    if a.topic in ("result", "retraction") and not a.evidence:
        print(f"--evidence is required for topic '{a.topic}'. {EVIDENCE_HINT}", file=sys.stderr)
        return 1
    mid = _new_id()
    msg = {
        "id": mid, "ts": _now_iso(), "from": a.sender, "topic": a.topic,
        "subject": a.subject, "body": body, "evidence": a.evidence,
        "refs": a.refs or [], "reply_to": a.reply_to, "priority": a.priority,
    }
    _write_atomic(os.path.join(dirs["messages"], f"{mid}.json"), msg)
    print(f"posted {mid}  [{a.topic}] {a.subject}")
    return 0


def cmd_read(a, dirs):
    msgs = _load_messages(dirs)
    cursor_path = os.path.join(dirs["cursors"], f"{a.agent}.txt") if a.agent else None
    last = ""
    if cursor_path and os.path.exists(cursor_path) and not a.all:
        last = open(cursor_path).read().strip()
    sel = [m for m in msgs if m["id"] > last]
    if a.topic:
        sel = [m for m in sel if m["topic"] == a.topic]
    if a.sender:
        sel = [m for m in sel if m["from"] == a.sender]
    if a.since:
        sel = [m for m in sel if m["ts"] >= a.since]
    if a.limit:
        sel = sel[-a.limit:]
    if not sel:
        print("no new messages" if not a.all else "no messages match")
        return 0
    for m in sel:
        mark = "!" if m.get("priority") == "high" else " "
        print(f"\n{'='*78}\n{mark}[{m['topic']}] {m['subject']}\n"
              f"  from {m['from']}  {m['ts']}  id {m['id']}"
              + (f"\n  reply to {m['reply_to']}" if m.get("reply_to") else ""))
        if m.get("evidence"):
            print(f"  evidence: {m['evidence']}")
        if m.get("refs"):
            print(f"  refs: {', '.join(m['refs'])}")
        print(f"{'-'*78}\n{m['body'].rstrip()}")
    print(f"\n{'='*78}\n{len(sel)} message(s)")
    if cursor_path and sel and not a.no_cursor:
        with open(cursor_path, "w") as fh:
            fh.write(sel[-1]["id"])
    return 0


def _task_paths(dirs, task):
    return (os.path.join(dirs["tasks"], f"{task}.claim.json"),
            os.path.join(dirs["tasks"], f"{task}.done.json"),
            os.path.join(dirs["tasks"], f"{task}.open.json"))


def cmd_open_task(a, dirs):
    _, done, openp = _task_paths(dirs, a.task)
    if os.path.exists(done):
        print(f"task '{a.task}' is already done", file=sys.stderr)
        return 1
    _write_atomic(openp, {"task": a.task, "description": a.description,
                          "opened_by": a.agent, "opened": _now_iso()})
    print(f"opened task '{a.task}'")
    return 0


def cmd_claim(a, dirs):
    claim, done, _ = _task_paths(dirs, a.task)
    if os.path.exists(done):
        rec = _read_json(done) or {}
        print(f"task '{a.task}' is already DONE by {rec.get('agent')} "
              f"({rec.get('finished')}); not claiming", file=sys.stderr)
        return 1
    payload = json.dumps({"task": a.task, "agent": a.agent, "claimed": _now_iso(),
                          "host": socket.gethostname(), "note": a.note},
                         indent=1, sort_keys=True) + "\n"
    try:
        # O_EXCL is the atomic primitive we trust on NFS: exactly one creator wins.
        fd = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o664)
    except FileExistsError:
        rec = _read_json(claim) or {}
        owner = rec.get("agent", "?")
        if owner == a.agent:
            print(f"you already hold '{a.task}' (claimed {rec.get('claimed')})")
            return 0
        print(f"REFUSED: '{a.task}' is held by {owner} since {rec.get('claimed')} "
              f"on {rec.get('host')}.\nTalk to them, or pick another task. "
              f"Do not do this work in parallel.", file=sys.stderr)
        return 1
    with os.fdopen(fd, "w") as fh:
        fh.write(payload)
    print(f"claimed '{a.task}' for {a.agent}")
    return 0


def cmd_release(a, dirs):
    claim, _, _ = _task_paths(dirs, a.task)
    rec = _read_json(claim)
    if rec is None:
        print(f"'{a.task}' is not claimed", file=sys.stderr)
        return 1
    if rec.get("agent") != a.agent and not a.force:
        print(f"'{a.task}' is held by {rec.get('agent')}, not you. "
              f"Use --force only after agreeing with them.", file=sys.stderr)
        return 1
    os.unlink(claim)
    print(f"released '{a.task}' (was {rec.get('agent')})")
    return 0


def cmd_done(a, dirs):
    claim, done, openp = _task_paths(dirs, a.task)
    rec = _read_json(claim) or {"task": a.task, "agent": a.agent}
    if rec.get("agent") != a.agent and os.path.exists(claim) and not a.force:
        print(f"'{a.task}' is held by {rec.get('agent')}, not you", file=sys.stderr)
        return 1
    rec.update({"agent": a.agent, "finished": _now_iso(), "result": a.result})
    _write_atomic(done, rec)
    for p in (claim, openp):
        if os.path.exists(p):
            os.unlink(p)
    print(f"task '{a.task}' marked done")
    return 0


def cmd_tasks(a, dirs):
    states = {}
    for f in sorted(os.listdir(dirs["tasks"])):
        if not f.endswith(".json") or f.startswith("."):
            continue
        rec = _read_json(os.path.join(dirs["tasks"], f))
        if not rec:
            continue
        name = rec.get("task")
        state = "done" if f.endswith(".done.json") else (
            "claimed" if f.endswith(".claim.json") else "open")
        # done beats claimed beats open
        rank = {"open": 0, "claimed": 1, "done": 2}
        if name not in states or rank[state] > rank[states[name][0]]:
            states[name] = (state, rec)
    if not states:
        print("no tasks")
        return 0
    for name, (state, rec) in sorted(states.items()):
        if a.open and state != "open":
            continue
        who = rec.get("agent") or rec.get("opened_by") or "-"
        when = rec.get("finished") or rec.get("claimed") or rec.get("opened") or ""
        print(f"[{state:<7}] {name:<28} {who:<14} {when}")
        if rec.get("description"):
            print(f"{'':<11}{rec['description']}")
        if rec.get("result"):
            print(f"{'':<11}-> {rec['result']}")
    return 0


def cmd_ledger(a, dirs):
    """Validated append to shared/LEDGER.csv. Refuses rows missing provenance."""
    import csv
    path = a.path or os.path.join(REPO, "shared", "LEDGER.csv")
    cols = ["design_id", "class", "length", "metric", "value", "oracle",
            "n_seeds", "batch", "agent", "timestamp", "note"]
    if not os.path.exists(path):
        with open(path, "w", newline="") as fh:
            csv.writer(fh).writerow(cols)
    row = {"design_id": a.design_id, "class": a.cls, "length": a.length,
           "metric": a.metric, "value": a.value, "oracle": a.oracle,
           "n_seeds": a.n_seeds, "batch": a.batch, "agent": a.agent,
           "timestamp": _now_iso(), "note": a.note or ""}
    missing = [k for k in ("design_id", "metric", "value", "oracle", "batch", "agent")
               if row[k] in (None, "")]
    if missing:
        print(f"refusing to write a row without provenance; missing: {missing}", file=sys.stderr)
        return 1
    with open(path, "a", newline="") as fh:
        csv.DictWriter(fh, fieldnames=cols).writerow(row)
    print(f"ledger += {a.design_id} {a.metric}={a.value} ({a.oracle}, batch {a.batch})")
    return 0


def cmd_status(a, dirs):
    msgs = _load_messages(dirs)
    print(f"campaign bus — {dirs['bus']}")
    print(f"{_now_iso()}\n")

    recs = [_read_json(os.path.join(dirs["agents"], f))
            for f in sorted(os.listdir(dirs["agents"])) if f.endswith(".json")]
    recs = [r for r in recs if r]
    print(f"AGENTS ({len(recs)})")
    if not recs:
        print("  none registered")
    for r in recs:
        print(f"  {r['name']:<14} {str(r.get('role')):<12} {str(r.get('host')):<13} "
              f"seen {r.get('last_seen')}")
        if r.get("running"):
            print(f"       RUNNING: {r['running']}")
            print(f"       stop:    {r.get('stop_cmd') or 'NOT DECLARED'}")

    states = {}
    for f in sorted(os.listdir(dirs["tasks"])):
        if not f.endswith(".json") or f.startswith("."):
            continue
        rec = _read_json(os.path.join(dirs["tasks"], f))
        if not rec:
            continue
        state = "done" if f.endswith(".done.json") else (
            "claimed" if f.endswith(".claim.json") else "open")
        rank = {"open": 0, "claimed": 1, "done": 2}
        n = rec.get("task")
        if n not in states or rank[state] > rank[states[n][0]]:
            states[n] = (state, rec)
    counts = {}
    for st, _ in states.values():
        counts[st] = counts.get(st, 0) + 1
    print(f"\nTASKS  open {counts.get('open',0)}  claimed {counts.get('claimed',0)}  "
          f"done {counts.get('done',0)}")
    for n, (st, rec) in sorted(states.items()):
        if st != "done":
            who = rec.get("agent") or rec.get("opened_by") or "-"
            print(f"  [{st:<7}] {n:<28} {who}")

    print(f"\nMESSAGES ({len(msgs)}) — last 5")
    for m in msgs[-5:]:
        print(f"  {m['ts']}  [{m['topic']:<10}] {m['from']:<12} {m['subject']}")
    retr = [m for m in msgs if m["topic"] == "retraction"]
    if retr:
        print(f"\n  {len(retr)} RETRACTION(S) — read these before trusting earlier results:")
        for m in retr:
            print(f"    {m['ts']} {m['from']}: {m['subject']}")
    print("\nProtocol: docs/AGENT_BUS.md")
    return 0


# --------------------------------------------------------------------------- selftest
def selftest(_a, _dirs):
    import shutil
    tmp = tempfile.mkdtemp(prefix="bus-selftest-")
    fails = []

    def ck(cond, msg):
        if not cond:
            fails.append(msg)

    try:
        d = _dirs_for(tmp)

        class A:
            pass

        # ids are monotonic and unique
        ids = [_new_id() for _ in range(500)]
        ck(len(set(ids)) == 500, "ids collided")
        ck(ids == sorted(ids), "ids are not monotonically sortable")

        # register + heartbeat
        a = A(); a.name = "alice"; a.role = "lead"; a.host = "h1"; a.gpus = "0"; a.note = None
        ck(cmd_register(a, d) == 0, "register failed")
        a = A(); a.name = "alice"; a.running = "wave 1 scoring"; a.stop_cmd = "kill $(cat run.pid)"
        ck(cmd_heartbeat(a, d) == 0, "heartbeat failed")
        ck(cmd_heartbeat_missing(d) == 1, "heartbeat on unregistered agent must fail")

        # post: evidence required for results, empty body refused
        base = dict(sender="alice", subject="s", body="b", evidence=None, refs=None,
                    reply_to=None, priority="normal")
        a = A(); a.__dict__.update(base, topic="result")
        ck(cmd_post(a, d) == 1, "result without evidence must be refused")
        a = A(); a.__dict__.update(base, topic="result", evidence="3 seeds, scramble control")
        ck(cmd_post(a, d) == 0, "valid result post failed")
        a = A(); a.__dict__.update(base, topic="plan", body="   ")
        ck(cmd_post(a, d) == 1, "empty body must be refused")
        a = A(); a.__dict__.update(base, topic="nonsense")
        ck(cmd_post(a, d) == 1, "unknown topic must be refused")

        # claim is exclusive
        a = A(); a.task = "t1"; a.agent = "alice"; a.note = None
        ck(cmd_claim(a, d) == 0, "first claim failed")
        a = A(); a.task = "t1"; a.agent = "bob"; a.note = None
        ck(cmd_claim(a, d) == 1, "second agent must NOT be able to claim a held task")
        a = A(); a.task = "t1"; a.agent = "alice"; a.note = None
        ck(cmd_claim(a, d) == 0, "re-claim by the holder should be a no-op success")
        a = A(); a.task = "t1"; a.agent = "bob"; a.force = False
        ck(cmd_release(a, d) == 1, "a non-holder must not release")
        a = A(); a.task = "t1"; a.agent = "alice"; a.force = False
        ck(cmd_release(a, d) == 0, "holder release failed")
        a = A(); a.task = "t1"; a.agent = "bob"; a.note = None
        ck(cmd_claim(a, d) == 0, "claim after release failed")
        a = A(); a.task = "t1"; a.agent = "bob"; a.result = "ok"; a.force = False
        ck(cmd_done(a, d) == 0, "done failed")
        a = A(); a.task = "t1"; a.agent = "alice"; a.note = None
        ck(cmd_claim(a, d) == 1, "a done task must not be claimable")

        # cursor advances: second read sees nothing new
        a = A(); a.__dict__.update(agent="carol", topic=None, sender=None, since=None,
                                   limit=None, all=False, no_cursor=False)
        ck(cmd_read(a, d) == 0, "read failed")
        n_before = len(_load_messages(d))
        a = A(); a.__dict__.update(agent="carol", topic=None, sender=None, since=None,
                                   limit=None, all=False, no_cursor=False)
        cmd_read(a, d)
        ck(len(_load_messages(d)) == n_before, "read must not mutate messages")
        cur = os.path.join(d["cursors"], "carol.txt")
        ck(os.path.exists(cur), "cursor was not written")

        # ledger validation
        led = os.path.join(tmp, "L.csv")
        a = A(); a.__dict__.update(path=led, design_id="d1", cls="B", length=70,
                                   metric="agonism", value=0.8, oracle="boltz",
                                   n_seeds=3, batch="b1", agent="alice", note=None)
        ck(cmd_ledger(a, d) == 0, "valid ledger row refused")
        a = A(); a.__dict__.update(path=led, design_id="d2", cls="B", length=70,
                                   metric="agonism", value=0.8, oracle="",
                                   n_seeds=3, batch="b1", agent="alice", note=None)
        ck(cmd_ledger(a, d) == 1, "ledger row without oracle must be refused")
        ck(sum(1 for _ in open(led)) == 2, "ledger should hold header + 1 row")

        # atomic write leaves no temp files behind
        leftover = [f for f in os.listdir(d["messages"]) if f.startswith(".tmp-")]
        ck(not leftover, f"temp files left behind: {leftover}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    for f in fails:
        print(f"FAIL {f}", file=sys.stderr)
    print(f"selftest: {'PASS' if not fails else str(len(fails)) + ' FAILURES'}", file=sys.stderr)
    return 1 if fails else 0


def _dirs_for(root):
    return _dirs(os.path.join(root, "bus"))


def cmd_heartbeat_missing(d):
    class A:
        pass
    a = A(); a.name = "nobody-here"; a.running = None; a.stop_cmd = None
    return cmd_heartbeat(a, d)


# --------------------------------------------------------------------------- cli
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bus", default=DEFAULT_BUS, help=f"bus directory (default {DEFAULT_BUS})")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("register", help="announce yourself")
    p.add_argument("--name", required=True)
    p.add_argument("--role", default="agent", help="lead / validator / property / agent")
    p.add_argument("--host", default=None)
    p.add_argument("--gpus", default=None)
    p.add_argument("--note", default=None)
    p.set_defaults(fn=cmd_register)

    p = sub.add_parser("heartbeat", help="say you are alive and what you are running")
    p.add_argument("--name", required=True)
    p.add_argument("--running", default=None, help="what is running now ('' to clear)")
    p.add_argument("--stop-cmd", dest="stop_cmd", default=None,
                   help="how another agent stops it if you vanish")
    p.set_defaults(fn=cmd_heartbeat)

    p = sub.add_parser("agents", help="who is here")
    p.set_defaults(fn=cmd_agents)

    p = sub.add_parser("post", help="post a message")
    p.add_argument("--from", dest="sender", required=True)
    p.add_argument("--topic", required=True, choices=TOPICS)
    p.add_argument("--subject", required=True)
    p.add_argument("--body", required=True, help="text, or '-' to read stdin")
    p.add_argument("--evidence", default=None, help=EVIDENCE_HINT)
    p.add_argument("--refs", nargs="*", default=None, help="paths backing this claim")
    p.add_argument("--reply-to", dest="reply_to", default=None, help="message id")
    p.add_argument("--priority", default="normal", choices=["normal", "high"])
    p.set_defaults(fn=cmd_post)

    p = sub.add_parser("read", help="read new messages (advances your cursor)")
    p.add_argument("--agent", default=None, help="your name, for the read cursor")
    p.add_argument("--topic", default=None, choices=TOPICS)
    p.add_argument("--from", dest="sender", default=None)
    p.add_argument("--since", default=None, help="ISO timestamp")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--all", action="store_true", help="ignore the cursor, show everything")
    p.add_argument("--no-cursor", action="store_true", help="do not advance the cursor")
    p.set_defaults(fn=cmd_read)

    p = sub.add_parser("open", help="declare a task that needs an owner")
    p.add_argument("--task", required=True)
    p.add_argument("--description", default=None)
    p.add_argument("--agent", default=None)
    p.set_defaults(fn=cmd_open_task)

    p = sub.add_parser("claim", help="take exclusive ownership of a task")
    p.add_argument("--task", required=True)
    p.add_argument("--agent", required=True)
    p.add_argument("--note", default=None)
    p.set_defaults(fn=cmd_claim)

    p = sub.add_parser("release", help="give a task back")
    p.add_argument("--task", required=True)
    p.add_argument("--agent", required=True)
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_release)

    p = sub.add_parser("done", help="mark a task finished")
    p.add_argument("--task", required=True)
    p.add_argument("--agent", required=True)
    p.add_argument("--result", default=None)
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_done)

    p = sub.add_parser("tasks", help="list tasks")
    p.add_argument("--open", action="store_true", help="only unowned ones")
    p.set_defaults(fn=cmd_tasks)

    p = sub.add_parser("ledger", help="append a validated row to shared/LEDGER.csv")
    p.add_argument("--design-id", dest="design_id", required=True)
    p.add_argument("--class", dest="cls", default=None)
    p.add_argument("--length", default=None)
    p.add_argument("--metric", required=True)
    p.add_argument("--value", required=True)
    p.add_argument("--oracle", required=True)
    p.add_argument("--n-seeds", dest="n_seeds", default=None)
    p.add_argument("--batch", required=True)
    p.add_argument("--agent", required=True)
    p.add_argument("--note", default=None)
    p.add_argument("--path", default=None)
    p.set_defaults(fn=cmd_ledger)

    p = sub.add_parser("status", help="one-screen overview")
    p.set_defaults(fn=cmd_status)

    p = sub.add_parser("selftest", help="verify the bus in a temp dir")
    p.set_defaults(fn=selftest)

    a = ap.parse_args()
    dirs = _dirs(a.bus) if a.cmd != "selftest" else {}
    sys.exit(a.fn(a, dirs))


if __name__ == "__main__":
    main()
