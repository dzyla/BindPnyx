#!/usr/bin/env python3
"""Pnyx: addressed messages, liveness leases and read receipts on top of the existing bus.

WHY A SECOND ENTRY POINT INSTEAD OF EDITING agent_bus.py
--------------------------------------------------------
Agents are mid-lane with `B="python3 scripts/agent_bus.py"` in their shells and claims they believe
they hold. Changing claim or cursor semantics underneath them is the failure this is meant to fix,
not a fix. So this is ADDITIVE: same bus directory, same message and claim files, same validation
gates. `agent_bus.py` keeps working unchanged and nothing it writes becomes invalid. After the
deadline, `agent_bus.py` becomes a shim and this becomes the only entry point.

WHAT IT ADDS, and the measurement behind each
---------------------------------------------
Measured on one campaign's bus at 260 messages, 9 registered agents:

  * 7 of 8 read cursors were 177-251 messages behind  -> `--to`, `inbox`, `receipts`
  * 9 of 9 claims held by agents last seen 62-79 h ago -> leases: lapsed presence = GONE,
    held claims of a GONE agent read EXPIRED and are takeable via `takeover`
  * 6 of 15 senders never registered, one claim was
    created under a name last seen 69 h ago            -> `post` refuses an unregistered sender
  * median body 3,118 chars, ~225k tokens of log       -> body cap, with --refs required above it

THE LIVENESS RULE THIS IMPLEMENTS
---------------------------------
An agent that has not heartbeated inside its lease is treated as GONE. Its lane is takeable. This
is an operator decision (2026-10-15) and it replaces waiting indefinitely on a reply.

Two things it does NOT mean, and conflating them would cause real damage:

  * GONE is not agreement. A shared rule, a threshold or a panel row still does not change on
    silence. Absence frees a LANE; it does not grant CONSENT.
  * GONE is a statement about the agent, not about its processes. A dead session's CUDA job can
    still be burning a card. `takeover` prints the host check to run first and the holder's own
    declared stop command; it does not kill anything and never will.

    python3 scripts/pnyx.py status
    python3 scripts/pnyx.py inbox --agent <you>
    python3 scripts/pnyx.py post --from <you> --to <them> --topic question --subject ... --body -
    python3 scripts/pnyx.py receipts --agent <you>
    python3 scripts/pnyx.py takeover --task <t> --agent <you>
    python3 scripts/pnyx.py selftest
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import socket
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agent_bus as ab  # noqa: E402  the bus is the storage layer; this is a second face on it

LEASE_H = 6.0
BODY_WARN = 1200
BODY_MAX = 2000


# --------------------------------------------------------------------------- liveness
def _age_h(ts):
    """Hours since an ISO timestamp, or None if it cannot be parsed."""
    if not ts:
        return None
    try:
        then = _dt.datetime.fromisoformat(ts)
    except (TypeError, ValueError):
        return None
    now = _dt.datetime.now().astimezone()
    if then.tzinfo is None:
        then = then.replace(tzinfo=now.tzinfo)
    return (now - then).total_seconds() / 3600.0


def agent_state(rec, lease_h=LEASE_H):
    """LIVE / GONE / UNKNOWN for one agent record, plus the age in hours.

    UNKNOWN is deliberate: an unparseable or missing last_seen must not read as LIVE (which would
    protect a dead claim) nor as GONE (which would license taking a lane on a parsing bug).
    """
    age = _age_h((rec or {}).get("last_seen"))
    if age is None:
        return "UNKNOWN", None
    return ("LIVE" if age <= lease_h else "GONE"), age


def _agents(dirs):
    out = {}
    for f in sorted(os.listdir(dirs["agents"])):
        if f.endswith(".json"):
            r = ab._read_json(os.path.join(dirs["agents"], f))
            if r and r.get("name"):
                out[r["name"]] = r
    return out


def _claims(dirs):
    out = []
    for f in sorted(os.listdir(dirs["tasks"])):
        if f.endswith(".claim.json"):
            r = ab._read_json(os.path.join(dirs["tasks"], f))
            if r:
                out.append(r)
    return out


def _cursor(dirs, name):
    p = os.path.join(dirs["cursors"], f"{name}.txt")
    try:
        return open(p).read().strip()
    except OSError:
        return ""


def _addressed_to(m, name):
    to = m.get("to") or []
    return name in to if isinstance(to, list) else False


# --------------------------------------------------------------------------- commands
def cmd_status(a, dirs):
    lease = a.lease_hours
    msgs = ab._load_messages(dirs)
    agents = _agents(dirs)
    print(f"pnyx — {dirs['bus']}\n{ab._now_iso()}   lease {lease:g} h\n")

    print(f"AGENTS ({len(agents)})")
    gone = set()
    for n, r in sorted(agents.items()):
        st, age = agent_state(r, lease)
        if st == "GONE":
            gone.add(n)
        age_s = "never" if age is None else f"{age:.1f} h ago"
        print(f"  {n:<16}{st:<8}seen {age_s:<12}{str(r.get('role'))[:38]}")
        run = r.get("running")
        if run and "idle" in str(run).lower():
            run = None          # an agent that declared itself idle is not claiming to be busy
        if run and st == "GONE":
            print(f"       declared RUNNING but GONE — probably dead: {str(run)[:88]}")
            print(f"       its own stop command: {r.get('stop_cmd') or 'NOT DECLARED'}")
        elif run:
            print(f"       RUNNING: {str(run)[:88]}")

    cl = _claims(dirs)
    exp = [c for c in cl if c.get("agent") in gone]
    print(f"\nCLAIMS ({len(cl)})   {len(exp)} EXPIRED (holder GONE)")
    for c in sorted(cl, key=lambda x: str(x.get("task"))):
        holder = c.get("agent")
        st, age = agent_state(agents.get(holder), lease)
        mark = "EXPIRED" if st == "GONE" else ("held" if st == "LIVE" else "unknown")
        hage = _age_h(c.get("claimed"))
        print(f"  [{mark:<8}] {str(c.get('task')):<30}{str(holder):<14}"
              f"claimed {('?' if hage is None else f'{hage:.1f} h'):>8} ago, holder "
              f"{('?' if age is None else f'{age:.1f} h'):>8} ago")
    if exp:
        print("\n  Takeable:  python3 scripts/pnyx.py takeover --task <t> --agent <you>")
        print("  It prints the host check to run BEFORE you launch anything. 'GONE' describes the")
        print("  agent, not its GPU job.")

    unread = [m for m in msgs if not a.agent or m["id"] > _cursor(dirs, a.agent)]
    print(f"\nMESSAGES ({len(msgs)})")
    if a.agent:
        mine = [m for m in unread if _addressed_to(m, a.agent)]
        print(f"  unread by {a.agent}: {len(unread)}   ADDRESSED TO YOU and unread: {len(mine)}")
        for m in mine:
            print(f"    ! [{m['topic']}] {m['from']}: {m['subject'][:84]}")
    retr = [m for m in msgs if m["topic"] == "retraction"]
    ur = [m for m in retr if not a.agent or m["id"] > _cursor(dirs, a.agent)]
    if ur:
        print(f"  {len(ur)} UNREAD RETRACTION(S) — read before trusting any earlier result:")
        for m in ur[-8:]:
            print(f"    {m['ts'][:16]} {m['from']}: {m['subject'][:80]}")
    print("\nProtocol and post-mortem: docs/AGENT_BUS.md")
    return 0


def cmd_inbox(a, dirs):
    """Full bodies for what you must read; one line for everything else."""
    msgs = ab._load_messages(dirs)
    last = "" if a.all else _cursor(dirs, a.agent)
    sel = [m for m in msgs if m["id"] > last]
    must = [m for m in sel if _addressed_to(m, a.agent)
            or m["topic"] in ("retraction", "warning")
            or m.get("priority") == "high"]
    rest = [m for m in sel if m not in must]
    for m in must:
        why = ("addressed to you" if _addressed_to(m, a.agent) else m["topic"])
        print(f"\n{'=' * 78}\n[{m['topic']}] {m['subject']}\n  from {m['from']}  {m['ts']}  "
              f"id {m['id']}  ({why})")
        if m.get("to"):
            print(f"  to: {', '.join(m['to'])}")
        if m.get("evidence"):
            print(f"  evidence: {m['evidence']}")
        if m.get("refs"):
            print(f"  refs: {', '.join(m['refs'])}")
        print(f"{'-' * 78}\n{m['body'].rstrip()}")
    if rest:
        print(f"\n{'=' * 78}\nDIGEST — {len(rest)} other message(s), subjects only:")
        for m in rest:
            print(f"  {m['ts'][:16]} [{m['topic']:<10}] {m['from']:<14} {m['subject'][:72]}")
    print(f"\n{len(must)} in full, {len(rest)} in digest")
    if sel and not a.no_cursor:
        with open(os.path.join(dirs["cursors"], f"{a.agent}.txt"), "w") as fh:
            fh.write(sel[-1]["id"])
    return 0


def cmd_post(a, dirs):
    """Same gates as the bus, plus: sender must be registered, body is capped, --to is recorded."""
    agents = _agents(dirs)
    if a.sender not in agents:
        print(f"refusing: '{a.sender}' is not registered. Run `register` first — an unregistered "
              f"sender makes presence meaningless and cannot be told it is GONE.", file=sys.stderr)
        return 1
    unknown = [t for t in (a.to or []) if t not in agents]
    if unknown:
        print(f"refusing: unknown addressee(s) {unknown}. Registered: {sorted(agents)}",
              file=sys.stderr)
        return 1
    body = sys.stdin.read() if a.body == "-" else a.body
    if not body or not body.strip():
        print("refusing to post an empty body", file=sys.stderr)
        return 1
    if a.topic in ("result", "retraction") and not a.evidence:
        print(f"--evidence is required for topic '{a.topic}'. {ab.EVIDENCE_HINT}", file=sys.stderr)
        return 1
    n = len(body)
    if n > BODY_MAX and not a.refs:
        print(f"refusing: body is {n} chars (cap {BODY_MAX}) and no --refs. Write the long form to "
              f"a file and cite it. Reading is paid by every other agent; writing is not.",
              file=sys.stderr)
        return 1
    if n > BODY_WARN:
        print(f"note: body is {n} chars. A verdict plus a pointer reads better than an essay.",
              file=sys.stderr)
    mid = ab._new_id()
    msg = {"id": mid, "ts": ab._now_iso(), "from": a.sender, "to": a.to or [],
           "topic": a.topic, "subject": a.subject, "body": body, "evidence": a.evidence,
           "refs": a.refs or [], "reply_to": a.reply_to, "priority": a.priority,
           "reply_by_h": a.reply_by_h, "fallback": a.fallback}
    ab._write_atomic(os.path.join(dirs["messages"], f"{mid}.json"), msg)
    print(f"posted {mid}  [{a.topic}] {a.subject}")
    if a.to and not a.fallback:
        print("note: you addressed someone but declared no --fallback. Say what you will do if "
              "nobody answers; silence is the normal case here.", file=sys.stderr)
    return 0


def cmd_receipts(a, dirs):
    """Has anyone read what you addressed to them? The cursors already know."""
    msgs = [m for m in ab._load_messages(dirs) if m["from"] == a.agent and m.get("to")]
    if not msgs:
        print(f"{a.agent} has posted no addressed messages (no --to). Nothing to chase.")
        return 0
    agents = _agents(dirs)
    for m in msgs[-a.limit:]:
        age = _age_h(m["ts"])
        print(f"\n[{m['topic']}] {m['subject'][:72]}\n  sent {('?' if age is None else f'{age:.1f}')}"
              f" h ago to {', '.join(m['to'])}")
        for t in m["to"]:
            read = _cursor(dirs, t) >= m["id"]
            st, ag = agent_state(agents.get(t), a.lease_hours)
            note = "READ" if read else "NOT READ"
            print(f"    {t:<16}{note:<10}{st}"
                  + ("" if ag is None else f", last seen {ag:.1f} h ago"))
            if not read and st == "GONE":
                print(f"      -> treat as gone. Do not wait. "
                      f"{'Fallback: ' + m['fallback'] if m.get('fallback') else 'No fallback declared.'}")
    return 0


def cmd_takeover(a, dirs):
    """Take a lane whose holder is GONE. Refuses while the holder is LIVE."""
    claim, done, _ = ab._task_paths(dirs, a.task)
    if os.path.exists(done):
        print(f"'{a.task}' is already done", file=sys.stderr)
        return 1
    rec = ab._read_json(claim)
    if rec is None:
        print(f"'{a.task}' is not claimed — just claim it:\n"
              f"  python3 scripts/agent_bus.py claim --task {a.task} --agent {a.agent}",
              file=sys.stderr)
        return 1
    holder = rec.get("agent")
    if holder == a.agent:
        print(f"you already hold '{a.task}'")
        return 0
    agents = _agents(dirs)
    hrec = agents.get(holder) or {}
    st, age = agent_state(hrec, a.lease_hours)
    if st != "GONE":
        print(f"REFUSED: '{a.task}' is held by {holder}, which is {st}"
              + ("" if age is None else f" (seen {age:.1f} h ago)")
              + f".\nThe lease is {a.lease_hours:g} h. Ask them, state a fallback, and do the "
                f"fallback if nobody answers.", file=sys.stderr)
        return 1
    host = hrec.get("host") or "?"
    print(f"{holder} is GONE (last seen {age:.1f} h ago, lease {a.lease_hours:g} h).")
    print(f"\nBEFORE YOU LAUNCH ANYTHING, check its host for live work. 'GONE' describes the agent,")
    print(f"not its GPU job, and a dead session's CUDA process can still hold a card:")
    print(f"  ssh -o BatchMode=yes {host} \"nvidia-smi; ps -eo pid,etime,args --no-headers | "
          f"grep -E '[p]rotenix pred|[b]oltz predict|[r]un_rfd'\"")
    print(f"  its own declared stop command: {hrec.get('stop_cmd') or 'NOT DECLARED'}")
    print(f"  it declared running: {hrec.get('running') or '-'}")
    print("\nTake PIDs from ps and kill those PIDs. Never a broad pkill pattern — one has already")
    print("matched the shell issuing it.")
    if not a.yes:
        print(f"\nNothing changed. Re-run with --yes to take '{a.task}'.")
        return 0
    ab._write_atomic(os.path.join(dirs["tasks"], f"{a.task}.takeover-{ab._new_id()}.json"),
                     {"task": a.task, "from_agent": holder, "to_agent": a.agent,
                      "holder_last_seen": hrec.get("last_seen"), "holder_age_h": round(age, 2),
                      "lease_h": a.lease_hours, "at": ab._now_iso(), "note": a.note})
    os.unlink(claim)
    try:
        fd = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o664)
    except FileExistsError:
        print("someone claimed it in the same instant; re-read status", file=sys.stderr)
        return 1
    with os.fdopen(fd, "w") as fh:
        json.dump({"task": a.task, "agent": a.agent, "claimed": ab._now_iso(),
                   "host": socket.gethostname(),
                   "note": f"takeover from {holder}, GONE {age:.1f} h; {a.note or ''}"},
                  fh, indent=1, sort_keys=True)
        fh.write("\n")
    print(f"\ntook '{a.task}' from {holder}. Post it — a takeover nobody is told about is a "
          f"duplicate waiting to happen.")
    return 0


def selftest(_a, _dirs):
    import shutil
    import tempfile
    tmp = tempfile.mkdtemp(prefix="pnyx-selftest-")
    fails = []

    def ck(c, m):
        if not c:
            fails.append(m)

    class A:
        pass

    try:
        d = ab._dirs(os.path.join(tmp, "bus"))
        now = _dt.datetime.now().astimezone()

        def seen(h):
            return (now - _dt.timedelta(hours=h)).isoformat(timespec="seconds")

        ck(agent_state({"last_seen": seen(1)})[0] == "LIVE", "a 1 h-old heartbeat is not LIVE")
        ck(agent_state({"last_seen": seen(70)})[0] == "GONE", "a 70 h-old heartbeat is not GONE")
        ck(agent_state({"last_seen": seen(70)}, lease_h=99)[0] == "LIVE",
           "lease_hours is ignored")
        ck(agent_state({})[0] == "UNKNOWN", "a missing last_seen must be UNKNOWN, never LIVE/GONE")
        ck(agent_state({"last_seen": "not-a-date"})[0] == "UNKNOWN",
           "an unparseable last_seen must be UNKNOWN")

        for nm, h in (("live1", 0.5), ("dead1", 70.0)):
            ab._write_atomic(os.path.join(d["agents"], f"{nm}.json"),
                             {"name": nm, "role": "r", "host": "h", "last_seen": seen(h),
                              "running": "something", "stop_cmd": "kill $(cat x.pid)"})

        base = dict(sender="live1", to=None, subject="s", body="b", evidence=None, refs=None,
                    reply_to=None, priority="normal", reply_by_h=None, fallback=None)
        a = A(); a.__dict__.update(base, topic="plan")
        ck(cmd_post(a, d) == 0, "a registered sender could not post")
        a = A(); a.__dict__.update(base, topic="plan", sender="ghost")
        ck(cmd_post(a, d) == 1, "an UNREGISTERED sender must be refused")
        a = A(); a.__dict__.update(base, topic="plan", to=["nobody"])
        ck(cmd_post(a, d) == 1, "an unknown addressee must be refused")
        a = A(); a.__dict__.update(base, topic="result")
        ck(cmd_post(a, d) == 1, "a result without evidence must still be refused")
        a = A(); a.__dict__.update(base, topic="plan", body="x" * (BODY_MAX + 1))
        ck(cmd_post(a, d) == 1, "an over-cap body with no --refs must be refused")
        a = A(); a.__dict__.update(base, topic="plan", body="x" * (BODY_MAX + 1), refs=["f"])
        ck(cmd_post(a, d) == 0, "an over-cap body WITH --refs must be allowed")
        a = A(); a.__dict__.update(base, topic="question", to=["dead1"], fallback="proceed alone")
        ck(cmd_post(a, d) == 0, "an addressed question was refused")

        # takeover: refuses a LIVE holder, allows a GONE one, and only with --yes
        for who in ("live1", "dead1"):
            cl, _, _ = ab._task_paths(d, f"t_{who}")
            with open(cl, "w") as fh:
                json.dump({"task": f"t_{who}", "agent": who, "claimed": seen(70)}, fh)
        a = A(); a.__dict__.update(task="t_live1", agent="live1", lease_hours=LEASE_H,
                                   yes=True, note=None)
        ck(cmd_takeover(a, d) == 0, "the holder re-taking its own task should be a no-op success")
        a = A(); a.__dict__.update(task="t_live1", agent="dead1", lease_hours=LEASE_H,
                                   yes=True, note=None)
        ck(cmd_takeover(a, d) == 1, "taking a task from a LIVE holder MUST be refused")
        a = A(); a.__dict__.update(task="t_dead1", agent="live1", lease_hours=LEASE_H,
                                   yes=False, note=None)
        ck(cmd_takeover(a, d) == 0, "a dry-run takeover should succeed and change nothing")
        ck((ab._read_json(ab._task_paths(d, "t_dead1")[0]) or {}).get("agent") == "dead1",
           "a takeover WITHOUT --yes must not reassign the claim")
        a = A(); a.__dict__.update(task="t_dead1", agent="live1", lease_hours=LEASE_H,
                                   yes=True, note="n")
        ck(cmd_takeover(a, d) == 0, "taking a GONE holder's task failed")
        ck((ab._read_json(ab._task_paths(d, "t_dead1")[0]) or {}).get("agent") == "live1",
           "the claim was not reassigned")
        ck(any(f.startswith("t_dead1.takeover-") for f in os.listdir(d["tasks"])),
           "no takeover record was written")

        # inbox: addressed + retraction/warning in full, the rest in digest; cursor advances
        a = A(); a.__dict__.update(agent="dead1", all=True, no_cursor=False)
        ck(cmd_inbox(a, d) == 0, "inbox failed")
        ck(os.path.exists(os.path.join(d["cursors"], "dead1.txt")), "inbox did not set a cursor")
        a = A(); a.__dict__.update(agent="dead1", all=False, no_cursor=False)
        cmd_inbox(a, d)
        a = A(); a.__dict__.update(agent="live1", limit=10, lease_hours=LEASE_H)
        ck(cmd_receipts(a, d) == 0, "receipts failed")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    for f in fails:
        print(f"FAIL {f}", file=sys.stderr)
    print(f"selftest: {'PASS' if not fails else str(len(fails)) + ' FAILURES'}", file=sys.stderr)
    return 1 if fails else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bus", default=ab.DEFAULT_BUS)
    ap.add_argument("--lease-hours", dest="lease_hours", type=float, default=LEASE_H,
                    help=f"no heartbeat inside this window = GONE (default {LEASE_H:g})")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("status", help="who is live, which lanes are expired, what you have not read")
    p.add_argument("--agent", default=None, help="your name, for unread counts")
    p.set_defaults(fn=cmd_status)

    p = sub.add_parser("inbox", help="full bodies for what you must read, digest for the rest")
    p.add_argument("--agent", required=True)
    p.add_argument("--all", action="store_true")
    p.add_argument("--no-cursor", action="store_true")
    p.set_defaults(fn=cmd_inbox)

    p = sub.add_parser("post", help="post, addressed, with the gates plus a body cap")
    p.add_argument("--from", dest="sender", required=True)
    p.add_argument("--to", nargs="*", default=None, help="registered agent names")
    p.add_argument("--topic", required=True, choices=ab.TOPICS)
    p.add_argument("--subject", required=True)
    p.add_argument("--body", required=True, help="text, or '-' for stdin")
    p.add_argument("--evidence", default=None, help=ab.EVIDENCE_HINT)
    p.add_argument("--refs", nargs="*", default=None)
    p.add_argument("--reply-to", dest="reply_to", default=None)
    p.add_argument("--reply-by-h", dest="reply_by_h", type=float, default=None,
                   help="hours you will wait before acting on your fallback")
    p.add_argument("--fallback", default=None, help="what you do if nobody answers")
    p.add_argument("--priority", default="normal", choices=["normal", "high"])
    p.set_defaults(fn=cmd_post)

    p = sub.add_parser("receipts", help="who has not read what you addressed to them")
    p.add_argument("--agent", required=True)
    p.add_argument("--limit", type=int, default=10)
    p.set_defaults(fn=cmd_receipts)

    p = sub.add_parser("takeover", help="take a lane whose holder is GONE")
    p.add_argument("--task", required=True)
    p.add_argument("--agent", required=True)
    p.add_argument("--note", default=None)
    p.add_argument("--yes", action="store_true", help="without this it is a dry run")
    p.set_defaults(fn=cmd_takeover)

    p = sub.add_parser("selftest", help="verify in a temp dir; touches nothing real")
    p.set_defaults(fn=selftest)

    a = ap.parse_args()
    dirs = ab._dirs(a.bus) if a.cmd != "selftest" else {}
    sys.exit(a.fn(a, dirs))


if __name__ == "__main__":
    main()
