#!/usr/bin/env python3
"""Pnyx sentinel: an always-running observer so an agent cannot go dark unnoticed.

THE PROBLEM THIS SOLVES
-----------------------
Claude science agents end their session when a round finishes. The session simply stops; nothing
runs an exit handler, nothing releases a claim, nothing posts a handoff. From inside the bus that
is indistinguishable from a crash, and from a healthy agent that is thinking hard between
heartbeats. Measured on one campaign's bus: 9 of 9 claims were held by agents last seen 62-79 h earlier, and
four of them still advertised RUNNING.

**No agent can be relied on to report its own death.** Liveness has to be observed from outside the
agent, by something whose own lifetime does not depend on any agent session. That is this.

TWO FACTS, NEVER CONFLATED
--------------------------
The sentinel reports two independent things and refuses to merge them into one verdict:

  1. **Is the AGENT present?** From heartbeat age alone. Unambiguous, cheap, no ssh.
  2. **Is its WORK alive?** From the declared host: matching processes and whether its output
     directory is still growing. Observed, not self-reported.

     This evidence is **host-level, not agent-level**. Several agents declaring the same host share
     one process list, so `DARK_BUT_WORKING` means "this agent is gone and its host has live work",
     not "this agent's own job is running". That is deliberately the conservative direction: it
     over-protects a GPU rather than freeing one that is busy. To attribute work to one agent, give
     it a `watch_dir` in its observation file and the sweep will use that directory's growth.

The cross product is the whole point, because the dangerous cell is not the obvious one:

  | agent | work    | state                | what it means                                        |
  |-------|---------|----------------------|------------------------------------------------------|
  | LIVE  | running | OK                   | normal                                               |
  | GONE  | running | DARK_BUT_WORKING     | orphaned job still burning a card. DO NOT take this  |
  |       |         |                      | lane's GPU, and do not kill blind: adopt it.         |
  | GONE  | none    | GONE_LANE_FREE       | the lane is genuinely takeable                       |
  | LIVE  | none    | DECLARED_WORK_DEAD   | it thinks it is running something that is not        |
  |       |         |                      | running. Its job died and it has not noticed.        |

`DARK_BUT_WORKING` is why "no heartbeat = gone = take the lane" is not safe on its own, and
`DECLARED_WORK_DEAD` catches the silent-failure mode where a tool exits 0 and produces nothing.

WHAT IT WRITES, AND WHAT IT NEVER DOES
--------------------------------------
Writes `bus/liveness/<agent>.json` (one observation per agent) and `bus/liveness/_sentinel.json`
(its own last run, so a reader can tell whether the OBSERVER is alive -- an observation older than
one interval is not evidence of anything). It posts a bus `warning` when a state changes, at most
once per state transition, so the log does not fill with heartbeats of its own.

It never kills a process, never releases or reassigns a claim, and never edits an agent's own
record. It observes and reports. Taking a lane stays a deliberate act (`pnyx.py takeover`).

WHERE TO RUN IT
---------------
On the machine that exports the shared filesystem, under `systemd --user` (survives logout with
`loginctl enable-linger`) or cron. `--install-systemd` and `--install-cron` print the exact units.

**Never on a cluster login node.** CURC Alpine auto-penalises the account for any python on a
login node, including a few seconds every ten minutes. Run it on the workstation that owns the
export; it reaches the cluster over ssh like anything else.

    python3 scripts/pnyx_sentinel.py --once
    python3 scripts/pnyx_sentinel.py --loop --interval 600
    python3 scripts/pnyx_sentinel.py --install-systemd
    python3 scripts/pnyx_sentinel.py --selftest
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import shlex
import socket
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agent_bus as ab  # noqa: E402
import pnyx  # noqa: E402

SENTINEL = "pnyx-sentinel"
# structure-prediction and design engines; bracketed so the grep never matches its own ssh command
ENGINES = r"[p]rotenix pred|[b]oltz predict|[r]un_rfd|[p]xdesign|[c]olabfold|[r]un_af3"
STATES = ("OK", "DARK_BUT_WORKING", "GONE_LANE_FREE", "DECLARED_WORK_DEAD", "UNKNOWN")


def _liveness_dir(dirs):
    p = os.path.join(dirs["bus"], "liveness")
    os.makedirs(p, exist_ok=True)
    return p


def probe_host(host, timeout=45):
    """One ssh per host: matching processes, GPU utilisation. Returns {} when unreachable.

    Unreachable is NOT 'no work'. A network blip must never be reported as a dead job, so the
    caller sees `reachable: False` and the state stays UNKNOWN.
    """
    if host in ("localhost", socket.gethostname(), None, "", "?"):
        cmd = ["bash", "-lc", f"ps -eo pid,etime,args --no-headers | grep -E '{ENGINES}' || true"]
    else:
        cmd = ["ssh", "-o", "BatchMode=yes", "-o", f"ConnectTimeout={min(15, timeout)}", host,
               f"ps -eo pid,etime,args --no-headers | grep -E '{ENGINES}' || true"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (subprocess.TimeoutExpired, OSError):
        return {"reachable": False, "procs": [], "gpu_util": None}
    if r.returncode not in (0, 1):
        return {"reachable": False, "procs": [], "gpu_util": None}
    procs = [ln.strip() for ln in r.stdout.splitlines() if ln.strip()]
    return {"reachable": True, "procs": procs, "gpu_util": None}


def artefact_count(path):
    """How many files are under a declared output directory, or None if it is not one we can see."""
    if not path or not os.path.isdir(path):
        return None
    n = 0
    for _root, _d, files in os.walk(path):
        n += len(files)
        if n > 200000:
            break
    return n


def classify(agent_rec, host_info, lease_h, prev=None):
    """Agent presence x work presence -> one of STATES, plus the evidence for it."""
    state_agent, age = pnyx.agent_state(agent_rec, lease_h)
    declared = bool((agent_rec or {}).get("running")) and \
        "idle" not in str((agent_rec or {}).get("running", "")).lower()
    if state_agent == "UNKNOWN" or not host_info.get("reachable", False):
        return "UNKNOWN", age, declared, "agent presence or host unreadable"
    work = bool(host_info.get("procs"))
    # a growing output directory counts as work even when the process name does not match
    if not work and prev is not None:
        now_n, was_n = prev.get("artefacts_now"), prev.get("artefacts_prev")
        if isinstance(now_n, int) and isinstance(was_n, int) and now_n > was_n:
            work = True
    if state_agent == "LIVE":
        return ("OK" if (work or not declared) else "DECLARED_WORK_DEAD"), age, declared, \
            f"agent LIVE; its host has {len(host_info['procs'])} matching process(es)"
    return ("DARK_BUT_WORKING" if work else "GONE_LANE_FREE"), age, declared, \
        f"agent GONE {0 if age is None else round(age, 1)} h; its host has " \
        f"{len(host_info['procs'])} matching process(es)"


def sweep(dirs, lease_h, prober=probe_host, post=True, counter=artefact_count):
    """One pass. Returns the list of observations and posts a warning on each state change."""
    ld = _liveness_dir(dirs)
    agents = pnyx._agents(dirs)
    claims = {}
    for c in pnyx._claims(dirs):
        claims.setdefault(c.get("agent"), []).append(c.get("task"))

    hosts = sorted({(r.get("host") or "localhost") for r in agents.values()})
    host_info = {h: prober(h) for h in hosts}

    obs, changed = [], []
    for name, rec in sorted(agents.items()):
        if name == SENTINEL:
            continue
        path = os.path.join(ld, f"{name}.json")
        prev = ab._read_json(path) or {}
        host = rec.get("host") or "localhost"
        hi = host_info.get(host, {"reachable": False, "procs": []})
        outdir = prev.get("watch_dir")
        n_now = counter(outdir)
        ctx = {"artefacts_now": n_now, "artefacts_prev": prev.get("artefacts_now")}
        state, age, declared, why = classify(rec, hi, lease_h, ctx)
        o = {"agent": name, "observed_at": ab._now_iso(), "observer": socket.gethostname(),
             "state": state, "self_last_seen": rec.get("last_seen"),
             "self_age_h": None if age is None else round(age, 2), "lease_h": lease_h,
             "host": host, "host_reachable": hi.get("reachable"),
             "n_matching_procs": len(hi.get("procs") or []),
             "procs": (hi.get("procs") or [])[:6],
             "declared_running": rec.get("running"), "stop_cmd": rec.get("stop_cmd"),
             "claims_held": claims.get(name, []), "watch_dir": outdir,
             "artefacts_now": n_now, "artefacts_prev": prev.get("artefacts_now"),
             "why": why, "prev_state": prev.get("state"),
             "work_attribution": "host-level: matching processes on the declared host, NOT proven "
                                 "to belong to this agent. Several agents sharing a host share "
                                 "this evidence."}
        ab._write_atomic(path, o)
        obs.append(o)
        if prev.get("state") != state:
            changed.append(o)

    ab._write_atomic(os.path.join(ld, "_sentinel.json"),
                     {"agent": SENTINEL, "last_run": ab._now_iso(),
                      "observer": socket.gethostname(), "lease_h": lease_h,
                      "n_agents": len(obs),
                      "counts": {s: sum(1 for o in obs if o["state"] == s) for s in STATES}})
    if post and changed:
        _post_changes(dirs, changed, lease_h)
    return obs


def _post_changes(dirs, changed, lease_h):
    """One bus warning per sweep that saw a transition. Never one per agent per sweep."""
    lines = []
    for o in changed:
        lines.append(f"{o['agent']}: {o['prev_state'] or 'new'} -> {o['state']}  "
                     f"({o['why']}; host {o['host']}"
                     f"{', claims ' + ','.join(o['claims_held']) if o['claims_held'] else ''})")
        if o["state"] == "DARK_BUT_WORKING":
            lines.append(f"    its job is ALIVE and orphaned. Do NOT take its GPU and do not kill "
                         f"blind. Its declared stop: {o['stop_cmd'] or 'NOT DECLARED'}")
        if o["state"] == "GONE_LANE_FREE" and o["claims_held"]:
            lines.append(f"    lane(s) takeable: python3 scripts/pnyx.py takeover --task "
                         f"{o['claims_held'][0]} --agent <you>")
        if o["state"] == "DECLARED_WORK_DEAD":
            lines.append(f"    it is LIVE and thinks it is running "
                         f"{str(o['declared_running'])[:60]!r}, which is not running. Its job died "
                         f"and it has not noticed; these tools exit 0 on failure.")
    body = ("Observed from outside the agents, by the sentinel. Agent presence is heartbeat age; "
            f"work presence is processes on the declared host. Lease {lease_h:g} h.\n\n"
            + "\n".join(lines)
            + "\n\nFull observations: bus/liveness/<agent>.json. Observer heartbeat: "
              "bus/liveness/_sentinel.json — an observation older than one interval is not "
              "evidence.\nGONE frees a LANE; it does not grant CONSENT.")

    class A:
        pass
    a = A()
    a.__dict__.update(sender=SENTINEL, to=None, topic="warning",
                      subject=f"liveness: {', '.join(o['agent'] + ' ' + o['state'] for o in changed)[:150]}",
                      body=body, evidence=None, refs=["bus/liveness"], reply_to=None,
                      priority="high" if any(o["state"] == "DARK_BUT_WORKING" for o in changed)
                      else "normal",
                      reply_by_h=None, fallback=None)
    try:
        pnyx.cmd_post(a, dirs)
    except SystemExit:
        pass


def ensure_registered(dirs):
    """The observer registers and heartbeats like anyone else, so `pnyx status` can see it."""
    p = os.path.join(dirs["agents"], f"{SENTINEL}.json")
    rec = ab._read_json(p) or {"name": SENTINEL, "registered": ab._now_iso()}
    rec.update({"name": SENTINEL, "role": "liveness observer", "host": socket.gethostname(),
                "last_seen": ab._now_iso(), "running": "liveness sweep", "gpus": "none",
                "stop_cmd": "systemctl --user stop pnyx-sentinel.timer"})
    ab._write_atomic(p, rec)


SYSTEMD = """\
# ~/.config/systemd/user/pnyx-sentinel.service
[Unit]
Description=Pnyx liveness sentinel (observes agents from outside their sessions)

[Service]
Type=oneshot
WorkingDirectory={repo}
ExecStart=/usr/bin/python3 {repo}/scripts/pnyx_sentinel.py --once --interval {interval}

# ~/.config/systemd/user/pnyx-sentinel.timer
[Unit]
Description=Run the Pnyx liveness sentinel every {interval_min} min

[Timer]
OnBootSec=2min
OnUnitActiveSec={interval}s
AccuracySec=30s
Persistent=true

[Install]
WantedBy=timers.target

# install, and make it survive logout:
#   loginctl enable-linger $USER
#   systemctl --user daemon-reload
#   systemctl --user enable --now pnyx-sentinel.timer
#   systemctl --user list-timers pnyx-sentinel.timer
"""

CRON = """\
# crontab -e   (fallback when systemd --user is not available)
*/{interval_min} * * * * cd {repo} && /usr/bin/python3 scripts/pnyx_sentinel.py --once \\
  --interval {interval} >> {repo}/bus/liveness/sentinel.log 2>&1

# NEVER add this to a cluster login node's crontab. CURC Alpine auto-penalises the account for
# python on a login node, and a few seconds every {interval_min} minutes still counts.
"""


def selftest():
    import shutil
    import tempfile
    tmp = tempfile.mkdtemp(prefix="sentinel-selftest-")
    fails = []

    def ck(c, m):
        if not c:
            fails.append(m)

    try:
        d = ab._dirs(os.path.join(tmp, "bus"))
        now = _dt.datetime.now().astimezone()

        def seen(h):
            return (now - _dt.timedelta(hours=h)).isoformat(timespec="seconds")

        busy = {"reachable": True, "procs": ["123 10:00 protenix predict x"]}
        quiet = {"reachable": True, "procs": []}
        down = {"reachable": False, "procs": []}
        live = {"last_seen": seen(0.2), "running": "scoring wave 2", "host": "h"}
        dead = {"last_seen": seen(70), "running": "scoring wave 2", "host": "h"}
        idle_live = {"last_seen": seen(0.2), "running": "idle - cards released", "host": "h"}

        ck(classify(live, busy, 6)[0] == "OK", "live agent with a running job is not OK")
        ck(classify(dead, busy, 6)[0] == "DARK_BUT_WORKING",
           "a gone agent whose job still runs MUST be DARK_BUT_WORKING, never lane-free")
        ck(classify(dead, quiet, 6)[0] == "GONE_LANE_FREE",
           "a gone agent with no job should free its lane")
        ck(classify(live, quiet, 6)[0] == "DECLARED_WORK_DEAD",
           "a live agent whose declared job vanished is not flagged")
        ck(classify(idle_live, quiet, 6)[0] == "OK",
           "an idle live agent must not be flagged as having dead work")
        ck(classify(dead, down, 6)[0] == "UNKNOWN",
           "an UNREACHABLE host must be UNKNOWN, never GONE_LANE_FREE")
        ck(classify({"last_seen": "nonsense"}, busy, 6)[0] == "UNKNOWN",
           "an unparseable heartbeat must be UNKNOWN")
        # lease_h must move the verdict: the same agent and host read GONE_LANE_FREE at a 6 h lease
        # and DECLARED_WORK_DEAD at a 999 h one, because it is then still LIVE with a vanished job
        ck(classify(dead, quiet, 6)[0] == "GONE_LANE_FREE" and
           classify(dead, quiet, 999)[0] == "DECLARED_WORK_DEAD",
           "lease_h does not change the verdict: %s / %s"
           % (classify(dead, quiet, 6)[0], classify(dead, quiet, 999)[0]))
        # a growing output dir counts as work even when no process name matches
        ck(classify(dead, quiet, 6, {"artefacts_now": 9, "artefacts_prev": 4})[0]
           == "DARK_BUT_WORKING", "a growing output directory was not read as live work")
        ck(classify(dead, quiet, 6, {"artefacts_now": 4, "artefacts_prev": 4})[0]
           == "GONE_LANE_FREE", "a static output directory was read as live work")

        for nm, rec in (("live1", live), ("dead1", dead)):
            ab._write_atomic(os.path.join(d["agents"], f"{nm}.json"), dict(rec, name=nm, role="r"))
        cl, _, _ = ab._task_paths(d, "lane1")
        with open(cl, "w") as fh:
            json.dump({"task": "lane1", "agent": "dead1", "claimed": seen(70)}, fh)

        obs = sweep(d, 6.0, prober=lambda h: quiet, post=False, counter=lambda p: None)
        by = {o["agent"]: o for o in obs}
        ck(set(by) == {"live1", "dead1"}, f"sweep saw {sorted(by)}")
        ck(by["dead1"]["state"] == "GONE_LANE_FREE", "dead1 not lane-free")
        ck(by["dead1"]["claims_held"] == ["lane1"], "the held lane was not reported")
        ck(by["live1"]["state"] == "DECLARED_WORK_DEAD", "live1 misclassified")
        ck(os.path.exists(os.path.join(d["bus"], "liveness", "_sentinel.json")),
           "the observer did not record its own run")

        n0 = len(ab._load_messages(d))
        sweep(d, 6.0, prober=lambda h: quiet, post=False, counter=lambda p: None)
        ck(len(ab._load_messages(d)) == n0, "a sweep with no transition must post nothing")

        ensure_registered(d)
        ab._write_atomic(os.path.join(d["agents"], "dead1.json"),
                         dict(dead, name="dead1", role="r", last_seen=seen(0.1)))
        before = len(ab._load_messages(d))
        sweep(d, 6.0, prober=lambda h: busy, post=True, counter=lambda p: None)
        ck(len(ab._load_messages(d)) == before + 1,
           "a state transition should post exactly one warning")
        ck(SENTINEL in pnyx._agents(d), "the sentinel did not register itself")
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
    ap.add_argument("--lease-hours", dest="lease_hours", type=float, default=pnyx.LEASE_H)
    ap.add_argument("--interval", type=int, default=600, help="seconds between sweeps in --loop")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--no-post", action="store_true", help="observe and write, post nothing")
    ap.add_argument("--install-systemd", action="store_true")
    ap.add_argument("--install-cron", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        return selftest()
    if a.install_systemd or a.install_cron:
        tpl = SYSTEMD if a.install_systemd else CRON
        print(tpl.format(repo=ab.REPO, interval=a.interval,
                         interval_min=max(1, a.interval // 60)))
        return 0
    if not (a.once or a.loop):
        ap.error("pass --once, --loop, --install-systemd, --install-cron or --selftest")

    dirs = ab._dirs(a.bus)
    while True:
        ensure_registered(dirs)
        obs = sweep(dirs, a.lease_hours, post=not a.no_post)
        for o in obs:
            age = "never" if o["self_age_h"] is None else f"{o['self_age_h']:.1f} h"
            print(f"{o['agent']:<16}{o['state']:<20}seen {age:<10}"
                  f"procs {o['n_matching_procs']}  claims {len(o['claims_held'])}")
        if not a.loop:
            return 0
        time.sleep(max(60, a.interval))


if __name__ == "__main__":
    sys.exit(main())
