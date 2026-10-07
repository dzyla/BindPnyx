"""Append-only experiment log: what an agent changed, why, and what it measured. One JSON object per line in out/phbind/experiments.jsonl.
An agent that changes a scorer, setting or model without logging it leaves the next agent unable to tell a result from an artifact."""
from __future__ import annotations
import hashlib, json, subprocess, time
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
LOG = REPO / "out" / "phbind" / "experiments.jsonl"

def _commit():
    try: return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip() or None
    except Exception: return None

def log(name: str, hypothesis: str, change: dict, result: dict, decision: str, path: Path | None = None) -> dict:
    """result must contain MEASURED numbers (counts, not adjectives); decision is one of keep / revert / inconclusive."""
    if decision not in ("keep", "revert", "inconclusive"): raise ValueError("decision is keep | revert | inconclusive")
    if not result: raise ValueError("an experiment without a measured result is not an experiment")
    rec = dict(t=time.strftime("%Y-%m-%dT%H:%M:%S"), name=name, hypothesis=hypothesis, change=change, result=result, decision=decision, commit=_commit(),
               config_sha=hashlib.sha1(json.dumps(change, sort_keys=True).encode()).hexdigest()[:10])
    p = Path(path or LOG); p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as fh: fh.write(json.dumps(rec) + "\n")
    return rec

def read(path: Path | None = None) -> list[dict]:
    p = Path(path or LOG)
    return [json.loads(l) for l in open(p)] if p.exists() else []
