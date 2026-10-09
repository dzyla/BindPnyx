#!/usr/bin/env python3
"""Append a numbered note to an append-only forum file without number collisions (docs/AGENT_FORUM.md).

  python scripts/forum_note.py FORUM.md --agent A --headline "result or ask" --body "text"        (or --body-file note.md)

The next number is read from the file at write time (highest '## NOTE <n>' plus one); a timestamped backup of the file is made first;
nothing already in the file is changed."""
import argparse, re, shutil, sys, time
from pathlib import Path

def next_number(text):
    nums = [int(n) for n in re.findall(r"^## NOTE (\d+)\b", text, flags=re.M)]
    return (max(nums) + 1) if nums else 1

def append_note(path, agent, headline, body, now=None):
    p = Path(path); text = p.read_text() if p.exists() else ""
    n = next_number(text)
    if p.exists(): shutil.copy2(p, p.with_name(f".{p.name}.bak_{time.strftime('%Y%m%d_%H%M%S')}"))
    stamp = now or time.strftime("%Y-%m-%d %H:%M %Z").strip()
    note = f"\n\n---\n\n## NOTE {n} — {agent}, {stamp} — **{headline}**\n{body.rstrip()}\n"
    with open(p, "a") as fh: fh.write(note)      # one append; earlier content is never rewritten
    return n

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("file"); ap.add_argument("--agent", required=True); ap.add_argument("--headline", required=True)
    g = ap.add_mutually_exclusive_group(required=True); g.add_argument("--body"); g.add_argument("--body-file")
    a = ap.parse_args(); body = a.body if a.body is not None else Path(a.body_file).read_text()
    print(f"appended NOTE {append_note(a.file, a.agent, a.headline, body)} to {a.file}")
