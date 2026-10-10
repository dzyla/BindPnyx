#!/usr/bin/env python3
"""Link a private/bulky workspace into a fresh clone, so the clone can start work immediately.

  python scripts/link_workspace.py [WORKSPACE] [--dry-run]

WORKSPACE (default: $BINDPNYX_WORKSPACE, else <repo>/../bindpnyx_workspace) mirrors the repository layout and holds everything that
must never be in git: environments (.pxd), checkpoints, run outputs (out/), built targets (data/, targets/), benchmark data, private
notes. This script walks the workspace and, for every entry the clone does not already have, creates a symlink at the same relative path.

Rules (it is safe to re-run, and it never destroys anything):
  * a file or non-empty directory that already exists in the clone is NEVER replaced - tracked files win;
  * where both sides are directories it descends (so `bench/` keeps its tracked scripts and gains the untracked data next to them);
  * an EMPTY directory in the clone is replaced by the link (the only thing it ever removes);
  * every symlink it creates is added to `.git/info/exclude`, because `.gitignore` patterns written as `out/` do not match a symlink.
Absolute paths stored inside old tables (for example `/path/to/repo/out/...`) keep working if the clone lives at the same path as the old
working tree, because `out` resolves through the link.
"""
import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SKIP_NAMES = {".git", "__pycache__", ".pytest_cache"}
SKIP_PREFIXES = (".copy_",)          # bookkeeping files a workspace builder may leave at its root
MAX_DEPTH = 4


def _skip(name):
    return name in SKIP_NAMES or name.startswith(SKIP_PREFIXES)


def overlay(src, dst, rel, report, depth=0):
    """Link every entry of directory `src` that `dst` lacks. Records outcomes in `report`."""
    for entry in sorted(os.listdir(src)):
        if _skip(entry):
            continue
        s, d = src / entry, dst / entry
        r = f"{rel}/{entry}" if rel else entry
        if d.is_symlink():
            ok = os.path.realpath(d) == os.path.realpath(s)
            report["ok" if ok else "conflict"].append(r)
        elif not d.exists():
            report["link"].append((r, s, d))
        elif d.is_dir() and s.is_dir() and not s.is_symlink():
            if not any(d.iterdir()):
                report["link"].append((r, s, d))            # empty directory: the link replaces it
                report["replace_empty"].append(r)
            elif depth < MAX_DEPTH:
                overlay(s, d, r, report, depth + 1)
            else:
                report["skip"].append(r)
        else:
            report["skip"].append(r)                          # a real file/dir in the clone wins


def apply(report, dry_run):
    made = []
    for r, s, d in report["link"]:
        if dry_run:
            made.append(r)
            continue
        if d.is_dir() and not d.is_symlink():
            d.rmdir()                                         # only reached for empty directories
        d.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(s, d)
        made.append(r)
    return made


def add_excludes(repo, rels):
    info = repo / ".git" / "info"
    if not (repo / ".git").exists() or not rels:
        return 0
    info.mkdir(parents=True, exist_ok=True)
    f = info / "exclude"
    have = set(f.read_text().splitlines()) if f.exists() else set()
    new = [f"/{r}" for r in rels if f"/{r}" not in have]
    if new:
        with open(f, "a") as fh:
            fh.write("\n# links created by scripts/link_workspace.py\n" + "\n".join(new) + "\n")
    return len(new)


def health(repo):
    """Cheap sanity checks on what the links should provide. Returns a list of warning strings."""
    w = []
    for p in sorted((repo / ".pxd" / "envs").glob("*")) if (repo / ".pxd" / "envs").exists() else []:
        if not (p / "bin" / "python").exists():
            w.append(f".pxd/envs/{p.name}: interpreter not found ({os.path.realpath(p)})")
    for d in ("checkpoints",):
        if (repo / d).is_dir():
            for f in sorted((repo / d).iterdir()):
                if f.is_symlink() and not f.exists():
                    w.append(f"{d}/{f.name}: dangling link -> {os.readlink(f)}")
    if (repo / "release_data").is_symlink() and not (repo / "release_data").exists():
        w.append(f"release_data: dangling link -> {os.readlink(repo / 'release_data')}")
    return w


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("workspace", nargs="?", default=os.environ.get("BINDPNYX_WORKSPACE") or str(REPO.parent / "bindpnyx_workspace"))
    ap.add_argument("--repo", default=str(REPO), help="clone to link into (default: this repository)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    ws, repo = Path(a.workspace).resolve(), Path(a.repo).resolve()
    if not ws.is_dir():
        print(f"workspace not found: {ws}\n  build one (see docs/START_HERE.md) or pass its path.", file=sys.stderr)
        return 2
    report = {k: [] for k in ("link", "ok", "skip", "conflict", "replace_empty")}
    overlay(ws, repo, "", report)
    made = apply(report, a.dry_run)
    n_excl = 0 if a.dry_run else add_excludes(repo, made)
    verb = "would link" if a.dry_run else "linked"
    print(f"workspace: {ws}\nclone:     {repo}")
    print(f"{verb} {len(made)}; already linked {len(report['ok'])}; kept the clone's own copy of {len(report['skip'])}; conflicts {len(report['conflict'])}")
    for r in made[:40]:
        print(f"  + {r}")
    if len(made) > 40:
        print(f"  ... {len(made) - 40} more")
    for r in report["conflict"]:
        print(f"  ! {r} is a symlink to somewhere else; left alone", file=sys.stderr)
    if n_excl:
        print(f"added {n_excl} entries to .git/info/exclude (so `git status` stays clean)")
    warns = [] if a.dry_run else health(repo)
    for x in warns:
        print(f"  warning: {x}", file=sys.stderr)
    return 1 if (report["conflict"] or warns) else 0


if __name__ == "__main__":
    sys.exit(main())
