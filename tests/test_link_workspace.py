"""scripts/link_workspace.py: link a private workspace into a fresh clone without ever destroying anything."""
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import link_workspace as lw  # noqa: E402


def _tree(tmp_path):
    ws, repo = tmp_path / "ws", tmp_path / "clone"
    (ws / "out/run1").mkdir(parents=True); (ws / "out/run1/table.csv").write_text("a\n")
    (ws / "bench/out").mkdir(parents=True); (ws / "bench/out/big.bin").write_text("x")
    (ws / "bench/README.md").write_text("WORKSPACE COPY")                 # a stale copy of a tracked file
    (ws / ".pxd/ext").mkdir(parents=True)
    (ws / ".copy_done").write_text("rc=0")                                 # bookkeeping: must be ignored
    (repo / "bench").mkdir(parents=True); (repo / "bench/README.md").write_text("TRACKED")
    (repo / "data").mkdir()                                                # an EMPTY directory the link may replace
    (ws / "data/targets").mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    return ws, repo


def test_links_missing_entries_and_descends_into_shared_directories(tmp_path):
    ws, repo = _tree(tmp_path)
    assert lw.main([str(ws), "--repo", str(repo)]) in (0, 1)
    assert (repo / "out").is_symlink() and (repo / "out/run1/table.csv").read_text() == "a\n"
    assert (repo / ".pxd").is_symlink()
    assert (repo / "bench").is_dir() and not (repo / "bench").is_symlink()          # bench keeps its tracked content ...
    assert (repo / "bench/out").is_symlink()                                       # ... and gains the untracked data
    assert not (repo / ".copy_done").exists()


def test_a_tracked_file_always_wins_and_nothing_is_overwritten(tmp_path):
    ws, repo = _tree(tmp_path)
    lw.main([str(ws), "--repo", str(repo)])
    assert (repo / "bench/README.md").read_text() == "TRACKED" and not (repo / "bench/README.md").is_symlink()


def test_empty_directory_is_replaced_but_nothing_else_is_removed(tmp_path):
    ws, repo = _tree(tmp_path)
    lw.main([str(ws), "--repo", str(repo)])
    assert (repo / "data").is_symlink()
    ws2, repo2 = tmp_path / "ws2", tmp_path / "clone2"
    (ws2 / "data/new").mkdir(parents=True); (repo2 / "data").mkdir(parents=True); (repo2 / "data/mine.txt").write_text("keep")
    lw.main([str(ws2), "--repo", str(repo2)])
    assert (repo2 / "data/mine.txt").read_text() == "keep" and (repo2 / "data/new").is_symlink()   # non-empty: descended, not replaced


def test_second_run_changes_nothing(tmp_path, capsys):
    ws, repo = _tree(tmp_path)
    lw.main([str(ws), "--repo", str(repo)])
    before = sorted(str(p) for p in repo.rglob("*") if ".git" not in p.parts)
    capsys.readouterr()
    lw.main([str(ws), "--repo", str(repo)])
    assert "linked 0" in capsys.readouterr().out
    assert before == sorted(str(p) for p in repo.rglob("*") if ".git" not in p.parts)


def test_dry_run_touches_nothing(tmp_path):
    ws, repo = _tree(tmp_path)
    before = sorted(str(p) for p in repo.rglob("*") if ".git" not in p.parts)
    lw.main([str(ws), "--repo", str(repo), "--dry-run"])
    assert before == sorted(str(p) for p in repo.rglob("*") if ".git" not in p.parts)


def test_links_are_excluded_from_git_status(tmp_path):
    ws, repo = _tree(tmp_path)
    lw.main([str(ws), "--repo", str(repo)])
    status = subprocess.run(["git", "status", "--short"], cwd=repo, capture_output=True, text=True).stdout
    assert "out" not in status and ".pxd" not in status, status                    # symlinks do not show up as untracked


def test_a_symlink_to_somewhere_else_is_reported_not_replaced(tmp_path):
    ws, repo = _tree(tmp_path)
    other = tmp_path / "elsewhere"; other.mkdir()
    os.symlink(other, repo / "out")
    rc = lw.main([str(ws), "--repo", str(repo)])
    assert rc == 1 and os.path.realpath(repo / "out") == str(other.resolve())


def test_missing_workspace_is_a_clear_error(tmp_path, capsys):
    assert lw.main([str(tmp_path / "nope"), "--repo", str(tmp_path)]) == 2
    assert "workspace not found" in capsys.readouterr().err
