"""The forum helper must number notes from the file at write time, never rewrite earlier content, and keep a backup."""
import importlib.util, re
from pathlib import Path
spec = importlib.util.spec_from_file_location("forum_note", Path(__file__).resolve().parent.parent / "scripts" / "forum_note.py")
fn = importlib.util.module_from_spec(spec); spec.loader.exec_module(fn)

def test_numbers_come_from_the_file_and_content_is_preserved(tmp_path):
    f = tmp_path / "FORUM.md"; f.write_text("# Forum\n\n## NOTE 7 — A, t — **old**\nbody\n\n## NOTE 12 — B, t — **older**\nbody\n")
    before = f.read_text()
    n = fn.append_note(f, "C", "new result", "evidence line", now="2026-01-01 00:00")
    after = f.read_text()
    assert n == 13 and after.startswith(before)                                   # highest number + 1, earlier text untouched
    assert re.findall(r"^## NOTE (\d+)", after, re.M) == ["7", "12", "13"]
    assert len(list(tmp_path.glob(".FORUM.md.bak_*"))) == 1                       # a backup was made first
    assert fn.append_note(f, "A", "second", "x", now="t") == 14                   # consecutive calls do not collide

def test_empty_or_missing_file_starts_at_one(tmp_path):
    f = tmp_path / "new.md"
    assert fn.append_note(f, "A", "first", "x", now="t") == 1 and "## NOTE 1 " in f.read_text()
