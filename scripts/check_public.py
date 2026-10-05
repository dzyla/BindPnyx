#!/usr/bin/env python3
"""Scan a directory for content that must not be published: personal paths, emails, host names, credentials, structure/MSA/data files, large files, symlinks.

  python scripts/check_public.py DIR          exit code 1 and a list of findings if anything is found
Allowed on purpose: the public ProteinMPNN weights under colabdesign/mpnn/weights*/ and the project's own figures (png)."""
import re, sys
from pathlib import Path

TEXT_BAD = [
    (r"/home/(?!someone\b|user\b|you\b|runner\b|alice\b|out/|my_pdbs/)[A-Za-z0-9_.-]+", "personal home path"),
    (r"/mnt/(?:HDD\d|nas|scratch|nas-home)\b", "machine-specific mount path"),
    (r"dzyla(?!/fastPISA)", "personal handle (only github.com/dzyla/fastPISA is allowed)"),
    (r"cuanschutz|\bdawid\b|\bzyla\b", "personal / institutional name"),
    (r"[A-Za-z0-9._%+-]+@(?!example\.|users\.noreply)[A-Za-z0-9.-]+\.(?:edu|com|org|net|io)\b", "email address"),
    (r"mev_screen|HDD1|egfr_px_obj|dzyla-lab", "internal project / host name"),
    (r"ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|AKIA[0-9A-Z]{16}|sk-[A-Za-z0-9]{32,}|BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY", "credential"),
]
BAD_EXT = {".pdb", ".cif", ".ent", ".mmcif", ".a3m", ".npz", ".npy", ".pt", ".ckpt", ".mrc", ".h5", ".parquet", ".gz", ".zip", ".tar", ".safetensors"}
ALLOW_BIN = ("tests/fixtures/", "colabdesign/af/weights/", "colabdesign/mpnn/weights/", "colabdesign/mpnn/weights_soluble/", "docs/figures/", "bench/results/figures/")
MAX_BYTES = 2_000_000

def scan(root):
    root = Path(root); bad = []
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root).as_posix()
        if ".git/" in rel + "/" or rel == ".git" or p.is_dir() or rel == "scripts/check_public.py": continue
        if p.is_symlink(): bad.append((rel, 0, "symlink (points outside the repository?)")); continue
        allow = rel.startswith(ALLOW_BIN)
        if p.suffix.lower() in BAD_EXT and not allow: bad.append((rel, 0, f"data/structure file type {p.suffix}"))
        if p.stat().st_size > MAX_BYTES and not allow: bad.append((rel, 0, f"large file ({p.stat().st_size // 1000} KB)"))
        if allow and p.suffix in (".pkl", ".png"): continue
        try: txt = p.read_text(errors="strict")
        except Exception: continue
        for n, line in enumerate(txt.splitlines(), 1):
            for pat, why in TEXT_BAD:
                if re.search(pat, line, re.I if "dawid" in pat else 0): bad.append((rel, n, f"{why}: {line.strip()[:110]}"))
    return bad

if __name__ == "__main__":
    d = sys.argv[1] if len(sys.argv) > 1 else "."
    bad = scan(d)
    for rel, n, why in bad: print(f"{rel}:{n}: {why}")
    print(f"\n{len(bad)} finding(s) in {d}"); sys.exit(1 if bad else 0)
