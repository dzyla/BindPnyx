#!/usr/bin/env python3
"""Build a clean, publishable copy of this repository from an ALLOWLIST (no history, no private data), scan it, and optionally run the tests inside it.

  python scripts/export_public.py --dest ../bindpnyx-public [--run-tests]

Not exported: .archive/, out/, data/, .pxd/, checkpoints, structures, MSAs, run outputs, per-design benchmark tables, private target manifests, anything the scanner flags.
The destination gets `git init` but NO commit and NO remote: review it, then commit with your own identity and push.  Nothing is uploaded by this script."""
import argparse, os, re, shutil, subprocess, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "scripts")); import check_public

INCLUDE = ["LICENSE", "COMMERCIAL.md", "THIRD_PARTY.md", "README.md", "AGENTS.md", "CLAUDE.md", "pyproject.toml", "setup.py", "requirements.txt", "pytest.ini",
           "funnel", "pxdesign", "pxdbench", "colabdesign", "tests", "phbind", "docs/HOMO_OLIGOMER_TARGETS.md",
           "scripts/setup.sh", "scripts/pxd_env.py", "scripts/run_campaign.sh", "scripts/prepare_target.py", "scripts/preflight_target.py", "scripts/target_spec.py",
           "scripts/check_msa_match.sh", "scripts/forum_note.py", "scripts/setup_extras.sh", "funnel/evaluate_overnight.py", "scripts/fetch_checkpoints.sh", "scripts/boltz_light.py", "scripts/hotspot_e2e_check.py", "scripts/check_public.py", "scripts/link_workspace.py", "scripts/campaign", "scripts/export_public.py",
           "docs/REPORT.md", "docs/RECOMMENDATIONS.md", "docs/DESIGN_PATH.md", "docs/START_HERE.md", "docs/WORKFLOW.md", "docs/OVERNIGHT_PLAN.md", "docs/OVERNIGHT_RESULTS.md", "docs/ADAPTYV_DESIGN_FLOW.md", "docs/ADAPTYV_STATS.md", "docs/AGENT_FORUM.md", "docs/EXTENDING.md", "docs/PROTOCOL.md", "docs/HOTSPOTS.md", "docs/pipeline-flow.md", "docs/make_workflow_figure.py", "docs/figures",
           "bench/README.md", "bench/run_benchmark.sh", "bench/prepare_labels.py", "bench/make_inputs.py", "bench/make_msas.py", "bench/metrics.py", "bench/score_boltz.py", "bench/score_protenix.py",
           "bench/analyze.py", "bench/remp.py", "bench/check1.py", "bench/cycle.py", "bench/mpnn_round.py", "bench/make_figures.py", "bench/make_followup_figure.py",
           "bench/results/auroc_by_target.csv", "bench/results/sc_sweep.csv", "bench/results/analysis.txt", "bench/results/figures"]
EXCLUDE_NAMES = {"__pycache__", ".pytest_cache", ".git", ".DS_Store"}
SANITIZE = [(r"<repo>", "<repo>"), (r"<external>/", "<external>/"), (r"~/", "~/")]
PUBLIC_GITIGNORE = """__pycache__/\n*.pyc\n.pytest_cache/\n# environments, weights, data and outputs never belong in git\n.pxd/\ncheckpoints/\nrelease_data\ndata/\nout/\n*.pkl.gz\n*.a3m\n*.cif\n*.pdb\n!tests/fixtures/*\n"""

def copy_tree(src, dst):
    if src.is_dir():
        for p in sorted(src.rglob("*")):
            if any(part in EXCLUDE_NAMES for part in p.parts) or p.suffix in (".pyc", ".ttf"): continue      # .ttf: Times New Roman is proprietary; only the upstream web-demo plots use it
            rel = p.relative_to(src); t = dst / rel
            if p.is_symlink(): continue
            if p.is_dir(): t.mkdir(parents=True, exist_ok=True)
            else: t.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p, t)
    else: dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src, dst)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dest", required=True); ap.add_argument("--run-tests", action="store_true"); a = ap.parse_args()
    dest = Path(a.dest).resolve()
    if dest.exists(): sys.exit(f"{dest} exists: choose a new directory (nothing is overwritten)")
    for item in INCLUDE:
        s = HERE / item
        if s.exists(): copy_tree(s, dest / item)
        else: print("skip (absent):", item)
    shutil.rmtree(dest / "phbind" / "target", ignore_errors=True)       # generated target data (structures, MSAs): never exported
    for p in dest.rglob("*"):                              # sanitize obviously private absolute paths in text files
        if p.is_file() and p.suffix in (".py", ".md", ".sh", ".json", ".txt", ".toml", ".cfg", ".yaml", ".csv"):
            try: s = p.read_text()
            except Exception: continue
            o = s
            for pat, rep in SANITIZE: s = re.sub(pat, rep, s)
            if s != o: p.write_text(s)
    (dest / ".gitignore").write_text(PUBLIC_GITIGNORE)
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=dest)
    bad = check_public.scan(dest)
    for rel, n, why in bad: print(f"FINDING {rel}:{n}: {why}")
    size = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file() and ".git/" not in f.as_posix()) / 1e6
    print(f"\nexported to {dest}: {sum(1 for f in dest.rglob('*') if f.is_file() and '.git/' not in f.as_posix())} files, {size:.1f} MB; scanner findings: {len(bad)}")
    if bad: sys.exit(1)
    if a.run_tests:
        py = os.environ.get("PXD_PYTHON") or str(HERE / ".pxd/envs/pxd/bin/python")
        rc = subprocess.run([py, "-m", "pytest", "tests", "-q", "-p", "no:cacheprovider"], cwd=dest, env={**os.environ, "PYTHONPATH": str(dest)}).returncode
        sys.exit(rc)
    print("next: cd", dest, "&& git add -A && git commit  (with your identity)  && git remote add origin ... && git push")

if __name__ == "__main__": main()
