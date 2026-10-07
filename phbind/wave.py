"""A WAVE = one self-contained campaign on one machine/GPU, resumable, returning only small CSVs. Run the same config on another machine and merge by CSV.

  PHBIND_CONFIG=wave.json PYTHONPATH=<repo> python phbind/wave.py --plan        what exists, what would run (no GPU)
  PHBIND_CONFIG=wave.json PYTHONPATH=<repo> python phbind/wave.py               run the first unfinished step, and the next, until done
  ... --until <step>      stop after a step (steps: target shard generate pack novelty prescreen gate export)

Every step's "done" is decided from REAL artifacts (rows in designs.csv, the shard file, batch CSVs, a gate table covering the survivors), never from a marker or an exit code.
Two machines must not collide: give each wave its own `wave.campaign`, its own `generate.seed_base`, and its own hotspot-set ALIASES (same residues, different names) so run ids and
backbones differ. The `novelty` step does not search: it waits (wave.novelty_wait_min) for `novelty_backbones_<campaign>.csv` in wave.export_dir, written by whoever holds the structure
database, then proceeds unscreened if nothing arrives. `export` writes `wave_<campaign>_results.csv` and a predicted-binder pack for a final novelty check on the survivors.
One GPU job at a time per card: pin a wave to a card with CUDA_VISIBLE_DEVICES.
"""
from __future__ import annotations
import os, subprocess, sys, time
from pathlib import Path
import pandas as pd
REPO = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "phbind")); sys.path.insert(0, str(REPO / "funnel"))
from phbind import config as _config
STEPS = ["target", "shard", "generate", "pack", "novelty", "prescreen", "gate", "export"]

def _paths(c, root: Path):
    camp = c["wave"]["campaign"] or "wave"; ex = Path(c["wave"]["export_dir"]) if c["wave"]["export_dir"] else root / "out/phbind/export"
    out = root / "out/phbind"
    return dict(out=out, gen=out / "gen", camp=camp, export=ex, shard=out / "shard" / "tnf_dimerBC.pkl.gz", manifest=root / "phbind/target/target_manifest.json",
                pack=ex / f"backbones_{camp}.tar.gz", novelty=ex / f"novelty_backbones_{camp}.csv", results=ex / f"wave_{camp}_results.csv", gate=out / f"gate_{camp}" / "gate.csv")

def run_ids(c):
    g = c["generate"]; return [f"{s}_L{L}" for s in g["sets"] for L in g["lengths"]]

def generated(c, root: Path) -> tuple[int, int]:
    g = c["generate"]; P = _paths(c, root); want = g["backbones_per_run"] * g["seqs_per_backbone"]; ok = 0
    for r in run_ids(c):
        f = P["gen"] / r / "designs.csv"
        if f.exists() and sum(1 for _ in open(f)) - 1 == want: ok += 1
    return ok, len(run_ids(c))

def survivors(c, root: Path) -> pd.DataFrame:
    """Scored designs of this campaign that clear the gate threshold on the first model and are not novelty-failed, best first, capped (the AF3 leg is the expensive one)."""
    P = _paths(c, root); fs = sorted((P["out"] / "s2").glob("batch_*.csv"))
    if not fs: return pd.DataFrame(columns=["id", "seq", "ipsae_min"])
    a = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True); a = a[~a.id.str.startswith("carrier_")].drop_duplicates("id")
    a = a[a.id.str.extract(r"^(.*_L\d+)_\d+_\d+$")[0].isin(run_ids(c))]; a = a[a.ipsae_min >= c["gate"]["gate"]]       # only THIS wave's runs (set + length), never an earlier campaign's designs
    if P["novelty"].exists():
        nv = pd.read_csv(P["novelty"]); key = "bbid" if "bbid" in nv else "id"; bad = set(nv[nv.n_strict_hits > 0][key].astype(str).str.replace(r"_\d+$", "", regex=True)) if key == "id" else set(nv[nv.n_strict_hits > 0].bbid)
        a = a[~a.id.str.rsplit("_", n=1).str[0].isin(bad)]
    seq = pd.read_csv(P["out"] / "designs_all.csv").set_index("id").seq; a = a.sort_values("ipsae_min", ascending=False).head(c["wave"]["af3_cap"]); a["seq"] = a.id.map(seq)
    return a[["id", "seq", "ipsae_min"]].reset_index(drop=True)

def plan(c, root: Path = REPO) -> list[tuple[str, bool, str]]:
    P = _paths(c, root); ok, n = generated(c, root); now = time.time(); pk = P["pack"].exists()
    nov = P["novelty"].exists() or (pk and now - P["pack"].stat().st_mtime > 60 * c["wave"]["novelty_wait_min"])
    sv = survivors(c, root); gated = pd.read_csv(P["gate"]).id if P["gate"].exists() else pd.Series([], dtype=str)
    from phbind import s2_prescreen as s2     # eligible() is pure; reading the module also reads PHBIND_CONFIG
    try:
        d = pd.read_csv(P["out"] / "designs_all.csv"); d["k"] = d.id.str.rsplit("_", n=1).str[1].astype(int); done = {i for f in (P["out"] / "s2").glob("batch_*.csv") for i in pd.read_csv(f).id}
        left = len(s2.eligible(d, done, c["prescreen"]["min_length"], c["prescreen"]["max_seq_index"], pd.read_csv(P["novelty"]) if P["novelty"].exists() else None, c["prescreen"].get("sets") or c["generate"]["sets"]))
    except FileNotFoundError: left = -1
    return [("target", P["manifest"].exists(), str(P["manifest"])), ("shard", P["shard"].exists(), str(P["shard"])), ("generate", ok == n, f"{ok}/{n} runs complete"),
            ("pack", pk, str(P["pack"])), ("novelty", bool(nov), str(P["novelty"]) + ("" if P["novelty"].exists() else f" (proceeds unscreened after {c['wave']['novelty_wait_min']} min)")),
            ("prescreen", ok == n and left == 0, f"{left} designs left to screen" if left >= 0 else "designs_all.csv missing"), ("gate", set(sv.id) <= set(gated) and (len(sv) == 0 or P["gate"].exists()), f"{len(sv)} survivors, {len(set(sv.id) & set(gated))} gated"),
            ("export", P["results"].exists() and (not P["gate"].exists() or P["results"].stat().st_mtime >= P["gate"].stat().st_mtime), str(P["results"]))]

def resolved_config(c, root: Path = REPO) -> str:
    """Effective config written to disk for the subprocess steps: the wave's novelty table and its own hotspot-set aliases filled in."""
    import json
    P = _paths(c, root); r = json.loads(json.dumps(c)); r["prescreen"]["novelty_file"] = r["prescreen"]["novelty_file"] or str(P["novelty"]); r["prescreen"]["sets"] = r["prescreen"].get("sets") or r["generate"]["sets"]; r["prescreen"]["min_length"] = max(r["prescreen"]["min_length"], min(r["generate"]["lengths"]))
    f = P["out"] / f"wave_{P['camp']}_config.json"; f.parent.mkdir(parents=True, exist_ok=True); f.write_text(json.dumps(r, indent=1)); return str(f)

def _sub(args, root, env=None):
    r = subprocess.run(args, cwd=root, env={**os.environ, "PYTHONPATH": str(root), **(env or {})}); return r.returncode

def run_step(step, c, root: Path = REPO):
    P = _paths(c, root); py = sys.executable; rc = resolved_config(c, root)
    if step == "target": _sub([py, "phbind/s0_target.py"], root)
    elif step == "shard": _sub([py, "phbind/s0b_shard.py"], root)
    elif step == "generate": _sub([py, "-u", "phbind/s1_queue.py"], root, {"PHBIND_CONFIG": rc})
    elif step == "pack":
        from phbind import handoff
        f, idx = handoff.backbone_files(min(c["generate"]["lengths"]), P["gen"], set(run_ids(c))); assert len(f) == len(run_ids(c)) * c["generate"]["backbones_per_run"], "backbone count != configured"
        P["export"].mkdir(parents=True, exist_ok=True); handoff.write_tar(f, P["pack"], f"backbones_{P['camp']}"); idx.to_csv(P["export"] / f"backbones_{P['camp']}_index.csv", index=False)
        print(f"PACK READY: {len(f)} backbones -> {P['pack']}  (screen for structural novelty, write {P['novelty'].name} with columns bbid,n_strict_hits)", flush=True)
    elif step == "novelty":
        t0 = P["pack"].stat().st_mtime; deadline = t0 + 60 * c["wave"]["novelty_wait_min"]
        while not P["novelty"].exists() and time.time() < deadline: time.sleep(20)
        print("novelty table found" if P["novelty"].exists() else "no novelty table: proceeding UNSCREENED", flush=True)
    elif step == "prescreen":
        _sub([py, "-u", "phbind/s2_prescreen.py"], root, {"PHBIND_CONFIG": rc})        # reads the novelty table before every batch
    elif step == "gate":
        sv = survivors(c, root); P["gate"].parent.mkdir(parents=True, exist_ok=True); sv[["id", "seq"]].to_csv(P["gate"].parent / "survivors.csv", index=False)
        _sub([py, "-u", "phbind/s3_gate.py", str(P["gate"].parent / "survivors.csv"), str(P["gate"].parent / "run"), str(P["out"] / "s2"), c["gate"]["legs"][1]], root, {"PHBIND_CONFIG": rc})
        src = P["gate"].parent / "run" / "gate.csv"
        if src.exists(): P["gate"].write_bytes(src.read_bytes())
    elif step == "export":
        g = pd.read_csv(P["gate"]); seq = pd.read_csv(P["out"] / "designs_all.csv").set_index("id").seq; g["sequence"] = g.id.map(seq)
        P["export"].mkdir(parents=True, exist_ok=True); g.drop(columns=[x for x in g.columns if x.endswith("cif")]).to_csv(P["results"], index=False)
        from phbind import handoff
        f = handoff.predicted_files(c["gate"]["gate"], P["out"] / "s2", min(c["generate"]["lengths"])); handoff.write_tar(f, P["export"] / f"predicted_{P['camp']}.tar.gz", "predicted_binder")
        print(f"EXPORT READY: {P['results']} ({len(g)} designs) + {len(f)} predicted binder chains for a final novelty check", flush=True)

def claim_checkout(c, root: Path = REPO):
    """Two waves in one checkout share out/phbind (batch numbering, designs_all.csv) and corrupt each other. Refuse to run if a DIFFERENT campaign's process is alive in this checkout."""
    import json, socket
    f = Path(root) / "out/phbind/.wave_owner.json"; f.parent.mkdir(parents=True, exist_ok=True); camp = c["wave"]["campaign"] or "wave"
    if f.exists():
        o = json.loads(f.read_text())
        alive = False
        if o.get("host") == socket.gethostname() and o.get("pid") != os.getpid():
            try: os.kill(int(o["pid"]), 0); alive = True
            except (OSError, ValueError): alive = False
        if alive and o.get("campaign") != camp:
            raise SystemExit(f"this checkout is in use by campaign '{o['campaign']}' (pid {o['pid']}); give wave '{camp}' its own clone (two waves must not share out/phbind)")
    f.write_text(json.dumps({"campaign": camp, "pid": os.getpid(), "host": socket.gethostname()}))

def main(argv):
    c = _config.load(); root = REPO
    if "--plan" not in argv: claim_checkout(c, root)
    st = plan(c, root)
    for s, done, note in st: print(f"{'done' if done else 'TODO':5s} {s:10s} {note}")
    if "--plan" in argv: return 0
    until = argv[argv.index("--until") + 1] if "--until" in argv else STEPS[-1]
    for s, done, _ in st:
        if not done: print(f"\n== {s} ==", flush=True); run_step(s, c, root)
        if s == until: break
    for s, done, note in plan(c, root): print(f"{'done' if done else 'TODO':5s} {s:10s} {note}")
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
