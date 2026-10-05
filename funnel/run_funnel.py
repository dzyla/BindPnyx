"""The funnel: designs -> fast Protenix screen -> elitist refold-redesign cycling -> Boltz-2 (3 seeds) + Protenix-v2 consensus -> de-duplicated shortlist
-> final_design/ review package.

  python funnel/run_funnel.py --target pdl1 --out out/funnel/pdl1 --n-backbones 500            # built-in generator: PXDesign diffusion + ProteinMPNN
  python funnel/run_funnel.py --target fimh --out out/x --designs-csv my_designs.csv            # any other generator: bring a table of sequences
  python funnel/run_funnel.py --out out/funnel/pdl1 --status                                   # what is done, what is missing

RESUMING (nothing is ever lost to a crash or Ctrl-C)
  * Re-run the SAME command: finished work is skipped, an interrupted stage continues from its last completed unit.
    Units: generation chunks (--chunk backbones), per-chunk design + fast screen, every cycling round, every Boltz/Protenix prediction.
  * Continue / optimise from last: raise --n-backbones (adds chunks and screens only the new ones), --rounds (cycling continues from the saved round),
    --final-m (only the new candidates get the expensive models). Re-ranking and the review package are always rebuilt.
  * Settings that change what a finished unit means (target, hotspots, binder length, chunk size, MPNN settings, --steps, --seqs, input designs) are
    recorded in run_config.json; a resume with different values stops with an explanation unless --allow-config-change.
  * Every table is written atomically (temp file + rename); stage state is in run_state.json; timings in timers.json.

DESIGN TABLE FOR OTHER GENERATORS (--designs-csv): columns `seq` (required), `bb` (optional, groups sequences that share a parent structure), `id` (optional).
One GPU job at a time: stages run strictly in sequence."""
import argparse, hashlib, json, os, shutil, subprocess, sys, time
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent)); import common, make_manifest, oracles
REPO = common.REPO

def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)

def save_csv(df, path):
    path = Path(path); tmp = path.with_name(path.name + ".tmp"); df.to_csv(tmp, index=False); os.replace(tmp, path)

def save_json(obj, path):
    path = Path(path); tmp = path.with_name(path.name + ".tmp"); json.dump(obj, open(tmp, "w"), indent=1, default=str); os.replace(tmp, path)

import threading
_LOCK = threading.RLock()

class Timers(dict):
    def __init__(self, path): super().__init__(); self.path = path; self.update(json.load(open(path)) if path.exists() else {})
    def add(self, k, s):
        with _LOCK: self[k] = self.get(k, 0) + s; save_json(dict(self), self.path)

class RunState:
    """run_state.json: per-stage status and counts, for --status and for post-mortems."""
    def __init__(self, path): self.path = path; self.d = json.load(open(path)) if path.exists() else {"stages": {}}
    def mark(self, stage, status, **info):
        with _LOCK: self.d["stages"][stage] = dict(status=status, updated=time.strftime("%Y-%m-%d %H:%M:%S"), **info); save_json(self.d, self.path)

def cif_to_pdb(cif, pdb):
    from Bio.PDB import MMCIFParser, PDBIO
    io = PDBIO(); io.set_structure(MMCIFParser(QUIET=True).get_structure("x", str(cif))); io.save(str(pdb))

def run_mpnn(pdb_dir, n, out_csv, weights="soluble", temp="0.1", bias="none", scope="full"):
    env = {**common.gpu_env(), "PYTHONPATH": str(REPO), "XLA_PYTHON_CLIENT_PREALLOCATE": "false"}
    p = subprocess.run([common.PXD_PY(), str(Path(__file__).with_name("mpnn_run.py")), str(pdb_dir), str(n), str(out_csv), weights, temp, bias, scope], env=env, capture_output=True, text=True)
    if p.returncode: raise RuntimeError("mpnn failed:\n" + p.stderr[-1500:])
    return pd.read_csv(out_csv)

# ------------------------------------------------------------------ config guard
GUARD = ("target", "hotspot_idx", "binder_length", "chunk", "steps", "seqs", "mpnn_weights", "mpnn_bias", "cycle_scope", "designs_csv_sha",
         "judge", "second_oracle", "boltz_seeds")      # rank_rule is NOT guarded: it only re-orders finished predictions
# What a run folder started before --judge existed means. Without this, `--judge new` on such a folder would silently mix two judges in one consensus table.
LEGACY = dict(judge="legacy", second_oracle="protenix-v2", boltz_seeds="1,2,3")      # rank_rule: "min"
# judge regime -> defaults. `new` = ONE Boltz-2 seed + ONE AlphaFold3 seed (two architectures beat three Boltz seeds at lower cost on the release benchmark).
JUDGES = {"legacy": dict(second_oracle="protenix-v2", boltz_seeds="1,2,3", rank_rule="min"), "new": dict(second_oracle="af3", boltz_seeds="1", rank_rule="mean")}
def check_config(out, cfg, allow):
    f = out / "run_config.json"
    if f.exists():
        old = json.load(open(f)); diff = {k: (old.get(k, LEGACY.get(k)), cfg.get(k, LEGACY.get(k))) for k in GUARD if (k in old or k in cfg or k in LEGACY) and old.get(k, LEGACY.get(k)) != cfg.get(k, LEGACY.get(k))}
        if diff and not allow:
            raise SystemExit("Refusing to resume: these settings differ from the ones this run was started with (finished units would be inconsistent):\n  "
                             + "\n  ".join(f"{k}: was {a!r}, now {b!r}" for k, (a, b) in diff.items()) + "\nUse the original values, a new --out, or --allow-config-change.")
        if diff: log("WARNING: config changed on resume:", diff)
    save_json({**(json.load(open(f)) if f.exists() else {}), **cfg}, f)

# ------------------------------------------------------------------ stage 1: backbones (chunked, resumable)
def gen_units(out, n, chunk):
    """[(key, offset, count)] of generation units, planned from what is already finished so that nothing completed is ever redone: finished chunks keep their own
    offset/count (even a short last one), new chunks continue after them. A legacy single-shot run (gen/converted.json without chunks) is unit 'legacy' [0, legacy_n)."""
    gen = out / "gen"; legacy = gen / "converted.json"; units = []; off = 0; k = 0
    if legacy.exists() and not any(gen.glob("chunk_*")):
        ln = json.load(open(legacy))["n"]; units.append(("legacy", 0, ln)); off = ln
    for d in sorted(gen.glob("chunk_*")):
        if (d / "done.json").exists():
            j = json.load(open(d / "done.json")); units.append((d.name, j["offset"], j["count"])); off = max(off, j["offset"] + j["count"]); k = max(k, int(d.name.split("_")[1]) + 1)
    while off < n:
        cnt = min(chunk, n - off); units.append((f"chunk_{k:03d}", off, cnt)); off += cnt; k += 1
    return units

def stage_generate(t, out, n, chunk, steps, T, state, force, gpus=None):
    gen = out / "gen"; gen.mkdir(parents=True, exist_ok=True); units = gen_units(out, n, chunk); prefix = f"{t['name']}_L{t['binder_length']}"
    legacy = gen / "converted.json"; pdb_dir = Path(json.load(open(legacy))["pdb_dir"]) if (legacy.exists() and units and units[0][0] == "legacy") else gen / "pdbs"
    pdb_dir = pdb_dir if pdb_dir.is_absolute() else REPO / pdb_dir; pdb_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(REPO)); from pxdbench.utils import convert_cifs_to_pdbs
    pending = []
    for key, off, cnt in units:
        if key == "legacy": continue
        if (gen / key / "done.json").exists() and not force: continue
        pending.append((key, off, cnt))
    def gen_chunk(u):
        key, off, cnt = u; cd = gen / key
        shutil.rmtree(cd, ignore_errors=True); cd.mkdir(parents=True); mf = cd / "input.json"; save_json(make_manifest.manifest(t), mf)
        log(f"generate {key}: backbones {off}..{off + cnt - 1}"); t0 = time.time()
        p = subprocess.run([common.PXD_PY(), "-u", "-m", "pxdesign.runner.inference", "--input_json_path", str(mf), "--dump_dir", str(cd / "run"),
                            "--load_checkpoint_dir", common.CKPT(), "--N_sample", str(cnt), "--N_step", str(steps)], env={**common.gpu_env(), "PYTHONPATH": str(REPO)}, capture_output=True, text=True)
        (cd / "log.txt").write_text(p.stdout[-30000:] + p.stderr[-30000:])
        if p.returncode: raise RuntimeError("diffusion failed:\n" + p.stderr[-2000:])
        pdirs = [d for d in (cd / "run").glob("*/seed_*/predictions") if d.is_dir()]
        if not pdirs: raise RuntimeError("diffusion produced no predictions")
        cdir, names, cond, bind = convert_cifs_to_pdbs(str(pdirs[0]))
        for nm in names:
            i = int(nm.rsplit("_sample_", 1)[1]); shutil.copy2(Path(cdir) / f"{nm}.pdb", pdb_dir / f"{prefix}_sample_{off + i}.pdb")
        save_json(dict(offset=off, count=len(names)), cd / "done.json"); T.add("1_generate", time.time() - t0)
        state.mark("generate", "running", chunks_done=len(list(gen.glob("chunk_*/done.json"))), chunks_total=len(units))
    common.parallel_map(gen_chunk, pending, gpus)
    total = sum(1 for _ in pdb_dir.glob(f"{prefix}_sample_*.pdb")); info = dict(pdb_dir=str(pdb_dir), n=total, prefix=prefix, cond=["A"], binder=["B"])
    save_json(info, gen / "converted.json"); state.mark("generate", "done", backbones=total); return info, units

# ------------------------------------------------------------------ stages 2+3: sequences + fast screen (per unit, resumable)
def screen_part(t, out, unit_key, d, T, state, gpus=None):
    """fast-screen one table of designs (id, bb, seq) and persist it as screen_parts/part_<key>.csv"""
    pf = out / "screen_parts" / f"part_{unit_key}.csv"
    t0 = time.time(); sc = common.protenix_fold_multi(d, t, out / "screen_fast" / unit_key, arm="fast", gpus=gpus) if gpus and len(gpus) > 1 and len(d) >= 2 * len(gpus) else common.protenix_fold(d, t, out / "screen_fast" / unit_key, arm="fast"); T.add("3_fast_screen", time.time() - t0)
    d = d.merge(sc, on="id"); save_csv(d, pf); return d

def stage_screen(t, out, gen, units, seqs, T, state, force, mp, designs_csv=None, gpus=None):
    parts = out / "screen_parts"; parts.mkdir(exist_ok=True)
    if designs_csv is not None:                                              # external generator: design table instead of backbones + MPNN
        x = pd.read_csv(designs_csv); x["bb"] = x["bb"] if "bb" in x else range(len(x)); x["id"] = x["id"] if "id" in x else [f"x{i:05d}" for i in range(len(x))]
        x["bb"] = pd.factorize(x["bb"])[0]
        for j in range(0, len(x), 250):
            key = f"ext_{j // 250:03d}"
            if (parts / f"part_{key}.csv").exists() and not force: continue
            log(f"fast-screen external designs {j}..{min(j + 250, len(x)) - 1}"); screen_part(t, out, key, x.iloc[j:j + 250][["id", "bb", "seq"]].reset_index(drop=True), T, state)
    else:
        legacy_csv = out / "screen.csv"
        if legacy_csv.exists() and not any(parts.glob("part_*.csv")) and units and units[0][0] == "legacy":
            shutil.copy2(legacy_csv, parts / "part_legacy.csv")             # adopt a finished single-shot run as unit 'legacy'
        pdb_dir = Path(gen["pdb_dir"]); prefix = gen["prefix"]
        todo = [(k, o, c) for k, o, c in units if force or not (parts / f"part_{k}.csv").exists()]
        def do_unit(u):
            key, off, cnt = u
            log(f"design + screen {key}: backbones {off}..{off + cnt - 1}")
            indir = out / "seq" / f"in_{key}"; shutil.rmtree(indir, ignore_errors=True); indir.mkdir(parents=True)
            for g in range(off, off + cnt): os.symlink((pdb_dir / f"{prefix}_sample_{g}.pdb").resolve(), indir / f"{prefix}_sample_{g}.pdb")
            t0 = time.time(); m = run_mpnn(indir, seqs, out / "seq" / f"mpnn_{key}.csv", mp[0], "0.1", mp[1], "full"); T.add("2_mpnn", time.time() - t0)
            m["bb"] = m.name.str.rsplit("_", n=1).str[1].astype(int); m["id"] = [f"b{b:04d}s{k}" for b, k in zip(m.bb, m.seq_idx)]
            screen_part(t, out, key, m[["id", "bb", "sequence"]].rename(columns={"sequence": "seq"}), T, state)
            state.mark("design_screen", "running", units_done=len(list(parts.glob("part_*.csv"))), units_total=len(units))
        common.parallel_map(do_unit, todo, gpus)
    allp = pd.concat([pd.read_csv(f) for f in sorted(parts.glob("part_*.csv"))]).drop_duplicates("id").reset_index(drop=True)
    save_csv(allp, out / "screen.csv"); state.mark("design_screen", "done", designs=len(allp), fast_ok=int(allp.fast_ok.sum())); return allp

# ------------------------------------------------------------------ stage 4: cycling (state saved after every round)
def stage_cycle(t, out, screen, k, rounds, n_new, T, state, force, mp, scope="full", gpus=None):
    ok = screen.dropna(subset=["fast_ipsae"])
    start = ok.sort_values("fast_ipsae", ascending=False).groupby("bb").head(1).sort_values("fast_ipsae", ascending=False).head(k).copy(); start["round"] = 0
    save_csv(start, out / "start_parents.csv"); cdir = out / "cycle"; sf = cdir / "state.json"; sids = list(start.id)
    if sf.exists() and not force and json.load(open(sf)).get("start_ids") == sids:
        st = json.load(open(sf)); parents = {int(b): p for b, p in st["parents"].items()}; done_rounds = st["rounds_done"]; log(f"cycling: resuming after round {done_rounds}")
    else:
        if cdir.exists(): shutil.move(str(cdir), str(out / f"cycle_old_{time.strftime('%Y%m%d_%H%M%S')}")); log("cycling: the start set changed (more backbones or --force): old cycling state archived")
        cdir.mkdir(parents=True); done_rounds = 0
        parents = {int(r.bb): dict(id=r.id, seq=r.seq, score=r.fast_ipsae, cif=r.fast_cif, rnd=0) for r in start.itertuples()}
    for rd in range(done_rounds + 1, rounds + 1):
        t0 = time.time(); pdb_dir = cdir / f"pdb_r{rd}"; shutil.rmtree(pdb_dir, ignore_errors=True); pdb_dir.mkdir(parents=True)
        for b, p in parents.items(): cif_to_pdb(_p(p["cif"]), pdb_dir / f"bb{b}.pdb")
        m = run_mpnn(pdb_dir, n_new, cdir / f"mpnn_r{rd}.csv", mp[0], "0.1", mp[1], scope)
        m["bb"] = m.name.str.replace("bb", "").astype(int); m["id"] = [f"c{rd}_{b:04d}_{i}" for b, i in zip(m.bb, m.seq_idx)]
        ch = m[["id", "bb", "sequence"]].rename(columns={"sequence": "seq"})
        sc = common.protenix_fold_multi(ch, t, cdir / f"fast_r{rd}", arm="fast", gpus=gpus); ch = ch.merge(sc, on="id"); ch["round"] = rd; save_csv(ch, cdir / f"children_r{rd}.csv")
        for b, g in ch.dropna(subset=["fast_ipsae"]).groupby("bb"):
            top = g.sort_values("fast_ipsae", ascending=False).iloc[0]
            if top.fast_ipsae > parents[b]["score"]: parents[b] = dict(id=top.id, seq=top.seq, score=top.fast_ipsae, cif=top.fast_cif, rnd=rd)
        save_json(dict(start_ids=sids, rounds_done=rd, parents={str(b): p for b, p in parents.items()}), sf)
        T.add("4_cycling", time.time() - t0); state.mark("cycling", "running", rounds_done=rd, rounds_target=rounds)
        log(f"cycle round {rd}: mean parent fast ipSAE {sum(p['score'] for p in parents.values()) / len(parents):.3f}")
    kids = [pd.read_csv(f) for f in sorted(cdir.glob("children_r*.csv"))]
    if kids: save_csv(pd.concat(kids), out / "cycle_children.csv")
    d = pd.DataFrame([dict(bb=b, **p) for b, p in parents.items()]).rename(columns={"score": "fast_ipsae", "cif": "fast_cif"}); save_csv(d, out / "cycled_parents.csv")
    state.mark("cycling", "done", rounds_done=max(done_rounds, rounds)); return d

def _p(x):
    x = Path(str(x)); return x if x.is_absolute() else REPO / x

# ------------------------------------------------------------------ stage 5: consensus
def consensus(t, cands, outdir, seeds, T, tag, gpus=None, oracle2="protenix-v2", o2_gate=0.5, rank_rule="min"):
    """Boltz-2 (`seeds`) + a second oracle on cands (id, seq). Returns the table with gate flags, hotspot contact, site occlusion and interface-quality columns.
    The second oracle's columns are called v2_* whichever model it is (downstream files depend on that); `o2_name` records the model.
    Resumable: predictions already on disk are reused."""
    outdir = Path(outdir)
    gpus = gpus or [None]; res = {}; boltz_done = threading.Event()
    def run_b():
        try:
            common.set_gpu(None); t0 = time.time(); res["b"] = common.boltz_fold(cands[["id", "seq"]], t, outdir / "boltz", seeds=seeds, gpus=gpus[:max(1, len(gpus) - 1)] if len(gpus) > 1 else gpus); T.add(f"5_boltz_{tag}", time.time() - t0)
        finally: boltz_done.set()
    def run_v():
        t0 = time.time()
        if oracle2 == "protenix-v2":
            common.set_gpu(gpus[-1]); res["v"] = common.protenix_fold(cands[["id", "seq"]], t, outdir / "v2", arm="v2", seed=seeds[0]); T.add(f"5_v2_{tag}", time.time() - t0)
        elif oracle2 == "af3":                     # every GPU pulls designs from one queue; the GPU Boltz-2 is using joins when Boltz-2 is done
            hold = {gpus[0]: boltz_done} if len(gpus) > 1 and gpus[0] is not None else None
            res["v"] = oracles.af3_fold(cands[["id", "seq"]], t, outdir / "af3", seed=seeds[0], gpus=gpus, hold=hold); T.add(f"5_af3_{tag}", time.time() - t0); T.add(f"5_af3_gpu_s_{tag}", res["v"].attrs.get("gpu_seconds", 0.0))
        else: raise ValueError(f"unknown second oracle {oracle2!r}; choose from {oracles.ORACLES}")
    if len(gpus) > 1:                                                       # Boltz seeds on gpus[:-1], second oracle on the last GPU (AF3: on all), at the same time
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(2) as ex: list(ex.map(lambda f: f(), [run_b, run_v]))
    else: run_b(); run_v()
    b, v = res["b"], res["v"]
    d = cands.merge(b, on="id", how="left").merge(v, on="id", how="left")
    d["o2_name"] = oracle2
    d["b_gate"] = (d.b_ipsae >= 0.5) & (d.b_paemin <= 2.0); d["v2_pass"] = d.v2_ipsae >= o2_gate; d["consensus_pass"] = d.b_gate & d.v2_pass
    d["consensus_min"] = oracles.consensus_score(d, ("b_ipsae", "v2_ipsae"), "min")                                  # gate-style: worst judge
    d["rank_rule"] = rank_rule; d["consensus"] = oracles.consensus_score(d, ("b_ipsae", "v2_ipsae"), rank_rule)       # what the shortlist is ordered by
    hc = [common.hotspot_contacts(r.b_cif, len(t["seq"]), t["hotspot_idx"]) if isinstance(r.b_cif, str) else (float("nan"), 0) for r in d.itertuples()]
    d["hotspot_frac"] = [h[0] for h in hc]; d["hotspot_n"] = [h[1] for h in hc]
    if t.get("site_dir"):                                  # ligand-site occlusion (e.g. FimH mannose pocket)
        so = pd.DataFrame([common.site_occlusion(r.b_cif, t) if isinstance(r.b_cif, str) else {} for r in d.itertuples()], index=d.index); d = pd.concat([d, so], axis=1)
    return add_pisa(d, t)

SC_FLAG = 0.53   # ~5th percentile of wet-lab binders on this SC implementation (x1 scale: 0.532); see docs/RECOMMENDATIONS.md

def add_pisa(d, t, workers=4):
    """fastPISA + shape-complementarity columns on the Boltz-2 complexes (CPU, ~1 s each). Report-only: they never enter `consensus` or the ranking.
    `pisa_flags` lists interfaces below the 5th percentile of wet-lab binders in the ProteinBase benchmark (see docs)."""
    try:
        import pisa
        items = [(r.id, r.b_cif, t["hotspot_idx"]) for r in d.itertuples() if isinstance(r.b_cif, str)]
        m = pd.DataFrame(pisa.batch(items, workers=workers)).add_prefix("pisa_").rename(columns={"pisa_id": "id"})
        d = d.merge(m, on="id", how="left")
        def flags(r):
            f = []
            if r.pisa_interface_area < 630: f.append("thin_interface")
            if r.pisa_n_hydrogen_bonds < 4: f.append("few_hbonds")
            if r.pisa_n_aromatic_iface == 0: f.append("no_aromatic_contact")
            if r.pisa_bsa_apolar_frac < 0.33: f.append("polar_interface")
            if r.pisa_sc < SC_FLAG: f.append("low_shape_compl")
            return ";".join(f)
        d["pisa_flags"] = [flags(r) for r in d.itertuples()]
    except Exception as e:                       # optional dependency: never block a run
        print("fastPISA metrics skipped:", repr(e)[:150], flush=True)
    return d

def shortlist(d, top, thr=0.6):
    keep = []
    for r in d.sort_values("consensus", ascending=False).itertuples():
        if r.consensus != r.consensus: continue
        if not any(common.identity(r.seq, k.seq) >= thr for k in keep): keep.append(r)
        if len(keep) >= top: break
    return pd.DataFrame(keep).drop(columns="Index", errors="ignore")

def finalize_variant(t, out, tag, src, final_m, top, T, state, force, gpus=None, judge=None):
    judge = judge or dict(seeds=(1, 2, 3), oracle2="protenix-v2", o2_gate=0.5, rank_rule="min")
    c = src.sort_values("fast_ipsae", ascending=False).head(final_m)[["id", "seq"]].copy(); cf = out / f"consensus_{tag}.csv"
    if cf.exists() and not force and set(pd.read_csv(cf).id) == set(c.id) and (out / f"final_{tag}.csv").exists():
        d = pd.read_csv(cf)
        if (d["rank_rule"].iloc[0] if "rank_rule" in d else "min") != judge["rank_rule"]:               # predictions are final; only the ordering changes -> cheap
            d["consensus_min"] = oracles.consensus_score(d, ("b_ipsae", "v2_ipsae"), "min"); d["rank_rule"] = judge["rank_rule"]
            d["consensus"] = oracles.consensus_score(d, ("b_ipsae", "v2_ipsae"), judge["rank_rule"]); save_csv(d, cf); save_csv(shortlist(d, top), out / f"final_{tag}.csv")
            log(f"[{tag}] predictions reused; re-ranked by {judge['rank_rule']}"); return
        log(f"[{tag}] already complete for these {len(c)} candidates"); return
    d = consensus(t, c, out / f"consensus_{tag}", judge["seeds"], T, tag, gpus, judge["oracle2"], judge["o2_gate"], judge["rank_rule"]); save_csv(d, cf); s = shortlist(d, top); save_csv(s, out / f"final_{tag}.csv")
    state.mark(f"consensus_{tag}", "done", evaluated=len(d), passed=int(d.consensus_pass.sum()), shortlist=len(s))
    log(f"[{tag}] consensus-pass {int(d.consensus_pass.sum())}/{len(d)} of the top-{final_m}; shortlist {len(s)}; best consensus {d.consensus.max():.3f}")

JUDGE_SPECIFIC = ("consensus_", "final_")               # prefixes of per-judge outputs; everything else in a run folder is judge-independent

def fork_run(src, dst, judge_cfg=None):
    """Start run folder `dst` from the judge-independent work of `src` (generation, design, fast screen, cycling): files are hard-linked, so it costs no
    disk and no GPU time, and a second judge sees exactly the same candidates. Boltz-2 seed predictions are linked too, so a judge that reuses seed 1
    does not fold it again. Timers and stage state of the judging stages are NOT carried over (each judge reports its own cost)."""
    src, dst = Path(src), Path(dst)
    if not (src / "run_config.json").exists(): raise SystemExit(f"--fork-from {src}: not a run folder (no run_config.json)")
    if dst.exists() and any(dst.iterdir()): raise SystemExit(f"--fork-from needs an empty or new --out, found files in {dst}")
    dst.mkdir(parents=True, exist_ok=True)
    def link(a, b):
        try: os.link(a, b)
        except OSError: shutil.copy2(a, b)                                  # different filesystem
    for item in sorted(src.iterdir()):
        if item.name.startswith(JUDGE_SPECIFIC) or item.name in ("final_design", "run_args.json", "run_state.json", "timers.json", "run_config.json"): continue
        if item.is_dir(): shutil.copytree(item, dst / item.name, copy_function=link)
        else: link(item, dst / item.name)
    for cdir in sorted(src.glob("consensus_*")):                           # share Boltz-2 predictions only (the other judge's predictions are not ours)
        if cdir.is_dir() and (cdir / "boltz").is_dir(): shutil.copytree(cdir / "boltz", dst / cdir.name / "boltz", copy_function=link)
    cfg = json.load(open(src / "run_config.json")); [cfg.pop(k, None) for k in LEGACY]; cfg.update(judge_cfg or {}); save_json(cfg, dst / "run_config.json")      # the fork is bound to the judge it was made for
    st = json.load(open(src / "run_state.json")) if (src / "run_state.json").exists() else {"stages": {}}
    st["stages"] = {k: v for k, v in st["stages"].items() if not k.startswith(("consensus_", "complete"))}; save_json(st, dst / "run_state.json")
    tm = json.load(open(src / "timers.json")) if (src / "timers.json").exists() else {}
    save_json({k: v for k, v in tm.items() if not k.startswith("5_")}, dst / "timers.json")
    log(f"forked {src} -> {dst}: judge-independent stages linked, judging stages left to run")

def print_status(out):
    out = Path(out); st = json.load(open(out / "run_state.json")) if (out / "run_state.json").exists() else {"stages": {}}
    print(f"run folder: {out}"); cfg = out / "run_config.json"
    if cfg.exists(): print("config:", {k: v for k, v in json.load(open(cfg)).items() if k in GUARD + ("n_backbones", "rounds", "final_m")})
    for k, v in st["stages"].items(): print(f"  {k:18s} {v['status']:8s} {v['updated']}  " + "  ".join(f"{a}={b}" for a, b in v.items() if a not in ('status', 'updated')))
    for nm in ("gen/converted.json", "screen.csv", "start_parents.csv", "cycled_parents.csv", "consensus_nocycle.csv", "consensus_cycled.csv", "final_nocycle.csv", "final_cycled.csv", "final_design/README.md"):
        print(f"  [{'x' if (out / nm).exists() else ' '}] {nm}")
    gens = sorted((out / "gen").glob("chunk_*/done.json")); parts = sorted((out / "screen_parts").glob("part_*.csv")); cyc = out / "cycle" / "state.json"
    print(f"  generation chunks done: {len(gens)}; screened parts: {len(parts)}; cycling rounds done: {json.load(open(cyc))['rounds_done'] if cyc.exists() else 0}")

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target"); ap.add_argument("--out", required=True); ap.add_argument("--status", action="store_true", help="print progress of --out and exit")
    ap.add_argument("--n-backbones", type=int, default=500); ap.add_argument("--chunk", type=int, default=100, help="backbones per generation unit (resume granularity)")
    ap.add_argument("--seqs", type=int, default=4); ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--cycle-k", type=int, default=100, help="backbones entering cycling (= top fraction by fast score)")
    ap.add_argument("--rounds", type=int, default=3, help="cycling rounds; raise it later to continue cycling from the saved round"); ap.add_argument("--n-new", type=int, default=8)
    ap.add_argument("--cycle-scope", default="full", help="full | interface[:CUTOFF_A]: which binder residues cycling may change (interface keeps the rest fixed)")
    ap.add_argument("--final-m", type=int, default=60, help="designs given the expensive consensus"); ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--no-cycle-ablation", action="store_true", help="skip scoring the pre-cycling shortlist")
    ap.add_argument("--mpnn-weights", default="soluble", choices=["soluble", "original"]); ap.add_argument("--mpnn-bias", default="none", help="none | iface | iface:SCALE  (interface-only hydrophobic/aromatic logit bias, SCALE multiplies it)")
    ap.add_argument("--designs-csv", help="skip generation + MPNN: screen these sequences (columns: seq, [bb], [id])")
    ap.add_argument("--gpus", default=None, help="comma list, e.g. 0,1,2,3 (or $FUNNEL_GPUS): units run in parallel, one worker per GPU; seeds of the consensus run on separate GPUs")
    ap.add_argument("--judge", default="legacy", choices=sorted(JUDGES), help="legacy: Boltz-2 x3 seeds + Protenix-v2 (as published). new: Boltz-2 x1 + AlphaFold3 x1 (needs $PXD_AF3_*). Opt-in until validated on your targets")
    ap.add_argument("--second-oracle", choices=oracles.ORACLES, help="override the judge's second model"); ap.add_argument("--boltz-seeds", help="override the Boltz-2 seeds, e.g. 1,2,3")
    ap.add_argument("--rank-rule", choices=oracles.RANK_RULES[:2], help="order the shortlist by min (worst judge) or mean of the two ipSAE values; default per judge (legacy: min, new: mean). Changing it re-ranks finished predictions without recomputing")
    ap.add_argument("--o2-gate", type=float, default=0.5, help="ipSAE gate on the second oracle. 0.5 was set for Protenix-v2; not calibrated for other models")
    ap.add_argument("--fork-from", help="start --out from the judge-independent stages of this finished run (hard links) so two judges compare on identical candidates")
    ap.add_argument("--force", action="store_true", help="redo finished units"); ap.add_argument("--allow-config-change", action="store_true")
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.status: return print_status(out)
    if not a.target: ap.error("--target is required")
    jd = JUDGES[a.judge]; second = a.second_oracle or jd["second_oracle"]; bseeds = a.boltz_seeds or jd["boltz_seeds"]
    if a.fork_from: fork_run(a.fork_from, out, dict(judge=a.judge, second_oracle=second, boltz_seeds=bseeds))
    judge = dict(seeds=tuple(int(x) for x in bseeds.split(",")), oracle2=second, o2_gate=a.o2_gate, rank_rule=a.rank_rule or jd["rank_rule"])
    t = common.load_target(a.target); T = Timers(out / "timers.json"); state = RunState(out / "run_state.json")
    sha = hashlib.sha1(open(a.designs_csv, "rb").read()).hexdigest()[:12] if a.designs_csv else None
    cfg = dict(target=t["name"], hotspot_idx=t["hotspot_idx"], binder_length=t["binder_length"], chunk=a.chunk, steps=a.steps, seqs=a.seqs, mpnn_weights=a.mpnn_weights, mpnn_bias=a.mpnn_bias,
               cycle_scope=a.cycle_scope, designs_csv_sha=sha, judge=a.judge, second_oracle=second, boltz_seeds=bseeds, n_backbones=a.n_backbones, rounds=a.rounds, final_m=a.final_m, top=a.top, started=time.strftime("%Y-%m-%d %H:%M:%S"))
    check_config(out, cfg, a.allow_config_change); gpus = common.gpu_list(a.gpus); log("GPUs:", gpus)
    log(f"target {t['name']}: {len(t['seq'])} aa, hotspots {t['hotspots']} -> shard idx {t['hotspot_idx']}, binder {t['binder_length']} aa")
    mp = (a.mpnn_weights, a.mpnn_bias)
    if a.designs_csv: gen, units = None, []
    else: gen, units = stage_generate(t, out, a.n_backbones, a.chunk, a.steps, T, state, a.force, gpus); log("backbones:", gen["n"])
    screen = stage_screen(t, out, gen, units, a.seqs, T, state, a.force, mp, a.designs_csv, gpus); log("screened designs:", len(screen), "fast ok:", int(screen.fast_ok.sum()))
    parents = stage_cycle(t, out, screen, a.cycle_k, a.rounds, a.n_new, T, state, a.force, mp, a.cycle_scope, gpus) if a.rounds > 0 else None
    if parents is None:                                                      # no cycling requested: still record the start set
        ok = screen.dropna(subset=["fast_ipsae"]); st_ = ok.sort_values("fast_ipsae", ascending=False).groupby("bb").head(1).sort_values("fast_ipsae", ascending=False).head(a.cycle_k)
        save_csv(st_, out / "start_parents.csv")
    start = pd.read_csv(out / "start_parents.csv")
    for tag, src in ([] if a.no_cycle_ablation else [("nocycle", start)]) + ([("cycled", parents)] if parents is not None else []):
        finalize_variant(t, out, tag, src, a.final_m, a.top, T, state, a.force, gpus, judge)
    save_json(dict(vars(a), command=" ".join(sys.argv), finished=time.strftime("%Y-%m-%d %H:%M:%S")), out / "run_args.json")
    try:
        import final_design; final_design.build(out, a.target, a.top)          # review package: out/final_design/
    except Exception as e:                                                      # never lose a finished run to a reporting error
        print("final_design skipped:", repr(e)[:200], flush=True)
    state.mark("complete", "done"); log("done. timers:", {k: round(v) for k, v in T.items()})

if __name__ == "__main__": main()
