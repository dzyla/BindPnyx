"""Pluggable second oracle for the funnel's expensive consensus (Boltz-2 + one other co-folding model).

Why this exists: on the 1,320 wet-lab-labelled designs of the Anthropic binder-design release, Boltz-2 alone ranks mid-pack among ten co-folding models
(within-target AUROC 0.70), and a second, architecturally different model adds more than extra Boltz seeds do (see docs/JUDGE_REGIME.md and
tests/test_consensus_regime.py for the pinned numbers). The legacy second oracle is Protenix-v2 (funnel/common.py:protenix_fold). This module adds AlphaFold3
(real weights, MSA injected, no data pipeline) behind the same interface and the shared pure helpers: the consensus rule, input/naming helpers and the
PAE -> ipSAE parser.

Every backend returns the same columns as `common.protenix_fold(..., arm="v2")`: v2_ok, v2_ipsae (min of the two directions), v2_ipsae_max, v2_iptm,
v2_rank, v2_paemin, v2_cif. The `v2_` prefix is kept ON PURPOSE so final_design.py, compare.py, controls.py and saved run folders keep working; the
`o2_name` column says which model produced them.

AF3 configuration (no paths are hard-coded; same contract as PXD_PYTHON / PXD_BOLTZ_BIN):
    PXD_AF3_PYTHON   python of an environment that can run AlphaFold3 (jax with CUDA)
    PXD_AF3_DIR      directory containing run_alphafold.py
    PXD_AF3_MODELS   directory with the model parameters (af3.bin or af3.bin.zst)
The target MSA is the same a3m the Boltz-2 / Protenix path uses (target["msa"]), sanitised once; the binder gets a single-sequence MSA, as in the other paths.
ipSAE is the mean over the diffusion samples of one seed (what the calibration used), not the top-ranked sample alone.
"""
import hashlib, json, os, queue, re, subprocess, threading, time
from pathlib import Path
import numpy as np
import pandas as pd
import common

ORACLES = ("protenix-v2", "af3")
V2_COLS = ("ok", "ipsae", "ipsae_max", "iptm", "rank", "paemin", "cif")

# ----------------------------------------------------------------------------- consensus rule
def consensus_score(df, cols=("b_ipsae", "v2_ipsae"), rule="min"):
    """Per-design score over the oracle columns. A design missing ANY oracle gets NaN (an earlier pandas default silently ranked it on the remaining oracle).
      min    the funnel's historical rule: a design is as good as its worst judge. Right for a PASS/FAIL gate; poor for RANKING.
      mean   mean of the raw ipSAE values. On the 1,320 labelled designs of the Anthropic release it ranks better than min (within-target AUROC +0.025,
             95% CI +0.011..+0.041, paired bootstrap over targets, Boltz-2 + OpenFold3; Boltz-2 + Protenix-v2: min is no better than Boltz-2 alone, mean is
             +0.035 [+0.010, +0.063]). See tests/test_consensus_regime.py for the pinned numbers.
      zmean  mean of within-table z-scores; only meaningful inside one candidate table."""
    x = df[list(cols)].astype(float)
    if rule == "min": return x.min(axis=1, skipna=False)
    if rule == "mean": return x.mean(axis=1, skipna=False)
    if rule == "zmean": return ((x - x.mean()) / x.std(ddof=0).replace(0, np.nan)).mean(axis=1, skipna=False)
    raise ValueError(f"unknown consensus rule {rule!r} (min | mean | zmean)")

RANK_RULES = ("min", "mean", "zmean")

# ----------------------------------------------------------------------------- naming, MSA and input helpers (pure)
_SAFE = re.compile(r"[a-z0-9_.\-]+")

def af3_name(design_id):
    """AF3 lower-cases job names and drops characters outside [a-z0-9_.-] when it names the output folder; use ids that already survive that, else a hash."""
    s = str(design_id)
    return s if _SAFE.fullmatch(s) else "x" + hashlib.sha1(s.encode()).hexdigest()[:12]

def sanitize_a3m(src, dst, query):
    """Copy an a3m keeping only records AF3's parser accepts: no NUL/non-ASCII bytes, only letters and '-', and an aligned length equal to the query's.
    The first record must be the query (checked after removing gaps). Returns {total, kept, dropped}. A truncated record from an interrupted MSA download
    would otherwise abort the whole prediction."""
    raw = Path(src).read_bytes().decode("ascii", errors="replace").replace("\x00", "\ufffd")
    recs, name, seq = [], None, []
    for line in raw.splitlines():
        if line.startswith(">"):
            if name is not None: recs.append((name, "".join(seq)))
            name, seq = line, []
        elif name is not None: seq.append(line.strip())
    if name is not None: recs.append((name, "".join(seq)))
    if not recs or recs[0][1].replace("-", "") != query:
        raise ValueError(f"first record of {src} is not the target query ({len(query)} aa)")
    ok = lambda s: re.fullmatch(r"[A-Za-z\-]+", s) is not None and len(re.sub(r"[a-z]", "", s)) == len(query)
    keep = [(n, s) for n, s in recs if ok(s)]
    Path(dst).write_text("".join(f"{n}\n{s}\n" for n, s in keep))
    return dict(total=len(recs), kept=len(keep), dropped=len(recs) - len(keep))

def af3_input(name, target_seq, msa_path, binder_seq, seed):
    """AF3 fold-input dict for a 1:1 target/binder complex, MSAs injected (run with --norun_data_pipeline)."""
    return {"name": name, "modelSeeds": [int(seed)], "dialect": "alphafold3", "version": 1, "sequences": [
        {"protein": {"id": "A", "sequence": target_seq, "unpairedMsaPath": str(Path(msa_path).resolve()), "pairedMsa": "", "templates": []}},
        {"protein": {"id": "B", "sequence": binder_seq, "unpairedMsa": f">query\n{binder_seq}\n", "pairedMsa": "", "templates": []}}]}

def af3_env(gpu):
    """Hard isolation for one AF3 process: one visible GPU, no VRAM pre-allocation (several processes may share a large card), per-GPU compilation cache."""
    e = dict(os.environ)
    if gpu is not None: e["CUDA_VISIBLE_DEVICES"] = str(gpu)
    e.update(XLA_PYTHON_CLIENT_PREALLOCATE="false", XLA_PYTHON_CLIENT_ALLOCATOR="platform")
    return e

def af3_command(py, run_dir, json_path, out_dir, model_dir, cache_dir, n_samples=5):
    return [str(py), str(Path(run_dir) / "run_alphafold.py"), f"--json_path={json_path}", f"--output_dir={out_dir}", f"--model_dir={model_dir}",
            "--norun_data_pipeline", "--gpu_device=0", f"--num_diffusion_samples={n_samples}", f"--jax_compilation_cache_dir={cache_dir}"]

# ----------------------------------------------------------------------------- reading an AF3 result (pure given the files)
def parse_af3_result(design_dir, name, n_target):
    """Summarise one AF3 output folder. ipSAE/paemin/ipTM are means over all finished diffusion samples; `rank` is the best ranking_score; cif is the
    top-ranked model. Returns {ok: False} when nothing usable is there."""
    d = Path(design_dir); lo, hi, pm, ip, rk = [], [], [], [], []
    for cj in sorted(d.glob("seed-*_sample-*/*_confidences.json")):
        if cj.name.endswith("summary_confidences.json"): continue
        sj = cj.with_name(cj.name.replace("_confidences.json", "_summary_confidences.json"))
        try: J = json.load(open(cj)); S = json.load(open(sj))
        except Exception: continue
        tc = J.get("token_chain_ids"); pae = np.asarray(J.get("pae"), float) if J.get("pae") is not None else None
        if not tc or pae is None: continue
        nt = sum(1 for x in tc if x == tc[0])
        if nt != n_target: raise ValueError(f"{name}: AF3 target chain has {nt} tokens, expected {n_target}")
        a, b = common.ipsae(pae, nt, len(tc) - nt); lo.append(a); hi.append(b); pm.append(common.pae_interface_min(pae, nt))
        ip.append(float(S["iptm"])); rk.append(float(S["ranking_score"]))
    cif = d / f"{name}_model.cif"
    if not lo or not cif.exists(): return dict(ok=False)
    return dict(ok=True, ipsae=float(np.mean(lo)), ipsae_max=float(np.mean(hi)), iptm=float(np.mean(ip)), rank=float(max(rk)), paemin=float(np.mean(pm)), cif=str(cif),
                n_samples=len(lo))

# ----------------------------------------------------------------------------- AF3 runner
def af3_config():
    cfg = {k: os.environ.get(k) for k in ("PXD_AF3_PYTHON", "PXD_AF3_DIR", "PXD_AF3_MODELS")}
    miss = [k for k, v in cfg.items() if not v]
    if miss: raise FileNotFoundError("AlphaFold3 backend not configured: set " + ", ".join("$" + k for k in miss) + " (see funnel/oracles.py)")
    for k in ("PXD_AF3_PYTHON", "PXD_AF3_DIR", "PXD_AF3_MODELS"):
        if not Path(cfg[k]).exists(): raise FileNotFoundError(f"${k} points to a missing path: {cfg[k]}")
    return cfg

def af3_fold(df, target, out, seed=1, gpus=None, n_samples=5, hold=None):
    """df: columns id, seq. Folds target + binder with AF3 (MSA injected), one process per design, one worker thread per GPU pulling from a SHARED queue, so a
    faster card (e.g. an RTX 6000 next to RTX 4000s, ~2x) takes proportionally more designs without hand-tuned weights. `hold` = {gpu: threading.Event}: that
    worker waits for the event before starting (lets a GPU that is busy with Boltz-2 join afterwards). RESUMABLE: designs with a complete result are skipped;
    a failing design is logged and left as ok=False instead of aborting the batch. Returns the v2_* table."""
    cfg = af3_config(); out = Path(out); (out / "in").mkdir(parents=True, exist_ok=True); (out / "logs").mkdir(exist_ok=True); gpus = list(gpus or [None]); hold = hold or {}
    nt = len(target["seq"]); t0 = time.time()
    msa = out / "target_msa.a3m"
    if not msa.exists(): json.dump(sanitize_a3m(target["msa"], msa, target["seq"]), open(out / "target_msa.stats.json", "w"))
    names = {r.id: af3_name(r.id) for r in df.itertuples()}
    json.dump(names, open(out / "name_map.json", "w"))
    for r in df.itertuples(): json.dump(af3_input(names[r.id], target["seq"], msa, r.seq, seed), open(out / "in" / f"{names[r.id]}.json", "w"))
    done = lambda n: bool(parse_af3_result(out / "pred" / n, n, nt).get("ok"))
    q = queue.Queue(); [q.put(r.id) for r in df.itertuples() if not done(names[r.id])]
    errors, busy = [], []                                # busy: seconds of GPU time per design (wall time x 1 card), the cost that matters when cards differ
    def worker(gpu):
        if gpu in hold: hold[gpu].wait()
        cache = out / f"jax_cache_gpu{gpu}"; cache.mkdir(exist_ok=True)
        while True:
            try: i = q.get_nowait()
            except queue.Empty: return
            n = names[i]; ts = time.time()
            p = subprocess.run(af3_command(cfg["PXD_AF3_PYTHON"], cfg["PXD_AF3_DIR"], out / "in" / f"{n}.json", out / "pred", cfg["PXD_AF3_MODELS"], cache, n_samples),
                               capture_output=True, text=True, env=af3_env(gpu))
            busy.append(time.time() - ts); (out / "logs" / f"{n}.log").write_text(p.stdout[-8000:] + p.stderr[-8000:])
            if p.returncode: errors.append((i, p.returncode))
    ths = [threading.Thread(target=worker, args=(g,)) for g in gpus]
    [t.start() for t in ths]; [t.join() for t in ths]
    rows = []
    for r in df.itertuples():
        res = parse_af3_result(out / "pred" / names[r.id], names[r.id], nt); rows.append(dict(id=r.id, **{k: res.get(k) for k in V2_COLS}))
    d = pd.DataFrame(rows).rename(columns={k: f"v2_{k}" for k in V2_COLS}); d.attrs["seconds"] = time.time() - t0; d.attrs["gpu_seconds"] = float(sum(busy)); d.attrs["failures"] = errors
    if errors: print(f"AF3: {len(errors)} design(s) failed (see {out / 'logs'}): {errors[:5]}", flush=True)
    return d
