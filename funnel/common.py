"""Shared pieces for the funnel: target loading, folding wrappers (Boltz-2 / Protenix), ipSAE, hotspot contacts.

No package __init__ on purpose: a directory with __init__.py at the repo root would shadow installed packages.
Everything here runs in the project's `pxd` env; Boltz-2 is a subprocess in the `boltz` env."""
import json, os, re, subprocess, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent

def _env_bin(var, rel):
    p = os.environ.get(var)
    if p: return str(Path(p))
    q = REPO / ".pxd" / "envs" / rel
    if not q.exists(): raise FileNotFoundError(f"{rel} not found: set ${var} or run scripts/setup.sh (looked in {q})")
    return str(q)

PXD_PY = lambda: _env_bin("PXD_PYTHON", "pxd/bin/python")
PXD_BIN = lambda name: str(Path(PXD_PY()).parent / name)
BOLTZ = lambda: _env_bin("PXD_BOLTZ_BIN", "boltz/bin/boltz")
OF3 = lambda: _env_bin("PXD_OF3_BIN", "openfold3/bin/run_openfold")
OF3_CKPT = lambda: os.environ.get("PXD_OF3_CKPT", str(Path.home() / ".openfold3" / "of3-p2-155k.pt"))   # the checkpoint the benchmark's `of3` oracle used; another one invalidates the measured gain
CKPT = lambda: os.environ.get("PXD_CHECKPOINTS", str(REPO / "checkpoints"))

# ----------------------------------------------------------------------------- GPUs (multi-GPU workstations)
import threading
_tls = threading.local()

def set_gpu(g): _tls.gpu = g

def gpu_env():
    """os.environ with CUDA_VISIBLE_DEVICES set for the calling worker thread (unchanged if the worker has no GPU assigned)."""
    e = dict(os.environ); g = getattr(_tls, "gpu", None)
    if g is not None: e["CUDA_VISIBLE_DEVICES"] = str(g)
    return e

def gpu_list(arg=None):
    """'0,1,2,3' (argument or $FUNNEL_GPUS) -> ['0','1','2','3']; [None] means 'whatever the process sees' (single-GPU behaviour)."""
    s = arg or os.environ.get("FUNNEL_GPUS") or ""
    return [x.strip() for x in s.split(",") if x.strip() != ""] or [None]

def parallel_map(fn, items, gpus):
    """Run fn(item) for every item, one worker thread per GPU, each pinned to its GPU via CUDA_VISIBLE_DEVICES for the subprocesses it launches.
    Sequential when there is one GPU. Exceptions propagate. Results keep the order of `items`."""
    items = list(items); gpus = gpus or [None]
    if len(gpus) == 1 or len(items) < 2:
        set_gpu(gpus[0]); return [fn(x) for x in items]
    from concurrent.futures import ThreadPoolExecutor
    res = [None] * len(items); n = min(len(gpus), len(items))
    def worker(w):
        set_gpu(gpus[w])
        for i in range(w, len(items), n): res[i] = fn(items[i])
    with ThreadPoolExecutor(n) as ex: list(ex.map(worker, range(n)))
    return res

# ----------------------------------------------------------------------------- targets
def load_target(name_or_path):
    p = Path(name_or_path)
    if not p.exists(): p = REPO / "funnel" / "targets" / f"{name_or_path}.json"
    t = json.load(open(p))
    if "shard" not in t:                      # public-style definition: built from a PDB id by funnel/fetch_target.py into data/targets/<name>/
        base = REPO / "data" / "targets" / t["name"]
        if not (base / "prepared" / f"{t['name']}.pkl.gz").exists():
            raise FileNotFoundError(f"target '{t['name']}' not built yet: run  python funnel/fetch_target.py {p}")
        t.update(shard=str((base / "prepared" / f"{t['name']}.pkl.gz").relative_to(REPO)), provenance=str((base / "prepared" / "provenance.json").relative_to(REPO)),
                 msa_dir=str((base / "msa" / "A" / "0").relative_to(REPO)), source_chain=t["source"]["chain"])
        if (base / "site" / "probe.npy").exists(): t["site_dir"] = str(base / "site")
    rmap = json.load(open(REPO / t["provenance"]))["residue_map"]; rm = rmap.get(t["source_chain"]) or next(iter(rmap.values()))   # keyed by source or output chain depending on how the shard was built
    by = {e["source_resnum"]: e for e in rm}
    t["seq"] = "".join(e["residue"] for e in sorted(rm, key=lambda e: e["shard_resnum"]))
    t["hotspot_idx"] = []                       # 1-based index into the shard / folded target chain
    for h in t["hotspots"]:
        e = by.get(int(h[1:]))
        if e is None or e["residue"] != h[0]:
            raise ValueError(f"{t['name']}: hotspot {h} not found with that identity in the structure's residue map")
        t["hotspot_idx"].append(e["shard_resnum"])
    t["msa"] = str((REPO / t["msa_dir"] / "non_pairing.a3m").resolve())
    q = [l.strip() for l in open(t["msa"]) if not l.startswith(">")][0].replace("-", "")
    if q != t["seq"]: raise ValueError(f"{t['name']}: MSA query does not equal the structure sequence")
    t["shard_abs"] = str((REPO / t["shard"]).resolve())
    return t

# ----------------------------------------------------------------------------- site (ligand) occlusion
def _kabsch(P, Q):
    """Rotation R and translation t with  R @ q + t ~ p  for matched point sets (rows)."""
    pc, qc = P.mean(0), Q.mean(0); U, S, Vt = np.linalg.svd((Q - qc).T @ (P - pc)); d = np.sign(np.linalg.det(Vt.T @ U.T)); R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, pc - R @ qc

def _two_chains(ch, what):
    """funnel/common assumes target = first chain, binder = second. On a homo-oligomer target (3+ chains) that silently measures target-vs-target
    (the binder is read as a target chain). Refuse; the trimer-aware path is phbind/ (group_indices, epitope_metrics)."""
    ids = sorted(ch)
    if len(ids) != 2:
        raise ValueError(f"{what}: expected exactly 2 chains (target, binder), found {len(ids)} {ids}. Multi-chain targets must use phbind/ "
                         "(sequence-based grouping); this function would score target against target.")
    return ids

def site_occlusion(cif, t):
    """Where a bound ligand sat in the reference structure, does the predicted binder sit? Aligns the predicted target onto the reference by CA, maps the ligand
    'probe' atoms (the part deepest in the pocket) into the prediction and measures binder heavy atoms near them.
    Returns dict(site_min_dist, site_clash [binder atoms < 2.0 A from a probe atom], site_near [probe atoms with a binder atom < 3.5 A], site_frac_near)."""
    from Bio.PDB import MMCIFParser
    if not t.get("site_dir"): return {}
    d = Path(t["site_dir"]); ref_ca = np.load(d / "ref_ca.npy"); probe = np.load(d / "probe.npy")
    ch = {c.id: c for c in MMCIFParser(QUIET=True).get_structure("x", str(cif))[0]}; ids = _two_chains(ch, "site_occlusion")
    ca = np.array([r["CA"].coord for r in ch[ids[0]] if "CA" in r]); b = np.array([a.coord for r in ch[ids[1]] for a in r if a.element != "H"])
    if len(ca) != len(ref_ca): return {}
    R, tr = _kabsch(ca, ref_ca); pr = probe @ R.T + tr
    D = np.linalg.norm(pr[:, None, :] - b[None, :, :], axis=-1); m = D.min(1)
    return dict(site_min_dist=float(m.min()), site_clash=int((D < 2.0).sum()), site_near=int((m < 3.5).sum()), site_frac_near=float((m < 3.5).mean()))

# ----------------------------------------------------------------------------- metrics
def _d0(n): return 1.24 * np.cbrt(np.maximum(n, 27) - 15) - 1.8

def _ipsae_dir(pae, a, b, cutoff=10.0):
    sub = pae[np.ix_(a, b)]; valid = sub < cutoff; n0 = valid.sum(1)
    ptm = 1.0 / (1.0 + (sub / _d0(n0)[:, None]) ** 2)
    return float(np.where(n0 > 0, (ptm * valid).sum(1) / np.maximum(n0, 1), 0.0).max())

def ipsae(pae, n_target, n_binder):
    """(min, max) of the two directional ipSAE values; chain order target, binder."""
    pae = np.asarray(pae, float); t = np.arange(n_target); b = np.arange(n_target, n_target + n_binder)
    x, y = _ipsae_dir(pae, b, t), _ipsae_dir(pae, t, b); return min(x, y), max(x, y)

def ipsae_directional(pae, n_target, n_binder):
    """(binder->target, target->binder) ipSAE: PAE rows are the aligned residues (binder rows scored on target columns for the first). Report-only:
    on the 1,320-design release the binder->target direction ranked slightly better within target (AUROC 0.718 vs 0.701 for the min), which is not
    enough to move the funnel's gate, calibrated on the min. Whether the release's `ipsae_binder_to_target` uses this exact row/column convention
    is unverified here (the PAE files are not on this machine)."""
    pae = np.asarray(pae, float); t = np.arange(n_target); b = np.arange(n_target, n_target + n_binder)
    return _ipsae_dir(pae, b, t), _ipsae_dir(pae, t, b)

def pae_interface_min(pae, nt):
    return float(min(pae[nt:, :nt].min(), pae[:nt, nt:].min()))

def hotspot_contacts(cif, nt, hotspot_idx, cutoff=5.0):
    """Fraction of hotspot residues with a binder heavy atom within `cutoff` A; also the number contacted."""
    from Bio.PDB import MMCIFParser
    ch = {c.id: c for c in MMCIFParser(QUIET=True).get_structure("x", str(cif))[0]}
    ids = _two_chains(ch, "hotspot_contacts")
    tgt, bnd = ch[ids[0]], ch[ids[1]]
    batoms = np.array([a.coord for r in bnd for a in r if a.element != "H"])
    tres = [r for r in tgt]
    hit = 0
    for i in hotspot_idx:
        xyz = np.array([a.coord for a in tres[i - 1] if a.element != "H"])
        if len(xyz) and np.min(np.linalg.norm(xyz[:, None, :] - batoms[None, :, :], axis=-1)) <= cutoff: hit += 1
    return hit / len(hotspot_idx), hit

def identity(a, b): return sum(x == y for x, y in zip(a, b)) / max(len(a), len(b))

def cluster_count(seqs, thr=0.6):
    reps = []
    for s in seqs:
        if not any(identity(s, r) >= thr for r in reps): reps.append(s)
    return len(reps)

# ----------------------------------------------------------------------------- folding
def boltz_fold(df, target, out, seeds=(1,), recycles=3, steps=200, keep_structures=True, gpus=None):
    """df: columns id, seq. Returns one row per id with per-seed and mean metrics. RESUMABLE: designs whose prediction files already exist for a seed are not
    folded again; only the missing ones are run (an interrupted job loses at most the batch in flight). Designs are always ONE batch per seed (never split);
    with several `gpus` the seeds run concurrently on different GPUs."""
    import shutil
    out = Path(out); out.mkdir(parents=True, exist_ok=True); nt = len(target["seq"]); rows = {}
    ya = out / "yaml_all"; ya.mkdir(exist_ok=True)
    cyc = "      cyclic: true\n" if target.get("binder_cyclic") else ""
    for r in df.itertuples():
        (ya / f"{r.id}.yaml").write_text(f"version: 1\nsequences:\n  - protein:\n      id: A\n      sequence: {target['seq']}\n      msa: {target['msa']}\n"
                                         f"  - protein:\n      id: B\n      sequence: {r.seq}\n      msa: empty\n{cyc}")
    t0 = time.time()
    def run_seed(sd):
        od = out / f"seed{sd}"; root = od / "boltz_results_yaml" / "predictions"
        done = lambda i: (root / i / f"confidence_{i}_model_0.json").exists() and (root / i / f"pae_{i}_model_0.npz").exists()
        missing = [r.id for r in df.itertuples() if not done(r.id)]
        if not missing: return
        todo = out / f"todo_seed{sd}" / "yaml"; shutil.rmtree(todo.parent, ignore_errors=True); todo.mkdir(parents=True)     # dir must be named 'yaml': boltz names its output after it
        for i in missing: shutil.copy(ya / f"{i}.yaml", todo / f"{i}.yaml")
        shutil.rmtree(od / "boltz_results_yaml" / "processed", ignore_errors=True)               # force fresh preprocessing of just the todo set
        p = subprocess.run([BOLTZ(), "predict", str(todo), "--out_dir", str(od), "--recycling_steps", str(recycles), "--sampling_steps", str(steps),
                            "--diffusion_samples", "1", "--write_full_pae", "--accelerator", "gpu", "--override", "--no_kernels", "--seed", str(sd)],
                           capture_output=True, text=True, env=gpu_env())
        if p.returncode: raise RuntimeError("boltz failed:\n" + p.stderr[-1500:])
    parallel_map(run_seed, list(seeds), gpus)
    for sd in seeds:
        root = out / f"seed{sd}" / "boltz_results_yaml" / "predictions"
        for r in df.itertuples():
            d = root / r.id; cj, pn = d / f"confidence_{r.id}_model_0.json", d / f"pae_{r.id}_model_0.npz"
            row = rows.setdefault(r.id, dict(id=r.id))
            if not (cj.exists() and pn.exists()): row.setdefault("n_fail", 0); row["n_fail"] += 1; continue
            c = json.load(open(cj)); pae = np.load(pn)["pae"]; lo, hi = ipsae(pae, nt, len(r.seq))
            row.setdefault("ipsae_s", []).append(lo); row.setdefault("b2t_s", []).append(ipsae_directional(pae, nt, len(r.seq))[0]); row.setdefault("paemin_s", []).append(pae_interface_min(pae, nt)); row.setdefault("iptm_s", []).append(c["iptm"])
            if keep_structures and "cif" not in row:
                cifs = list(d.glob("*.cif")); row["cif"] = str(cifs[0]) if cifs else None
    res = []
    for i, row in rows.items():
        if "ipsae_s" not in row: res.append(dict(id=i, b_ok=False)); continue
        res.append(dict(id=i, b_ok=True, b_ipsae=float(np.mean(row["ipsae_s"])), b_ipsae_sd=float(np.std(row["ipsae_s"])), b_ipsae_b2t=float(np.mean(row["b2t_s"])), b_paemin=float(np.mean(row["paemin_s"])),
                        b_iptm=float(np.mean(row["iptm_s"])), b_nseeds=len(row["ipsae_s"]), b_cif=row.get("cif")))
    d = pd.DataFrame(res); d.attrs["seconds"] = time.time() - t0; return d

PTX_MODELS = {  # arm -> (model name, pairformer cycles, diffusion steps)
    "fast": ("protenix_mini_default_v0.5.0", 2, 5),
    "v2": ("protenix-v2", 4, 200),
}

def protenix_fold(df, target, out, arm="fast", seed=101, chunk=1500, tag=""):
    """df: columns id, seq. Returns ipSAE/ipTM/ranking per id and the path of the predicted complex (cif).
    RESUMABLE: designs with a complete result on disk (summary json + full-data json + cif) are not folded again; only the missing ones are run."""
    model, cyc, steps = PTX_MODELS[arm]; out = Path(out); (out / "msa").mkdir(parents=True, exist_ok=True); nt = len(target["seq"])
    tq = out / "msa" / "target_query.a3m"; tq.write_text(f">query\n{target['seq']}\n")
    rows = []; t0 = time.time()
    paths = lambda i: (lambda pr: (pr / f"{i}_summary_confidence_sample_0.json", pr / f"{i}_full_data_sample_0.json", pr / f"{i}_sample_0.cif"))(out / "pred" / i / f"seed_{seed}" / "predictions")
    todo = df[[not all(p.exists() for p in paths(r.id)) for r in df.itertuples()]]
    for c0 in range(0, len(todo), chunk):
        part = todo.iloc[c0:c0 + chunk]; jobs = []
        for r in part.itertuples():
            bq = out / "msa" / f"{r.id}.a3m"; bq.write_text(f">query\n{r.seq}\n")
            jobs.append({"name": r.id, "covalent_bonds": [], "sequences": [
                {"proteinChain": {"sequence": target["seq"], "count": 1, "unpairedMsaPath": target["msa"], "pairedMsaPath": str(tq.resolve())}},
                {"proteinChain": {"sequence": r.seq, "count": 1, "unpairedMsaPath": str(bq.resolve()), "pairedMsaPath": str(bq.resolve())}}]})
        jf = out / f"in_{tag}{c0}.json"; json.dump(jobs, open(jf, "w"))
        p = subprocess.run([PXD_BIN("protenix"), "pred", "-i", str(jf), "-o", str(out / "pred"), "-s", str(seed), "-n", model, "-e", "1", "-c", str(cyc), "-p", str(steps),
                            "--use_msa", "true", "--need_atom_confidence", "true"], capture_output=True, text=True, env={**gpu_env(), "PYTHONPATH": str(REPO)})
        (out / f"log_{tag}{c0}.txt").write_text(p.stdout[-20000:] + p.stderr[-20000:])
        if p.returncode: raise RuntimeError("protenix failed:\n" + p.stderr[-1500:])
    for r in df.itertuples():
        sj, fj, cf = paths(r.id)
        if not all(p.exists() for p in (sj, fj, cf)): rows.append(dict(id=r.id, ok=False)); continue
        s = json.load(open(sj)); pae = np.array(json.load(open(fj))["token_pair_pae"]); lo, hi = ipsae(pae, nt, len(r.seq))
        rows.append(dict(id=r.id, ok=True, ipsae=lo, ipsae_max=hi, ipsae_b2t=ipsae_directional(pae, nt, len(r.seq))[0], iptm=s["iptm"], rank=s["ranking_score"], paemin=pae_interface_min(pae, nt), cif=str(cf)))
    d = pd.DataFrame(rows).add_prefix(f"{arm}_").rename(columns={f"{arm}_id": "id"}); d.attrs["seconds"] = time.time() - t0; return d


def protenix_fold_multi(df, target, out, arm="fast", seed=101, gpus=None):
    """protenix_fold with the designs sharded across GPUs (each design is folded independently, so sharding is safe for Protenix). Same output layout and resume logic."""
    gpus = gpus or [None]
    if len(gpus) == 1 or len(df) < 2 * len(gpus): set_gpu(gpus[0]); return protenix_fold(df, target, out, arm, seed)
    shards = [df.iloc[i::len(gpus)] for i in range(len(gpus))]
    parts = parallel_map(lambda k: protenix_fold(shards[k], target, out, arm, seed, tag=f"g{k}_"), range(len(gpus)), gpus)
    d = pd.concat(parts, ignore_index=True); d.attrs["seconds"] = max(p.attrs.get("seconds", 0) for p in parts); return d


# ----------------------------------------------------------------------------- OpenFold3 (second judge)
def a3m_rectangular(src, dst):
    """Write `src` to `dst` as a rectangular a3m (every row has the query's number of match columns). Returns the number of rows.
    Our target MSAs drop trailing gaps, which OpenFold3's parser rejects; padding on the right is exact. Raises if a row is longer than the query or the file is empty."""
    rows, h = [], None
    for l in open(src):
        l = l.rstrip("\n")
        if l[:1] == ">": h = l
        elif h is not None: rows.append((h, l)); h = None
    if not rows: raise ValueError(f"empty MSA: {src}")
    w = len(re.sub("[a-z]", "", rows[0][1])); out = []
    for h, l in rows:
        n = len(re.sub("[a-z]", "", l))
        if n > w or not l: raise ValueError(f"malformed MSA row in {src}: {h!r} has {n} match columns, query has {w}")
        out += [h, l + "-" * (w - n)]
    Path(dst).write_text("\n".join(out) + "\n"); return len(rows)

def of3_fold(df, target, out, seed=101, chunk=1500, tag=""):
    """OpenFold3 (p2-155k) on target + binder, template-free, target with its unpaired MSA, binder a query-only MSA: the benchmark's `of3` setup.
    df: columns id, seq. Same return layout as protenix_fold with the prefix `of3_` (ipsae, ipsae_max, iptm, rank, paemin, cif). RESUMABLE per design.
    The model is loaded once per chunk, so a chunk should be large."""
    out = Path(out); (out / "msa").mkdir(parents=True, exist_ok=True); nt = len(target["seq"]); tmsa = out / "msa" / "target" / "colabfold_main.a3m"; tmsa.parent.mkdir(exist_ok=True)
    depth = a3m_rectangular(target["msa"], tmsa)
    if depth < 2: raise ValueError(f"target MSA has {depth} row(s): refusing to fold without a real MSA ({target['msa']})")
    base = lambda i: out / "pred" / i / f"seed_{seed}"
    files = lambda i: (base(i) / f"{i}_seed_{seed}_sample_1_confidences.json", base(i) / f"{i}_seed_{seed}_sample_1_confidences_aggregated.json", base(i) / f"{i}_seed_{seed}_sample_1_model.cif")
    todo = df[[not all(p.exists() for p in files(r.id)) for r in df.itertuples()]]; t0 = time.time()
    (out / "runner.yml").write_text(f"experiment_settings:\n  seeds:\n    - {seed}\n")
    for c0 in range(0, len(todo), chunk):
        qs = {}
        for r in todo.iloc[c0:c0 + chunk].itertuples():
            bm = out / "msa" / r.id / "colabfold_main.a3m"; bm.parent.mkdir(exist_ok=True); bm.write_text(f">query\n{r.seq}\n")
            qs[r.id] = {"chains": [{"molecule_type": "protein", "chain_ids": ["A"], "sequence": target["seq"], "main_msa_file_paths": str(tmsa.resolve())},
                                   {"molecule_type": "protein", "chain_ids": ["B"], "sequence": r.seq, "main_msa_file_paths": str(bm.resolve())}]}
        qf = out / f"query_{tag}{c0}.json"; json.dump({"queries": qs}, open(qf, "w"))
        p = subprocess.run([OF3(), "predict", "--query-json", str(qf), "--use-msa-server=False", "--output-dir", str(out / "pred"), "--num-diffusion-samples", "1",
                            "--runner-yaml", str(out / "runner.yml"), "--inference-ckpt-path", OF3_CKPT()], capture_output=True, text=True, env={**gpu_env()})
        (out / f"log_{tag}{c0}.txt").write_text(p.stdout[-20000:] + p.stderr[-20000:])
        if p.returncode: raise RuntimeError("openfold3 failed:\n" + p.stderr[-1500:])   # per-design failures do not raise: they show as ok=False below
    rows = []
    for r in df.itertuples():
        cj, aj, cf = files(r.id)
        if not all(x.exists() for x in (cj, aj, cf)): rows.append(dict(id=r.id, ok=False)); continue
        a = json.load(open(aj)); pae = np.array(json.load(open(cj))["pae"], float)
        if pae.shape != (nt + len(r.seq),) * 2: rows.append(dict(id=r.id, ok=False)); continue
        lo, hi = ipsae(pae, nt, len(r.seq))
        rows.append(dict(id=r.id, ok=True, ipsae=lo, ipsae_max=hi, ipsae_b2t=ipsae_directional(pae, nt, len(r.seq))[0], iptm=a["iptm"], rank=a["sample_ranking_score"], paemin=pae_interface_min(pae, nt), cif=str(cf)))
    d = pd.DataFrame(rows).add_prefix("of3_").rename(columns={"of3_id": "id"}); d.attrs["seconds"] = time.time() - t0; return d
