"""Boltz-2 co-folding of binder + the INTACT human TNF-alpha trimer, and the scoring of what comes out.

Contract (the project's scoring contract): 3 target chains (471 residues, `_nulfix` MSA on every copy), binder
single-sequence with no MSA and no templates, one batch per seed, ipSAE grouped (3 chains = one group) with
the max convention reported beside min. Every failure raises; nothing is gated on an exit status.

Chain order in every YAML is A,B,C (target) then D (binder), so token order == PAE order.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "funnel"))
import trimer as T  # noqa: E402

_MANIFEST = None


def manifest():
    """target_manifest.json written by s0_target.py (generated, never committed). Loaded lazily so importing this module needs no target files."""
    global _MANIFEST
    if _MANIFEST is None:
        f = HERE / "target" / "target_manifest.json"
        if not f.exists():
            raise FileNotFoundError(f"{f} not found: run phbind/s0_target.py first (it needs $TNF_BUNDLE)")
        _MANIFEST = json.load(open(f))
    return _MANIFEST


def msa_human() -> str:
    return manifest()["files"]["msa_human"]["path"]


def msa_mouse() -> str:
    return manifest()["files"]["msa_mouse"]["path"]


def target_for(species: str):
    """(construct sequence, canonical NUL-fixed MSA, residues in the 3-copy oracle target). The species picks all three together; mixing them is the error this prevents."""
    if species == "human":
        return T.TNF_HUMAN, msa_human(), 3 * len(T.TNF_HUMAN)
    if species == "mouse":
        return T.TNF_MOUSE, msa_mouse(), 3 * len(T.TNF_MOUSE)
    raise ValueError(f"species must be 'human' or 'mouse', got {species!r}")


FOOT = {"B": [87, 88, 90], "C": [21, 33, 65, 67, 113, 115, 144, 145, 146]}   # confirmed-binder footprint (12)
GROOVES = [("A", "B"), ("B", "C"), ("C", "A")]        # the three C3-equivalent (face-on-X, face-on-Y) pairs
BINDER_RE = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]{10,250}$")


def boltz_bin():
    import common
    return common.BOLTZ()


def write_yamls(df: pd.DataFrame, out: Path, species: str = "human") -> Path:
    """df: id, seq.  One YAML per design in <out>/yaml. Binder validity is asserted, not assumed."""
    tseq, tmsa, _ = target_for(species)
    d = out / "yaml"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    assert df["id"].is_unique, "duplicate design ids"
    for r in df.itertuples():
        assert BINDER_RE.match(r.seq), f"{r.id}: invalid binder sequence"
        T.write_boltz_yaml(d / f"{r.id}.yaml", tseq, tmsa, r.seq)
    assert len(list(d.glob("*.yaml"))) == len(df)
    return d


def _done(root: Path, i: str) -> bool:
    return (root / i / f"confidence_{i}_model_0.json").exists() and (root / i / f"pae_{i}_model_0.npz").exists()


def run_seed(df: pd.DataFrame, out: Path, seed: int, recycles=3, steps=200, gpu=None, species: str = "human", extra_args=()) -> Path:
    """One Boltz-2 batch at `seed`. Resumable on real outputs. Raises if any design lacks output afterwards."""
    od = out / f"seed{seed}"
    root = od / "boltz_results_yaml" / "predictions"
    missing = [i for i in df["id"] if not _done(root, i)]
    if missing:
        ydir = write_yamls(df[df["id"].isin(missing)], od / "todo", species)
        # the input dir must be named 'yaml': boltz names its output after it
        shutil.rmtree(od / "boltz_results_yaml" / "processed", ignore_errors=True)
        env = None
        if gpu is not None:
            import os
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
        t0 = time.time()
        cmd = [boltz_bin(), "predict", str(ydir), "--out_dir", str(od), "--recycling_steps", str(recycles),
               "--sampling_steps", str(steps), "--diffusion_samples", "1", "--write_full_pae",
               "--accelerator", "gpu", "--override", "--no_kernels", "--seed", str(seed), *map(str, extra_args)]
        # the exact command is part of the result: a flag such as --num_workers changes the prediction (reproducibly), so it must travel with the numbers
        (od / "boltz.cmd").write_text(" ".join(cmd) + "\n")
        p = subprocess.run(cmd, capture_output=True, text=True, env=env)
        (od / "boltz.log").write_text(p.stdout[-30000:] + "\n" + p.stderr[-30000:])
        print(f"seed {seed}: {len(missing)} designs, rc={p.returncode}, {time.time() - t0:.0f}s", flush=True)
    still = [i for i in df["id"] if not _done(root, i)]
    if still:      # the counted-artifact guard: Boltz exits 0 when it skips an input
        raise RuntimeError(f"seed {seed}: {len(still)}/{len(df)} designs have no output (first: {still[:3]}); see {od / 'boltz.log'}")
    return root


def _contacts(coords, meta, tgt_chain, binder_chain, cutoff=5.0):
    """set of resnums (ints) on `tgt_chain` with a heavy atom within cutoff of any binder atom."""
    bm = np.array([m[0] == binder_chain for m in meta])
    tm = np.array([m[0] == tgt_chain for m in meta])
    B, Tt = coords[bm], coords[tm]
    tres = [int(m[1]) for m, k in zip(meta, tm) if k]
    D = np.linalg.norm(Tt[:, None, :] - B[None, :, :], axis=-1).min(1)
    return {r for r, d in zip(tres, D) if d <= cutoff}


def epitope_metrics(cif: str):
    """Descriptors only (handoff section 7 retires them as ranking terms). For each of the three C3-equivalent
    grooves (X = chain carrying face B, Y = chain carrying face C): recall of P1x (33) and of the confirmed
    footprint (12). Reports the best groove and which one. Also per-protomer contact counts."""
    coords, meta = T._heavy_atoms(cif)
    tab = T.chain_table(cif)
    tg = [c["chain_id"] for c in tab if c["seq"] == T.TNF_HUMAN]
    bd = [c["chain_id"] for c in tab if c["seq"] != T.TNF_HUMAN]
    assert len(tg) == 3 and len(bd) == 1, (tg, bd)
    for c in tab:        # a co-fold is numbered 1..N: local numbering is the residue number
        if c["chain_id"] in tg:
            assert c["resnums"][0] == "1" and c["resnums"][-1] == "157", "target numbering is not construct-local"
    con = {c: _contacts(coords, meta, c, bd[0]) for c in tg}
    best = None
    for X, Y in GROOVES:
        p = len(con[X] & set(manifest()["p1x_faces"]["B"])) + len(con[Y] & set(manifest()["p1x_faces"]["C"]))
        f = len(con[X] & set(FOOT["B"])) + len(con[Y] & set(FOOT["C"]))
        if best is None or (f, p) > (best[1], best[0]):
            best = (p, f, X + Y)
    return dict(p1x_recall=best[0] / 33, foot_recall=best[1] / 12, groove=best[2],
                n_prot_engaged=sum(1 for c in tg if len(con[c]) >= 3),
                contacts_per_protomer=";".join(f"{c}={len(con[c])}" for c in tg))


def score_seed(df: pd.DataFrame, out: Path, seed: int, species: str = "human") -> pd.DataFrame:
    n_target = target_for(species)[2]
    root = out / f"seed{seed}" / "boltz_results_yaml" / "predictions"
    rows = []
    for r in df.itertuples():
        d = root / r.id
        pae = np.load(d / f"pae_{r.id}_model_0.npz")["pae"].astype(float)
        cj = json.load(open(d / f"confidence_{r.id}_model_0.json"))
        cif = sorted(d.glob("*_model_0.cif"))[0]
        ti, bi, info = T.group_indices(str(cif), species=species, binder_seq=r.seq, pae_n=pae.shape[0])
        assert info["n_target_residues"] == n_target and info["target_chains"] == ["A", "B", "C"]
        assert info["chain_order"] == ["A", "B", "C", "D"] and pae.shape == (n_target + len(r.seq),) * 2
        lo, hi = T.ipsae_grouped(pae, ti, bi)
        pc = cj["pair_chains_iptm"]      # {i: {j: v}} keyed by chain index as strings; binder is chain 3 (asserted above)
        b2t = float(np.mean([pc["3"][str(k)] for k in range(3)] + [pc[str(k)]["3"] for k in range(3)]))
        row = dict(id=r.id, seed=seed, binder_len=len(r.seq), ipsae_min=lo, ipsae_max=hi,
                   pae_iface_min=T.pae_interface_min_grouped(pae, ti, bi),
                   iptm_global=cj.get("iptm"), b2t_pair_iptm=b2t,
                   cif=str(cif))
        # epitope descriptors use human numbering (P1x faces, footprint); not defined for the mouse target, where they are reported as NaN/'na'
        row.update(epitope_metrics(str(cif)) if species == "human" else dict(p1x_recall=float("nan"), foot_recall=float("nan"), groove="na", n_prot_engaged=-1, contacts_per_protomer=""))
        rows.append(row)
    return pd.DataFrame(rows)


def run(df: pd.DataFrame, out: Path, seeds, species: str = "human", **kw) -> pd.DataFrame:
    """All seeds, ONE batch per seed (never split: --seed is global, so batch composition matters).
    Returns the per-seed table. Reference designs must be inside df."""
    out = Path(out)
    parts = []
    for s in seeds:
        run_seed(df, out, s, species=species, **kw)
        parts.append(score_seed(df, out, s, species))
    res = pd.concat(parts, ignore_index=True)
    res.to_csv(out / "per_seed.csv", index=False)
    return res


def summarise(res: pd.DataFrame) -> pd.DataFrame:
    """Per design over seeds. THE GATE IS CALIBRATED ON THE MIN-DIRECTION GROUPED ipSAE, not on max.

    the project's scoring contract says to use `max`, but its gate (unanimous AND mean >= 0.65 AND worst >= 0.50) comes from the
    sibling campaign's 5-seed table, and re-scoring those 20 designs here (accept_regression.py, 3 seeds, 1 batch) reproduces
    that table with `min` (Spearman 0.93, median |d| 0.032, 5/20 pass == sibling's 5/20) and NOT with `max`
    (bias +0.20, 14/20 pass). Applying 0.65/0.50 to max would admit designs the sibling rejected. So: gate on min, report max.
    """
    g = res.groupby("id")
    s = g.agg(n_seeds=("seed", "nunique"), binder_len=("binder_len", "first"),
              ipsae_mean=("ipsae_min", "mean"), ipsae_worst=("ipsae_min", "min"), ipsae_sd=("ipsae_min", "std"),
              ipsae_max_mean=("ipsae_max", "mean"), ipsae_max_worst=("ipsae_max", "min"),
              b2t_pair_iptm=("b2t_pair_iptm", "mean"), p1x_recall=("p1x_recall", "mean"), foot_recall=("foot_recall", "mean"),
              groove=("groove", lambda x: x.mode().iat[0])).reset_index()
    s["unanimous"] = s["ipsae_worst"] >= 0.5
    s["gate_pass"] = s["unanimous"] & (s["ipsae_mean"] >= 0.65)
    return s
