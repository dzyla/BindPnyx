"""Scorer registry: every co-folding model the trimer path can use, behind ONE contract, with its measured validation record.

Contract: run(df[id, seq], out_dir, **kw) -> DataFrame with `id` and `<prefix>_ipsae_min` (+ `<prefix>_ipsae_max`, epitope descriptors). ipSAE is grouped
(all target copies = one group, binder = the other), MIN direction is what the 0.5 / 0.65 thresholds were fitted on.

Validation records live in phbind/validation.json and are checked by `require()`: a scorer whose record says it cannot do a role (screen / gate) is REFUSED for that role
unless the caller passes force=True. Add or change a record only through phbind/validate.py (positive controls + reference set), never by hand-waving: a benchmark average
on other targets did not transfer to this homotrimer (OpenFold3 gave exactly 0.0 interface confidence to every real binder here).
"""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import pandas as pd

HERE = Path(__file__).resolve().parent
ROLES = ("screen", "gate")

class ScorerRefused(RuntimeError):
    """Raised when a scorer is used for a role its validation record does not support."""

@dataclass(frozen=True)
class Scorer:
    name: str
    prefix: str                      # column prefix of its output
    s_per_design: float              # measured, one RTX 5090, ~560-token complex
    needs: tuple                     # environment variables required
    run: Callable                    # (df, out_dir, **kw) -> DataFrame, lazy-imports its heavy deps

def _boltz(df, out, seed=101, **kw):
    from phbind import boltz_trimer as B
    r = B.run(df, out, [seed], **kw).drop(columns=["cif"], errors="ignore")
    return r.rename(columns={c: "boltz_" + c for c in r.columns if c != "id"})

def _af3(df, out, seed=1, n_samples=5, **kw):
    from phbind import af3_trimer as A
    return A.run(df, out, seed, n_samples=n_samples, **kw)

def _ptx(df, out, seed=101, **kw):
    from phbind import protenix_trimer as P
    return P.run(df, out, seed, arm="v2", **kw)

def _ptx_fast(df, out, seed=101, **kw):
    from phbind import protenix_trimer as P
    r = P.run(df, out, seed, arm="fast", **kw)
    return r.rename(columns={c: "ptxf_" + c[4:] for c in r.columns if c.startswith("ptx_")})

def _of3(df, out, seed=101, **kw):
    from phbind import of3_trimer as O
    return O.run(df, out, seed, **kw)

SCORERS = {s.name: s for s in (
    Scorer("boltz", "boltz", 18.0, (), _boltz),
    Scorer("af3", "af3", 36.0, ("PXD_AF3_PYTHON", "PXD_AF3_DIR", "PXD_AF3_MODELS"), _af3),
    Scorer("ptx", "ptx", 14.0, (), _ptx),
    Scorer("ptx_fast", "ptxf", 0.0, (), _ptx_fast),          # s_per_design filled in from validation.json once measured
    Scorer("of3", "of3", 16.5, (), _of3),
)}

def records(path: Path | None = None) -> dict:
    return json.load(open(path or HERE / "validation.json"))

def require(name: str, role: str, path: Path | None = None, force: bool = False, allow_untested: bool = False) -> Scorer:
    """Return the scorer if its validation record supports `role`; raise ScorerRefused otherwise (with the measured reason)."""
    if role not in ROLES: raise ValueError(f"role must be one of {ROLES}")
    if name not in SCORERS: raise KeyError(f"unknown scorer {name!r}; registered: {sorted(SCORERS)}")
    rec = records(path).get(name)
    if force: return SCORERS[name]
    if rec is None or rec.get("status") == "untested":
        if allow_untested: return SCORERS[name]
        raise ScorerRefused(f"{name}: no validation record on this target. Run phbind/validate.py first (positive controls + reference set), or pass allow_untested=True for an experiment.")
    ok = rec.get("roles", {}).get(role)
    if not ok: raise ScorerRefused(f"{name} is not validated for role '{role}': {rec.get('why', '')} (evidence: {rec.get('evidence', '')})")
    return SCORERS[name]

def check_contract(df: pd.DataFrame, scorer: Scorer) -> pd.DataFrame:
    need = {"id", f"{scorer.prefix}_ipsae_min"}
    if not need <= set(df.columns): raise ValueError(f"{scorer.name} output lacks {sorted(need - set(df.columns))}")
    if df[f"{scorer.prefix}_ipsae_min"].isna().any(): raise ValueError(f"{scorer.name}: NaN scores (a model result is missing; never fall back to another model)")
    return df
