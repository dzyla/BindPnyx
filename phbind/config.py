"""One config for the whole trimer path, JSON, defaults = the settings the current campaign was run with. An agent changes behaviour by writing a config (never by editing stage code):

  PHBIND_CONFIG=/path/config.json python phbind/s1_queue.py     (or python phbind/run.py --config ... <command>)

Unknown keys are REJECTED (a typo must not silently run the default: BindCraft2 'campaign refused' and Boltz-2 'Skipping' taught the same lesson from the other side).
Every change you make should be logged with phbind/experiments.py, with the reason and what you measured.
"""
from __future__ import annotations
import copy, json, os
from pathlib import Path

DEFAULTS = {
    "generate": {
        "hotspot_sets": {"foot12": ["B87", "B88", "B90", "C21", "C33", "C65", "C67", "C113", "C115", "C144", "C145", "C146"],
                         "decl8": ["B75", "B86", "B87", "C32", "C33", "C115", "C145", "C146"],
                         "core6": ["B87", "B90", "C33", "C65", "C115", "C145"]},
        "sets": ["foot12", "decl8", "core6"],            # which of hotspot_sets to run
        "lengths": [62, 68, 74, 78, 86, 96, 104, 112],   # binder lengths; >= 60 (novelty floor), keep out of the 110-130 Ig-like trap unless you can screen it
        "backbones_per_run": 100, "seqs_per_backbone": 2, "diffusion_steps": 400,
        "mpnn": "soluble",                               # 'soluble' only: plain ProteinMPNN weights are refused
    },
    "prescreen": {"scorer": "boltz", "seed": 101, "batch": 72, "cut": 0.35},   # cut: 1-seed grouped ipSAE (min); 0.45 was too tight (7/7 recall has CI [0.59, 1])
    "gate": {"legs": ["boltz", "af3"], "gate": 0.5, "strong": 0.65, "af3_samples": 5, "rank": "mean"},
    "budget": {"max_gpu_hours": None},                   # informational for agents; stages do not enforce it
}
ALLOWED_MPNN = ("soluble",)

def _merge(base, over, path=""):
    out = copy.deepcopy(base)
    for k, v in over.items():
        if k not in base: raise KeyError(f"unknown config key '{path}{k}' (allowed: {sorted(base)})")
        if isinstance(base[k], dict) and isinstance(v, dict) and k != "hotspot_sets": out[k] = _merge(base[k], v, path + k + ".")
        else: out[k] = v
    return out

def validate(c: dict) -> dict:
    g = c["generate"]
    if g["mpnn"] not in ALLOWED_MPNN: raise ValueError(f"generate.mpnn must be one of {ALLOWED_MPNN} (plain ProteinMPNN gave poor binders)")
    for s in g["sets"]:
        if s not in g["hotspot_sets"]: raise ValueError(f"set '{s}' has no entry in generate.hotspot_sets")
    for name, hs in g["hotspot_sets"].items():
        if not hs or not all(isinstance(h, str) and h[0] in "ABC" and h[1:].isdigit() for h in hs): raise ValueError(f"hotspot set '{name}': use organiser chain+number like 'B87'")
    if any(l < 60 or l > 250 for l in g["lengths"]): raise ValueError("generate.lengths must be within 60..250 (60 is the novelty floor, 250 the contract maximum)")
    if not (0 < c["prescreen"]["cut"] < 1) or not (0 < c["gate"]["gate"] < 1): raise ValueError("cut and gate must be in (0, 1)")
    if len(c["gate"]["legs"]) != 2 or len(set(c["gate"]["legs"])) != 2: raise ValueError("gate.legs must be TWO different models (one seed each); seeds of one model are not a gate")
    if c["gate"]["rank"] not in ("min", "mean"): raise ValueError("gate.rank is 'min' or 'mean'")
    return c

def load(path: str | os.PathLike | None = None) -> dict:
    p = path or os.environ.get("PHBIND_CONFIG")
    over = json.load(open(p)) if p else {}
    return validate(_merge(DEFAULTS, over))
