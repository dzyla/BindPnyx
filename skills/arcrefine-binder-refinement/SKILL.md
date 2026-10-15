---
name: arcrefine-binder-refinement
description: Refine an EXISTING protein binder with ArcRefine (structural carryover through Boltz-2, built on Mosaic) while preserving its fold and binding mode. Use when a design pool has plateaued and you want to improve specific candidates rather than generate more, or when asked to "improve/refine/optimise these binders", "polish the panel before ordering", "make this design better without changing where it binds". Carries the measured outcome statistics from two campaigns (n = 15 and n = 12), including the three facts that decide whether a refinement is usable - ArcRefine's own ipTM is not predictive, low identity to the parent means the binder is gone, and the gains did not replicate on a second target. Also records why a search for "arcrefine" finds nothing on a working install. Not a generator; it needs a parent sequence plus a target.
metadata:
  authored_via: "claude-code"
  measured_on: "two Adaptyv/Anthropic binder campaigns, 2026-10"
---

# ArcRefine: refining an existing binder without losing its binding mode

ArcRefine (FoldArc, github.com/ken-osumi/ArcRefine) optimises a **given** binder sequence while
keeping its fold and binding mode, by carrying Boltz-2's single and pair representations between
sequence updates ("structural carryover"). It is built on Mosaic and keeps Mosaic's
`structural-carryover` command names. Paper: foldarc.com/research/structural-carryover/paper/,
Zenodo DOI 10.5281/zenodo.23115832.

**It is not a generator.** It takes a parent binder + target and returns one optimised sequence. Use
it after a pool exists and has plateaued, as the "improve what you have" lever - the complement to
MPNN-based refold/redesign cycling and to generating more backbones.

**Licence: PolyForm Noncommercial 1.0.0.** Universities and public research institutions may use it
without a separate patent licence; commercial use needs permission from the author. Do not ship its
outputs into a commercial deliverable without checking.

## FIRST: a search for "arcrefine" finds nothing, even on a working install

**Nothing on disk is called `arcrefine`.** The distribution installs as
`foldarc-structural-carryover`; it provides the modules `mosaic` and `structural_carryover` and the
console script `structural-carryover`. Therefore all of these come back empty on a perfectly healthy
install:

```
pip list | grep -i arcrefine        # nothing
python -c "import arcrefine"        # ModuleNotFoundError
find / -iname "*arcrefine*"         # only whatever the operator named their own job folder
```

This has already caused an agent to report the tool as "not installed anywhere" and argue against a
refinement step that the lab had been running for weeks. **Probe for the real names:**

```bash
pip list 2>/dev/null | grep -iE "foldarc|structural.carryover|^mosaic"
command -v structural-carryover
python -c "import structural_carryover, mosaic; print(structural_carryover.__file__)"
```

A name-based miss is **no evidence either way**. Ask the operator where the environment is before
concluding anything about availability, and check the lab's local guidelines file - the install is
typically a self-contained conda env parked next to the campaign that used it, not on `PATH`.

## Install (own environment - do not mix with another Mosaic)

```bash
H=<install dir>; cd $H
git clone https://github.com/ken-osumi/ArcRefine.git
conda create -y -p $H/env python=3.12
$H/env/bin/python -m pip install uv
cd ArcRefine
export UV_PROJECT_ENVIRONMENT=$H/env
$H/env/bin/uv pip sync    --python $H/env/bin/python requirements-linux-cuda12.lock
$H/env/bin/uv pip install --python $H/env/bin/python --no-deps .
$H/env/bin/python -c "import jax; print(jax.__version__, jax.devices())"   # end with an import test
```

Needs Linux + NVIDIA CUDA (the lock file is CUDA 12), `libgomp1`, `uv`. Boltz-2 weights download on
first use; soluble ProteinMPNN weights ship with it. `$H/env/bin/structural-carryover` is the
`ARC_BIN` that `scripts/campaign/arcrefine_worker.sh` expects.

## Run

`config.json` - one entry per target chain, paths relative to the JSON,
`template_pdb`/`template_chain`/`use_msa` optional:

```json
{"name": "my_design",
 "parent_sequence": "<binder to improve>",
 "targets": [{"id": "A", "sequence": "<target chain>"}],
 "carryover": true,
 "phase_steps": [50, 50, 15],
 "seed": 37,
 "recycling_steps": 1,
 "sampling_steps": 25,
 "num_samples": 1}
```

```bash
export CUDA_VISIBLE_DEVICES=<one card> XLA_PYTHON_CLIENT_PREALLOCATE=false
export HF_HOME=$HOME/.cache/huggingface
$H/env/bin/structural-carryover config.json --validate-only      # always first; no GPU, no MSA server
$H/env/bin/structural-carryover config.json --output run
```

`--output` must name a directory that does not yet exist. Other verified flags: `--carryover
{on,off}`, `--steps SOFT SHARPEN FINAL`, `--seed`, `--no-predict`. Outputs land in `run/`:
`optimized.fasta`, `result.json` (has `optimized_sequence`), loss history, `parent_prediction.cif`
and `fresh_prediction.cif` with PAE/pLDDT `.npy` arrays. **CIF B-factors are placeholders - read the
saved pLDDT arrays (0-1 scale), not the B-factors.** One GPU per job; several jobs fit a large card
with preallocation off.

## The three facts that decide whether a refinement is usable

### 1. ArcRefine's own ipTM tells you nothing. Re-judge independently.

Correlation between its reported refined ipTM and the actual change in independent Boltz-2 ipSAE:
**r = +0.08 (n = 14)** - noise. It failed in both directions:

| ArcRefine ipTM | independent Boltz-2 ipSAE (parent -> refined) | verdict |
|---|---|---|
| 0.813 -> **0.534** (down) | 0.76 -> **0.88** (up) | kept |
| 0.844 -> **0.850** (up) | 0.77 -> **0.08** (collapse) | discarded |

Re-derived from the full run log: its ipTM rose by more than 0.02 in 9 runs, **6 of which were still
discarded**. Never filter on its reported confidence, in either direction.

### 2. Low identity to the parent means the binder is gone. High identity does not mean it survived.

ArcRefine rewrites most of the sequence - identity to parent ran **0.07-0.43**. It is not a
conservative tweak. On the first target, identity vs change in independent ipSAE gave r = +0.88:

| identity to parent | n | mean change in ipSAE | kept |
|---|---|---|---|
| 0.00-0.20 | 5 | **-0.646** | 0/5 |
| 0.20-0.34 | 4 | -0.358 | 0/4 |
| 0.34-1.00 | 6 | **+0.212** | 4/6 |

Re-checked 2026-10-15 against the per-run log: **every run below 0.35 identity failed, 9 of 9**,
while at >= 0.35 it was 4 kept of 6. So the rule is **asymmetric** - compute identity first and
discard below ~0.35 before spending oracle time, but do not read surviving identity as success.

### 3. A second target did not replicate rule 2, and refinement net-hurt it

12 refinements of already-passing, cycled single-chain designs (parents 0.68-0.89 consensus), each
re-judged against its own parent in ONE fresh-seed batch (Boltz-2 x3 seeds + a second oracle family):

| | parents | refined |
|---|---|---|
| passing both oracles | 12 / 12 | **5 / 12** |
| identity >= 0.35 band (n = 10): mean change in consensus | - | **-0.357** (same band was +0.212 on target 1) |
| improved by > 0.03 | - | 2 / 10 (best +0.129) |
| r(identity, change in consensus) | - | **+0.22** (target 1: +0.88) |

The loss was oracle-specific: several refined sequences kept their Boltz-2 ipSAE (8/10 still >= 0.5)
while the second oracle collapsed (0.860 vs 0.077 on one pair); two did the reverse. ArcRefine
optimises through Boltz-2, so that is the expected failure and the reason a second-family re-judge is
mandatory. Untested hypothesis: headroom - target 1's keepers started at 0.56-0.63, these parents at
0.68-0.89, so there was little to gain and much to lose.

**Practical rule: treat ArcRefine as a lottery ticket per design, never as an upgrade.** Keep the
parent; accept the refined sequence only where it wins on the independent oracle in the same batch.

## Validation protocol (the part that is easy to get wrong)

Judge the refined sequence **and its parent in the SAME batch**, with fresh seeds, on an oracle family
that was not inside ArcRefine's loop. ArcRefine optimises against Boltz-2, so Boltz-2 alone is partly
circular - carry a second family (AF3 or OpenFold3) alongside.

```python
pd.DataFrame([dict(id=f"{job}_ARC", seq=result["optimized_sequence"]),
              dict(id=job,          seq=config["parent_sequence"])]).to_csv("val_shard.csv")
```

Then one judging run over `val_shard.csv`. Comparing a refined score against a parent score from an
earlier batch is invalid - batch effects are small but real (SD ~ 0.02).

## Failure modes seen

| symptom | cause | action |
|---|---|---|
| `input error (template residues differ from configured target)` | `template_pdb` residues do not match the `targets` sequence | rerun without the template; it completed fine that way |
| refined sequence scores ~0.00 on every independent judge | carryover drifted off the binding mode | check identity to parent; below ~0.2 discard, do not try to rescue |
| only ArcRefine's ipTM looks good | fact 1 | ignore it; the independent re-judge is the decision |
| "not installed" after a name search | the distribution is `foldarc-structural-carryover` | see the first section; probe the real names and ask the operator |

## What this does not show

n = 15 on the first target and n = 12 on the second, heterogeneous parents, and on the first target
identity is confounded with length (every keeper was a 160-192 aa parent). The 0.35 cut was a post-hoc
split on 15 points and **did not replicate** - re-derive it per target or do not use it.

## In this repository

`docs/RECOMMENDATIONS.md` section 20 (the same evidence in campaign-playbook form),
`scripts/campaign/arcrefine_jobs.py` (build one job folder per design from a judged table; omits
`template_pdb` on purpose, since a residue mismatch aborts the run) and
`scripts/campaign/arcrefine_worker.sh` (static job assignment, resumable, takes `ARC_BIN`).
Per-run outcome logs and the operator's install path are private campaign data and stay in the lab's
local guidelines, not here.
