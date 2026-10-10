# Start here: a fresh clone to a running campaign

For someone (or an AI assistant) who has just cloned this repository and needs to start a design campaign. Read this, then
`docs/DESIGN_PATH.md` (the rules and the evidence behind them), then `AGENTS.md` (guardrails). Everything here is computational:
a "pass" means independent predictors agree on a confident interface, not that a protein binds.

## 1. Ten minutes: make the clone usable

The repository holds code and docs. Everything bulky or private lives in a **workspace folder next to it**, never in git:

| in the workspace (mirrors the repo layout) | what it is |
|---|---|
| `.pxd/` | environment links (`envs/pxd`, `envs/boltz`, `envs/openfold3`) and `ext/` (fastPISA and friends) |
| `checkpoints/`, `release_data` | model weights (never redistributed, never in git) |
| `out/` | every run's outputs: funnel runs, judge batches, controls, panels |
| `data/`, `targets/`, `bench/`, `manifests/` | built targets, the benchmark data, private target manifests |
| `scripts/.private_terms` | your deny-list for the public-export scanner (one regex per line) |

```bash
git clone <the repository> <same path as the old working tree, if there was one> && cd <it>
python scripts/link_workspace.py /path/to/workspace --dry-run   # shows what it would link; it never overwrites a file and never deletes non-empty folders
python scripts/link_workspace.py /path/to/workspace
PYTHONPATH=$(pwd) .pxd/envs/pxd/bin/python -m pytest tests -q   # CPU-only; data-dependent tests skip with an instruction
```

* Cloning to the **same path** as the old working tree keeps absolute paths stored in old tables valid (`out` resolves through the link) and, for AI assistants
  that key their notes to the working directory, keeps those notes attached. At a different path, copy the notes across (your private notes folder says how).
* The script puts the links it makes into `.git/info/exclude`, because `out/` in `.gitignore` does not match a symlink; `git status` stays clean.
* No workspace yet? Create the folders above, run `scripts/setup.sh` for the two core environments, and rebuild targets with `funnel/fetch_target.py <target json>`.
* Verify the GPU with a real matmul, never `torch.cuda.is_available()`; run **one GPU job at a time** per card.

## 2. The first day of a new challenge (in this order)

1. **Read the rules page yourself** and write down, with the date, the verified facts: target(s) and construct sequences, oligomeric state, assay conditions, objectives
   in the order the ranking section lists them, size categories, number of designs and format, deadline (and any earlier internal deadline), the novelty rule.
2. **Pick the site from structure, not convention.** `python scripts/campaign/epitope_from_complex.py COMPLEX.pdb TARGET_CHAIN PARTNER [...] --compare OTHER.pdb ...`
   prints the contact table, the face geometry (a 60-85 aa binder covers about 20-28 A) and hotspots in the `"E35"` form. A second, independent partner that touches the
   same residues is the best check. Multi-chain targets use `phbind/`, not `funnel/`.
3. **Calibrate against the native interface**: run fastPISA on the real complex first. If the natural binder trips an interface flag (apolar fraction, aromatics), do not rank on it.
4. **Choose the oracle pair for this target class and validate it on solved complexes** (`docs/DESIGN_PATH.md` section 2). An oracle that scores a known binder 0 cannot rank
   designs. Put carriers (one expected failure, two expected passers, a shuffle of the native binder) in every batch.
5. **Write the selection rule down before the first result** (`scripts/campaign/select_panel.py` documents one). Keep a copy next to the panel.
6. **Use every machine from the first hour**, with static job assignment and results read on the machine that wrote them.
7. If novelty is a rule, screen it **before** spending oracle time.

## 3. Tool map

| step | tool |
|---|---|
| target from a public PDB id, with MSA and a ligand-site probe | `funnel/fetch_target.py`, `scripts/prepare_target.py` |
| epitope and hotspots from a complex | `scripts/campaign/epitope_from_complex.py` |
| generate, screen, cycle, judge | `funnel/run_funnel.py`, `funnel/judge.py --second-oracle`, `funnel/controls.py` |
| another generator's designs | `scripts/campaign/harvest_boltzgen.py` -> `run_funnel.py --designs-csv` |
| a third model family (AF3) on a cluster | `scripts/campaign/make_af3_inputs.py`, parse with `funnel/oracles.py:parse_af3_result` |
| panel selection by a pre-registered rule | `scripts/campaign/select_panel.py` |
| refinement (use with care) | `scripts/campaign/arcrefine_jobs.py`, `arcrefine_worker.sh`; read `docs/RECOMMENDATIONS.md` section 20 first |
| publish without private data | `scripts/export_public.py --dest <new dir> --run-tests`, `scripts/check_public.py` |

## 4. Defaults after two campaigns (re-check each on your own target)

* Three cycling rounds; more bought nothing measurable. On a hard target (fast-screen median near zero) cycling is not optional.
* A joint sequence-structure generator and the diffusion route both belong in the pool; judge them in the **same** batch.
* Binder length 60-70 aa for a compact epitope; longer is a novelty lever, not an affinity lever.
* Soluble ProteinMPNN weights, interface bias off. Never plain ProteinMPNN.
* Boltz-2 (3 seeds) plus a second oracle family that recognised a native binder of this target; AF3 or another family as a report column.
* Judge refinements against their parents in one batch; do not assume a refiner helps.

These come from two campaigns on different targets; the length and refinement findings in particular rest on one target each.

## 5. Open ideas, with what would settle each

| idea | what settles it |
|---|---|
| a co-design / hallucination generator (BoltzDesign1, Mosaic) as a third route (`docs/RECOMMENDATIONS.md` section 12) | judge its designs in the same batch as the other two routes, with controls |
| does a refiner help when its parents are weak rather than already passing? | refine a deliberately mixed-quality set and judge parents and refinements together |
| how many cycling rounds on an easy vs a hard target | same-batch comparison of rounds 0, 3, 6 on two targets |
| structural diversity of a panel (sequence identity is not enough) | structural clustering of the predicted complexes |
| a native binder as a calibration anchor for the second oracle | fold several known binders of the target class, not one |

## 6. Rules that were paid for

Never `pkill -f` a pattern in your own command line. Capture `$?` on the next line. `State=COMPLETED`, exit 0 and a non-empty table are not success: check the per-row status.
Never type hotspot indices by hand. Never compare scores across batches. Never report a pass as a binder. Never push the working tree: build a clean export, scan it, review it.
Full list with the failure behind each: `docs/DESIGN_PATH.md` section 12 and `AGENTS.md`.
