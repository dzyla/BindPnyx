# Waves: running the trimer pipeline on another machine or GPU

A **wave** is one self-contained campaign (generate -> pack -> novelty -> screen -> gate -> export) on one card, resumable, that returns only small CSVs. Two machines (or two cards) run two
waves in parallel and the results are merged by CSV. Nothing is shared at run time except files in one drop folder.

See also `docs/CLUSTER_LSF.md` for LSF.

## One checkout per wave (enforced)
Two waves in one checkout share `out/phbind` (batch numbering, `designs_all.csv`) and corrupt each other; `wave.py` refuses to start if another live campaign owns the checkout. Clone the repo once per concurrent wave.

## What must differ between waves (or you regenerate the same backbones)
| thing | why | how |
|---|---|---|
| `wave.campaign` | names every output (`backbones_<c>.tar.gz`, `wave_<c>_results.csv`) | any short tag |
| `generate.seed_base` | PXDesign seed = seed_base + 1000*index(set) + length | e.g. 100000 / 300000 / 400000 |
| hotspot-set **aliases** | run ids are `<set>_L<len>`; the same residues under a new name keeps ids unique | `"decl8z": [same residues as decl8]` |

## Prerequisites on the second machine (all via environment variables; nothing is hard-coded)
```bash
git clone https://github.com/dzyla/binder-design && cd binder-design          # code only; no data in git
export PYTHONPATH=$(pwd)
export PXD_PYTHON=<python of the PXDesign env>   PXD_BOLTZ_BIN=<boltz executable>   PXD_CHECKPOINTS=<dir with pxdesign_v0.1.0.pt>
export PXD_AF3_PYTHON=<af3 env python>  PXD_AF3_DIR=<dir with run_alphafold.py>  PXD_AF3_MODELS=<dir with af3.bin.zst>     # the gate's second model
export TNF_BUNDLE=<campaign bundle: targets/, msa/, tables/>                     # organiser structures + repaired MSAs
export PHBIND_CARRIERS=<csv id,seq: >=3 reference designs, ids start with carrier_>   # 2 known passers + 1 known failure, carried in every batch
export CUDA_VISIBLE_DEVICES=<one card>                                          # one GPU job at a time per card; a 24 GB card cannot hold AF3 on a ~650-token complex, use a 48+ GB card
export PHBIND_CONFIG=<wave config json>
```
## Commands
```bash
$PXD_PYTHON phbind/wave.py --plan        # which steps are done (decided from real artifacts), which would run; no GPU
$PXD_PYTHON phbind/wave.py               # runs target -> shard -> generate -> pack -> novelty -> prescreen -> gate -> export, resuming wherever it stopped
python phbind/run.py status         # counts of real artifacts at any time
```
Re-running the same command after any interruption is safe. Steps never gate on exit codes: `generate` is done when every run's `designs.csv` has its expected rows, `gate` when the gate table covers
every survivor, and so on.

## The novelty step does no search
`pack` writes `backbones_<campaign>.tar.gz` (+ index) into `wave.export_dir`. Whoever holds a structure database screens it and writes `novelty_backbones_<campaign>.csv` (columns `bbid,n_strict_hits`,
optionally `best_tm_at_70cov`) into the same folder. The prescreen re-reads that file before **every** batch: backbones with 0 strict hits first, unscreened next, any with a hit never. If nothing arrives within
`wave.novelty_wait_min` the wave proceeds unscreened. (A structural-novelty failure -- a PDB hit at TM >= 0.8 over >= 70% of the chain -- is the portal's own rejection rule; 18 of 21 first-campaign gate passers failed it.)

## What a wave returns
`wave_<campaign>_results.csv` (gate table: Boltz-2 and AF3 scores, consensus, tier, sequence) and `predicted_<campaign>.tar.gz` (predicted binder chains of every design that cleared the first model,
for a final novelty check on exactly what would be submitted). These are consensus passes of two predictors, not validated binders.
