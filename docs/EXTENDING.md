# Extending the software

The funnel is built around a few narrow, documented seams, so new ideas plug in without touching the rest. Each seam has a test or a smoke command.
**Rule for contributions:** a new route or metric is merged as *experimental* until it has been compared with the default under the identical judge
(`funnel/judge.py`) and its controls (`funnel/controls.py`) on at least two targets; the README table is updated with the result, positive or negative.

## 1. A new target (no code)
```bash
cp funnel/targets/fimh.json funnel/targets/mytarget.json     # edit: pdb_id, chain, range, hotspots like "Y48", binder_length, optional ligand for a site probe
python funnel/fetch_target.py funnel/targets/mytarget.json   # downloads from RCSB, MSA from the ColabFold server, builds the shard; identity-checks the hotspots
python funnel/run_funnel.py --target mytarget --out out/funnel/mytarget --n-backbones 500
```
Choose the site from the structure (contacts with the natural partner or ligand), 3-6 residues on one face. `source.ligand` + `probe_radius` give an occlusion test (does the design sit where the ligand sat).

## 2. A new generator (a table of sequences)
Anything that proposes binders (RFdiffusion, BoltzGen, BindCraft, Mosaic, hallucination, directed evolution, a spreadsheet) enters the funnel as a CSV:
```
seq,bb,id          # seq required; bb = parent-structure group (cycling works per bb); id optional
```
```bash
python funnel/run_funnel.py --target fimh --out out/x --designs-csv my_designs.csv --rounds 0
```
The funnel then does fast screening, optional cycling, Boltz-2 + Protenix-v2 consensus, quality flags and the `final_design/` package. `funnel/dock_redesign.py` is the worked example
(dock scaffolds -> interface-only MPNN -> `--designs-csv`).

## 3. A new design route inside the loop
Cycling calls ProteinMPNN through `funnel/mpnn_run.py` (`bias=`, `scope=`). To change how sequences are proposed (LigandMPNN, an ESM-based sampler, gradient hallucination), keep the contract:
*input = a directory of complexes (chain A target, chain B binder); output = CSV with `name, seq_idx, sequence`*; point `run_funnel.run_mpnn` at it.

## 4. A new metric or filter
Add columns in `run_funnel.add_pisa` (report-only flags) or `funnel/common.py` (a function of the predicted complex). Metrics never change the ranking unless you change `consensus()`;
the rule in this repository is that a metric earns a place in ranking only after it is shown, on labelled data (`bench/`), to add information beyond the two ipSAE scores.
`funnel/sc.py` (shape complementarity) and `funnel/pisa.py` are the templates, including how they were validated.

## 5. A new predictor (judge)
`common.boltz_fold` / `common.protenix_fold` return one row per design (`ipsae`, `iptm`, structure path) and are resumable. A third predictor (AF2-multimer, Chai, OpenFold3) follows the same
shape and is run next to them in `consensus()`; the benchmark in `bench/` shows how to measure whether it adds information (ProteinBase labels).

## 6. Benchmarks and honesty checks that come with the code
`bench/run_benchmark.sh` (scorers vs wet-lab labels), `funnel/controls.py` (decoys and shuffles), `funnel/screen_calibration.py` (how well the fast screen predicts the slow models),
`funnel/judge.py` (identical-protocol comparison of any set of shortlists). Use them before claiming an improvement.

## Resumability contract (keep it when you add stages)
Write results per unit (chunk, round, design) with atomic writes (`run_funnel.save_csv/save_json`), skip units whose result exists, never delete finished units, record settings that change
meaning in `run_config.json` (`GUARD` in `run_funnel.py`). `python funnel/run_funnel.py --out <run> --status` must stay truthful.
