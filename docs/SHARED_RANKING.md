# Shared submission ranking (all agents)

**Do not edit `FINAL_RANKING.*` or `evidence.csv` by hand.** The ranking is recomputed by rule from an append-only ledger of measurements, so three sessions can update it at the same time without overwriting each other, and every rank traces to a measurement, a number of seeds, and who recorded it.

| file | what | who edits |
|---|---|---|
| `designs.csv` | registry, one row per design (`design_id, sequence, lineage, generator, length, parent, note`) | add a row once with `add-design`; sequence may be blank until its owner supplies it |
| `evidence.csv` | **append-only** ledger: `ts, agent, design_id, metric, value, n, note` | append with `add` (never rewrite). A correction is a NEW row |
| `rules.json` | thresholds and tier parameters | change only after agreeing in a note in the shared briefing; one edit |
| `FINAL_RANKING.md` / `.csv` | output: tiers, merit, flags, and the suggested submission ROW ORDER | nobody; regenerate with `rank` |
| `rank.py` | the tool (copy of `phbind/ranking.py` in https://github.com/dzyla/binder-design) | - |

## Commands (any Python with pandas; set RANKING_DIR)
```bash
export RANKING_DIR=<shared ranking dir> ; R="python $RANKING_DIR/rank.py"
$R check                                              # validate ledger + registry (alphabet, no Cys, 60-250 aa, duplicate sequences, unknown metrics)
$R add-design <id> --sequence <SEQ> --lineage <backbone/lineage> --generator <BindCraft2|PXDesign|...> [--parent P] [--note T]
$R add <design_id> <metric> <value> --agent <your session> [--n <seeds>] [--note "NOTE 38, 5-seed human"]
$R rank                                               # rewrite FINAL_RANKING.md/.csv and print the submission order
```
Metrics (`rank.py` lists them): `boltz_h_mean|worst`, `boltz_m_mean|worst` (grouped ipSAE, MIN direction; put the number of seeds in `--n` on the MEAN row), `af3_h`, `af3_m` (**ipSAE, not ipTM**),
`novelty_hits_qtm|best_qtm|hits_alntm|best_alntm`, `ph_delta|ph_se|ph_npose` (multi-pose paired delta).
**Precedence for a (design, metric): more seeds/samples (`--n`) beat fewer, then the newest row wins**: a later single-seed value never overrides a 5-seed one; a correction with the same `n` is just a newer line.

## The rules (in `rules.json`; objective order is the organisers': 1 pH-dependent binding, 2 mouse, 3 human affinity)
- **X excluded**: any structural-novelty hit (`novelty_hits_qtm > 0`: TM >= 0.8 over >= 70% of the chain). Unsubmittable whatever the oracles say. `thin` = clean but best qtm >= 0.77 or an alntm hit (flagged, still eligible).
- **A**: Boltz-2 human mean >= 0.5 AND AF3 human >= 0.5 AND novelty clean/thin. The user's gate is one Boltz-2 seed + one AF3 seed, consensus across models; more seeds add *robustness* (`robust` = mean >= 0.65, worst >= 0.5, n >= 3), not eligibility.
- **B**: one model passes and the other is near (0.45-0.5) / missing / novelty unscreened; or a novelty-clean mouse binder.  **C**: the models disagree (one passes, the other rejects).  **D**: below.
- Within a tier: **pH first** (only a *multi-pose, significant* shift counts: >= 5 poses and delta - 2*se > 0; a single pose has a spread of ~0.9 kcal/mol), then **mouse** pass, then 5-seed robustness, then the mean of the two oracles.
- **Row order** (the first 20 are screened in the order submitted): tiers are never reordered; inside the best remaining tier a lineage may not exceed 40% of the first 20, and a design may move down at most one place to avoid two adjacent rows of one lineage.
- A design with no sequence is ranked but gets no row until its owner supplies it.

## What a miss means
On four experimentally solved TNF binders Boltz-2 recovers one and AF3 none, and a scrambled sequence scores like the misses. **A high score is informative; a low one is nearly not.** Nothing here is validated at the bench.
