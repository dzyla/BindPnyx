# ProteinBase benchmark

Scores experimentally labelled designs from [ProteinBase](https://proteinbase.com) with several
structure predictors and reports AUROC against the wet-lab binding label. Findings: `docs/REPORT.md`.

```bash
bench/run_benchmark.sh        # everything, ~2 h on one RTX 5090 (idle GPU!)
```

| file | purpose |
|---|---|
| `prepare_labels.py` | bulk CSV -> `labels.pkl` (binding = any experimental `binding` evaluation True) |
| `make_inputs.py` | per-target balanced sample (≤30 per class, 40–200 aa) -> `inputs/manifest.csv`; target constructs are UniProt ranges listed at the top of the file |
| `make_msas.py` | one ColabFold-server MSA per target -> `msa/<target>.a3m` |
| `score_boltz.py` | Boltz-2: target MSA, binder single-sequence, 3 recycles, 200 steps |
| `score_protenix.py ARM` | `ptx_v2` (protenix-v2), `ptx_v1` (v1.0.0), `ptx05_fast` (0.5-mini, 2 cycles, 5 steps) |
| `metrics.py` | ipSAE (Dunbrack, PAE cutoff 10) and AUROC; shared by every arm |
| `analyze.py` | joins scores with labels -> `out/auroc.csv`, `out/joined.csv` |
| `remp.py` | re-run ProteinMPNN on existing backbones under several settings |
| `manifests/{pdl1,mdm2}.json`, `targets/` | generation campaigns used in the report (PDB 3BIK, 1YCR) |
| `check1.py`, `cycle.py`, `mpnn_round.py`, `make_followup_figure.py` | follow-up checks: fast-vs-Boltz agreement, refold-redesign cycling, figure |
| `make_figures.py` | figures 1–7 and `results/stats.json` |
| `results/` | tracked outputs: AUROC tables, ProteinBase scores, rescored PXDesign designs, top-20 candidates |

Caveats that matter when reading the numbers: the negatives were mostly pre-filtered by other
predictors (so AUROC is pessimistic); method and label are confounded; n is 26–60 per target;
Protenix model names on this machine do not match their contents (CLAUDE.md).

## Data not included
The ProteinBase bulk export used here is not redistributed. Download `https://storage.proteinbase.com/proteinbase_all_data_28_01_2026.csv` into `bench/` (see proteinbase.com for terms), then run `prepare_labels.py`.
