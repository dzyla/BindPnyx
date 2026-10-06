# funnel: many backbones → fast screen → cycling → consensus

A binder-design workflow built on the same PXDesign diffusion model, ProteinMPNN and Boltz-2 as the main pipeline, restructured so
the *expensive* models only see designs that survived cheap ones. Evidence for each choice: `docs/FUNNEL_REPORT.md`.

```
 500 backbones ──► 4 MPNN sequences each ──► Protenix-0.5-mini "fast" fold of all 2000 ──► best sequence of the top 100 backbones
 (diffusion,        (soluble, T=0.1,           (0.4 s/design; ipSAE)                          │
  hotspot-conditioned) target fixed)                                                          ▼
                                                    3 rounds: MPNN on the PREDICTED complex, 8 children, fast-fold, elitist keep
                                                                                              │
                                  top 60 ──► Boltz-2 (3 seeds) + Protenix-v2 ──► consensus = min(ipSAE) ──► de-duplicate ──► shortlist
```

## Run it

```bash
./scripts/setup.sh                                        # once: .pxd/envs/{pxd,boltz}
python3 scripts/prepare_target.py ...                     # build the target shard (see main README)
$EDITOR funnel/targets/mytarget.json                      # target definition (below)
.pxd/envs/pxd/bin/python funnel/run_funnel.py --target mytarget --out out/funnel/mytarget --n-backbones 500
```

Resumable (each stage's outputs are reused; `--force` redoes). One GPU job at a time. Typical cost on one RTX 5090 for a 70–85-residue binder:
**~1.1 GPU-hours** (500 backbones). `--second-judge of3` swaps Protenix-v2 for OpenFold3 p2-155k as the second judge (see docs/RECOMMENDATIONS.md §19); the default stays `v2`. Outputs in `--out`: `final_cycled.csv` (the shortlist: sequence, Boltz-2/Protenix-v2 scores, hotspot contact,
`consensus_pass`), `consensus_cycled/boltz/seed1/**/*.cif` (predicted complexes), `timers.json`, and every intermediate table.

### Target definition (`funnel/targets/*.json`)

| key | meaning |
|---|---|
| `shard`, `provenance` | prebuilt shard and its `provenance.json` from `scripts/prepare_target.py` |
| `source_chain` | chain id in the *source* PDB (hotspot numbers refer to it) |
| `msa_dir` | folder with `non_pairing.a3m` whose query equals the shard sequence (checked) |
| `hotspots` | e.g. `["Y56","E58"]`: **PDB number with residue letter**. Converted to shard indices through the provenance residue map and *identity-checked*; a wrong letter or missing residue raises |
| `binder_length` | residues |

The identity check exists because the pipeline reads `hotspot` numbers on a `.pkl.gz` shard as the shard's own numbering (restarting at 1).

**Not for multi-chain targets.** A homo-oligomer target (binding site between protomers) needs the trimer-aware path in `phbind/` (`docs/HOMO_OLIGOMER_TARGETS.md`); `common.hotspot_contacts`/`site_occlusion` now raise on more than two chains.

## Judge any shortlist identically

```bash
.pxd/envs/pxd/bin/python funnel/judge.py --target mytarget --out out/judge/mytarget \
   --arm funnel=out/funnel/mytarget/final_cycled.csv --arm other=path/to/summary.csv
```
Fresh seeds (Boltz-2 101–103, Protenix-v2 101); **consensus pass** = Boltz-2 mean ipSAE ≥ 0.5 and interface PAE ≤ 2 Å and Protenix-v2 ipSAE ≥ 0.5.
A pass means "two structure predictors agree", not "binds"; see the report for what that is worth on each target.

## Files

| file | role |
|---|---|
| `run_funnel.py` | the five stages, resumable |
| `common.py` | target loading, Boltz-2 / Protenix wrappers, ipSAE, hotspot contacts, clustering |
| `mpnn_run.py` | ProteinMPNN worker (target fixed) |
| `judge.py`, `compare.py` | identical-protocol judging and cross-arm tables/figure |
| `run_baseline.sh`, `run_all.sh` | pipeline baseline arms with timing; the whole evaluation unattended |
| `make_manifest.py` | pipeline/diffusion input JSON from a target definition |
