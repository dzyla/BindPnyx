# How a campaign runs, and where a bad MSA is caught

Companion to CLAUDE.md §5, which describes the scientific stages. This one
describes the *guards*: what refuses to proceed, and where.

Everything marked NEW landed 2026-10-02 with the input-integrity branch
(`docs/superpowers/plans/2026-10-02-input-integrity.md`); the target
pre-flight marked NEW2 landed 2026-10-03 with `scripts/prepare_target.py`.

```
+- LAUNCH -------------------------------------------------------------+
| scripts/run_campaign.sh                                              |
|                                                                      |
|  resolve_python --> $PXD_PYTHON -> .pxd/envs/pxd -> conda roots  NEW |
|  resolve_boltz  --> $PXD_BOLTZ_BIN -> .pxd/envs/boltz -> PATH        |
|       (lazy: stops at the first match; was: probe all ~17)       NEW |
|                                                                      |
|  preflight:  package shadowing ....... --shadowing, 0.02s        NEW |
|              target/MSA pairs ....... preflight_target.py, 0.4s NEW2 |
|                 `- FIRST, before the ~80 s CUDA/protenix probes      |
|                 `- rebuilds orig_seqs from the SHARD, then runs      |
|                    the same target_chains_from_orig_seqs ->          |
|                    validate_target_chains that prepare_json runs     |
|                    (loaded by path: importing the package costs      |
|                     68 s of torch/protenix)                          |
|                 `- a .pdb structure_file is a SKIP, not a failure:   |
|                    its shard does not exist yet                      |
|              real CUDA matmul ........ never cuda.is_available()     |
|              checkpoints                                             |
|              boltz resolves                                          |
|              boltz warning string .... greps featurizerv2.py     NEW |
+------------------------------+---------------------------------------+
                               v
         diffusion  -->  N backbones, COORDINATES ONLY (binder = GLY)
                               v
         ProteinMPNN -->  every residue identity, one shot
                               v
+- BoltzBackend.prepare_json --- the chokepoint every fold passes -----+
|                                                                      |
|   orig_seqs --> target_chains_from_orig_seqs --> {id, seq, msa}      |
|                   (applies target_msa override, a3m_name, and a      |
|                    string-or-dict msa - so validate the RESULT,      |
|                    not the entity list)                              |
|                               v                                      |
|   validate_target_chains  <---- MANDATORY, no off switch         NEW |
|      - query parsed as boltz parses it (lowercase dropped, - kept)   |
|      - boltz's own tolerance (all MET<->UNK, or all input UNK)       |
|      - depth = UNIQUE count, not headers (boltz dedups)              |
|      x MsaMismatch --> STOP, before a single YAML is written         |
|                               v                                      |
|   write YAMLs  -->  msa_validation.json (seq + a3m sha256)       NEW |
+------------------------------+---------------------------------------+
                               v
+- _fold_batch, once per seed -----------------------------------------+
|   _batch_digest = composition + YAML bytes + MSA CONTENTS        NEW |
|                   + predictor identity + seed + params               |
|   clear boltz_results_*  -- UNCONDITIONAL                        NEW |
|        `- this is what invalidates boltz's INPUT cache.              |
|           boltz/main.py:724-742 skips records already in             |
|           processed/, regardless of --override, and processed/       |
|           lives under boltz_results_*. Without this clear, the       |
|           YAML refresh below is INERT and the previous sequence      |
|           is re-predicted and reported ok.                           |
|   stage YAMLs -- ALWAYS overwrite (was: only if absent)          NEW |
|                               v                                      |
|   run_seed --> keeps stdout AND stderr separately                NEW |
|                counts the dummy-MSA warning on the FULL stream   NEW |
|                (boltz prints it early; a fixed tail loses it)        |
|                               v                                      |
|   warning seen? --> every row bz_status = msa_discarded          NEW |
|   artifacts?    --> pdb + pae + plddt, three distinct messages   NEW |
+------------------------------+---------------------------------------+
                               v
     _mean_scores --> rows stamped with their seed                 NEW
                      (without it every design -> "unevaluable")
                               v
     aggregate_policy --> policy_status   [NOT wired to ranking yet]
                               v
     triage -> common batch -> final rank -> export
                               v
     summarise_outcome --> scored only if notna AND bz_status==ok  NEW
                           exit 1 if nothing scored                NEW
```

## The four layers against a discarded MSA

| # | where | catches |
|---|---|---|
| 0 | `scripts/prepare_target.py` | when the shard is BUILT - it refuses to write one whose sequence disagrees with the declared WT, or whose MSA would be discarded |
| 1 | campaign-script pre-flight | before screening/MPNN - `run_campaign.sh` included, since 2026-10-03 |
| 2 | `prepare_json` | before any YAML or GPU work |
| 3 | `run_seed` full-stream scan | boltz discarding one anyway, at fold time |
| 4 | `summarise_outcome` | a run that scored nothing reporting success |

Layer 2 fired on its first real campaign (2026-10-02), on EGFR chain B: one
residue at mature 516, which would have discarded 3,109 sequences and folded
the chain unconditioned. See `targets/egfr_ecd/PROVENANCE.md`.

## Where a target comes from

`scripts/prepare_target.py` is the only documented way to produce a shard. It
takes a source structure, an authoritative WT sequence, the chain/crop choices
and any structural edits, and writes the `.pkl.gz` **with** `provenance.json`:
source and WT hashes, the crop in a stated numbering, every edit named, a
source <-> prepared <-> shard <-> WT residue map, the result hashes, and the
MSA validation record. It refuses to emit a shard whose sequence disagrees with
the declared WT unless the reconciling edit is declared - and the only mutating
edit it offers deletes the side chain past CB, because relabelling a residue
leaves the donor's atoms wearing the new name.

The shard keeps none of this: it renumbers every chain from 1 and renames
chains A, B, ... The provenance record is the only place the source numbering
survives, which is what made the 2026-10-02 failure hard to diagnose.

## What this diagram deliberately shows as NOT done

- **A `.pdb`/`.cif` `structure_file` is still not pre-flighted.** The pipeline
  builds that shard at run time, so there are no entity sequences to compare
  yet; `preflight_target.py` says so and lets the run proceed to layer 2.
- **`policy_status` is computed but does not reach ranking.** Selection is on
  confidence alone, so a design that binds the wrong site can still be
  exported. See `docs/superpowers/plans/2026-10-02-policy-enforcement.md`.
