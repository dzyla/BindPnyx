# Local patches to the vendored `pxdbench` tree

Generated from the pre-release git history on 2026-10-01. That history is
not carried into the public repository, so this file is the only record of
how this tree diverges from upstream. Do not edit by hand.

```
2026-10-01  e0bf2cd  feat(metrics): vendor the gate scorer so a clean machine can score (B1)
2026-10-01  f48266b  feat(boltz): carry the policy verdict through aggregation and rescoring
2026-10-01  f81e346  feat(targets): stage consistency, and close the A1 padding bypass
2026-10-01  93f1393  feat(targets): schema-2 resolution record, one artifact every stage reads
2026-10-01  b63b5f2  feat(targets): residue specs with identity assertions, and three epitope roles
2026-10-01  794955a  feat(targets): bounded source-to-work residue map with identity assertions
2026-10-01  9739e4c  feat(targets): adjustable gate presets with a shipped basic set
2026-10-01  a3e92f7  feat(targets): per-target calibration as data, and a pose-targeting readout
2026-10-01  4746d1d  fix: the app owns its environments instead of borrowing someone else's
2026-09-30  4b613f6  feat(epitope): patch-aware evaluation - "all of them" is often unsatisfiable
2026-09-30  5e814fd  feat: epitope policy, provisional-gate early-stop guard, stable design ids
2026-09-30  1c17a75  bench(boltz): worker scaling, time breakdown, and the kernels arm
2026-09-30  3446f28  bench(boltz): measure the GPU-utilisation change; reject dataloader workers
2026-09-30  6cdaa07  fix(mpnn): the soluable typo, hotspot target chains, and fix_interface
2026-09-30  1b39adc  fix(boltz): parallelise SEEDS, not designs - batch composition changes scores
2026-09-30  635441d  feat(boltz): concurrent fold workers to fill the GPU's idle time
2026-09-30  d2780f9  fix(boltz): resolve BoltzBackend directly, not through the global ptx factory
2026-09-30  f2bf13b  fix(boltz): batched prediction paths and pre_filter_boltz idempotency
2026-09-30  bce9610  feat(boltz): re-score the pooled shortlist in one common batch
2026-09-30  1548f1e  feat(boltz): route the binder task through the Boltz scorer
2026-09-30  3d54b54  feat(boltz): BoltzBackend with batched two-tier scoring
2026-09-30  76092ce  feat(boltz): batched per-seed subprocess runner
2026-09-30  5689085  feat(boltz): parse predictions into bz_* metrics with failure statuses
2026-09-30  58cde12  feat(boltz): add per-design YAML writer with per-chain target MSA
2026-09-30  3721890  fix(filters): fail closed on nonfinite values and unknown operators
2026-09-30  764e5d6  feat(metrics): add i_pDAE and ipSAE interface metrics
2026-09-30  2fe1f4c  chore: vendor baseline of pxdesign, pxdbench and colabdesign
```

Files changed relative to the first commit that touched this tree:

```
 pxdbench/metrics/_common_scorer.py                | 130 +++++
 pxdbench/metrics/epitope.py                       | 238 +++++++++
 pxdbench/metrics/interface.py                     | 207 ++++++++
 pxdbench/metrics/pose_targeting.py                | 253 +++++++++
 pxdbench/pxd_configs/eval.py                      |  66 +++
 pxdbench/targets/__init__.py                      |   1 +
 pxdbench/targets/consistency.py                   | 244 +++++++++
 pxdbench/targets/eligibility.py                   | 140 +++++
 pxdbench/targets/epitope_policy.py                | 188 +++++++
 pxdbench/targets/gate_presets/default.json        |  53 ++
 pxdbench/targets/gate_presets/none.json           |  53 ++
 pxdbench/targets/presets.py                       | 189 +++++++
 pxdbench/targets/record.py                        | 160 ++++++
 pxdbench/targets/registry.json                    |  45 ++
 pxdbench/targets/registry.py                      | 287 ++++++++++
 pxdbench/targets/residue_map.py                   | 408 +++++++++++++++
 pxdbench/targets/residue_spec.py                  | 132 +++++
 pxdbench/tasks/base.py                            | 103 +++-
 pxdbench/tasks/binder.py                          |   4 +
 pxdbench/toolenv.py                               | 148 ++++++
 pxdbench/tools/biopython_utils.py                 |  24 +-
 pxdbench/tools/boltz/__init__.py                  |   4 +
 pxdbench/tools/boltz/backend.py                   | 603 ++++++++++++++++++++++
 pxdbench/tools/boltz/final_batch.py               |  89 ++++
 pxdbench/tools/boltz/parse.py                     | 194 +++++++
 pxdbench/tools/boltz/result_schema.py             | 142 +++++
 pxdbench/tools/boltz/runner.py                    | 111 ++++
 pxdbench/tools/boltz/targets.py                   |  80 +++
 pxdbench/tools/boltz/yaml_writer.py               | 123 +++++
 pxdbench/tools/protmpnn/main_mpnn.py              |  51 +-
 pxdbench/tools/protmpnn/vanilla_mpnn_predictor.py |   9 +-
 31 files changed, 4462 insertions(+), 17 deletions(-)
```
