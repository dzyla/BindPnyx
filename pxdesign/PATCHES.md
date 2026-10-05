# Local patches to the vendored `pxdesign` tree

Generated from the pre-release git history on 2026-10-01. That history is
not carried into the public repository, so this file is the only record of
how this tree diverges from upstream. Do not edit by hand.

```
2026-10-01  f81e346  feat(targets): stage consistency, and close the A1 padding bypass
2026-10-01  9d83282  refactor(numbering): extract the pure helpers so the map contracts are testable
2026-10-01  a3e92f7  feat(targets): per-target calibration as data, and a pose-targeting readout
2026-09-30  58bbce5  fix(hotspots): validate and record the numbering; verify it end to end
2026-09-30  5e814fd  feat: epitope policy, provisional-gate early-stop guard, stable design ids
2026-09-30  1b39adc  fix(boltz): parallelise SEEDS, not designs - batch composition changes scores
2026-09-30  afeb54a  fix(pipeline): make the AF2 and Protenix startup gates conditional
2026-09-30  f2bf13b  fix(boltz): batched prediction paths and pre_filter_boltz idempotency
2026-09-30  6141e64  fix(early-stop): resolve the success key from the mode and raise when absent
2026-09-30  8ee4b1a  feat(output): route the boltz mode through trim, figure and export
2026-09-30  29e6cd3  feat(ranking): add pre_filter_boltz and a boltz mode
2026-09-30  1548f1e  feat(boltz): route the binder task through the Boltz scorer
2026-09-30  104df32  fix(configs): make the empty seeds default constructible under protenix 0.5.5
2026-09-30  2fe1f4c  chore: vendor baseline of pxdesign, pxdbench and colabdesign
```

Files changed relative to the first commit that touched this tree:

```
 pxdesign/configs/configs_infer.py   |  22 ++-
 pxdesign/runner/helpers.py          | 299 +++++++++++++++++++++++++++++++-
 pxdesign/runner/pipeline.py         | 273 ++++++++++++++++++++++-------
 pxdesign/utils/infer.py             | 333 +++++++++--------------------------
 pxdesign/utils/pipeline.py          |  32 +++-
 pxdesign/utils/residue_numbering.py | 335 ++++++++++++++++++++++++++++++++++++
 6 files changed, 977 insertions(+), 317 deletions(-)
```
