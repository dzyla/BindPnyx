# Third-party code

This repository **vendors** three trees and patches them locally. They are not
dependencies fetched at install time: they are in the source tree, modified,
and their upstream licences continue to govern them.

Every vendored file keeps its original copyright and licence header. Nothing in
this project's own licence (AGPL-3.0, see `LICENSE`) restricts the rights those
licences already grant you in those files.

| tree | upstream | licence | why vendored |
| --- | --- | --- | --- |
| `pxdesign/` | PXDesign (ByteDance) | Apache-2.0, `Copyright 2025 ByteDance and/or its affiliates` | locally patched; diffusion and orchestration |
| `pxdbench/` | PXDesign benchmark suite (ByteDance), plus 4 files adapted from BindCraft | Apache-2.0; those 4 files MIT | locally patched; the evaluation stack |
| `colabdesign/` | ColabDesign | Apache-2.0, `Copyright 2021 DeepMind Technologies Limited`, `Copyright 2021 Levinthal Limited` | the build that accepts `weights=`, which PyPI's does not |

Per-tree divergence from upstream is recorded in `pxdesign/PATCHES.md`,
`pxdbench/PATCHES.md` and `colabdesign/PATCHES.md`, generated from the
pre-release git history. That history is not carried into this repository, so
those files are the only record.

## Individually attributed files

| file | origin | licence |
| --- | --- | --- |
| `pxdbench/tools/biopython_utils.py` | BindCraft `functions/biopython_utils.py`, commit pinned in the header | MIT |
| `pxdbench/metrics/interface.py` | `i_pDAE` ports BindCraft2 `bindcraft/filters.py:117-168` | MIT, attributed in-file |
| `pxdbench/metrics/secondary.py` | references a pinned BindCraft commit | MIT, attributed in-file |
| `pxdbench/metrics/_common_scorer.py` | independent implementation of the published ipSAE definition (Dunbrack lab, bioRxiv 2025.02.10.637595); verified against pinned golden values | this project (AGPL-3.0-or-later) |

## Resolved: `_common_scorer.py`

An earlier revision vendored this module from a private project whose licence could not be established. It has been replaced by an independent
implementation of the published ipSAE definition (see the module docstring); the pinned golden values in `tests/fixtures/ipsae_golden.json` are reproduced by it.

## External tools called as subprocesses (not bundled)

Boltz-2, Protenix, fastPISA (github.com/dzyla/fastPISA), and optionally LightDock (GPL-3.0; <https://lightdock.org>) are installed separately and keep their own licences.

## Not redistributed
- `pxdesign/pxd_server/TimesNewRoman.ttf` (a proprietary font used only by upstream PXDesign's web-demo plots) is omitted from the public export; those two demo modules need a font file placed there if you use them.
- Model weights (PXDesign, Protenix, Boltz-2) and all target structures / MSAs are not included; see `scripts/fetch_checkpoints.sh` and `funnel/fetch_target.py`.
- ProteinMPNN weights under `colabdesign/mpnn/` are redistributed with ColabDesign (MIT/Apache-2.0 per upstream).
- `bench/` contains only code and result summaries; the ProteinBase export (Adaptyv) used for it must be downloaded separately (URL in bench/README.md).
