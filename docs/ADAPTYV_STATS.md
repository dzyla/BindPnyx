# Adaptyv x Anthropic 2026: counts for both challenges (aggregate, no sequences or identifiers)

**Wet-lab results: none yet.** No design from either challenge has been tested; every number below is a computational prediction. "Successful" therefore means "passed our computational gates", and nothing here shows binding. As of 2026-10-09 no experimental results have been received.

Source: the per-design tables (one row per design, with sequences) are kept privately. Only counts are published here. Both campaigns kept separate id spaces per generation pool, so a design is joined to a later stage only where an id or an exact sequence matches; rows that could not be joined are counted at the stage they reached, not guessed forward.

## Challenge 1: EGFR (human and mouse cross-reactive binder; 20 sequences submitted)
Cumulative: a design counted at a stage also counts at every earlier one. Rows are design records, so a sequence that appears in two tables (candidate pool and a screening pool) is counted in each.

| stage (reached at least) | n_designs |
|---|---|
| 1 generated, not screened further | 5110 |
| 2 rejected by the BindCraft2 filters | 3274 |
| 3 passed the BindCraft2 filters, not screened further | 1987 |
| 4 one-seed screen on both species | 1116 |
| 5 multi-seed confirmation, both species | 140 |
| 6 multi-seed evidence (5 or more seeds, both species) | 83 |
| 7 submitted (final panel of 20) | 20 |

Records by pool: candidate_pool 3994, evidence_master 83, full_pool_screen 311, short_pool_screen 722.
Pool-level numbers: of 2,208 BindCraft2 designs, 871 passed its own filters in the candidate table (the raw design table counts 880 passed). The short-binder screen scored 722 designs on both species at one seed: 241 had human ipSAE ≥ 0.5 and 161 had both species ≥ 0.5. The full-pool screen scored 309 designs (reference controls excluded): 137 human ≥ 0.5, 83 both species ≥ 0.5. The final 20: 15 binders and 5 pH-switch designs, 36–81 residues, 5–30 seeds each.

## Challenge 2: TNF-α homotrimer (pH-switchable, mouse cross-reactive; 40 sequences submitted)
Cumulative, as above.

| stage (reached at least) | n_designs |
|---|---|
| 0 generated, no score on record | 7902 |
| 1 prescreened, did not pass | 5505 |
| 3 novelty screened and passed prescreen | 339 |
| 4 AlphaFold 3 run | 117 |
| 5 passed both models (1 Boltz seed + AF3) | 82 |
| 8 in the evidence ledger (final candidate pool) | 66 |
| 9 submitted (v1 and/or v2 list) | 42 |

By generation lane (`lane` is a batch of work, not a machine type):

| lane | generated | prescreened (1 Boltz seed) | ipSAE≥0.5 | AlphaFold 3 run | passed both models | in evidence ledger | on a submitted list |
|---|---|---|---|---|---|---|---|
| cluster_wave_0 | 6 | 6 | 0 | 0 | 0 | 0 | 0 |
| cluster_wave_1 | 60 | 60 | 3 | 3 | 0 | 0 | 0 |
| cluster_wave_2 | 60 | 60 | 0 | 0 | 0 | 0 | 0 |
| cluster_wave_3 | 60 | 60 | 1 | 1 | 0 | 0 | 0 |
| cluster_wave_5 | 60 | 60 | 2 | 0 | 0 | 1 | 1 |
| cluster_wave_6 | 60 | 60 | 4 | 4 | 1 | 1 | 1 |
| local_lane | 5520 | 3679 | 237 | 55 | 30 | 18 | 8 |
| other_agents_or_later_arms | 36 | 0 | 0 | 0 | 0 | 36 | 23 |
| server_wave_1 | 720 | 692 | 27 | 16 | 9 | 10 | 7 |
| server_wave_2 | 600 | 432 | 15 | 0 | 0 | 0 | 0 |
| server_wave_3 | 720 | 360 | 14 | 0 | 0 | 0 | 0 |

Notes. (1) The one-seed prescreen scored 5469 designs; most score 0 because the binder does not contact the trimer in the prediction. (2) "Generated" counts designs with a record in a lane's design table; 2397 have no score on record because their scoring batches were not run, not because they failed. (3) Designs from other agents' large pools are not in this count; only the 36 that reached the ledger are, listed as `other_agents_or_later_arms`. (4) "Submitted" counts the union of two candidate lists; the first 20 of the novelty-adjusted list had portal structural-novelty levels of 4/4 for 5 rows, 3/4 for 13 and 2/4 for 2. (5) The robust-gate funnel on the first long-design pool: 1,231 scored, 53 passed one Boltz-2 seed (4.3%), 51 went to AlphaFold 3, 19 passed both models, 8 held at 3 seeds, 3 held on both species at 5 seeds.

## What this does not show
Binding, affinity, pH switching, expression, or mouse cross-reactivity were not measured. Two independent models agreeing is stronger than one, but both are trained on overlapping data. Thresholds were set on one target.
