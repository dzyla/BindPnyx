"""Re-score a pooled shortlist in ONE Boltz batch.

Final selection pools rows from every shard, so it would otherwise compare
scores produced in different batches. A measured ~0.065 ipSAE shift on an
identical unmutated sequence exceeds typical per-round gains, so cross-batch
ranking is not a valid operation.

Anchor-offset adjustment is deliberately NOT offered: subtracting a per-batch
anchor delta assumes the shift is a constant additive offset, and the evidence
establishes that a shift exists, not its form. Re-scoring removes the artefact
by construction.

This runs AFTER a bounded triage pass, so the batch is the shortlist (bounded by
max_success_return), not the pooled rows of every shard.
"""
import json

import pandas as pd

ANCHOR_NAME = "__anchor__"
BZ_PREFIX = "bz_"  # retained for callers that import it

from pxdbench.tools.boltz.result_schema import propagated_columns  # noqa: E402


def _entries_from_manifest(manifest_path, data_list):
    """Manifest entries when the manifest is readable, else derived names."""
    try:
        with open(manifest_path) as handle:
            return json.load(handle)["designs"]
    except (OSError, ValueError, KeyError, TypeError):
        return [
            {
                "sample_name": f"{item['name']}_seq{item['seq_idx']}",
                "yaml": "",
                "binder_id": "C",
            }
            for item in data_list
        ]


def rescore_shortlist(
    shortlist_df,
    backend,
    dump_dir,
    orig_seqs,
    seeds=3,
    anchor_sequence="",
):
    """Fold every shortlist row again in one batch; return the updated frame."""
    if shortlist_df is None or len(shortlist_df) == 0:
        return pd.DataFrame(shortlist_df)

    data_list = [
        {
            "name": row["name"],
            "seq_idx": int(row["seq_idx"]),
            "sequence": row["sequence"],
        }
        for _, row in shortlist_df.iterrows()
    ]
    if anchor_sequence:
        data_list.append(
            {"name": ANCHOR_NAME, "seq_idx": 0, "sequence": anchor_sequence}
        )

    manifest_path = backend.prepare_json(
        "", data_list, dump_dir=dump_dir, orig_seqs=orig_seqs
    )

    # score_flat, NOT predict: the two-tier predict would re-select winners and
    # give three seeds only to those, leaving the rest of the shortlist at one
    # seed. Equal depth across the whole shortlist is the point of this batch.
    entries = _entries_from_manifest(manifest_path, data_list)
    scored_by_name = backend.score_flat(entries, dump_dir, seeds)

    out = shortlist_df.copy().reset_index(drop=True)
    # An EXPLICIT schema, not a `bz_` prefix test. The prefix whitelist
    # silently dropped every ep_* and policy column, so a design's epitope
    # verdict never survived the common final batch (R1).
    available = {k for row in scored_by_name.values() for k in row}
    bz_cols = propagated_columns(available)
    for col in bz_cols:
        # matched by sample_name, never positionally
        out[col] = [
            scored_by_name.get(f"{row['name']}_seq{int(row['seq_idx'])}", {}).get(col)
            for _, row in out.iterrows()
        ]
    return out
