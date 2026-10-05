import copy
import json
import os
import shutil
import subprocess
import sys
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
from protenix.utils.distributed import DIST_WRAPPER
from pxdbench.run import run_task
from pxdbench.tools.ptx.ptx_utils import populate_msa_with_cache
from pxdbench.utils import convert_cifs_to_pdbs

from pxdbench.targets.registry import Registry as _Registry
from pxdesign.utils.pipeline import (
    AF2_SUBDIR,
    ORIG_SUBDIR,
    PTX_SUBDIR,
    convert_strlist_col,
    save_af2_docked,
    save_design_cif,
    save_ptx_docked,
    trim_summary_df,
)

# ============================================================================
# Ranking and selection helpers (AF2-only / AF2+PTX)
# ============================================================================


def pre_filter_preview(
    df: pd.DataFrame,
    col_af2_ig: str = "af2_opt_success",
    col_af2_ig_easy: str = "af2_easy_success",
    col_af2_score: str = "unscaled_i_pAE",
    min_total_return: int = 5,
    max_success_return: int = 25,
    rmsd_col: str = "af2_complex_pred_design_rmsd",
    rmsd_threshold: float = 4.0,
) -> pd.DataFrame:
    """
    AF2-only pre-filtering and ranking.

    Rules:
      1) Identify SUCCESS rows where either `col_af2_ig` or `col_af2_ig_easy`
         is True, and also use RMSD < `rmsd_threshold` as a soft criterion.
      2) Assign discrete buckets based on (af2_ig, af2_ig_easy, RMSD).
         Lower bucket index means better.
      3) Within SUCCESS rows (bucket < 5), sort by (bucket ASC,
         `col_af2_score` ASC). Return up to `max_success_return`.
      4) If the number of successes is < `min_total_return`, pad with FAILED
         rows (bucket >= 5) sorted in the same way, until we reach
         `min_total_return` total rows.
      5) Add a 1-based `rank` column and return the combined DataFrame.
    """
    for c in [col_af2_ig, col_af2_ig_easy, col_af2_score, rmsd_col]:
        assert c in df.columns
        if c in [col_af2_ig, col_af2_ig_easy]:
            df[c] = df[c].astype(bool)

    # Aliases
    A = df[col_af2_ig]
    Ae = df[col_af2_ig_easy]
    R = df[rmsd_col] < rmsd_threshold

    # Bucket assignment for SUCCESS rows (1..5), FAILED = 6
    conds = [A & Ae, A, Ae & R, Ae, R]
    choices = [1, 2, 3, 4, 5]
    df["bucket"] = np.select(conds, choices, default=6).astype(int)
    df["pass_af2"] = df["bucket"].isin([1, 2, 3, 4])
    is_success = df["bucket"].lt(5)

    # Per-bucket tie score for successes (lower is better -> store as negative)
    df["tie_score"] = -df[col_af2_score].astype(float)

    # STEP 1: successes (up to max_success_return)
    success_df = df[is_success].copy()
    success_df = success_df.sort_values(
        by=["bucket", "tie_score"],
        ascending=[True, False],
        kind="mergesort",
        na_position="last",
    )
    success_pick = success_df.head(max_success_return)

    if len(success_pick) >= min_total_return:
        out = success_pick.reset_index(drop=True)
        out.insert(0, "rank", np.arange(1, len(out) + 1))
        return out

    # STEP 2: failures, sorted with the same (bucket, tie_score) rule
    need = min_total_return - len(success_pick)
    failed_df = df[~is_success].copy()
    failed_df = failed_df.sort_values(
        by=["bucket", "tie_score"],
        ascending=[True, False],
        kind="mergesort",
        na_position="last",
    )
    failed_pick = failed_df.head(need)

    combined = pd.concat([success_pick, failed_pick], axis=0).reset_index(drop=True)
    combined.insert(0, "rank", np.arange(1, len(combined) + 1))
    return combined


def resolve_ptx_columns(df: pd.DataFrame):
    """
    Automatically resolve PTX-related column names depending on which Protenix
    variant was used.

    Supported patterns (in priority order):
      1) ptx_success, ptx_basic_success, ptx_iptm
      2) ptx_mini_success, ptx_mini_basic_success, ptx_mini_iptm

    Returns
    -------
    (col_ptx, col_ptx_basic, col_ptx_score)
        Column names for PTX success, basic success, and score.
    (None, None, None)
        If none of the supported patterns are found in the DataFrame.
    """
    candidates = [
        ("ptx_success", "ptx_basic_success", "ptx_iptm"),
        ("ptx_mini_success", "ptx_mini_basic_success", "ptx_mini_iptm"),
    ]
    for c_succ, c_basic, c_score in candidates:
        if c_succ in df.columns and c_basic in df.columns and c_score in df.columns:
            return c_succ, c_basic, c_score
    return None, None, None


def pre_filter_extended(
    df: pd.DataFrame,
    # success mark columns (booleans)
    col_af2_ig: str = "af2_opt_success",
    col_af2_ig_easy: str = "af2_easy_success",
    col_ptx: str | None = None,
    col_ptx_basic: str | None = None,
    # score columns
    col_ptx_score: str | None = None,  # higher is better
    col_af2_score: str = "unscaled_i_pAE",  # lower is better
    # output caps
    min_total_return: int = 20,
    max_success_return: int = 100,
    w_af2: float = 0.5,
    w_ptx: float = 0.5,
):
    """
    Joint AF2 + Protenix ranking.

    PTX-related column names are auto-resolved from the DataFrame if not
    explicitly provided:
      - If df has ptx_success/ptx_basic_success/ptx_iptm, use them.
      - Else if df has ptx_mini_success/ptx_mini_basic_success/ptx_mini_iptm, use them.
      - Otherwise, raise an error (caller should ensure PTX columns exist).

    Rules:
      1) Identify SUCCESS rows where any of the four success flags are True:
         af2_opt, af2_easy, ptx, ptx_basic.
      2) Assign buckets 1..8 (success) or 9 (failure) based on combinations
         of AF2 and PTX signals. Lower bucket index means better.
      3) For success rows, define a tie score:
           - For PTX-like buckets: higher PTX score is better
           - For AF2-like buckets: lower AF2 score is better (stored as negative)
      4) Sort successes by (bucket ASC, tie_score DESC) and return up to
         `max_success_return`.
      5) If successes < `min_total_return`, pad with failure rows (bucket == 9)
         using a weighted rank fusion:
           - r_af2: rank by AF2 score ASC (lower is better)
           - r_ptx: rank by PTX score DESC (higher is better)
           - rank_w = w_af2 * r_af2 + w_ptx * r_ptx
         Then sort failures by (rank_w ASC, r_ptx ASC, r_af2 ASC) and take as many
         as needed. Finally add a 1-based `rank` column.
    """
    df = df.copy()

    # Resolve PTX column names if not explicitly specified
    if col_ptx is None or col_ptx_basic is None or col_ptx_score is None:
        auto_ptx, auto_basic, auto_score = resolve_ptx_columns(df)
        if auto_ptx is None:
            raise ValueError(
                "pre_filter_extended: could not find PTX columns in df. "
                "Expected (ptx_* or ptx_mini_*)."
            )
        col_ptx = col_ptx or auto_ptx
        col_ptx_basic = col_ptx_basic or auto_basic
        col_ptx_score = col_ptx_score or auto_score

    # Ensure success columns exist (default False) and are boolean
    for c in [
        col_af2_ig,
        col_af2_ig_easy,
        col_ptx,
        col_ptx_basic,
        col_ptx_score,
        col_af2_score,
    ]:
        assert c in df.columns, c
        if c in [col_af2_ig, col_af2_ig_easy, col_ptx, col_ptx_basic]:
            df[c] = df[c].astype(bool)

    # Aliases
    A = df[col_af2_ig]
    Ae = df[col_af2_ig_easy]
    P = df[col_ptx]
    Pb = df[col_ptx_basic]

    # Bucket assignment for SUCCESS rows (1..8), FAILED = 9
    conds = [
        A & P,
        A & Pb,
        Ae & P,
        Ae & Pb,
        P & ~A & ~Ae,
        Pb & ~A & ~Ae,
        A & ~P & ~Pb,
        Ae & ~P & ~Pb,
    ]
    choices = [1, 2, 3, 4, 5, 6, 7, 8]
    df["bucket"] = np.select(conds, choices, default=9).astype(int)
    df["pass_af2"] = df["bucket"].isin([1, 2, 3, 4, 7, 8])
    df["pass_ptx"] = df["bucket"].isin([1, 2, 3, 4, 5, 6])

    is_success = df["bucket"].lt(9)

    # Per-bucket tie score for successes (unified "higher is better")
    ptx_like = df["bucket"].isin([1, 2, 3, 4, 5, 6, 9])  # include 9 harmlessly
    af2_like = df["bucket"].isin([7, 8])
    df["tie_score"] = np.where(
        ptx_like,
        df[col_ptx_score].astype(float),
        np.where(af2_like, -df[col_af2_score].astype(float), np.nan),
    )

    # STEP 1: successes (up to max_success_return)
    success_df = df[is_success].copy()
    success_df = success_df.sort_values(
        by=["bucket", "tie_score"],
        ascending=[True, False],
        kind="mergesort",
        na_position="last",
    )
    success_pick = success_df.head(max_success_return)

    if len(success_pick) >= min_total_return:
        out = success_pick.reset_index(drop=True)
        out.insert(0, "rank", np.arange(1, len(out) + 1))
        return out

    # STEP 2: failures — weighted rank fusion
    need = min_total_return - len(success_pick)
    failed_df = df[~is_success].copy()  # bucket == 9

    def ordinal_rank(series: pd.Series, *, ascending: bool) -> pd.Series:
        """
        Ordinal rank consistent with sort order.
        NaNs are always placed last.
        """
        order = series.sort_values(
            ascending=ascending, na_position="last", kind="mergesort"
        )
        ranks = pd.Series(np.arange(1, len(order) + 1, dtype=float), index=order.index)
        return ranks.reindex(series.index)

    # r_af2: lower AF2 score is better (ascending=True)
    failed_df["_r_af2"] = ordinal_rank(
        failed_df[col_af2_score].astype(float), ascending=True
    )
    # r_ptx: higher PTX score is better (ascending=False)
    failed_df["_r_ptx"] = ordinal_rank(
        failed_df[col_ptx_score].astype(float), ascending=False
    )

    # Weighted rank (smaller is better)
    failed_df["_rank_w"] = w_af2 * failed_df["_r_af2"] + w_ptx * failed_df["_r_ptx"]

    failed_sorted = failed_df.sort_values(
        by=["_rank_w", "_r_ptx", "_r_af2"],
        ascending=[True, True, True],
        kind="mergesort",
        na_position="last",
    )
    failed_pick = failed_sorted.head(need)

    combined = pd.concat([success_pick, failed_pick], axis=0).reset_index(drop=True)
    combined.insert(0, "rank", np.arange(1, len(combined) + 1))
    return combined


def top_pct_mean(s: pd.Series, pct: float = 0.10, *, ascending: bool = False) -> float:
    """
    Compute the mean of the top `pct` fraction of values in a Series.

    Parameters
    ----------
    s : pd.Series
        Input score series.
    pct : float
        Fraction in (0, 1]. For example, 0.1 means top 10%.
    ascending : bool
        If False (default), higher values are better (use nlargest).
        If True, lower values are better (use nsmallest).

    Returns
    -------
    float
        Mean of the selected subset, or NaN if the series is empty.
    """
    s = s.dropna()
    if s.empty:
        return np.nan

    pct = float(pct)
    if not 0 < pct <= 1:
        raise ValueError("pct must be in (0, 1].")

    k = max(1, int(np.ceil(pct * len(s))))
    return (s.nsmallest(k) if ascending else s.nlargest(k)).mean()


def infer_mode_from_df(df: pd.DataFrame) -> str:
    """
    Decide which ranking rule to use based on whether Protenix filter actually
    produced results.

    Logic:
      - If Boltz columns are present, use 'boltz' ranking. This check runs
        FIRST and on its own columns: a stray ptx_* column would otherwise
        route selection to the mixed AF2+Protenix rule.
      - Else if PTX-related columns are present (ptx_* or ptx_mini_*),
        use 'extended' ranking.
      - Otherwise, use 'preview' (AF2-only ranking).
    """
    if "bz_ipsae" in df.columns or "bz_ipdae" in df.columns:
        return "boltz"
    col_ptx, col_ptx_basic, col_ptx_score = resolve_ptx_columns(df)
    has_ptx = (
        col_ptx is not None and col_ptx_basic is not None and col_ptx_score is not None
    )
    return "extended" if has_ptx else "preview"


# Resolved from the target registry; see pxdbench/targets/registry.py for why
# this is not a literal.
BOLTZ_GATE_NAME = _Registry.load_default().default_gate_id()
BOLTZ_GATE_COL = f"{BOLTZ_GATE_NAME}_success"


def pre_filter_boltz(
    df: pd.DataFrame,
    min_total_return: int = 20,
    max_success_return: int = 100,
    per_backbone_cap: int = 1,
    ranking_key: str = "bz_ipsae",
    gate_col: str = BOLTZ_GATE_COL,
    within_batch: bool = False,
    *,
    mode: str | None = None,
    context=None,
):
    """Boltz-only ranking: two buckets, ordered by `ranking_key` descending.

    bucket 1 = gate pass, bucket 9 = everything else. The per-backbone cap is
    applied to the COMBINED selection, successes and padding together: a sparse
    gate is exactly when padding dominates, so capping successes alone would let
    the padded tail fill with siblings of one backbone.

    Raises if rows span more than one `bz_final_batch_id`. Co-fold scores are
    not comparable across batches (a measured ~0.065 ipSAE shift on an identical
    sequence), so ranking a mixed pool is not a valid operation.

    `within_batch=True` is the triage mode. It does NOT relax the rule: instead
    of sorting across batches it allocates places inside each batch and returns
    the union, so no design is ever excluded by a cross-batch comparison. The
    union is then re-scored in one common batch and ranked properly.

    `mode`/`context` are keyword-only and default to the LEGACY path, byte for
    byte: an existing caller that passes neither behaves exactly as before,
    including the padding that legacy campaigns depend on.

    In `policy_v1` there is still ONE ranking algorithm (targeting-contracts
    section 5). Mandatory eligibility is filtered FIRST, then the existing
    quality buckets, ordering, per-backbone cap and padding run over the
    eligible subset only. That is what closes review A1: padding previously
    drew from bucket 9 - the gate failures - so a design whose epitope policy
    had FAILED was still selected and still exported, correctly labelled and
    entirely shipped.
    """
    from pxdbench.targets.eligibility import (
        POLICY_V1,
        SelectionContext,
        resolve_eligibility,
    )

    resolved_mode = mode or (getattr(context, "mode", None) or "legacy")
    if resolved_mode == POLICY_V1 and mode is None:
        raise ValueError(
            "pre_filter_boltz received a policy_v1 context but no mode= "
            "argument, so it would have taken the legacy path. A "
            "manifest-declared policy task must not be ranked as legacy: pass "
            "mode='policy_v1' explicitly."
        )
    if resolved_mode == POLICY_V1:
        selection_context = context or SelectionContext(mode=POLICY_V1)
        df = resolve_eligibility(df, selection_context)
        df = df[df["eligible"]].copy()
    # Columns this function OWNS are dropped first: it is called twice in the
    # boltz dispatch (bounded triage, then the final rank after the common
    # batch), so the second call receives a frame that already carries them.
    df = df.copy().drop(
        columns=["rank", "bucket", "pass_boltz", "_score"], errors="ignore"
    )
    if ranking_key not in df.columns:
        raise ValueError(
            f"pre_filter_boltz: ranking key {ranking_key!r} not in df columns "
            f"{list(df.columns)}"
        )
    if "bz_final_batch_id" in df.columns:
        batches = sorted(set(df["bz_final_batch_id"].dropna().unique().tolist()))
        if len(batches) > 1 and not within_batch:
            raise ValueError(
                f"pre_filter_boltz: rows span {len(batches)} bz_final_batch_id "
                f"values {batches}. Co-fold scores are not comparable across "
                f"batches; re-score the pooled shortlist in one batch. For the "
                f"triage pass that CHOOSES that shortlist, pass "
                f"within_batch=True, which allocates places per batch instead "
                f"of sorting across them."
            )
        if len(batches) > 1 and within_batch:
            # Allocate places INSIDE each batch and take the union. A pooled
            # sort would exclude designs on a cross-batch comparison, and an
            # excluded design is never recoverable.
            per_batch = max(1, int(np.ceil(max_success_return / len(batches))))
            parts = []
            for batch in batches:
                part = pre_filter_boltz(
                    df[df["bz_final_batch_id"] == batch],
                    min_total_return=min(min_total_return, per_batch),
                    max_success_return=per_batch,
                    per_backbone_cap=per_backbone_cap,
                    ranking_key=ranking_key,
                    gate_col=gate_col,
                )
                parts.append(part.drop(columns=["rank"], errors="ignore"))
            union = pd.concat(parts, axis=0).reset_index(drop=True)
            union.insert(0, "rank", np.arange(1, len(union) + 1))
            return union

    gate = (
        pd.to_numeric(df[gate_col], errors="coerce").fillna(0) > 0
        if gate_col in df.columns
        else pd.Series(False, index=df.index)
    )
    df["bucket"] = np.where(gate, 1, 9).astype(int)
    df["pass_boltz"] = df["bucket"].eq(1)
    df["_score"] = pd.to_numeric(df[ranking_key], errors="coerce")

    def _ordered(frame):
        return frame.sort_values(
            by=["bucket", "_score"],
            ascending=[True, False],
            kind="mergesort",
            na_position="last",
        )

    ordered = _ordered(df)
    if per_backbone_cap and per_backbone_cap > 0 and "name" in ordered.columns:
        ordered = _ordered(ordered.groupby("name", sort=False).head(per_backbone_cap))

    successes = ordered[ordered["bucket"].eq(1)].head(max_success_return)
    if len(successes) >= min_total_return:
        out = successes
    else:
        need = min_total_return - len(successes)
        out = pd.concat(
            [successes, ordered[ordered["bucket"].eq(9)].head(need)], axis=0
        )

    out = out.drop(columns=["_score"]).reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out


# ============================================================================
# Writing designs, summaries, and difficulty figures
# ============================================================================


def process_preview_results(
    selected_df: pd.DataFrame,
    base_dir: str,
    output_dir: str,
):
    """
    Serialize designs for AF2-only (preview-like) runs.

    For each selected row:
      - Write the original design CIF file under ORIG_SUBDIR.
      - If `pass_af2` is True, also write AF2-docked CIF under AF2_SUBDIR.
      - For each row, record which structure is chosen ("orig" or "af2")
        and its relative path in `chosen_struct_path`.

    Then:
      - Save a full summary CSV under <output_dir>/<task_name>/summary.csv
    """
    if selected_df.empty:
        return
    task_name = selected_df["task_name"].iloc[0]
    if DIST_WRAPPER.world_size > 1:
        task_name = task_name[: -len("_chunk*")]
    df = selected_df.copy()

    # Save original design CIFs
    for _, task in df.iterrows():
        save_design_cif(
            task,
            base_dir,
            os.path.join(output_dir, task_name),
            output_subdir=ORIG_SUBDIR,
            rank_col="rank",
        )

    rows = []
    for _, task in df.iterrows():
        chosen_type, chosen_path = "orig", ORIG_SUBDIR
        if task.get("pass_af2", False):
            save_af2_docked(
                task,
                base_dir,
                os.path.join(output_dir, task_name),
                output_subdir=AF2_SUBDIR,
            )
            chosen_type, chosen_path = "af2", AF2_SUBDIR

        chosen_path = os.path.join(chosen_path, f"rank_{task['rank']}.cif")
        assert os.path.exists(os.path.join(output_dir, task_name, chosen_path))
        r = dict(task)
        r.update({"chosen_struct_type": chosen_type, "chosen_struct_path": chosen_path})
        rows.append(r)

    saved_df = pd.DataFrame(rows)
    trimmed_df = trim_summary_df(saved_df)
    trimmed_df.task_name = task_name

    summary = os.path.join(output_dir, task_name, "summary.csv")
    os.makedirs(os.path.dirname(summary), exist_ok=True)
    trimmed_df.to_csv(summary, index=False)
    print(f"Saved summary CSV to {summary}")

    return os.path.join(output_dir, task_name)


BOLTZ_SUBDIR = "boltz_docked"


def _export_structure(task, sample_name, struct_dir):
    """COPY the prediction into the deliverable; report honestly if absent.

    The source comes from `bz_struct_path`, which the backend persists on each
    row at scoring time. Searching a base directory does not work: task
    predictions land under the task's own out_dir, the final common batch writes
    under `final_batch/`, and each pass has its own tag subdirectory - the
    export stage cannot reconstruct which combination produced a given row.

    A selected row may have no retained prediction: failure-padded rows are
    non-winners by definition, and keep_structures="winners" removes exactly
    their structures. That case gets chosen_struct_type="none" with a reason,
    never a path that does not resolve.
    """
    rank = int(task.get("rank", 0))
    source = task.get("bz_struct_path")
    resolvable = (
        isinstance(source, str) and source and os.path.isfile(source)
    )
    if resolvable:
        dst = os.path.join(struct_dir, f"rank_{rank}_{sample_name}.pdb")
        shutil.copyfile(source, dst)
        return {
            "chosen_struct_type": "boltz",
            "chosen_struct_path": dst,
            "chosen_struct_source": source,
            "chosen_struct_note": "",
        }
    if not isinstance(source, str) or not source:
        note = (
            "no bz_struct_path on the row; the design was not gate-confirmed "
            "and keep_structures removed its prediction"
            if not task.get("pass_boltz", False)
            else "gate-confirmed but no bz_struct_path was recorded"
        )
    else:
        note = f"bz_struct_path does not resolve: {source}"
    print(f"[WARN] {sample_name}: {note}")
    return {
        "chosen_struct_type": "none",
        "chosen_struct_path": "",
        "chosen_struct_source": "",
        "chosen_struct_note": note,
    }


def process_boltz_results(
    selected_df: pd.DataFrame,
    base_dir: str,
    output_dir: str,
):
    """Serialize designs for Boltz runs.

    No rerun_ptx and no AF2 branch: the 3-seed gate pass already IS the
    confirmation re-run, so the extended path's rationale does not carry over.

    The exported structure is whatever `bz_struct_path` points at, which the
    backend sets to the lowest-seed prediction of the pass that produced the
    row. Exporting the best-scoring seed would bias the structure toward the
    score it illustrates.
    """
    if selected_df is None or len(selected_df) == 0:
        return None
    task_name = selected_df["task_name"].iloc[0]
    if DIST_WRAPPER.world_size > 1:
        task_name = task_name[: -len("_chunk*")]

    task_out = os.path.join(output_dir, task_name)
    os.makedirs(task_out, exist_ok=True)
    struct_dir = os.path.join(task_out, BOLTZ_SUBDIR)
    os.makedirs(struct_dir, exist_ok=True)

    rows = []
    for _, task in selected_df.iterrows():
        sample_name = f"{task.get('name')}_seq{task.get('seq_idx')}"
        record = task.to_dict()
        record.update(_export_structure(task, sample_name, struct_dir))
        rows.append(record)

    trimmed = trim_summary_df(pd.DataFrame(rows))
    summary = os.path.join(task_out, "summary.csv")
    trimmed.to_csv(summary, index=False)
    print(f"Saved summary CSV to {summary}")
    return task_out


def process_extended_results(
    selected_df: pd.DataFrame,
    base_dir: str,
    output_dir: str,
    orig_seqs: list,
    configs_eval,
):
    """
    Serialize designs for extended runs (AF2 + PTX).

    For each selected row:
      - Save original design CIF under ORIG_SUBDIR.
      - Optionally save AF2-docked CIF under AF2_SUBDIR (if pass_af2).
      - Optionally save PTX-docked CIF under PTX_SUBDIR (if pass_ptx).
      - The chosen structure for each row is selected with priority:
          PTX > AF2 > original (ORIG).
        The chosen type and path are stored in columns:
          chosen_struct_type, chosen_struct_path.

    If there are any rows with pass_ptx == True, re-run Protenix on the
    subset via `rerun_ptx` before final serialization.

    Finally:
      - Save full summary CSV under <output_dir>/<task_name>/summary.csv
    """
    if selected_df.empty:
        return
    task_name = selected_df["task_name"].iloc[0]
    if DIST_WRAPPER.world_size > 1:
        task_name = task_name[: -len("_chunk*")]

    # Optional PTX rerun for pass_ptx rows
    if "pass_ptx" in selected_df.columns and selected_df["pass_ptx"].any():
        _ = rerun_ptx(
            selected_df[selected_df["pass_ptx"]],
            base_dir,
            task_name,
            orig_seqs,
            copy.deepcopy(configs_eval),
        )

    df = selected_df.copy()
    for _, task in df.iterrows():
        save_design_cif(
            task,
            base_dir,
            os.path.join(output_dir, task_name),
            output_subdir=ORIG_SUBDIR,
            rank_col="rank",
        )

    rows = []
    for _, task in df.iterrows():
        chosen_type, chosen_path = "orig", ORIG_SUBDIR

        if task.get("pass_af2", False):
            save_af2_docked(
                task,
                base_dir,
                os.path.join(output_dir, task_name),
                output_subdir=AF2_SUBDIR,
            )
            chosen_type, chosen_path = "af2", AF2_SUBDIR
        if task.get("pass_ptx", False):
            save_ptx_docked(
                task,
                os.path.join(base_dir, "ptx_final_outputs", task_name),
                os.path.join(output_dir, task_name),
                output_subdir=PTX_SUBDIR,
                is_large=True,
            )
            chosen_type, chosen_path = "ptx", PTX_SUBDIR

        chosen_path = os.path.join(chosen_path, f"rank_{task['rank']}.cif")
        assert os.path.exists(os.path.join(output_dir, task_name, chosen_path))
        r = dict(task)
        r.update({"chosen_struct_type": chosen_type, "chosen_struct_path": chosen_path})
        rows.append(r)

    saved_df = pd.DataFrame(rows)
    trimmed_df = trim_summary_df(saved_df)
    trimmed_df.task_name = task_name

    summary = os.path.join(output_dir, task_name, "summary.csv")
    os.makedirs(os.path.dirname(summary), exist_ok=True)
    trimmed_df.to_csv(summary, index=False)
    print(f"Saved summary CSV to {summary}")

    return os.path.join(output_dir, task_name)


def save_difficulty_fig(df, mode, save_dir, use_template=False):
    """
    Render a difficulty figure for the server UI.

    Parameters
    ----------
    df : pd.DataFrame
        Merged summary DataFrame with AF2/PTX metrics.
    mode : str
        "preview"  -> AF2-only difficulty figure
        "extended" -> AF2 + PTX difficulty figure
    save_dir : str
        Directory where the figure will be saved.
    use_template : bool
        Whether the PTX filter used template-based model (for extended mode).
    """
    current_path = os.path.abspath(__file__)
    runner_dir = os.path.dirname(current_path)
    project_root = os.path.dirname(runner_dir)

    if mode == "boltz":
        # The server UI figures are AF2/Protenix-specific; there is no Boltz
        # equivalent, and the preview branch below would KeyError on
        # unscaled_i_pAE / af2_easy_success, which the Boltz path does not emit.
        print("[INFO] save_difficulty_fig: no figure for boltz mode, skipping.")
        return None
    if mode == "extended":
        af2_sr = df["af2_easy_success"].mean() * 100
        _, col_ptx_basic, _ = resolve_ptx_columns(df)
        if col_ptx_basic is None:
            raise ValueError(
                "save_difficulty_fig: extended mode requires PTX columns in df."
            )
        ptx_sr = df[col_ptx_basic].mean() * 100
        length = len(df["sequence"][0])
        cmd = [
            sys.executable,
            f"{project_root}/pxd_server/server_extended_mode.py",
            "--out_af2_sr",
            f"{af2_sr}",
            "--out_ptx_sr",
            f"{ptx_sr}",
            "--use_temp_model",
            str(use_template),
            "--length",
            f"{length}",
            "--save_path",
            os.path.join(save_dir, "server_extended_mode.png"),
        ]
        subprocess.run(cmd, check=True)
    else:
        af2_ipae = top_pct_mean(df["unscaled_i_pAE"], ascending=True)
        af2_sr = df["af2_easy_success"].mean() * 100
        length = len(df["sequence"][0])
        cmd = [
            "python3",
            f"{project_root}/pxd_server/server_preview_mode.py",
            "--out_af2_ipae",
            f"{af2_ipae}",
            "--out_af2_sr",
            f"{af2_sr}",
            "--length",
            f"{length}",
            "--save_path",
            os.path.join(save_dir, "server_preview_mode.png"),
        ]
        subprocess.run(cmd, check=True)


def save_top_designs(p, configs, orig_seqs, use_template=False):
    """
    Collect all sample-level CSVs, decide which ranking mode to use, and
    write out top designs + difficulty figure.

    Steps:
      1) Collect `sample_level_output.csv` from all runs via `collect_sample_csvs`.
      2) Normalize string-list columns and drop all-null columns.
      3) Infer whether to use AF2-only ("preview") or AF2+PTX ("extended")
         ranking based on presence of PTX columns.
      4) Apply the corresponding pre-filter and selection:
           - preview  -> `pre_filter_preview`
           - extended -> `pre_filter_extended`
      5) Serialize designs via `process_preview_results` or `process_extended_results`.
      6) Call `save_difficulty_fig` with the actual mode used.
    """
    output_dir = os.path.join(configs.dump_dir, "design_outputs")
    os.makedirs(output_dir, exist_ok=True)

    # Collect per-sample CSVs for final selection/serialization
    merged_df = collect_sample_csvs(configs.dump_dir)
    merged_df = convert_strlist_col(merged_df)
    merged_df = merged_df.dropna(axis=1, how="all")
    if merged_df.empty:
        return

    merged_df.to_csv(os.path.join(configs.dump_dir, "all_summary.csv"), index=False)

    # Decide ranking mode based on presence of PTX columns
    mode_used = infer_mode_from_df(merged_df)

    if mode_used == "preview":
        pre_filtered = pre_filter_preview(
            merged_df,
            min_total_return=p["min_total_return"],
            max_success_return=p["max_success_return"],
            rmsd_col="af2_complex_pred_design_rmsd",
        )
        pre_filtered.to_csv(
            os.path.join(configs.dump_dir, "filtered_summary.csv"), index=False
        )

        save_dir = process_preview_results(pre_filtered, configs.dump_dir, output_dir)

    elif mode_used == "extended":
        pre_filtered = pre_filter_extended(
            merged_df,
            min_total_return=p["min_total_return"],
            max_success_return=p["max_success_return"],
            w_af2=p["extended_w_af2"],
            w_ptx=p["extended_w_ptx"],
        )
        pre_filtered.to_csv(
            os.path.join(configs.dump_dir, "filtered_summary.csv"), index=False
        )

        save_dir = process_extended_results(
            pre_filtered,
            configs.dump_dir,
            output_dir,
            orig_seqs=list(orig_seqs.values())[0],
            configs_eval=configs.eval,
        )
    elif mode_used == "boltz":
        # Local imports: these pull in the pxdbench task stack, which this
        # module does not otherwise need.
        from pxdbench.tasks.base import BaseTask
        from pxdbench.tools.boltz.backend import BoltzBackend
        from pxdbench.tools.boltz.final_batch import rescore_shortlist

        boltz_cfg = configs.eval.binder.tools.boltz
        ranking_key = boltz_cfg.get("ranking_key", "bz_ipsae")
        cap = p.get("per_backbone_cap", 1)

        # Resolve the selection mode from the SAME config the scorer read, so
        # the two cannot disagree. Previously nothing passed mode= here, so a
        # campaign that declared an epitope policy was still ranked as legacy
        # and the policy changed nothing about what shipped.
        from pxdbench.targets.eligibility import (
            POLICY_V1,
            SelectionContext,
            assert_policy_config_consistent,
        )

        _policy = dict(boltz_cfg).get("epitope_policy", {}) or {}
        _enforced = bool(_policy.get("required") or _policy.get("forbidden"))
        assert_policy_config_consistent(_policy, _enforced)
        sel_mode = POLICY_V1 if _enforced else None
        sel_ctx = (
            SelectionContext(mode=POLICY_V1, enforced=True)
            if _enforced
            else None
        )
        if _enforced:
            print(
                "    selection: policy_v1 - mandatory eligibility is applied "
                "before quality buckets, and padding may only draw on "
                "eligible rows"
            )

        # 1) BOUNDED triage. within_batch allocates places INSIDE each batch
        #    and unions them: a pooled sort would exclude designs on a
        #    cross-batch comparison, and an exclusion is never recoverable.
        triage = pre_filter_boltz(
            merged_df,
            min_total_return=p["min_total_return"],
            max_success_return=p["max_success_return"],
            per_backbone_cap=cap,
            ranking_key=ranking_key,
            within_batch=True,
            mode=sel_mode,
            context=sel_ctx,
        )

        # 2) One common batch over the shortlist, every row at the same depth.
        #    MANDATORY, with no config switch: an off switch would silently
        #    restore cross-batch ranking. Runs even for a SINGLE shard, because
        #    within one shard the rank-pass rows carry 1 seed and the gate-pass
        #    rows 3, under different per-pass batch ids.
        triage = rescore_shortlist(
            triage,
            BoltzBackend(cfg=dict(boltz_cfg), device="cuda:0"),
            os.path.join(configs.dump_dir, "final_batch"),
            list(orig_seqs.values())[0],
            seeds=int(boltz_cfg.get("seeds_gate", 3)),
            anchor_sequence=boltz_cfg.get("anchor_sequence", ""),
        )
        # the gate columns were replaced with common-batch values, so the
        # per-shard _success column no longer describes the frame
        triage = BaseTask.compute_success_rate(
            {BOLTZ_GATE_NAME: dict(configs.eval.binder.filters[BOLTZ_GATE_NAME])},
            triage,
        )

        # 3) Final rank on one batch; the batch assertion now holds.
        pre_filtered = pre_filter_boltz(
            triage,
            min_total_return=p["min_total_return"],
            max_success_return=p["max_success_return"],
            per_backbone_cap=cap,
            ranking_key=ranking_key,
            mode=sel_mode,
            context=sel_ctx,
        )
        pre_filtered.to_csv(
            os.path.join(configs.dump_dir, "filtered_summary.csv"), index=False
        )
        save_dir = process_boltz_results(pre_filtered, configs.dump_dir, output_dir)
    else:
        raise ValueError(f"Unknown mode_used={mode_used!r}")

    # Save difficulty figure based on the mode actually used
    save_difficulty_fig(merged_df, mode_used, save_dir, use_template)

    # Clean unnecessary files
    cleanup_outputs(configs.dump_dir)


# ============================================================================
# Protenix target template helpers
# ============================================================================


def create_target_ptx_json(cif_path, orig_seqs, task_name, dump_dir):
    """
    Build a Protenix input JSON for target-only prediction from a CIF file.

    This:
      - Converts CIF to a Protenix JSON with entity/asym information.
      - Optionally patches sequences with `orig_seqs`.
      - Drops the last sequence (binder), so only targets remain.
      - Saves the JSON to <dump_dir>/target_protenix_inputs.json and returns the path.
    """
    from protenix.data.json_maker import cif_to_input_json
    from pxdbench.tools.ptx.ptx_utils import patch_with_orig_seqs

    d = cif_to_input_json(cif_path, sample_name=task_name, save_entity_and_asym_id=True)
    if orig_seqs is not None:
        d = patch_with_orig_seqs([d], orig_seqs, use_template=False)[0]
    d["sequences"].pop(-1)

    os.makedirs(dump_dir, exist_ok=True)
    json_path = os.path.join(dump_dir, "target_protenix_inputs.json")
    with open(json_path, "w") as f:
        json.dump([d], f, indent=4)
    return json_path


def keep_target_chains(input_pdb: str, output_pdb: str):
    """
    Keep only the target chains in a complex PDB and drop the last chain
    (binder). The output PDB is overwritten.

    Assumes the last chain corresponds to the binder, and all preceding chains
    are target chains.
    """
    from Bio import PDB

    parser = PDB.PDBParser(QUIET=True)
    io = PDB.PDBIO()
    structure = parser.get_structure("structure", input_pdb)
    all_chains = []
    for model in structure:
        for chain in model:
            all_chains.append(chain)
    n = len(all_chains)

    class ChainSelect(PDB.Select):
        def accept_chain(self, chain):
            return chain in all_chains[: n - 1]

    io.set_structure(structure)
    io.save(output_pdb, select=ChainSelect())


def predict_target_structure(
    configs_eval, input_json_path, gt_cif_path, task_name, dump_dir, device, seed
):
    """
    Run Protenix filter for target-only prediction and compute RMSD
    between predicted target and the ground-truth target structure.

    Steps:
      1) Run ProtenixFilter.inference_only with the provided input JSON.
      2) Convert GT CIF to PDB and drop the binder chain(s).
      3) Permute the predicted complex to minimize target RMSD.
      4) Align predicted and GT target structures and compute RMSD.
      5) Return the Protenix stats dict with an extra 'rmsd' field.
    """
    from pxdbench.metrics.Kalign import align_and_calculate_target_rmsd
    from pxdbench.permutation import permute_generated_min_complex_rmsd
    from pxdbench.tools.ptx.ptx import ProtenixFilter
    from pxdbench.utils import convert_cif_to_pdb

    ptx_cfg = configs_eval.binder.tools.ptx
    ptx_filter = ProtenixFilter(ptx_cfg, device)

    pred_pdb_paths, pred_stats = ptx_filter.inference_only(
        input_json_path=input_json_path,
        dump_dir=dump_dir,
        seed=seed,
        N_sample=ptx_cfg.N_sample,
        N_step=ptx_cfg.N_step,
        step_scale_eta=ptx_cfg.step_scale_eta,
        gamma0=ptx_cfg.gamma0,
        N_cycle=ptx_cfg.N_cycle,
        use_msa=True,
    )
    pred_pdb_path = pred_pdb_paths[task_name]
    pred_stat = pred_stats[task_name][0]

    gt_pdb_path = os.path.join(dump_dir, f"gt_{task_name}.pdb")
    convert_cif_to_pdb(gt_cif_path, gt_pdb_path)
    keep_target_chains(gt_pdb_path, gt_pdb_path)

    permute_generated_min_complex_rmsd(pred_pdb_path, gt_pdb_path, pred_pdb_path)
    rmsd = align_and_calculate_target_rmsd(pred_pdb_path, gt_pdb_path)
    pred_stat["rmsd"] = rmsd
    return pred_stat


def use_target_template_or_not(
    configs_eval,
    pipeline_args,
    gt_cif_path,
    orig_seqs,
    task_name,
    dump_dir,
    device,
    seed,
):
    """
    Decide whether to use the target structure as a template in PTX filter.

    Procedure:
      - Build a target-only PTX JSON from the GT CIF.
      - Populate MSA from cache (if available).
      - Run Protenix prediction on the target and compute RMSD between
        predicted and GT target structures.
      - If the RMSD is below `pipeline_args['target_template_rmsd_thres']`,
        treat the prediction as template-like and return False
        (i.e., do NOT use target template for binder scoring).
      - Otherwise, return True (use target template in PTX filter).
    """
    json_path = create_target_ptx_json(gt_cif_path, orig_seqs, task_name, dump_dir)

    # Add precomputed MSA if needed
    with open(json_path, "r") as f:
        input_dicts = json.load(f)
    input_dicts = populate_msa_with_cache(input_dicts)
    with open(json_path, "w") as f:
        json.dump(input_dicts, f, indent=4)

    pred_stat = predict_target_structure(
        configs_eval, json_path, gt_cif_path, task_name, dump_dir, device, seed
    )
    print(f"[INFO] Target RMSD: {pred_stat['rmsd']}")
    if pred_stat["rmsd"] < pipeline_args["target_template_rmsd_thres"]:
        return False
    else:
        return True


# ============================================================================
# PTX re-run helpers (final refinement)
# ============================================================================


def rerun_ptx(
    result_df: pd.DataFrame,
    base_dir: str,
    task_name: str,
    orig_seqs: list,
    configs_eval,
):
    """
    Rerun the full Protenix filter on a subset of designs and write outputs.

    This is typically used for a small number of top-ranked designs to get
    more accurate PTX scores.

    Steps:
      - Copy selected CIFs into <base_dir>/ptx_final_outputs/<task_name>.
      - Convert CIFs to PDBs and build PTX evaluation inputs.
      - Configure eval to run only the full Protenix filter.
      - Run `run_task` once (num_seqs=1) and return the eval results list.
    """
    final_tmp_dir = os.path.join(base_dir, "ptx_final_outputs", task_name)
    os.makedirs(final_tmp_dir, exist_ok=True)

    pdb_name_to_binder_seq_list = {}
    for _, row in result_df.iterrows():
        src = os.path.join(
            base_dir,
            f"global_run_{row['run_idx']}",
            row["task_name"],
            f"seed_{row['seed']}",
            "predictions",
            row["name"] + ".cif",
        )
        dst_name = f"run_{row['run_idx']}_{row['name']}_seq{row['seq_idx']}"
        dst = os.path.join(final_tmp_dir, dst_name + ".cif")
        shutil.copy(src, dst)
        pdb_name_to_binder_seq_list[dst_name] = [row["sequence"]]

    pdb_dir, pdb_names, cond_chains, binder_chains = convert_cifs_to_pdbs(final_tmp_dir)
    eval_inputs = {
        "task": "binder",
        "name": task_name,
        "pdb_dir": pdb_dir,
        "pdb_names": pdb_names,
        "cond_chains": cond_chains,
        "binder_chains": binder_chains,
        "out_dir": final_tmp_dir,
        "orig_seqs": orig_seqs,
        "pdb_name_to_binder_seq_list": pdb_name_to_binder_seq_list,
    }

    configs_eval.binder.eval_protenix_mini = False
    configs_eval.binder.eval_protenix = True
    configs_eval.binder.eval_complex = False
    configs_eval.binder.eval_binder_monomer = False
    configs_eval.binder.use_binder_seq_list = True
    configs_eval.binder.tools.ptx_mini.N_step = 20
    configs_eval.binder.tools.ptx.N_step = 20
    configs_eval.binder.num_seqs = 1

    return [run_task(eval_inputs, configs_eval, seed=2025)]


# ============================================================================
# Aggregating per-run CSVs
# ============================================================================


def collect_sample_csvs(base_dir: str) -> pd.DataFrame:
    """
    Collect all sample_level_output.csv files from multiple global runs
    into a single DataFrame.

    Expected directory layout:
      base_dir/global_run_*/<task_name>/seed_*/predictions/sample_level_output.csv

    For each CSV, the following metadata columns are prepended:
      - task_name
      - run_idx  (derived from global_run_X)
      - seed     (derived from seed_Y)

    Returns
    -------
    pd.DataFrame
        Concatenated DataFrame of all sample-level outputs. If no CSVs
        are found, returns an empty DataFrame.
    """
    pattern = os.path.join(
        base_dir,
        "global_run_*",
        "*",
        "seed_*",
        "predictions",
        "sample_level_output.csv",
    )
    file_list = glob(pattern)
    if not file_list:
        print("No data found.")
        return pd.DataFrame()

    dfs = []
    for fp in file_list:
        parts = fp.split(os.sep)
        run_idx, task_name, seed = (
            parts[-5].replace("global_run_", ""),
            parts[-4],
            parts[-3].replace("seed_", ""),
        )
        df = pd.read_csv(fp)
        df.insert(0, "task_name", task_name)
        df.insert(1, "run_idx", run_idx)
        df.insert(2, "seed", seed)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def cleanup_outputs(root) -> None:
    root = Path(root)
    # Remove redundant summary files
    for fname in ["all_summary.csv", "filtered_summary.csv"]:
        p = root / fname
        if p.exists():
            p.unlink()

    # Remove PTX re-run outputs
    ptx_dir = root / "ptx_final_outputs"
    if ptx_dir.exists():
        shutil.rmtree(ptx_dir)

    # Remove empty ERR directory (if exists)
    err_dir = root / "ERR"
    if err_dir.exists() and err_dir.is_dir():
        # rmdir() only succeeds if directory is empty
        try:
            err_dir.rmdir()
        except OSError:
            # Directory not empty → keep it
            pass
