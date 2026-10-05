import os

import pandas as pd
import pytest

from pxdesign.runner.helpers import process_boltz_results, save_difficulty_fig
from pxdesign.utils.pipeline import trim_summary_df

BZ_COLS = [
    "bz_ipsae",
    "bz_ipsae_sd",
    "bz_ipdae",
    "bz_ipdae_sd",
    "bz_pae_interface_min",
    "bz_interface_plddt",
    "bz_iptm",
    "bz_ptm",
    "bz_complex_plddt",
    "bz_n_seeds",
    "bz_status",
    "bz_pass_tag",
    "bz_struct_path",
    "bz_batch_id",
    "bz_final_batch_id",
    "bz_gate_egfr_provisional_v1_success",
]


def test_trim_summary_keeps_bz_columns():
    """The silent-loss case: summary.csv is the deliverable."""
    row = {"rank": 1, "task_name": "t", "sequence": "AAAA"}
    row.update({c: 1.0 for c in BZ_COLS})
    kept = trim_summary_df(pd.DataFrame([row]))
    for col in BZ_COLS:
        assert col in kept.columns, col


def test_trim_summary_keeps_a_new_bz_column_too():
    """A prefix sweep, not an allowlist: a new metric must not be dropped."""
    row = {"rank": 1, "task_name": "t", "sequence": "AAAA", "bz_future_metric": 0.5}
    kept = trim_summary_df(pd.DataFrame([row]))
    assert "bz_future_metric" in kept.columns


def test_trim_summary_still_renames_the_legacy_columns():
    """Regression: trim_summary_df renames AF2/Protenix columns on export, and
    that mapping must not change."""
    row = {
        "rank": 1, "task_name": "t", "sequence": "AAAA",
        "af2_easy_success": 1, "ptx_success": 1, "pLDDT": 0.9, "i_pTM": 0.7,
        "ptx_iptm": 0.8,
    }
    kept = trim_summary_df(pd.DataFrame([row]))
    for col in (
        "AF2-IG-easy-success",
        "Protenix-success",
        "af2_plddt",
        "af2_iptm",
        "ptx_iptm",
    ):
        assert col in kept.columns, col


def test_difficulty_fig_does_not_crash_in_boltz_mode(tmp_path):
    """mode='boltz' used to fall into the preview branch and KeyError on
    unscaled_i_pAE / af2_easy_success, which no longer exist."""
    df = pd.DataFrame([{"sequence": "AAAA", "bz_ipsae": 0.8}])
    assert save_difficulty_fig(df, "boltz", str(tmp_path)) is None


def _plant_prediction(base_dir, sample_name, tag="gate", seed=1,
                      text="ATOM  fake\n"):
    """Write a prediction where _fold_batch puts it, and return its path."""
    d = os.path.join(
        base_dir, "boltz_pred", tag, f"seed_{seed}",
        f"boltz_results_{sample_name}", "predictions", sample_name,
    )
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{sample_name}_model_0.pdb")
    with open(path, "w") as handle:
        handle.write(text)
    return path


def _sel(**over):
    row = {
        "rank": 1, "task_name": "t", "name": "bb1", "seq_idx": 0,
        "sequence": "AAAA", "bz_ipsae": 0.8, "pass_boltz": True,
        "bz_struct_path": None,
    }
    row.update(over)
    return pd.DataFrame([row])


def test_process_boltz_results_writes_summary(tmp_path):
    base = str(tmp_path / "base")
    src = _plant_prediction(base, "bb1_seq0")
    out = process_boltz_results(_sel(bz_struct_path=src), base, str(tmp_path / "out"))
    summary = os.path.join(out, "summary.csv")
    assert os.path.exists(summary)
    written = pd.read_csv(summary)
    assert "bz_ipsae" in written.columns
    assert written["chosen_struct_type"].iloc[0] == "boltz"


def test_export_copies_a_real_file_into_the_deliverable(tmp_path):
    """A directory path is not an export; the deliverable must hold the file."""
    base = str(tmp_path / "base")
    src = _plant_prediction(base, "bb1_seq0", text="ATOM  marker\n")
    out = process_boltz_results(_sel(bz_struct_path=src), base, str(tmp_path / "out"))
    written = pd.read_csv(os.path.join(out, "summary.csv"))
    path = written["chosen_struct_path"].iloc[0]
    assert os.path.isfile(path), path
    assert path.startswith(out)
    assert open(path).read() == "ATOM  marker\n"


def test_export_uses_the_persisted_path_not_a_directory_search(tmp_path):
    """The source may live under the task out_dir, under final_batch/, or under
    any pass tag. Only the row knows which."""
    elsewhere = str(tmp_path / "final_batch")
    src = _plant_prediction(elsewhere, "bb1_seq0", tag="final", text="ATOM  fb\n")
    out = process_boltz_results(
        _sel(bz_struct_path=src), str(tmp_path / "unrelated_base"),
        str(tmp_path / "out"),
    )
    written = pd.read_csv(os.path.join(out, "summary.csv"))
    assert written["chosen_struct_type"].iloc[0] == "boltz"
    assert open(written["chosen_struct_path"].iloc[0]).read() == "ATOM  fb\n"


def test_missing_prediction_is_reported_not_faked(tmp_path):
    """Failure-padded rows are non-winners, so keep_structures removed theirs."""
    out = process_boltz_results(
        _sel(pass_boltz=False, bz_struct_path=None),
        str(tmp_path / "base"),
        str(tmp_path / "out"),
    )
    written = pd.read_csv(os.path.join(out, "summary.csv"))
    assert written["chosen_struct_type"].iloc[0] == "none"
    assert str(written["chosen_struct_path"].iloc[0]) in ("", "nan")
    assert "keep_structures" in str(written["chosen_struct_note"].iloc[0])


def test_unresolvable_path_is_reported_with_the_path(tmp_path):
    out = process_boltz_results(
        _sel(bz_struct_path="/gone/x.pdb"), str(tmp_path / "base"),
        str(tmp_path / "out"),
    )
    written = pd.read_csv(os.path.join(out, "summary.csv"))
    assert written["chosen_struct_type"].iloc[0] == "none"
    assert "/gone/x.pdb" in str(written["chosen_struct_note"].iloc[0])


def test_process_boltz_results_never_reruns_protenix(tmp_path, monkeypatch):
    """The 3-seed gate pass IS the confirmation rerun."""
    import pxdesign.runner.helpers as helpers

    called = []
    monkeypatch.setattr(helpers, "rerun_ptx", lambda *a, **k: called.append(1))
    df = pd.DataFrame(
        [
            {
                "rank": 1, "task_name": "t", "name": "bb1", "seq_idx": 0,
                "sequence": "AAAA", "bz_ipsae": 0.8, "pass_ptx": True,
                "bz_struct_path": None, "pass_boltz": True,
            }
        ]
    )
    process_boltz_results(df, str(tmp_path / "base"), str(tmp_path / "out"))
    assert called == []


def test_process_boltz_results_handles_empty(tmp_path):
    assert process_boltz_results(
        pd.DataFrame(), str(tmp_path / "base"), str(tmp_path / "out")
    ) is None


def test_export_names_files_by_rank(tmp_path):
    base = str(tmp_path / "base")
    rows = []
    for rank, idx in ((1, 0), (2, 1)):
        src = _plant_prediction(base, f"bb1_seq{idx}")
        rows.append(
            {
                "rank": rank, "task_name": "t", "name": "bb1", "seq_idx": idx,
                "sequence": "AAAA", "bz_ipsae": 0.8, "pass_boltz": True,
                "bz_struct_path": src,
            }
        )
    out = process_boltz_results(pd.DataFrame(rows), base, str(tmp_path / "out"))
    names = sorted(os.listdir(os.path.join(out, "boltz_docked")))
    assert names == ["rank_1_bb1_seq0.pdb", "rank_2_bb1_seq1.pdb"]


def test_trim_summary_keeps_the_per_design_identifier():
    """summary.csv is the only surviving deliverable: cleanup_outputs deletes
    all_summary.csv and filtered_summary.csv, which were the files carrying
    name/seq_idx. Without them here, an exported design cannot be traced back
    to its backbone."""
    row = {
        "rank": 1, "task_name": "t", "name": "obj2_L55_sample_3", "seq_idx": 2,
        "sequence": "AAAA", "bz_ipsae": 0.8,
    }
    kept = trim_summary_df(pd.DataFrame([row]))
    assert "name" in kept.columns
    assert "seq_idx" in kept.columns
    assert kept["name"].iloc[0] == "obj2_L55_sample_3"
    assert kept["seq_idx"].iloc[0] == 2
