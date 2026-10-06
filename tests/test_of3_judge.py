"""CPU-only tests for the OpenFold3 second judge (funnel/common.py:of3_fold, a3m_rectangular; funnel/run_funnel.py --second-judge). The binary is faked."""
import json, stat, sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "funnel"))
sys.path.insert(0, str(REPO))
import common  # noqa: E402


def test_a3m_rectangular_pads_trailing_gaps_and_keeps_insertions(tmp_path):
    src = tmp_path / "in.a3m"; src.write_text(">q\nACDEF\n>h1\nACabD\n>h2\nAC-EF\n")      # h1: 3 match columns + 2 lowercase insertions
    assert common.a3m_rectangular(src, tmp_path / "out.a3m") == 3
    assert (tmp_path / "out.a3m").read_text() == ">q\nACDEF\n>h1\nACabD--\n>h2\nAC-EF\n"


@pytest.mark.parametrize("body", ["", ">q\nACD\n>h\nACDEF\n", ">q\nACD\n>h\n\n"])
def test_a3m_rectangular_refuses_empty_or_overlong_rows(tmp_path, body):
    src = tmp_path / "in.a3m"; src.write_text(body)
    with pytest.raises(ValueError): common.a3m_rectangular(src, tmp_path / "out.a3m")


def _fake_of3(tmp_path, nt, fail_ids=()):
    """A run_openfold stand-in: writes the files OpenFold3 writes for each query, with a PAE block that makes the interface confident."""
    f = tmp_path / "fake_run_openfold"
    f.write_text(f'''#!{sys.executable}
import json, sys
from pathlib import Path
a = sys.argv; q = json.load(open(a[a.index("--query-json") + 1])); out = Path(a[a.index("--output-dir") + 1])
seed = [l.split()[-1] for l in open(a[a.index("--runner-yaml") + 1]) if l.strip().startswith("-")][0]
for name, v in q["queries"].items():
    if name in {list(fail_ids)!r}: continue
    n = sum(len(c["sequence"]) for c in v["chains"]); d = out / name / f"seed_{{seed}}"; d.mkdir(parents=True)
    json.dump(dict(pae=[[0.5] * n] * n, plddt=[90.0] * n, pde=[[0.5] * n] * n), open(d / f"{{name}}_seed_{{seed}}_sample_1_confidences.json", "w"))
    json.dump(dict(iptm=0.9, sample_ranking_score=0.8), open(d / f"{{name}}_seed_{{seed}}_sample_1_confidences_aggregated.json", "w"))
    (d / f"{{name}}_seed_{{seed}}_sample_1_model.cif").write_text("data_x\\n")
''')
    f.chmod(f.stat().st_mode | stat.S_IEXEC); return f


def _target(tmp_path, rows=3):
    seq = "ACDEFGHIKL"; a3m = tmp_path / "t.a3m"; a3m.write_text(f">q\n{seq}\n" + "".join(f">h{i}\n{seq}\n" for i in range(rows - 1)))
    return dict(seq=seq, msa=str(a3m))


def test_of3_fold_parses_outputs_and_flags_missing_designs(tmp_path, monkeypatch):
    monkeypatch.setenv("PXD_OF3_BIN", str(_fake_of3(tmp_path, 10, fail_ids=("bad",))))
    df = pd.DataFrame(dict(id=["good", "bad"], seq=["MKVLAAGIVG", "MKVLAAGIVA"]))
    d = common.of3_fold(df, _target(tmp_path), tmp_path / "o", seed=7).set_index("id")
    assert bool(d.loc["good", "of3_ok"]) and d.loc["good", "of3_ipsae"] > 0.8 and d.loc["good", "of3_iptm"] == 0.9 and d.loc["good", "of3_rank"] == 0.8
    assert d.loc["good", "of3_cif"].endswith("good_seed_7_sample_1_model.cif")
    assert not bool(d.loc["bad", "of3_ok"])                                           # a design OpenFold3 failed on is reported, not invented
    assert (tmp_path / "o/msa/good/colabfold_main.a3m").read_text() == ">query\nMKVLAAGIVG\n"   # the binder gets a query-only MSA


def test_of3_fold_is_resumable(tmp_path, monkeypatch):
    monkeypatch.setenv("PXD_OF3_BIN", str(_fake_of3(tmp_path, 10)))
    df = pd.DataFrame(dict(id=["a"], seq=["MKVLAAGIVG"])); t = _target(tmp_path)
    common.of3_fold(df, t, tmp_path / "o", seed=7)
    monkeypatch.setenv("PXD_OF3_BIN", "/nonexistent")                                  # a second call must not need the binary
    assert bool(common.of3_fold(df, t, tmp_path / "o", seed=7)["of3_ok"].iloc[0])


def test_of3_fold_refuses_a_target_without_a_real_msa(tmp_path, monkeypatch):
    monkeypatch.setenv("PXD_OF3_BIN", str(_fake_of3(tmp_path, 10)))
    with pytest.raises(ValueError, match="real MSA"):
        common.of3_fold(pd.DataFrame(dict(id=["a"], seq=["MKVLAAGIVG"])), _target(tmp_path, rows=1), tmp_path / "o")


def test_finalize_variant_does_not_reuse_a_table_from_another_second_oracle(tmp_path, monkeypatch):
    import run_funnel as rf
    src = pd.DataFrame(dict(id=["a", "b"], seq=["AAA", "CCC"], fast_ipsae=[.9, .8]))
    pd.DataFrame(dict(id=["a", "b"], consensus_pass=[True, False], consensus=[.7, .2], rank_rule="min")).to_csv(tmp_path / "consensus_x.csv", index=False)   # old run: no o2_name -> Protenix-v2
    pd.DataFrame(dict(id=["a"])).to_csv(tmp_path / "final_x.csv", index=False)
    calls = []
    def fake(t, c, outdir, seeds, T, tag, gpus=None, oracle2="protenix-v2", o2_gate=0.5, rank_rule="min"):
        calls.append(oracle2); return pd.DataFrame(dict(id=c.id, consensus_pass=True, consensus=.9, o2_name=oracle2))
    monkeypatch.setattr(rf, "consensus", fake); monkeypatch.setattr(rf, "shortlist", lambda d, top: d.head(1))
    class S:
        def mark(self, *a, **k): pass
    J = lambda o: dict(seeds=(1, 2, 3), oracle2=o, o2_gate=0.5, rank_rule="min")
    rf.finalize_variant({}, tmp_path, "x", src, 2, 1, None, S(), False, None, J("protenix-v2")); assert calls == []       # same oracle: reused
    rf.finalize_variant({}, tmp_path, "x", src, 2, 1, None, S(), False, None, J("of3")); assert calls == ["of3"]          # different oracle: recomputed


def test_of3_backend_returns_the_shared_v2_columns(tmp_path, monkeypatch):
    import oracles
    monkeypatch.setenv("PXD_OF3_BIN", str(_fake_of3(tmp_path, 10)))
    d = oracles.of3_fold(pd.DataFrame(dict(id=["a"], seq=["MKVLAAGIVG"])), _target(tmp_path), tmp_path / "o", seed=7)
    assert "of3" in oracles.ORACLES and bool(d.v2_ok.iloc[0]) and d.v2_ipsae.iloc[0] > 0.8 and not any(c.startswith("of3_") for c in d.columns)


def test_final_design_tier_reads_the_shared_second_oracle_column():
    import final_design as fdz
    base = dict(consensus_pass=True, b_ipsae=.75, b_paemin=.6, hotspot_frac=1.0, pisa_flags="")
    assert fdz.tier({**base, "v2_ipsae": .78}) == "A" and fdz.tier({**base, "v2_ipsae": .60}) == "B"
    assert fdz.O2_LABEL["of3"] == "OpenFold3"
