"""scripts/campaign/*: the generic tools extracted from a real campaign. Each test pins a behaviour that went wrong, or nearly did, in use."""
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
CAMP = REPO / "scripts" / "campaign"
sys.path.insert(0, str(CAMP)); sys.path.insert(0, str(REPO / "funnel")); sys.path.insert(0, str(REPO))
import arcrefine_jobs as aj  # noqa: E402
import epitope_from_complex as ep  # noqa: E402
import harvest_boltzgen as hb  # noqa: E402
import select_panel as sp  # noqa: E402


# ----------------------------------------------------------------------------------------------- epitope
def _atom(serial, name, res, chain, num, x, y, z):
    elem = name[0]
    return f"ATOM  {serial:5d} {name:<4s} {res:>3s} {chain}{num:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {elem:>2s}"


def _pdb(tmp_path, partner_x=3.5, name="c.pdb"):
    """Target chain A: GLU1 at the origin, LYS2 20 A away, TYR3 near GLU1. Partner chain B: one atom partner_x A from GLU1's CA."""
    L = [_atom(1, " CA ", "GLU", "A", 1, 0, 0, 0), _atom(2, " CB ", "GLU", "A", 1, 1, 0, 0),
         _atom(3, " CA ", "LYS", "A", 2, 20, 0, 0), _atom(4, " CA ", "TYR", "A", 3, 0, 6, 0),
         _atom(5, " CA ", "ALA", "B", 7, partner_x, 0, 0), _atom(6, " CB ", "ALA", "B", 7, partner_x, 1, 0), "END"]
    p = tmp_path / name; p.write_text("\n".join(L) + "\n"); return p


def test_epitope_contacts_pick_only_residues_near_the_partner(tmp_path):
    atoms, res = ep.analyse(str(_pdb(tmp_path)), "A", ["B"], cutoff=4.5)
    assert set(res) == {1} and res[1][0] == "E" and res[1][1] >= 2        # LYS2 is 20 A away, TYR3 6 A from the nearest partner atom: both out
    assert ep.suggest(res, 5) == ["E1"]


def test_epitope_geometry_is_span_and_radius_of_gyration(tmp_path):
    atoms, _ = ep.analyse(str(_pdb(tmp_path)), "A", ["B"])
    span, rg = ep.geometry(atoms, [1, 2])                                    # CA of GLU1 (0,0,0) and LYS2 (20,0,0)
    assert span == pytest.approx(20.0) and rg == pytest.approx(10.0)


def test_epitope_compare_reports_the_shared_residues(tmp_path, capsys):
    a, b = _pdb(tmp_path, name="a.pdb"), _pdb(tmp_path, name="b.pdb")
    ep.main([str(a), "A", "B", "--compare", str(b), "A", "B"])
    out = capsys.readouterr().out
    assert "shared with the first: 1" in out and "touched by BOTH partners: ['E1']" in out


def test_epitope_unknown_chain_is_a_clear_error(tmp_path):
    with pytest.raises(SystemExit, match="not found"):
        ep.analyse(str(_pdb(tmp_path)), "Z", ["B"])


# ----------------------------------------------------------------------------------------------- panel selection
def _judged():
    base = dict(consensus_pass=True, b_paemin=.6, hotspot_frac=1.0, pisa_flags=np.nan)
    rows = [
        dict(id="a1", seq="A" * 30, b_ipsae=.90, v2_ipsae=.88, **base),
        dict(id="a1dup", seq="A" * 29 + "C", b_ipsae=.89, v2_ipsae=.87, **base),              # same 60% cluster as a1: must not take a second slot
        dict(id="b_thin", seq="D" * 30, b_ipsae=.92, v2_ipsae=.91, **{**base, "pisa_flags": "thin_interface"}),      # ranked-on flag: tier B
        dict(id="a_polar", seq="E" * 30, b_ipsae=.85, v2_ipsae=.84, **{**base, "pisa_flags": "polar_interface;no_aromatic_contact"}),   # NOT ranked on: stays tier A
        dict(id="offsite", seq="F" * 30, b_ipsae=.95, v2_ipsae=.95, **{**base, "hotspot_frac": 0.0}),                # passes but not on target
        dict(id="fails", seq="G" * 30, b_ipsae=.40, v2_ipsae=.40, **{**base, "consensus_pass": False}),
    ]
    d = pd.DataFrame(rows); d["consensus"] = d[["b_ipsae", "v2_ipsae"]].min(axis=1); return d


def test_selection_rule_tier_cluster_and_ranking():
    panel, elig = sp.select(_judged(), top=10)
    assert panel.id.tolist() == ["a1", "a_polar", "b_thin"]                  # tier A by consensus, then tier B; dup, off-site and failing rows are out
    assert set(panel.tier[panel.id.isin(["a1", "a_polar"])]) == {"A"} and panel.tier[panel.id == "b_thin"].iloc[0] == "B"
    assert "a1dup" in set(elig.id) and not bool(elig.keep[elig.id == "a1dup"].iloc[0])


def test_polar_and_aromatic_flags_are_reported_but_not_ranked_on_unless_asked():
    d = _judged()
    assert sp.tier(d[d.id == "a_polar"].iloc[0]) == "A"
    assert sp.tier(d[d.id == "a_polar"].iloc[0], rank_flags=("polar_interface",)) == "B"


def test_selection_cli_excludes_controls_and_writes_an_audit_table(tmp_path):
    d = _judged(); d["arms"] = "arm1"; c = d.iloc[[0]].copy(); c["id"] = "ctl"; c["seq"] = "W" * 30; c["arms"] = "controls"
    f = tmp_path / "j.csv"; pd.concat([d, c]).to_csv(f, index=False)
    sp.main(["--judged", str(f), "--out", str(tmp_path / "panel.csv"), "--top", "5"])
    assert "ctl" not in pd.read_csv(tmp_path / "panel.csv").id.tolist() and (tmp_path / "panel_eligible.csv").exists()


# ----------------------------------------------------------------------------------------------- harvest
def test_harvest_keeps_a_binder_longer_than_130_aa():
    """Regression: an earlier cap of 130 aa silently dropped every 140-aa design ('extracted 0')."""
    chains = {"A": "A" * 140, "B": "C" * 115}
    assert hb.binder_sequence(chains, target_len=115, min_len=20, max_len=300) == "A" * 140


def test_harvest_refuses_an_ambiguous_or_missing_binder():
    assert hb.binder_sequence({"A": "A" * 70, "B": "C" * 80, "C": "D" * 115}, 115, 20, 300) is None       # two non-target chains
    assert hb.binder_sequence({"A": "C" * 115}, 115, 20, 300) is None                                     # only the target
    assert hb.binder_sequence({"A": "AXA" * 20, "B": "C" * 115}, 115, 20, 300) is None                   # unknown residue


# ----------------------------------------------------------------------------------------------- arcrefine
def test_arcrefine_parent_choice_takes_passing_on_target_designs_best_first():
    d = _judged()
    assert aj.pick(d, 2).id.tolist() == ["b_thin", "a1"]                       # offsite and failing designs excluded; highest consensus first


def _worker_env(tmp_path, n_jobs):
    arc = tmp_path / "arc.sh"
    arc.write_text('#!/bin/bash\nmkdir "$3" && echo \'{"optimized_sequence": "AAAA"}\' > "$3/result.json"\n')
    arc.chmod(arc.stat().st_mode | stat.S_IEXEC)
    root = tmp_path / "jobs"
    for i in range(n_jobs):
        (root / f"job{i}").mkdir(parents=True); (root / f"job{i}/config.json").write_text("{}")
    return arc, root


def _run_worker(arc, root, k, n):
    return subprocess.run(["bash", str(CAMP / "arcrefine_worker.sh"), str(k), str(n)], capture_output=True, text=True,
                          env={**os.environ, "ARC_BIN": str(arc), "JOBS_ROOT": str(root)})


def test_arcrefine_workers_take_disjoint_jobs_by_static_assignment(tmp_path):
    arc, root = _worker_env(tmp_path, 5)
    r0, r1 = _run_worker(arc, root, 0, 2), _run_worker(arc, root, 1, 2)
    done0 = [l.split()[3] for l in r0.stdout.splitlines() if " done " in l]
    done1 = [l.split()[3] for l in r1.stdout.splitlines() if " done " in l]
    assert done0 == ["job0", "job2", "job4"] and done1 == ["job1", "job3"]       # no overlap, nothing missed
    assert all((root / f"job{i}/run/result.json").exists() for i in range(5))


def test_arcrefine_worker_reports_the_real_exit_code_and_is_resumable(tmp_path):
    arc, root = _worker_env(tmp_path, 2)
    arc.write_text("#!/bin/bash\nexit 3\n")                                       # a failing refiner
    r = _run_worker(arc, root, 0, 1)
    assert "rc=3 result=no" in r.stdout                                           # not a masked 0 (the $? trap)
    arc.write_text('#!/bin/bash\nmkdir "$3" && echo \'{}\' > "$3/result.json"\n')
    assert "result=yes" in _run_worker(arc, root, 0, 1).stdout
    assert "skip job0 (done)" in _run_worker(arc, root, 0, 1).stdout               # second pass: already done


# ----------------------------------------------------------------------------------------------- AF3 inputs
def test_af3_inputs_use_the_sanitised_alignment_and_collision_safe_names(tmp_path, monkeypatch):
    import common, make_af3_inputs as m
    seq = "ACDEFGHIKL"
    msa = tmp_path / "t.a3m"; msa.write_bytes(f">q\n{seq}\n>bad\nAC\x00EFGHIKL\n>ok\n{seq}\n".encode())     # one NUL-poisoned record
    monkeypatch.setattr(common, "load_target", lambda n: dict(seq=seq, msa=str(msa)))
    d = tmp_path / "d.csv"; pd.DataFrame(dict(id=["Design One", "x2"], seq=["MKV" * 5, "MKV" * 6])).to_csv(d, index=False)
    m.main(["--target", "t", "--designs", str(d), "--out", str(tmp_path / "o"), "--remote-msa", "/remote/t.a3m"])
    clean = (tmp_path / "o/t_target_msa.a3m").read_bytes()
    assert b"\x00" not in clean and clean.count(b">") == 2                       # the poisoned record was dropped, not shipped
    js = sorted((tmp_path / "o/json").glob("*.json")); assert len(js) == 2
    j = json.loads(js[0].read_text()); assert j["sequences"][0]["protein"]["unpairedMsaPath"] == "/remote/t.a3m"
