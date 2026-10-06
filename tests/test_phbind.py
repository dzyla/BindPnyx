"""CPU-only tests for phbind/ (homo-oligomer target path): grouping, the min-direction gate, submission contract, MSA crop, His variants,
and the funnel/common.py multi-chain guard. No GPU, no models, no target files, no private data (sequences are synthetic or the public TNF construct)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO), str(REPO / "phbind"), str(REPO / "funnel")]
import trimer as T  # noqa: E402
import s8_assemble as S8  # noqa: E402
import msa_crop  # noqa: E402

NT, NB = 471, 40


def _ca_pdb(path, chains):
    """CA-only PDB, chains = [(id, sequence)]; residues 1..N, 3.8 A apart on a line per chain."""
    rev = {v: k for k, v in T.AA3to1.items() if k not in ("MSE", "SEC", "PYL", "UNK")}
    lines, n = [], 0
    for ci, (cid, seq) in enumerate(chains):
        for i, aa in enumerate(seq, 1):
            n += 1
            lines.append(f"ATOM  {n:5d}  CA  {rev[aa]:>3s} {cid}{i:4d}    {3.8 * i:8.3f}{10.0 * ci:8.3f}{0.0:8.3f}  1.00  0.00           C")
    Path(path).write_text("\n".join(lines) + "\nEND\n")


def _pae(seed=0):
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.5, 25.0, size=(NT + NB, NT + NB))
    return p


def test_grouped_ipsae_matches_funnel_when_target_chains_come_first():
    import common
    pae = _pae()
    t, b = np.arange(NT), np.arange(NT, NT + NB)
    assert T.ipsae_grouped(pae, t, b) == pytest.approx(common.ipsae(pae, NT, NB), abs=1e-12)


def test_grouping_is_the_point_one_protomer_only_does_not_equal_whole_target():
    pae = np.full((NT + NB, NT + NB), 30.0)
    b = np.arange(NT, NT + NB)
    pae[np.ix_(b, np.arange(157, 314))] = 0.5            # binder confident ONLY against protomer 2
    pae[np.ix_(np.arange(157, 314), b)] = 0.5
    lo, hi = T.ipsae_grouped(pae, np.arange(NT), b)
    assert lo > 0.5 and hi > 0.5                          # whole target as one group sees the interface
    assert T.ipsae_grouped(pae, np.arange(0, 157), b) == (0.0, 0.0)   # protomer 1 alone sees none


def test_group_indices_is_sequence_based_and_asserts_the_trimer(tmp_path):
    binder = "MKTAYIAKQRQISFVKSHFSRQ" * 2
    f = tmp_path / "ok.pdb"
    _ca_pdb(f, [("A", T.TNF_HUMAN), ("B", T.TNF_HUMAN), ("C", T.TNF_HUMAN), ("D", binder)])
    ti, bi, info = T.group_indices(str(f), species="human", binder_seq=binder)
    assert len(ti) == 471 and len(bi) == len(binder) and info["target_chains"] == ["A", "B", "C"]
    g = tmp_path / "two.pdb"                              # a dimer where a trimer is expected must raise, not score
    _ca_pdb(g, [("A", T.TNF_HUMAN), ("B", T.TNF_HUMAN), ("D", binder)])
    with pytest.raises(T.GroupingError):
        T.group_indices(str(g), species="human")


def test_funnel_common_refuses_multichain_complexes(tmp_path):
    """Without _two_chains this returns a number (target chain B read as the binder). Delete the guard and this test must fail."""
    import common
    from Bio.PDB import PDBParser, MMCIFIO
    f = tmp_path / "t.pdb"
    _ca_pdb(f, [("A", "ACDEFGHIKL"), ("B", "ACDEFGHIKL"), ("C", "ACDEFGHIKL"), ("D", "MKTAYIAKQR")])
    io = MMCIFIO(); io.set_structure(PDBParser(QUIET=True).get_structure("x", str(f))); cif = tmp_path / "t.cif"; io.save(str(cif))
    with pytest.raises(ValueError, match="exactly 2 chains"):
        common.hotspot_contacts(str(cif), 10, [1, 2])


def test_gate_is_calibrated_on_the_min_direction_not_max():
    sys.path.insert(0, str(REPO))
    from phbind import boltz_trimer as B
    rows = []
    for i, (mn, mx) in {"stable": (0.72, 0.88), "max_only": (0.30, 0.85)}.items():
        for s in (101, 202, 303):
            rows.append(dict(id=i, seed=s, binder_len=70, ipsae_min=mn, ipsae_max=mx, b2t_pair_iptm=0.8, p1x_recall=0.5, foot_recall=0.5, groove="AB"))
    s = B.summarise(pd.DataFrame(rows)).set_index("id")
    assert s.loc["stable", "gate_pass"] and not s.loc["max_only", "gate_pass"]      # high max with low min must NOT pass
    assert s.loc["max_only", "ipsae_max_mean"] == pytest.approx(0.85)                # but max is still reported
    split = pd.DataFrame([dict(id="split", seed=s_, binder_len=70, ipsae_min=m, ipsae_max=0.9, b2t_pair_iptm=.8, p1x_recall=.5, foot_recall=.5, groove="AB")
                          for s_, m in zip((101, 202, 303), (0.9, 0.9, 0.2))])
    assert not B.summarise(split).iloc[0].gate_pass                                  # one bad seed breaks unanimity even with a high mean


def _cand(n=120):
    rng = np.random.default_rng(1)
    aas = list("ADEFGHIKLMNPQRSTVWY")
    return pd.DataFrame([dict(name=f"d{i}", sequence="".join(rng.choice(aas, 70 + i % 40)), block=["ph", "affinity", "xreact"][i % 3], lineage=f"L{i % 6}", rank=i) for i in range(n)])


def test_submission_contract_accepts_a_good_panel_and_rejects_each_violation():
    sub, _ = S8.assemble(_cand())
    assert len(sub) == 40 and S8.check_submission(sub)
    bad = {"cysteine": sub.assign(sequence=sub.sequence.str.replace("A", "C", n=1)), "short": sub.assign(sequence=sub.sequence.str[:30]),
           "lowercase": sub.assign(sequence=sub.sequence.str.lower()), "header": sub.rename(columns={"name": "id"}),
           "duplicate": sub.assign(sequence=sub.sequence.iloc[0])}
    for k, v in bad.items():
        with pytest.raises(AssertionError):
            S8.check_submission(v)
    with pytest.raises(AssertionError):                    # one lineage cannot fill the panel
        S8.assemble(_cand().assign(lineage="L0"))


def test_lineages_are_interleaved_inside_a_block():
    d = pd.DataFrame([dict(name=f"{l}{i}", sequence="A" * 70 + chr(65 + i) * 3 + l, block="ph", lineage=l, rank=i) for l in "XY" for i in range(5)])
    order = list(S8.interleave(d).lineage)
    assert all(a != b for a, b in zip(order, order[1:]))


def test_msa_crop_keeps_match_columns_and_inside_insertions(tmp_path):
    src = tmp_path / "a.a3m"
    src.write_text(">q\nABCDEFGH\n>h1\nAB-DeFGHI\n>h2\nAxBCDEFGH\n")        # h1 has an insertion 'e' between match cols 4 and 5; h2 'x' before col 2
    out = tmp_path / "c.a3m"
    assert msa_crop.crop_a3m(src, out, 3, 6) == 3
    rows = out.read_text().split("\n")
    assert rows[1] == "CDEF" and rows[3] == "-DeFG" and rows[5] == "CDEF"


def test_his_variants_differ_only_at_declared_positions_and_never_add_cys():
    from s4_variants import variants
    seq = "MKTAYIAKQRQISFVKSHFSRQLEEAGKKLNPDWQ"
    v = variants(seq, m3=[(5, 9), (10, 14), (20, 24)], m1=[7, 27])
    assert v and all(len(s) == len(seq) and "C" not in s for s, _ in v.values())
    for name, (s, pos) in v.items():
        assert {i + 1 for i, (a, b) in enumerate(zip(seq, s)) if a != b} <= set(pos) and all(s[p - 1] == "H" for p in pos)
    assert max(len(p) for _, p in v.values()) <= 3         # one-shot His count is capped: three broke 4/6 scaffolds in a measured campaign
