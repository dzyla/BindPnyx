"""CPU-only tests for funnel/common.py and funnel/run_funnel.py:shortlist. No GPU, no models."""
import json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "funnel"))
sys.path.insert(0, str(REPO))
import common  # noqa: E402


def test_ipsae_is_high_for_a_confident_interface_and_zero_for_none():
    nt, nb = 40, 30
    good = np.full((nt + nb, nt + nb), 0.5)
    assert common.ipsae(good, nt, nb)[0] > 0.8
    bad = np.full((nt + nb, nt + nb), 30.0)          # nothing under the PAE cutoff
    assert common.ipsae(bad, nt, nb) == (0.0, 0.0)


def test_ipsae_min_is_the_conservative_direction():
    nt, nb = 40, 30
    pae = np.full((nt + nb, nt + nb), 0.5)
    pae[:nt, nt:] = 30.0                              # target->binder uncertain, binder->target confident
    lo, hi = common.ipsae(pae, nt, nb)
    assert lo == 0.0 and hi > 0.8


def test_pae_interface_min():
    pae = np.full((10, 10), 20.0); pae[7, 2] = 0.7
    assert common.pae_interface_min(pae, 5) == pytest.approx(0.7)


def test_identity_and_cluster_count():
    a, b, c = "AAAAAAAAAA", "AAAAAAAAAB", "CDEFGHIKLM"
    assert common.identity(a, b) == pytest.approx(0.9)
    assert common.cluster_count([a, b, c], thr=0.6) == 2 and common.cluster_count([a, b, c], thr=0.95) == 3


def _load_or_skip(name):
    try: return common.load_target(name)
    except FileNotFoundError as e: pytest.skip(str(e))


def test_every_shipped_target_loads_with_residue_identity_checked():
    for name in ("pdl1", "mdm2", "fima"):
        t = _load_or_skip(name)
        assert len(t["hotspot_idx"]) == len(t["hotspots"]) and max(t["hotspot_idx"]) <= len(t["seq"])
        for h, i in zip(t["hotspots"], t["hotspot_idx"]):
            assert t["seq"][i - 1] == h[0], f"{name}: hotspot {h} -> shard index {i} has the wrong residue"


def test_a_hotspot_with_the_wrong_identity_is_refused(tmp_path):
    _load_or_skip("mdm2")
    t = json.load(open(REPO / "funnel" / "targets" / "mdm2.json")); t["hotspots"] = ["A54"]   # residue 54 is LEU
    p = tmp_path / "bad.json"; json.dump(t, open(p, "w"))
    with pytest.raises(ValueError, match="not found with that identity"):
        common.load_target(p)


def test_the_pdb_numbered_pdl1_list_that_once_ran_silently_is_what_the_guard_exists_for():
    """Shard index = PDB number - 17 for PD-L1 (chain starts at 18). Using the PDB numbers as indices hits other residues."""
    t = _load_or_skip("pdl1")
    wrong = [t["seq"][i - 1] for i in (56, 58, 113, 115, 123)]
    assert wrong != [h[0] for h in t["hotspots"]]


def _two_chain_cif(path, gap):
    from Bio.PDB import MMCIFIO, Structure, Model, Chain, Residue, Atom
    s = Structure.Structure("x"); m = Model.Model(0); s.add(m)
    for cid, z0 in (("A", 0.0), ("B", gap)):
        ch = Chain.Chain(cid); m.add(ch)
        for i in range(1, 4):
            r = Residue.Residue((" ", i, " "), "GLY", " "); r.add(Atom.Atom("CA", np.array([float(i * 3), 0.0, z0]), 0, 1, " ", "CA", i, "C")); ch.add(r)
    io = MMCIFIO(); io.set_structure(s); io.save(str(path))


def test_hotspot_contacts_counts_only_residues_within_the_cutoff(tmp_path):
    near, far = tmp_path / "near.cif", tmp_path / "far.cif"
    _two_chain_cif(near, gap=4.0); _two_chain_cif(far, gap=20.0)
    assert common.hotspot_contacts(near, 3, [1, 2, 3]) == (1.0, 3)
    assert common.hotspot_contacts(far, 3, [1, 2, 3]) == (0.0, 0)


def test_shortlist_orders_by_consensus_and_drops_near_duplicates():
    import run_funnel
    d = pd.DataFrame(dict(seq=["A" * 20, "A" * 19 + "C", "D" * 20, "E" * 20], consensus=[0.9, 0.85, 0.8, float("nan")]))
    s = run_funnel.shortlist(d, top=5)
    assert s.seq.tolist() == ["A" * 20, "D" * 20]       # near-duplicate and NaN rows dropped


def test_pisa_batch_reports_errors_instead_of_raising_on_a_bad_structure(tmp_path):
    import pisa; pytest.importorskip("freesasa")
    bad = tmp_path / "bad.cif"; bad.write_text("not a structure")
    out = pisa.batch([("x", str(bad), None)], workers=1)
    assert out[0]["id"] == "x" and "error" in out[0]


def test_add_pisa_never_blocks_a_run_when_the_metrics_are_unavailable(monkeypatch):
    import run_funnel, pisa
    monkeypatch.setattr(pisa, "batch", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    d = pd.DataFrame(dict(id=["a"], b_cif=["x.cif"], consensus=[0.9]))
    out = run_funnel.add_pisa(d, {"hotspot_idx": [1]})
    assert list(out.columns) == ["id", "b_cif", "consensus"]


def test_shape_complementarity_is_high_for_matching_surfaces_and_falls_when_pulled_apart():
    """Two interlocking bumpy 'surfaces' (complementary by construction) vs the same pair displaced. No structure files needed."""
    import sc
    rng = np.random.default_rng(1)
    a = np.array([[x, y, 0.0] for x in np.arange(-6, 6.1, 1.5) for y in np.arange(-6, 6.1, 1.5)]); a[:, 2] += 0.4 * np.sin(a[:, 0]) * np.cos(a[:, 1])
    b = a.copy(); b[:, 2] = -a[:, 2] + 3.6                       # mirror image placed on top: surfaces fit
    ra, rb = np.full(len(a), 1.9), np.full(len(b), 1.9)
    fit = sc.shape_complementarity(a, ra, b, rb)["sc"]
    far = sc.shape_complementarity(a, ra, b + np.array([0.0, 0.0, 3.0]), rb)["sc"]
    assert fit > 0.5 and (np.isnan(far) or far < fit - 0.2)


def test_sc_shipped_reference_complexes_are_in_the_expected_range():
    from pathlib import Path as P
    f = REPO / "bench/sc_validation/2PTC.pdb"
    if not f.exists(): pytest.skip("reference PDB not downloaded")
    import sc
    r = sc.complex_sc(f, ("E",), ("I",)); assert 0.55 < r["sc"] < 0.8      # trypsin-BPTI; published protease-inhibitor values are ~0.7


def test_final_design_tiers():
    import final_design as fdz
    base = dict(consensus_pass=True, b_ipsae=.75, v2_ipsae=.78, b_paemin=.6, hotspot_frac=1.0, pisa_flags="")
    assert fdz.tier(base) == "A"
    assert fdz.tier({**base, "pisa_flags": "no_aromatic_contact"}) == "B"          # any flag demotes A to B
    assert fdz.tier({**base, "b_ipsae": .65}) == "B"
    assert fdz.tier({**base, "consensus_pass": False}) == "C"


# ----------------------------------------------------------------------------- resume / extend logic (funnel/run_funnel.py)
def test_gen_units_plans_from_finished_chunks_and_never_redoes_them(tmp_path):
    import run_funnel as rf
    (tmp_path / "gen").mkdir()
    units = rf.gen_units(tmp_path, 250, 100)
    assert units == [("chunk_000", 0, 100), ("chunk_001", 100, 100), ("chunk_002", 200, 50)]
    for k, o, c in units:
        (tmp_path / "gen" / k).mkdir(); json.dump(dict(offset=o, count=c), open(tmp_path / "gen" / k / "done.json", "w"))
    ext = rf.gen_units(tmp_path, 400, 100)
    assert ext[:3] == units and ext[3:] == [("chunk_003", 250, 100), ("chunk_004", 350, 50)]      # the short last chunk is kept, new ones follow it
    (tmp_path / "gen" / "chunk_003").mkdir()                                                          # an interrupted chunk (no done.json) is simply planned again
    assert rf.gen_units(tmp_path, 400, 100) == ext


def test_a_legacy_single_shot_run_is_adopted_as_the_first_unit(tmp_path):
    import run_funnel as rf
    (tmp_path / "gen").mkdir(); json.dump(dict(pdb_dir="x", n=500), open(tmp_path / "gen" / "converted.json", "w"))
    assert rf.gen_units(tmp_path, 700, 100) == [("legacy", 0, 500), ("chunk_000", 500, 100), ("chunk_001", 600, 100)]


def test_resume_with_a_changed_guarded_setting_is_refused_but_growing_a_run_is_not(tmp_path):
    import run_funnel as rf
    cfg = dict(target="a", hotspot_idx=[1], binder_length=70, chunk=100, steps=400, seqs=4, mpnn_weights="soluble", mpnn_bias="none", cycle_scope="full", designs_csv_sha=None, n_backbones=100)
    rf.check_config(tmp_path, cfg, False)
    rf.check_config(tmp_path, dict(cfg, n_backbones=900, rounds=6, final_m=200), False)                # extending is allowed
    with pytest.raises(SystemExit, match="hotspot_idx"):
        rf.check_config(tmp_path, dict(cfg, hotspot_idx=[2]), False)
    rf.check_config(tmp_path, dict(cfg, hotspot_idx=[2]), True)                                        # explicit override


def test_atomic_writers_never_leave_a_partial_file(tmp_path):
    import run_funnel as rf
    rf.save_csv(pd.DataFrame(dict(a=[1, 2])), tmp_path / "x.csv"); rf.save_json({"k": 1}, tmp_path / "x.json")
    assert pd.read_csv(tmp_path / "x.csv").a.tolist() == [1, 2] and json.load(open(tmp_path / "x.json")) == {"k": 1}
    assert not list(tmp_path.glob("*.tmp"))


def test_parallel_map_keeps_order_and_pins_workers_to_gpus():
    seen = []
    def f(i): seen.append((i, common._tls.gpu)); return i * i
    out = common.parallel_map(f, range(6), ["0", "1", "2"])
    assert out == [0, 1, 4, 9, 16, 25] and {g for _, g in seen} == {"0", "1", "2"}
    assert common.gpu_list("0, 1,2") == ["0", "1", "2"] and common.gpu_list("") == [None]


def test_site_occlusion_maps_the_reference_ligand_into_the_prediction(tmp_path):
    """Predicted target = the reference target moved rigidly; a binder atom placed on the ligand's (moved) position must register as a clash."""
    from Bio.PDB import MMCIFIO, Structure, Model, Chain, Residue, Atom
    rng = np.random.default_rng(0); ref_ca = rng.normal(size=(12, 3)) * 8; probe = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
    d = tmp_path / "site"; d.mkdir(); np.save(d / "ref_ca.npy", ref_ca); np.save(d / "probe.npy", probe)
    th = 0.7; R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]]); tr = np.array([5.0, -3.0, 2.0])
    pred_ca, pred_probe = ref_ca @ R.T + tr, probe @ R.T + tr
    s = Structure.Structure("x"); m = Model.Model(0); s.add(m)
    for cid, pts in (("A", pred_ca), ("B", pred_probe + np.array([0.0, 0.0, 1.0]))):                # binder atoms 1.0 A from the moved ligand atoms
        ch = Chain.Chain(cid); m.add(ch)
        for i, p in enumerate(pts, 1):
            r = Residue.Residue((" ", i, " "), "GLY", " "); r.add(Atom.Atom("CA", p, 0, 1, " ", "CA", i, "C")); ch.add(r)
    f = tmp_path / "pred.cif"; io = MMCIFIO(); io.set_structure(s); io.save(str(f))
    out = common.site_occlusion(f, {"site_dir": str(d)})
    assert out["site_clash"] >= 2 and out["site_min_dist"] == pytest.approx(1.0, abs=1e-3)


def test_fetch_target_parser_keeps_resolved_standard_residues_in_range():
    import fetch_target
    pdb = "\n".join([
        "ATOM      1  N   ALA A  10      0.000   0.000   0.000  1.00  0.00           N",
        "ATOM      2  CA  ALA A  10      1.000   0.000   0.000  1.00  0.00           C",
        "ATOM      3  CA  GLY A  11      2.000   0.000   0.000  1.00  0.00           C",
        "ATOM      4  CA AASP A  12      3.000   0.000   0.000  1.00  0.00           C",
        "ATOM      5  CA  LYS B  10      9.000   0.000   0.000  1.00  0.00           C",
        "HETATM    6  CA  XYZ A  13      4.000   0.000   0.000  1.00  0.00           C"])
    res = fetch_target.parse_chain(pdb, "A", 10, 12)
    assert [(r[0], r[1]) for r in res] == [(10, "A"), (11, "G"), (12, "D")]


# ----------------------------------------------------------------------------- site finder (funnel/hotspots.py)
def _hs():
    import hotspots as hs                                  # adds .pxd/ext to sys.path if present
    pytest.importorskip("freesasa", reason="run ./scripts/setup_extras.sh (installs freesasa)")
    return hs


def _toy_chain(tmp_path):
    """A 6x6x4 block of residues (5 A grid): top face has a hydrophobic LEU centre among LYS, the bottom face is SER, the two interior layers are THR (genuinely buried)."""
    lines = []; n = 0
    for layer in range(4):                                  # layer 3 = top face
        for i in range(6):
            for j in range(6):
                n += 1; z = layer * 5.0
                name = {3: ("LEU" if 2 <= i <= 3 and 2 <= j <= 3 else "LYS"), 0: "SER"}.get(layer, "THR")
                for an, dz, el in (("N", -0.5, "N"), ("CA", 0.0, "C"), ("C", 0.4, "C"), ("O", 0.8, "O"), ("CB", 2.5, "C")):
                    lines.append(f"ATOM  {len(lines)+1:5d} {an:^4s} {name} A{n:4d}    {i*5.0:8.3f}{j*5.0:8.3f}{z+dz:8.3f}  1.00  0.00           {el}")
    p = tmp_path / "toy.pdb"; p.write_text("\n".join(lines) + "\nEND\n"); return p


def test_site_finder_prefers_the_exposed_hydrophobic_patch(tmp_path):
    hs = _hs()
    w = {k: 0.0 for k in hs.FEATURES}; w["hydrophobic"] = 1.0                                                   # isolate the mechanism: only hydrophobicity counts
    patches, res, s = hs.propose(_toy_chain(tmp_path), "A", None, None, n_hot=4, top=2, weights=w)
    assert patches
    leu = {r["num"] for r in res if r["aa"] == "L"}
    assert len({int("".join(c for c in h if c.isdigit())) for h in patches[0]["hotspots"]} & leu) >= 2          # the top patch includes the hydrophobic centre


def test_site_finder_never_proposes_buried_residues(tmp_path):
    hs = _hs()
    res = hs.read_chain(_toy_chain(tmp_path), "A"); s, rel, d = hs.score_residues(res)
    core = [l * 36 + i * 6 + j for l in (1, 2) for i in (2, 3) for j in (2, 3)]        # centre of the interior layers: surrounded on all sides
    assert all(not np.isfinite(s[k]) for k in core) and np.isfinite(s).any()


def test_site_finder_patches_are_spatially_separated(tmp_path):
    hs = _hs()
    patches, res, s = hs.propose(_toy_chain(tmp_path), "A", None, None, n_hot=3, top=3)
    pos = {f"{r['aa']}{r['num']}": r["ca"] for r in res}
    seeds = [np.array(pos[p["seed"]]) for p in patches]
    assert all(np.linalg.norm(a - b) >= 14.0 for i, a in enumerate(seeds) for b in seeds[i + 1:])


def test_conservation_requires_an_msa_that_fits_the_query(tmp_path):
    hs = _hs()
    f = tmp_path / "m.csv"; f.write_text("key,sequence\n0,ACDE\n1,ACDF\n2,ACDE\n3,AC-E\n4,ACDE\n5,ACDE\n")
    assert list(hs.conservation(f, "ACDE")) == pytest.approx([1.0, 1.0, 5 / 6, 5 / 6])
    assert hs.conservation(f, "ACD") is None                                                                     # length mismatch -> no silent misalignment


def test_ipsae_directional_separates_the_two_directions_and_agrees_with_min_max():
    nt, nb = 40, 30
    pae = np.full((nt + nb, nt + nb), 0.5); pae[:nt, nt:] = 30.0       # target->binder uncertain, binder->target confident
    b2t, t2b = common.ipsae_directional(pae, nt, nb)
    assert b2t > 0.8 and t2b == 0.0
    assert (min(b2t, t2b), max(b2t, t2b)) == common.ipsae(pae, nt, nb)
    pae2 = np.full_like(pae, 0.5); pae2[nt:, :nt] = 30.0               # and the other way round
    assert common.ipsae_directional(pae2, nt, nb)[0] == 0.0
