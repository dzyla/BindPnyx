"""scripts/prepare_target.py - the refusals, mostly.

A tool that silently emits a shard disagreeing with its MSA is the failure this
exists to prevent, so the refusal paths carry the weight here. Every case below
is driven with files that are really in the tree - the 1NQL crystal, the
UniProt P00533 record, the FimA crop - because a fixture written to match the
code proves only that the code matches the fixture.
"""
import json
import os
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))

import prepare_target as pt  # noqa: E402

NQL = os.path.join(REPO, "targets", "egfr_ecd", "1nql_full.pdb")
WT = os.path.join(REPO, "targets", "egfr_ecd", "P00533_uniprot_wt.fasta")
FIMA_PDB = os.path.join(REPO, "data", "targets", "fima", "raw", "4DWH.pdb")   # python funnel/fetch_target.py funnel/targets/fima.json
FIMA_FASTA = os.path.join(REPO, "data", "targets", "fima", "wt.fasta")

needs_nql = pytest.mark.skipif(
    not os.path.isfile(NQL), reason="targets/egfr_ecd/1nql_full.pdb not present"
)

#: The crop the EGFR campaign conditions on, in 1NQL's own (mature) numbering.
CROP = ["A:227-306=B", "A:512-614=D"]
K516N = "D:516:LYS>ASN:truncate_to_cb"


def a3m_dir(tmp_path, seq, name, depth=140):
    """A real a3m whose query is `seq`, with `depth` genuinely unique records.

    The depth is asserted with the project's own counter rather than assumed:
    an earlier generation of helpers in this repo produced four unique
    sequences while claiming a hundred, and every test built on them passed.
    """
    from pxdbench.tools.boltz.msa_check import a3m_depths

    d = tmp_path / name / "0"
    d.mkdir(parents=True, exist_ok=True)
    out = [f">query\n{seq}\n"]
    for i in range(depth - 1):
        pos = i % len(seq)
        sub = "ACDEFGHIKLMNPQRSTVWY"[(i // len(seq) + 1) % 20]
        if sub == seq[pos]:
            sub = "W" if sub != "W" else "Y"
        out.append(f">hom{i}\n{seq[:pos]}{sub}{seq[pos + 1:]}\n")
    path = d / "non_pairing.a3m"
    path.write_text("".join(out))
    headers, unique = a3m_depths(str(path))
    assert unique >= depth - 2, (headers, unique)
    return str(d)


def run(tmp_path, *extra, chains=CROP, source=NQL, wt=WT, out=None):
    """-> (exit code, provenance record)."""
    out = out or str(tmp_path / "out")
    argv = ["--source", source, "--wt-fasta", wt, "--out-dir", out, "--no-shard"]
    for c in chains:
        argv += ["--chain", c]
    argv += list(extra)
    code = pt.main(argv)
    with open(os.path.join(out, "provenance.json")) as fh:
        return code, json.load(fh), out


# ---------------------------------------------------------------- refusals

@needs_nql
def test_the_1nql_crop_is_refused_against_uniprot_wt(tmp_path):
    """The whole point: K516 in the crystal, N in UniProt WT."""
    code, rec, out = run(tmp_path)
    assert code == 2
    assert rec["status"] == "refused"
    assert "516" in rec["refusal"] and "LYS" in rec["refusal"]
    assert "position 540" in rec["refusal"]          # precursor numbering
    assert not os.path.exists(os.path.join(out, "1nql_full_prepared.pdb"))
    assert not [f for f in os.listdir(out) if f.endswith((".pdb", ".pkl.gz"))]


@needs_nql
def test_the_refusal_prints_an_edit_that_actually_works(tmp_path):
    """The suggested command is the fix, not a plausible-looking string."""
    _, rec, _ = run(tmp_path)
    line = [l for l in rec["refusal"].splitlines() if "--edit" in l][0]
    spec = line.split("--edit", 1)[1].strip().split("   ")[0].strip("'\" ")
    code, rec2, _ = run(tmp_path, "--edit", spec, out=str(tmp_path / "b"))
    assert code == 0, rec2.get("refusal")
    assert spec == K516N


@needs_nql
def test_an_edit_may_only_move_a_residue_towards_the_declared_wt(tmp_path):
    code, rec, _ = run(tmp_path, "--edit", "D:516:LYS>ALA:truncate_to_cb")
    assert code == 2
    assert "WT position 540 is N" in rec["refusal"]


@needs_nql
def test_an_edit_whose_from_disagrees_with_the_structure_is_refused(tmp_path):
    code, rec, _ = run(tmp_path, "--edit", "D:516:ARG>ASN:truncate_to_cb")
    assert code == 2
    assert "declares FROM=ARG but the structure has LYS" in rec["refusal"]


@needs_nql
def test_an_edit_that_matches_no_residue_is_refused(tmp_path):
    code, rec, _ = run(
        tmp_path, "--edit", K516N, "--edit", "D:700:CYS>ASN:truncate_to_cb"
    )
    assert code == 2
    assert "match no residue" in rec["refusal"] and "D:700" in rec["refusal"]


@needs_nql
def test_an_identity_edit_is_refused_rather_than_stripping_a_side_chain(tmp_path):
    """D600 is a real CYS. A CYS>CYS truncation would delete its SG - and the
    disulfide with it - while reconciling nothing with the WT."""
    code, rec, _ = run(
        tmp_path, "--edit", K516N, "--edit", "D:600:CYS>CYS:truncate_to_cb"
    )
    assert code == 2
    assert "reconciles nothing" in rec["refusal"]


@needs_nql
def test_there_is_no_relabel_only_operation(tmp_path):
    """Relabelling alone leaves LYS atoms wearing an ASN name."""
    code, rec, _ = run(tmp_path, "--edit", "D:516:LYS>ASN:relabel")
    assert code == 2
    assert "unknown operation" in rec["refusal"]
    assert "donor's side-chain atoms" in rec["refusal"]


@needs_nql
def test_a_wrong_declared_offset_is_refused(tmp_path):
    code, rec, _ = run(tmp_path, "--wt-offset", "0")
    assert code == 2
    assert "declared WT has" in rec["refusal"]


def test_a_wt_that_is_not_an_integer_shift_is_refused(tmp_path):
    if not os.path.isfile(FIMA_PDB): pytest.skip("build it with: python funnel/fetch_target.py funnel/targets/fima.json")
    """FimA's fasta is the OBSERVED residues concatenated across two gaps, so
    no constant offset places the structure on it. Real files, real refusal."""
    code, rec, _ = run(
        tmp_path, "--allow-gaps", chains=["B:17-159"],
        source=FIMA_PDB, wt=FIMA_FASTA,
    )
    assert code == 2
    assert "could not place the structure on the WT sequence" in rec["refusal"]


def test_a_gap_in_the_requested_range_is_refused_by_default(tmp_path):
    if not os.path.isfile(FIMA_PDB): pytest.skip("build it with: python funnel/fetch_target.py funnel/targets/fima.json")
    code, rec, _ = run(
        tmp_path, chains=["B:17-159"], source=FIMA_PDB, wt=FIMA_FASTA
    )
    assert code == 2
    assert "91-94" in rec["refusal"] and "--allow-gaps" in rec["refusal"]


@needs_nql
def test_allow_gaps_records_which_residues_were_missing(tmp_path):
    """1NQL starts at residue 3; asking for 1-306 asks for two it does not have."""
    code, rec, _ = run(
        tmp_path, "--allow-gaps", chains=["A:1-306=B"]
    )
    assert code == 0, rec.get("refusal")
    sel = rec["selection"]["chains"][0]
    assert sel["unresolved_in_range"] == [1, 2]
    assert sel["n_residues"] == 304


@needs_nql
def test_a_chain_that_is_not_in_the_structure_is_refused(tmp_path):
    code, rec, _ = run(tmp_path, chains=["Z:1-10"])
    assert code == 2 and "not in" in rec["refusal"]


@needs_nql
def test_two_selections_cannot_share_an_output_chain(tmp_path):
    code, rec, _ = run(tmp_path, chains=["A:227-306=B", "A:512-614=B"])
    assert code == 2 and "share an output id" in rec["refusal"]


def test_a_cif_source_is_refused_rather_than_guessed(tmp_path):
    cif = tmp_path / "x.cif"
    cif.write_text("data_x\n")
    code, rec, _ = run(tmp_path, chains=["A"], source=str(cif))
    assert code == 2 and "only .pdb is read here" in rec["refusal"]


# ------------------------------------------------------------- the edit

@needs_nql
def test_the_edited_residue_carries_no_lysine_atoms(tmp_path):
    """The load-bearing one. A relabelled LYS would still have CG CD CE NZ."""
    code, rec, out = run(tmp_path, "--edit", K516N)
    assert code == 0, rec.get("refusal")
    pdb = os.path.join(out, "1nql_full_prepared.pdb")
    atoms = [
        line for line in open(pdb)
        if line.startswith("ATOM") and line[21] == "D"
        and int(line[22:26]) == 516
    ]
    assert {line[17:20].strip() for line in atoms} == {"ASN"}
    assert sorted(line[12:16].strip() for line in atoms) == [
        "C", "CA", "CB", "N", "O"
    ]
    edit = rec["edits"][0]
    assert edit["op"] == "truncate_to_cb"
    assert edit["atoms_removed"] == ["CG", "CD", "CE", "NZ"]
    assert edit["from"] == "LYS" and edit["to"] == "ASN"


@needs_nql
def test_the_emitted_sequence_matches_uniprot_wt_exactly(tmp_path):
    code, rec, _ = run(tmp_path, "--edit", K516N)
    assert code == 0
    wt = "".join(
        l.strip() for l in open(WT) if not l.startswith(">")
    )
    for chain, sel in zip(rec["result"]["chains"], rec["selection"]["chains"]):
        lo = int(sel["ranges"].split("-")[0]) + sel["wt_offset"] - 1
        assert chain["sequence"] == wt[lo:lo + chain["length"]]


@needs_nql
def test_truncating_a_glycine_is_refused_rather_than_modelled(tmp_path):
    """G -> anything needs a CB that is not in the source. Not this tool's job."""
    res = pt.Residue("D", 1, " ", "GLY", False)
    res.atoms = [
        "ATOM      1  N   GLY D   1      00.000  00.000  00.000  1.00  0.00",
        "ATOM      2  CA  GLY D   1      00.000  00.000  00.000  1.00  0.00",
        "ATOM      3  C   GLY D   1      00.000  00.000  00.000  1.00  0.00",
        "ATOM      4  O   GLY D   1      00.000  00.000  00.000  1.00  0.00",
    ]
    with pytest.raises(pt.Refused, match="no CB to keep"):
        pt.apply_truncate_to_cb(res, "ASN")


# --------------------------------------------------------------- the record

@needs_nql
def test_the_provenance_record_answers_every_question_it_must(tmp_path):
    msa_b = a3m_dir(tmp_path, _wt_slice(227, 306), "msaB")
    msa_d = a3m_dir(tmp_path, _wt_slice(512, 614), "msaD")
    code, rec, _ = run(
        tmp_path, "--edit", K516N, "--msa", f"B={msa_b}", "--msa", f"D={msa_d}"
    )
    assert code == 0, rec.get("refusal")
    assert rec["status"] == "written"
    assert len(rec["source"]["sha256"]) == 64
    assert rec["source"]["path"].endswith("1nql_full.pdb")
    assert len(rec["wt"]["sequence_sha256"]) == 64
    assert rec["wt"]["numbering"] == "mature"
    sel = {c["output_chain"]: c for c in rec["selection"]["chains"]}
    assert sel["D"]["source_chain"] == "A" and sel["D"]["ranges"] == "512-614"
    assert sel["D"]["numbering"] == "auth" and sel["D"]["wt_offset"] == 24
    assert rec["edits"][0]["atoms_removed"]
    assert all(len(c["sequence_sha256"]) == 64 for c in rec["result"]["chains"])
    assert {m["output_chain"] for m in rec["msa"]} == {"B", "D"}
    assert all(m["status"] == "valid" for m in rec["msa"])
    assert all(len(m["a3m_sha256"]) == 64 for m in rec["msa"])
    md = open(os.path.join(os.path.dirname(rec["prepared_pdb"]["path"]),
                           "PROVENANCE.md")).read()
    assert "truncate_to_cb" in md and "NOT relabelled" in md


@needs_nql
def test_the_residue_map_links_source_output_and_wt(tmp_path):
    code, rec, _ = run(tmp_path, "--edit", K516N)
    assert code == 0
    d = {r["source_resnum"]: r for r in rec["residue_map"]["D"]}
    assert d[512]["source"] == "A512" and d[512]["wt_position"] == 536
    assert d[512]["shard_resnum"] == 1
    assert d[516] == {
        "source": "A516", "source_resnum": 516, "prepared_resnum": 516,
        "shard_resnum": 5, "wt_position": 540, "residue": "N", "edited": True,
    }
    assert d[614]["wt_position"] == 638
    assert len(rec["residue_map"]["D"]) == 103
    assert sum(r["edited"] for r in rec["residue_map"]["D"]) == 1


# ------------------------------------------------------------------ the MSA

@needs_nql
def test_an_msa_that_does_not_apply_refuses_and_writes_no_structure(tmp_path):
    """Chain B's crop paired with chain D's alignment."""
    wrong = a3m_dir(tmp_path, _wt_slice(512, 614), "msaD")
    code, rec, out = run(tmp_path, "--edit", K516N, "--msa", f"B={wrong}")
    assert code == 2
    assert "does not match the MSA" in rec["refusal"]
    assert not [f for f in os.listdir(out) if f.endswith((".pdb", ".pkl.gz"))]
    assert rec["msa"][0]["status"] == "INVALID"


@needs_nql
def test_a_shallow_msa_refuses(tmp_path):
    shallow = a3m_dir(tmp_path, _wt_slice(227, 306), "shallow", depth=10)
    code, rec, _ = run(tmp_path, "--edit", K516N, "--msa", f"B={shallow}")
    assert code == 2 and "below 100" in rec["refusal"]


@needs_nql
def test_a_chain_without_an_msa_is_recorded_as_unvalidated(tmp_path):
    msa_b = a3m_dir(tmp_path, _wt_slice(227, 306), "msaB")
    code, rec, _ = run(tmp_path, "--edit", K516N, "--msa", f"B={msa_b}")
    assert code == 0
    by = {m["output_chain"]: m for m in rec["msa"]}
    assert by["B"]["status"] == "valid"
    assert by["D"]["status"] == "not supplied"
    assert "NOT validated" in by["D"]["note"]


# ----------------------------------------------------------------- plumbing

def test_boltz_light_loads_the_module_the_package_exposes():
    """The pre-flight must not drift from the code prepare_json runs."""
    import pxdbench.tools.boltz.msa_check as packaged
    import pxdbench.tools.boltz.targets as packaged_targets
    import boltz_light

    assert os.path.realpath(boltz_light.load("msa_check").__file__) == \
        os.path.realpath(packaged.__file__)
    assert os.path.realpath(boltz_light.load("targets").__file__) == \
        os.path.realpath(packaged_targets.__file__)


def test_prepare_target_is_executable_as_a_command():
    out = subprocess.run(
        [sys.executable, os.path.join(REPO, "scripts", "prepare_target.py"),
         "--help"],
        capture_output=True, text=True, timeout=120,
    )
    assert out.returncode == 0
    assert "truncate_to_cb" in out.stdout


def _wt_slice(lo, hi):
    """The UniProt WT residues for a 1NQL mature-numbered range."""
    wt = "".join(l.strip() for l in open(WT) if not l.startswith(">"))
    return wt[lo + 24 - 1:hi + 24]
