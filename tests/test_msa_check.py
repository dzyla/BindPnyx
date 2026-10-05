import os

import pytest

from pxdbench.tools.boltz.msa_check import (
    DEFAULT_MIN_DEPTH,
    MsaMismatch,
    a3m_depths,
    compare,
    file_sha256,
    query_residues,
    sequence_sha256,
    validate_a3m,
    validate_msa_dir,
    validate_target_chains,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIMA_A3M = os.path.join(REPO, "data", "targets", "fima", "msa", "A", "0", "non_pairing.a3m")   # built by: python funnel/fetch_target.py funnel/targets/fima.json
FIMA_SEQ = (
    "MNAACAVDAGSVDQTVQLGQVRTASLAQEGATSSAVGFNIQLNDCDTNVASKAAVAFLGTAIDAGHTNVL"
    "ALQSSATNVGVQILDRTGAALTLDGATFSSGTNTIPFQARYFATGAATPGAANADATFKVQYQ"
)


def _a3m(tmp_path, *records, name="non_pairing.a3m"):
    """Write an a3m. Each record is (header, sequence_line)."""
    body = "".join(f">{h}\n{s}\n" for h, s in records)
    d = tmp_path / "msa"
    d.mkdir(parents=True, exist_ok=True)
    p = d / name
    p.write_text(body)
    return str(d), str(p)


def _deep(seq, n=DEFAULT_MIN_DEPTH + 20):
    """A query plus n-1 genuinely DISTINCT homologues.

    An earlier version of this helper cycled `i % 3` and produced **four**
    unique sequences out of a hundred records. `validate_msa_dir` counts UNIQUE
    sequences, because that is what boltz loads, so every test asking for depth
    >= 100 was really supplying depth 4 and would have failed. The fixture, not
    the code, was wrong - which is the failure mode CLAUDE.md s8 warns about.
    """
    aa = "ACDEFGHIKLMNPQRSTVWY"
    recs = [("query", seq)]
    seen = {seq}
    i = 0
    limit = len(seq) * len(aa)
    while len(recs) < n and i < limit:
        pos, letter = i % len(seq), aa[(i // len(seq)) % len(aa)]
        cand = seq[:pos] + letter + seq[pos + 1:]
        if cand not in seen:
            seen.add(cand)
            recs.append((f"hom{len(recs)}", cand))
        i += 1
    if len(recs) < n:
        raise AssertionError(
            f"fixture can only reach depth {len(recs)} for a {len(seq)}-mer; "
            f"use a longer sequence or ask for less"
        )
    return recs


def test_the_deep_fixture_really_is_deep(tmp_path):
    """Guards the FIXTURE, not the code. Every depth assertion below rests on
    this, and the previous helper silently supplied 4 unique sequences."""
    _, p = _a3m(tmp_path, *_deep("MKTAYIAKQR"))
    headers, unique = a3m_depths(p)
    assert unique >= DEFAULT_MIN_DEPTH, f"{headers} records, {unique} unique"


# --- query_residues: boltz's parser, not a naive read

def test_query_is_the_first_sequence_line():
    if not os.path.isfile(FIMA_A3M): pytest.skip("build it with: python funnel/fetch_target.py funnel/targets/fima.json")
    assert query_residues(FIMA_A3M) == FIMA_SEQ


def test_lowercase_insertion_columns_are_dropped(tmp_path):
    """boltz's a3m parser skips lowercase inserts; they are not residues."""
    _, p = _a3m(tmp_path, ("q", "MKTaayIAK"))
    assert query_residues(p) == "MKTIAK"


def test_gaps_are_kept_as_residues(tmp_path):
    """'-' becomes a gap TOKEN in boltz, so it must not be stripped."""
    _, p = _a3m(tmp_path, ("q", "MK-TA"))
    assert query_residues(p) == "MK-TA"


def test_comment_and_blank_lines_are_skipped(tmp_path):
    p = tmp_path / "c.a3m"
    p.write_text("# a comment\n\n>q\nMKTA\n")
    assert query_residues(str(p)) == "MKTA"


def test_an_a3m_with_no_sequence_line_gives_none(tmp_path):
    p = tmp_path / "e.a3m"
    p.write_text(">only_a_header\n")
    assert query_residues(str(p)) is None


# --- a3m_depths: headers overcount what boltz loads

def test_depth_reports_headers_and_unique_separately(tmp_path):
    _, p = _a3m(tmp_path, ("a", "MKTA"), ("b", "MKTA"), ("c", "QRST"))
    assert a3m_depths(p) == (3, 2)


def test_dedup_ignores_gaps_and_case_like_boltz(tmp_path):
    """boltz dedups on line.replace('-','').upper(), so these are one sequence."""
    _, p = _a3m(tmp_path, ("a", "MKTA"), ("b", "mk-ta"))
    assert a3m_depths(p) == (2, 1)


def test_the_real_fima_msa_depth():
    if not os.path.isfile(FIMA_A3M): pytest.skip("build it with: python funnel/fetch_target.py funnel/targets/fima.json")
    """4024 records, 4023 unique. It read (4024, 4024) until 2026-10-03,
    when the 4024th 'sequence' turned out to be a lone NUL byte on the last
    line - which boltz's a3m parser cannot tokenise, so it skipped every
    design and exited 0. The byte was removed; see test_a3m_bytes.py."""
    assert a3m_depths(FIMA_A3M) == (4024, 4023)


# --- compare: boltz's comparison INCLUDING its tolerance

def test_an_exact_match_is_ok():
    ok, reason = compare("MKTA", "MKTA")
    assert ok is True
    assert "exact" in reason


def test_a_single_substitution_fails_and_names_the_position():
    ok, reason = compare("MKNA", "MKTA")
    assert ok is False
    assert "pos 2" in reason and "sequence=N" in reason and "msa=T" in reason


def test_a_length_difference_fails_as_boltz_warning_2():
    ok, reason = compare("MKT", "MKTA")
    assert ok is False
    assert "length differs" in reason and "'2'" in reason


def test_met_against_unk_is_tolerated_because_boltz_repairs_it():
    """boltz replaces the msa residue instead of dummying. Rejecting this would
    be STRICTER than boltz and would throw away a usable MSA."""
    ok, reason = compare("MKTA", "XKTA")
    assert ok is True
    assert "MET/UNK" in reason


def test_an_unknown_input_residue_is_tolerated():
    ok, _ = compare("XKTA", "MKTA")
    assert ok is True


def test_a_mixed_mismatch_set_is_not_tolerated():
    """One tolerable and one real mismatch is still a discarded MSA."""
    ok, reason = compare("MKNA", "XKTA")
    assert ok is False
    assert "'1'" in reason


def test_mixed_tolerated_input_x_and_met_unk_is_not_tolerated():
    """Boltz: (all met/msa-unk) or (all input-unk) is false for a mixed set."""
    ok, reason = compare("XKMA", "MKXA")
    assert ok is False
    assert "'1'" in reason


def test_mixed_with_input_u_is_also_not_tolerated():
    """U is UNK to boltz, so it is an input-X position here too."""
    ok, _ = compare("UKMA", "MKXA")
    assert ok is False


def test_u_against_x_is_equal_to_boltz():
    assert compare("MUTA", "MXTA")[0] is True


def test_b_against_z_is_equal_to_boltz():
    assert compare("MBTA", "MZTA")[0] is True


def test_u_in_msa_against_a_non_met_residue_is_still_rejected():
    """msa U is UNK; only (input MET, msa UNK) is repaired, so input K fails."""
    assert compare("MKTA", "MUTA")[0] is False


def test_input_u_against_m_is_tolerated_because_input_unk_is_repaired():
    """Input U is UNK to boltz, so `all is_unk` holds exactly as for input X."""
    assert compare("MUTA", "MMTA")[0] is True


def test_met_against_u_msa_is_tolerated_as_unk():
    assert compare("MKTA", "UKTA")[0] is True


def test_a_gap_is_not_an_unk():
    assert compare("MATA", "M-TA")[0] is False


def test_mismatch_message_names_original_letters_not_normalised():
    ok, reason = compare("MKTA", "MUTA")
    assert ok is False
    assert "sequence=K" in reason and "msa=U" in reason
    ok, reason = compare("MBKA", "MKTA")
    assert ok is False
    assert "pos 1 sequence=B msa=K" in reason and "pos 2 sequence=K msa=T" in reason


def test_a_gap_is_not_an_unk_message():
    ok, reason = compare("MATA", "M-TA")
    assert "sequence=A" in reason and "msa=-" in reason


def test_a_gap_in_the_query_is_a_mismatch():
    ok, _ = compare("MKTA", "MK-A")
    assert ok is False


def test_a_missing_query_fails_rather_than_raising():
    ok, reason = compare("MKTA", None)
    assert ok is False
    assert "no sequence line" in reason


# --- hashes

def test_sequence_hash_is_stable_and_differs_on_one_residue():
    assert sequence_sha256("MKTA") == sequence_sha256("MKTA")
    assert sequence_sha256("MKTA") != sequence_sha256("MKNA")


def test_file_hash_matches_a_known_value(tmp_path):
    p = tmp_path / "f.txt"
    p.write_text("abc")
    assert file_sha256(str(p)) == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


# --- validate_msa_dir: the gate

def test_a_valid_msa_returns_a_provenance_record(tmp_path):
    d, p = _a3m(tmp_path, *_deep("MKTAYIAKQR"))
    rec = validate_msa_dir("MKTAYIAKQR", d, label="chain A")
    assert rec["label"] == "chain A"
    assert rec["depth_unique"] >= DEFAULT_MIN_DEPTH
    assert rec["sequence_sha256"] == sequence_sha256("MKTAYIAKQR")
    assert rec["a3m_sha256"] == file_sha256(p)


def test_a_mismatched_query_raises_and_says_what_is_lost(tmp_path):
    """The 101-sequence mismatched MSA from the review."""
    d, _ = _a3m(tmp_path, *_deep("MKTAYIAKQR", n=101))
    with pytest.raises(MsaMismatch) as e:
        validate_msa_dir("MKTAYIAKNR", d, label="chain A")
    assert "chain A" in str(e.value)
    assert "101" in str(e.value)
    assert "unconditioned" in str(e.value)


def test_a_missing_a3m_raises(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    with pytest.raises(MsaMismatch, match="no such MSA file"):
        validate_msa_dir("MKTA", str(d))


def test_a_shallow_msa_raises_even_when_the_query_matches(tmp_path):
    d, _ = _a3m(tmp_path, ("q", "MKTA"))
    with pytest.raises(MsaMismatch, match="below") as e:
        validate_msa_dir("MKTA", d)
    # M5: the floor is now mandatory on paths that never had one; name the knob
    assert "tools.boltz.min_msa_depth" in str(e.value)
    assert "min_depth" in str(e.value)


def test_depth_is_judged_on_unique_not_header_count(tmp_path):
    """Ten identical records are depth 1 to boltz, however many headers."""
    d, _ = _a3m(tmp_path, *[(f"h{i}", "MKTA") for i in range(10)])
    with pytest.raises(MsaMismatch) as e:
        validate_msa_dir("MKTA", d, min_depth=5)
    assert "MSA depth 1 (from 10 records)" in str(e.value)


def test_min_depth_is_tunable_but_identity_is_not(tmp_path):
    """A caller may accept a shallow MSA; nobody may accept a mismatched one."""
    d, _ = _a3m(tmp_path, ("q", "MKTA"), ("h", "MKTV"))
    assert validate_msa_dir("MKTA", d, min_depth=2)["depth_unique"] == 2
    with pytest.raises(MsaMismatch):
        validate_msa_dir("MKNA", d, min_depth=1)


def test_the_real_fima_target_validates():
    if not os.path.isfile(FIMA_A3M): pytest.skip("build it with: python funnel/fetch_target.py funnel/targets/fima.json")
    """Positive control on a real cached MSA, so the gate is not vacuous."""
    rec = validate_msa_dir(FIMA_SEQ, os.path.dirname(FIMA_A3M), label="fima B")
    assert rec["depth_unique"] == 4023


def test_the_real_fima_msa_rejects_a_one_residue_change():
    if not os.path.isfile(FIMA_A3M): pytest.skip("build it with: python funnel/fetch_target.py funnel/targets/fima.json")
    """The measured failure: one substitution discards every sequence."""
    mutated = FIMA_SEQ[:51] + "N" + FIMA_SEQ[52:]
    with pytest.raises(MsaMismatch, match="pos 51"):
        validate_msa_dir(mutated, os.path.dirname(FIMA_A3M))


# --- validate_a3m: the core gate, exercised directly

def test_validate_a3m_takes_a_FILE_path_not_a_directory(tmp_path):
    """The resolver hands over a path to the a3m itself, which is why this is
    the function production calls."""
    _, p = _a3m(tmp_path, *_deep("MKTAYIAKQR"))
    rec = validate_a3m("MKTAYIAKQR", p, label="chain A")
    assert rec["a3m"] == p
    assert rec["label"] == "chain A"


def test_validate_a3m_rejects_a_directory_path(tmp_path):
    """Passing the directory is the mistake the old signature invited."""
    d, _ = _a3m(tmp_path, *_deep("MKTAYIAKQR"))
    with pytest.raises(MsaMismatch, match="no such MSA file"):
        validate_a3m("MKTAYIAKQR", d)


# --- validate_target_chains: the RESOLVED pairs the YAML writer receives

def _resolve(entities, **kw):
    """Build resolved chains through the REAL resolver, never by hand."""
    from pxdbench.tools.boltz.targets import target_chains_from_orig_seqs

    return target_chains_from_orig_seqs(entities, **kw)


def _entity(seq, msa, chain="A0", use_msa=True):
    """`msa` may be a dict OR a bare string - msa_path_of accepts both."""
    inner = {"sequence": seq, "count": 1, "label_asym_id": [chain]}
    if use_msa:
        inner["use_msa"] = True
    if msa is not None:
        inner["msa"] = (
            msa if isinstance(msa, str) else {"precomputed_msa_dir": msa}
        )
    return {"proteinChain": inner}


def test_every_resolved_chain_is_validated(tmp_path):
    da, _ = _a3m(tmp_path / "a", *_deep("MKTAYIAKQR"))
    db, _ = _a3m(tmp_path / "b", *_deep("QRSTVWYCDE"))
    recs = validate_target_chains(_resolve([
        _entity("MKTAYIAKQR", da, "A0"),
        _entity("QRSTVWYCDE", db, "B0"),
    ]))
    # the resolver renames A0 -> A
    assert [r["label"] for r in recs] == ["target chain A", "target chain B"]


def test_the_second_chain_is_not_skipped_when_the_first_is_fine(tmp_path):
    """Pairing MSAs by position is how a target silently loses one chain."""
    da, _ = _a3m(tmp_path / "a", *_deep("MKTAYIAKQR"))
    db, _ = _a3m(tmp_path / "b", *_deep("QRSTVWYCDE"))
    with pytest.raises(MsaMismatch, match="chain B"):
        validate_target_chains(_resolve([
            _entity("MKTAYIAKQR", da, "A0"),
            _entity("QRSTVWYCDD", db, "B0"),
        ]))


def test_a_target_msa_override_is_the_thing_validated(tmp_path):
    """THE CASE validating orig_seqs misses: the entity's MSA is correct and
    the override that replaces it is not."""
    good, _ = _a3m(tmp_path / "g", *_deep("MKTAYIAKQR"))
    bad, _ = _a3m(tmp_path / "b", *_deep("QRSTVWYCDE"))
    chains = _resolve(
        [_entity("MKTAYIAKQR", good, "A0")],
        target_msa_override={"A": os.path.join(bad, "non_pairing.a3m")},
    )
    with pytest.raises(MsaMismatch, match="length differs|mismatch"):
        validate_target_chains(chains)


def test_a_nondefault_a3m_name_is_honoured(tmp_path):
    """a3m_name is configurable; a hardcoded filename checks the wrong file."""
    seq = "MKTAYIAKQR"
    d, _ = _a3m(tmp_path / "c", *_deep(seq), name="custom.a3m")
    # the default name is absent from that directory
    chains = _resolve([_entity(seq, d, "A0")], a3m_name="custom.a3m")
    assert validate_target_chains(chains)[0]["depth_unique"] >= DEFAULT_MIN_DEPTH

    with pytest.raises(MsaMismatch, match="no such MSA file"):
        validate_target_chains(_resolve([_entity(seq, d, "A0")]))


def test_a_string_msa_path_is_validated(tmp_path):
    """msa may be a bare path string, not only a dict."""
    seq = "MKTAYIAKQR"
    _, p = _a3m(tmp_path / "s", *_deep(seq))
    chains = _resolve([_entity(seq, p, "A0")])
    assert chains[0]["msa"] == p
    assert validate_target_chains(chains)[0]["sequence_len"] == len(seq)


def test_use_msa_false_does_not_exempt_a_chain(tmp_path):
    """msa_path_of never consults use_msa, so the MSA still reaches the YAML.
    Skipping validation on that flag left exactly this hole open."""
    good, _ = _a3m(tmp_path / "g", *_deep("MKTAYIAKQR"))
    chains = _resolve([_entity("MKTAYIAKNR", good, "A0", use_msa=False)])
    with pytest.raises(MsaMismatch, match="mismatch"):
        validate_target_chains(chains)


def test_a_resolved_chain_with_no_sequence_raises():
    with pytest.raises(MsaMismatch, match="no sequence"):
        validate_target_chains([{"id": "A", "msa": "/x.a3m"}])


def test_a_resolved_chain_with_no_msa_raises():
    with pytest.raises(MsaMismatch, match="no MSA path"):
        validate_target_chains([{"id": "A", "seq": "MKTA"}])


def test_a_dict_instead_of_a_list_is_rejected():
    with pytest.raises(MsaMismatch, match="list of resolved chain dicts"):
        validate_target_chains({"B": "MKTAYIAK"})


def test_the_warning_string_matches_the_installed_boltz():
    """Pins it against the real source, so a boltz upgrade that reworded it
    fails here rather than silently disabling detection."""
    import inspect

    try:
        from boltz.data.feature import featurizerv2
    except ImportError:
        pytest.skip("boltz is not importable from this interpreter")
    from pxdbench.tools.boltz.msa_check import DUMMY_MSA_WARNING

    assert DUMMY_MSA_WARNING in inspect.getsource(featurizerv2)
