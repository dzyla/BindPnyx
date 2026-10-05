"""One stray byte in an alignment skips every design and still exits 0."""
import os, sys
import pytest
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from pxdbench.tools.boltz.msa_check import MsaMismatch, unparseable_a3m_bytes, validate_a3m

def _a3m(tmp_path, body, name="m.a3m"):
    p = tmp_path / name
    p.write_bytes(body)
    return str(p)

def test_a_clean_a3m_has_no_offenders(tmp_path):
    p = _a3m(tmp_path, b">q\nACDEFGHIKL\n>h1\nACDEFGHIKM\n")
    assert unparseable_a3m_bytes(p) == []

def test_a_nul_byte_is_found_with_its_line(tmp_path):
    p = _a3m(tmp_path, b">q\nACDEFGHIKL\n>h1\nACDEFGHIKM\n\x00\n")
    got = unparseable_a3m_bytes(p)
    assert got and got[0][0] == 5 and "x00" in got[0][1]

def test_lowercase_insertions_and_gaps_are_fine(tmp_path):
    p = _a3m(tmp_path, b">q\nACDEFGHIKL\n>h1\nACDefGHIK-L\n")
    assert unparseable_a3m_bytes(p) == []

def test_a_header_may_contain_anything(tmp_path):
    p = _a3m(tmp_path, b">q | tax=9606 (x)\nACDEFGHIKL\n")
    assert unparseable_a3m_bytes(p) == []

def test_validate_refuses_the_file_and_says_why(tmp_path):
    seq = "ACDEFGHIKL"
    body = (f">q\n{seq}\n").encode() + b"".join(
        f">h{i}\n{seq[:-1]}{c}\n".encode() for i, c in enumerate("ACDEFGHIKLMNPQRSTVWY" * 6)
    ) + b"\x00\n"
    p = _a3m(tmp_path, body)
    with pytest.raises(MsaMismatch, match="cannot"):
        validate_a3m(seq, p, "fima chain B", min_depth=1)

def test_the_real_fima_alignment_is_clean_now():
    f = os.path.join(REPO, "data/targets/fima/msa/A/0/non_pairing.a3m")
    if not os.path.isfile(f):
        pytest.skip("fima msa not on this machine")
    assert unparseable_a3m_bytes(f) == []
